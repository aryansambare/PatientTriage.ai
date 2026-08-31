# PatientTriage.ai — Solution

A triage assistant that gives the nurse a second opinion at the door and then keeps
watching the waiting room she has to walk away from.

An ED nurse at triage has on the order of a minute per patient and is making a
5-level acuity call under fatigue and interruption. The failure that costs lives is
**undertriage** — a sick patient parked in the waiting room. So this is not "an AI
that triages." It is two things:

1. A **second opinion** at the moment of arrival that is never allowed to downgrade
   a nurse's own judgement.
2. A **watcher for the waiting room**, which is where nobody is currently looking
   once triage is done.

The second is where most of the value sits. The first is the part every triage
project builds; the second is the part that catches the worst case, and it drives
the whole design below.

We validated the approach against 32,232 real emergency department visits (CDC
NHAMCS, 2021–2022) before building the full system around it — see
[EVALUATION.md](EVALUATION.md) for the numbers, including the ones that argue
against deploying this as it stands today.

**Run the prototype:**

```bash
python -m patienttriage.demo.run                    # district general, 250 visits/day
python -m patienttriage.demo.run --profile rural    # 100/day, rules-only, no paediatrics
python -m patienttriage.demo.run --profile urban    # 550/day, major trauma centre
python -m patienttriage.ui.server                    # browser console
```

---

## 1. What the system decides, and what it only recommends

| The system DECIDES (autonomous) | The system RECOMMENDS (nurse confirms) | The system NEVER does |
|---|---|---|
| Re-check timing / who to re-assess and when | ESI acuity level (1–5) | Assign a final acuity itself |
| Waiting-room re-scan cadence | Queue order within the same acuity band | Lower an acuity a human set |
| Which patients to surface on the watch board | A recommended destination zone | Discharge, or refuse care |
| Firing a deterioration alert | — | Act without an audit record |

The rule that follows from this: **the model can only raise an acuity, never lower
one.** A nurse's override is one tap, needs no justification to record, and is
logged as both an accountability record and a training label.

---

## 2. Stated assumptions

| Assumption | Value | Why |
|---|---|---|
| Regulatory jurisdiction | **HIPAA (United States)**, with a GDPR profile implemented alongside | Named because it changes the code, not just the paperwork — see §9 |
| Department size | 100–550 visits/day across three site profiles | Covers a rural unit through to a major trauma centre |
| Severity scale | 5-level, 1 most acute (ESI-compatible) | Standard, and the validation data uses it |
| Time-to-clinician targets | ATS targets: 0 / 10 / 30 / 60 / 120 min | ESI defines no times; a department still needs numbers |
| Prior record availability | ~50% of arrivals | Matched in the demo cohort |
| Surge threshold | 3× normal hourly volume | See §10 |
| Undertriage : overtriage cost | Not a constant — calibrated per site | See §4 |

---

## 3. Data strategy — designing for data that is mostly not there

Realistically present within the first few minutes of arrival: vitals (HR, SBP/DBP,
RR, SpO2, temperature, pain score), age and sex, a short chief complaint, arrival
mode, and — if the record matches — a handful of prior-history flags (anticoagulant,
beta-blocker, immunosuppressed, prior ICU admission). Not available and never
assumed: labs, imaging, an ECG interpretation, the eventual diagnosis. The model has
to work with vitals, one short string, and age/sex alone, and treat everything else
as an optional bonus feature.

**Availability is a property of the field, declared once.** `features/schema.py`
records when each field becomes knowable, and `assert_triage_time_only` refuses any
feature matrix containing something that does not exist yet at prediction time.
Most published triage models look strong because they quietly train on data that
arrives later; this one is built so it cannot.

**Missingness is signal, never imputed.** A patient too combative, too young or too
unwell to measure is a different patient from one whose observations are normal.
Absent vitals become explicit features (`heart_rate__missing`,
`missing_vital_count`) and are shown on screen as a completeness indicator beside
the confidence. Nothing is filled in with a population median the patient never had.

**Unknown is not False.** Where there is no prior record, every history field is
`None` — not `False`. The distinction matters: "no record found" must never be read
as "confirmed not anticoagulated," which is the input to a red-flag rule. Half the
demo cohort arrives with no record at all, and a test enforces that they carry
`None`.

**Implausible values are treated as failed measurements.** The 2022 NHAMCS extract
contains a recorded pulse of 998. Three options exist and only one is honest:
crashing drops the record (and strange values cluster in sick patients), passing it
through has the model learn from a heart rate of 998, so `data/quality.py` treats it
as not recorded *and counts how often that happens*. The count is reported at
ingest.

**Two forms of chief complaint.** Departments record it as free text or as a code
from a picklist. Both are supported. Coded reasons deliberately do **not** drive the
rule layer — band-level codes are too coarse, and wiring them into red flags would
escalate every sprained ankle in the injury module.

**The complaint is read by a cited lexicon, not a language model.** `rules/lexicon.py`
matches free text against a maintained set of clinical patterns (stroke, arrest,
anaphylaxis, sepsis language, and so on), each pattern chosen to avoid the obvious
false positives — "under arrest" must never trigger a cardiac-arrest rule. This is
deterministic and auditable: a nurse can read the exact pattern that fired. A
transformer-based text branch is a plausible upgrade once a dataset with genuine
free-text complaints is available (an optional `text` dependency group is reserved
for it), but it is not part of the shipped system, and nothing here claims it is.

The largest constraint this creates: the primary validation dataset (NHAMCS) has no
free-text chief complaint at all, only coded reasons for visit — see
[EVALUATION.md](EVALUATION.md) §2 for what that costs the model, and why MIMIC-IV-ED
credentialing is the next real step.

---

## 4. Age-stratified physiology

Age-dependent thresholds are the largest single source of the undertriage risk in a
one-size-fits-all vitals model, and closing that gap is the biggest structural
change in this build. `clinical/agebands.py` is the single source of truth for every
age-dependent threshold — heart-rate and respiratory ranges, shock threshold, fever
and hypothermia bars — across nine bands from neonate to geriatric. Three effects
are encoded beyond simple lookup tables, because each can kill someone and none of
them is a threshold:

**Children compensate, then crash.** A child maintains blood pressure until they are
nearly arrested, so paediatric hypotension is a pre-arrest sign, not an early
warning. Rule **R22** fires on marked tachycardia and tachypnoea *with a normal
pressure for age* — the normal pressure is what makes it dangerous.

**Older patients mount a blunted fever response.** A septic 85-year-old is often
normothermic, and not rarely hypothermic. Rule **R21** lowers the bar to 37.5 °C in
the geriatric band and treats hypothermia as more alarming than fever.

**The same number means different things at different ages, by design, not by
accident.** `fever_significance(3.0, 38.5)` and `fever_significance(75.0, 38.5)`
return different clinical statements for the same temperature, and a test asserts
they differ. SBP 95 is unremarkable in a 40-year-old, shock in an 82-year-old, and
fine for a toddler. RR 34 is respiratory distress in an adult and entirely normal in
an infant. Both are tested.

---

## 5. Decision model — hybrid, with the value judgement kept separate

```
L0  rules/engine.py       24 deterministic red flags, each with a clinical citation.
                          Escalation only — returns an acuity FLOOR, never a ceiling.
L1  models/ordinal.py     Cumulative-link LightGBM over acuity 1–5, isotonic calibrated
                          per threshold, so the ordering is structural.
L2  deterioration.py      P(dies in ED or needs critical care) — drives the watch board.
L3  models/conformal.py   Split conformal. Abstains when the prediction set straddles
                          the urgent / can-wait boundary. 90.9% empirical coverage.
L4  models/explain.py     SHAP selects the facts; phrasing is written against the
                          feature so a nurse can disagree with a specific observation.
L5  service/pipeline.py   Assembles the stack. Never raises; degrades to L0.
```

**Rules and model answer different questions, and the rules win.** A fired red flag
is not a probabilistic claim that can be uncertain — it is a fact about the
handover. A bug found during development had model abstention *masking* a level-1
rule, so a cardiac arrest displayed as "uncertain." Now a fired rule always owns the
display, and abstention describes only what the model declined to add on top of it.
There is a regression test for exactly this.

**The undertriage cost is not a constant.** Early simulation showed that escalating
30% of arrivals makes critically unwell patients wait *ten hours longer* than doing
nothing, while 7% helps — the same architecture, opposite sign, purely from where the
escalation budget is set. So the cost ratio is calibrated per site from the
department's real escalation budget, on held-out data. For the district profile that
resolves to 2.0, flagging 15% of arrivals.

**Confidence is reported for the band, not the level.** The decision rule
deliberately recommends a more acute level than the single most likely one, so
"probability of the recommended level" reads as absurdly low exactly when the system
is being appropriately cautious. What is reported is confidence in *urgent* versus
*can wait* — the only distinction that changes what happens next.

**Every output carries a confidence indicator, without exception**, including the
degraded path where no model exists at all. A test asserts it over the whole cohort.

**NEWS2 rides alongside as a second opinion, never as a decision.** `clinical/news2.py`
reports the Royal College of Physicians' six-parameter early warning score next to
our own number — a scale a nurse already trusts without having to learn ours. It
answers `Assessment.news2_score` only, never `recommended_acuity`; a test constructs
a patient NEWS2 scores "high" and asserts the recommended acuity is exactly what the
rule engine alone would give the same vitals. It is also scoped honestly: Scale 1 is
an adult, non-pregnant tool, so it reports nothing for anyone under 18 or pregnant
rather than silently applying an adult scale.

**A recommended destination follows the acuity, not just a number.**
`clinical/routing.py` maps the acuity to a named zone, a bed type, and the same
time-to-clinician target the waiting-room monitor uses, and names it when a site
lacks a capability a patient needs (no paediatric service, no obstetric service, not
a designated trauma centre). It deliberately does not track or invent bed-by-bed
occupancy — no feed for that exists — so it reports the zone a patient of this
acuity normally goes to, for a charge nurse to check against the real board, not a
claim that a specific bed is free right now.

---

## 6. Workflow — the waiting room is the product

The most explicit requirement this design answers: monitor patients already in the
queue, and trigger re-assessment on wait-time breach or worsening vitals.
`monitoring/waitingroom.py` raises four triggers:

| Trigger | Fires when | Changes acuity? |
|---|---|---|
| `WAIT_BREACH` | Past the time-to-clinician target for their level | No — puts them in front of a human |
| `WAIT_APPROACHING` | At 80% of target | No |
| `VITALS_WORSENING` | Repeat observations moved the wrong way | **Yes — escalates to level 2** |
| `UNOBSERVED` | Nothing recorded for twice the re-check interval | No |

Deterioration is detected two ways, because either alone misses cases: **trend** (a
fall of 15 mmHg, a rise of 15 bpm — a patient can deteriorate substantially while
every value stays inside "normal") and **absolute crossing** of an age-appropriate
threshold (no single step large enough to read as a trend, but the patient is now in
shock *for their age*).

Moving a patient to a bed ends the wait, not the watch: **`SENT_FOR_TREATMENT`** and
**`STABILISED_ON_RECHECK`** take them off the wait-time clock but keep the vitals
watch running, because a monitored bed is not immunity from deterioration. Only
`DISCHARGED`, `TRANSFERRED` and `OTHER` close the record entirely.

The board is **capped at what one person can act on**. In the demo's surge, dozens
of alerts are raised and only a handful are shown. An uncapped board during a surge
is a board nobody reads, and what gets cut is always the least urgent.

**Surge is proposed by the system and declared by a human.** The department can
notice that volume has tripled; switching the objective function of a triage
department is a rationing decision with clinical and ethical weight, and it belongs
to the person in charge. Under surge the cost matrix narrows the gap between the two
errors — and the red-flag floor does not move at all, because nothing about a busy
department makes a cardiac arrest less urgent.

---

## 7. Designing for the worst case, not the average

| Worst case | Design response |
|---|---|
| Model is down or slow | Degraded mode: rule layer runs alone. A 500 ms latency budget, and exceeding it is reported rather than hidden. `assess()` never raises. |
| Silent deterioration in the waiting room | The waiting-room monitor — the single highest-value component in the system. |
| Missing vitals (uncooperative, paediatric, language barrier) | Never silently imputed. Missingness becomes a feature, is shown as a completeness indicator, and two-or-more-missing is itself a red flag (R23). |
| Vitals that lie | Beta-blockers mask tachycardia; children compensate until they crash. Age- and medication-adjusted thresholds are explicit, not left to a single global cutoff. |
| Automation bias | Confidence and abstention are always shown. The acuity field is never pre-filled — the nurse sets it, the model sits beside it. |
| Demographic bias | Undertriage rates are measured per stratum as a release gate, not a report appendix — see [EVALUATION.md](EVALUATION.md) §5, where this is currently failing. |
| Mass casualty / surge | A department-declared surge mode narrows the undertriage/overtriage cost gap; the red-flag floor is untouched. Not automatic — see §6. |

Two items are named here deliberately as **not yet built**, because a worst-case
table that only lists solved problems is not honest: a formal mass-casualty
objective switch (SALT/START-style resource allocation under absolute scarcity) and
automated drift monitoring on live inputs and outputs. Both are natural next steps
once the system has a live deployment to monitor.

---

## 8. Evaluation approach — undertriage is not symmetric with overtriage

Primary metrics:

- **Undertriage rate**: share of true ESI 1–2 predicted as 3–5 — the number the
  whole project is optimised against.
- **Sensitivity for the critical composite** (ICU transfer, death, or a critical
  intervention within 4 hours) at a fixed nurse-alert budget, because alert fatigue
  is a real failure mode and an unbounded alert count is not a deployable one.
- **Calibration**: expected calibration error and reliability curves, per acuity
  band.

Secondary: overtriage rate (resource cost), AUROC/AUPRC, and a discrete-event
simulation of the ED queue replayed on historical arrivals, comparing nurse-only
ordering against assisted ordering — this is how a claim about wait-time impact gets
evidence instead of assertion.

The loss function is cost-sensitive, with undertriage weighted several times
overtriage; the exact ratio is not picked for a nice-looking F1, it is set against a
department's real escalation capacity (§5). Validation uses a **temporal split**
(train on earlier data, test on later) rather than a random split, because a random
split leaks seasonality and flatters the model. Full results:
[EVALUATION.md](EVALUATION.md).

---

## 9. Patient data protection

**HIPAA assumed; GDPR implemented alongside.** They disagree about things that
change the code, and a single "privacy module" that ignores the difference is
compliant nowhere.

| | HIPAA | GDPR |
|---|---|---|
| Lawful basis | Treatment operations, 45 CFR 164.506 | Art. 9(2)(h) healthcare provision |
| Audit retention | 6 years (45 CFR 164.316) | 10 years |
| Training corpus | 3 years | 2 years — storage limitation bites hardest on secondary use |
| Right to erasure | Not applicable | Yes, but **cannot reach the audit trail** (Art. 17(3)) |
| DPIA required | No | Yes |

An unrecognised jurisdiction defaults to the **stricter** regime: needlessly strict
is a deployment inconvenience, accidentally lax is a notifiable breach.

**The model never sees identity.** Nothing in the feature set is a name, address or
record number. `scoring_projection()` returns exactly what the model receives, so a
reviewer can confirm minimum-necessary themselves rather than trusting a diagram.
Pseudonymisation is **keyed (HMAC)**, not a bare hash — a plain SHA-256 of a medical
record number is reversible by enumeration.

**Access control denies what is not needed.** Data scientists cannot view identified
patients; only a charge nurse may declare surge. The surest way to prevent a
re-identification incident is to not grant the access.

**GDPR Art. 22 is satisfied by the architecture, not a disclaimer.** The system
makes no solely-automated decision that significantly affects a patient, because a
licensed clinician sets every acuity. The design constraint *is* the legal basis.

---

## 10. Clinical accountability — overrides

An override is the system working. The capture path is one tap, a reason from a
short list, no free text required, no confirmation dialogue — friction here is what
makes staff work around a tool instead of through it. **Validation failures never
block**: the system records the decision and then complains about its own
incompleteness, never the other way round.

Every decision is logged, agreements included, because an override rate needs a
denominator. Three things come out of the same rows:

- **Accountability** — attributable to the clinician who made it, as both regimes
  require.
- **Training data** — a disagreement marks exactly where the model is wrong, on a
  real patient, judged by someone who saw them.
- **The trust signal** — an override rate collapsing toward zero usually means staff
  have stopped reading the recommendation, the failure mode that looks like success
  on every adoption dashboard.

The audit log is **hash-chained**: editing a record breaks every hash downstream,
which `verify()` detects. It does not prevent tampering — nothing local can — but it
makes it visible, which is what an investigation needs.

---

## 11. Scalability and integration

Three axes differ between departments, and each changes behaviour rather than
branding.

| | Rural general | District general | Urban trauma centre |
|---|---|---|---|
| Volume | 100/day | 250/day | 550/day |
| Integration | Manual entry | Read-only EHR | Bidirectional |
| Model tier | **Rules only** | Rules + model | Rules + model |
| Paediatrics | No — warns | Yes | Yes |
| Escalation budget | 8/shift (24%) | 16/shift (19%) | 32/shift (17%) |

**Budget scales with staffing, not volume** — the constraint is who does the
re-assessing. A department with one triage nurse cannot act on forty alerts however
many patients it sees.

**The rural site starts on rules alone, and that is the honest configuration**, not
a lesser product: no local validation cohort exists yet, and a site with one triage
nurse has no capacity to monitor a model for drift. The deterministic layer needs no
local data and has no drift.

**Integration degrades to a web form.** Most departments cannot offer bidirectional
EHR integration on day one, and a design that requires it does not deploy.

---

## 12. Adoption path

The tool has to earn its way onto the screen.

1. **Shadow** — runs silent beside the nurse for a period, recommending to nobody.
   Compare what it would have said to what happened.
2. **Watch board only** — the waiting-room list goes live first. It is purely
   additive, takes nothing away from anyone, and is where the evidence is strongest.
3. **Second opinion at triage** — only once the department trusts the board. Never
   pre-filled, never blocking, always beside the nurse's own field.

Throughout, **the escalation budget belongs to the charge nurse**, set against
tonight's capacity rather than fixed in a config file by whoever trained the model.

---

## 13. What the prototype demonstrates

20 simulated arrivals over an evening shift, each written to exercise a named
behaviour: 3 ambiguous presentations, 3 paediatric (neonate, toddler, adolescent —
three different physiologies), 5 geriatric, 10 with no prior record, 2 with no
obtainable vitals at all.

Every case carries a **pre-registered expectation** written before the system was
run, so the demo scores itself rather than being tuned to agree with its own
numbers:

| | |
|---|---|
| As expected | **13 / 20** |
| More acute than expected (safe side) | 3 / 20 |
| **Missed** | **4 / 20** |

The misses are the useful part, and they are reported rather than tuned away:

- **ED-003** (ambiguous, "just feels unwell") — the system confidently said *can
  wait*. She then deteriorated in the waiting room and was escalated by the monitor
  35 minutes later on repeat observations. This is the whole thesis of the project
  in one patient: the triage-time judgement was wrong, and the continuous one caught
  it.
- **ED-014** (sickle cell crisis, 10/10 pain, normal vitals) — undertriaged to level
  3. This is documented pain-assessment bias appearing in this system, and it is a
  release blocker of exactly the kind §5 of [EVALUATION.md](EVALUATION.md)
  identified.
- **ED-018** (DKA) and **ED-020** (language barrier) — abstained where a level was
  expected.

A demo that agrees with itself on every case demonstrates nothing.

The browser console (`python -m patienttriage.ui.server`) puts all of this in front
of a nurse rather than a terminal: intake, a recommended destination, a live
waiting-room board split between who's still waiting and who's already in a
monitored bed, and a command centre showing real session statistics — nothing on
that screen is fabricated to look busier than the session actually is.

---

## 14. Status and honest limits

**Not deployable as it stands.** Evaluation found paediatric undertriage at 23.1%
against 5.7% for adults, and walk-in patients undertriaged 12× more than ambulance
arrivals. The age-stratification work in §4 is aimed directly at the first; neither
number has been re-measured against a fresh cohort since, and both remain release
gates.

The acuity model **matches rather than beats** the triage nurse (AUROC 0.783 vs
0.789). The case for the system was never that it judges better than a nurse — it is
that it judges *again*, for everyone, all shift. The waiting-room monitor is the
part that does something no nurse has time to do.

The largest remaining data gap is unchanged: the validation dataset carries no
free-text chief complaint, which bounds the model and leaves the majority of the
complaint-reading rules with nothing to read during evaluation. Credentialed access
to a dataset with genuine free text (MIMIC-IV-ED) is the critical path to closing
that gap.

Full evaluation results: [EVALUATION.md](EVALUATION.md).
