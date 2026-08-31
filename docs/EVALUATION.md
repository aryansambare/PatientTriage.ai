# Findings — PatientTriage.ai on NHAMCS 2021/2022

Everything below comes from `python -m patienttriage.models.train` and
`python -m patienttriage.sim.run`, on CDC NHAMCS Emergency Department public-use
files. Train: 2021. Test: 2022. Temporal split, never random.

**Cohort.** 32,232 visits; 20,702 carry a usable acuity (IMMEDR 1–5). Test set is
10,207 labelled 2022 visits. 126 recorded values were discarded as physiologically
impossible (116 pulses, 8 respiratory rates, 2 diastolic pressures) and treated as
not measured. GCS is 100% missing — NHAMCS does not collect it, so every GCS-based
red-flag rule is inert on this dataset and untested by these numbers.

---

## 1. The headline: the model matches the nurse, and does not beat her

Ranking patients for the composite critical outcome (died in the department, or
admitted to critical care — 2.4% of arrivals):

| Ranking by | AUROC | Avg precision | Lift over base rate |
|---|---|---|---|
| Deterioration model (vitals, age, arrival mode) | 0.783 | 0.109 | 4.52× |
| **The triage nurse's own acuity** | **0.789** | 0.108 | 4.24× |

Within a 10%-of-arrivals alert budget, the model catches 40.7% of critical outcomes
and the nurse catches 40.8%. At a 2% budget the nurse is clearly better (18.1% vs
13.2%).

**This is the result, and it should not be dressed up as a win.** From triage vitals
alone the model reproduces the quality of a nurse's judgement — it does not exceed it.

The case for the system therefore is not "it is better than the nurse". It is that
the nurse makes this judgement **once, at the door, and never again**, while the model
can remake it every ten minutes for everyone in the waiting room. The value is in the
repetition, not the accuracy.

## 2. The acuity model is a mediocre mimic, and that is informative

Predicting the nurse's own level 1–2 assignment: **AUROC 0.743**. Calibration is good
(ECE 0.0093), so the weakness is discrimination, not probability quality.

The reason is visible in the data: NHAMCS has **no free-text chief complaint**, only
coded reasons for visit. A nurse assigns level 2 to chest pain largely *because it is
chest pain*, and that signal is not in the file. This is direct empirical support for
the MIMIC-IV-ED track — it is an observed ceiling, not an assumption.

The rule layer alone reaches only 15.2% sensitivity here, for the same reason: 14 of
its 21 rules read the complaint text, and there is none to read.

## 3. Choosing the cost ratio in the abstract produces an unusable system

An uncalibrated first choice set undertriage at 8× overtriage. On real data that recommends level 1–2 for
**52.8% of the department**:

| Undertriage cost | Flagged as 1–2 | Undertriage rate | Sensitivity | Precision |
|---|---|---|---|---|
| 1.5 | 12.7% | 73.5% | 26.5% | 35.4% |
| 3.0 | 30.5% | 41.5% | 58.5% | 32.6% |
| 8.0 | 52.8% | 19.1% | 80.9% | 26.0% |
| 12.0 | 71.7% | 9.0% | 91.0% | 21.6% |

A 5% undertriage rate is purchasable, and it costs 78% overtriage. No department can
resuscitate four fifths of its waiting room, so staff would learn within one shift to
click past the recommendation.

**The fix is to set the operating point against a real capacity.** Nurses assign level
1–2 to 17.0% of arrivals, so the department is already staffed for roughly that
volume. Within that budget the model catches 26.5% of level 1–2 patients. That is a
poor number, and it is the honest one.

## 4. Abstention works, and shows how little the model knows

Split conformal, calibrated on 2,624 visits held out from fitting entirely:

- target coverage 90%, **empirical coverage 90.9%** — the guarantee holds
- mean prediction set 2.40 levels
- **abstention rate 54.7%**

The system declines to score more than half of all patients. Given the signal
available without complaint text, that is the correct behaviour rather than a defect:
it is the model reporting, accurately, that it cannot place most patients.

## 5. Fairness gaps large enough to block release

| Stratum | Undertriage rate |
|---|---|
| Children | 23.1% |
| Adults | 5.7% |
| Older adults | 0.8% |
| Walk-in | 7.9% |
| Arrived by ambulance | 0.7% |

Children are undertriaged **4× more than adults**; walk-in patients **12× more than
ambulance arrivals**. The ambulance gap is mechanical — arrival mode is the single
highest-gain feature in the deterioration model — but that makes it worse, not better:
the model has largely learned to trust the paramedic, and a walk-in who is quietly
very unwell is exactly the patient this project exists to catch.

These are release gates. Neither number is acceptable for deployment.

## 6. The simulation: escalation budget flips the sign of the whole intervention

A week of arrivals, 9/hour, replayed through the same department twice — once ordered
by the nurses' acuity, once by the assisted ordering. 25 replications. The assisted
ordering **escalates only**: the nurse's level stands unless the model argues for
sooner.

At 115% utilisation (surge — the regime worth designing for), change in 90th-percentile
wait for patients who turned out to be critically unwell:

| Assisted ordering | Escalates | Critical p90 | Everyone else, p90 |
|---|---|---|---|
| Acuity model | 30.3% of arrivals | **+631.6 min (far worse)** | −2,445 min |
| Watch board, top 5% risk | 7.4% of arrivals | **−37.0 min (better)** | +162 min |

**This is the most useful thing the project has produced.** Escalating 30% of arrivals
into the priority band dilutes it so thoroughly that critically unwell patients wait
*ten hours longer* than if the system had done nothing at all. The same architecture,
with a 7.4% budget, helps.

The size of the escalation budget is not a tuning parameter. It is the difference
between a system that helps and one that kills people, and it is invisible to every
metric in sections 1–5.

At 85% utilisation the department barely queues at all and no ordering changes
anything. Re-ordering is zero-sum at fixed capacity: it can only ever move waiting
from one group to another.

*Caveat: ~41 critical patients per replication. The dilution effect is large and
robust; the watch board's benefit is small and within noise.*

## 7. Two things in the data worth reporting on their own

**The nonurgent bucket is not empty.** Critical-outcome rate by the nurse's assigned
level: 30.9% (L1), 7.6% (L2), 1.6% (L3), 0.25% (L4), **0.65% (L5)**. Level 5 carries
*more* risk than level 4 — a small group of genuinely sick patients is being sorted
into "nonurgent". That is the undertriage population, visible in national data.

**Acuity barely translates into priority.** Median wait to be seen: 11 minutes for
level 1, 17 minutes for level 5. Six minutes separates the most and least acute
patients in the department. Whatever the triage scale is achieving, it is not
sequencing.

---

## What this means for the build

1. **The text branch is not optional.** Sections 2 and 5 are both bounded by the
   absence of chief-complaint text. MIMIC-IV-ED credentialing is the critical path.
2. **Lead with the watch board, not the acuity assistant.** The acuity model does not
   beat the nurse and actively harms the queue when it escalates freely. The
   deterioration model at a tight budget is where the evidence points.
3. **The escalation budget belongs in the interface**, set by the charge nurse against
   the department's actual capacity, not fixed in a config file.
4. **Do not deploy on these numbers.** Paediatric and walk-in undertriage are
   disqualifying, and shadow mode is the only honest next step.
