# PatientTriage.ai

AI decision-support for emergency department triage. It gives the triage nurse a second
opinion at the door, and then keeps re-scoring the waiting room she has to walk away from.

**The system may raise a patient's acuity. It may never lower one a human has set, and it
never assigns a final acuity on its own.** That invariant is enforced in code and locked
by tests, not just described in the docs.

- [docs/SOLUTION.md](docs/SOLUTION.md) — the full solution: architecture, the
  decide/recommend boundary, data strategy, age-stratified scoring, waiting-room
  monitoring, governance, scalability, and what the prototype shows
- [docs/EVALUATION.md](docs/EVALUATION.md) — what the evaluation actually found,
  including the results that argue against deploying it

## Quick start

Requires Python 3.12 (3.13+ has no LightGBM wheel yet).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Fetch the data — CDC NHAMCS Emergency Department public-use files, no credentialing:

```bash
mkdir -p data/raw
BASE=https://ftp.cdc.gov/pub/Health_Statistics/NCHS
for y in 2021 2022; do
  curl -o data/raw/ed$y.zip      $BASE/Datasets/NHAMCS/ed$y.zip
  curl -o data/raw/eddict$y.dct  $BASE/dataset_documentation/nhamcs/stata/eddict$y.dct
done
```

Then:

```bash
python -m patienttriage.demo.run       # the prototype: a full shift, 20 patients
python -m patienttriage.models.train   # train, evaluate, fairness audit
python -m patienttriage.sim.run        # queue simulation
pytest -q                              # 183 tests
```

The demo runs without the NHAMCS download too, on the deterministic rule layer alone —
that is the `RULES_ONLY` deployment tier a small site would start on, not a broken mode.
Try `--profile rural` and `--profile urban` to see the same assistant reconfigure.

### Browser console

On Windows, double-click **`PatientTriage.bat`** in the repository root. It creates the
virtual environment on first run, installs what the console needs, starts the server and
opens the browser. It takes the same options as the module: `PatientTriage.bat --profile
urban --port 8010`.

Otherwise, or to run it from an environment you already have:

```bash
uv pip install --python .venv/bin/python -e ".[ui]"
python -m patienttriage.ui.server            # opens http://127.0.0.1:8000
python -m patienttriage.ui.server --profile urban --port 8010
```

A FastAPI + vanilla-JS front end over the same `TriageAssistant` the CLI demo drives —
nothing clinical is re-implemented in the browser layer. A sidebar app shell, six views:

- **Triage intake** — a blank intake by default. Type or dictate the complaint; get
  the recommendation, its confidence, its drivers, a NEWS2 second opinion, and a
  recommended destination zone ([clinical/routing.py](src/patienttriage/clinical/routing.py)
  — acuity mapped to a bed type and target time, flagged when the site lacks the
  capability a patient needs, never a claim about which bed is physically free);
  record a nurse's decision; and place the patient on the waiting-room board at the
  acuity the nurse assigned.
- **Example cases** — the 20-case demo cohort as cards, kept on its own tab rather
  than mixed into the intake flow, because there is no example patient in a real ED.
  Clicking one loads it into Triage intake and switches there.
- **Bed routing** — the five zones from `clinical/routing.py`, each paired with how
  many real patients this session actually put there right now. No bed-by-bed
  occupancy is tracked or invented; the count is real, the floor plan is not.
- **Waiting room** — live, not scripted: it shows whoever a nurse has actually added,
  re-assessed against wait-time and worsening vitals in real time, with a visible and
  audible alert on a new escalation. **Resolve** ends a patient's wait, but not always
  their record: *sent for treatment* or *stabilised on recheck* moves them to a second
  "in treatment" list — off the wait-time clock, still swept for deterioration — while
  *discharged* or *transferred* closes the record entirely. `Load 20-patient demo
  scenario` seeds it with the scripted cohort — including the ED-003 storyline —
  anchored to real "now" rather than a fixed date, for showing the feature off without
  a real patient list yet.
- **Command center** — KPI cards, an acuity donut, and a bar chart, all computed over
  this session's own decisions (nothing fabricated), plus the hash-chained audit log
  every assessment, override, and waiting-room event writes to.
- **Site & governance** — the selected hospital profile's retention and access policy.
  Switching site profiles retrains in the background rather than blocking the page.

## How it fits together

```
arrival
  |
  |-- L0  rules/engine.py       24 red-flag rules, each with a citation.
  |                             Escalation only - returns an acuity FLOOR.
  |
  |-- L1  models/ordinal.py     Cumulative-link LightGBM over acuity 1-5,
  |                             isotonic-calibrated per threshold.
  |
  |-- L2  models/deterioration.py   P(dies in ED or needs critical care).
  |                                 Drives the waiting-room watch board.
  |
  |-- L3  models/conformal.py   Split conformal. Abstains when the prediction set
  |                             straddles the urgent/non-urgent boundary.
  |
  |-- L4  models/explain.py     SHAP selects the facts; the phrasing is written
  |                             against the feature so a nurse can disagree with it.
  |
  '-- L5  service/pipeline.py   Assembles all of it. Never raises; degrades to L0.
          service/audit.py      Hash-chained, append-only decision log.
          service/override.py   Clinician overrides — accountability and training data.

after triage, continuously
  |
  '--    monitoring/           Re-assesses everyone still waiting: wait-time breach,
                               worsening repeat observations, unobserved patients.

cross-cutting
       clinical/agebands.py    Every age-dependent threshold, in one place.
       config/profile.py       Per-site scale, integration tier, escalation budget.
       compliance/             Jurisdiction, retention, pseudonymisation, access control.
       demo/run.py             A full-shift prototype demonstration, end to end.
```

## Three things worth knowing before you read the code

**`features/schema.py` is the contract.** Every field declares *when it becomes knowable*.
`assert_triage_time_only` rejects any feature matrix containing something that does not
exist yet at prediction time — labs, imaging, the physician's note. Most published triage
models look strong because they quietly train on data that arrives later; this refuses to.

**`models/decision.py` holds the value judgement.** Undertriage costs more than
overtriage, and that number lives alone in one readable file rather than smeared through
a loss function — so a department can set it, and `eval/operating_point.py` can sweep it
against real capacity instead of picking it in the abstract.

**`clinical/agebands.py` is why one model can serve a neonate and a pensioner.** Every
age-dependent threshold reads from here. A fever of 38.5 °C means something different at
3 and at 75, and the system says which — rather than applying one adult-calibrated scale
to everybody, which is a failure that reads as normal on every dashboard.

## Status

Runs end to end on 32,232 real visits, trained on 2021 and tested on 2022. **Not
deployable.** Paediatric undertriage is 23.1% against 5.7% for adults, and walk-in
patients are undertriaged 12× more than ambulance arrivals. Both are release gates.
See [docs/EVALUATION.md](docs/EVALUATION.md).
