"""Discrete-event simulation of the emergency department queue.

The claim "reduces waiting times" is the easiest thing in this project to assert and
the hardest to earn. A model that ranks patients better does not by itself make anyone
wait less: the department has a fixed number of treatment spaces, and re-ordering a
queue is zero-sum. Every minute saved for one patient is a minute somebody else waits.

So the honest question is not "does the wait go down" — it is:

    Does the time-to-clinician for the patients who turn out to be critically
    unwell fall, and what does that cost the people it is taken from?

This module answers both, by replaying the same arrival stream through the same
department twice: once ordered by the nurses' own acuity, once by the assisted
ordering. Identical arrivals, identical service times, identical capacity. The only
thing that differs is the order, which is the only thing the system changes.

What this does not model, and should not be read as modelling: any effect on how long
treatment takes, on staffing, on boarding, or on patients who deteriorate while
waiting. It is a queueing argument, not a clinical trial.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import simpy


@dataclass(frozen=True)
class Arrival:
    patient_id: str
    minute: float
    service_minutes: float
    nurse_priority: int
    assisted_priority: int
    critical: bool


def build_arrival_stream(
    acuity: np.ndarray,
    assisted: np.ndarray,
    critical: np.ndarray,
    service_minutes: np.ndarray,
    arrivals_per_hour: float,
    hours: int,
    rng: np.random.Generator,
) -> list[Arrival]:
    """A Poisson arrival stream drawing patients from the observed cohort.

    Patients are sampled with replacement from the test cohort, so the mix of acuity,
    outcome and service time matches the real one. Arrival *times* are simulated
    because NHAMCS deliberately withholds visit dates; the comparison is between two
    orderings of the same stream, so the stream only needs to be realistic in shape.
    """
    n_arrivals = rng.poisson(arrivals_per_hour * hours)
    picks = rng.integers(0, len(acuity), size=n_arrivals)
    minutes = np.sort(rng.uniform(0, hours * 60, size=n_arrivals))

    return [
        Arrival(
            patient_id=f"sim-{i}",
            minute=float(minutes[i]),
            service_minutes=float(service_minutes[p]),
            nurse_priority=int(acuity[p]),
            assisted_priority=int(assisted[p]),
            critical=bool(critical[p]),
        )
        for i, p in enumerate(picks)
    ]


def simulate(
    arrivals: list[Arrival], n_clinicians: int, use_assisted: bool
) -> pd.DataFrame:
    """Run the stream through a department with `n_clinicians` treatment spaces."""
    env = simpy.Environment()
    clinicians = simpy.PriorityResource(env, capacity=n_clinicians)
    results: list[dict[str, float | bool | str]] = []

    def patient(arrival: Arrival) -> object:
        yield env.timeout(arrival.minute - env.now)
        started_waiting = env.now
        priority = arrival.assisted_priority if use_assisted else arrival.nurse_priority

        # simpy treats lower priority values as more urgent, which happens to match the
        # acuity scale: level 1 is the most acute. No inversion needed, but the
        # coincidence is worth naming so nobody "fixes" it later.
        with clinicians.request(priority=priority) as slot:
            yield slot
            waited = env.now - started_waiting
            results.append(
                {
                    "patient_id": arrival.patient_id,
                    "wait_minutes": waited,
                    "nurse_priority": arrival.nurse_priority,
                    "assisted_priority": arrival.assisted_priority,
                    "critical": arrival.critical,
                }
            )
            yield env.timeout(arrival.service_minutes)

    for arrival in arrivals:
        env.process(patient(arrival))
    env.run()

    return pd.DataFrame(results)


def compare(
    arrivals: list[Arrival], n_clinicians: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same patients, same capacity, two orderings."""
    return (
        simulate(arrivals, n_clinicians, use_assisted=False),
        simulate(arrivals, n_clinicians, use_assisted=True),
    )


def summarise(nurse: pd.DataFrame, assisted: pd.DataFrame) -> pd.DataFrame:
    """Who waited longer, who waited less, and by how much.

    Reported separately for the patients who turned out to be critically unwell and
    for everyone else, because a single average conceals exactly the trade being made.
    """
    rows = []
    for label, mask_fn in (
        ("critically unwell", lambda d: d["critical"]),
        ("everyone else", lambda d: ~d["critical"]),
        ("all patients", lambda d: pd.Series(True, index=d.index)),
    ):
        a, b = nurse[mask_fn(nurse)], assisted[mask_fn(assisted)]
        rows.append(
            {
                "group": label,
                "n": len(a),
                "nurse_median": a["wait_minutes"].median(),
                "assisted_median": b["wait_minutes"].median(),
                "nurse_p90": a["wait_minutes"].quantile(0.9),
                "assisted_p90": b["wait_minutes"].quantile(0.9),
                "median_change": b["wait_minutes"].median() - a["wait_minutes"].median(),
                "p90_change": b["wait_minutes"].quantile(0.9) - a["wait_minutes"].quantile(0.9),
            }
        )
    return pd.DataFrame(rows)
