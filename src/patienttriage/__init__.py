"""PatientTriage.ai — AI decision-support for emergency department triage.

Design invariant, enforced throughout this package: the system may raise a patient's
acuity but may never lower an acuity that a human has set, and may never assign a
final acuity on its own. See docs/SOLUTION.md section 1.
"""

__version__ = "0.1.0"
