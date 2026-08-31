"""Where a patient goes, given an acuity — separate from how urgently they're seen."""

from __future__ import annotations

from patienttriage.clinical.routing import recommend_destination
from patienttriage.config.profile import PRESETS


def test_each_acuity_gets_a_named_zone_and_the_shared_time_target():
    for acuity, target in [(1, 0), (2, 10), (3, 30), (4, 60), (5, 120)]:
        rec = recommend_destination(acuity)
        assert rec.zone
        assert rec.bed_type
        assert rec.target_minutes == target
        assert rec.note is None


def test_a_child_at_a_site_without_paediatric_service_gets_the_capability_note():
    rec = recommend_destination(2, age_years=6.0, profile=PRESETS["rural"])
    assert rec.note and "paediatric" in rec.note


def test_the_same_child_at_a_paediatric_capable_site_gets_no_note():
    rec = recommend_destination(2, age_years=6.0, profile=PRESETS["district"])
    assert rec.note is None


def test_pregnancy_without_an_obstetric_service_is_flagged():
    rec = recommend_destination(2, is_pregnant=True, profile=PRESETS["rural"])
    assert rec.note and "obstetric" in rec.note


def test_level_one_at_a_non_trauma_centre_names_the_transfer_pathway():
    rec = recommend_destination(1, profile=PRESETS["district"])
    assert rec.note and "trauma" in rec.note


def test_level_one_at_the_trauma_centre_has_nothing_to_flag():
    rec = recommend_destination(1, profile=PRESETS["urban"])
    assert rec.note is None


def test_an_unrecognised_acuity_falls_back_to_the_least_acute_zone():
    rec = recommend_destination(9)
    assert rec.zone == recommend_destination(5).zone
