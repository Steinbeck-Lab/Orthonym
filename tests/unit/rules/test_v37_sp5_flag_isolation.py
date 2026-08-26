"""v37 SP5.4 -- the load-bearing ISOLATION property for the
``--emit-tier full-coverage`` flag.

With the flag OFF (the default everywhere) behavior is byte-identical to
today, and merely adding the flag CHOICE + its plumbing must not change any
existing default-tier or D1-table output. This test file must NEVER regress
for the life of the D2 (flag-tier coordination additive namer) build -- it is
the guard that proves D2 is fully isolated from the 0-wrong default tier.
"""
import pytest

from orthonym import name_compound, Orthonym
from orthonym.cli import _emit_tier_flags, main


def test_default_tier_byte_identical_with_flag_plumbing_present():
    # Adding the flag CHOICE must not change any existing default-tier output.
    assert name_compound("CCO") == "ethanol"
    assert name_compound("c1ccccc1") == "benzene"


def test_full_coverage_flag_does_not_change_a_table_hit():
    # A D1 exact-InChIKey table hit is unaffected by the new flag existing
    # (default namer, flag OFF -- the table wins deterministically as before).
    from tests.unit.rules.test_d1_coordination_v36 import HEME_B
    assert Orthonym().name(HEME_B["smiles"]) == HEME_B["name"]


def test_emit_tier_full_coverage_maps_to_its_own_flag_set():
    # full-coverage is a valid choice mapping to its OWN flag set: a SUPERSET
    # of best-effort's production triple PLUS its own `full_coverage` marker
    # bit (never silently aliased to best-effort's triple).
    flags = _emit_tier_flags("full-coverage")
    assert isinstance(flags, dict)
    assert flags["general_fallback"] is True
    assert flags["general_fallback_unverified"] is True
    assert flags["allow_aromatic_general"] is True
    assert flags.get("full_coverage") is True
    # the four pre-existing tiers never carry the marker.
    for tier in ("pin", "valid", "complete", "best-effort"):
        assert _emit_tier_flags(tier).get("full_coverage") in (False, None), tier


def test_full_coverage_is_a_valid_cli_choice(capsys):
    # argparse must ACCEPT the new choice, and an ordinary organic still names
    # identically under the flag (isolation: flag on, non-macrocycle input ->
    # unchanged output).
    rc = main(["CCO", "--emit-tier", "full-coverage"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "ethanol"


def test_name_tiered_carries_verified_field_for_default_tier():
    # The tiered dict gains a `verified` field; an ordinary PIN emission that
    # round-trips through OPSIN is labelled "opsin" (or, with the suite's gate
    # disabled, "unverified" -- never a D2/identity label for a PIN organic).
    row = Orthonym().name_tiered("CCO")
    assert row["name"] == "ethanol"
    assert "verified" in row
    assert row["verified"] in ("opsin", "unverified")


def test_name_tiered_labels_a_d1_table_hit_identity():
    from tests.unit.rules.test_d1_coordination_v36 import HEME_B
    row = Orthonym().name_tiered(HEME_B["smiles"])
    assert row["name"] == HEME_B["name"]
    assert row["verified"] == "identity"
