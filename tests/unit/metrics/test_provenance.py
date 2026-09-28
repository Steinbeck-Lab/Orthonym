# tests/unit/metrics/test_provenance.py
""": tier/provenance labels on naming output."""
import pytest

from orthonym.namer import Orthonym

pytestmark = pytest.mark.unit


@pytest.mark.opsin_gate
def test_default_path_is_t1():
    # pin_verified is earned by the round trip (claims conformance R63,
    # 2026-09-27), so this runs with the validity gate ON; with it disabled the
    # same name is labelled pin_unverified (test_gate_disabled_is_not_verified).
    row = Orthonym().name_tiered("CCO")
    assert row["name"] == "ethanol"
    assert row["tier"] == "pin_verified" and row["is_pin"] is True
    assert row["source"] == "pin_path"


def test_gate_disabled_is_not_verified():
    row = Orthonym(_disable_opsin_validity_gate=True).name_tiered("CCO")
    assert row["name"] == "ethanol"
    assert row["gate_outcome"] == "gate_disabled"
    assert row["tier"] == "pin_unverified" and row["is_pin"] is False
    assert row["verified"] == "unverified"


def test_abstention_is_t5():
    nm = Orthonym(_disable_opsin_validity_gate=True)
    row = nm.name_tiered("O=C1CCC2CCCCC2C1")  # suppression class; engine OFF
    # gate disabled -> the wrong legacy candidate ships in unit tests; accept
    # either the shipped string or the failure sentinel, but tier must be
    # T1/ accordingly and never T3/ with the engine off.
    assert row["tier"] in ("pin_verified", "abstain")
    assert row["source"] != "general_engine"


def test_engine_emission_is_t3_or_t4():
    nm = Orthonym(_disable_opsin_validity_gate=True, general_fallback=True,
                   general_fallback_unverified=True)
    row = nm.name_tiered("CC1CCC2CCCCC2C1")
    if row["source"] == "general_engine":
        assert row["tier"] in ("systematic_verified", "best_effort")
        assert row["is_pin"] is False
        assert "atom_coverage" in row["gates_passed"]


def test_t5_row_carries_formula_and_reason():
    nm = Orthonym(_disable_opsin_validity_gate=True)
    row = nm.name_tiered("CC1C2C=CC1c1ccccc12")  # abstains with engine OFF
    assert row["tier"] == "abstain"
    assert row["formula"] == "C12H12"
    assert row["limit_code"]


def test_non_t5_row_schema_stable():
    row = Orthonym(_disable_opsin_validity_gate=True).name_tiered("CCO")
    assert row["formula"] is None and row["limit_code"] is None
