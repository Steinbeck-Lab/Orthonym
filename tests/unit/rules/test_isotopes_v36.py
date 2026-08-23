"""v36 Milestone A3 -- isotope-descriptor PLACEMENT gaps (P-82.2.1 / P-82.2.2.1 /
P-82.6.3.2). The decorator (``rules/isotopes.py``) already exists and already
names simple isotopologues; this file covers three measured placement gaps
plus a regression guard for the working simple cases.

All assertions are OPSIN-round-trip-gated via ``opsin_roundtrip_check`` (PLAIN
OPSIN -- isotopes do not use the radical ``-r`` path). A gap that cannot be
made to round-trip stays abstained (xfail), never forced.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


def _name(smiles, tier="pin"):
    with jvm_slots(1, purpose="test-isotope"):
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles).get("name")


# ---------------------------------------------------------------------------
# Task 1 -- systematic-parent fallback (P-82.6.3.2) + multi-attachment-group
# placement, per-D-glycine
# ---------------------------------------------------------------------------

PER_D_GLYCINE = "[2H]OC(=O)C([2H])([2H])N([2H])[2H]"


def test_per_deuterated_glycine_systematic_parent():
    name = _name(PER_D_GLYCINE)
    assert name and name not in ("unknown organic compound", None), "abstained"
    assert opsin_roundtrip_check(PER_D_GLYCINE, name)["passed"], name
    # P-82.6.3.2: retained 'glycine' has no numbered positions for the label
    # -> systematic parent. The default systematic-STYLE skeleton itself
    # still keeps the retained "acetic acid" stem (measured 2026-08-23), so
    # this also asserts the P-82.6.3.2 fallback actually fired.
    assert "glycine" not in name.lower()
    assert "acetic" not in name.lower()


# ---------------------------------------------------------------------------
# Regression guard -- the working simple cases must not regress
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("[13CH3]CO", "(2-13C1)ethan-1-ol"),
        ("[2H]C([2H])([2H])O", "(2H3)methanol"),
    ],
)
def test_simple_isotopologues_unchanged(smiles, expected):
    name = _name(smiles)
    assert name == expected
    assert opsin_roundtrip_check(smiles, name)["passed"], name
