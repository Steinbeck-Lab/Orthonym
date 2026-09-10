""" Milestone A3 -- isotope-descriptor PLACEMENT gaps / /
. The decorator (``rules/isotopes.py``) already exists and already
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
# Task 1 -- systematic-parent fallback + multi-attachment-group
# placement, per-D-glycine
# ---------------------------------------------------------------------------

PER_D_GLYCINE = "[2H]OC(=O)C([2H])([2H])N([2H])[2H]"


def test_per_deuterated_glycine_systematic_parent():
    name = _name(PER_D_GLYCINE)
    assert name and name not in ("unknown organic compound", None), "abstained"
    assert opsin_roundtrip_check(PER_D_GLYCINE, name)["passed"], name
    #: retained 'glycine' has no numbered positions for the label
    # -> systematic parent. The default systematic-STYLE skeleton itself
    # still keeps the retained "acetic acid" stem (measured 2026-08-23), so
    # this also asserts the fallback actually fired.
    assert "glycine" not in name.lower()
    assert "acetic" not in name.lower()


# ---------------------------------------------------------------------------
# Task 2 -- ring single-label placement, 18F tracer
# ---------------------------------------------------------------------------
#
# VERIFIED 2026-08-23 (Task 0 trace): the DECORATOR'S PLACEMENT LOGIC already
# works generally for a single isotope label on a ring/parent atom that
# carries its own substituent locant -- see the 3 positive witnesses below,
# none of which needed any code change. This specific witness's skeleton
# carries a pseudoasymmetric ring-stereo descriptor, "(1r,3r)-", and
# OPSIN 2.9.0 cannot parse THAT at all -- verified directly:
# opsin_parse("(1r,3r)-1,3-difluorocyclobutane") -> None
# opsin_parse("(1R,3R)-1-amino-3-fluorocyclobutane-1-carboxylic acid") -> None
# opsin_parse("rel-(1R,3R)-...") -> None
# opsin_parse("cis-3-fluoro-1-aminocyclobutane-1-carboxylic acid") -> None
# opsin_parse("trans-...") -> None
# every stereo notation tried, even on the isotope-free skeleton and even on
# the simplest possible instance of the pattern. Placing the isotope
# descriptor correctly cannot make an unparseable base name round-trip: the
# SAME skeleton with 18F correctly inserted right before "fluoro" (offset
# already offered by the existing ``_insertion_offsets`` -- it is an alpha
# boundary) round-trips fine once the stereo descriptor is dropped from the
# comparison (see the 3 positive witnesses). So this is an OPSIN ring-stereo
# grammar limitation, orthogonal to isotope placement -- 0-wrong-safe
# abstain, not a placement bug for this module to fix.
F18_TRACER = "N[C@]1(C(=O)O)C[C@@H]([18F])C1"


@pytest.mark.xfail(
    reason=(
        "OPSIN 2.9.0 cannot parse the base skeleton's pseudoasymmetric ring-"
        "stereo descriptor '(1r,3r)-' AT ALL (verified with R,R / r,r / "
        "rel-(R,R) / cis- / trans- notations, even on the simplest instance "
        "'(1r,3r)-1,3-difluorocyclobutane' with no isotope). The isotope-"
        "descriptor placement itself is correct (offset before 'fluoro' "
        "round-trips once stereo is dropped from the comparison) -- this is "
        "an OPSIN ring-stereo grammar gap, not a placement defect. Abstain "
        "is 0-wrong-safe; see test_ring_single_label_placement_general below "
        "for the (already-working) general mechanism."
    ),
    strict=True,
)
def test_f18_ring_tracer():
    name = _name(F18_TRACER)
    assert name and name not in ("unknown organic compound", None), "abstained"
    assert opsin_roundtrip_check(F18_TRACER, name)["passed"], name


@pytest.mark.parametrize(
    "smiles",
    [
        "OC1CCC([2H])CC1",             # 4-D-cyclohexan-1-ol (D on an unlabelled ring position)
        "Clc1ccc([18F])cc1",           # 1-chloro-4-fluorobenzene, 18F-labelled (existing substituent locant)
        "NC1CCC([2H])(C(=O)O)CC1",     # 4-amino-1-D-cyclohexane-1-carboxylic acid
    ],
)
def test_ring_single_label_placement_general(smiles):
    """General coverage (no stereo wall): a single isotope label on a ring
    atom that already carries -- or sits next to -- its own substituent
    locant places correctly via the EXISTING single-descriptor enumeration.
    Confirms Task 2's placement mechanism is not broken; the F18 witness
    above fails for an unrelated (OPSIN grammar) reason.
    """
    name = _name(smiles)
    assert name and name not in ("unknown organic compound", None), "abstained"
    assert opsin_roundtrip_check(smiles, name)["passed"], name


# ---------------------------------------------------------------------------
# Task 3 -- salt / multi-fragment skeleton placement, 11C-choline
# ---------------------------------------------------------------------------

C11_CHOLINE = "C[N+](C)([11CH3])CCO.[Cl-]"


def test_c11_choline_salt_skeleton():
    name = _name(C11_CHOLINE)
    assert name and name not in ("unknown organic compound", None), "abstained"
    assert opsin_roundtrip_check(C11_CHOLINE, name)["passed"], name


# ---------------------------------------------------------------------------
# Regression guard -- the working simple cases must not regress
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("[13CH3]CO", "(2-13C)ethan-1-ol"),
        ("[2H]C([2H])([2H])O", "(2H3)methanol"),
    ],
)
def test_simple_isotopologues_unchanged(smiles, expected):
    name = _name(smiles)
    assert name == expected
    assert opsin_roundtrip_check(smiles, name)["passed"], name
