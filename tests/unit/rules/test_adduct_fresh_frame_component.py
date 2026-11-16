"""A mixture component that the nested naming refuses only because it is a part of the caller's input is named once
more as a fresh outermost naming (``rules.adducts._name_component_once``, ``fragment_naming.name_in_fresh_frame``).

"Names are formed by citing the names of individual compounds" "Mixed organic – inorganic adducts",
the Blue Book): a component is a compound of its own. The whole name still ships only when OPSIN reads it back
to the drawing, and "preferred IUPAC names cannot be assigned to mixed adducts".
"""
import pytest

from orthonym.assembly import fragment_naming as fn
from orthonym.cli import _tiered_namer
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

_ROWS = [
    ("O.[O-]P=O", "phosphanonyloxidanide—water (1/1)"),
    ("C[Al+]C1CC1.C[Al+]C1CC1.[Br-].[Br-]", "cyclopropyl(methyl)alumanylium—bromide (2/2)"),
    ("N#C[NH-].[Na+].O", "cyanoazanide—sodium(1+)—water (1/1/1)"),
]


@pytest.mark.parametrize("smiles,expected", _ROWS)
def test_component_named_as_a_compound_of_its_own(smiles, expected):
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"], res["is_pin"]) == (expected, "systematic_verified", False), res
    assert name_is_rt_exact(res["name"], smiles)


# The component producers behind these rows build names the Blue Book does not use (an 'a' chain that ends on S or N,
#; an 'a' chain without carbon,: only the read-back and the tier are asserted here, not the
# spelling (roadmap lead).
_READ_BACK_ONLY = ["NS(=O)[O-].NCl"]


@pytest.mark.parametrize("smiles", _READ_BACK_ONLY)
def test_component_named_and_read_back(smiles):
    res = name_best_effort(smiles)
    assert res["name"] and (res["tier"], res["is_pin"]) == ("systematic_verified", False), res
    assert name_is_rt_exact(res["name"], smiles)


@pytest.mark.parametrize("smiles", [s for s, _ in _ROWS] + _READ_BACK_ONLY)
def test_pin_tier_declines(smiles):
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"


@pytest.mark.parametrize("smiles,expected", [
    ("CCO.O", "ethanol—water (1/1)"),
    ("CC(=O)[O-].[Na+]", "sodium acetate"),
])
def test_names_made_before_are_unchanged(smiles, expected):
    assert name_best_effort(smiles)["name"] == expected


def _reads_radical_adducts() -> bool:
    from orthonym import opsin_capabilities as caps
    try:
        return caps.radical_adduct_enabled()
    except Exception:
        return False


@pytest.mark.skipif(not _reads_radical_adducts(), reason="the OPSIN in use reads no adduct with a radical component")
@pytest.mark.parametrize("smiles", ["N.N.O([Si])[Si]", "[B][B].B(F)(F)F"])
def test_radical_component_named_as_a_compound_of_its_own(smiles):
    # radical parts: the last producer names them only where OPSIN reads adducts with a radical component; the
    # spelling ('a' chains without carbon) is not asserted, roadmap lead)
    res = name_best_effort(smiles)
    assert res["name"] and res["verified"] in ("opsin", "opsin_twin") and not res["is_pin"], res


def test_fresh_frame_keeps_the_callers_input_and_counts_inner_trips(monkeypatch):
    import orthonym.namer as nm
    seen = {}
    real = fn.name_in_fresh_frame

    def spy(f):
        seen["before"] = nm._caller_input()
        out = real(f)
        seen["after"] = nm._caller_input()
        return out
    monkeypatch.setattr(fn, "name_in_fresh_frame", spy)
    assert name_best_effort("O.[O-]P=O")["name"] == "phosphanonyloxidanide—water (1/1)"
    assert seen and seen["before"] == seen["after"] and seen["before"] is not None, seen
    before = fn.hang_budget_trips()
    fn.name_in_fresh_frame(fn._count_hang_budget_trip)
    assert fn.hang_budget_trips() == before + 1


def test_fresh_frame_restores_the_naming_state():
    fn._fragment_guard.probe_attr = "outer"
    try:
        seen = fn.name_in_fresh_frame(lambda: getattr(fn._fragment_guard, "probe_attr", None))
        assert seen is None
        assert fn._fragment_guard.probe_attr == "outer"
    finally:
        del fn._fragment_guard.probe_attr
