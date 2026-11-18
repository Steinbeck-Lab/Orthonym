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


# A component larger than the fresh-frame cap (15 heavy atoms) is retried only when it has a charged or radical atom
# outside C, N, O, S, P, Se and the halogens (``adducts._has_exotic_centre``): the producers of such a centre
# (stannylium, alumanylium) run for the caller's whole input only. A large component without one that fails nested
# fails alone too, and a retry would cost a whole search (2026-11-18 PubChem rerun-5 triage item 2b).
_LARGE_ROWS = [
    ("C1CCC(CC1)[Sn+](C2CCCCC2)C3CCCCC3.C1=CC=C2C(=C1)C=CC(=C2C(=O)O)[O-]",
     "1-carboxy-2-oxidonaphthalene—tricyclohexylstannylium (1/1)"),
    ("CC(C1=CC(=CC=C1)OC2=CC=CC=C2)C(=O)O[Al+]OC(=O)C(C)C3=CC(=CC=C3)OC4=CC=CC=C4.[OH-]",
     "bis{[2-(3-phenoxyphenyl)propanoyl]oxy}alumanylium—hydroxide (1/1)"),
]


@pytest.mark.parametrize("smiles,expected", _LARGE_ROWS)
def test_large_component_with_an_exotic_centre_is_retried(smiles, expected):
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"], res["is_pin"]) == (expected, "systematic_verified", False), res
    assert name_is_rt_exact(res["name"], smiles)


def test_without_the_exotic_centre_gate_the_large_component_declines(monkeypatch):
    """Mutation: the cap applied to every large component again, the stannylium mixture has no name."""
    from orthonym.rules import adducts
    monkeypatch.setattr(adducts, "_has_exotic_centre", lambda smiles: False)
    assert name_best_effort(_LARGE_ROWS[0][0])["name"] is None


@pytest.mark.parametrize("smiles,exotic", [
    ("C1CCC(CC1)[Sn+](C2CCCCC2)C3CCCCC3", True),
    ("CCCCCCCCCC(=O)O[Al+]OC(=O)CCCCCCCCC", True),
    ("CCCC[N+](CCCC)(CCCC)CCCC", False),                        # an ammonium: the nested producers name it
    ("CCCCCCCCCCCCCCCCCC(=O)[O-]", False),
    ("CCCCCCCCCCCCCCCCCC[Se+](C)C", False),
])
def test_exotic_centre(smiles, exotic):
    from orthonym.rules.adducts import _has_exotic_centre
    assert _has_exotic_centre(smiles) is exotic


def test_a_radical_centre_is_retried_only_where_opsin_reads_radical_adducts(monkeypatch):
    """A boron radical: no name of the mixture reads back unless the OPSIN in use reads an adduct with a radical
    component, so the retry is spent only then."""
    from orthonym import opsin_capabilities as caps
    from orthonym.rules.adducts import _has_exotic_centre
    for enabled in (False, True):
        monkeypatch.setattr(caps, "radical_adduct_enabled", lambda enabled=enabled: enabled)
        assert _has_exotic_centre("[B]c1ccc(S(=O)(=O)c2ccccc2)cc1") is enabled


@pytest.mark.parametrize("exotic", [False, True])
def test_a_large_plain_component_spends_no_retry(monkeypatch, exotic):
    """Spy: a large component without an exotic centre that the nested naming leaves without a name is not named
    again in the fresh frame; the same input with the gate forced open is (the trace's positive control)."""
    from orthonym.rules import adducts
    real = adducts._name_component_once
    retried = []

    def once(frag_smi, *a, **k):
        if adducts._heavy_atoms(frag_smi) > adducts._FRESH_FRAME_MAX_HEAVY_ATOMS:
            retried.append(getattr(adducts._FRESH_FRAME, "active", False))
            return None
        return real(frag_smi, *a, **k)
    monkeypatch.setattr(adducts, "_name_component_once", once)
    if exotic:
        monkeypatch.setattr(adducts, "_has_exotic_centre", lambda smiles: True)
    name_best_effort("CCCCCCCCCCCCCCCCCC(=O)O.O")
    assert retried and any(retried) is exotic, retried
