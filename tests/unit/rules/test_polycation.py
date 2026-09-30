"""/.2 (a phase): symmetric bis-quaternary-ammonium dications.

C[N+](C)(C)CCCCCC[N+](C)(C)C (hexamethonium core) used to abstain
('unknown organic compound') because route_charged fails closed on EVERY
multi-cation shape except a homogeneous poly-aminium (charged_router.py
~1289-1302, "poly-quaternary... FAILS CLOSED here"), and the fallback
_try_neutralize_and_name over-valences the quaternary N (dropping its +
charge with no compensating bond removal), matching the observed RDKit
"Explicit valence for atom # N, 4, is greater than permitted".

emit_bis_quaternary_ammonium (rules/ions.py) closes this ONE narrow shape:
exactly 2 quaternary (0-H, degree-4, +1, acyclic) N cations joined by a
single unbranched, saturated, unsubstituted all-carbon bridge, with
BYTE-IDENTICAL onium substituent sets on both ends -> named
'{bridge-diyl}bis({onium unit})' (BB, cf.
'(1,4-phenylene)bis(phosphanium)'). Everything else (asymmetric,
>2 cations, ring-borne, branched/heteroatom linker) fails closed ('').
"""
import pytest
from rdkit import Chem
from orthonym import Orthonym
from orthonym.rules.ions import emit_bis_quaternary_ammonium
from orthonym.perception.ions import get_ion_sites
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "C[N+](C)(C)CCCC(=O)[O-]",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_obj_name(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES:
        return _declined_pin_row(smiles)["name"]
    return namer_obj.name(smiles)


def _dt_obj_row(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES:
        return _declined_pin_row(smiles)
    return namer_obj.name_tiered(smiles)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)



def _sites(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol, get_ion_sites(mol)['cations']


def test_hexamethonium_unit():
    mol, cations = _sites("C[N+](C)(C)CCCCCC[N+](C)(C)C")
    assert emit_bis_quaternary_ammonium(mol, cations) == \
        "hexane-1,6-diylbis(trimethylazanium)"


def test_decamethonium_unit():
    mol, cations = _sites("C[N+](C)(C)CCCCCCCCCC[N+](C)(C)C")
    assert emit_bis_quaternary_ammonium(mol, cations) == \
        "decane-1,10-diylbis(trimethylazanium)"


def test_short_ethylene_bridge_unit():
    # scope naturally extends to any straight saturated bridge, not just the
    # two named examples (n=2 minimum enforced -> ethane-1,2-diyl)
    mol, cations = _sites("C[N+](C)(C)CC[N+](C)(C)C")
    assert emit_bis_quaternary_ammonium(mol, cations) == \
        "ethane-1,2-diylbis(trimethylazanium)"


def test_asymmetric_dication_failclosed():
    # different N-substituents at each end -> the two onium units are not
    # identical -> fail closed, never a wrong symmetric name
    mol, cations = _sites("C[N+](C)(C)CCCCCC[N+](C)(CC)CC")
    assert emit_bis_quaternary_ammonium(mol, cations) == ''


def test_wrong_cation_count_failclosed():
    mol, cations = _sites("C[N+](C)(C)C")
    assert emit_bis_quaternary_ammonium(mol, cations) == ''


def test_branched_linker_failclosed():
    # a methyl branch on the bridge breaks the "unbranched" scope requirement
    mol, cations = _sites("C[N+](C)(C)CC(C)CCC[N+](C)(C)C")
    assert emit_bis_quaternary_ammonium(mol, cations) == ''


def test_ring_borne_cation_failclosed():
    # two ring-borne quaternary N+ (piperidinium-like) joined by a hexyl
    # bridge -- both cations are otherwise "quaternary", but IN A RING is
    # out of scope for this narrow acyclic-linker builder.
    mol, cations = _sites("C[N+]1(CCCCCC[N+]2(C)CCCCC2)CCCCC1")
    assert emit_bis_quaternary_ammonium(mol, cations) == ''


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# The PIN is the SUBSTITUTIVE '-bis(aminium)' name (emit_bis_quaternary_aminium):
# names a carbon-substituted N+ by the 'aminium' suffix
# ('N,N,N-trimethylmethanaminium (PIN)', the Blue Book, not
# 'tetramethylazanium':41354); 'N1,N1,N3,N3,N3-hexamethylpropane-
# bis(amidium) (PIN)' (:42154) and 'butanebis(nitrilium) (PIN)' beside the non-PIN
# 'butanediylidynebis(azanium)' (:42160-42162); '3-(azaniumylmethyl)pentane-
# 1,5-bis(aminium) (PIN)' (:42366). The multiplicative builder above stays as the
# verified general-tier fallback.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("C[N+](C)(C)CCCCCC[N+](C)(C)C",
     "N1,N1,N1,N6,N6,N6-hexamethylhexane-1,6-bis(aminium)"),
    ("C[N+](C)(C)CCCCCCCCCC[N+](C)(C)C",
     "N1,N1,N1,N10,N10,N10-hexamethyldecane-1,10-bis(aminium)"),
])
def test_integration_bis_quaternary_ammonium(namer, smi, expected):
    assert _dt_obj_name(namer, smi) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("C[N+](C)(C)C", "N,N,N-trimethylmethanaminium"),
    ("C[n+]1ccccc1", "1-methylpyridin-1-ium"),
    ("C[N+](C)(C)CCCC(=O)[O-]", "4-(trimethylazaniumyl)butanoate"),
])
def test_integration_regressions_unchanged(namer, smi, expected):
    assert _dt_obj_name(namer, smi) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", [
    "C[N+](C)(C)CCCCCC[N+](C)(CC)CC",           # asymmetric onium units
    "C[N+](C)(C)CC[N+](C)(C)CC[N+](C)(C)C",     # 3 cations, out of scope
])
def test_integration_failclosed_never_wrong(namer, smi):
    # honest fail: abstain rather than a wrong symmetric-looking bis(...) name
    # for a molecule whose cationic centres are not actually a clean
    # symmetric pair. Both verified (pre-fix) to abstain to this sentinel.
    out = _dt_obj_name(namer, smi)
    assert out == "unknown organic compound"
    assert "diylbis(" not in out


@pytest.mark.opsin_gate
def test_mixed_protonation_dication_fails_closed(namer):
    """One NH3+ (protonated amine) + one N+(C)(C) quaternary -> mixed protonation.

    Scope requires IDENTICAL onium units (both must be quaternary N+);
    mixing a protonated amine with a quaternary N+ fails closed.
    """
    out = _dt_obj_name(namer, "[NH3+]CCCCCC[N+](C)(C)C")
    assert out == "unknown organic compound"


@pytest.mark.opsin_gate
def test_mixed_onium_dication_fails_closed(namer):
    """N+ (pyridinium-like) + S+ (sulfonium-like) -> mixed onium types.

    Scope requires both cations to be identical quaternary N+;
    mixing N+ with S+ is out of scope, fails closed.
    """
    out = _dt_obj_name(namer, "C[N+](C)(C)CCCCCC[S+](C)C")
    assert out == "unknown organic compound"
