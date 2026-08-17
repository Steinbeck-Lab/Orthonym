"""P-73.5.1.1/.2 (v33 Phase 3): symmetric bis-quaternary-ammonium dications.

C[N+](C)(C)CCCCCC[N+](C)(C)C (hexamethonium core) used to abstain
('unknown organic compound') because route_charged fails closed on EVERY
multi-cation shape except a homogeneous poly-aminium (charged_router.py
~1289-1302, "poly-quaternary ... FAILS CLOSED here"), and the fallback
_try_neutralize_and_name over-valences the quaternary N (dropping its +
charge with no compensating bond removal), matching the observed RDKit
"Explicit valence for atom # N, 4, is greater than permitted".

emit_bis_quaternary_ammonium (rules/ions.py) closes this ONE narrow shape:
exactly 2 quaternary (0-H, degree-4, +1, acyclic) N cations joined by a
single unbranched, saturated, unsubstituted all-carbon bridge, with
BYTE-IDENTICAL onium substituent sets on both ends -> named
'{bridge-diyl}bis({onium unit})' (BB P-73.5.1.1, cf.
'(1,4-phenylene)bis(phosphanium)'). Everything else (asymmetric,
>2 cations, ring-borne, branched/heteroatom linker) fails closed ('').
"""
import pytest
from rdkit import Chem
from orthonym import Orthonym
from orthonym.rules.ions import emit_bis_quaternary_ammonium
from orthonym.perception.ions import get_ion_sites


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


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("C[N+](C)(C)CCCCCC[N+](C)(C)C", "hexane-1,6-diylbis(trimethylazanium)"),
    ("C[N+](C)(C)CCCCCCCCCC[N+](C)(C)C", "decane-1,10-diylbis(trimethylazanium)"),
])
def test_integration_bis_quaternary_ammonium(namer, smi, expected):
    assert namer.name(smi) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("C[N+](C)(C)C", "N,N,N-trimethylmethanaminium"),
    ("C[n+]1ccccc1", "1-methylpyridin-1-ium"),
    ("C[N+](C)(C)CCCC(=O)[O-]", "4-(trimethylazaniumyl)butanoate"),
])
def test_integration_regressions_unchanged(namer, smi, expected):
    assert namer.name(smi) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", [
    "C[N+](C)(C)CCCCCC[N+](C)(CC)CC",           # asymmetric onium units
    "C[N+](C)(C)CC[N+](C)(C)CC[N+](C)(C)C",     # 3 cations, out of scope
])
def test_integration_failclosed_never_wrong(namer, smi):
    # honest fail: abstain rather than a wrong symmetric-looking bis(...) name
    # for a molecule whose cationic centres are not actually a clean
    # symmetric pair. Both verified (pre-fix) to abstain to this sentinel.
    out = namer.name(smi)
    assert out == "unknown organic compound"
    assert "diylbis(" not in out


@pytest.mark.opsin_gate
def test_mixed_protonation_dication_fails_closed(namer):
    """One NH3+ (protonated amine) + one N+(C)(C) quaternary -> mixed protonation.

    Scope requires IDENTICAL onium units (both must be quaternary N+);
    mixing a protonated amine with a quaternary N+ fails closed.
    """
    out = namer.name("[NH3+]CCCCCC[N+](C)(C)C")
    assert out == "unknown organic compound"


@pytest.mark.opsin_gate
def test_mixed_onium_dication_fails_closed(namer):
    """N+ (pyridinium-like) + S+ (sulfonium-like) -> mixed onium types.

    Scope requires both cations to be identical quaternary N+;
    mixing N+ with S+ is out of scope, fails closed.
    """
    out = namer.name("C[N+](C)(C)CCCCCC[S+](C)C")
    assert out == "unknown organic compound"
