"""Aminium/betaine zwitterions with a MULTI-carboxylate (poly-anion) parent
(v33 Phase 3).

Generalizes the P-74.1.3 GUARD-4 zwitterion path
(`charged_router._route_zwitterion` -> `_name_polyacid_zwitterion`) from
exactly-one-anion to >= 2 carboxylate anions on one acyclic skeleton, so a
single cation riding on a poly-acid parent (the glutamate/aspartate
zwitterion ANION -- net charge -1, both carboxyls deprotonated, the amine
still protonated) gets a real name instead of abstaining
("unknown organic compound").

Root cause (measured, SPY): TWO stacked gaps, not the one hypothesized.
1. `detect_species_type` classifies a net-NONZERO mixed-sign fragment as
   `'ion'`, not `'zwitterion'` (its own docstring: "Zwitterion - net zero
   charge but has both + and - atoms"), so `assemble_ion_name`'s `'ion'`
   branch (composer.py) never called `name_zwitterion`/`route_charged` for
   this shape at all -- neither its "cations only" nor "anions only" branch
   matches a MIXED ion_sites, so it fell straight to the empty-string
   fallback.
2. Even if reached, `_route_zwitterion`'s exactly-one-anion scope check
   (`len(cations) != 1 or len(anions) != 1: return ''`) would still have
   declined the 2-carboxylate shape.

Both are fixed: composer.py routes a mixed cation+anion `'ion'` through
`name_zwitterion` (pure ADD, RT-gated); `_name_polyacid_zwitterion` is tried
BEFORE the exactly-one-anion guard for >= 2 carboxylate anions.

RT-gated (`@pytest.mark.opsin_gate`); the top-level SELF-01/OPSIN gate
(which also checks net-charge preservation) is the 0-wrong backstop.
"""
import pytest
from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# --- the 2 goal molecules (RT-verified via OPSIN 2026-08-17) --------------
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # Glutamate zwitterion anion: [NH3+]-CH(-COO-)-CH2-CH2-COO-.
    # Parent = pentanedioic acid (severed cation); attach locant = min(4, 2) = 2.
    ("[NH3+]C(CCC(=O)[O-])C(=O)[O-]", "2-azaniumylpentanedioate"),
    # Aspartate zwitterion anion: [NH3+]-CH(-COO-)-CH2-COO-.
    # Parent = butanedioic acid (severed cation); attach locant = min(3, 2) = 2.
    ("[NH3+]C(CC(=O)[O-])C(=O)[O-]", "2-azaniumylbutanedioate"),
])
def test_aminium_polyacid_zwitterion(namer, smi, expected):
    assert namer.name(smi) == expected


# --- regressions: untouched paths ----------------------------------------
@pytest.mark.opsin_gate
def test_carboxylate_betaine_unchanged(namer):
    # Single-anion GUARD-4 betaine path (P-74.1.3) is a completely separate
    # branch (len(anions) == 1) -- must stay byte-identical.
    assert namer.name("C[N+](C)(C)CCC(=O)[O-]") == "3-(trimethylazaniumyl)propanoate"


@pytest.mark.opsin_gate
def test_l_carnitine_unchanged(namer):
    assert namer.name("C[N+](C)(C)C[C@H](O)CC(=O)[O-]") == "L-carnitine"


@pytest.mark.opsin_gate
def test_l_serine_zwitterion_unchanged(namer):
    # Single-anion protonated-amine zwitterion -- the established
    # retained/neutral-form amino-acid path (D-06 defer), unaffected by the
    # new multi-anion branch (len(anions) == 1 here).
    assert namer.name("[NH3+][C@@H](CO)C(=O)[O-]") == "L-serine"


@pytest.mark.opsin_gate
def test_dicarboxylate_dianion_no_cation_unchanged(namer):
    # Pure poly-anion (no cation at all): route_charged never calls
    # _route_zwitterion when there is no cation site -- a completely
    # separate dispatch path, must stay byte-identical.
    assert namer.name("[O-]C(=O)CCC(=O)[O-]") == "butanedioate"


@pytest.mark.opsin_gate
def test_phase3_choline_sulfate_unchanged(namer):
    assert namer.name("C[N+](C)(C)CCOS(=O)(=O)[O-]") == "2-(trimethylazaniumyl)ethyl sulfate"


@pytest.mark.opsin_gate
def test_phase3_bis_quaternary_ammonium_unchanged(namer):
    assert namer.name("C[N+](C)(C)CCCCCC[N+](C)(C)C") == \
        "hexane-1,6-diylbis(trimethylazanium)"


@pytest.mark.opsin_gate
def test_phase3_sulfonatobenzoate_unchanged(namer):
    assert namer.name("[O-]C(=O)c1ccc(cc1)S(=O)(=O)[O-]") == "4-sulfonatobenzoate"


# --- fail-closed: out-of-scope shapes must never emit a wrong name --------
@pytest.mark.opsin_gate
def test_mixed_carboxylate_sulfonate_zwitterion_failclosed(namer):
    # 1 cation + 1 carboxylate anion + 1 sulfonate anion -- mixed anion
    # types are out of scope for the new poly-carboxylate branch (it only
    # fires when EVERY anion classifies 'carboxylate'). Abstain (or some
    # non-fabricated fallback) is acceptable; a WRONG name is not.
    from orthonym.errors import is_failure_name
    out = namer.name("[NH3+]C(CS(=O)(=O)[O-])C(=O)[O-]")
    assert out is not None
    assert is_failure_name(out), f"expected an honest abstain, got: {out!r}"


@pytest.mark.opsin_gate
def test_two_cations_failclosed(namer):
    # >1 cation is out of scope -- must decline rather than fabricate.
    from orthonym.errors import is_failure_name
    out = namer.name("[NH3+]C(CC(=O)[O-])C([NH3+])C(=O)[O-]")
    assert out is not None
    assert is_failure_name(out), f"expected an honest abstain, got: {out!r}"


def test_ring_cation_declines_the_new_branch_directly():
    # Unit-level (no JVM needed): a ring-borne cation must be declined by
    # _name_polyacid_zwitterion itself (P-74.1.2 is out of scope here), not
    # silently accepted with a wrong locant/prefix.
    from rdkit import Chem
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.charged_router import _name_polyacid_zwitterion

    smi = "C[n+]1ccc(CC(CC(=O)[O-])C(=O)[O-])cc1"
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    sites = get_ion_sites(mol)
    result = _name_polyacid_zwitterion(
        mol, sites['cations'], sites['anions'], 'pin')
    assert result == ''
