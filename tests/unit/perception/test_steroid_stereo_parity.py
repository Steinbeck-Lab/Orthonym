"""Inverse-parity sign-convention tripwire for steroid α/β derivation (a phase, -02).

Asserts `alpha_beta_at` returns the expected ring-face descriptor for the 6 pinned
gold structures (internal notes Pattern 1 evidence). This is the empirical sign-convention
tripwire : the `+1 → beta, -1 → alpha` mapping is pinned against this RDKit build;
if a future RDKit upgrade flips neighbour-ordering semantics, these assertions fail loudly
and `_SIGN` is flipped ONCE globally — never per-molecule.

WAVE 0 CONTRACT: imports the not-yet-built `alpha_beta_at` / `collect_steroid_alpha_beta`
symbols INSIDE each test body (NOT at module level) so `pytest --collect-only` succeeds;
RED at run time until Wave 1 builds steroid_stereo.py.
"""

import pytest
from rdkit import Chem


def _mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"bad test SMILES: {smiles}"
    return m


def _wiring(info):
    """Build (ringorder, loc2idx, idx2loc) from the detected scaffold using existing helpers.

    Mirrors steroid_stereo._ring_wiring but stays test-local so the parity test exercises
    the deriver primitive directly rather than the citing filter.
    """
    from orthonym.data.opsin_imports.natural_products_opsin import OPSIN_NATURAL_PRODUCTS
    from orthonym.rules.natural_products import _build_target_to_iupac

    scaffold_smiles = info["scaffold_smiles"]
    entry = OPSIN_NATURAL_PRODUCTS.get(scaffold_smiles)
    assert entry is not None, f"no OPSIN entry for scaffold {scaffold_smiles}"
    abo = entry.get("alphaBetaClockWiseAtomOrdering")
    assert abo is not None, f"no ABO for scaffold {scaffold_smiles}"
    ringorder = [int(x) for x in abo.split("/")]
    idx2loc = _build_target_to_iupac(info)
    assert idx2loc, "numbering map empty"
    loc2idx = {loc: idx for idx, loc in idx2loc.items()}
    return ringorder, loc2idx, idx2loc


# (smiles, {locant: expected α/β}, {locants that must be ABSENT/None})
PINNED = [
    ("CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)O",
     {5: "alpha", 3: "beta"}, ()),                                                   # row 1
    ("CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@@H](CC[C@]4(C)[C@H]3CC[C@]12C)O",
     {5: "alpha", 3: "alpha"}, ()),                                                  # row 2
    ("CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@@H]4C[C@@H](CC[C@]4(C)[C@H]3CC[C@]12C)O",
     {5: "beta", 3: "alpha"}, ()),                                                   # row 3
    ("O[C@@H]1C[C@@H]2CC[C@H]3[C@@H]4CC[C@@H]([C@@]4(C)CC[C@@H]3[C@]2(CC1)C)O",
     {5: "alpha", 3: "beta", 17: "beta"}, ()),                                       # row 4
    ("C[C@]12CC[C@H]3[C@@H](CC=C4C[C@@H](O)CC[C@]34C)[C@@H]1CCC2",
     {3: "beta"}, (5,)),                                                             # row 5 (Δ5 androstenol: C-5 sp2)
    ("CC([C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)O)O",
     {5: "alpha", 3: "beta"}, ()),                                                   # row 6
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected,absent", PINNED, ids=[f"pinned{i+1}" for i in range(len(PINNED))])
def test_alpha_beta_at_pinned(smiles, expected, absent):
    """alpha_beta_at returns the pinned α/β for each expected ring locant; None for sp2 (Δ5 C-5)."""
    from orthonym.rules.steroid_stereo import alpha_beta_at
    from orthonym.perception.natural_products import detect_natural_product
    from orthonym.perception.stereo import assign_stereochemistry

    mol = _mol(smiles)
    info = detect_natural_product(mol)
    assert info and info.get("scaffold_class") == "steroid", f"not routed as steroid: {smiles}"
    assign_stereochemistry(mol)
    ringorder, loc2idx, idx2loc = _wiring(info)
    for loc, want in expected.items():
        got = alpha_beta_at(mol, loc, loc2idx, idx2loc, ringorder)
        assert got == want, f"locant {loc}: expected {want}, got {got} ({smiles})"
    for loc in absent:
        got = alpha_beta_at(mol, loc, loc2idx, idx2loc, ringorder)
        assert got is None, f"locant {loc} should be None (sp2/aromatic), got {got} ({smiles})"


@pytest.mark.unit
def test_collect_returns_structured_result():
    """collect_steroid_alpha_beta returns {'ring_ab':..., 'side_rs':...} with 3/5 cited for row 1."""
    from orthonym.rules.steroid_stereo import collect_steroid_alpha_beta
    from orthonym.perception.natural_products import detect_natural_product
    from orthonym.rules.natural_products import _build_target_to_iupac

    mol = _mol("CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)O")
    info = detect_natural_product(mol)
    numbering = _build_target_to_iupac(info)
    result = collect_steroid_alpha_beta(mol, info, numbering)
    assert result is not None, "expected a structured result for a resolvable steroid"
    ring_ab = result["ring_ab"]
    assert ring_ab.get(3) == "beta" and ring_ab.get(5) == "alpha", ring_ab
