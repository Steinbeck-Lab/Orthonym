"""
Wave2 Tier 5b — located acyclic alkyl N-substituents WITH simple internal
substituents (P-29.2 / P-46.1.12 / P-14.5.2).

Reproduce-first: both master-plan witnesses were fail-closed 'unknown' at HEAD
(NOT the plan-claimed halogen double-count):

  CC(Br)C(C(C)Cl)NC(C)=O -> N-(2-bromo-4-chloropentan-3-yl)acetamide
  CC(CO)NC(C)=O          -> N-(1-hydroxypropan-2-yl)acetamide

Root cause: `_located_acyclic_alkyl_name` (the tier-1.8 located-alkyl deriver)
rejected ANY heteroatom in the fragment, and the polyfunctional acyclic path
is terminal-attachment-only, so heteroatom-bearing internally-attached
N-fragments had no producer.  Fix: the deriver's CHAIN stays all-carbon, but
degree-1 halogens and hydroxyl oxygens are now allowed as BRANCHES (named by
the shared substituent namer: bromo/chloro/hydroxy).  Everything else (ethers,
amino, carbonyl — the intra-fragment double-bond check, charges) still
declines fail-closed.

Locant rules exercised: free valence lowest (P-46.1.8); direction tie broken
by lowest branch-locant set at first point of difference (P-29.4.1, shipped
T3c); NEW final tie-break — equal locant sets assign the lowest locant to the
substituent cited first in alphanumerical order (P-14.5.2(e)): the Br/Cl
pentan-3-yl witness must number bromo=2, never '4-bromo-2-chloro...'.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.assembly.substituent_naming import _located_acyclic_alkyl_name


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CC(Br)C(C(C)Cl)NC(C)=O", "N-(2-bromo-4-chloropentan-3-yl)acetamide"),
        ("CC(CO)NC(C)=O", "N-(1-hydroxypropan-2-yl)acetamide"),
        ("CC(C)(CO)NC(C)=O", "N-(1-hydroxy-2-methylpropan-2-yl)acetamide"),
    ],
)
def test_tier5b_substituted_n_alkyl_amides(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # shipped located-alkyl behavior must not move (SEN-04 / E2 golds)
        ("CC(=O)NC(C)CCCC", "N-(hexan-2-yl)acetamide"),
        ("CCC(CC)NC(C)=O", "N-(pentan-3-yl)acetamide"),
    ],
)
def test_tier5b_regression_controls(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# Deriver-level behavior.
# ---------------------------------------------------------------------------

def _fragment(smiles_with_attach, attach_smarts_idx=0):
    """Build (mol, sub_atoms, attach_idx) for a free-standing fragment where
    atom 0 of the SMILES is the attachment atom."""
    mol = Chem.MolFromSmiles(smiles_with_attach)
    return mol, list(range(mol.GetNumAtoms())), attach_smarts_idx


@pytest.mark.unit
def test_tier5b_deriver_bromo_chloro_pentanyl():
    # C(attach)(C(C)Br)(C(C)Cl) — pentan-3-yl skeleton, Br and Cl on the 2/4
    # carbons: the alphanumerical tie-break gives bromo locant 2.
    mol = Chem.MolFromSmiles("C(C(C)Br)C(C)Cl")
    # attachment = atom 0 (the central CH)
    res = _located_acyclic_alkyl_name(mol, list(range(mol.GetNumAtoms())), 0)
    assert res is not None
    name, k = res
    assert name == "2-bromo-4-chloropentan-3-yl"
    assert k == 3


@pytest.mark.unit
def test_tier5b_deriver_hydroxy_propanyl():
    mol = Chem.MolFromSmiles("C(C)CO")
    res = _located_acyclic_alkyl_name(mol, list(range(mol.GetNumAtoms())), 0)
    assert res is not None
    name, k = res
    assert name == "1-hydroxypropan-2-yl"
    assert k == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,attach,why",
    [
        ("C(C)OC", 0, "ether oxygen (degree 2) — not a simple branch"),
        ("C(C)CN", 0, "amino branch — out of scope, must decline"),
        ("C(C)C=O", 0, "carbonyl — unsaturated bond inside fragment"),
        ("C(C)C(=O)O", 0, "carboxyl — must decline"),
        ("C(C)C[O-]", 0, "charged oxygen — must decline"),
    ],
)
def test_tier5b_deriver_fail_closed(smiles, attach, why):
    mol = Chem.MolFromSmiles(smiles)
    res = _located_acyclic_alkyl_name(mol, list(range(mol.GetNumAtoms())), attach)
    assert res is None, f"expected decline for {smiles} ({why})"


@pytest.mark.unit
def test_tier5b_deriver_pure_alkyl_unchanged():
    # pentan-3-yl and 2-methylpentan-3-yl behavior byte-identical (T3c tie).
    mol = Chem.MolFromSmiles("C(CC)CC")
    assert _located_acyclic_alkyl_name(
        mol, list(range(mol.GetNumAtoms())), 0
    ) == ("pentan-3-yl", 3)
    mol2 = Chem.MolFromSmiles("C(C(C)C)CC")
    assert _located_acyclic_alkyl_name(
        mol2, list(range(mol2.GetNumAtoms())), 0
    ) == ("2-methylpentan-3-yl", 3)
