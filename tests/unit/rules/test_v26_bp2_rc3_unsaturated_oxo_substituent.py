"""v26 BP-2 RC-3 — unsaturated substituent chain with an in-chain carbonyl -> oxo.

An acyclic all-carbon substituent chain that is UNSATURATED and carries an
in-chain / terminal aldehyde or ketone fell between two tiers: the pure-alkenyl
namer declines (the =O is not C) and the saturated polyfunctional builder
declines (unsaturation + internal attachment). The recursive path then named the
capped fragment as a free molecule (`prop-2-enal`) and `parent_to_prefix` (post
RC-1) returned None -> `unknown`. Pre-RC-1 it fabricated `prop-2-enalyl`.

RC-3 expresses the carbonyl as the detachable prefix `oxo` on the chain numbered
from the free valence (P-33 oxo; P-14.4 free valence > unsaturation > oxo), e.g.
`-C(=CH2)-CHO` -> `3-oxoprop-1-en-2-yl`. Fail-closed on rings, any heteroatom
other than a ketone/aldehyde =O, branched/non-single chains, no unsaturation
(saturated tier owns it), no oxo, or oxo ON the free-valence carbon (acyl).
"""
from rdkit import Chem

from orthonym.assembly.substituent_naming import (
    _name_unsaturated_oxo_substituent,
)
from orthonym.namer import name_compound


def _sub_fragment(smiles, parent_placeholder_last=True):
    """Build (mol, sub_atoms, attach_idx) from a SMILES whose LAST atom is a
    placeholder 'parent' carbon bonded to the substituent's attach atom."""
    mol = Chem.MolFromSmiles(smiles)
    n = mol.GetNumAtoms()
    parent = n - 1
    attach = [nb.GetIdx() for nb in mol.GetAtomWithIdx(parent).GetNeighbors()][0]
    sub_atoms = [i for i in range(n) if i != parent]
    return mol, sub_atoms, attach


# --- full name (authoritative + RT via gate) -------------------------------

def test_full_name_unsaturated_oxo_on_ring():
    assert name_compound("C=C(C=O)C1CCC(C)C1C=O") == \
        "2-methyl-5-(3-oxoprop-1-en-2-yl)cyclopentane-1-carbaldehyde"


# --- builder positive ------------------------------------------------------

def test_builder_prop_en_oxo():
    # C=C(C=O)C : last C is the parent placeholder; sub = -C(=CH2)-CHO
    mol, sub, attach = _sub_fragment("C=C(C=O)C")
    assert _name_unsaturated_oxo_substituent(mol, sub, attach, set()) == \
        "3-oxoprop-1-en-2-yl"


# --- builder fail-closed ---------------------------------------------------

def test_builder_declines_pure_alkenyl():
    # no oxo -> None (pure-alkenyl tier owns it)
    mol, sub, attach = _sub_fragment("C=CCC")
    assert _name_unsaturated_oxo_substituent(mol, sub, attach, set()) is None


def test_builder_declines_saturated_oxo():
    # saturated + oxo -> None (saturated polyfunctional tier owns it)
    mol, sub, attach = _sub_fragment("O=CCCC")
    assert _name_unsaturated_oxo_substituent(mol, sub, attach, set()) is None


def test_builder_declines_acyl_on_attach():
    # oxo ON the free-valence carbon = acyl -> None (P-66 note (m))
    # C(=O)C=C  with the attach being the carbonyl carbon
    mol = Chem.MolFromSmiles("O=CC=CC")           # penta? -> acyl at C1
    # attach = the carbonyl C (idx 1), sub = atoms 0..3, parent = last C(4)
    sub = [0, 1, 2, 3]
    assert _name_unsaturated_oxo_substituent(mol, sub, 1, set()) is None


def test_builder_declines_ring():
    mol = Chem.MolFromSmiles("O=CC1=CC1C")        # contains a ring
    assert _name_unsaturated_oxo_substituent(
        mol, [0, 1, 2, 3], 1, set()) is None
