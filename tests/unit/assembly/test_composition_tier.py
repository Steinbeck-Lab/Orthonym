"""Slice-1 composition tier — the covered-atom foundation.

The covered-atom computation is the foundation the whole composition step rests on, and the
Task-1/2 spy proved the obvious source is WRONG: ``features.principal_chain`` alone undercounts,
reporting ``propan-1-ol`` as 3/4 (it misses the ``-ol`` oxygen). These tests pin the correct
computation — chain PLUS principal-group atoms — and the uncovered-fragment split that finds the
dropped remainder. Pure functions, no OPSIN, so this stays in the fast suite.
"""

import sys
from pathlib import Path

from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from orthonym.assembly.composition_tier import (  # noqa: E402
    fragment_attachment_atom,
    parent_covered_atoms,
    uncovered_fragments,
)
from orthonym.cli import _emit_tier_flags  # noqa: E402
from orthonym.namer import Orthonym  # noqa: E402


def _features(smiles):
    flags = _emit_tier_flags("best-effort")
    nm = Orthonym(
        general_fallback=flags["general_fallback"],
        general_fallback_unverified=flags["general_fallback_unverified"],
        allow_aromatic_general=flags["allow_aromatic_general"],
    )
    mol = Chem.MolFromSmiles(smiles)
    feats = nm._perceive(mol, smiles, Chem.MolToSmiles(mol))
    nm._classify(feats)
    return mol, feats


def test_covered_includes_the_suffix_oxygen_not_just_the_chain():
    """The exact bug the spy exposed: propan-1-ol is fully covered (4/4), not 3/4."""
    mol, feats = _features("CCCO")
    covered = parent_covered_atoms(mol, feats)
    assert len(covered) == 4, f"propan-1-ol must cover all 4 heavy atoms, got {covered}"
    assert covered == {0, 1, 2, 3}


def test_covered_includes_all_carboxyl_atoms():
    """A multi-atom principal group (carboxylic acid = C,O,O) is fully counted."""
    mol, feats = _features("CCC(=O)O")
    covered = parent_covered_atoms(mol, feats)
    assert len(covered) == 5, f"propanoic acid must cover all 5 heavy atoms, got {covered}"


def test_atom_short_parent_leaves_the_dropped_substituent_uncovered():
    """COS(=O)(=O)O names 'methane' (covers only the methyl C); the sulfate is the remainder."""
    mol, feats = _features("COS(=O)(=O)O")
    covered = parent_covered_atoms(mol, feats)
    assert covered == {0}, f"methane parent covers only atom 0, got {covered}"

    frags = uncovered_fragments(mol, covered)
    assert len(frags) == 1, f"the sulfate is one connected remainder, got {frags}"
    assert set(frags[0]) == {1, 2, 3, 4, 5}


def test_no_remainder_when_fully_covered():
    """A fully-named molecule has no uncovered fragment."""
    mol, feats = _features("CCCO")
    frags = uncovered_fragments(mol, parent_covered_atoms(mol, feats))
    assert frags == []


def test_uncovered_fragments_are_disjoint_and_complete():
    """Every uncovered heavy atom lands in exactly one fragment (no overlap, none missed)."""
    mol = Chem.MolFromSmiles("OC(O)c1ccc(Br)c(C(O)O)c1")
    # pretend the aromatic ring is the covered parent; the two CH(OH)2 arms are the remainder
    covered = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
    frags = uncovered_fragments(mol, covered)
    flat = [i for f in frags for i in f]
    assert len(flat) == len(set(flat)), "fragments overlap"
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    assert set(flat) == heavy - covered, "fragments do not cover exactly the uncovered set"


def test_fragment_attachment_atom_is_inside_the_fragment_and_bonds_outward():
    mol, feats = _features("COS(=O)(=O)O")
    frags = uncovered_fragments(mol, parent_covered_atoms(mol, feats))
    att = fragment_attachment_atom(mol, frags[0])
    assert att in frags[0]
    # it must bond to an atom outside the fragment (the methyl carbon, atom 0)
    outside = [nb.GetIdx() for nb in mol.GetAtomWithIdx(att).GetNeighbors()
               if nb.GetIdx() not in set(frags[0])]
    assert outside, "attachment atom must bond outside the fragment"
