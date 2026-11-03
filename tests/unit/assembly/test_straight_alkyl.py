"""The carbon-count alkyl name only for the shape it names (``assembly.straight_alkyl``).

 (1) (the Blue Book): 'methyl', 'ethyl',..., 'heptyl' are the prefixes of an unbranched saturated chain
whose free valence is at an end carbon. The producers that named an unnamed branch by its count of carbons gave
'4-heptylcyclohexane-1-carboxylic acid' for the phenylcarbamoyl branch -C(=O)NHC6H5 (seven carbons) and 'heptyl'
for a 4-methylphenyl ring. Over 7,701 molecules (a dev split, dev2000, druglike2000, the Blue Book oracle; PIN and
best-effort tiers) the composer's fallback ran 217 times on 71 molecules, and never on that one shape.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.straight_alkyl import straight_alkyl_name

pytestmark = pytest.mark.unit


def _branch(smiles, first):
    """The atoms of the branch that starts at atom ``first`` of ``smiles``, away from atom 0 (the parent)."""
    mol = Chem.MolFromSmiles(smiles)
    seen, todo = {first}, [first]
    while todo:
        for nb in mol.GetAtomWithIdx(todo.pop()).GetNeighbors():
            if nb.GetIdx() not in seen and nb.GetIdx() != 0:
                seen.add(nb.GetIdx())
                todo.append(nb.GetIdx())
    return mol, seen


@pytest.mark.parametrize("smiles,name", [
    ("NC", "methyl"),
    ("NCC", "ethyl"),
    ("NCCCCCCC", "heptyl"),
])
def test_a_straight_chain_bound_by_its_end_takes_the_count_name(smiles, name):
    mol, atoms = _branch(smiles, 1)
    assert straight_alkyl_name(mol, atoms) == name


@pytest.mark.parametrize("smiles", [
    "NC(=O)Nc1ccccc1",       # the phenylcarbamoyl branch: 7 carbons, not 'heptyl'
    "Nc1ccc(C)cc1",          # a 4-methylphenyl ring, not 'heptyl'
    "NC(C)C",                # branched: propan-2-yl, not 'propyl'
    "NCC(C)C",               # branched below the end
    "NC=CC",                 # unsaturated
    "NCCO",                  # a heteroatom in the branch
    "NC[13CH3]",             # a labelled carbon
    "NCCC1CC1",              # a ring
])
def test_any_other_branch_gets_no_count_name(smiles):
    mol, atoms = _branch(smiles, 1)
    assert straight_alkyl_name(mol, atoms) is None


def test_a_chain_bound_by_an_inner_carbon_gets_no_count_name():
    """'NC(CC)CC' bound at its middle carbon is pentan-3-yl."""
    mol, atoms = _branch("NC(CC)CC", 1)
    assert straight_alkyl_name(mol, atoms) is None
