"""The longest carbon path from a carboxyl carbon (``rules.carbon_path``), found by a branch and bound.

The plain DFS it replaces kept, of several longest paths, the first in DFS order; the bound cuts only branches that
cannot make a strictly longer path, so it returns the same path (speed audit 2026-11-24, lever K7).
"""
import pytest
from rdkit import Chem

from orthonym.rules.carbon_path import first_longest_carbon_path

pytestmark = pytest.mark.unit


def _plain_dfs(mol, start, excluded):
    def walk(i, visited):
        best = [i]
        for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nbr.GetIdx()
            if j in visited or j in excluded or nbr.GetSymbol() != "C":
                continue
            visited.add(j)
            cand = [i] + walk(j, visited)
            if len(cand) > len(best):
                best = cand
            visited.discard(j)
        return best
    return walk(start, {start})


@pytest.mark.parametrize("smi", [
    "CCCCCCCC(=O)[O-]",
    "CC(C)CC(CC)CC(=O)[O-]",                        # branches: ties broken as the DFS did
    "[O-]C(=O)C1CCCCC1CC(C)C=CC",
    "[O-]C(=O)c1ccc2ccccc2c1",                      # many equal paths round a fused ring
    "[O-]C(=O)CC1C2CC3CC1CC(C2)C3",                 # a cage
    "[O-]C(=O)CCOCCC",                              # the walk stops at a heteroatom
])
def test_the_same_path_as_the_plain_dfs(smi):
    mol = Chem.MolFromSmiles(smi)
    o = next(a for a in mol.GetAtoms() if a.GetFormalCharge() == -1)
    c = o.GetNeighbors()[0].GetIdx()
    excluded = {n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetSymbol() == "O"}
    assert first_longest_carbon_path(mol, c, excluded) == _plain_dfs(mol, c, excluded)
