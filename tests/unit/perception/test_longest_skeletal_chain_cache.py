"""``chains.find_longest_skeletal_chain`` keeps its answer per (elements, neighbour lists, excluded atoms).

 /: the longest chain of skeletal atoms (C, N, O, S) is a candidate parent for the pool.
The depth-first search is a function of the elements of the atoms, their neighbour lists and the excluded atoms
only, and every pass that names a structure asks for it again, so the answer is kept for the life of the
process. The tests compare it with the search written out as it was, over the atoms and the exclusions the pool
passes, in any order of the calls and from a cold or a warm cache.
"""
import random

import pytest
from rdkit import Chem

from orthonym.perception import chains

pytestmark = pytest.mark.unit

_SKELETAL = {6, 7, 8, 16}

SMILES = [
    "CCCCC", "CC(C)CC(C)C", "OCCOCCOCCO", "NCCNCCNCCN", "CCSCCSCC", "CC(=O)NCC(=O)NCC(=O)O",
    "CC(C)C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CCCCN)N)C(=O)O", "C1CCCCC1CCOCC", "c1ccccc1CCNCCc1ccccc1",
    "CCOCC(=O)OCC", "OC(=O)CNCC(=O)O", "CC(C)(C)OC(C)(C)C", "C(C(C(C)C)C(C)C)C(C)C(C)C", "CCCCCCCCCCCCCCCCOCC",
    "CN(C)CCOCCN(C)C", "[Na+].[O-]C(=O)COCC(=O)[O-]", "CCCOCCC.CC", "N#CCCOCCC#N", "OCC(O)COCC(O)CO",
    "C=CCOCC=C", "FC(F)(F)CCOC", "CS(=O)(=O)NCCOCC", "CCP(CC)CC", "[2H]OC([2H])([2H])CC", "C1CC1CNCC1CC1",
]


def _as_it_was(mol, exclude_atoms=None):
    """``find_longest_skeletal_chain`` before the answer was kept."""
    exclude = exclude_atoms or set()

    def dfs(atom_idx, visited, path, results):
        if atom_idx in exclude:
            return
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() not in _SKELETAL:
            return
        visited.add(atom_idx)
        path.append(atom_idx)
        if len(path) > len(results[0]):
            results[0] = path.copy()
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited:
                dfs(nbr_idx, visited, path, results)
        path.pop()
        visited.discard(atom_idx)

    results = [[]]
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in _SKELETAL and atom.GetIdx() not in exclude:
            dfs(atom.GetIdx(), set(), [], results)
    return results[0]


def _cases():
    """Every molecule with no exclusion, with its ring atoms excluded (as the pool does) and with random
    atoms excluded."""
    rng = random.Random(20261026)
    cases = []
    for smiles in SMILES:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        ring_atoms = {a for ring in mol.GetRingInfo().AtomRings() for a in ring}
        n = mol.GetNumAtoms()
        cases += [(mol, None), (mol, set()), (mol, ring_atoms)]
        cases += [(mol, set(rng.sample(range(n), k))) for k in (1, 2, max(1, n // 3)) if k <= n]
    return cases


def test_answer_equals_the_search_as_it_was():
    for mol, exclude in _cases():
        assert chains.find_longest_skeletal_chain(mol, exclude) == _as_it_was(mol, exclude), (
            Chem.MolToSmiles(mol), exclude)


def test_answer_does_not_depend_on_the_order_of_the_calls_or_the_cache():
    cases = _cases()
    chains._longest_skeletal_path.cache_clear()
    forward = [chains.find_longest_skeletal_chain(m, e) for m, e in cases]
    chains._longest_skeletal_path.cache_clear()
    backward = [chains.find_longest_skeletal_chain(m, e) for m, e in reversed(cases)][::-1]
    warm = [chains.find_longest_skeletal_chain(m, e) for m, e in cases]
    expected = [_as_it_was(m, e) for m, e in cases]
    assert forward == backward == warm == expected


def test_the_answer_is_kept_across_molecule_objects():
    """Two objects of one structure give one answer from one search."""
    chains._longest_skeletal_path.cache_clear()
    first = chains.find_longest_skeletal_chain(Chem.MolFromSmiles("CCOCCOCC"), set())
    before = chains._longest_skeletal_path.cache_info()
    second = chains.find_longest_skeletal_chain(Chem.MolFromSmiles("CCOCCOCC"), set())
    after = chains._longest_skeletal_path.cache_info()
    assert first == second == list(range(8))
    assert (after.hits, after.misses) == (before.hits + 1, before.misses)


def test_the_atom_order_of_a_structure_is_part_of_the_answer():
    """The path is a list of atom indices: the same structure written in another order has its own answer."""
    a = Chem.MolFromSmiles("CCOCC(N)C")
    b = Chem.MolFromSmiles("NC(C)COCC")
    assert chains.find_longest_skeletal_chain(a, set()) == _as_it_was(a, set())
    assert chains.find_longest_skeletal_chain(b, set()) == _as_it_was(b, set())


def test_the_list_returned_is_the_callers():
    """A caller may change the list: the kept answer is not touched."""
    mol = Chem.MolFromSmiles("CCOCC")
    first = chains.find_longest_skeletal_chain(mol, set())
    first.append(99)
    assert chains.find_longest_skeletal_chain(mol, set()) == _as_it_was(mol, set())
