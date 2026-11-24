"""The bound of the longest-cycle search (``main_ring_cycles._blocks_bound``) on a bit mask of free atoms.

The bound is the number of atoms in the blocks (biconnected components) on the way from ``v`` to ``s``. It is now
found by Tarjan's algorithm in the vertex-stack form with each block a bit mask (speed audit 2026-11-24, lever K6).
The reference below is the edge-stack form on a set that the search used before; the two must agree on every call.
"""
import random

import pytest
from rdkit import Chem

from orthonym.rules import main_ring_cycles as mrc
from tests.unit.rules.test_main_ring_levels import _polyhex, _random_cage

pytestmark = pytest.mark.unit


def _reference(nbrs, free, v, s):
    """The edge-stack form on a set of free atoms (the code before lever K6)."""
    members = free | {v, s}
    disc, low, parent = {v: 0}, {v: 0}, {v: None}
    edges, block_of, blocks, clock = [], {}, [], 1
    stack = [(v, iter(nbrs[v]))]
    while stack:
        x, it = stack[-1]
        for w in it:
            if w not in members:
                continue
            if w not in disc:
                parent[w] = x
                disc[w] = low[w] = clock
                clock += 1
                edges.append((x, w))
                stack.append((w, iter(nbrs[w])))
                break
            if w != parent[x] and disc[w] < disc[x]:
                low[x] = min(low[x], disc[w])
                edges.append((x, w))
        else:
            stack.pop()
            if stack:
                p = stack[-1][0]
                low[p] = min(low[p], low[x])
                if low[x] >= disc[p]:
                    block = set()
                    while True:
                        e = edges.pop()
                        block.update(e)
                        if e == (p, x):
                            break
                    blocks.append(block)
                    for a in block:
                        if a != p and parent.get(a) is not None and parent[a] in block:
                            block_of.setdefault(a, len(blocks) - 1)
    if s not in disc:
        return None
    on_way, y = set(), s
    while y != v:
        on_way.add(block_of[y])
        y = parent[y]
    return len(set().union(*(blocks[k] for k in on_way)))


def _graph(mol):
    return {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors()] for a in mol.GetAtoms()}


@pytest.mark.parametrize("seed", range(40))
def test_the_bit_mask_bound_agrees_with_the_set_bound(seed):
    rng = random.Random(seed)
    mol = _random_cage(rng) if seed % 2 else _polyhex(rng, rng.randint(3, 9))
    nbrs = _graph(mol)
    atoms = sorted(nbrs)
    for _ in range(60):
        v, s = rng.sample(atoms, 2)
        free = {a for a in atoms if a not in (v, s) and rng.random() < 0.7}
        mask = sum(1 << a for a in free)
        assert mrc._blocks_bound(nbrs, mask, v, s) == _reference(nbrs, free, v, s)


def test_the_longest_cycles_of_a_fused_ring_system_are_found():
    mol = Chem.MolFromSmiles("c1ccc2cc3ccccc3cc2c1")          # anthracene: the 14-atom perimeter
    length, cycles, complete = mrc.longest_cycles(_graph(mol), sorted(_graph(mol)), lambda k: None)
    assert (length, len(cycles), complete) == (14, 1, True)
