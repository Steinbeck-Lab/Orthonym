"""``namer._same_canonical_smiles`` reads each canonical form from a process-wide cache.

The mixture read-back compares the same few SMILES thousands of times on one input (speed audit 2026-11-24, lever K3:
6,285 calls on 41 strings); the canonical form is a pure function of the string, so the cache is exact.
"""
import pytest

from orthonym.namer import _same_canonical_smiles
from orthonym.perception.molcache import canon_smiles


@pytest.mark.parametrize("a,b,same", [
    ("OCC", "CCO", True),                              # one graph, two writings
    ("CCO", "CC=O", False),
    ("F[C@H](Cl)Br", "F[C@@H](Cl)Br", False),          # the stereo layer counts
    ("C1CC", "CCO", False),                            # does not parse
    ("xx", "xx", False),                               # neither parses
    ("", "", True),                                    # RDKit reads an empty SMILES as an empty molecule
])
def test_the_answer(a, b, same):
    assert _same_canonical_smiles(a, b) is same


def test_a_repeat_comes_from_the_cache():
    _same_canonical_smiles("CC(=O)[O-]", "[O-]C(C)=O")
    hits = canon_smiles.cache_info().hits
    _same_canonical_smiles("CC(=O)[O-]", "[O-]C(C)=O")
    assert canon_smiles.cache_info().hits == hits + 2
