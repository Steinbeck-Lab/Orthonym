"""Cycloalkane numbering: (j) decides a tie that survives the earlier
criteria (pre-existing-failures plan, Task 4 continuation, 2026-09-25).

 (j) (the Blue Book): "the lower locant is assigned to CIP
stereodescriptors Z, R, M, and r (pseudoasymmetry) that are preferred to E, S,
P, and s". The Blue Book's own ring example is "13-norgermacrane
(1R,4s,7S)-4-ethyl-1,7-dimethylcyclodecane" (:51471). The orientation used to
leave this tie to the input atom order.
"""
import pytest

from orthonym import name_compound

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,expected", [
    # (j): R before S at a tied locant pair (a meso cis-1,3 ring)
    ("C[C@@H]1CCC[C@H](C)C1", "(1R,3S)-1,3-dimethylcyclohexane"),
    # the same molecule from the other end of the input
    ("C[C@@H]1C[C@H](C)CCC1", "(1R,3S)-1,3-dimethylcyclohexane"),
    # an equal-code pair is unaffected
    ("C[C@H]1CCC[C@H](C)C1", "(1S,3S)-1,3-dimethylcyclohexane"),
])
def test_cycloalkane_stereo_tie_break(smiles, expected):
    assert name_compound(smiles) == expected
