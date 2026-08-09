"""v30 #29 — heteroaromatic parent + COMPOUND N-substituent + PCG suffix must
name at the PIN tier (not only best-effort).

Root cause (pre-fix): name_substituted_heterocycle declined every N/O/S-anchored
exocyclic substituent because name_substituent_fragment mis-roots them
(-NHC(=O)CH3 -> 'carbamoylmethyl'). The recursive name_substituent DOES root them
correctly (acetamido / methylamino / methanesulfonamido); the handler now routes
N-anchored substituents through it, guarded by a gate-independent OPSIN atom-coverage
check so a partial name (an arenesulfonamido degrading to bare 'amino') fails closed.

Orthonym emits the PIN heteroatom locants '1,3-thiazole' that BOTH reference
engines drop (they emit bare 'thiazole').
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.unit


def _pin():
    return Orthonym(style="pin")  # PIN default — SELF-01 on


@pytest.mark.parametrize("smi,expected", [
    ("OC(=O)c1cnc(NC(C)=O)s1", "2-acetamido-1,3-thiazole-5-carboxylic acid"),
    ("OC(=O)c1cnc(NC)s1", "2-(methylamino)-1,3-thiazole-5-carboxylic acid"),
    ("OC(=O)c1cnc(NS(C)(=O)=O)s1", "2-(methanesulfonamido)-1,3-thiazole-5-carboxylic acid"),
])
def test_compound_n_substituent_thiazole_pin(smi, expected):
    assert _pin().name(smi) == expected


def test_baseline_amino_unchanged():
    # the simple-substituent path must be byte-identical
    assert _pin().name("OC(=O)c1cnc(N)s1") == "2-amino-1,3-thiazole-5-carboxylic acid"


def test_arene_sulfonamido_still_fails_closed_at_pin():
    # gap (a): the 2-ring arenesulfonamido substituent is NOT yet nameable;
    # it must abstain (fail closed), NEVER ship a wrong atom-dropped 'amino'.
    r = _pin().name("OC(=O)c1cnc(NS(=O)(=O)c2ccc(N)cc2)s1")
    assert r == "unknown organic compound"
