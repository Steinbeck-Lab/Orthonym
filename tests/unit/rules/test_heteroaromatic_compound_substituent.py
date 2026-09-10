""" #29 — heteroaromatic parent + COMPOUND N-substituent + PCG suffix must
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
    return Orthonym(style="pin")  # PIN default — on


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


@pytest.mark.parametrize("smi,expected", [
    # #29 gap-a (was abstaining): the substituted-arene arenesulfonamido
    # substituent now names at PIN — the exact BB row:33034 —
    # after the _acid_stem_unsaturated_oxide_prefix piece-selection fix. Generalises
    # across heteroaromatic hosts (thiazole/thiophene/pyridine).
    ("OC(=O)c1cnc(NS(=O)(=O)c2ccc(N)cc2)s1",
     "2-(4-aminobenzene-1-sulfonamido)-1,3-thiazole-5-carboxylic acid"),
    ("OC(=O)c1cnc(NS(=O)(=O)c2ccc(C)cc2)s1",
     "2-(4-methylbenzene-1-sulfonamido)-1,3-thiazole-5-carboxylic acid"),
    ("OC(=O)c1ccc(NS(=O)(=O)c2ccc(N)cc2)s1",
     "5-(4-aminobenzene-1-sulfonamido)thiophene-2-carboxylic acid"),
])
def test_arene_sulfonamido_names_at_pin(smi, expected):
    assert _pin().name(smi) == expected
