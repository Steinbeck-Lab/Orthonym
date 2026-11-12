"""Spelling check /: a phosphonic or arsonic acid on a hydrocarbon skeleton is the parent of the
name, and two of them are multiplied ('bis(phosphonic acid)').

 "Names of mononuclear noncarbon oxoacids" (the Blue Book): ':35472 (naphthalene-2,6-diyl)bis(phosphonous
acid) (PIN)'. "Identical structural units" (:5166) (d): "functional parent compounds having substitutable
hydrogen atoms, for example, acetic acid or phosphonic acid". "Preferred IUPAC multiplicative names" (:23178):
"Multiplicative nomenclature is preferred to substitutive nomenclature for generating preferred IUPAC names to express
multiple occurrences of identical parent structures"; (:6295) allows an unsymmetrical central group.

The engine named '2,7-diphosphonobicyclo[2.2.1]heptane' and '(6-phosphonodecahydronaphthalen-2-yl)phosphonic acid'
pin_verified (audit 2026-11-11). The ring PINs are built now from a carboxylic surrogate of the same numbering
(``multiplicative._carboxylic_surrogate_name``).
"""
import pytest

from orthonym.validation import pin_spelling as ps
from orthonym.validation.spelling import checks_p_oxoacid_parent as chk
from tests.support.pin_tiers import assert_not_pin_labelled, assert_pin_at_both_tiers
from tests.support.read_back import mol as _mol

pytestmark = pytest.mark.unit

FAILS = [  # (smiles, name, rule)
    ("OP(=O)(O)C1CC2CCC1C2P(=O)(O)O", "2,7-diphosphonobicyclo[2.2.1]heptane", "P-67.1.1.1"),
    ("OP(=O)(O)C1CCC2CC(P(=O)(O)O)CCC2C1", "(6-phosphonodecahydronaphthalen-2-yl)phosphonic acid", "P-51.3.1"),
    ("O[As](=O)(O)C1CC2CCC1C2[As](=O)(O)O", "2,7-diarsonobicyclo[2.2.1]heptane", "P-67.1.1.1"),
    ("OP(=O)(O)CCP(=O)(O)O", "(2-phosphonoethyl)phosphonic acid", "P-51.3.1"),
    ("OP(=O)(O)c1ccc(cc1)-c1ccc(cc1)P(=O)(O)O", "(4'-phosphono[1,1'-biphenyl]-4-yl)phosphonic acid", "P-51.3.1"),
]
NO_FAILURE = [  # (smiles, name): passes, or the check abstains
    ("OP(=O)(O)C1CCC(P(=O)(O)O)CC1", "(cyclohexane-1,4-diyl)bis(phosphonic acid)"),
    ("CCP(=O)(O)O", "ethylphosphonic acid"),
    ("OP(=O)(O)CP(=O)(O)O", "methylenebis(phosphonic acid)"),
    ("OC(=O)CP(=O)(O)O", "phosphonoacetic acid"),                       # a senior acid: abstain
    ("OP(=O)(O)C1CCN(CC1)P(=O)(O)O", "(1-phosphonopiperidin-4-yl)phosphonic acid"),  # ring N: abstain
    ("OP(=O)(O)C1CC(P(=O)(O)O)CC(P(=O)(O)O)C1", "(cyclohexane-1,3,5-triyl)tris(phosphonic acid)"),
    # a ring and a chain are no single multiplying group, the Blue Book): substitutive PIN
    ("OP(=O)(O)Cc1ccc(cc1)P(=O)(O)O", "[(4-phosphonophenyl)methyl]phosphonic acid"),
    ("OP(=O)(O)CCc1ccccc1P(=O)(O)O", "[2-(2-phosphonophenyl)ethyl]phosphonic acid"),
]


@pytest.mark.parametrize("smiles,name,rule", FAILS, ids=[n for _, n, _ in FAILS])
def test_failures(smiles, name, rule):
    failure = chk.phosphonic_acid_parent(_mol(smiles), name)
    assert isinstance(failure, ps.SpellingFailure) and failure.rule == rule
    assert rule in [f.rule for f in ps.check_pin_spelling(_mol(smiles), name)]


@pytest.mark.parametrize("smiles,name", NO_FAILURE, ids=[n for _, n in NO_FAILURE])
def test_no_failure(smiles, name):
    assert not isinstance(chk.phosphonic_acid_parent(_mol(smiles), name), ps.SpellingFailure)


def test_interface():
    assert "orthonym.validation.spelling.checks_p_oxoacid_parent" in ps.BUILTIN_CHECK_MODULES
    assert chk.phosphonic_acid_parent in ps._REGISTRY["P-51.3.1"]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", [
    ("OP(=O)(O)C1CC2CCC1C2P(=O)(O)O", "(bicyclo[2.2.1]heptane-2,7-diyl)bis(phosphonic acid)"),
    ("OP(=O)(O)C1CC2CCC1CC2P(=O)(O)O", "(bicyclo[2.2.2]octane-2,5-diyl)bis(phosphonic acid)"),
    ("OP(=O)(O)C1CCC2CC(P(=O)(O)O)CCC2C1", "(decahydronaphthalene-2,6-diyl)bis(phosphonic acid)"),
    ("OP(=O)(O)c1ccc(cc1)-c1ccc(cc1)P(=O)(O)O", "([1,1'-biphenyl]-4,4'-diyl)bis(phosphonic acid)"),
    ("OP(=O)(O)c1cccc2c(P(=O)(O)O)cccc12", "(naphthalene-1,5-diyl)bis(phosphonic acid)"),
    #: a linker holding a stereo block is enclosed in square brackets
    ("O=P(O)(O)[C@H]1C[C@@H]2CC[C@H]1[C@H]2P(=O)(O)O",
     "[(1R,2S,4S,7S)-bicyclo[2.2.1]heptane-2,7-diyl]bis(phosphonic acid)"),
])
def test_ring_bisphosphonic_acids_get_the_multiplicative_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", [
    ("OP(=O)(O)Cc1ccc(cc1)P(=O)(O)O", "[(4-phosphonophenyl)methyl]phosphonic acid"),
])
def test_a_ring_and_chain_linker_keeps_its_substitutive_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.opsin_gate
def test_the_arsonic_acid_keeps_no_false_pin():
    assert_not_pin_labelled("O[As](=O)(O)C1CC2CCC1C2[As](=O)(O)O", "2,7-diarsonobicyclo[2.2.1]heptane")
