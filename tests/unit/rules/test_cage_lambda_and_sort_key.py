"""'7,7-dioxo-7λ6-thia' and the alphanumerical place of '(1λ1-aluminamethyl)'.

A skeletal sulfur with two oxo prefixes has the bonding number 6: "Nonstandard bonding numbers"
(the Blue Book, 'λn' cited with the locant) and (:29436, the group -SO2- in a ring system, "the sulfur atoms
are designated as λ4 or λ6 atoms"). The substituent cage core is the ring alone, so its prefix ('7-thia-8-aza') carried no
λ; ``rules.cage_lambda`` reads the bonding number from the molecule. (:3446) leaves Greek letters out of the
alphanumerical order and locants are never letters: '(1λ1-aluminamethyl)' is alphabetized at 'a', ahead of 'oxo'.
"""
import pytest

from orthonym.assembly.naming_utils import alpha_sort_key
from tests.support.rt_assert import name_best_effort, name_is_rt_exact


@pytest.mark.parametrize("name,key", [
    ("(1λ1-aluminamethyl)", "aluminamethyl"),
    ("(1λ1-thallamethyl)", "thallamethyl"),
    ("(1-indamethyl)", "indamethyl"),
])
def test_lambda_after_the_locant_is_not_part_of_the_sort_key(name, key):
    assert alpha_sort_key(name) == key


@pytest.mark.parametrize("smiles,expected", [
    ("C1=CC=C2C(=C1)C=CC(=N2)/C=C\\3/C4=CC=CC=C4S(=O)(=O)N3[Al]",
     "2-{[(9Z)-8-(1λ1-aluminamethyl)-7,7-dioxo-7λ6-thia-8-azabicyclo[4.3.0]nona-1,3,5-trien-9-ylidene]methyl}quinoline"),
    ("C1=CC=C2C(=C1)C=CC(=N2)/C=C\\3/C4=CC=CC=C4S(=O)(=O)N3C",
     "2-{[(9Z)-8-methyl-7,7-dioxo-7λ6-thia-8-azabicyclo[4.3.0]nona-1,3,5-trien-9-ylidene]methyl}quinoline"),
])
def test_ring_sulfonyl_cites_its_bonding_number(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected, res
    assert name_is_rt_exact(expected, smiles)
