"""Ring -OH and -SH suffixes on arene parents and ring assemblies (``rules.polycyclics``, ``rules.ring_assemblies``).

 class 17 (the Blue Book): hydroxy compounds and their chalcogen analogues; the most principal groups,
then a ring before a chain ('4-(hydroxymethyl)phenol'). Ring assemblies take the suffix on the enclosed parent
('[1,1'-binaphthalene]-2,2'-diol (PIN)', '[1,1'-biphenyl]-2,4,4',6-tetrol (PIN)':26892). These rows had no PIN.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("Sc1ccc2ccccc2c1-c1c(S)ccc2ccccc12", "[1,1'-binaphthalene]-2,2'-dithiol"),
    ("Sc1ccc(-c2ccc(S)cc2)cc1", "[1,1'-biphenyl]-4,4'-dithiol"),
    ("Sc1ccc(-c2ccccc2)cc1", "[1,1'-biphenyl]-4-thiol"),
    ("Oc1ccc(-c2ccc(S)cc2)cc1", "4'-sulfanyl[1,1'-biphenyl]-4-ol"),
    ("Nc1ccc(-c2ccc(S)cc2)cc1", "4'-amino[1,1'-biphenyl]-4-thiol"),
    ("OCc1ccc2cc(O)ccc2c1", "6-(hydroxymethyl)naphthalen-2-ol"),
    ("OC1c2ccccc2-c2cc(S)ccc12", "3-sulfanyl-9H-fluoren-9-ol"),
    ("OCc1cc(S)c2ccccc2c1O", "2-(hydroxymethyl)-4-sulfanylnaphthalen-1-ol"),
])
def test_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


def test_a_branch_with_more_hydroxy_groups_is_the_parent():
    #: the ethane-1,2-diol chain holds two -OH against the ring's one
    assert tier_row("OCC(O)c1ccc2cc(O)ccc2c1", "pin")["name"] == "1-(6-hydroxynaphthalen-2-yl)ethane-1,2-diol"


@pytest.mark.parametrize("smiles,expected", [
    # the ring -OH suffix takes the lowest locants first (c), the Blue Book)
    ("OCc1cccc2c1ccc1cc(O)ccc12", "8-(hydroxymethyl)phenanthren-2-ol"),
    ("OCc1ccc2ccc3cccc(O)c3c2c1", "6-(hydroxymethyl)phenanthren-4-ol"),
    ("OCc1cc(O)cc2ccccc12", "4-(hydroxymethyl)naphthalen-2-ol"),
    ("OCc1ccc2cc(O)ccc2c1CO", "5,6-bis(hydroxymethyl)naphthalen-2-ol"),
])
def test_the_ol_suffix_is_numbered_first(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


def test_a_stereo_ring_thiol_keeps_the_substitutive_route():
    # the assembly name writes no ring cis/trans descriptor: the substitutive name keeps it
    assert tier_row("S[C@H]1CC[C@@H](CC1)C1CCCCC1", "best-effort")["name"] == "trans-4-cyclohexylcyclohexane-1-thiol"
