"""A substituted fused ring named as a prefix numbers its free valence first.

 "NUMBERING" (the Blue Book): "(c) principal characteristic groups and free valences
(suffixes)" (:3256) take low locants before "(f) detachable alphabetized prefixes, all considered
together in a series of increasing numerical order" (:3301). The catalogue matcher used to give the
lowest locants to the free valence and the ring's own prefixes together, so the methyl groups won
('2,2-dimethyl-...-7-yl'); ``rules.fused_free_valence`` gives the free valence its lowest locant first.
"""
import pytest

from tests.support.large_polycycles import be, pin

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles, name", [
    ("OC(=O)CCc1ccc2OCC(C)(C)Oc2c1", "3-(3,3-dimethyl-2,3-dihydro-1,4-benzodioxin-6-yl)propanoic acid"),
    ("CC1(C)COc2ccc(CCC(=O)O)cc2O1", "3-(3,3-dimethyl-2,3-dihydro-1,4-benzodioxin-6-yl)propanoic acid"),
    ("OCc1ccc2OCC(C)Oc2c1", "(3-methyl-2,3-dihydro-1,4-benzodioxin-6-yl)methanol"),
    ("OC(=O)Cc1ccc2NCC(C)Nc2c1", "(3-methyl-1,2,3,4-tetrahydroquinoxalin-6-yl)acetic acid"),
    ("OC(=O)CCc1ccc2Sc3cc(Br)ccc3Sc2c1", "3-(7-bromothianthren-2-yl)propanoic acid"),
    ("OC(=O)CCc1ccc2c(c1)-c1ccc(Br)cc1C2", "3-(7-bromo-9H-fluoren-3-yl)propanoic acid"),
    ("OC(=O)CCc1ccc2c(c1)oc1ccc(Br)cc12", "3-(8-bromodibenzo[b,d]furan-3-yl)propanoic acid"),
    # already right: the matcher's numbering gives the free valence 6
    ("OC(=O)Cc1ccc2SCC(C)(C)Sc2c1", "(3,3-dimethyl-2,3-dihydro-1,4-benzodithiin-6-yl)acetic acid"),
    ("NC(=O)Cc1ccc2OC(C)C(C)Oc2c1", "2-(2,3-dimethyl-2,3-dihydro-1,4-benzodioxin-6-yl)acetamide"),
])
def test_free_valence_takes_the_lowest_locant_first(smiles, name):
    for row in (pin(smiles), be(smiles)):
        assert row["name"] == name
        assert row["tier"] == "pin_verified"


@pytest.mark.parametrize("smiles, name", [
    # no free valence: the prefixes alone decide (f)), as before
    ("Cc1ccc2OCC(C)(C)Oc2c1", "2,2,7-trimethyl-2,3-dihydro-1,4-benzodioxine"),
    ("CC1(C)COc2ccc(Cl)cc2O1", "7-chloro-2,2-dimethyl-2,3-dihydro-1,4-benzodioxine"),
    # a suffix on the ring: (c) as well
    ("OC(=O)c1ccc2OCC(C)(C)Oc2c1", "3,3-dimethyl-2,3-dihydro-1,4-benzodioxine-6-carboxylic acid"),
])
def test_the_ring_as_parent_is_unchanged(smiles, name):
    assert pin(smiles)["name"] == name
