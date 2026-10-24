import pytest

from ._helpers import needs_phane_opsin, phane_row

pytestmark = [pytest.mark.opsin_gate, needs_phane_opsin]

#: structures whose tests describe OPSIN 2.9.0 (a decline, or a substitutive or von Baeyer name below the PIN) and are
#: marked release_opsin_only: with an OPSIN that reads phane names the structure gets its phane PIN, read back to the
#: full InChIKey (1) for the cyclic rows, for the linear ones)
ROWS = [
    ("C1=CC=C2C(=C1)OC3=CC=C(C=C3)OOS2", "a ten-atom macrocycle with an ortho and a para benzene ring"),
    ("Oc1c2cccc1Cc1cccc(c1O)Cc1cccc(c1O)Cc1cccc(c1O)Cc1cccc(c1O)Cc1cccc(c1O)C2", "calix[6]arene, six meta benzene rings"),
    ("C1=CC2=CC=C1C=C=CC3=CC=C(C=C3)C=C=C2", "cumulated double bonds in the bridges"),
    ("C1CC2=CC=C(C=C2)C#CC3=CC=C1C=C3", "an alkyne in a bridge"),
    ("OC(=O)c1cc(Oc2ccc(Oc3ccccc3)cc2)ccc1OC1CCCCO1", "four ring systems and seven nodes, an oxane terminal ring"),
    ("OC(=O)c1ccc(OC2CCCCO2)cc1NC(=O)c1ccc(cn1)N1CCCCC1", "a linear phane with an amide in the chain"),
    ("c1ccc2c3ccc(c2c1)CCCCCCCCC3", "a nonano chain across the 1,4-positions of naphthalene (:15039 shape)"),
    ("c1ccc2c3ccc(c2c1)CCCCCCCCCC3", "a decano chain across the 1,4-positions of naphthalene (:15039 shape)"),
    ("OC(=O)c1ccc(Nc2ccc(Nc3ccc(Nc4ccccc4)cc3)cc2)cc1", "four benzene rings and three NH on one chain"),
    ("OC(=O)c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cn2)cc1", "four ring systems and seven nodes on one chain"),
    ("c1ccc(-c2ccc(-c3ccc(-c4ccc(-c5ccc(-c6ccc(-c7ccc(-c8ccc(-c9ccc(-c%10ccc(-c%11ccccc%11)cc%10)cc9)cc8)cc7)cc6)cc5)cc4)cc3)cc2)cc1",
     "a ring assembly of eleven benzene rings (P-52.2.5.3 names ring assemblies of seven nodes or more as phanes)"),
]


@pytest.mark.parametrize("smiles,what", ROWS, ids=[w[:30] for _, w in ROWS])
def test_the_structure_gets_its_phane_pin(smiles, what):
    row = phane_row(smiles)
    assert row["tier"] == "pin_verified" and row["spelling_failures"] == [], row
