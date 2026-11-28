"""A name with no suffix is built on the senior ring system: "general criteria for determining ring seniority... are
applied successively until there are no alternatives remaining" ('the senior ring or ring system: (a) is a heterocycle;
(b) has at least one nitrogen atom;... (d) has the greater number of rings; (e) has the greater number of skeletal
atoms; (f) has the greater number of heteroatoms of any kind', the Blue Book-19417; '2-[(naphthalen-2-yl)methyl]
pyridine (PIN) (pyridine > naphthalene)',:19426). A name on a ring system that loses keeps its name below the PIN label
(rules/ring_parent_seniority.py).
"""
import pytest

from orthonym.rules.ring_parent_seniority import ring_parent_seniority_rows

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("C1C2=CC=CC=C2C3=C1C=C(C=C3)C4=CC5=CC=CC=C5N4", "2-(1H-indol-2-yl)-9H-fluorene"),                  # (a)
    ("C1CN=C(N1)CC2=CNC3=CC=CC=C32", "2-[(1H-indol-3-yl)methyl]-4,5-dihydro-1H-imidazole"),           # (d)
    ("C1C=C(C2=CC=CC=C21)C3=CC=CC4=CC=CC=C43", "3-(naphthalen-1-yl)-1H-indene"),                       # (e)
    ("c1ccc2oc(Cc3c[nH]c4ccccc34)nc2c1", "3-[(1,3-benzoxazol-2-yl)methyl]-1H-indole"),                # (f)
    ("Cc1ccc(cc1)-c1ccccn1", "1-methyl-4-(pyridin-2-yl)benzene"),                                     # (a)
    ("C1CN=C(N1)SCc1ccccn1", "2-{[(pyridin-2-yl)methyl]sulfanyl}-4,5-dihydro-1H-imidazole"),         # (e) 6 > 5
    ("C[C@@H]1CC[C@@H](C)P1c1ccccc1C1OCCO1", "(2R,5R)-1-[2-(1,3-dioxolan-2-yl)phenyl]-2,5-dimethylphospholane"),  # (c) O > P
])
def test_a_name_on_the_junior_ring_system_is_flagged(smiles, name):
    rows = ring_parent_seniority_rows(smiles, name)
    assert rows and rows[0]["rule"] == "P-44.2.1", name


@pytest.mark.parametrize("smiles,name", [
    ("c1ccc(Cc2ccc3ccccc3c2)nc1", "2-[(naphthalen-2-yl)methyl]pyridine"),                             #:19426
    ("C1C2=CC=CC=C2C3=C1C=C(C=C3)C4=CC5=CC=CC=C5N4", "2-(9H-fluoren-2-yl)-1H-indole"),
    ("c1ccc(-c2ccc3ccccc3c2)cc1", "2-phenylnaphthalene"),
    ("Oc1ccc(-c2ccccn2)cc1", "4-(pyridin-2-yl)phenol"),                                              # a suffix
    ("C1=CC=C2C(=C1)C=NN2C3=NC4=CC=CC=C4S3", "1-(1,3-benzothiazol-2-yl)-1H-indazole"),              # (g): not judged
    ("c1ccc(cc1)-c1ccccc1", "1,1'-biphenyl"),
    ("C[Si](C)(C)c1ccccn1", "2-(trimethylsilyl)pyridine"),
    # an amine is a characteristic group: the amine parent stands,:19422)
    ("CN(C)C1=CC=C(C=C1)C2=NC3=CC=CC=C3C=C2", "N,N-dimethyl-4-(quinolin-2-yl)aniline"),
    ("Cc1ccccc1N(CC1CO1)CC1CO1", "2-methyl-N,N-bis(oxiranylmethyl)aniline"),
    ("CC1=NC2=CC=CC=C2C(=C1)COC3CCN(CC3)S(=O)(=O)C", "1-(methanesulfonyl)-4-[(2-methylquinolin-4-yl)methoxy]piperidine"),
    # an unlocanted suffix is no bare parent ring
    ("C1CCC(=Nc2ccccn2)CC1", "N-(pyridin-2-yl)cyclohexanimine"),
    ("[SeH]c1ccccc1-c1ccncc1", "2-(pyridin-4-yl)benzeneselenol"),
    ("c1ccc2c(c1)CCC2c1ccc2ccccc2c1C12CCC(CC1)CC2", "1-(bicyclo[2.2.2]octan-1-yl)-2-(2,3-dihydro-1H-inden-1-yl)naphthalene"),
    ("c1ccc2ncccc2c1CC1CC2CCN1CC2", "5-[(1-azabicyclo[2.2.2]octan-2-yl)methyl]quinoline"),      # (d) 2 = 2, (e) 10 > 8
])
def test_other_names_pass(smiles, name):
    assert ring_parent_seniority_rows(smiles, name) == [], name
