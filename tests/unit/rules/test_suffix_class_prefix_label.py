"""A senior thiol, hydroperoxide, alcohol or ketone takes the suffix classes 16-18, the Blue Book;
,:18875; 'cyclohexanethiol (PIN)',:2917; 'ethaneperoxol (PIN)',:24316). A name that ends in a bare parent
hydride and cites the group only as a prefix is correct but no PIN; the spelling check
(validation/spelling/checks_suffix_class_prefix.py) lowers its label and keeps the name.
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_suffix_class_prefix import suffix_class_as_prefix_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("SC1=NCCN1C", "1-methyl-2-sulfanyl-4,5-dihydro-1H-imidazole"),
    ("CC(=O)C1CCN(CC1)S(C)(=O)=O", "4-acetyl-1-(methanesulfonyl)piperidine"),
    ("OOC1CC2CCC1C2", "2-hydroperoxybicyclo[2.2.1]heptane"),
    ("C1=CC2=C3C(=C1)C(=C(C3=CC=C2)O)O", "1,2-dihydroxyacenaphthylene"),             # a multiplied prefix
    ("O=C1C(=O)C2CCC1C21CC1", "2,3-dioxospiro[bicyclo[2.2.1]heptane-7,1'-cyclopropane]"),
    ("O=C1CC2CCC(C1)C21OCCO1", "3-oxospiro[bicyclo[3.2.1]octane-8,2'-[1,3]dioxolane]"),  # a ketone as 'oxo'
    # an acyl on a ring N is a pseudoketone "'Hidden' amides",:33125), senior to the alcohol: the PIN is
    # '1-[2-(hydroxymethyl)piperidin-1-yl]ethan-1-one' (this row was listed as passing, read as an amide, before)
    ("OCC1CCCCN1C(C)=O", "1-acetyl-2-(hydroxymethyl)piperidine"),
    # a ring C=O/C=S on a ring heteroatom is a pseudoketone or heterone, class 16,:28267;,:29551)
    ("S=C1NCc2ccccc2O1", "2-sulfanylidene-3,4-dihydro-2H-1,3-benzoxazine"),
    ("O=C1NCCC12CC1CCC2CC1", "2'-oxospiro[bicyclo[2.2.2]octane-2,3'-pyrrolidine]"),
    ("S=C1[Se]c2ccccc2[Se]1", "2-sulfanylidene-2H-1,3-benzodiselenole"),
    ("O=c1ccc2ccccc2o1", "2-oxo-2H-1-benzopyran"),                         # an aromatic lactone, a '-pyran' ending
    ("S=C1CC(c2ccccc2)c2c(ccc3ccccc23)O1", "1-phenyl-3-sulfanylidene-2,3-dihydro-1H-naphtho[2,1-b]pyran"),
    ("CC1(C)C2CCC1(C)C(=S)C2", "1,7,7-trimethyl-2-sulfanylidenebicyclo[2.2.1]heptane"),   # a thioketone
    ("O=C1OCCO1", "2-oxo-1,3-dioxolane"),                                  # a cyclic carbonate
    ("[Se]=C1CCCCN1", "2-selanylidenepiperidine"),
])
def test_a_prefix_only_name_is_flagged(smiles, name):
    failure = suffix_class_as_prefix_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-44.1.1", name


@pytest.mark.parametrize("smiles,name", [
    ("SC1=NCCN1C", "1-methyl-4,5-dihydro-1H-imidazole-2-thiol"),
    ("OCCP(O)(O)=O", "(2-hydroxyethyl)phosphonic acid"),                  # a noncarbon acid is senior
    ("OCC1CCCN1S(N)(=O)=O", "2-(hydroxymethyl)pyrrolidine-1-sulfonamide"),  # a sulfonamide is senior
    ("NC(=N)N1CCN(CCO)CC1", "4-(2-hydroxyethyl)piperazine-1-carboximidamide"),
    ("C[Si](C)(C)CS", "(trimethylsilyl)methanethiol"),
    ("OC1CCCCC1", "cyclohexanol"),
    ("OCC1OC(O)C(O)C(O)C1O", "glucopyranose"),                             # a retained name: not read
    ("CC1(C)OC(=O)CC1O", "4-hydroxy-5,5-dimethyloxolan-2-one"),
    ("NCCO.Cl", "2-aminoethan-1-ol hydrochloride"),
    ("OCC1CCC(=O)N1", "5-(hydroxymethyl)pyrrolidin-2-one"),                # the lactam is the suffix
    ("COC(=O)C1CCC(=O)N1", "methyl 5-oxopyrrolidine-2-carboxylate"),       # an open ester: senior, exempt
    ("Cn1ccccc1=O", "1-methylpyridin-2(1H)-one"),                          # an aromatic c(=O): the suffix
    ("OCc1ccc2ccc(=O)oc2c1", "7-(hydroxymethyl)-2H-1-benzopyran-2-one"),
    ("S=C1NCCN1", "imidazolidine-2-thione"),
    ("O=CC1CCNC(=S)N1", "2-sulfanylidenehexahydropyrimidine-4-carbaldehyde"),   # the aldehyde is senior
    ("OC1CCN(CC1)C(N)=O", "4-hydroxypiperidine-1-carboxamide"),
    ("OCC1CCC(C#N)CC1", "4-(hydroxymethyl)cyclohexane-1-carbonitrile"),
    ("OCCc1ccncc1", "2-(pyridin-4-yl)ethan-1-ol"),
])
def test_other_names_pass(smiles, name):
    assert suffix_class_as_prefix_check(Chem.MolFromSmiles(smiles), name) is None, name
