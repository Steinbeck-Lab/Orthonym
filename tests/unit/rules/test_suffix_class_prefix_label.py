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
    ("OCC1CCCCN1C(C)=O", "1-acetyl-2-(hydroxymethyl)piperidine"),        # an amide (ring-N acyl) is senior
    ("OCC1CCC(C#N)CC1", "4-(hydroxymethyl)cyclohexane-1-carbonitrile"),
    ("OCCc1ccncc1", "2-(pyridin-4-yl)ethan-1-ol"),
])
def test_other_names_pass(smiles, name):
    assert suffix_class_as_prefix_check(Chem.MolFromSmiles(smiles), name) is None, name
