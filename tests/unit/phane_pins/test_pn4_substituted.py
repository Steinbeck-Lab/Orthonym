import pytest
from rdkit import Chem

from orthonym.rules.phane_pin import general_phane_name

from ._helpers import mol, needs_phane_opsin, substituted


@pytest.mark.parametrize("line,name,smiles", substituted("prefix"), ids=lambda v: str(v)[:30])
def test_book_prefix_names_with_composite_locants(line, name, smiles):
    # -.3: 1^4,1^5,1^6,3,3,4^2,4^3-heptachloro-... and its two companions
    assert general_phane_name(mol(smiles)) == name


@pytest.mark.parametrize("line,name,smiles", substituted("suffix"), ids=lambda v: str(v)[:30])
def test_book_suffix_name_with_ene(line, name, smiles):
    # (e)(ii)::29005 '...phan-11-en-2-one'
    assert general_phane_name(mol(smiles)) == name


@pytest.mark.parametrize("smiles,name", [
    ("C1(CC2=CC=C(C=C2)CCC2=CC=C1C=C2)O", "1,4(1,4)-dibenzenacyclohexaphan-2-ol"),
    ("C1(CC2=CC=C(C=C2)CCC2=CC=C1C=C2)C(=O)O", "1,4(1,4)-dibenzenacyclohexaphane-2-carboxylic acid"),
    ("C1(CC2=CC=C(C=C2)CCC2=CC=C1C=C2)N", "1,4(1,4)-dibenzenacyclohexaphan-2-amine"),
    ("C1(CC2=CC=C(C=C2)C(CC2=CC=C1C=C2)=O)=O", "1,4(1,4)-dibenzenacyclohexaphane-2,5-dione"),
    ("CC1=CC2=CC=C1CCC1=CC=C(C=C1)CC2", "1^2-methyl-1,4(1,4)-dibenzenacyclohexaphane"),
])
def test_suffix_classes_of_the_reader(smiles, name):
    # (c): the principal characteristic group is the suffix; OPSIN main reads each of these names
    assert general_phane_name(mol(smiles)) == name


@pytest.mark.parametrize("name,smiles", [
    ("1,4(1,4)-dibenzenacyclohexaphane-2,5-diene", "C1=Cc2ccc(cc2)C=Cc2ccc1cc2"),                           #:49692
    ("1,9(1,3)-dibenzenacyclohexadecaphan-2-ene", "C1=CCCCCCc2cccc(c2)CCCCCCCc2cccc1c2"),                   #:16821
    ("1,7(1,3)-dibenzenacyclotridecaphan-4-en-2-yne", "C1#CC=CCc2cccc(c2)CCCCCCc2cccc1c2"),                 #:16829
    ("1,7(1,3)-dibenzenacyclotridecaphan-2-en-5-yne", "C1=CCC#Cc2cccc(c2)CCCCCCc2cccc1c2"),                 #:16833
    # (a) (:7595): 'dien' and 'diyne' start with a consonant, so 'phane' keeps its 'e'
    ("1,4(1,4)-dibenzenacycloundecaphane-5,7-dien-9-yne", "c1cc2ccc1C=CC=CC#CCc1ccc(cc1)CC2"),
    ("1,4(1,4)-dibenzenacycloundecaphan-5-en-7,9-diyne", "c1cc2ccc1C=CC#CC#CCc1ccc(cc1)CC2"),
    ("1,4(1,4)-dibenzenacyclononaphane-5,7-diyne", "c1cc2ccc1C#CC#CCc1ccc(cc1)CC2"),
    ("1,4(1,4)-dibenzenacyclododecaphane-5,7-dien-9,11-diyne", "c1cc2ccc1C=CC=CC#CC#Cc1ccc(cc1)CC2"),
])
def test_the_endings_agree_with_the_legacy_builder(name, smiles):
    # the ene and yne endings are spelled twice (rules.phane._unsaturated_phane_name:1127 and naming.endings): they must agree
    from orthonym.rules.phane import build_phane_pin
    assert general_phane_name(mol(smiles)) == build_phane_pin(mol(smiles)) == name


@needs_phane_opsin
@pytest.mark.parametrize("line,name,smiles", substituted("prefix") + substituted("suffix"),
                         ids=lambda v: str(v)[:30])
def test_round_trip(line, name, smiles):
    from tests.support.rt_assert import assert_full_rt
    assert_full_rt(general_phane_name(mol(smiles)), smiles)


@pytest.mark.parametrize("line,name,smiles",
                         substituted("indicated_hydrogen") + substituted("hydro"),
                         ids=lambda v: str(v)[:30])
def test_hydro_and_indicated_hydrogen_decline_until_pn4c(line, name, smiles):
    assert general_phane_name(mol(smiles)) is None


@pytest.mark.parametrize("smiles", [
    "C1(CC2=CC=C(C=C2)CCC2=CC=C1C=C2)C(=O)OC",         # an ester: two-word names, not built
    "C1(CC2=CC=C(C=C2)CCC2=CC=C1C=C2)C(=O)NC",         # an N-substituted amide
    "C1(CC2=CC=C(C=C2)CCC2=CC=C1C=C2)NC",              # a secondary amine
])
def test_suffix_groups_that_are_not_built_decline(smiles):
    assert general_phane_name(mol(smiles)) is None


@pytest.mark.parametrize("smiles,name", [
    # (g) (:3307): the substituent cited first in the name takes the lowest locants
    ("CCc1cc2ccc1CCc1ccc(cc1F)CC2", "1^2-ethyl-4^2-fluoro-1,4(1,4)-dibenzenacyclohexaphane"),
    ("CCc1cc2ccc1CCc1ccc(c(F)c1)CC2", "1^2-ethyl-4^3-fluoro-1,4(1,4)-dibenzenacyclohexaphane"),
])
def test_the_prefix_cited_first_takes_the_lower_locant(smiles, name):
    assert general_phane_name(mol(smiles)) == name


@pytest.mark.parametrize("smiles", [
    "c1cc2ccc1CN(O)c1ccc(cc1)CC2",              # a heterol: 2-aza-...phan-2-ol,:27230; piperidin-1-ol:27238)
    "OC1Cc2ccc(cc2)CN(O)c2ccc1cc2",             # two hydroxy groups, one on N: a diol, never a prefix
    "NC1Cc2ccc(cc2)CN(O)c2ccc1cc2",             # the heterol is senior to the amine (class 17 before 19)
    "c1cc2ccc1C[SiH](O)c1ccc(cc1)CC2",          # a silanol
    "c1cc2ccc1CN(N)c1ccc(cc1)CC2",              # an amino group on N: (:26213) makes it a suffix
    "OC(=O)C1Cc2ccc(cc2)CB(O)c2ccc1cc2",        # a boron hydroxy group is no plain prefix
])
def test_a_hydroxy_or_amino_group_on_a_heteroatom_is_not_cited_as_a_prefix(smiles):
    assert general_phane_name(mol(smiles)) is None


@pytest.mark.parametrize("smiles,name", [
    ("OC(=O)C1Cc2ccc(cc2)CN(O)c2ccc1cc2", "2-hydroxy-2-aza-1,4(1,4)-dibenzenacyclohexaphane-6-carboxylic acid"),
    ("O=C1Cc2ccc(cc2)CN(O)c2ccc1cc2", "2-hydroxy-2-aza-1,4(1,4)-dibenzenacyclohexaphan-6-one"),
])
def test_a_heteroatom_hydroxy_group_is_a_prefix_under_a_senior_class(smiles, name):
    # the principal group is an acid or a ketone: class 7 and 16 come before the alcohols (class 17)
    assert general_phane_name(mol(smiles)) == name


@needs_phane_opsin
@pytest.mark.parametrize("smiles", [
    "OC(=O)C1Cc2ccc(cc2)CN(O)c2ccc1cc2",
    "O=C1Cc2ccc(cc2)CN(O)c2ccc1cc2",
])
def test_a_heteroatom_prefix_round_trips(smiles):
    from tests.support.rt_assert import assert_full_rt
    assert_full_rt(general_phane_name(mol(smiles)), smiles)


DIAZA = ("C1=CC=C(C=C1)N2C3=CC4=C(C=C3)C=CC(=C4N(C5=CC6=C(C=C5)C=CC(=C62)C7=CC=CC(=C7)C8=CN=CC=C8)C9=CC=CC=C9)"
         "C1=CC=CC(=C1)C1=CN=CC=C1")      # two naphthalenes joined by two N-phenyl atoms: RDKit reads the macrocycle as aromatic


def test_a_macrocycle_that_rdkit_reads_as_aromatic_is_named_from_its_kekule_form():
    # the bonds between the naphthalenes and the N atoms are single amplificants, 'a' prefixes)
    m = mol(DIAZA)
    assert any(b.GetBondType() == Chem.BondType.AROMATIC and {m.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol(),
                                                                m.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol()} == {"C", "N"}
               for b in m.GetBonds())
    assert general_phane_name(m) == ("2,4-diphenyl-1^2,3^2-bis[3-(pyridin-3-yl)phenyl]-2,4-diaza-"
                                     "1(1,7),3(7,1)-dinaphthalenacyclotetraphane")


@needs_phane_opsin
def test_the_kekule_named_macrocycle_round_trips():
    from tests.support.rt_assert import assert_full_rt
    assert_full_rt(general_phane_name(mol(DIAZA)), DIAZA)
