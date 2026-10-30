"""Labelled molecules whose skeleton name took no isotopic descriptor.

- A substituent prefix that opens with a primed locant set, '(9,9'-spirobi[fluoren]-4-yl)', had no slot in front of
  the set (``rules.isotope_scopes._candidate_offsets``): a label on the primed component is cited there, as for a
  parent, the Blue Book), '[(2'-2H)-9,9'-spirobi[fluoren]-4-yl]'.
- The decorator is given a second skeleton when the first takes no descriptor (``rules.isotope_alternative``): a
  hydrocarbon-series member OPSIN draws in the other Kekule set ('tetraphenylen-2-yl') and a senior mononuclear hub
  whose like substituents the labels make unlike ('bis[...]di(phenyl)silane') give their place to the next producer.
- Of two identical groups the modified one is cited first, "Different isotopic modifications on otherwise
  identical substituents" (the Blue Book): "The isotopically modified substituent is preferred alphabetically
  to the unmodified substituent"; the reading of a symmetric skeleton that cites the descriptor earlier wins a tie
  of locants (``isotope_scopes._lowest_locants``), and every reading is tried.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    pytest.param('c1ccc(nc1)-c1cccc2c1-c1ccccc1C21c2ccccc2-c2ccc([2H])cc21',
                 "2-[(2'-2H)-9,9'-spirobi[fluoren]-4-yl]pyridine",
                 id='spirobi_label_on_the_substituent'),
    pytest.param('[2H]C1=C(C2=C(C(=C1)N(C3=C(C(=C(C(=C3[2H])[2H])C4=C(C(=C(C5=C4OC6=C(C(=C(C(=C56)[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])C7=C(C(=C(C8=C7C9=C(C(=C(C(=C9C8(C([2H])([2H])[2H])C([2H])([2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])C1=C(C(=C(C(=C1C21C2=C(C(=C(C(=C2C2=C(C(=C(C(=C12)[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H]',
                 "4-[4-({9,9-di[(2H3)methyl](2H7)-9H-fluoren-4-yl}{(3',4'-2H2)-9,9'-spirobi[(1,2,5,6,7,8-2H6)fluoren]-4-yl}amino)(2H4)phenyl](2H7)dibenzo[b,d]furan",
                 id='spirobi_cid_176725668'),
    pytest.param('[2H]c1c([2H])c([2H])c2c(c1[2H])-c1c([2H])c([2H])c([2H])c([2H])c1-c1c([2H])c([2H])c(-c3ccc4c(c3)c3ccccc3n4C)c([2H])c1-c1c([2H])c([2H])c([2H])c([2H])c1-2',
                 '9-methyl-3-[(2H15)pentacyclo[18.4.0.0^2,7.0^8,13.0^14,19]tetracosa-1(24),2,4,6,8,10,12,14,16,18,20,22-dodecaen-4-yl]-9H-carbazole',
                 id='tetraphenylenyl_fully_labelled'),
    pytest.param('[2H]C([2H])([2H])C1=C(C=C(C=C1)[Si](C2=CC=CC=C2)(C3=CC=CC=C3)C4=CC(=C(C=C4)C)C5=C6C(=CC=C5)C(C7=CC=CC=C7O6)(C)C([2H])([2H])[2H])C8=C9C(=CC=C8)C(C1=CC=CC=C1O9)(C)C([2H])([2H])[2H]',
                 '4-[5-({3-[9-(2H3)methyl-9-methyl-9H-xanthen-4-yl]-4-(2H3)methylphenyl}di(phenyl)silyl)-2-methylphenyl]-9-(2H3)methyl-9-methyl-9H-xanthene',
                 id='senior_hub_cid_159371810'),
    pytest.param('[2H]C1(CCCC1)C2=C(C=CC(=C2)[Si](C3=CC=CC=C3)(C4=CC=CC=C4)C5=CC(=C(C=C5)C6=C7C(=CC=C6)C(C8=CC=CC=C8O7)(C)C([2H])([2H])[2H])C9(CCCC9)[2H])C1=C2C(=CC=C1)C(C1=CC=CC=C1O2)(C)C',
                 '4-[2-(1-2H)cyclopentyl-4-({3-(1-2H)cyclopentyl-4-[9-(2H3)methyl-9-methyl-9H-xanthen-4-yl]phenyl}di(phenyl)silyl)phenyl]-9,9-dimethyl-9H-xanthene',
                 id='senior_hub_cid_161429857'),
    pytest.param('c1ccc(cc1)[Si](c1ccccc1)(c1ccc(-c2ccccc2)cc1)c1ccc(-c2ccccc2[2H])cc1',
                 "[(2'-2H)[1,1'-biphenyl]-4-yl]([1,1'-biphenyl]-4-yl)di(phenyl)silane",
                 id='modified_group_cited_first'),
])
def test_the_labelled_molecule_is_named(smiles, expected):
    row = name_best_effort(smiles)
    assert row["name"] == expected, row
    assert row["tier"] == "systematic_verified", row
    assert name_is_rt_exact(row["name"], smiles), row


def test_the_nuclide_of_higher_mass_number_takes_the_lower_parent_locant():
    """ (the Blue Book): "The preferred IUPAC name is based on the senior parent structure that has the
    lowest locant(s) for nuclides of higher mass number", '4-[2-13C]ethyl-2-(2-[2-14C]ethylpentyl)heptan-1-ol (PIN)
    [not 4-[2-14C]ethyl-2-(2-[2-13C]ethylpentyl)heptan-1-ol]'. The two readings of the symmetric skeleton cite the
    same locants inside their descriptors; the one with 14C in the prefix at 2 wins (``isotope_scopes._reading_key``)."""
    smiles = "CCCC(C[13CH3])CC(CO)CC(C[14CH3])CCC"
    row = name_best_effort(smiles)
    assert row["name"] == "4-(2-13C)ethyl-2-[2-(2-14C)ethylpentyl]heptan-1-ol", row
    assert name_is_rt_exact(row["name"], smiles), row
