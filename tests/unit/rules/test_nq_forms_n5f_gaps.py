"""Lane L2 proper fix: locants no isomer needs, in the lane's writers (roadmap N5f).

* (b) (the Blue Book) "in monosubstituted homogeneous chains consisting of
  only two identical atoms"; 'methoxyethane (PIN)' (:27745).
* (:2953); 'vinyl (ethenyl)' (:24547); 'ethenylbenzene (PIN)' (:2002);
  '2-chloroethen-1-yl (preferred prefix)' (:3003).
"""
import pytest
from rdkit import Chem

from orthonym.assembly.universal_substituent import (
    name_universal_substituent_prefix, name_universal_substitutive)
from orthonym.rules.terminal_fragment import terminal_fragment_name


@pytest.mark.parametrize("smiles,expected", [
    ("CCOC", "methoxyethane"),
    ("CCCl", "chloroethane"),
    ("CCNCC", "(ethylamino)ethane"),
    ("CC(Cl)Cl", "1,1-dichloroethane"),          # disubstituted: the locants stay
])
def test_a_monosubstituted_two_carbon_parent_cites_no_locant(smiles, expected):
    assert name_universal_substitutive(Chem.MolFromSmiles(smiles)).name == expected


@pytest.mark.parametrize("smiles,frag,attach,expected", [
    ("C=Cc1ccccc1", {0, 1}, 1, "ethenyl"),
    ("ClC=Cc1ccccc1", {0, 1, 2}, 2, "2-chloroethen-1-yl"),
    ("C=COCc1ccccc1", {0, 1, 2, 3}, 3, "(ethenyloxy)methyl"),
])
def test_ethene_cites_no_locant_it_does_not_need(smiles, frag, attach, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert terminal_fragment_name(mol, frag, attach).name == expected
    assert name_universal_substituent_prefix(mol, sorted(frag), attach) == expected
