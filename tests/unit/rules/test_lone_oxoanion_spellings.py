"""The oxoanions of one carbon-free centre, spelt on the anionic parent hydride of the oxygen, so a mixture beside them reads back.

 "Mixed organic – inorganic adducts" (the Blue Book): the mixture cites the names of its components. Inside
such a name OPSIN reads an '-ite' / '-ate' anion word as the neutral acid, so the mixture of a sulfite, phosphite,
perchlorate, bromate or periodate with neutral parts had no name. "Anionic centers on other characteristic
groups" (the Blue Book) names an anionic centre on a group "on the basis of the corresponding anionic parent hydrides";
these are oxoacid anions, which that rule leaves to the acid endings, so the spellings are valid names and no PIN
(``rules.hetero_group_anions.one_centre_chalcogenides``). A carbon-free P/S oxoacid group is cited by its retained prefix:
'sulfino', "Retained traditional prefixes" (the Blue Book, ":17971 sulfino (preselected prefix)").
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import name_best_effort, name_is_rt_exact


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("CCC(C)CC.O=S([O-])[O-]", "3-methylpentane—oxo-λ4-sulfanediylbis(oxidanide) (1/1)"),
    ("CCC(C)CC.[O-]P([O-])[O-]", "3-methylpentane—phosphanetriyltris(oxidanide) (1/1)"),
    # RDKit reads a halate as the charge-separated drawing (Cl+3 beside four O-); its group is the halogen acid group,
    # (the Blue Book): "O3Cl– perchloryl (preselected prefix)... The corresponding Br, F and I groups
    # are named in a similar manner"
    ("CCC(C)CC.[O-]Cl(=O)(=O)=O", "3-methylpentane—perchloryloxidanide (1/1)"),
    ("CCC(C)CC.[O-]Br(=O)=O", "3-methylpentane—bromyloxidanide (1/1)"),
    ("CCC(C)CC.[O-]I(=O)(=O)=O", "3-methylpentane—periodyloxidanide (1/1)"),
    ("CCC(C)CC.OS(=O)[O-]", "3-methylpentane—sulfinooxidanide (1/1)"),
])
def test_a_one_centre_oxoanion_beside_a_neutral_part(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,spellings", [
    ("O=S([O-])[O-]", ["oxo-λ4-sulfanediylbis(oxidanide)"]),
    # the first cited ligand has no enclosing marks,:7272)
    ("NP([O-])[O-]", ["aminophosphanediylbis(oxidanide)"]),
    # the preselected prefix of the group first; phosphoryl, Table 5.1:36044), then the λ drawing
    ("[O-]Cl(=O)(=O)=O", ["perchloryloxidanide", "(trioxo-λ7-chloranyl)oxidanide"]),
    ("O=P([O-])([O-])[O-]", ["phosphoryltris(oxidanide)", "oxo-λ5-phosphanetriyltris(oxidanide)"]),
    ("O=S([O-])O", ["sulfinooxidanide"]),
    # one anion on a neutral centre stays with the group-prefix path; a charged centre that is no halogen is declined
    ("O=[N+]([O-])[O-]", []),
    ("[O-]P(O)O", []),
    # an organic group keeps its spelling: the retained oxoacid prefix is asked for carbon-free groups only
    ("CS(=O)[O-]", ["(methanesulfinyl)oxidanide"]),
])
def test_the_spellings_of_an_anion(smiles, spellings):
    from orthonym.rules.hetero_group_anions import hetero_group_chalcogenide_spellings
    assert hetero_group_chalcogenide_spellings(Chem.MolFromSmiles(smiles)) == spellings


@pytest.mark.opsin_gate
def test_a_lone_anion_alone_keeps_its_name_and_a_salt_is_untouched():
    assert name_best_effort("O=S([O-])[O-]")["name"] == "sulfite"
    assert name_best_effort("[Na+].[Na+].O=S([O-])[O-]")["name"] == "disodium sulfite"
