"""The anions X=C=[N-] and X=[N-]: the azanide of their '-ylidene' group, never an acyl azanide.

 "Anionic centers on other characteristic groups" (the Blue Book): anionic centres on other
characteristic groups "are named on the basis of the corresponding anionic parent hydrides" ('acetylazanide (PIN)',
'hydroxyazanide (preselected name)'). The cumulated anion of HN=C=O has no acyl group (its N-C bond is double), so
it is the azanide of the one-carbon group =C=O ('(oxomethylidene)azanide'), and O=[N-] the azanide of =O
('oxoazanide'): ``rules.ylidene_azanide``. No name for these anions in the Blue Book is a PIN, so they ship below it.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("[N-]=C=O", "(oxomethylidene)azanide"),
    ("[N-]=C=S", "(sulfanylidenemethylidene)azanide"),
    ("[N-]=C=[Se]", "(selanylidenemethylidene)azanide"),
    ("[N-]=O", "oxoazanide"),
    ("C(=[N-])=S.[K+]", "potassium (sulfanylidenemethylidene)azanide"),
    ("C(=[N-])=O.C(=[N-])=O.C(=[N-])=O.[Fe+3]", "iron tris[(oxomethylidene)azanide]"),
])
def test_an_n_side_chalcogen_anion_is_the_azanide_of_its_ylidene_group(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


def test_only_a_single_bonded_acyl_carbon_makes_an_acyl_azanide():
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.ions import classify_anion
    for smiles, kind in (("CC(=O)[NH-]", "acyl_azanide"), ("[N-]=C=O", "iminide"), ("[N-]=C=S", "iminide")):
        mol = Chem.MolFromSmiles(smiles)
        assert classify_anion(mol, get_ion_sites(mol)["anions"][0]) == kind, smiles


def test_the_acyl_azanide_keeps_its_pin():
    r = name_best_effort("CC(=O)[N-]C")
    assert r["name"] == "acetyl(methyl)azanide" and r["tier"] == "pin_verified", r


@pytest.mark.parametrize("smiles,name", [
    ("S=C=N[Zn]N=C=S", "diisothiocyanatozinc"),
    ("O=C=N[Ag]", "isocyanatosilver"),
    ("O=N[Ru](Cl)(Cl)Cl", "trichlorido(nitroso)ruthenium"),
])
def test_a_group_drawn_bonded_to_a_metal_keeps_the_name_of_the_drawing(smiles, name):
    """The metal offer names the ionic drawing beside the drawing as written and takes the ionic name on a tie; the
    new anion names are not offered inside that run (``metal_offer.INSIDE_METAL_REWRITE``), so the name as drawn stays."""
    r = name_best_effort(smiles)
    assert r["name"] == name, r
