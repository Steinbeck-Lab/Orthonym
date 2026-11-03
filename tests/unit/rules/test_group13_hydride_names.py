"""Group 13 organyls on the preselected parent hydrides alumane, gallane, indigane and thallane.

 (the Blue Book): "Although systematic names alumane, gallane, indigane and thallane are preselected names,
the names based on these parent hydrides currently do not have PIN status. However such names can be used in general
nomenclature"; (:1671) "trimethylalumane for (CH3)3Al"; (:27236) '(CH3-CH2)2AlOH diethylalumanol'. A centre
that is no derivative of its hydride (a radical, an ion, a double bond) keeps the metal name."""
import pytest

from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("C[Al](C)C", "trimethylalumane"),
    ("CC[Al](CC)O", "diethylalumanol"),
    ("C[AlH]c1ccccc1", "methyl(phenyl)alumane"),
    ("CC(C)C[Al](CC(C)C)CC(C)C", "tris(2-methylpropyl)alumane"),
    ("CC(C)O[Al](OC(C)C)OC(C)C", "tris[(propan-2-yl)oxy]alumane"),
    ("CC[Al](CC)Cl", "chlorodi(ethyl)alumane"),
    ("CC[Ga](CC)O", "diethylgallanol"),
    ("C[In](O)C", "dimethylindiganol"),
    # a cation drawn beside its anion is named through its bonded drawing (CONNECT_ANION)
    ("[Cl-].C[Al+]C", "chlorodi(methyl)alumane"),
    ("C[Al+]C", "dimethylaluminium(1+)"),          # an ion keeps the metal name
    ("CCO[Al+]C(=O)C", "acetyl(ethoxy)alumanylium"),  # where no metal name is built, the cation of alumane
    # no derivative of the hydride: the metal name, the first simple ligand bare,:7272)
    ("C[Al]c1ccccc1", "methyl(phenyl)aluminium"),
    ("O=[AlH]", "hydrido(oxo)aluminium"),
    ("C[Al+2].[O-2]", "methylaluminium(2+) oxide"),
])
def test_group13_names(smiles, name):
    row = name_best_effort(smiles)
    assert row["name"] == name, row
    assert not row["is_pin"], row
    assert_full_rt(row["name"], smiles)
