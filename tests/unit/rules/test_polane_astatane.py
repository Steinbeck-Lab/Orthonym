"""The mononuclear hydrides of polonium and astatine "Systematic names", Table 2.1, the Blue Book).

Table 2.1 names PoH2 'polane' and AtH 'astane'; OPSIN reads 'polane' and 'astatane', not 'astane', and a name ships
only when it reads back, so AtH is 'astatane'. Carbon-free names are preselected at most, never PINs.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [("[PoH2]", "polane"), ("[AtH]", "astatane"), ("[PoH2].O", "polane—water (1/1)")])
def test_polonium_and_astatine_hydrides(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


@pytest.mark.parametrize("smiles,name", [("F[At](F)F", "trifluoro-λ3-astatane"), ("Cl[At](Cl)Cl", "trichloro-λ3-astatane")])
def test_an_astatine_halide_cites_its_bonding_number(smiles, name):
    """P-14.1 / P-31.1.4.2: a non-standard bonding number is cited as λn, as 'trifluoro-λ3-iodane'."""
    r = name_best_effort(smiles)
    assert r["name"] == name and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)
