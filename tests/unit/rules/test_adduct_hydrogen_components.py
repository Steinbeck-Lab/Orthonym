"""Hydrogen, hydride and ammonia components of a mixture, and the neutral metal atom of a charged assembly.

 (the Blue Book): "Names are formed by citing the names of individual compounds in the order of the formula
connected by long (em) dashes". gives a mixed organic-inorganic adduct no PIN (:4667), so these names are made at
the best-effort tier only, and each ships only when OPSIN reads it back to the drawn structure. Dihydrogen and the hydride
ion have no heavy atom and no name from the single-component pipeline; bare nitrogen stays out of the PIN table. A neutral
metal atom beside charged components keeps the oxidation number zero in the name ('zinc(0)'): OPSIN reads 'zinc—benzenide
(1/2)' as zinc(2+), 'zinc(0)—benzenide (1/2)' as the neutral atom.
"""
import pytest

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

ROWS = [
    ("[HH].CCO", "ethanol—dihydrogen (1/1)"),
    ("[HH].[HH].CCNC1=C(N=CC(=C1)C(C)C)N", "N3-ethyl-5-(propan-2-yl)pyridine-2,3-diamine—dihydrogen (1/2)"),
    ("N.CCO", "ethanol—ammonia (1/1)"),
    ("[Na+].[H-].CCO", "ethanol—sodium(1+)—hydride (1/1/1)"),
    ("[Zn].[c-]1ccccc1.[c-]1ccccc1", "benzenide—zinc(0) (2/1)"),
]


def _reads_dihydrogen() -> bool:
    """OPSIN main reads dihydrogen as a component of a mixture; the release 2.9.0 reads it alone only, and there
    such a name fails closed."""
    from orthonym.namer import _validity_gate_name_to_smiles
    return bool(_validity_gate_name_to_smiles("ethanol—dihydrogen (1/1)"))


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_best_effort_name(smiles, expected):
    res = name_best_effort(smiles)
    if "dihydrogen" in expected and not _reads_dihydrogen():
        assert res["name"] is None, res          # the read-back cannot confirm it: no name, never a wrong one
        return
    assert res["name"] == expected and res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res


@pytest.mark.parametrize("smiles,_expected", ROWS)
def test_pin_tier_ships_no_pin(smiles, _expected):
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] != "pin_verified"


@pytest.mark.parametrize("smiles,expected", [
    # no charged component: the metal atom keeps its bare element name
    ("[Zn].CCO", "ethanol—zinc (1/1)"),
    # dihydrogen spelled with two hydrogen atoms keeps the pipeline's name
    ("CCO.[H][H]", "ethanol—molecular hydrogen (1/1)"),
    ("O.CCO", "ethanol—water (1/1)"),
])
def test_unchanged(smiles, expected):
    assert name_best_effort(smiles)["name"] == expected


def test_zero_oxidation_number_is_for_metals_only():
    # boron is no metal: no 'boron(0)' (the name stays as main gives it, or none)
    name = name_best_effort("[B].[c-]1ccccc1")["name"] or ""
    assert "boron(0)" not in name


def test_unreadable_hydrogen_spellings_decline():
    # OPSIN reads no name for the proton or for isotopic dihydrogen: those mixtures stay unnamed
    for smiles in ("[H+].Oc1ccccc1", "[2HH].CCO"):
        assert not name_best_effort(smiles)["name"]
