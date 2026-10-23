"""A radical centre on a ring atom is named as the ring's free-valence group, every other group a prefix.

 "Radicals... substituent group" (the Blue Book) and (the Blue Book), class 1
"Radicals": the free valence is the most senior class, so an amino, amide or hydroxy group on the same molecule is a prefix
('2,5-dioxopyrrolidin-1-yl', the Blue Book). The charged route cannot take a ring that carries a principal group (its
neutral parent becomes the '-ol' or '-amine'), so a ring nitrogen, ring boron (one electron) and a ring carbene carbon (two
electrons, '-ylidene', "General method", the Blue Book) take the group name of the whole molecule attached at the centre.
"""
import pytest

from tests.support.pin_tiers import name_default
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

BORON_RING = [
    ("[B]1C(N)C1", "2-aminoboriran-1-yl"),
    ("[B]1C(C(=O)O)C1", "2-carboxyboriran-1-yl"),
]

CARBENE_RING = "CN1CCN(CCO)[C]1"
CARBENE_STEREO = ("CC1=CC(=C(C(=C1)C)N2[C]N([C@H]([C@@H]2C3=CC=CC=C3)C4=CC=CC=C4)C5=CC=CC=C5C6=CC=CC=C6O)C")
BORON_DIAZABOROLE = "[B]1C(=NN=C1C(F)(F)F)/C(=C/C/C=C(/CN(C2=CC=CC=C2)C(=C)C3CN(C3)CCC)\\C=C)/C"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", BORON_RING)
def test_ring_boron_radical_with_a_group_is_a_prefix(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected, res
    assert name_is_rt_exact(expected, smiles)


@pytest.mark.opsin_gate
def test_carbene_ring_with_a_hydroxyl_is_a_ylidene():
    res = name_best_effort(CARBENE_RING)
    assert res["name"].endswith("ylidene"), res
    assert name_is_rt_exact(res["name"], CARBENE_RING)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [CARBENE_STEREO, BORON_DIAZABOROLE])
def test_ring_radical_with_a_principal_group_and_a_stereo_prefix_round_trips(smiles):
    res = name_best_effort(smiles)
    assert res["name"] and name_is_rt_exact(res["name"], smiles), res
    # Stereodescriptors inside a radical prefix have no derived PIN placement (radicals._ship): never a PIN.
    assert res["tier"] != "pin_verified", res
    assert name_default(smiles).get("tier") != "pin_verified"


def _declined(smiles):
    res = name_best_effort(smiles)
    assert res["name"] and name_is_rt_exact(res["name"], smiles), res


@pytest.mark.opsin_gate
@pytest.mark.xfail(strict=True, reason=(
    "a spiro ring system with a fused component and a stereocentre has no substituent name: "
    "name_ring_system_substituent returns None for the radical boron (the same molecule without the "
    "stereocentre is named), so only a stereo-less offer exists and the read-back refuses it"))
def test_spiro_borirane_radical_with_a_stereocentre():
    _declined("[B]1C([C@@]12C3=C(C=CC(=C3)C4=CSC=C4)C(=O)N(C2=O)C)NC5=CC=C(C=C5)CN6CCCCC6")


@pytest.mark.opsin_gate
@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "two spiro systems in a 46-atom molecule: the spiro assemblies are named only as separate parts and the "
    "parts never join into one name that reads back"))
def test_two_spiro_dioxanedione_and_dioxine_amide():
    _declined("CC(C)(C)OC(=O)N1CCC2(CC1)OC(=O)C(C(=O)O2)C(=O)NCC(=O)OC3=C(C(=O)OC4(O3)C5CC6CC(C5)CC4C6)"
              "C(=O)NCC(=O)O")


@pytest.mark.opsin_gate
@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "an iodine(III) ylide on a pyran ring bridged to a julolidine-type cage: the name built for the cage reads "
    "back with an extra NH; root cause not isolated"))
def test_iodane_pyran_julolidine_cage():
    _declined("CCCCI(C)C1=CC(=C(C#N)C#N)C=C(O1)/C=C/C2=CC3=C4C(=C2)C5(CS6(CC=C(C6)C)CC5CN4CCC3(C)C)C")


@pytest.mark.opsin_gate
@pytest.mark.xfail(strict=True, reason=(
    "indium is not admitted to the strict naming path in data/metal_scope.py; admitting a metal is a panel "
    "measurement and decision (CLAUDE.md, Scope), not part of this lane"))
def test_indium_dihydride_pyridinium_bromide():
    _declined("CCCC(C(CCC(C=C1CC(=O)[NH+](C=C1)CC2=CC=CC=C2)[InH2])OC)OC.[Br-]")
