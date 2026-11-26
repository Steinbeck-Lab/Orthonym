"""The azanidyl rung of the floor names the free amide anions of a metal's ionic drawing, as it does at top level.

The rung (``assembly.universal_substituent._azanidyl_rung``; "Anionic centers in both parent compounds and
substituent groups", the Blue Book: "one anion must be chosen as the parent anion and the other expressed as anionic
substituent group(s)") used to decline in every run of another drawing of a metal input, to keep a metal drawn bonded to a
nitrogen named as drawn. That name is kept by ``rules.ylidene_azanide`` ('diisothiocyanatozinc'), and the decline left a
halide of ruthenium or zirconium beside free amide anions with no name: its ionic drawing named at top level but not
inside the offer. Every name is of another drawing, so below the PIN.
"""
import pytest

from tests.support.rt_assert import name_best_effort


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("S=C=N[Zn]N=C=S", "diisothiocyanatozinc"),
    ("O=C=N[Cd]N=C=O", "diisocyanatocadmium"),
])
def test_a_metal_bonded_to_the_nitrogen_keeps_its_name_as_drawn(smiles, name):
    assert name_best_effort(smiles)["name"] == name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,tail", [
    ("[CH3-].CCCC(CCC)CS(=O)(=O)[N-]C(C1=CC=CC=C1)C(C2=CC=CC=C2)[NH-].Cl[Ru+3]",
     "ethyl}benzene—methanide—chloride—ruthenium(4+) (1/1/1/1)"),
    ("CC(=O)[N-]CC[N-]C(C)=O.Cl[Zr+2]Cl", "zirconium 1-{[2-(acetylazanidyl)ethyl]azanidyl}-1-oxoethane dichloride"),
])
def test_a_metal_halide_beside_free_amide_anions_is_named_through_its_ionic_drawing(smiles, tail):
    r = name_best_effort(smiles)
    assert r["tier"] == "systematic_verified" and not r["is_pin"] and r["name"].endswith(tail), r
