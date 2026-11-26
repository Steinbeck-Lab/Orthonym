"""Iodides and monohalides of a mononuclear parent hydride.

 "Halogen atoms attached to heteroatoms" (the Blue Book),: "The prefixes 'bromo', 'chloro',
'fluoro', and 'iodo' are used to name halogen compounds when the halogen atoms are attached to B, Al, In, Ga, Tl, Si,
Ge, Sn, Pb, and Bi." Iodine is a hub element of its own (λ3-iodane), so an iodide had several hub atoms and no name; a
terminal iodine is now an 'iodo' substituent of the one hub that is no halogen. A hub that is no halogen takes one
halogen too ('chlorostannane'); a halogen hub still needs two (ICl, IBr stay out). A carbon-free name is never a PIN
; 'diiododi(methyl)stannane' takes the label of 'dichlorodi(methyl)stannane' (same producer, same read-back).
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("I[Sn](I)(I)I", "tetraiodostannane"),
    ("I[Al](I)I", "triiodoalumane"),
    ("I[Bi](I)I", "triiodobismuthane"),
    ("Cl[SnH3]", "chlorostannane"),
    ("Cl[AlH2]", "chloroalumane"),
])
def test_a_carbon_free_hydride_halide_is_named_below_the_pin(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
def test_the_iodide_twin_takes_the_label_of_the_chloride():
    for smiles, name in (("C[Sn](C)(Cl)Cl", "dichlorodi(methyl)stannane"), ("C[Sn](C)(I)I", "diiododi(methyl)stannane")):
        r = name_best_effort(smiles)
        assert (r["name"], r["tier"]) == (name, "pin_verified"), r


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [("CI", "iodomethane"), ("II", "molecular iodine")])
def test_iodine_compounds_outside_the_class_keep_their_names(smiles, name):
    assert name_best_effort(smiles)["name"] == name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["ICl", "IBr"])
def test_a_diatomic_interhalogen_is_still_not_named_here(smiles):
    assert name_best_effort(smiles)["tier"] == "abstain"


@pytest.mark.opsin_gate
def test_a_one_halogen_sulfur_hub_gives_no_pin():
    """ names halogen compounds with halogen prefixes only on B, Al, In, Ga, Tl, Si, Ge, Sn, Pb, Bi; 'CH3-S-Cl
    methyl thiohypochlorite (PIN; see ' (the Blue Book), so a chlorosulfanyl prefix is no PIN."""
    from orthonym import cli
    smiles = "ClSC1CCCC2CCCCC12"
    assert cli._tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"
    r = name_best_effort(smiles)
    assert r["tier"] == "systematic_verified" and not r["is_pin"], r


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["IS(I)(I)C1CCCC2CCCCC12", "IS(I)(I)(I)(I)C1CCCC2CCCCC12"])
def test_an_iodo_sulfur_group_of_a_larger_molecule_gives_no_pin(smiles):
    """ (the Blue Book): halogens on phosphorus or chalcogen atoms make acid halides or esters of inorganic
    acids; the iodide rule names such a hub only as a carbon-free compound or part, never as the group of a larger one."""
    from orthonym import cli
    assert cli._tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"
    assert not name_best_effort(smiles)["is_pin"]


@pytest.mark.opsin_gate
def test_a_carbon_free_tellurium_iodide_part_of_a_mixture_is_named():
    assert name_best_effort("CCO.I[Te](I)(I)I")["name"] == "ethanol—tetraiodo-λ4-tellane (1/1)"
