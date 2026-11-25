"""A mixture of neutral parts and one carbon-free anion reads back once the anion is spelt on its anionic parent hydride.

 "Mixed organic – inorganic adducts" (the Blue Book): the mixture cites the names of its components.
Inside such a name OPSIN reads an '-ite' / '-ate' anion word as the neutral acid ('3-methylpentane—hypochlorite (1/1)'
gives HOCl beside the alkane), so the name never read back. The respelling step of the mixture names
(``rules.adducts._component_spellings``) now also offers the anion's name on the anionic parent hydride
, ``rules.hetero_group_anions``: 'chlorooxidanide'), which OPSIN keeps charged. The name is no PIN.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("CCC(C)CC.[O-]Cl", "3-methylpentane—chlorooxidanide (1/1)"),
    ("CCC(C)(C)CC.[O-]Cl", "3,3-dimethylpentane—chlorooxidanide (1/1)"),
])
def test_a_lone_oxoanion_beside_a_neutral_part(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
def test_a_lone_chloride_keeps_its_name():
    assert name_best_effort("CCO.[Cl-]")["name"] == "ethanol—chloride (1/1)"


@pytest.mark.opsin_gate
def test_a_mixture_another_spelling_repairs_keeps_its_name():
    """The anion's hydride spelling is offered after every other spelling (review of the lane: offered first, it
    renamed 19 mixtures that another spelling already repaired), and only in a best-effort request."""
    from orthonym import cli
    assert name_best_effort("CCO.OS(=O)(=O)[O-]")["name"] == (
        "ethanol—2-hydroxy-2-oxido-1,3-dioxa-2-thiapropa-1,2-diene (1/1)")
    assert cli._tiered_namer("pin", "complete").name_tiered("CCC(C)CC.[O-]Cl")["tier"] == "abstain"
