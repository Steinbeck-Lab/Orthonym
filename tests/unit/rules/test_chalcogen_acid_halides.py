"""Halides of the mononuclear oxoacids of sulfur, selenium and tellurium (``rules.chalcogen_acid_halides``).

 "Acid halides and pseudohalides" (the Blue Book): the name is the acid name with the class
name(s) of the halides, "Exceptionally... halides... with identical atoms or groups and derived from phosphoric
acid, sulfuric acid, selenic acid and telluric acid are named by adding the class name(s) to the acyl group name
'phosphoryl', 'sulfuryl',... 'selenonyl' and 'telluronyl'". A compound without carbon gets no PIN.
"""
import pytest

from orthonym import Orthonym
from orthonym.rules.chalcogen_acid_halides import name_chalcogen_oxoacid_halide
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("O=S(Cl)Cl", "sulfurous dichloride"),              # thionyl chloride: sulfurous acid is not one of the four
    ("[O-][S+](Cl)Cl", "sulfurous dichloride"),         # the charge-separated drawing of the same compound
    ("O=S(Cl)F", "sulfurous chloride fluoride"),
    ("O=S(=O)(F)Cl", "sulfuric chloride fluoride"),     # not identical: the acid name
    ("O=S(=O)(Br)Br", "sulfuryl dibromide"),            # identical: the acyl word
    ("O=[Se](Cl)Cl", "selenous dichloride"),
    ("O=[Se](=O)(F)F", "selenonyl difluoride"),
    ("O=[Te](Cl)Cl", "tellurous dichloride"),
    ("CCO.O=S(Cl)Cl", "ethanol—sulfurous dichloride (1/1)"),
])
def test_named_by_the_acid_and_its_halides(smiles, name):
    r = name_best_effort(smiles)
    assert (r["name"], r["tier"], r["is_pin"]) == (name, "systematic_verified", False), r
    assert name_is_rt_exact(name, smiles)
    assert Orthonym().name_tiered(smiles)["tier"] == "abstain"


@pytest.mark.parametrize("smiles", ["CS(=O)Cl", "O=S(Cl)N", "OS(=O)Cl", "O=S(=O)(Cl)Cl.O=S(Cl)Cl", "[2H]S(=O)Cl",
                                    "O=S(Cl)[35Cl]"])
def test_other_shapes_are_left_to_their_producers(smiles):
    from rdkit import Chem
    assert name_chalcogen_oxoacid_halide(Chem.MolFromSmiles(smiles)) is None


def test_without_the_producer_the_chain_name_returns(monkeypatch):
    """Mutation: the producer gives nothing, thionyl chloride is named as a heteroatom chain again."""
    from orthonym.rules import chalcogen_acid_halides as M
    monkeypatch.setattr(M, "name_chalcogen_oxoacid_halide", lambda mol: None)
    assert name_best_effort("O=S(Cl)Cl")["name"] == "1,1-dichloro-2-oxa-1-thiaeth-1-ene"


@pytest.mark.parametrize("smiles, name", [
    ("O=[34S](Cl)Cl", "(34S)sulfurous dichloride"),                       # a label on the centre
    ("[18O]=S(Cl)Cl", "1,1-dichloro(18O)-2-oxa-1-thiaeth-1-ene"),         # OPSIN reads no '(18O)sulfurous'
    ("O=S(Cl)[37Cl]", "1-(37Cl)chloro-1-chloro-2-oxa-1-thiaeth-1-ene"),   # nor a label on a class word
])
def test_a_label_the_class_name_cannot_carry_keeps_the_chain_name(smiles, name):
    assert name_best_effort(smiles)["name"] == name
