"""An acid salt of a disulfonic or tricarboxylic acid cites the free acid as a 'sulfo' prefix of the anion (method (1)).

IUPAC 2013 Acid salts (the Blue Book), "Method (1) generates preferred IUPAC names":
'potassium 6-carboxyhexanoate (PIN)' (:31602); the same construction for -SO3H.
"""

import pytest
from rdkit import Chem

from orthonym.rules.acid_salt_anions import (
    name_dicarboxylate_acid_salt_anion,
    name_sulfonate_acid_salt_anion,
)
from orthonym.rules.salts import _anion_word_expresses_acid_hydrons
from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("OS(=O)(=O)CCS(=O)(=O)[O-]", "2-sulfoethane-1-sulfonate"),
    ("OS(=O)(=O)CCCS(=O)(=O)[O-]", "3-sulfopropane-1-sulfonate"),
    ("OS(=O)(=O)CC(C)S(=O)(=O)[O-]", ""),          # a branch: not built here
    ("OS(=O)(=O)CCS(=O)(=O)O", ""),                # no anion
])
def test_builder(smiles, name):
    assert name_sulfonate_acid_salt_anion(Chem.MolFromSmiles(smiles)) == name


def test_the_sulfo_prefix_states_the_acid_hydron():
    assert _anion_word_expresses_acid_hydrons("2-sulfoethane-1-sulfonate")
    assert not _anion_word_expresses_acid_hydrons("ethane-1,2-disulfonate")


@pytest.mark.opsin_gate
def test_salt_pin():
    assert_pin_at_both_tiers("OS(=O)(=O)CCS(=O)(=O)[O-].[Na+]", "sodium 2-sulfoethane-1-sulfonate")


@pytest.mark.parametrize("smiles,name", [
    ("[O-]C(=O)CC(C(=O)[O-])C(=O)O", "2-carboxybutanedioate"),
    ("[O-]C(=O)CCC(C(=O)O)CC(=O)[O-]", "3-carboxyhexanedioate"),
    ("[O-]C(=O)CC(=O)[O-]", ""),                   # no free acid
    ("[O-]C(=O)CC(C(=O)O)C(C)C(=O)[O-]", ""),      # a branch: not built here
])
def test_dicarboxylate_builder(smiles, name):
    assert name_dicarboxylate_acid_salt_anion(Chem.MolFromSmiles(smiles)) == name


@pytest.mark.opsin_gate
def test_dicarboxylate_salt_pin():
    assert_pin_at_both_tiers("[O-]C(=O)CC(C(=O)[O-])C(=O)O.[Na+].[Na+]", "disodium 2-carboxybutanedioate")
