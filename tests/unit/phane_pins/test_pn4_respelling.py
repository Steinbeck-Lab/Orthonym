import random

import pytest
from rdkit import Chem

from orthonym.rules.phane_pin import general_phane_name

from ._helpers import gold, mol, substituted

SPELLINGS = 30

# the structures whose two ortho-cited prefixes could take either of two equal locant sets
TIED_LOCANTS = [
    "C1Cc2c(Br)cc(cc2Cl)CCc2ccc1cc2",
    "C1Cc2c(C)cc(cc2CC)CCc2ccc1cc2",
    "OC1Cc2c(Cl)cc(cc2Br)CCc2ccc1cc2",
]


def _respelled(smiles, seed):
    """The same structure with its atoms in another order."""
    m = mol(smiles)
    order = list(range(m.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    return Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RenumberAtoms(m, order), canonical=False))


def _names(smiles):
    return {general_phane_name(_respelled(smiles, seed)) for seed in range(SPELLINGS)}


@pytest.mark.parametrize("smiles", TIED_LOCANTS)
def test_the_name_does_not_depend_on_the_order_of_the_atoms(smiles):
    # (g) (:3307): of equal locant sets, the prefix cited first takes the lower locant
    names = _names(smiles)
    assert len(names) == 1 and None not in names, names


def test_the_prefix_cited_first_gets_the_lower_composite_locant():
    assert _names("C1Cc2c(Br)cc(cc2Cl)CCc2ccc1cc2") == {"1^2-bromo-1^6-chloro-1,4(1,4)-dibenzenacyclohexaphane"}


@pytest.mark.parametrize("line,name,smiles", gold("cycle"), ids=lambda v: str(v)[:24])
def test_a_book_name_is_the_same_for_every_order_of_the_atoms(line, name, smiles):
    assert _names(smiles) == {name}


@pytest.mark.parametrize("line,name,smiles", substituted("prefix") + substituted("suffix"),
                         ids=lambda v: str(v)[:24])
def test_a_substituted_book_name_is_the_same_for_every_order_of_the_atoms(line, name, smiles):
    assert _names(smiles) == {name}


@pytest.mark.parametrize("line,name,smiles", gold("path"), ids=lambda v: str(v)[:24])
def test_a_linear_book_name_is_the_same_for_every_order_of_the_atoms(line, name, smiles):
    from orthonym.rules.phane_pin import linear_phane_name
    assert {linear_phane_name(_respelled(smiles, seed)) for seed in range(8)} == {name}


def test_two_chains_that_tie_give_one_name_for_every_order_of_the_atoms():
    #:21631: two naphthalene rings could end the chain; (:21604) takes the one that leaves two simple
    # prefixes (the chloro group and the naphthalenyl group) on the parent
    from orthonym.rules.phane_pin import linear_phane_name
    (_, name, smiles), = [r for r in substituted("compound_prefix") if r[0] == "21631"]
    assert {linear_phane_name(_respelled(smiles, seed)) for seed in range(SPELLINGS)} == {name}


def test_a_macrocycle_read_as_aromatic_is_named_the_same_for_every_order_of_the_atoms():
    smiles = ("C1=CC=C(C=C1)N2C3=CC4=C(C=C3)C=CC(=C4N(C5=CC6=C(C=C5)C=CC(=C62)C7=CC=CC(=C7)C8=CN=CC=C8)C9=CC=CC=C9)"
              "C1=CC=CC(=C1)C1=CN=CC=C1")
    names = _names(smiles)
    assert len(names) == 1 and None not in names, names
