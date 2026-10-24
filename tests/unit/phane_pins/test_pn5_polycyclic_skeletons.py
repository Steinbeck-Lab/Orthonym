import pytest

from orthonym.rules.phane_pin import general_phane_name
from orthonym.rules.phane_pin.skeleton import perceive_cyclic

from ._helpers import mol, substituted

# the [2.2.2](1,3,5)cyclophane of the skeleton rows with a third benzene joining two chain atoms:
# the node graph has two edges more than nodes (three rings)
THREE_RINGS = "c1c2cc3cc1CC1c4ccc(cc4)C3Cc3cc(cc1c3)CC2"


@pytest.mark.parametrize("line,name,smiles", substituted("skeleton"), ids=lambda v: str(v)[:30])
def test_book_and_derived_polycyclic_skeletons(line, name, smiles):
    #:15065 bicyclo[6.6.0],:19659 spiro[7.7], and the [2.2.2](1,3,5)cyclophane
    assert general_phane_name(mol(smiles)) == name


def test_a_three_ring_skeleton_declines():
    assert perceive_cyclic(mol(THREE_RINGS)) is None
    assert general_phane_name(mol(THREE_RINGS)) is None


def test_a_spiro_node_that_is_an_amplificant_declines():
    # shows the spiro atom as a chain atom; two [2.2]paracyclophane layers sharing a benzene
    # (a degree-4 amplificant) have no book example
    assert general_phane_name(mol("c1cc2ccc1CCc1cc3c(cc1CCc1ccc(cc1)CC3)CC2")) is None


def test_two_cyclophanes_joined_by_a_chain_are_no_bicycle():
    # two rings joined by a chain (a dumbbell) have two nodes of degree three but share no path of nodes
    dumbbell = "c1cc2ccc1CCc1ccc(c(CCCc3cc4ccc3CCc3ccc(cc3)CC4)c1)CC2"
    assert perceive_cyclic(mol(dumbbell)) is None


@pytest.mark.parametrize("line,name,smiles", substituted("skeleton"), ids=lambda v: str(v)[:30])
def test_the_vocabulary_of_non_pin_forms_does_not_read_a_skeleton_word(line, name, smiles):
    # (:14878) names a skeleton of superatoms by von Baeyer nomenclature; 'bicyclo[6.6.0]' there is no
    # ortho-fused ring system, which (:23831) would give a fusion name
    from orthonym.rules.pin_vocabulary import non_pin_vocabulary
    assert non_pin_vocabulary(name) is None


@pytest.mark.parametrize("name", [
    "bicyclo[4.4.0]decane",                                  # (:24217): decahydronaphthalene is the PIN
    "bicyclo[4.4.0]decylphosphane",                          # 'phosphane' is no phane
    "bicyclo[4.4.0]decylphane",                              # no node count in front of 'phane'
])
def test_the_ortho_fused_von_baeyer_tokens_outside_a_skeleton_word_stay_non_pin(name):
    from orthonym.rules.pin_vocabulary import non_pin_vocabulary
    assert non_pin_vocabulary(name) == "bicyclo[4.4.0]"
