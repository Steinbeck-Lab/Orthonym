""": two isotope name shapes the decorator had no place for (PubChem rows that abstained).

Cited rules (the Blue Book, with their section headings):

* (heading NAMES,:43718): the descriptor is inserted "before the part of the compound
  that is isotopically substituted"; after the closing mark of a prefix, the part is the fused parent:
  '2-(propan-2-yl)(7-3H)[1,3]thiazolo[4,5-c]pyridine' (the shape of '...3-(2H3)methyl(1-2H1)pentane
  (PIN)',:43722).
* 'Different isotopic modifications on otherwise identical substituents' (:43756): "they are
  cited separately. The isotopically modified substituent is preferred alphabetically to the unmodified
  substituent" (:43758); the alkyl words of a functional-class ester are cited apart as in 'ethyl methyl
  butanedioate (PIN)' (:31669): '(2H1)methyl methyl sulfate'. (:44190): no locant for the one
  carbon of a methyl.
"""
import pytest
from rdkit import Chem

from orthonym.rules import isotopes as I
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.unit


def _unlabelled_tier(smiles):
    """The tier of the molecule with its labels removed, from the same namer: a label never moves it."""
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return name_best_effort(Chem.MolToSmiles(mol))["tier"]


@pytest.mark.parametrize("smiles, name", [
    #: the slot before a fused parent that follows a bracketed prefix
    ("[3H]C1=CN=CC2=C1SC(=N2)C(C)C", "2-(propan-2-yl)(7-3H)[1,3]thiazolo[4,5-c]pyridine"),
    #: one labelled copy of the multiplied word of a functional-class name
    ("[2H]COS(=O)(=O)OC", "(2H1)methyl methyl sulfate"),
    ("[2H]C([2H])([2H])OP(=O)(OC)OC", "(2H3)methyl dimethyl phosphate"),
])
def test_named_at_the_tier_of_the_skeleton_and_read_back(smiles, name):
    got = name_best_effort(smiles)
    assert got["name"] == name
    assert got["tier"] == _unlabelled_tier(smiles)
    assert name_is_rt_exact(got["name"], smiles)


@pytest.mark.parametrize("smiles, name", [
    ("COS(=O)(=O)OC", "dimethyl sulfate"),                                         # no label
    ("[2H]C([2H])([2H])OS(=O)(=O)OC([2H])([2H])[2H]", "di[(2H3)methyl] sulfate"),  # every copy alike
    ("[2H]N([2H])C(=O)C[2H]", "(N,N,2-2H3)acetamide"),        # two labels on one N: the letter repeated
])
def test_names_built_before_are_unchanged(smiles, name):
    got = name_best_effort(smiles)
    assert got["name"] == name
    assert got["tier"] == _unlabelled_tier(smiles)


def test_an_unsymmetrical_diester_is_not_split():
    # the two alkyl words stand on different acid oxygens: OPSIN's read-back cannot place a split label
    got = name_best_effort("[2H]C([2H])([2H])OC(=O)CC(C)C(=O)OC")["name"] or ""
    assert "methyl methyl" not in got


def test_fused_opening_slot_after_a_prefix_mark_only():
    skel = "2-(propan-2-yl)[1,3]thiazolo[4,5-c]pyridine"
    assert skel.index("[1,3]") in I._insertion_offsets(skel)
    fused = "7,14-methano[1,3]benzimidazolo[1,2-b][2,5]benzodiazocin-5(14H)-one"
    assert fused.index("[2,5]") not in I._insertion_offsets(fused)   # a fusion descriptor's bracket


def test_functional_class_split_needs_the_multiplied_first_word():
    assert I._FC_MULTIPLIED_WORD_RE.match("dimethyl sulfate")
    assert not I._FC_MULTIPLIED_WORD_RE.match("1,2-dimethylbenzene")
    assert not I._FC_MULTIPLIED_WORD_RE.match("methyl acetate")
