""": carboxy groups labelled in different ways, one suffix and the others prefixes.

Cited rules (the Blue Book, with their section headings):

* (heading 'Isotopically modified compounds',:43770): "the isotopically modified
  characteristic group with the greater number of modifications is chosen as the principal characteristic
  group to be cited as a suffix; the other characteristic is then cited as a prefix"
  ('1-(13C)carboxycyclohexane-1-(14C)carboxylic acid (PIN)',:43778).
* (heading 'Location of nuclides on positions not normally denoted by locants',:44251):
  'benzene(13C)carboxylic acid (PIN)' (:44255), not the retained 'benzoic acid'.
* numbering: with one suffix the parent is numbered again; the engine numbers the carboxamide
  surrogate (``rules.isotope_suffix_split``), so the label on the other group moves the methyl from 4 to 5.
"""
import pytest
from rdkit import Chem

from orthonym.rules import isotope_suffix_split as S
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.unit

_PIN_TIERS = ("pin_verified", "pin_unverified")


@pytest.mark.parametrize("smiles, name", [
    ("C1=CC=C(C(=C1)C(=O)O)[13C](=O)O", "2-carboxybenzene-1-(13C)carboxylic acid"),
    ("OC(=O)c1cc(C(=O)O)cc([13C](=O)O)c1", "3,5-dicarboxybenzene-1-(13C)carboxylic acid"),
    ("CC1=CC(=C(C=C1)[13C](=O)O)C(=O)O", "2-carboxy-4-methylbenzene-1-(13C)carboxylic acid"),
    ("CC1=CC(=C(C=C1)C(=O)O)[13C](=O)O", "2-carboxy-5-methylbenzene-1-(13C)carboxylic acid"),
    ("OC(=O)C1CCC(CC1)[13C](=O)O", "4-carboxycyclohexane-1-(13C)carboxylic acid"),
    ("OC(=O)c1c(C(=O)O)c(C(=O)O)c(C(=O)O)c(C(=O)O)c1[13C](=O)O",
     "2,3,4,5,6-pentacarboxybenzene-1-(13C)carboxylic acid"),
    #: the locant set ties, the first prefix in alphanumerical order takes the lower locant
    ("OC(=O)c1cccc(Br)c1[13C](=O)O", "2-bromo-6-carboxybenzene-1-(13C)carboxylic acid"),
])
def test_one_suffix_named_below_the_pin_and_read_back(smiles, name):
    got = name_best_effort(smiles)
    assert got["name"] == name
    assert got["tier"] not in _PIN_TIERS        # the PIN status of the split is not checked
    assert name_is_rt_exact(got["name"], smiles)


@pytest.mark.parametrize("smiles, name", [
    ("C1=CC=C(C(=C1)C(=O)O)C(=O)O", "benzene-1,2-dicarboxylic acid"),                  # no label
    ("OC(=O)C1(CCCCC1)[13C](=O)O", "1-carboxycyclohexane-1-(13C)carboxylic acid"),     # built before
    ("OC(=O)CC1=CC=CC=C1[13C](=O)O", "2-(carboxymethyl)(13C)benzoic acid"),            # no multiplied suffix
])
def test_names_built_before_are_unchanged(smiles, name):
    assert name_best_effort(smiles)["name"] == name


@pytest.mark.parametrize("smiles", [
    "OC(=O)C1(CCCCC1)[13C](=O)O[2H]",       # the suffix spelling of 13C,2H does not read back: no prefix label
    "OC(=O)c1ccccc1[13C](=[18O])O",
    "COC(=O)c1ccc(C(=O)O)cc1[13C](=O)O",    # an ester word ('methoxycarbonyl'): no split
])
def test_the_label_never_lands_on_a_carboxy_prefix(smiles):
    got = name_best_effort(smiles)["name"] or ""
    assert ")carboxybenzene" not in got and ")carboxycyclohexane" not in got


def test_the_pin_tier_does_not_take_the_split():
    from orthonym import name_compound
    got = name_compound("C1=CC=C(C(=C1)C(=O)O)[13C](=O)O", emit_tier="pin")
    assert "carboxybenzene-1-(13C)" not in (got or "")


@pytest.mark.parametrize("smiles", [
    "OC(=O)c1ccccc1[13C](=O)N",                    # one carboxy group only
    "NC(=O)c1ccc(C(=O)O)cc1[13C](=O)O",            # an amide of the input's own
    "[13C](=O)(O)c1ccccc1[13C](=O)O",              # labels on two groups
    "OC(=O)c1ccccc1C(=O)O[2H].[13CH4]",            # a label outside the groups
])
def test_no_surrogate(smiles):
    assert S.surrogate_smiles(Chem.MolFromSmiles(smiles)) is None


def test_surrogate_draws_the_other_group_as_a_carboxamide():
    sur = Chem.MolFromSmiles(S.surrogate_smiles(Chem.MolFromSmiles("C1=CC=C(C(=C1)C(=O)O)[13C](=O)O")))
    assert sur.HasSubstructMatch(Chem.MolFromSmarts("NC(=O)c1ccccc1C(=O)[OH]"))
    assert all(a.GetIsotope() == 0 for a in sur.GetAtoms())


def test_only_a_multiplied_suffix_is_split():
    assert S._MULTIPLIED_SUFFIX.search("benzene-1,2-dicarboxylic acid")
    assert not S._MULTIPLIED_SUFFIX.search("2-(carboxymethyl)benzoic acid")
