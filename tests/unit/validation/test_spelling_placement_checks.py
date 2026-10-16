"""Spelling checks of where a name places a stereodescriptor and an enclosing mark.

 "NAMING OF STEREOISOMERS" (the Blue Book): "In preferred IUPAC names, stereodescriptors are
placed immediately at the front of the part of the name to which they relate. They are placed at the front of
the complete name when related to the parent structure;... When they relate to substituent groups, they are
cited at the front of the corresponding prefix."
 "Parentheses" (:7232): "Parentheses are used around compound and complex prefixes".

The checks read the name only (the structure is not used); the precision on the Blue Book's own names is
measured in ``tests_local/unit/validation/test_placement_checks_precision.py`` (7,967 names, 0 hits).
Changed asserted values: none, the checks are new.
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import check_pin_spelling, registered_rules

pytestmark = pytest.mark.unit

MOL = Chem.MolFromSmiles("CC")      # the checks do not read the structure


def _rules(name):
    return [f.rule for f in check_pin_spelling(MOL, name, strict=True)]


def test_both_rules_are_registered():
    assert "P-91.3" in registered_rules() and "P-16.5.1.1" in registered_rules()


@pytest.mark.parametrize("name", [
    # the descriptors of the substituent ring and of the benzylic carbon at the front of an ethane
    "(1R,4S)-2-(4-methoxyphenyl)-N-{[9-(1-methyl-1H-pyrazol-4-yl)-5-oxa-2,7-diazabicyclo[4.4.0]deca-1(6),"
    "7,9-trien-4-yl](phenyl)methyl}ethan-1-amine",
    "(3R)-2-chloroethan-1-ol",
    "(2S,5R)-N-methylpropan-2-amine",
])
def test_a_descriptor_locant_beyond_the_parent_is_flagged(name):
    assert "P-91.3" in _rules(name)


@pytest.mark.parametrize("name", [
    "(2S)-N-[(1R)-1-phenylethyl]propan-2-amine",        # the substituent's own descriptor, inside its prefix
    "(2S,3R)-3-bromobutan-2-ol",
    "(1R,2S)-2-methylcyclohexan-1-ol",
    "(4S)-4-[(1R)-1-hydroxyethyl]piperidine",
    "(2E)-but-2-enoic acid",
    "(2S)-2-amino-3-methylbutanoic acid",
    "(4aR,8aS)-decahydronaphthalene",                  # a parent whose positions the reader does not know
    "(1R)-N-benzylethan-1-amine",
])
def test_a_descriptor_that_fits_its_unit_is_not_flagged(name):
    assert "P-91.3" not in _rules(name)


@pytest.mark.parametrize("name", [
    "3-(5-chlorofuran-2-yl)-N-2-methoxyethylprop-2-en-1-amine",
    "N-2-methoxyethyl-3-phenylprop-2-en-1-amine",
    "N,N-1-phenylethylpropan-2-amine",
    "2-methoxy-N-2-phenylethylethan-1-amine",
])
def test_a_prefix_with_its_own_locant_after_an_n_locant_needs_enclosing_marks(name):
    assert "P-16.5.1.1" in _rules(name)


@pytest.mark.parametrize("name", [
    "N-(2-methoxyethyl)but-2-en-1-amine",
    "N,2-dimethylpropanamide",
    "N-benzyl-N-methylethan-1-amine",
    "N,N-bis(2-chloroethyl)aniline",
    "(N-2H1)acetamide",                                # an isotope descriptor
    "N-(1H-indol-3-yl)acetamide",
    "N-tert-butylaniline",
    "N,4-dimethyl-N-(3-methylphenyl)benzamide",
])
def test_an_enclosed_or_simple_n_prefix_is_not_flagged(name):
    assert "P-16.5.1.1" not in _rules(name)
