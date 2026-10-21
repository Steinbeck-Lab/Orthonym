"""A compound prefix inside an '-amino' prefix is enclosed in its own marks.

``CSCNc1ccc(C(=O)O)cc1`` was named '4-[(methylsulfanyl)methylamino]benzoic acid', with
'(methylsulfanyl)methyl' bare inside the 'amino' prefix. 'is_complex_substituent' reads a name that
opens with a mark as already enclosed; the nitrogen-group builder of the benzene rules decided the
inner marks with it alone. The shared decision is ``enclose_if_compound``, the union of
``needs_brackets``, ``is_complex_substituent`` and an inner enclosing mark.

 (the Blue Book, under DEFINITIONS): "A compound substituent group consists of a
simple substituent group (the parent substituent group) to which is attached one or more simple
substituent groups". (:7232): parentheses "around compound... prefixes";
(:7416): "Brackets enclose substituent prefixes in which parentheses have already been used",
'4-[(hydroxyselanyl)methyl]benzoic acid (PIN)' (:7420); (:7436): "Braces are used to enclose
substituent prefixes in which brackets and parentheses have already been used",
'4-{[(4-chlorophenyl)methylidene]amino}aniline (PIN)' (:26529).
"""
import pytest

from orthonym.assembly.naming_utils import enclose_if_compound
from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("name,marked", [
    ("(methylsulfanyl)methyl", "[(methylsulfanyl)methyl]"),
    ("(chloromethyl)phenyl", "[(chloromethyl)phenyl]"),
    ("(2-chloroethyl)", "(2-chloroethyl)"),
])
def test_prefix_that_holds_an_enclosed_substituent_is_compound(name, marked):
    assert enclose_if_compound(name) == marked


@pytest.mark.parametrize("smiles,pin", [
    ("CSCNc1ccc(C(=O)O)cc1", "4-{[(methylsulfanyl)methyl]amino}benzoic acid"),
    ("CSC(C)Nc1ccc(C(=O)O)cc1", "4-{[1-(methylsulfanyl)ethyl]amino}benzoic acid"),
    ("CSCNc1ccc(C(=O)O)cc1C(=O)O", "4-{[(methylsulfanyl)methyl]amino}benzene-1,3-dicarboxylic acid"),
    # a simple inner prefix keeps its single level of marks
    ("CCOCNc1ccc(C(=O)O)cc1", "4-[(ethoxymethyl)amino]benzoic acid"),
    ("ClCNc1ccc(C(=O)O)cc1", "4-[(chloromethyl)amino]benzoic acid"),
])
def test_compound_prefix_inside_amino_is_enclosed(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
