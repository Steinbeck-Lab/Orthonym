"""PIN class program, Task 15 (RETAINED_XYLENE): the xylenes are retained preferred IUPAC names.

 "Parent hydrocarbons having retained names" (the Blue Book): "Toluene and xylene are
preferred IUPAC names, but are not freely substitutable" (:8089); 'xylene (1,2-, 1,3-, and 1,4-isomers,
PINs) (no substitution)' (:8097); "Cyclic hydrocarbons" (:25669): '1,2-xylene (PIN)'
(:25699); "The name 'xylene' is a preferred IUPAC name, but cannot be substituted" (:25691). So the three
unsubstituted isomers are named by the retained name and a substituted xylene keeps its benzene parent.

An isotopically modified xylene cannot be sited on a numbered position of the retained name
 "Location of nuclides on positions not normally denoted by locants" (:44221),, "When the nuclide is located at a position in a retained name that is not numbered a
systematic name that identifies separately the relevant atom is used for the IUPAC preferred name",
the Blue Book), so its PIN is the systematic name: "Names" (:43716), '1,2-di[(13C)methyl]benzene
(PIN)' with '(alpha,alpha'-13C2)-1,2-xylene' the general-nomenclature form (:43734).
"""
import pytest

from tests.support.pin_tiers import assert_not_pin_labelled, assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    ("Cc1ccccc1C", "1,2-xylene"),                                             #:25699
    ("Cc1cccc(C)c1", "1,3-xylene"),                                           #:8097
    ("Cc1ccc(C)cc1", "1,4-xylene"),                                           #:8097
    # no substitution on xylene: the benzene parent carries every group
    ("Cc1cccc(Cl)c1C", "1-chloro-2,3-dimethylbenzene"),
    ("Cc1ccc(C)c(Br)c1", "2-bromo-1,4-dimethylbenzene"),
    ("Cc1ccccc1C(=O)O", "2-methylbenzoic acid"),
    # the isotopically modified xylene is named systematically
    ("[13CH3]c1ccccc1[13CH3]", "1,2-di[(13C)methyl]benzene"),                 #:43734
    ("[13CH3]c1cccc([13CH3])c1", "1,3-di[(13C)methyl]benzene"),
    ("Cc1ccccc1[13CH3]", "1-(13C)methyl-2-methylbenzene"),
    # the boundary: the other retained and the other multiplied arenes keep their names
    ("Cc1ccccc1", "toluene"),
    ("Cc1cc(C)cc(C)c1", "1,3,5-trimethylbenzene"),                            #:8097
    ("CCc1ccc(CC)cc1", "1,4-diethylbenzene"),
]
NOT_PIN_ROWS = [
    ("Cc1ccccc1C", "1,2-dimethylbenzene"),                                    #:25699 'not' form
    ("Cc1cccc(C)c1", "1,3-dimethylbenzene"),
    ("Cc1ccc(C)cc1", "1,4-dimethylbenzene"),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_xylene_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", NOT_PIN_ROWS)
def test_xylene_systematic_spelling_is_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)
