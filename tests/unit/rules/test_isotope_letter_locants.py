"""When an isotopic descriptor may cite a bare element letter ('O', 'N'), and when it cites 'O1', 'N1'.

 LOCANTS, Omission of locants, (the Blue Book): "In preferred IUPAC
names, locants are omitted if no locants are necessary in unmodified names. However, if isotopic
modification requires a locant to specify its position, then all locants must be specified and none
are omitted." (:44190): "Locants are omitted when there is only one atom of a given
element." (:44202): "Locants are not omitted when there is a possibility of isomers".
 Letter and/or numeral locants (:44208): "When locants are needed for defining the structure of
the parent structure or of a unit of structure as defined by its appropriate enclosing marks, then all
locants must be cited for the parent structure or that structural unit."

The book prints the bare letter in '(O-2H, 18O)acetic acid (PIN)',:44294) and
'(N-2H1)acetamide (PIN)',:43824), and, beside numerals, in '(O,2,2,2-2H4)ethan-1-ol' (the
comparative table,:44506): one atom the letter can denote. The project's rule
(``rules/isotope_letters.py``): a bare letter only where it leaves no possibility of isomers, that is,
every atom the letter could denote is one symmetry class of the structure (judged without the labels the
letter itself denotes, with every other label kept); otherwise the letter with its number ('O1', 'O4',
'N1'), which OPSIN 2.9.0 reads. A letter cited beside other locants is not numbered for that reason
alone.

Before lane I (fix a performance pass) the decorator printed '(2-13C,O-18O)pentanedioic acid' (pin_verified at both
tiers). OPSIN reads a bare 'O' as the first oxygen, so that name reads back exact, but its text licenses
the oxygen of the other carboxyl too, which is a different isotopomer: '(2-13C,O5-18O)pentanedioic acid'
reads as the other molecule (machine-confirmed below). The 13C in the chain tells the two acid oxygens
apart; without it they are one class and the letter stays bare, '(O-18O)butanedioic acid'.
"""
import pytest
from rdkit import Chem

from orthonym.rules import isotope_letters as L
from tests.support.pin_tiers import assert_pin_at_both_tiers
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.unit


# -- no JVM: the predicate -----------------------------------------------------------------------------


@pytest.mark.parametrize("smiles,element,classes", [
    ("OC(=O)CCC(=O)O", "O", 1),                    # butanedioic acid: the two hydroxyls are one orbit
    ("[2H]OC(=O)CCC(=O)O", "O", 1),                # the label the letter denotes is not part of the classes
    ("[2H]OC(=O)CCC(=O)[18OH]", "O", 1),           # nor are two such labels, one on each hydroxyl
    ("OC(=O)CC[13CH2]C(=O)O", "O", 2),             # a 13C in the chain tells the two acid ends apart
    ("OC(=O)CC[13CH2]C(=O)[18OH]", "O", 2),
    ("[2H]Oc1ccc(O)cc1", "O", 1),                  # benzene-1,4-diol
    ("[2H]Oc1ccc(O)cc1[2H]", "O", 2),              # a ring D tells the two hydroxyls apart
    ("OCCO", "O", 1),                              # ethane-1,2-diol
    ("CC(O)CO", "O", 2),                           # propane-1,2-diol: primary and secondary hydroxyl
    ("CC(=O)O", "O", 1),                           # the carbonyl oxygen is not a hydroxyl
    ("Cl[13CH2]C(=O)[18OH]", "O", 1),              # one acid group: a 13C elsewhere leaves one atom
    ("CC(=O)Nc1ccccc1", "N", 1),                   # one amide nitrogen
    ("[2H]C([2H])C(=O)N([2H])c1ccccc1", "N", 1),   #... whatever else is labelled
    ("CC(=O)Nc1ccc(N)cc1", "N", 2),                # an amide and an amine nitrogen
    ("c1ccncc1CC(=O)N", "N", 1),                   # a ring nitrogen is numbered by its ring, not denoted
])
def test_the_symmetry_classes_a_letter_could_denote(smiles, element, classes):
    assert L.distinct_classes(Chem.MolFromSmiles(smiles), element) == classes


def test_a_bare_letter_needs_one_class():
    diacid = Chem.MolFromSmiles("OC(=O)CC[13CH2]C(=O)O")
    assert not L.bare_letter_allowed(diacid, "O")
    assert L.bare_letter_allowed(Chem.MolFromSmiles("OC(=O)CCC(=O)O"), "O")
    # the atoms of one unit: a hydroxyl alone is one class whatever the molecule holds
    hydroxyls = L.denotable_atoms(diacid, "O")
    assert len(hydroxyls) == 2 and L.bare_letter_allowed(diacid, "O", hydroxyls[:1])


# -- the decorator, both tiers ----------------------------------------------------------------------------


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
def test_pentanedioic_acid_cites_the_number_of_its_oxygen():
    smiles = "OC(=O)CC[13CH2]C(=O)[18OH]"
    name = "(2-13C,O1-18O)pentanedioic acid"
    assert_pin_at_both_tiers(smiles, name)
    # the other numeral is the other molecule; the bare letter, which OPSIN reads as O1, reads back too and
    # so is not told from the numbered name by the read-back: the rule decides, not the oracle
    assert not name_is_rt_exact("(2-13C,O5-18O)pentanedioic acid", smiles)
    assert name_is_rt_exact("(4-13C,O5-18O)pentanedioic acid", smiles)      # the other numbering of the same molecule
    assert name_is_rt_exact("(2-13C,O-18O)pentanedioic acid", smiles)


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,name", [
    # one symmetry class of hydroxyl oxygens: the bare letter stays
    ("OC(=O)CCC(=O)[18OH]", "(O-18O)butanedioic acid"),
    ("[2H]OC(=O)CCC(=O)O", "(O-2H)butanedioic acid"),
    ("[2H]OCCO", "(O-2H)ethane-1,2-diol"),
    # the book's own printed bare letters (one atom the letter can denote)
    ("CC(=O)[18OH]", "(O-18O)acetic acid"),
    ("[2H]OC(=O)C", "(O-2H)acetic acid"),
    ("CC(=O)N[2H]", "(N-2H1)acetamide"),
    # one acid group or one amide nitrogen: the letter was never ambiguous, so it stays bare beside
    # other locants of the descriptor
    ("Cl[13CH2]C(=O)[18OH]", "2-chloro(2-13C,O-18O)acetic acid"),
    ("[2H]C([2H])(Cl)C(=O)[18OH]", "2-chloro(2,2-2H2,O-18O)acetic acid"),
    ("[2H]C([2H])C(=O)N([2H])c1ccccc1", "N-phenyl(N,2,2-2H3)acetamide"),
    # two classes of hydroxyl oxygens: the number
    ("[2H]OCC(C)O", "(O1-2H)propane-1,2-diol"),
    ("OC(=O)CC[13CH2]C(=O)[18OH]", "(2-13C,O1-18O)pentanedioic acid"),
])
def test_bare_or_numbered_letter(smiles, name):
    assert_pin_at_both_tiers(smiles, name)


# -- the composer follows the same predicate -------------------------------------------------------------------


@pytest.mark.parametrize("smiles,skeleton,name", [
    ("[2H]OC(=O)CCC(=O)O", "butanedioic acid", "(O-2H)butanedioic acid"),
    ("[2H]OC(=O)CC[13CH2]C(=O)O", "pentanedioic acid", "(4-13C,O1-2H)pentanedioic acid"),
    ("[2H]C([2H])C(=O)N([2H])c1ccccc1", "N-phenylacetamide", "N-phenyl(N,2,2-2H3)acetamide"),
    ("CC(=O)N([2H])c1ccc(O)cc1", "N-(4-hydroxyphenyl)acetamide", "N-(4-hydroxyphenyl)(N-2H)acetamide"),
])
def test_the_composer_cites_a_letter_as_the_decorator_does(smiles, skeleton, name):
    from orthonym.rules import isotope_scopes as S, isotopes as I
    original = Chem.MolFromSmiles(smiles)
    stripped, label_map = I.strip_isotopes(original)
    got = S.compose_scopes(skeleton, original, label_map, stripped.GetNumHeavyAtoms())
    assert got == name
    assert name_is_rt_exact(got, smiles)
