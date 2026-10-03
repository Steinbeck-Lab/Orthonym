"""PIN class program, batch 2 fix a performance pass: a nuclide on the chalcogen of an '-XH' suffix.

The rule
--------
 (the Blue Book): "The name of an isotopically substituted compound is formed by
adding or inserting the nuclide symbol(s) enclosed in parentheses, preceded by any necessary
locant(s), letters, and/or numerals, before the part of the compound that is isotopically
substituted."; '1-(aminomethyl)cyclopentan-1-(18O)ol (PIN)' (:43744).

 (:44180): "In preferred IUPAC names, locants are omitted if no locants are necessary
in unmodified names. However, if isotopic modification requires a locant to specify its
position, then all locants must be specified and none are omitted."; 'ethan(2H)ol (PIN) (as in
ethanol)' (:44184); '(2-13C)ethan-1-ol [not (2-13C)ethanol]' (:44186). (:44190):
"Locants are omitted when there is only one atom of a given element."

 (:43834): 'methan(2H,18O)ol (PIN)' (:43840). (:44208): '(1-2H1)ethan-1-(2H)ol
(PIN)' (:44214) -- a carbon label in front of the parent, the label of the '-ol' oxygen before
the suffix.

So the nuclide of the oxygen (sulfur, selenium, tellurium) that a '-ol' ('-thiol', '-selenol',
'-tellurol') suffix names is cited immediately before that suffix, and the name keeps the locant-
free spelling of the unlabelled name when that label needs no locant ('ethan(18O)ol',
'methane(34S)thiol').

A multiplied suffix
-------------------
 (:43770): "When two characteristic groups are isotopically modified in different ways
so that they cannot be combined using multiplicative terms such as 'di-', 'bis-', etc., the
isotopically modified characteristic group with the greater number of modifications is chosen as
the principal characteristic group to be cited as a suffix; the other characteristic is then
cited as a prefix."; copies modified the same way: 'cyclohexane-1,1-di[(14C)carboxylic acid]
(PIN)' (:43774). Neither form is built, so a name with the descriptor in front of a multiplied
'-XH' suffix ('(18O)ethane-1,2-diol', PIN '2-hydroxyethan-1-(18O)ol') is never labelled a PIN;
best-effort keeps its RT-exact name.

At the batch-2 head the engine kept a locant-free alcohol 18O in front of the parent
('(18O)ethan-1-ol', '(18O)methanol', '1-(aminomethyl)(18O)cyclopentan-1-ol') and the chalcogen
analogue on a locanted spelling ('ethane-1-(34S)thiol'), all labelled pin_verified.

Controls: a carbon label keeps its locant and its front place; an amine nitrogen keeps the
combined front descriptor (gold V47-/03); a label on a 'hydroxy' prefix stays before the
prefix ('(2R)-2-(2H)hydroxy-3-hydroxy(1-2H)propanal (PIN)',:44219).

A retained one-word name (fix a performance pass)
--------------------------------------
 (:43824): "In a name consisting of one word, the isotopic descriptor is placed before
the name, with an appropriate locant. This method is preferred to that of placing the
descriptor before the implied name of the characteristic group." ('(N-2H1)acetamide (PIN)',
:43828; '(N-2H2)aniline (PIN)',:43830). The retained name 'phenol',:26764) is not
a parent-hydride stem plus an '-ol' suffix, so it offers no slot before its 'ol' ('phen(18O)ol'
is the placement disprefers): the descriptor stands before the name, after any
substituent prefix ('2-methoxy(3,4,5,6-3H4)phenol (PIN)',,:44168), and the one oxygen
takes no locant,:44190): '(18O)phenol', '4-methyl(18O)phenol'.

One letter locant cited both for a hydrogen nuclide and for the nuclide of the element it
names ('(O-2H,O-18O)phenol') is not a Blue Book spelling: the Blue Book gives that heteroatom
nuclide no locant, '(O-2H,18O)acetic acid (PIN)',:43808), '(18O-2H,18O)acetic acid
(PIN)' (:43810). The search shares one locant across the descriptor and does not build that
form, so the name is labelled below the PIN; best-effort keeps it RT-exact.
"""
import re

import pytest

from orthonym.rules.isotopes import (
    _RETAINED_ONE_WORD_OL_NAMES,
    _STEM_ADJACENT_CHALCOGEN_OL_RE,
    _is_heteroatom_suffix_slot,
)
from tests.support.pin_tiers import (
    assert_not_pin_labelled,
    assert_pin_at_both_tiers,
)

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    ("NCC1([18OH])CCCC1", "1-(aminomethyl)cyclopentan-1-(18O)ol"),     #:43744
    ("CC[18OH]", "ethan(18O)ol"),                                        # as:44184
    ("C[18OH]", "methan(18O)ol"),                                        # as:43840
    ("CC(C)[18OH]", "propan-2-(18O)ol"),
    ("[18OH]C1CCCCC1", "cyclohexan(18O)ol"),
    ("[SiH3][18OH]", "silan(18O)ol"),
    ("CC[34SH]", "ethane(34S)thiol"),
    ("C[34SH]", "methane(34S)thiol"),
    ("C[13CH2][18OH]", "(1-13C)ethan-1-(18O)ol"),                        # as:44214
    ("[2H]C([2H])([2H])C[18OH]", "(2,2,2-2H3)ethan-1-(18O)ol"),          # as:44214
]

CONTROL_ROWS = [
    ("[13CH3]CO", "(2-13C)ethan-1-ol"),                                  #:44186
    ("CCC[34SH]", "propane-1-(34S)thiol"),
    ("C[13CH2][15NH2]", "(1-13C,15N)ethan-1-amine"),                     # gold V47-ISO
    ("OC(=O)CC[18OH]", "3-(18O)hydroxypropanoic acid"),
    ("N[14CH2]C1(O)CCCC1", "1-[amino(14C)methyl]cyclopentan-1-ol"),     #:43742
    ("OC1CCC(CC[18OH])CC1", "4-[2-(18O)hydroxyethyl]cyclohexan-1-ol"),
    ("CC(=[18O])C", "propan-2-(18O)one"),
    ("OC[13CH2]O", "(13C)ethane-1,2-diol"),
]

# (smiles, the spelling with the descriptor in front of a multiplied '-XH' suffix)
MULTIPLIED_SUFFIX_ROWS = [
    ("OCC[18OH]", "(18O)ethane-1,2-diol"),
    ("[18OH]CC[18OH]", "(18O2)ethane-1,2-diol"),
    ("CC(O)C[18OH]", "(18O)propane-1,2-diol"),
    ("OC1CCC([18OH])CC1", "(18O)cyclohexane-1,4-diol"),
    ("OCC(O)C[18OH]", "(18O)propane-1,2,3-triol"),
    ("SCC[34SH]", "(34S)ethane-1,2-dithiol"),
]


# fix a performance pass: the retained one-word name 'phenol',:43824)
RETAINED_NAME_ROWS = [
    ("c1ccccc1[18OH]", "(18O)phenol"),
    ("Cc1ccc(cc1)[18OH]", "4-methyl(18O)phenol"),
    ("COc1ccccc1[18OH]", "2-methoxy(18O)phenol"),                       # as:44168
]

RETAINED_NAME_CONTROL_ROWS = [
    ("c1ccccc1O[2H]", "(O-2H)phenol"),                                   # as:43830
    ("COc1c([3H])c([3H])c([3H])c([3H])c1O", "2-methoxy(3,4,5,6-3H4)phenol"),  #:44168
]

# (smiles, a letter locant cited for a hydrogen nuclide and for its own element's nuclide)
SHARED_LETTER_LOCANT_ROWS = [
    ("c1ccccc1[18O][2H]", "(O-2H,O-18O)phenol"),
    ("CC(=O)[18O][2H]", "(O-2H,O-18O)acetic acid"),                      # PIN:43810
]

# (unlabelled name, offset of its '-ol' / '-thiol', a slot before that suffix)
SLOT_ROWS = [
    ("phenol", 4, False),
    ("4-methylphenol", 12, False),
    ("2-methoxyphenol", 13, False),
    ("thiophenol", 8, False),
    ("eugenol", 5, False),
    ("ethanol", 5, True),
    ("methanol", 6, True),
    ("ethenol", 5, True),
    ("cyclohexanol", 10, True),
    ("phenylmethanol", 12, True),
    ("benzenethiol", 7, True),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", RETAINED_NAME_ROWS)
def test_retained_one_word_name_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", RETAINED_NAME_CONTROL_ROWS)
def test_retained_one_word_name_control(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", SHARED_LETTER_LOCANT_ROWS)
def test_shared_letter_locant_on_its_own_element_never_labelled_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


@pytest.mark.parametrize("skel,off,slot", SLOT_ROWS)
def test_suffix_slot_of_a_retained_name(skel, off, slot):
    assert _is_heteroatom_suffix_slot(skel, off) is slot


def test_retained_ol_names_of_the_engine_tables_offer_no_slot():
    """Every one-word name of the engine's retained-name tables whose '-ol' follows an 'an' /
    'en' / 'yn' stem is either a parent-hydride stem plus the suffix ('methanol': 'methane' is
    itself a name of the tables), which offers the slot, or one of
    ``_RETAINED_ONE_WORD_OL_NAMES``, which does not."""
    from orthonym.data import amino_acids, natural_products
    from orthonym.data.opsin_imports import OPSIN_RETAINED_NAMES
    from orthonym.data.retained_names import RETAINED_NAMES

    tables = (RETAINED_NAMES, OPSIN_RETAINED_NAMES,
              natural_products.NATURAL_PRODUCT_DERIVATIVES,
              amino_acids.NON_STANDARD_AMINO_ACIDS)
    names = {v for t in tables for v in t.values() if isinstance(v, str)}
    words = sorted(n for n in names if re.fullmatch(r"[a-z]+(?:an|en|yn)ol", n))
    assert "phenol" in words and "methanol" in words
    for word in words:
        off = len(word) - 2
        if word[:off] + "e" in names:
            assert _STEM_ADJACENT_CHALCOGEN_OL_RE.match(word, off), word
        else:
            assert word in _RETAINED_ONE_WORD_OL_NAMES, word
            assert not _STEM_ADJACENT_CHALCOGEN_OL_RE.match(word, off), word


@pytest.mark.parametrize("smiles,pin", CONTROL_ROWS)
def test_control(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", MULTIPLIED_SUFFIX_ROWS)
def test_multiplied_suffix_never_labelled_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)
