"""Trivial names of the retained-name table that the Blue Book does not make PINs.

Each name below was shipped ``pin_verified`` at the default tier from the hand-curated table
``data/retained_names.py`` although the Blue Book makes a systematic name the PIN, or does not
print the name at all. The home of "this retained name is not a PIN" is a ``pin: false`` row of
``data/iupac_2013_pin_list.json``; a denied name leaves the PIN-path lookup and the systematic
producer names the molecule (the name stays reachable through ``GENERAL_RETAINED_NAMES``).

Blue Book basis, each anchor opened (the Blue Book Blue Book; the heading is given with the line):

* "HYDROXY COMPOUNDS AND CHALCOGEN ANALOGUES" (:26748),:26750: "Names generated
  substitutively are preferred IUPAC names rather than functional class names or retained names,
  with the exception of the retained name 'phenol' that can be fully substituted."
  -> propargyl alcohol, mesitol.
* "Names of aldehydes derived from retained names of carboxylic acids" (:34960):
   (:34982) "Systematic names are the preferred IUPAC names" (only
  furaldehyde, phthalaldehyde,... stay for general nomenclature;:35074 '5-(hydroxymethyl)furan-2-
  carbaldehyde (PIN)... [not 5-(hydroxymethyl)furfural]'); (:34994) "Aldehydes derived
  from retained acid names given in are only used in general nomenclature; no
  substitution is allowed. Preferred IUPAC names are systematic names" -> furfural,
  nicotinaldehyde, isonicotinaldehyde.
* "ACYLOINS" (:29661),:29677 '1,2-di(furan-2-yl)-2-hydroxyethan-1-one (PIN)' and
   "Retained names" (:28295),:28369 '1-phenylethan-1-one (PIN)': a ketone is expressed by
  the suffix on the chain that carries the C=O -> 2-acetylfuran.
* "Polyamines" (:26324),:26404 'N1-(4-aminophenyl)-N4-phenylbenzene-1,4-diamine (PIN)':
  the ring diamine is 'benzene-1,4-diamine'; 'phenylene' is the divalent group of a multiplicative
  name only (:26402) -> 1,4-phenylenediamine.
* "Retained names as preferred IUPAC names" (:29715),:29717: "Only the following five
  carboxylic acids retained names and are also preferred IUPAC names." (formic, oxalic, acetic,
  benzoic, oxamic) -> suberic, azelaic, sebacic acid, and (the name is not in the book) orotic acid.
* (:33246; paragraph " Sultams, sultims, and intramolecular amides of
  sulfinic acids",:33244),:33252 "Method (1) leads to preferred IUPAC names." with
  '1λ6,2-thiazinane-1,1-dione (PIN)' (:33258): the sultam PIN is the heterocyclic heterone, so
  'saccharin' is not a PIN. No PIN is built for the structure yet (the lambda-6 ring-sulfur heterone
  class), so the default tier declines it and best-effort names it systematically.

NOT denied, three leads of the lane that the Blue Book refutes:

* 'dimethylformamide': (:32774) prints 'dimethylformamide (PIN)' (:32782). Formamide's
  only substitutable hydrogens are its two N-H, so complete substitution in the same way omits the
  locants: (:3007), "All locants are omitted in compounds or substituent groups in which
  all substitutable positions are completely substituted or modified".
* 'chalcone': the retained PIN of the (E) isomer only,:28297); the name of an input
  without the (E) stereo is systematic, and both are pinned in test_chalcone_retained_pin.py.
* 'acetylene', 'anisole': PINs,, untouched.

Changed asserted values: ``test_pin_policy.py`` (furfural -> furan-2-carbaldehyde) and the saccharin
entry of ``test_pin_deny_amino_and_trivial.py``'s not-gated list. Independent check: every PIN below
reads back to the full InChIKey of its structure with OPSIN 2.9.0 (``name_is_rt_exact`` is a separate
OPSIN call, not the code under test). The tests fail on the tree before the change (an A/B check).
Unverified by the book: mesitol, 2-acetylfuran, 1,4-phenylenediamine, orotic acid and the
nicotinaldehydes are not printed; their PINs are derived by the rules cited.
"""
import pytest
from rdkit import Chem

from orthonym.data import ALL_RETAINED_NAMES, GENERAL_RETAINED_NAMES
from orthonym.data.pin_policy import is_pin_denied
from orthonym.data.retained_names import RETAINED_NAMES
from tests.support.pin_tiers import (
    assert_declined_at_default, assert_not_pin_labelled, assert_pin_at_both_tiers,
)

pytestmark = pytest.mark.opsin_gate

# (input SMILES, trivial table name, PIN)
_WITH_PIN = [
    ("C#CCO", "propargyl alcohol", "prop-2-yn-1-ol"),
    ("Cc1cc(C)c(O)c(C)c1", "mesitol", "2,4,6-trimethylphenol"),
    ("O=Cc1ccco1", "furfural", "furan-2-carbaldehyde"),
    ("CC(=O)c1ccco1", "2-acetylfuran", "1-(furan-2-yl)ethan-1-one"),
    ("O=Cc1cccnc1", "nicotinaldehyde", "pyridine-3-carbaldehyde"),
    ("O=Cc1ccncc1", "isonicotinaldehyde", "pyridine-4-carbaldehyde"),
    ("Nc1ccc(N)cc1", "1,4-phenylenediamine", "benzene-1,4-diamine"),
    ("O=C(O)c1cc(=O)[nH]c(=O)[nH]1", "orotic acid",
     "2,6-dioxo-1,2,3,6-tetrahydropyrimidine-4-carboxylic acid"),
    ("O=C(O)CCCCCCC(=O)O", "suberic acid", "octanedioic acid"),
    ("O=C(O)CCCCCCCC(=O)O", "azelaic acid", "nonanedioic acid"),
    ("O=C(O)CCCCCCCCC(=O)O", "sebacic acid", "decanedioic acid"),
]
_SACCHARIN = "O=C1NS(=O)(=O)c2ccccc21"
_ALL = [(s, t) for s, t, _ in _WITH_PIN] + [(_SACCHARIN, "saccharin")]


@pytest.mark.parametrize("smiles, trivial", _ALL)
def test_the_trivial_name_is_denied_and_demoted_not_deleted(smiles, trivial):
    canonical = Chem.CanonSmiles(smiles)
    assert is_pin_denied(trivial), trivial
    # the PIN-path lookup no longer holds it...
    assert canonical not in ALL_RETAINED_NAMES
    #... the name is still reachable as a general-nomenclature name (--trivial), and the raw table keeps it
    assert GENERAL_RETAINED_NAMES.get(canonical) == trivial
    assert RETAINED_NAMES.get(canonical) == trivial


@pytest.mark.parametrize("smiles, trivial, pin", _WITH_PIN)
def test_the_systematic_pin_is_emitted_at_both_tiers(smiles, trivial, pin):
    assert_pin_at_both_tiers(smiles, pin)
    assert_not_pin_labelled(smiles, trivial)


def test_saccharin_is_no_pin_and_the_default_tier_declines_it():
    """ (:33246): the sultam PIN is a heterone, not a trivial name. No PIN is built for this
    structure (the lambda-6 ring-sulfur heterone class), so the default tier declines; best-effort
    still names it, systematically and RT-exact."""
    assert_declined_at_default(_SACCHARIN)
    assert_not_pin_labelled(_SACCHARIN, "saccharin")


def test_dimethylformamide_stays_the_pin():
    """ (:32774) 'dimethylformamide (PIN)' (:32782); (:3007) omits the locants."""
    assert not is_pin_denied("dimethylformamide")
    assert ALL_RETAINED_NAMES.get(Chem.CanonSmiles("CN(C)C=O")) == "dimethylformamide"
    assert_pin_at_both_tiers("CN(C)C=O", "dimethylformamide")
