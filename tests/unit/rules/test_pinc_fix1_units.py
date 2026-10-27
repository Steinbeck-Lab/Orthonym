"""PINC lane, fix a performance pass: the pure parts (no OPSIN) of the adduct class order, the parent amide chain check, the added
hydrogen check and the multiplier of an anion word. The names at both tiers are pinned in ``test_pin_class_pinc_built.py``
and ``test_pin_class_pinc_labels.py``.

* (the Blue Book) with (:3769): class 21 "Nitrogen compounds: heterocycles, polyazanes,
  hydrazines..." (:18197), 26 "Silicon compounds" (:18202), 35 "Oxygen compounds: heterocycles" (:18211), 40 "Carbon
  compounds: rings, chains" (:18216), 41 ethers (:18217); (:21033) "has the greater number of multiple bonds".
* (:21033) before (:21604) for the parent chain of two amides.
* (:45978), example (:45994) for the cation at the nitrogen of a hydro-azine.
* (:7029) (a) and (c) (:7035), (e) (:7104) for 'di' against 'bis' before an anion word.
"""
import pytest
from rdkit import Chem

from orthonym.rules.adducts import component_sort_key
from orthonym.rules.salt_multiplier import is_unsubstituted_locanted_word
from orthonym.validation.spelling.checks_amide_chains import amide_chain_parent_check
from orthonym.validation.spelling.checks_hydrogen import added_hydrogen_at_ring_nitrogen_check


def _ordered(smiles):
    return sorted((Chem.CanonSmiles(s) for s in smiles), key=component_sort_key)


@pytest.mark.parametrize("first,second", [
    ("NNc1ccccc1", "c1ccoc1"),                    # 21 hydrazines before 35 oxygen heterocycles
    ("c1ccc(cc1)[Si](C)(C)C", "c1ccoc1"),         # 26 silanes before 35
    ("C[Si](C)(C)C", "c1ccoc1"),
    ("c1ccncc1", "c1ccoc1"),                      # 21 before 35
    ("c1ccoc1", "c1ccccc1"),                      # 35 before 40
    ("c1ccccc1", "C1CCCCC1"),                     #: three double bonds before none
    ("c1ccccc1", "C1=CCCCC1"),                    # three before one
    ("c1ccc2ccccc2c1", "c1ccc2CCCCc2c1"),         # naphthalene before tetralin
    ("c1ccccc1", "c1ccc(Oc2ccccc2)cc1"),          # 40 before the ether, 41
    ("c1ccoc1", "c1ccc(Oc2ccccc2)cc1"),           # 35 before the ether, 41
])
def test_component_class_order(first, second):
    assert _ordered([second, first]) == [Chem.CanonSmiles(first), Chem.CanonSmiles(second)]
    assert _ordered([first, second]) == [Chem.CanonSmiles(first), Chem.CanonSmiles(second)]


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def test_amide_chain_of_equal_length_with_the_multiple_bond_is_the_parent():
    # the primary amide chain has the double bond: it is the parent although the N-methyl chain has more prefixes
    assert amide_chain_parent_check(_mol("NC(=O)/C=C/c1ccc(CCC(=O)NC)cc1"),
                                    "(2E)-3-{4-[3-(methylamino)-3-oxopropyl]phenyl}prop-2-enamide") is None
    # saturated chains: the N-methyl chain has the locant 'N', the name of the primary amide is no PIN
    assert amide_chain_parent_check(_mol("NC(=O)CCc1ccccc1CCC(=O)NC"),
                                    "3-{2-[3-(methylamino)-3-oxopropyl]phenyl}propanamide") is not None
    # the N-methyl chain carries the double bond: the name on the primary chain is no PIN
    assert amide_chain_parent_check(_mol("NC(=O)CCc1ccc(/C=C/C(=O)NC)cc1"),
                                    "3-{4-[(1E)-3-(methylamino)-3-oxoprop-1-en-1-yl]phenyl}propanamide") is not None


@pytest.mark.parametrize("name,flagged", [
    ("1,1-dimethyl-1,2,3,4-tetrahydroquinolin-1-ium", False),               #:45994
    ("1,2,3,4-tetrahydroquinolin-1-yl", True),
    ("1,2,3,4-tetrahydroquinoline-1-carboxylic acid", True),
    ("1,2,3,4-tetrahydroquinolin-1-ol", True),
    ("1-methyl-1,2,3,4-tetrahydroquinoline", False),
    ("1,2,3,4-tetrahydroquinolin-4-ol", False),
])
def test_added_hydrogen_check_leaves_the_ium_site_alone(name, flagged):
    assert (added_hydrogen_at_ring_nitrogen_check(None, name) is not None) is flagged


@pytest.mark.parametrize("name,simple", [
    ("(9Z)-octadec-9-enoate", True),
    ("prop-2-enoate", True),
    ("(2E,4E)-hexa-2,4-dienoate", True),
    ("(3E,5E)-hepta-3,5-dienoate", True),
    ("cyclohex-2-ene-1-carboxylate", True),
    ("benzene-1,4-dicarboxylate", True),
    ("2-ethylhexanoate", False),                  # substituted: 'bis',:7035; 'bis(chloroacetate)':31825
    ("(2S)-2-hydroxypropanoate", False),
    ("4-chlorobenzoate", False),
    ("4-oxopent-2-enoate", False),
    ("octadecanoate", False),                     # no locant: the plain multiplier
    ("acetate", False),
])
def test_unsubstituted_word_with_locants_takes_di(name, simple):
    assert is_unsubstituted_locanted_word(name) is simple
