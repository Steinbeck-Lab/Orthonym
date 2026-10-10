"""Leads program L4, item 30(b): an oxoacid with a substitutable hydrogen is a functional parent.

 "Substitution of mononuclear noncarbon oxoacids with hydrogen atoms attached to the
central atom (substitutable hydrogen)" (the Blue Book): the carbon framework is a prefix on
the acid word -- 'ethylphosphonic acid (PIN) (not ethanephosphonic acid)' (:35461),
'[2-(methoxysulfonyl)phenyl]phosphonic acid (PIN)',:36540). Its Note (:35457):
"Another method has been suggested which would treat the acid as a suffix (like sulfonic acid)
leading to names such as benzenephosphonic acid. This suggestion has been rejected...".

The polyfunctional suffix assembly wrote '<parent>-1-phosphonic acid' for an acid with a junior
group; it now OFFERS the dedicated namer first (``rules.phosphorus.name_oxoacid_functional_parent``)
and builds the suffix form only when the namer declines. The suffix form is labelled below the PIN
(``non_pin_vocabulary``), so the default tier declines it.
"""
import importlib
import inspect

import pytest
from rdkit import Chem

from orthonym.rules.phosphorus import name_oxoacid_functional_parent
from tests.support.pin_tiers import assert_declined_at_default, assert_pin_at_both_tiers
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

# every expected name is read back by OPSIN to the full InChIKey (test_expected_name_reads_back)
PIN_ROWS = [
    ("O=P(O)(O)CCO", "(2-hydroxyethyl)phosphonic acid"),
    ("O=P(O)(O)CCCCOC", "(4-methoxybutyl)phosphonic acid"),
    ("OCCCCCP(O)(O)=O", "(5-hydroxypentyl)phosphonic acid"),
    ("OP(O)(=O)c1ccc(cc1)[N+](=O)[O-]", "(4-nitrophenyl)phosphonic acid"),
    # stereodescriptors stay inside the organyl's marks
    ("OC[C@@H](O)CP(O)(O)=O", "[(2R)-2,3-dihydroxypropyl]phosphonic acid"),
    ("O[C@H](CP(O)(O)=O)c1ccccc1", "[(2S)-2-hydroxy-2-phenylethyl]phosphonic acid"),
    # controls built right before and after
    ("OP(O)(=O)c1ccccc1", "phenylphosphonic acid"),
    ("OP(O)(=O)c1ccc(O)cc1", "(4-hydroxyphenyl)phosphonic acid"),
    ("O=P(O)(O)CCC(=O)O", "3-phosphonopropanoic acid"),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_expected_name_reads_back(smiles, pin):
    assert name_is_rt_exact(pin, smiles), pin


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_functional_parent_name_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_suffix_form_is_declined_at_the_default_tier():
    """The wider tier still names the molecule (the suffix form is a correct systematic name),
    the default tier no longer labels it a PIN."""
    # the non-PIN suffix spelling of a molecule above is a correct name (OPSIN's read-back)
    assert name_is_rt_exact("2-hydroxyethane-1-phosphonic acid", "O=P(O)(O)CCO")
    assert_declined_at_default("OB(O)/C=C/C[C@H](C)O")


# A charge-separated nitro group is a neutral substituent; a real cation is senior to the acid.
def test_nitro_is_not_an_ion():
    mol = Chem.MolFromSmiles("OP(O)(=O)c1ccc(cc1)[N+](=O)[O-]")
    acid = tuple(mol.GetSubstructMatches(Chem.MolFromSmarts("[PX4](=O)(O)(O)c"))[0])
    assert name_oxoacid_functional_parent(mol, "phosphonic_acid", [acid]) == "(4-nitrophenyl)phosphonic acid"


def test_cation_substituent_declines_the_offer():
    mol = Chem.MolFromSmiles("OP(O)(=O)c1ccc(cc1)[N+](C)(C)C")
    acid = tuple(mol.GetSubstructMatches(Chem.MolFromSmarts("[PX4](=O)(O)(O)c"))[0])
    assert name_oxoacid_functional_parent(mol, "phosphonic_acid", [acid]) is None


def test_offer_needs_exactly_one_acid_group():
    mol = Chem.MolFromSmiles("OP(O)(=O)CCCCP(O)(O)=O")
    acids = [tuple(m) for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[PX4](=O)(O)(O)C"))]
    assert len(acids) == 2
    assert name_oxoacid_functional_parent(mol, "phosphonic_acid", acids) is None


# Rows whose PIN the default tier cannot build yet: the producer of the organyl prefix is a
# separate namer (assembly/substituent_*), listed in the report of lane L4.
@pytest.mark.xfail(strict=True, reason=(
    "assembly.substituent_enumerator.name_substituent refuses the 4-(dimethylsulfamoyl)phenyl "
    "branch (recursion_depth_fallback on the sulfamoyl group, token 'substituent'), so the "
    "functional-parent name cannot be built at the default tier; the best-effort tier names it"))
def test_sulfamoylphenyl_phosphonic_acid_at_default_tier():
    assert_pin_at_both_tiers("OP(O)(=O)c1ccc(cc1)S(=O)(=O)N(C)C",
                             "[4-(dimethylsulfamoyl)phenyl]phosphonic acid")


@pytest.mark.xfail(strict=True, reason=(
    "BB P-67.1.4.1 'arsono' is not a prefix of the substituent namer: the organyl "
    "'4-arsonobutyl' cannot be spelled; the book name is '(4-arsonobutyl)phosphonic acid'"))
def test_arsonobutyl_phosphonic_acid_book_name():
    assert_pin_at_both_tiers("O=P(O)(O)CCCC[As](=O)(O)O", "(4-arsonobutyl)phosphonic acid")


@pytest.mark.xfail(strict=True, reason=(
    "name_substituent refuses an organyl that carries both a double-bond and a tetrahedral "
    "descriptor ('OB(O)/C=C/C[C@H](C)O': token 'substituent'; the E-only and S-only organyls "
    "name), so the functional-parent name cannot be built; the PIN is "
    "'[(1E,4S)-4-hydroxypent-1-en-1-yl]boronic acid'"))
def test_stereo_boronic_acid_pin():
    assert_pin_at_both_tiers("OB(O)/C=C/C[C@H](C)O", "[(1E,4S)-4-hydroxypent-1-en-1-yl]boronic acid")


# A stereo-bearing organyl on a single-group acid goes through the dedicated handler, which must
# declare the scope to the stereo injector (patch L4-30b-oxoacid-handlers-stereo): strict xfail until
# the patch is applied, an ordinary test afterwards.
NEEDS_OXOACID_HANDLER_PATCH = pytest.mark.xfail(
    "retained_no_locants" not in inspect.getsource(
        importlib.import_module("orthonym.assembly.handlers.phosphonic_acid")),
    strict=True, reason="needs patch L4-30b-oxoacid-handlers-stereo (assembly/handlers/phosphonic_acid.py)")


@NEEDS_OXOACID_HANDLER_PATCH
def test_stereo_organyl_on_a_single_acid():
    assert name_is_rt_exact("[(2E)-but-2-en-1-yl]phosphonic acid", "C/C=C/CP(O)(O)=O")
    assert_pin_at_both_tiers("C/C=C/CP(O)(O)=O", "[(2E)-but-2-en-1-yl]phosphonic acid")
