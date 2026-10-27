"""Fix a performance pass of lane ION: the review of the ion names, class by class.

* the triazenium PIN is not taken by the diazenium row: (the Blue Book) 'Azo imides... Method (1)
  in gives preferred IUPAC names', its example (:43121) 'the preferred name is based on the unbroken nitrogen
  chain'; the molecules keep the names and the PIN label that main gives them at both tiers;
* the trivalent '-ous' acids of P, As and Sb are named by the substituent prefix, never as a suffix:
  (:35453) "Substitution of mononuclear noncarbon oxoacids with hydrogen atoms attached to the central atom": the Note
  (:35457) rejects the suffix method, 'ethylphosphonic acid (PIN) (not ethanephosphonic acid)' (:35461), 'diphenylarsinous
  acid (PIN)' (:35465), and the anion takes 'ite' for 'ous acid', 'disodium methylphosphonite (PIN)' (:35904,
   :40955); where the book's form cannot be built the name is labelled below the PIN;
* the first group cited on a mononuclear parent has no enclosing marks unless it includes a locant, (:7272),
  'acetyl(methyl)chloranium (PIN)' (:41488): 'acetyl(dimethylamino)azanide';
* the junction cation of a spiro system is the ylium of the lambda5 hydride, (:41752), '5λ5-arsaspiro[4.4]
  nonan-5-ylium (PIN)' (:41802), at both tiers;
* the italic designator of 's-indacene' is set off by a hyphen from the hydro prefixes, (d) (:6960);
* a zwitterion whose anion is an amide or a sulfonamide nitrogen has no PIN in the book: its name is valid and is
  labelled below the PIN;
* the sulfinyl anion is 'oxo(phenyl)-λ4-sulfanide (PIN)', (:40880), and not the sulfanuide of the same ion;
  the anion of another centre with a double bond has no example and is labelled below the PIN;
* the descriptor of an isotopically modified name stands before the whole locant of the parent, 'lambda' number
  included: (:43718) 'when the name, or a part of a name, includes a preceding locant, a hyphen is inserted'.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.roundtrip, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,pin", [
    ("[N-]=[N+](C)N(C)C", "2,3,3-trimethyltriaz-1-en-2-ium-1-ide"),
    ("[N-]=[N+](C)N=NC", "2,4-dimethyltetraaz-1,3-dien-2-ium-1-ide"),
    ("[N-]=[N+](C)N", "2-methyltriaz-1-en-2-ium-1-ide"),
])
def test_the_unbroken_nitrogen_chain_keeps_its_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", [
    ("CCOC(C)(OCC)P(O)O", "(1,1-diethoxyethyl)phosphonous acid"),
    ("CCOC(C)(OCC)[As](O)O", "(1,1-diethoxyethyl)arsonous acid"),
    ("OC(C)(C)P(O)O", "(2-hydroxypropan-2-yl)phosphonous acid"),
    ("NCCP(O)O", "(2-aminoethyl)phosphonous acid"),
    ("ClCC[Sb](O)O", "(2-chloroethyl)stibonous acid"),
    ("CCOC(C)(OCC)P(O)C", "(1,1-diethoxyethyl)(methyl)phosphinous acid"),
    ("CCOC(C)(OCC)P([O-])[O-]", "(1,1-diethoxyethyl)phosphonite"),
    ("CCOC(OCC)([Si](C)(C)C)P([O-])[O-]", "[diethoxy(trimethylsilyl)methyl]phosphonite"),
])
def test_the_ous_acids_and_their_anions_are_functional_parents(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles", ["CC(=O)NCP(O)O", "C(C(=O)NCP(O)O)N", "C(C(=O)NCP(=O)(O)O)N"])
def test_an_organyl_prefix_that_names_an_isomer_is_not_offered(smiles):
    """The functional-parent offer of (:35453) reads back. The substituent namer names '-CH2-NH-C(=O)CH3' as
    '(2-amino-2-oxoethyl)methyl', an isomer: with no validity gate behind it the offer shipped that name. Declined, the
    molecule is named by the assemblies that follow it (the suffix form of the rejected method, Note:35457, or the
    substitutive form of the wider tier), and the name is read back by OPSIN independently of the namer."""
    from orthonym.namer import Orthonym
    row = Orthonym(_disable_opsin_validity_gate=True).name_tiered(smiles)
    assert row["name"] and name_is_rt_exact(row["name"], smiles), row
    assert row["tier"] != "pin_verified", row


def test_a_phosphonite_with_two_acid_groups_is_no_pin():
    """The suffix form of a polyacid ('3-phenylbutane-1,1-diphosphonite') is the rejected method of."""
    smiles = "CC(CC(P([O-])[O-])P([O-])[O-])c1ccccc1"
    row = name_breadth(smiles)
    assert row["name"] and row["tier"] != "pin_verified", row
    assert name_is_rt_exact(row["name"], smiles)


@pytest.mark.parametrize("smiles,pin", [
    ("CC(=O)[N-]N(C)C", "acetyl(dimethylamino)azanide"),
    ("CC(=O)[N-]C", "acetyl(methyl)azanide"),
    ("CS(=O)(=O)[N-]CC", "ethyl(methanesulfonyl)azanide"),
])
def test_the_first_group_of_an_azanide_has_no_marks(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_the_hypervalent_group_of_an_azanide_keeps_the_name_of_main():
    smiles = "CS(=N)(=O)[N-]C"
    row = name_breadth(smiles)
    assert row["name"] == "(1-imino-1-oxo-1-thiaethan-1-yl)(methyl)azanide", row
    assert name_is_rt_exact(row["name"], smiles)


@pytest.mark.parametrize("smiles,pin", [
    ("C1CC[N+]2(C1)CCCC2", "5λ5-azaspiro[4.4]nonan-5-ylium"),
    ("C1CCCC[N+]12CCCC2", "5λ5-azaspiro[4.5]decan-5-ylium"),
    ("CC1CC[N+]2(C1)CCCC2", "2-methyl-5λ5-azaspiro[4.4]nonan-5-ylium"),
    ("C1CCCC[P+]12CCCCC2", "6λ5-phosphaspiro[5.5]undecan-6-ylium"),
])
def test_a_spiro_junction_cation_is_the_ylium_of_its_lambda5_hydride(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_the_arsenic_spiro_cation_of_the_book_is_built():
    """'5λ5-arsaspiro[4.4]nonan-5-ylium (PIN)' (the Blue Book): the builder spells the book's name. The namer
    declines every arsenic compound before it reaches the builder (a limit of its scope, not of the builder)."""
    from rdkit import Chem

    from orthonym.rules.spiro import name_charged_spiro_system
    mol = Chem.MolFromSmiles("C1CC[As+]2(C1)CCCC2")
    cation = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1)
    assert name_charged_spiro_system(mol, cation) == "5λ5-arsaspiro[4.4]nonan-5-ylium"


@pytest.mark.parametrize("smiles,pin", [
    ("C1CC2=CC3=C(CCC3)C(=C2C1)N", "1,2,3,5,6,7-hexahydro-s-indacen-4-amine"),
    ("C1CC2=CC3=CC=CC3=CC2=C1", "1,2-dihydro-s-indacene"),
    ("CC1CCC2=C1C(=C3CCCC3=C2)O", "3-methyl-1,2,3,5,6,7-hexahydro-s-indacen-4-ol"),
])
def test_the_italic_designator_of_s_indacene_has_its_hyphen(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles", [
    "[NH3+]CCC[N-]S(C)(=O)=O",
    "C[NH2+]CC[N-]S(C)(=O)=O",
])
def test_the_zwitterion_of_an_amide_anion_has_no_pin(smiles):
    default, breadth = name_default(smiles), name_breadth(smiles)
    assert default["tier"] == "abstain" and breadth["tier"] != "pin_verified", (default, breadth)
    assert name_is_rt_exact(breadth["name"], smiles)


def test_the_sulfinyl_anion_is_the_sulfanide_of_the_book():
    """'benzenesulfinyl anion oxo(phenyl)-λ4-sulfanide (PIN)' (the Blue Book); 'oxo(phenyl)sulfanuide' named the
    same ion and was labelled as a PIN."""
    assert_pin_at_both_tiers("O=[S-]c1ccccc1", "oxo(phenyl)-λ4-sulfanide")
    assert_pin_at_both_tiers("O=[S-]C", "methyl(oxo)-λ4-sulfanide")


@pytest.mark.parametrize("smiles,name", [
    ("N=[S-]c1ccccc1", "imino(phenyl)-λ4-sulfanide"),
    ("O=[Se-]c1ccccc1", "oxo(phenyl)-λ4-selanide"),
    ("F[S-](F)(F)(F)c1ccccc1", "tetrafluoro(phenyl)-λ4-sulfanuide"),
])
def test_the_anion_of_another_hypervalent_centre_is_no_pin(smiles, name):
    row = name_breadth(smiles)
    assert row["name"] == name and row["tier"] != "pin_verified", row
    assert name_is_rt_exact(name, smiles)
    assert name_default(smiles)["tier"] == "abstain"


@pytest.mark.parametrize("smiles,pin", [
    ("[2H]C1(CC[N+]2(C1)CCCC2)[2H]", "(2,2-2H2)-5λ5-azaspiro[4.4]nonan-5-ylium"),
    ("[2H]C1CCC[N+]12CCCC2", "(1-2H1)-5λ5-azaspiro[4.4]nonan-5-ylium"),
    ("[2H]C1CCCC[P+]12CCCCC2", "(1-2H1)-6λ5-phosphaspiro[5.5]undecan-6-ylium"),
    ("CC1([2H])CC[N+]2(C1)CCCC2", "2-methyl(2-2H)-5λ5-azaspiro[4.4]nonan-5-ylium"),
])
def test_the_isotope_descriptor_goes_before_the_lambda_locant_of_the_parent(smiles, pin):
    """P-82.2.1 (BlueBookV2.md:43718): 'when the name, or a part of a name, includes a preceding locant, a hyphen is
    inserted', '(15N)-1H-indole (PIN)' (:43790). The locant '5λ5' of the parent is one locant: the descriptor stands
    before it, as before '5-azaspiro' (main: '(2,2-2H2)-5-azaspiro[4.4]nonan-5-ium'), never between the 5 and its λ."""
    assert_pin_at_both_tiers(smiles, pin)


def test_a_labelled_spiro_cation_with_substituents_is_named():
    smiles = "[2H]C1(C(N(C([N+]12CCCCC2)C)C3=C(C=CC4=C3OC5=CC=CC=C45)C)(C)C)[2H]"
    row = name_breadth(smiles)
    assert row["name"] and "λ5-diazaspiro[4.5]decan-5-ylium" in row["name"] and row["tier"] == "pin_verified", row
    assert name_is_rt_exact(row["name"], smiles)
