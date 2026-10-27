"""Silicon, boron and germanium centres as the parent of a cation, a radical anion or a radical pair.

Every name is read back by a fresh OPSIN call to the input's full InChIKey (``assert_full_rt``, with ``-r``
only for an input that has radical electrons), so a test passes with the engine's own gate off or on.

Rules (the Blue Book):
* "Cationic centers in parent hydrides", "Specific method" (:41553): the Group 14
  mononuclear hydride loses a hydride ion and 'ane' becomes 'ylium'; "(C6H5)3Si+ triphenylsilylium (PIN)"
  (:41557). The general method of the same section (:41559) gives the Group 13 centre the same ending.
* "Examples of radical anions" (:43424): the radical anion of an anionic parent hydride takes 'yl'
  on the anion's name, 'azanidyl' (:43426), 'trimethylboranuidyl (PIN)' (:43430).
* "ASSEMBLIES OF PARENT RADICALS" (:40739): identical parent radicals in different parts of the
  structure are named by the principles of; "Retained prefixes that are preferred prefixes"
  (:16270,:16282): 'methylene', not 'methanediyl'.
"""
import pytest

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("c1ccccc1[Si+](c1ccccc1)c1ccccc1", "triphenylsilylium"),   # (PIN) the Blue Book
    ("C[Si+](C)C", "trimethylsilylium"),
    ("C[SiH2+]", "methylsilylium"),
    ("C[SiH+]C", "dimethylsilylium"),
    ("C[Ge+](C)C", "trimethylgermylium"),                       # Group 14, specific method
    ("C[B+]C", "dimethylboranylium"),                           # Group 13, general method:41559
    ("CO[Si+](C)C", "methoxydi(methyl)silylium"),               # citation of the groups
])
def test_group14_and_group13_ylium_is_the_preferred_name(smiles, expected):
    row = default_tier_row(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(expected, smiles)


def test_silylium_with_a_ligand_that_holds_rings_reads_back():
    # a census row: the ligands go through the substituent-prefix namer, the centre stays the parent
    smiles = "COC1=CN=C(N=C1OCC[Si+](C)C)Cl"
    row = name_best_effort(smiles)
    assert row["name"] == "{2-[(2-chloro-5-methoxypyrimidin-4-yl)oxy]ethyl}di(methyl)silylium", row
    assert_full_rt(row["name"], smiles)


def test_silylium_is_the_parent_even_with_a_carboxylic_acid_in_a_ligand():
    # (the Blue Book): a cation outranks an acid, which stays a 'carboxy' prefix
    smiles = "C[Si+](C)CC(=O)O"
    row = name_best_effort(smiles)
    assert row["name"] == "(carboxymethyl)di(methyl)silylium", row
    assert_full_rt(row["name"], smiles)


def test_tin_cation_is_not_named_here():
    # tin is a metal of data/metal_scope.py, admitted to no cation mechanism: route_charged hands it on
    assert name_best_effort("C[Sn+](C)C")["tier"] == "abstain"


@pytest.mark.parametrize("smiles", [
    "C[Si+]([Si](C)(C)C)[Si](C)(C)C",     # 'heptamethyltrisilan-2-ylium (PIN)', the Blue Book: the chain is the parent
    "C[Si-]([Si](C)(C)C)(C)C",            # the same for the lambda silanide
])
def test_centre_in_a_silicon_chain_is_not_named_as_a_mononuclear_parent(smiles):
    from rdkit import Chem
    from orthonym.rules.group14_centres import emit_group_ylium, name_lambda_ide_anion
    mol = Chem.MolFromSmiles(smiles)
    idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge())
    assert emit_group_ylium(mol, idx) == "" and name_lambda_ide_anion(mol) == ""


def test_ylium_centre_in_a_ring_is_left_alone():
    # a ring atom is a skeletal atom of a ring parent hydride, not of this mononuclear one
    from rdkit import Chem
    from orthonym.rules.group14_centres import emit_group_ylium
    mol = Chem.MolFromSmiles("C1CC[SiH+]C1")
    idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Si")
    assert emit_group_ylium(mol, idx) == ""


@pytest.mark.parametrize("smiles,expected", [
    ("C[Si-]C", "dimethylsilanidyl"),
    ("COC[C@H]1CC(CN1C(=O)C=C)[Si-]C",
     "[(5R)-5-(methoxymethyl)-1-(prop-2-enoyl)pyrrolidin-3-yl](methyl)silanidyl"),   # nested marks,
])
def test_silicon_radical_anion_is_silanidyl(smiles, expected):
    row = name_best_effort(smiles)
    assert row["name"] == expected, row
    assert_full_rt(expected, smiles)


def test_radical_anion_with_a_stereodescriptor_in_its_prefix_is_not_a_pin():
    # brief item B10 (radicals._ship): a descriptor in a radical's prefix is never a PIN
    smiles = "COC[C@H]1CC(CN1C(=O)C=C)[Si-]C"
    assert not name_best_effort(smiles)["is_pin"]
    assert default_tier_row(smiles)["tier"] == "abstain"


@pytest.mark.parametrize("smiles,expected", [
    ("C[Si-](C)(C)C", "tetramethyl-\u03bb5-silanide"),
    ("CC(C)(C)[Si-](C)(C)OC", "tert-butyl(methoxy)di(methyl)-\u03bb5-silanide"),
    ("c1ccccc1[Si-](c1ccccc1)(c1ccccc1)c1ccccc1", "tetraphenyl-\u03bb5-silanide"),
])
def test_four_bonded_silicon_anion_is_a_lambda_silanide_below_the_pin(smiles, expected):
    # "ANIONS FORMED BY ADDITION OF HYDRIDE IONS", method (2) (the Blue Book): 'ide' on a parent
    # hydride whose bonding number at the anionic site is expressed by the lambda-convention
    # ('tetramethyl-lambda5-phosphanide',:41106). Method (1) would name R4Si(-) as the radical anion
    # 'tetramethylsilanuidyl',:43424), which OPSIN reads only under -r; the read-back policy gives
    # -r to an input with radical electrons and RDKit counts none on this atom, so the lambda name, which
    # OPSIN reads to the input without -r, is the one shipped, and never as a PIN.
    row = name_best_effort(smiles)
    assert row["name"] == expected and row["tier"] == "systematic_verified" and not row["is_pin"], row
    assert_full_rt(expected, smiles)
    assert default_tier_row(smiles)["tier"] == "abstain"


@pytest.mark.parametrize("smiles,expected", [
    ("[B]C[B]", "methylenebis(boranylidene)"),
    ("[B]C([B])O", "(hydroxymethylene)bis(boranylidene)"),
    ("[B]C([B])(c1ccccc1)O", "[hydroxy(phenyl)methylene]bis(boranylidene)"),
    ("[B]C([B])(Cl)Cl", "(dichloromethylene)bis(boranylidene)"),
])
def test_two_boranylidene_on_one_carbon_are_an_assembly_of_parent_radicals(smiles, expected):
    row = name_best_effort(smiles)
    assert row["name"] == expected, row
    assert_full_rt(expected, smiles)
    # in preferred-name form, whose status no example of the book certifies: not a PIN
    assert not row["is_pin"] and row["tier"] != "pin_verified", row
    assert default_tier_row(smiles)["tier"] == "abstain"


@pytest.mark.parametrize("smiles,expected", [
    ("[B]C([B])C", "1,1-di(λ1-boranyl)ethane"),         # a carbon chain would be the parent of the linking group
    ("[B]CC[B]", "1,2-di(λ1-boranyl)ethane"),           # two carbons: no producer builds ethane-1,2-diylbis(boranylidene)
    ("[B]C1(CCCC1)[B]", "1,1-di(λ1-boranyl)cyclopentane"),   # the carbon is a ring atom
])
def test_other_radical_pairs_are_named_only_by_the_last_resort(smiles, expected):
    """No producer of the assembly of parent radicals names these pairs (they were declined here), and the radical
    offer, the last resort of the best-effort tier (``assembly/radical_offer.py``), names them with each boron cited by
    the lambda convention. Each boron has one bond and two radical electrons: a divalent radical centre, which
    "The λ-convention" (the Blue Book, sentence:40562) says "may be described by the λ-convention... This method
    is only for general nomenclature", with 'phenyl-λ1-azane' (:40570) as its own example of a lambda of one: so the
    name is a verified systematic name and never a PIN. Read back by a fresh OPSIN call with ``-r`` to the input's full
    InChIKey and its radical graph (``assert_full_rt``); the default tier, which never makes this offer, still declines
    them."""
    row = name_best_effort(smiles)
    assert row["name"] == expected and row["tier"] == "systematic_verified" and not row["is_pin"], row
    assert_full_rt(expected, smiles)
    assert default_tier_row(smiles)["tier"] == "abstain"
