"""A mono-anion whose centre is a hydroxy or thiol group is named on that group, however senior a
neutral group of the molecule is (``rules.charged_router.olate_with_senior_neutral_group``).

 "SENIORITY ORDER FOR CLASSES" (the Blue Book), Table 4.1: class 4 'Anions' is senior to
class 7 'Acids' and to every class below it. The anion is the parent, its group the principal
characteristic group, and the senior neutral group is cited as a prefix, as in '5-carboxypentanoate
(PIN)' (:40997). "Anions derived from hydroxy compounds" (:41011): the centre is named
by the suffix 'olate' or 'thiolate'. "Partial salts of polyols and their chalcogen
analogues are named substitutively on the basis of the corresponding anion" (:28203): 'sodium
3-hydroxypropan-1-olate (PIN)', 'sodium 2-sulfanylbenzene-1-thiolate (PIN)'.

The charged route names the neutral form and gives its suffix the ending of the anion. Since an
oxoacid with a substitutable hydrogen became a functional parent "Substitution of
mononuclear noncarbon oxoacids with hydrogen atoms attached to the central atom",:35453), the
neutral form of a phenolate with an arsonous acid group is '(4-hydroxy-3-nitrophenyl)arsonous acid',
which does not end in 'ol', and the phenolate lost its name. Every name below is read back by an
independent OPSIN call (``tests.support.rt_assert.name_is_rt_exact``: the full standard InChIKey).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.rules import charged_router
from orthonym.perception.ions import get_ion_sites
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

#: the lost row, its sibling without the nitro group, and the class members whose neutral form
#: has another senior group: the oxoacids of C, S, P, As and Sb, an amide, a nitrile, an aldehyde,
#: a ketone, an acid halide and an amino acid, on a chain, on benzene (the ester too,,
#: the Blue Book: 'alkoxycarbonyl' on a ring), on an aliphatic ring and on a heteroaromatic
#: ring, with a thiol centre as well as a hydroxy one
PIN_ROWS = [
    ("O=[N+]([O-])c1cc([As](O)O)ccc1[O-]", "4-(dihydroxyarsanyl)-2-nitrophenolate"),
    ("[O-]c1ccc([As](O)O)cc1", "4-(dihydroxyarsanyl)phenolate"),
    ("[S-]c1ccc([As](O)O)cc1", "4-(dihydroxyarsanyl)benzene-1-thiolate"),
    ("[O-]CCC(=O)O", "2-carboxyethan-1-olate"),
    ("[O-]CCP(=O)(O)O", "2-phosphonoethan-1-olate"),
    ("[O-]CCS(=O)(=O)O", "2-sulfoethan-1-olate"),
    ("[S-]CCC(=O)O", "2-carboxyethane-1-thiolate"),
    ("O=C(O)CC([S-])C(=O)O", "1,2-dicarboxyethane-1-thiolate"),
    ("O=CCc1ccc([O-])c([N+](=O)[O-])c1", "2-nitro-4-(2-oxoethyl)phenolate"),
    # a carboxylic acid, a sulfonic acid, an aldehyde, an amide, a nitrile on benzene
    ("[O-]c1ccc(cc1)C(=O)O", "4-carboxyphenolate"),
    ("[O-]c1ccccc1C(=O)O", "2-carboxyphenolate"),
    ("[O-]c1ccc(cc1)S(=O)(=O)O", "4-sulfophenolate"),
    ("[O-]c1ccc(cc1)C=O", "4-formylphenolate"),
    ("[O-]c1ccc(cc1)C(N)=O", "4-carbamoylphenolate"),
    ("[O-]c1ccc(C#N)cc1", "4-cyanophenolate"),
    ("O=[N+]([O-])c1cc(C(=O)O)ccc1[O-]", "4-carboxy-2-nitrophenolate"),
    ("[O-]c1ccc(OC)cc1C(=O)O", "2-carboxy-4-methoxyphenolate"),
    # phosphonic, arsonic and stibonic acid groups ('arsono' and 'stibono':
    ("[O-]c1ccc(cc1)P(=O)(O)O", "4-phosphonophenolate"),
    ("[O-]c1ccc([As](=O)(O)O)cc1", "4-arsonophenolate"),
    ("[O-]c1ccc([Sb](=O)(O)O)cc1", "4-stibonophenolate"),
    ("O=[N+]([O-])c1cc([As](=O)(O)O)ccc1[O-]", "4-arsono-2-nitrophenolate"),
    ("[O-]CC[As](=O)(O)O", "2-arsonoethan-1-olate"),
    # a ketone and an ester on benzene are prefixes ('acetyl', 'alkoxycarbonyl')
    ("CC(=O)c1ccc([O-])cc1", "4-acetylphenolate"),
    ("[O-]c1ccc(cc1)C(=O)c1ccccc1", "4-benzoylphenolate"),
    ("[O-]c1ccc(cc1)C(=O)OC", "4-(methoxycarbonyl)phenolate"),
    ("[O-]c1ccccc1C(=O)OC", "2-(methoxycarbonyl)phenolate"),
    # a thiol centre, a benzylic and a chain alcohol, an aliphatic ring, a heteroaromatic ring
    ("[S-]c1ccc(cc1)C(=O)O", "4-carboxybenzene-1-thiolate"),
    ("[S-]c1ccccc1C(=O)O", "2-carboxybenzene-1-thiolate"),
    ("[S-]c1ccc(C(C)=O)cc1", "4-acetylbenzene-1-thiolate"),
    ("[O-]Cc1ccc(C(=O)O)cc1", "(4-carboxyphenyl)methanolate"),
    ("[O-]CCC(C)=O", "3-oxobutan-1-olate"),
    ("[O-]CCC=O", "3-oxopropan-1-olate"),
    ("[O-]CCC#N", "2-cyanoethan-1-olate"),
    ("[O-]CCC(N)=O", "3-amino-3-oxopropan-1-olate"),
    ("[O-]C1CCC(C(=O)O)CC1", "4-carboxycyclohexan-1-olate"),
    ("[O-]C1CCC(C(=O)OC)CC1", "4-(methoxycarbonyl)cyclohexan-1-olate"),
    ("[O-]c1ccc(C(=O)O)nc1", "6-carboxypyridin-3-olate"),
    ("[O-]c1ccc(C(=O)O)o1", "5-carboxyfuran-2-olate"),
    # an amino acid: the retained name is not the parent of the anion
    ("[O-]C[C@H](N)C(=O)O", "(2S)-2-amino-2-carboxyethan-1-olate"),
]

#: members whose name is not built: the producer that names the neutral form decides the principal
#: group of its own, or the anion class is not known. Each expected name is the shape the Blue Book
#: gives; the first test below reads it back, so that the shape is checked, and the second asserts
#: that the row is still declined, strictly: building the member turns the row into an error.
DECLINED_ROWS = [
    # an ester on a chain: no producer cites 'alkoxy...oxo' for a sulfonic acid, an aminium or an
    # olate,:31696; '3-ethoxy-3-oxopropan-1-aminium bromide (PIN)',:31709)
    ("[O-]CCC(=O)OC", "3-methoxy-3-oxopropan-1-olate",
     "no producer cites an ester as 'alkoxy...oxo' on a chain whose principal group is not an acid"),
    # a lactone is named on a ring parent that has its own suffix
    ("[O-]CC1CCC(=O)O1", "(5-oxooxolan-2-yl)methanolate",
     "the heterocycle producer names the lactone as the suffix 'one'"),
    # a fused ring system: the polycyclic writers decide the suffix from the molecule
    ("[O-]c1ccc2ccccc2c1C(=O)O", "1-carboxynaphthalen-2-olate",
     "the polycyclic and fused-ring writers decide the principal group from the molecule"),
    # classify_anion does not know a selenium or tellurium anion (benzeneselenolate abstains too)
    ("[Se-]CCC(=O)O", "2-carboxyethane-1-selenolate",
     "classify_anion does not classify a selenolate"),
    # the ring writer's phosphorus prefix ignores the hydroxy groups of -P(OH)2 ('4-phosphanylphenol')
    ("[O-]c1ccc(P(O)O)cc1", "4-(dihydroxyphosphanyl)phenolate",
     "get_phosphanyl_prefix names -P(OH)2 'phosphanyl', a different molecule"),
    # the chain writer has no prefix for -As(OH)2
    ("[O-]CC[As](O)O", "2-(dihydroxyarsanyl)ethan-1-olate",
     "no producer names -As(OH)2 as a prefix on a chain"),
]

TIERS = ["pin", "best-effort"]


def _row(smiles, tier):
    engine = Orthonym(style="pin", **(_emit_tier_flags(tier) if tier != "pin" else {}))
    return engine.name_tiered(smiles)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("smiles, pin", PIN_ROWS)
def test_the_anion_is_the_parent_and_the_senior_neutral_group_a_prefix(smiles, pin, tier):
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("smiles, pin", PIN_ROWS)
def test_the_name_is_the_one_the_helper_builds(smiles, pin):
    """The route returns the name, not a later producer (no other route names these)."""
    mol = Chem.MolFromSmiles(smiles)
    assert charged_router.route_charged(mol, "pin") == pin


def test_the_neutral_phenol_is_still_named_on_its_oxoacid():
    # an oxoacid with a substitutable hydrogen is senior to a hydroxy compound,
    # class 7c over class 17): the neutral form keeps the acid as the parent; only the anion
    # (class 4) reverses the order
    for smiles, name in [("Oc1ccc([As](O)O)cc1", "(4-hydroxyphenyl)arsonous acid"),
                         ("O=[N+]([O-])c1cc([As](O)O)ccc1O", "(4-hydroxy-3-nitrophenyl)arsonous acid")]:
        row = _row(smiles, "pin")
        assert (row["name"], row["tier"]) == (name, "pin_verified"), row


def test_a_second_hydroxy_group_leaves_the_name_to_the_other_routes():
    # a second -OH would be a second suffix of the forced neutral name ('...-1,4-diol') and the
    # ending would give the dianion: the route does not name it, and nothing it names is wrong
    smiles = "[O-]c1ccc(O)cc1"
    mol = Chem.MolFromSmiles(smiles)
    neutral = "Oc1ccc(O)cc1"
    assert charged_router.olate_with_senior_neutral_group(
        mol, get_ion_sites(mol), neutral, "pin") == ""
    row = _row(smiles, "best-effort")
    if row["name"] and row["tier"] != "abstain":
        assert name_is_rt_exact(row["name"], smiles), row


def test_a_cation_or_a_radical_is_not_this_routes_molecule():
    for smiles in ("[O-]CC[NH3+]", "[O-]CC[CH2]"):
        mol = Chem.MolFromSmiles(smiles)
        sites = get_ion_sites(mol)
        assert charged_router.olate_with_senior_neutral_group(
            mol, sites, Chem.MolToSmiles(mol), "pin") == ""


@pytest.mark.parametrize("smiles, expected, why", DECLINED_ROWS)
def test_the_expected_name_of_a_declined_member_is_the_molecule(smiles, expected, why):
    assert name_is_rt_exact(expected, smiles), expected


@pytest.mark.parametrize("smiles, expected, why", DECLINED_ROWS)
@pytest.mark.xfail(strict=True, reason="declined member of the olate class, see DECLINED_ROWS")
def test_a_declined_member_is_named(smiles, expected, why):
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), (why, row)


@pytest.mark.parametrize("smiles, expected, why", DECLINED_ROWS)
def test_a_declined_member_is_never_named_wrong(smiles, expected, why):
    for tier in TIERS:
        row = _row(smiles, tier)
        if row["name"] and row["tier"] != "abstain":
            assert name_is_rt_exact(row["name"], smiles), (tier, row)


#: neutral molecules whose ring writer now reads the molecule's principal group: their names do not
#: change (the principal group of each is the senior group, so no hydroxy group is promoted over it)
NEUTRAL_ROWS = [
    ("Oc1ccc(C(=O)O)cc1", "4-hydroxybenzoic acid"),
    ("Oc1ccccc1C(=O)O", "2-hydroxybenzoic acid"),
    ("CC(=O)c1ccc(O)cc1", "1-(4-hydroxyphenyl)ethan-1-one"),
    ("COC(=O)c1ccc(O)cc1", "methyl 4-hydroxybenzoate"),
    ("Oc1ccc(C=O)cc1", "4-hydroxybenzaldehyde"),
    ("Sc1ccc(C(=O)O)cc1", "4-sulfanylbenzoic acid"),
    ("Oc1ccc(S)cc1", "4-sulfanylphenol"),
    ("O=C(O)c1ccc(C(=O)OC)cc1", "4-(methoxycarbonyl)benzoic acid"),
]


@pytest.mark.parametrize("smiles, name", NEUTRAL_ROWS)
def test_a_neutral_name_is_unchanged(smiles, name):
    row = _row(smiles, "pin")
    assert row["name"] == name, row
    assert name_is_rt_exact(row["name"], smiles), row


def test_the_ring_promotion_follows_a_hydroxy_principal_group():
    from orthonym.rules.benzene import benzene_prefix_suffix_promotion as promote
    # a hydroxy principal group is promoted over a senior suffix; without the argument it is not
    assert promote({"carboxylic acid"}, {"hydroxy"}) == (None, frozenset())
    assert promote({"carboxylic acid"}, {"hydroxy"}, principal_group="phenol") == ("ol", frozenset({"hydroxy"}))
    # a junior principal group that holds the suffix slot keeps it
    assert promote({"thiol"}, {"hydroxy"}, principal_group="thiol") == ("thiol", frozenset())
    assert promote({"thiol"}, {"hydroxy"}) == ("ol", frozenset({"hydroxy"}))
    # a senior principal group changes nothing
    assert promote({"carboxylic acid"}, {"hydroxy"}, principal_group="carboxylic_acid") == (None, frozenset())


@pytest.mark.parametrize("smiles, name", [
    ("Oc1ccc([As](=O)(O)O)cc1", "arsono"), ("Oc1ccc([Sb](=O)(O)O)cc1", "stibono"),
    ("Oc1ccc([As](=O)(O)C)cc1", None), ("Oc1ccc([As](O)O)cc1", None),
    ("Oc1ccc([As](=O)(O)OC)cc1", None), ("Oc1ccc(P(=O)(O)O)cc1", None),
])
def test_the_pnictogen_oxoacid_prefix(smiles, name):
    from orthonym.rules.benzene import _bfs_substituent_atoms
    from orthonym.rules.benzene_prefix_forms import pnictogen_oxoacid_prefix
    mol = Chem.MolFromSmiles(smiles)
    ring = {a for r in mol.GetRingInfo().AtomRings() for a in r}
    centre = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in ("P", "As", "Sb"))
    assert pnictogen_oxoacid_prefix(mol, _bfs_substituent_atoms(mol, centre, ring), centre) == name
