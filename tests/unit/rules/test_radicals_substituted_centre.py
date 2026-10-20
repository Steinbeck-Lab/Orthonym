"""UN lane R: a substituted B / Si / Ge / Sn / Pb / Al / Ga / P / As / Sb radical centre is the parent of its name.

``C[B]c1ccccc1`` is 'methyl(phenyl)boranyl' because the parent of the neutral 'methyl(phenyl)borane' IS the
radical atom, so the radical ending fits ``charged_router._textual_radical_name``. When the neutral's parent is a
ring or a chain ('2-boranyloxolane', '2-boranylacetic acid') no ending fits, and the producer
``radicals.name_heteroatom_radical`` used to return '' for every non-nitrogen substituted centre ("out of
scope, fail closed"); the molecule then went to a skeletal-replacement name ('3-hydroxy-4-oxa-1-borabut-3-en-1-ylidene'
for B-CH2-COOH) or to no name at all.

Blue Book, Group 14 (Si, Ge, Sn, Pb): (the Blue Book), "A radical formally derived by the
removal of one hydrogen atom from a mononuclear parent hydride of an element of Group 14... is named by
replacing the 'ane' ending of the systematic name of the parent hydride by 'yl'"; 'Specific method
and retained names' (:40474) is the same with '-ylidene' and '-ylidyne' ('benzylsilylidene (PIN)',:40490).
B, Al, Ga, P, As, Sb: 'Monovalent radicals' (:40426), "A radical formally derived by the removal of
one hydrogen atom from any position of a parent hydride, or a modified parent hydride other than those
described by... is named by adding the suffix 'yl' to the name of the parent hydride, eliding the
final letter 'e'"; the example 'tert-butoxytri(phenyl)-lambda5-phosphanyl (PIN)' (:40442) is a substituted
mononuclear parent hydride radical. 'General method' (:40498) is the same for 'ylidene' and
'ylidyne'. 'SENIORITY ORDER FOR CLASSES', Table 4.1 (:18162): "1 Radicals", the most senior class, so a
group a suffix would otherwise express stays a prefix ('carboxymethyl', 'oxo', 'octanoyl').
(:7272): "the first cited substituent never has enclosing marks unless it includes a locant. The second and
further substituents are each enclosed with parentheses even for simple substituents. When the simple
substituent groups are accompanied by multiplicative prefixes such as 'di' and 'tri', the multiplicative
prefixes are not included in the parentheses" ('ethyldi(methyl)phosphane (PIN)'). 'Standard bonding
numbers' (:2744), Table 1.3: 3 for B, Al, Ga, 4 for Si, Ge, Sn, Pb, 3 for P, As, Sb.

A centre that is a skeletal atom of a longer parent hydride of its OWN element is left to the chain names:
 'Heterogeneous parent hydrides composed of alternating heteroatoms' (:8014), "an unbranched chain of
alternating atoms terminated by two identical atoms", 'disiloxane (preselected name, ' (:7631),
 'Mononuclear and acyclic parent hydrides' (:38010), 'trisilan-2-yl (preselected prefix)' (:16005).
A neighbour of another element makes no chain: 'Compounds with groups of higher seniority' (:37465),
'(difluoroboranyl)tri(methyl)silane (PIN) (Si is senior to B)' (:37520).

Every expected name is read back by OPSIN to the input's full InChIKey and radical profile
(``tests.support.rt_assert.name_is_rt_exact``).
"""
import pytest
from rdkit import Chem

import orthonym.rules.radicals as rad
from orthonym.perception.ions import get_radical_sites
from tests.support.pin_tiers import (assert_not_pin_labelled, assert_pin_at_both_tiers, name_breadth,
                                     name_default)
from tests.support.rt_assert import name_is_rt_exact

# (SMILES, PIN): the neutral compound's parent is the ring or the chain, never the centre; each name is read
# back by OPSIN to the input's full key and radical profile.
PINS = [
    ("[B]CC(=O)O", "(carboxymethyl)boranylidene"),
    ("[B]CCO", "(2-hydroxyethyl)boranylidene"),
    ("[B]CCCl", "(2-chloroethyl)boranylidene"),
    ("[B]C#N", "cyanoboranylidene"),
    ("[B]C1CCC(=O)CC1", "(4-oxocyclohexyl)boranylidene"),
    ("[B]C(=O)CCCCCCC", "octanoylboranylidene"),
    ("[SiH2]CC=O", "(2-oxoethyl)silyl"),
    ("C[Si](C)CC(=O)O", "(carboxymethyl)di(methyl)silyl"),
    ("[Si](OC)(OC)CC(=O)O", "(carboxymethyl)di(methoxy)silyl"),
    ("C[Si](C)C(=O)C", "acetyldi(methyl)silyl"),
    ("[Ge](C)(C)CC(=O)O", "(carboxymethyl)di(methyl)germyl"),
    ("CC[P]CC(=O)O", "(carboxymethyl)(ethyl)phosphanyl"),
    ("C[As]CC(=O)O", "(carboxymethyl)(methyl)arsanyl"),
    ("[B](OC)OC", "dimethoxyboranyl"),      # the best-effort tier named it below the PIN, the default tier not at all
    # Group 14 below carbon: RDKit's default valence of Sn and Pb is 2, the standard bonding number 4
    ("C[Ge](C)C1CCCO1", "dimethyl(oxolan-2-yl)germyl"),
    ("C[Sn](C)C1CCCO1", "dimethyl(oxolan-2-yl)stannyl"),
    ("C[Pb](C)C1CCCO1", "dimethyl(oxolan-2-yl)plumbyl"),
    ("C[Sn](C)CC(=O)O", "(carboxymethyl)di(methyl)stannyl"),
    # a neighbour of ANOTHER element is no chain: the boron stays on the silicon,:37520)
    ("C[B][Si](C)(C)C", "methyl(trimethylsilyl)boranyl"),
    ("[B]O[Si](C)(C)C", "[(trimethylsilyl)oxy]boranylidene"),
    ("C[Si](C)(C)[B]F", "fluoro(trimethylsilyl)boranyl"),
    # controls: the neutral's parent is the centre, the name is the one the radical ending always gave
    ("C[B]c1ccccc1", "methyl(phenyl)boranyl"),
    ("[B]c1ccccc1", "phenylboranylidene"),
    ("C[Si](C)(C)", "trimethylsilyl"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", PINS, ids=[s for s, _ in PINS])
def test_the_substituted_centre_is_named_on_its_parent_hydride_radical(smiles, pin):
    """The default tier and the best-effort tier both give the name, labelled PIN."""
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("[B](CCC/C=C/C)C(=O)CCCCCCC/C=C/C", "[(4E)-hex-4-en-1-yl][(9E)-undec-9-enoyl]boranyl"),
    ("CC1=CC2=C(C=C1)N(C(=C2)[Si](C)C)C(=O)OC(C)(C)C",
     "[1-(tert-butoxycarbonyl)-5-methyl-1H-indol-2-yl]di(methyl)silyl"),
    ("[B][C@@H]1CCCO1", "[(2R)-oxolan-2-yl]boranylidene"),
], ids=["enoyl", "indolyl", "oxolanyl"])
def test_a_ligand_that_states_a_descriptor_is_named_but_not_labelled_a_pin(smiles, name):
    """The hand-written names of the lane brief, read back by the independent OPSIN call. A stereodescriptor
    inside a radical's prefix keeps the name out of the PIN tier (``radicals._ship``, brief item B10): the
    default tier does not name the row as a PIN, the best-effort tier names it below the PIN."""
    assert name_is_rt_exact(name, smiles), name
    d = name_default(smiles)
    assert not (d.get("tier") == "pin_verified" and d.get("is_pin")), d
    b = name_breadth(smiles)
    assert b.get("name") == name and b.get("tier") != "pin_verified", b


# ---------------------------------------------------------------------------
# The centre that is a skeletal atom of a longer parent hydride is not the mononuclear construction's
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_the_trisilanyl_pin_is_kept():
    """'trisilan-2-yl (preselected prefix)' (:16005): the mononuclear construction would say 'disilylsilyl'."""
    assert_pin_at_both_tiers("[SiH3][SiH]([SiH3])", "trisilan-2-yl")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,not_a_pin", [
    ("[SiH2][SiH3]", "silylsilyl"),                                       # disilanyl
    ("C[Si](C)O[Si](C)(C)C", "dimethyl[(trimethylsilyl)oxy]silyl"),       # disiloxanyl
    ("[B](C)OB(C)C", "[(dimethylboranyl)oxy](methyl)boranyl"),            # diboroxanyl
], ids=["disilanyl", "disiloxanyl", "diboroxanyl"])
def test_a_chain_parent_is_not_spelled_as_a_substituted_mononuclear_one(smiles, not_a_pin):
    assert_not_pin_labelled(smiles, not_a_pin)


def _sites(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol, get_radical_sites(mol)


@pytest.mark.parametrize("smiles", [
    "[SiH2][SiH3]",                # a Group-14 neighbour: disilane
    "C[Si](C)[Si](C)(C)C",
    "[B]B",                        # B-B
    "C[Si](C)O[Si](C)(C)C",        # Si-O-Si: disiloxane
    "[B](C)N(C)B(C)C",             # B-N-B
    "[B](F)[Si](C)(C)B(F)F",       # B-Si-B: an a(ba)n chain with a silicon between two borons
    "C=[Si]",                      # a double bond at the centre
    "[SiH]1CCCC1",                 # the centre is a ring atom
    "COC(=O)C[B-]",                # a charged centre
    "[B][N+](C)(C)C",              # an ionic ligand
    "C[P](C)(C)C",                 # non-standard valence (lambda5)
    "[B]",                         # no ligand: the mononuclear producer's, not this one's
    "[2H][B]C",                    # a labelled hydrogen the name could not state
], ids=lambda s: s)
def test_the_construction_declines_what_is_not_a_substituted_mononuclear_centre(smiles):
    """The producer returns '' before any name is built, so the caller's own routes run unchanged."""
    mol, sites = _sites(smiles)
    centre = [s for s in sites if s["element"] in rad._CENTRE_RADICAL_ELEMENTS]
    assert centre, smiles
    assert rad._name_substituted_centre_radical(mol, centre[0]) == ""


def test_standard_bonding_numbers_are_those_of_table_1_3():
    """ / Table 1.3 (the Blue Book-2746): 3 for B, Al, Ga and P, As, Sb; 4 for Si, Ge, Sn, Pb."""
    assert rad._STANDARD_BONDING_NUMBER == {
        "B": 3, "Al": 3, "Ga": 3, "Si": 4, "Ge": 4, "Sn": 4, "Pb": 4, "P": 3, "As": 3, "Sb": 3}
    assert rad._CENTRE_RADICAL_ELEMENTS == frozenset(rad._STANDARD_BONDING_NUMBER)


@pytest.mark.parametrize("smiles,expected", [
    ("[SiH2][SiH3]", True), ("[B]B", True), ("C[Si](C)O[Si](C)(C)C", True), ("[B](C)N(C)B(C)C", True),
    ("[B](F)[Si](C)(C)B(F)F", True), ("[B](C)OB(C)C", True),
    # another element beside the centre, or between the centre and a different element: no chain of one element
    ("C[B][Si](C)(C)C", False), ("[B]O[Si](C)(C)C", False), ("C[Si](C)(C)[B]F", False),
    ("[B](C)OC", False), ("[B](C)N(C)C", False), ("[Si](C)(C)C[Si](C)(C)C", False), ("[B]c1ccccc1", False),
])
def test_centre_extends_to_chain(smiles, expected):
    mol, sites = _sites(smiles)
    assert rad._centre_extends_to_chain(mol.GetAtomWithIdx(sites[0]["atom_idx"])) is expected


@pytest.mark.parametrize("text,found", [
    ("[(2R)-oxolan-2-yl]boranylidene", True),
    ("[(2R,4aS,8aS)-decahydronaphthalen-2-yl]boranylidene", True),
    ("[(4E)-hex-4-en-1-yl]boranyl", True),
    ("{[(1E)-prop-1-en-1-yl]}silyl", True),
    ("[(3'R)-oxolan-2-yl]boranylidene", True),
    ("[(4'aR,8'aS)-decahydronaphthalen-2-yl]boranylidene", True),
    ("[(Z)-but-2-en-2-yl]silyl", True),
    ("[(2RS,3SR)-3-methyloxolan-2-yl]boranylidene", True),
    ("[(1R*,2S*)-2-methylcyclohexyl]silyl", True),
    ("[(1r,4r)-4-methylcyclohexyl]boranylidene", True),
    ("[(Ra)-buta-1,2-dien-1-yl]silyl", True),
    ("[(aS)-biphenyl-2-yl]boranylidene", True),
    ("[(M)-hexahelicen-1-yl]boranylidene", True),
    ("[(P)-hexahelicen-1-yl]boranylidene", True),
    ("(2H,4H-1,3-benzodioxin-6-yl)di(chloro)silyl", False),      # indicated hydrogen is no descriptor
    ("[1-(tert-butoxycarbonyl)-5-methyl-1H-indol-2-yl]di(methyl)silyl", False),
    ("(carboxymethyl)boranylidene", False),
    ("[(phenylsulfanyl)methyl]boranyl", False),
    ("(propan-2-yl)(phosphanyl)silyl", False),
])
def test_ligand_descriptor_test_sees_every_form_of_a_stereodescriptor(text, found):
    """``radicals._ship``'s own test needs a number and one of R, S, E, Z; a fused-ring descriptor '(4aR,8aS)', a
    primed one "(4'aR)", a racemic or relative one '(2RS,3SR)', '(1R*,2S*)', an axial or helical one '(Ra)', '(M)',
    '(P)' has more, and the producer must not ship such a name from the PIN tier either."""
    assert bool(rad._LIGAND_DESCRIPTOR.search(text)) is found


# ---------------------------------------------------------------------------
# What the construction must not change
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("[BH]c1ccccc1", "phenylboranyl"),
    ("[B](F)F", "difluoroboranyl"),
    ("C[Sn](C)C", "trimethylstannyl"),
    ("[SiH2]c1ccccc1", "phenylsilyl"),
])
def test_a_centre_whose_neutral_parent_is_the_centre_keeps_its_name(smiles, name):
    assert_pin_at_both_tiers(smiles, name)


# ---------------------------------------------------------------------------
# Only the molecule the outermost name call was given (universal_halane.names_the_input)
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_a_nested_frame_is_not_named_by_the_construction(monkeypatch):
    """The engine names a substituent group, or a capped fragment, by sending the fragment through the radical
    namer. Inside such a frame the outermost input is another molecule, and the construction names nothing: its
    name would be a prefix of the outer name and its label would lift the whole name."""
    import orthonym.assembly.fragment_naming as fragment_naming
    mol, sites = _sites("[B]CCO")
    assert rad._name_substituted_centre_radical(mol, sites[0]) == "(2-hydroxyethyl)boranylidene"
    monkeypatch.setattr(fragment_naming, "outermost_name_input", lambda: "CC[SiH2]C(=O)OC")
    assert rad._name_substituted_centre_radical(mol, sites[0]) == ""
    # the same molecule, drawn another way, is the molecule the call was asked about
    monkeypatch.setattr(fragment_naming, "outermost_name_input", lambda: "OCC[B]")
    assert rad._name_substituted_centre_radical(mol, sites[0]) == "(2-hydroxyethyl)boranylidene"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name,label", [
    # BB 61.5.1: the book PIN is 'trimethyl(nitro)silane'; 68.1.5.2.3: '(difluoroboranyl)tri(methyl)silane'
    ("C[Si](C)(C)[N+](=O)[O-]", "[dimethyl(nitro)silyl]methane", "systematic_verified"),
    ("C[Si](C)(C)B(F)F", "[(difluoroboranyl)di(methyl)silyl]methane", "systematic_verified"),
    ("CC(Cl)[AsH]CCCl", "1-chloro-2-[(1-chloroethyl)arsanyl]ethane", "systematic_verified"),
    ("CCS(=O)(=O)[S][Ge]([CH3])([CH3])[CH3]",
     "1-(3,3-dimethyl-1,1-dioxo-1λ6,2-dithia-3-germabutyl)ethane", "systematic_verified"),
    ("CCO[Si](SC)(SC)SC", "1-{[tris(methylsulfanyl)silyl]oxy}ethane", "systematic_verified"),
    ("CCN(CC)C(=S)[S][Sb]([CH](C)C)[CH](C)C",
     "2-[4-ethyl-1-(propan-2-yl)-3-sulfanylidene-2-thia-4-aza-1-stibahexyl]propane", "systematic_verified"),
    ("CSC[SiH2]CSC[SiH2]CCN",
     "2-({[({[(methylsulfanyl)methyl]silyl}methyl)sulfanyl]methyl}silyl)ethan-1-amine", "systematic_verified"),
    ("C=C[Si](C)(N(C)C(C)=O)N(C)C(C)=O",
     "N-acetyl-1-{[acetyl(methyl)amino](methyl)(methylamino)silyl}eth-1-ene", "systematic_verified"),
    ("CCC(=O)O[Si](C)(C)OOC(C)(C)C", "(tert-butylperoxy)di(methyl)silyl propanoate", "pin_unverified"),
    ("C[SiH2]C[SiH2]C[SiH2]C[SiH2]CCCl",
     "1-chloro-2-({[({[(methylsilyl)methyl]silyl}methyl)silyl]methyl}silyl)ethane", "systematic_verified"),
    ("C[SiH2]C[SiH2]C[SiH2]C[SiH2]CC(C)=O",
     "1-({[({[(methylsilyl)methyl]silyl}methyl)silyl]methyl}silyl)propan-2-one",
     "systematic_verified"),
], ids=["nitrosilane", "boranylsilane", "arsane", "germane", "silane", "stibane", "sulfanyl-chain", "acetamido",
        "silyl-ester", "chloro-chain", "oxo-chain"])
def test_a_closed_shell_name_built_from_radical_fragments_is_the_name_before_lane_r(smiles, name, label):
    """These closed-shell molecules are named through capped fragments with a radical centre (a silyl, a germyl,
    an arsanyl group). Before the outermost-input gate, the construction named those fragments, its label lifted
    the whole name to pin_verified for names the book says are not the PIN (BB 61.5.1, 68.1.5.2.3, 68.3.2.3.2.1,
    51.4.1.2), and several names changed. The name and label are those of the tree before lane R (a
    characterisation, not a claim that they are the PIN), and the name reads back."""
    res = name_breadth(smiles)
    assert (res.get("name"), res.get("tier")) == (name, label), res
    assert name_is_rt_exact(name, smiles), res


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "COCOCOCOCC1COCC(C[SiH2]C[SiH2]C[SiH2]C[SiH2]C)C1",       # BB 44.1.2.1
    "COP(=O)(OC)OC[SiH2]C[SiH2]C[SiH2]C[SiH2]C",              # BB 51.4.1.2
], ids=["oxane", "phosphate"])
def test_a_replacement_chain_of_the_book_is_not_renamed_by_its_radical_fragments(smiles):
    """ replacement ('a') chains: the book names the silicon chain '2,4,6,8-tetrasilanonyl'; the
    construction named its radical fragments and the name became a chain of '(methylsilyl)methyl' groups."""
    res = name_breadth(smiles)
    assert "2,4,6,8-tetrasilanonyl" in res["name"], res
    assert "(methylsilyl)methyl" not in res["name"], res


@pytest.mark.opsin_gate
def test_the_silyl_radical_of_a_fused_core_is_named_below_the_pin():
    """The whole-molecule silyl radical that tests/unit/test_g0_fail_closed.py used as its 'unnameable branch'
    witness. The construction names it on silyl, the Blue Book) with the fused ring as a
    prefix, and the independent OPSIN -r read-back (full InChIKey and radical graph) accepts the name.
    UNVERIFIED: the spelling of the fused prefix and its enclosing marks (the ligand namer's; the book is not
    checked for this core), which is why the label stays below the PIN; the default tier does not name the row."""
    smiles = "[Si](C)(C)c1cc2nc3ccoc3cc2o1"
    name = "[difuro[3,2-b:2',3'-e]pyridin-2-yl]di(methyl)silyl"
    assert name_is_rt_exact(name, smiles), name
    b = name_breadth(smiles)
    assert b.get("name") == name and b.get("tier") != "pin_verified", b
    d = name_default(smiles)
    assert d.get("tier") != "pin_verified", d

