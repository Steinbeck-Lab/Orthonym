"""The group of an ester, an aryl ether and an alkoxycarbonyl is named from STRUCTURE.

Four producers wrote a group name from a count or from the alcohol's own name, and so named
another molecule (OPSIN reads each of the old names as a different constitution):

* ``get_alkoxycarbonyl_prefix`` spelled the alkyl of a partial ester from its carbon COUNT:
  8-methylnonyl was 'decyloxycarbonyl', isopropyl 'propoxycarbonyl', allyl 'propoxycarbonyl',
  vinyl 'ethoxycarbonyl', 4-methylphenyl '(benzyloxy)carbonyl' (measured at the commit this
  lane branched from). "Partial esters of polybasic acids and their salts"
  (the Blue Book): "Method (1) generates preferred IUPAC names." (:31940), the ester
  cited as a prefix ('2-chloro-6-(ethoxycarbonyl)benzoic acid (PIN)',:31950); the alkoxy
  part follows "Retained names" (:27665).
* ``_assemble_ester`` glued the alcohol's own name in front of the acid word when the alcohol's
  oxygen was a 'hydroxy' prefix, or when it had several hydroxy groups:
  (the Blue Book), "All preferred IUPAC names for esters are named by functional class
  nomenclature", the group name first ('ethyl acetate').
* ``_assemble_glycoside`` read a polyol aglycone as '-diyl' / '-triyl' ('benzene-1,4-diyl
  beta-D-galactopyranoside': a group with two free valences, OPSIN cannot parse it). The group
  word of an aglycone with other hydroxy groups is built from the aglycone's STRUCTURE
  (``_aglycone_polyol_group``) and shipped only after it round-trips to the parent:
  "Names" (the Blue Book), "Glycosides are named by using functional class
  nomenclature.... The class name is preceded, as a separate word, by the name of the
  substituent group that is part of the acetal or ketal function." Only a group senior to
  hydroxy turns a glycoside substitutive,:53915: "not 4-acetylphenyl
  beta-D-glucopyranoside; a ketone is senior to a hydroxy compound").
* ``get_alkoxy_prefix`` declined a decorated fused or heterocyclic aryl and its caller dropped
  the whole aryloxy group ('butanoic acid' for 4-[(5-chloroquinolin-8-yl)oxy]butanoic acid):
   (the Blue Book); the Blue Book '(5-chloropyridin-2-yl)oxy'.

Each row names the molecule at the default tier (the suite runs with the OPSIN gate off, so the
raw producer output is what is asserted) and checks the name with a FRESH OPSIN parse against the
input's full InChIKey, outside the engine (tests/support/rt_assert.py::name_is_rt_exact).
"""
import pytest

from orthonym import name_compound

pytestmark = [pytest.mark.unit, pytest.mark.integration]

ROWS = [
    # --- alkoxycarbonyl: the alkyl is a structure, not a count -------------------------------
    ("CC(C)CCCCCCCOC(=O)c1ccccc1C(=O)O", "2-{[(8-methylnonyl)oxy]carbonyl}benzoic acid"),
    ("CC(C)OC(=O)c1ccccc1C(=O)O", "2-{[(propan-2-yl)oxy]carbonyl}benzoic acid"),
    ("CCC(C)OC(=O)c1ccccc1C(=O)O", "2-{[(butan-2-yl)oxy]carbonyl}benzoic acid"),
    ("CC(C)(C)OC(=O)c1ccccc1C(=O)O", "2-(tert-butoxycarbonyl)benzoic acid"),
    ("CC(C)COC(=O)c1ccccc1C(=O)O", "2-[(2-methylpropoxy)carbonyl]benzoic acid"),
    ("C=COC(=O)c1ccccc1C(=O)O", "2-[(ethenyloxy)carbonyl]benzoic acid"),
    ("C=CCOC(=O)c1ccccc1C(=O)O", "2-{[(prop-2-en-1-yl)oxy]carbonyl}benzoic acid"),
    ("C#CCOC(=O)c1ccccc1C(=O)O", "2-{[(prop-2-yn-1-yl)oxy]carbonyl}benzoic acid"),
    ("Cc1ccc(OC(=O)c2ccccc2C(=O)O)cc1", "2-[(4-methylphenoxy)carbonyl]benzoic acid"),
    ("O=C(Oc1cccc2ccccc12)c1ccccc1C(=O)O",
     "2-{[(naphthalen-1-yl)oxy]carbonyl}benzoic acid"),
    ("O=C(OCc1ccc(C)cc1)c1ccccc1C(=O)O",
     "2-{[(4-methylphenyl)methoxy]carbonyl}benzoic acid"),
    ("O=C(OCCc1ccccc1)c1ccccc1C(=O)O", "2-[(2-phenylethoxy)carbonyl]benzoic acid"),
    # the unbranched chain keeps the count spelling it always had
    ("CCCCCCCCCCOC(=O)c1ccccc1C(=O)O", "2-(decyloxycarbonyl)benzoic acid"),
    ("CCCCOC(=O)c1ccccc1C(=O)O", "2-(butoxycarbonyl)benzoic acid"),
    # --- aryl ether: a decorated fused / heterocyclic aryl is named, never dropped -----------
    ("OC(=O)CCCOc1ccc(Cl)c2cccnc12", "4-[(5-chloroquinolin-8-yl)oxy]butanoic acid"),
    ("O=C(O)COc1ccc(Cl)c2cccnc12", "[(5-chloroquinolin-8-yl)oxy]acetic acid"),
    ("NCCCOc1ccc(Cl)c2cccnc12", "3-[(5-chloroquinolin-8-yl)oxy]propan-1-amine"),
    # --- ester: the group word of an alcohol whose name cannot carry it -----------------------
    ("CCCCCC(C)OC(=O)COc1ccc(Cl)c2cccnc12",
     "heptan-2-yl [(5-chloroquinolin-8-yl)oxy]acetate"),
    ("CCCCCCCCCCCCC1=C(OC(C)=O)C(=O)c2ccccc2C1=O",
     "3-dodecyl-1,4-dioxo-1,4-dihydronaphthalen-2-yl acetate"),
    ("CCC(C(CC)C(=O)OC[N+](C)(C)C)C(=O)N",
     "(trimethylazaniumyl)methyl 3-carbamoyl-2-ethylpentanoate"),
    # an alcohol with several hydroxy groups: the linking oxygen is the one that rebuilds the parent
    ("O=C(O[C@H]1Cc2c(O)cc(O)cc2O[C@@H]1c1ccc(O)c(O)c1)c1cc(O)c(O)c(O)c1",
     "(2R,3S)-2-(3,4-dihydroxyphenyl)-5,7-dihydroxy-3,4-dihydro-2H-1-benzopyran-3-yl "
     "3,4,5-trihydroxybenzoate"),
    # --- glycoside of a polyol: the group word is built from the aglycone's structure, never the
    # --- fabricated '-diyl' / '-triyl' functional-class word
    ("OC[C@H]1O[C@@H](Oc2ccc(O)cc2)[C@H](O)[C@@H](O)[C@H]1O",
     "4-hydroxyphenyl β-D-galactopyranoside"),
    ("C([C@@H]1[C@@H]([C@H]([C@H]([C@H](O1)OC(CO)CO)O)O)O)O",
     "1,3-dihydroxypropan-2-yl α-D-gulopyranoside"),
    ("OCCO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O",
     "2-hydroxyethyl β-D-glucopyranoside"),
    ("OC[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O",
     "(2R,3S,4R,5S)-2,3,4,5,6-pentahydroxyhexyl α-D-glucopyranoside"),
    # a monovalent string word that is right stays (it round-trips): salicin, gastrodin
    ("OC[C@H]1O[C@@H](Oc2ccccc2CO)[C@H](O)[C@@H](O)[C@@H]1O",
     "2-(hydroxymethyl)phenyl β-D-glucopyranoside"),
    ("OC[C@H]1O[C@@H](Oc2ccc(CO)cc2)[C@H](O)[C@@H](O)[C@@H]1O",
     "4-(hydroxymethyl)phenyl β-D-glucopyranoside"),
    # a polyol aglycone whose group the substituent namer cannot spell with the locant of its
    # stereocentre ('(S)-6-hydroxy-...-8-yl') is not offered; the general name stands
    ("CCCCC[C@H]1Cc2c(C)c(O)c(C)c(O[C@@H]3O[C@@H](C)[C@H](O)[C@@H](O)[C@H]3O)c2CO1",
     "(3S)-5,7-dimethyl-3-pentyl-8-(α-L-rhamnopyranosyloxy)-3,4-dihydro-1H-2-benzopyran-6-ol"),
    # --- aryl ether whose aryl holds a ketone / lactone / pyranone: recognised groups, named
    ("CC(C)=CCc1cc2ccc(=O)oc2c(O)c1OC1OC(C(=O)O)C(O)C(O)C1O",
     "3,4,5-trihydroxy-6-{[8-hydroxy-6-(3-methylbut-2-en-1-yl)-2-oxo-2H-1-benzopyran-7-yl]oxy}"
     "oxane-2-carboxylic acid"),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_group_name_is_built_from_structure(smiles, expected):
    from tests.support.rt_assert import name_is_rt_exact

    name = name_compound(smiles)
    assert name == expected, name
    assert name_is_rt_exact(name, smiles), name


# An aryl that holds a group the functional-group perception does not know (an O-alkyl
# thiocarbamate) must not be hidden inside an aryloxy prefix: the producer that chose the
# parent cannot rank it, so the name would carry an alcohol as the parent under the label of
# a preferred name. "Seniority order for classes" (the Blue Book): an ester-type
# group outranks an alcohol.
UNRECOGNISED_GROUP_IN_ARYL = [
    "COC(=S)NCc1ccc(OC2OC(C)C(O)C(O)C2O)cc1",
    "COC(=S)NCc1ccc(OCC(O)CO)cc1",
    "COC(=S)NCc1ccc2ccccc2c1OCC(O)CO",
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", UNRECOGNISED_GROUP_IN_ARYL)
def test_unrecognised_senior_group_in_an_aryl_is_not_labelled_a_pin(smiles):
    from tests.support.default_tier import default_tier_row

    row = default_tier_row(smiles)
    assert row["tier"] != "pin_verified" and row["is_pin"] is False, row


@pytest.mark.parametrize("alcohol", [
    "ethane-1,2-diol", "propane-1,2,3-triol", "butane-1,2,3,4-tetrol",
    "pentane-1,2,3,4,5-pentol", "hexane-1,2,3,4,5,6-hexol",
    "(2R,3S,4R)-pentane-1,2,3,4,5-pentol",
])
def test_a_multiplied_ol_is_not_read_as_one_group_word(alcohol):
    """A multiplied '-ol' (the multiplier's final 'a' elided before the vowel, "Elision of
    vowels", (c), the Blue Book) names an alcohol with several hydroxy groups; no
    monovalent group word can be read from it ('pentane-1,2,3,4,5-pentyl' was shipped)."""
    from orthonym.decomposition.fragment_assembly import (
        _alcohol_to_alkyl,
        _is_monovalent_group_word,
    )

    assert not _is_monovalent_group_word(_alcohol_to_alkyl(alcohol))


@pytest.mark.parametrize("alcohol,word", [
    ("ethanol", "ethyl"), ("propan-2-ol", "propan-2-yl"), ("cyclopentanol", "cyclopentyl"),
])
def test_a_single_ol_is_still_one_group_word(alcohol, word):
    from orthonym.decomposition.fragment_assembly import (
        _alcohol_to_alkyl,
        _is_monovalent_group_word,
    )

    got = _alcohol_to_alkyl(alcohol)
    assert got == word and _is_monovalent_group_word(got)


@pytest.mark.parametrize("word,bare", [
    ("(S)-6-hydroxy-5,7-dimethyl-3-pentyl-3,4-dihydro-1H-2-benzopyran-8-yl", True),
    ("(R,S)-1-phenylethyl", True),
    ("(3S)-6-hydroxy-5,7-dimethyl-3-pentyl-3,4-dihydro-1H-2-benzopyran-8-yl", False),
    ("(2R,3S,4R)-2,3,4,5-tetrahydroxypentyl", False),
    ("(E)-2-phenylethenyl", False),
])
def test_a_cip_descriptor_without_its_locant_is_seen(word, bare):
    """ "Naming of stereoisomers" (the Blue Book, the paragraph after the heading):
    descriptors "are preceded by a numerical or letter locant to describe the position of the
    stereogenic unit when such locants are present". Only R/S are tested: '(E)-2-phenylethenyl'
    has one possible position."""
    from orthonym.decomposition.fragment_assembly import _has_unlocanted_cip_descriptor

    assert _has_unlocanted_cip_descriptor(word) is bare
