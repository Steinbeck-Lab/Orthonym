"""Functional-class polyol polyester names method (1)): read back through the acyloxy twin.

OPSIN 2.9.0 has no grammar for the multi-anion form 'propane-1,2,3-triyl 1,3-diacetate 2-propanoate', and
the name used to ship with no read-back (the ``polyol_polyester`` grammar-gap carve-out): it carried the
name of an anion for a half ester of a polybasic acid ('propane-1,2,3-triyl 1,3-diacetate 2-butanedioate'
for the glycerol diacetate hemisuccinate, a molecule with a free COOH), and the CO audit sample (2026-10-17)
had 8 such names among 121.

The rules:

* (the Blue Book) "When anions are different, two methods are used: (1) names of anions
  are cited in alphanumerical order preceded by a locant, when required; multiplicative prefixes are used to
  denote a multiplicity of identical anionic components; (2) one anion is chosen as principal anion and all
  other ester groups are expressed as prefixes in the name of the organyl group." (:31833-:31834); "Method (1)
  generates preferred IUPAC names but names formed by using method (2) are acceptable in general
  nomenclature." (:31836); 'propane-1,2,3-triyl 1,2-diacetate 3-propanoate (PIN)' (:31840).
* (:31842) 'propane-1,2,3-triyl 2-acetate 1-hexadecanoate 3-[(9Z)-octadec-9-enoate] (PIN)'
  (:31846) and, with the configuration of C-2, '(2S)-propane-1,2,3-triyl 2-acetate 1-hexadecanoate
  3-[(9Z)-octadec-9-enoate]' (:55113).
* (:31968): "Substitutive nomenclature is senior to functional class nomenclature for preferred
  IUPAC for esters": the acyloxy spelling names the same ester ('4-(acetyloxy)benzoic acid (PIN)').
* Partial esters of polybasic acids and their salts (:31935): "(1) substitutively on the basis of
  the anion, the free acid group(s) and the ester group(s) being cited as prefixes" (:31937); "Method (1)
  generates preferred IUPAC names." (:31941); Acid esters of organic acids (:40993): "Preferred
  IUPAC names of acid esters of 'organic acids' as discussed in are formed substitutively [...] rather
  than by the method of 'hydrogen salts'" ('5-carboxypentanoate (PIN)',:40997). A half ester is named as the
  acid: the dianion word states no acid hydrogen.
* (:31819): "When anions are identical functional class multiplicative nomenclature is used. Names
  are formed by citing the multivalent group, the multiplicative prefix, and the multiplied anionic component name.
  Multiplicative prefixes 'di', 'tri', etc. are used when anions are unsubstituted; when substituted, prefixes
  'bis', 'tris', etc. are used"; 'ethane-1,2-diyl diacetate (PIN)' and 'propane-1,3-diyl bis(chloroacetate) (PIN)'
  (:31825). 'dichloroacetate' is the anion of dichloroacetic acid, so two chloroacetate anions are 'bis(chloroacetate)'.
* (:7255): "Parentheses are used around simple substituent prefixes (see to separate
  locants of the same type referring to different structural elements": an anion that starts with a locant of
  its own is enclosed after the attachment locant, '2-(4-hydroxybutanoate)'.
"""
import pytest
from rdkit import Chem

from orthonym.validation.polyester_twin import polyester_twin
from orthonym.validation.twin_spellings import twin_of, twin_round_trip
from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import _canon, _independent_parse, assert_full_rt, name_best_effort

# (smiles, the name the producer gives, the twin) -- the first two are the Blue Book's printed PINs
BB_ROWS = [
    ("CC(=O)OCC(OC(=O)C)COC(=O)CC", "propane-1,2,3-triyl 1,2-diacetate 3-propanoate",
     "1,2-bis[acetyloxy]-3-[propanoyloxy]propane"),                                             #:31840
    ("CC(=O)OC(COC(=O)CCCCCCCCCCCCCCC)COC(=O)CCCCCCC/C=C\\CCCCCCCC",
     "propane-1,2,3-triyl 2-acetate 1-hexadecanoate 3-[(9Z)-octadec-9-enoate]",
     "2-[acetyloxy]-1-[hexadecanoyloxy]-3-[(9Z)-octadec-9-enoyloxy]propane"),                   #:31846
    ("CC(=O)O[C@@H](COC(=O)CCCCCCCCCCCCCCC)COC(=O)CCCCCCC/C=C\\CCCCCCCC",
     "(2S)-propane-1,2,3-triyl 2-acetate 1-hexadecanoate 3-[(9Z)-octadec-9-enoate]",
     "(2S)-2-[acetyloxy]-1-[hexadecanoyloxy]-3-[(9Z)-octadec-9-enoyloxy]propane"),              #:55113
    ("CC(=O)OCC(COC(=O)C)OC(=O)CC", "propane-1,2,3-triyl 1,3-diacetate 2-propanoate",
     "1,3-bis[acetyloxy]-2-[propanoyloxy]propane"),
    ("CC(=O)OCC(COC(C)=O)OC(=O)C(Cl)C", "propane-1,2,3-triyl 1,3-diacetate 2-(2-chloropropanoate)",
     "1,3-bis[acetyloxy]-2-[2-chloropropanoyloxy]propane"),
    # an anion that starts with a locant is enclosed after the attachment locant
    ("CCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCC)OC(=O)CCCO",
     "propane-1,2,3-triyl 1,3-didodecanoate 2-(4-hydroxybutanoate)",
     "1,3-bis[dodecanoyloxy]-2-[4-hydroxybutanoyloxy]propane"),
    ("CCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCC)OC(=O)CCCOC=O",
     "propane-1,2,3-triyl 1,3-didodecanoate 2-[4-(formyloxy)butanoate]",
     "1,3-bis[dodecanoyloxy]-2-[4-(formyloxy)butanoyloxy]propane"),
]
# a half ester of a polybasic acid: (smiles, the name it used to get, the substitutive PIN, (1))
HALF_ESTERS = [
    ("CC(=O)OCC(COC(=O)C)OC(=O)CCC(=O)O", "propane-1,2,3-triyl 1,3-diacetate 2-butanedioate",
     "4-{[1,3-bis(acetyloxy)propan-2-yl]oxy}-4-oxobutanoic acid"),
    ("CC(=O)OCC(COC(=O)C)OC(=O)CCCC(=O)O", "propane-1,2,3-triyl 1,3-diacetate 2-pentanedioate",
     "5-{[1,3-bis(acetyloxy)propan-2-yl]oxy}-5-oxopentanoic acid"),
    ("CCCCCCCC(=O)OCC(COC(=O)CCCCCCC)OC(=O)CCCCCCCCC(=O)O",
     "propane-1,2,3-triyl 1,3-dioctanoate 2-sebacate",
     "10-{[1,3-bis(octanoyloxy)propan-2-yl]oxy}-10-oxodecanoic acid"),
]


@pytest.mark.parametrize("smiles,name,twin", BB_ROWS)
def test_the_twin_is_the_acyloxy_spelling_of_the_name(smiles, name, twin):
    assert polyester_twin(name) == twin
    assert twin_of(name) == (twin, ("polyester_acyloxy",))


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name,twin", BB_ROWS)
def test_opsin_reads_the_twin_to_the_input(smiles, name, twin):
    """The check the engine does not make: a fresh OPSIN parse of the twin, compared on the canonical
    SMILES with the input (the stereo of C-2 and of the double bond included)."""
    parsed = _independent_parse(twin, smiles)
    assert parsed is not None and _canon(parsed) == _canon(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name,twin", BB_ROWS)
def test_the_default_tier_gives_the_blue_book_name_through_its_twin(smiles, name, twin):
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert row["gate_outcome"] == "twin_verified:polyester_acyloxy", row
    assert row["verified"] == "opsin_twin" and row["opsin"] == "verified_twin", row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name,twin", BB_ROWS)
def test_best_effort_names_every_row_the_default_tier_names(smiles, name, twin):
    row = name_best_effort(smiles)
    assert row["name"] and row["tier"] != "abstain", row
    assert_full_rt(row["name"], smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name,twin", BB_ROWS[:2] + BB_ROWS[3:4])
def test_a_word_of_another_acid_is_not_read_back(smiles, name, twin):
    """A name that cites another anion is read to another molecule: the twin does not ship it."""
    assert twin_round_trip(name, smiles) == "polyester_acyloxy"
    wrong = name.replace("propanoate", "butanoate").replace("hexadecanoate", "octadecanoate")
    assert wrong != name and polyester_twin(wrong) != wrong
    assert twin_round_trip(wrong, smiles) is None


@pytest.mark.opsin_gate
def test_a_locant_that_moves_an_acid_is_not_read_back():
    smiles = "CC(=O)OCC(OC(=O)C)COC(=O)CC"        # acetate at 1 and 2, propanoate at 3
    assert twin_round_trip("propane-1,2,3-triyl 1,2-diacetate 3-propanoate", smiles) == "polyester_acyloxy"
    assert twin_round_trip("propane-1,2,3-triyl 1,3-diacetate 2-propanoate", smiles) is None


@pytest.mark.parametrize("smiles,old,pin", HALF_ESTERS)
def test_a_half_ester_gets_no_functional_class_name(smiles, old, pin):
    """The dianion word of a half ester ('2-pentanedioate') states no acid hydrogen: the glyceride
    assembler no longer builds the functional-class form (1),:31937)."""
    from orthonym.rules.lipids import name_lipid
    assert name_lipid(Chem.MolFromSmiles(smiles)) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,old,pin", HALF_ESTERS)
def test_a_half_ester_is_named_as_the_acid(smiles, old, pin):
    """The substitutive PIN of the acid, at both tiers, read back by OPSIN."""
    row = default_tier_row(smiles)
    assert row["name"] == pin and row["tier"] == "pin_verified" and row["verified"] == "opsin", row
    assert name_best_effort(smiles)["name"] == pin
    assert_full_rt(pin, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,old,pin", HALF_ESTERS)
def test_the_old_half_ester_name_has_no_acyloxy_twin_opsin_reads(smiles, old, pin):
    """The twin of the old name ('2-[butanedioyloxy]') has a free valence, which OPSIN does not read."""
    twin = polyester_twin(old)
    assert twin != old
    assert _independent_parse(twin, smiles) is None
    assert twin_round_trip(old, smiles) is None


@pytest.mark.opsin_gate
def test_a_free_carboxy_in_the_acyl_chain_is_a_half_ester_too():
    """[1-(carboxymethyl)cyclopropyl]acetate states its free carboxy as a prefix, but the acid is the senior
    class (1)): no functional-class polyester name at the default tier, the acid at best-effort."""
    smiles = "CCCCCCCC(=O)OCC(COC(=O)CCCCCCC)OC(=O)CC1(CC(=O)O)CC1"
    row = default_tier_row(smiles)
    assert not (row["name"] or "").startswith("propane-1,2,3-triyl"), row
    be = name_best_effort(smiles)
    assert be["name"].endswith("acetic acid"), be
    assert_full_rt(be["name"], smiles)


@pytest.mark.parametrize("name", [
    "ethanol", "propane-1,2,3-triyl triacetate", "propane-1,2,3-triyl 1,3-diacetate",
    "propane-1,2,3-triyl 1,3-diacetate 2-propanoic acid", "propane-1,2,3-triyl 1,2-diacetate 2-propanoate",
    "butane-1,2-diyl 1-acetate 2-propanoate 3-butanoate"])
def test_no_twin_for_a_name_that_is_not_a_polyester_of_one_polyol(name):
    """One anion word (OPSIN reads it), a word that is no '-ate' anion, locants that do not make up the
    parent's, a word that cites a locant twice: the name is returned as it stands."""
    assert polyester_twin(name) == name


@pytest.mark.parametrize("name,twin", [
    ("propane-1,2,3-triyl 1,3-bis(chloroacetate) 2-propanoate", "1,3-bis[chloroacetyloxy]-2-[propanoyloxy]propane"),
    ("propane-1,2,3-triyl 2-benzoate 1,3-dicyclohexanecarboxylate",
     "2-[benzoyloxy]-1,3-bis[cyclohexanecarbonyloxy]propane"),
    ("propane-1,2,3-triyl 2-carbonochloridate 1,3-dihexadecanoate",
     "2-[carbonochloridoyloxy]-1,3-bis[hexadecanoyloxy]propane"),
    ("propane-1,2,3-triyl 2-acetate 1,3-dicarbamate", "2-[acetyloxy]-1,3-bis[carbamoyloxy]propane"),
    ("butane-1,2,3,4-tetrayl 1,4-diacetate 2,3-dipropanoate",
     "1,4-bis[acetyloxy]-2,3-bis[propanoyloxy]butane"),
])
def test_the_twin_words(name, twin):
    assert polyester_twin(name) == twin


def test_the_twin_is_pure_and_not_rewritten_again():
    for _, name, twin in BB_ROWS:
        assert polyester_twin(name) == polyester_twin(name)
        assert polyester_twin(twin) == twin


# ---------------------------------------------------------------------------------------------------------
# a SUBSTITUTED anion takes bis/tris,:31819); an unsubstituted one di/tri
# ---------------------------------------------------------------------------------------------------------
SUBSTITUTED_ROWS = [
    ("ClCC(=O)OCC(OC(=O)C)COC(=O)CCl", "propane-1,2,3-triyl 2-acetate 1,3-bis(chloroacetate)",
     "2-[acetyloxy]-1,3-bis[chloroacetyloxy]propane"),
    ("CNC(=O)OCC(OC(=O)C)COC(=O)NC", "propane-1,2,3-triyl 2-acetate 1,3-bis(methylcarbamate)",
     "2-[acetyloxy]-1,3-bis[methylcarbamoyloxy]propane"),
    ("OCC(=O)OCC(OC(=O)C)COC(=O)CO", "propane-1,2,3-triyl 2-acetate 1,3-bis(hydroxyacetate)",
     "2-[acetyloxy]-1,3-bis[hydroxyacetyloxy]propane"),
    ("O=C(Nc1ccccc1)OCC(OC(=O)c1ccccc1)COC(=O)Nc1ccccc1",
     "propane-1,2,3-triyl 2-benzoate 1,3-bis(phenylcarbamate)",
     "2-[benzoyloxy]-1,3-bis[phenylcarbamoyloxy]propane"),
    ("CC(=O)OCC(OC(=O)C)COC(=O)C(C)C", "propane-1,2,3-triyl 1,2-diacetate 3-(2-methylpropanoate)",
     "1,2-bis[acetyloxy]-3-[2-methylpropanoyloxy]propane"),
]
# two anions of a substituted acid that is not the look-alike dichloroacetate
LOOK_ALIKE = ("ClC(Cl)C(=O)OCC(OC(=O)C)COC(=O)C(Cl)Cl", "propane-1,2,3-triyl 2-acetate 1,3-bis(dichloroacetate)",
              "2-[acetyloxy]-1,3-bis[dichloroacetyloxy]propane")
# unsubstituted anions keep di/tri
UNSUBSTITUTED_ROWS = [
    ("NC(=O)OCC(OC(=O)C)COC(N)=O", "propane-1,2,3-triyl 2-acetate 1,3-dicarbamate"),
    ("CC(=O)OCC(COC(=O)C)OC(=O)CC", "propane-1,2,3-triyl 1,3-diacetate 2-propanoate"),
    ("CCCCCCCCCCCCCCCCCC(=O)OCC(OC(=O)C)COC(=O)CCCCCCCCCCCCCCCCC",
     "propane-1,2,3-triyl 2-acetate 1,3-dioctadecanoate"),
]


@pytest.mark.parametrize("smiles,name,twin", SUBSTITUTED_ROWS + [LOOK_ALIKE])
def test_a_substituted_anion_takes_bis_with_enclosing_marks(smiles, name, twin):
    from orthonym.rules.lipids import name_lipid
    assert name_lipid(Chem.MolFromSmiles(smiles)) == name
    assert polyester_twin(name) == twin


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name,twin", SUBSTITUTED_ROWS + [LOOK_ALIKE])
def test_the_twin_reads_the_bis_spelling_to_the_full_key(smiles, name, twin):
    parsed = _independent_parse(twin, smiles)
    assert parsed is not None and _canon(parsed) == _canon(smiles)
    assert twin_round_trip(name, smiles) == "polyester_acyloxy"
    row = default_tier_row(smiles)
    assert row["name"] == name and row["gate_outcome"] == "twin_verified:polyester_acyloxy", row


@pytest.mark.opsin_gate
def test_chloroacetate_and_dichloroacetate_are_two_names():
    """'1,3-dichloroacetate' was the name of both (two ClCH2COO-, one Cl2CHCOO- cited twice)."""
    one, two = SUBSTITUTED_ROWS[0], LOOK_ALIKE
    assert default_tier_row(one[0])["name"] == one[1]
    assert default_tier_row(two[0])["name"] == two[1]
    assert one[1] != two[1] and "1,3-dichloroacetate" not in (one[1], two[1])


@pytest.mark.parametrize("smiles,name", UNSUBSTITUTED_ROWS)
def test_an_unsubstituted_anion_keeps_di_and_tri(smiles, name):
    from orthonym.rules.lipids import name_lipid
    assert name_lipid(Chem.MolFromSmiles(smiles)) == name


# ---------------------------------------------------------------------------------------------------------
# a half ester of an amino acid (a free COOH) is no polyester name: 'glutamate' states the dianion
# ---------------------------------------------------------------------------------------------------------
AMINO_ACID_HALF_ESTERS = [
    ("CC(=O)OCC(COC(=O)C)OC(=O)[C@@H](N)CCC(=O)O", "propane-1,2,3-triyl 1,3-diacetate 2-glutamate"),
    ("CC(=O)OCC(COC(=O)C)OC(=O)[C@@H](N)CC(=O)O", "propane-1,2,3-triyl 1,3-diacetate 2-aspartate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", AMINO_ACID_HALF_ESTERS)
def test_a_retained_dianion_word_of_a_half_ester_is_not_read_back(smiles, name):
    """OPSIN reads the twin ('glutamyloxy') to the structure of the input, a monovalent acyl with a free COOH:
    the read-back alone would verify the word. The input holds a free carboxylic acid, so its residue is that
    of a polybasic acid whose PIN is the acid (1),:31937,:31941)."""
    twin = polyester_twin(name)
    assert twin != name
    parsed = _independent_parse(twin, smiles)
    assert parsed is not None and _canon(parsed) == _canon(smiles)      # the structure is the same...
    assert twin_round_trip(name, smiles) is None                         #... and the name is not shipped


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", AMINO_ACID_HALF_ESTERS)
def test_a_half_ester_of_an_amino_acid_is_never_named_as_a_polyester(smiles, name):
    for row in (default_tier_row(smiles), name_best_effort(smiles)):
        assert not (row["name"] or "").startswith("propane-1,2,3-triyl"), row


@pytest.mark.opsin_gate
def test_a_polyester_with_no_free_acid_is_still_read_back():
    smiles = "CC(=O)OCC(COC(=O)C)OC(=O)CC"
    assert twin_round_trip("propane-1,2,3-triyl 1,3-diacetate 2-propanoate", smiles) == "polyester_acyloxy"
