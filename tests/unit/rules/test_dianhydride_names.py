"""Di- and polyanhydride names: every acid residue is named as the acid it is, the multiplier
follows, and a name ships only when something reads it back (``validation.anhydride_acids``).

The ``dianhydride`` and ``chalcogen_dianhydride`` carve-outs shipped every name of the class with no read-back
(OPSIN 2.9.0 has no grammar for the class word), and 117 of the 210 names of the audited PubChem rows were the
name of another molecule: ``_diacid_stem`` accepted any O atom (an OH, an oxo group, an ether or ester O of the
residue passed as a carbonyl O) and ``_name_acyl_acid`` chain-named a residue with a ring, a double bond or a
stereo centre. A name that never had the Blue Book spelling went with it: ``di2-methylbenzoic`` for the
multiplied acid of a substituted name.

* (the Blue Book): "Dianhydrides are named by citing the acid groups in their order of
  occurrence in the structure beginning with the end acid group lower in alphabetical order followed by the
  class term 'dianhydride'. The numerical prefix 'di-' is used to generate preferred IUPAC names."; 'diacetic
  butanedioic dianhydride (PIN)' (:32386), 'butanoic butanedioic propanoic dianhydride (PIN)' (:32392).
* (:32394): "Linear polyanhydrides are named by one of the following methods: (1) by selecting the
  preferred dicarboxylic acid and citing the adjoining acid groups one of which will be substituted...; (2) by
  citing the acid groups, in their order of appearance... followed by a class term 'dianhydride',
  'trianhydride', etc.... Method (1) generates preferred IUPAC names.": the 'trianhydride' of method (2) is no
  PIN, and the default tier declines it.
* (:32443): 'diacetic butanedioic bis(thioanhydride) (PIN)' (:32446).
* Symmetric anhydrides (:32225-32249): 'bis(chloroacetic) anhydride (PIN)'; (:32565): "if
  symmetrically substituted, are named by prefixing 'bis' to the name of the acid".
* General methodology (:7029): "(a) (:7033) determine whether the component to be multiplied is simple or
  compound. Simple components are unsubstituted parent hydrides...; functionalized parent hydrides, such as
  benzenesulfonic acid; or retained names, such as acetic acid. All of these are multiplied by the
  multiplicative prefixes 'di', 'tri', etc.;... (c) (:7035) any component which is substituted automatically requires
  use of the multiplicative forms 'bis', 'tris', etc."
* (:29723): 'oxalic acid (PIN)'; 2,3-dihydroxybutanedioic acid (PIN) (:29834).
* (:36925): 'tetraacetic hypodiphosphoric tetraanhydride (PIN)' (:36933).
"""
import re

import pytest
from rdkit import Chem

from orthonym.errors import is_failure_name
from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import _independent_parse, assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

TWIN = "acid_split_verified:anhydride_acids"

# (smiles, the name the Blue Book rules give)
RIGHT_PIN = [
    ("CC(=O)OC(=O)CCC(=O)OC(=O)C", "diacetic butanedioic dianhydride"),                       #:32386
    ("CCCC(=O)OC(=O)CCC(=O)OC(=O)CC", "butanoic butanedioic propanoic dianhydride"),         #:32392
    ("CC(=O)SC(=O)CCC(=O)SC(C)=O", "diacetic butanedioic bis(thioanhydride)"),               #:32446
    ("CC(=O)OP(=O)(OC(=O)C)P(=O)(OC(=O)C)OC(=O)C", "tetraacetic hypodiphosphoric tetraanhydride"),  #:36933
    ("CC(=O)OP(=O)(OC(=O)C)OP(=O)(OC(=O)C)OC(=O)C", "tetraacetic diphosphoric tetraanhydride"),
    ("CC(=O)OS(=O)(=O)OS(=O)(=O)OC(=O)C", "diacetic disulfuric dianhydride"),
    ("CC(=O)OC(=O)CCCCCCCCCCC(=O)OC(=O)C", "diacetic dodecanedioic dianhydride"),
    ("CCCCCCCC(=O)OC(=O)CCCCC(=O)OC(=O)C1=CC=CC=C1", "benzoic hexanedioic octanoic dianhydride"),
    ("CC(=O)OC(=O)CC(=O)OC(=O)C", "diacetic propanedioic dianhydride"),
]
# rows whose name was the name of another molecule or a spelling the rules do not give, now named as they are
RENAMED = [
    # the two OH groups of the central acid were lost: 'diacetic butanedioic dianhydride'
    ("CC(=O)OC(=O)C(O)C(O)C(=O)OC(C)=O", "diacetic 2,3-dihydroxybutanedioic dianhydride"),
    # a substituted acid is multiplied with 'bis' (c)), not 'di2-'
    ("Cc1ccccc1C(=O)OC(=O)CCC(=O)OC(=O)c1ccccc1C", "bis(2-methylbenzoic) butanedioic dianhydride"),
    ("ClCC(=O)OC(=O)CCC(=O)OC(=O)CCl", "bis(chloroacetic) butanedioic dianhydride"),
    # the retained name of oxalic acid is the PIN (:29723)
    ("CC(=O)OC(=O)C(=O)OC(=O)C", "diacetic oxalic dianhydride"),
    # a polyacid anhydride: the substituted acid is multiplied with 'bis', not 'di2,3-'
    ("OCC(O)C(=O)OS(=O)(=O)OS(=O)(=O)OC(=O)C(O)CO", "bis(2,3-dihydroxypropanoic) disulfuric dianhydride"),
]
# the words that state a configuration, a name of the retained table that is no PIN, enclosing marks and the label
# of a long chain: each was lost, wrong or lowered on the way to the name
FURTHER_PINS = [
    #: the acid word states its own descriptors; the composer adds no front block of the whole name
    ("N[C@@H](CCC(=O)OC=O)C(=O)OC=O", "diformic (2S)-2-aminopentanedioic dianhydride"),
    ("C[C@@H](C(=O)OC(=O)CCCCC(=O)OC(=O)[C@H](C)N)N", "bis[(2S)-2-aminopropanoic] hexanedioic dianhydride"),
    # (:29715): only five retained acid names are PINs, 'suberic' and 'sebacic' (the retained-name
    # table) are not: the acid is octanedioic acid
    ("CC(C)CC(=O)OC(=O)CCCCCCC(=O)OC(=O)CC(C)C", "bis(3-methylbutanoic) octanedioic dianhydride"),
    ("CC(=O)OC(=O)CCCCCCCCC(=O)OC(=O)C", "diacetic decanedioic dianhydride"),
    # (:7446): the enclosing marks of a multiplied acid nest:   { }
    ("CCCCCCCCCOC(=O)CCCCCCCCC(=O)OC(=O)CCCCCCCCC(=O)OC(=O)CCCCCCCCC(=O)OCCCCCCCCC",
     "bis[10-(nonyloxy)-10-oxodecanoic] decanedioic dianhydride"),
    # (c): 'hexadecanoic' begins with a multiplying syllable ('hexa')
    ("CCCCCCCCCCCCCCCC(=O)SC(=O)CCCCCCC(=O)SC(=O)CCCCCCCCCCCCCCC", "di(hexadecanoic) octanedioic bis(thioanhydride)"),
]

# (smiles, the name of another molecule the carve-out shipped): no tier may give it, and a name that ships is
# read back to the input by the independent check below
WRONG_BEFORE = [
    ("CC(=O)OC(=O)C(C(C(=O)OC(=O)C)O)O", "diacetic butanedioic dianhydride"),                 # OH x2 lost
    ("C=CC(=O)OC(=O)C=CC(=O)OC(=O)C=C", "diprop-2-enoic butanedioic dianhydride"),            # C=C lost
    ("CC(=O)OC(=O)C(=O)C(=O)OC(=O)C", "diacetic propanedioic dianhydride"),                  # oxo lost
    ("CC(=O)OC(=O)CC(CC(=O)OC(=O)C)O", "diacetic pentanedioic dianhydride"),                 # OH lost
    ("CC(=O)OC(=O)CCOCCOCCC(=O)OC(=O)C", "diacetic octanedioic dianhydride"),                # ether O counted as C=O
    ("C1=CC=C(C=C1)C(=O)OC(=O)C(=O)C(=O)OC(=O)C2=CC=CC=C2", "dibenzoic propanedioic dianhydride"),
    ("C1CC1C(=O)OC(=O)C(C(C(=O)OC(=O)C2CC2)O)O", "dicyclopropanecarboxylic butanedioic dianhydride"),
    ("C1=CC=C(C=C1)/C=C/C(=O)OC(=O)C(=O)OC(=O)/C=C/C2=CC=CC=C2", "dinonanoic ethanedioic dianhydride"),
    ("C[C@@H](C(=O)OC(=O)CCCCC(=O)OC(=O)[C@H](C)N)N", "di2-aminopropanoic hexanedioic dianhydride"),
    ("CC(C)(C)C(=O)OC(=O)C=CC(=O)OC(=O)C(C)(C)C", "di2,2-dimethylpropanoic butanedioic dianhydride"),
    ("C=C(CO)C(=O)OC(=O)C=CC(=O)OC(=O)C(=C)CO", "di2-(hydroxymethyl)propanoic butanedioic dianhydride"),
    ("CC1=CC=CC=C1C(=O)OC(=O)C(C(C(=O)OC(=O)C2=CC=CC=C2C)O)O", "di2-methylbenzoic butanedioic dianhydride"),
    ("CCCCCCCCCCCCCCCC(=O)OC(=O)CC(C(=O)OC(=O)CCCCCCCCCCCCCCC)O", "dihexadecanoic butanedioic dianhydride"),
    # a chalcogen dianhydride whose central acid holds an ester O
    ("CC(=O)SC(=O)COC(=O)CCC(=O)OCC(=O)SC(=O)C", "diacetic octanedioic bis(thioanhydride)"),
]
# method (2) of (three links or more) is no PIN
NO_PIN_POLYANHYDRIDE = [
    ("CC(=O)OC(=O)CCC(=O)OC(=O)CCC(=O)OC(=O)C", "diacetic dibutanedioic trianhydride"),
]

# a decorated variant of a Blue Book name (a Cl on the central acid): it never reproduces the bare name (the
# decoration-mutation certificate of the carve-out table, tests/unit/validation/test_pin_carveout_atom_coverage.py,
# for the two classes that left it)
MUTANTS = [
    ("CC(=O)OC(=O)CCC(=O)OC(C)=O", "diacetic butanedioic dianhydride", "CC(=O)OC(=O)C(Cl)CC(=O)OC(C)=O"),
    ("CC(=O)SC(=O)CCC(=O)SC(C)=O", "diacetic butanedioic bis(thioanhydride)", "CC(=O)SC(=O)C(Cl)CC(=O)SC(C)=O"),
]

_NUMBER = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6}
_COMPLEX = {"bis": 2, "tris": 3, "tetrakis": 4, "pentakis": 5, "hexakis": 6}


def _key(smiles_or_mol):
    mol = Chem.MolFromSmiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    return Chem.MolToInchiKey(mol)


def _split_into_acids(smiles):
    """The input split at each anhydride link (C(=O)-X-C(=O), X an O, S, Se or Te atom; an acyl-O-oxoacid link),
    the bridge atom kept as the OH of one side and a new OH on the other: the keys of the acids, sorted."""
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts("[#6;$([#6]=O)]-[O,S,Se,Te;X2]-[#6,#15,#16;$(*=O)]")
    bridges = sorted({m[1] for m in mol.GetSubstructMatches(pattern)})
    rw = Chem.RWMol(mol)
    for bridge in bridges:
        atom = rw.GetAtomWithIdx(bridge)
        first, second = [n.GetIdx() for n in atom.GetNeighbors()]
        rw.RemoveBond(bridge, second)
        atom.SetAtomicNum(8)
        atom.SetNoImplicit(False)
        rw.AddBond(second, rw.AddAtom(Chem.Atom(8)), Chem.BondType.SINGLE)
    pieces = rw.GetMol()
    Chem.SanitizeMol(pieces)
    return len(bridges), sorted(_key(p) for p in Chem.GetMolFrags(pieces, asMols=True))


def _acids_of_name(name):
    """The acids a poly-anhydride name cites (each word, multiplied as its spelling says, read by a fresh OPSIN
    call as '<word> acid'), as the keys, sorted; None when a word is no acid."""
    keys = []
    for word in name.split(" ")[:-1]:
        count, stem = 1, word
        found = re.fullmatch(r"(bis|tris|tetrakis|pentakis|hexakis)[(\[{](.+)[)\]}]", word)
        if found:
            count, stem = _COMPLEX[found.group(1)], found.group(2)
        else:
            found = re.fullmatch(r"(di|tri|tetra|penta|hexa)[(\[{](.+)[)\]}]", word)
            if found:
                count, stem = _NUMBER[found.group(1)], found.group(2)
            elif _independent_parse(f"{word} acid") is None:
                found = re.fullmatch(r"(di|tri|tetra|penta|hexa)(.+)", word)
                if found:
                    count, stem = _NUMBER[found.group(1)], found.group(2)
        smiles = _independent_parse(f"{stem} acid")
        if smiles is None or Chem.MolFromSmiles(smiles) is None:
            return None
        keys += [_key(smiles)] * count
    return sorted(keys)


def _links_of_class_word(name):
    word = name.split(" ")[-1]
    found = re.fullmatch(r"(di|tri|tetra|penta|hexa)anhydride", word)
    if found:
        return _NUMBER[found.group(1)]
    found = re.fullmatch(r"(bis|tris|tetrakis|pentakis|hexakis)\((?:thio|seleno|telluro)anhydride\)", word)
    return _COMPLEX[found.group(1)] if found else None


def _independent_split_check(name, smiles):
    """The check the engine does not make: the acids of the input (split at its links) are the acids of the
    name, and the number of links is the class word's. The bridge is not compared, nor the order."""
    links, acids = _split_into_acids(smiles)
    return links == _links_of_class_word(name) and acids == _acids_of_name(name)


@pytest.mark.parametrize("smiles,name", RIGHT_PIN + RENAMED + FURTHER_PINS)
def test_default_tier_gives_the_pin_and_reads_it_back_through_its_acids(smiles, name):
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert row["verified"] == "opsin_acids" and row["opsin"] == "verified_acid_split", row
    assert row["gate_outcome"] == TWIN, row
    assert row["spelling_failures"] == [], row


@pytest.mark.parametrize("smiles,name", RIGHT_PIN + RENAMED + FURTHER_PINS)
def test_the_independent_split_check_reads_the_name_to_the_input(smiles, name):
    assert _independent_split_check(name, smiles)


@pytest.mark.parametrize("smiles,name", RIGHT_PIN + RENAMED + FURTHER_PINS)
def test_best_effort_gives_a_name_for_every_row_the_default_tier_names(smiles, name):
    """The wider tier ships the names OPSIN reads back (the carve-out names are withdrawn there): the row is
    named, by a name that reads back to the full input."""
    row = name_best_effort(smiles)
    assert row["name"] and not is_failure_name(row["name"]), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,wrong", WRONG_BEFORE)
def test_the_wrong_name_is_refuted_by_the_split_check(smiles, wrong):
    assert not _independent_split_check(wrong, smiles)


@pytest.mark.parametrize("smiles,wrong", WRONG_BEFORE)
def test_no_tier_ships_the_wrong_name(smiles, wrong):
    """A name that ships is the name of the input: the independent split check holds for a poly-anhydride
    name, and a name OPSIN reads reads back to the full input. A decline is a label, never another molecule."""
    default = default_tier_row(smiles)
    assert default["name"] != wrong, default
    if not is_failure_name(default["name"]):
        if _links_of_class_word(default["name"]) is not None:
            assert _independent_split_check(default["name"], smiles), default
        else:
            assert_full_rt(default["name"], smiles)
    best = name_best_effort(smiles)
    assert best["name"] != wrong, best
    if not is_failure_name(best["name"]):
        assert_full_rt(best["name"], smiles)


@pytest.mark.parametrize("smiles,wrong", WRONG_BEFORE)
def test_a_multiplied_acid_is_never_cited_with_di_before_a_locant(smiles, wrong):
    """ (c): a substituted acid word takes 'bis', never 'di2-' (the locant of the acid after the 'di')."""
    name = default_tier_row(smiles)["name"] or ""
    assert not re.search(r"(?:^| )(?:di|tri|tetra|penta|hexa)\d", name), name


@pytest.mark.parametrize("smiles,name", NO_PIN_POLYANHYDRIDE)
def test_method_two_of_p65762_is_no_pin(smiles, name):
    """: 'Method (1) generates preferred IUPAC names'; the 'trianhydride' name of method (2) is a
    correct name that is not the PIN, so the default tier gives it no name (the substitutive method (1) name is
    not built here) and the best-effort tier gives a name that reads back."""
    row = default_tier_row(smiles)
    assert row["tier"] == "abstain" and row["name"] != name, row
    best = name_best_effort(smiles)
    assert best["name"] and not is_failure_name(best["name"]) and best["name"] != name, best
    assert_full_rt(best["name"], smiles)


# two links whose central acid has two different ends and no locants: the acids do not say which way round it lies,
# so no dianhydride name is offered and the substitutive spelling of the anhydride (read by OPSIN) names it
UNDETERMINED = [
    ("CCCC(=O)OC(=O)C(C)CC(=O)OC(=O)CC", "butanoic 2-methyl-4-oxo-4-(propanoyloxy)butanoic anhydride"),
]


@pytest.mark.parametrize("smiles,name", UNDETERMINED)
def test_a_name_the_acids_do_not_determine_is_not_offered(smiles, name):
    for row in (default_tier_row(smiles), name_best_effort(smiles)):
        assert row["name"] == name, row
        assert row["verified"] == "opsin", row
    assert _independent_parse(name, smiles) is not None


def test_a_name_the_acids_do_not_read_is_declined(monkeypatch):
    """No poly-anhydride name ships unless the check reads it: a producer that gives the old wrong name for the
    tartaric dianhydride is declined at the default tier."""
    import orthonym.rules.anhydrides as anhydrides
    smiles = "CC(=O)OC(=O)C(O)C(O)C(=O)OC(C)=O"
    monkeypatch.setattr(anhydrides, "_name_dianhydride", lambda mol, matches: "diacetic butanedioic dianhydride")
    row = default_tier_row(smiles)
    assert row["name"] != "diacetic butanedioic dianhydride", row
    assert name_best_effort(smiles)["name"] != "diacetic butanedioic dianhydride"


@pytest.mark.parametrize("bare_smiles,bare_name,smiles", MUTANTS)
def test_a_decorated_variant_never_gets_the_bare_name(bare_smiles, bare_name, smiles):
    """The recogniser fired on a bigger molecule and named only its core: the check that reads the acids sees
    the Cl, so the bare name is not verified for the decorated structure and the name that ships is."""
    assert _independent_split_check(bare_name, bare_smiles)
    assert not _independent_split_check(bare_name, smiles)
    for row in (default_tier_row(smiles), name_best_effort(smiles)):
        assert row["name"] != bare_name, row
        if not is_failure_name(row["name"]) and _links_of_class_word(row["name"]) is not None:
            assert _independent_split_check(row["name"], smiles), row
        elif not is_failure_name(row["name"]):
            assert_full_rt(row["name"], smiles)


# ---------------------------------------------------------------------------------------------------------------
# CO lane A review a performance pass
# ---------------------------------------------------------------------------------------------------------------
import random

from orthonym.perception.stereo import assign_stereochemistry

# MF1: the descriptor of the free acid is not the descriptor of the molecule. The alpha carbon of the third acid has
# two carboxyl-type groups; in the free acid its COOH ranks below the ester COOMe, in the molecule its anhydride
# C(=O)O-C(=O) ranks above it: (2S) in the acid word, R in the molecule. (the Blue Book): "In preferred
# IUPAC names, stereodescriptors are placed immediately at the front of the part of the name to which they relate".
FREE_LABEL_IS_NOT_THE_MOLECULES = (
    "CC(=O)OC(=O)CCC(=O)OC(=O)[C@H](C)C(=O)OC",
    "acetic butanedioic (2S)-3-methoxy-2-methyl-3-oxopropanoic dianhydride",
)
# rows whose acid words state a configuration the molecule has
STEREO_ROWS = [
    "C[C@H](O)C(=O)OC(=O)CCC(=O)OC(=O)[C@H](C)O",
    "C[C@@H](O)C(=O)OC(=O)CCC(=O)OC(=O)[C@H](C)O",
    "C[C@@H](C(=O)OC(=O)CCCCC(=O)OC(=O)[C@H](C)N)N",
    "C/C=C/C(=O)OC(=O)CCC(=O)OC(=O)/C=C/C",
    "C/C=C/C(=O)OC(=O)CCC(=O)OC(=O)/C=C\\C",
    "CC(=O)OC(=O)[C@H](O)[C@@H](O)C(=O)OC(C)=O",
    "N[C@@H](CCC(=O)OC=O)C(=O)OC=O",
]
# SF (a): a linked carbonyl cited through a prefix ('carboxymethoxy') is no suffix group: (:32433)
# "Polyanhydrides consisting of a polybasic acid residue and the equivalent number of monobasic residues are named by
# citing the monobasic acid groups... followed by the polybasic acid residue"
PREFIX_CITED_CARBONYL = [
    ("CC(=O)OC(=O)CC(=O)OCC(=O)OC(C)=O", "diacetic 3-(carboxymethoxy)-3-oxopropanoic dianhydride"),
    ("CC(=O)OC(=O)CCC(=O)NCC(=O)OC(C)=O", "diacetic 4-[(carboxymethyl)amino]-4-oxobutanoic dianhydride"),
]
# MF2: isotopically labelled dianhydrides (the best-effort tier named them on main through the general engine)
ISOTOPE_ROWS = [
    ("[2H]C([2H])([2H])C(=O)OC(=O)CCC(=O)OC(=O)C([2H])([2H])[2H]", "1,4-bis[(2H3)acetyloxy]-1,4-dioxobutane"),
    ("[2H]C([2H])([2H])C(=O)OC(=O)CCCCC(=O)OC(=O)C([2H])([2H])[2H]", "1,6-bis[(2H3)acetyloxy]-1,6-dioxohexane"),
    ("[2H]C([2H])([2H])C(=O)SC(=O)CCC(=O)SC(=O)C([2H])([2H])[2H]", "1,4-bis[(2H3)acetylsulfanyl]-1,4-dioxobutane"),
    ("[2H]C([2H])([2H])C(=O)OC(=O)C(=O)OC(=O)C([2H])([2H])[2H]", "1,2-bis[(2H3)acetyloxy]-1,2-dioxoethane"),
]


def _molecule_descriptors(smiles):
    """The CIP letters of the whole molecule, by the project's labelling, sorted."""
    mol = Chem.MolFromSmiles(smiles)
    assign_stereochemistry(mol)
    letters = [a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")]
    letters += [b.GetProp("_CIPCode") for b in mol.GetBonds() if b.HasProp("_CIPCode")]
    return sorted(letters)


def _name_descriptors(name):
    blocks = re.findall(r"\((\d+[a-z']*[RSEZrsez](?:,\d+[a-z']*[RSEZrsez])*)\)", name)
    return sorted(letter for block in blocks for letter in re.findall(r"[RSEZrsez]", block))


def test_a_descriptor_the_link_changes_is_not_shipped():
    smiles, wrong = FREE_LABEL_IS_NOT_THE_MOLECULES
    assert _molecule_descriptors(smiles) == ["R"]            # the molecule is R, the acid word says S
    from orthonym.validation.anhydride_acids import anhydride_acids_round_trip
    assert anhydride_acids_round_trip(wrong, smiles) is None
    assert anhydride_acids_round_trip(wrong.replace("(2S)", "(2R)"), smiles) is None
    for row in (default_tier_row(smiles), name_best_effort(smiles)):
        assert row["name"] != wrong and "dianhydride" not in (row["name"] or ""), row
        if not is_failure_name(row["name"]):
            assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", STEREO_ROWS)
def test_the_descriptors_of_a_shipped_name_are_those_of_the_molecule(smiles):
    row = default_tier_row(smiles)
    if not is_failure_name(row["name"]) and _links_of_class_word(row["name"]) is not None:
        # the letters, not their number: a multiplied word ('bis[(2S)-...]') states its descriptors once
        assert set(_name_descriptors(row["name"])) == set(_molecule_descriptors(smiles)), row
        assert _independent_split_check(row["name"], smiles), row


@pytest.mark.parametrize("smiles,wrong", PREFIX_CITED_CARBONYL)
def test_a_carbonyl_cited_through_a_prefix_gets_no_dianhydride_name(smiles, wrong):
    for row in (default_tier_row(smiles), name_best_effort(smiles)):
        assert row["name"] != wrong and "carboxy" not in (row["name"] if _links_of_class_word(row["name"] or "x")
                                                          else ""), row
        if not is_failure_name(row["name"]):
            assert_full_rt(row["name"], smiles) if _links_of_class_word(row["name"]) is None else None


@pytest.mark.parametrize("smiles,main_name", ISOTOPE_ROWS)
def test_best_effort_names_the_labelled_dianhydride_as_main_did(smiles, main_name):
    """The substitutive 'acetic 4-(acetyloxy)-4-oxobutanoic anhydride' cannot take two labelled acetyl groups (one
    is the class word, the other a prefix), and the best-effort tier lost the row once it was built there: the
    dianhydride name is offered at the wider tiers too and withdrawn by the gate, so the general engine names it."""
    row = name_best_effort(smiles)
    assert row["name"] == main_name, row
    assert_full_rt(main_name, smiles)
    default = default_tier_row(smiles)
    assert is_failure_name(default["name"]) or default["name"] == main_name, default


def test_the_best_effort_name_of_a_dianhydride_is_labelled_a_non_pin_name():
    row = name_best_effort("CC(=O)OC(=O)CCC(=O)OC(=O)C")
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row


def _orderings(smiles, count, seed):
    mol = Chem.MolFromSmiles(smiles)
    rng = random.Random(seed)
    out = {smiles}
    for _ in range(count):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        out.add(Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False))
    return sorted(out)


# MF3: the end acid of a poly-anhydride is the one lower in alphabetical order, and where two acids differ in
# configuration only the one with 'R' (or 'Z') precedes (j), the Blue Book "'R' precedes 'S'"): the name
# is the same whichever way the structure is drawn
DETERMINISM_ROWS = [
    "C[C@@H](O)C(=O)OC(=O)CCC(=O)OC(=O)[C@H](C)O",
    "C/C=C/C(=O)OC(=O)CCC(=O)OC(=O)/C=C\\C",
    "C[C@H](N)C(=O)OC(=O)CCC(=O)OC(=O)[C@@H](C)N",
]


@pytest.mark.parametrize("smiles", DETERMINISM_ROWS)
def test_the_name_does_not_depend_on_the_drawing(smiles):
    names = {default_tier_row(s)["name"] for s in _orderings(smiles, 20, 11)}
    assert len(names) == 1, names
    (name,) = names
    assert "dianhydride" in name and name.index("(2R)") < name.index("(2S)") if "(2R)" in name else True, name


@pytest.mark.xfail(reason="the general engine numbers the 1,4-disubstituted chain by atom order where two groups differ in "
                          "configuration only (P-31.1.4.3.4): the best-effort names of main vary the same way "
                          "('1-{[(2S)-...]}-4-{[(2R)-...]}' and the reverse); not a producer of this lane", strict=False)
@pytest.mark.parametrize("smiles", DETERMINISM_ROWS)
def test_the_best_effort_name_does_not_depend_on_the_drawing(smiles):
    names = {name_best_effort(s)["name"] for s in _orderings(smiles, 8, 11)}
    assert len(names) == 1, names


def test_the_multiplying_syllables_are_the_numerical_terms_of_p1421():
    """ (c) (the Blue Book): a functionalized parent hydride 'beginning with a multiplicative prefix' is
    enclosed when multiplied: 'di(pentanal)'. The terms are those of (:2790-2821), undeca- to docosa- too."""
    from orthonym.rules.anhydrides import _begins_with_multiplier
    for stem in ("pentanoic", "hexanoic", "octanoic", "decanoic", "undecanoic", "dodecanoic", "hexadecanoic",
                 "icosanoic", "henicosanoic", "docosanoic", "tricosanoic", "triacontanoic"):
        assert _begins_with_multiplier(stem), stem
    for stem in ("acetic", "propanoic", "butanoic", "benzoic", "oxalic", "cyclohexanecarboxylic", "formic"):
        assert not _begins_with_multiplier(stem), stem


@pytest.mark.parametrize("carbons,stem", [(11, "undecanoic"), (12, "dodecanoic"), (20, "icosanoic"), (22, "docosanoic")])
def test_a_multiplied_acid_beginning_with_a_multiplying_syllable_is_enclosed(carbons, stem):
    chain = "C" * (carbons - 1)
    smiles = f"{chain}C(=O)OC(=O)CCC(=O)OC(=O){chain}"
    row = default_tier_row(smiles)
    assert row["name"] == f"di({stem}) butanedioic dianhydride", row
    assert _independent_split_check(row["name"], smiles)


def test_the_gate_outcome_and_the_verified_value_name_what_was_done():
    """OPSIN reads the acids, not the name or a spelling of it: its own outcome, label and verified value."""
    row = default_tier_row("CC(=O)OC(=O)CCC(=O)OC(=O)C")
    assert row["gate_outcome"] == "acid_split_verified:anhydride_acids", row
    assert row["opsin"] == "verified_acid_split" and row["verified"] == "opsin_acids", row
    assert "acid_split_round_trip" in row["gates_passed"], row
