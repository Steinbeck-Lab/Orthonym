"""A name with a count word above 50 is read back by the reconstructor, never by OPSIN
(``validation.long_count_read_back``).

OPSIN 2.9.0 has a numerical term up to 50 ('pentaconta') and none above, so ``tetrapentacontafluorohexacosane``
and a cumulene of 55 carbon atoms have no OPSIN reading. The reader takes the facts from the TEXT of the name and
the reconstructor builds the structure from them: nothing here needs a JVM.

* Basic multiplicative prefixes (the Blue Book, Table 1.4) and Derivation of basic
  numerical terms (:2811): the terms are cited "in the order opposite to that of the constituent digits in the
  arabic numbers", joined directly, and "The letter 'i' in 'icosa' is elided after a vowel" (52 dopentaconta,
  111 undecahecta, 363 trihexacontatricta, 486 hexaoctacontatetracta).
* Hydrocarbons (:7984): 'heptacontane (PIN)'.
* Omission of locants (:3007): all locants are omitted when every substitutable position is
  substituted in the same way; "In case of partial substitution or modification, all numerical prefixes must be
  indicated."
* Elision of vowels (:7593) (c): the terminal 'a' of a numerical term is elided before a suffix that
  begins with 'a' or 'o', and so 'tetrapentacontaene' keeps it.
"""
import pytest
from rdkit import Chem

from orthonym.data.chain_names import multiplying_prefix
from orthonym.validation.long_count_read_back import (
    LONG_COUNT_RULE, long_count_round_trip, name_facts_from_text, numerical_term)

# Table 1.4 and the worked examples of (the Blue Book-2822), then the terms the 24 rows cite
BLUE_BOOK_TERMS = {
    11: "undeca", 14: "tetradeca", 20: "icosa", 21: "henicosa", 22: "docosa", 23: "tricosa", 24: "tetracosa",
    30: "triaconta", 40: "tetraconta", 41: "hentetraconta", 50: "pentaconta", 52: "dopentaconta", 60: "hexaconta",
    70: "heptaconta", 80: "octaconta", 90: "nonaconta", 100: "hecta", 101: "henhecta", 111: "undecahecta",
    200: "dicta", 300: "tricta", 363: "trihexacontatricta", 400: "tetracta", 486: "hexaoctacontatetracta",
    500: "pentacta", 900: "nonacta",
    51: "henpentaconta", 53: "tripentaconta", 54: "tetrapentaconta", 55: "pentapentaconta", 61: "henhexaconta",
    77: "heptaheptaconta", 112: "dodecahecta", 113: "tridecahecta", 120: "icosahecta", 124: "tetracosahecta",
}


@pytest.mark.parametrize("number,term", sorted(BLUE_BOOK_TERMS.items()))
def test_the_numerical_term_is_the_blue_books(number, term):
    assert numerical_term(number) == term


def test_the_terms_agree_with_the_chain_name_table_of_the_producer():
    """The reader composes the terms again, from the Blue Book; the producer's table is only compared."""
    assert [numerical_term(n) for n in range(2, 1000)] == [multiplying_prefix(n) for n in range(2, 1000)]


def test_a_term_names_one_number():
    words = [numerical_term(n) for n in range(2, 1000)]
    assert len(set(words)) == len(words)


def _locants(first, last, step=1):
    return ",".join(str(i) for i in range(first, last + 1, step))


def _chain(length, substituents):
    """The SMILES of an unbranched chain with ``substituents`` ({position: [SMILES,...]}) and the locants of
    each kind of substituent, position by position, as the name cites them ({SMILES: '1,1,1,2,2,...'})."""
    smiles, cited = "", {}
    for position in range(1, length + 1):
        smiles += "C" + "".join(f"({s})" for s in substituents.get(position, []))
        for s in substituents.get(position, []):
            cited[s] = cited.get(s, []) + [position]
    return smiles, {s: ",".join(str(p) for p in positions) for s, positions in cited.items()}


def _perfluoro(length, bare=()):
    """A chain of ``length`` carbon atoms with every hydrogen atom fluorine, but those of the positions ``bare``."""
    return {p: ["F"] * (3 if p in (1, length) else 2) for p in range(1, length + 1) if p not in bare}


# name, SMILES: the class of the BR study (acyclic cumulenes, perhalo alkanes, polyynes, a branched cumulene)
C27_SMILES, C27_LOCANTS = _chain(27, {**_perfluoro(27), 27: ["F", "F"]})
C42 = {**_perfluoro(42, bare=range(14, 30)), 15: ["I"], 28: ["I"]}
C42_SMILES, C42_LOCANTS = _chain(42, C42)
REAL = [
    # C53 cumulene: 52 double bonds, 'dopentaconta'; C55: 54, 'tetrapentaconta'; C113: 112, 'dodecahecta'
    ("tripentaconta-" + _locants(1, 52) + "-dopentacontaene", "=".join(["C"] * 53)),
    ("pentapentaconta-" + _locants(1, 54) + "-tetrapentacontaene", "=".join(["C"] * 55)),
    ("tridecahecta-" + _locants(1, 112) + "-dodecahectaene", "=".join(["C"] * 113)),
    # perfluorohexacosane C26F54: every locant omitted
    ("tetrapentacontafluorohexacosane", _chain(26, _perfluoro(26))[0]),
    # C27F55H: one hydrogen atom left, so every locant is cited
    (C27_LOCANTS["F"] + "-pentapentacontafluoroheptacosane", C27_SMILES),
    # C42: 54 fluorine and two iodine atoms (a parent below 51 and a count above)
    ("15,28-diiodo-" + C42_LOCANTS["F"] + "-tetrapentacontafluorodotetracontane", C42_SMILES),
    # the cumulene of the BR study with two alkyl branches: 79-methyl-78-(propan-2-yl)octaconta-1,...,77-ene
    ("79-methyl-78-(propan-2-yl)octaconta-" + _locants(1, 77) + "-heptaheptacontaene",
     "=".join(["C"] * 77) + "=C(C(C)C)C(C)C"),
    # the polyyne of the BR study: C124, 61 triple bonds at 2, 4,..., 122
    ("tetracosahecta-" + _locants(2, 122, 2) + "-henhexacontayne", "C" + "C#C" * 61 + "C"),
]


@pytest.mark.parametrize("name,smiles", REAL, ids=[r[0][:36] for r in REAL])
def test_the_reader_confirms_the_name_of_the_structure(name, smiles):
    assert long_count_round_trip(name, smiles) == LONG_COUNT_RULE


def test_the_perhalo_name_states_every_halogen_atom():
    name, smiles = REAL[3]
    assert long_count_round_trip(name, smiles) == LONG_COUNT_RULE
    # a count one off: a name of another molecule, or no name that the reader understands
    assert long_count_round_trip("tripentacontafluorohexacosane", smiles) is None
    assert long_count_round_trip("pentapentacontafluorohexacosane", smiles) is None
    # a hydrogen atom left: with no locants this is partial substitution
    partial = _chain(26, {**_perfluoro(26), 26: ["F", "F"]})[0]
    assert long_count_round_trip("tripentacontafluorohexacosane", partial) is None
    assert long_count_round_trip(name, partial) is None
    # one halogen atom of another kind
    assert long_count_round_trip("tetrapentacontachlorohexacosane", smiles) is None
    # the parent one carbon atom longer or shorter
    assert long_count_round_trip("tetrapentacontafluoroheptacosane", smiles) is None
    assert long_count_round_trip("tetrapentacontafluoropentacosane", smiles) is None


def test_the_halo_prefixes_without_locants_are_read_only_for_a_full_substitution():
    smiles = REAL[3][1]
    # no locants with another prefix beside, or a prefix that is no halogen: no reading
    assert long_count_round_trip("methyltetrapentacontafluorohexacosane", smiles) is None
    assert long_count_round_trip("tetrapentacontamethylhexacosane", smiles) is None
    assert long_count_round_trip("1-methyl-tetrapentacontafluorohexacosane", smiles) is None


def test_a_perhalogenated_alkene_is_complete_when_every_hydrogen_atom_is_replaced():
    # C27 with one double bond: 54 hydrogen atoms, all fluorine
    smiles = "C(F)(F)=C(F)" + "C(F)(F)" * 24 + "C(F)(F)F"
    assert long_count_round_trip("tetrapentacontafluoroheptacos-1-ene", smiles) == LONG_COUNT_RULE
    assert long_count_round_trip("pentapentacontafluoroheptacos-1-ene", smiles) is None
    assert long_count_round_trip("tripentacontafluoroheptacos-1-ene", smiles) is None
    assert long_count_round_trip("tetrapentacontafluoroheptacos-2-ene", smiles) is None


CUMULENE_NAME, CUMULENE_SMILES = REAL[1]

# one corruption of a name each: a locant, a count, the parent, the ending, and an extra token
MUTATIONS = [
    ("a locant out of the chain", "pentapentaconta-" + _locants(1, 53) + ",55-tetrapentacontaene"),
    ("a locant repeated", "pentapentaconta-" + _locants(1, 53) + ",53-tetrapentacontaene"),
    ("a locant dropped", "pentapentaconta-" + _locants(1, 53) + "-tetrapentacontaene"),
    ("the count lowered", CUMULENE_NAME.replace("-tetrapentacontaene", "-tripentacontaene")),
    ("the count raised", CUMULENE_NAME.replace("-tetrapentacontaene", "-pentapentacontaene")),
    ("the parent one carbon atom shorter", CUMULENE_NAME.replace("pentapentaconta-", "tetrapentaconta-")),
    ("the parent one carbon atom longer", CUMULENE_NAME.replace("pentapentaconta-", "hexapentaconta-")),
    ("the double bonds triple", CUMULENE_NAME[:-3] + "yne"),
    ("a token left over", CUMULENE_NAME + "x"),
    ("a token in front", "x" + CUMULENE_NAME),
    ("a stereo descriptor in front", "(2E)-" + CUMULENE_NAME),
    ("a substituent the structure has not", "2-methyl" + CUMULENE_NAME),
    ("a space", CUMULENE_NAME.replace("-1,2", " 1,2", 1)),
]


@pytest.mark.parametrize("what,name", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_a_corrupted_name_is_refused(what, name):
    assert long_count_round_trip(CUMULENE_NAME, CUMULENE_SMILES) == LONG_COUNT_RULE
    assert long_count_round_trip(name, CUMULENE_SMILES) is None


def test_a_name_of_another_cumulene_is_refused():
    assert long_count_round_trip(CUMULENE_NAME, "=".join(["C"] * 54)) is None
    assert long_count_round_trip(CUMULENE_NAME, "=".join(["C"] * 56)) is None
    assert long_count_round_trip(CUMULENE_NAME, "C" + "=".join(["C"] * 53) + "C") is None


HALO_NAME, HALO_SMILES = REAL[4]


@pytest.mark.parametrize("what,name", [
    ("a locant moved", HALO_NAME.replace("13,13,14", "13,13,15", 1)),
    ("a locant dropped", HALO_NAME.replace("13,13,14", "13,14", 1)),
    ("a locant added", HALO_NAME.replace("13,13,14", "13,13,13,14", 1)),
    ("one halogen of another kind", HALO_NAME.replace("fluoro", "chloro")),
    ("the count one off", HALO_NAME.replace("pentapentaconta", "tetrapentaconta")),
    ("the parent one carbon atom longer", HALO_NAME.replace("heptacosane", "octacosane")),
    ("a different parent", HALO_NAME.replace("heptacosane", "hexacosane")),
    ("an extra prefix", "2-methyl-" + HALO_NAME),
])
def test_a_corrupted_halo_name_is_refused(what, name):
    assert long_count_round_trip(HALO_NAME, HALO_SMILES) == LONG_COUNT_RULE
    assert long_count_round_trip(name, HALO_SMILES) is None


def test_the_branched_name_is_refused_when_a_branch_is_changed():
    name, smiles = REAL[6]
    assert long_count_round_trip(name, smiles) == LONG_COUNT_RULE
    for what, wrong in [
        ("methyl for ethyl", name.replace("79-methyl", "79-ethyl")),
        ("the isopropyl branch at 77", name.replace("78-(propan-2-yl)", "77-(propan-2-yl)")),
        ("propyl for propan-2-yl", name.replace("(propan-2-yl)", "propyl")),
        ("no enclosing marks", name.replace("(propan-2-yl)", "propan-2-yl")),
        ("the methyl branch at 80", name.replace("79-methyl", "80-methyl")),
        ("the methyl branch dropped", name.replace("79-methyl-", "")),
        ("the isopropyl branch dropped", name.replace("78-(propan-2-yl)", "")),
    ]:
        assert long_count_round_trip(wrong, smiles) is None, what


def test_the_polyyne_name_is_refused_when_a_bond_moves():
    name, smiles = REAL[7]
    assert long_count_round_trip(name, smiles) == LONG_COUNT_RULE
    assert long_count_round_trip(name.replace("122-henhexacontayne", "121-henhexacontayne"), smiles) is None
    assert long_count_round_trip(name.replace("tetracosahecta-2,", "tetracosahecta-1,"), smiles) is None
    assert long_count_round_trip(name.replace("henhexacontayne", "hexacontayne"), smiles) is None


def test_a_name_with_no_count_above_50_is_not_this_readers():
    """OPSIN reads these; the reader keeps to the names it cannot."""
    assert long_count_round_trip("hexacosane", "C" * 26) is None
    assert long_count_round_trip("pentacontane", "C" * 50) is None
    assert name_facts_from_text("hexacosane") is None
    assert name_facts_from_text("hexacosane", 0) is not None
    assert long_count_round_trip("henpentacontane", "C" * 51) == LONG_COUNT_RULE
    assert long_count_round_trip("heptacontane", "C" * 70) == LONG_COUNT_RULE          #
    assert long_count_round_trip("heptacontane", "C" * 69) is None


def test_the_reader_confirms_nothing_for_an_input_with_stereo():
    """The reconstructor is constitution-only: it never confirms a name for an input that asserts stereo."""
    assert long_count_round_trip("henpentacont-2-ene", "C/C=C/" + "C" * 48) is None
    assert long_count_round_trip("henpentacont-2-ene", "C/C=C\\" + "C" * 48) is None
    assert long_count_round_trip("henpentacont-2-ene", "CC=C" + "C" * 48) == LONG_COUNT_RULE
    # without the guard the constitution alone would be confirmed
    from orthonym.validation.reconstruct import Verdict, reconstruct_and_verify
    assert reconstruct_and_verify(name_facts_from_text("henpentacont-2-ene"),
                                  Chem.MolFromSmiles("C/C=C/" + "C" * 48)).verdict == Verdict.CONFIRMED


@pytest.mark.parametrize("name", [
    "",
    "x",
    "ane",
    "henpentacontan",
    "henpentacontaneX",
    "henpentacontane (2E)",
    "henpentacontane-1-ol",                       # a suffix this reader does not know
    "(2E)-henpentacont-2-ene",
    "1-hydroxyhenpentacontane",                   # a prefix outside this reader's vocabulary
    "henpentacontane-1,2-diyl",
    "1,1,1-trifluorohenpentacontane-2-amine",
    "henpentaconta-1,2-dien-4-yne",               # 'ene' and 'yne' in one name: not read
    "cyclohenpentacontane",
    "henpentaconta-1,2,3-diene",                  # a locant list that does not match the multiplier
    "henpentaconta-1-diene",
    "henpentaconta-0-ene",                        # a locant out of the chain
    "henpentaconta-51-ene",
    "henpentaconta-1,1-diene",
])
def test_a_name_the_reader_does_not_wholly_understand_gets_no_verdict(name):
    for smiles in ("C" * 51, "C" * 52, "C=C" + "C" * 49, "C=C=C" + "C" * 48):
        assert long_count_round_trip(name, smiles) is None, (name, smiles)


def test_a_methyl_prefix_with_its_locant_is_read_as_written():
    # 2-methyl on a chain of 51: CH3-CH(CH3)-... the structure has 52 carbon atoms in all
    assert long_count_round_trip("2-methylhenpentacontane", "CC(C)" + "C" * 49) == LONG_COUNT_RULE
    assert long_count_round_trip("2-methylhenpentacontane", "C" * 52) is None
    assert long_count_round_trip("1,2-dimethylhenpentacontane", "CCC(C)" + "C" * 49) == LONG_COUNT_RULE
    assert long_count_round_trip("1,2-dimethylhenpentacontane", "CC(C)C(C)" + "C" * 48) is None
    assert long_count_round_trip("1-methylhenpentacontane", "C" * 52) == LONG_COUNT_RULE      # read as written


def test_the_substituent_table_holds_propan_2_yl():
    from orthonym.validation.reconstruct import NameFacts, Verdict, reconstruct_and_verify
    facts = NameFacts(parent_kind="chain", parent_length=3, substituents=(("propan-2-yl", 2),))
    # 2,3-dimethylbutane is propane with a propan-2-yl group at 2; 3-methylpentane is not
    assert reconstruct_and_verify(facts, Chem.MolFromSmiles("CC(C)C(C)C")).verdict == Verdict.CONFIRMED
    assert reconstruct_and_verify(facts, Chem.MolFromSmiles("CCC(C)CC")).verdict == Verdict.MISMATCH


def test_the_reader_never_raises_on_odd_text():
    for name in ["-" * 80, "1," * 300 + "ane", "a" * 3000, "1-" + "tri" * 40 + "fluorohenpentacontane",
                 "éne", "1,2-", "-1-ene", "1-ene"]:
        assert long_count_round_trip(name, "C" * 51) is None


def test_every_prefix_the_reader_knows_is_in_the_substituent_table_of_the_reconstructor():
    """The reader names a prefix, the reconstructor builds it: the two vocabularies stay one."""
    from orthonym.validation import long_count_read_back as reader
    from orthonym.validation.reconstruct import _SUBSTITUENT_TABLE
    assert set(reader._BARE + reader._ENCLOSED) <= set(_SUBSTITUENT_TABLE)


@pytest.mark.parametrize("name", [
    "1-fluoro-henpentacontane",          # a hyphen before the parent
    "01-fluorohenpentacontane",          # a locant with a leading zero
    "1-chloro-fluorohenpentacontane",    # a hyphen not followed by locants
])
def test_a_badly_formed_spelling_is_not_read(name):
    """ Hyphens, (a) (the Blue Book): in substitutive names hyphens separate locants from
    words, as in '2-chloro-2-methylpropane (PIN)'; a hyphen that precedes no locant, or a locant with a leading
    zero, is a spelling the producer never writes, so it is not read and cannot ship as a verified name."""
    assert name_facts_from_text(name) is None


def test_the_well_formed_spellings_are_still_read():
    assert name_facts_from_text("1-chloro-2-fluorohenpentacontane") is not None
    assert name_facts_from_text("1-chloro-2-(propan-2-yl)henpentacontane") is not None
