"""Isotope descriptors the Blue Book prints and OPSIN 2.9.0 has no grammar for, at the default tier.

OPSIN reads a letter locant once per atom ('(N,N-2H2)aniline') and rejects a hydrogen nuclide that has
neither a count nor a locant ('ethan(2H1)ol' is read, 'ethan(2H)ol' is not). The default tier spells
the printed form and ships it through its twin spelling (``validation.twin_spellings``); a wider tier
keeps the spelling OPSIN reads.

* (heading NAMES, the Blue Book; the rule:43824): '(N-2H2)aniline (PIN)' (:43830);
  a letter locant is cited once, while integer locants are repeated per atom Letter and/or
  numeral locants:44208, '(1,1,1,3,3-2H5)pentan-2-one':44208).
* (heading Omission of locants,:44178; the rule:44180): 'ethan(2H)ol (PIN) (as in
  ethanol)' (:44184); (:44208): '(1-2H1)ethan-1-(2H)ol (PIN)' (:44214).
* (heading ORDER OF NUCLIDE SYMBOLS,:43832; the rule:43834): 'methan(2H,18O)ol (PIN)'
  (:43840).
* (:43800): '(1-14C)pentan(3H)oic acid (PIN)' (:43812) and 'cyclohexane(2H)carboxylic acid (PIN)'
  (:43816).

A hydrogen nuclide is cited without count and locant only in a slot of a group that holds one hydrogen or in
a unit that holds exactly one hydrogen and cites no locant in its unmodified name (``rules.isotope_licence``);
elsewhere the Blue Book cites the count or the locant: (:43720) "When polysubstitution at a single
position is possible, the number of atoms substituted is always specified as a right subscript";
(:44180) "locants are omitted if no locants are necessary in unmodified names"; (:44202) "Locants
are not omitted when there is a possibility of isomers"; (heading:44208,:44210) "When locants are
needed... then all locants must be cited"; printed '(2-2H1)acetic acid (PIN)' (:43804), '(O-2H)acetic acid
(PIN)' (:43806), '1,1,1-trifluoro(2-2H1)ethane (PIN)' (:44155, under Numbering in relation to the
unmodified compound).
"""
import pytest
from rdkit import Chem

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import _canon, _independent_parse, name_best_effort
from tests.support.wider_tier import wider_tier

pytestmark = pytest.mark.opsin_gate

# (smiles, name, twin); the first group is printed in the Blue Book, the second is the same rule on
# another structure (the spelling is the producer's, the twin proves the structure)
PRINTED_ROWS = [
    ("[2H]N([2H])c1ccccc1", "(N-2H2)aniline", "(N,N-2H2)aniline"),                    #:43830
    ("[2H]OCC", "ethan(2H)ol", "ethan(2H1)ol"),                                       #:44184
    ("CC([2H])O[2H]", "(1-2H1)ethan-1-(2H)ol", "(1-2H1)ethan-1-(2H1)ol"),             #:44214
    ("C[18O][2H]", "methan(2H,18O)ol", "methan(2H1,18O)ol"),                          #:43840
    ("[3H]O[14C](=O)CCCC", "(1-14C)pentan(3H)oic acid", "(1-14C)pentan(3H1)oic acid"),   #:43812
    ("[2H]OC(=O)C1CCCCC1", "cyclohexane(2H)carboxylic acid", "cyclohexane(2H1)carboxylic acid"),  #:43816
]
DERIVED_ROWS = [
    ("O=C([2H])c1ccccc1", "benzene(2H)carbaldehyde", "benzene(2H1)carbaldehyde"),
    ("[2H]OC", "methan(2H)ol", "methan(2H1)ol"),
    ("[2H]SCC", "ethane(2H)thiol", "ethane(2H1)thiol"),
    ("[2H][Se]CC", "ethane(2H)selenol", "ethane(2H1)selenol"),
    ("CC(C)O[2H]", "propan-2-(2H)ol", "propan-2-(2H1)ol"),
    ("[2H]OC1CCCCC1", "cyclohexan(2H)ol", "cyclohexan(2H1)ol"),
    ("[3H]OCC", "ethan(3H)ol", "ethan(3H1)ol"),
    ("[2H]N([2H])CC", "(N-2H2)ethanamine", "(N,N-2H2)ethanamine"),
    ("CC[N+]([2H])([2H])[2H]", "(N-2H3)ethanaminium", "(N,N,N-2H3)ethanaminium"),
    ("[3H]N([3H])c1ccccc1", "(N-3H2)aniline", "(N,N-3H2)aniline"),
]
# a descriptor inside a substituent: the unit cites locants in its unmodified name, so the descriptor cites
# its locant:44180,:44210), and the enclosing marks step up,
# the Blue Book, as '{[(2H1)methoxy(2H2)methyl]sulfanyl}methaneperoxol (PIN)':44188) although '(2H)'
# reads as an indicated hydrogen to the mark primitive
NESTED_ROW = ("FC(F)(F)C(F)([2H])C1CCCCC1", "[1,2,2,2-tetrafluoro(1-2H)ethyl]cyclohexane",
              "[1,2,2,2-tetrafluoro(2H1)ethyl]cyclohexane")
PIN_ROWS = PRINTED_ROWS + DERIVED_ROWS

# the same structures at the best-effort tier: the spelling OPSIN reads, byte-identical to the name
# before the default tier spelled the printed forms. Includes the class the hydroxyl-hydrogen skeleton
# choice reaches (a D on -OH / -SH / -SeH): ungated, it would turn 'ethan-1-(2H1)ol' into 'ethan(2H1)ol'
BEST_EFFORT = {
    "[2H]N([2H])c1ccccc1": "(N,N-2H2)aniline",
    "[2H]OCC": "ethan-1-(2H1)ol",
    "CC([2H])O[2H]": "(1-2H1)ethan-1-(2H1)ol",
    "C[18O][2H]": "methan(2H1,18O1)ol",
    "O=C([2H])c1ccccc1": "benzene(2H1)carbaldehyde",
    "CCO[2H]": "ethan-1-(2H1)ol",
    "[2H]SCC": "ethane-1-(2H1)thiol",
    "CCS[2H]": "ethane-1-(2H1)thiol",
    "CC(C)O[2H]": "propan-2-(2H1)ol",
    "CO[2H]": "methan(2H1)ol",
    "[2H][Se]CC": "ethane(2H1)selenol",
    "[2H]OC1CCCCC1": "cyclohexan(2H1)ol",
    "[3H]OCC": "ethan-1-(3H1)ol",
    "[2H]N([2H])CC": "(N,N-2H2)ethanamine",
    "[3H]O[14C](=O)CCCC": "(1-14C)pentan(3H1)oic acid",
    "[2H]OC(=O)C1CCCCC1": "cyclohexane(2H1)carboxylic acid",
    "[2H]C1=CC2SC1c1ccccc12": "1,4-dihydro-1,4-epithio(2-2H)naphthalene",
    "[2H]N1C2C=CC1c1ccccc12": "1,4-dihydro-1,4-epimino(9-2H)naphthalene",
    NESTED_ROW[0]: NESTED_ROW[2],
    "[2H]C(=O)O": "(2H1)formic acid",
    "OC(=O)C([2H])(Cl)Cl": "dichloro(2H1)acetic acid",
    "[2H]OP(=O)(O)O": "(2H1)phosphoric acid",
    "[2H]OS(=O)(=O)O": "(2H1)sulfuric acid",
    "[2H]OC(=O)C(=O)O": "(2H1)oxalic acid",
    "FC(F)(F)C([2H])(F)Cl": "1-chloro-1,2,2,2-tetrafluoro(2H1)ethane",
    "FC(F)(F)C(F)([2H])c1ccccc1": "[1,2,2,2-tetrafluoro(2H1)ethyl]benzene",
    "OC([2H])(C(F)(F)F)C(F)(F)F": "1,1,1,3,3,3-hexafluoro(2H1)propan-2-ol",
}
# read by OPSIN as printed: no twin
CONTROLS = [
    ("CC(=O)N[2H]", "(N-2H1)acetamide"),
    ("CC[18OH]", "ethan(18O)ol"),
    ("CC(=O)O[2H]", "(O-2H)acetic acid"),
    ("Cl[12CH](Cl)Cl", "trichloro(12C)methane"),
    ("[2H]OCCO", "(O-2H)ethane-1,2-diol"),                 # 3b must not touch the multiplied-suffix record
    ("[2H]Oc1ccccc1", "(O-2H)phenol"),
    ("[2H]OC=O", "(O-2H)formic acid"),
]
# a labelled bridged ring system: the descriptor stands in front of the parent, after the bridge prefix
BRIDGE_ROWS = [
    ("[2H]C1=CC2SC1c1ccccc12", "1,4-dihydro-1,4-sulfano(2-2H)naphthalene", "1,4-dihydro-1,4-epithio(2-2H)naphthalene"),
    ("[2H]N1C2C=CC1c1ccccc12", "1,4-dihydro-1,4-azano(9-2H)naphthalene", "1,4-dihydro-1,4-epimino(9-2H)naphthalene"),
]


@pytest.mark.parametrize("smiles,name,twin", PIN_ROWS + [(s, n, t) for s, n, t in BRIDGE_ROWS])
def test_default_tier_gives_the_printed_spelling(smiles, name, twin):
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert row["verified"] == "opsin_twin" and row["opsin"] == "verified_twin", row
    assert row["gate_outcome"].startswith("twin_verified:"), row
    assert row["spelling_failures"] == [], row


@pytest.mark.parametrize("smiles,name,twin", PIN_ROWS + [(s, n, t) for s, n, t in BRIDGE_ROWS])
def test_the_twin_of_the_default_name_is_the_structure_of_the_input(smiles, name, twin):
    """OPSIN rejects the printed spelling and a fresh read of the twin gives the input."""
    from orthonym.validation.twin_spellings import twin_of
    assert twin_of(name)[0] == twin
    assert _independent_parse(name, smiles) is None
    assert _canon(_independent_parse(twin, smiles)) == _canon(smiles)


@pytest.mark.parametrize("smiles,name", sorted(BEST_EFFORT.items()))
def test_best_effort_is_unchanged(smiles, name):
    row = name_best_effort(smiles)
    assert row["name"] == name and row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert _canon(_independent_parse(name, smiles)) == _canon(smiles)


@pytest.mark.parametrize("smiles,name", CONTROLS)
def test_controls_that_opsin_reads_do_not_use_the_twin(smiles, name):
    row = default_tier_row(smiles)
    assert row["name"] == name and row["verified"] == "opsin" and row["tier"] == "pin_verified", row


def test_label_on_the_hydroxyl_hydrogen_keeps_the_locant_free_skeleton():
    # (:44180): "locants are omitted if no locants are necessary in unmodified names"; a twin
    # alone would ship 'ethan-1-(2H)ol', which cites a locant the unlabelled name 'ethanol' omits
    for smiles, name in (("[2H]OCC", "ethan(2H)ol"), ("CCO[2H]", "ethan(2H)ol"), ("[2H]OC", "methan(2H)ol"),
                         ("CO[2H]", "methan(2H)ol"), ("[2H]SCC", "ethane(2H)thiol"),
                         ("CCS[2H]", "ethane(2H)thiol")):
        assert default_tier_row(smiles)["name"] == name


def test_isotope_tier_order_independence():
    """The skeleton choice and the descriptor spelling read the tier of the call: a best-effort call
    leaves nothing behind for the default call, and the other way round."""
    from orthonym import Orthonym
    default = Orthonym(style="pin")
    for smiles, pin, general in (("[2H]OCC", "ethan(2H)ol", "ethan-1-(2H1)ol"),
                                 ("[2H]N([2H])c1ccccc1", "(N-2H2)aniline", "(N,N-2H2)aniline"),
                                 ("[2H]SCC", "ethane(2H)thiol", "ethane-1-(2H1)thiol")):
        first = name_best_effort(smiles)["name"]
        middle = default.name_tiered(smiles)["name"]
        last = name_best_effort(smiles)["name"]
        assert first == last == general and middle == pin, (first, middle, last)


# ---- the count and the locant the Blue Book cites ------------------------------------------------------

# the locant is cited because an isomer exists or the unit's unmodified name cites locants; OPSIN reads these
# spellings itself
LOCANTED_ROWS = [
    ("[2H]C(=O)O", "(1-2H)formic acid"),                       # isomer '(O-2H)formic acid',:44202
    ("OC(=O)C([2H])(Cl)Cl", "dichloro(2-2H)acetic acid"),      # isomer on the OH; '(2-2H1)acetic acid (PIN)':43804
    ("[2H]CC(F)(F)F", "1,1,1-trifluoro(2-2H1)ethane"),         # the printed control,:44155
    ("FC(F)(F)C([2H])(F)Cl", "1-chloro-1,2,2,2-tetrafluoro(1-2H)ethane"),    #:44180,:44210
    ("FC(F)(F)C(F)([2H])c1ccccc1", "[1,2,2,2-tetrafluoro(1-2H)ethyl]benzene"),
    NESTED_ROW[:2],
    ("OC([2H])(C(F)(F)F)C(F)(F)F", "1,1,1,3,3,3-hexafluoro(2-2H)propan-2-ol"),
]
# the count-less spelling is refused and no locanted one is read: the default tier declines
DECLINED_ROWS = ["[2H]OP(=O)(O)O", "[2H]OS(=O)(=O)O", "[2H]OC(=O)C(=O)O",    # three and two equivalent OH,:43720
                 "[2H]c1ccccc1"]                                              # six equivalent CH
REFUSED_NAMES = ["(2H)formic acid", "dichloro(2H)acetic acid", "(2H)phosphoric acid", "(2H)sulfuric acid",
                 "(2H)oxalic acid", "1-chloro-1,2,2,2-tetrafluoro(2H)ethane",
                 "[1,2,2,2-tetrafluoro(2H)ethyl]cyclohexane", "[1,2,2,2-tetrafluoro(2H)ethyl]benzene",
                 "1,1,1,3,3,3-hexafluoro(2H)propan-2-ol"]


@pytest.mark.parametrize("smiles,name", LOCANTED_ROWS)
def test_the_locant_is_cited_where_the_blue_book_cites_it(smiles, name):
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["verified"] == "opsin", row
    assert row["spelling_failures"] == [], row


@pytest.mark.parametrize("smiles", DECLINED_ROWS)
def test_a_count_less_front_descriptor_over_several_alike_hydrogens_is_declined(smiles):
    row = default_tier_row(smiles)
    assert row["tier"] == "abstain" and row["limit_code"] == "NO_VERIFIED_PIN", row


@pytest.mark.parametrize("name", REFUSED_NAMES)
def test_the_refused_spellings_have_a_twin_but_are_not_licensed(name):
    """The twin of each reads (so the oracle could accept it); the licence is what refuses it."""
    from orthonym.validation.twin_spellings import twin_of
    assert twin_of(name) is not None


@pytest.mark.parametrize("name,smiles,expected", [
    ("(2H)formic acid", "[2H]C(=O)O", False), ("dichloro(2H)acetic acid", "OC(=O)C([2H])(Cl)Cl", False),
    ("(2H)phosphoric acid", "[2H]OP(=O)(O)O", False), ("(2H)oxalic acid", "[2H]OC(=O)C(=O)O", False),
    ("1-chloro-1,2,2,2-tetrafluoro(2H)ethane", "FC(F)(F)C([2H])(F)Cl", False),
    ("[1,2,2,2-tetrafluoro(2H)ethyl]benzene", "FC(F)(F)C(F)([2H])c1ccccc1", False),
    ("trichloro(2H)methane", "[2H]C(Cl)(Cl)Cl", True), ("(2H)formonitrile", "[2H]C#N", True),
    ("pentafluoro(2H)ethane", "[2H]C(F)(F)C(F)(F)F", True),
    ("[difluoro(2H)methyl]benzene", "[2H]C(F)(F)c1ccccc1", True),
    ("ethan(2H)ol", "[2H]OCC", True), ("methane(2H)sulfonic acid", "[2H]OS(=O)(=O)C", True),
    ("2-(2H)hydroxypropanoic acid", "[2H]OC(C)C(=O)O", True), ("(1-14C)pentan(3H)oic acid", "[3H]O[14C](=O)CCCC", True),
    ("cyclohexane(2H)carboxylic acid", "[2H]OC(=O)C1CCCCC1", True), ("benzene(2H)carbaldehyde", "O=C([2H])c1ccccc1", True),
    ("methan(2H,18O)ol", "C[18O][2H]", True), ("trichloro(13C,2H)methane", "Cl[13C]([2H])(Cl)Cl", True),
    ("(1-2H)formic acid", "[2H]C(=O)O", True),                  # no count-less item: nothing to license
])
def test_the_licence_of_a_count_less_hydrogen(name, smiles, expected):
    from orthonym.rules.isotope_licence import twin_licensed
    assert twin_licensed(name, Chem.MolFromSmiles(smiles)) is expected


def test_the_locant_is_needed_only_where_the_hydrogens_are_not_alike_or_the_name_cites_locants():
    from orthonym.rules.isotope_licence import needs_locant

    def needed(smiles, skel, before):
        return needs_locant(Chem.MolFromSmiles(smiles), skel, skel.index(before))
    assert needed("[2H]C(=O)O", "formic acid", "formic") is True                   # the isomer '(O-2H)formic acid'
    assert needed("OC(=O)C([2H])(Cl)Cl", "dichloroacetic acid", "acetic") is True   # the isomer on the OH
    assert needed("FC(F)(F)C([2H])(F)Cl", "1-chloro-1,2,2,2-tetrafluoroethane", "ethane") is True   # locants cited
    # the hydrogens of the unit are alike and its name cites no locant: the count decides, not a locant
    assert needed("[2H]c1ccccc1", "benzene", "benzene") is False
    assert needed("[2H]OP(=O)(O)O", "phosphoric acid", "phosphoric") is False
    assert needed("[2H]C(Cl)(Cl)Cl", "trichloromethane", "methane") is False
    assert needed("[2H]OCC", "ethanol", "ol") is False                              # a slot of a one-hydrogen group
    with wider_tier():
        assert needed("[2H]C(=O)O", "formic acid", "formic") is False


# ---- the descriptor formatter and the oracle ----------------------------------------------------

def test_the_letter_locant_is_cited_once_only_where_the_printed_form_does():
    from orthonym.rules.isotopes import format_isotope_descriptor as fmt
    assert fmt([("N", 2, "H", 2, 2)]) == "(N,N-2H2)"
    assert fmt([("N", 2, "H", 2, 2)], letter_once=True) == "(N-2H2)"
    assert fmt([("N", 2, "H", 3, 3)], letter_once=True) == "(N-2H3)"
    # a count of one has one letter either way
    assert fmt([("N", 2, "H", 1, 2)], letter_once=True) == "(N-2H1)"
    # integer locants repeat per atom,:44208); a tuple of locants is a list of positions
    assert fmt([(1, 2, "H", 3, 3)], letter_once=True) == "(1,1,1-2H3)"
    assert fmt([((2, 4), 2, "H", 2, 2)], letter_once=True) == "(2,4-2H2)"
    # two nuclides share one locant: the letter stays repeated, the twin rule rewrites one nuclide only
    assert fmt([("O", 2, "H", 2, 2), ("O", 18, "O", 1, 1)], letter_once=True) == "(O,O-2H2,O-18O)"
    # a two-letter element symbol is not a single capital letter
    assert fmt([("Se", 2, "H", 2, 2)], letter_once=True) == "(Se,Se-2H2)"


def test_the_marks_around_a_bare_hydrogen_descriptor_step_up():
    from orthonym.rules.isotopes import _escalate_marks_for_descriptor as escalate
    unstepped = "(1,2,2,2-tetrafluoro(2H)ethyl)cyclohexane"
    assert escalate(unstepped, unstepped.index("(2H)")) == "[1,2,2,2-tetrafluoro(2H)ethyl]cyclohexane"
    unstepped = "(1,2,2,2-tetrafluoro(2H,18O)ethyl)cyclohexane"
    assert escalate(unstepped, unstepped.index("(2H,")) == "[1,2,2,2-tetrafluoro(2H,18O)ethyl]cyclohexane"
    # a descriptor with a count or a locant took the same step before
    counted = "(1,2,2,2-tetrafluoro(2H1)ethyl)cyclohexane"
    assert escalate(counted, counted.index("(2H1)")) == "[1,2,2,2-tetrafluoro(2H1)ethyl]cyclohexane"


def test_the_oracle_reads_the_twin_of_a_candidate_opsin_rejects():
    from orthonym.rules import isotopes as iso
    mol = Chem.MolFromSmiles("[2H]OCC")
    assert iso._opsin_parse("ethan(2H)ol") is None
    assert iso._isotope_round_trips("ethan(2H)ol", mol) is True
    # the twin is of another structure: the candidate fails
    assert iso._isotope_round_trips("ethan(2H)ol", Chem.MolFromSmiles("CC([2H])O")) is False
    # a wider tier reads the candidate itself, as before
    with wider_tier():
        assert iso._isotope_round_trips("ethan(2H)ol", mol) is False
        assert iso._isotope_round_trips("ethan(2H1)ol", mol) is True


def test_the_oracle_does_not_ask_for_a_twin_when_opsin_reads_the_candidate(monkeypatch):
    from orthonym.rules import isotopes as iso
    asked = []
    monkeypatch.setattr(iso, "_twin_parse", lambda name, original=None: asked.append(name))
    assert iso._isotope_round_trips("(O-2H)acetic acid", Chem.MolFromSmiles("CC(=O)O[2H]")) is True
    assert iso._isotope_round_trips("ethan(18O)ol", Chem.MolFromSmiles("CC[18OH]")) is True
    assert asked == []
    iso._isotope_round_trips("ethan(2H)ol", Chem.MolFromSmiles("[2H]OCC"))
    assert asked == ["ethan(2H)ol"]


def test_the_twin_is_refused_when_opsin_reads_it_as_ambiguous(monkeypatch):
    from orthonym.rules import isotopes as iso
    monkeypatch.setattr("orthonym.validation.twin_spellings.ambiguity_messages",
                        lambda name: ("Position of isotope on eth",))
    assert iso._isotope_round_trips("ethan(2H)ol", Chem.MolFromSmiles("[2H]OCC")) is False


def test_the_spelling_reader_tells_a_hydrogen_nuclide_from_an_added_hydrogen():
    """'(2H)' after a locant is the added-hydrogen mark ('naphthalen-4a(2H)-amine'); after a name part it is
    a hydrogen nuclide of an isotope descriptor, whose parentheses count in the nesting order
    , the Blue Book)."""
    from orthonym.validation.spelling.checks_marks import enclosing_marks_check
    from orthonym.validation.spelling.lexer import enclosures, normalise

    def kinds(name):
        return [e.kind for e in enclosures(normalise(name))]

    assert kinds("ethan(2H)ol") == ["isotope"]
    assert kinds("trichloro(2H)methane") == ["isotope"]
    for name in ("naphthalen-4a(2H)-amine", "pyrimidine-2,4(1H,3H)-dione", "naphthalen-2(3H)-one"):
        assert kinds(name) == ["added_ih"], name
    assert enclosing_marks_check(None, "[1,2,2,2-tetrafluoro(2H)ethyl]cyclohexane") is None
    assert enclosing_marks_check(None, "(1,2,2,2-tetrafluoro(2H)ethyl)cyclohexane") is not None


def test_the_offer_selection_keeps_a_twin_verified_isotope_name():
    """An isotope-labelled name is gated at the exit of the call, after the offers: the offer check reads its
    twin, so the selection carries the name instead of leaving it to the single-offer fallback."""
    from orthonym import Orthonym, namer
    engine = Orthonym(style="pin")
    for smiles, name, unlabelled in (("[2H]OCC", "ethan(2H)ol", "CCO"),
                                     ("[2H]N([2H])c1ccccc1", "(N-2H2)aniline", "Nc1ccccc1"),
                                     ("C[18O][2H]", "methan(2H,18O)ol", "CO")):
        assert engine.name_tiered(smiles)["name"] == name
        assert engine._last_selected_offer is not None and engine._last_selected_offer.name == name
        assert namer._offer_rt_ok(name, smiles) is True
        assert namer._offer_rt_ok(name, unlabelled) is False   # the twin reads to the labelled structure only
