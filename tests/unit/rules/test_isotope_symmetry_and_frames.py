""",: a label on one of several atoms the graph cannot tell apart, a frame in the other Kekule
alternation, a slot after a bridge prefix, a skeleton that needs several hang budgets.

Cited rules (the Blue Book, with their section headings):

* NAMES,: the descriptor stands "before the part of the compound that is isotopically
  substituted" ('2,3-... (2H)' before the fused system that follows its bridge prefix,.
* Omission of locants,: "if isotopic modification requires a locant to specify its
  position, then all locants must be specified and none are omitted";: "Locants are not omitted
  when there is a possibility of isomers". The two hydroxyl oxygens of one phosphono group are one position
  of the molecule: there is no possibility of isomers, so '(2H1)phosphonooxy' cites none.
* Different isotopic modifications on otherwise identical substituents: the copies are "cited
  separately" ('1-(phosphonooxy)-3-[(2H1)phosphonooxy]propan-2-ol').
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.errors import is_failure_name
from orthonym.rules import isotope_passes as P
from orthonym.rules import isotope_scopes as S
from orthonym.rules import isotopes as I
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.unit

GLYCEROL_PHOSPHATE = "[2H]OP(=O)(O)OCC(COP(=O)(O)O)O"


# -- the symmetry of the molecule --------------------------------------------------------------------------


def test_a_label_is_taken_modulo_the_symmetry_of_the_molecule():
    model = S._Model(Chem.MolFromSmiles("[2H]OP(=O)(O)OC"))
    (position, record), = model.need.items()
    hydroxyl = [p for p, a in enumerate(model.heavy_atoms)
                if p != position and model.cls[p] == model.cls[position]]
    assert len(hydroxyl) == 1                                  # the other hydroxyl oxygen of the phosphono
    assert S._class_positions(model, lambda p: record, position, tuple(hydroxyl)) == (position,)
    carbonyl = next(p for p, a in enumerate(model.heavy_atoms) if model.cls[p] not in
                    {model.cls[position]} and p != position and model.heavy.GetAtomWithIdx(p).GetSymbol() == "O"
                    and model.heavy.GetAtomWithIdx(p).GetDegree() == 1)
    assert S._class_positions(model, lambda p: record, position, (carbonyl,)) is None


def test_one_deuterium_of_one_phosphono_group_is_cited_without_a_locant():
    original = Chem.MolFromSmiles(GLYCEROL_PHOSPHATE)
    stripped, label_map = I.strip_isotopes(original)
    name = S.compose_scopes("1,3-bis(phosphonooxy)propan-2-ol", original, label_map,
                            stripped.GetNumHeavyAtoms())
    assert name == "1-(phosphonooxy)-3-[(2H1)phosphonooxy]propan-2-ol"
    assert name_is_rt_exact(name, GLYCEROL_PHOSPHATE)


@pytest.mark.opsin_gate
def test_glycerol_bisphosphate_with_one_deuterium_is_named():
    res = name_best_effort(GLYCEROL_PHOSPHATE)
    assert res["name"] == "1-(phosphonooxy)-3-[(2H1)phosphonooxy]propan-2-ol"
    assert name_is_rt_exact(res["name"], GLYCEROL_PHOSPHATE)


def test_two_units_of_one_nuclide_cited_without_locants_are_one_descriptor():
    """: "When two or more nuclide symbols appear at the same place in the name they are cited first in
    alphabetical order and then by their mass number": two CD3 of one tert-butyl are '(2H6)', not '(2H3,2H3)'."""
    model = S._Model(Chem.MolFromSmiles("[2H]C([2H])([2H])C(C)(C)C([2H])([2H])[2H]"))
    first, second = sorted(model.need)
    assert S._descriptor(model, [("*3", (first,)), ("*3", (second,))], False, frozenset()) == "(2H6)"


# -- a frame in the other Kekule alternation ---------------------------------------------------------------


def test_a_conjugated_ring_in_the_other_alternation_is_the_same_frame():
    # the periphery of '10b,10c-dihydropyren-1-yl' as the input draws it and as OPSIN writes it
    drawn = Chem.MolFromSmiles("C1=CC2=CC=C3C=CC(c4ccccc4)=C4C=CC(=C1)C2C34")
    read = Chem.MolFromSmiles("C1=CC2=CC=C3C(c4ccccc4)=CC=C4C=CC(=C1)C2C43")
    assert Chem.MolToSmiles(drawn) != Chem.MolToSmiles(read)
    assert S._other_alternation(read, drawn)
    assert not S._other_alternation(Chem.MolFromSmiles("C1=CC=CC=C1"), Chem.MolFromSmiles("C1CCCCC1"))


# -- the slot after a bridge prefix ------------------------------------------------------------------------

BRIDGED = ("(7R,14R)-7-(difluoromethoxy)-1-methyl-6,7-dihydro-7,14-methano[1,3]benzimidazolo[1,2-b]"
           "[2,5]benzodiazocin-5(14H)-one")


def test_the_fused_system_that_follows_its_bridge_prefix_has_a_slot_of_its_own():
    slot = BRIDGED.index("[1,3]")
    offsets = I._insertion_offsets(BRIDGED)
    assert slot in offsets
    assert BRIDGED.index("[2,5]") not in offsets               # a bracket after a bracket opens no system
    assert BRIDGED[:slot].endswith("methano")


# -- a skeleton that needs several hang budgets ------------------------------------------------------------


class _Trips:
    """A naming that runs out of the hang budget the first ``trips`` times it is asked."""

    def __init__(self, trips):
        self.trips, self.calls = trips, 0

    def name(self, smiles):
        from orthonym.assembly.fragment_naming import PerfBudgetExceeded
        self.calls += 1
        if self.calls <= self.trips:
            raise PerfBudgetExceeded()
        return "name of " + smiles


def test_a_skeleton_naming_is_begun_again_while_the_call_has_attempts_left(monkeypatch):
    from orthonym.assembly.fragment_naming import PerfBudgetExceeded
    rearmed = []
    monkeypatch.setattr("orthonym.assembly.fragment_naming.rearm_hang_budgets", lambda: rearmed.append(1))
    token = I._SKELETON_NAMINGS.set([0])
    try:
        naming = _Trips(3)
        assert I._name_skeleton(naming, "CCO") == "name of CCO"
        assert (naming.calls, len(rearmed)) == (4, 3)
        with pytest.raises(PerfBudgetExceeded):                # two attempts are left of the five
            I._name_skeleton(_Trips(9), "CCO")
    finally:
        I._SKELETON_NAMINGS.reset(token)


# -- names (OPSIN), the three molecules whose skeleton or frame needed these -------------------------------

SLOW_CASES = [
    ("a cage of thirteen rings: its systematic spelling needs more than one hang budget",
     "[2H]C1=C(C(=C(C(=C1[2H])[2H])N2C3=C(C4=CC=CC=C42)C5=C(C=C3)C6=C7N5C8=CC9=C(C=C8C7=CC=C6)C1=C2N9C3=C(C2=CC=C1)C=C1C(=C3)C2=CC=CC=C2N1C1=C(C(=C(C(=C1[2H])[2H])[2H])[2H])[2H])[2H])[2H]",
     "14,35-di[(2H5)phenyl]-1,5,14,35-tetraazatridecacyclo[25.17.1.1^5,19.0^2,26.0^4,24.0^6,18.0^7,15.0^8,13."
     "0^31,45.0^32,44.0^34,42.0^36,41.0^23,46]hexatetraconta-2,4(24),6(18),7(15),8,10,12,16,19(46),20,22,25,"
     "27(45),28,30,32(44),33,36,38,40,42-henicosaene"),
    ("the ring D of a bridged fused parent: the slot after the bridge prefix",
     "[2H]C1=C(C(=C2[C@H]3C[C@@](C4=NC5=CC=CC=C5N34)(N(C(=O)C2=C1[2H])C6=NC(=NC=C6)CN7CC(C7C(F)(F)F)O)OC(F)F)C)[2H]",
     "(7R,14R)-7-(difluoromethoxy)-6-(2-{[3-hydroxy-2-(trifluoromethyl)azetidin-1-yl]methyl}pyrimidin-4-yl)-1-"
     "methyl-6,7-dihydro-7,14-methano(2,3,4-2H3)[1,3]benzimidazolo[1,2-b][2,5]benzodiazocin-5(14H)-one"),
    ("a dihydropyrenyl whose frame OPSIN writes in the other Kekule alternation",
     "[2H]C1=C(C(=C(C(=C1[2H])C2=C3C=CC4=CC=CC5=CC=C(C3C54)C=C2)[2H])C6=C(C(=C(C(=C6[2H])[2H])C7=C(C(=C(C(=C7[2H])[2H])C8=C9C(=C(C1=C2C9=C(C(=C8[2H])[2H])OC2=C(C(=C1[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H])[2H]",
     "4-(4-{4-[3-(10b,10c-dihydropyren-1-yl)(2H4)phenyl](2H4)phenyl}(2H4)phenyl)(2H7)-15-oxatetracyclo"
     "[10.2.1.0^5,14.0^8,13]pentadeca-1,3,5,7,9,11,13-heptaene"),
]


@pytest.mark.slow
@pytest.mark.opsin_gate
@pytest.mark.parametrize("why,smiles,expected", SLOW_CASES, ids=[c[0][:40] for c in SLOW_CASES])
def test_the_molecules_these_were_made_for_are_named(why, smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected, why
    assert name_is_rt_exact(res["name"], smiles)


# -- the slot after a bridge prefix is not one the book prints --------------------------------------------


def test_only_the_slot_right_after_a_bridge_prefix_is_unprinted():
    assert I._unprinted_slot(BRIDGED, BRIDGED.index("[1,3]"))
    assert not I._unprinted_slot(BRIDGED, BRIDGED.index("[2,5]"))
    # after a hydro prefix the book prints the descriptor: '2,3-dihydro(15N)-1H-indole (PIN)'
    hydro = "5,6-dihydro[1,2,4]triazolo[4,3-a]pyridine"
    assert not I._unprinted_slot(hydro, hydro.index("["))
    # after a substituent prefix too: the descriptor of the parent stands after the prefixes
    methyl = "2,3-dimethyl[1,2,4]triazolo[4,3-a]pyridine"
    assert not I._unprinted_slot(methyl, methyl.index("["))
    epoxy = "1,4-epoxy[1,3]benzodioxole"
    assert I._unprinted_slot(epoxy, epoxy.index("["))


@pytest.mark.parametrize("text", [
    "7,14-methano", "4a,8a-ethano", "1,4:5,8-dimethano", "4a,8a-di(metheno)", "3,5-(epoxymethano)",
    "2,3-epoxy", "1,2-prop[1]eno", "1,3-buta[1,3]dieno", "1,4-benzeno",
    "1,2-sulfano", "3,4-diazeno", "1,2-epidioxy", "4,5-epithio", "1,6-epimino", "2,3-epicyclopenta",
    "(7R)-7,14-dihydro-7,14-methano", "4a,8a-metheno", "1,4-naphthaleno", "2,5-furano", "1,3-epoxireno",
])
def test_a_bridge_prefix_is_told_from_the_project_s_bridge_data(text):
    """ Names of bridges (the Blue Book), (:14130) 'metheno'... the prefixes the
    producer spells (``bridged_fused_pin/prefixes.py``), the reader of names reads
    (``validation/spelling/units.py``) and the book lists (a closed list of eleven words missed 'metheno',
    'epoxymethano', 'propeno', 'sulfano' and every 'epi' prefix)."""
    from orthonym.rules.bridged_fused_pin.prefixes import ends_in_bridge_prefix
    assert ends_in_bridge_prefix(text)


@pytest.mark.parametrize("text", [
    "6,7-dihydro", "2,3-dimethyl", "2,3-dicyano", "2,3-dichloro", "5,6-diphenyl", "5,6-di(methyl)",
    "7-methano", "2,3-dihydroxy", "1,2-ditrimethylsilyl", "2-methyl", "",
])
def test_a_substituent_prefix_or_a_single_locant_is_no_bridge(text):
    from orthonym.rules.bridged_fused_pin.prefixes import ends_in_bridge_prefix
    assert not ends_in_bridge_prefix(text)


def test_the_bridge_vocabulary_holds_every_word_of_the_producer_and_of_the_reader():
    """One home of the rule: the words of the producer's tables and of the reader's lexicon are in it, so
    a prefix either of them learns is a bridge here too."""
    import re
    from orthonym.rules.bridged_fused import BRIDGE_PREFIXES, GENERAL_ONLY_BRIDGE_PREFIXES
    from orthonym.rules.bridged_fused_pin import prefixes
    from orthonym.validation.spelling.units import BRIDGE_RE
    words = prefixes.bridge_prefix_words()
    assert set(re.findall(r"[a-z]+", BRIDGE_RE)) <= words
    assert set(prefixes._ALKANO.values()) | set(prefixes._ONE_ATOM.values()) <= words
    assert set(prefixes._PRESELECTED) | set(prefixes._PRESELECTED.values()) <= words
    assert {re.sub(r"\[[\d,]+\]", "", w) for w in prefixes._ALKENO.values()} <= words
    assert set(BRIDGE_PREFIXES.values()) | set(GENERAL_ONLY_BRIDGE_PREFIXES) <= words
    assert {"metheno", "epoxymethano", "propeno", "sulfano", "diazeno", "benzeno"} <= words


@pytest.mark.slow
@pytest.mark.opsin_gate
def test_that_name_is_below_the_pin_so_the_default_tier_declines_it():
    smiles, _expected = SLOW_CASES[1][1], SLOW_CASES[1][2]
    best = name_best_effort(smiles)
    assert best["name"] == _expected and best["tier"] != "pin_verified"
    assert is_failure_name(Orthonym(style="pin").name_tiered(smiles)["name"])


#: fused parents whose name starts at a bracket of heteroatom locants after a letter or a bracket: the names
#: are the preferred names whatever the tier, and a label on them is no slot after a bridge prefix
TRIAZOLO = [
    ("[2H]c1nc2ncccn2n1", "(2-2H)[1,2,4]triazolo[1,5-a]pyrimidine"),
    ("[2H]c1nnc2ccccn12", "(3-2H)[1,2,4]triazolo[4,3-a]pyridine"),
    ("[2H]C([2H])([2H])c1ccc2nnc(C(F)(F)F)n2n1",
     "6-(2H3)methyl-3-(trifluoromethyl)[1,2,4]triazolo[4,3-b]pyridazine"),
    ("[2H]C([2H])([2H])Oc1ccc2nnc(C(F)(F)F)n2n1",
     "6-(2H3)methoxy-3-(trifluoromethyl)[1,2,4]triazolo[4,3-b]pyridazine"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", TRIAZOLO, ids=[t[1][:40] for t in TRIAZOLO])
def test_a_label_on_a_triazolo_fused_parent_keeps_its_pin_at_both_tiers(smiles, expected):
    from orthonym.cli import _emit_tier_flags
    for flags in ({}, _emit_tier_flags("best-effort")):
        row = Orthonym(style="pin", **flags).name_tiered(smiles)
        assert (row["name"], row["tier"]) == (expected, "pin_verified")
        assert name_is_rt_exact(row["name"], smiles)


# -- a cover cannot pool two pieces of one nuclide in the first pass ---------------------------------------


def test_the_pooling_of_two_unlocanted_pieces_is_reached_only_by_the_second_pass():
    """A piece is the unit of a slot with the positions OPSIN labels, fixed by the frame; the first pass
    offers a piece only for the positions it labels, and a cover takes no position twice, so two pieces of
    one slot and nuclide ('(2H3,2H3)') need a piece mapped onto another atom of its class: the second pass."""
    model = S._Model(Chem.MolFromSmiles("[2H]C([2H])([2H])C(C)(C)C([2H])([2H])[2H]"))
    first, _second = sorted(model.need)
    key = model.need[first][0][0]
    reach = {(0, "*3", key): (first, first, first), (0, "*1", key): (first,)}     # 'unit holds more' pieces

    class Slot:
        def sort_key(self):
            return (0,)

    assert not list(S._covers(model, reach, [Slot()], {"nodes": 0}))
    covers = list(S._covers(model, reach, [Slot()], {"nodes": 0}, extras=True))
    assert covers and all(len(c) == 2 and c[0][0] == c[1][0] for c in covers)      # the one piece, twice
    assert S._descriptor(model, [("*3", (first,)), ("*3", (_second,))], False, frozenset()) == "(2H6)"


# -- a labelled substituent prefix of a mononuclear parent is enclosed ----------------------------------------


def test_the_copies_of_a_group_without_locants_are_cited_enclosed_first():
    """ (heading '', the Blue Book): "For mononuclear parent hydrides with two or
    more substituents the first cited substituent never has enclosing marks unless it includes a locant. The
    second and further substituents are each enclosed with parentheses even for simple substituents." A copy
    that carries a descriptor includes a locant and its marks step up, 'di[(13C)methyl]')."""
    stripped = Chem.MolFromSmiles("OC(c1ccccc1)c1ccccc1")
    variants = S._apart_variants("diphenylmethanol", stripped)
    assert [v[0] for v in variants][:2] == ["(phenyl)(phenyl)methanol", "phenylphenylmethanol"]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("[2H]c1ccc(cc1)C(O)c1ccccc1", "[(4-2H)phenyl](phenyl)methanol"),
    ("[2H]c1ccc(cc1)[C@@H](O)c1ccccc1", "(1S)-[(4-2H)phenyl](phenyl)methanol"),
    ("[2H]c1ccc(cc1)[C@H](O)c1ccccc1", "(1R)-[(4-2H)phenyl](phenyl)methanol"),
])
def test_a_labelled_phenyl_of_diphenylmethanol_is_enclosed(smiles, expected):
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert row["name"] == expected
    assert name_is_rt_exact(row["name"], smiles)
