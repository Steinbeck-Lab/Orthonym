"""lane/l18 item 2, fix a performance pass: the locants of an isotope descriptor that cite an atom of a phane amplificant.

 Numbering, Numbering of phane parent hydrides, (the Blue Book): "locants for
amplificant atoms are constructed by citing the actual locants of the amplificant as superscripts to the locant of
the superatom that represents the amplificant in the simplified skeleton" (atom 4 of the amplificant at superatom 1
is ``1^4``); (:15115) orders composite locants by their primary locants first; (:44208)
"Specific positions of nuclides must be indicated in the isotopic descriptor by appropriate locants".

The placement search writes integers, and OPSIN reads ``14`` as ``1^4`` in a phane name, so the integer spelling
reads back and the read-back cannot tell it from the composite one: the spelling is settled on the name the search
returns (``rules.isotope_phane_locants``).
"""
import pytest

from orthonym import Orthonym
from orthonym import opsin_capabilities as caps
from orthonym.metrics import provenance as prov
from orthonym.rules import isotopes as iso
from orthonym.rules.isotope_phane_locants import (
    CompositeLocant, composite_locants, composite_sort_key, composite_tokens, decorated_with_composite_locants,
    skeleton_node_count, superatom_locants)

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

HEPTAPHANE = "2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphane"
HEXAPHANE = "1,4(1,4)-dibenzenacyclohexaphane"


def _accepts(*needles):
    """A read-back that accepts a candidate holding every needle (the OPSIN of these tests is the pure function)."""
    return lambda candidate: all(n in candidate for n in needles)


def test_superatom_locants_are_read_from_the_amplification_blocks():
    """ superatom locants, attachment locants in parentheses."""
    assert superatom_locants(HEPTAPHANE) == [1, 3, 5, 7]
    assert superatom_locants("(14-2H)-" + HEPTAPHANE) == [1, 3, 5, 7]
    assert superatom_locants("3(2,5)-pyridina-1,7(1),5(1,4)-tribenzenaheptaphane") == [1, 3, 5, 7]
    assert superatom_locants("1(1,3),4(1,4)-dibenzenacycloheptaphane") == [1, 4]
    assert superatom_locants("2,2'-bi-1,3-thiazole") == []


def test_an_integer_locant_of_an_amplificant_atom_becomes_a_composite_locant():
    """: atom 4 of the benzene at superatom 1 is cited ``1^4``."""
    assert composite_locants("(14-2H)-" + HEPTAPHANE, _accepts("1^4")) == "(1^4-2H)-" + HEPTAPHANE
    assert composite_locants("(12-2H)-" + HEXAPHANE, _accepts("1^2")) == "(1^2-2H)-" + HEXAPHANE


def test_the_composite_locant_is_the_lowest_primary_locant_that_reads_back():
    """A split of ``36`` into superatom ``3`` and atom 6, never ``3`` and ``6`` of another reading."""
    tried = []

    def reads_back(candidate):
        tried.append(candidate)
        return "3^6" in candidate
    assert composite_locants("(36-2H)-" + HEPTAPHANE, reads_back) == "(3^6-2H)-" + HEPTAPHANE
    assert tried == ["(3^6-2H)-" + HEPTAPHANE]


def test_a_node_of_the_simplified_skeleton_keeps_its_integer_locant():
    """: the atoms that do not belong to amplificants keep the locants of the simplified skeleton, and a
    locant that no split reads back as is the number of such a node."""
    plain = "(2,2-2H2)-1,3(1,4)-dibenzenacyclopentaphane"
    assert composite_locants(plain, _accepts("never")) == plain
    assert composite_locants("(14-2H)-" + HEPTAPHANE, _accepts("never")) == "(14-2H)-" + HEPTAPHANE
    assert composite_locants("(21-2H)-" + HEXAPHANE, _accepts("2^1")) == "(21-2H)-" + HEXAPHANE   # 2 is no superatom


def test_composite_locants_of_a_group_follow_the_primary_locants():
    """ (:15115): the primary locants first, so ``1^3`` comes before ``2``."""
    got = composite_locants("(2,13-2H2)-" + HEXAPHANE, _accepts("1^3"))
    assert got == "(1^3,2-2H2)-" + HEXAPHANE
    got = composite_locants("(12,13,15,16-2H4)-" + HEXAPHANE, lambda c: True)
    assert got == "(1^2,1^3,1^5,1^6-2H4)-" + HEXAPHANE


def test_every_group_of_a_mixed_descriptor_is_spelt_on_its_own():
    got = composite_locants("(2-13C,13-2H)-" + HEXAPHANE, _accepts("1^3-2H"))
    assert got == "(2-13C,1^3-2H)-" + HEXAPHANE


def test_composite_locant_is_an_integer_that_is_written_and_ordered_as_a_composite_locant():
    """ (:15115): the primary locant first, then the superscript, a letter after its number."""
    one_four, one_five, seven_two, four_a = (CompositeLocant(1, 4), CompositeLocant(1, 5), CompositeLocant(7, 2),
                                             CompositeLocant(1, 4, "a"))
    assert [str(x) for x in (one_four, four_a, seven_two)] == ["1^4", "1^4a", "7^2"]
    assert f"{one_four}" == "1^4" and ",".join(map(str, (one_four, one_five))) == "1^4,1^5"
    assert sorted([seven_two, one_five, one_four, four_a]) == [one_four, four_a, one_five, seven_two]
    assert one_four < one_five < seven_two and isinstance(one_four, int) and one_four >= 1
    assert composite_sort_key("1^4") < composite_sort_key("1^4a") < composite_sort_key("1^5") < composite_sort_key("7^2")
    assert composite_sort_key("12") is None and iso._locant_sort_key("1^2") < iso._locant_sort_key("2")


def test_the_search_is_offered_the_composite_locants_of_the_amplificants_lowest_first():
    """ for each superatom locant of the name, the amplificant locants 1... the atoms of the ring system."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("C1CCCCCc2ccc(cc2)CCCCCc2cc([2H])c1cc2")
    tokens = [str(t) for t in composite_tokens("1,7(1,4)-dibenzenacyclotridecaphane", mol)]
    assert tokens == [f"{p}^{q}" for p in (1, 7) for q in range(1, 7)]
    assert composite_tokens("(2,2-2H2)-1,3-diphenylpropane", mol) == []
    naphthalene = Chem.MolFromSmiles("c1ccc2cc(ccc2c1)CCCC")
    assert str(composite_tokens("1(2,6)-naphthalenacyclopentaphane", naphthalene)[-1]) == "1^10a"


def test_composites_are_asked_where_opsin_reads_them_and_not_beyond(monkeypatch):
    """A place that reads no first atom of the first superatom is dead for every composite locant (one read), and the
    amplificant locants of a superatom stop at the first one OPSIN does not read."""
    from orthonym.rules import isotope_search as search
    reads = []

    def parse(name):
        reads.append(name)
        # OPSIN reads a descriptor only in front of the name, and the amplificants of this stand-in have four atoms
        return "C" if name.startswith("(") and int(name[name.index("^") + 1]) <= 4 else None
    monkeypatch.setattr(iso, "_opsin_parse", parse)
    search._COMPOSITE_READS.clear()
    skel = "1,7(1,4)-dibenzenacyclotridecaphane"
    reach = search.LocantReach(skel, 13, lambda locant: f"({locant}-2H1)")
    assert not reach.beyond(0, CompositeLocant(1, 4)) and reach.beyond(0, CompositeLocant(1, 5))
    assert not reach.beyond(0, CompositeLocant(7, 2)) and reach.beyond(0, CompositeLocant(7, 6))
    del reads[:]
    assert all(reach.beyond(5, CompositeLocant(p, q)) for p in (1, 7) for q in range(1, 7))
    assert len(reads) == 1                       # the first atom of the first superatom: no place for any composite
    del reads[:]
    assert all(reach.beyond(0, CompositeLocant(p, q)) is (q > 4) for p in (1, 7) for q in range(1, 7))
    assert reads == []                           # everything was asked before and is remembered
    assert search.LocantReach.number_of(CompositeLocant(1, 4)) is None and search.LocantReach.number_of(14) == 14
    search._COMPOSITE_READS.clear()


def test_node_count_is_read_from_the_skeleton_word():
    """: the number of nodes, superatoms included, and 'phane'."""
    assert skeleton_node_count("1,7(1,4)-dibenzenacyclotridecaphane") == 13
    assert skeleton_node_count(HEPTAPHANE) == 7
    assert skeleton_node_count(HEXAPHANE) == 6
    assert skeleton_node_count("1,15(1),3,5,7,9,11,13(1,4)-octabenzenapentadecaphane") == 15
    assert skeleton_node_count("2,4,6,8,10-pentaoxa-1,11(1),3,5,7,9(1,4)-hexabenzenaundecaphan-1^2-ol") == 11
    assert skeleton_node_count("1,1'-biphenyl") is None


def _recorded(name, original_smiles, reads_back):
    """The name ``decorated_with_composite_locants`` returns and whether it recorded that name below the PIN."""
    from rdkit import Chem
    token = prov._NON_PIN_FRAGMENTS.set(())
    real = iso._isotope_round_trips
    iso._isotope_round_trips = lambda candidate, original, **k: reads_back(candidate)
    try:
        got = decorated_with_composite_locants(name, Chem.MolFromSmiles(original_smiles))
        return got, any(f in got for f in prov._NON_PIN_FRAGMENTS.get())
    finally:
        iso._isotope_round_trips = real
        prov._NON_PIN_FRAGMENTS.reset(token)


def test_a_respelling_in_a_skeleton_where_an_integer_can_be_a_node_is_labelled_below_the_pin():
    """From 11 nodes on, ``11``... ``19`` can be nodes, so a search that wrote integers could not reach the composite
    locant of that integer and the set it found is not proven the lowest: the name is recorded below the PIN."""
    smiles = "C1CCCCCc2ccc(cc2)CCCCCc2cc([2H])c1cc2"
    got, below = _recorded("(15-2H)-1,7(1,4)-dibenzenacyclotridecaphane", smiles, _accepts("1^5"))
    assert got == "(1^5-2H)-1,7(1,4)-dibenzenacyclotridecaphane" and below
    got, below = _recorded("(15-2H)-1,5(1,4)-dibenzenacyclohexaphane", smiles, _accepts("1^5"))
    assert got == "(1^5-2H)-1,5(1,4)-dibenzenacyclohexaphane" and not below      # six nodes: no integer is a node
    got, below = _recorded("(1^5-2H)-1,7(1,4)-dibenzenacyclotridecaphane", smiles, lambda c: True)
    assert not below                                                          # spelt by the search: nothing to prove


def test_a_name_without_a_phane_parent_is_left_alone():
    name = "(13-2H)-1,1'-biphenyl"
    assert composite_locants(name, lambda c: pytest.fail("no read-back for a name that has no phane parent")) == name
    # a group that is not a nuclide ('3S': no sulfur of mass 3) is no isotope descriptor
    other = "(12-3S)-" + HEXAPHANE
    assert composite_locants(other, lambda c: pytest.fail("not an isotope descriptor")) == other


# --- the names (OPSIN main reads phane names; the release 2.9.0 reads none) -----------------------------------

needs_phane = pytest.mark.skipif(not caps.phane_enabled(), reason="needs an OPSIN that reads phane names (OPSIN main)")

NAMES = [
    # linear phane, atom 4 of the terminal benzene (the sample of the review)
    ("[2H]c1ccc(cc1)Oc1ccc(cc1)Oc1ccc(cc1)Oc1ccc(cc1)", "(1^4-2H)-" + HEPTAPHANE),
    # cyclophane, atom 2 of the benzene at superatom 1
    ("[2H]c1cc2ccc1CCc1ccc(cc1)CC2", "(1^2-2H)-" + HEXAPHANE),
    # four atoms of one amplificant
    ("[2H]c1c([2H])c2c([2H])c([2H])c1CCc1ccc(cc1)CC2", "(1^2,1^3,1^5,1^6-2H4)-" + HEXAPHANE),
    # a node of the simplified skeleton and an amplificant atom, two nuclides
    ("[2H]c1cc2ccc1CCc1ccc(cc1)C[13CH2]2", "(2-13C,1^3-2H)-" + HEXAPHANE),
    # atom 6 of the naphthalene amplificant
    ("[2H]c1ccc2cc(ccc2c1)Oc1ccc(cc1)Oc1ccc(cc1)Oc1ccc(cc1)",
     "(1^6-2H)-2,4,6-trioxa-1(2)-naphthalena-3,5(1,4),7(1)-tribenzenaheptaphane"),
    # skeleton of 13 nodes: 12 and 13 are nodes, so the integer search skipped 1^2 and 1^3, lowest set)
    ("C1CCCCCc2ccc(cc2)CCCCCc2c([2H])cc1cc2", "(1^2-2H)-1,7(1,4)-dibenzenacyclotridecaphane"),
    ("C1CCCCCc2ccc(cc2)CCCCCc2cc([2H])c1cc2", "(1^3-2H)-1,7(1,4)-dibenzenacyclotridecaphane"),
    ("[2H]C1CCCCCc2ccc(cc2)CCCCCc2c([2H])cc1cc2", "(1^2,13-2H2)-1,7(1,4)-dibenzenacyclotridecaphane"),
    # atoms of the second amplificant, which the integer search could not reach (4^3 is 43 of 16 atoms)
    ("[2H]c1cc2ccc1CCc1cc([2H])c(cc1)CC2", "(1^2,4^3-2H2)-" + HEXAPHANE),
    # an atom of the first amplificant of a skeleton of 15 nodes, where 14 is a node
    ("[2H]c1ccc(cc1)Cc1ccc(cc1)Cc1ccc(cc1)Cc1ccc(cc1)Cc1ccc(cc1)Cc1ccc(cc1)Cc1ccc(cc1)Cc1ccccc1",
     "(1^4-2H)-1,15(1),3,5,7,9,11,13(1,4)-octabenzenapentadecaphane"),
    # the substituent path already spelt the composite locant (control)
    ("[13CH3]c1ccc(cc1)Oc1ccc(cc1)Oc1ccc(cc1)Oc1ccc(cc1)", "1^4-(13C)methyl-" + HEPTAPHANE),
    # a node of the simplified skeleton keeps its integer locant (control)
    ("[2H]C1(c2ccc(cc2)CCc2ccc1cc2)[2H]", "(2,2-2H2)-1,3(1,4)-dibenzenacyclopentaphane"),
]


@needs_phane
@pytest.mark.parametrize("smiles,expected", NAMES)
def test_phane_isotope_descriptor_cites_the_composite_locant(smiles, expected):
    r = Orthonym().name_tiered(smiles)
    assert r.get("name") == expected
    assert r.get("tier") == "pin_verified"
