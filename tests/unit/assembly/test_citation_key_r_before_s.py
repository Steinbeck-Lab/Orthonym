"""R is cited before S whatever locant each is attached at: the citation keys (no JVM).

``### **** Sequence Rule **4a**`` (``the Blue Book``), its example (``:45465``)::

    (2R,3s,4S,6R)-2,6-dichloro-5-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]heptan-4-ol (PIN)

cites the (1R)-chloroethyl first, at the HIGHER locant 5. ``****`` (``:22603``, under ``### ****
CRITERIA RELATED ONLY TO CONFIGURATION``): "When names based on alphanumerical order and isotopic descriptors are
the same, further choice depends on the alphabetic order of the stereochemical descriptors 'R' and 'S'"
(``1-[(1R)-1-bromoethyl]-1-[(1S)-1-bromoethyl]cyclopentane (PIN)``, ``:22609``; ``4-{3,4-bis[(1R)-...]phenoxy}-
1,2-bis[(1S)-...]benzene (PIN)``, ``:22597``). ``****`` (``:3517``, under ``### **** ALPHANUMERICAL
ORDER``) compares the locants INSIDE the prefix names, and "the fact that 'N' is lower than '4' is irrelevant"
(``:3527``): the locant a prefix is attached at is no tier of the order of citation.

The book gives no order of citation for a Z and an E prefix, so the E/Z order of every key here is what it was.
"""
import pytest

from orthonym.assembly.naming_utils import (alpha_sort_key, alpha_sort_key_rs, alpha_sort_key_total,
                                            alpha_sort_key_with_locants, rendered_prefix_citation_key,
                                            rs_descriptor_rank_key)

pytestmark = pytest.mark.unit

R, S = "(2R)-butan-2-yl", "(2S)-butan-2-yl"
Z, E = "(1Z)-prop-1-en-1-yl", "(1E)-prop-1-en-1-yl"


def test_rs_rank_reads_only_the_upper_case_r_and_s_in_order_of_appearance():
    assert rs_descriptor_rank_key("(1R)-1-chloroethyl") == (0,)
    assert rs_descriptor_rank_key("(1S)-1-chloroethyl") == (1,)
    assert rs_descriptor_rank_key("(2R,3S)-3-chlorobutan-2-yl") < rs_descriptor_rank_key("(2S,3R)-3-chlorobutan-2-yl")
    assert rs_descriptor_rank_key("(2R,3S)-3-chlorobutan-2-yl") > rs_descriptor_rank_key("(2R,3R)-3-chlorobutan-2-yl")
    # no order of citation in the book for these: left to the callers' convention
    for other in (Z, E, "(1r,4r)-4-methylcyclohexyl", "cis-4-methylcyclohexyl", "methyl"):
        assert rs_descriptor_rank_key(other) == ()


@pytest.mark.parametrize("locants_of_r, locants_of_s", [((1,), (3,)), ((3,), (1,)), ((2,), (2,))])
def test_r_is_cited_first_whatever_the_locants_a_site_passes(locants_of_r, locants_of_s):
    """ (:45465): R first at the HIGHER locant; the locants come after the configuration."""
    rows = [(S, locants_of_s), (R, locants_of_r)]
    assert [n for n, _ in sorted(rows, key=lambda x: alpha_sort_key_with_locants(x[0], x[1]))] == [R, S]
    rows.reverse()
    assert [n for n, _ in sorted(rows, key=lambda x: alpha_sort_key_with_locants(x[0], x[1]))] == [R, S]


def test_the_locants_a_site_passes_decide_e_and_z_as_they_did():
    """ names R and S only: an E and a Z prefix are cited by the locants of the site, then Z before E."""
    rows = [(Z, (3,)), (E, (1,))]
    assert [n for n, _ in sorted(rows, key=lambda x: alpha_sort_key_with_locants(x[0], x[1]))] == [E, Z]
    rows = [(Z, (2,)), (E, (2,))]
    assert [n for n, _ in sorted(rows, key=lambda x: alpha_sort_key_with_locants(x[0], x[1]))] == [Z, E]


def test_a_name_cited_with_enclosing_marks_ties_with_the_same_name_cited_without():
    """The C-prefix of an amine is cited in marks, its N-prefix is not: orders them alike, R before S."""
    bare, marked = "(1S)-1-chloroethyl", "[(1R)-1-chloroethyl]"
    assert sorted([bare, marked], key=alpha_sort_key_total) == [marked, bare]
    assert sorted([marked, bare], key=alpha_sort_key_total) == [marked, bare]


def test_alpha_sort_key_rs_adds_r_before_s_and_nothing_else():
    assert sorted([S, R], key=alpha_sort_key_rs) == [R, S]
    assert sorted([R, S], key=alpha_sort_key_rs) == [R, S]
    # Z and E tie, as in alpha_sort_key: the order the site met them in stands
    assert sorted([E, Z], key=alpha_sort_key_rs) == [E, Z]
    assert sorted([Z, E], key=alpha_sort_key_rs) == [Z, E]
    # the letters of come first
    assert sorted(["(1S)-1-chloroethyl", "(2R)-butan-2-yl"], key=alpha_sort_key_rs)[0] == "(2R)-butan-2-yl"
    assert alpha_sort_key_rs("methyl")[0] == alpha_sort_key("methyl")


def test_a_rendered_prefix_is_cited_by_its_configuration_not_by_the_locant_it_is_attached_at():
    """``4-[(2R)-..]`` before ``2-[(2S)-..]``: (:45465), (:22603)."""
    rows = ["2-[(2S)-butan-2-yl]", "4-[(2R)-butan-2-yl]"]
    assert sorted(rows, key=rendered_prefix_citation_key) == ["4-[(2R)-butan-2-yl]", "2-[(2S)-butan-2-yl]"]
    rows = ["N-[(2S)-butan-2-yl]", "4-[(2R)-butan-2-yl]"]
    assert sorted(rows, key=rendered_prefix_citation_key) == ["4-[(2R)-butan-2-yl]", "N-[(2S)-butan-2-yl]"]
    rows = ["4-[(2S)-butan-2-yl]", "N-[(2R)-butan-2-yl]"]
    assert sorted(rows, key=rendered_prefix_citation_key) == ["N-[(2R)-butan-2-yl]", "4-[(2S)-butan-2-yl]"]


def test_a_rendered_prefix_is_cited_by_the_locants_inside_its_name_not_by_those_it_is_attached_at():
    """ (:3517): ``4-(2-methylbutyl)-N-(3-methylbutyl)aniline (PIN)`` -- "the fact that 'N' is lower than
    '4' is irrelevant" (:3527); ``6-(1-chloroethyl)-5-(2-chloroethyl)-1H-indole (PIN)`` (:3529) cites the higher
    locant first."""
    rows = ["2-(3-methylbutyl)", "4-(2-methylbutyl)"]
    assert sorted(rows, key=rendered_prefix_citation_key) == ["4-(2-methylbutyl)", "2-(3-methylbutyl)"]
    rows = ["N-(3-methylbutyl)", "4-(2-methylbutyl)"]
    assert sorted(rows, key=rendered_prefix_citation_key) == ["4-(2-methylbutyl)", "N-(3-methylbutyl)"]
    rows = ["5-(2-chloroethyl)", "6-(1-chloroethyl)"]
    assert sorted(rows, key=rendered_prefix_citation_key) == ["6-(1-chloroethyl)", "5-(2-chloroethyl)"]


def test_a_rendered_prefix_keeps_the_order_of_its_attachment_locants_when_the_book_orders_nothing():
    assert sorted(["4-[(2R)-butan-2-yl]", "2-[(2R)-butan-2-yl]"], key=rendered_prefix_citation_key) == [
        "2-[(2R)-butan-2-yl]", "4-[(2R)-butan-2-yl]"]
    # E and Z of one name: the attachment locants first, then Z before E
    assert sorted([f"3-[{E}]", f"1-[{Z}]"], key=rendered_prefix_citation_key) == [f"1-[{Z}]", f"3-[{E}]"]
    assert sorted([f"3-[{Z}]", f"1-[{E}]"], key=rendered_prefix_citation_key) == [f"1-[{E}]", f"3-[{Z}]"]


def test_the_locants_inside_the_names_come_before_the_configuration():
    """ (:3517): 1-chloropropyl before 2-chloropropyl; (:22603) is for names that are the same."""
    s1, r2 = "(1S)-1-chloropropyl", "(2R)-2-chloropropyl"
    for key in (alpha_sort_key_with_locants, alpha_sort_key_total, alpha_sort_key_rs):
        assert sorted([r2, s1], key=key) == [s1, r2]
        assert sorted([s1, r2], key=key) == [s1, r2]
    rows = ["3-[(2R)-2-chloropropyl]", "5-[(1S)-1-chloropropyl]"]
    assert sorted(rows, key=rendered_prefix_citation_key) == ["5-[(1S)-1-chloropropyl]", "3-[(2R)-2-chloropropyl]"]


def test_a_name_without_r_or_s_is_cited_before_one_with_it_whatever_the_locants_ASSUMED():
    """The book has no example of the pair; the empty rank tuple sorts first and the tier is ahead of the locants."""
    rows = ["2-[(2R)-butan-2-yl]", "4-(butan-2-yl)"]
    assert sorted(rows, key=rendered_prefix_citation_key) == ["4-(butan-2-yl)", "2-[(2R)-butan-2-yl]"]


def test_the_citation_key_of_a_c_entry_keeps_the_marker_and_takes_the_locants_apart_from_the_text():
    """A site that sorts N-entries and C-entries together (the polyamine) passes the first element of the locants
    tier and, for a prefix whose text has no attachment locant in front, its locants."""
    n = alpha_sort_key_with_locants("(2S)-butan-2-yl", (0, ("N", 1)))
    c_baked = rendered_prefix_citation_key("3-[(2R)-butan-2-yl]", marker=1)
    c_apart = rendered_prefix_citation_key("[(2R)-butan-2-yl]", marker=1, locants=(3,))
    assert c_baked == c_apart
    assert sorted([n, c_baked]) == [c_baked, n]          # R before S, ahead of the N before C of the callers
    n_r = alpha_sort_key_with_locants("(2R)-butan-2-yl", (0, ("N", 1)))
    assert sorted([c_baked, n_r]) == [n_r, c_baked]      # the same name: the N-entry first


def test_the_general_engine_cites_r_before_s_and_leaves_e_and_z_in_the_order_met():
    from orthonym.assembly.general_engine import _citation_key
    assert sorted(["[(1S)-1-chloroethyl]amino", "[(1R)-1-chloroethyl]amino"], key=_citation_key) == [
        "[(1R)-1-chloroethyl]amino", "[(1S)-1-chloroethyl]amino"]
    assert sorted([E, Z], key=_citation_key) == [E, Z]
    assert sorted([Z, E], key=_citation_key) == [Z, E]
    # the locants inside the names come before R/S,:3517)
    assert sorted(["[(2R)-2-chloropropyl]amino", "[(1S)-1-chloropropyl]amino"], key=_citation_key) == [
        "[(1S)-1-chloropropyl]amino", "[(2R)-2-chloropropyl]amino"]
    assert sorted(["[(2R)-2-chloropropyl]amino", "[(1R)-1-chloropropyl]amino"], key=_citation_key) == [
        "[(1R)-1-chloropropyl]amino", "[(2R)-2-chloropropyl]amino"]


def test_the_name_tree_cites_its_prefixes_as_the_assembler_does():
    """The serializer's order is kept only when it is the assembler's byte for byte
    (``composer._serializer_flip_or_name``): a node whose text carries its attachment locant is cited by its
    configuration, R before S, and by the locants inside its name, not by the locant it is attached at."""
    from orthonym.assembly.name_tree import NameTreeNode, _alphabetize_prefixes
    nodes = (NameTreeNode(parent_stem="2-[(2S)-butan-2-yl]"), NameTreeNode(parent_stem="4-[(2R)-butan-2-yl]"))
    assert [n.parent_stem for n in _alphabetize_prefixes(nodes)] == ["4-[(2R)-butan-2-yl]", "2-[(2S)-butan-2-yl]"]
    nodes = (NameTreeNode(parent_stem="2-(3-methylbutyl)"), NameTreeNode(parent_stem="4-(2-methylbutyl)"))
    assert [n.parent_stem for n in _alphabetize_prefixes(nodes)] == ["4-(2-methylbutyl)", "2-(3-methylbutyl)"]
