"""SL lane G, item G1: identical prefixes of one scope are one multiplied prefix.

 "MULTIPLICATIVE PREFIXES 'DI', 'TRI', ETC. vs. 'BIS', 'TRIS', ETC." (the Blue Book),
 (:7027): "Multiplicative prefixes... are the principal method for describing a
multiplicity of identical features of a structure in chemical names." The group takes the
locants of all its members, the italic N first: "Lowest set of locants" (:3189),
"Italic capital and lower-case letter locants are lower than Greek letter locants, which, in
turn, are lower than numerals" (:3195). (:3448): "multiplicative prefixes, if
necessary, are then inserted and do not alter the alphabetical order already established."
Blue Book names: 'N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)' (:32879),
'N,N,2-trimethyl-3-{...}propanamide (PIN)' (:21624), 'N,N,N,1-tetramethylquinolin-1-ium-3-
aminium (PIN)' (:42213).

The module reads a rendered prefix into its parts and uses it only when the engine's own writer
of a multiplied component writes the same text from them; these tests pin that reading, the
grouping, the aniline entries and the audit of a prefix block put in front of an amide name.
"""
import pytest

from orthonym.assembly import identical_prefixes as ip
from orthonym.assembly.composition_primitives import identical_prefixes_cited_apart

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------ reading a rendered prefix

@pytest.mark.parametrize("text,locants,name,marked", [
    ("2-methyl", ["2"], "methyl", "methyl"),
    ("N-methyl", ["N"], "methyl", "methyl"),
    ("N,N-dimethyl", ["N", "N"], "methyl", "methyl"),
    ("3,3-dimethyl", ["3", "3"], "methyl", "methyl"),
    ("2,3-dihydroxy", ["2", "3"], "hydroxy", "hydroxy"),
    ("2-(4-methoxyphenyl)", ["2"], "4-methoxyphenyl", "(4-methoxyphenyl)"),
    ("N,2-bis(4-methoxyphenyl)", ["N", "2"], "4-methoxyphenyl", "(4-methoxyphenyl)"),
    ("N,N-di(propan-2-yl)", ["N", "N"], "propan-2-yl", "(propan-2-yl)"),
    ("2,3-di-tert-butyl", ["2", "3"], "tert-butyl", "tert-butyl"),
    ("2-{(2R)-2-amino}", ["2"], "(2R)-2-amino", "{(2R)-2-amino}"),
    ("N1,N4-dimethyl", ["N1", "N4"], "methyl", "methyl"),
    ("2',3-dichloro", ["2'", "3"], "chloro", "chloro"),
])
def test_a_rendered_prefix_is_read_into_its_parts(text, locants, name, marked):
    unit = ip.read_prefix(text)
    assert unit is not None, text
    assert (unit.locants, unit.name, unit.marked) == (locants, name, marked)


@pytest.mark.parametrize("text", [
    "methyl",                       # no locant: a mononuclear parent, left to its producer
    "2,3-methyl",                   # two locants, no multiplier: not what the writer writes
    "2,3,4-dimethyl",               # three locants, 'di': the multiplier does not agree
    "(2R)-2-methyl",                # a stereodescriptor is no locant
    "",
])
def test_a_text_the_writer_would_not_write_is_not_read(text):
    assert ip.read_prefix(text) is None


# ------------------------------------------------------------------ grouping rendered prefixes

@pytest.mark.parametrize("prefixes,expected", [
    # 2-(4-methoxyphenyl)-2-methyl-N-methylpropanamide ->: methoxyphenyl before methyl
    (["2-(4-methoxyphenyl)", "2-methyl", "N-methyl"], ["2-(4-methoxyphenyl)", "N,2-dimethyl"]),
    (["3-hydrazinyl", "2-methyl", "N,N-dimethyl"], ["3-hydrazinyl", "N,N,2-trimethyl"]),
    (["2-methyl", "3-hydroxy", "N,N-dimethyl", "3-(4-methylphenyl)"],
     ["3-hydroxy", "N,N,2-trimethyl", "3-(4-methylphenyl)"]),
    (["2-hydroxy", "3-hydroxy", "2-(1-hydroxytridecyl)", "4-methoxy"],
     ["2,3-dihydroxy", "2-(1-hydroxytridecyl)", "4-methoxy"]),
    (["2,3,4-trihydroxy", "5-hydroxy", "5-[(phosphonooxy)methyl]"],
     ["2,3,4,5-tetrahydroxy", "5-[(phosphonooxy)methyl]"]),
    (["N-(4-methoxyphenyl)", "2-(4-methoxyphenyl)"], ["N,2-bis(4-methoxyphenyl)"]),
    (["N-ethyl", "N-methyl", "2-methyl"], ["N-ethyl", "N,2-dimethyl"]),
])
def test_identical_prefixes_are_one_group_in_the_order_of_p_14_5(prefixes, expected):
    assert ip.group_identical_prefixes(prefixes) == expected


@pytest.mark.parametrize("prefixes", [
    ["2-methyl", "3-ethyl"],                    # nothing shared
    ["N-methyl"],
    ["2-methyl", "2-methyl"],                   # one group an earlier step listed twice
    ["methyl", "N-methyl"],                     # a prefix without a locant is left to its producer
    ["2-[(1R)-1-hydroxyethyl]", "N-[(1S)-1-hydroxyethyl]"],     # two configurations: two substituents
    ["2-tert-butyl", "N-butyl"],                # 'tert-butyl' is not 'butyl'
])
def test_what_is_not_the_same_substituent_is_left_as_it_was(prefixes):
    assert ip.group_identical_prefixes(prefixes) == sorted(
        prefixes, key=_key)


def _key(p):
    from orthonym.assembly.naming_utils import prefix_citation_sort_key
    return prefix_citation_sort_key(p, parent_locants=True)


def test_a_group_keeps_the_pairs_of_its_members():
    """The set of (locant, substituent) pairs of the input is the set of the output."""
    prefixes = ["2-(4-methoxyphenyl)", "2-methyl", "N-methyl", "4-chloro", "N,N-dimethyl"]
    out = ip.group_identical_prefixes(prefixes)
    assert ip._pairs(prefixes) == ip._pairs(out)


def test_prefixes_cited_apart_stay_apart():
    """ (the Blue Book): groups that differ in an isotopic modification are
    cited separately; ``identical_prefixes_cited_apart`` asks the producers not to combine."""
    with identical_prefixes_cited_apart():
        assert ip.group_identical_prefixes(["2-methyl", "N-methyl"], sort=False) == [
            "2-methyl", "N-methyl"]


# ------------------------------------------------------------------ the aniline entries

def test_the_substituents_of_the_ring_and_of_the_nitrogen_are_one_group():
    ring, n, entries = ip.shared_prefix_entries({"methyl": [4], "chloro": [2]}, ["methyl"])
    assert ring == {"chloro": [2]} and n == []
    assert [text for _key_, text in entries] == ["N,4-dimethyl"]


def test_two_nitrogen_substituents_and_a_ring_substituent():
    ring, n, entries = ip.shared_prefix_entries({"ethyl": [2, 6], "methyl": [4]}, ["ethyl", "methyl"])
    assert ring == {} and n == []
    assert sorted(text for _key_, text in entries) == ["N,2,6-triethyl", "N,4-dimethyl"]


def test_nothing_shared_changes_nothing():
    ring, n, entries = ip.shared_prefix_entries({"methyl": [4]}, ["ethyl"])
    assert (ring, n, entries) == ({"methyl": [4]}, ["ethyl"], [])


# ------------------------------------------------------------------ an N block in front of an amide

@pytest.mark.parametrize("name,units,parent", [
    ("3,3-dimethylbutanamide", ["3,3-dimethyl"], "butanamide"),
    ("3,3-dimethyl-2-propanamidobutanamide", ["3,3-dimethyl", "2-propanamido"], "butanamide"),
    ("2-{(2R)-2-amino-4-methylpentanamido}-3,3-dimethylbutanamide",
     ["2-{(2R)-2-amino-4-methylpentanamido}", "3,3-dimethyl"], "butanamide"),
    ("2-(4-chlorophenyl)butanamide", ["2-(4-chlorophenyl)"], "butanamide"),
])
def test_the_prefixes_and_the_parent_of_a_chain_amide_are_read(name, units, parent):
    assert ip.split_prefix_zone(name) == (units, parent)


@pytest.mark.parametrize("name", [
    "butanamide",                           # no prefix
    "2-methylbenzamide",                    # not a chain amide
    "3,3-dimethylbutanoic acid",
    "2-methyl-3-butanamide-hexanamide",
])
def test_any_other_shape_is_not_read(name):
    assert ip.split_prefix_zone(name) is None


@pytest.mark.parametrize("block,name,merged", [
    ("N-methyl", "3,3-dimethyl-2-propanamidobutanamide", "N,3,3-trimethyl-2-propanamidobutanamide"),
    ("N-methyl", "2-{(2R)-2-amino-4-methylpentanamido}-3,3-dimethylbutanamide",
     "2-{(2R)-2-amino-4-methylpentanamido}-N,3,3-trimethylbutanamide"),
    ("N,N-dimethyl", "2-methylpropanamide", "N,N,2-trimethylpropanamide"),
    ("N-ethyl-N-methyl", "2-methylpropanamide", "N-ethyl-N,2-dimethylpropanamide"),
    ("N-(4-methoxyphenyl)", "2-(4-methoxyphenyl)butanamide", "N,2-bis(4-methoxyphenyl)butanamide"),
])
def test_an_n_block_and_the_prefixes_of_the_amide_name_are_cited_together(block, name, merged):
    assert ip.merge_n_block_into_name(block, name) == merged


@pytest.mark.parametrize("block,name", [
    ("N-ethyl", "3,3-dimethyl-2-propanamidobutanamide"),        # nothing shared
    ("N-methyl", "2-methylbenzamide"),                          # not read: stays as it was
    ("N-(4-methoxyphenyl)", "2-(4-chlorophenyl)butanamide"),
])
def test_no_merge_when_nothing_is_shared_or_the_name_is_not_read(block, name):
    assert ip.merge_n_block_into_name(block, name) is None
