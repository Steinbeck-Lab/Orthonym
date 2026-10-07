"""The lexical readers behind the spelling checks (``validation/spelling/lexer.py``, ``units.py``).

 (the Blue Book) "the presence of square brackets and/or parentheses that are an
integral part of the name of a parent structure does not affect the nesting order":
/.2 (:7465,:7469) added indicated hydrogen and fusion, spiro, ring-assembly and von
Baeyer descriptors are ignored; /.4 (:7478,:7501) compound locants, stereo and
isotope descriptors count. (:3477) "The name of a prefix for a substituent is considered
to begin with the first letter of its complete name"; (:2869) a unit is "the parent
structure or... a unit of structure as defined by its appropriate enclosing marks".
"""
import pytest

from orthonym.validation.spelling.lexer import (
    UnbalancedMarksError,
    enclosures,
    letters,
    normalise,
)
from orthonym.validation.spelling.units import parse_name, split_simple


def test_normalise_writes_primes_and_plain_digits():
    assert normalise("1,1′-biphenyl") == "1,1'-biphenyl"
    assert normalise("N¹,N³-dimethyl") == "N1,N3-dimethyl"


@pytest.mark.parametrize("name,content,kind", [
    ("imidazo[1,2-a]pyridine", "1,2-a", "fusion"),
    ("bicyclo[2.2.1]heptane", "2.2.1", "vonbaeyer"),
    ("spiro[4.5]decane", "4.5", "vonbaeyer"),
    ("3-([1,1'-biphenyl]-4-yl)pyridine", "1,1'-biphenyl", "assembly"),
    ("(2R)-butan-2-ol", "2R", "stereo"),
    ("3,4-dihydronaphthalen-1(2H)-one", "2H", "added_ih"),
    ("1-(amino[14C]methyl)cyclopentan-1-ol", "14C", "isotope_br"),
    ("1-(propan-2-yl)-4-methylbenzene", "propan-2-yl", "subst"),
    ("[1,2,4]triazolo[1,5-a]pyridine", "1,2,4", "component_loc"),
])
def test_each_mark_is_classified(name, content, kind):
    found = {e.content: e.kind for e in enclosures(normalise(name))}
    assert found[content] == kind, found


def test_counted_marks_are_those_of_p_16_5_4_1_3():
    kinds = {e.content: e.counted for e in enclosures("(2R)-2-[(1H-indol-3-yl)methyl]-3,4-dihydronaphthalen-1(2H)-one")}
    assert kinds["2R"] and kinds["(1H-indol-3-yl)methyl"] and kinds["1H-indol-3-yl"]
    assert not kinds["2H"]


@pytest.mark.parametrize("name", ["benzo[8]annulene", "pyrrolo[1,2-a]pyrazine", "bicyclo[3.2.1]octane",
                                  "spiro[4.5]decane", "naphthalen-1(2H)-one", "[1,1'-biphenyl]-4-ol"])
def test_integral_marks_are_not_counted(name):
    assert not any(e.counted for e in enclosures(normalise(name)))


def test_marks_that_do_not_pair_raise():
    with pytest.raises(UnbalancedMarksError):
        enclosures("2-[(propan-2-yl)oxy)ethan-1-ol")


@pytest.mark.parametrize("prefix,key", [
    ("(4-chlorophenyl)", "chlorophenyl"),          # complete name
    ("(dimethylamino)", "dimethylamino"),          #:3491 'dimethylpentyl' begins with 'd'
    ("tert-butyl", "butyl"),                       # italic prefixes not alphabetized
    ("[(2S)-butan-2-yl]", "butanyl"),              #:3446 stereodescriptors not alphabetized
    ("(2H-1-benzopyran-3-yl)", "benzopyranyl"),    # indicated hydrogen and locants dropped
])
def test_the_alphanumerical_key_is_the_complete_prefix_name(prefix, key):
    assert letters(prefix) == key


def test_a_unit_reads_into_prefixes_parent_and_endings():
    _s, _encs, units = parse_name("3,3-dibromo-3-cyclohexylpropanoic acid")
    top = units[0]
    assert [(p.locs, p.key, p.mult) for p in top.prefixes] == [(("3", "3"), "bromo", 2),
                                                               (("3",), "cyclohexyl", 1)]
    assert top.parent == "propanoic"


def test_hydro_prefixes_indicated_hydrogen_and_suffix_locants():
    _s, _encs, units = parse_name("5-methyl-1,2,3,4-tetrahydronaphthalen-1-ol")
    top = units[0]
    assert top.hydro == [(("1", "2", "3", "4"), "tetrahydro")]
    assert top.parent == "naphthalen" and top.endings == [(("1",), "ol", None)]
    _s, _encs, units = parse_name("2H-1-benzopyran-2-one")
    assert units[0].ih == [("2H",)]


def test_each_enclosure_is_its_own_unit():
    _s, _encs, units = parse_name("4-[(4-chlorophenyl)methyl]morpholine")
    inner = {u.unit.encl.content: u for u in units if u.unit.encl is not None}
    assert inner["4-chlorophenyl"].parent == "phenyl"
    assert [p.key for p in inner["4-chlorophenyl"].prefixes] == ["chloro"]
    assert inner["(4-chlorophenyl)methyl"].parent == "methyl"


def test_a_run_of_simple_prefixes_splits_by_the_lexicon_and_stops_at_an_unknown_word():
    assert split_simple("pentyloxycarbonyl") == ([(None, "pentyl"), (None, "oxy"), (None, "carbonyl")], "")
    assert split_simple("dichloromethyl") == ([("di", "chloro"), (None, "methyl")], "")
    pre, rest = split_simple("oxolan")
    assert pre == [] and rest == "oxolan"


def test_a_numbering_model_reads_the_locant_sets_of_a_unit():
    from orthonym.validation.spelling.models import read_units, ring_or_chain_model
    _s, _encs, units = read_units("6-(4-ethylphenoxy)cyclohexane-1-carbonitrile")
    kind, n, perms, feats = ring_or_chain_model(units[0], False)
    assert (kind, n, len(perms)) == ("ring", 6, 12)           # the dihedral numberings of the ring
    assert feats["suffix"] == [1] and [locs for _k, locs in feats["prefix"]] == [[6]]
    _s, _encs, units = read_units("2-methylpent-1-en-4-yn-3-ol")
    kind, n, perms, feats = ring_or_chain_model(units[0], False)
    assert (kind, n, len(perms)) == ("chain", 5, 2)
    assert feats["unsat_b"] == [1, 4] and feats["double_b"] == [1] and feats["suffix"] == [3]


def test_the_model_keeps_the_prefixes_of_its_locant_sets_in_their_order():
    """``feats['prefix_src']`` is the prefix of each ``feats['prefix']`` entry, in the same order,
    so a check that pairs the two (the principal-chain check of / reads the
    model's own selection; an element-locant prefix ('N-...') is in neither list."""
    from orthonym.validation.spelling.models import read_units, ring_or_chain_model
    _s, _encs, units = read_units("N,N-diethyl-3-methyl-2-propylpentanamide")
    kind, n, _perms, feats = ring_or_chain_model(units[0], False)
    assert (kind, n) == ("chain", 5)
    assert [(pr.key, [int(x) for x in pr.locs]) for pr in feats["prefix_src"]] == feats["prefix"]
    assert [pr.raw for pr in feats["prefix_src"]] == ["methyl", "propyl"]
    assert feats["letter"] == [("N", None), ("N", None)]


def test_a_parent_outside_the_models_is_not_read():
    from orthonym.validation.spelling.models import read_units, ring_or_chain_model
    _s, _encs, units = read_units("5-methylbicyclo[2.2.1]heptan-2-ol")
    assert ring_or_chain_model(units[0], False) is None


@pytest.mark.parametrize("name", ["β-D-glucopyranose", "L-alanyl-L-alanine", "cholest-5-en-3β-ol",
                                  "(η5-cyclopentadienyl)iron"])
def test_the_p10_and_coordination_marks_are_recognised(name):
    from orthonym.validation.spelling.models import P10
    assert P10.search(name)


@pytest.mark.parametrize("name,content", [
    # (:43718) the nuclide symbols in parentheses are "preceded by any necessary
    # locant(s), letters, and/or numerals"; an element locant: '(N-2H1)acetamide (PIN)' (:43828)
    ("1-phenyl(N-2H1)methanamine", "N-2H1"),
    ("(N,N-2H2)ethanamine", "N,N-2H2"),
    ("(2-13C)propane", "2-13C"),
])
def test_an_isotope_descriptor_is_not_a_substituent_prefix(name, content):
    found = {e.content: e.kind for e in enclosures(normalise(name))}
    assert found[content] == "isotope", found


@pytest.mark.parametrize("word,split", [
    # radical parents and the ion endings: the rest that begins with an ending is the
    # parent's own name ('methoxyl (PIN)' the Blue Book, 'bis(chloromethyl)aminoxyl (PIN)'
    #:40703, 'methoxyboranylidene':36271, 'oxidoazaniumylidyne':43386, 'phenyldisulfanylium'
    #:41668)
    ("pentyloxyl", ([(None, "pentyl")], "oxyl")),
    ("methoxyl", ([], "methoxyl")),
    ("aminoxyl", ([], "aminoxyl")),
    ("methylboranylidene", ([(None, "methyl")], "boranylidene")),
    ("oxidoazaniumylidyne", ([(None, "oxido")], "azaniumylidyne")),
    ("phenyldisulfanylium", ([(None, "phenyl")], "disulfanylium")),
    # a Hantzsch-Widman name whose replacement prefix elides its 'a':
    # 'ioda' + 'ocine', 'oxa' + 'ocine', 'oxa' + 'onine', 'broma' + 'olane'
    ("iodocine", ([], "iodocine")),
    ("oxocin", ([], "oxocin")),
    ("oxonine", ([], "oxonine")),
    ("bromolane", ([], "bromolane")),
    # and the ordinary splits stay
    ("chloromethyl", ([(None, "chloro"), (None, "methyl")], "")),
    ("benzylsilylidene", ([(None, "benzyl"), (None, "silylidene")], "")),   #:40490
])
def test_a_rest_that_begins_with_an_ending_or_a_ring_stem_is_the_parent(word, split):
    assert split_simple(word) == split


def test_a_compound_locant_is_never_compared():
    """ (the priority of compound locants) does not apply to the reader: a unit whose
    locants include a compound locant ('1(10)') has no numbering model, so never compares it."""
    from orthonym.validation.spelling.models import read_units, ring_or_chain_model
    found = {e.content: e.kind for e in enclosures(normalise("5-methylcyclodeca-1(10),2-diene"))}
    assert found["10"] == "compound_locant"
    _s, _encs, units = read_units("5-methylcyclodeca-1(10),2-diene")
    assert ring_or_chain_model(units[0], False) is None


@pytest.mark.parametrize("name", ["4,4-dimethylpiperazin-4-ium-1-ylium",       # the Blue Book
                                  "anthracen-9(10H)-yl-10-ylidene"])            #:24711
def test_a_unit_with_two_kinds_of_suffix_has_no_numbering_model(name):
    from orthonym.validation.spelling.models import read_units, ring_or_chain_model
    _s, _encs, units = read_units(name)
    assert ring_or_chain_model(units[0], False) is None


def test_the_replacement_prefixes_the_reader_knows_are_those_of_tables_1_5_and_2_4():
    """The reader keeps its own map of 'a' prefixes (a check must not inherit a defect of the
    writer's tables); it must equal Table 1.5 and Table 2.4, 'aluma',
    'indiga', the Blue Book) together, without 'carba', so the copies cannot drift apart."""
    from orthonym.data.hw_heteroatoms import HW_PREFIXES
    from orthonym.rules.ring_replacement import TABLE_1_5
    from orthonym.validation.spelling.units import REPL, REPL_ELEMENT
    tables = {spelling: el for el, (spelling, *_rest) in TABLE_1_5.items() if spelling != "carba"}
    tables.update({spelling: el for el, spelling in HW_PREFIXES.items()})
    assert REPL_ELEMENT == tables
    assert set(REPL) == set(REPL_ELEMENT)
