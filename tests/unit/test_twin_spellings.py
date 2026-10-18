"""The twin spellings of ``validation.twin_spellings`` (no JVM): the rewrite table, the names that
have no twin, and the composition of two rules."""
import pytest

from orthonym.validation.twin_spellings import TWIN_RULES, twin_of

pytestmark = pytest.mark.unit

PAIRS = [
    # (the Blue Book) preselected 'sulfano'; the PIN example '1,4-dihydro-1,4-
    # sulfanonaphthalene (PIN)' under CYCLIC ETHERS, SULFIDES, SELENIDES AND TELLURIDES (:28063)
    ("1,4-dihydro-1,4-sulfanonaphthalene", "1,4-dihydro-1,4-epithionaphthalene", "bridge_sulfano"),
    ("(1R,4S)-1,4-dihydro-1,4-sulfanonaphthalene", "(1R,4S)-1,4-dihydro-1,4-epithionaphthalene",
     "bridge_sulfano"),
    ("1,2,3,4-tetrahydro-1,4-sulfanonaphthalene", "1,2,3,4-tetrahydro-1,4-epithionaphthalene",
     "bridge_sulfano"),
    ("(1,4-dihydro-1,4-sulfanonaphthalen-1-yl)acetic acid",
     "(1,4-dihydro-1,4-epithionaphthalen-1-yl)acetic acid", "bridge_sulfano"),
    # the table row '–S-S– disulfano (preselected prefix) / epidithio' (:14105)
    ("9,10-dihydro-9,10-disulfanoanthracene", "9,10-dihydro-9,10-epidithioanthracene", "bridge_disulfano"),
    # the table row '–NH– azano (preselected prefix) / epimino' (:14110)
    ("1,4-dihydro-1,4-azanonaphthalene", "1,4-dihydro-1,4-epiminonaphthalene", "bridge_azano"),
    ("9-methyl-1,4-dihydro-1,4-azanonaphthalene", "9-methyl-1,4-dihydro-1,4-epiminonaphthalene",
     "bridge_azano"),
    # (:12658) interior atom '3a^1' for the former '9b'; (:14470,:14491),
    # (:23474,:23486), (:25319,:25350)
    ("1,3a^1,4,9-tetraazaphenalene", "1,9b,4,9-tetraazaphenalene", "interior_phenalene"),
    ("1,3a^1,4,9-tetraazaphenalene-3-carboxylic acid", "1,9b,4,9-tetraazaphenalene-3-carboxylic acid",
     "interior_phenalene"),
    ("1,3a^1,4-triazaphenalene", "1,9b,4-triazaphenalene", "interior_phenalene"),
    ("1,3a^1,4,9-tetraazaphenalen-3-amine", "1,9b,4,9-tetraazaphenalen-3-amine", "interior_phenalene"),
    ("3-methyl-1,3a^1,4,9-tetraazaphenalene", "3-methyl-1,9b,4,9-tetraazaphenalene", "interior_phenalene"),
    # (:43824) '(N-2H2)aniline (PIN)' (:43830)
    ("(N-2H2)aniline", "(N,N-2H2)aniline", "isotope_letter_once"),
    # Omission of locants (:44178,:44184): 'ethan(2H)ol (PIN) (as in ethanol)'
    ("ethan(2H)ol", "ethan(2H1)ol", "isotope_count_one"),
    # Letter and/or numeral locants (:44208,:44214): '(1-2H1)ethan-1-(2H)ol (PIN)'
    ("(1-2H1)ethan-1-(2H)ol", "(1-2H1)ethan-1-(2H1)ol", "isotope_count_one"),
    # ORDER OF NUCLIDE SYMBOLS (:43832), (:43834): 'methan(2H,18O)ol (PIN)'
    # (:43840); the 2H is followed by a comma
    ("methan(2H,18O)ol", "methan(2H1,18O)ol", "isotope_count_one"),
    ("ethane(2H)thiol", "ethane(2H1)thiol", "isotope_count_one"),
    ("propan-2-(2H)ol", "propan-2-(2H1)ol", "isotope_count_one"),
    ("ethan(3H)ol", "ethan(3H1)ol", "isotope_count_one"),
    # (:43800): '(1-14C)pentan(3H)oic acid (PIN)' (:43812), 'cyclohexane(2H)carboxylic acid (PIN)' (:43816)
    ("(1-14C)pentan(3H)oic acid", "(1-14C)pentan(3H1)oic acid", "isotope_count_one"),
    ("cyclohexane(2H)carboxylic acid", "cyclohexane(2H1)carboxylic acid", "isotope_count_one"),
    # a letter locant of a nuclide counted three times is cited three times
    ("(N-2H3)ethan-1-aminium", "(N,N,N-2H3)ethan-1-aminium", "isotope_letter_once"),
]


@pytest.mark.parametrize("pin,twin,rule", PAIRS)
def test_twin(pin, twin, rule):
    assert twin_of(pin) == (twin, (rule,))


@pytest.mark.parametrize("name", [
    "1H-3,10-dioxa-2a^1,5-ethanocycloocta[cd]pentalene",         # an interior locant on another parent
    "1,3a^1,4,9-tetraazaphenaleno[1,2-a]benzene",                # 'phenaleno' is an attached component
    "1H-phenalene", "1,4,9,9b-tetraazaphenalene",                # no superscript locant to rewrite
    "1,4-dihydro-1,4-selanonaphthalene",                         # 'episeleno' reads as a [SeH2] bridge: no twin
    "1,4-dihydro-1,4-episelenonaphthalene",                      # already the general spelling
    "1,4-dihydro-1,4-epithionaphthalene", "1,4-dihydro-1,4-epiminonaphthalene",
    "1,5-(epoxymethanosulfano)octalene",                         # a composite bridge
    # a bridge cited more than once: 'disulfano' there would read as the -S-S- bridge
    "4,8-dihydro-1H-1,4:5,8-disulfanobenzo[7]annulene",
    "propylidene-λ4-sulfanone", "bis(sulfanyl)ethane", "1,4-diazanonaphthalene",
    "1,4-dihydro-1,4-λ4-sulfanonaphthalene",
    "ethanol", "ethan(18O)ol", "(N-2H1)acetamide", "(2,4-2H2,15N)pyridine", "(13C)methyl acetate",
    "(2H3)acetonitrile", "(2H6)benzene", "trichloro(12C)methane",
    # '(2H)' after a locant is an added-hydrogen mark
    "naphthalen-4a(2H)-amine", "anthracene-1,9,10(2H)-trione",
    "pyrimidine-2,4(1H,3H)-dione", "quinoxaline-2,3(1H,4H)-dione",
    "naphthalen-2(3H)-one", "pyridin-2(1H)-one",
    "[1,1'-biphenyl]-4'(2H)-one", "naphthalen-4a'(2H)-amine",      # after a primed locant
    # a locant in the descriptor: OPSIN reads the hydrogen nuclide as written
    "(O-2H)acetic acid", "(1-2H)ethan-1-ol", "(O-2H,18O)acetic acid",
])
def test_no_twin(name):
    assert twin_of(name) is None


def test_two_rules_compose():
    assert twin_of("(N-2H2)-1,4-dihydro-1,4-sulfanonaphthalen-1-amine") == (
        "(N,N-2H2)-1,4-dihydro-1,4-epithionaphthalen-1-amine",
        ("bridge_sulfano", "isotope_letter_once"))


def test_the_rules_apply_in_table_order():
    assert twin_of("(N-2H2)-1,4-dihydro-1,4-sulfanonaphthalen-1-yl(2H)ol")[1] == (
        "bridge_sulfano", "isotope_letter_once", "isotope_count_one")
    assert [rule for rule, _ in TWIN_RULES] == [
        "bridge_sulfano", "bridge_disulfano", "bridge_azano", "interior_phenalene", "isotope_letter_once",
        "isotope_count_one", "polyester_acyloxy", "halido_o"]


def test_twin_of_is_pure_and_a_twin_is_not_rewritten_again():
    for pin, twin, _ in PAIRS:
        assert twin_of(pin) == twin_of(pin)
        assert twin_of(twin) is None


def test_the_memo_key_carries_the_tier_flag():
    """A name memoized in a wider-tier call (general bridge and isotope spellings) is not served to a
    default-tier call (PIN spellings), nor the other way round; a pure namespace keeps its key."""
    from orthonym.assembly import memo
    from orthonym.metrics.provenance import wider_tier_ctx
    default = memo._ck("name_pipeline_only", ("C", "pin"))
    token = wider_tier_ctx.set(True)
    try:
        wider = memo._ck("name_pipeline_only", ("C", "pin"))
        assert memo._ck("fg_detect", "C") == ("fg_detect", "C")
    finally:
        wider_tier_ctx.reset(token)
    assert default != wider
    assert memo._ck("name_pipeline_only", ("C", "pin")) == default
