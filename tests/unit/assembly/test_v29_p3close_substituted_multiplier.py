"""Item A — the multiplier choice asks "is it SUBSTITUTED?", not "is it complex?".

`get_multiplier_prefix` used to consult `is_complex_substituent`, whose actual
test is "contains a digit / hyphen / a known compound morpheme". That is the
**enclosure** question, not the **multiplier** question
(c) / (a)), and the two rules disagree on two whole classes:

* a SIMPLE prefix that merely carries a locant — `propan-2-yl`,
    `naphthalen-2-yl`, `bicyclo[3.2.1]octan-3-yl` — is enclosed in parentheses
    but multiplied by the SIMPLE `di`/`tri`;
* a SUBSTITUTED prefix with no digit at all — `hydroxymethyl`,
    `carboxymethyl`, `oxomethyl`, `silylamino`, `phenyldiazenyl` — takes
    `bis`/`tris`.

So the predicate was wrong in BOTH directions: 16 of the 49 Blue-Book-derived
rows below were wrong at.

Blue Book authority, with headings (de-garbled from the OCR of chapter,
in which `G` is a hyphen, `!` a space and `P"16.x`/`PG16.x` a rule number):

* `**P"16.3** MULTIPLICATIVE!PREFIXES!'DI',!'TRI',!ETC. *vs*.'BIS',!'TRIS',!ETC.`
    (`the Blue Book`) — the chapter that owns this decision.
* `### **P"16.3.2** General!methodology.` (`:7031`) clause **(a)**: "*determine
    whether the component to be multiplied is simple or compound. Simple
    components are unsubstituted parent hydrides, such as naphthalene;
    unsubstituted prefixes, such as ethyl or tert-butyl; functionalized parent
    hydrides, such as benzenesulfonic acid; or retained names, such as acetic
    acid. All of these are multiplied by the multiplicative prefixes 'di',
    'tri', etc.*"; clause **(c)**: "*any component which is substituted
    automatically requires use of the multiplicative forms 'bis', 'tris',
    etc.*"; clause **(d)** reserves the bis-form for a simple component that
    would be AMBIGUOUS if multiplied by 'di' — the existing
    `CATENATION_AMBIGUOUS_PREFIXES` leg, which this change preserves untouched.
* `P"16.3.4 Parentheses!(round!brackets)!...!are!used!to!enclose!multiplied!
    components!that!are:` (`:7085`) clause **(a)** "*simple substituent prefixes
    having locants*" with the verbatim example `di(propanG2Gyl)!(preferred!
    prefix)` at `:7087` and `tetra(naphthalenG2Gyl)!(preferred!prefix)` at
    `:7089` — parentheses AND a simple multiplier, together, on a locant-bearing
    prefix. Clause (f) adds `di(bicyclo[3.2.1]octanG3Gyl)!(preferred!prefix)`.
* `**P"16.3.5**` (`:7104`) clause **(a)**: "*The numerical prefixes 'bis',
    'tris', 'tetrakis', etc. are used to indicate a multiplicity of: (a) compound
    or complex (i.e. **substituted**) prefixes*" — examples
    `bis(2GchloropropanG2Gyl)`, `bis(dimethylamino)`, `bis(bromomethyl)`, all
    "preferred prefix".

The decisive matched pair, one molecule apart, both PINs:

* `1,4-di(propan-2-yl)cyclohexane (PIN)` (`the Blue Book`)
* `1,4-bis(2-chloropropan-2-yl)benzene (PIN)` (`:25793`)

Same parent, same locants; `di` while the prefix is unsubstituted, `bis` the
moment it is substituted. That pair, not the presence of a digit, is the rule.

Two further verbatim PINs pin the Group-14 and negative cases:
`ethyldi(propanG2Gyl)silane!(PIN)` (`:7294`) and
`1,3,5-tri(decyl)cyclohexane (PIN) (not 1,3,5-tris(decyl)cyclohexane)` (`:25715`).
"""

import pytest

from orthonym.assembly.naming_utils import (
    get_multiplier_prefix,
    is_complex_substituent,
    is_substituted_substituent,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# The Blue-Book-derived oracle. Every row is a name that literally appears in
# the Blue Book inside a `di(...)` or a `bis(...)`, or is the systematic prefix
# for one of this repo's own live emissions.
# ---------------------------------------------------------------------------

# (a): unsubstituted parent hydrides / unsubstituted prefixes /
# retained names -> the SIMPLE multiplier, parentheses or not.
BB_SIMPLE = [
    # bare alkyl/aryl, no locant: `di(methyl)`:7104-region, `di(phenyl)`
    "methyl", "ethyl", "propyl", "decyl", "phenyl", "benzyl", "cyclohexyl",
    # (c)/(d): a chain root that merely BEGINS with a multiplier
    # (`di(dodecyl)`, `di(tridecyl)`, `tri(decyl)`) is still simple.
    "dodecyl", "tridecyl", "tetradecyl", "undecyl", "octadecyl",
    # (a): simple prefixes HAVING LOCANTS -> di(...)
    "propan-2-yl", "butan-2-yl", "pentan-2-yl", "prop-1-en-2-yl",
    "propan-2-ylidene", "propan-2-ylium",
    # ring prefixes with locants: `di(naphthalen-2-yl)`, `di(furan-2-yl)`,
    # `di(isoxazol-3-yl)`, `di(tetraphen-1-yl)`, `tetra(naphthalen-2-yl)`
    "naphthalen-2-yl", "furan-2-yl", "isoxazol-3-yl", "tetraphen-1-yl",
    # (f): simple components containing brackets
    "bicyclo[3.2.1]octan-3-yl",
    # `di(metheno)` — a simple divalent bridge prefix
    "metheno",
]

# (a): compound or complex (i.e. substituted) prefixes -> bis/tris.
BB_SUBSTITUTED = [
    # with a digit (already correct at HEAD, must stay correct)
    "2-chloropropan-2-yl", "2-chloroethyl", "2-methylbutyl",
    "2,2-dimethylpropyl", "2-methylpropyl",
    # NO digit at all — the direction the digit test misses entirely
    "hydroxymethyl", "carboxymethyl", "oxomethyl", "silylamino",
    "phenyldiazenyl",
    # multiplied substituent prefixes
    "trifluoromethyl", "trimethylsilyl", "dimethylamino", "dimethylphosphanyl",
    # single substituent prefix on a simple carrier
    "methylamino", "bromomethyl", "chloromethyl", "ethylsulfanyl",
    "cyclohexylmethyl",
    # acyloxy compounds: `bis(acetyloxy)`, `tetrakis(acetyloxy)`,
    # and this repo's live `bis(hexadecanoyloxy)` gold rows
    "acetyloxy", "hexadecanoyloxy", "octadecanoyloxy",
]

# (d): a SIMPLE component that would be ambiguous if multiplied by
# 'di' keeps the bis-form. `disulfanyl` is -S-SH, so two -SH must be
# `bis(sulfanyl)` (BB has `bis(sulfanyl)` x7). This leg already existed and
# must survive the change untouched.
BB_AMBIGUOUS = ["sulfanyl", "selanyl", "phosphanyl", "azanyl", "tellanyl"]


@pytest.mark.parametrize("name", BB_SIMPLE)
def test_simple_prefix_is_not_substituted(name):
    """(a): an unsubstituted prefix is not substituted, locant or not."""
    assert is_substituted_substituent(name) is False, (
        f"{name!r} is an unsubstituted prefix (P-16.3.2(a)) and must not be "
        f"classed substituted"
    )


@pytest.mark.parametrize("name", BB_SUBSTITUTED)
def test_substituted_prefix_is_substituted(name):
    """(a): a compound/complex (i.e. substituted) prefix."""
    assert is_substituted_substituent(name) is True, (
        f"{name!r} carries a detachable substituent prefix of its own "
        f"(P-16.3.2(c) / P-16.3.5(a)) and must be classed substituted"
    )


@pytest.mark.parametrize("name", BB_SIMPLE)
def test_simple_prefix_takes_the_simple_multiplier(name):
    """(a) end-to-end: di/tri, never bis/tris."""
    assert get_multiplier_prefix(2, name).rstrip("-") == "di"
    assert get_multiplier_prefix(3, name).rstrip("-") == "tri"


@pytest.mark.parametrize("name", BB_SUBSTITUTED)
def test_substituted_prefix_takes_the_bis_multiplier(name):
    """(a) end-to-end: bis/tris, never di/tri."""
    assert get_multiplier_prefix(2, name) == "bis"
    assert get_multiplier_prefix(3, name) == "tris"


@pytest.mark.parametrize("name", BB_AMBIGUOUS)
def test_ambiguity_leg_still_takes_bis(name):
    """(d): preserved untouched by the predicate swap."""
    assert get_multiplier_prefix(2, name) == "bis"


# ---------------------------------------------------------------------------
# The two questions must stay SEPARATE. `is_complex_substituent` keeps its own
# job (enclosure,; only the multiplier moved off it.
# ---------------------------------------------------------------------------

def test_the_two_predicates_provably_disagree():
    """The whole point: enclosure and multiplier are different questions.

    `propan-2-yl` is enclosed (a)) AND simply multiplied
    (a)) — `di(propan-2-yl)`. A single predicate cannot say both, and
    conflating them is what produced `bis(propan-2-yl)`.
    """
    assert is_complex_substituent("propan-2-yl") is True    # -> parentheses
    assert is_substituted_substituent("propan-2-yl") is False  # -> `di`
    #... and the reverse disagreement, on a name with no digit to key off.
    assert is_complex_substituent("hydroxymethyl") is False
    assert is_substituted_substituent("hydroxymethyl") is True


def test_enclosure_predicate_is_unchanged_for_the_italicized_carve_out():
    """`tert-butyl` stays simple for BOTH questions (a) names it)."""
    assert is_complex_substituent("tert-butyl") is False
    assert is_substituted_substituent("tert-butyl") is False
    # and the (d) hyphen still comes from the primitive
    assert get_multiplier_prefix(2, "tert-butyl") == "di-"
    assert get_multiplier_prefix(3, "tert-butyl") == "tri-"


def test_count_one_still_has_no_multiplier():
    for n in ("propan-2-yl", "hydroxymethyl", "methyl", "tert-butyl"):
        assert get_multiplier_prefix(1, n) == ""


def test_empty_name_is_not_substituted():
    assert is_substituted_substituent("") is False
    assert is_substituted_substituent(None) is False


# ---------------------------------------------------------------------------
# The BB's own matched pair, as an executable statement of the rule.
# ---------------------------------------------------------------------------

def test_bb_matched_pair_di_vs_bis():
    """BB:25721 `1,4-di(propan-2-yl)cyclohexane (PIN)` vs
    BB:25793 `1,4-bis(2-chloropropan-2-yl)benzene (PIN)`.

    One chloro apart; the multiplier flips. Nothing about the digit.
    """
    assert get_multiplier_prefix(2, "propan-2-yl") == "di"
    assert get_multiplier_prefix(2, "2-chloropropan-2-yl") == "bis"


def test_bb_tri_decyl_negative_control():
    """BB :25715 `1,3,5-tri(decyl)cyclohexane (PIN)
    (not 1,3,5-tris(decyl)cyclohexane)` — verbatim negative control."""
    assert get_multiplier_prefix(3, "decyl") == "tri"


# ---------------------------------------------------------------------------
# (a) — SUFFIXES. `get_multiplier_prefix` is called for suffixes too,
# and this is where the first version of this change broke 10 gold rows.
# ---------------------------------------------------------------------------

# `****` (the Blue Book): "*The basic numerical prefixes 'di',
# 'tri', 'tetra', etc. are used to indicate a multiplicity of: (a) functional and
# cumulative suffixes*", examples `diol`, `dicarboxylic!acid`, `disulfonic!acid`.
FUNCTIONAL_SUFFIXES = [
    "carboxylic acid", "sulfonic acid", "sulfinic acid", "phosphonic acid",
    "ol", "amide", "carboxamide", "nitrile", "carbonitrile", "carbaldehyde",
    "carboxylate", "sulfonate", "oate",
]


@pytest.mark.parametrize("suffix", FUNCTIONAL_SUFFIXES)
def test_a_functional_suffix_always_takes_the_basic_multiplier(suffix):
    """(a). A suffix is NEVER decomposed for substituted-ness.

    REGRESSION GUARD. The first version of `is_substituted_substituent`
    decomposed `carboxylic acid` as `carboxy` + `lic acid` and `sulfonic acid` as
    `sulfo` + `nic acid`, so it emitted `benzene-1,4-biscarboxylic acid` and
    `4-methylbenzene-1,3-bissulfonic acid`. That broke 10 gold targets and was
    caught only by measuring the flip set before committing.
    """
    assert is_substituted_substituent(suffix) is False
    assert get_multiplier_prefix(2, suffix) == "di"
    assert get_multiplier_prefix(3, suffix) == "tri"


def test_the_thioic_acid_suffix_exception_takes_bis():
    """(b) is an EXPLICIT exception to (a).

    BB `:7104`: `bis(thioic acid)` is a "multiple preferred suffix" describing
    two -C(O/S)H suffixes, "*whereas dithioic acid describes a -CSSH suffix*",
    and `bis(dithioic acid)`... "*not didithoic acid*".
    """
    assert get_multiplier_prefix(2, "thioic acid") == "bis"
    assert get_multiplier_prefix(2, "dithioic acid") == "bis"


def test_an_already_enclosed_prefix_is_decided_on_its_interior():
    """A fully enclosed token is one component; the marks are not the answer.

    Producers hand this primitive both bare and already-enclosed names, so
    `(2-chloroethyl)` must reach the same verdict as `2-chloroethyl`. Without
    the recursion the leading `(` blocks every test below it and a substituted
    prefix silently returns False.
    """
    assert is_substituted_substituent("(2-chloroethyl)") is True
    assert is_substituted_substituent("2-chloroethyl") is True
    assert is_substituted_substituent("(propan-2-yl)") is False
    assert is_substituted_substituent("propan-2-yl") is False


def test_a_trailing_syllable_is_not_a_substituted_remainder():
    """`methylene` is `methyl` + `ene`, and `ene` is not a group name.

    Without the remainder-must-look-like-a-group-name guard, every `-ene`/`-yne`
    prefix decomposes into "a substituent plus a syllable" and wrongly takes
    `bis`. (BB does write `bis(methylene)`, but that is the (d)
    AMBIGUITY leg — `dimethylene` could read as -CH2CH2- — not substitution, and
    it must come from `CATENATION_AMBIGUOUS_PREFIXES`, not from here.)
    """
    assert is_substituted_substituent("methylene") is False
    assert is_substituted_substituent("ethylene") is False
    assert is_substituted_substituent("carbonyl") is False


def test_an_italicized_prefix_on_a_COMPOUND_remainder_is_substituted():
    """`tert-butylsulfanyl` reduces to the compound `butylsulfanyl`.

    (a) makes `tert-butyl` simple, but the carve-out is about the
    ITALICIZED PREFIX only — it does not make everything it leads simple. So
    this takes `bis` (a)) and, being enclosed, takes NO hyphen:
    `****` (`the Blue Book`) "*No hyphen is placed after a
    numerical prefix cited in front of a compound substituent enclosed by
    parentheses*".

    Before this task the hyphen leg ran first and emitted
    `di-tert-butylsulfanyl`, on an explicit code comment asserting that a
    substituted italicized-led prefix could not exist.
    """
    assert is_substituted_substituent("tert-butylsulfanyl") is True
    assert is_substituted_substituent("sec-butylsulfanyl") is True
    assert get_multiplier_prefix(2, "tert-butylsulfanyl") == "bis"
    assert get_multiplier_prefix(3, "tert-butylsulfanyl") == "tris"
    #...while the SIMPLE italicized prefix keeps its (d) hyphen.
    assert get_multiplier_prefix(2, "tert-butyl") == "di-"


def test_a_chain_root_beginning_with_a_multiplier_is_not_substituted():
    """(c)/(d): `tridecyl` is C13, not `tri` + `decyl`.

    This is the trap that defeats a peel-the-multiplier string rule, and the
    reason the full name is tested against the chain-root table FIRST.
    """
    assert is_substituted_substituent("tridecyl") is False
    assert get_multiplier_prefix(2, "tridecyl") == "di"
    # contrast: `trimethylsilyl` really is a multiplied substituent prefix
    assert is_substituted_substituent("trimethylsilyl") is True


# ===========================================================================
# -FINAL — the classes NO oracle row covered, which is why four
# OPSIN-clean wrong PINs shipped through a full PASSING gate.
#
# Every row below carries its Blue Book authority WITH the section heading.
# The OCR of chapter mangles hyphen->`G` and space->`!`, so quotations from
# it are de-garbled; quotations from chapters /// are literal.
# ===========================================================================

# ---------------------------------------------------------------------------
# (1) CHARACTERISTIC-GROUP prefixes are SIMPLE -> basic `di`.
#
# `**P"16.3.3**` (BB 7038) "*The basic numerical prefixes 'di', 'tri', 'tetra',
# etc. are used to indicate a multiplicity of:*" clause **(b)** (BB 7067,
# continuing at BB 7094) "*simple substituent prefixes, including parent hydrides
# with 'ene' and 'yne' endings (without locants), and characteristic groups*",
# with the verbatim examples `diimino` (BB 7072) and `dibromo` (BB 7073).
#
# THE REGRESSION THIS PINS: `nitroso` was decomposed as `nitro` + `so`, and `so`
# passed a tail test that accepted any remainder ending in a group-shaped letter
# (bare `'o'` was in `_GROUP_NAME_ENDINGS`). Result: `1,4-bis(nitroso)benzene`,
# OPSIN-clean, against the verbatim PIN.
# `## **** Nitro and nitroso compounds` (BB 25933) ->
# `1,4-dinitrosobenzene (PIN)` (BB 25943).
# ---------------------------------------------------------------------------
BB_SIMPLE_CHARACTERISTIC_GROUP = [
    # verbatim-attested multiplied forms
    "nitroso",   # BB 25943 `1,4-dinitrosobenzene (PIN)`
    "nitro",     # BB 25947 `2-methyl-1,3,5-trinitrobenzene (PIN)`
    "cyano",     # BB 40908 `tricyanomethanide (PIN)`
    "azido",     # BB 53968 `2,3-diazido-6-bromo-...`
    "imino",     # BB 7072 `diimino` (b) example)
    "bromo",     # BB 7073 `dibromo` (b) example)
    "chloro", "fluoro", "iodo",
    # simple by rule (b) "characteristic groups"); the Blue Book
    # happens to print no multiplied instance of these four, so they are a
    # SOUND INFERENCE from the clause, not verbatim examples.
    "hydroxy", "oxo", "amino", "carboxy",
    "carbamoyl", "sulfamoyl", "hydroperoxy", "hydrazinyl",
    "sulfo", "sulfino", "isocyano", "isocyanato", "guanidino",
    "carbamimidoyl", "sulfanylidene", "selanylidene", "thioxo",
    "diazenyl", "hydrazinylidene", "phosphono", "phosphino", "borono",
]

# ---------------------------------------------------------------------------
# (2) The SIX retained contracted alkoxy prefixes are SIMPLE -> `di`, and every
# OTHER `<R>oxy` prefix is a concatenated COMPOUND prefix -> `bis`.
#
# This is the sharpest boundary in the whole predicate and the Blue Book states
# BOTH halves outright, three lines apart in the same section:
#
# `### **** Retained names` (BB 27665), BB 27667: "*Some contracted
# names are retained for R-O– substituent groups.... they are fully
# substitutable (with the exception of tert-butoxy) and are **considered as
# simple prefixes requiring the numerical prefixes 'di', 'tri', etc.** They
# are:*" -> `methoxy`, `ethoxy`, `propoxy`, `butoxy`, `phenoxy`,
# `tert-butoxy` (all "(preferred prefix)"). Confirmed in a PIN at BB 27705:
# `1,2-dimethoxybenzene (PIN)`.
#
# `### **** Systematic names` (BB 27631) -> `****`
# (BB 27633): "*Substituent prefix names for R′-O– groups are formed by
# concatenation, i.e., by adding the prefix 'oxy' to the substituent prefix
# name for the group R′. **These compound prefixes require the numerical
# multiplying prefixes 'bis', 'tris', etc.**\*" -> first example
# `pentyloxy (preferred prefix)`.
#
# So the list is CLOSED at six, and membership — not spelling shape — decides.
# ---------------------------------------------------------------------------
BB_SIMPLE_RETAINED_ALKOXY = ["methoxy", "ethoxy", "propoxy", "butoxy", "phenoxy"]

BB_SUBSTITUTED_CONCATENATED_OXY = [
    "pentyloxy",        # BB 27637, the rule's own first example
    "hexyloxy", "benzyloxy",
    # concatenated hetero-oxy prefixes. `phosphonooxy` is filed by the Blue
    # Book under `**** Compound and complex substituent groups`
    # (BB 36327; Appendix 2 row BB 56614 cites that same rule), and `sulfooxy`
    # under `### **** Substituent groups formed by concatenation`
    # (BB 31276). Neither is ever printed multiplied, so `bis` follows from
    # (a) + rather than from a verbatim example — but the
    # CLASSIFICATION is verbatim.
    "phosphonooxy", "sulfooxy", "nitrooxy", "carbamoyloxy", "aminooxy",
    "sulfanyloxy",
    # BB 31346 `sulfonylbis(sulfanediyl) (preselected prefix)` and BB 31348
    # `sulfinobis(oxy) (preselected prefix)` are the sibling witnesses.
]

# ---------------------------------------------------------------------------
# (3) ALKOXY-ALKYL and every other two-unit compound prefix -> `bis`.
#
# `**P"16.3.5**`(a) (BB 7104): "*The numerical prefixes 'bis', 'tris',
# 'tetrakis', etc. are used to indicate a multiplicity of: (a) compound or
# complex (i.e. substituted) prefixes (see and *", with the
# verbatim example `bis(bromomethyl) (preferred prefix)`.
#
# `### **** GENERAL METHODOLOGY` (BB 17950), BB 17954: "*Compound and
# mixed substituent prefixes require the derived multiplying terms 'bis', 'tris',
# 'tetrakis', etc., to designate their multiplicity in substitutive
# nomenclature.*"
#
# THE REGRESSION THIS PINS: `methoxymethyl` was absent from a closed hand-list
# of prefix morphemes, so it read as SIMPLE and shipped
# `1,4-di(methoxymethyl)benzene` where the identically-shaped `bromomethyl`
# correctly gave `1,4-bis(bromomethyl)benzene`. A closed spelling table cannot
# be the class boundary: the class is open, so the table is wrong on its
# complement.
# ---------------------------------------------------------------------------
BB_SUBSTITUTED_TWO_UNIT = [
    # alkoxy + carrier — the class that shipped wrong
    "methoxymethyl", "2-methoxyethyl", "3-methoxypropyl", "2-ethoxyethyl",
    "4-methoxyphenyl", "2-methoxy-2-oxoethyl", "2-hydroxyethoxy",
    # amido / sulfamoyl on a ring carrier
    "4-acetamidophenyl", "4-sulfamoylphenyl", "acetamido",
    # heteroaryl carrier + alkyl (`pyridin-2-yl` + `methyl`)
    "pyridin-2-ylmethyl", "oxolan-2-ylmethyl", "furan-2-ylmethyl",
    # carbamoyl carrying its own substituent. NB the Blue Book spells this
    # WITHOUT an italic locant: `4-(dimethylcarbamoyl)benzoic acid (PIN)`
    # (BB 30382, `## **** Amic acids`), and BB 24611
    # `(dimethylcarbamoyl)hydrazinylidene (preferred prefix)`.
    "dimethylcarbamoyl", "methylcarbamoyl",
    #...and the italic-letter-locant spelling must reach the same verdict.
    # `di(*N*-...` occurs ZERO times in the book; `bis(*N*-...` occurs 5 times
    # (e.g. BB 6198 `...bis(*N*-methylacetamide) (PIN)`), always earned by the
    # substitution rather than by the locant.
    "N-pentylcarbamoyl", "N-methylcarbamoyl", "N,N-dimethylcarbamoyl",
    # enclosed leader then a carrier — the C3 class
    "3-(4-methylphenyl)propyl", "2-(4-chlorophenyl)ethyl",
    "(4-methylphenyl)methyl",
    # verbatim `bis` witnesses
    "1-methylpropyl",      # BB 28170 `2-[bis(1-methylpropyl)amino]butan-2-ol`
    "cyanomethyl",         # BB 33087 `...bis(cyanomethyl)oxamide (PIN)`
    "methylsulfanyl",      # BB 35344 `1,1-bis(methylsulfanyl)pentane (PIN)`
    "2-chloropropan-2-yl",  # BB 7104 / BB 25793
]

# ---------------------------------------------------------------------------
# (4) An UNSUBSTITUTED ring-yl carrying only its own attachment locant is
# SIMPLE -> `di(...)`: the BASIC multiplier, WITH parentheses. The parentheses
# come from a different rule than the multiplier, and conflating them is the
# original sin this whole task exists to undo.
#
# `P"16.3.4 Parentheses (round brackets)... are used to enclose multiplied
# components that are:` (BB 7085) clause **(a)** "*simple substituent prefixes
# having locants*" -> `di(propanG2Gyl) (preferred prefix)` (BB 7087),
# `tetra(naphthalenG2Gyl) (preferred prefix)` (BB 7090); clause **(f)**
# "*simple components containing brackets*" -> `di(bicyclo[3.2.1]octanG3Gyl)
# (preferred prefix)` and `di([4G2H]benzoyl) (preferred prefix)`.
#
# `**P"16.5.1.1**` (BB 7232) says the same from the marks side: parentheses
# enclose "*a multiplied substituent prefix **even though preceded by a basic
# numerical prefix, such as 'di', 'tri', etc.***"
#
# Verbatim PIN witnesses for the bare-ring-yl class:
# `1,2-di(furan-2-yl)-2-hydroxyethan-1-one (PIN)` BB 29677 (`## ****
# ACYLOINS`, BB 29661)
# `di(1*H*-imidazol-1-yl)methanethione (PIN)` BB 29544
# `di(naphthalen-2-yl)ethanedione (PIN)` BB 28380
# `1,3-di(naphthalen-2-yl)triaz-1-ene (PIN)` BB 39047
# `2,6-di(tetraphen-1-yl)pyridine (PIN)` BB 25762
# `2-[di(butan-2-yl)amino]butan-2-ol (PIN)` BB 28170
#
# THE DEFECTS THIS PINS: `oxolan-2-yl` peeled as `oxo` + `lan-2-yl` and shipped
# the live `bis(oxolan-2-yl)methanol`; `triphenylen-2-yl` peeled as `tri` +
# `phenyl` + `en-2-yl`; and a leading bracketed FUSION descriptor was misread as
# an enclosed substituent, so `[1,2,4]triazolo[1,5-a]pyrimidin-2-yl` and
# `[4-2H]benzoyl` were called substituted against the verbatim (f).
# ---------------------------------------------------------------------------
# a performance pass: the Hantzsch-Widman ring-yls that open with an 'a' prefix are simple
# but take the derived multiplier by (c) (the Blue Book,
# "before skeletal replacement ('a') prefixes, such as 'aza', 'oxa', etc. that are
# used in the construction of Hantzsch-Widman names";:7178 'bis(1,2-oxazol-3-yl)
#... whereas di(1,2-oxazol-3-yl) might be interpreted as a 'dioxazole' ring
# system'): 'bis(oxolan-2-yl)methanol', read back exact by OPSIN.
BB_REPLACEMENT_FRONT_RING_YL = [
    "oxolan-2-yl", "oxan-4-yl", "oxocan-2-yl", "oxonan-2-yl",
    "1,3-dioxolan-2-yl", "1,3-dioxan-2-yl",
]
BB_SIMPLE_BARE_RING_YL = [
    "triphenylen-2-yl", "pyridin-2-yl", "thiophen-2-yl",
    "1H-imidazol-1-yl", "piperidin-1-yl", "pyrrolidin-1-yl",
    "cyclohexen-1-yl", "isoxazol-3-yl",
    # bracket-leading FUSION descriptors — (f), verbatim `di([4G2H]benzoyl)`
    "[1,2,4]triazolo[1,5-a]pyrimidin-2-yl", "[1,3]oxazolo[4,5-b]pyridin-2-yl",
    "[1,2,4]triazin-3-yl", "[1,3]oxazol-2-yl", "[4-2H]benzoyl",
    # retained hydrocarbyls: BB 23258 `...(2,2-dibenzylpropane-1,3-diyl)...
    # (PIN)` uses the BASIC `di` on `benzyl`, and BB 24420 says `benzyl` "cannot
    # be substituted" in a preferred name, so (a) can never fire on it.
    "benzyl", "benzylidene",
    # BB 7324 `di(benzenesulfinyl)acetic acid (PIN)`
    "benzenesulfinyl",
]


@pytest.mark.parametrize("name", BB_SIMPLE_CHARACTERISTIC_GROUP
                         + BB_SIMPLE_RETAINED_ALKOXY
                         + BB_SIMPLE_BARE_RING_YL)
def test_p3final_simple_class_takes_the_basic_multiplier(name):
    assert is_substituted_substituent(name) is False, (
        f"{name!r} is an unsubstituted/simple prefix and must take di/tri"
    )
    assert get_multiplier_prefix(2, name).rstrip("-") == "di"
    assert get_multiplier_prefix(3, name).rstrip("-") == "tri"


@pytest.mark.parametrize("name", BB_REPLACEMENT_FRONT_RING_YL)
def test_replacement_front_ring_yl_is_simple_but_takes_bis(name):
    assert is_substituted_substituent(name) is False, name
    assert get_multiplier_prefix(2, name) == "bis"
    assert get_multiplier_prefix(3, name) == "tris"


@pytest.mark.parametrize("name", BB_SUBSTITUTED_CONCATENATED_OXY
                         + BB_SUBSTITUTED_TWO_UNIT)
def test_p3final_substituted_class_takes_the_derived_multiplier(name):
    assert is_substituted_substituent(name) is True, (
        f"{name!r} is a compound/complex (substituted) prefix "
        f"(P-16.3.5(a)) and must take bis/tris"
    )
    assert get_multiplier_prefix(2, name) == "bis"
    assert get_multiplier_prefix(3, name) == "tris"


def test_the_four_shipped_wrong_pins_as_one_executable_statement():
    """The exact four CRITICAL emissions, at the predicate level."""
    # C1 — BB 25943 `1,4-dinitrosobenzene (PIN)`
    assert get_multiplier_prefix(2, "nitroso") == "di"
    # C4 — (a); cf. verbatim `bis(bromomethyl)`
    assert get_multiplier_prefix(2, "methoxymethyl") == "bis"
    assert get_multiplier_prefix(2, "bromomethyl") == "bis"
    # C3 — BB 25729 `1,2,4-tris[3-(4-methylphenyl)propyl]benzene`
    assert get_multiplier_prefix(3, "3-(4-methylphenyl)propyl") == "tris"
    # C2 — BB 30313 `benzene-1,2-dicarbodithioic acid (PIN)`
    assert get_multiplier_prefix(2, "carbodithioic acid") == "di"


def test_the_locant_peel_runs_before_the_enclosed_leader_test():
    """C3's mechanism, isolated.

    `is_substituted_substituent('3-(4-methylphenyl)propyl')` returned False
    while `('(4-methylphenyl)propyl')` returned True — the enclosed-leader test
    ran on the RAW name, so a leading locant blocked it permanently and the
    morpheme loop could never match a body starting with `(`.
    """
    assert is_substituted_substituent("(4-methylphenyl)propyl") is True
    assert is_substituted_substituent("3-(4-methylphenyl)propyl") is True


def test_a_leading_bracketed_fusion_descriptor_is_not_an_enclosed_prefix():
    """m1. `[1,2,4]` is a locant list, not a substituent.

     clause (f) "*simple components containing brackets*" is verbatim on
    this: `di([4G2H]benzoyl) (preferred prefix)` and
    `8,8′Goxydi(spiro[4.5]decane) (PIN)` — bracket-leading prefixes with the
    BASIC multiplier.
    """
    for nm in ("[1,2,4]triazolo[1,5-a]pyrimidin-2-yl", "[4-2H]benzoyl",
               "spiro[4.5]decan-2-yl"):
        assert is_substituted_substituent(nm) is False, nm
    #...while a bracketed SUBSTITUENT leader still counts.
    assert is_substituted_substituent("[4-(methoxy)phenyl]methyl") is True


def test_a_ring_stem_is_never_peeled_as_a_characteristic_group_prefix():
    """I2. `oxolan-2-yl` is not `oxo` + `lan-2-yl`.

    The old tail test accepted any remainder ending in a group-shaped letter, so
    `lan-2-yl` looked like a group name and the live emission was
    `bis(oxolan-2-yl)methanol` where BB 29677's `di(furan-2-yl)` class requires
    `di(oxolan-2-yl)methanol`. The negative-control list it relied on covered
    only the 20 acyclic alkyl roots, so no ring name could ever be protected.

    (a performance pass: the multiplier of a Hantzsch-Widman prefix is 'bis' all the same, by
     (c), the Blue Book, "before skeletal replacement ('a')
    prefixes... used in the construction of Hantzsch-Widman names", 'bis(1,2-
    oxazol-3-yl)... whereas di(1,2-oxazol-3-yl) might be interpreted as a
    'dioxazole' ring system'; furan is a retained name, not a Hantzsch-Widman
    one. This test asserts the predicate only, which is unchanged.)
    """
    for nm in ("oxolan-2-yl", "oxocan-2-yl", "oxonan-2-yl",
               "1,3-dioxolan-2-yl", "triphenylen-2-yl"):
        assert is_substituted_substituent(nm) is False, nm
    # and the siblings that were already right stay right
    for nm in ("oxan-4-yl", "1,3-dioxan-2-yl"):
        assert is_substituted_substituent(nm) is False, nm


def test_the_retained_alkoxy_list_is_closed_at_six():
    """ vs — membership decides, not spelling shape.

    Both halves are verbatim, and they are three lines apart in the book, so a
    predicate that keys on "ends in -oxy" is wrong whichever answer it gives.
    """
    for simple in ("methoxy", "ethoxy", "propoxy", "butoxy", "phenoxy"):
        assert is_substituted_substituent(simple) is False, simple
    for compound in ("pentyloxy", "hexyloxy", "benzyloxy", "phosphonooxy"):
        assert is_substituted_substituent(compound) is True, compound


# ---------------------------------------------------------------------------
# (5) SUFFIXES. `get_multiplier_prefix` is asked BOTH questions by different
# callers, and the suffix answer is not the substituent answer.
#
# `**P"16.3.3**` (BB 7038) clause (a) (BB 7040): "*functional and cumulative
# suffixes, basic or modified by functional replacement, **with the exception of
# 'thioic acid' and 'dithioic acid' described in (b)***".
#
# `**P"16.3.5**`(b) (BB 7104): "*'thioic acid' and 'dithioic acid' suffixes, and
# their Se and Te analogues, as exceptions to suffixes described in and
# *".
#
# The Blue Book prints the decisive boundary pair four lines apart under
# `### **** Functional replacement in systematic names of carboxylic
# acids`:
# `benzene-1,2-dicarbodithioic acid (PIN) (not tetrathiophthalic acid)` BB 30313
# `ethanebis(dithioic acid) (PIN) (not tetrathiooxalic acid)` BB 30317
#
# THE REGRESSION THIS PINS: the exception was matched with
# `endswith('dithioic acid')`, and `'carbodithioic acid'.endswith('dithioic
# acid')` is True — so the ring form shipped as
# `cyclohexane-1,2-biscarbodithioic acid`. The exception must match the WHOLE
# suffix token.
# ---------------------------------------------------------------------------
BB_BASIC_SUFFIXES = [
    "carbodithioic acid", "carbothioic acid", "carboxylic acid",
    "sulfonic acid", "carboxamide", "carbonitrile", "carbaldehyde", "ol",
]

BB_BIS_SUFFIXES = [
    "thioic acid", "dithioic acid",
    "selenoic acid", "diselenoic acid", "telluroic acid", "ditelluroic acid",
    # I1 — the italic-locant spellings of the SAME suffixes.
    # (BB 30215) confirms the `O-`/`S-` forms and that the locants are "normally
    # omitted", so keying the exception on the locant-free spelling alone made
    # the whole `<alkane>bis(thioic O-acid)` family abstain.
    "thioic O-acid", "thioic S-acid", "dithioic O-acid", "dithioic S-acid",
    "selenoic Se-acid",
]


@pytest.mark.parametrize("suffix", BB_BASIC_SUFFIXES)
def test_p3final_a_suffix_takes_the_basic_multiplier(suffix):
    assert get_multiplier_prefix(2, suffix) == "di", suffix
    assert get_multiplier_prefix(3, suffix) == "tri", suffix


@pytest.mark.parametrize("suffix", BB_BIS_SUFFIXES)
def test_p3final_the_p1635b_suffix_exception_takes_bis(suffix):
    assert get_multiplier_prefix(2, suffix) == "bis", suffix


def test_the_carbodithioic_boundary_pair():
    """BB 30313 vs BB 30317, both PINs, four lines apart."""
    assert get_multiplier_prefix(2, "carbodithioic acid") == "di"
    assert get_multiplier_prefix(2, "dithioic acid") == "bis"
