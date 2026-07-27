"""Item A — the multiplier choice asks "is it SUBSTITUTED?", not "is it complex?".

`get_multiplier_prefix` used to consult `is_complex_substituent`, whose actual
test is "contains a digit / hyphen / a known compound morpheme".  That is the
**enclosure** question (P-16.3.4), not the **multiplier** question
(P-16.3.2(c) / P-16.3.5(a)), and the two rules disagree on two whole classes:

*   a SIMPLE prefix that merely carries a locant — `propan-2-yl`,
    `naphthalen-2-yl`, `bicyclo[3.2.1]octan-3-yl` — is enclosed in parentheses
    but multiplied by the SIMPLE `di`/`tri`;
*   a SUBSTITUTED prefix with no digit at all — `hydroxymethyl`,
    `carboxymethyl`, `oxomethyl`, `silylamino`, `phenyldiazenyl` — takes
    `bis`/`tris`.

So the predicate was wrong in BOTH directions: 16 of the 49 Blue-Book-derived
rows below were wrong at `e5233ef8`.

Blue Book authority, with headings (de-garbled from the OCR of chapter P-16,
in which `G` is a hyphen, `!` a space and `P"16.x`/`PG16.x` a rule number):

*   `**P"16.3** MULTIPLICATIVE!PREFIXES!'DI',!'TRI',!ETC. *vs*.'BIS',!'TRIS',!ETC.`
    (`BlueBookV2.md:7026`) — the chapter that owns this decision.
*   `### **P"16.3.2** General!methodology.` (`:7031`) clause **(a)**: "*determine
    whether the component to be multiplied is simple or compound.  Simple
    components are unsubstituted parent hydrides, such as naphthalene;
    unsubstituted prefixes, such as ethyl or tert-butyl; functionalized parent
    hydrides, such as benzenesulfonic acid; or retained names, such as acetic
    acid.  All of these are multiplied by the multiplicative prefixes 'di',
    'tri', etc.*"; clause **(c)**: "*any component which is substituted
    automatically requires use of the multiplicative forms 'bis', 'tris',
    etc.*"; clause **(d)** reserves the bis-form for a simple component that
    would be AMBIGUOUS if multiplied by 'di' — the existing
    `CATENATION_AMBIGUOUS_PREFIXES` leg, which this change preserves untouched.
*   `P"16.3.4 Parentheses!(round!brackets)!...!are!used!to!enclose!multiplied!
    components!that!are:` (`:7085`) clause **(a)** "*simple substituent prefixes
    having locants*" with the verbatim example `di(propanG2Gyl)!(preferred!
    prefix)` at `:7087` and `tetra(naphthalenG2Gyl)!(preferred!prefix)` at
    `:7089` — parentheses AND a simple multiplier, together, on a locant-bearing
    prefix.  Clause (f) adds `di(bicyclo[3.2.1]octanG3Gyl)!(preferred!prefix)`.
*   `**P"16.3.5**` (`:7104`) clause **(a)**: "*The numerical prefixes 'bis',
    'tris', 'tetrakis', etc. are used to indicate a multiplicity of: (a) compound
    or complex (i.e. **substituted**) prefixes*" — examples
    `bis(2GchloropropanG2Gyl)`, `bis(dimethylamino)`, `bis(bromomethyl)`, all
    "preferred prefix".

The decisive matched pair, one molecule apart, both PINs:

*   `1,4-di(propan-2-yl)cyclohexane (PIN)`  (`BlueBookV2.md:25721`)
*   `1,4-bis(2-chloropropan-2-yl)benzene (PIN)` (`:25793`)

Same parent, same locants; `di` while the prefix is unsubstituted, `bis` the
moment it is substituted.  That pair, not the presence of a digit, is the rule.

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
# The Blue-Book-derived oracle.  Every row is a name that literally appears in
# BlueBookV2.md inside a `di(...)` or a `bis(...)`, or is the systematic prefix
# for one of this repo's own live emissions.
# ---------------------------------------------------------------------------

# P-16.3.2(a): unsubstituted parent hydrides / unsubstituted prefixes /
# retained names -> the SIMPLE multiplier, parentheses or not.
BB_SIMPLE = [
    # bare alkyl/aryl, no locant: `di(methyl)` :7104-region, `di(phenyl)`
    "methyl", "ethyl", "propyl", "decyl", "phenyl", "benzyl", "cyclohexyl",
    # P-16.3.4(c)/(d): a chain root that merely BEGINS with a multiplier
    # (`di(dodecyl)`, `di(tridecyl)`, `tri(decyl)`) is still simple.
    "dodecyl", "tridecyl", "tetradecyl", "undecyl", "octadecyl",
    # P-16.3.4(a): simple prefixes HAVING LOCANTS -> di(...)
    "propan-2-yl", "butan-2-yl", "pentan-2-yl", "prop-1-en-2-yl",
    "propan-2-ylidene", "propan-2-ylium",
    # ring prefixes with locants: `di(naphthalen-2-yl)`, `di(furan-2-yl)`,
    # `di(isoxazol-3-yl)`, `di(tetraphen-1-yl)`, `tetra(naphthalen-2-yl)`
    "naphthalen-2-yl", "furan-2-yl", "isoxazol-3-yl", "tetraphen-1-yl",
    # P-16.3.4(f): simple components containing brackets
    "bicyclo[3.2.1]octan-3-yl",
    # `di(metheno)` — a simple divalent bridge prefix
    "metheno",
]

# P-16.3.5(a): compound or complex (i.e. substituted) prefixes -> bis/tris.
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

# P-16.3.2(d): a SIMPLE component that would be ambiguous if multiplied by
# 'di' keeps the bis-form.  `disulfanyl` is -S-SH, so two -SH must be
# `bis(sulfanyl)` (BB has `bis(sulfanyl)` x7).  This leg already existed and
# must survive the change untouched.
BB_AMBIGUOUS = ["sulfanyl", "selanyl", "phosphanyl", "azanyl", "tellanyl"]


@pytest.mark.parametrize("name", BB_SIMPLE)
def test_simple_prefix_is_not_substituted(name):
    """P-16.3.2(a): an unsubstituted prefix is not substituted, locant or not."""
    assert is_substituted_substituent(name) is False, (
        f"{name!r} is an unsubstituted prefix (P-16.3.2(a)) and must not be "
        f"classed substituted"
    )


@pytest.mark.parametrize("name", BB_SUBSTITUTED)
def test_substituted_prefix_is_substituted(name):
    """P-16.3.5(a): a compound/complex (i.e. substituted) prefix."""
    assert is_substituted_substituent(name) is True, (
        f"{name!r} carries a detachable substituent prefix of its own "
        f"(P-16.3.2(c) / P-16.3.5(a)) and must be classed substituted"
    )


@pytest.mark.parametrize("name", BB_SIMPLE)
def test_simple_prefix_takes_the_simple_multiplier(name):
    """P-16.3.2(a) end-to-end: di/tri, never bis/tris."""
    assert get_multiplier_prefix(2, name).rstrip("-") == "di"
    assert get_multiplier_prefix(3, name).rstrip("-") == "tri"


@pytest.mark.parametrize("name", BB_SUBSTITUTED)
def test_substituted_prefix_takes_the_bis_multiplier(name):
    """P-16.3.5(a) end-to-end: bis/tris, never di/tri."""
    assert get_multiplier_prefix(2, name) == "bis"
    assert get_multiplier_prefix(3, name) == "tris"


@pytest.mark.parametrize("name", BB_AMBIGUOUS)
def test_ambiguity_leg_still_takes_bis(name):
    """P-16.3.2(d): preserved untouched by the predicate swap."""
    assert get_multiplier_prefix(2, name) == "bis"


# ---------------------------------------------------------------------------
# The two questions must stay SEPARATE.  `is_complex_substituent` keeps its own
# job (enclosure, P-16.3.4); only the multiplier moved off it.
# ---------------------------------------------------------------------------

def test_the_two_predicates_provably_disagree():
    """The whole point: enclosure and multiplier are different questions.

    `propan-2-yl` is enclosed (P-16.3.4(a)) AND simply multiplied
    (P-16.3.2(a)) — `di(propan-2-yl)`.  A single predicate cannot say both, and
    conflating them is what produced `bis(propan-2-yl)`.
    """
    assert is_complex_substituent("propan-2-yl") is True    # -> parentheses
    assert is_substituted_substituent("propan-2-yl") is False  # -> `di`
    # ... and the reverse disagreement, on a name with no digit to key off.
    assert is_complex_substituent("hydroxymethyl") is False
    assert is_substituted_substituent("hydroxymethyl") is True


def test_enclosure_predicate_is_unchanged_for_the_italicized_carve_out():
    """`tert-butyl` stays simple for BOTH questions (P-16.3.2(a) names it)."""
    assert is_complex_substituent("tert-butyl") is False
    assert is_substituted_substituent("tert-butyl") is False
    # and the P-16.2.4.1(d) hyphen still comes from the primitive
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
    """BB :25721 `1,4-di(propan-2-yl)cyclohexane (PIN)` vs
    BB :25793 `1,4-bis(2-chloropropan-2-yl)benzene (PIN)`.

    One chloro apart; the multiplier flips.  Nothing about the digit.
    """
    assert get_multiplier_prefix(2, "propan-2-yl") == "di"
    assert get_multiplier_prefix(2, "2-chloropropan-2-yl") == "bis"


def test_bb_tri_decyl_negative_control():
    """BB :25715 `1,3,5-tri(decyl)cyclohexane (PIN)
    (not 1,3,5-tris(decyl)cyclohexane)` — verbatim negative control."""
    assert get_multiplier_prefix(3, "decyl") == "tri"


# ---------------------------------------------------------------------------
# P-16.3.3(a) — SUFFIXES.  `get_multiplier_prefix` is called for suffixes too,
# and this is where the first version of this change broke 10 gold rows.
# ---------------------------------------------------------------------------

# `**P-16.3.3**` (BlueBookV2.md:7038): "*The basic numerical prefixes 'di',
# 'tri', 'tetra', etc. are used to indicate a multiplicity of: (a) functional and
# cumulative suffixes*", examples `diol`, `dicarboxylic!acid`, `disulfonic!acid`.
FUNCTIONAL_SUFFIXES = [
    "carboxylic acid", "sulfonic acid", "sulfinic acid", "phosphonic acid",
    "ol", "amide", "carboxamide", "nitrile", "carbonitrile", "carbaldehyde",
    "carboxylate", "sulfonate", "oate",
]


@pytest.mark.parametrize("suffix", FUNCTIONAL_SUFFIXES)
def test_a_functional_suffix_always_takes_the_basic_multiplier(suffix):
    """P-16.3.3(a).  A suffix is NEVER decomposed for substituted-ness.

    REGRESSION GUARD.  The first version of `is_substituted_substituent`
    decomposed `carboxylic acid` as `carboxy` + `lic acid` and `sulfonic acid` as
    `sulfo` + `nic acid`, so it emitted `benzene-1,4-biscarboxylic acid` and
    `4-methylbenzene-1,3-bissulfonic acid`.  That broke 10 gold targets and was
    caught only by measuring the flip set before committing.
    """
    assert is_substituted_substituent(suffix) is False
    assert get_multiplier_prefix(2, suffix) == "di"
    assert get_multiplier_prefix(3, suffix) == "tri"


def test_the_thioic_acid_suffix_exception_takes_bis():
    """P-16.3.5(b) is an EXPLICIT exception to P-16.3.3(a).

    BB `:7104`: `bis(thioic acid)` is a "multiple preferred suffix" describing
    two -C(O/S)H suffixes, "*whereas dithioic acid describes a -CSSH suffix*",
    and `bis(dithioic acid)` ... "*not didithoic acid*".
    """
    assert get_multiplier_prefix(2, "thioic acid") == "bis"
    assert get_multiplier_prefix(2, "dithioic acid") == "bis"


def test_an_already_enclosed_prefix_is_decided_on_its_interior():
    """A fully enclosed token is one component; the marks are not the answer.

    Producers hand this primitive both bare and already-enclosed names, so
    `(2-chloroethyl)` must reach the same verdict as `2-chloroethyl`.  Without
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
    `bis`.  (BB does write `bis(methylene)`, but that is the P-16.3.2(d)
    AMBIGUITY leg — `dimethylene` could read as -CH2CH2- — not substitution, and
    it must come from `CATENATION_AMBIGUOUS_PREFIXES`, not from here.)
    """
    assert is_substituted_substituent("methylene") is False
    assert is_substituted_substituent("ethylene") is False
    assert is_substituted_substituent("carbonyl") is False


def test_an_italicized_prefix_on_a_COMPOUND_remainder_is_substituted():
    """`tert-butylsulfanyl` reduces to the compound `butylsulfanyl`.

    P-16.3.2(a) makes `tert-butyl` simple, but the carve-out is about the
    ITALICIZED PREFIX only — it does not make everything it leads simple.  So
    this takes `bis` (P-16.3.5(a)) and, being enclosed, takes NO hyphen:
    `**P-16.2.4.2**` (`BlueBookV2.md:6968`) "*No hyphen is placed after a
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
    # ...while the SIMPLE italicized prefix keeps its P-16.2.4.1(d) hyphen.
    assert get_multiplier_prefix(2, "tert-butyl") == "di-"


def test_a_chain_root_beginning_with_a_multiplier_is_not_substituted():
    """P-16.3.4(c)/(d): `tridecyl` is C13, not `tri` + `decyl`.

    This is the trap that defeats a peel-the-multiplier string rule, and the
    reason the full name is tested against the chain-root table FIRST.
    """
    assert is_substituted_substituent("tridecyl") is False
    assert get_multiplier_prefix(2, "tridecyl") == "di"
    # contrast: `trimethylsilyl` really is a multiplied substituent prefix
    assert is_substituted_substituent("trimethylsilyl") is True
