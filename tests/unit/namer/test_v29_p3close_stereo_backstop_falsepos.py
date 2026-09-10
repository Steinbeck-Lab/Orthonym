"""Item D — the stereo backstop was crying wolf 1147 times.

`_final_stereo_check`'s log fired **1147 times over 122 distinct names** in the
certified full gate at, every one with `handler='unknown'`. It is
detection-only, so nothing shipped wrong — but the diagnostic was accusing
names that are configurationally COMPLETE.

Root cause: `needs_stereo_injection`'s descriptor vocabulary is `{R,S,r,s,E,Z}`
inside parentheses, plus `alpha|beta-D/L-` (the ANOMERIC sugar form only) and a
bare `D-`/`L-` token. The Blue Book's PIN descriptor vocabulary is wider —
`## **P-91.2.1.2.2** Stereodescriptors used in the nomenclature of natural
products` (`the Blue Book`), items at `:44628-44632`, additionally
legitimises italic cyclitol/carbohydrate prefixes and, item (iii), "*the
stereodescriptors 'alpha', 'beta'... to describe the absolute configuration of
alkaloids, terpenes and terpenoids, steroids*". On top of that,
`### **P-101.2.6**` (`:51047`) makes a stereoparent's own name carry its
skeleton's configuration.

So four channels express configuration invisibly to the counter, and the
backstop flagged all four.

**This is a LOG-SITE exemption and changes no emitted name.**
`needs_stereo_injection` gates the injection path too, and
`rules/stereochemistry.py:456` carries an explicit "*Per, do NOT
broaden*" — widening it would stop injection firing on names that currently
receive a correct block, trading a noisy log for real stereo loss.
"""

import logging

import pytest

from orthonym.namer import _stereo_is_implied_by_name

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# The families that ARE configurationally complete -> the warning was false.
# Every name below is a verbatim string from the certified gate log.
# ---------------------------------------------------------------------------

# Family 4 — the name ALREADY carries alpha/beta descriptors (P-101.2.6).
STEROID_ALPHA_BETA = [
    "cholest-5-en-3β-yl hydrogen sulfate",
    "3β-hydroxy-5α-androstan-17β-yl hydrogen sulfate",
    "5α-cholestan-3β-ol",
    "5β-cholestan-3α-ol",
    "17β-hydroxyandrost-4-en-3-one",
    "5α-androstane-3β,17β-diol",
    "androst-5-en-3β-ol",
    "3β-methoxy-5β-androstan-17-one",
    "5α-pregnane-3β,20-diol",
    "3-oxo-5α-androstan-17β-yl hydrogen sulfate",
    "17β-methoxy-5α-androstan-3α-ol",
]

# Family 1 — the italic prefix IS the descriptor (P-104.2.1).
INOSITOLS = [
    "myo-inositol", "scyllo-inositol", "neo-inositol", "muco-inositol",
    "epi-inositol", "allo-inositol", "cis-inositol",
]

# Family 3 — retained nucleosides/nucleotides (P-105.1 / P-106.1).
NUCLEOSIDES = [
    "adenosine", "inosine",
    "5'-adenylic acid", "5'-inosinic acid",
    "2'-deoxyadenosine", "2'-deoxyguanosine",
    "1-methyladenosine", "N6-methyladenosine", "7-methylguanosine",
    "5-methyluridine", "5-methylcytidine", "1-methylguanosine",
    "adenosine 5'-(tetrahydrogen triphosphate)",
    "adenosine 5'-(trihydrogen diphosphate)",
    "adenosine 2',3',5'-triacetate",
    "adenosine 5'-acetate",
]

# Family 5 — stereoparent hydrides (P-101.2.1.3 / P-101.2.6 / P-101.3.6).
STEREOPARENTS = [
    "trichothecane", "taxane", "ursane", "oleanane", "lupane", "hopane",
    "tropane", "yohimban", "vincane", "strychnidine", "stigmastane",
    "spirostan", "sparteine", "solanidane", "rosane", "prostane",
    "thromboxane", "podocarpane", "pimarane", "picrasane", "lanostane",
    "labdane", "kaurane", "ibogamine", "guaiane", "gibbane", "germacrane",
    "galanthan", "furostan", "eudesmane", "erythrinan", "drimane",
    "dammarane", "corynan", "conanine", "cevane", "cadinane", "beyerane",
    "atisane", "aspidospermidine", "aristolane", "ambrosane", "abietane",
]

# Family 2b — sphinganine, implied by definition (P-107.4.3.1).
SPHINGOIDS = ["sphinganine"]


@pytest.mark.parametrize("name", STEROID_ALPHA_BETA)
def test_steroid_alpha_beta_names_already_carry_descriptors(name):
    """P-101.2.6: a locant-prefixed alpha/beta IS a stereodescriptor."""
    assert _stereo_is_implied_by_name(name) is True


@pytest.mark.parametrize("name", INOSITOLS)
def test_inositol_prefix_is_the_descriptor(name):
    """P-104.2.1: "Names denoted by the prefixes are preferred"."""
    assert _stereo_is_implied_by_name(name) is True


@pytest.mark.parametrize("name", NUCLEOSIDES)
def test_retained_nucleoside_stereo_is_implied(name):
    """P-105.1 / P-106.1: retained names, BB examples descriptor-free."""
    assert _stereo_is_implied_by_name(name) is True


@pytest.mark.parametrize("name", STEREOPARENTS)
def test_stereoparent_name_carries_its_configuration(name):
    """P-101.2.6: "...usually implies the absolute configuration of all
    chirality centers... without further specification"."""
    assert _stereo_is_implied_by_name(name) is True


@pytest.mark.parametrize("name", SPHINGOIDS)
def test_sphinganine_is_implied_by_definition(name):
    """P-107.4.3.1: the retained name is for the amino alcohol "having the
    described absolute configuration"."""
    assert _stereo_is_implied_by_name(name) is True


# ---------------------------------------------------------------------------
# The NEGATIVE half — this is what makes the exemption a claim and not a mute
# button. A too-broad exemption would silence a REAL stereo gap.
# ---------------------------------------------------------------------------

# The free amino acids: the warning here is TRUE. `P-103.1.3.1` (:54291)
# requires the alpha-carbon configuration to be designated D or L; `P-103.3.4`
# (:54715) scopes L-omission to PEPTIDES only; and Table 10.4 pairs each
# retained name with a `rel-` (RELATIVE) systematic equivalent (:54204,:54211),
# so the bare name is not enantiospecific. Deliberately NOT exempted.
# -CLEANUP: the last two were written HYPHENATED here. That spelling never
# shipped -- settled on the FUSED form that `## **P-103.1.3.2.2** Use of
# the prefix 'allo'` (:54320) writes (`allothreonine`), and the hyphenated italic
# `*allo*-` in the Blue Book is a CARBOHYDRATE/cyclitol prefix (:53011,:53021,
#:54890), a different device. The assertion below passes either way -- both forms
# are correctly un-exempt -- so nothing caught the drift; it is corrected because a
# stale string in a test reads as a claim about what the system emits.
#
# `cystine` stays in this list and stays correct: the predicate is asked about the
# BARE name, which genuinely lacks its descriptor. Production now emits
# `L-cystine`/`D-cystine` (P3-CLEANUP Item 1), and the backstop is silent on those.
GENUINE_AMINO_ACID_GAP = [
    "alanine", "serine", "cysteine", "cystine", "threonine", "isoleucine",
    "allothreonine", "alloisoleucine",
]


@pytest.mark.parametrize("name", GENUINE_AMINO_ACID_GAP)
def test_free_amino_acids_are_NOT_exempted(name):
    """A real gap must keep warning — P-103.1.3.1 requires D/L on a free
    amino acid, and P-103.3.4 licenses omission only inside a peptide."""
    assert _stereo_is_implied_by_name(name) is False, (
        f"{name!r} genuinely lacks its configurational descriptor; exempting "
        f"it would silence a real defect"
    )


def test_a_plain_systematic_name_is_not_exempted():
    """The exemption must not leak onto ordinary systematic names."""
    for n in ("butan-2-ol", "2-chlorobutane", "cyclohexane-1,2-diol",
              "1,1'-vinylenedibenzene", "hexan-3-yl acetate"):
        assert _stereo_is_implied_by_name(n) is False


def test_empty_and_none_are_not_exempted():
    assert _stereo_is_implied_by_name("") is False
    assert _stereo_is_implied_by_name(None) is False


def test_alpha_beta_needs_a_locant_to_count_as_a_descriptor():
    """A bare 'beta' with no locant is not the P-101.2.6 descriptor form.

    `β-D-glucopyranose` is already handled upstream by the anomeric pattern
    in `needs_stereo_injection`; a stray 'alpha'/'beta' inside an ordinary word
    must not buy an exemption.

    ⚠ -FINAL: `betaine` and `alphaprodine` do NOT pin what this test's name
    claims. Both are rejected by the trailing `\\b`, not by the `\\d+`, so
    mutating `\\d+(?:alpha|beta|xi)\\b` to `(?:alpha|beta|xi)\\b` left all 92
    tests green. `N-acetylalpha-neuraminic acid` -- a name the build really
    emits -- is the row that pins the locant, because `alpha` there IS followed by
    a word boundary.
    """
    assert _stereo_is_implied_by_name("betaine") is False
    assert _stereo_is_implied_by_name("alphaprodine") is False
    # M1 KILLER: locant-free `alpha` at a word boundary, in a live emission.
    assert _stereo_is_implied_by_name("N-acetylalpha-neuraminic acid") is False, (
        "a locant-free 'alpha' must not buy an exemption; without this row the "
        "\\d+ in the descriptor regex is unpinned"
    )


# ---------------------------------------------------------------------------
# -FINAL I11 — the three kept-VISIBLE names, and one anchor row per leg.
#
# Three over-broadening mutations previously survived all 92 tests of this file:
#
# M1 `\d+(?:alpha|beta|xi)\b` -> `(?:alpha|beta|xi)\b` 92 passed
# M2 `'sphing' in name` -> `'sph' in name` 92 passed
# M5 append `'itol','neuraminic'` to the nucleoside stems 92 passed
#
# M5 silenced verbatim the three names the change designates must stay VISIBLE,
# and none of the three appeared anywhere in the file. They do now.
# ---------------------------------------------------------------------------

KEPT_VISIBLE = [
    # `xylitol` -- the project's own `data/sugar_names.py` comment cites the
    # governing rule, so "no Blue Book text found" was wrong; either way the name
    # carries no configurational descriptor and must keep warning.
    "xylitol",
    # `## **P-103.1.3.1** The stereodescriptors 'D' and 'L'` (:54291) requires the
    # alpha-carbon configuration to be designated on a free amino acid, and the
    # neuraminic acids are not peptides, so P-103.3.4's omission licence
    # (:54715) does not reach them.
    "N-acetylneuraminic acid",
    "N-glycolylneuraminic acid",
]


@pytest.mark.parametrize("name", KEPT_VISIBLE)
def test_the_kept_visible_names_are_never_exempted(name):
    """M5 KILLER. These three are the names the change says must stay visible."""
    assert _stereo_is_implied_by_name(name) is False, (
        f"{name!r} must keep warning; exempting it silences a real gap and is "
        f"exactly what an over-broad stem list does"
    )


def test_the_sphingoid_leg_is_anchored_to_the_retained_name():
    """M2 KILLER: a `sph`-containing non-sphingoid must not be exempted.

    Every `phosph...` name in the corpus contains `sph`, so the mutation from
    `'sphing'` to `'sph'` is not academic -- it exempts a whole live class.
    """
    for n in ("2-aminoethylphosphonic acid", "triphosphane",
              "phosphanyl", "phosphonooxyacetic acid"):
        assert _stereo_is_implied_by_name(n) is False, n
    #...while the two spellings the Blue Book retains still are.
    assert _stereo_is_implied_by_name("sphinganine") is True
    assert _stereo_is_implied_by_name("(4E)-sphing-4-enine") is True
    #...and a sphingoid-ADJACENT non-Blue-Book name is not (P-107.4.3.1 names
    # `sphinganine`; `sphingosine` is "not a Blue Book name" per the project's
    # own `data/natural_products.py` comment).
    assert _stereo_is_implied_by_name("sphingomyelin") is False


def test_the_steroid_leg_requires_a_stereoparent_not_just_a_descriptor():
    """The alpha/beta leg must be anchored to a PARENT.

    Before this, any name containing `3β` was exempted with no steroid
    anywhere in it.
    """
    assert _stereo_is_implied_by_name("3β-hydroxyoctadecanoic acid") is False
    #...while the real steroid derivatives stay exempt.
    for n in ("5α-cholestan-3β-ol",
              "cholest-5-en-3β-yl 2-hydroxypropanoate"):
        assert _stereo_is_implied_by_name(n) is True, n
    # M1 KILLER on the steroid leg. `**P-101.2.6**` gives the descriptor form
    # WITH a locant (`3β`, `5α`); an unlocanted `beta-` designates no
    # centre, so a steroid carrying one is not self-describing and must keep
    # warning. Under the locant-free mutation this is exempted.
    assert _stereo_is_implied_by_name("beta-cholestan-3-ol") is False, (
        "an unlocanted 'beta-' is not the P-101.2.6 descriptor form even on a "
        "genuine steroid stem"
    )


def test_the_steroid_stem_must_start_at_a_token_boundary():
    """A plain substring test on 4-character stems is too loose.

    `tropane` contributes the stem `trop`, which occurs inside
    `1,6-anhydro-β-D-altropyranose` -- a carbohydrate, measured as the single
    spurious hit across 1963 gold + pack + table names.
    """
    assert _stereo_is_implied_by_name("1,6-anhydro-β-D-altropyranose") is False
    #...and the genuine tropane stem still matches at a boundary.
    assert _stereo_is_implied_by_name("tropan-3β-ol") is True


def test_the_nucleoside_leg_is_derived_from_the_producer_table():
    """I10: the stems must come FROM `rules/nucleosides.py`, not a copy of it.

    A hand-written duplicate is what let a mutation append two unrelated stems.
    Deriving it means there is no literal tuple to append to, and a drift between
    the two tables becomes impossible rather than merely unlikely.
    """
    from orthonym.namer import _IMPLIED_STEREO_NUCLEOSIDE_STEMS
    from orthonym.rules.nucleosides import _NUCLEOSIDE_STEM

    produced = {v.lower() for v in _NUCLEOSIDE_STEM.values() if isinstance(v, str)}
    assert produced, "producer table empty -- test would pass vacuously"
    missing = {p for p in produced if p not in _IMPLIED_STEREO_NUCLEOSIDE_STEMS}
    assert not missing, (
        f"every name the nucleoside constructor can emit must be covered; "
        f"missing {sorted(missing)}"
    )
    # anchored, so an unrelated longer word is not swallowed...
    assert _stereo_is_implied_by_name("uridinelike-compound") is False
    #...but an acylated derivative still is (BB:54981
    # `2',3',5'-tri-O-acetyladenosine`).
    assert _stereo_is_implied_by_name("2',3',5'-tri-O-acetyladenosine") is True


# ---------------------------------------------------------------------------
# The log call itself: the name must no longer be truncated.
# ---------------------------------------------------------------------------

def test_exempted_name_produces_NO_warning_through_the_real_check(caplog):
    """The exemption must be WIRED, not merely present.

    Exercises `_final_stereo_check` end-to-end so a mutation of the call site
    (rather than of the predicate) is caught.
    """
    from rdkit import Chem
    from orthonym.namer import _final_stereo_check

    mol = Chem.MolFromSmiles("C[C@@H](N)C(=O)O")   # one carbon R/S centre
    with caplog.at_level(logging.WARNING, logger="orthonym.namer"):
        out = _final_stereo_check(mol, "myo-inositol", handler="unknown")
    assert out == "myo-inositol"
    assert "Stereo backstop" not in "\n".join(
        r.getMessage() for r in caplog.records
    ), "an implied-stereo name must not trigger the backstop warning"


def test_NON_exempted_name_STILL_warns_through_the_real_check(caplog):
    """...and the exemption must not be a mute button.

    `alanine` is a genuine gap (P-103.1.3.1), so the warning must survive. This
    is the test that fails if the call site is short-circuited to always exempt.
    """
    from rdkit import Chem
    from orthonym.namer import _final_stereo_check

    mol = Chem.MolFromSmiles("C[C@@H](N)C(=O)O")
    with caplog.at_level(logging.WARNING, logger="orthonym.namer"):
        out = _final_stereo_check(mol, "alanine", handler="unknown")
    assert out == "alanine"
    assert "Stereo backstop" in "\n".join(
        r.getMessage() for r in caplog.records
    ), "a real stereo gap must keep warning"


def test_backstop_log_does_not_truncate_the_name(caplog):
    """`namer.py` logged `name[:50]`, so every longer name was cut mid-word.

    The certified gate log carries
    `'3β-hydroxy-5α-androstan-17β-yl hydrogen '` — 49 characters and
    a dangling word. A diagnostic that mangles its own subject wastes every
    reader's time, and it silently corrupted the distinct-name census (the
    truncated pair collapses two different sulfate esters onto one string).
    """
    from rdkit import Chem
    from orthonym.namer import _final_stereo_check

    # A molecule with a defined R/S centre whose name carries no descriptor and
    # is longer than 50 characters, and is in NO exempted family.
    mol = Chem.MolFromSmiles("C[C@H](O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
    long_name = (
        "a-deliberately-long-systematic-name-well-over-fifty-characters-long"
    )
    assert len(long_name) > 50
    with caplog.at_level(logging.WARNING, logger="orthonym.namer"):
        out = _final_stereo_check(mol, long_name, handler="unknown")
    assert out == long_name          # detection-only: never mutates the name
    logged = "\n".join(r.getMessage() for r in caplog.records)
    if "Stereo backstop" in logged:
        assert long_name in logged, (
            "the backstop truncated the name it is complaining about"
        )
