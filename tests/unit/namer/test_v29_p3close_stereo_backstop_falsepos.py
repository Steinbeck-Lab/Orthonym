"""Item D — the stereo backstop was crying wolf 1147 times.

`_final_stereo_check`'s log fired **1147 times over 122 distinct names** in the
certified full gate at `e5233ef8`, every one with `handler='unknown'`.  It is
detection-only, so nothing shipped wrong — but the diagnostic was accusing
names that are configurationally COMPLETE.

Root cause: `needs_stereo_injection`'s descriptor vocabulary is `{R,S,r,s,E,Z}`
inside parentheses, plus `alpha|beta-D/L-` (the ANOMERIC sugar form only) and a
bare `D-`/`L-` token.  The Blue Book's PIN descriptor vocabulary is wider —
`## **P-91.2.1.2.2** Stereodescriptors used in the nomenclature of natural
products` (`BlueBookV2.md:44626`), items at `:44628-44632`, additionally
legitimises italic cyclitol/carbohydrate prefixes and, item (iii), "*the
stereodescriptors 'alpha', 'beta' ... to describe the absolute configuration of
alkaloids, terpenes and terpenoids, steroids*".  On top of that,
`### **P-101.2.6**` (`:51047`) makes a stereoparent's own name carry its
skeleton's configuration.

So four channels express configuration invisibly to the counter, and the
backstop flagged all four.

**This is a LOG-SITE exemption and changes no emitted name.**
`needs_stereo_injection` gates the injection path too, and
`rules/stereochemistry.py:456` carries an explicit "*Per D-20, do NOT
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
    "cholest-5-en-3beta-yl hydrogen sulfate",
    "3beta-hydroxy-5alpha-androstan-17beta-yl hydrogen sulfate",
    "5alpha-cholestan-3beta-ol",
    "5beta-cholestan-3alpha-ol",
    "17beta-hydroxyandrost-4-en-3-one",
    "5alpha-androstane-3beta,17beta-diol",
    "androst-5-en-3beta-ol",
    "3beta-methoxy-5beta-androstan-17-one",
    "5alpha-pregnane-3beta,20-diol",
    "3-oxo-5alpha-androstan-17beta-yl hydrogen sulfate",
    "17beta-methoxy-5alpha-androstan-3alpha-ol",
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
    chirality centers ... without further specification"."""
    assert _stereo_is_implied_by_name(name) is True


@pytest.mark.parametrize("name", SPHINGOIDS)
def test_sphinganine_is_implied_by_definition(name):
    """P-107.4.3.1: the retained name is for the amino alcohol "having the
    described absolute configuration"."""
    assert _stereo_is_implied_by_name(name) is True


# ---------------------------------------------------------------------------
# The NEGATIVE half — this is what makes the exemption a claim and not a mute
# button.  A too-broad exemption would silence a REAL stereo gap.
# ---------------------------------------------------------------------------

# The free amino acids: the warning here is TRUE.  `P-103.1.3.1` (:54291)
# requires the alpha-carbon configuration to be designated D or L; `P-103.3.4`
# (:54715) scopes L-omission to PEPTIDES only; and Table 10.4 pairs each
# retained name with a `rel-` (RELATIVE) systematic equivalent (:54204, :54211),
# so the bare name is not enantiospecific.  Deliberately NOT exempted.
GENUINE_AMINO_ACID_GAP = [
    "alanine", "serine", "cysteine", "cystine", "threonine", "isoleucine",
    "allo-threonine", "allo-isoleucine",
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

    `beta-D-glucopyranose` is already handled upstream by the anomeric pattern
    in `needs_stereo_injection`; a stray 'alpha'/'beta' inside an ordinary word
    must not buy an exemption.
    """
    assert _stereo_is_implied_by_name("betaine") is False
    assert _stereo_is_implied_by_name("alphaprodine") is False


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

    `alanine` is a genuine gap (P-103.1.3.1), so the warning must survive.  This
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
    `'3beta-hydroxy-5alpha-androstan-17beta-yl hydrogen '` — 49 characters and
    a dangling word.  A diagnostic that mangles its own subject wastes every
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
