"""a phase CFR dispatch consistency — byte-identical vs Plan-01 frozen baseline.

Per internal notes + line 230-234: ≥ 33 tests asserting ``Orthonym.name(smi)``
produces the exact ``name_output`` from ``tests/canary/canary_pre_cfr_158.csv``
(Plan-01 frozen pre-CFR baseline).

Per internal notes honest-fail-on-data: any single byte difference is a
HARD-gate failure. NO ``@pytest.mark.xfail`` markers; NO special-case
allowances. The fix is upstream (predicate purity per OR priority
ordering per) — not relaxation per + +.

The CSV-diff layer is verified separately by ``verify_cfr_byte_identical.py
--mode post`` + ``diff -q pre.csv post.csv``. This pytest tier asserts NAME
equality only; the CSV-diff catches OPSIN-parse + InChI-L1 column drift that
pytest equality misses.

Two test functions:

- ``test_cfr_dispatch_byte_identical`` — parametrized over one representative
  fixture per StoutClass (StoutClass.NAME ID): 18 tests covering every active
  enum member. Mined from the live canary corpus via Plan-03 dev bench.
- ``test_cfr_supplementary_byte_identical`` — parametrized over 15 fixtures
  sampled stride-wise from PRE_CANARY_ROWS for cross-class coverage.

Total: 33 tests at the floor; the parametrize expansion yields exactly this.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import pytest

from orthonym import name_compound
from orthonym.routing.dispatch_table import StoutClass


CANARY_PRE_CSV = Path(__file__).parent.parent / "canary" / "canary_pre_cfr_158.csv"


def _load_pre_canary() -> List[Tuple[str, str, str, str]]:
    """Load ``(canary_tier, fixture_id, smiles, expected_name)`` from frozen baseline.

    Per PATTERNS anti-pattern "No silent missing-canary-fixture handling":
    raise FileNotFoundError if the frozen baseline is missing — Plan-01 MUST
    have shipped the CSV. If it is absent, the test infrastructure is broken,
    not the CFR substrate.
    """
    if not CANARY_PRE_CSV.exists():
        raise FileNotFoundError(
            f"Frozen pre-CFR baseline missing: {CANARY_PRE_CSV}. "
            f"Plan-01 must have shipped this artifact."
        )
    rows = []
    with CANARY_PRE_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append((
                r["canary_tier"],
                r["fixture_id"],
                r["smiles_input"],
                r["name_output"],
            ))
    return rows


PRE_CANARY_ROWS: List[Tuple[str, str, str, str]] = _load_pre_canary()


# ---------------------------------------------------------------------------
# Per-StoutClass byte-identical tests
#
# One (smiles, expected_name) per StoutClass mined from the live canary
# corpus + Plan-03 development bench. Identical seed-list to
# tests/unit/routing/test_dispatcher.py::STOUTCLASS_REPRESENTATIVES so a
# regression in either layer surfaces against the same fixture.
# ---------------------------------------------------------------------------


CFR_DISPATCH_CANARY: "OrderedDict[StoutClass, Tuple[str, str]]" = OrderedDict([
    (StoutClass.SALT,                    ("[Na+].[Cl-]",            "sodium chloride")),
    (StoutClass.RADICAL,                 ("[CH3]",                  "methyl")),
    (StoutClass.ZWITTERION,              ("[NH3+]CC(=O)[O-]",       "glycine")),
    (StoutClass.ANION_RETAINED,          ("CC(=O)[O-]",             "acetate")),
    # Commit 12541211e (alkylammonium -> substitutive aminium PIN):
    # (the Blue Book) 'N,N,N-trimethylmethanaminium (PIN) tetramethylammonium';
    # 'Cation and anion names' (:26660): "Method (1) leads to preferred IUPAC names".
    (StoutClass.CATION_RETAINED,         ("C[N+](C)(C)C",           "N,N,N-trimethylmethanaminium")),
    (StoutClass.ANION_SMALL,             ("C(=O)([O-])CCCCCC",      "heptanoate")),
    (StoutClass.POLY_ANION,              ("[O-]C(=O)CCCC(=O)[O-]",  "pentanedioate")),
    (StoutClass.MULTI_COMPONENT_NEUTRAL, ("CCO.OCC",                "ethanol ethanol")),
    (StoutClass.MULTIPLICATIVE,          ("c1ccc(Cc2ccccc2)cc1",    "1,1'-methylenedibenzene")),
    (StoutClass.CARBOHYDRATE_LOOKUP,     ("OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
                                          "α-D-glucopyranose")),
    (StoutClass.NATURAL_PRODUCT,         (
        "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]34C)[C@@H]1CC[C@@H]2O",
        "(8R,9S,10S,13S,14S,17S)-17-hydroxyandrost-4-en-3-one",
    )),
    #: peptide PIN is the SUBSTITUTIVE form (V38-PEPTIDE-PIN-VERDICT.md); RT verified.
    (StoutClass.PEPTIDE,                 ("NCC(=O)NCC(=O)O",        "(2-aminoacetamido)acetic acid")),
    (StoutClass.RETAINED_NAME,           ("CCO",                    "ethanol")),
    # Commit a6cadc255 (-p3reg-I12): a free amino acid keeps its L; 'The
    # stereodescriptors D and L' (the Blue Book). Gold row.0 pins 'L-alanine'.
    (StoutClass.AMINO_ACID,              ("C[C@H](N)C(=O)O",        "L-alanine")),
    # Commit 36464b71c (simple 1,2-O ethers are substitutive, not oxa-replacement):
    # 'Systematic names of ethers' (the Blue Book) '(1) 1,2-dimethoxyethane
    # (PIN)'. The row therefore no longer reaches the skeletal-replacement producer; it keeps
    # pinning the dispatch byte-identity of the PIN name for this fixture.
    (StoutClass.SKELETAL_REPLACEMENT,    ("COCCOC",                 "1,2-dimethoxyethane")),
    (StoutClass.CYCLOPHANE,              ("C1CCc2ccccc2CCCc2ccccc21",
                                          "[3.3]orthocyclophane")),
    #: was 'propyl palmitate'. The ester acyl word follows the PIN
    # acid stem -- (the Blue Book) retains only formic/oxalic/
    # acetic/benzoic/oxamic as PINs, (:29860) sends the rest to general
    # nomenclature, and:29787 prints '(PIN)' on 'hexadecanoic acid'. This row
    # pins DISPATCH byte-identity, so the spelling is incidental to its purpose.
    (StoutClass.DECOMPOSITION_PRE_GENERAL,
                                          ("CCCCCCCCCCCCCCCC(=O)OCCC",
                                          "propyl hexadecanoate")),
    (StoutClass.GENERAL,                 ("CCCCCCCO",               "heptan-1-ol")),
])


# Rows whose frozen value is NOT the PIN and whose PIN the engine does not build
# yet. The expected value is the PIN, under a strict xfail naming the missing
# producer, so the row turns red (XPASS) the day the PIN ships and the frozen
# non-PIN can never be re-accepted. Keyed by StoutClass name (per-class test) or
# fixture id (supplementary test).
_PIN_NOT_BUILT = {
    # Suite fix j4 (TRIAGE g3 C10a / g6 C20). Two benzene rings ortho-fused to a
    # 10-membered alicyclic ring: (1) (the Blue Book) needs a
    # mancude ring attached "at nonadjacent ring positions";
    # (:23835-23841) "Mancude systems attached to adjacent atoms of an alicyclic
    # ring are either fused systems or bridged fused systems [...] A cyclophane
    # name is not allowed" (BB class example:23839 '5,6,7,8,9,10,11,12,13,14,15,
    # 16-dodecahydrobenzo[14]annulene (PIN)'). The frozen '[3.3]orthocyclophane'
    # is a general-nomenclature phane name; the phane producer now declines, the
    # PIN tier abstains, the best-effort tier ships the OPSIN-exact von Baeyer
    # name. PIN spelling ASSUMED (OPSIN full-InChIKey exact).
    "CYCLOPHANE": (
        "5,6,7,12,13,14-hexahydrodibenzo[a,f][10]annulene",
        "needs the hydro fusion name for mancude rings ortho-fused to a large "
        "alicyclic ring (P-52.2.5.2.1); PIN spelling ASSUMED; TODO "
        ".planning/preexisting-triage/TRIAGE.md 'Suite fix -- j4-pin-labels-a'"),
    # Suite fix j4 (TRIAGE g3 C10b). A glycerol diester ether:
    # (:31836) "Method (1) generates preferred IUPAC names" (:31840 'propane-
    # 1,2,3-triyl 1,2-diacetate 3-propanoate (PIN)'). The engine cites both
    # esters as acyloxy prefixes on 'propane' (OPSIN-exact, label now below the
    # PIN tier); the frozen value is not a parseable name. The method-(1) PIN is
    # not parseable by OPSIN 2.9.0, spelling ASSUMED.
    "test_canary_name_stability_228": (
        "(2R)-3-(octadecyloxy)propane-1,2-diyl 2-[(9Z,12Z)-octadeca-9,12-"
        "dienoate] 1-tetracosanoate",
        "needs the P-65.6.3.3.3.2 method-(1) functional-class name for polyol "
        "esters with different acids; PIN spelling ASSUMED (not OPSIN-parseable); "
        "TODO .planning/preexisting-triage/TRIAGE.md 'Suite fix -- "
        "j4-pin-labels-a'"),
    # Suite fix j7 (TRIAGE g3 C05). Asn-Glu-Leu: the frozen 'L-asparaginyl-L-
    # glutamyl-L-leucine' and the current 'asparaginylglutamylleucine' are both the
    # Chapter peptide name, the Blue Book: 'L' is not
    # indicated for Table 10.4 residues), which is not a PIN:50943;
    # controller ruling: the PIN is the substitutive name, method (1)
    #:32995). j7 builds the nested amido prefix for tripeptides, but the Asn-Glu
    # acid fragment itself still fails (its raw name is a different molecule), so
    # the retained name ships best_effort. PIN spelling ASSUMED; OPSIN 2.9.0 full
    # InChIKey exact.
    "test_canary_rt75_595": (
        "(2S)-2-{(2S)-4-carboxy-2-[(2S)-2,4-diamino-4-oxobutanamido]butanamido}-"
        "4-methylpentanoic acid",
        "needs the substitutive name of the Asn-Glu acid fragment (the glutamyl "
        "residue's carboxy side chain); PIN spelling ASSUMED; TODO "
        ".planning/preexisting-triage/TRIAGE.md 'Suite fix -- j7-defects-misc'"),
    # The next two rows: the frozen value names a DIFFERENT molecule (OPSIN 2.9.0 full
    # InChIKey, inchi_l1_match False in the CSV), and the default (pin) tier DECLINES the
    # input -- 'PIN or decline' (the paper-conformant default tier, v1.0.2) -- so there is no
    # PIN string to pin. The expected value below is NOT a PIN: it is the best-effort tier's
    # name (--emit-tier best-effort: tier systematic_verified, is_pin false, OPSIN-exact),
    # the stand-in for 'a name that round-trips'. The strict xfail turns red (XPASS) only
    # if the default tier ships exactly this string; until a PIN is built the decline
    # stays unasserted here (a test must not accept the abstention).
    "test_canary_name_stability_398": (
        "4-(3-oxa-2-azaprop-2-en-2-ium-1-ylidene)-1-({[4-(3-oxa-2-azaprop-2-en-2-ium-1-"
        "ylidene)-1,4-dihydropyridin-1-yl]methoxy}methyl)-1,4-dihydropyridine",
        "cfr-frozen-wrong-molecule-default-tier-declines -- see "
        ".planning/preexisting-triage/TRIAGE-2026-10-09.md (frozen value is another "
        "molecule; default tier declines; no PIN built; stand-in is the best-effort name)"),
    "test_canary_name_stability_483": (
        "S-{2-[(3-{[(2R)-3-{[(3-{[(2R,3S,4R,5R)-5-(6-amino-9H-purin-9-yl)-4-hydroxy-3-"
        "(phosphonooxy)oxolan-2-yl]methoxy}-1,3-dihydroxy-1,3-dioxo-1λ5,3λ5-"
        "diphosphoxan-1-yl)oxy]methyl}-2-hydroxy-3-methyl-1-oxobutyl]amino}-1-oxopropyl)"
        "amino]ethyl} (9Z,12Z)-octadeca-9,12-dienethioate",
        "cfr-frozen-wrong-molecule-default-tier-declines -- see "
        ".planning/preexisting-triage/TRIAGE-2026-10-09.md (acyl-CoA: frozen value is an "
        "atom-dropped different molecule; default tier declines; no PIN built; stand-in is "
        "the best-effort name)"),
}


def _pin_not_built_param(key, node_id, *values):
    """pytest.param for a row: the PIN under a strict xfail when ``key`` is in
    ``_PIN_NOT_BUILT`` (the expected value is the LAST of ``values``)."""
    if key in _PIN_NOT_BUILT:
        pin, reason = _PIN_NOT_BUILT[key]
        return pytest.param(*values[:-1], pin, id=node_id,
                            marks=pytest.mark.xfail(strict=True, reason=reason))
    return pytest.param(*values, id=node_id)


@pytest.mark.parametrize(
    "class_id,smiles,expected_name",
    [_pin_not_built_param(c.name, c.name, c, s, n)
     for c, (s, n) in CFR_DISPATCH_CANARY.items()],
)
def test_cfr_dispatch_byte_identical(class_id, smiles, expected_name):
    """internal notes-CFR.md: name(smi) byte-identical to cascade output.

    Per internal notes: a single byte difference is a HARD-gate failure.
    Fix the upstream cause (predicate purity / priority ordering) — never
    relax this assertion via xfail or special-case allowance.
    """
    result = name_compound(smiles, style="pin")
    assert result == expected_name, (
        f"CFR DISPATCH REGRESSION (StoutClass={class_id.value}):\n"
        f"  SMILES:   {smiles}\n"
        f"  Expected: {expected_name}\n"
        f"  Got:      {result}"
    )


# ---------------------------------------------------------------------------
# Supplementary stride-wise sample from PRE_CANARY_ROWS
#
# 15 fixtures sampled at regular index strides across the 1282-row corpus
# to give cross-class coverage (rt75 + connectivity + name_stability tiers).
# Per internal notes: aim ≥ 33 total integration tests; 18 per-class +
# 15 supplementary = 33.
# ---------------------------------------------------------------------------


_STRIDE = max(1, len(PRE_CANARY_ROWS) // 15)
SUPPLEMENTARY_CANARY: List[Tuple[str, str, str, str]] = [
    PRE_CANARY_ROWS[i] for i in range(0, len(PRE_CANARY_ROWS), _STRIDE)
][:15]


# Frozen rows whose value names a DIFFERENT molecule (OPSIN: not the input), or
# the right molecule under a name that is not the PIN, and whose current name is the
# verified PIN. The CSV stays the frozen record
# (tests/integration/test_orgm_byte_identical_v18_canary.py md5-pins it against
# the post-CFR CSV); the expected value comes from here, with its evidence.
_SUPPLEMENTARY_REBASELINE = {
    # Same molecule, non-PIN spelling: the frozen '1-oxa-4-azacyclooctane' is a
    # skeletal-replacement name for an 8-membered ring. (the Blue Book):
    # "Hantzsch-Widman names, except for azine and oxine, are preferred IUPAC names for
    # both the unsaturated and saturated compounds" (3- to 10-membered rings), so the PIN of
    # C1CCNCCOC1 is the Hantzsch-Widman name. OPSIN 2.9.0: full InChIKey exact.
    "test_canary_name_stability_143": "1,4-oxazocane",
    # (the Blue Book, under ALPHANUMERICAL ORDER): "The name of a
    # prefix for a substituent is considered to begin with the first letter of its
    # complete name." '(4-hydroxy-3,5-dimethoxyphenyl)' begins with 'h' and
    # '(2,4,6-trihydroxy-3-methoxyphenyl)' with 't' (the multiplying prefix is part of a
    # compound prefix), so 'h' is cited first; the frozen value cites them t before h.
    # Same locants, same molecule: OPSIN 2.9.0 full InChIKey exact for both orders.
    "test_canary_rt75_0": (
        "3-(4-hydroxy-3,5-dimethoxyphenyl)-1-(2,4,6-trihydroxy-3-methoxyphenyl)"
        "propane-1,2-dione"),
    # N5b step 2. The frozen value is the abstention. A fused ring system with a ring heteroatom
    # has its fusion name, (the Blue Book, "Five-membered ring requirement"):
    # naphtho[2,3-c]furan is a two-component fusion name,:11903). OPSIN 2.9.0:
    # full InChIKey and canonical SMILES exact; labelled pin_verified.
    "test_canary_name_stability_313": (
        "5,7,8-trihydroxy-6-methoxy-1-methylnaphtho[2,3-c]furan-4,9-dione"),
    # Suite fix j4 (TRIAGE g3 C10c). (the Blue Book): "the
    # nesting order is as follows: {[({})]}";: the
    # stereodescriptor's parentheses count. OPSIN 2.9.0: full InChIKey and
    # canonical SMILES exact; the frozen value is another molecule.
    "test_canary_name_stability_58": (
        "4-[4-(4-{(2S,3R)-4-amino-3-methoxy-2-[4-(4-nitrobenzamido)benzamido]-"
        "4-oxobutanamido}benzamido)-2-hydroxy-3-[(propan-2-yl)oxy]benzamido]-"
        "3-ethoxybenzoic acid"),
    # Suite fix j6 (TRIAGE g3 C17b). The frozen '(3R)-9-(1,3-dioxolan-5-yl)-3,7-
    # dimethylnona-1,6-dien-3-ol' drops the four ring methyls (OPSIN: another
    # molecule; '1,3-dioxolan-5-yl' also misnumbers the ring). The PIN tier had
    # abstained: a decorated saturated chalcogen heteromonocycle had no stem.
    # Hantzsch-Widman names are PINs for saturated rings,
    # the Blue Book); heteroatoms lowest, then the free valence;
    # the chain carries the -OH suffix,:18875). OPSIN 2.9.0: full
    # InChIKey and canonical SMILES exact.
    "test_canary_rt75_170": (
        "(3R)-3,7-dimethyl-9-(2,2,5,5-tetramethyl-1,3-dioxolan-4-yl)nona-1,6-"
        "dien-3-ol"),
    # Suite fix j7 (TRIAGE g3 C05). Thr-Val-Lys: the peptide name (frozen with L-,
    # current without,:54720) is not a PIN:50943; controller
    # ruling); the tripeptide now takes the substitutive name, the nested amido
    # prefix built from the substitutive name of its dipeptide acid fragment
    # method (1),:32995). OPSIN 2.9.0: full InChIKey and canonical
    # SMILES exact; labelled pin_verified.
    "test_canary_rt75_680": (
        "(2S)-6-amino-2-{(2S)-2-[(2S,3R)-2-amino-3-hydroxybutanamido]-3-methyl"
        "butanamido}hexanoic acid"),
}


def _supplementary_id(row: Tuple[str, str, str, str]) -> str:
    """Pytest test ID from canary row (tier_fixture-id)."""
    return f"{row[0]}_{row[1]}"


@pytest.mark.parametrize(
    "tier,fixture_id,smiles,expected_name",
    [_pin_not_built_param(r[1], _supplementary_id(r), *r)
     for r in SUPPLEMENTARY_CANARY],
)
def test_cfr_supplementary_byte_identical(tier, fixture_id, smiles, expected_name):
    """Supplementary fixtures sampled from PRE_CANARY_ROWS for cross-class coverage.

    Per internal notes: a single byte difference here is the same HARD-gate
    failure as the per-class test above. No xfail; no relaxation.

    Note: the pre-CFR baseline records exception output as
    ``<ERROR: TYPE: msg>``; for ``name_compound`` invocations here we let the
    exception propagate naturally (a behavioral change between and CFR
    surfaces as a test error, not a name mismatch). The CSV-diff layer
    (verify_cfr_byte_identical.py --mode post) preserves the exception-as-cell
    format and catches the same drift loudly.
    """
    if expected_name.startswith("<ERROR:"):
        # The frozen baseline recorded an exception for this fixture.
        # Plan-03 contract: re-raising the same exception type produces
        # byte-identical post-CFR CSV; the pytest layer skips here so a
        # spurious exception type bubble does not look like a CFR routing bug.
        # The CSV-diff layer catches any drift in exception type/message.
        pytest.skip(
            f"frozen baseline recorded exception for fixture {fixture_id!r}; "
            f"CSV-diff layer (verify_cfr_byte_identical.py --mode post) covers it."
        )
    expected_name = _SUPPLEMENTARY_REBASELINE.get(fixture_id, expected_name)
    result = name_compound(smiles, style="pin")
    assert result == expected_name, (
        f"CFR DISPATCH REGRESSION (tier={tier}, fixture_id={fixture_id}):\n"
        f"  SMILES:   {smiles}\n"
        f"  Expected: {expected_name}\n"
        f"  Got:      {result}"
    )


# ---------------------------------------------------------------------------
# Tier contract for the two supplementary rows whose PIN is not built
#
# ``test_cfr_supplementary_byte_identical`` keeps those rows (398, 483) as strict xfails
# (``_PIN_NOT_BUILT``): there is no PIN string to pin, and the test must not accept the
# default tier's decline. What can be pinned today is the contract around them, the same
# one ``test_parent_mismatch_tier_contract`` asserts for its declared rows: the best-effort
# tier names the molecule and the name round-trips to the input's FULL InChIKey (breadth
# never drops), and whatever the PIN tier ships is RT-exact too (never another molecule).
# Run with the OPSIN validity gate on, as production does.
# ---------------------------------------------------------------------------


@pytest.mark.opsin_gate
@pytest.mark.parametrize("fixture_id", [
    "test_canary_name_stability_398",
    "test_canary_name_stability_483",
])
def test_cfr_supplementary_tier_contract(fixture_id):
    from tests.support.rt_assert import assert_tier_contract
    smiles = next(r[2] for r in PRE_CANARY_ROWS if r[1] == fixture_id)
    assert_tier_contract(smiles)
