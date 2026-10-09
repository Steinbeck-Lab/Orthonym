"""
a phase Stereo Benchmark: CIP audit and stereo-mismatch tracking.

This test suite tracks all small+medium (<=40 HA) benchmark compounds that have
stereo in the molecule but did NOT achieve InChI round-trip in v6.0 baseline.

=== a phase CIP Audit Results ===
Total benchmark compounds with stereo: 309 (of 500)
  Correct stereo labels emitted: 244 (79%)
  Missing stereo (wrong parent): 65 (21%)
  WRONG_LABEL (incorrect CIP): 0
  WRONG_LOCANT (count mismatch): 0

=== a phase Stereo Accounting ===
Small (<=20 HA):
  Already RT in v6.0: 27
  Newly RT after a phase: 8 (7 from stereo integration, 1 from a phase)
  Still failing: 50 (all due to wrong parent, NOT stereo)

Medium (21-40 HA):
  Already RT in v6.0: 23
  Newly RT after a phase: 1
  Still failing: 138 (all due to wrong parent, NOT stereo)

=== Key Finding ===
After Plans 01-03, ZERO compounds have stereo-only InChI mismatch. Every remaining
failure has connectivity/parent differences. The stereo integration is complete for
all naming paths that produce correct parent structures. Remaining stereo improvements
require a phase+ parent selection fixes first.

Tests use current names as baselines. When a compound's name improves (e.g., via
a phase+ parent fixes), update the expected name here.
"""

import re

import pytest

from orthonym import name_compound
from tests.support.rt_assert import assert_tier_contract
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "CC(=O)O[C@H]1CC[C@]2(C)C3=C(CC[C@H]2C1(C)C)[C@]1(C)C[C@@H](O)[C@H]([C@@H](C/C=C/C(C)(C)O)C(=O)O)[C@@]1(C)CC3",
    "CC(C)C[C@H](N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O",
    "CC12CCC(=O)C=C1C=CC1[C@@H]2CCC2(C)[C@H]1CCC21CCC(=O)O1",
    "COC(=O)[C@@H]1CC23CCCN4CC[C@@]5(c6ccccc6N(C)C15CC2)[C@@H]43",
    "C[C@@H]([NH3+])P(=O)([O-])[O-]",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)


# Baselines that described a DIFFERENT molecule (pre-existing-failures plan,
# Task 4 continuation, 2026-09-25). With the gate off, the PIN tier shipped the
# natural-product scaffold name, which silently dropped a decoration its finders
# do not recognise: m11 lost its 2-(2-hydroxypropan-2-yl)-5-methyloxolan-5-yl
# side chain (the C30H50O3 input named as a bare 'pentamethylgonan-3-one'), m16
# lost the ester's O-methyl ('...cholan-24-one', which OPSIN reads as C24H40O4
# for the C25H42O5 input).
# (the Blue Book): every substituent is cited as a prefix or suffix. The NP
# producer now declines (its completeness invariant runs for every decorated
# scaffold); no NP-parent name is a PIN,:50943), the PIN tier has no
# other producer and fails closed, as it already did in production. These rows
# assert the tier contract instead (tests/support/rt_assert.py), gate ON.
#
# Task 12 fix a performance pass (wp6-tests, 2026-09-26; t12-research rest-of-suite items 0-12):
# the 92 red trackers were re-computed at HEAD in one process -- raw (gate off, as this
# file runs), PIN tier (gate on) and best-effort -- and every name went through one
# OPSIN 2.9.0 batch outside the engine (full standard InChIKey against the input).
# Each row now takes ONE of four routes:
# * a tracked name (expected is a string, row in neither dict below): the current
# raw name, full-InChIKey exact, the gate-on PIN-tier name at pin_verified, and not
# in a known non-PIN class. 33 rows were re-baselined this way; most old values
# were different molecules, the rest were spelling moves toward the Blue Book
# ('isopropyl' -> 'propan-2-yl',; 'isochromane' ->
# '3,4-dihydro-1H-2-benzopyran (PIN)' the Blue Book; 'tetrahydroxy-7-oxa-
# bicyclo...' -> '...heptane-2,3,4,5-tetrol',:26834 and RB-VB
#:23710); the enclosing marks of; the (g) order of s05, (c)
# spiro numbering of s43 and the pyrimidine-2,4(1H,3H)-dione of s37, all fixed in
# wp2/wp5). The sugar and amino-acid names (s11, s24, m21, m28, m31, m32) rest on
# user decision D-a, m45 (was m48) on the controller ruling for stereoparents
# (TRIAGE.md 'User decisions', 'Controller rulings').
# wp7 (verification panel): an RT sweep over EVERY string row (not only the 92
# formerly red ones) found six more that denote a different molecule or do not parse
# (s50, m15, m17-m19, m30); they now take the tier contract, and m17/m18 carry their
# derived 24(28) spelling as a strict-xfail target (_DERIVED_TARGETS). Every tracked
# row is now full-InChIKey exact (test_every_tracked_row_is_rt_exact).
# * _TIER_CONTRACT (expected None): the PIN tier fails closed and best-effort names
# the molecule RT-exact. 27 rows are D-abstain (raw gives the whole failure
# sentinel) and 26 rows are raw names that are NOT RT-exact (13 wrong molecule, 10
# OPSIN-unparseable, 3 stereo/protonation omission; wp7 added s50, m18, m30
# (wrong molecule) and m15, m17, m19 (unparseable) to wp6's 20) -- known gate-off 0-wrong
# producers that the gate voids (TODO section J, 'Latent 0-wrong class'). A
# sentinel or a wrong raw name is never accepted as a value (TRIAGE.md 'Exit
# criteria'); the tier contract checks what ships (gate on).
# * _KNOWN_NON_PIN: the name is RT-exact but in a class that is not the PIN, and the
# PIN tier now labels it below pin_verified (wp3/wp5). The row asserts that honest
# demotion; test_stereo_pin_not_built_yet carries a strict xfail per row, which
# XPASSes once a PIN producer lands (or if the label regresses).
# * _PIN_TARGET: a pin_verified name that is not the PIN, with a Blue-Book-derived
# target; the row is a strict xfail on the target string.
# * _LIST_NAME_NOT_THE_PIN: a natural-product list name, labelled pin_verified by its
# list identity at every tier (user decision 2026-09-30); a strict xfail on the
# systematic PIN string in test_stereo_pin_not_built_yet.
_TIER_CONTRACT = {
    "CC(C)(O)[C@@H]1CC[C@@](C)([C@H]2CC[C@]3(C)[C@@H]2CC[C@@H]2[C@@]4(C)CCC"
    "(=O)C(C)(C)[C@@H]4CC[C@]23C)O1",
    "COC(=O)CC[C@@H](C)[C@H]1C[C@@H](O)[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)"
    "CC[C@]4(C)[C@H]3CC[C@@]21C",
}


# ---------------------------------------------------------------------------
# Small stereo compounds (<=20 HA, 58 non-RT at v6.0 baseline)
# ---------------------------------------------------------------------------
SMALL_STEREO_COMPOUNDS = [
    ('C[C@@H]([NH3+])P(=O)([O-])[O-]', '(1S)-1-azaniumylethane-1-phosphonate'),  # MISSING_STEREO - wrong parent
    ('O=C([O-])/C=C/C(=O)O.[Na+]', None),  # a phase: partial salt hydrogen prefix
    ('CC[C@H](C)[C@H](N)C(=O)[O-]', '(2S,3S)-2-amino-3-methylpentanoate'),  # a phase: amino acid stereo injection
    ('N[C@H](C[13C](=O)O)[13C](=O)O', '(2R)-2-amino(1,4-13C2)butanedioic acid'),  # fix: dioic acid via polyfunctional pipeline
    ('C/N=C(\\N)NCCCCN', None),  # MISSING_STEREO - wrong parent; alpha order fixed P80
    ('C=C1C=C[C@H](C(C)C)CC1', '(6S)-3-methylidene-6-(propan-2-yl)cyclohex-1-ene'),
    ('O=C1N[C@H]2NC(=O)N[C@H]2N1', None),  # a phase-02: stereo injection for urea handler
    ('NC(=O)N/C=C\\C(=O)OO', '(2Z)-3-(carbamoylamino)prop-2-eneperoxoic acid'),  # alpha order fixed P80
    ('C=CC/C=C/CCC(=O)OC', 'methyl (4E)-octa-4,7-dienoate'),  # NEWLY_RT
    ('O=C([O-])C(=O)C[C@H](O)C(=O)[O-]', '(2S)-2-hydroxy-4-oxopentanedioate'),  # NEWLY_RT
    ('O[C@H]1[C@H](O)[C@@H](O)[C@@H]2O[C@@H]2[C@@H]1O', '(1R,2R,3S,4S,5R,6S)-7-oxabicyclo[4.1.0]heptane-2,3,4,5-tetrol'),
    ('OC[C@@H]1O[C@@](O)(CO)[C@@H](O)[C@@H]1O', 'β-L-sorbofuranose'),
    ('C[C@H](CC(=O)[O-])OC(=O)C[C@@H](C)O', '(3R)-3-{[(3R)-3-hydroxybutanoyl]oxy}butanoate'),
    ('C[C@@H]1Cc2cc(O)cc(O)c2CO1', '(3R)-3-methyl-3,4-dihydro-1H-2-benzopyran-6,8-diol'),  # NEWLY_RT
    ('O=P([O-])([O-])OC[C@@H](O)[C@H](O)[C@@H](O)CO', '(2R,3R,4S)-2,3,4,5-tetrahydroxypentyl phosphate'),  # P80-01 phosphonooxy prefix; P131 merges hydroxy prefixes
    ('O=C([O-])[C@@H](O)[C@H](O)[C@H](O)[C@@H](O)C(=O)[O-]', '(2R,3S,4R,5S)-2,3,4,5-tetrahydroxyhexanedioate'),  # NEWLY_RT
    # s16, decision A part 2 (2026-09-27): was '4-[(S)-2-amino-2-carboxyethoxy]-...'
    # (labelled below pin_verified). (the Blue Book) 'stereodescriptors,
    # preceded by a locant, must be cited'; the substituent producer's free-valence
    # numbering now locates the centre. OPSIN 2.9.0 full-InChIKey exact.
    ('N[C@@H](COC(=O)CCC(=O)O)C(=O)O', '4-[(2S)-2-amino-2-carboxyethoxy]-4-oxobutanoic acid'),
    ('C=C(C(=O)OC)N1C(=O)C[C@@H](C)C1=O', 'methyl 2-[(3R)-3-methyl-2,5-dioxopyrrolidin-1-yl]prop-2-enoate'),  # MISSING_STEREO - wrong parent; depth-independent naming v11
    ('CC(=O)[C@@H](C)Nc1ccccc1C(=O)O', None),
    # wp7 change-asserted-value (s19, s20): compares LETTERS first, locants and hyphens
    # never (the Blue Book;:3477 "begins with the first letter of its complete
    # name"): propanylidene < propenyl, and hydroxymethyl, an initial segment of hydroxymethylpentyl,
    # comes first (:21663 '...-4-methyl-3-methylidenehexanoic acid (PIN)'). The old values let '-'
    # sort before a letter ('prop-' < 'propa'). OPSIN 2.9.0 full-InChIKey exact (outside the engine).
    ('C=C[C@]1(C)CCC(=C(C)C)C[C@H]1C(=C)C', '(1S,2S)-1-ethenyl-1-methyl-4-(propan-2-ylidene)-2-(prop-1-en-2-yl)cyclohexane'),  # was '...-2-(prop-1-en-2-yl)-4-(propan-2-ylidene)cyclohexane'
    ('CC(C)CC[C@@H](O)[C@H]1C(=O)OC[C@@H]1CO', '(3S,4S)-4-(hydroxymethyl)-3-[(1R)-1-hydroxy-4-methylpentyl]oxolan-2-one'),  # was '(3S,4S)-3-[(1R)-1-hydroxy-4-methylpentyl]-4-(hydroxymethyl)oxolan-2-one'
    ('CC(C)[C@@H]1CC[C@H](C)CCC[C@H](C)CC1', 'germacrane'),  # lowercase s for pseudoasymmetric center per IUPAC
    ('COc1c(C)c(O)cc2c1C(=O)N[C@H]2C', '(3S)-5-hydroxy-7-methoxy-3,6-dimethyl-2,3-dihydro-1H-isoindol-1-one'),
    ('CC(=N)NCCCC[C@H](N)C(=O)O.Cl.Cl', None),  # alpha order fixed P80
    # s24, decision A part 2 (user decision A, 2026-09-26): an amino acid substituted on its
    # NITROGEN takes the systematic substitutive name at the PIN tier.
    # (the Blue Book): -NH-CO-R is named "(1) substitutively, by using a prefix formed
    # by changing the final letter 'e' in the complete name of the amide to 'o'" (:32995);
    # "Method (1) generates preferred IUPAC names." (:32998); an N-substituted glycine is
    # printed on the acetic acid parent at:33213. The N-acyl retained form ('N-[(6E)-4-
    # hydroxyoct-6-enoyl]glycine',:54480) is general nomenclature:50943).
    # OPSIN 2.9.0 full-InChIKey exact (outside the engine).
    ('C/C=C/CC(O)CCC(=O)NCC(=O)O', '[(6E)-4-hydroxyoct-6-enamido]acetic acid'),  # was 'N-[(6E)-4-hydroxyoct-6-enoyl]glycine'
    ('CCC[C@@H]1OCc2c(O)cccc2[C@H]1O', '(3S,4R)-3-propyl-3,4-dihydro-1H-2-benzopyran-4,8-diol'),  # NEWLY_RT
    # a phase cleanup: stereo descriptor `(1S,4R,5R)-` rebaselined per
    # a phase spiro stereo injection + IUPAC mandatory rules
    # for stereodefined sp3 centers. The pre-Phase-153 expectation lacked
    # the descriptor; stereo pipeline correctly emits it.
    ('C=C(C)[C@@H]1CC[C@@H](C)[C@@]12CC=C(C)CC2', '(1R,4S,5R)-1,8-dimethyl-4-(prop-1-en-2-yl)spiro[4.5]dec-7-ene'),  # P119-02: subs discovered; a phase stereo
    ('NC(C(=O)O)C(CC[C@H](N)C(=O)O)C(=O)O', '(5S)-1,5-diaminopentane-1,2,5-tricarboxylic acid'),
    ('COc1c(O)c(O)cc2c1CO[C@@H](C)C2=O', '(3S)-6,7-dihydroxy-8-methoxy-3-methyl-1H-2-benzopyran-4(3H)-one'),  # NEWLY_RT
    ('O=C(O)/C=C/c1ccc(OS(=O)(=O)O)cc1', '(2E)-3-[4-(sulfooxy)phenyl]prop-2-enoic acid'),  # NEWLY_RT
    ('CC[C@@H](O)C[C@@H](O)c1cc(OC)cc(=O)o1', '6-[(1R,3R)-1,3-dihydroxypentyl]-4-methoxy-2H-pyran-2-one'),
    ('CC(C)=C[C@H](O)C1=CC(=O)[C@@H](O)[C@H](O)[C@H]1O', None),
    ('CCC(C)C1=C2C(=O)OC[C@H]2[C@@H](C)[C@H](C)O1', None),  # Fixed: fabricated diethyl from ring boundary leak
    ('COc1c(Cl)c2c(c(C(=O)O)c1Cl)C[C@H](C)O2', '(2S)-5,7-dichloro-6-methoxy-2-methyl-2,3-dihydro-1-benzofuran-4-carboxylic acid'),  # MISSING_STEREO - wrong parent
    ('C/C=C/C=C/C(=O)C1=C(O)C(=C(C)C)NC1=O', None),  # MISSING_STEREO - wrong parent
    ('CCCCCCC(=O)NC1=CC(=O)[C@@H]2CCCN12', None),  # a phase improved: was heptanamide (MISSING_STEREO - wrong parent)
    ('C/C=C1\\[C@H]2C=C(C)C[C@]1([NH3+])c1ccc(=O)[nH]c1C2', None),
    ('CO[C@@H]1[C@H](O)[C@@H](CO)O[C@H]1n1ccc(=O)[nH]c1=O', '1-[(2R,3R,4R,5R)-4-hydroxy-5-(hydroxymethyl)-3-methoxyoxolan-2-yl]pyrimidine-2,4(1H,3H)-dione'),
    ('CCCCC[C@@H](O)[C@@H](O)c1cc(OC)cc(=O)o1', '6-[(1R,2R)-1,2-dihydroxyheptyl]-4-methoxy-2H-pyran-2-one'),
    ('C[C@H]1C[C@H](O)[C@@H]2[C@H]1[C@@H]1[C@H](CC[C@]2(C)O)[C@@]1(C)CO', None),
    # j7 (TRIAGE g5 C15): the tripeptide's substitutive PIN (nested amido prefix,
    # method (1):32995; controller ruling: peptide names are not
    # PINs); was the retained 'prolylalanylalanine'. OPSIN full-InChIKey exact.
    ('C[C@H](NC(=O)[C@H](C)NC(=O)[C@@H]1CCCN1)C(=O)O', '(2S)-2-{(2S)-2-[(2S)-pyrrolidine-2-carboxamido]propanamido}propanoic acid'),
    ('CCCCCC=CC1=C(CO)C(=O)C[C@H](O)[C@@H]1O', '(4R,5S)-3-(hept-1-en-1-yl)-4,5-dihydroxy-2-(hydroxymethyl)cyclohex-2-en-1-one'),
    ('CC(=O)[C@@]1(C)C(C)=C[C@H](O)[C@H]2C[C@](C)(O)CC[C@@H]21', None),  # MISSING_STEREO - wrong parent
    # a phase cleanup: stereo descriptor `(5S,6R,9R,10R)-` rebaselined
    # per a phase spiro stereo injection (same rationale as the
    # spiro entry above).
    ('CC1=C[C@]2(CC1=O)[C@H](C)CC[C@@H](C(C)(C)O)[C@H]2O', '(5S,6R,7R,10R)-6-hydroxy-7-(2-hydroxypropan-2-yl)-3,10-dimethylspiro[4.5]dec-3-en-2-one'),  # P119-02: subs discovered; a phase stereo
    ('COCC1=C2[C@@H]3CC(C)(C)C[C@@H]3C[C@@]2(O)CC1=O', None),
    ('O=C([O-])C(=O)C[C@@H](O)[C@H](O)[C@H](O)COP(=O)([O-])[O-]', None),  # P80-01 phosphonooxy prefix now generated
    ('CCC(O)CC(=O)O[C@H](CC(=O)[O-])C[N+](C)(C)C', None),  # MISSING_STEREO - wrong parent
    ('NC(N)=NCCC[C@H](NC(=O)[C@@H](N)CO)C(=O)O', '(2S)-2-[(2S)-2-amino-3-hydroxypropanamido]-5-[(diaminomethylidene)amino]pentanoic acid'),  # alpha order fixed P80
    ('CCCC/C=C\\CCCCCCCCCOC(C)=O', '(10Z)-pentadec-10-en-1-yl acetate'),  # NEWLY_RT (a phase)
    ('CC(C)[C@H]1CC[C@@H](CO)c2c(O)cc(C(=O)O)cc21', None),  # a phase: enrichment adds substituents
    ('CCOC(=O)C[C@@H](SP(=O)(OC)OC)C(=O)OCC', None),  # wp7: was 'diethyl butanedioate' (drops the phosphorodithioate; OPSIN DKMROQRQHGEIOW vs input WSORODGWGUUOBO)
    ('CC1C/C(=C\\CC(CC(N)=O)CC(=O)O)C(=O)C(C)C1', None),  # a phase-01: chain exclusion changes path
    ('CC(C)=CCc1ccc(O)c2c1C=C[C@H]1O[C@@H]2O[C@H]1C', None),
    ('COc1cccc2c1CO[C@@H]2C[C@@H](O)[C@@H](O)[C@@H]1O[C@@H]1C', None),  # a phase: alphabetical tiebreaker (cyclononyl < oxiranyl)
    ('C[C@H]1C[C@@H](O)[C@H]2C(=O)c3c(O)cccc3O[C@]2(C)[C@@H]1O', None),
    ('C[C@H]1CCC/C=C/[C@@H]2CC[C@H](O)[C@H]2[C@H](O)/C=C/C(=O)O1', None),  # Fixed: fabricated dipropyl from ring boundary leak
    ('CC(C)=CCc1ccc(O)c2c1[C@H](CC(=O)O)OC2=O', None),  # MISSING_STEREO - wrong parent
    ('C/C=C/C(=O)O[C@H]1/C=C\\C(=O)[C@@H](O)CCC(=O)O[C@@H]1C', None),
]


# ---------------------------------------------------------------------------
# Medium stereo compounds (21-40 HA, 139 non-RT at v6.0 baseline)
# Representative subset: 50 compounds spanning common failure patterns
# ---------------------------------------------------------------------------
MEDIUM_STEREO_COMPOUNDS = [
    ('Oc1ccc2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@@H](O)[C@@H]2O', '(2R,3S,4R)-2-(3,4-dihydroxyphenyl)-3,4-dihydro-2H-1-benzopyran-3,4,7-triol'),
    # a phase Plan 02 Task 03: estra ring locant correction 1,2,4 → 1,3,5
    # (canonical estra-1,3,5-triene numbering for aromatic A-ring per
    #; per Plan 01 SUMMARY this is "unrelated to a phase"
    # incidental locant correction). Acceptable churn.
    # j7: the estrane numbering map had C-11/C-12 swapped; with it fixed the
    # alpha/beta form resolves:51053 "This method is
    # preferred"); OPSIN full-InChIKey exact.
    ('C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1[C@@H](O)[C@@H](O)[C@@H]2O', 'estra-1,3,5(10)-triene-3,15α,16α,17β-tetrol'),
    ('COc1cc(O)c2c(c1)C(=O)C1=C(C2=O)[C@@H](O)C[C@@](C)(O)C1', '(1S,3S)-1,3,8-trihydroxy-6-methoxy-3-methyl-1,2,3,4-tetrahydroanthracene-9,10-dione'),
    ('C[C@@H]1CC(=O)O[C@@H](C)[C@H](O)/C=C\\C(=O)O[C@@H](C)C/C=C\\C(=O)O1', '(4R,7Z,10S,13Z,15R,16S)-15-hydroxy-4,10,16-trimethyl-1,5,11-trioxacyclohexadeca-7,13-diene-2,6,12-trione'),
    ('CN1CCC2=C[C@H](O)[C@H]3OC(=O)c4cc5c(cc4[C@H]3[C@@H]21)OCO5', None),  # Updated P72: IUPAC citation order
    ('CCCCC[C@H](O)/C=C/[C@H]1CCC(=O)[C@@H]1C/C=C\\CCCC(=O)O', None),
    ('CCCCC[C@H](O)/C=C/[C@@H]1[C@@H](C/C=C\\CCCC(=O)O)[C@H](O)C[C@H]1O', None),
    # j7 (TRIAGE g3 C05): the substitutive PIN, was the retained 'tyrosylvalylarginine'.
    ('CC(C)[C@H](NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O', '(2S)-2-{(2S)-2-[(2S)-2-amino-3-(4-hydroxyphenyl)propanamido]-3-methylbutanamido}-5-(carbamimidoylamino)pentanoic acid'),
    # wp7 change-asserted-value (m08): 'dimethyl' keys at 'methyl', before 'oxo':3448); a primed
    # locant is a locant:3442), not text that sorts before letters. OPSIN full-InChIKey exact.
    # quick-wins F-Q1: von Baeyer superscripts as low as possible, the Blue Book);
    # was "(10'S,17'S)-9',13'-dimethyl-5,6'-dioxospiro[oxolane-2,14'-tetracyclo[8.7.0.0^4,9.0^13,17]heptadeca-2,4-diene]"
    ('CC12CCC(=O)C=C1C=CC1[C@@H]2CCC2(C)[C@H]1CCC21CCC(=O)O1', "(1'S,11'S)-2',15'-dimethyl-5,5'-dioxospiro[oxolane-2,14'-tetracyclo[8.7.0.0^2,7.0^11,15]heptadeca-6,8-diene]"),
    ('C=C(C)[C@H]1CC[C@]2(C)[C@@H]1CC[C@]1(C)C/C=C(\\C)CC/C=C(\\C)CC[C@H]12', None),
    ('C/C(=C\\CC/C(C)=C/C/C=C(/CC(=O)c1cc(O)ccc1O)C(=O)O)CO', None),  # NEWLY_RT
    # _TIER_CONTRACT (was '...-4,4,8,10,14-pentamethylgonan-3-one', side chain dropped)
    ('CC(C)(O)[C@@H]1CC[C@@](C)([C@H]2CC[C@]3(C)[C@@H]2CC[C@@H]2[C@@]4(C)CCC(=O)C(C)(C)[C@@H]4CC[C@]23C)O1', None),
    ('C/C1=C/C[C@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)/C(C)=C/[C@H]2OC(=O)[C@H](C)[C@@H]2CC1', None),
    ('CC1(C)OC[C@]2(C)[C@@H](CC[C@@]3(C)[C@H]2[C@@H](O)C[C@H]2C[C@@H]4C[C@@]23CC[C@]4(O)CO)O1', None),  # Updated P72: IUPAC citation order
    # quick-wins F-Q1: superscripts {1,3,8,9,9,12,16,21} before {1,5,8,8,9,14,16,21},:9685);
    # was 'methyl (8R,17R,21S)-15-methyl-5,15-diazahexacyclo[14.2.2.1^1,5.0^8,16.0^9,14.0^8,21]henicosa-9,11,13-triene-17-carboxylate'
    ('COC(=O)[C@@H]1CC23CCCN4CC[C@@]5(c6ccccc6N(C)C15CC2)[C@@H]43', 'methyl (9R,18R,21S)-2-methyl-2,12-diazahexacyclo[14.2.2.1^9,12.0^1,9.0^3,8.0^16,21]henicosa-3,5,7-triene-18-carboxylate'),
    ('COc1cc2c(cc1OC)[C@H]1Cc3ccc(OC)c(OC)c3CN1CC2', None),  # wp7: was 'berberine' (a different compound, the unsaturated alkaloid; tetrahydropalmatine is the input)
    # _TIER_CONTRACT (was '...-3,7,15-trihydroxycholan-24-one', the ester O-methyl dropped)
    ('COC(=O)CC[C@@H](C)[C@H]1C[C@@H](O)[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@@]21C', None),
    # wp7 (verification panel): these three raw names are not RT-exact -- OPSIN reads
    # '24-ene' as C24=C25, the inputs carry the 24(28)-methylidene (stigmastadienol,
    # ergostadienol) or cannot be parsed (the trienone). Gate on, the PIN tier abstains and
    # best-effort names them exactly: tier contract. The two derived 24(28) spellings are
    # strict-xfail targets in _DERIVED_TARGETS below. They were listed twice (also as
    # m43-m45); the duplicates are removed.
    ('C/C=C(/CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)C(C)C', None),
    ('C=C(CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1CC3)C(C)C', None),
    ('C=C(CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4[C@H](C)C(=O)CC[C@]4(C)C3=C[C@@H](O)[C@]12C)C(C)C', None),
    ('CC(C)C[C@H](N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O', 'leucylaspartylarginine'),  # alpha order fixed P80
    ('CC(=O)N[C@@H]1[C@@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2NC(C)=O)[C@@H](O)[C@@H](CO)O[C@@H]1O', '2-acetamido-2-deoxy-β-D-glucopyranosyl-(1->3)-2-acetamido-2-deoxy-α-D-galactopyranose'),  # Updated a phase: sugar routing returns retained sugar name
    ('CC(=O)N[C@H]1C(OP(=O)(O)OP(=O)(O)OC[C@H]2O[C@@H](n3ccc(=O)[nH]c3=O)[C@H](O)[C@@H]2O)O[C@H](CO)[C@H](O)[C@@H]1O', None),
    ('CC(C)[C@@H](C)[C@@H](O)[C@H]1CC[C@@H]([C@@]2(C)CCC(=O)[C@@]3(C)CC[C@H](O)C[C@]34C=C[C@@](O)(O4)C2=O)[C@@H]1C', None),
    ('CC(=CCC(O)C(C)[C@H]1CC(=O)[C@@]2(C)C3=C(C(=O)[C@@H](O)[C@]12C)[C@@]1(C)CCC(=O)[C@](C)(CO)[C@@H]1CC3=O)C(=O)O', None),
    ('C=C(CC[C@@H](C(=O)O)[C@H]1[C@H](O)[C@H](O)[C@@]2(C)C3=CC[C@H]4C(C)(C)C(=O)CC[C@]4(C)C3=CC[C@]12C)C(C)C', None),
    ('CC(CCCC(C)(O)COS(=O)(=O)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)CCC4(C)[C@H]3C[C@H](O)C12C', None),
    ('CC(=O)O[C@H]1CC[C@]2(C)C3=C(CC[C@H]2C1(C)C)[C@]1(C)C[C@@H](O)[C@H]([C@@H](C/C=C/C(C)(C)O)C(=O)O)[C@@]1(C)CC3', '(3S,5R,10S,13R,14R,16R,17R,20R,23E)-3-(acetyloxy)-16,25-dihydroxy-4,4,14-trimethylcholesta-8,23-dien-21-oic acid'),
    ('O=C1c2c(O)cc(O)cc2O[C@@H](c2ccc(O)c(O)c2)[C@@H]1O[C@@H]1OC[C@@H](O)[C@H](O)[C@H]1O', '(2S,3S)-2-(3,4-dihydroxyphenyl)-5,7-dihydroxy-3-(β-D-xylopyranosyloxy)-2,3-dihydro-4H-1-benzopyran-4-one'),
    ('CC1=C[C@]2(C)C[C@@H](C)CC[C@@H]2[C@H](C(=O)[C@@H]2C(=O)N3CC[C@@H]4C(=O)O[C@H]2[C@@]43O)[C@@H]1C', None),
    ('CC(=O)O[C@H]1[C@@H](OC(C)=O)C(C)(C)[C@]2(O)CC[C@H]3C(=O)c4ccoc4C[C@@H]3[C@@]2(C)[C@H]1OC(C)=O', None),  # wp7: was 'bis(acetyloxy)ethanone acetate' (OPSIN: a two-component mixture NPMYZOILKOAWDI vs input SJZIFWGVHGVKAH); a phase-05 garbled multi-ester path
    ('OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O[C@@H]1OC[C@@H](O)[C@H](O)[C@H]1O', 'β-D-xylopyranosyl-(1->4)-β-D-glucopyranose'),  # Updated a phase: sugar routing returns retained sugar name
    ('OC[C@H]1O[C@H](O)[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O)[C@@H](O)[C@H]1O', 'α-D-mannopyranosyl-(1->2)-α-D-galactopyranose'),  # Updated a phase: sugar routing returns retained sugar name
    ('OC[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O', None),
    ('CC(=O)OC[C@H]1O[C@@H](N2CCC(=O)NC2=O)[C@H](OC(C)=O)[C@@H]1OC(C)=O', None),  # a phase: stereo now included
    ('C=C1NC(=O)[C@H]([C@@H](C)[C@]2(O)C(=O)N(C)c3ccccc32)NC1=O', None),
    ('C[C@H]1C=C[C@H]2C[C@@H](O)CC[C@H]2[C@@H]1c1cc(N)c(C=O)c(=O)o1', None),  # Fixed: ring boundary fix changed substituent/locant assignment
    ('CC(C)=CCC/C(C)=C/CC[C@]1(C)Cc2c(c(O)cc3c2CN([C@H]2CCCNC2=O)C3=O)C[C@@H]1O', None),
    ('CC(C)[C@@H](C)[C@@H](O)[C@H]1CC[C@@H]([C@@]2(C)CCC(=O)[C@@]3(C)CC[C@H](O)C[C@]34C=C[C@@](O)(O4)C2=O)[C@@H]1C', None),
    # a phase Plan 02 Task 03: indole+peptide chain. Pre-148 ring-as-parent
    # (`-3-(2-aminoethyl)-1H-indole` suffix); post-148 cascade per (a)
    # selects propanamide chain as parent with indole as `1H-indol-3-yl`
    # substituent. Acceptable churn (cascade decision IUPAC-correct).
    ('CC(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N/C=C\\c1c[nH]c2ccccc12', None),  # a phase cascade unblock; depth-independent naming v11; N-bracket fix v15
    ('O=C1N[C@@H](C[C@@]2(O)c3ccccc3N3C(=O)[C@@H]4CCCCN4[C@@H]32)C(=O)N[C@H]1Cc1ccccc1', None),
    ('CC(C)[C@@H](C)[C@@H](O)[C@H]1CC[C@@H]([C@@]2(C)CCC(=O)[C@@]3(C)CC[C@H](O)C[C@]34C=C[C@@](O)(O4)C2=O)[C@@H]1C', None),
    ('COc1c(Cl)c(C)cc2cc(O)c3c(c12)C(=O)c1cc2c(c(O)c1C3=O)[C@H](C)OC2=O', None),  # Updated P72: IUPAC citation order
    ('CC(=O)O[C@H]1CC[C@]2(C)C3=C(CC[C@H]2C1(C)C)[C@]1(C)C[C@@H](O)[C@H]([C@@H](C/C=C/C(C)(C)O)C(=O)O)[C@@]1(C)CC3', '(3S,5R,10S,13R,14R,16R,17R,20R,23E)-3-(acetyloxy)-16,25-dihydroxy-4,4,14-trimethylcholesta-8,23-dien-21-oic acid'),
    ('C[C@@H]1O[C@@H](OCCCCCCCCCCCCCCCCCCCC[C@@H](O)CC(=O)O)[C@H](O)C[C@H]1O', None),
    ('CC(C)[C@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)OS(=O)(=O)[O-]', '(24S)-3β-hydroxycholest-5-en-24-yl sulfate'),
]


# ---------------------------------------------------------------------------
# Test IDs
# ---------------------------------------------------------------------------
_SMALL_IDS = [
    f"s{i:02d}_HA{ha}"
    for i, (smi, _) in enumerate(SMALL_STEREO_COMPOUNDS)
    for ha in [len([a for a in __import__('rdkit').Chem.MolFromSmiles(smi).GetAtoms()
                    if a.GetAtomicNum() > 1]) if __import__('rdkit').Chem.MolFromSmiles(smi) else 0]
]

_MEDIUM_IDS = [
    f"m{i:02d}_HA{ha}"
    for i, (smi, _) in enumerate(MEDIUM_STEREO_COMPOUNDS)
    for ha in [len([a for a in __import__('rdkit').Chem.MolFromSmiles(smi).GetAtoms()
                    if a.GetAtomicNum() > 1]) if __import__('rdkit').Chem.MolFromSmiles(smi) else 0]
]


_TIER_CONTRACT |= {smi for smi, exp in SMALL_STEREO_COMPOUNDS + MEDIUM_STEREO_COMPOUNDS
                   if exp is None}

# SMILES -> why the RT-exact name is not the PIN (the PIN tier labels it below
# pin_verified), and the PIN string where it is derived (else None).
_KNOWN_NON_PIN = {
    # s00
    "C[C@@H]([NH3+])P(=O)([O-])[O-]": (
        "zwitterion named by the 'amino' -> 'azaniumyl' swap, which wp3 labels below "
        "pin_verified (name-scoped record); the P-74 zwitterion PIN is not derived",
        None),
    # s16 left this map in decision A part 2 (2026-09-27): the '(2S)-' emitter is built
    # (substituent_naming.polyfunctional_substituent_located) and the row ships its PIN
    # '4-[(2S)-2-amino-2-carboxyethoxy]-4-oxobutanoic acid' at pin_verified.
    # s21 ('germacrane') left this map: _LIST_NAME_NOT_THE_PIN below.
    # m20: retained peptide name (s40 and m07 left this list in j7: the tripeptide
    # substitutive PIN is built, TRIAGE g3 C05)
    "CC(C)C[C@H](N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O": (
        "peptide name; controller ruling: peptides are not PINs, the PIN is the "
        "substitutive name (not built)", None),
    # m08 (canrenone)
    "CC12CCC(=O)C=C1C=CC1[C@@H]2CCC2(C)[C@H]1CCC21CCC(=O)O1": (
        "von Baeyer descriptor for the ortho-fused steroid core inside a spiro name and "
        "carbonyls as 'oxo' (P-41 :18158); the fused spiro PIN is not built (P-52.2.4.1 "
        ":23710). The P-14.5.1 prefix order (:3448) is fixed (T12 wp7)", None),
    # m14
    "COC(=O)[C@@H]1CC23CCCN4CC[C@@]5(c6ccccc6N(C)C15CC2)[C@@H]43": (
        "von Baeyer name of a system with ortho-fused 5+/6 rings; the (bridged) fusion "
        "name is the PIN (P-52.2.4.1 :23710; large-polycycle Task 6a); shipped at "
        "systematic_verified", None),
    # m27 == m43 (one SMILES; m43 was m46 before wp7 removed three duplicate rows)
    "CC(=O)O[C@H]1CC[C@]2(C)C3=C(CC[C@H]2C1(C)C)[C@]1(C)C[C@@H](O)[C@H]([C@@H](C/C=C/C(C)(C)O)C(=O)O)[C@@]1(C)CC3": (
        "a free COOH written as '21-hydroxy...21-oxo' prefixes with the ester principal; "
        "acids outrank esters (P-41 :18158, 7a :18172 > 9 :18182; P-65.6.3.2.3 :31696); "
        "wp5 demotes it", None),
}

# SMILES -> (why the name is not the PIN, the systematic PIN string) for a name of the
# natural-product list (NAME_EXACT_NP_PARENTS, matched by the exact structure). User
# decision 2026-09-30: as in the paper's run, the list names are labelled
# pin_verified with verified 'identity' (the exact structure match, not an OPSIN
# verdict) at every tier, and trivial and semisystematic natural-product names get real
# systematic PINs later (the PIN class program, Task 32); the deviation from
# ("Preferred IUPAC names (PINs) are not identified for the compounds in this
# Chapter", the Blue Book) is recorded in TRIAGE.
_LIST_NAME_NOT_THE_PIN = {
    # s21: 'germacrane':51375, Table 10.1 (c) terpenes:51413). OPSIN 2.9.0
    # reads neither it nor its systematic name (no lowercase pseudoasymmetric
    # descriptor; '13-norgermacrane (1R,4s,7S)-4-ethyl-1,7-dimethylcyclodecane',
    #:51471); '(1R,7S)-1,7-dimethyl-4-(propan-2-yl)cyclodecane', the systematic name
    # without the '4s', reads back to the input's constitution and R/S centres.
    "CC(C)[C@@H]1CC[C@H](C)CCC[C@H](C)CC1": (
        "'germacrane', the np_stereoparent list name (P-100 :50943, no PINs for natural "
        "products; labelled pin_verified by its list identity, user decision "
        "2026-09-30). The systematic name's pseudoasymmetric '4s' is verified by the "
        "centres labeller, OPSIN 2.9.0 cannot assign it",
        "(1R,4s,7S)-1,7-dimethyl-4-(propan-2-yl)cyclodecane"),
}

# SMILES -> (Blue-Book-derived PIN, reason) for pin_verified names that are not the PIN.
_PIN_TARGET = {
    # s28 (canary row 133, DK-INDH)
    "COc1c(O)c(O)cc2c1CO[C@@H](C)C2=O": (
        "(3S)-6,7-dihydroxy-8-methoxy-3-methyl-1H-2-benzopyran-4(3H)-one",
        "DEFECT (DK-INDH): ships '(3S)-6,7-dihydroxy-8-methoxy-3-methyl-3,4-dihydro-"
        "1H-2-benzopyran-4-one' at pin_verified. No 4H-2-benzopyran isomer exists, so "
        "the indicated hydrogen takes C1 and the ketone takes added indicated hydrogen: "
        "P-58.2.3.1.4 (BlueBookV2.md:24841) -> P-58.2.3.1.3 (:24806) (1)/(3); P-58.2.2 "
        "(:24687) added hydrogen is preferred to hydro prefixes; BB 'hexahydro-1H-2-"
        "benzopyran-1,3(4H)-dithione (PIN)' (:32549), '3,4-dihydronaphthalen-1(2H)-one "
        "(PIN)' (:3276). Producer: partial_saturation.name_cyclic_oxo_compound. "
        ".planning/TODO-2026-09-24.md 'Open from T12 fix round 2 (wp6)'."),
}


def _assert_demoted(smiles, expected_name):
    """A known non-PIN class: the gate-on PIN tier ships the (RT-exact) name, but never
    labelled pin_verified."""
    r = _dt_row(smiles)
    reason = _KNOWN_NON_PIN[smiles][0]
    assert r["name"] == expected_name, (
        f"KNOWN-NON-PIN ROW CHANGED: {smiles}: expected {expected_name!r}, got "
        f"{r['name']!r} -- re-verify (OPSIN full InChIKey + the Blue Book)")
    assert r["tier"] != "pin_verified" and not r["is_pin"], (
        f"a known non-PIN name is labelled pin_verified ({reason}): {r}")
    from tests.support.rt_assert import assert_full_rt
    assert_full_rt(expected_name, smiles)


def _assert_list_identity(smiles, expected_name):
    """A natural-product list name (_LIST_NAME_NOT_THE_PIN): the default and the
    best-effort tier ship it, labelled pin_verified by its list identity (verified
    'identity', gate outcome carveout:np_stereoparent). OPSIN cannot read it, so there
    is no round trip to assert."""
    from tests.support.rt_assert import name_best_effort
    for r in (_dt_row(smiles), name_best_effort(smiles)):
        assert r["name"] == expected_name, (
            f"LIST-NAME ROW CHANGED: {smiles}: expected {expected_name!r}, got "
            f"{r['name']!r} -- re-verify (the list entry + the Blue Book)")
        assert r["tier"] == "pin_verified" and r["verified"] == "identity", r
        assert r["gate_outcome"] == "carveout:np_stereoparent", r


def _params(rows):
    out = []
    for smi, exp in rows:
        if smi in _TIER_CONTRACT or smi in _KNOWN_NON_PIN or smi in _LIST_NAME_NOT_THE_PIN:
            out.append(pytest.param(smi, exp, marks=pytest.mark.opsin_gate))
        elif smi in _PIN_TARGET:
            out.append(pytest.param(smi, _PIN_TARGET[smi][0], marks=pytest.mark.xfail(
                strict=True, reason=_PIN_TARGET[smi][1])))
        else:
            out.append((smi, exp))
    return out


def _check(smiles, expected_name):
    if smiles in _TIER_CONTRACT:
        assert_tier_contract(smiles)
        return
    if smiles in _KNOWN_NON_PIN:
        _assert_demoted(smiles, expected_name)
        return
    if smiles in _LIST_NAME_NOT_THE_PIN:
        _assert_list_identity(smiles, expected_name)
        return
    result = _dt_name_compound(smiles)
    assert result == expected_name, (
        f"STEREO BASELINE CHANGED: {smiles}\n"
        f"  Expected: {expected_name}\n"
        f"  Got:      {result}"
    )


@pytest.mark.parametrize("smiles,expected_name", _params(SMALL_STEREO_COMPOUNDS),
                         ids=_SMALL_IDS)
def test_small_stereo_baseline(smiles, expected_name):
    """Track stereo naming for small (<=20 HA) benchmark compounds.

    These compounds had stereo in the molecule but did not round-trip in the v6.0
    baseline. A tracked row asserts its re-verified current name; the routed rows are
    described above _TIER_CONTRACT. When a name changes, re-verify it (OPSIN full
    InChIKey + the Blue Book) before updating the expected value.
    """
    _check(smiles, expected_name)


@pytest.mark.parametrize("smiles,expected_name", _params(MEDIUM_STEREO_COMPOUNDS),
                         ids=_MEDIUM_IDS)
def test_medium_stereo_baseline(smiles, expected_name):
    """Track stereo naming for medium (21-40 HA) benchmark compounds (same routes as
    test_small_stereo_baseline)."""
    _check(smiles, expected_name)


# wp7: tier-contract rows whose raw name misreads a 24(28)-methylidene as '24-ene'.
# The derived spelling cites the exocyclic bond with its compound locant;
# stereoparent names kept by the controller ruling on names) and is OPSIN 2.9.0
# full-InChIKey exact. j7: the steroid unsaturation finder now emits every compound
# locant (1), the Blue Book), so both rows pass; m17's stigmastane
# map (C-11/C-12 swapped) is fixed too, so its name takes the preferred
# alpha/beta form:51053) instead of the whole-graph R/S target.
_DERIVED_TARGETS = {
    # m17 (stigmastadienol)
    "C/C=C(/CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)C(C)C":
        "(24Z)-5α-stigmasta-7,24(28)-dien-3β-ol",
    # m18 (ergostadienol)
    "C=C(CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1CC3)C(C)C":
        "(3S,5R,10S,13R,14R,17R,20R)-4,4,14-trimethylergosta-8,24(28)-dien-3-ol",
}


@pytest.mark.parametrize("smiles,target", list(_DERIVED_TARGETS.items()))
def test_steroid_24_28_methylidene_target(smiles, target):
    from tests.support.rt_assert import name_is_rt_exact
    assert _dt_name_compound(smiles) == target
    assert name_is_rt_exact(target, smiles)


def test_every_tracked_row_is_rt_exact():
    """Every tracked string row (neither a tier contract, a known non-PIN nor a PIN
    target) denotes the input molecule: independent OPSIN full-InChIKey round trip
    (tests/support/rt_assert.name_is_rt_exact). wp7: the header promised this for
    all tracked rows while six of them failed it."""
    from tests.support.rt_assert import name_is_rt_exact
    bad = [(smi, exp) for smi, exp in SMALL_STEREO_COMPOUNDS + MEDIUM_STEREO_COMPOUNDS
           if exp is not None and smi not in _TIER_CONTRACT and smi not in _KNOWN_NON_PIN
           and smi not in _LIST_NAME_NOT_THE_PIN and smi not in _PIN_TARGET
           and not name_is_rt_exact(exp, smi)]
    assert not bad, bad


def test_steroid_rows_are_listed_once():
    """ wp7: the three 24(28)-steroid rows were listed twice (m17-m19 and m43-m45)."""
    smiles = [smi for smi, _ in SMALL_STEREO_COMPOUNDS + MEDIUM_STEREO_COMPOUNDS]
    for smi in list(_DERIVED_TARGETS) + [
            "C=C(CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4[C@H](C)C(=O)CC[C@]4(C)C3=C"
            "[C@@H](O)[C@]12C)C(C)C"]:
        assert smiles.count(smi) == 1, smi


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    pytest.param(smi, marks=pytest.mark.xfail(strict=True, reason=(
        "PIN not built: " + reason + ". .planning/TODO-2026-09-24.md 'Open from T12 "
        "fix round 2 (wp5)' / '(wp6)'")))
    for smi, (reason, _target) in list(_KNOWN_NON_PIN.items())
    + list(_LIST_NAME_NOT_THE_PIN.items())])
def test_stereo_pin_not_built_yet(smiles):
    """The PIN tier ships a PIN for a known non-PIN row (the derived string where there
    is one). Strict xfail: it XPASSes when a PIN producer lands -- or if the demoted
    label regresses to pin_verified -- and forces the row to be re-verified."""
    from orthonym import Orthonym
    r = _dt_row(smiles)
    target = {**_KNOWN_NON_PIN, **_LIST_NAME_NOT_THE_PIN}[smiles][1]
    assert r["tier"] == "pin_verified", r
    if target is not None:
        assert r["name"] == target, r


def test_cip_audit_no_wrong_labels():
    """Verify CIP audit: all emitted stereo labels match rdCIPLabeler.

    a phase CIP audit confirmed 0 WRONG_LABEL and 0 WRONG_LOCANT across
    all 500 benchmark compounds. This test verifies the invariant holds.
    """
    import re
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    # Test a representative set of compounds that DO emit stereo
    stereo_compounds = [
        (smi, name) for smi, name in SMALL_STEREO_COMPOUNDS
        if name and re.search(r'\(\d+[RSEZ]', name)
    ]

    for smiles, expected_name in stereo_compounds[:20]:  # Check first 20
        result = _dt_name_compound(smiles)
        # Parse stereo from name
        stereo_in_name = re.findall(r'(\d+)([RSEZ])', result)

        # All CIP codes must be valid
        for locant, code in stereo_in_name:
            assert code in ('R', 'S', 'E', 'Z'), (
                f"Invalid CIP code '{code}' at locant {locant} in: {result}"
            )

        # Verify molecule has at least as many stereo as name claims
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            rdCIPLabeler.AssignCIPLabels(mol)
            mol_rs = sum(1 for a in mol.GetAtoms() if a.HasProp('_CIPCode'))
            mol_ez = sum(1 for b in mol.GetBonds() if b.HasProp('_CIPCode'))
            name_rs = sum(1 for _, c in stereo_in_name if c in ('R', 'S'))
            name_ez = sum(1 for _, c in stereo_in_name if c in ('E', 'Z'))
            assert name_rs <= mol_rs, (
                f"More R/S in name ({name_rs}) than in molecule ({mol_rs}): {smiles}"
            )
            assert name_ez <= mol_ez, (
                f"More E/Z in name ({name_ez}) than in molecule ({mol_ez}): {smiles}"
            )
