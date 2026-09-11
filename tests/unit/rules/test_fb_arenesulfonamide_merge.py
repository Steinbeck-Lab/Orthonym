"""F-B — N,ring-locant merging for N-substituted arenesulfonamides + the
 ``-1-`` locant on a ring-substituted (di-substituted) benzenesulfonamide.

Blue Book authority (verified verbatim 2026-08-08)
--------------------------------------------------
**(c)** (``the Blue Book Blue Book``): the locant ``1`` is omitted
"in monosubstituted homogeneous monocyclic rings" — example ``cyclohexanethiol
(PIN)``. A benzene ring bearing BOTH the sulfonamide suffix AND a ring substituent
is DI-substituted, so (``:2869``, deny-by-default) cites the ``1``.
The whole arenesulfon* family carries it: ``4-aminobenzene-1-sulfonic acid (PIN)``
(``:31174``), ``2-(4-aminobenzene-1-sulfonamido)-1,3-thiazole-5-carboxylic acid
(PIN)`` (``:33034``). Zero no-``-1-`` di-substituted ``methylbenzenesulfon*`` PINs
exist in the Blue Book.

**** N-substitution + ** / ** alphanumerical order:
the italic-``N`` locants and the ring numerals form ONE merged, alphabetised
prefix list. ``N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)`` (``:32879``) and
``3-chloro-N-(2-chlorophenyl)naphthalene-2-sulfonamide (PIN)`` (``:32881``) fix the
ordering: same substituent name is pooled (``N,4-dimethyl``), the italic ``N``
sorts ahead of a numeral within the pool, and distinct names order alphabetically
(``3-chloro`` before the N-prefix because "chloro" < "chloro-phenyl").

Every expected string below was OPSIN round-trip InChIKey-verified against its
SMILES before being asserted here.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.sulfonamides import n_substituted_sulfonamide_name
from orthonym.rules.benzene import name_benzene_derivative


def _producer(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"bad test SMILES {smiles!r}"
    return n_substituted_sulfonamide_name(mol)


# ----------------------------------------------------------------------------
# PART 1 — the ``-1-`` locant on a ring-substituted PRIMARY benzenesulfonamide
# (each substituent already names alone; only the -1- was omitted). name_compound
# with the gate ON is safe here: every target round-trips, so the gate cannot
# suppress it.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("NS(=O)(=O)c1ccc(C)cc1",   "4-methylbenzene-1-sulfonamide"),
    ("NS(=O)(=O)c1cccc(Cl)c1",  "3-chlorobenzene-1-sulfonamide"),
    ("NS(=O)(=O)c1ccc(C)cc1C",  "2,4-dimethylbenzene-1-sulfonamide"),
])
def test_part1_ring_substituted_primary_cites_locant_one(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # mono benzenesulfonamide stays WITHOUT -1- (monosubstituted ring, (c))
    ("NS(=O)(=O)c1ccccc1", "benzenesulfonamide"),
    # sulfonic-acid sibling was already correct — a regression witness
    ("OS(=O)(=O)c1ccc(C)cc1", "4-methylbenzene-1-sulfonic acid"),
    # retained benzoic / benzamide never take -1- (retained-name path, untouched)
    ("OC(=O)c1ccc(C)cc1", "4-methylbenzoic acid"),
    ("NC(=O)c1ccc(C)cc1", "4-methylbenzamide"),
])
def test_part1_protects_monosubstituted_and_retained(smiles, expected):
    assert name_compound(smiles) == expected


# ----------------------------------------------------------------------------
# PART 2 — N,ring merge. The producer returns the merged PIN for a benzene parent.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("Cc1ccc(S(=O)(=O)NC)cc1",   "N,4-dimethylbenzene-1-sulfonamide"),
    ("CCNS(=O)(=O)c1ccc(C)cc1",  "N-ethyl-4-methylbenzene-1-sulfonamide"),
    ("CNS(=O)(=O)c1cccc(Cl)c1",  "3-chloro-N-methylbenzene-1-sulfonamide"),
    ("Cc1ccc(S(=O)(=O)N(C)C)cc1", "N,N,4-trimethylbenzene-1-sulfonamide"),
    ("Cc1ccc(S(=O)(=O)Nc2ccccc2)cc1", "4-methyl-N-phenylbenzene-1-sulfonamide"),
])
def test_part2_merged_n_ring_producer(smiles, expected):
    assert _producer(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    ("Cc1ccc(S(=O)(=O)NC)cc1",   "N,4-dimethylbenzene-1-sulfonamide"),
    ("CCNS(=O)(=O)c1ccc(C)cc1",  "N-ethyl-4-methylbenzene-1-sulfonamide"),
    ("CNS(=O)(=O)c1cccc(Cl)c1",  "3-chloro-N-methylbenzene-1-sulfonamide"),
])
def test_part2_merged_n_ring_end_to_end(smiles, expected):
    """The gate is ON here; these all round-trip, so the gate never suppresses."""
    assert name_compound(smiles) == expected


# The bare (ring-monosubstituted) N-substituted cases must be UNCHANGED — no -1-.
@pytest.mark.parametrize("smiles,expected", [
    ("CNS(=O)(=O)c1ccccc1",    "N-methylbenzenesulfonamide"),
    ("CCNS(=O)(=O)c1ccccc1",   "N-ethylbenzenesulfonamide"),
    ("CN(C)S(=O)(=O)c1ccccc1", "N,N-dimethylbenzenesulfonamide"),
])
def test_part2_bare_ring_n_substituted_unchanged(smiles, expected):
    assert _producer(smiles) == expected


# ----------------------------------------------------------------------------
#, the Blue Book `2-(dimethylsulfamoyl)benzene-1-sulfonic acid`,
#:32985 `phenylsulfamoyl`): a demoted N-substituted sulfonamide is the ENCLOSED
# `{N-substituents}sulfamoyl` prefix, the N-substituents cited WITHOUT the italic N.
# (Supersedes F-B's earlier fail-closed stopgap for this class -- building the whole
# class per a project rule. Each expected string is OPSIN round-trip InChIKey-verified.)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)c1ccc(S(=O)(=O)NC)cc1",       "4-(methylsulfamoyl)benzoic acid"),
    ("OC(=O)c1ccc(S(=O)(=O)N(C)C)cc1",     "4-(dimethylsulfamoyl)benzoic acid"),
    ("OS(=O)(=O)c1ccccc1S(=O)(=O)N(C)C",   "2-(dimethylsulfamoyl)benzene-1-sulfonic acid"),
    ("OC(=O)c1ccc(S(=O)(=O)Nc2ccccc2)cc1", "4-(phenylsulfamoyl)benzoic acid"),
])
def test_risk7_n_substituted_sulfamoyl_prefix(smiles, expected):
    assert name_compound(smiles) == expected


def test_distinct_n_substituents_sulfamoyl_fails_closed():
    """Distinct N-substituents need the nested `ethyl(methyl)sulfamoyl` form, which
    is not built -> fail closed (abstains), never a wrong molecule."""
    from orthonym.errors import is_failure_name
    out = name_compound("OC(=O)c1ccc(S(=O)(=O)N(C)CC)cc1")
    assert not out or is_failure_name(out)


# The PRIMARY sulfonamide demotion (`4-sulfamoylbenzoic acid`) must survive: it has
# no N-substituent to drop, so it is a legitimate PIN and is a regression witness.
def test_primary_sulfonamide_demotion_still_names():
    assert name_compound("OC(=O)c1ccc(S(=O)(=O)N)cc1") == "4-sulfamoylbenzoic acid"


# A genuine CHAIN branch point (parent 2-methylpropane-2-sulfonamide) is out of
# scope — the merge is built for arenes; the chain-branch case stays refused.
def test_chain_branch_parent_still_refused():
    assert _producer("CC(C)(C)S(=O)(=O)NC") is None


# A PRIMARY di-sulfonamide (no N-substituents) names normally.
def test_primary_disulfonamide_names():
    assert name_compound("NS(=O)(=O)c1ccc(S(=O)(=O)N)cc1") == "benzene-1,4-disulfonamide"


# 0-WRONG regression guard (review-caught): a MULTI-instance sulfonamide carrying
# N-substituents needs the N^1/N^3 superscript locants (not built) — the benzene
# producer must FAIL CLOSED, never emit `benzene-1,4-disulfonamide` (which silently
# drops the N-methyls, a wrong constitution).
@pytest.mark.parametrize("smiles", [
    "O=S(=O)(NC)c1ccc(S(=O)(=O)NC)cc1",   # N,N'-dimethyl disulfonamide
    "NS(=O)(=O)c1ccc(S(=O)(=O)NC)cc1",     # mixed primary + N-methyl
])
def test_multi_instance_n_substituted_sulfonamide_fails_closed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    out = name_benzene_derivative(mol)
    assert out is None or "disulfonamide" not in out or "N" in out


# Adversarial N-substituent shapes (review pre-empt) — each must name RT-exact
# end-to-end (some route through the F-B benzene producer, some through a sibling
# handler; the end-to-end contract is what matters). Each string is OPSIN-verified,
# and the F-B producer that cannot claim one (e.g. N-acyl, where the acyl competes
# in name_benzene_derivative) FAILS CLOSED honestly rather than emit a wrong name.
@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)NS(=O)(=O)c1ccc(C)cc1",  "N-acetyl-4-methylbenzene-1-sulfonamide"),
    ("OCCNS(=O)(=O)c1ccc(C)cc1",     "N-(2-hydroxyethyl)-4-methylbenzene-1-sulfonamide"),
    ("CC(C)(C)NS(=O)(=O)c1ccc(C)cc1", "N-tert-butyl-4-methylbenzene-1-sulfonamide"),
    ("c1ccncc1NS(=O)(=O)c1ccc(C)cc1", "4-methyl-N-(pyridin-3-yl)benzene-1-sulfonamide"),
    ("ClCNS(=O)(=O)c1ccc(C)cc1",     "N-(chloromethyl)-4-methylbenzene-1-sulfonamide"),
])
def test_adversarial_n_substituent_shapes(smiles, expected):
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------
# 0-WRONG blockers found by the cross-model review review (the deterministic
# sweep missed all four). The F-B producer must FAIL CLOSED on each — a
# gate-independent OPSIN InChIKey re-anchor (8afa533c precedent) rejects any
# merged name that denotes a different molecule, plus a fused-ring guard.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,why", [
    # B1: a FUSED arene (naphthalene) must NOT be renamed as benzene (drops carbons).
    ("Clc1ccccc1NS(=O)(=O)c1cc2ccccc2cc1Cl", "fused arene, not benzene"),
    # B2: a benzylic stereocentre the fragment namer drops -> re-anchor rejects.
    ("C[C@H](c1ccccc1)NS(=O)(=O)c1ccc(C)cc1", "benzylic stereodescriptor dropped"),
    # B3: an N-substituent bearing a functional group mis-named (-COOH -> formyl).
    ("Cc1ccc(S(=O)(=O)N(C)C(=O)O)cc1", "N-branch -COOH mis-named formyl"),
])
def test_review_blockers_producer_fails_closed(smiles, why):
    """The F-B producer must never emit a wrong molecule for these (0-wrong)."""
    assert _producer(smiles) is None, why


@pytest.mark.opsin_gate
def test_b2_benzylic_stereo_never_ships_wrong_at_gate_on():
    """B2 was SHIPPING the stereo-dropped wrong molecule at the default gate via
    the BBR stereo carve-out. With the re-anchor it must not (it abstains)."""
    out = name_compound("C[C@H](c1ccccc1)NS(=O)(=O)c1ccc(C)cc1")
    assert out != "4-methyl-N-(1-phenylethyl)benzene-1-sulfonamide"
