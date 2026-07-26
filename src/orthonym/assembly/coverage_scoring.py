"""Graduated confidence scoring for coverage gate candidate selection.

Replaces the binary accept/reject coverage gate (DROP-20) with a continuous
multi-factor scoring system.  Each handler (complex_ring, heterocycle, benzene)
produces a CandidateName with a 0.0-1.0 confidence score derived from four
factors.  The best candidate is selected and returned; no computed name is
ever discarded.

Four scoring factors:
  1. ratio         -- name-length / heavy-atom-count heuristic (normalised 0-1)
  2. atom_coverage -- fraction of heavy atoms covered by the parent structure
     ONLY when ``compute_confidence`` is given ``parent_atom_indices``.  See
     the PROVENANCE WARNING below: in the production path it never is.
  3. fg_recognition -- fraction of detected FGs with known naming forms
  4. substituent_completeness -- fraction of substituents reflected in the name

Thread-local confidence store allows callers to retrieve metadata after
assemble_name() returns without changing its str return type.

PROVENANCE WARNING (v29 C4) -- ``factors['atom_coverage']`` is NOT a coverage
measurement in the production path.  ``CandidatePool.add()``
(``candidate_pool.py``, "Risk 1" mitigation) deliberately calls
``compute_confidence`` WITHOUT ``parent_atom_indices`` in order to keep
confidence byte-identical, and attaches ``parent_atom_indices`` to the
CandidateName only POST-HOC.  Consequently ``atom_coverage`` is either

  * ``ratio_score`` -- a pure name-LENGTH proxy, numerically identical to
    ``factors['ratio']`` (provenance ``estimated_name_length``), or
  * a hard ``1.0`` from the retained-name boost (provenance
    ``retained_name_boost``).

It is a real fraction-of-atoms measurement (provenance ``measured``) only when
a caller passes ``parent_atom_indices`` -- which no production caller does.
Every record therefore carries ``coverage_provenance`` so that no consumer can
mistake the estimate for a measurement.  Do not remove that field, and do not
present ``atom_coverage`` as verified coverage without checking it.
"""

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:  # pragma: no cover - typing-only import (avoids runtime cycle)
    from .name_tree import NameTreeNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verification tri-state and coverage provenance (v29 C4)
# ---------------------------------------------------------------------------
# Project invariant: a proof must never be CONFIDENTLY WRONG. Insufficient
# evidence => 'unverified', never 'verified' and never 'refuted'. Applied here
# to atom coverage: a record that measured nothing must say so, rather than
# reporting a number (1.0 OR 0.0) that implies a verdict was reached.
VERIFICATION_VERIFIED = 'verified'
VERIFICATION_UNVERIFIED = 'unverified'
VERIFICATION_REFUTED = 'refuted'

# Where factors['atom_coverage'] came from. See the module PROVENANCE WARNING.
COVERAGE_MEASURED = 'measured'                        # real |parent| / |heavy|
COVERAGE_ESTIMATED_NAME_LENGTH = 'estimated_name_length'  # name-length proxy
COVERAGE_RETAINED_NAME_BOOST = 'retained_name_boost'      # hard-set to 1.0

# Handler sentinel for "no candidate was ever scored". Distinct from 'unknown'
# (the store's never-written state) so the two can be told apart, and NOT a
# real handler id -- 'direct' was, which let an unmeasured record masquerade
# as a scored one.
HANDLER_UNMEASURED = 'unmeasured'


def unmeasured_confidence(name: str = '', handler: str = HANDLER_UNMEASURED) -> dict:
    """The single honest record for "nothing measured this name's coverage".

    One shared constructor so the ``name()`` path and the
    ``name_with_confidence()`` path cannot disagree (they did: ``name()``
    reported ``handler='unknown', confidence=0.0``; ``name_with_confidence()``
    fabricated ``handler='direct', confidence=1.0`` with all four factors at
    ``1.0``, i.e. a perfect score on a name nothing had scored).

    ``confidence`` is ``None`` -- not ``1.0`` (a fabricated pass) and not
    ``0.0`` (a fabricated fail). ``factors`` is empty so no consumer can read
    an ``atom_coverage`` that was never computed.
    """
    return {
        'name': name,
        'confidence': None,
        'verification': VERIFICATION_UNVERIFIED,
        'factors': {},
        'coverage_provenance': None,
        'handler': handler,
        'atom_to_locant': None,
        'is_phenol_benzene': None,
    }


# ---------------------------------------------------------------------------
# CandidateName dataclass
# ---------------------------------------------------------------------------

@dataclass
class CandidateName:
    """A candidate IUPAC name with multi-factor confidence metadata."""

    name: str
    handler: str  # 'complex_ring', 'heterocycle', 'benzene', 'chain'
    # 0.0-1.0 aggregate score, or None for "nothing scored this candidate"
    # (v29 C4). None is NOT a low score -- consumers must branch on it, never
    # coerce it to 0.0 (a fabricated FAIL) or 1.0 (a fabricated PASS).
    confidence: Optional[float] = 0.0
    factors: Dict[str, float] = field(default_factory=dict)
    # factors keys: 'ratio', 'atom_coverage', 'fg_recognition',
    #               'substituent_completeness'
    # v29 C4: provenance of factors['atom_coverage'] -- one of
    # COVERAGE_MEASURED / COVERAGE_ESTIMATED_NAME_LENGTH /
    # COVERAGE_RETAINED_NAME_BOOST, or None when factors is empty. Set by
    # compute_confidence(); purely descriptive, so it cannot perturb the
    # weighted sum (byte-identical safe). Consumers MUST check it before
    # treating atom_coverage as a coverage measurement -- in the production
    # path it is never COVERAGE_MEASURED (see module PROVENANCE WARNING).
    coverage_provenance: Optional[str] = None
    # New in Phase 145.1: parent atom indices populated POST-HOC by
    # CandidatePool.add() (see candidate_pool.py). Used by
    # ParentCorrectnessScorer to compare against OPSIN-extracted
    # reference parent. None when handler doesn't report parent atoms
    # (most direct-return handlers; benzene/heterocycle/complex_ring
    # populate via features.benzene_ring / features.principal_ring /
    # ring atoms from _assemble_complex_ring_name()).
    # CRITICAL: this field is set AFTER compute_confidence() returns.
    # Do NOT pass it as a positional arg to compute_confidence — that
    # changes atom_coverage and breaks byte-identical (see RESEARCH §9.1
    # Risk 3 / PATTERNS Risk 1).
    parent_atom_indices: Optional[set] = None
    # Phase 146 CD-01: principal-characteristic-group count for the parent
    # structure. Populated POST-HOC by CandidatePool.add() via
    # _count_pcgs_in_parent. Same Risk 1 mitigation as parent_atom_indices:
    # NEVER passed into compute_confidence() — that would break the
    # byte-identical guarantee. Used by Tier-1 cascade filter
    # _filter_max_pcg_count (P-44.1.1).
    # Default None means "not yet computed" (the cascade treats None as 0).
    # Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1.1
    parent_pcg_count: Optional[int] = None
    # Phase 147 D-03: ring-type authoritative IUPAC locants used by
    # Tier-1 cascade step 6 (_filter_lowest_locants) per P-44.4.1.4+.
    # Populated POST-HOC by CandidatePool.add() — same Risk 1 mitigation
    # as parent_atom_indices: NEVER passed into compute_confidence()
    # (would break byte-identical guarantee).
    # Shape: {"iupac_locants": {atom_idx: int | (int, str) tuple}} or None.
    # None means "no authoritative locants available" (spiro/VB stubs
    # per D-06, or unsupported ring type) — cascade step 6 short-circuits
    # via _has_iupac_locants probe at candidate_pool.py:501-514.
    # Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.4+
    ring_info: Optional[Dict[str, Any]] = None
    # Phase 165 SCORE-01: structured Name-Tree IR for this candidate. Populated
    # POST-HOC by CandidatePool.add(tree=...) — same Risk 1 mitigation as
    # parent_atom_indices / ring_info: NEVER passed into compute_confidence()
    # (would break the byte-identical guarantee). None means "no tree yet"
    # (coarse-bucket counted in Plan 04). pool.best().tree surfaces the winner's.
    tree: Optional["NameTreeNode"] = None
    # Phase 166 SCORE-03: POST-HOC per-node {id(node): NodeScores}, same Risk 1
    # mitigation as tree / parent_atom_indices / ring_info: NEVER passed into
    # compute_confidence() (would break the byte-identical guarantee). None / {}
    # means "no per-substring signal" — the production no-reference path, or a
    # coarse tree. pool.best().node_scores surfaces the winner's map for the
    # Plan 04 SCORE-06 measurement.
    node_scores: Optional[dict] = None
    # Phase 177 WSB-01 (D-04): authoritative {atom_idx: 1-indexed locant} for the
    # parent structure, threaded to the namer backstop _final_stereo_check so it
    # can inject stereodescriptors on the chain / non-phenol-benzene cohort.
    # POST-HOC — NEVER passed into compute_confidence() (byte-identity Risk 1),
    # same mitigation as parent_atom_indices / ring_info / tree / node_scores.
    # None for direct-return handlers / complex_ring / polycyclic / phenol benzene
    # (no authoritative map -> backstop stays log-only there, D-05/D-06).
    atom_to_locant: Optional[Dict[int, int]] = None
    # Phase 177 WSB-01 (D-05): True when the benzene parent is a PHENOL, so the
    # backstop excludes it from the inject allowlist (phenol benzene stays
    # log-only). None/False otherwise. POST-HOC, never into compute_confidence.
    is_phenol_benzene: Optional[bool] = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Handler priority for tiebreaking (higher = more specific)
HANDLER_PRIORITY: Dict[str, int] = {
    # Ring handlers (Tier A: use candidate collection + select_best_candidate)
    'complex_ring': 4,
    'heterocycle': 3,
    'benzene': 2,
    'chain': 1,
    # Tier B specialized handlers (use confidence gating before return)
    'oxime': 5,
    'hydrazone': 5,
    'isocyanate': 5,
    'isothiocyanate': 5,
    'carbamic_acid': 5,
    'carbamate': 5,
    'urea': 5,
    'guanidine': 5,
    'cyanamide': 5,
    'sulfoxide': 5,
    'sulfone': 5,
    'thioether': 5,
    'boronic_acid': 5,
    'partial_sat': 4,
    # Direct-return handlers (no gating needed: self-gating or retained names)
    'polycyclic': 6,
    'acid_halide': 5,
    'anhydride': 5,
    'lactone': 5,
    'lactam': 5,
    'ring_ester': 5,
    'polyfunctional': 4,
    'ester': 5,
    'multi_ester': 5,
    'phosphine_oxide': 5,
    'phosphate_ester': 5,
    'phosphine': 5,
    'phosphinic_acid': 5,
    'ring_assembly': 5,
    'ring_nitrile': 5,
    'amide': 5,
    # Phase 145.1 ADDS (ISS-002 remediation): single source of truth for
    # candidate_pool.HANDLER_POLICIES. Plan 01's HANDLER_POLICIES dict pulls
    # every priority from this dict; missing entries break with KeyError.
    # n_oxide and amine handlers exist at composer.py:740 and composer.py:1383
    # respectively; both have direct-return semantics with priority 5
    # (matches their Tier B / direct-return peers). simple_molecule is the
    # terminal fallback at composer.py:1389; priority 1 matches chain
    # (lowest priority — only wins when nothing else fires). polyfunctional
    # is ALREADY present above with priority 4.
    'n_oxide': 5,
    'amine': 5,
    'simple_molecule': 1,
}

# Confidence threshold for Tier B handler gating.
# Handlers producing names with confidence below this threshold fall through
# to the next handler in the cascade. Calibrated in Phase 81 on ChEBI 500.
CONFIDENCE_GATE_THRESHOLD: float = 0.40

# Calibrated weights for multi-factor confidence scoring
# Derivation: 
# Benchmark: ChEBI 500-sample, seed=123, n=500
# Date: 2026-02-27
# Grid steps: 11, cross-validated (400 train / 100 test)
# InChI RT accuracy: 103/500 (20.6%) -- matches pre-Phase-81 baseline (20.0%)
# Calibration method: confidence separation maximization (all molecules are
# single-candidate under current architecture, so weights affect confidence
# quality rather than selection outcomes). Weights proportional to per-factor
# discriminative power between InChI-matching and non-matching names:
#   fg_recognition (0.062) > substituent_completeness (0.047) >
#   ratio (0.042) = atom_coverage (0.042)
# Pre-calibration baseline: ratio=0.30, atom_cov=0.30, fg=0.25, sub=0.15
#
# Phase 146 (D-07): two dicts FACTOR_WEIGHTS_V17 and FACTOR_WEIGHTS_V18 are
# declared; FACTOR_WEIGHTS is bound to one or the other at module-import time
# based on the ORTHONYM_USE_V18_WEIGHTS env var (default 'false' = V17).
# The V18 dict adds the multiple_bond_count factor (D-06, P-44.4.1.2).
# Calibrated weight values are written by Plan 04 grid search.
#
# ----------------------------------------------------------------
# Phase 146 D-07/D-08: feature-flag-controlled FACTOR_WEIGHTS dispatch.
# ORTHONYM_USE_V18_WEIGHTS env var read at module-import time.
# Default 'false' (V17 active) during the post-148 soak week per D-07.
# Rollback one-liner:
#   ORTHONYM_USE_V18_WEIGHTS=false ORTHONYM_SELECTION_MODE=first_applicable pytest tests/
# Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1.1, P-44.4.1.2
# ----------------------------------------------------------------
_USE_V18 = os.getenv('ORTHONYM_USE_V18_WEIGHTS', 'false').strip().lower() == 'true'

# V17 weights (Phase 145.1 + 145.2 baseline — byte-identical preserved).
# ratio=0.0 from 145.2 D-09-a.1 (zero IUPAC justification per Blue Book grep).
# parent_correctness=0.0 from 145.1 D-10 (scaffolded; raised to calibrated value in V18).
#
# BYTE-IDENTICAL PROOF (D-14):
# - Python 3.7+ dict iteration is insertion-order-deterministic.
# - compute_confidence's sum-loop iterates FACTOR_WEIGHTS in insertion order.
# - parent_correctness inserted LAST so the existing 4 weighted-sum terms
#   accumulate first; the 5th term (0.0 * factor) contributes exactly 0.0
#   by IEEE 754 (x + 0.0 = x for finite x).
# - Therefore confidence values are byte-identical to pre-145.1 code.
#
# RISK 2 (PATTERNS Risk 2): inserting parent_correctness in the MIDDLE would
# change weighted-sum order; floating-point summation is non-associative,
# so the rounded result may differ in the 5th decimal. DO NOT REORDER.
FACTOR_WEIGHTS_V17: Dict[str, float] = {
    'ratio': 0.0,
    'atom_coverage': 0.20,
    'fg_recognition': 0.35,
    'substituent_completeness': 0.25,
    'parent_correctness': 0.0,
}

# V18 weights (Phase 146 calibrated by grid search in Plan 04).
# CALIBRATED by Plan 04 grid search (2026-04-24, 52 configs, anti-overfit delta=0.0).
# Grid: 5^5 with sum-to-1.0 constraint; parent_correctness DROPPED per CD-03
# (std=0.0 until Phase 148 wires features.parent_selection_result); 52 surviving
# configs evaluated over ~2.5 hrs wall-clock on 4 workers.
# multiple_bond_count is the NEW factor per D-06 (P-44.4.1.2) — calibration
# retained it at 0.25 (D-20 NOT triggered; factor has meaningful discriminative
# signal even under CD-03's production path).
# Insertion order: multiple_bond_count is APPENDED LAST per D-14 IEEE 754
# invariant (when V17 loop encounters this dict via reload, the 6th term
# is added LAST so prior 5 sums are byte-identical to V17).
# Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.2
FACTOR_WEIGHTS_V18: Dict[str, float] = {
    'ratio': 0.0,                       # permanent 0.0 (Phase 145.2 D-09-a.1)
    'atom_coverage': 0.05,              # RECALIBRATED via Phase 148.2 boundary-extension grid
    'fg_recognition': 0.25,             # RECALIBRATED (post-148 cascade unblock shifts optimum)
    'substituent_completeness': 0.25,   # RECALIBRATED
    'parent_correctness': 0.35,         # RECALIBRATED — Phase 148.2 INTERIOR optimum (was 0.25 at 148.1 corner)
    'multiple_bond_count': 0.10,        # RECALIBRATED (lowered to keep sum-to-1.0)
}
# Sum = 1.00 (0.00 + 0.05 + 0.25 + 0.25 + 0.35 + 0.10); sum-to-1.0 constraint satisfied.
# Phase 148.2 partial (100/255 configs; D-07 5-hr abort): Wilson-95-LB winner;
# CD-02 SATISFIED: parent_correctness=0.35 is INTERIOR to [0.30, 0.50] grid; no further
# boundary extension needed. Top-10 cluster: PC=0.30 (5/10), PC=0.35 (3/10), PC=0.40 (1/10), PC=0.45 (1/10).

# Active weights — flipped via ORTHONYM_USE_V18_WEIGHTS env var.
FACTOR_WEIGHTS: Dict[str, float] = (
    FACTOR_WEIGHTS_V18 if _USE_V18 else FACTOR_WEIGHTS_V17
)

# Confidence bands for structured logging
CONFIDENCE_HIGH: float = 0.75    # DEBUG level
CONFIDENCE_MEDIUM: float = 0.45  # INFO level
# Below MEDIUM: WARNING level


# ---------------------------------------------------------------------------
# Factor computation helpers
# ---------------------------------------------------------------------------

def _is_retained_scaffold_name(name: str) -> bool:
    """Check if a name is a recognised scaffold name from naming databases.

    Checks RETAINED_NAMES (trivial/common names) and fused heterocycle
    tables (adenine, indole, purine, etc.).

    This is a general rule: "retained names for recognised scaffolds are
    always valid regardless of molecule size."
    """
    name_lower = name.lower()

    # Check RETAINED_NAMES (Phase 150 D-05: consult merged ALL_RETAINED_NAMES)
    try:
        from ..data import ALL_RETAINED_NAMES as RETAINED_NAMES
        for _smi, retained in RETAINED_NAMES.items():
            if retained.lower() == name_lower:
                return True
    except ImportError:
        pass

    # Check fused heterocycle database (nucleobases, indole, purine, etc.)
    try:
        from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
        for _smi, entry in FUSED_HETEROCYCLE_DATA.items():
            entry_name = entry.get('name', '')
            if entry_name.lower() == name_lower:
                return True
    except ImportError:
        pass

    return False


def _is_core_retained_name(name: str) -> bool:
    """Check if a name is a core retained name that should ALWAYS be boosted.

    Core retained names identify biologically significant scaffolds that are
    valid as names regardless of molecule size (e.g., adenine in a nucleotide).
    These correspond to fused heterocycle entries with ``is_retained_name: True``
    in the FUSED_HETEROCYCLE_DATA.

    Regular retained names (benzene, toluene, etc.) are only boosted for small
    molecules (total_heavy <= 15) to prevent incorrect boosting.
    """
    name_lower = name.lower()
    try:
        from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
        for _smi, entry in FUSED_HETEROCYCLE_DATA.items():
            if (entry.get('name', '').lower() == name_lower
                    and entry.get('is_retained_name', False)):
                return True
    except ImportError:
        pass
    return False


def _compute_fg_recognition(features: Any) -> float:
    """Fraction of detected functional groups with known naming forms.

    The principal group is always counted as recognised (it becomes the
    suffix).  Non-principal groups are counted if they have a known prefix
    form via ``get_prefix()``.

    Returns 1.0 when there are no functional groups to miss.
    """
    fg_dict = getattr(features, 'functional_groups', {})
    if not fg_dict:
        return 1.0

    total_fg_instances = sum(len(matches) for matches in fg_dict.values())
    if total_fg_instances == 0:
        return 1.0

    # Principal group is always recognised (becomes suffix)
    principal = getattr(features, 'principal_group', None)
    principal_atoms = getattr(features, 'principal_group_atoms', None) or []
    recognised = len(principal_atoms)

    # Non-principal groups: count those with known prefix forms
    # Also count FGs that are named by substitution (ethers, thioethers,
    # aromatic ethers) even though they don't have a standard prefix in
    # the PREFIX_FORMS table -- they are handled by dedicated naming code.
    _SUBSTITUTIVE_FGS = frozenset({
        'ether', 'thioether', 'aromatic_ether', 'sulfoxide', 'sulfone',
    })
    from ..rules.seniority import get_prefix
    for fg_name, matches in fg_dict.items():
        if fg_name == principal:
            continue  # Already counted
        prefix = get_prefix(fg_name)
        if prefix or fg_name in _SUBSTITUTIVE_FGS:
            recognised += len(matches)

    return min(recognised / total_fg_instances, 1.0)


def _compute_substituent_completeness(name: str, features: Any) -> float:
    """Estimate fraction of substituents represented in the name.

    Heuristic: count expected substituent attachment points on the parent
    structure, then count locant-prefixed groups in the name as evidence
    of named substituents.

    Returns 1.0 when there are no expected substituents.
    """
    chain_subs = len(getattr(features, 'substituents', {}))
    ring_subs = len(getattr(features, 'ring_substituents', {}))
    het_subs = len(getattr(features, 'heterocycle_substituents', {}))
    benz_subs = len(getattr(features, 'benzene_substituents', {}))
    total_expected = chain_subs + ring_subs + het_subs + benz_subs

    if total_expected == 0:
        return 1.0

    # Heuristic: count locant-prefixed groups in the name.
    # Monosubstituted rings (total_expected == 1) often omit the locant
    # (e.g., "hexoxybenzene" not "1-hexoxybenzene"), so if the name is
    # longer than the parent ring name, count the substituent as present.
    locant_groups = re.findall(r'\d+[,-]', name)
    named_subs = min(len(locant_groups), total_expected)

    # For monosubstituted cases without locants, check if the name
    # contains more than just the parent ring name (indicates a prefix).
    if named_subs == 0 and total_expected <= 2 and len(name) > 8:
        # The name has enough characters to suggest substituent prefixes
        # even without explicit locants.
        named_subs = min(1, total_expected)

    return min(named_subs / total_expected, 1.0)


def _compute_multiple_bond_count(
    features: Any,
    parent_atom_indices: Optional[Set[int]],
) -> float:
    """Count (double + triple) bonds where both endpoints are in parent_atom_indices.

    Phase 146 D-06: parent atoms ONLY (NOT entire molecule), per P-44.4.1.2
    which says "ring system or chain" = the parent skeleton. Substituent
    multiple bonds (e.g. a nitrile substituent's C#N triple bond) do NOT
    contribute because the bond's endpoints are not both in parent_atom_indices.

    Returns a numeric count (e.g., 0, 1, 2, 3) as a float. The weighted-sum
    in compute_confidence multiplies by FACTOR_WEIGHTS_V18['multiple_bond_count']
    and clamps the final confidence to [0, 1] — so a parent with 3 multiple
    bonds and a 0.10 weight contributes 0.30 to confidence (capped if total > 1).

    Note: This helper is functionally identical to
    candidate_pool._count_multiple_bonds_in_atom_set; the duplication is
    intentional to avoid a coverage_scoring <-> candidate_pool circular import
    (candidate_pool imports compute_confidence from coverage_scoring, so the
    reverse direction is forbidden). See RESEARCH §2.5.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.2
    """
    mol = getattr(features, 'mol', None)
    if mol is None or not parent_atom_indices:
        return 0.0
    try:
        from rdkit import Chem
    except ImportError:
        return 0.0
    atom_set = set(parent_atom_indices)
    count = 0
    for bond in mol.GetBonds():
        if (bond.GetBeginAtomIdx() in atom_set
                and bond.GetEndAtomIdx() in atom_set):
            bt = bond.GetBondType()
            if bt == Chem.BondType.DOUBLE or bt == Chem.BondType.TRIPLE:
                count += 1
    return float(count)


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_confidence(
    name: str,
    handler: str,
    features: Any,
    parent_atom_indices: Optional[Set[int]] = None,
) -> CandidateName:
    """Score a candidate name on the 4 confidence factors.

    Args:
        name: The generated IUPAC name string.
        handler: Which handler produced this name (e.g. 'complex_ring').
        features: MolecularFeatures object with perceived molecular data.
        parent_atom_indices: Set of atom indices covered by the parent name.
            When ``None``, atom_coverage is estimated from name length.

    Returns:
        CandidateName with individual factor scores and aggregate confidence.
    """
    mol = features.mol
    total_heavy = mol.GetNumHeavyAtoms()

    # Factor 1: name-length / HA ratio (normalised 0-1)
    ratio_raw = len(name) / max(total_heavy, 1)
    ratio_score = min(ratio_raw / 1.5, 1.0)

    # Factor 2: heavy atom coverage.
    # v29 C4: record WHICH of the three things this number actually is. No
    # production caller passes parent_atom_indices (candidate_pool.add()
    # withholds it as its "Risk 1" byte-identical mitigation), so in practice
    # this is the name-length proxy, numerically equal to ratio_score.
    if parent_atom_indices is not None:
        atom_cov = len(parent_atom_indices) / max(total_heavy, 1)
        coverage_provenance = COVERAGE_MEASURED
    else:
        atom_cov = ratio_score  # fallback: ESTIMATE from name length
        coverage_provenance = COVERAGE_ESTIMATED_NAME_LENGTH

    # Retained name confidence boost (general rule):
    # If the candidate name matches a recognised scaffold name, set all
    # factors to 1.0. Two tiers:
    #
    # 1. Core retained names (is_retained_name=True in fused heterocycle
    #    data, e.g. adenine): boosted ALWAYS, regardless of molecule size.
    #    These identify biologically significant scaffolds that are valid
    #    as names even in large molecules (nucleotides, cofactors).
    #
    # 2. Regular retained names (benzene, toluene, indole, etc.): boosted
    #    only for small molecules (total_heavy <= 15) to prevent incorrect
    #    boosting of e.g. "1H-indole" on a 36-atom benzamide.
    is_retained = _is_retained_scaffold_name(name)
    is_core = _is_core_retained_name(name) if is_retained else False

    if is_core or (is_retained and total_heavy <= 15):
        ratio_score = 1.0
        atom_cov = 1.0
        fg_recognition = 1.0
        sub_completeness = 1.0
        # v29 C4: this 1.0 is an assertion about the NAME being a recognised
        # scaffold, not a count of covered atoms. Never report it as measured.
        coverage_provenance = COVERAGE_RETAINED_NAME_BOOST
    else:
        # Factor 3: functional group recognition rate
        fg_recognition = _compute_fg_recognition(features)

        # Factor 4: substituent completeness
        sub_completeness = _compute_substituent_completeness(name, features)

    factors = {
        'ratio': round(ratio_score, 4),
        'atom_coverage': round(min(atom_cov, 1.0), 4),
        'fg_recognition': round(fg_recognition, 4),
        'substituent_completeness': round(sub_completeness, 4),
        # Phase 145.1: placeholder set to 0.0 so the sum-loop at lines 324-326
        # doesn't raise KeyError on the new FACTOR_WEIGHTS key. CandidatePool.add()
        # overwrites this POST-HOC with the real ParentCorrectnessScorer.score()
        # result. With FACTOR_WEIGHTS['parent_correctness'] = 0.0, the placeholder
        # contributes exactly 0.0 to confidence (IEEE 754) -- byte-identical safe.
        'parent_correctness': 0.0,
    }
    # Phase 146 D-06: multiple_bond_count factor (V18-only).
    # Guard ensures V17 path (where the key is absent from FACTOR_WEIGHTS)
    # produces byte-identical factors dict. POST-HOC overwrite in pool.add()
    # for V18 mode handles the case where this is initially 0.0
    # (parent_atom_indices=None at compute_confidence call time per Risk 1).
    # Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.2
    if 'multiple_bond_count' in FACTOR_WEIGHTS:
        factors['multiple_bond_count'] = round(
            _compute_multiple_bond_count(features, parent_atom_indices), 4
        )

    # Weighted linear combination
    confidence = sum(
        FACTOR_WEIGHTS[k] * factors[k] for k in FACTOR_WEIGHTS
    )
    confidence = round(min(max(confidence, 0.0), 1.0), 4)

    return CandidateName(
        name=name,
        handler=handler,
        confidence=confidence,
        factors=factors,
        # v29 C4: descriptive only -- assigned AFTER the weighted sum above,
        # and never a FACTOR_WEIGHTS key, so confidence stays byte-identical.
        coverage_provenance=coverage_provenance,
    )


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def select_best_candidate(candidates: List[CandidateName]) -> CandidateName:
    """Select the best candidate from a list of scored candidates.

    Selection criteria:
      1. Highest confidence score wins.
      2. On tie (within EPSILON=0.01): prefer more specific handler
         (complex_ring > heterocycle > benzene > chain).

    Args:
        candidates: Non-empty list of CandidateName objects.

    Returns:
        The CandidateName with the highest score (or best tiebreaker).

    Raises:
        AssertionError: If candidates list is empty.
    """
    assert candidates, "candidates list must be non-empty"

    EPSILON = 0.01
    best = candidates[0]
    for c in candidates[1:]:
        if c.confidence > best.confidence + EPSILON:
            best = c
        elif abs(c.confidence - best.confidence) <= EPSILON:
            # Tiebreak 1: prefer more specific handler
            c_pri = HANDLER_PRIORITY.get(c.handler, 0)
            b_pri = HANDLER_PRIORITY.get(best.handler, 0)
            if c_pri > b_pri:
                best = c
            elif c_pri == b_pri:
                # Tiebreak 2 (P-45.5): earlier in alphanumerical order.
                # Replaces the input-order first-wins fallback so equal
                # candidates resolve identically regardless of registration
                # or SMILES-spelling order.
                from .name_comparison import compare_names
                if compare_names(c.name, best.name) == -1:
                    best = c
    return best


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

def log_confidence(candidate: CandidateName) -> None:
    """Log confidence at structured severity levels.

    - confidence >= CONFIDENCE_HIGH:   DEBUG
    - confidence >= CONFIDENCE_MEDIUM: INFO
    - confidence <  CONFIDENCE_MEDIUM: WARNING
    """
    if candidate.confidence >= CONFIDENCE_HIGH:
        logger.debug(
            "Coverage gate: accepted handler=%s confidence=%.4f",
            candidate.handler, candidate.confidence,
        )
    elif candidate.confidence >= CONFIDENCE_MEDIUM:
        logger.info(
            "Coverage gate: moderate handler=%s confidence=%.4f factors=%s",
            candidate.handler, candidate.confidence, candidate.factors,
        )
    else:
        logger.warning(
            "Coverage gate: low-confidence handler=%s confidence=%.4f factors=%s",
            candidate.handler, candidate.confidence, candidate.factors,
        )


# ---------------------------------------------------------------------------
# Thread-local confidence store
# ---------------------------------------------------------------------------

_confidence_store = threading.local()


def store_confidence(candidate: CandidateName) -> None:
    """Store confidence metadata for current naming call (top-level only)."""
    _confidence_store.last_candidate = candidate


def retrieve_confidence() -> dict:
    """Retrieve stored confidence metadata as a dict.

    Returns a dict with keys: name, confidence, verification, factors,
    coverage_provenance, handler (plus atom_to_locant / is_phenol_benzene).

    When no metadata has been stored the record is the shared honest
    "unmeasured" record (v29 C4): ``confidence=None``,
    ``verification='unverified'``, ``factors={}``. ``handler`` stays
    ``'unknown'`` for that case because the ``name()`` quality gates use
    ``handler != 'unknown'`` as their "was anything actually scored?" guard.

    ``confidence`` was previously ``0.0`` here. That is a fabricated FAIL --
    the project invariant is that insufficient evidence yields 'unverified',
    never a verdict in either direction -- so it is now ``None``. Every
    production consumer already guards with ``is not None``
    (``namer.py`` quality gates) or ``.get(..., default)``.
    """
    candidate = getattr(_confidence_store, 'last_candidate', None)
    if candidate is None:
        return unmeasured_confidence(name='', handler='unknown')
    provenance = getattr(candidate, 'coverage_provenance', None)
    return {
        'name': candidate.name,
        'confidence': candidate.confidence,
        # v29 C4: a scored candidate is only 'verified' with respect to atom
        # coverage when the coverage number was actually MEASURED. In the
        # production path it never is (see module PROVENANCE WARNING), so
        # 'unverified' is the normal -- and honest -- outcome here.
        'verification': (
            VERIFICATION_VERIFIED if provenance == COVERAGE_MEASURED
            else VERIFICATION_UNVERIFIED
        ),
        'factors': dict(candidate.factors),
        'coverage_provenance': provenance,
        'handler': candidate.handler,
        # Phase 177 WSB-01 (D-04): surface the POST-HOC authoritative parent
        # locant map + phenol flag for the namer backstop.
        'atom_to_locant': getattr(candidate, 'atom_to_locant', None),
        'is_phenol_benzene': getattr(candidate, 'is_phenol_benzene', None),
    }


def clear_confidence() -> None:
    """Clear stored confidence metadata."""
    _confidence_store.last_candidate = None
