"""Graduated confidence scoring for coverage gate candidate selection.

Replaces the binary accept/reject coverage gate (DROP-20) with a continuous
multi-factor scoring system.  Each handler (complex_ring, heterocycle, benzene)
produces a CandidateName with a 0.0-1.0 confidence score derived from four
factors.  The best candidate is selected and returned; no computed name is
ever discarded.

Four scoring factors:
  1. ratio         -- name-length / heavy-atom-count heuristic (normalised 0-1)
  2. atom_coverage -- fraction of heavy atoms covered by the parent structure
  3. fg_recognition -- fraction of detected FGs with known naming forms
  4. substituent_completeness -- fraction of substituents reflected in the name

Thread-local confidence store allows callers to retrieve metadata after
assemble_name() returns without changing its str return type.
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CandidateName dataclass
# ---------------------------------------------------------------------------

@dataclass
class CandidateName:
    """A candidate IUPAC name with multi-factor confidence metadata."""

    name: str
    handler: str  # 'complex_ring', 'heterocycle', 'benzene', 'chain'
    confidence: float = 0.0  # 0.0-1.0 aggregate score
    factors: Dict[str, float] = field(default_factory=dict)
    # factors keys: 'ratio', 'atom_coverage', 'fg_recognition',
    #               'substituent_completeness'


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
FACTOR_WEIGHTS: Dict[str, float] = {
    'ratio': 0.20,
    'atom_coverage': 0.20,
    'fg_recognition': 0.35,
    'substituent_completeness': 0.25,
}

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

    # Check RETAINED_NAMES
    try:
        from ..data.retained_names import RETAINED_NAMES
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


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_confidence(
    name: str,
    handler: str,
    features: Any,
    parent_atom_indices: Optional[set] = None,
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

    # Factor 2: heavy atom coverage
    if parent_atom_indices is not None:
        atom_cov = len(parent_atom_indices) / max(total_heavy, 1)
    else:
        atom_cov = ratio_score  # fallback: estimate from name length

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
    }

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
            # Tiebreak: prefer more specific handler
            if HANDLER_PRIORITY.get(c.handler, 0) > HANDLER_PRIORITY.get(best.handler, 0):
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

    Returns a dict with keys: name, confidence, factors, handler.
    If no metadata has been stored, returns a default dict with empty values.
    """
    candidate = getattr(_confidence_store, 'last_candidate', None)
    if candidate is None:
        return {
            'name': '',
            'confidence': 0.0,
            'factors': {},
            'handler': 'unknown',
        }
    return {
        'name': candidate.name,
        'confidence': candidate.confidence,
        'factors': dict(candidate.factors),
        'handler': candidate.handler,
    }


def clear_confidence() -> None:
    """Clear stored confidence metadata."""
    _confidence_store.last_candidate = None
