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
    'complex_ring': 4,
    'heterocycle': 3,
    'benzene': 2,
    'chain': 1,
}

# Calibrated weights -- initial values, will be tuned in Plan 02
# These initial weights reproduce approximately the old binary gate behavior:
# ratio-dominated scoring, so names that pass the old threshold still win.
# Derivation: pre-calibration defaults, to be updated by calibrate_coverage_gate.py
FACTOR_WEIGHTS: Dict[str, float] = {
    'ratio': 0.30,
    'atom_coverage': 0.30,
    'fg_recognition': 0.25,
    'substituent_completeness': 0.15,
}

# Confidence bands for structured logging
CONFIDENCE_HIGH: float = 0.75    # DEBUG level
CONFIDENCE_MEDIUM: float = 0.45  # INFO level
# Below MEDIUM: WARNING level


# ---------------------------------------------------------------------------
# Factor computation helpers
# ---------------------------------------------------------------------------

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
    from ..rules.seniority import get_prefix
    for fg_name, matches in fg_dict.items():
        if fg_name == principal:
            continue  # Already counted
        prefix = get_prefix(fg_name)
        if prefix:  # Has a known prefix form -> will be named
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

    # Heuristic: count locant-prefixed groups in the name
    locant_groups = re.findall(r'\d+[,-]', name)
    named_subs = min(len(locant_groups), total_expected)

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
    # If the candidate name matches a retained name for a recognised
    # scaffold, set atom_coverage to 1.0. This handles nucleobases,
    # benzene derivatives, etc. without molecule-specific hacks.
    try:
        from ..data.retained_names import RETAINED_NAMES
        name_lower = name.lower()
        for _smi, retained in RETAINED_NAMES.items():
            if retained.lower() == name_lower:
                atom_cov = 1.0
                break
    except ImportError:
        pass  # Graceful degradation if retained_names unavailable

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
