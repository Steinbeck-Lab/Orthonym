"""Fragment naming infrastructure with cycle-detection guard.

Provides thread-safe cycle detection to prevent infinite loops when
fragment naming calls name_compound() recursively. Uses a visited-SMILES
set (threading.local() pattern) instead of an arbitrary depth counter.

The visited set tracks which SMILES are currently being named up the call
stack. If a SMILES is encountered that's already being processed, cycle
detection returns None to break the recursion. A safety-net maximum
visited set size (20) prevents unbounded recursion from decomposition
chains where every fragment SMILES is unique.

Usage:
    from orthonym.assembly.fragment_naming import name_fragment_recursively

    # Inside a naming function that needs to recursively name a sub-fragment:
    fragment_name = name_fragment_recursively("CCO")
    if fragment_name is None:
        # Cycle detected or naming failed -- use fallback
        ...
"""

import logging
import threading as _threading
from typing import Dict, Optional

from rdkit import Chem

logger = logging.getLogger(__name__)

_fragment_guard = _threading.local()

# Legacy constant kept for backward compatibility — no longer enforced
# as the primary guard. Cycle detection via visited set is the primary
# mechanism. A safety-net size limit (_MAX_VISITED_SIZE) replaces this.
MAX_NAMING_DEPTH = 7
MAX_TOTAL_CALLS = 100

# Safety-net maximum: even without exact cycle, limit recursion depth
# to prevent unbounded decomposition chains where every fragment SMILES
# is different. Generous limit (20 vs old limit of 7) to allow deep
# but finite naming chains.
_MAX_VISITED_SIZE = 20


def _get_visited() -> set:
    """Get the current visited-SMILES set (thread-safe).

    Returns an empty set if no naming session is active.
    """
    visited = getattr(_fragment_guard, 'visited', None)
    if visited is None:
        visited = set()
        _fragment_guard.visited = visited
    return visited


# Pre-computed names for common fragments that frequently hit the depth limit.
# Checked BEFORE the cycle guard so these fragments are always nameable,
# regardless of recursion state. Analogous to retained_names.py but for
# fragments produced during recursive decomposition of complex molecules.
#
# Every entry was verified against name_compound() at depth 0 (2026-02-25).
# Only fragments with CORRECT verified names are included.
FRAGMENT_NAME_CACHE: Dict[str, str] = {
    # --- Simple alkanes ---
    "CC": "ethane",
    "CCC": "propane",
    "CCCC": "butane",
    "CCCCC": "pentane",
    "CCCCCC": "hexane",
    "CCCCCCC": "heptane",
    "CCCCCCCC": "octane",
    "CCCCCCCCC": "nonane",
    "CCCCCCCCCC": "decane",
    "CCCCCCCCCCC": "undecane",
    "CCCCCCCCCCCC": "dodecane",
    # --- Simple alcohols ---
    "CO": "methanol",
    "CCO": "ethanol",
    "CCCO": "propan-1-ol",
    "CCCCO": "butan-1-ol",
    "CCCCCO": "pentan-1-ol",
    "CC(C)O": "propan-2-ol",
    "CC(C)(C)O": "2-methylpropan-2-ol",
    # --- Simple carboxylic acids ---
    "O=CO": "formic acid",
    "CC(=O)O": "acetic acid",
    "CCC(=O)O": "propanoic acid",
    "CCCC(=O)O": "butanoic acid",
    "CCCCC(=O)O": "pentanoic acid",
    "O=C(O)c1ccccc1": "benzoic acid",
    # --- Simple aldehydes ---
    "C=O": "formaldehyde",
    "CC=O": "acetaldehyde",
    "CCC=O": "propanal",
    "CCCC=O": "butanal",
    # --- Simple ketones ---
    "CC(C)=O": "acetone",
    "CCC(C)=O": "butan-2-one",
    # --- Simple amines ---
    "CN": "methylamine",
    "CCN": "ethylamine",
    "CCCN": "propan-1-amine",
    "CCCCN": "butan-1-amine",
    "NCCCCCN": "pentane-1,5-diamine",
    # --- Common amides / nitriles ---
    "CC(N)=O": "acetamide",
    "CC#N": "acetonitrile",
    # --- Common aromatics ---
    "c1ccccc1": "benzene",
    "Oc1ccccc1": "phenol",
    "Nc1ccccc1": "aniline",
    "O=Cc1ccccc1": "benzaldehyde",
    "CC(=O)c1ccccc1": "acetophenone",
    "c1ccc(-c2ccccc2)cc1": "1,1'-biphenyl",
    "c1ccc2ccccc2c1": "naphthalene",
    # --- Common heterocycles ---
    "c1ccncc1": "pyridine",
    "c1ccoc1": "furan",
    "c1cc[nH]c1": "pyrrole",
    "c1ccsc1": "thiophene",
    # --- Amino acids (retained names, common in peptide fragments) ---
    "NCC(=O)O": "glycine",
    "CC(N)C(=O)O": "alanine",
    "NC(CO)C(=O)O": "serine",
    "NC(CS)C(=O)O": "cysteine",
    "NC(Cc1ccc(O)cc1)C(=O)O": "tyrosine",
    "NC(Cc1c[nH]cn1)C(=O)O": "histidine",
    "NC(Cc1c[nH]c2ccccc12)C(=O)O": "tryptophan",
    "NCCCCC(N)C(=O)O": "lysine",
    "NC(CCC(=O)O)C(=O)O": "glutamic acid",
    "NC(CC(=O)O)C(=O)O": "aspartic acid",
    "NC(=O)CC(N)C(=O)O": "asparagine",
    "CSCCC(N)C(=O)O": "methionine",
    # --- Other common fragments ---
    "CCCCC(CC)CO": "2-ethylhexan-1-ol",
    "ClCCCl": "1,2-dichloroethane",
    "ClC(Cl)Cl": "chloroform",
    # --- Fatty acids (common in phospholipids/sphingolipids) ---
    "CCCCCC(=O)O": "hexanoic acid",
    "CCCCCCC(=O)O": "heptanoic acid",
    "CCCCCCCC(=O)O": "octanoic acid",
    "CCCCCCCCC(=O)O": "nonanoic acid",
    "CCCCCCCCCC(=O)O": "decanoic acid",
    "CCCCCCCCCCCC(=O)O": "dodecanoic acid",
    "CCCCCCCCCCCCCC(=O)O": "tetradecanoic acid",
    "CCCCCCCCCCCCCCCC(=O)O": "hexadecanoic acid",
    "CCCCCCCCCCCCCCCCCC(=O)O": "octadecanoic acid",
    "CCCCCCCCCCCCCCCCCCCC(=O)O": "icosanoic acid",
    # --- Common biological fragments ---
    "OCC(O)CO": "glycerol",
    "OCCO": "ethylene glycol",
    "NCCO": "2-aminoethanol",
    "O=P(O)(O)O": "phosphoric acid",
}


def start_naming_session():
    """Initialize runtime fragment cache and visited set for a naming call.

    The runtime cache stores (canonical SMILES -> name) pairs discovered
    during a single top-level naming call. The visited set tracks which
    SMILES are currently being named to detect cycles.

    Only the outermost call (visited set empty) should start a session.
    Nested calls inherit the parent's cache and visited set.
    """
    visited = getattr(_fragment_guard, 'visited', None)
    if visited is None or len(visited) == 0:
        _fragment_guard.cache = {}
        _fragment_guard.visited = set()


def end_naming_session():
    """Clear runtime fragment cache and visited set after naming completes.

    Only the outermost call (visited set empty) should end the session to
    avoid clearing a parent session's state during nested calls.
    """
    visited = getattr(_fragment_guard, 'visited', None)
    if visited is None or len(visited) == 0:
        _fragment_guard.cache = None
        _fragment_guard.visited = set()


def get_naming_depth() -> int:
    """Get current recursion depth proxy for fragment naming.

    Returns the size of the visited set, which represents how many
    fragments are currently being named up the call stack.

    Returns:
        Number of fragments currently being named (0 = top-level).
    """
    visited = getattr(_fragment_guard, 'visited', None)
    return len(visited) if visited else 0


def is_top_level_naming() -> bool:
    """Check if we're at the top level (not inside any recursive naming).

    Returns:
        True if no fragments are currently being named.
    """
    return get_naming_depth() == 0


def name_fragment_recursively(smiles: str, max_depth: int = MAX_NAMING_DEPTH) -> Optional[str]:
    """Name a molecular fragment with cycle-detection guard.

    Uses a visited-SMILES set to detect and break circular recursion
    instead of an arbitrary depth counter. Before naming a fragment,
    checks if its canonical SMILES is already being processed up the
    call stack. If yes (cycle detected), returns a cached name or None.

    A safety-net maximum visited set size (_MAX_VISITED_SIZE=20) prevents
    unbounded recursion from decomposition chains where every fragment
    SMILES is unique (different capping produces different SMILES).

    SMILES is canonicalized before processing to ensure consistent keys.

    Args:
        smiles: SMILES string of the fragment to name.
        max_depth: Legacy parameter, kept for API compatibility. Not enforced.

    Returns:
        IUPAC name if successful, None if cycle detected or naming fails.
    """
    # Canonicalize early so cache lookup uses consistent keys
    try:
        canonical = Chem.CanonSmiles(smiles)
    except Exception:
        return None
    if canonical is None:
        return None

    # Tier 1: static fragment cache — always available, cycle-independent
    cached = FRAGMENT_NAME_CACHE.get(canonical)
    if cached is not None:
        return cached

    # Tier 2: runtime dynamic cache — populated during this naming session
    runtime_cache = getattr(_fragment_guard, 'cache', None)
    if runtime_cache is not None:
        dynamic = runtime_cache.get(canonical)
        if dynamic is not None:
            return dynamic

    # Cycle detection: if this SMILES is already being named up the call
    # stack, we have circular recursion. Return None (caller will fallback).
    visited = _get_visited()
    if canonical in visited:
        logger.info(
            "CYCLE detected: smiles=%s already in visited set (size=%d)",
            smiles[:60], len(visited),
        )
        return None

    # Safety net: even without exact cycle, limit the recursion depth
    # to prevent unbounded decomposition chains where every fragment
    # SMILES is different (different capping) but naming never terminates.
    if len(visited) >= _MAX_VISITED_SIZE:
        logger.warning(
            "DEPTH safety net: visited set size=%d >= %d, smiles=%s",
            len(visited), _MAX_VISITED_SIZE, smiles[:60],
        )
        return None

    # Mark as in-progress, name it, then unmark
    visited.add(canonical)
    try:
        from ..namer import name_compound
        result = name_compound(canonical)
        if result:
            # Populate runtime cache with successful result
            if runtime_cache is not None:
                runtime_cache[canonical] = result
            return result
        return None
    except Exception as e:
        logger.debug(
            "Fragment naming exception: smiles=%s error=%s",
            smiles[:60], e,
        )
        return None
    finally:
        visited.discard(canonical)
