"""Fragment naming infrastructure with recursion depth guard.

Provides thread-safe recursion depth tracking to prevent infinite loops
when fragment naming calls name_compound() recursively. Uses
threading.local() pattern established by _n_oxide_guard in composer.py.

Max depth = 7 levels:
  Level 0: Initial molecule (normal naming)
  Level 1: Decomposition fragment (e.g., ester acid/alkyl fragment)
  Level 2: Sub-fragment from recursive decomposition
  Level 3: Substituent naming within a fragment
  Level 4: Deep nesting (phospholipids, complex multi-fragment)
  Level 5: Very deep nesting (triglycerides, iterative decomposition)
  Level 6: Maximum practical depth (multi-bond iterative)
  Level 7: STOP -- return None as graceful fallback

Usage:
    from orthonym.assembly.fragment_naming import name_fragment_recursively

    # Inside a naming function that needs to recursively name a sub-fragment:
    fragment_name = name_fragment_recursively("CCO")
    if fragment_name is None:
        # Depth limit reached or naming failed -- use fallback
        ...
"""

import logging
import threading as _threading
from typing import Dict, Optional

from rdkit import Chem

logger = logging.getLogger(__name__)

_fragment_guard = _threading.local()

MAX_NAMING_DEPTH = 7
MAX_TOTAL_CALLS = 100


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
# Checked BEFORE the depth counter so these fragments are always nameable,
# regardless of recursion depth.  Analogous to retained_names.py but for
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
    """Initialize runtime fragment cache for a naming call.

    The runtime cache stores (canonical SMILES -> name) pairs discovered
    during a single top-level naming call.  This eliminates redundant
    re-naming of the same fragment at different recursion depths.

    Only the outermost call (depth == 0) should start a session.
    Nested calls inherit the parent's cache.
    """
    if getattr(_fragment_guard, 'depth', 0) == 0:
        _fragment_guard.cache = {}


def end_naming_session():
    """Clear runtime fragment cache after a naming call completes.

    Only the outermost call (depth == 0) should end the session to
    avoid clearing a parent session's cache during nested calls.
    """
    if getattr(_fragment_guard, 'depth', 0) == 0:
        _fragment_guard.cache = None


def get_naming_depth() -> int:
    """Get current recursion depth for fragment naming.

    Returns:
        Current depth (0 = top-level, not inside any recursive call).
    """
    return getattr(_fragment_guard, 'depth', 0)


def name_fragment_recursively(smiles: str, max_depth: int = MAX_NAMING_DEPTH) -> Optional[str]:
    """Name a molecular fragment by calling name_compound() with depth guard.

    Increments the thread-local depth counter, calls name_compound(),
    and restores the counter on exit (even on exception). If the depth
    limit is reached, returns None immediately instead of recursing.

    SMILES is canonicalized before calling name_compound() to ensure
    consistent input regardless of the original SMILES notation.

    Args:
        smiles: SMILES string of the fragment to name.
        max_depth: Maximum recursion depth (default 7).

    Returns:
        IUPAC name if successful and within limits, None otherwise.

    Notes:
        - Input SMILES is canonicalized via RDKit CanonSmiles
        - Thread-safe via threading.local() and GIL

    Example:
        >>> name_fragment_recursively("CCO")
        'ethanol'
        >>> name_fragment_recursively("CC(=O)O")
        'acetic acid'
    """
    # Canonicalize early so cache lookup uses consistent keys
    try:
        canonical = Chem.CanonSmiles(smiles)
    except Exception:
        return None
    if canonical is None:
        return None

    # Tier 1: static fragment cache — depth-independent
    cached = FRAGMENT_NAME_CACHE.get(canonical)
    if cached is not None:
        return cached

    # Tier 2: runtime dynamic cache — populated during this naming session
    runtime_cache = getattr(_fragment_guard, 'cache', None)
    if runtime_cache is not None:
        dynamic = runtime_cache.get(canonical)
        if dynamic is not None:
            return dynamic

    depth = get_naming_depth()
    if depth >= max_depth:
        logger.warning(
            "DROP-13 substituent_skip: reason=depth_limit_reached depth=%d max=%d smiles=%s",
            depth, max_depth, smiles[:60],
        )
        return None  # Graceful fallback at depth limit

    _fragment_guard.depth = depth + 1
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
            "DROP-14 substituent_skip: reason=fragment_naming_exception depth=%d smiles=%s error=%s",
            depth, smiles[:60], e,
        )
        return None
    finally:
        _fragment_guard.depth = depth  # Restore previous depth
