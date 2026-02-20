"""Fragment naming infrastructure with recursion depth guard.

Provides thread-safe recursion depth tracking to prevent infinite loops
when fragment naming calls name_compound() recursively. Uses
threading.local() pattern established by _n_oxide_guard in composer.py.

Max depth = 10 levels:
  Level 0: Initial molecule (normal naming)
  Level 1: Decomposition fragment (e.g., ester acid/alkyl fragment)
  Level 2: Sub-fragment from recursive decomposition
  Level 3: Substituent naming within a fragment
  Level 4: Deep nesting (phospholipids, complex multi-fragment)
  Level 5: Very deep nesting (triglycerides, iterative decomposition)
  Level 6: Maximum practical depth (multi-bond iterative)
  Level 7-9: Extended nesting for medium-size molecules
  Level 10: STOP -- return None as graceful fallback
  Total-call cap: 100 calls per top-level naming operation

Usage:
    from orthonym.assembly.fragment_naming import name_fragment_recursively

    # Inside a naming function that needs to recursively name a sub-fragment:
    fragment_name = name_fragment_recursively("CCO")
    if fragment_name is None:
        # Depth limit reached or naming failed -- use fallback
        ...
"""

import functools
import logging
import threading as _threading
from typing import Optional

from rdkit import Chem

logger = logging.getLogger(__name__)

_fragment_guard = _threading.local()

MAX_NAMING_DEPTH = 10
MAX_TOTAL_CALLS = 100


def get_naming_depth() -> int:
    """Get current recursion depth for fragment naming.

    Returns:
        Current depth (0 = top-level, not inside any recursive call).
    """
    return getattr(_fragment_guard, 'depth', 0)


@functools.lru_cache(maxsize=1024)
def _name_compound_cached(canonical_smiles: str) -> Optional[str]:
    """Name a compound from canonical SMILES with result caching.

    Only called from name_fragment_recursively() with pre-canonicalized SMILES.
    Cache hit avoids redundant name_compound() calls for repeated fragments.

    The cache is module-level (per-process), thread-safe under CPython GIL,
    and requires no invalidation since naming is deterministic for a given
    canonical SMILES.
    """
    from ..namer import name_compound
    result = name_compound(canonical_smiles)
    return result if result else None


def name_fragment_recursively(smiles: str, max_depth: int = MAX_NAMING_DEPTH) -> Optional[str]:
    """Name a molecular fragment by calling name_compound() with depth guard.

    Increments the thread-local depth counter, calls name_compound(),
    and restores the counter on exit (even on exception). If the depth
    limit or total-call cap is reached, returns None immediately instead
    of recursing.

    Args:
        smiles: SMILES string of the fragment to name.
        max_depth: Maximum recursion depth (default 10).

    Returns:
        IUPAC name if successful and within limits, None otherwise.

    Notes:
        - Results are cached by canonical SMILES via lru_cache
        - Total-call cap (100) prevents exponential blowup
        - Thread-safe via threading.local() and GIL

    Example:
        >>> name_fragment_recursively("CCO")
        'ethanol'
        >>> name_fragment_recursively("CC(=O)O")
        'acetic acid'
    """
    depth = get_naming_depth()
    total = getattr(_fragment_guard, 'total_calls', 0)

    # Reset total counter at top level (new molecule)
    if depth == 0:
        _fragment_guard.total_calls = 0
        total = 0

    if depth >= max_depth:
        logger.warning(
            "DROP-13 substituent_skip: reason=depth_limit_reached depth=%d max=%d smiles=%s",
            depth, max_depth, smiles[:60],
        )
        return None  # Graceful fallback at depth limit

    if total >= MAX_TOTAL_CALLS:
        logger.warning(
            "DROP-13 substituent_skip: reason=total_call_limit_reached calls=%d max=%d smiles=%s",
            total, MAX_TOTAL_CALLS, smiles[:60],
        )
        return None  # Graceful fallback at total-call cap

    _fragment_guard.depth = depth + 1
    _fragment_guard.total_calls = total + 1
    try:
        canonical = Chem.CanonSmiles(smiles)
        if canonical is None:
            return None
        return _name_compound_cached(canonical)
    except Exception as e:
        logger.warning(
            "DROP-14 substituent_skip: reason=fragment_naming_exception depth=%d smiles=%s error=%s",
            depth, smiles[:60], e,
        )
        return None
    finally:
        _fragment_guard.depth = depth  # Restore depth (stack)
        # total_calls NOT restored -- accumulates across tree
