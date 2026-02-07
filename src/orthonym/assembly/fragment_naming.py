"""Fragment naming infrastructure with recursion depth guard.

Provides thread-safe recursion depth tracking to prevent infinite loops
when fragment naming calls name_compound() recursively. Uses
threading.local() pattern established by _n_oxide_guard in composer.py.

Max depth = 3 levels:
  Level 0: Initial molecule (normal naming)
  Level 1: Fragment naming (e.g., peptide residue, ester acid fragment)
  Level 2: Sub-fragment (e.g., named substituent on a fragment)
  Level 3: STOP -- return None as graceful fallback

Usage:
    from orthonym.assembly.fragment_naming import name_fragment_recursively

    # Inside a naming function that needs to recursively name a sub-fragment:
    fragment_name = name_fragment_recursively("CCO")
    if fragment_name is None:
        # Depth limit reached or naming failed -- use fallback
        ...
"""

import threading as _threading
from typing import Optional

_fragment_guard = _threading.local()

MAX_NAMING_DEPTH = 3


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

    Args:
        smiles: SMILES string of the fragment to name.
        max_depth: Maximum recursion depth (default 3).

    Returns:
        IUPAC name if successful and within depth limit, None otherwise.

    Example:
        >>> name_fragment_recursively("CCO")
        'ethanol'
        >>> name_fragment_recursively("CC(=O)O")
        'acetic acid'
    """
    depth = get_naming_depth()
    if depth >= max_depth:
        return None  # Graceful fallback at depth limit

    _fragment_guard.depth = depth + 1
    try:
        from ..namer import name_compound
        result = name_compound(smiles)
        return result if result else None
    except Exception:
        return None
    finally:
        _fragment_guard.depth = depth  # Restore previous depth
