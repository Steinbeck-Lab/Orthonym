"""Compiled-SMARTS cache (Lever J, 2026-09-12).

81 rule functions compiled their SMARTS with ``Chem.MolFromSmarts`` on every call: 32,313
compiles of 105 distinct patterns per 300 molecules. A query mol is immutable in matching, so
one compiled object per pattern string is exact. ``compiled(pattern)`` returns the same object
``Chem.MolFromSmarts(pattern)`` would (or None for an invalid pattern), cached process-wide.
"""
from __future__ import annotations

import functools

from rdkit import Chem


@functools.lru_cache(maxsize=4096)
def compiled(pattern: str):
    """``Chem.MolFromSmarts(pattern)``, compiled once per distinct pattern string."""
    return Chem.MolFromSmarts(pattern)
