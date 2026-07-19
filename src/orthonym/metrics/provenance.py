"""v25 G3: per-call naming provenance (tier derivation inputs).

Observation-only contextvars — the default naming path's BEHAVIOR is
untouched; sites merely record where the shipped name came from.
"""
from __future__ import annotations

import contextvars
from typing import Optional

_SOURCE = contextvars.ContextVar("orthonym_prov_source", default=None)
_OPSIN = contextvars.ContextVar("orthonym_prov_opsin", default=None)
# v25 G3: top-level general_fallback flag propagated into fragment /
# component recursion (name_compound inherits it).
general_fallback_ctx = contextvars.ContextVar(
    "orthonym_general_fallback", default=False)


def clear_provenance() -> None:
    _SOURCE.set(None)
    _OPSIN.set(None)


def record_source(source: str, opsin: Optional[str] = None) -> None:
    _SOURCE.set(source)
    if opsin is not None:
        _OPSIN.set(opsin)


def get_provenance() -> dict:
    return {"source": _SOURCE.get(), "opsin": _OPSIN.get()}
