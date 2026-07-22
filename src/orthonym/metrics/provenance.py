"""v25 G3: per-call naming provenance (tier derivation inputs).

Observation-only contextvars — the default naming path's BEHAVIOR is
untouched; sites merely record where the shipped name came from.
"""
from __future__ import annotations

import contextvars
from typing import Optional

_SOURCE = contextvars.ContextVar("orthonym_prov_source", default=None)
_OPSIN = contextvars.ContextVar("orthonym_prov_opsin", default=None)
# v27 Phase 6 T6.4: honest flag for a best-effort emission that ships a
# CONSTITUTION-ONLY name (defined stereo the engine could not express was
# omitted). The name string stays a clean IUPAC name; this metadata is the
# only place the omission is surfaced. Never set for pin/valid/complete (those
# tiers abstain on dropped stereo — P-91.2.1).
_STEREO_UNEXPRESSED = contextvars.ContextVar(
    "orthonym_prov_stereo_unexpressed", default=False)
# v25 G3: top-level general_fallback flag propagated into fragment /
# component recursion (name_compound inherits it).
general_fallback_ctx = contextvars.ContextVar(
    "orthonym_general_fallback", default=False)


def clear_provenance() -> None:
    _SOURCE.set(None)
    _OPSIN.set(None)
    _STEREO_UNEXPRESSED.set(False)


def record_source(source: str, opsin: Optional[str] = None) -> None:
    _SOURCE.set(source)
    if opsin is not None:
        _OPSIN.set(opsin)


def record_stereo_unexpressed(flag: bool) -> None:
    """v27 P6 T6.4: mark the current emission as constitution-only (stereo
    defined on the input but not expressed in the name). Set at the flagged
    best-effort ship site; read by ``name_tiered``."""
    _STEREO_UNEXPRESSED.set(bool(flag))


def get_provenance() -> dict:
    return {
        "source": _SOURCE.get(),
        "opsin": _OPSIN.get(),
        "stereo_unexpressed": _STEREO_UNEXPRESSED.get(),
    }
