"""Phase 162 ML Fallback Gate package.

Encapsulates the opt-in STOUT 2.0.5 fallback for compounds where the
rule-based pipeline produces a degraded name. The full architectural
contract is locked in :doc:`162-AUDIT-MLF` (Plan-01 audit-as-spec).

Components:

* :mod:`orthonym.ml_fallback.quality_gate` — the
  :func:`is_name_quality_inadequate` predicate (CONTEXT D-04 + D-12)
* :mod:`orthonym.ml_fallback.invoker` — :class:`MLFallbackInvoker`
  lazy-load + SHA-pin (CONTEXT D-09 + D-10 + D-11), shipped in
  Plan-02 T02

Requirements (MLF-01..06):

* MLF-01: default-OFF (``Orthonym(allow_ml_fallback=False)`` baseline)
* MLF-02: opt-in (``allow_ml_fallback=True`` enables)
* MLF-03: INFO-level structured logging
* MLF-04: dual-config measurement (--ml-fallback + benchmark report)
* MLF-05: SHA-256 reproducibility pin
* MLF-06: byte-identical preservation on rule-based-success canary
"""

from __future__ import annotations

from orthonym.data.ml_model_pin import (
    STOUT_MODEL_FILES_MANIFEST,
    STOUT_MODEL_SHA256,
    STOUT_MODEL_VERSION,
    MLModelPinViolation,
    pin_violation_error,
)
from .quality_gate import (
    _DESCRIPTIVE_FALLBACK_NAMES,
    _GARBLED_TOKENS,
    is_name_quality_inadequate,
)

# Plan-02 T02 ships the invoker; guard the import to allow Plan-02 T01
# atomic commit to land before T02 (the package still imports cleanly even
# if the invoker module is not yet on disk).
try:
    from .invoker import MLFallbackInvoker, MLFallbackResult
except ImportError:  # pragma: no cover — guard for inter-commit ordering
    MLFallbackInvoker = None  # type: ignore[assignment]
    MLFallbackResult = None  # type: ignore[assignment]


__all__ = [
    # Re-exports from data.ml_model_pin (Plan-01)
    "STOUT_MODEL_VERSION",
    "STOUT_MODEL_FILES_MANIFEST",
    "STOUT_MODEL_SHA256",
    "MLModelPinViolation",
    "pin_violation_error",
    # quality_gate exports (Plan-02 T01)
    "is_name_quality_inadequate",
    "_GARBLED_TOKENS",
    "_DESCRIPTIVE_FALLBACK_NAMES",
    # invoker exports (Plan-02 T02; may be None at T01 commit time)
    "MLFallbackInvoker",
    "MLFallbackResult",
]
