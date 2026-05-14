"""Orthonym Routing Module.

Class-first dispatcher routing input mol -> handler before naming pipeline
(Phase 158, CFR-01..04).

Public exports per CONTEXT D-04:
- ``ClassFirstRouter`` — per-instance class-first dispatcher (D-03).
- ``ClassDispatchResult`` — frozen dataclass returned by ``dispatch()`` (D-03).
- ``StoutClass`` — StrEnum of dispatch class identifiers (D-11).
- ``ClassDispatchEntry`` — frozen dataclass per ``DISPATCH_TABLE`` row (D-05).
- ``DISPATCH_TABLE`` — ``OrderedDict[StoutClass, ClassDispatchEntry]``;
  module-level frozen post-import (D-05).
- ``dispatch`` / ``get_dispatch_stats`` / ``reset_dispatch_stats`` —
  module-level convenience wrappers (D-04).

Internal helpers (``_register_dispatch``, ``_is_*``, ``_handle_*``,
``_invoke_audit_log``) are PRIVATE and intentionally not exported per
CONTEXT line 89 (no public plugin API; v20+ extracts one if needed).
"""

from .dispatcher import (
    ClassFirstRouter,
    ClassDispatchResult,
    dispatch,
    get_dispatch_stats,
    reset_dispatch_stats,
)
from .dispatch_table import (
    StoutClass,
    ClassDispatchEntry,
    DISPATCH_TABLE,
)

__all__ = [
    "ClassFirstRouter",
    "ClassDispatchResult",
    "StoutClass",
    "ClassDispatchEntry",
    "DISPATCH_TABLE",
    "dispatch",
    "get_dispatch_stats",
    "reset_dispatch_stats",
]
