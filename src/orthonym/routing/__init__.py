"""Orthonym Routing Module.

Class-first dispatcher routing input mol -> handler before naming pipeline
(CFR-01..04).

Public exports per CONTEXT:
- ``ClassFirstRouter`` — per-instance class-first dispatcher.
- ``ClassDispatchResult`` — frozen dataclass returned by ``dispatch()``.
- ``StoutClass`` — StrEnum of dispatch class identifiers.
- ``ClassDispatchEntry`` — frozen dataclass per ``DISPATCH_TABLE`` row.
- ``DISPATCH_TABLE`` — ``OrderedDict[StoutClass, ClassDispatchEntry]``;
  module-level frozen post-import.
- ``dispatch`` / ``get_dispatch_stats`` / ``reset_dispatch_stats`` —
  module-level convenience wrappers.

Internal helpers (``_register_dispatch``, ``_is_*``, ``_handle_*``,
``_invoke_audit_log``) are PRIVATE and intentionally not exported per
CONTEXT line 89 (no public plugin API; + extracts one if needed).
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
