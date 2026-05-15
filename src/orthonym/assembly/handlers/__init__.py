"""Orthonym Handler Package (Phase 160).

Per-class IUPAC-name handlers lifted from
``composer.py:_assemble_name_impl`` into per-file modules under this
package (DECOMP-01, DECOMP-05). Each handler exposes ONE public
``name_<handler_id>(features, mol=None, style='pin') -> Optional[NamingResult]``
callable registered in ``src/orthonym/assembly/inner_dispatch.py`` per
CONTEXT D-03 + D-05.

Substrate commit 02-00 ships the package SKELETON: this ``__init__.py``,
the private ``_enrichment.py`` + ``_handler_shared.py`` modules, and the
re-exports of NameTreeNode + NamingResult + name_tree_to_string. No
handler files yet — those land in commits 02-01..02-29 (Plan-02 Tier-1 +
Tier-1.5) + 03-01..03-09 (Plan-03 Tier-2 + Tier-3).

Public exports per CONTEXT D-04 mirror (Phase 158 ``routing/__init__.py``
shape):

- ``name_<handler_id>`` callables (28 handlers + ``general_acyclic`` +
  ``ion_dispatch`` + ``simple_molecule``; appended per atomic commit).
- ``NameTreeNode`` re-export from ``assembly/name_tree.py`` (convenience).
- ``NamingResult`` re-export from ``assembly/name_tree.py`` (convenience).
- ``name_tree_to_string`` re-export from ``assembly/name_tree_to_string.py``.
- ``dispatch_inner`` + ``INNER_DISPATCH_TABLE`` re-export from
  ``assembly/inner_dispatch.py`` (convenience for tests + Plan-04 CLI).

Internal helpers (``_enrichment.py``, ``_handler_shared.py``) are PRIVATE
and intentionally not exported per CONTEXT D-03 + Phase 158 D-04 mirror
(no public plugin API; v20+ extracts one if needed).

Anti-pattern hygiene:
- AP-160-14 banned: silent ImportError fallback in handlers package
  (no ``try: from orthonym.assembly.handlers import ...; except: pass``).
- AP-160-06 banned: invent-as-you-go handler_id outside HANDLER_POLICIES
  (only ion_dispatch + simple_molecule + general_acyclic are Phase 160
  additions per CONTEXT line 191-192).

References:
- 160-AUDIT-DECOMP.md § 1 + § 3 — handler enumeration + dependency graph.
- 160-CONTEXT.md D-03 — one file per HANDLER_POLICIES handler_id.
- 160-PATTERNS.md § 4 — analog: routing/__init__.py:1-43.
"""
from ..inner_dispatch import (
    INNER_DISPATCH_TABLE,
    InnerDispatchEntry,
    dispatch_inner,
)
from ..name_tree import NameTreeNode, NamingResult
from ..name_tree_to_string import name_tree_to_string

# Per-handler imports appended per atomic commit 02-01..02-29 (Plan-02)
# + 03-01..03-09 (Plan-03). See 160-AUDIT-DECOMP.md § 3 for topological
# extraction order. Substrate commit 02-00 ships ZERO handler imports.
from .oxime import name_oxime  # commit 02-01

__all__ = [
    "NameTreeNode",
    "NamingResult",
    "name_tree_to_string",
    "dispatch_inner",
    "INNER_DISPATCH_TABLE",
    "InnerDispatchEntry",
    # Handler exports appended per atomic commit:
    "name_oxime",  # 02-01
    # "name_hydrazone",      # 02-02
    # "name_n_oxide",        # 02-03
    # ... (29 total at Plan-02 end; 39 at Plan-03 end)
]
