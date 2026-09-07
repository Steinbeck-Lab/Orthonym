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
from .acid_halide import name_acid_halide  # commit 02-11
from .amide import name_amide  # commit 03-02
from .amine import name_amine  # commit 03-03
from .anhydride import name_anhydride  # commit 02-12
from .boronic_acid import name_boronic_acid  # commit 02-10
from .carbamate import name_carbamate  # commit 02-07
from .carbamic_acid import name_carbamic_acid  # commit 02-06
from .cyanamide import name_cyanamide  # AM-1
from .guanidine import name_guanidine  # commit 02-09
from .hydrazone import name_hydrazone  # commit 02-02
from .imine import name_imine  # Wave2 T2a
from .ion_dispatch import name_ion_dispatch  # commit 02-26
from .isocyanate import name_isocyanate  # commit 02-04
from .isothiocyanate import name_isothiocyanate  # commit 02-05
from .lactam import name_lactam  # commit 02-14
from .lactone import name_lactone  # commit 02-13
from .n_oxide import name_n_oxide  # commit 02-03
from .organometallic import name_organometallic  # commit 02-04 (Phase 161)

# Per-handler imports appended per atomic commit 02-01..02-29 (Plan-02)
# + 03-01..03-09 (Plan-03). See 160-AUDIT-DECOMP.md § 3 for topological
# extraction order. Substrate commit 02-00 ships ZERO handler imports.
from .oxime import name_oxime  # commit 02-01
from .partial_sat import name_partial_sat  # commit 02-24
from .phosphate_ester import name_phosphate_ester  # commit 02-19
from .phosphine import name_phosphine  # commit 02-20
from .phosphine_oxide import name_phosphine_oxide  # commit 02-18
from .phosphinic_acid import name_phosphinic_acid  # commit 02-21
from .polycyclic import name_polycyclic  # commit 02-23
from .ring_assembly import name_ring_assembly  # commit 02-22
from .ring_ester import name_ring_ester  # commit 03-04
from .ring_nitrile import name_ring_nitrile  # commit 03-01
from .simple_molecule import name_simple_molecule  # commit 02-25
from .sulfone import name_sulfone  # commit 02-16
from .sulfoxide import name_sulfoxide  # commit 02-15 (renumbered; polyfunctional deferred)
from .thioether import name_thioether  # commit 02-17
from .thiourea import name_thiourea  # R3 (P-66.1.6.1.3)
from .urea import name_urea  # commit 02-08

__all__ = [
    "NameTreeNode",
    "NamingResult",
    "name_tree_to_string",
    "dispatch_inner",
    "INNER_DISPATCH_TABLE",
    "InnerDispatchEntry",
    # Handler exports appended per atomic commit:
    "name_oxime",  # 02-01
    "name_hydrazone",  # 02-02
    "name_n_oxide",  # 02-03
    "name_isocyanate",  # 02-04
    "name_isothiocyanate",  # 02-05
    "name_carbamic_acid",  # 02-06
    "name_carbamate",  # 02-07
    "name_urea",  # 02-08
    "name_thiourea",  # R3 (P-66.1.6.1.3)
    "name_guanidine",  # 02-09
    "name_cyanamide",  # AM-1
    "name_boronic_acid",  # 02-10
    "name_acid_halide",  # 02-11
    "name_anhydride",  # 02-12
    "name_lactone",  # 02-13
    "name_lactam",  # 02-14
    "name_sulfoxide",  # 02-15 (renumbered)
    "name_sulfone",  # 02-16
    "name_thioether",  # 02-17
    "name_phosphine_oxide",  # 02-18
    "name_phosphate_ester",  # 02-19
    "name_phosphine",  # 02-20
    "name_phosphinic_acid",  # 02-21
    "name_ring_assembly",  # 02-22
    "name_polycyclic",  # 02-23
    "name_partial_sat",  # 02-24
    "name_simple_molecule",  # 02-25
    "name_ion_dispatch",  # 02-26
    "name_ring_nitrile",  # 03-01
    "name_amide",  # 03-02
    "name_amine",  # 03-03
    "name_imine",  # Wave2 T2a
    "name_ring_ester",  # 03-04
    "name_organometallic",  # 02-04 (Phase 161)
    # ... (26 total at Plan-02 end after ester-family deferrals; 39 at Plan-03 end)
]
