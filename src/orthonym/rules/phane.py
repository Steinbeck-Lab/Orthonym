"""Cyclic phane parent hydride nomenclature (IUPAC P-26.4).

Detects molecules whose topology fits the IUPAC P-26.4 "cyclic phane"
class (two or more ring systems joined by acyclic chain segments of
length >= 2 atoms; mutually exclusive with ring-assembly,
spiro/fused/bridged-fused, and multiplicative cases).

Public API (Phase 155.A Plan 01 -- Wave 0 SCAFFOLD; logic ships in
Task 3):

* ``is_cyclophane(mol) -> bool``                            -- topology gate
* ``name_cyclophane(mol) -> Optional[str]``                 -- top-level handler
* ``_classify_phane_topology(mol) -> PhaneTopology``        -- sub-class enum
* ``_build_composite_locant(ring_idx, ring_locant, style)`` -- composite-locant emitter
* ``_enumerate_inter_ring_chains(mol, ring_systems)``       -- BFS chain walker
* ``PhaneTopology``                                         -- enum

Source:

* 155-CONTEXT.md D-03 (graph-topology classification: >=2 ring systems,
  >=6 ring nodes, >=1 inter-ring acyclic chain >=2 atoms; mutually
  exclusive with ring-assembly / spiro / fused / bridged).
* 155-CONTEXT.md D-04 (sub-class enum {PARACYCLOPHANE, METACYCLOPHANE,
  ORTHOCYCLOPHANE, GENERIC_CYCLOPHANE}).
* 155-CONTEXT.md D-05 (composite-locant dual rendering: ASCII ``n(a)``
  production form; Unicode superscript documentation form).
* 155-CONTEXT.md D-16 (mutual-exclusion contract enforced at the
  topology gate; Phase 154 D-11 pattern).
* IUPAC 2013 Blue Book P-26.4 "Cyclic Phane Parent Hydrides".
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem


# ---------------------------------------------------------------------------
# Sub-class enum (D-04)
# ---------------------------------------------------------------------------


class PhaneTopology(Enum):
    """Cyclophane sub-class per IUPAC P-26.4.

    Source: 155-CONTEXT.md D-04.
    """

    PARACYCLOPHANE = "paracyclophane"
    METACYCLOPHANE = "metacyclophane"
    ORTHOCYCLOPHANE = "orthocyclophane"
    GENERIC_CYCLOPHANE = "generic"


# ---------------------------------------------------------------------------
# Public API stubs (Phase 155.A Task 3 will replace bodies)
# ---------------------------------------------------------------------------


def is_cyclophane(mol: Chem.Mol) -> bool:
    """Return True iff ``mol`` matches the IUPAC P-26.4 cyclophane topology.

    Stub: Phase 155.A Task 3 fills the body per 155-CONTEXT.md D-03.
    """
    raise NotImplementedError("Phase 155.A Task 3 fills this")


def name_cyclophane(mol: Chem.Mol) -> Optional[str]:
    """Return the IUPAC PIN cyclophane name for ``mol``, or None.

    Stub: Phase 155.A Task 3 fills the body per 155-CONTEXT.md D-04 + D-05.
    """
    raise NotImplementedError("Phase 155.A Task 3 fills this")


def _classify_phane_topology(mol: Chem.Mol) -> PhaneTopology:
    """Classify cyclophane sub-class per the inter-ring chain attachment positions.

    Stub: Phase 155.A Task 3 fills the body per 155-CONTEXT.md D-04.
    """
    raise NotImplementedError("Phase 155.A Task 3 fills this")


def _build_composite_locant(
    ring_idx: int,
    ring_locant: int,
    style: str = "ascii",
) -> str:
    """Emit composite locant ``n(a)`` (OPSIN ASCII) or ``n^a`` (Blue Book superscript).

    Stub: Phase 155.A Task 3 fills the body per 155-CONTEXT.md D-05.
    """
    raise NotImplementedError("Phase 155.A Task 3 fills this")


def _enumerate_inter_ring_chains(
    mol: Chem.Mol,
    ring_systems: List[Set[int]],
) -> List[List[int]]:
    """Enumerate acyclic chain paths whose endpoints lie in two distinct ring systems.

    Stub: Phase 155.A Task 3 fills the body per 155-CONTEXT.md D-03 + RESEARCH.md
    Code Examples §1.
    """
    raise NotImplementedError("Phase 155.A Task 3 fills this")
