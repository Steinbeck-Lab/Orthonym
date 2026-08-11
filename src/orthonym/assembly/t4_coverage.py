"""T4 universal coverage-by-construction namer (best-effort tier ONLY).

Reached only on the general-fallback best-effort recovery path -- master
switch ``self._general_fallback`` in ``namer.py`` (its early ``return None``
when that flag is False; ``general_fallback_unverified`` is a secondary,
narrower flag on the same path) -- which itself only runs AFTER the
PIN/systematic path has already abstained. So this module can never change a
PIN emission; it only ever fills in where the PIN tiers were silent.

Produces an atom-complete, E1-certified name or ``None`` (clean abstain,
never a partial name). ``validation.e1_certificate.verify_certificate`` is
the atom-coverage certificate; SELF-01 downstream (OPSIN round-trip) is the
second half of the 0-wrong net, not a substitute for E1 here.

This module is the Task 2 SKELETON only: the control flow + the locked
``_Candidate`` interface. Task 3 fills in ``_best_effort_candidate`` with the
real parent+substituents producer; Tasks 4-6 add the per-class cascade and
wire this namer into ``namer.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..validation.e1_certificate import verify_certificate

if TYPE_CHECKING:  # type-only; no runtime dependency on general_engine
    from .general_engine import GeneralEngineResult


@dataclass(frozen=True)
class _Candidate:
    """Locked Task-2 interface: a best-effort name plus its E1 proof object.

    ``result_obj`` is a ``general_engine.GeneralEngineResult`` (carries the
    atom->token ``bindings`` E1 verifies) when the candidate can be
    E1-certified, or ``None`` when there is nothing for E1 to check --
    ``name_t4_complete`` then ships the name unchecked.
    """
    name: str
    result_obj: "Optional[GeneralEngineResult]"


def name_t4_complete(mol, features) -> Optional[str]:
    """Best-effort T4 name for ``mol``, or ``None`` (clean abstain).

    Control flow (locked for Tasks 3-6):
      1. Ask ``_best_effort_candidate`` for a name + its E1 proof object.
      2. No candidate at all -> abstain (``None``).
      3. A candidate with no proof object (``result_obj is None``) ships
         unchecked -- E1 has nothing to verify.
      4. Otherwise the candidate must pass ``verify_certificate`` or it is
         discarded -- never patched, never shipped anyway.
    """
    candidate = _best_effort_candidate(mol, features)
    if candidate is None:
        return None
    if candidate.result_obj is None:
        return candidate.name
    verdict = verify_certificate(mol, candidate.result_obj)
    if not verdict.ok:
        return None
    return candidate.name


def _best_effort_candidate(mol, features) -> Optional[_Candidate]:
    """The producer cascade. Stub for Task 2 -- returns ``None`` (clean
    abstain) unconditionally. Task 3 fills this in with the parent +
    complete-recursive-substituents producer; Task 5 adds the per-class
    cascade (lactam-unsat, fused, spiro, peptide, sugar, acyclic,
    OPSIN-format) on top of it.
    """
    return None
