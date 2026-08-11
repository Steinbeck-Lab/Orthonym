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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..validation.e1_certificate import verify_certificate

if TYPE_CHECKING:  # type-only; no runtime dependency on general_engine
    from .general_engine import GeneralEngineResult

logger = logging.getLogger(__name__)


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
    """Senior parent + every other atom as a COMPLETE recursive substituent.

    Task 3 (happy path). Diagnosed on the CLASS-A target
    ``CN(C)C[C@H]1CCCC[C@H]1O``: the general engine's own ring/chain producers
    ALREADY do exactly what this task requires -- select the senior parent
    (``select_principal_ring_system`` / ``principal_chain``), name EVERY
    off-parent fragment through the C4 recursive ``name_substituent`` (e.g.
    ``-CH2N(CH3)2`` -> ``(dimethylamino)methyl``), and assemble the name with a
    full atom->token ``bindings`` partition. Run under the T4-permissive flags
    (``allow_aromatic_general=True`` opens the lone-monocycle / mancude-cage
    paths; ``allow_suffix_free=True`` is the best-effort terminal-ring
    assembly), they produced the complete, correct, E1-passing name
    ``(1R,2R)-2-((dimethylamino)methyl)cyclohexan-1-ol``.

    The molecule abstains in production only because the recovery lane runs the
    engine with ``allow_aromatic_general=self._allow_aromatic_general`` (off by
    default), gating off ``name_general_monocycle``. So the lever is NOT a
    from-scratch re-implementation of ``_assemble`` (that would duplicate the
    engine's parent-selection + ``name_substituent`` + ordering + stereo +
    elision machinery and risk divergence) -- it is to run the SAME engine at
    the most permissive T4 setting and audit the result with E1. This is the
    Blue-Book-reference "re-anchor the bindings and audit afterward" pattern:
    ``name_general`` builds the atom->token bindings, ``verify_certificate``
    proves every heavy atom is covered before we hand the candidate up.

    Bounded to the happy path: if the engine DECLINES (``name_general`` returns
    ``None`` -- e.g. the unsupported-suffix ester ``pg='ester'`` on a short
    acetyl chain) or its result is not atom-complete, return ``None`` (clean
    abstain). Task 5 adds the route-around-declines cascade here; Task 6 routes
    a polyfunctional remainder. We never hand up a partial: E1 is re-run in
    ``name_t4_complete`` as the backstop, but the coverage check below means we
    return ``None`` ourselves rather than rely on it.
    """
    from .general_engine import name_general

    try:
        # T4 best-effort: the most permissive engine setting, independent of
        # the calling instance's flags. allow_aromatic_general=True opens the
        # lone-monocycle and mancude-cage paths and lifts the charge/mancude
        # refusals; allow_suffix_free=True is the best-effort terminal-ring
        # assembly. name_general is fail-closed (returns None, never a wrong
        # name), but wrap defensively so the T4 tier can never raise.
        result = name_general(
            mol, features,
            allow_aromatic_general=True,
            allow_suffix_free=True)
    except Exception as exc:  # fail-closed: an engine error is a clean abstain
        logger.info("t4 _best_effort_candidate: name_general raised: %s", exc)
        return None

    if result is None:
        return None  # genuine engine decline -> Task 5 cascade territory

    # Never hand up a partial. verify_certificate is the SAME E1 gate
    # name_t4_complete re-runs; matching its default (allow_charged=False)
    # keeps our accept/reject decision identical to the backstop's, so a
    # candidate we approve is never voided one call later.
    if not verify_certificate(mol, result).ok:
        return None

    return _Candidate(name=result.name, result_obj=result)
