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


def _run_general_e1(mol, features) -> Optional[_Candidate]:
    """One bounded engine attempt: run ``name_general`` at the T4-permissive
    setting and E1-audit the result. Returns a complete ``_Candidate`` or
    ``None`` -- NEVER a partial, NEVER a raise.

    ``allow_aromatic_general=True`` opens the lone-monocycle / mancude-cage
    paths and lifts the charge/mancude refusals; ``allow_suffix_free=True`` is
    the best-effort terminal-ring assembly. Chosen independent of the calling
    instance's flags -- T4 is best-effort, so it always uses the most permissive
    engine. ``name_general`` is fail-closed (returns None, never a wrong name),
    but the call is wrapped so any engine error is a clean abstain.

    ``verify_certificate`` here is the SAME E1 gate ``name_t4_complete`` re-runs;
    matching its default (``allow_charged=False``) keeps our accept/reject
    decision identical to the backstop's, so a candidate we approve is never
    voided one call later. The coverage check means we return ``None`` ourselves
    rather than rely on the backstop, so no partial is ever handed up.
    """
    from .general_engine import name_general

    try:
        result = name_general(
            mol, features,
            allow_aromatic_general=True,
            allow_suffix_free=True)
    except Exception as exc:  # fail-closed: an engine error is a clean abstain
        logger.info("t4 _run_general_e1: name_general raised: %s", exc)
        return None

    if result is None:
        return None
    if not verify_certificate(mol, result).ok:
        return None
    return _Candidate(name=result.name, result_obj=result)


def _clone_features_with(features, **overrides):
    """Return a shallow COPY of ``features`` with the given attributes set, or
    ``None`` if it is not clonable.

    The caller owns ``features`` and reuses it (namer.py's T4 dispatch), so it
    is never mutated in place -- exactly the ``_copy.copy(features)`` + reassign
    pattern ``general_engine._name_terminal_ring_assembly`` uses to suppress a
    principal group. Reassigning ``principal_group_atoms`` to a fresh ``[]``
    (never mutating the shared list) keeps the original object intact.
    """
    import copy as _copy

    try:
        clone = _copy.copy(features)
        for attr, value in overrides.items():
            setattr(clone, attr, value)
        return clone
    except (AttributeError, TypeError) as exc:
        logger.info("t4 _clone_features_with: features not clonable (%s)",
                    type(exc).__name__)
        return None


def _best_effort_candidate(mol, features) -> Optional[_Candidate]:
    """Senior parent + every other atom as a COMPLETE recursive substituent,
    with a bounded route-around-declines cascade (Task 5).

    Rung 0 (Task 3, happy path). The general engine's own ring/chain producers
    ALREADY select the senior parent (``select_principal_ring_system`` /
    ``principal_chain``), name EVERY off-parent fragment through the C4 recursive
    ``name_substituent`` (e.g. ``-CH2N(CH3)2`` -> ``(dimethylamino)methyl``),
    and assemble a full atom->token ``bindings`` partition. Run at the
    T4-permissive setting and E1-audited, they name the CLASS-A target
    ``CN(C)C[C@H]1CCCC[C@H]1O`` as
    ``(1R,2R)-2-((dimethylamino)methyl)cyclohexan-1-ol``. So the lever is NOT a
    from-scratch re-implementation of ``_assemble`` -- it is to run the SAME
    engine and audit the bindings afterward (the Blue-Book-reference
    "re-anchor the bindings and audit afterward" pattern).

    The cascade (Task 5). When rung 0 DECLINES -- ``name_general`` returns
    ``None`` or its result is not atom-complete -- the molecule is NOT
    terminated. The dominant decline (measured, ``diag_t5``) is the
    UNSUPPORTED-SUFFIX class: with ``pg='ester'`` and a short acyl chain
    selected as the parent, ``name_general_chain._partition`` refuses
    "unsupported suffix for pg='ester'". The route-around is the same move
    ``_name_terminal_ring_assembly`` already makes for von-Baeyer cages, applied
    universally through the engine so it also reaches MONOCYCLES: SUPPRESS the
    principal group (``principal_group=None``) so every functional group becomes
    a detachable PREFIX -- an ester's acyl-oxy becomes ``acetyloxy`` -- and no
    unsupported suffix is required. Two ordered rungs, each a SINGLE bounded
    ``name_general`` call, each independently E1-gated:

      * rung 1 -- suppress PG AND demote ``chain_is_parent`` so the RING becomes
        the parent and the acyl-oxy a prefix. Converts the ester-on-ring class,
        e.g. ``CC(=O)O[C@H]1C(=C)C=C(C=C1OC)OC`` ->
        ``(6S)-6-(acetyloxy)-1,3-dimethoxy-5-methylidenecyclohexa-1,3-diene``
        (OPSIN-round-tripping, stereo included).
      * rung 2 -- suppress PG only, keeping the perceived parent, for ring-less /
        chain-preferred esters where a suffix-free chain name is the only
        complete form available.

    Each rung is complete-or-nothing: E1 proves every heavy atom is bound before
    a candidate is handed up, and SELF-01 (OPSIN round-trip) downstream is the
    second half of the 0-wrong net. When a decline needs a capability the
    codebase lacks -- e.g. the dichlorophosphoryl-carbamate substituent of
    ``C1CCC(CC1)OC(=O)NP(=O)(Cl)Cl``, whose recursive substituent namer hits its
    depth cap -- every rung fails E1 and the producer HONESTLY abstains
    (``None``), never a fabricated or partial name. A suffix-free prefix name is
    a valid description of the right structure that is not a well-formed PIN;
    that is licit here (invariant 1: "a table miss must degrade to an uglier
    name, never to a refusal") and confined to this best-effort T4 branch.
    """
    # Rung 0: the full engine with the perceived principal group (Task 3).
    candidate = _run_general_e1(mol, features)
    if candidate is not None:
        return candidate

    # Cascade: route around the decline. Each rung suppresses the principal
    # group so every FG becomes a detachable prefix; ordered most-specific
    # first. A fixed, bounded list -- no unbounded recursion.
    cascade = (
        # rung 1: ring-first -- demote chain_is_parent so a ring parent is
        # chosen and the acyl-oxy is cited as an ...oyloxy/acetyloxy prefix.
        dict(principal_group=None, principal_group_atoms=[],
             chain_is_parent=False),
        # rung 2: keep the perceived parent, PG suppressed (ring-less esters).
        dict(principal_group=None, principal_group_atoms=[]),
    )
    for overrides in cascade:
        alt_features = _clone_features_with(features, **overrides)
        if alt_features is None:
            continue
        candidate = _run_general_e1(mol, alt_features)
        if candidate is not None:
            return candidate

    return None  # every applicable strategy tried -> honest abstain
