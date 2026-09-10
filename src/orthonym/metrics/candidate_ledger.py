"""Append-only candidate ledger (, the ``audit`` instrument's recorder).

Every oracle this project owns answers *"does the EMITTED name denote the right
molecule?"* — round-trip,, E1, ``bb_conformance``. None answers **"was a
correct name ever BUILT, and if so what threw it away?"**, which is the question
that defeated PB task 5 and: three correct fixes moved a dev split by ~0
because hand-picked target lists kept landing off the mass.

This module is that missing recorder. It is deliberately modelled on
``metrics/abstention.py`` — thread-local, side-effect-only, every recording call
wrapped so telemetry can never raise into naming — with three deliberate
differences, each forced by a measurement (`internal notes`
):

* **Append-only, not first-writer-wins.** ``abstention.py`` keeps one code per
  session because it answers "which site declined". The selection question needs
  *every* candidate, including the ones that lost.

* **Meaningful for SUCCESSES too.** ``abstention_code_for`` returns None unless the
  result is a failure sentinel. A ledger that did that could never see the
  interesting case — a row that emits a *wrong* name while a correct candidate was
  built and discarded.

* **``scope`` is a recorded field.** Measured: the pipeline builds fragment names
  (``N,N-diethylethanamine``) that are correct names *for a fragment* and can never
  round-trip to the whole input. Round-tripping them against the molecule would
  file every one under "wrong" and manufacture a large fake producer-correctness
  class, so consumers filter on ``scope == 'molecule'``.

**OFF BY DEFAULT.** ``enable`` must be called explicitly, so the production
naming path carries nothing but a single ``if not _enabled`` test. The naming
output must be byte-identical with the ledger on and off; that contract is what
makes this an instrument rather than a behaviour change, and it is asserted in
``tests/unit/metrics/test_candidate_ledger.py``.
"""

import threading
from typing import Any, Dict, List, NamedTuple, Optional

__all__ = [
    "LedgerEntry",
    "Scope",
    "Stage",
    "clear_ledger",
    "disable",
    "enable",
    "is_enabled",
    "read_ledger",
    "record_candidate",
]


class Stage:
    """What happened to the candidate. Not an Enum: these are written into JSON
    artifacts and compared as plain strings by the analysis scripts."""

    #: A producer built this name string.
    PRODUCED = "produced"
    #: A pool/confidence gate declined it (``CandidatePool.add`` returned None).
    GATE_REJECTED = "gate_rejected"
    #: A correctness gate suppressed it (, OPSIN validity, coverage downgrade).
    SUPPRESSED = "suppressed"
    #: A later producer replaced it without any gate firing.
    SUPERSEDED = "superseded"
    #: The name the caller actually returned. Recorded EXPLICITLY rather than
    #: inferred by looking the emitted string up in the ledger, because measured:
    #: the pool held '(2R)-piperidine-2-carboxylic acid' and naming emitted
    #: '(2R)-piperidine-2-carboxylate'. A lookup would mis-report every anion row.
    EMITTED = "emitted"


class Scope:
    """Whether the name names the WHOLE input or one fragment of it."""

    #: A candidate for the whole input molecule. Only these may be round-tripped
    #: against the input's InChIKey.
    MOLECULE = "molecule"
    #: A substituent/fragment name. Correct fragment names never round-trip to the
    #: whole molecule, so comparing them against the input is a category error.
    FRAGMENT = "fragment"


class LedgerEntry(NamedTuple):
    site: str
    stage: str
    scope: str
    name: Optional[str]
    detail: Optional[str]
    #: Fragment-recursion depth at record time (0 = top-level molecule).
    depth: int = 0


_slot = threading.local()


def resolve_scope() -> tuple:
    """``(scope, depth)`` for the naming currently in progress.

    Every hook calls this rather than testing the depth itself, so the four hook
    sites cannot drift apart on the one question that decides whether a name is
    comparable to the input molecule.

    ⚠ **``scope`` is a PRIOR, not ground truth, and the classifier must treat it
    that way.** Depth is the only signal available at record time and it is not a
    perfect proxy for "names a fragment": measured on ``O=C([O-])[C@H]1CCCCN1``,
    the *whole-molecule* candidate ``(2R)-piperidine-2-carboxylic acid`` is built
    inside a NESTED component naming (the anion is named by naming its neutral
    acid), so depth reports 1 and this function returns FRAGMENT for a name that
    denotes the entire input.

    The consumer therefore resolves the question by evidence: a recorded name that
    round-trips to the input IS a molecule candidate whatever its scope, while
    ``scope == FRAGMENT`` is used only to keep genuine fragment names out of the
    "our producers build wrong molecules" count. That way an imperfect prior
    cannot corrupt either direction.
    """
    try:
        from ..assembly.fragment_naming import get_naming_depth
        depth = get_naming_depth()
        return (Scope.MOLECULE if depth == 0 else Scope.FRAGMENT), depth
    except Exception:  # pragma: no cover - telemetry must never break naming
        return Scope.MOLECULE, 0


def enable() -> None:
    """Start recording on this thread, and clear anything already recorded."""
    _slot.enabled = True
    _slot.entries = []


def disable() -> None:
    """Stop recording on this thread and drop the entries."""
    _slot.enabled = False
    _slot.entries = []


def is_enabled() -> bool:
    return bool(getattr(_slot, "enabled", False))


def clear_ledger() -> None:
    """Drop recorded entries but stay enabled. Called between molecules by a
    batch consumer that names more than one input per process."""
    _slot.entries = []


def record_candidate(
    site: str,
    stage: str,
    name: Optional[str],
    scope: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    """Append one event. No-op unless ``enable`` was called on this thread.

    ``scope=None`` (the default) resolves scope and depth from the current naming
    depth via:func:`resolve_scope`; pass an explicit scope only to override that,
    as the substituent-cascade hook does (it knows it produced a fragment name
    regardless of the depth it was called at).

    Never raises: a telemetry failure must not change a name. The bare ``except``
    mirrors ``abstention.record_abstention`` for the same reason.
    """
    try:
        if not getattr(_slot, "enabled", False):
            return
        resolved, depth = resolve_scope()
        entries = getattr(_slot, "entries", None)
        if entries is None:
            entries = _slot.entries = []
        entries.append(LedgerEntry(
            site, stage, scope if scope is not None else resolved,
            name, detail, depth,
        ))
    except Exception:  # pragma: no cover - telemetry must never break naming
        pass


def read_ledger() -> List[LedgerEntry]:
    """The entries recorded on this thread, in order. Empty when disabled."""
    return list(getattr(_slot, "entries", ()) or ())


def as_dicts() -> List[Dict[str, Any]]:
    """``read_ledger`` in JSON-serialisable form, for artifact writers."""
    return [e._asdict() for e in read_ledger()]
