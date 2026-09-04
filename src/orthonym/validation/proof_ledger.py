""": the proof ledger -- assert the spine on the FINAL name.

WHY this module exists
----------------------
``e1_certificate.verify_certificate`` is checked on the string the PRODUCER
built. That string is not the string the caller receives. Between the
certificate and the ``return``, ``name()`` runs the universal stereo
backstop, the OPSIN-grammar repair backstop, the real-OPSIN validity gate,
the empty-string normalisation, a last-resort decomposition retry and the
trivial-name fallback -- and, on the late-recovery path, a retained/catalog
substitution that can replace the WHOLE name. Every one of those may hand
back a different string, and nothing re-asserts the producer's proof against
it. A proof that is checked on a string nobody ships is not a proof of what
was shipped.

The ledger closes exactly that gap and nothing else. A producer RECORDS its
``(mol, spine)`` mid-pipeline; the exit of ``name()`` FINALIZES, re-running
``verify_spine`` against the string actually being returned. Where the two
strings differ, the re-anchor is *supposed* to report findings -- that
visibility is the deliverable, not a defect to suppress.

Design
------
Module-level ``contextvars``, the established pattern in this codebase (see
``orthonym/metrics/provenance.py``, mirrored deliberately): the naming
pipeline threads no proof object through its dozens of call sites, and a
ContextVar is per-context state that a recursive/threaded run cannot
cross-contaminate the way a module global would.

AUDIT-ONLY and side-effect-free with respect to names. Nothing in this
module returns, mutates or influences a name string; it only observes.

Fail-safe discipline
--------------------
``finalize`` NEVER raises. A bug in a proof must never turn a successful
naming into a crash or an abstention, so an internal error is converted into
a failed ``SpineProof`` carrying a single ``PROOF_INTERNAL_ERROR`` finding --
which is honest (the proof did not establish anything) without being fatal to
the caller.
"""
from __future__ import annotations

import contextvars
from typing import Any, Dict, NamedTuple, Optional

from orthonym.validation.binding_spine import BindingSpine, Finding, SpineProof, verify_spine

# The ledger's own finding code: the proof machinery itself failed, so
# nothing was established. Distinct from every binding_spine code, all of
# which report something the proof actually observed about the name.
PROOF_INTERNAL_ERROR = "PROOF_INTERNAL_ERROR"


class _Record(NamedTuple):
    """What a producer recorded, kept verbatim until the exit finalizes it."""

    mol: Any
    spine: Any
    stage: str
    name_at_record: str
    allow_charged: bool


_RECORD: contextvars.ContextVar[Optional[_Record]] = contextvars.ContextVar(
    "orthonym_proof_record", default=None)
_PROOF: contextvars.ContextVar[Optional[SpineProof]] = contextvars.ContextVar(
    "orthonym_proof_result", default=None)
_FINAL_NAME: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "orthonym_proof_final_name", default=None)


def clear_ledger() -> None:
    """Reset the ledger. Called once per top-level molecule.

    Without this, a molecule that records no spine would finalize against
    the PREVIOUS molecule's record -- the exact cross-molecule contamination
    class the per-molecule confidence/pool/abstention resets exist to
    prevent.
    """
    _RECORD.set(None)
    _PROOF.set(None)
    _FINAL_NAME.set(None)


def record_spine(mol, spine: BindingSpine, *, stage: str,
                 name_at_record: str, allow_charged: bool = False) -> None:
    """Record the spine a producer built, to be re-asserted at the exit.

    ``name_at_record`` is the producer's own string. It is stored ONLY as
    diagnostic evidence: ``finalize`` verifies against the FINAL name, and
    keeping both is what makes "a post-processor changed the name" legible
    instead of appearing as an unexplained proof failure.

    The last record wins. A later producer on the same molecule is the one
    whose emission is closer to what ships, and any previous verdict is
    invalidated because it was computed for a spine no longer on the ledger.
    """
    _RECORD.set(_Record(mol=mol, spine=spine, stage=stage,
                        name_at_record=name_at_record,
                        allow_charged=bool(allow_charged)))
    _PROOF.set(None)
    _FINAL_NAME.set(None)


def finalize(final_name: str, *, mode: str = "audit") -> Optional[SpineProof]:
    """Re-assert the recorded spine against ``final_name``.

    Returns ``None`` when nothing was recorded -- the honest answer for the
    great majority of molecules in this phase, since only the two
    general-engine sites record. ``None`` means "no proof was attempted",
    never "the proof passed"; callers must not read it as a pass.

    Never raises: see the module docstring.
    """
    record = _RECORD.get()
    if record is None:
        return None
    _FINAL_NAME.set(final_name)
    try:
        proof = verify_spine(record.mol, record.spine, final_name,
                             mode=mode, allow_charged=record.allow_charged)
    except Exception as exc:  # a proof bug must never break naming
        proof = SpineProof(
            ok=False,
            findings=(Finding(PROOF_INTERNAL_ERROR,
                              f"{type(exc).__name__}: {exc}", "error"),),
            stats={"stage": record.stage, "mode": mode},
        )
    _PROOF.set(proof)
    return proof


def get_proof() -> Dict[str, Any]:
    """The ledger's current state, for telemetry and the Task 7 census.

    ``ok is None`` and ``codes == ()`` mean no verdict exists yet (nothing
    recorded, or recorded but not finalized) -- deliberately distinct from
    ``ok is False`` with codes, which is a real failed proof.
    """
    record = _RECORD.get()
    proof = _PROOF.get()
    return {
        "stage": record.stage if record is not None else None,
        "codes": proof.codes() if proof is not None else (),
        "ok": proof.ok if proof is not None else None,
        "name_at_record": (record.name_at_record
                           if record is not None else None),
        "final_name": _FINAL_NAME.get(),
    }
