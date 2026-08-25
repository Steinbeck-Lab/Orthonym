"""v25 G3: per-call naming provenance (tier derivation inputs).

Observation-only contextvars — the default naming path's BEHAVIOR is
untouched; sites merely record where the shipped name came from.
"""
from __future__ import annotations

import contextvars
from typing import Optional

# ---------------------------------------------------------------------------
# v29 Phase 7 Task 1: what the OPSIN validity gate ACTUALLY DID for this name
# ---------------------------------------------------------------------------
# `name_tiered` used to derive `opsin`/`gates_passed` from
#     gate_active = (not self._disable_opsin_validity_gate
#                    and _validity_gate_jar_present())
# i.e. from *jar presence*. That is not evidence that THIS name passed
# anything: `namer._final_opsin_validity_gate` has ten `return name`
# carve-outs that never reach `_self_consistency_decision`, plus a BBR-GATE
# stereo branch that judges only the stereo-STRIPPED parse. Measured at
# `a6cf6157` ( §2): three names
# OPSIN cannot parse at all reported `gates_passed: ['SELF-01']`.
#
# The gate now records its per-name outcome here and the report is derived
# from it. Deny-by-default in three ways: the default is NOT_RUN, the outcome
# is stored WITH the name it was recorded for (an outcome belonging to a
# different string never counts — see `resolve_gate_outcome`), and only the
# two outcomes explicitly listed in `_VERIFIED_GATE_OUTCOMES` may claim any
# verification (see `opsin_label_for_gate_outcome`).

#: no gate call was recorded for this naming session (the default).
GATE_OUTCOME_NOT_RUN = "not_run"
#: an outcome WAS recorded, but for a different string than the one shipped
#: (e.g. `_apply_trivial_fallback` ships a retained name the gate never saw).
GATE_OUTCOME_BYPASSED = "bypassed"
#: `_DISABLE_VALIDITY_GATE` (env) or `_disable_opsin_validity_gate` (instance).
GATE_OUTCOME_DISABLED = "gate_disabled"
#: jar absent / transient OPSIN failure — the gate failed OPEN.
GATE_OUTCOME_UNAVAILABLE = "unavailable"
#: the name handed to the gate was already a descriptive-fallback sentinel,
#: so the gate skipped it (nothing was verified).
GATE_OUTCOME_DESCRIPTIVE_FALLBACK = "descriptive_fallback"
#: the gate suppressed a candidate to the descriptive fallback.
GATE_OUTCOME_SUPPRESSED = "suppressed"
#: OPSIN parsed the FULL name and SELF-01 returned verdict "ok". The ONE
#: state that may claim a bare SELF-01.
GATE_OUTCOME_SELF01 = "self01_verified"
#: the BBR-GATE stereo carve-out: SELF-01 judged the stereo-STRIPPED parse, so
#: the CONSTITUTION is verified and the stereo layer is NOT.
GATE_OUTCOME_SELF01_CONSTITUTION_ONLY = "self01_verified_constitution_only"
#: SELF-01 ran and could not compare (fail-OPEN) — nothing was proven.
GATE_OUTCOME_SELF01_INCONCLUSIVE = "self01_inconclusive"
#: SELF-01 was a no-op (`_SC_MODE == "off"`, or no input SMILES to compare to).
GATE_OUTCOME_SELF01_SKIPPED = "self01_skipped"
#: SELF-01 PROVED a different molecule but `_SC_MODE == "warn"` shipped it.
GATE_OUTCOME_SELF01_WARN_MISMATCH = "self01_warn_mismatch"
#: prefix for the ten by-design `return name` carve-outs: `carveout:<slug>`.
GATE_OUTCOME_CARVEOUT_PREFIX = "carveout:"

#: The ONLY outcomes that may report anything other than "unverified".
#: An allowlist, never a blocklist — an outcome string this code has never
#: seen (a future carve-out, a path someone forgot to instrument) falls
#: through to "unverified" rather than inheriting a claim it did not earn.
_VERIFIED_GATE_OUTCOMES = {
    GATE_OUTCOME_SELF01: "verified",
    GATE_OUTCOME_SELF01_CONSTITUTION_ONLY: "verified_constitution_only",
}

#: `gates_passed` token per verified outcome. `SELF-01(constitution)` is
#: deliberately a DIFFERENT token from `SELF-01`: the stereo carve-out checked
#: the constitution only, and folding it into either bucket would lose that.
_GATE_TOKENS = {
    GATE_OUTCOME_SELF01: "SELF-01",
    GATE_OUTCOME_SELF01_CONSTITUTION_ONLY: "SELF-01(constitution)",
}

_SOURCE = contextvars.ContextVar("orthonym_prov_source", default=None)
_OPSIN = contextvars.ContextVar("orthonym_prov_opsin", default=None)
# v27 Phase 6 T6.4: honest flag for a best-effort emission that ships a
# CONSTITUTION-ONLY name (defined stereo the engine could not express was
# omitted). The name string stays a clean IUPAC name; this metadata is the
# only place the omission is surfaced. Never set for pin/valid/complete (those
# tiers abstain on dropped stereo — P-91.2.1).
_STEREO_UNEXPRESSED = contextvars.ContextVar(
    "orthonym_prov_stereo_unexpressed", default=False)
# v30: this emission cites the principal characteristic group as a detachable
# PREFIX with NO suffix (``…-5-oxo-…-4-oxabicyclo[6.4.0]dodeca-…`` for a ketone).
#
# That is ILL-FORMED, not merely non-preferred. `BlueBookV2.md:25009`, P-41
# "Seniority order of classes": *"If characteristic groups other than those
# given in Table 5.1 are present, one (and only one) kind must be cited as
# suffix (the principal characteristic group) for classes other than radicals"*.
# So the suffix is REQUIRED and these names omit it; OPSIN tolerates them and
# they denote the right structure, but they are not valid IUPAC.
#
# Shipped deliberately on T4 ONLY, where the alternative is silence (the contributor guide
# invariant 1: for T4 an abstention is a DEFECT and a table miss must degrade to
# an uglier name). Recorded per row rather than merely counted, because a number
# in a report is not recoverable and a field is: this is precisely the spelling
# blind spot the BB-conformance audit sized at 516 rows, and the reason it went
# unnoticed is that nothing marked the rows. v31 must be able to ENUMERATE them.
_SUFFIX_FREE_PREFIX_NAME = contextvars.ContextVar(
    "orthonym_prov_suffix_free_prefix_name", default=False)
# v25 G3: top-level general_fallback flag propagated into fragment /
# component recursion (name_compound inherits it).
general_fallback_ctx = contextvars.ContextVar(
    "orthonym_general_fallback", default=False)
# v30: the BEST-EFFORT discriminator (`general_fallback_unverified`), published
# for code that must choose a VOCABULARY by tier but is called from handlers that
# do not know the tier.
#
# `composer._integrate_universal_prefixes` is the case that forced this. It names
# every substituent for every enriching handler, and it is called from
# `acid_halides`, `anhydrides`, `esters`, `lactones` and `polyfunctional` -- all
# PIN-path handlers, none of which receives a tier. Threading a parameter through
# all of them would mean touching every handler signature to move one boolean, so
# the flag is published once here and read where it is needed, exactly as
# `general_fallback_ctx` already does for fragment recursion.
#
# ⚠ This is deliberately NOT `allow_aromatic_general`: that flag is True for
# `complete` as well as `best-effort`, so using it would change `complete` output
# and break hard bound H3 (PIN default byte-identical) at the tier above PIN.
best_effort_ctx = contextvars.ContextVar(
    "orthonym_best_effort", default=False)
# v37 SP1.1b: the ``allow_aromatic_general`` opt-in propagated into fragment /
# component recursion, exactly as ``general_fallback_ctx`` / ``best_effort_ctx``
# above. ``name_compound`` (the recursive re-entry point, namer.py) builds a FRESH
# namer, so a top-level ``allow_aromatic_general=True`` was silently lost on
# recursion (the free function had no such parameter). Unlike ``best_effort_ctx``,
# this DOES publish the raw ``allow_aromatic_general`` value -- that is safe here
# because it is read back ONLY by the recursive re-entry (to give a recursively
# named fragment the SAME tier its top-level call ran at), never used as the
# best-effort discriminator the ⚠ note above warns against. At the PIN default
# tier the value is False, so PIN/complete output is byte-identical.
allow_aromatic_general_ctx = contextvars.ContextVar(
    "orthonym_allow_aromatic_general", default=False)
# v30 P3-T1c: a composer (``pin_path``) name whose ring substituent prefix could
# only be produced by the GENERAL tier -- see ``record_general_ring_prefix``.
_GENERAL_RING_PREFIX = contextvars.ContextVar(
    "orthonym_prov_general_ring_prefix", default=False)
# v29 P7 T1: the validity gate's per-name outcome, and the name string it was
# recorded FOR. Defaults fail closed (NOT_RUN / no name).
_GATE_OUTCOME = contextvars.ContextVar(
    "orthonym_prov_gate_outcome", default=GATE_OUTCOME_NOT_RUN)
_GATE_OUTCOME_NAME = contextvars.ContextVar(
    "orthonym_prov_gate_outcome_name", default=None)


def clear_provenance() -> None:
    _SOURCE.set(None)
    _OPSIN.set(None)
    _STEREO_UNEXPRESSED.set(False)
    # v29 P7 T1: a gate outcome left over from the PREVIOUS molecule would
    # mislabel this one, so it resets with the rest of the provenance.
    _GATE_OUTCOME.set(GATE_OUTCOME_NOT_RUN)
    _GATE_OUTCOME_NAME.set(None)
    # v30 P3-T1c: same reason -- a flag left from the previous molecule would
    # demote this one's tier for a prefix it does not contain.
    _GENERAL_RING_PREFIX.set(False)
    # v30: same reason again -- a suffix-free flag left from the previous
    # molecule would tag this one's name as ill-formed when it is not.
    _SUFFIX_FREE_PREFIX_NAME.set(False)


def restore_provenance(snapshot: dict) -> None:
    """Re-set every provenance ContextVar from a ``get_provenance()`` dict.

    The inverse of :func:`get_provenance`. A caller that speculatively runs a
    producer which records provenance (``record_source`` et al.) and then
    REJECTS its candidate must restore the pre-attempt provenance, otherwise the
    kept emission carries the rejected producer's label. Used by the v33
    Engine-3 NP→von-Baeyer downgrade, whose ``_try_general_engine_recovery``
    probe stamps ``source="general_engine"`` before the RT gate can decline it.
    """
    _SOURCE.set(snapshot.get("source"))
    _OPSIN.set(snapshot.get("opsin"))
    _STEREO_UNEXPRESSED.set(bool(snapshot.get("stereo_unexpressed")))
    _GATE_OUTCOME.set(snapshot.get("gate_outcome", GATE_OUTCOME_NOT_RUN))
    _GATE_OUTCOME_NAME.set(snapshot.get("gate_outcome_name"))
    _GENERAL_RING_PREFIX.set(bool(snapshot.get("general_ring_prefix")))
    _SUFFIX_FREE_PREFIX_NAME.set(bool(snapshot.get("suffix_free_prefix_name")))


def record_source(source: str, opsin: Optional[str] = None) -> None:
    _SOURCE.set(source)
    if opsin is not None:
        _OPSIN.set(opsin)


def record_stereo_unexpressed(flag: bool) -> None:
    """v27 P6 T6.4: mark the current emission as constitution-only (stereo
    defined on the input but not expressed in the name). Set at the flagged
    best-effort ship site; read by ``name_tiered``."""
    _STEREO_UNEXPRESSED.set(bool(flag))


def record_suffix_free_prefix_name(flag: bool) -> None:
    """v30: mark the current emission as citing the principal characteristic
    group as a PREFIX with no suffix -- ill-formed per P-41 (see the ContextVar
    comment). Set only at the T4 PG-suppressed assembly site; read by
    ``name_tiered`` and surfaced per row so v31 can enumerate the debt."""
    _SUFFIX_FREE_PREFIX_NAME.set(bool(flag))


def record_gate_outcome(outcome: str, name: Optional[str]) -> None:
    """v29 P7 T1: record what `_final_opsin_validity_gate` DID, and for WHICH
    string. Called at every return of the gate (and of
    `_self_consistency_decision`, which owns the gate's SELF-01 exits).

    ``name`` must be the string the gate is about to RETURN, not the one it
    was handed — on a suppression those differ, and the outcome belongs to
    what actually ships.
    """
    _GATE_OUTCOME.set(outcome)
    _GATE_OUTCOME_NAME.set(name)


def carveout_outcome(slug: str) -> str:
    """``carveout:<slug>`` — a by-design `return name` carve-out. The name is
    shipped deliberately despite an OPSIN coverage gap; it is NOT verified."""
    return GATE_OUTCOME_CARVEOUT_PREFIX + slug


def resolve_gate_outcome(outcome: Optional[str], outcome_name: Optional[str],
                         shipped_name: Optional[str]) -> str:
    """The outcome that applies to ``shipped_name`` — deny-by-default.

    A recorded outcome only counts for the exact string it was recorded for.
    Paths that ship a name the gate never saw (`_apply_trivial_fallback`
    returns a retained name only AFTER the gate has suppressed the systematic
    candidate, so the recorded outcome then belongs to a DIFFERENT string) get
    ``bypassed``, never the previous name's verdict.
    """
    if not outcome or outcome == GATE_OUTCOME_NOT_RUN:
        return GATE_OUTCOME_NOT_RUN
    if outcome_name is None or outcome_name != shipped_name:
        return GATE_OUTCOME_BYPASSED
    return outcome


def opsin_label_for_gate_outcome(outcome: Optional[str]) -> str:
    """``verified`` / ``verified_constitution_only`` / ``unverified``.

    Allowlist, so it fails closed: any outcome not explicitly listed —
    including a state that does not exist yet — reports ``unverified``.
    """
    return _VERIFIED_GATE_OUTCOMES.get(outcome, "unverified")


def gate_token_for_gate_outcome(outcome: Optional[str]) -> Optional[str]:
    """The `gates_passed` token this outcome earns, or None if it earns none."""
    return _GATE_TOKENS.get(outcome)


def record_general_ring_prefix() -> None:
    """v30 P3-T1c: mark that a PIN-path (``composer``) name contains a ring
    substituent prefix that only the GENERAL tier could produce.

    Tier and ``is_pin`` are decided from ``source`` alone (``namer.py:2728``),
    and every composer emission reports ``source='pin_path'`` -> ``T1``,
    ``is_pin=True``. A systematic replacement / von Baeyer substituent form is a
    VALID name but not the PREFERRED one, so shipping it under that label would
    assert PIN status for a non-PIN name -- measured on
    ``OC(=O)CC12CC3CC(O)(CC(C3)C1)C2``, whose ring PIN is the retained name
    *adamantane*, not ``tricyclo[3.3.1.1^3,7]decane``.

    This flag lets ``name_tiered`` demote exactly those emissions to the general
    tier while leaving every other composer emission untouched. It is
    deliberately one-way (never cleared) for the duration of a naming call.
    """
    _GENERAL_RING_PREFIX.set(True)


def get_provenance() -> dict:
    return {
        "source": _SOURCE.get(),
        "opsin": _OPSIN.get(),
        "stereo_unexpressed": _STEREO_UNEXPRESSED.get(),
        "gate_outcome": _GATE_OUTCOME.get(),
        "gate_outcome_name": _GATE_OUTCOME_NAME.get(),
        "general_ring_prefix": _GENERAL_RING_PREFIX.get(),
        "suffix_free_prefix_name": _SUFFIX_FREE_PREFIX_NAME.get(),
    }
