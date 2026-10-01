""": per-call naming provenance (tier derivation inputs).

Observation-only contextvars — the default naming path's BEHAVIOR is
untouched; sites merely record where the shipped name came from.
"""
from __future__ import annotations

import contextvars
from typing import Optional

# ---------------------------------------------------------------------------
# a phase Task 1: what the OPSIN validity gate ACTUALLY DID for this name
# ---------------------------------------------------------------------------
# `name_tiered` used to derive `opsin`/`gates_passed` from
# gate_active = (not self._disable_opsin_validity_gate
# and _validity_gate_jar_present)
# i.e. from *jar presence*. That is not evidence that THIS name passed
# anything: `namer._final_opsin_validity_gate` has ten `return name`
# carve-outs that never reach `_self_consistency_decision`, plus an OPSIN-validity stereo carve-out
# branch that judges only the stereo-STRIPPED parse. Measured at
# (internal notes): three names
# OPSIN cannot parse at all reported `gates_passed: ['']`.
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
#: OPSIN parsed the FULL name and returned verdict "ok". The ONE
#: state that may claim a bare.
GATE_OUTCOME_SELF01 = "self_consistency_verified"
#: the OPSIN-validity stereo carve-out: judged the stereo-STRIPPED parse, so
#: the CONSTITUTION is verified and the stereo layer is NOT.
GATE_OUTCOME_SELF01_CONSTITUTION_ONLY = "self_consistency_constitution_only"
#: ran and could not compare (fail-OPEN) — nothing was proven.
GATE_OUTCOME_SELF01_INCONCLUSIVE = "self_consistency_inconclusive"
#: was a no-op (`_SC_MODE == "off"`, or no input SMILES to compare to).
GATE_OUTCOME_SELF01_SKIPPED = "self_consistency_skipped"
#: PROVED a different molecule but `_SC_MODE == "warn"` shipped it.
GATE_OUTCOME_SELF01_WARN_MISMATCH = "self_consistency_warn_mismatch"
#:: the general-fallback stereo-OMISSION reclaim COMPOSED the input's
#: dropped stereo back onto a constitution-verified flat name and re-verified
#: that the composed name recomputes to the input's FULL InChIKey. That check is
#: strictly STRONGER than (it compares the full stereo layer, not just
#: the skeleton), so this is a genuine VERIFIED state — not a by-design carve-out
#: that ships unproven. (It used to be recorded as ``carveout:stereo_omission_reanchor``,
#: which mislabels a full-key-verified name as ``unverified`` — a review -P2 F3.)
GATE_OUTCOME_STEREO_RECOMPOSED = "stereo_omission_full_key_recomposed"
#: prefix for the ten by-design `return name` carve-outs: `carveout:<slug>`.
GATE_OUTCOME_CARVEOUT_PREFIX = "carveout:"
#: claims conformance (2026-09-27): at a general tier, ``name_tiered`` checked the
#: shipped string itself because no verified outcome was recorded for it (an offer
#: the pool picked after the gate had seen another string, or an outcome recorded
#: for a different string): OPSIN read the name and the FULL InChIKey of what it
#: read equals the input's. That is the round trip the paper claims for every
#: shown name, and stronger than, so it claims the plain "verified" label.
GATE_OUTCOME_FULL_KEY_VERIFIED = "full_key_round_trip_verified"

#: The ONLY outcomes that may report anything other than "unverified".
#: An allowlist, never a blocklist — an outcome string this code has never
#: seen (a future carve-out, a path someone forgot to instrument) falls
#: through to "unverified" rather than inheriting a claim it did not earn.
_VERIFIED_GATE_OUTCOMES = {
    GATE_OUTCOME_SELF01: "verified",
    GATE_OUTCOME_SELF01_CONSTITUTION_ONLY: "verified_constitution_only",
    # Full-InChIKey recomposition is a stronger proof than, so it claims
    # the plain "verified" label (full stereo AND constitution match the input).
    GATE_OUTCOME_STEREO_RECOMPOSED: "verified",
    GATE_OUTCOME_FULL_KEY_VERIFIED: "verified",
}

#: `gates_passed` token per verified outcome. `(constitution)` is
#: deliberately a DIFFERENT token from ``: the stereo carve-out checked
#: the constitution only, and folding it into either bucket would lose that.
_GATE_TOKENS = {
    GATE_OUTCOME_SELF01: "self_consistency",
    GATE_OUTCOME_SELF01_CONSTITUTION_ONLY: "self_consistency_constitution_only",
    GATE_OUTCOME_STEREO_RECOMPOSED: "stereo_omission_full_key_recomposed",
    GATE_OUTCOME_FULL_KEY_VERIFIED: "full_key_round_trip",
}

_SOURCE = contextvars.ContextVar("orthonym_prov_source", default=None)
_OPSIN = contextvars.ContextVar("orthonym_prov_opsin", default=None)
# a phase T6.4: honest flag for a best-effort emission that ships a
# CONSTITUTION-ONLY name (defined stereo the engine could not express was
# omitted). The name string stays a clean IUPAC name; this metadata is the
# only place the omission is surfaced. Never set for pin/valid/complete (those
# tiers abstain on dropped stereo —.
_STEREO_UNEXPRESSED = contextvars.ContextVar(
    "orthonym_prov_stereo_unexpressed", default=False)
#: this emission cites the principal characteristic group as a detachable
# PREFIX with NO suffix (``…-5-oxo-…-4-oxabicyclo[6.4.0]dodeca-…`` for a ketone).
#
# That is ILL-FORMED, not merely non-preferred. `the Blue Book`,
# "Seniority order of classes": *"If characteristic groups other than those
# given in Table 5.1 are present, one (and only one) kind must be cited as
# suffix (the principal characteristic group) for classes other than radicals"*.
# So the suffix is REQUIRED and these names omit it; OPSIN tolerates them and
# they denote the right structure, but they are not valid IUPAC.
#
# Shipped deliberately on ONLY, where the alternative is silence (the contributor guide
# a project rule: for an abstention is a DEFECT and a table miss must degrade to
# an uglier name). Recorded per row rather than merely counted, because a number
# in a report is not recoverable and a field is: this is precisely the spelling
# blind spot the BB-conformance audit sized at 516 rows, and the reason it went
# unnoticed is that nothing marked the rows. must be able to ENUMERATE them.
_SUFFIX_FREE_PREFIX_NAME = contextvars.ContextVar(
    "orthonym_prov_suffix_free_prefix_name", default=False)
#: top-level general_fallback flag propagated into fragment /
# component recursion (name_compound inherits it).
general_fallback_ctx = contextvars.ContextVar(
    "orthonym_general_fallback", default=False)
#: the BEST-EFFORT discriminator (`general_fallback_unverified`), published
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
#: the ``allow_aromatic_general`` opt-in propagated into fragment /
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
#: the ``full_coverage`` opt-in (``--emit-tier full-coverage``)
# propagated into fragment / component recursion, exactly as the three
# contextvars above. This is the SINGLE bit that arms the D2 general
# coordination-additive namer (donor-set perception + additive renderer),
# which lives strictly ABOVE best-effort: full-coverage is best-effort's
# production superset PLUS this marker. Default False EVERYWHERE, so with the
# flag off (the default) D2 dispatch is never reached and every tier's output
# is byte-identical -- the load-bearing isolation property SP5.4 tests. Read
# back ONLY by the recursive re-entry (``name_compound``) and D2's own dispatch
# guard; never used as a naming-vocabulary discriminator, so it cannot alter
# pin/valid/complete/best-effort output.
full_coverage_ctx = contextvars.ContextVar(
    "orthonym_full_coverage", default=False)
# The tier of the OUTERMOST naming request: True for a best-effort request
# (``general_fallback_unverified``), False for any other, None outside a request.
# Set by the outermost ``name`` (``namer._budget_scope`` at name-scope depth 1,
# which no nested or isolated naming re-enters), with the value ``name``
# publishes as ``best_effort_ctx``, for the whole call -- the last-resort rescue
# that runs after the body has hit the hang budget included -- and reset on
# return. It differs from ``best_effort_ctx`` on three paths, all measured on the
# large-molecule census (TRIAGE 'Large molecules -- part 3'): the best-effort
# clean fall-through resets ``best_effort_ctx`` to False so that substituent
# recursion takes the PIN vocabulary; the last-resort rescue runs after the
# body has reset it; and the fragment rescue builds a fresh best-effort engine
# inside a request of ANY tier, the PIN tier included. Read only by the von Baeyer
# ring-size ceilings (``vonbaeyer_universal.cage_caps``), which take their
# best-effort values in a best-effort request and nowhere else.
best_effort_request_ctx = contextvars.ContextVar(
    "orthonym_best_effort_request", default=None)
# -T1c: a composer (``pin_path``) name whose ring substituent prefix could
# only be produced by the GENERAL tier -- see ``record_general_ring_prefix``.
_GENERAL_RING_PREFIX = contextvars.ContextVar(
    "orthonym_prov_general_ring_prefix", default=False)
# Name fragments a producer built that are valid but never part of a PIN -- see
# ``record_non_pin_fragment``. A tuple (ordered, duplicate-free); it only grows
# during a naming call and resets with the rest of the provenance.
_NON_PIN_FRAGMENTS = contextvars.ContextVar(
    "orthonym_prov_non_pin_fragments", default=())
# Branch review fixes: the LABEL-ONLY sibling of ``_NON_PIN_FRAGMENTS`` -- name
# fragments whose non-PIN status a structural check found (a polyacid mono-ester, a
# lactone parent beside a senior ring ketone). They lower the label of a name that
# contains them exactly as a non-PIN fragment does, but the PIN tier's promotion
# re-run does not read them when it decides whether one of its names ships at all
# (``name_carries_non_pin_part(..., label_forms=False)``): every re-run name is
# labelled below the PIN anyway, so they would only remove a name the default tier
# shipped before. Same lifetime and roll-back rule as the fragments.
_NON_PIN_LABELS = contextvars.ContextVar(
    "orthonym_prov_non_pin_labels", default=())
# Review a performance pass (F-06): names in PIN form whose preferred status the Blue Book itself
# leaves open -- the book prints a different (PIN) for the same structure. A shipped
# name that contains one is labelled pin_unverified (not systematic_verified: the code
# does not know it is not the PIN). Name-scoped, with the same lifetime and roll-back
# rule as the labels above.
_UNCERTIFIED_PIN_NAMES = contextvars.ContextVar(
    "orthonym_prov_uncertified_pin_names", default=())
# Branch review fixes (2026-09-28): the shipped name was built by the PIN tier's
# promotion re-run (``Orthonym._name_with_pin_promotion``), i.e. with a best-effort
# composition producer admitted under the PIN-vocabulary guard. Such a name is not
# certified as the PIN -- the paper: "The label pin_unverified means a name in PIN
# form that only a breadth producer built" -- so ``name_carries_non_pin_part`` reads
# it and ``name_tiered`` labels the name pin_unverified, is_pin False. Set only by the
# wrapper, after its re-run's name was kept; whole-call, like ``general_ring_prefix``.
_PIN_PROMOTION_RERUN = contextvars.ContextVar(
    "orthonym_prov_pin_promotion_rerun", default=False)
# T1: the validity gate's per-name outcome, and the name string it was
# recorded FOR. Defaults fail closed (NOT_RUN / no name).
_GATE_OUTCOME = contextvars.ContextVar(
    "orthonym_prov_gate_outcome", default=GATE_OUTCOME_NOT_RUN)
_GATE_OUTCOME_NAME = contextvars.ContextVar(
    "orthonym_prov_gate_outcome_name", default=None)

# a performance pass (a lever replay-memo hardening): a scope-bound log of WHICH
# provenance vars a nested naming call TOUCHED (wrote), so the replay-memo can
# reproduce the EXACT provenance state the call left — including a var written to
# a value it already held, or written A->B->A, which a before/after DELTA misses.
# Armed only around a memoised nested call (assembly/nested_memo). A STACK (list
# of sets) so a nested memoised call captures its own touches AND propagates them
# to every enclosing armed call (each setter, and a hit's replay via
# ``note_touched``, adds to every set currently on the stack).
_TOUCHED = contextvars.ContextVar("orthonym_prov_touched", default=None)


def _touch(*names: str) -> None:
    stack = _TOUCHED.get()
    if stack:
        for s in stack:
            s.update(names)


def note_touched(*names: str) -> None:
    """Public: record that ``names`` were written by a path that bypasses the
    setter functions (a replay-memo HIT writes the ContextVars directly). Lets an
    enclosing touched-log still capture them."""
    _touch(*names)


def push_touched_log() -> set:
    """Arm a fresh touched-var log for the current nested call and return it.
    Pair with:func:`pop_touched_log` in a try/finally (LIFO)."""
    stack = _TOUCHED.get()
    if stack is None:
        stack = []
        _TOUCHED.set(stack)
    s: set = set()
    stack.append(s)
    return s


def pop_touched_log() -> set:
    """Disarm the innermost touched-var log and return it (LIFO)."""
    stack = _TOUCHED.get()
    if stack:
        return stack.pop()
    return set()


# ChEBI speed 2: a scope-bound log of the ``record_non_pin_fragment`` CALLS a
# memoised substituent-fragment naming made (every call, also one whose fragment
# was already recorded), so a memo hit can make the same calls again. A read of
# the recorded fragments (``get_provenance``, ``name_carries_non_pin_part``,
# ``record_derived_non_pin_fragment``) appends ``PROVENANCE_READ`` instead: a call
# whose path read provenance is not memoised. A STACK of lists, like _TOUCHED.
_NON_PIN_LOG = contextvars.ContextVar("orthonym_prov_non_pin_log", default=None)
PROVENANCE_READ = object()


def _log_non_pin(item) -> None:
    stack = _NON_PIN_LOG.get()
    if stack:
        for log in stack:
            log.append(item)


def push_non_pin_log() -> list:
    """Arm a fresh record-call log; pair with:func:`pop_non_pin_log` (LIFO)."""
    stack = _NON_PIN_LOG.get()
    if stack is None:
        stack = []
        _NON_PIN_LOG.set(stack)
    log: list = []
    stack.append(log)
    return log


def pop_non_pin_log() -> list:
    """Disarm the innermost record-call log and return it (LIFO)."""
    stack = _NON_PIN_LOG.get()
    if stack:
        return stack.pop()
    return []


def clear_provenance() -> None:
    _touch("source", "opsin", "stereo_unexpressed", "gate_outcome",
           "gate_outcome_name", "general_ring_prefix", "suffix_free_prefix_name",
           "non_pin_fragments", "pin_promotion_rerun", "non_pin_labels",
           "uncertified_pin_names")
    _SOURCE.set(None)
    _OPSIN.set(None)
    _STEREO_UNEXPRESSED.set(False)
    # T1: a gate outcome left over from the PREVIOUS molecule would
    # mislabel this one, so it resets with the rest of the provenance.
    _GATE_OUTCOME.set(GATE_OUTCOME_NOT_RUN)
    _GATE_OUTCOME_NAME.set(None)
    # -T1c: same reason -- a flag left from the previous molecule would
    # demote this one's tier for a prefix it does not contain.
    _GENERAL_RING_PREFIX.set(False)
    #: same reason again -- a suffix-free flag left from the previous
    # molecule would tag this one's name as ill-formed when it is not.
    _SUFFIX_FREE_PREFIX_NAME.set(False)
    _NON_PIN_FRAGMENTS.set(())
    _PIN_PROMOTION_RERUN.set(False)
    _NON_PIN_LABELS.set(())
    _UNCERTIFIED_PIN_NAMES.set(())


def restore_provenance(snapshot: dict) -> None:
    """Re-set every provenance ContextVar from a ``get_provenance`` dict.

    The inverse of:func:`get_provenance`. A caller that speculatively runs a
    producer which records provenance (``record_source`` et al.) and then
    REJECTS its candidate must restore the pre-attempt provenance, otherwise the
    kept emission carries the rejected producer's label. Used by the
    Engine-3 NP→von-Baeyer downgrade, whose ``_try_general_engine_recovery``
    probe stamps ``source="general_engine"`` before the RT gate can decline it.

    The name-scoped non-PIN record (``record_non_pin_fragment``) is the one
    exception: it only grows within a naming call (``clear_provenance`` resets it
    at a call boundary), so a restore keeps the fragments recorded since the
    snapshot. A rejected attempt leaves cache entries behind -- the scope memo
    (``assembly.memo.cache_or_compute``) and the fragment cache return a stored
    string without the record its computation made -- so dropping the record let
    a later cache hit ship a recorded non-PIN spelling as if none had been
    recorded, and the result then depended on ``ORTHONYM_MEMO`` (branch review,
    2026-09-28: 'C[C@@H](CN(C[C@@H]1CCC=CC1)C)O' was named at the PIN tier with the
    memo on and abstained with it off, through the '(R)-(cyclohex-3-en-1-yl)meth'
    record that ``rules.pin_vocabulary.promote_at_pin_tier`` rolled back). A kept
    record only demotes a name that CONTAINS the recorded spelling.
    """
    _SOURCE.set(snapshot.get("source"))
    _OPSIN.set(snapshot.get("opsin"))
    _STEREO_UNEXPRESSED.set(bool(snapshot.get("stereo_unexpressed")))
    _GATE_OUTCOME.set(snapshot.get("gate_outcome", GATE_OUTCOME_NOT_RUN))
    _GATE_OUTCOME_NAME.set(snapshot.get("gate_outcome_name"))
    _GENERAL_RING_PREFIX.set(bool(snapshot.get("general_ring_prefix")))
    _SUFFIX_FREE_PREFIX_NAME.set(bool(snapshot.get("suffix_free_prefix_name")))
    _kept = tuple(snapshot.get("non_pin_fragments") or ())
    _NON_PIN_FRAGMENTS.set(_kept + tuple(
        f for f in _NON_PIN_FRAGMENTS.get() if f not in _kept))
    _PIN_PROMOTION_RERUN.set(bool(snapshot.get("pin_promotion_rerun")))
    _kept_labels = tuple(snapshot.get("non_pin_labels") or ())
    _NON_PIN_LABELS.set(_kept_labels + tuple(
        f for f in _NON_PIN_LABELS.get() if f not in _kept_labels))
    _kept_unc = tuple(snapshot.get("uncertified_pin_names") or ())
    _UNCERTIFIED_PIN_NAMES.set(_kept_unc + tuple(
        f for f in _UNCERTIFIED_PIN_NAMES.get() if f not in _kept_unc))
    _touch("source", "opsin", "stereo_unexpressed", "gate_outcome",
           "gate_outcome_name", "general_ring_prefix", "suffix_free_prefix_name",
           "non_pin_fragments", "pin_promotion_rerun", "non_pin_labels",
           "uncertified_pin_names")


def record_source(source: str, opsin: Optional[str] = None) -> None:
    _SOURCE.set(source)
    _touch("source")
    if opsin is not None:
        _OPSIN.set(opsin)
        _touch("opsin")


def record_stereo_unexpressed(flag: bool) -> None:
    """ T6.4: mark the current emission as constitution-only (stereo
    defined on the input but not expressed in the name). Set at the flagged
    best-effort ship site; read by ``name_tiered``."""
    _STEREO_UNEXPRESSED.set(bool(flag))
    _touch("stereo_unexpressed")


def record_suffix_free_prefix_name(flag: bool) -> None:
    """: mark the current emission as citing the principal characteristic
    group as a PREFIX with no suffix -- ill-formed per (see the ContextVar
    comment). Set only at the PG-suppressed assembly site; read by
    ``name_tiered`` and surfaced per row so can enumerate the debt."""
    _SUFFIX_FREE_PREFIX_NAME.set(bool(flag))
    _touch("suffix_free_prefix_name")


def record_gate_outcome(outcome: str, name: Optional[str]) -> None:
    """ T1: record what `_final_opsin_validity_gate` DID, and for WHICH
    string. Called at every return of the gate (and of
    `_self_consistency_decision`, which owns the gate's exits).

    ``name`` must be the string the gate is about to RETURN, not the one it
    was handed — on a suppression those differ, and the outcome belongs to
    what actually ships.
    """
    _GATE_OUTCOME.set(outcome)
    _GATE_OUTCOME_NAME.set(name)
    _touch("gate_outcome", "gate_outcome_name")


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
    """-T1c: mark that a PIN-path (``composer``) name contains a ring
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
    _touch("general_ring_prefix")


def record_pin_promotion_rerun() -> None:
    """Mark the name about to ship as built by the PIN tier's promotion re-run
    (see ``_PIN_PROMOTION_RERUN``): labelled pin_unverified, never pin_verified."""
    _PIN_PROMOTION_RERUN.set(True)
    _touch("pin_promotion_rerun")


def record_non_pin_fragment(fragment: str) -> None:
    """Record a name fragment that a producer built and that is valid (the whole
    name is still round-trip verified) but is never part of a PIN -- e.g. a
    carbon-substituted N+ cited as 'trimethylazaniumyl', whose PIN form
    '...methanaminiumyl' cannot be verified.

    Unlike ``record_general_ring_prefix`` this is NAME-SCOPED: ``name_tiered``
    demotes the shipped name only if it CONTAINS a recorded fragment. A producer
    that runs speculatively (a // sort key naming a prefix) or
    whose candidate is discarded (a later gate rejects it and another path names
    the molecule) therefore cannot demote a name that does not carry its
    fragment."""
    if not fragment:
        return
    _log_non_pin(fragment)
    cur = _NON_PIN_FRAGMENTS.get()
    if fragment not in cur:
        _NON_PIN_FRAGMENTS.set(cur + (fragment,))
    _touch("non_pin_fragments")


def record_non_pin_label(fragment: str) -> None:
    """Record ``fragment`` as a LABEL-ONLY non-PIN part (see ``_NON_PIN_LABELS``):
    a shipped name that contains it is labelled below the PIN; the PIN tier's
    promotion re-run still ships it."""
    if not fragment:
        return
    cur = _NON_PIN_LABELS.get()
    if fragment not in cur:
        _NON_PIN_LABELS.set(cur + (fragment,))
    _touch("non_pin_labels")


def record_uncertified_pin_name(name: str) -> None:
    """Record ``name`` as a PIN-form name whose preferred status the Blue Book leaves
    open (see ``_UNCERTIFIED_PIN_NAMES``): a shipped name that contains it is labelled
    pin_unverified, is_pin False."""
    if not name:
        return
    cur = _UNCERTIFIED_PIN_NAMES.get()
    if name not in cur:
        _UNCERTIFIED_PIN_NAMES.set(cur + (name,))
    _touch("uncertified_pin_names")


def record_derived_non_pin_fragment(source: str, derived: str) -> None:
    """Record ``derived`` as a non-PIN fragment when ``source`` contains a recorded one:
    a form built from a non-PIN name by a string conversion (an acid name turned into
    its acyl prefix) keeps the source's status, since the recorded substring does not
    survive the conversion."""
    if not source or not derived:
        return
    _log_non_pin(PROVENANCE_READ)
    if any(f in source for f in _NON_PIN_FRAGMENTS.get()):
        record_non_pin_fragment(derived)


def name_carries_non_pin_part(prov: dict, name: Optional[str], *,
                              label_forms: bool = True) -> bool:
    """True when ``name`` must not be labelled a PIN: the whole-call
    ``general_ring_prefix`` flag, the whole-call ``pin_promotion_rerun`` flag (the
    PIN tier's promotion re-run built the name), or a recorded non-PIN fragment it
    contains. ``label_forms`` is passed to ``rules.pin_vocabulary.
    non_pin_vocabulary``."""
    _log_non_pin(PROVENANCE_READ)
    if prov.get("general_ring_prefix") or prov.get("pin_promotion_rerun"):
        return True
    if not name:
        return False
    if any(f in name for f in (prov.get("uncertified_pin_names") or ())):
        return True
    if any(f in name for f in (prov.get("non_pin_fragments") or ())):
        return True
    if label_forms and any(f in name for f in (prov.get("non_pin_labels") or ())):
        return True
    from ..rules.pin_vocabulary import non_pin_vocabulary
    return non_pin_vocabulary(name, label_forms=label_forms) is not None


def get_provenance() -> dict:
    _log_non_pin(PROVENANCE_READ)
    return {
        "source": _SOURCE.get(),
        "opsin": _OPSIN.get(),
        "stereo_unexpressed": _STEREO_UNEXPRESSED.get(),
        "gate_outcome": _GATE_OUTCOME.get(),
        "gate_outcome_name": _GATE_OUTCOME_NAME.get(),
        "general_ring_prefix": _GENERAL_RING_PREFIX.get(),
        "suffix_free_prefix_name": _SUFFIX_FREE_PREFIX_NAME.get(),
        "non_pin_fragments": _NON_PIN_FRAGMENTS.get(),
        "pin_promotion_rerun": _PIN_PROMOTION_RERUN.get(),
        "non_pin_labels": _NON_PIN_LABELS.get(),
        "uncertified_pin_names": _UNCERTIFIED_PIN_NAMES.get(),
    }
