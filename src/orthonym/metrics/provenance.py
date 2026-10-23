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
#: the gate suppressed a candidate that OPSIN parsed to the input but read as ambiguous
#: (warning ``APPEARS_AMBIGUOUS``, ``validation.opsin_ambiguity``); the name is named once more
#: with every locant cited before it is withdrawn.
GATE_OUTCOME_AMBIGUOUS_SUPPRESSED = "opsin_ambiguous_suppressed"
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
#: prefix of ``twin_verified:<rule ids>``: OPSIN has no grammar for the name, and it reads a twin
#: spelling of the same structure (``validation.twin_spellings``) to the input's FULL InChIKey.
#: The producer owns the spelling; the twin proves the structure.
GATE_OUTCOME_TWIN_PREFIX = "twin_verified:"
#: prefix of ``acid_split_verified:<rule>``: OPSIN has no grammar for the name (a di- or poly-anhydride), and it
#: reads each acid the name cites, to the structures a poly-anhydride built from those acids has the input's FULL
#: InChIKey and descriptors of (``validation.anhydride_acids``). OPSIN never reads the name, or a spelling of it.
GATE_OUTCOME_ACID_SPLIT_PREFIX = "acid_split_verified:"

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

#: the ``opsin`` label of the twin outcome and its ``gates_passed`` token
_TWIN_LABEL, _TWIN_TOKEN = "verified_twin", "twin_round_trip"
#: the same of the acid-split outcome
_ACID_SPLIT_LABEL, _ACID_SPLIT_TOKEN = "verified_acid_split", "acid_split_round_trip"

#: the ``opsin`` labels of a name that was read back to the input's full structure: the name
#: itself, its twin spelling, or the acids of a poly-anhydride name
VERIFIED_OPSIN_LABELS = ("verified", _TWIN_LABEL, _ACID_SPLIT_LABEL)

#: the ``verified`` field of a row by the ``opsin`` label of its gate outcome
VERIFIED_FIELD_BY_OPSIN_LABEL = {
    "verified": "opsin",
    "verified_constitution_only": "opsin_constitution",
    _TWIN_LABEL: "opsin_twin",
    _ACID_SPLIT_LABEL: "opsin_acids",
}

#: the values of ``verified`` that earn a tier named "verified": a read-back of the name, or of
#: its twin spelling, of the acids of a poly-anhydride name, or the exact-match list identity
VERIFIED_FIELD_FOR_TIER = ("opsin", "opsin_twin", "opsin_acids", "identity")

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
# True while a wider-tier call (``Orthonym(general_fallback=True)``: complete, best-effort) runs.
# ``namer.Orthonym.name`` publishes it and never lowers it: a nested naming by a namer built with
# the default flags (the ion and charged-router re-entries) must not turn the PIN spellings on
# (``validation.twin_spellings``) inside a name the wider tier has to hand to OPSIN.
# ``rules.oxoacid_group_prefix.ps_tier_ctx`` is lowered by such a namer, and the isolated naming
# sessions of the fragment recursion reset ``general_fallback_ctx``.
wider_tier_ctx = contextvars.ContextVar("orthonym_wider_tier", default=False)
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
# The Blue Book rule of a label-only record that names one
# (``record_non_pin_label(fragment, rule=..., detail=...)``): ``(fragment, rule,
# detail)`` triples, read by ``non_pin_label_rule_failures``. A name lowered by such
# a record carries the rule in its row (``spelling_failures``), as a name lowered by
# a spelling check does. Same lifetime and roll-back rule as the labels.
_NON_PIN_LABEL_RULES = contextvars.ContextVar(
    "orthonym_prov_non_pin_label_rules", default=())
# Review a performance pass (F-06): names in PIN form whose preferred status the Blue Book itself
# leaves open -- the book prints a different (PIN) for the same structure. A shipped
# name that contains one is labelled pin_unverified (not systematic_verified: the code
# does not know it is not the PIN). Name-scoped, with the same lifetime and roll-back
# rule as the labels above.
_UNCERTIFIED_PIN_NAMES = contextvars.ContextVar(
    "orthonym_prov_uncertified_pin_names", default=())
# ZN10: names that were read back to the drawing of the input without its dative bonds, not to the
# drawing as written (``record_verified_through_split``): RDKit gives a drawing with dative bonds no
# InChIKey, so the key of the drawing without them is the reference. A name-scoped tuple, like the
# uncertified PIN names, with the same lifetime and roll-back rule.
_SPLIT_VERIFIED_NAMES = contextvars.ContextVar(
    "orthonym_prov_split_verified_names", default=())
# PB lane R: the names ``rules.pin_vocabulary.promote_at_pin_tier`` KEPT while the PIN tier's
# promotion re-run ran, i.e. pieces a best-effort composition producer built. The re-run's name
# stays certified (no ``pin_promotion_rerun`` flag) when this is empty: every piece of it came
# from a strict producer (``assembly.strict_ring_prefix``). Name-scoped like the lists above (it
# survives ``restore_provenance`` and is replayed by ``nested_memo``), and kept apart from
# ``_UNCERTIFIED_PIN_NAMES`` because ``name_carries_non_pin_part`` would drop the re-run name
# itself if it read these.
_BREADTH_PROMOTED_NAMES = contextvars.ContextVar(
    "orthonym_prov_breadth_promoted_names", default=())
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
# the recorded fragments (``get_provenance``, ``name_carries_non_pin_part``)
# appends ``PROVENANCE_READ`` instead: a call whose path read provenance is not
# memoised. ``record_derived_non_pin_fragment`` reads the record only to decide
# whether it records the derived form, and nothing it returns depends on that, so
# it appends the typed item ``('D', source, derived)``: a hit makes the call again
# and the record decides again. A STACK of lists, like _TOUCHED.
_NON_PIN_LOG = contextvars.ContextVar("orthonym_prov_non_pin_log", default=None)
PROVENANCE_READ = object()
# The most items a stored log that holds a derived call may have after ``compact_non_pin_log``: a bound
# on what a hit replays and on what a memo entry holds. A log of plain records only is not bounded, as
# before. It decides only whether the call is stored, never a name.
NON_PIN_LOG_CAP = 256


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
           "non_pin_label_rules", "uncertified_pin_names", "split_verified_names",
           "breadth_promoted_names")
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
    _NON_PIN_LABEL_RULES.set(())
    _UNCERTIFIED_PIN_NAMES.set(())
    _SPLIT_VERIFIED_NAMES.set(())
    _BREADTH_PROMOTED_NAMES.set(())


def append_missing(base: tuple, extra: tuple) -> tuple:
    """``base`` followed by the items of ``extra`` that ``base`` does not hold, in order: the merge
    ``restore_provenance`` makes of a snapshot and the live value of an accumulating record, and
    the merge a memo hit makes of the record its fresh call left. The records only grow within a
    naming call, so a giant molecule's tuples hold thousands of names; the membership test is a
    set lookup, not a scan of ``base`` for each item (that scan made every merge quadratic in the
    length of the record)."""
    try:
        held = frozenset(base)
        return base + tuple(f for f in extra if f not in held)
    except TypeError:   # an unhashable item in either record: the scan the merge always made
        return base + tuple(f for f in extra if f not in base)


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
    _NON_PIN_FRAGMENTS.set(append_missing(
        tuple(snapshot.get("non_pin_fragments") or ()), _NON_PIN_FRAGMENTS.get()))
    _PIN_PROMOTION_RERUN.set(bool(snapshot.get("pin_promotion_rerun")))
    _NON_PIN_LABELS.set(append_missing(
        tuple(snapshot.get("non_pin_labels") or ()), _NON_PIN_LABELS.get()))
    _NON_PIN_LABEL_RULES.set(append_missing(
        tuple(snapshot.get("non_pin_label_rules") or ()), _NON_PIN_LABEL_RULES.get()))
    _UNCERTIFIED_PIN_NAMES.set(append_missing(
        tuple(snapshot.get("uncertified_pin_names") or ()), _UNCERTIFIED_PIN_NAMES.get()))
    _SPLIT_VERIFIED_NAMES.set(append_missing(
        tuple(snapshot.get("split_verified_names") or ()), _SPLIT_VERIFIED_NAMES.get()))
    _BREADTH_PROMOTED_NAMES.set(append_missing(
        tuple(snapshot.get("breadth_promoted_names") or ()), _BREADTH_PROMOTED_NAMES.get()))
    _touch("source", "opsin", "stereo_unexpressed", "gate_outcome",
           "gate_outcome_name", "general_ring_prefix", "suffix_free_prefix_name",
           "non_pin_fragments", "pin_promotion_rerun", "non_pin_labels",
           "non_pin_label_rules", "uncertified_pin_names", "split_verified_names",
           "breadth_promoted_names")


class isolated_provenance:
    """Context manager: every provenance variable, the name-scoped records included,
    is exactly what it was on entry when the body exits, and the record calls the
    body made are taken out of the armed record-call logs again.

    For a check that names prefixes only to compare two parents
    (``perception.chains.chain_parent_prefix_seniority``) and whose own verdict is
    recorded after it, outside the body. Unlike:func:`restore_provenance`, the
    non-PIN records the body made are dropped too: the body's namings run through
    ``name_substituent_for_ordering``, whose memo and fragment-cache writes are
    discarded on exit (``memo.push_sandbox``, ``speculative_fragment_naming``), so no
    later cache hit can return a string whose record was dropped; a record kept from
    such a body would lower the label of a name the real naming built by another
    route only because the check ran. A provenance read in the body stays in the
    logs (``PROVENANCE_READ``), so an enclosing memo still declines to store a call
    whose path read provenance. Only a variable the body changed is written back."""

    _VARS = (_SOURCE, _OPSIN, _STEREO_UNEXPRESSED, _GATE_OUTCOME, _GATE_OUTCOME_NAME,
             _GENERAL_RING_PREFIX, _SUFFIX_FREE_PREFIX_NAME, _NON_PIN_FRAGMENTS,
             _PIN_PROMOTION_RERUN, _NON_PIN_LABELS, _UNCERTIFIED_PIN_NAMES, _SPLIT_VERIFIED_NAMES)
    # not ``_BREADTH_PROMOTED_NAMES``: ``assembly.memo`` replays no provenance, so a breadth piece built
    # in the body and served from the memo later would ship unrecorded; a record that outlives the body
    # only ever adds a piece ``record_pin_promotion_rerun`` looks for in the final name

    def __enter__(self):
        self._values = [(var, var.get()) for var in self._VARS]
        stack = _NON_PIN_LOG.get()
        self._logs = [(log, len(log)) for log in stack] if stack else []
        return self

    def __exit__(self, *exc):
        for var, value in self._values:
            if var.get() is not value:
                var.set(value)
        for log, n in self._logs:
            tail = log[n:]
            del log[n:]
            if any(item is PROVENANCE_READ for item in tail):
                log.append(PROVENANCE_READ)
        return False


def record_source(source: str, opsin: Optional[str] = None) -> None:
    _SOURCE.set(source)
    _touch("source")
    if opsin is not None:
        _OPSIN.set(opsin)
        _touch("opsin")


def producer_record() -> tuple:
    """The producer record of the call so far: ``(source, opsin)`` as
    :func:`record_source` left them. Pair with:func:`restore_producer_record` around
    an attempt whose name may be discarded. The value read is only written back to
    the record, never used to build a name, so the read is not logged as a
    provenance read (``PROVENANCE_READ``)."""
    return (_SOURCE.get(), _OPSIN.get())


def restore_producer_record(record: tuple) -> None:
    """Re-set ``source`` and ``opsin`` to ``record`` (a:func:`producer_record`
    value). A variable that already holds its value is not written, so a call that
    left the record unchanged touches nothing."""
    source, opsin = record
    if _SOURCE.get() != source:
        _SOURCE.set(source)
        _touch("source")
    if _OPSIN.get() != opsin:
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


def twin_outcome(rules: str) -> str:
    """``twin_verified:<rule ids>`` — the name ships on its construction, and a twin
    spelling OPSIN reads round-trips to the input's full InChIKey."""
    return GATE_OUTCOME_TWIN_PREFIX + rules


def acid_split_outcome(rule: str) -> str:
    """``acid_split_verified:<rule>`` — the name ships on its construction, and the acids it cites, read by
    OPSIN, build a structure that is the input's (see:data:`GATE_OUTCOME_ACID_SPLIT_PREFIX`)."""
    return GATE_OUTCOME_ACID_SPLIT_PREFIX + rule


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
    """``verified`` / ``verified_constitution_only`` / ``verified_twin`` / ``verified_acid_split`` /
    ``unverified``.

    Allowlist, so it fails closed: any outcome not explicitly listed —
    including a state that does not exist yet — reports ``unverified``.
    """
    if outcome and outcome.startswith(GATE_OUTCOME_TWIN_PREFIX):
        return _TWIN_LABEL
    if outcome and outcome.startswith(GATE_OUTCOME_ACID_SPLIT_PREFIX):
        return _ACID_SPLIT_LABEL
    return _VERIFIED_GATE_OUTCOMES.get(outcome, "unverified")


def gate_token_for_gate_outcome(outcome: Optional[str]) -> Optional[str]:
    """The `gates_passed` token this outcome earns, or None if it earns none."""
    if outcome and outcome.startswith(GATE_OUTCOME_TWIN_PREFIX):
        return _TWIN_TOKEN
    if outcome and outcome.startswith(GATE_OUTCOME_ACID_SPLIT_PREFIX):
        return _ACID_SPLIT_TOKEN
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
    (see ``_PIN_PROMOTION_RERUN``): labelled pin_unverified, never pin_verified. Only when a breadth
    piece the re-run kept (``record_breadth_promotion``) is part of the name: a name whose every piece
    a strict producer built (``assembly.strict_ring_prefix``) is the strict path's own name. A piece
    kept in a route the re-run then left does not count (the label would depend on the input atom
    order); with no name to read the whole call is flagged."""
    pieces = _BREADTH_PROMOTED_NAMES.get()
    name = _GATE_OUTCOME_NAME.get()
    if not pieces or (name and not any(piece in name for piece in pieces)):
        return
    _PIN_PROMOTION_RERUN.set(True)
    _touch("pin_promotion_rerun")


def record_breadth_promotion(name: str) -> None:
    """Record a name ``promote_at_pin_tier`` kept: a piece a best-effort producer built (see
    ``_BREADTH_PROMOTED_NAMES``)."""
    if not name:
        return
    cur = _BREADTH_PROMOTED_NAMES.get()
    if name not in cur:
        _BREADTH_PROMOTED_NAMES.set(cur + (name,))
    _touch("breadth_promoted_names")


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
    _add_non_pin_fragment(fragment)


def _add_non_pin_fragment(fragment: str) -> None:
    """The write of:func:`record_non_pin_fragment`, without the log item."""
    cur = _NON_PIN_FRAGMENTS.get()
    if fragment not in cur:
        _NON_PIN_FRAGMENTS.set(cur + (fragment,))
    _touch("non_pin_fragments")


def record_non_pin_label(fragment: str, rule: Optional[str] = None,
                         detail: str = "") -> None:
    """Record ``fragment`` as a LABEL-ONLY non-PIN part (see ``_NON_PIN_LABELS``):
    a shipped name that contains it is labelled below the PIN; the PIN tier's
    promotion re-run still ships it.

    ``rule`` (a Blue Book rule id, with ``detail`` saying how) is the reason the
    fragment is not the PIN form: the row of a name the record lowers lists it in
    ``spelling_failures`` (see ``non_pin_label_rule_failures``)."""
    if not fragment:
        return
    cur = _NON_PIN_LABELS.get()
    if fragment not in cur:
        _NON_PIN_LABELS.set(cur + (fragment,))
    _touch("non_pin_labels")
    if rule:
        entry = (fragment, rule, detail)
        cur_rules = _NON_PIN_LABEL_RULES.get()
        if entry not in cur_rules:
            _NON_PIN_LABEL_RULES.set(cur_rules + (entry,))
        _touch("non_pin_label_rules")


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


def record_verified_through_split(name: str) -> None:
    """Record that ``name`` was read back to the drawing of the input without its dative bonds
    (``assembly.metal_offer``), not to the drawing as written. ``verified_field`` then reports
    ``opsin_split`` for the name in place of ``opsin``."""
    if not name:
        return
    cur = _SPLIT_VERIFIED_NAMES.get()
    if name not in cur:
        _SPLIT_VERIFIED_NAMES.set(cur + (name,))
    _touch("split_verified_names")


def verified_field(verified: str, prov: dict, name: Optional[str]) -> str:
    """The ``verified`` field of a row: ``opsin`` becomes ``opsin_split`` for a name recorded by
    :func:`record_verified_through_split` (OPSIN read the whole name back to the drawing of the
    input without its dative bonds, whose standard InChIKey is the reference, and an independent
    count of atoms and charges ties that drawing to the input); every other value is unchanged."""
    _log_non_pin(PROVENANCE_READ)
    if verified == "opsin" and name and name in (prov.get("split_verified_names") or ()):
        return "opsin_split"
    return verified


def record_derived_uncertified_pin_name(source: str, derived: str) -> None:
    """Record ``derived`` as an uncertified PIN-form name when ``source`` contains a
    recorded one: a name built from ``source`` by a splice that leaves every locant of
    ``source`` as it is (an isotopic descriptor that needs no locant) keeps the status of
    ``source``, since the recorded string does not survive the splice."""
    if not source or not derived:
        return
    _log_non_pin(PROVENANCE_READ)
    if any(n in source for n in _UNCERTIFIED_PIN_NAMES.get()):
        record_uncertified_pin_name(derived)


def record_derived_non_pin_fragment(source: str, derived: str, _item=None) -> None:
    """Record ``derived`` as a non-PIN fragment when ``source`` contains a recorded one:
    a form built from a non-PIN name by a string conversion (an acid name turned into
    its acyl prefix) keeps the source's status, since the recorded substring does not
    survive the conversion.

    The call is logged as it was made (``('D', source, derived)``), and the record it makes when it
    fires is not: a replay of the call decides again against the record of the replay, where a replay
    of the record would add ``derived`` also when the record holds no fragment of ``source``. A replay
    passes the stored item as ``_item``, so the logs share it."""
    if not source or not derived:
        return
    _log_non_pin(_item if _item is not None else ('D', source, derived))
    if any(f in source for f in _NON_PIN_FRAGMENTS.get()):
        _add_non_pin_fragment(derived)


def compact_non_pin_log(items, cap: Optional[int] = None) -> Optional[tuple]:
    """The record-call log ``items`` of a call, as an equivalent tuple with the no-op repeats removed: its replay
    (:func:`replay_non_pin_log`) leaves any record as the replay of ``items`` leaves it; None when the
    call read provenance (``PROVENANCE_READ``) or the result holds a derived call and more than ``cap``
    items (``NON_PIN_LOG_CAP``; a log of plain items is stored whatever its length, as it always was).
    The record only grows under a replay, so an item is left out when it changes nothing after the items
    kept before it:

    * a fragment recorded before (``record_non_pin_fragment`` adds it whatever the record holds);
    * a derived call whose ``derived`` is such a fragment (it adds that or nothing);
    * a repeat of a derived call, unless a kept item since its last kept occurrence adds a fragment
      that is a substring of its ``source``: with none, the record holds a fragment of ``source`` at
      the repeat only if it held one at the earlier call, which then added ``derived`` already.
      Dropping every repeat would lose the ``derived`` of a call that found no fragment of ``source``
      the first time and finds one that a later item recorded.

    Each kept item is tested once against each derived call, and a log with a derived call stops at the
    cap, so the cost is bounded by the square of the cap."""
    limit = NON_PIN_LOG_CAP if cap is None else cap
    kept: list = []
    recorded: set = set()    # the fragments a kept plain item adds
    checked: dict = {}       # (source, derived) -> the number of kept items up to its last kept call
    derived_calls = False
    for item in items:
        if item is PROVENANCE_READ:
            return None
        if isinstance(item, str):
            if item in recorded:
                continue
            recorded.add(item)
        else:
            _tag, source, derived = item
            if derived in recorded:
                continue
            key = (source, derived)
            seen = checked.get(key)
            if seen is not None and not any(
                    (other if isinstance(other, str) else other[2]) in source for other in kept[seen:]):
                checked[key] = len(kept)
                continue
            checked[key] = len(kept) + 1
            derived_calls = True
        kept.append(item)
        if derived_calls and len(kept) > limit:
            return None
    return tuple(kept)


def replay_non_pin_log(items) -> None:
    """Make the record calls of a stored log again, in order, into every armed log: a plain item
    through:func:`record_non_pin_fragment`, a derived call through
    :func:`record_derived_non_pin_fragment`."""
    for item in items:
        if isinstance(item, str):
            record_non_pin_fragment(item)
        else:
            record_derived_non_pin_fragment(item[1], item[2], item)


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


def non_pin_label_rule_failures(prov: dict, name: Optional[str]) -> list:
    """The reasons of the label-only records that ``name`` contains and that name a
    rule (``record_non_pin_label(..., rule=...)``), as ``spelling_failures`` entries
    ``{"rule":..., "detail":...}`` in record order, one per rule and detail."""
    _log_non_pin(PROVENANCE_READ)
    if not name:
        return []
    out = []
    for fragment, rule, detail in (prov.get("non_pin_label_rules") or ()):
        entry = {"rule": rule, "detail": detail}
        if fragment in name and entry not in out:
            out.append(entry)
    return out


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
        "non_pin_label_rules": _NON_PIN_LABEL_RULES.get(),
        "uncertified_pin_names": _UNCERTIFIED_PIN_NAMES.get(),
        "split_verified_names": _SPLIT_VERIFIED_NAMES.get(),
        "breadth_promoted_names": _BREADTH_PROMOTED_NAMES.get(),
    }
