""" universal coverage-by-construction namer (best-effort tier ONLY).

Reached only on the general-fallback best-effort recovery path -- master
switch ``self._general_fallback`` in ``namer.py`` (its early ``return None``
when that flag is False; ``general_fallback_unverified`` is a secondary,
narrower flag on the same path) -- which itself only runs AFTER the
PIN/systematic path has already abstained. So this module can never change a
PIN emission; it only ever fills in where the PIN tiers were silent.

Produces an atom-complete, E1-certified name or ``None`` (clean abstain,
never a partial name). ``validation.e1_certificate.verify_certificate`` is
the atom-coverage certificate; downstream (OPSIN round-trip) is the
second half of the 0-wrong net, not a substitute for E1 here.

This module is the Task 2 SKELETON only: the control flow + the locked
``_Candidate`` interface. Task 3 fills in ``_best_effort_candidate`` with the
real parent+substituents producer; Tasks 4-6 add the per-class cascade and
wire this namer into ``namer.py``.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..validation.coverage_gate import certify_general_result

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
    # True when ``_prefer_verified_floor`` put the universal floor's spelling in
    # place of a rung candidate that failed the round trip (see
    # ``name_t4_complete_candidate``).
    floor_substitute: bool = False
    # True when the candidate is the final rung's universal floor itself: every
    # engine rung above it declined.
    final_floor: bool = False


#: Whether the last whole-molecule ``name_t4_complete`` call returned the universal
#: floor in place of a rung candidate that failed its round trip. Written at the END
#: of each call (so a nested call for a fragment cannot leave its value behind for
#: the outer one); read by the namer's recovery right after its own call.
_LAST_FLOOR_SUBSTITUTE: ContextVar[bool] = ContextVar(
    "t4_last_floor_substitute", default=False)


#: Whether the last whole-molecule ``name_t4_complete`` call returned the final
#: rung's universal floor (every engine rung declined); written like the flag above.
_LAST_FINAL_FLOOR: ContextVar[bool] = ContextVar("t4_last_final_floor", default=False)


def last_t4_was_final_floor() -> bool:
    """True iff the most recent ``name_t4_complete`` call in this context returned
    the final rung's universal floor: no engine rung (the general engine with the
    perceived features or a feature override, the polyol rung) gave a candidate."""
    return _LAST_FINAL_FLOOR.get()


def last_t4_was_floor_substitute() -> bool:
    """True iff the most recent ``name_t4_complete`` call in this context returned
    the universal floor standing in for a rung candidate that failed the full-
    InChIKey round trip (``_prefer_verified_floor``). Such a rung candidate is the
    mark of a degraded naming context (the recovery inside ``name``; see
    ``namer._try_besteffort_clean_general_fallthrough``), where a clean context can
    still build the engine's own verified name."""
    return _LAST_FLOOR_SUBSTITUTE.get()


def reset_t4_floor_substitute() -> None:
    """Clear the flags before a call whose producer may be replaced (tests stub
    ``name_t4_complete``), so a stale value is never read."""
    _LAST_FLOOR_SUBSTITUTE.set(False)
    _LAST_FINAL_FLOOR.set(False)


def name_t4_complete_candidate(mol, features) -> Optional[_Candidate]:
    """``name_t4_complete`` returning the whole candidate (``floor_substitute``
    included). Same gates, same name."""
    candidate = _best_effort_candidate(mol, features)
    if candidate is not None and candidate.result_obj is not None:
        # a phase Part A: the shared best-effort certification gate (E1 + the
        # binding spine). ``allow_charged=False`` preserves the T4
        # net-charge-out-of-scope contract (see ``name_t4_complete``).
        if not certify_general_result(mol, candidate.result_obj,
                                      allow_charged=False):
            candidate = None
    _LAST_FLOOR_SUBSTITUTE.set(bool(candidate is not None
                                    and candidate.floor_substitute))
    _LAST_FINAL_FLOOR.set(bool(candidate is not None and candidate.final_floor))
    return candidate


def name_t4_complete(mol, features) -> Optional[str]:
    """Best-effort name for ``mol``, or ``None`` (clean abstain).

    Control flow (locked for Tasks 3-6):
      1. Ask ``_best_effort_candidate`` for a name + its E1 proof object.
      2. No candidate at all -> abstain (``None``).
      3. A candidate with no proof object (``result_obj is None``) ships
         unchecked -- E1 has nothing to verify.
      4. Otherwise the candidate must pass ``verify_certificate`` or it is
         discarded -- never patched, never shipped anyway.

    Audit-coverage note (Fix 3a, final review): this returns a bare ``str``, not
    the ``GeneralEngineResult`` with its atom->token ``bindings``. So when
    ``namer.py`` ships a name it CANNOT populate the binding-proof ledger --
    ``_record_binding_proof`` runs only on the engine's own-name ``else`` branch,
    never on the branch. That is an audit-coverage gap, not a correctness one:
    it is benign under the default (``binding_proof`` off), and T4's 0-wrong net
    is E1 (proven internally here) + (the OPSIN round-trip in namer.py),
    neither of which needs the ledger. A follow-on wanting binding proofs
    would return the ``_Candidate`` (which carries ``result_obj``) instead of a
    bare string.

    a phase Task 2b: ALSO requires ``verify_spine`` (audit mode) to pass.
    ``verify_certificate`` proves only the flat atom partition (P1's job); the
    binding spine additionally proves bond totality (P2), token-span anchoring
    (P4/P5) and arity (P6) -- axes E1 cannot see at all (a name whose bindings
    silently re-fragment a ring, e.g. spelling cyclohexane as two disjoint
    propyl halves, passes E1 outright). ``mode="audit"`` stays the base mode
    (never "strict" here): a blanket strict flip would ALSO change P2's
    bond-linkage inference policy and P5/P6's unrelated unproven-codes
    severity, none of which this task's scope covers. ``allow_charged=False``
    matches the ``verify_certificate`` call above so the two proofs never
    disagree about scope. A spine failure voids the candidate exactly like an
    E1 failure does -- no new control flow.

    a phase Task 4: ``escalate=STRICT_STEREO_CHARGE_AXES`` promotes JUST the
    P8 stereo axis and P3's ``CHARGE_UNVERIFIED`` to error severity, on top of
    ``mode="audit"`` -- the coverage certificate's stereo axis was shipped
    audit-only in Task 3 (findings recorded, `ok` never affected); the Task 4
    diagnostic re-scan found 0 remaining false positives and 0 genuine stereo
    drops, so this makes it enforcing. ``allow_charged=False`` means
    ``NET_CHARGE_OUT_OF_SCOPE`` (unconditional "error", not mode/escalate-
    gated) already voids any nonzero-net-charge candidate reaching this
    point regardless of this promotion; what newly blocks is a ZWITTERION
    (net charge 0, but individual atoms charged) whose charges no producer
    threaded through ``charge_atom_ids`` -- measured byte-identical on
    a dev split best-effort (see the Task 4 report).

    Task 2a first wired this and measurably false-voided 3 correct,
    OPSIN-round-tripping a dev split rows via a pre-existing P6 (``token_arity``)
    false-positive on replacement-nomenclature substituent tokens; that bug is
    fixed (Task 2b, ``name_morphemes.py::_evaluate``'s multiplier short-circuit
    now also skips zero-atom REPL segments) and the a dev split best-effort
    before/after re-measurement is BYTE-IDENTICAL emit/rt_exact -- see
    internal notes.
    """
    # a phase Part A: the shared best-effort certification gate (E1 + the
    # binding spine) runs in ``name_t4_complete_candidate``: behaviour-identical
    # to the former inline verify_certificate + verify_spine(escalate=
    # STRICT_STEREO_CHARGE_AXES) pair; the ONE place all three lanes route
    # through so they cannot drift (validation/coverage_gate.py).
    candidate = name_t4_complete_candidate(mol, features)
    return candidate.name if candidate is not None else None


def _run_general_e1(mol, features) -> Optional[_Candidate]:
    """One bounded engine attempt: run ``name_general`` at the T4-permissive
    setting and E1-audit the result. Returns a complete ``_Candidate`` or
    ``None`` -- NEVER a partial, NEVER a raise.

    ``allow_aromatic_general=True`` opens the lone-monocycle / mancude-cage
    paths and lifts the charge/mancude refusals; ``allow_suffix_free=True`` is
    the best-effort terminal-ring assembly. Chosen independent of the calling
    instance's flags -- is best-effort, so it always uses the most permissive
    engine. ``name_general`` is fail-closed (returns None, never a wrong name),
    but the call is wrapped so any engine error is a clean abstain.

    ``verify_certificate`` here is the SAME E1 gate ``name_t4_complete`` re-runs;
    matching its default (``allow_charged=False``) keeps our accept/reject
    decision identical to the backstop's, so a candidate we approve is never
    voided one call later. The coverage check means we return ``None`` ourselves
    rather than rely on the backstop, so no partial is ever handed up.

    a phase Task 2b: the SAME additional ``verify_spine`` (audit mode) check as
    ``name_t4_complete`` -- see that docstring for why. Duplicated here (not
    only at the outer gate) so a rung this function rejects never reaches the
    cascade's next rung believing it merely failed E1; it is rejected for
    exactly the reason the outer gate would reject it, one call earlier.

    a phase Task 4: the SAME ``escalate=STRICT_STEREO_CHARGE_AXES`` promotion
    as ``name_t4_complete`` -- see that docstring.

    A molecule with a NET formal charge returns ``None`` before the engine runs.
    The certification below passes ``allow_charged=False``, and E1 then refuses
    every result for such a molecule
    (``validation.e1_certificate._verify_partition``: "net formal charge nonzero
    (G1 charge scope)"; every other exit of that check is a refusal too), so the
    engine's result for it was always discarded here. Running it cost the time
    anyway: the engine names every N-acyl substituent through its acid
    (``_name_amino_branch`` -> ``acyl_amido_prefix_from_branch``, 9e74facde),
    and for the ChEBI lipid II-type glycopeptide (130 heavy atoms) the discarded
    runs on its charged fragments took 139 of its 211 s (trace, fresh process:
    612 outermost calls on charged fragments, 0 results certified).
    """
    from rdkit import Chem

    try:
        if Chem.GetFormalCharge(mol) != 0:
            return None
    except Exception:  # an unreadable charge keeps the engine run (and its verdict)
        pass

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
    # a phase Part A: the SAME shared certification gate as name_t4_complete
    # (E1 + binding spine). Duplicated here so a rung this function rejects is
    # rejected for exactly the reason the outer gate would reject it, one call
    # earlier -- behaviour-identical to the former inline pair.
    if not certify_general_result(mol, result, allow_charged=False):
        return None
    return _Candidate(name=result.name, result_obj=result)


def _clone_features_with(features, **overrides):
    """Return a shallow COPY of ``features`` with the given attributes set, or
    ``None`` if it is not clonable.

    The caller owns ``features`` and reuses it (namer.py's dispatch), so it
    is never mutated in place -- exactly the ``_copy.copy(features)`` + reassign
    pattern ``general_engine._name_terminal_ring_assembly`` uses to suppress a
    principal group. Reassigning ``principal_group_atoms`` to a fresh ````
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


def _prefer_verified_floor(mol, candidate: "_Candidate") -> "_Candidate":
    """ Track A #2 -- verify-before-commit for every best-effort cascade rung.

    Each early rung here (``_run_general_e1`` rung 0, the polyol rung, the
    feature-override cascade) is E1-complete -- E1 proves atom COVERAGE (every
    heavy atom is bound to a token), but it is blind to BOND ORDER and to
    STEREO. So a rung can emit a name that E1 passes yet is constitution- or
    stereo-WRONG, e.g. an alkynyloxy ether named ``...butoxy`` (the C#C triple
    bond silently dropped), or a defined stereocentre omitted. Such a name
    short-circuits the cascade at ``return candidate`` before the final
    ``name_universal_substitutive`` rung, which re-derives the whole graph
    constitution-complete (and, since 33f5e47c, branch-stereo-complete). The
    wrong rung name then FAILS the downstream full-InChIKey gate -- and because
    it is the only offer, the whole molecule ABSTAINS, even though the floor
    names it correctly (a project rule offer-not-return; council Track A #2).

    So: full-InChIKey verify the rung's name. If it verifies, keep it (the
    engine rungs give better nomenclature than the ugly floor). If it does NOT,
    prefer the universal floor when IT verifies. Both branches are full-InChIKey
    gated (``verify_or_none``), so this can never ship a wrong molecule; and it
    never abstains (keeps the rung when neither verifies -- e.g. OPSIN absent,
    where ``verify_or_none`` fails closed).

    This is strictly non-regressive: a rung name that fails ``verify_or_none``
    here would ALSO be rejected by the offer pool's own ``_offer_rt_ok`` in
    namer.py, so a correct rung name (which passes both) is never displaced --
    the change only converts an about-to-be-rejected rung (-> abstain) into the
    floor's correct name (-> emit). Best-effort only; PIN is untouched.
    """
    from rdkit import Chem

    from ..validation.reconstruct import verify_or_none

    smi = Chem.MolToSmiles(mol)
    try:
        if verify_or_none(candidate.name, smi) is not None:
            return candidate  # rung already RT-correct: keep its nomenclature
    except Exception:
        pass  # treat an unverifiable rung as a candidate for the floor upgrade
    floor_name = _verified_universal_floor(mol, smi)
    if floor_name is not None:
        return _Candidate(name=floor_name, result_obj=None, floor_substitute=True)
    return candidate


def _reads_back_as_constitution(mol, name: str) -> bool:
    """True iff OPSIN parses ``name`` to the input's constitution (InChIKey
    skeleton block; ``validate_atom_coverage``). False for an unparseable name,
    a different molecule, or no parse at all (fail-closed: the caller then only
    looks for a fully verified alternative)."""
    try:
        from ..validation.atom_coverage import validate_atom_coverage
        cov = validate_atom_coverage(mol, name)
        return cov.method == "parse_back" and bool(cov.constitution_match)
    except Exception:
        return False


def _verified_universal_floor(mol, smi: str, first=None) -> Optional[str]:
    """The first spelling of the universal floor that FULL-InChIKey verifies
    (``verify_or_none``), or ``None``.

    Order: the floor as built; the von Baeyer form of a fused spiro component
    (Task C); then both forms without the '<acyl>oxy' leaf (see
    ``_leaf_checked_floor``). ``first`` is an already-built floor result for the
    first spelling (the final rung has one), so it is not built twice.

    Shared by ``_prefer_verified_floor`` (a rung's name failed the gate) and the
    final rung (the floor's own first spelling failed it). The final rung used to
    ship the first spelling unverified and stop, so a fused-spiro floor name that
    fails the round trip (e.g. a retained ring name whose locants read back as a
    different molecule) abstained although its von Baeyer form verifies; a rung
    candidate reaching ``_prefer_verified_floor`` got the retry, the last rung did
    not. Fail-closed: any error returns ``None`` (the caller keeps what it had).

    Roadmap N5: the ladder runs first with the book spellings
    (``universal_substituent._Ctx.book_forms``); when none of its names verifies and
    a book spelling was part of one, it runs again with the mechanical spellings the
    floor gave before, so a book spelling never costs a name."""
    name, fired = _verified_floor_ladder(mol, smi, first, book_forms=True)
    if name is not None or not fired:
        return name
    # One book spelling that fails the round trip (a chain, an acyl group, a fused ring) must
    # not take the ring names of the whole molecule with it. Ladder order: book -> the
    # keep-forms rungs (``book_prefixes.RETRY_KEEP_ORDER``: ring-only, then the other lanes'
    # kinds) -> mechanical. A rung runs only when it differs from the run it follows
    # (``retry_keeps``: every kept kind was given and another kind was too).
    from .book_prefixes import book_forms_minus, retry_rungs
    for how, kinds in retry_rungs(fired):
        if how == "drop":
            # every book spelling except one kind (item 12a, ``RETRY_DROP_ORDER``): the
            # spelling before that lane's forms, the other kinds' names kept
            with book_forms_minus(kinds):
                name, _kept_fired = _verified_floor_ladder(mol, smi, None, book_forms=True)
        else:
            name, _kept_fired = _verified_floor_ladder(mol, smi, None, book_forms=False,
                                                       keep_forms=kinds)
        if name is not None:
            return name
    name, _fired = _verified_floor_ladder(mol, smi, None, book_forms=False)
    return name


def _verified_floor_ladder(mol, smi: str, first, book_forms: bool,
                           keep_forms: frozenset = frozenset()):
    """``(name, kinds)``: the ladder of ``_verified_universal_floor`` at one spelling
    setting; ``kinds`` is the set of kinds of book spelling its floor names carried (empty
    when none: falsy, as the bool it replaces)."""
    from ..validation.reconstruct import verify_or_none
    fired = frozenset()
    try:
        from .universal_substituent import name_universal_substitutive
        uni = first if first is not None else name_universal_substitutive(
            mol, book_forms=book_forms, keep_forms=keep_forms)
        fired = frozenset(uni.book_forms_kinds) if uni is not None else frozenset()
        if uni is not None and uni.name and verify_or_none(uni.name, smi) is not None:
            return uni.name, fired
        # Task C retry: the systematic fused-spiro name (a fusion descriptor the
        # numbering subsystem got wrong or OPSIN cannot parse) failed the gate.
        # Retry FORCING the von-Baeyer polyene form of the fused spiro
        # component -- always constitution-faithful, so it recovers breadth at
        # 0-wrong with an uglier (non-PIN, floor-only) covering name.
        uni_vb = name_universal_substitutive(mol, force_vonbaeyer_spiro=True,
                                             book_forms=book_forms,
                                             keep_forms=keep_forms)
        fired = fired | (frozenset(uni_vb.book_forms_kinds) if uni_vb is not None else frozenset())
        if uni_vb is not None and uni_vb.name \
                and verify_or_none(uni_vb.name, smi) is not None:
            return uni_vb.name, fired
        # Leaf-free retry: an '<acyl>oxy' leaf named a branch of the floor name
        # and the name failed the gate -- rebuild both forms WITHOUT that leaf,
        # the spelling the floor gave before the leaf existed (see
        # ``_leaf_checked_floor``).
        if any(u is not None and u.acyloxy_leaf_fired for u in (uni, uni_vb)):
            for force_vb in (False, True):
                alt = name_universal_substitutive(
                    mol, force_vonbaeyer_spiro=force_vb, acyloxy_leaf=False,
                    book_forms=book_forms, keep_forms=keep_forms)
                fired = fired | (frozenset(alt.book_forms_kinds) if alt is not None else frozenset())
                if alt is not None and alt.name \
                        and verify_or_none(alt.name, smi) is not None:
                    return alt.name, fired
    except Exception as exc:  # fail-closed: keep the rung, never abstain
        logger.info("t4 verified-floor preference raised: %s", exc)
    return None, fired


def name_prefix_order_fallback(mol) -> Optional[str]:
    """The universal floor's name of ``mol`` with its parent's prefixes cited out
    of the order, or ``None``.

    The last resort of the best-effort tier for a molecule it would otherwise not
    name (``namer.Orthonym._maybe_prefix_order_fallback`` is the only caller, and
    it adds the checks that make a shipped name the input's): the round trip
    (OPSIN's SMILES read by RDKit) can depend on the citation order of the
    prefixes, so a floor name in the order (the Blue Book order) can fail
    at the stereo layer although its descriptors are right -- see
    ``universal_substituent._prefix_order_fallback``. A name is returned only when
    that rung built it: the -ordered spelling failed at the stereo layer
    only and this spelling reads back to the input's full InChIKey. Such a name
    breaks on purpose and is never a PIN. Fail-closed: any error returns
    ``None``."""
    try:
        from .universal_substituent import name_universal_substitutive
        uni = name_universal_substitutive(mol, prefix_order_fallback=True)
        if (uni is None or not uni.prefix_order_fallback) and uni is not None \
                and uni.book_forms_fired:
            # roadmap N5: the mechanical spellings, as before the book forms
            uni = name_universal_substitutive(mol, prefix_order_fallback=True,
                                              book_forms=False)
    except Exception as exc:  # fail-closed: the abstention stands
        logger.info("t4 prefix-order fallback raised: %s", exc)
        return None
    if uni is None or not uni.prefix_order_fallback or not uni.name:
        return None
    return uni.name


def _book_checked_floor(mol, uni):
    """The universal floor's name ``uni``, or -- when a book spelling (roadmap N5,
    ``universal_substituent._Ctx.book_forms``) is part of it and the name does NOT
    pass the full-InChIKey round trip -- the floor rebuilt with the mechanical
    spellings it gave before. So a book spelling only ever replaces the mechanical
    one with a verified name; otherwise the mechanical name is offered and the
    caller's round-trip ladder judges it exactly as before. A name with no book
    spelling is returned untouched (no extra round trip)."""
    if not uni.book_forms_fired:
        return uni
    from rdkit import Chem

    from ..validation.reconstruct import verify_or_none
    from .universal_substituent import name_universal_substitutive
    try:
        if verify_or_none(uni.name, Chem.MolToSmiles(mol)) is not None:
            return uni
    except Exception:  # an unverifiable book name: prefer the mechanical one
        pass
    # the kept kinds stay the book's when the name that keeps only them verifies: one
    # failing chain or acyl spelling does not cost the molecule its ring names. Same rung
    # order as ``_verified_universal_floor``.
    from .book_prefixes import retry_rungs, retry_spelling
    for rung in retry_rungs(uni.book_forms_kinds):
        try:
            with retry_spelling(rung):
                kept = name_universal_substitutive(
                    mol, book_forms=rung[0] == "drop", keep_forms=rung[1] if rung[0] == "keep"
                    else frozenset())
            if (kept is not None and kept.name and kept.book_forms_fired
                    and verify_or_none(kept.name, Chem.MolToSmiles(mol)) is not None):
                return kept
        except Exception:  # a keep-forms rung that raises is a decline
            pass
    alt = name_universal_substitutive(mol, book_forms=False)
    return alt if alt is not None and alt.name else uni


def _leaf_checked_floor(mol, uni):
    """The universal floor's name ``uni``, or -- when an '<acyl>oxy' leaf named one
    of its branches and the name does NOT pass the full-InChIKey round trip --
    the floor rebuilt without that leaf.

    The leaf (``universal_substituent._acyloxy_leaf``, TRIAGE g8 C4) spells an
    ester branch through the shared acid engine, 'acetyloxy' instead of the
    replacement chain '2-oxo-1-oxapropan-1-yl'. Where that engine's token is
    wrong for the branch, the whole floor name fails the round trip and, the
    floor being the last rung, the molecule abstains -- although the floor
    named it, round-trip exact, before the leaf existed (measured: cholesteryl
    hydrogen succinate, 'butanedioyloxy' for -O-CO-CH2-CH2-COOH; best-effort
    lost it in 76fb18964). So the leaf only ever REPLACES the leaf-free
    spelling with a verified one; otherwise the leaf-free name is offered, and
    the caller's round-trip ladder judges it exactly as before. A name with no
    leaf is returned untouched (no extra round trip)."""
    if not uni.acyloxy_leaf_fired:
        return uni
    from rdkit import Chem

    from ..validation.reconstruct import verify_or_none
    from .universal_substituent import name_universal_substitutive
    try:
        if verify_or_none(uni.name, Chem.MolToSmiles(mol)) is not None:
            return uni
    except Exception:  # an unverifiable leaf name: prefer the leaf-free one
        pass
    alt = name_universal_substitutive(mol, acyloxy_leaf=False,
                                      book_forms=uni.book_forms_fired)
    return alt if alt is not None and alt.name else uni


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

    Feature-aliasing note (Fix 3b, final review): rung 0 passes the caller's
    ORIGINAL ``features`` object to ``name_general`` (only the cascade rungs
    below take a ``_clone_features_with`` COPY). This is safe -- ``name_general``
    reads the perceived features and never mutates the shared containers
    (``principal_group_atoms`` etc.), so rung 0 cannot corrupt the object the
    caller reuses. The clones exist only because rungs 1-2 must OVERRIDE
    ``principal_group`` / ``chain_is_parent`` without disturbing rung 0's view.

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
    a candidate is handed up, and (OPSIN round-trip) downstream is the
    second half of the 0-wrong net. When a decline needs a capability the
    codebase lacks -- e.g. the dichlorophosphoryl-carbamate substituent of
    ``C1CCC(CC1)OC(=O)NP(=O)(Cl)Cl``, whose recursive substituent namer hits its
    depth cap -- every rung fails E1 and the producer HONESTLY abstains
    (``None``), never a fabricated or partial name. A suffix-free prefix name is
    a valid description of the right structure that is not a well-formed PIN;
    that is licit here (a project rule: "a table miss must degrade to an uglier
    name, never to a refusal") and confined to this best-effort branch.

    Task 6 (measured 2026-08-11) -- WHY the cascade STOPS at rung 2, and where
    the remaining polyfunctional breadth lives. Two facts were established, not
    assumed (probes in a temp dir/diag_t6*.py):

      1. ``name_general`` is ALL-OR-NOTHING: it returns a complete E1-passing
         result or ``None``, NEVER a partial with a leftover remainder. So the
         "compute the unbound atoms of a partial and attach them" model does not
         apply -- there is no partial to complete. This cascade of rungs already
         IS the universal-decomposition mechanism (senior parent + every
         off-parent fragment as a complete recursive ``name_substituent`` +
         E1), because that is exactly what ``name_general`` does internally.

      2. The ONLY feature-override levers the engine reads are
         ``principal_group`` (+ ``principal_group_atoms``) and
         ``chain_is_parent`` (grep-confirmed: no other toggle gates naming).
         Rungs 0-2 plus a ``chain_is_parent=True`` variant cover the FULL 2x2+
         toggle space, and the ``chain_is_parent=True`` rung converts **0 of
         150** producer-``None`` polyfunctional molecules on a 400-molecule
         pubchem_2000 sample. The feature-override lever is EXHAUSTED; adding a
         further re-parametrisation rung would be dead code (invariants 8, 17).

    So the remaining polyfunctional abstentions are NOT reachable by any feature
    toggle here -- each needs an ENGINE-level capability that is a separate
    follow-on, never a ``t4_coverage`` rung (the brief forbids reimplementing
    ``name_substituent`` / ``_assemble``):

      * an UNCONDITIONAL recursive-substituent base case (longest-path chain +
        every off-path atom as a recursive prefix + skeletal heteroatoms as
        aza/oxa replacement) so a hard branch degrades to an ugly-but-complete
        prefix instead of "branch unnameable" -- the dominant class (~120/150,
        many also charged/out-of-scope);
      * acyl substituents built as ``name(R)`` + ``carbonyl`` / ``amino`` so an
        internal C=C never blocks them (the ``C/C=C/C(=O)NCC(=O)O`` enamide
        "branch unnameable" class);
      * OFFER-not-RETURN parent competition (a project rule): a parent choice that
        leaves an unnameable fragment should lose to a competitor parent rather
        than terminate the molecule;
      * a peptide-residue namer for deep peptide/ester side chains (perindopril's
        ``CCC[C@H](N[C@H](C)C=O)C(=O)OCC`` hits substituent_recursion_depth_exceeded recursion_depth_fallback);
      * charged-parent support (``allow_charged``, ~13/150, mostly out-of-scope
        salts) and spiro/fused/monocycle ring-parent producers (~9/150).

    Whichever of these ships, its output still flows through THIS producer's E1
    gate + unchanged, so 0-wrong / 0-partial is preserved by
    construction. NOTE (namer.py follow-on, symptom VERIFIED / cause a LEAD):
    for ``CC(=O)NCN(C)N=O`` (cid 43057) this producer builds a complete,
    OPSIN-round-tripping replacement name when called cleanly, but a DEGRADED
    atom-dropping name when invoked from ``namer._try_general_engine_recovery``
    after ``_name_impl`` has run -- the recovery path does not run the T4
    producer in a clean naming state (ruled out: the fragment cache and the
    ``best_effort_ctx`` / ``general_fallback_ctx`` contextvars). That is a
    namer.py state-isolation fix, outside this module.
    """
    # Rung 0: the full engine with the perceived principal group (Task 3).
    candidate = _run_general_e1(mol, features)
    if candidate is not None:
        return _prefer_verified_floor(mol, candidate)  #: verify-before-commit

    # Rung 0.5 (tail #21): the acyclic polyol / polyether / POLYESTER class.
    # name_general is a ring/von-Baeyer engine and returns None for an acyclic
    # polyol chain (the composer names a plain polyol, but an ester's
    # seniority makes the composer commit to the ester and LINEARISE the polyol
    # -- a wrong 'tricosyl 2-methylpropanoate' suppresses). This producer
    # supplies the degrade method 2): choose the polyol chain as
    # the parent, cite free -OH as the -ol suffix, and DEMOTE every ester to an
    # (Racyloxy) prefix. It is coverage-complete by construction and, carrying no
    # result_obj, is verified by the downstream round-trip (the 0-wrong
    # net). Fail-closed (None) for every shape outside its tight scope.
    try:
        from ..rules.polyol_polyester import name_acyclic_polyol_polyester
        _pp = name_acyclic_polyol_polyester(mol)
        if _pp:
            return _prefer_verified_floor(mol, _Candidate(name=_pp, result_obj=None))
    except Exception as exc:  # fail-closed: a producer bug keeps the abstention
        logger.info("t4 polyol-polyester producer raised: %s", exc)

    # Cascade: route around the decline. Each rung suppresses the principal
    # group so every FG becomes a detachable prefix; ordered most-specific
    # first. A fixed, bounded list -- no unbounded recursion.
    cascade = (
        # rung 1: ring-first -- demote chain_is_parent so a ring parent is
        # chosen and the acyl-oxy is cited as an...oyloxy/acetyloxy prefix.
        dict(principal_group=None, principal_group_atoms=[],
             chain_is_parent=False),
        # rung 2: keep the perceived parent, PG suppressed (ring-less /
        # chain-preferred esters+amides). NOT redundant with rung 1: measured
        # (final review, a temp dir/probe_rung2_wide.py, 400-molecule
        # pubchem_2000 sample) rung 2 is the SOLE producer for 5/400 molecules
        # where rung 1's chain_is_parent=False forces a ring parent the engine
        # cannot host -- e.g. cid 266765 CC(C)(CC(=O)NC1CCCCC1)CBr ->
        # 4-bromo-1-(cyclohexylamino)-3,3-dimethyl-1-oxobutane. Locked by
        # test_cascade_rung2_is_a_unique_producer.
        dict(principal_group=None, principal_group_atoms=[]),
    )
    for overrides in cascade:
        alt_features = _clone_features_with(features, **overrides)
        if alt_features is None:
            continue
        candidate = _run_general_e1(mol, alt_features)
        if candidate is not None:
            return _prefer_verified_floor(mol, candidate)  #: verify-before-commit

    # Final rung (Phase B4): the UNCONDITIONAL recursive substitutive namer.
    # Every feature-override rung above declined -- the remaining abstentions
    # are the ENGINE-CAPABILITY class the Task-6 note enumerated (a hard branch
    # the recursive substituent namer cannot spell, a ring parent no
    # producer hosts, a charge name_general has no suffix for). This rung is
    # the coverage floor that closes them: `name_universal_substitutive`
    # re-derives the WHOLE molecule from scratch as senior parent + every
    # off-parent atom rendered as a recursive `-yl` (skeletal heteroatoms as
    # aza/oxa replacement), bottoming out at an ugly-but-valid systematic
    # token rather than declining. It is coverage-complete BY CONSTRUCTION
    # (its own atom-coverage assertion) and, like the polyol-polyester rung
    # above, carries NO result_obj -- so `name_t4_complete` ships it straight
    # to the caller's round-trip ladder (namer.py), which is the
    # 0-wrong net: an unverifiable universal name (or a stereo CONFLICT) is
    # suppressed there, a constitution-correct one (or a safe stereo-OMISSION
    # via the `_rt_match` superset gate) ships. It renders no stereo
    # descriptors (constitution-complete only), which is exactly what that
    # superset gate is built to accept as a valid less-specific degrade.
    # Fail-closed: `None` (a genuine residual -- out-of-scope charge, an
    # isotope/radical/wildcard, or the work-budget/size ceiling) keeps the
    # honest abstention below. Reachable ONLY on the best-effort path (this
    # whole module is `_general_fallback`-gated in namer.py), so PIN is
    # untouched.
    try:
        # A2 (C2): a SINGLE free-valence centre that the dedicated
        # radicals.py/route_charged path already declined (this whole rung is
        # reached only after that path's own candidate -- if any -- failed
        # its OPSIN ``-r`` gate check, per this module's header) gets ONE more
        # attempt here, through ``name_universal_substituent_prefix`` rather
        # than ``name_universal_substitutive``. The whole-molecule entry point
        # has NO ``-yl``/``-ylidene``/``-ylidyne`` suffix logic at all, so it
        # can only ever name the FULLY SATURATED neutral constitution --
        # measured: '[CH2]CO' -> '1-oxapropane' (ethanol's skeletal-
        # replacement spelling, wrong formula for the radical) -- which the
        # caller's own -r round-trip gate correctly rejects, net a wasted
        # abstain, never a rescue. The substituent-prefix entry point treats
        # the radical centre as the attachment point of a "-yl" prefix over
        # the WHOLE heavy-atom set instead, so it CAN cite the missing bond --
        # measured: '[CH2]CO' -> '3-oxapropan-1-yl' (-r-RT VERIFIED MATCH).
        # A second radical centre is out of ``_build_ctx``'s scope (see its
        # docstring) and falls through to the whole-molecule rung below,
        # which stays a safe (if pointless) abstain for that shape.
        from ..perception.ions import get_radical_sites
        radical_sites = get_radical_sites(mol)
        if len(radical_sites) == 1 and radical_sites[0]['n_electrons'] in (1, 2, 3):
            from .universal_substituent import name_universal_substituent_prefix
            all_heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
            _prefix_name = name_universal_substituent_prefix(
                mol, all_heavy, radical_sites[0]['atom_idx'],
                bond_order=radical_sites[0]['n_electrons'], cap_attachment=True)
            if _prefix_name:
                return _Candidate(name=_prefix_name, result_obj=None,
                                  final_floor=True)
        else:
            from .universal_substituent import name_universal_substitutive
            _uni = name_universal_substitutive(mol)
            if _uni is not None and _uni.name:
                _uni = _book_checked_floor(mol, _uni)
                _uni = _leaf_checked_floor(mol, _uni)
                # The retry the rungs above get through ``_prefer_verified_floor``:
                # when the floor's first spelling does not even read back as the
                # input's CONSTITUTION (OPSIN cannot parse it, or it parses to a
                # different molecule), no later step can rescue it, so a verified
                # alternative spelling (the von Baeyer form of a fused spiro
                # component, the leaf-free form) is offered instead. A first
                # spelling with the right constitution ships to the caller's
                # ladder exactly as before (its stereo composition may still
                # complete it there).
                if not _reads_back_as_constitution(mol, _uni.name):
                    from rdkit import Chem as _Chem
                    _alt = _verified_universal_floor(
                        mol, _Chem.MolToSmiles(mol), first=_uni)
                    if _alt is not None:
                        return _Candidate(name=_alt, result_obj=None,
                                          final_floor=True)
                return _Candidate(name=_uni.name, result_obj=None, final_floor=True)
    except Exception as exc:  # fail-closed: a producer bug keeps the abstention
        logger.info("t4 universal-substitutive rung raised: %s", exc)

    return None  # every applicable strategy tried -> honest abstain
