"""The von Baeyer cage analysis, memoised for one top-level naming call with replay of its side effects.

``vonbaeyer_universal.analyze_cage_universal`` is asked the same cage many times in one naming: each producer, each
tier of the cascade and each sort key of a ring prefix ask again, and each ask runs the main-ring search and the
numbering choice from scratch (speed audit 2026-11-24, lever K1: 8,382 of 10,478 calls on the typical panel were
repeats; 7.4% less CPU on its 374 rows in time, names unchanged).

A hit must leave the naming as the computation would have left it, so the memo stores, beside the result:

* the perf and analysis budget units the computation charged; a hit charges them again (a hit with less room left
  raises the same ``PerfBudgetExceeded`` the computation would have raised);
* the ``record_non_pin_fragment`` calls it made (``metrics.provenance.push_non_pin_log``); a hit makes them again,
  so a name built on a cage descriptor beyond the PIN caps is still labelled below the PIN;
* the abstention it recorded and whether it reached the N-acyl float gate, as ``assembly.memo.cache_or_compute``
  keeps them.

A computation is stored only when those are its only side effects: it wrote no other provenance variable, read none,
and charged no unit of the work budget. Otherwise it is recomputed on every ask, as before. The key holds the molecule
(its binary form and its CIP labels, as the cage numbering reads them), the arguments, the cage ceilings, every
context value the nested sort-key namings read (``substituent_enumerator.name_substituent_for_ordering``: the four
breadth flags, the fragment session depth, ``allow_mancude`` of the general engine and the renaming state) and the run
tags of ``assembly.memo._ck`` (book or mechanical spellings and the other run scopes). A computation with no budget
armed is not stored either. An RWMol
(mutable) is never memoised. Every value handed out is a deep copy, so a caller that edits its result cannot change
the stored one. ``ORTHONYM_MEMO=verify`` recomputes every ask and raises ``MemoMismatch`` on a different result or a
different record log.
"""
import copy
import functools

from rdkit import Chem

_NS = "vonbaeyer_universal.analyze_cage_universal"
_REPLAYABLE = frozenset({"non_pin_fragments"})


def _key(mol, cage_atoms, allow_mancude, spiro_atom, substituent_atoms):
    from ..assembly.engine_prefix_ordering import engine_orders_with_mancude
    from ..assembly.fragment_naming import _session_depth
    from ..assembly.memo import pin_promotion_var
    from ..assembly.nested_memo import renaming_state
    from ..metrics import provenance as pv
    from .vonbaeyer_universal import cage_caps, lifted_ceilings_active
    return (mol.ToBinary(),
            tuple(a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else "" for a in mol.GetAtoms()),
            tuple(b.GetProp("_CIPCode") if b.HasProp("_CIPCode") else "" for b in mol.GetBonds()),
            None if cage_atoms is None else tuple(sorted(cage_atoms)), bool(allow_mancude), spiro_atom,
            None if substituent_atoms is None else tuple(sorted(substituent_atoms)),
            cage_caps(), lifted_ceilings_active(), pin_promotion_var.get(),
            pv.best_effort_ctx.get(), pv.general_fallback_ctx.get(), pv.allow_aromatic_general_ctx.get(),
            pv.full_coverage_ctx.get(), _session_depth(), engine_orders_with_mancude(), renaming_state())


def _budgets():
    from ..assembly.fragment_naming import _fragment_guard as g
    return (getattr(g, "perf_budget", None), getattr(g, "analysis_budget", None), getattr(g, "work_budget", None))


def _spent(before, after):
    """Units charged between two budget readings (0 where no budget is armed)."""
    return before - after if before is not None and after is not None and before > after else 0


def memoised_cage_analysis(analyze):
    """Decorate ``analyze_cage_universal`` with the memo described in the module docstring."""
    @functools.wraps(analyze)
    def wrapper(mol, cage_atoms=None, allow_mancude=False, spiro_atom=None, substituent_atoms=None):
        from ..assembly import fragment_naming as fn
        from ..assembly import memo
        from ..metrics import provenance as pv
        from ..metrics.abstention import peek_abstention, restore_abstention

        def plain():
            return analyze(mol, cage_atoms, allow_mancude, spiro_atom, substituent_atoms)

        cache = memo._cache_var.get()
        if memo._MODE == "off" or cache is None or mol is None or isinstance(mol, Chem.RWMol) or fn._PERF_MEASURE:
            return plain()
        try:
            # ``memo._ck`` adds the run tags every other memo carries (book or mechanical spellings, kept and dropped
            # forms, wider tier, the adduct and isotope scopes): the nested sort-key names, and with them the
            # (g) tie of the numbering, differ between those runs
            ck = memo._ck(_NS, _key(mol, cage_atoms, allow_mancude, spiro_atom, substituent_atoms))
        except Exception:  # noqa: BLE001 -- no key, no memo
            return plain()
        if memo._MODE == "on":
            hit = cache.get(ck)
            if hit is not None:
                result, records, perf_units, analysis_units = hit
                if analysis_units:
                    fn.spend_analysis_call(analysis_units)
                if perf_units:
                    fn.spend_perf_work(perf_units)
                pv.replay_non_pin_log(records)
                record = cache.get((memo._ABSTENTION_KEY, ck), memo._NO_RECORD)
                if record is not memo._NO_RECORD:
                    restore_abstention(record)
                if (memo._GATE_KEY, ck) in cache:
                    memo.note_nacyl_gate()
                return copy.deepcopy(result)
        before = _budgets()
        gate_mark = memo.nacyl_gate_mark()
        abstention = peek_abstention()
        pv.push_touched_log()
        pv.push_non_pin_log()
        try:
            result = plain()
        finally:
            log = pv.pop_non_pin_log()
            touched = pv.pop_touched_log()
        after = _budgets()
        records = pv.compact_non_pin_log(log)
        if (records is None or not touched <= _REPLAYABLE or before[2] != after[2]
                or before[0] is None or before[1] is None):
            return result               # (no budget armed: the units a hit must charge are unknown)
        stored = (copy.deepcopy(result), records, _spent(before[0], after[0]), _spent(before[1], after[1]))
        prev = cache.get(ck)
        if memo._MODE == "verify" and prev is not None and (prev[0], prev[1]) != (stored[0], stored[1]):
            memo._VERIFY_MISMATCHES.append((_NS, ck))
            raise memo.MemoMismatch(_NS, ck, prev[:2], stored[:2])
        cache[ck] = stored
        if memo.nacyl_gate_mark() != gate_mark:
            cache[(memo._GATE_KEY, ck)] = True
        if peek_abstention() is not abstention:
            cache[(memo._ABSTENTION_KEY, ck)] = peek_abstention()
        return result
    return wrapper
