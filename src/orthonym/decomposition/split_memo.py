"""The verdict of one single-bond split, memoised for the naming scope (``engine._try_single_bond_decompose``).

One naming asks the decomposition for the same split again and again: the charged-route re-entry and every fragment
naming that reaches the same remainder enter it once more (a peptide of the 2026-11-12 time-out audit, PubChem CID
166028644: 143 top-level splits in 40 groups of one molecule, bond and naming state; the repeats cost 79 of 165 s).
Only the assemblers that cannot prove their group word kept their verdict (``engine._unproven_split_key``); an amide
split that the assembler refuses was named again in full each time.

The key is the one of the unproven verdict (the atom-ordered graph, the bond, the style, the ambient breadth flags,
the N-acyl float scope, the peptide re-entrancy flags, the locant and isotope scopes, the fragment-session depth) and
the set of fragments named up the stack (a fragment met again there is a cycle, and the names read it).

A split already found unproven is not served here: its repeat takes the engine's own path, which costs nothing and
fixes the exploration floor of the budget armed now (``engine._note_unproven_split``). Any other hit replays what the
first call left: its provenance (``nested_memo.cached_nested_call``) and the cycle hits it logged
(``fragment_naming._note_cycle_hit``; the fragment named around this split reads them to keep its name in context).
A hit charges the budget units the first call charged (``replay_budgets=True``): the old repeat of a split on the
large molecules this memo is for named its remainder again and charged the analysis budget again, and that charge is
what ends the search there (``engine._exploration_exhausted``). Measured 2026-11-12 (PubChem CID 16396833, 300 s
limit): no charge on a hit 148.8 s, the first call's charge 98.1 s, no memo 92.1 s; the same name. A split found
unproven is the case where the old repeat cost nothing, and it does not reach this memo. A call that raises is not
stored.
``ORTHONYM_MEMO=verify`` recomputes every hit and compares the name and the cycle hits.
"""
from typing import Callable, Dict, Optional


def memoised_split(mol, bond: Dict, style: str, fresh: Callable[[], Optional[str]]) -> Optional[str]:
    """``fresh`` (the split named), once per naming scope for the same split and state."""
    from ..assembly import fragment_naming as _fn
    from ..assembly import memo as _memo
    try:
        from .engine import _unproven_split_key
        split_key = _unproven_split_key(mol, bond, style)
        key = split_key + (frozenset(_fn._get_visited()),)
        hash(key)
    except Exception:  # noqa: BLE001 - a split that cannot be keyed is named afresh
        return fresh()
    if _memo._MODE == "on" and _memo.side_get("unproven_split", split_key):
        return fresh()

    def compute():
        mark = _fn._cycle_hit_mark()
        name = fresh()
        log = getattr(_fn._fragment_guard, 'cycle_hits', None) or ()
        return name, tuple(log[mark:])

    from ..assembly.nested_memo import cached_nested_call
    state = {"fresh": False}

    def counted():
        state["fresh"] = True
        return compute()

    name, hits = cached_nested_call("single_bond_split", key, counted, replay_budgets=True)
    if not state["fresh"]:
        for entry in hits:
            _fn._note_cycle_hit(entry)
    return name
