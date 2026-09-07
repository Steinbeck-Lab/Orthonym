"""Best-effort composition tier (v30 slice 1).

When a producer builds an atom-short parent name on the best-effort tier, this module finishes
it: it computes which heavy atoms the parent actually accounts for, isolates the uncovered
remainder, and (in a later step) names that remainder as substituent prefixes and composes a
whole-molecule name.

Design + provenance: ``docs/superpowers/specs/2026-08-05-best-effort-composition-tier-design.md``
and ``docs/superpowers/plans/2026-08-05-best-effort-composition-slice1.md``. The covered-atom
source was settled by the Task-1/2 spy: ``features.principal_chain`` ALONE undercounts (it misses
the ``-ol`` oxygen of ``propan-1-ol`` → 3/4), so coverage must also include the principal-group
atoms. Every function here is pure (no OPSIN, no I/O).
"""

from __future__ import annotations

from typing import Any, List, Set, Tuple


def parent_covered_atoms(mol, features: Any) -> Set[int]:
    """Heavy-atom indices the parent name accounts for.

    Coverage = the principal chain plus every heavy atom of the principal characteristic group
    (the suffix). ``principal_group_atoms`` is a list of tuples, each tuple holding all atom
    indices of one group occurrence — e.g. carboxylic acid ``[(2, 3, 4)]`` = C,O,O, alcohol
    ``[(3, 2)]`` = O plus its attachment carbon. Both shapes are handled by taking every integer
    index in every tuple. The attachment carbon is often already in the chain; the set union
    de-duplicates it.

    ⚠ This is what the parent NAME covers, not what perception saw — a dropped substituent is
    perceived (it lives in ``features.substituents``) but is NOT in this set, which is exactly
    how :func:`uncovered_fragments` finds the remainder.
    """
    n = mol.GetNumHeavyAtoms()
    covered: Set[int] = set()

    for idx in getattr(features, "principal_chain", None) or ():
        if isinstance(idx, int):
            covered.add(idx)

    for group in getattr(features, "principal_group_atoms", None) or ():
        for idx in group:
            if isinstance(idx, int) and 0 <= idx < mol.GetNumAtoms() \
                    and mol.GetAtomWithIdx(idx).GetAtomicNum() > 1:
                covered.add(idx)

    # never claim a hydrogen or an out-of-range index
    return {i for i in covered if 0 <= i < mol.GetNumAtoms()
            and mol.GetAtomWithIdx(i).GetAtomicNum() > 1}


def _heavy_atoms(mol) -> Set[int]:
    return {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}


def uncovered_fragments(mol, covered: Set[int]) -> List[Tuple[int, ...]]:
    """Heavy atoms not in ``covered``, split into connected components (by heavy-atom bonds).

    Each returned tuple is one connected uncovered fragment — a candidate substituent whose
    atoms the parent dropped. Order within a tuple is ascending atom index; the list is ordered
    by each fragment's lowest atom index, so the result is deterministic.
    """
    uncovered = _heavy_atoms(mol) - set(covered)
    if not uncovered:
        return []

    seen: Set[int] = set()
    frags: List[Tuple[int, ...]] = []
    for start in sorted(uncovered):
        if start in seen:
            continue
        stack = [start]
        comp: List[int] = []
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            comp.append(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                j = nb.GetIdx()
                if j in uncovered and j not in seen:
                    stack.append(j)
        frags.append(tuple(sorted(comp)))
    return frags


def fragment_attachment_atom(mol, fragment: Tuple[int, ...]) -> int:
    """The atom inside ``fragment`` bonded to an atom outside it (the free-valence / attach atom).

    ``name_substituent`` expects a fragment-side attachment index. When a fragment has several
    external bonds (rare for a dropped substituent), the lowest such atom index is chosen for
    determinism; when it has none (a disconnected component — should not happen for a real
    substituent), the lowest atom index is returned.
    """
    fset = set(fragment)
    for idx in fragment:  # ascending
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() not in fset:
                return idx
    return fragment[0]
