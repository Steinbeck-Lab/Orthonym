"""The longest carbon path from one atom, found by a branch and bound.

``rules.ions._find_carboxylate_chain`` walks the carbons from a carboxyl carbon. The plain DFS lists every simple path,
which grows exponentially on a carbon cage (40 s on a fullerene adduct). Here a branch is cut when the carbons it can
still reach through unused atoms cannot make a strictly longer path. A path of equal length found later never replaced
the first one, so the cut never changes the path returned (speed audit 2026-11-24, lever K7).
"""
from typing import Collection, List


def first_longest_carbon_path(mol, start: int, excluded: Collection[int]) -> List[int]:
    """The longest simple path of carbon atoms from ``start`` that avoids ``excluded``; of several, the first in DFS
    order (neighbours in RDKit order), as a list of atom indices from ``start``."""
    steps = [[n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
              if n.GetSymbol() == 'C' and n.GetIdx() not in excluded] for i in range(mol.GetNumAtoms())]
    best = [[start]]
    path = [start]
    visited = {start}

    def reach(v):
        seen = {v}
        stack = [v]
        while stack:
            for w in steps[stack.pop()]:
                if w not in visited and w not in seen:
                    seen.add(w)
                    stack.append(w)
        return len(seen)

    def walk(v):
        if len(path) > len(best[0]):
            best[0] = list(path)
        for w in steps[v]:
            if w in visited or len(path) + reach(w) <= len(best[0]):
                continue
            visited.add(w)
            path.append(w)
            walk(w)
            path.pop()
            visited.discard(w)

    walk(start)
    return best[0]
