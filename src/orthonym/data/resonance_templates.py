"""
Resonance-shifted (bond_orders, charges) chain templates for terminal
nitrogen chains: azido, diazo, diazonium (a phase).

RDKit does NOT normalise resonance forms on ``MolFromSmiles`` -- each drawing
keeps its own literal bond orders / formal charges. Two independently-valid
resonance drawings exist for each of these three //
classes; a single closed 3-set of six ``(bond_orders, charges)`` vectors over
the ordered attach->terminal nitrogen chain distinguishes all of them
(a phase a trace, `internal notes` Q3):

    class | drawing | bond_orders | charges
    -----------|---------------------------------|-------------|------------------
    azido | R-N=[N+]=[N-] (canonical) | (1, 2, 2) | (0, 0, +1, -1)
    azido | R-[N-]-[N+]#N (charge-sep. alt) | (1, 1, 3) | (0, -1, +1, 0)
    diazonium | R-N+#N (canonical) | (1, 3) | (0, +1, 0)
    diazonium | R-N=N+ (charge-on-terminal alt) | (1, 2) | (0, 0, +1)
    diazo | R2C=[N+]=[N-] (canonical) | (2, 2) | (0, +1, -1)
    diazo | R2[C-]-[N+]#N (carbanion alt) | (1, 3) | (-1, +1, 0)

Resonance-template table (independent implementation) --
a per-cell shape: ``key`` / ``bond_orders`` / ``charges`` as
per-cell ``None`` wildcard / ``frozenset`` alternatives / exact ``int``, with
first-match priority. This table has no wildcard cells today: the six rows
are pairwise distinct as exact tuples (chain length alone separates azido's
4-atom chain from diazonium/diazo's 3-atom chain; diazonium vs diazo then
separate on the attach-bond order, 1 vs 2).

The chain is always ``(attach_atom, n1, n2, (n3))`` -- ``attach_atom`` is the
first NON-nitrogen atom reached walking outward from the charged nitrogen
group, and is a structural requirement of the walk (never a wildcarded
cell): a free azide/diazonium anion, with no organic attachment, has no
non-nitrogen atom to seed a walk from and can therefore never match (Q3
nuance) -- ``[N-]=[N+]=[N-]`` alone never enters ``find_resonance_chains``.
"""

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Set, Tuple, Union

from rdkit import Chem

Cell = Union[int, FrozenSet[int], None]


@dataclass(frozen=True)
class ResonanceChainTemplate:
    """One row of the closed 3-set. ``matches`` is a per-cell AND across
    ``bond_orders`` and ``charges``: ``None`` wildcards a cell, a
    ``frozenset`` accepts any listed alternative, an ``int`` requires an
    exact match. First-match wins in table order (a defined-order
    match) -- today every cell is an exact ``int`` so no two
    rows can double-match the same vector, but the priority contract is kept
    for any future wildcarded row."""

    key: str
    bond_orders: Tuple[Cell, ...]
    charges: Tuple[Cell, ...]

    def matches(self, bond_orders: Tuple[int, ...], charges: Tuple[int, ...]) -> bool:
        if len(bond_orders) != len(self.bond_orders):
            return False
        if len(charges) != len(self.charges):
            return False
        return (_row_matches(self.bond_orders, bond_orders)
                and _row_matches(self.charges, charges))


def _cell_matches(wanted: Cell, got: int) -> bool:
    if wanted is None:
        return True
    if isinstance(wanted, frozenset):
        return got in wanted
    return wanted == got


def _row_matches(wanted: Tuple[Cell, ...], got: Tuple[int, ...]) -> bool:
    return all(_cell_matches(w, g) for w, g in zip(wanted, got))


# Closed 3-set -- a phase a trace Q3. Chain order is (attach, n1, n2, (n3)).
RESONANCE_CHAIN_TEMPLATES: Tuple[ResonanceChainTemplate, ...] = (
    # azido R-N=[N+]=[N-] (canonical) / R-[N-]-[N+]#N (charge-separated alt)
    ResonanceChainTemplate('azido', (1, 2, 2), (0, 0, 1, -1)),
    ResonanceChainTemplate('azido', (1, 1, 3), (0, -1, 1, 0)),
    # diazonium R-N+#N (canonical) / R-N=N+ (charge-on-terminal-N alt)
    ResonanceChainTemplate('diazonium', (1, 3), (0, 1, 0)),
    ResonanceChainTemplate('diazonium', (1, 2), (0, 0, 1)),
    # diazo R2C=[N+]=[N-] (canonical) / R2[C-]-[N+]#N (carbanion-diazonium-ylide alt)
    ResonanceChainTemplate('diazo', (2, 2), (0, 1, -1)),
    ResonanceChainTemplate('diazo', (1, 3), (-1, 1, 0)),
)


def classify_resonance_chain(bond_orders: Tuple[int, ...],
                              charges: Tuple[int, ...]) -> Optional[str]:
    """First-match lookup: ``(bond_orders, charges)`` -> ``'azido'`` /
    ``'diazo'`` / ``'diazonium'``, or ``None`` if the vector matches none of
    the six rows -- fail closed: an ordinary amine / nitro / hydrazine /
    hydrazinium chain returns ``None``, never a guess."""
    for template in RESONANCE_CHAIN_TEMPLATES:
        if template.matches(bond_orders, charges):
            return template.key
    return None


_BOND_ORDER = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
}


def chain_vector(mol, chain: Tuple[int, ...]
                  ) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """``(bond_orders, charges)`` for an ordered atom-index chain. ``None``
    on any bond type the closed set never uses (aromatic / dative /
    quadruple), or if consecutive atoms in ``chain`` are not bonded."""
    bond_orders: List[int] = []
    for a, b in zip(chain, chain[1:]):
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            return None
        order = _BOND_ORDER.get(bond.GetBondType())
        if order is None:
            return None
        bond_orders.append(order)
    charges = tuple(mol.GetAtomWithIdx(idx).GetFormalCharge() for idx in chain)
    return tuple(bond_orders), charges


def _walk_n_chain(mol, attach_idx: int, first_n_idx: int,
                   max_n: int = 3) -> Optional[Tuple[int, ...]]:
    """Ordered ``(attach_idx, n1, n2,...)`` linear chain of consecutive
    nitrogen atoms starting at ``first_n_idx`` (bonded to ``attach_idx``),
    stopping at the first atom with no further heavy neighbour (the true
    terminal atom of the group).

    Returns ``None`` if the chain branches (more than one forward heavy
    neighbour -- a ring, or a genuine polyfunctional N this closed set does
    not cover), re-enters a non-nitrogen atom (e.g. an azo R-N=N-R', which
    never terminates in nitrogen), or exceeds ``max_n`` nitrogens (the
    closed set's longest row has 3)."""
    chain = [attach_idx, first_n_idx]
    prev, cur = attach_idx, first_n_idx
    while True:
        atom = mol.GetAtomWithIdx(cur)
        forward = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetIdx() != prev]
        if not forward:
            return tuple(chain)
        if len(forward) > 1:
            return None
        nxt = forward[0]
        if mol.GetAtomWithIdx(nxt).GetSymbol() != 'N':
            return None
        if len(chain) - 1 >= max_n:
            return None
        chain.append(nxt)
        prev, cur = cur, nxt


import weakref as _weakref

from ..perception.molcache import atoms_of  # audit 2026-09-03 (S2): per-call atom/bond tuples

# Per-mol memoization (perf 2026-08-28): ~130 calls/mol measured. Pure function of
# the molecule (reads only bond orders / charges / connectivity). Keyed by the RDKit
# Mol; returns a COPY of the cached list so callers can't corrupt the cache.
_RES_CACHE = _weakref.WeakKeyDictionary()


def find_resonance_chains(mol) -> List[Tuple[str, Tuple[int, ...]]]:
    cached = _RES_CACHE.get(mol)
    if cached is None:
        cached = _find_resonance_chains_impl(mol)
        try:
            _RES_CACHE[mol] = cached
        except TypeError:
            return list(cached)
    return list(cached)


def _find_resonance_chains_impl(mol) -> List[Tuple[str, Tuple[int, ...]]]:
    """Scan ``mol`` for every linear ``(non-N attach) -> N ->...`` chain and
    classify it against the closed 3-set.

    Returns a list of ``(class_key, chain)`` where ``chain`` is
    ``(attach_idx, n1, n2, (n3))``, attach atom first. The attach atom is
    REQUIRED to be non-nitrogen (never wildcarded), so a free azide /
    diazonium anion with no organic attachment never seeds a walk and can
    never match. Does not deduplicate against a caller's own SMARTS matches
    -- callers combine results and drop overlaps themselves."""
    results: List[Tuple[str, Tuple[int, ...]]] = []
    seen: Set[Tuple[int, int]] = set()
    for atom in atoms_of(mol):
        if atom.GetSymbol() == 'N':
            continue
        attach_idx = atom.GetIdx()
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() != 'N':
                continue
            seed = (attach_idx, nbr.GetIdx())
            if seed in seen:
                continue
            seen.add(seed)
            chain = _walk_n_chain(mol, attach_idx, nbr.GetIdx())
            if chain is None:
                continue
            vector = chain_vector(mol, chain)
            if vector is None:
                continue
            cls = classify_resonance_chain(*vector)
            if cls is not None:
                results.append((cls, chain))
    return results
