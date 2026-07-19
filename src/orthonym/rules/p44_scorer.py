"""v25 G1: unified Blue Book P-44 parent-structure scorer.

ONE deterministic comparator over a POOLED ring+chain candidate list,
replacing the staged class-specific ``select_parent`` cascade (the v25 P0
root-cause coverage gap: staged branches + fail-open default-to-ring).

Rule grounding (BlueBookV2/BlueBookV2.md, verified 2026-07-19):
- P-44.1.1  (~18850): max count of principal characteristic group as suffix.
- P-44.1.2  (18917): class order N>P>As>Sb>Bi>Si>Ge>Sn>Pb>B>Al>Ga>In>Tl>
  O>S>Se>Te>C; chooses between rings and chains, NOT among rings/'a'-chains.
- P-44.1.2.2(1) (19340): same class -> ring senior to chain, regardless of
  hydrogenation (heptylbenzene PIN).
- P-44.2.1 (19408): among rings (a) heterocycle (b) has-N (c) earliest
  heteroatom F>Cl>Br>I>O>S>Se>Te>P>... (d) more rings (e) more skeletal
  atoms (f) more heteroatoms (g) more of the earliest heteroatom.
- P-44.3 (20926): among chains (a) more skeletal heteroatoms (b) more
  skeletal atoms (c) more of the senior heteroatom O>S>Se>Te>N>P>...
- P-44.4.1 (21016): shared tiebreaks (a) multiple bonds (b) double bonds
  (h) lower suffix locants (j) lower ene/yne locants; mancude rings count
  as noncumulative double bonds.
- P-51.4: chain skeletal-replacement nomenclature admitted at >= 4
  bridging hetero units (candidate ADMISSION gate, not a comparison rule).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# P-44.1.2 (BB line 18917) — decides between rings and chains at class level.
P44_CLASS_ORDER = ['N', 'P', 'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb',
                   'B', 'Al', 'Ga', 'In', 'Tl', 'O', 'S', 'Se', 'Te', 'C']
P44_CLASS_RANK = {s: len(P44_CLASS_ORDER) - i for i, s in enumerate(P44_CLASS_ORDER)}

# P-44.2.1(c) (BB line 19455) — ring heteroatom seniority.
RING_HETERO_ORDER = ['F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'P', 'As',
                     'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga', 'In', 'Tl']
RING_HETERO_RANK = {s: len(RING_HETERO_ORDER) - i for i, s in enumerate(RING_HETERO_ORDER)}

# P-44.3.3 (BB line 21002) — chain heteroatom seniority (O first!).
CHAIN_HETERO_ORDER = ['O', 'S', 'Se', 'Te', 'N', 'P', 'As', 'Sb', 'Bi',
                      'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga', 'In', 'Tl']
CHAIN_HETERO_RANK = {s: len(CHAIN_HETERO_ORDER) - i for i, s in enumerate(CHAIN_HETERO_ORDER)}


@dataclass(frozen=True)
class ParentCandidate:
    """One pooled parent-structure candidate.

    kind: 'ring' | 'chain'.
    atoms: ring -> sorted ring-system atom indices; chain -> chain order.
    pg_count: P-44.1.1 count of principal-characteristic-group instances
        expressible as suffix on THIS candidate.
    """
    kind: str
    atoms: Tuple[int, ...]
    pg_count: int


def pool_candidates(mol, ring_systems, principal_chain,
                    principal_group, principal_group_atoms) -> List[ParentCandidate]:
    """Pool ALL admissible ring + chain parent candidates.

    Admission gates (these are NOT seniority rules — they decide which
    skeletons are legal parents at all):
    - every connected ring system is a candidate;
    - the carbon principal chain (if any) is a candidate;
    - a longer hetero skeletal chain is admitted iff EITHER it bears the
      principal group and has a bridging heteroatom (legacy P-44.3(b) gate)
      OR there is no principal group and it has >= 4 bridging hetero units
      (P-51.4 replacement-nomenclature admission).
    """
    from .parent_selection import (
        _count_pg_on_ring, _count_pg_on_chain,
        is_principal_group_on_chain,
        _has_bridging_heteroatom, _count_bridging_heteroatoms,
    )
    from ..perception.chains import find_longest_skeletal_chain
    from ..perception.functional_groups import get_chain_excluded_atoms

    pool: List[ParentCandidate] = []
    all_ring_atoms = set()
    for rs in ring_systems:
        all_ring_atoms.update(rs)

    for rs in ring_systems:
        pg_n = _count_pg_on_ring(mol, set(rs), principal_group_atoms or [],
                                 principal_group)
        pool.append(ParentCandidate('ring', tuple(sorted(rs)), pg_n))

    chains: List[Tuple[int, ...]] = []
    if principal_chain:
        chains.append(tuple(principal_chain))

    sk = find_longest_skeletal_chain(
        mol, exclude_atoms=all_ring_atoms | get_chain_excluded_atoms(mol))
    if sk and len(sk) > len(principal_chain or ()):
        if principal_group and principal_group_atoms:
            if (is_principal_group_on_chain(mol, sk, principal_group_atoms,
                                            principal_group)
                    and _has_bridging_heteroatom(mol, sk)):
                chains.append(tuple(sk))
        elif _count_bridging_heteroatoms(mol, sk) >= 4:
            chains.append(tuple(sk))

    for ch in chains:
        pg_n = _count_pg_on_chain(mol, list(ch), principal_group_atoms or [],
                                  principal_group)
        pool.append(ParentCandidate('chain', ch, pg_n))
    return pool


def _class_rank(mol, cand: ParentCandidate) -> int:
    """P-44.1.2: rank of the most senior skeletal element in the candidate."""
    return max((P44_CLASS_RANK.get(mol.GetAtomWithIdx(i).GetSymbol(), 0)
                for i in cand.atoms), default=0)


def _n_multiple_bonds(mol, atom_set) -> Tuple[int, int]:
    """(multiple, double) bond counts within the candidate skeleton.

    P-44.4.1.1: mancude (aromatic) bonds count as noncumulative double
    bonds -- GetBondTypeAsDouble() returns 1.5 for aromatic, > 1.0 counts.
    """
    n_mult = n_dbl = 0
    for b in mol.GetBonds():
        if b.GetBeginAtomIdx() in atom_set and b.GetEndAtomIdx() in atom_set:
            v = b.GetBondTypeAsDouble()
            if v > 1.0:
                n_mult += 1
            if 1.5 <= v <= 2.0:
                n_dbl += 1
    return n_mult, n_dbl


def _candidate_locants(mol, cand: ParentCandidate, target_atoms,
                       ring_info=None):
    """Lowest-locant set for `target_atoms` on this candidate.

    Chain: try both directions, keep the lower set (P-14.4 first point of
    difference). Ring: use IUPAC locants from ring_info when available for
    THIS ring system, else the deterministic sorted-position fallback
    (`_build_ring_pos` -- same fallback the legacy cascade used).
    """
    from .locants import compare_locant_sets
    from .parent_selection import _build_ring_pos

    targets = set(target_atoms) & set(cand.atoms)
    if not targets:
        return []
    if cand.kind == 'chain':
        fwd = {a: i + 1 for i, a in enumerate(cand.atoms)}
        rev = {a: i + 1 for i, a in enumerate(reversed(cand.atoms))}
        lf = sorted(fwd[a] for a in targets)
        lr = sorted(rev[a] for a in targets)
        return lf if compare_locant_sets(lf, lr) <= 0 else lr
    pos = None
    if ring_info and ring_info.get('iupac_locants'):
        iupac = ring_info['iupac_locants']
        if all(a in iupac for a in cand.atoms):
            pos = iupac
    if pos is None:
        pos = _build_ring_pos(set(cand.atoms), ring_info=ring_info)
    return sorted(pos[a] for a in targets if a in pos)


def _substituent_positions(mol, cand: ParentCandidate):
    """Atoms of the candidate bearing >=1 external heavy substituent."""
    cset = set(cand.atoms)
    out = []
    for i in cand.atoms:
        for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
            if nbr.GetIdx() not in cset and nbr.GetAtomicNum() > 1:
                out.append(i)
                break
    return out


def _ring_subrings(mol, atom_set) -> int:
    ri = mol.GetRingInfo()
    return sum(1 for ring in ri.AtomRings() if set(ring) <= atom_set)


def _hetero_rank_vector(mol, atoms, rank_table):
    """Descending rank vector of hetero skeletal atoms, for lexicographic
    'more of the most senior heteroatom' comparison."""
    ranks = sorted((rank_table.get(mol.GetAtomWithIdx(i).GetSymbol(), 0)
                    for i in atoms
                    if mol.GetAtomWithIdx(i).GetSymbol() != 'C'),
                   reverse=True)
    return tuple(ranks)


def _cmp(x, y) -> int:
    return (x > y) - (x < y)


def compare_parent_candidates(mol, a: ParentCandidate, b: ParentCandidate, *,
                              principal_group=None,
                              principal_group_atoms=None,
                              ring_info=None) -> int:
    """Blue Book P-44 seniority: >0 a senior, <0 b senior, 0 tie."""
    from .locants import compare_locant_sets
    from .parent_selection import _pg_attachment_atoms

    # P-44.1.1 -- more principal characteristic groups as suffix.
    if a.pg_count != b.pg_count:
        return _cmp(a.pg_count, b.pg_count)

    # P-44.1.2 -- senior skeletal atom class (ring-vs-chain / class level).
    r = _cmp(_class_rank(mol, a), _class_rank(mol, b))
    if r:
        return r

    # P-44.1.2.2(1) -- same class: ring senior to chain.
    if a.kind != b.kind:
        return 1 if a.kind == 'ring' else -1

    if a.kind == 'ring':
        # P-44.2.1 (a)..(g)
        sa = {mol.GetAtomWithIdx(i).GetSymbol() for i in a.atoms}
        sb = {mol.GetAtomWithIdx(i).GetSymbol() for i in b.atoms}
        r = _cmp(sa != {'C'}, sb != {'C'})                              # (a)
        if r:
            return r
        r = _cmp('N' in sa, 'N' in sb)                                  # (b)
        if r:
            return r
        ra = max((RING_HETERO_RANK.get(s, 0) for s in sa if s != 'C'), default=0)
        rb = max((RING_HETERO_RANK.get(s, 0) for s in sb if s != 'C'), default=0)
        r = _cmp(ra, rb)                                                # (c)
        if r:
            return r
        r = _cmp(_ring_subrings(mol, set(a.atoms)),
                 _ring_subrings(mol, set(b.atoms)))                     # (d)
        if r:
            return r
        r = _cmp(len(a.atoms), len(b.atoms))                            # (e)
        if r:
            return r
        ha = _hetero_rank_vector(mol, a.atoms, RING_HETERO_RANK)
        hb = _hetero_rank_vector(mol, b.atoms, RING_HETERO_RANK)
        r = _cmp(len(ha), len(hb))                                      # (f)
        if r:
            return r
        r = _cmp(ha, hb)                                                # (g)
        if r:
            return r
    else:
        # P-44.3 (a)..(c)
        ha = _hetero_rank_vector(mol, a.atoms, CHAIN_HETERO_RANK)
        hb = _hetero_rank_vector(mol, b.atoms, CHAIN_HETERO_RANK)
        r = _cmp(len(ha), len(hb))                                      # (a)
        if r:
            return r
        r = _cmp(len(a.atoms), len(b.atoms))                            # (b)
        if r:
            return r
        r = _cmp(ha, hb)                                                # (c)
        if r:
            return r

    # P-44.4.1 shared tiebreaks.
    ma, da = _n_multiple_bonds(mol, set(a.atoms))
    mb, db = _n_multiple_bonds(mol, set(b.atoms))
    r = _cmp(ma, mb)                                                    # (a)
    if r:
        return r
    r = _cmp(da, db)                                                    # (b)
    if r:
        return r

    # (h) lower locants for the suffix (PCG attachment atoms).
    if principal_group_atoms:
        att = set()
        for match in principal_group_atoms:
            att.update(_pg_attachment_atoms(principal_group, match))
        la = _candidate_locants(mol, a, att, ring_info=ring_info)
        lb = _candidate_locants(mol, b, att, ring_info=ring_info)
        if la or lb:
            r = -compare_locant_sets(la, lb)  # -1 == first-arg preferred
            if r:
                return r

    # (j) lower locants for unsaturation.
    def _unsat_atoms(cand):
        cset = set(cand.atoms)
        out = set()
        for bnd in mol.GetBonds():
            if (bnd.GetBeginAtomIdx() in cset and bnd.GetEndAtomIdx() in cset
                    and bnd.GetBondTypeAsDouble() > 1.0):
                out.update((bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()))
        return out

    la = _candidate_locants(mol, a, _unsat_atoms(a), ring_info=ring_info)
    lb = _candidate_locants(mol, b, _unsat_atoms(b), ring_info=ring_info)
    if la or lb:
        r = -compare_locant_sets(la, lb)
        if r:
            return r

    # Legacy-preserved final criteria (P-44.1(h)/(i) of the staged cascade):
    # more substituents, then lower substituent locants.
    pa = _substituent_positions(mol, a)
    pb = _substituent_positions(mol, b)
    r = _cmp(len(pa), len(pb))
    if r:
        return r
    la = _candidate_locants(mol, a, set(pa), ring_info=ring_info)
    lb = _candidate_locants(mol, b, set(pb), ring_info=ring_info)
    if la or lb:
        r = -compare_locant_sets(la, lb)
        if r:
            return r

    # Deterministic total order (never input-order-dependent).
    return _cmp(tuple(sorted(b.atoms)), tuple(sorted(a.atoms)))
