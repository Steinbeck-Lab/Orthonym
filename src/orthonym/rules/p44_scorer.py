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
