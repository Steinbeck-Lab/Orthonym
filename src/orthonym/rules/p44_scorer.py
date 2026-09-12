""": unified Blue Book parent-structure scorer.

ONE deterministic comparator over a POOLED ring+chain candidate list,
replacing the staged class-specific ``select_parent`` cascade (the
root-cause coverage gap: staged branches + fail-open default-to-ring).

Rule grounding (the Blue Book Blue Book, verified 2026-07-19):
- (~18850): max count of principal characteristic group as suffix.
- (18917): class order N>P>As>Sb>Bi>Si>Ge>Sn>Pb>B>Al>Ga>In>Tl>
  O>S>Se>Te>C; chooses between rings and chains, NOT among rings/'a'-chains.
- (1) (19340): same class -> ring senior to chain, regardless of
  hydrogenation (heptylbenzene PIN) or chain length (BB 35011:
  2-(7-oxoheptyl)cyclopentane-1-carbaldehyde PIN, ring 5 beats chain 7 on
  a PG tie).
- (19408): among rings (a) heterocycle (b) has-N (c) earliest
  heteroatom F>Cl>Br>I>O>S>Se>Te>P>... (d) more rings (e) more skeletal
  atoms (f) more heteroatoms (g) more of the earliest heteroatom.
- (20926): among chains (a) more skeletal heteroatoms (b) more
  skeletal atoms (c) more of the senior heteroatom O>S>Se>Te>N>P>...
- (21016): shared tiebreaks (a) multiple bonds (b) double bonds
  (h) lower suffix locants (j) lower ene/yne locants; mancude rings count
  as noncumulative double bonds.
-: chain skeletal-replacement nomenclature admitted at >= 4
  bridging hetero units (candidate ADMISSION gate, not a comparison rule).

Expressibility principle (class rank + hetero criteria): a chain
heteroatom only counts as SKELETON when the name can express it as such --
a pure non-carbon hydride chain (silane, phosphane, hydrazine: or a
 replacement-admissible mixed chain. Below that, a mixed C/het chain
is a carbon parent with functionalized bridges (a phase semantics) and
ranks as carbon. Principal-group match atoms are suffix, never skeleton.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ..perception.molcache import bonds_of

logger = logging.getLogger(__name__)

# (the Blue Book) - decides between rings and chains at class level.
P44_CLASS_ORDER = ['N', 'P', 'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb',
                   'B', 'Al', 'Ga', 'In', 'Tl', 'O', 'S', 'Se', 'Te', 'C']
P44_CLASS_RANK = {s: len(P44_CLASS_ORDER) - i for i, s in enumerate(P44_CLASS_ORDER)}

# (c) (the Blue Book) - ring heteroatom seniority.
RING_HETERO_ORDER = ['F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'P', 'As',
                     'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga', 'In', 'Tl']
RING_HETERO_RANK = {s: len(RING_HETERO_ORDER) - i for i, s in enumerate(RING_HETERO_ORDER)}

# (the Blue Book) - chain heteroatom seniority (O first!).
CHAIN_HETERO_ORDER = ['O', 'S', 'Se', 'Te', 'N', 'P', 'As', 'Sb', 'Bi',
                      'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga', 'In', 'Tl']
CHAIN_HETERO_RANK = {s: len(CHAIN_HETERO_ORDER) - i for i, s in enumerate(CHAIN_HETERO_ORDER)}


@dataclass(frozen=True)
class ParentCandidate:
    """One pooled parent-structure candidate.

    kind: 'ring' | 'chain'.
    atoms: ring -> sorted ring-system atom indices; chain -> chain order.
    pg_count: count of principal-characteristic-group instances
        expressible as suffix on THIS candidate.
    """
    kind: str
    atoms: Tuple[int, ...]
    pg_count: int


def pool_candidates(mol, ring_systems, principal_chain,
                    principal_group, principal_group_atoms) -> List[ParentCandidate]:
    """Pool ALL admissible ring + chain parent candidates.

    Admission gates (these are NOT seniority rules - they decide which
    skeletons are legal parents at all):
    - every connected ring system is a candidate;
    - the carbon principal chain (if any) is a candidate;
    - a longer PG-bearing skeletal hetero chain is admitted ONLY in the
      tied-PG context (PG on BOTH the ring system and the carbon principal
      chain) with a bridging heteroatom - exactly the branch where the
      legacy staged cascade considered it (a phase / (b)).
      Outside that context the skeletal chain must not displace the
      carbon-chain parent (e.g. the secondary-amine C-N-C shape);
    - with no principal group, a longer skeletal chain is admitted at
      >= 4 bridging hetero units replacement admission).
    """
    from ..perception.chains import find_longest_skeletal_chain
    from ..perception.functional_groups import get_chain_excluded_atoms
    from .parent_selection import (
        _count_bridging_heteroatoms,
        _count_pg_on_chain,
        _count_pg_on_ring,
        _has_bridging_heteroatom,
        is_principal_group_on_chain,
        is_principal_group_on_ring,
    )

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
            _tied_context = bool(
                principal_chain
                and ring_systems
                and is_principal_group_on_ring(
                    mol, all_ring_atoms, principal_group_atoms,
                    principal_group)
                and is_principal_group_on_chain(
                    mol, list(principal_chain), principal_group_atoms,
                    principal_group)
            )
            if (_tied_context
                    and is_principal_group_on_chain(
                        mol, sk, principal_group_atoms, principal_group)
                    and _has_bridging_heteroatom(mol, sk)):
                chains.append(tuple(sk))
        elif _count_bridging_heteroatoms(mol, sk) >= 4:
            chains.append(tuple(sk))

    for ch in chains:
        pg_n = _count_pg_on_chain(mol, list(ch), principal_group_atoms or [],
                                  principal_group)
        pool.append(ParentCandidate('chain', ch, pg_n))
    return pool


def _pg_atom_set(principal_group_atoms) -> set:
    out = set()
    for match in (principal_group_atoms or []):
        out.update(match)
    return out


def _chain_hetero_expressible(mol, atoms) -> bool:
    """True when the chain's heteroatoms are expressible as SKELETON:
    a pure non-carbon hydride chain or a replacement-
    admissible mixed chain (>= 4 bridging hetero units)."""
    from .parent_selection import _count_bridging_heteroatoms
    syms = {mol.GetAtomWithIdx(i).GetSymbol() for i in atoms}
    if 'C' not in syms:
        return True
    return _count_bridging_heteroatoms(mol, list(atoms)) >= 4


def _class_rank(mol, cand: ParentCandidate, principal_group_atoms=None) -> int:
    """: rank of the most senior EXPRESSIBLE skeletal element."""
    if cand.kind == 'ring':
        return max((P44_CLASS_RANK.get(mol.GetAtomWithIdx(i).GetSymbol(), 0)
                    for i in cand.atoms), default=0)
    pg_atoms = _pg_atom_set(principal_group_atoms)
    syms = [(i, mol.GetAtomWithIdx(i).GetSymbol()) for i in cand.atoms]
    has_c = any(s == 'C' for _, s in syms)
    het = [(i, s) for i, s in syms if s != 'C' and i not in pg_atoms]
    if not het:
        return P44_CLASS_RANK['C'] if has_c else 0
    if _chain_hetero_expressible(mol, cand.atoms):
        best_het = max(P44_CLASS_RANK.get(s, 0) for _, s in het)
        return max(best_het, P44_CLASS_RANK['C'] if has_c else 0)
    return P44_CLASS_RANK['C'] if has_c else 0


def _n_multiple_bonds(mol, atom_set) -> Tuple[int, int]:
    """(multiple, double) bond counts within the candidate skeleton.

    : mancude (aromatic) bonds count as noncumulative double
    bonds -- GetBondTypeAsDouble returns 1.5 for aromatic, > 1.0 counts.
    """
    n_mult = n_dbl = 0
    for b in bonds_of(mol):
        if b.GetBeginAtomIdx() in atom_set and b.GetEndAtomIdx() in atom_set:
            v = b.GetBondTypeAsDouble()
            if v > 1.0:
                n_mult += 1
            if 1.5 <= v <= 2.0:
                n_dbl += 1
    return n_mult, n_dbl


def _locant_sort_key(v):
    """ a phase cleanup T3: TOTAL, crash-proof sort key for the locant
    values `_candidate_locants` collects (below).

    The `_Locant` type contract (`locants.py`) only ever admits ``int`` or
    ``(int, str)`` fusion/prime tuples -- but `_candidate_locants`'s FAST
    PATH reads ``ring_info['iupac_locants']`` DIRECTLY (bypassing
    `_build_ring_pos`'s legacy-string filter, which drops any locant that
    is not ``int``/``tuple`` before the sorted-fallback proxy). When a
    producer leaves a bare legacy-string locant (e.g. ``'3a'``, never
    coerced to ``(3, 'a')``) in that dict and every one of ``cand.atoms``
    still happens to be a key, the fast path takes it uncoerced -- and
    plain ``sorted(locs)`` then compares that string against an ``int``/
    ``tuple`` neighbour and raises ``TypeError`` (measured, L3-2/L3-3 a trace:
    crashes ``_classify`` -> the molecule abstains; 0-wrong-safe but a
    robustness defect).

    This key bucket-separates by type -- ``(0, (base, suffix))`` for every
    ``int``/``tuple`` locant (ints padded to ``(n, '')`` so two bucket-0
    keys always compare tuple-to-tuple, never int-to-tuple) and
    ``(1, str_val)`` for a bare string -- so bucket 0 always sorts before
    bucket 1 and Python never has to compare across the two shapes.

    This is deliberately NOT a claim that a bare string locant sorts into
    the numerically "correct" position relative to the int/tuple ones --
    doing that correctly means root-causing the producer that leaks the
    unconverted string in the first place, out of scope for this fix (the
    task brief sanctions the minimal safe fix here). It only guarantees
    `sorted` cannot raise, matching the project's fail-closed contract.

    Byte-identical for every list this project has ever fed `sorted`
    here: a homogeneous-int or homogeneous-(int, str)-tuple list (the
    only two shapes the type contract already allowed) reduces to the
    SAME relative order `sorted(locs)` always gave it, since every
    element still lands in bucket 0 and the inner ``(base, suffix)``
    comparison reproduces plain int/tuple ordering exactly. A
    homogeneous-string list also reduces to its previous (lexicographic)
    order, since every element lands in bucket 1 alone.
    """
    if isinstance(v, str):
        return (1, v)
    if isinstance(v, tuple):
        return (0, v)
    return (0, (v, ''))


def _candidate_locants(mol, cand: ParentCandidate, target_atoms,
                       ring_info=None):
    """Lowest-locant set for `target_atoms` on this candidate.

    Chain: try both directions, keep the lower set first point of
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
    # a phase homogeneity: raw iupac_locants (and thus the fast path
    # above) can mix int locants (2) with (int, str) fusion/prime tuples
    # ((4, 'a'), (2, "'")). Coerce ints to (n, '') when any tuple is
    # present so BOTH this sort AND the downstream compare_locant_sets stay
    # type-safe (same idiom as _build_ring_pos:180-184 and
    # compare_locant_sets). #40: fused ring assembly Tier-5 non-crash.
    locs = [pos[a] for a in targets if a in pos]
    if any(isinstance(v, tuple) for v in locs):
        locs = [(v, '') if isinstance(v, int) else v for v in locs]
    return sorted(locs, key=_locant_sort_key)


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


def _ring_hetero_rank_vector(mol, atoms):
    """Descending rank vector of ring hetero skeletal atoms."""
    ranks = sorted((RING_HETERO_RANK.get(mol.GetAtomWithIdx(i).GetSymbol(), 0)
                    for i in atoms
                    if mol.GetAtomWithIdx(i).GetSymbol() != 'C'),
                   reverse=True)
    return tuple(ranks)


def _chain_hetero_rank_vector(mol, cand, principal_group_atoms=None):
    """Descending rank vector of EXPRESSIBLE chain skeletal heteroatoms
    (a)/(c)); suffix (PG-match) heteroatoms never count."""
    if not _chain_hetero_expressible(mol, cand.atoms):
        return ()
    pg_atoms = _pg_atom_set(principal_group_atoms)
    ranks = sorted((CHAIN_HETERO_RANK.get(mol.GetAtomWithIdx(i).GetSymbol(), 0)
                    for i in cand.atoms
                    if mol.GetAtomWithIdx(i).GetSymbol() != 'C'
                    and i not in pg_atoms),
                   reverse=True)
    return tuple(ranks)


def _cmp(x, y) -> int:
    return (x > y) - (x < y)


def compare_with_reason(mol, a: ParentCandidate, b: ParentCandidate, *,
                        principal_group=None,
                        principal_group_atoms=None,
                        ring_info=None) -> Tuple[int, str]:
    """Blue Book seniority with the deciding criterion.

    Returns (sign, label): sign >0 a senior, <0 b senior, 0 tie; label
    names the rule that decided.
    """
    from .locants import compare_locant_sets
    from .parent_selection import _pg_attachment_atoms

    # -- more principal characteristic groups as suffix.
    if a.pg_count != b.pg_count:
        return _cmp(a.pg_count, b.pg_count), "PG count (P-44.1.1)"

    # -- senior skeletal atom class (ring-vs-chain / class level).
    r = _cmp(_class_rank(mol, a, principal_group_atoms),
             _class_rank(mol, b, principal_group_atoms))
    if r:
        return r, "senior atom class (P-44.1.2)"

    # (1) -- same class: ring senior to chain (also the
    # ring-on-tie preference of the staged cascade).
    if a.kind != b.kind:
        return ((1 if a.kind == 'ring' else -1),
                "ring senior to chain (P-44.1.2.2/P-52.2.8)")

    if a.kind == 'ring':
        # (a)..(g)
        sa = {mol.GetAtomWithIdx(i).GetSymbol() for i in a.atoms}
        sb = {mol.GetAtomWithIdx(i).GetSymbol() for i in b.atoms}
        r = _cmp(sa != {'C'}, sb != {'C'})
        if r:
            return r, "heterocycle senior (P-44.2.1.2)"
        r = _cmp('N' in sa, 'N' in sb)
        if r:
            return r, "nitrogen ring senior (P-44.2.1.3)"
        ra = max((RING_HETERO_RANK.get(s, 0) for s in sa if s != 'C'), default=0)
        rb = max((RING_HETERO_RANK.get(s, 0) for s in sb if s != 'C'), default=0)
        r = _cmp(ra, rb)
        if r:
            return r, "earlier ring heteroatom (P-44.2.1.4)"
        r = _cmp(_ring_subrings(mol, set(a.atoms)),
                 _ring_subrings(mol, set(b.atoms)))
        if r:
            return r, "more rings (P-44.2.1.5)"
        r = _cmp(len(a.atoms), len(b.atoms))
        if r:
            return r, "more ring skeletal atoms (P-44.2.1.6)"
        ha = _ring_hetero_rank_vector(mol, a.atoms)
        hb = _ring_hetero_rank_vector(mol, b.atoms)
        r = _cmp(len(ha), len(hb))
        if r:
            return r, "more ring heteroatoms (P-44.2.1.7)"
        r = _cmp(ha, hb)
        if r:
            return r, "earlier ring heteroatom set (P-44.2.1.8)"
    else:
        # (a)..(c)
        ha = _chain_hetero_rank_vector(mol, a, principal_group_atoms)
        hb = _chain_hetero_rank_vector(mol, b, principal_group_atoms)
        r = _cmp(len(ha), len(hb))
        if r:
            return r, "more chain heteroatoms (P-44.3.1)"
        r = _cmp(len(a.atoms), len(b.atoms))
        if r:
            return r, "longer chain (P-44.3.2)"
        r = _cmp(ha, hb)
        if r:
            return r, "senior chain heteroatoms (P-44.3.3)"

    # shared tiebreaks.
    ma, da = _n_multiple_bonds(mol, set(a.atoms))
    mb, db = _n_multiple_bonds(mol, set(b.atoms))
    r = _cmp(ma, mb)
    if r:
        return r, "more multiple bonds (P-44.4.1.1)"
    r = _cmp(da, db)
    if r:
        return r, "more double bonds (P-44.4.1.2)"

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
                return r, "lower suffix locants (P-44.4.1.8)"

    # (j) lower locants for unsaturation.
    def _unsat_atoms(cand):
        cset = set(cand.atoms)
        out = set()
        for bnd in bonds_of(mol):
            if (bnd.GetBeginAtomIdx() in cset and bnd.GetEndAtomIdx() in cset
                    and bnd.GetBondTypeAsDouble() > 1.0):
                out.update((bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()))
        return out

    la = _candidate_locants(mol, a, _unsat_atoms(a), ring_info=ring_info)
    lb = _candidate_locants(mol, b, _unsat_atoms(b), ring_info=ring_info)
    if la or lb:
        r = -compare_locant_sets(la, lb)
        if r:
            return r, "lower unsaturation locants (P-44.4.1.10)"

    # Legacy-preserved final criteria (h)/(i) of the staged cascade):
    # more substituents, then lower substituent locants.
    pa = _substituent_positions(mol, a)
    pb = _substituent_positions(mol, b)
    r = _cmp(len(pa), len(pb))
    if r:
        return r, "more substituents"
    la = _candidate_locants(mol, a, set(pa), ring_info=ring_info)
    lb = _candidate_locants(mol, b, set(pb), ring_info=ring_info)
    if la or lb:
        r = -compare_locant_sets(la, lb)
        if r:
            return r, "lower substituent locants"

    # Deterministic total order (never input-order-dependent).
    return (_cmp(tuple(sorted(b.atoms)), tuple(sorted(a.atoms))),
            "deterministic atom-order tiebreak")


def compare_parent_candidates(mol, a: ParentCandidate, b: ParentCandidate, *,
                              principal_group=None,
                              principal_group_atoms=None,
                              ring_info=None) -> int:
    """Blue Book seniority: >0 a senior, <0 b senior, 0 tie."""
    return compare_with_reason(
        mol, a, b,
        principal_group=principal_group,
        principal_group_atoms=principal_group_atoms,
        ring_info=ring_info)[0]


def select_parent_unified(
    mol,
    ring_systems,
    principal_chain,
    principal_group,
    principal_group_atoms,
    ring_info: Optional[dict] = None,
    _offer_rank: int = 0,
):
    """Drop-in unified replacement for ``parent_selection.select_parent``.

    Pre-empts (admission/priority rules) are preserved from the staged
    implementation; everything else is ONE pooled comparator sort.

     (offer-not-return, a project rule): ``_offer_rank`` selects WHICH
    member of the -ranked pool becomes the parent. ``_offer_rank == 0`` (the
    default at every normal call site) commits to ``ranked[0]`` — byte-identical
    to the pre-SP1.3 behaviour for every molecule. A best-effort retry may pass
    ``_offer_rank == k`` to force the k-th-ranked candidate ONLY after
    ``ranked[0]`` has already abstained; the ranking is NEVER reordered and
    an out-of-range rank clamps back to ``ranked[0]``. Every returned result
    carries ``parent_pool_size`` so the caller can bound its retry.
    """
    from ..perception.natural_products import detect_natural_product
    from .parent_selection import (
        SKELETAL_SUFFIX_PGS,
        ParentSelectionResult,
        is_principal_group_on_ring,
    )

    all_ring_atoms = set()
    for ring in ring_systems:
        all_ring_atoms.update(ring)
    ring_tuples = [tuple(sorted(r)) for r in ring_systems]

    # --- Pre-empt 1: no chain -> ring (legacy:735) ---
    if not principal_chain:
        return ParentSelectionResult(
            parent_type='ring',
            parent_atoms=list(sorted(all_ring_atoms)),
            substituent_rings=[],
            reasoning="P-44.1 unified: no chain provided - ring is parent",
        )

    # --- Pre-empt 2: single-carbon chain (legacy:752-776, verbatim rule) ---
    if len(principal_chain) == 1:
        _single = principal_chain[0]
        _single_ring_attached = any(
            nbr.GetIdx() in all_ring_atoms
            for nbr in mol.GetAtomWithIdx(_single).GetNeighbors()
        )
        _skeletal = principal_group in SKELETAL_SUFFIX_PGS
        if (not principal_group or not principal_group_atoms
                or (_single_ring_attached and not _skeletal)
                or is_principal_group_on_ring(
                    mol, all_ring_atoms, principal_group_atoms, principal_group)):
            return ParentSelectionResult(
                parent_type='ring',
                parent_atoms=list(sorted(all_ring_atoms)),
                substituent_rings=[],
                reasoning="P-44.1 unified: single carbon chain - ring is parent",
            )
        # else fall through: the 1-carbon PCG chain joins the pool below.

    # --- Pre-empt 3: natural-product backbone, legacy:779) ---
    np_info = detect_natural_product(mol)
    if np_info is not None and ring_systems:
        pool = pool_candidates(mol, ring_systems, [], principal_group,
                               principal_group_atoms)
        rings_only = [c for c in pool if c.kind == 'ring']
        best = _sort_pool(mol, rings_only, principal_group,
                          principal_group_atoms, ring_info)[0]
        others = [t for t in ring_tuples if t != tuple(sorted(best.atoms))]
        return ParentSelectionResult(
            parent_type='ring',
            parent_atoms=list(best.atoms),
            substituent_rings=others,
            reasoning=(f"P-31.1.3.4: {np_info['scaffold_class']} NP backbone"
                       " - ring is parent"),
        )

    # --- The unified pool + comparator ---
    pool = pool_candidates(mol, ring_systems, principal_chain,
                           principal_group, principal_group_atoms)
    if not pool:
        return ParentSelectionResult(
            parent_type='ring',
            parent_atoms=list(sorted(all_ring_atoms)),
            substituent_rings=[],
            reasoning="P-44.1 unified: empty candidate pool - default ring",
        )
    ranked = _sort_pool(mol, pool, principal_group, principal_group_atoms,
                        ring_info)
    best = ranked[0]
    if len(ranked) > 1:
        _, label = compare_with_reason(
            mol, best, ranked[1],
            principal_group=principal_group,
            principal_group_atoms=principal_group_atoms,
            ring_info=ring_info)
    else:
        label = "only candidate"

    #: OFFER the ranked pool. ``_offer_rank == 0`` -> ``chosen`` IS
    # ``best`` (byte-identical). A best-effort retry may force a junior member;
    # an out-of-range rank clamps to ``best`` so the offer can never fabricate a
    # parent that was not in the -ranked pool.
    _pool_size = len(ranked)
    offer = _offer_rank if 0 <= _offer_rank < _pool_size else 0
    chosen = ranked[offer]

    if chosen.kind == 'chain':
        return ParentSelectionResult(
            parent_type='chain',
            parent_atoms=list(chosen.atoms),
            substituent_rings=ring_tuples,
            reasoning=(f"P-44.1 unified comparator: chain wins by {label} "
                       f"(len={len(chosen.atoms)}, pool={len(pool)})"),
            parent_pool_size=_pool_size,
        )
    others = [t for t in ring_tuples if t != tuple(sorted(chosen.atoms))]
    return ParentSelectionResult(
        parent_type='ring',
        parent_atoms=list(chosen.atoms),
        substituent_rings=others,
        reasoning=(f"P-44.1 unified comparator: ring wins by {label} "
                   f"(size={len(chosen.atoms)}, pool={len(pool)})"),
        parent_pool_size=_pool_size,
    )


def _sort_pool(mol, pool, principal_group, principal_group_atoms, ring_info):
    import functools
    key = functools.cmp_to_key(
        lambda a, b: compare_parent_candidates(
            mol, a, b,
            principal_group=principal_group,
            principal_group_atoms=principal_group_atoms,
            ring_info=ring_info))
    return sorted(pool, key=key, reverse=True)
