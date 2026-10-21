"""Wave2: the substituent prefix of a NON-AROMATIC monocycle with ONE heteroatom
and a skeletal double bond ('3,4-dihydro-2H-pyran-2-yl'), moved out of ``ring_substituents`` (the
size budget) and given the '-ylidene' mode of ``rules.ring_ylidene_prefix``.

``ring_substituents`` re-exports the names, so every earlier import keeps working.
"""
from typing import Dict, List, Optional, Set, Tuple

# Wave2 T6b: hydro-prefix multipliers for the heteromonocyclic substituent
# emitter (hydro counts are always even — each hydrogenated mancude double
# bond contributes two positions).
_HYDRO_MULTIPLIERS = {2: 'di', 4: 'tetra', 6: 'hexa', 8: 'octa', 10: 'deca'}


def _unsaturated_heteromonocyclic_substituent(
    mol, ring_atoms: Tuple[int, ...], attachment_atom: int, ylidene: bool = False
) -> Optional[str]:
    """Wave2: substituent name for a NON-AROMATIC monocyclic
    ring with exactly ONE heteroatom carrying >=1 skeletal double bond, named
    on its mancude parent with hydro prefixes and indicated hydrogen:
    ``3,4-dihydro-2H-pyran-2-yl``, ``2,3-dihydrofuran-...-yl``.

    Locant selection follows: heteroatom lowest (fixed = 1), then
    indicated hydrogen, then the free-valence suffix, and finally 'hydro'
    prefixes. The mancude double-bond patterns are enumerated as maximum
    matchings of the ring cycle (O/S/Se/Te excluded from double bonds); the
    actual double-bond set must be a subset of the chosen pattern, the
    hydrogenated pattern bonds become the hydro positions, and the unmatched
    atom (if any) is the indicated hydrogen. Returns None (fail-closed) for
    anything outside this narrow class — multiple heteroatoms, aromatic or
    fused rings, triple bonds, exocyclic decoration, attachment on the
    heteroatom (would need added indicated hydrogen), or an unnameable
    mancude parent.

    ``ylidene``: the attachment bond is a DOUBLE bond and the token ends '-ylidene'
    ('2H-pyrrol-2-ylidene', '4H-thiopyran-4-ylidene (PIN)',, the Blue Book):
    only when the ylidene atom is the indicated-hydrogen position of the mancude parent (the
    other placement needs the added-hydrogen form,, which is not built).
    """
    from itertools import combinations

    ring_set = set(ring_atoms)
    n = len(ring_set)
    if attachment_atom not in ring_set or n < 3 or n > 10:
        return None
    ri = mol.GetRingInfo()

    het_idx = None
    adj: Dict[int, List[int]] = {}
    for i in ring_atoms:
        a = mol.GetAtomWithIdx(i)
        if a.GetIsAromatic() or a.GetFormalCharge() != 0:
            return None
        if ri.NumAtomRings(i) != 1:
            return None  # fused / spiro atom -> not a simple monocycle
        sym = a.GetSymbol()
        if sym != 'C':
            if sym not in ('N', 'O', 'S', 'Se', 'Te') or het_idx is not None:
                return None  # unsupported element or >1 heteroatom
            het_idx = i
        nbrs = [x.GetIdx() for x in a.GetNeighbors() if x.GetIdx() in ring_set]
        if len(nbrs) != 2:
            return None
        adj[i] = nbrs
    if het_idx is None:
        return None  # carbocycle -> _unsaturated_carbocyclic_substituent
    if attachment_atom == het_idx:
        return None  # N-attachment needs added indicated hydrogen -> decline

    # No exocyclic heavy decoration; the parent bond at the attachment atom must be a
    # single bond ('-yl') or, with ``ylidene``, a double bond ('-ylidene').
    for i in ring_atoms:
        atom_i = mol.GetAtomWithIdx(i)
        for nbr in atom_i.GetNeighbors():
            j = nbr.GetIdx()
            if j in ring_set or nbr.GetAtomicNum() <= 1:
                continue
            if i != attachment_atom:
                return None
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond.GetBondTypeAsDouble() != (2.0 if ylidene else 1.0):
                return None

    # Ring bonds: actual double bonds (A) and double-bond-eligible bonds.
    ring_bonds: List[Tuple[int, int]] = []
    actual_dbl: Set[Tuple[int, int]] = set()
    db_eligible: List[Tuple[int, int]] = []
    for i in ring_atoms:
        for j in adj[i]:
            if i >= j:
                continue
            order = mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble()
            if order not in (1.0, 2.0):
                return None  # triple / aromatic / kekulization oddity
            pair = (i, j)
            ring_bonds.append(pair)
            if order == 2.0:
                actual_dbl.add(pair)
            if all(
                mol.GetAtomWithIdx(k).GetSymbol() in ('C', 'N') for k in pair
            ):
                db_eligible.append(pair)
    if not actual_dbl:
        return None  # saturated -> the ordinary saturated stems apply
    if not actual_dbl <= set(db_eligible):
        return None  # double bond on a divalent chalcogen — not mancude-based

    # Maximum matchings of the ring cycle over the eligible bonds = the
    # candidate mancude double-bond patterns.
    max_size = 0
    matchings: List[Set[Tuple[int, int]]] = []
    for size in range(len(db_eligible) // 2 + 1, 0, -1):
        for combo in combinations(db_eligible, size):
            atoms_seen: Set[int] = set()
            ok = True
            for x, y in combo:
                if x in atoms_seen or y in atoms_seen:
                    ok = False
                    break
                atoms_seen.add(x)
                atoms_seen.add(y)
            if ok:
                matchings.append(set(combo))
        if matchings:
            max_size = size
            break
    if not matchings or max_size == 0:
        return None
    candidates_m = [m for m in matchings if actual_dbl <= m]
    if not candidates_m:
        return None  # actual pattern incompatible with any mancude parent

    # Two numbering directions, heteroatom fixed at locant 1.
    def _ring_order(start_nbr: int) -> Optional[List[int]]:
        seq = [het_idx, start_nbr]
        while len(seq) < n:
            prev, cur = seq[-2], seq[-1]
            nxt = [x for x in adj[cur] if x != prev]
            if not nxt:
                return None
            seq.append(nxt[0])
        return seq

    best = None  # (ih_locs, yl_loc, hydro_locs, name_parts)
    for start in adj[het_idx]:
        seq = _ring_order(start)
        if seq is None or len(seq) != n:
            continue
        pos = {atom: idx + 1 for idx, atom in enumerate(seq)}
        for m in candidates_m:
            matched_atoms = {k for pair in m for k in pair}
            # Indicated hydrogen sites are unmatched sp3 C/N skeletal atoms of
            # the mancude parent; an unmatched divalent chalcogen (O/S/Se/Te)
            # bears no H and is NOT an indicated-H position (furan/pyran O).
            ih_locs = sorted(
                pos[a] for a in ring_set
                if a not in matched_atoms
                and mol.GetAtomWithIdx(a).GetSymbol() in ('C', 'N')
            )
            if len(ih_locs) > 1:
                continue  # multi-indicated-H parent — out of scope
            if ylidene and ih_locs != [pos[attachment_atom]]:
                continue  # the ylidene atom is not the indicated-hydrogen position
            hydro_locs = sorted(
                pos[k] for pair in (m - actual_dbl) for k in pair
            )
            yl_loc = pos[attachment_atom]
            key = (ih_locs, yl_loc, hydro_locs)
            if best is None or key < best[0]:
                best = (key, ih_locs, yl_loc, hydro_locs)
    if best is None:
        return None
    _, ih_locs, yl_loc, hydro_locs = best

    # Mancude parent stem via the heterocycle parent namer on a detached,
    # mancude-ized copy of the ring (any maximum matching gives the same
    # stem; the indicated-H token, if emitted, is stripped — the substituent
    # form re-derives it from the selected numbering above).
    stem = _mancude_monocycle_stem(mol, ring_atoms, candidates_m[0])
    if stem is None:
        return None

    hydro_part = ''
    if hydro_locs:
        mult = _HYDRO_MULTIPLIERS.get(len(hydro_locs))
        if mult is None:
            return None
        hydro_part = f"{','.join(str(loc) for loc in hydro_locs)}-{mult}hydro"
    ih_part = f"{ih_locs[0]}H-" if ih_locs else ''
    if hydro_part and ih_part:
        core = f"{hydro_part}-{ih_part}{stem}"
    elif hydro_part:
        core = f"{hydro_part}{stem}"
    else:
        core = f"{ih_part}{stem}"
    if stem.endswith('e'):
        core = core[:-1]
    return f"{core}-{yl_loc}-{'ylidene' if ylidene else 'yl'}"


def _mancude_monocycle_stem(
    mol, ring_atoms: Tuple[int, ...], matching: Set[Tuple[int, int]]
) -> Optional[str]:
    """Name the mancude parent of a detached heteromonocycle (double bonds
    set to ``matching``) via the heterocycle parent namer and return the bare
    stem with any leading indicated-H token stripped ('2H-pyran' -> 'pyran').
    Fail-closed: None unless the result is a single plain word."""
    import re

    from rdkit import Chem
    keep = set(ring_atoms)
    try:
        rw = Chem.RWMol(mol)
        for idx in sorted((a.GetIdx() for a in mol.GetAtoms()
                           if a.GetIdx() not in keep), reverse=True):
            rw.RemoveAtom(idx)
        sub = rw.GetMol()
        # Re-map original indices -> submol indices (removal preserves the
        # relative order of the kept atoms).
        old_order = sorted(keep)
        old_to_new = {old: new for new, old in enumerate(old_order)}
        for bond in sub.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
        for x, y in matching:
            b = sub.GetBondBetweenAtoms(old_to_new[x], old_to_new[y])
            if b is None:
                return None
            b.SetBondType(Chem.BondType.DOUBLE)
        for a in sub.GetAtoms():
            a.SetNoImplicit(False)
            a.SetNumExplicitHs(0)
        Chem.SanitizeMol(sub)
    except Exception:
        return None
    try:
        from .heterocycles import name_heterocycle
        raw = name_heterocycle(sub, tuple(range(sub.GetNumAtoms())))
    except Exception:
        return None
    if not raw or not isinstance(raw, str):
        return None
    stem = re.sub(r'^\d+H-', '', raw)
    if not re.fullmatch(r'[a-z]+', stem):
        return None  # locants / hyphens survived — not a bare mancude stem
    return stem
