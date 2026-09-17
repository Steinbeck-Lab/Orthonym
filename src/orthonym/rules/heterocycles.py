"""
Heterocycle detection, classification, ring numbering, and naming according to IUPAC 2013.

Handles:
- Heterocycle classification (ring size, aromaticity, heteroatom types)
- Ring numbering starting at highest-priority heteroatom
- Direction selection to minimize locants for other heteroatoms
- First-point-of-difference rule for direction ties
- Retained name lookup for common heterocycles
- Hantzsch-Widman (HW) systematic name generation

IUPAC 2013 Rules for heterocycle numbering:
- Position 1 goes to highest-priority heteroatom (O > S > Se > Te > N > P >...)
- Numbering direction chosen to give lowest locants to other heteroatoms
- First-point-of-difference comparison (not sum of locants)

Naming priority:
1. Check retained names FIRST (pyridine, furan, morpholine, etc.)
2. Fall back to HW systematic naming if no retained name

Reference: IUPAC 2013 Blue Book, Section (Heterocycles)
"""

import logging
import re
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from rdkit import Chem

from ..data.hw_heteroatoms import HETEROATOM_PRIORITY, get_heteroatom_priority, get_hw_prefix
from ..data.hw_stems import HW_STEMS, get_hw_stem
from ..data.retained_names import get_retained_name
from ..perception.rings import (
    is_aromatic_ring,
    is_saturated_ring,
)
from ..perception.smarts_cache import compiled as _compiled_smarts
from ..perception.molcache import bonds_of

# Simple multiplicative prefixes for HW naming
SIMPLE_MULTIPLIERS = {
    2: "di", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
}


def classify_heterocycle(mol, ring_atoms) -> Dict:
    """
    Classify a heterocyclic ring by its structural features.

    Returns detailed information about the ring for naming purposes:
    - Ring size (number of atoms)
    - Heteroatom identity and positions
    - Aromaticity
    - Saturation status
    - Dominant heteroatom (highest priority)

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring

    Returns:
        Dict with:
        - ring_size: int (3-10 typically)
        - heteroatoms: List[Tuple[int, str]] - (atom_idx, element_symbol)
        - is_aromatic: bool
        - is_saturated: bool (no double bonds within ring)
        - num_heteroatoms: int
        - dominant_heteroatom: str (highest priority element in ring)

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccncc1') # pyridine
        >>> info = classify_heterocycle(mol, mol.GetRingInfo.AtomRings[0])
        >>> info['ring_size']
        6
        >>> info['is_aromatic']
        True
        >>> info['dominant_heteroatom']
        'N'
    """
    ring_list = list(ring_atoms)
    ring_size = len(ring_list)

    # Get heteroatoms with their atom indices (not ring positions)
    # get_ring_heteroatoms returns (position_in_ring, element) but we need atom indices
    heteroatoms = []
    for idx in ring_list:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            heteroatoms.append((idx, symbol))

    # Determine aromaticity and saturation
    # Note: is_saturated_ring checks for explicit double bonds, but aromatic
    # rings have delocalized bonding. If a ring is aromatic, it's NOT saturated.
    is_arom = is_aromatic_ring(mol, ring_atoms)
    is_sat = is_saturated_ring(mol, ring_atoms) and not is_arom

    # Find dominant (highest priority) heteroatom
    dominant = None
    if heteroatoms:
        # Sort by priority (lower number = higher priority)
        sorted_heteroatoms = sorted(
            heteroatoms,
            key=lambda x: get_heteroatom_priority(x[1])
        )
        dominant = sorted_heteroatoms[0][1]

    return {
        'ring_size': ring_size,
        'heteroatoms': heteroatoms,
        'is_aromatic': is_arom,
        'is_saturated': is_sat,
        'num_heteroatoms': len(heteroatoms),
        'dominant_heteroatom': dominant,
    }


def number_heterocycle_ring(
    mol,
    ring_atoms,
    heteroatoms: Optional[List[Tuple[int, str]]] = None
) -> List[int]:
    """
    Reorder ring atoms for IUPAC numbering of heterocycles.

    Position 1 is assigned to the highest-priority heteroatom.
    Numbering direction is chosen to give lowest locants to other heteroatoms,
    using first-point-of-difference comparison.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring
        heteroatoms: Optional list of (atom_idx, element) tuples.
                     If None, will be computed from ring_atoms.

    Returns:
        List of ring atom indices reordered for IUPAC numbering.
        Position 1 (index 0) is the highest-priority heteroatom.

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccncc1') # pyridine
        >>> ring = mol.GetRingInfo.AtomRings[0]
        >>> oriented = number_heterocycle_ring(mol, ring)
        >>> # N atom should be at position 1 (index 0)
        >>> mol.GetAtomWithIdx(oriented[0]).GetSymbol
        'N'
    """
    ring_list = list(ring_atoms)
    n = len(ring_list)

    if n == 0:
        return []

    # Get heteroatoms if not provided
    if heteroatoms is None:
        heteroatoms = []
        for idx in ring_list:
            atom = mol.GetAtomWithIdx(idx)
            symbol = atom.GetSymbol()
            if symbol != 'C':
                heteroatoms.append((idx, symbol))

    if not heteroatoms:
        # No heteroatoms - not a true heterocycle, return as-is
        return ring_list

    #: find the highest-priority element (lowest priority number).
    best_prio = min(get_heteroatom_priority(sym) for _, sym in heteroatoms)
    best_starts = [(idx, sym) for idx, sym in heteroatoms
                   if get_heteroatom_priority(sym) == best_prio]

    # If only one heteroatom, direction doesn't matter
    if len(heteroatoms) == 1:
        start_pos = ring_list.index(best_starts[0][0])
        return _rotate_ring(ring_list, start_pos, 1)

    # Try ALL equal-priority start atoms in both directions; pick the
    # (start, direction) that gives the lexicographically minimum sorted
    # all-heteroatom locant set lowest-locant criterion).
    # Use canonical rank as a tiebreaker to ensure determinism.
    _canon_rank = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    best_oriented = None
    best_key = None
    for start_idx, _ in best_starts:
        start_pos = ring_list.index(start_idx)
        for direction in (1, -1):
            oriented = _rotate_ring(ring_list, start_pos, direction)
            other_locs = _get_other_heteroatom_locants(oriented, heteroatoms, start_idx)
            all_locs = sorted([1] + other_locs)
            # Tiebreaker: canonical rank sequence of the oriented ring
            canon_key = [_canon_rank[a] for a in oriented]
            key = (all_locs, canon_key)
            if best_key is None or key < best_key:
                best_oriented = oriented
                best_key = key
    return best_oriented


def orient_heterocycle(mol, ring_atoms) -> Tuple[List[int], Dict[int, int]]:
    """
    Convenience function combining classification and numbering.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring

    Returns:
        Tuple of:
        - oriented_ring: List of atom indices in IUPAC numbering order
        - atom_to_locant: Dict mapping atom_idx to locant (1-indexed)

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccncc1') # pyridine
        >>> ring = mol.GetRingInfo.AtomRings[0]
        >>> oriented, mapping = orient_heterocycle(mol, ring)
        >>> # N is at locant 1
        >>> n_idx = [i for i in oriented if mol.GetAtomWithIdx(i).GetSymbol == 'N'][0]
        >>> mapping[n_idx]
        1
    """
    # Classify to get heteroatoms
    info = classify_heterocycle(mol, ring_atoms)
    heteroatoms = info['heteroatoms']

    # Number the ring
    oriented = number_heterocycle_ring(mol, ring_atoms, heteroatoms)

    # Build atom-to-locant mapping
    atom_to_locant = {atom_idx: locant for locant, atom_idx in enumerate(oriented, 1)}

    return oriented, atom_to_locant


#: Group 15 ring heteroatoms -- the only ones that carry an indicated hydrogen a
#: substituent can displace while the atom stays NEUTRAL. An aromatic ring O, S
#: or Se with an exocyclic bond is an onium centre, excluded by the formal-charge
#: test below; naming the elements states the chemistry rather than relying on it.
_INDICATED_H_ELEMENTS = frozenset({7, 15, 33, 51, 83})  # N, P, As, Sb, Bi


def _occupies_indicated_h_position(mol, idx: int, ring_set: Set[int]) -> bool:
    """Does this ring atom hold the ring's indicated-hydrogen position?

     "NUMBERING" (``the Blue Book Blue Book``) assigns low locants
    "in the following decreasing order of seniority", listing
    **(b) indicated hydrogen** ahead of **(c) principal characteristic groups
    and free valences (suffixes)**. So the atom holding the indicated hydrogen
    has to be identified before the suffix can compete for locant 1. (b)'s own
    caveat -- "a higher locant may be needed at another position to accommodate
    a substituent suffix in accordance with structural feature (d)" -- points at
    (d) *added* indicated hydrogen (``3,4-dihydronaphthalen-1(2H)-one``), not at
    ordinary substitution.

    Testing ``GetTotalNumHs >= 1`` answers a different question. A substituent
    does not move the indicated hydrogen -- it stands in its place, and
     "Substitution rules for Type 1 retained names" (``:4916``),
    sentence ``:4918``, allows that substitution without limit. An N-substituted
    azole nitrogen has no hydrogen left to count, so it read as pyridine-type and
    surrendered locant 1: ``Cn1nccc1C(=O)O`` was named
    ``2-methyl-1H-pyrazole-3-carboxylic acid``, whose ``1H`` and whose
    ``2-methyl`` describe different atoms, and which OPSIN rejects outright.

    The displacement is *proved*, not assumed: a neutral group-15 aromatic ring
    atom with no hydrogen and exactly ONE exocyclic SINGLE bond has had exactly
    one hydrogen's worth of valence taken from it. Nothing here consults a ring
    size or a heteroatom count.

    R1-FIX / a review-B "Indicated hydrogen"): indicated hydrogen is a
    MANCUDE-ring concept -- it names the sp3 position of a ring that carries the
    maximum number of noncumulative double bonds. A FULLY SATURATED ring has no
    indicated hydrogen at all, so no atom occupies that position. Without this
    gate a saturated ring N-H (piperazine's second nitrogen) counted as
    indicated-H via ``GetTotalNumHs >= 1`` and out-ranked the SUFFIX-bearing
    nitrogen for locant 1, giving ``piperazine-4-carboxylic acid`` where
     NUMBERING (``the Blue Book``) criterion (c) "principal
    characteristic groups and free valences (suffixes)" — which outranks (f)
    "detachable alphabetized prefixes" — requires ``piperazine-1-carboxylic
    acid``. Partially-saturated
    mancude rings never reach here -- the ``_mancude_*`` numbering short-circuits
    ahead of the heteroatom sort in ``orient_heterocycle_with_substituents`` -- so
    the rings that survive to this test are either aromatic (unsaturated) or fully
    saturated; the gate flips only the fully-saturated ones.
    """
    atom = mol.GetAtomWithIdx(idx)
    ring_unsaturated = any(
        b.GetBondType() != Chem.BondType.SINGLE
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in ring_set and b.GetEndAtomIdx() in ring_set
    )
    if not ring_unsaturated:
        return False
    if atom.GetTotalNumHs() >= 1:
        return True
    if not atom.GetIsAromatic() or atom.GetFormalCharge() != 0:
        return False
    if atom.GetAtomicNum() not in _INDICATED_H_ELEMENTS:
        return False
    exocyclic = [nb.GetIdx() for nb in atom.GetNeighbors()
                 if nb.GetIdx() not in ring_set]
    if len(exocyclic) != 1:
        return False
    bond = mol.GetBondBetweenAtoms(idx, exocyclic[0])
    return bond is not None and bond.GetBondType() == Chem.BondType.SINGLE


def orient_heterocycle_with_substituents(
    mol,
    ring_atoms,
    substituent_positions: Optional[Set[int]] = None,
    principal_group_atoms: Optional[Set[int]] = None
) -> Tuple[List[int], Dict[int, int]]:
    """
    Orient heterocycle considering heteroatoms, the principal characteristic
    group, AND other substituent positions.

    IUPAC Rule low-locant order, applied to a ring whose senior
    heteroatom is fixed at position 1 per: after fixing the
    heteroatom at position 1, choose the numbering direction that gives lowest
    locants to, IN THIS ORDER:
    1. Other heteroatoms (if any)
    2. The principal characteristic group (suffix) [(c)]
    3. Other substituents (if the above tie) [(g)]

    ``principal_group_atoms`` is the set of RING atom indices that bear the
    principal characteristic group (e.g. the ring carbon double-bonded to =O for
    a ketone, or the ring carbon bearing -OH for an -ol). When None/empty the
    behaviour is exactly the prior heteroatom-then-substituent order.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring
        substituent_positions: Set of ring atom indices that have substituents.
                               If None, will be detected from the molecule.

    Returns:
        Tuple of:
        - oriented_ring: List of atom indices in IUPAC numbering order
        - atom_to_locant: Dict mapping atom_idx to locant (1-indexed)

    Examples:
        >>> mol = Chem.MolFromSmiles('Cc1ccccn1') # 2-methylpyridine
        >>> ring = mol.GetRingInfo.AtomRings[0]
        >>> # Find which ring atom has the methyl substituent
        >>> sub_positions = {idx for idx in ring if _has_substituent(mol, idx, ring)}
        >>> oriented, mapping = orient_heterocycle_with_substituents(mol, ring, sub_positions)
        >>> # Should give methyl the lowest possible locant (2, not 6)
    """
    ring_list = list(ring_atoms)
    ring_set = set(ring_list)
    n = len(ring_list)

    if n == 0:
        return [], {}

    # RISK 2 unification: a partially-saturated mancude heteromonocycle is
    # numbered by ONE authority carrying the ladder (a)-(f) + a canonical
    # tie-break — heteroatom cascade -> (b) indicated H `:3246` -> (c) suffix
    # `:3256` -> (e) hydro `:3288` -> (f) detachable prefixes `:3300` -> canon.
    # (g) `:3306` first-cited-prefix is NOT encoded — a KNOWN pre-existing
    # gap shared with orient's own cascade tail; see risk2-numbering-unification.md.)
    # The
    # heteroatom cascade below carried (c)/(f) but NOT the indicated-H (b) or
    # hydro (e) tiers, so it disagreed with the stem builder (which numbers with
    # (b) first) and abstained (`2H-pyran-6-carboxylic acid`, the verbatim
    # BB row `:3252`). ``_mancude_hydro_numbering`` returns the exact atom->locant
    # map the stem builder (`_mancude_hydro_name`) uses, or None for any ring that
    # is not one (aromatic / fully saturated / not mancude), leaving the cascade
    # below unchanged for those. It supersedes the cascade for partially-saturated
    # mancude rings; the added (f)/canon tiers keep decorated rings PIN-correct and
    # representation-stable (a review BLOCKER 1/2).
    _man = _mancude_hydro_numbering(mol, ring_set, principal_group_atoms)
    if _man is None:
        # Sibling for the fully-mancude indicated-H parent (2H-pyran-6-carboxylic
        # acid, the verbatim BB row at:3252): the hydro namer declines it
        # (d == max_match), but the suffix still had no indicated-H tier.
        _man = _mancude_parent_suffix_numbering(mol, ring_set, principal_group_atoms)
    if _man is not None:
        oriented_r = sorted(_man, key=lambda a: _man[a])
        return oriented_r, dict(_man)

    # Detect substituent positions if not provided
    if substituent_positions is None:
        substituent_positions = set()
        for idx in ring_list:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in ring_set:
                    substituent_positions.add(idx)
                    break

    # Get heteroatoms
    heteroatoms = []
    for idx in ring_list:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            heteroatoms.append((idx, symbol))

    if not heteroatoms:
        # No heteroatoms - return as-is
        atom_to_locant = {atom_idx: locant for locant, atom_idx in enumerate(ring_list, 1)}
        return ring_list, atom_to_locant

    # Find the best (priority, indicated-H tier) from sorted heteroatoms.
    # NUMBERING (the Blue Book) criterion (b) "indicated hydrogen":
    # a heteroatom bearing indicated H (pyrrole-type) takes
    # the lower locant; (a) element priority, (b) indicated-H tier
    # (lower=better), (c) canonical rank for deterministic tiebreaking.
    #
    # (b) used to be ``GetTotalNumHs >= 1`` -- an atom count a substituent has
    # already consumed, so an N-substituted azole nitrogen read as pyridine-type
    # and lost locant 1 to the suffix. See ``_occupies_indicated_h_position``.
    # For rings where all heteroatoms are equal-priority AND equal-H_count
    # (e.g. three sp2 N in 1,2,4-triazine) the old code fixed a single
    # start atom, producing wrong locants when that atom wasn't the optimal
    # choice. The new code collects ALL atoms sharing the best
    # (priority, h_count) tier and tries every one as position-1 candidate,
    # picking the (start, direction) pair that yields the globally minimum
    # tiered (hetero > pg > sub) locant comparison.
    _canon_rank = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    sorted_heteroatoms = sorted(
        heteroatoms,
        key=lambda x: (
            get_heteroatom_priority(x[1]),
            0 if _occupies_indicated_h_position(mol, x[0], ring_set) else 1,
            _canon_rank[x[0]],
        )
    )
    best_prio = get_heteroatom_priority(sorted_heteroatoms[0][1])
    best_hcount = (0 if _occupies_indicated_h_position(
        mol, sorted_heteroatoms[0][0], ring_set) else 1)
    # Collect all atoms that share the best (priority, h_count) tier
    start_candidates = [
        idx for idx, sym in heteroatoms
        if get_heteroatom_priority(sym) == best_prio
        and (0 if _occupies_indicated_h_position(mol, idx, ring_set) else 1) == best_hcount
    ]

    pg_set = principal_group_atoms or set()

    def _candidate_key(start_idx, direction):
        """Return the tiered comparison key for a given (start, direction)."""
        s_pos = ring_list.index(start_idx)
        oriented_r = _rotate_ring(ring_list, s_pos, direction)
        loc_map = {atom_idx: pos for pos, atom_idx in enumerate(oriented_r, 1)}
        hetero = sorted(loc_map[idx] for idx, _ in heteroatoms if idx != start_idx)
        pg = sorted(loc_map[idx] for idx in pg_set if idx in loc_map)
        sub = sorted(loc_map[idx] for idx in substituent_positions if idx in loc_map)
        canon = [_canon_rank[a] for a in oriented_r]
        return (hetero, pg, sub, canon), oriented_r

    best_key = None
    best_oriented = None
    for start_idx in start_candidates:
        for direction in (1, -1):
            key, oriented_r = _candidate_key(start_idx, direction)
            if best_key is None or key < best_key:
                best_key = key
                best_oriented = oriented_r

    oriented = best_oriented

    atom_to_locant = {atom_idx: locant for locant, atom_idx in enumerate(oriented, 1)}
    return oriented, atom_to_locant


def get_heteroatom_locants(oriented_ring: List[int], mol) -> List[Tuple[int, str]]:
    """
    Get locants and element symbols for all heteroatoms in an oriented ring.

    Args:
        oriented_ring: Ring atoms already in IUPAC numbering order
                       (position 1 at index 0)
        mol: RDKit Mol object

    Returns:
        List of (locant, element_symbol) tuples for each heteroatom,
        sorted by locant.

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ncnc1') # pyrimidine-like
        >>> ring = mol.GetRingInfo.AtomRings[0]
        >>> oriented, _ = orient_heterocycle(mol, ring)
        >>> locants = get_heteroatom_locants(oriented, mol)
        >>> # Returns something like [(1, 'N'), (3, 'N')]
    """
    result = []

    for locant, atom_idx in enumerate(oriented_ring, 1):
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            result.append((locant, symbol))

    return sorted(result, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rotate_ring(ring_list: List[int], start_pos: int, direction: int) -> List[int]:
    """
    Rotate a ring so that the atom at start_pos becomes first.

    Args:
        ring_list: List of atom indices
        start_pos: Index in ring_list to start from
        direction: 1 for clockwise, -1 for counterclockwise

    Returns:
        Reordered list with start_pos atom first
    """
    n = len(ring_list)
    if n == 0:
        return []

    result = []
    for i in range(n):
        idx = (start_pos + i * direction) % n
        result.append(ring_list[idx])

    return result


def _get_other_heteroatom_locants(
    oriented_ring: List[int],
    heteroatoms: List[Tuple[int, str]],
    start_idx: int
) -> List[int]:
    """
    Get locants for heteroatoms other than the one at position 1.

    Args:
        oriented_ring: Ring atoms in numbering order (position 1 at index 0)
        heteroatoms: List of (atom_idx, element) tuples
        start_idx: Atom index of the heteroatom at position 1 (to exclude)

    Returns:
        Sorted list of locants for other heteroatoms
    """
    # Build position lookup
    pos_map = {atom_idx: locant for locant, atom_idx in enumerate(oriented_ring, 1)}

    locants = []
    for atom_idx, _ in heteroatoms:
        if atom_idx != start_idx and atom_idx in pos_map:
            locants.append(pos_map[atom_idx])

    return sorted(locants)


# ---------------------------------------------------------------------------
# Heterocycle naming functions
# ---------------------------------------------------------------------------


def _ring_is_ortho_fused(mol, ring_atoms) -> bool:
    """True when a bond of this ring is shared with another ring.

    An ortho-fused ring is not a ring system of its own: indole contains a
    pyrrole ring and benzofuran a furan ring, yet each is its own retained
    name. Anything that reconstructs a ring's parent hydride must stop at a
    fusion bond or it starts naming indole as a substituted pyrrole.
    """
    ring_set = set(ring_atoms)
    ring_info = mol.GetRingInfo()
    for idx in ring_atoms:
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() not in ring_set:
                continue
            bond = mol.GetBondBetweenAtoms(idx, nb.GetIdx())
            if bond is not None and ring_info.NumBondRings(bond.GetIdx()) > 1:
                return True
    return False


def _ring_smiles_with_indicated_h_restored(mol, ring_atoms) -> Optional[str]:
    """Rebuild the ring key after a substituent displaced its indicated hydrogen.

     "Indicated hydrogen" (``the Blue Book``) -- a mancude azole
    carries one saturated ring position, the ``1H``. When a substituent sits
    there instead of the hydrogen, the ring atom's H count is zero, and the
    plain fragment extraction writes a bare aromatic ``n``: ``Cn1cccc1`` gives
    ``c1ccnc1``, which is not a kekulisable molecule at all. The parent hydride
    the retained-name table is keyed on still has the hydrogen, so it is put
    back before the fragment is written.

     "Substitution rules for Type 1 retained names" (``:4916``),
    sentence ``:4918``: "Type 1 retained names of parent hydrides described in
    Chapters and have unlimited substitution by substituent groups cited
    either as suffixes or prefixes." Substitution never demotes the retained
    name, so the key must survive it. The Blue Book prints such a PIN outright
    under (``:18871``) at ``:18940``:
    ``1-(trimethylsilyl)-1H-imidazole (PIN)``.

    Returns ``None`` -- and the caller then keeps its existing result byte for
    byte -- unless every one of these can be shown:

    * the ring is not ortho-fused (see ``_ring_is_ortho_fused``);
    * the atom is an aromatic, NEUTRAL group-15 ring atom with **no** hydrogen;
    * it has exactly **one** exocyclic neighbour, joined by a **single** bond --
      i.e. exactly one hydrogen's worth of valence was taken from it;
    * the rebuilt fragment parses.

    This is a reconstruction of the parent hydride, not a repair of a name: no
    string produced here is ever edited, and nothing about the ring's size or
    heteroatom count is consulted.
    """
    if _ring_is_ortho_fused(mol, ring_atoms):
        return None

    ring_set = set(ring_atoms)
    work = Chem.Mol(mol)
    restored = 0
    for idx in ring_atoms:
        atom = work.GetAtomWithIdx(idx)
        if not atom.GetIsAromatic():
            continue
        if atom.GetAtomicNum() not in _INDICATED_H_ELEMENTS:
            continue
        if atom.GetFormalCharge() != 0 or atom.GetTotalNumHs() != 0:
            continue
        exocyclic = [nb.GetIdx() for nb in atom.GetNeighbors()
                     if nb.GetIdx() not in ring_set]
        if len(exocyclic) != 1:
            continue
        bond = mol.GetBondBetweenAtoms(idx, exocyclic[0])
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        atom.SetNumExplicitHs(1)
        atom.SetNoImplicit(True)
        restored += 1

    if not restored:
        return None
    work.UpdatePropertyCache(strict=False)

    ring_smiles = Chem.MolFragmentToSmiles(work, atomsToUse=list(ring_atoms))
    ring_mol = Chem.MolFromSmiles(ring_smiles)
    if ring_mol is None:
        return None
    return Chem.MolToSmiles(ring_mol, canonical=True)


def get_ring_canonical_smiles(mol, ring_atoms) -> str:
    """
    Extract a ring as canonical SMILES for retained name lookup.

    Creates a new molecule containing only the ring atoms and their bonds,
    then returns the canonical SMILES representation.

    ``MolFragmentToSmiles`` writes each atom with the hydrogen count it has in
    ``mol``, so a ring whose indicated hydrogen has been replaced by a
    substituent comes back as a string that describes no molecule at all
    (``Cn1cccc1`` -> ``c1ccnc1``, unkekulisable). Such a string can never equal
    a table key, and the retained name is lost purely to the extraction. When
    -- and only when -- that happens, the displaced indicated hydrogen is
    restored and the fragment rewritten
    (``_ring_smiles_with_indicated_h_restored``).

    The repair is therefore **strictly additive**: a ring whose fragment already
    parses is returned exactly as before, so no key that resolves today can
    change. The only outcomes that move are keys that match nothing.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring

    Returns:
        Canonical SMILES string for the ring

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccncc1') # pyridine
        >>> ring = mol.GetRingInfo.AtomRings[0]
        >>> get_ring_canonical_smiles(mol, ring)
        'c1ccncc1'
        >>> mol = Chem.MolFromSmiles('Cn1cccc1') # 1-methyl-1H-pyrrole
        >>> ring = mol.GetRingInfo.AtomRings[0]
        >>> get_ring_canonical_smiles(mol, ring)
        'c1cc[nH]c1'
    """
    # Use MolFragmentToSmiles to extract just the ring
    # This handles aromaticity correctly
    ring_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=list(ring_atoms))

    # Canonicalize the SMILES
    ring_mol = Chem.MolFromSmiles(ring_smiles)
    if ring_mol:
        return Chem.MolToSmiles(ring_mol, canonical=True)

    restored = _ring_smiles_with_indicated_h_restored(mol, ring_atoms)
    if restored is not None:
        return restored

    return ring_smiles


def build_hw_name(
    heteroatoms: List[Tuple[int, str]],
    ring_size: int,
    is_saturated: bool,
    is_aromatic: bool,
    lambda_by_locant: Optional[Dict[int, int]] = None,
) -> Optional[str]:
    """
    Build Hantzsch-Widman systematic name for a heterocycle.

    Returns ``None`` -- fail closed -- when any heteroatom has no Table 2.4
    prefix. ``""`` still means "no heteroatoms supplied" and stays distinct.

    Assembles a systematic HW name from heteroatom prefixes and ring stem:
    1. Group heteroatoms by element
    2. Order by IUPAC priority (O > S > N >...)
    3. Add multipliers and locants for multiple same heteroatoms
    4. Get stem based on ring size and saturation
    5. Apply 'a' elision (drop terminal 'a' before vowel stem)

    Args:
        heteroatoms: List of (locant, element) tuples for heteroatoms in numbered order
        ring_size: Ring size (3-10)
        is_saturated: True if fully saturated
        is_aromatic: True if aromatic (overrides is_saturated for stem selection)

    Returns:
        HW systematic name (e.g., 'oxolane', '1,3-dioxolane', 'azine')

    Examples:
        >>> build_hw_name([(1, 'O')], 5, True, False)
        'oxolane'
        >>> build_hw_name([(1, 'O'), (3, 'O')], 5, True, False)
        '1,3-dioxolane'
        >>> build_hw_name([(1, 'N')], 6, False, True)
        'azine'
    """
    if not heteroatoms:
        return ""

    # Group heteroatoms by element: {element: [locants]}
    element_locants: Dict[str, List[int]] = {}
    for locant, elem in heteroatoms:
        if elem not in element_locants:
            element_locants[elem] = []
        element_locants[elem].append(locant)

    # Order elements by IUPAC (Hantzsch-Widman) seniority: F > Cl > Br > I >
    # O > S > Se > Te > N > P... / Table 2.8). Prefixes are cited
    # in this order.
    elements_by_priority = sorted(
        element_locants.keys(),
        key=lambda e: HETEROATOM_PRIORITY.get(e, 999)
    )

    # Collect the FULL heteroatom locant set: for a HW name with
    # more than one heteroatom, ALL locants are cited ONCE at the front of the
    # name — a single heteroatom carries no locant (oxolane, azepane). This is
    # what keeps mixed-element medium rings from dropping their locants (bare
    # 'oxazepane' is the 1,2-isomer — a different molecule).
    #
    # The ORDER of that citation is NOT a global ascending sort.
    # (the Blue Book), last sentence: "Locants are cited at the front of
    # the name, IN THE ORDER OF CITATION OF THE SKELETAL REPLACEMENT ('a')
    # PREFIXES." Set selection and set citation are two separate steps: the
    # numbering is chosen to give the lowest locant SET, then that set is spelled
    # grouped per element, each element's own locants ascending, in 'a'-prefix
    # seniority order. The rule's own example block is decisive —
    # 1,6,2-dioxazepane (PIN,:8300) dioxa owns 1 and 6, aza owns 2
    # 1,3,2-dioxaboretane (PIN,:37178) dioxa owns 1 and 3, bora owns 2
    # — and the Blue Book prints the tie-break reasoning for the first as the
    # SET "'1,2,6' is lower than '1,3,4'" while citing the name as "1,6,2-".
    #
    # A global flat sort spells those "1,2,6-" and "1,2,3-", and for an O-Si-O
    # ring it spells "1,2,3-dioxasilolane" — asserting an O-O bond that is not
    # in the structure (OPSIN rejects it: O in an unphysical valency state).
    # It is a no-op wherever the senior element already holds the lowest slot,
    # which is why the degenerate cases (1,4-oxazepane, 1,3,5-oxadiazinane,
    # 1,2,6-oxadithiepane) never exposed it.
    lambda_by_locant = lambda_by_locant or {}
    cited_locants = [
        loc for elem in elements_by_priority for loc in sorted(element_locants[elem])
    ]
    total_het = len(cited_locants)
    locant_prefix = ""
    #: a ring in which ONE heteroatom element occupies EVERY skeletal
    # position needs no heteroatom locants — the numbering is unambiguous
    # (hexasilinane, not '1,2,3,4,5,6-hexasilinane'; hexathiane). This holds only
    # when a single distinct element fills all ring positions AND no lambda is
    # present (a lambda always cites its locant,. A same-element ring
    # NOT spanning all positions (1,2-disilinane) still needs its locants.
    _one_element_all_positions = (
        len(element_locants) == 1 and total_het == ring_size
        and not lambda_by_locant
    )
    #: a lambda-bearing ring ALWAYS cites its heteroatom locants,
    # even for a single heteroatom (1lambda3-iodinane, not 'lambda3-iodinane'),
    # with the lambda token immediately after its locant (1,3lambda5-oxaphosphole).
    if (total_het > 1 or lambda_by_locant) and not _one_element_all_positions:
        from .lambda_convention import format_lambda_token
        locant_prefix = ','.join(
            format_lambda_token(loc, lambda_by_locant.get(loc))
            for loc in cited_locants
        ) + '-'

    # Build the element prefix chain in seniority order, each element carrying
    # only its di/tri/... multiplier (locants already collected above).
    prefix_parts = []
    for elem in elements_by_priority:
        hw_prefix = get_hw_prefix(elem)
        if not hw_prefix:
            # FAIL CLOSED. This used to ``continue``, which dropped the
            # heteroatom from the name while ``ring_size`` still counted it
            # toward the HW stem -- an aluminium ring came back as the bare stem
            # ``inane`` (6-membered), ``epane`` (7) or ``olane`` (5), i.e. a
            # carbocycle's name for a metallacycle. has no prefix for
            # Hg/Zn/Cd at all, so refusing is the only sound answer here.
            return None
        count = len(element_locants[elem])
        if count > 1:
            multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
            #: elide the multiplier's terminal 'a' before an 'a'
            # (aza/oxa/thia/...) term that begins with a vowel -> tetra+aza=tetraza.
            # (di/tri end in 'i'; only tetra/penta/hexa/... trigger this elision.)
            if multiplier.endswith('a') and hw_prefix and hw_prefix[0] in 'aeiou':
                multiplier = multiplier[:-1]
            prefix_parts.append(f"{multiplier}{hw_prefix}")
        else:
            prefix_parts.append(hw_prefix)

    # Join prefix parts with 'a' elision between them:
    # IUPAC rule: terminal 'a' of a prefix is elided before another vowel-starting
    # prefix (e.g., oxa + aza -> oxaza, not oxaaza)
    if prefix_parts:
        prefix = prefix_parts[0]
        for part in prefix_parts[1:]:
            if prefix.endswith('a') and part and part[0] in 'aeiou':
                prefix = prefix[:-1] + part
            else:
                prefix += part
    else:
        prefix = ""

    # Determine saturation for stem lookup
    # Aromatic = unsaturated for HW naming purposes
    saturated_for_stem = is_saturated and not is_aromatic

    # Stem selection. TWO DIFFERENT RULES, by ring size — do not merge them:
    #
    # 3-, 4- and 5-membered rings — (the Blue Book):
    # "The stems 'iridine', 'etidine', and 'olidine' are used when nitrogen
    # atoms are present in the ring; otherwise the 'ane' stems are used."
    # So the N-form is keyed on nitrogen being present ANYWHERE, not on nitrogen
    # being the senior heteroatom: 1,3-oxazolidine (O senior, N present) uses
    # -olidine, while N-free 1,3-oxathiolane keeps -olane. `stem_heteroatom`
    # below carries that "is there an N" answer, and get_hw_stem keys on it.
    #
    # 6-membered rings — (:8411): "The stem for six-membered rings
    # depends on the least senior heteroatom in the ring, i.e., the heteroatom
    # whose name directly precedes the stem." That is a property of the SET, so
    # get_hw_stem derives it itself from `ring_heteroatoms` and IGNORES
    # `stem_heteroatom` for size 6. Passing the senior element here used to
    # decide the 6-ring stem, which emitted 'oxarsane' for the O+As ring whose
    # PIN is 1,3-oxarsinane (:8455); nitrogen rings were right only because N
    # and the least senior atom happen to share Table 2.5 group B.
    has_nitrogen = 'N' in element_locants
    dominant_elem = elements_by_priority[0] if elements_by_priority else 'O'
    stem_heteroatom = 'N' if has_nitrogen else dominant_elem

    # Get stem based on ring size, saturation, and stem-driving heteroatom.
    # / Table 2.7 class 6C: pass the FULL ring heteroatom set so an
    # unsaturated 6-ring bearing a 6C atom (P/As/Sb/B/... e.g. 1,4-oxaphosphinine)
    # gets the '-inine' ending, not '-ine'.
    stem = get_hw_stem(ring_size, saturated_for_stem, stem_heteroatom,
                       ring_heteroatoms=set(element_locants.keys()))
    # Wave2: a 3-membered mancude ring with ONLY
    # nitrogen heteroatoms takes the 'irine' stem (1H-/2H-azirine,
    # 3H-diazirine), not 'irene'. get_hw_stem's single-element signature
    # cannot express "only N", so the variant is selected here where the
    # full heteroatom set is known.
    if (ring_size == 3 and not saturated_for_stem
            and set(element_locants) == {'N'}):
        stem = HW_STEMS[3].get('n_unsaturated', stem)
    if not stem:
        stem = ""

    # Apply 'a' elision: drop terminal 'a' before vowel stem
    if prefix.endswith('a') and stem and stem[0] in 'aeiou':
        prefix = prefix[:-1]

    return locant_prefix + prefix + stem


def _mancude_monocycle_parent(mol, ring_atoms):
    """Build the mancude (maximum-non-cumulative-double-bond) parent of a
    monocyclic ring by forcing every ring atom + bond aromatic and re-sanitizing.

    Returns ``(parent_mol, old_to_new_idx)`` where ``parent_mol`` is the ring-only
    aromatic parent and ``old_to_new_idx`` maps each input ring atom index to its
    index in ``parent_mol``. Returns ``(None, None)`` if the ring cannot aromatize
    (the parent is then not a mancude aromatic ring, so the hydro naming declines).
    """
    from itertools import combinations
    ring = set(ring_atoms)
    # A ring N may be pyridine-type (=N-, no H) or pyrrole-type (-NH-). RDKit
    # treats a bare aromatic N as pyridine-type, which aromatizes a 6-membered
    # azine (pyridine) but NOT a 5-membered azole (needs the pyrrole-type N-H
    # for the 6-pi count). Try the MOLECULE's actual N-H positions FIRST (so the
    # mancude parent's indicated-H aligns with the molecule's N-H — critical for
    # the 2-N azoles, e.g. 4,5-dihydro-1H-imidazole), then all-pyridine-type
    # (azines, where the N-H N is itself a hydro position, e.g. dihydropyridine),
    # then other combinations as a fallback.
    #: phosphole's ring P behaves like pyrrole's ring N (1H-phosphole)
    # — an explicit ring-H at that atom completes the mancude 6-pi aromatic count.
    ring_xh = [i for i in sorted(ring)
               if mol.GetAtomWithIdx(i).GetSymbol() in ('N', 'P')]
    mol_nh = frozenset(i for i in ring_xh if mol.GetAtomWithIdx(i).GetTotalNumHs() >= 1)
    nh_options = [mol_nh, frozenset()]
    for r in range(1, len(ring_xh) + 1):
        nh_options.extend(frozenset(c) for c in combinations(ring_xh, r))
    seen = set()
    nh_options = [o for o in nh_options if not (o in seen or seen.add(o))]
    for nh_set in nh_options:
        em = Chem.RWMol()
        old_to_new = {}
        for idx in sorted(ring):
            atom = Chem.Atom(mol.GetAtomWithIdx(idx).GetAtomicNum())
            if idx in nh_set:
                atom.SetNumExplicitHs(1)
                atom.SetNoImplicit(True)
            old_to_new[idx] = em.AddAtom(atom)
        for bond in bonds_of(mol):
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in ring and j in ring:
                em.AddBond(old_to_new[i], old_to_new[j], Chem.BondType.AROMATIC)
                em.GetAtomWithIdx(old_to_new[i]).SetIsAromatic(True)
                em.GetAtomWithIdx(old_to_new[j]).SetIsAromatic(True)
        parent = em.GetMol()
        try:
            Chem.SanitizeMol(parent)
        except Exception:
            continue
        if all(a.GetIsAromatic() for a in parent.GetAtoms()):
            return parent, old_to_new
    return None, None


def _mancude_bond_eligible(mol, oriented: List[int]) -> List[bool]:
    """Which ring positions may carry a mancude ring double bond.

    A ring atom already spends two of its valences on ring sigma bonds, so it
    can take a ring double bond only if its BONDING NUMBER is three or more.
    For an atom in a standard valence state that is a property of the element
    -- the divalent chalcogens O/S/Se/Te cannot, everything else can -- which
    is why this test used to be written on the element symbol alone. Under
    the lambda-convention it is not a property of the element: a lambda-4
    sulfur has bonding number 4 and takes a ring double bond exactly as a
    carbon does, while its standard-valence twin cannot.

    ** "Heteromonocyclic hydrides having heteroatoms with nonstandard
    bonding numbers."** (``the Blue Book``), (``:9158``) --
    "The indicated hydrogen symbol *H*, if required to denote saturated
    skeletal atoms, is cited at the front of the complete name". The decisive
    statement is the parenthetical printed under ``1lambda6-thiopyran (PIN)``
    (``:9511``), at ``:9513``: "this heteromonocycle has the maximum number of
    double bonds and one double bond at every position; hence, no indicated
    hydrogen is cited for the sulfur atom". Three things follow, and the
    whole lambda-hydro producer rests on them:

    * a lambda atom DOES join the mancude matching;
    * it takes at most ONE ring double bond (a second would be cumulative, and
      a mancude ring carries the maximum number of NONcumulative double bonds);
    * its spare valence becomes hydrogen, which is CITED as indicated hydrogen
      only when the atom carries no ring double bond -- so ``1lambda6-thiopyran``
      needs none even though its sulfur holds three hydrogens, while
      ``1H-1lambda4-thiophene`` (``:9167``) needs one.

    Outside the lambda class this is the OLD EXPRESSION VERBATIM, not a
    reimplementation of it: an atom for which ``nonstandard_bonding_number``
    returns None -- every standard-valence atom, and every charged atom, which
    the lambda helper excludes by design -- is still judged by
    ``symbol not in _DIVALENT``. So no ring without a lambda atom can change
    behaviour, and that needs no measurement to believe.
    """
    from .lambda_convention import nonstandard_bonding_number
    _DIVALENT = frozenset({'O', 'S', 'Se', 'Te'})
    ring = set(oriented)
    out: List[bool] = []
    for i in oriented:
        lam = nonstandard_bonding_number(mol, i)
        if lam is None:
            ok = mol.GetAtomWithIdx(i).GetSymbol() not in _DIVALENT
        else:
            ok = lam >= 3
        out.append(ok and _has_room_for_ring_double_bond(mol, i, ring, lam))
    return out


# Default bonding numbers, used ONLY by the exocyclic veto below and only for an
# atom that actually carries an exocyclic multiple bond. Deliberately not a
# general valence table: an element missing here simply skips the veto and keeps
# the element-based verdict above.
_STANDARD_BONDING_NUMBER = {
    'C': 4, 'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4,
    'N': 3, 'P': 3, 'As': 3, 'Sb': 3, 'Bi': 3, 'B': 3, 'Al': 3,
    'O': 2, 'S': 2, 'Se': 2, 'Te': 2,
}


def _has_room_for_ring_double_bond(mol, idx: int, ring: Set[int], lam) -> bool:
    """False when the atom's valence is already fully spent OUTSIDE the ring.

    :func:`_mancude_bond_eligible`'s own opening sentence is the whole rule --
    "A ring atom already spends two of its valences on ring sigma bonds, so it
    can take a ring double bond only if its BONDING NUMBER is three or more" --
    but it was applied as if the only other claim on an atom's valences were
    those two sigma bonds. An **exocyclic double bond** is a third claim, and
    it is not removable by hydrogenation the way a hydrogen is: in
    ``pyridin-2(1H)-one`` the C-2 carbon spends 2 valences on ring sigma bonds
    and 2 on the exocyclic ``C=O``, so it cannot carry a ring double bond at
    all, and the ring's mancude maximum is 2 rather than pyridine's 3.

    Without this, that ring reads as "2 double bonds where the parent has 3",
    i.e. as a HYDRO FORM, and the ``1 <= _d < _max_match`` guard in
    :func:`name_heterocycle` refuses it. That is what withdrew the correct
    ``5-(3-fluorophenyl)-1H-pyridin-2-one`` (regression from,
    internal notes). The compound is a
    **pseudoketone**, not a hydro form: "'Hidden' amides"
    (``the Blue Book``) and "Lactams and lactims" (``:33224``)
    both name this shape on the numbered mancude ring with an added suffix, and
    ``:33224``'s method (1) "generates preferred IUPAC names".

    ⚠ Only bonds of order **>= 2** are counted, and only to atoms OUTSIDE this
    ring. Counting single bonds would veto an N-methyl ring nitrogen, and
    counting RDKit's aromatic order 1.5 would veto the fusion carbons of
    naphthalene -- measured: both stay eligible under this test, as they must.
    An atom with no exocyclic multiple bond returns True immediately, so
    :func:`_mancude_bond_eligible` keeps its previous expression **verbatim**
    for every such atom -- which is every atom in every ring the mancude
    machinery handled before this change.
    """
    atom = mol.GetAtomWithIdx(idx)
    exo = 0.0
    for bond in atom.GetBonds():
        if bond.GetOtherAtomIdx(idx) in ring:
            continue
        order = bond.GetBondTypeAsDouble()
        if order >= 2:
            exo += order
    if exo == 0.0:
        return True
    bn = lam if lam is not None else _STANDARD_BONDING_NUMBER.get(atom.GetSymbol())
    if bn is None:
        return True  # unknown element -> keep the element-based verdict
    # 2 valences go to the ring sigma bonds; a ring double bond needs 1 more.
    return (bn - 2 - exo) >= 1


def _monocycle_numberings(mol, ordered: List[int]):
    """Every one of the ``2n`` ring numberings, keyed by the HETEROATOM cascade.

    Yields ``(het_key, loc)`` where ``loc`` maps a ring POSITION (index into
    ``ordered``) to its locant, and ``het_key`` ranks the numbering by the
    heteroatom criteria alone. Callers append their own tail to the key --
    indicated hydrogen, then hydro prefixes -- so that the heteroatom rules
    always outrank them.

     (``the Blue Book``) fixes the first two criteria, and
    their order is not the intuitive one: "The locant '1' is given to a
    heteroatom that occurs first in the seniority sequence used for citation
    of the skeletal replacement ('a') prefixes. The numbering is THEN chosen
    to give lowest locants to heteroatoms considered as a set in ascending
    numerical order." Senior-heteroatom-at-1 OUTRANKS the lowest locant set,
    which is why furazan is ``1,2,5-oxadiazole`` (``:14717``) and not
    ``2,1,3-oxadiazole``. (``:8806``) states the whole cascade
    verbatim and adds the third criterion, "and then, if necessary, according
    to the order of seniority above".

    Scoped to the Hantzsch-Widman range: above ten ring members the parent is
    named by skeletal replacement, whose numbering rule is the lowest
    heteroatom SET with no senior-at-1 clause, ``:16697``), so
    ``senior_at_one`` is neutralised there.

    Extracted so that the mancude lambda branch and the hydro namer cannot
    drift apart: the lambda branch used to inherit ``orient_heterocycle``'s
    numbering, which ranks heteroatoms but NOT indicated hydrogen, and so
    emitted ``5H-1lambda4-thiophene`` where (b) (``:3246``) requires
    ``2H-``. Neither OPSIN nor an InChIKey can see that -- both names denote
    the same molecule -- so it survived a 707/707 round-trip.
    """
    n = len(ordered)
    het_pos = [p for p in range(n)
               if mol.GetAtomWithIdx(ordered[p]).GetSymbol() != 'C']
    out = []
    for start in range(n):
        for direction in (1, -1):
            loc = {(start + k * direction) % n: k + 1 for k in range(n)}
            at_one = next(p for p in range(n) if loc[p] == 1)
            senior_at_one = (
                get_heteroatom_priority(
                    mol.GetAtomWithIdx(ordered[at_one]).GetSymbol())
                if n <= 10 else 0
            )
            het_locs = tuple(sorted(loc[p] for p in het_pos))
            seniority = tuple(sorted(
                (get_heteroatom_priority(
                    mol.GetAtomWithIdx(ordered[p]).GetSymbol()), loc[p])
                for p in het_pos
            ))
            out.append(((senior_at_one, het_locs, seniority), loc))
    return out


def _apply_retained_stem(hw_name: str) -> str:
    """: swap a Hantzsch-Widman stem for its RETAINED mancude name.

    The substitution is on the STEM -- the segment after the last hyphen --
    and it must match the WHOLE stem, never a suffix of it. A suffix test
    says ``1,2-dithiole`` ends in ``thiole`` and rewrites it to the
    non-existent ``1,2-dithiophene``, destroying the Blue Book's own
    ``1lambda4,3-dithiole (PIN)`` (``the Blue Book``); by the same route
    ``1,3-oxazole`` would become ``1,3-furan``. Whole-stem matching leaves
    both untouched, and a locant-carrying stem (``dioxine``, ``oxazole``) is
    never a key.

    ONE table, because it had drifted into three partial hand-copies that
    disagreed: ``_MANCUDE_RETAINED_STEM`` (10 rows, used by the hydro namer),
    a 3-row dict in:func:`name_heterocycle` (oxine/azine/thiine) and a 3-row
    tuple in:func:`_name_lambda_heteromonocycle` (thiole/selenole/tellurole,
    matched with ``endswith`` -- the trap above). Measured consequence on the
    27,687-ring enumeration: ``2H-azole`` was emitted where
    (``:8163``, "pyrrole (1H-isomer shown; the PIN is 1H-pyrrole)") and
    ``3,4-dihydro-2H-pyrrole (PIN)`` (``:16896``) require ``2H-pyrrole``, and
    ``1lambda4-thiine`` where ``:8141`` ("pyran... the PIN is 2H-pyran;
    thiopyran (S instead of O)") requires ``1lambda4-thiopyran``.

    All ten rows re-opened at write time: pyran/thiopyran/selenopyran/
    telluropyran ``:8141``; pyrrole ``:8163``; selenophene ``:8165``;
    tellurophene ``:8170``; thiophene ``:8174``; pyridine ``:8157``. Table
    2.2's own ``furan`` row survives only as an image in this OCR
    (``_page_149_Picture_3.jpeg``), so furan's PIN status is taken from
    ``:12047`` -- "furo (preferred prefix) (from furan, PIN)".
    """
    head, sep, stem = hw_name.rpartition('-')
    retained = _MANCUDE_RETAINED_STEM.get(stem)
    return head + sep + retained if retained else hw_name


def _mancude_max_matching(mol, oriented: List[int]) -> Tuple[int, List[bool]]:
    """Mancude maximum for a monocyclic ring given in cycle order.

    Returns ``(max_match, eligible)`` where ``max_match`` is the maximum number of
    noncumulative ring double bonds (= maximum matching on the ring cycle
    restricted to double-bond-eligible edges) and ``eligible[k]`` is the flag for
    ``oriented[k]``. Divalent chalcogens (O/S/Se/Te) never carry a mancude ring
    double bond; every other ring atom (C, N, P,...) is eligible. Ring size <= 10
    -> exact brute force over the ring edges; deterministic.

    Factored out of:func:`_monocycle_indicated_h_prefix` so the partial-saturation
    namer can reuse the exact mancude count for non-aromatizable (4-pi) HW rings.

    Eligibility is delegated to:func:`_mancude_bond_eligible`, which is
    lambda-aware; see there for why that is byte-identical outside the
    lambda-convention class.
    """
    from itertools import combinations
    n = len(oriented)
    eligible = _mancude_bond_eligible(mol, oriented)
    ok_edges = [p for p in range(n) if eligible[p] and eligible[(p + 1) % n]]
    max_match = 0
    for size in range(min(n // 2, len(ok_edges)), 0, -1):
        found = False
        for combo in combinations(ok_edges, size):
            used: Set[int] = set()
            good = True
            for p in combo:
                a, b = p, (p + 1) % n
                if a in used or b in used:
                    good = False
                    break
                used.add(a)
                used.add(b)
            if good:
                found = True
                break
        if found:
            max_match = size
            break
    return max_match, eligible


# retained mancude stems (the Blue Book — "pyran (2H-isomer
# shown; the PIN is 2H-pyran) thiopyran... the PIN is 2H-thiopyran...
# selenopyran... telluropyran"; the Blue Book — "2H-pyran (PIN) (not
# 2H-oxine, see "; the 5-ring row at:8165/:8174 gives
# "selenophene (PIN)" / "thiophene (PIN)"). EXACT-name substitution only: the
# key is the whole Hantzsch-Widman stem, so '1,3-oxazole' can never become
# '1,3-furan'.
_MANCUDE_RETAINED_STEM: Dict[str, str] = {
    'azole': 'pyrrole',
    'oxole': 'furan',
    'thiole': 'thiophene',
    'selenole': 'selenophene',
    'tellurole': 'tellurophene',
    'azine': 'pyridine',
    'oxine': 'pyran',
    'thiine': 'thiopyran',
    'selenine': 'selenopyran',
    'tellurine': 'telluropyran',
}


def _ring_perfect_matchings(n: int, allowed: Set[int], eligible: List[bool]) -> List[frozenset]:
    """Every perfect matching of the ring-cycle subgraph induced by ``allowed``.

    ``allowed`` is a set of ring POSITIONS (0..n-1). An edge joins consecutive
    positions ``p`` and ``(p+1) % n`` when both are in ``allowed`` and both are
    double-bond eligible. A *perfect* matching covers every allowed position.
    Returned as frozensets of edge start-positions. Ring size <= 10, so the
    exhaustive enumeration is cheap and deterministic.

    Used for two different structure proofs, never for counting:
      * the molecule's own ring double bonds must be the UNIQUE perfect matching
        over the unsaturated positions — otherwise the hydro name would denote
        more than one molecule and we must fail closed;
      * a candidate indicated-hydrogen set is legitimate only if removing it
        leaves a perfect matching, i.e. a real mancude parent exists.
    """
    from itertools import combinations
    if any(not eligible[p] for p in allowed):
        return []
    if not allowed:
        return [frozenset()]
    if len(allowed) % 2:
        return []
    edges = [p for p in range(n)
             if p in allowed and ((p + 1) % n) in allowed]
    out: List[frozenset] = []
    for combo in combinations(edges, len(allowed) // 2):
        used: Set[int] = set()
        ok = True
        for p in combo:
            a, b = p, (p + 1) % n
            if a in used or b in used:
                ok = False
                break
            used.add(a)
            used.add(b)
        if ok and used == allowed:
            out.append(frozenset(combo))
    return out


def _kekulized_if_aromatic(mol, ring_set) -> Optional['Chem.Mol']:
    """A copy of ``mol`` with the ring's real double bonds made explicit, or
    ``None`` if that cannot be done.

    RDKit's ``GetIsAromatic`` answers "does this ring satisfy RDKit's
    aromaticity model?". It does NOT answer "is this ring mancude?", and three
    separate sites in this module used it as though it did. The two are
    different whenever a ring atom contributes a LONE PAIR to the pi system
    instead of a double bond: two or more pyrrole-type heteroatoms make a ring
    RDKit-aromatic while it carries FEWER ring double bonds than
     (``the Blue Book``) demands of an unsaturated
    Hantzsch-Widman parent -- "Unsaturated compounds are those having the
    maximum number of noncumulative double bonds (mancude compounds) and at
    least one double bond."

    Kekulising is what makes the real count readable, and it is a no-op for a
    ring that was never RDKit-aromatic, so every caller keeps its previous
    behaviour on non-aromatic input by construction.

    Returns the ORIGINAL object when the ring is not aromatic, so callers can
    rebind unconditionally.
    """
    if not any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_set):
        return mol
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return None
    if any(kek.GetAtomWithIdx(i).GetIsAromatic() for i in ring_set):
        return None
    return kek


def _ring_double_bond_count(mol, ring_set) -> Optional[int]:
    """Number of DOUBLE bonds inside the ring once it is kekulised, or ``None``
    if the ring cannot be kekulised. See:func:`_kekulized_if_aromatic` for
    why RDKit aromaticity may not be read as unsaturation."""
    kek = _kekulized_if_aromatic(mol, ring_set)
    if kek is None:
        return None
    n = 0
    for bond in kek.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetBeginAtomIdx() in ring_set and bond.GetEndAtomIdx() in ring_set:
            n += 1
    return n


def _mancude_hydro_select(mol, ring_set: Set[int],
                          principal_group_atoms=None) -> Optional[dict]:
    """Shared numbering + saturation analysis for a partially saturated mancude
    heteromonocycle. ONE selection, consumed by both
    :func:`_mancude_hydro_name` (builds the stem string) and
    :func:`_mancude_hydro_numbering` (hands the atom->locant map to the
    composer's suffix/substituent placement).

    This is the ROOT-CAUSE fix for the "two numbering authorities" class
    (RISK 2): the indicated hydrogen OUTRANKS the suffix for low locants
    (b) ``the Blue Book``, verbatim ``2H-pyran-6-carboxylic acid
    (PIN)`` ``:3252``), so numbering the stem here and the suffix independently in
    ``orient_heterocycle_with_substituents`` (which had no indicated-H tier) let
    them disagree and abstained. read the suffix
    locant from the SAME numbering map the indicated-H prefix uses; so do we now.

     'hydro' name for a partially saturated mancude heteromonocycle,
    INCLUDING the case where the mancude parent itself needs indicated hydrogen.

    Examples this closes: ``5,6-dihydro-4H-1,3-oxazine``, ``3,4-dihydro-2H-1,4-
    oxazine``, ``3,4-dihydro-2H-pyrrole``, ``2,3-dihydro-1H-azepine``,
    ``2,3-dihydro-1,4-dioxine``. Supersedes the earlier v1 fallback, whose
    gates admitted only parents needing ZERO indicated hydrogen and a single
    heteroatom (``1,2-dihydrophosphete``).

    Rules (each opened; heading + decisive sentence):

    * ** "Hantzsch-Widman heteromonocycles"** (``the Blue Book``),
      sentence ``:24171`` — "'Hydro' prefixes added to names of fully
      unsaturated Hantzsch-Widman rings lead to preferred IUPAC names for
      partially unsaturated rings."
    * ** "General methodology"** (``:16878``), sentence ``:16880`` —
      "Indicated hydrogen atoms have priority over 'hydro' prefixes for low
      locants. If indicated hydrogen atoms are present in a name, the 'hydro'
      prefixes precede them." That single sentence fixes BOTH the numbering
      rank and the spelling order.
    * ** "NUMBERING"** (``:3219``) — decreasing seniority for low locants:
      (a) fixed numbering ``:3227``, (b) indicated hydrogen ``:3246``, …
      (e)(i) hydro/dehydro prefixes ``:3289``.
    * ** "Indicated hydrogen"** (``:3557``), sentence ``:3721`` — "in a
      preferred IUPAC name a locant and the symbol 'H' must be cited", so the
      indicated-hydrogen term is never dropped once the parent requires one.

    ⚠ The hydro positions are NOT "the parent's double-bonded atoms that lost
    their bond". The Blue Book's own ``2,7-dihydro-1H-azepine`` (``:16920``)
    refutes that reading: 1H-azepine is unsaturated at 2-3/4-5/6-7 while the
    molecule is unsaturated at 3-4/5-6, so the remaining double bonds RELOCATE
    when hydrogen is added. What is therefore verified here is the reader's
    reconstruction: saturating exactly (indicated-H ∪ hydro) must leave one and
    only one way to place the remaining noncumulative double bonds, and it must
    be the molecule's. Fail closed otherwise — never a wrong name.

    Nothing in this function counts hydrogens; saturation and every candidate
    indicated-hydrogen position are decided by matchings on the ring graph.
    """
    from .lambda_convention import nonstandard_bonding_number
    from .partial_saturation import SATURATION_PREFIXES

    n = len(ring_set)
    if n < 4 or n > 10:
        return None
    ordered = _macrocycle_ordered_ring(mol, ring_set)
    if ordered is None or len(ordered) != n:
        return None
    # (``the Blue Book``) -- "If a further choice is needed
    # between two or more of the same skeletal atom with different bonding
    # numbers, the lower locant is assigned in order of the decreasing value
    # of the bonding number" -- is NOT implemented in the numbering cascade
    # below, so a ring carrying a lambda on one of several same-element
    # heteroatoms must fail closed here rather than be numbered by a rule that
    # cannot see the bonding number. Self-defending: the lambda namer applies
    # the same guard before it delegates, but this function is also reachable
    # directly from ``name_partially_saturated_monocyclic_heterocycle``.
    for p, idx in enumerate(ordered):
        if nonstandard_bonding_number(mol, idx) is None:
            continue
        sym = mol.GetAtomWithIdx(idx).GetSymbol()
        if sum(1 for q in ordered
               if mol.GetAtomWithIdx(q).GetSymbol() == sym) > 1:
            return None
    # An RDKit-aromatic ring carries no DOUBLE ring bonds, so the saturation
    # census below cannot read it as written -- this used to ``return None``
    # for every such ring, "leaving them to the mancude namers". That is only
    # sound if RDKit-aromatic implies mancude, and it does not: see
    #:func:`_kekulized_if_aromatic`. Kekulise instead, so the census reads
    # the ring's REAL double bonds; a ring that is genuinely mancude then
    # leaves a few lines below on ``d == max_match`` exactly as before, and a
    # non-aromatic ring is untouched because the helper returns ``mol`` itself.
    mol = _kekulized_if_aromatic(mol, ring_set)
    if mol is None:
        return None  # cannot read the ring's double bonds -> fail closed
    # An exocyclic double bond makes its ring atom neither unsaturated-in-ring
    # nor a hydro position (it is a ketone / methylidene carbon,
    # 'added indicated hydrogen' territory) -> fail closed.
    for i in ring_set:
        for bond in mol.GetAtomWithIdx(i).GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                if bond.GetOtherAtomIdx(i) not in ring_set:
                    return None

    max_match, eligible = _mancude_max_matching(mol, ordered)
    if max_match < 1:
        return None
    n_eligible = sum(eligible)
    ih_count = n_eligible - 2 * max_match

    pos_of = {a: k for k, a in enumerate(ordered)}
    # (c): the ring position(s) bearing the principal characteristic group
    # (the suffix) get low locants AFTER indicated hydrogen (b) and BEFORE hydro
    # prefixes (e). Without this term the stem was numbered PCG-blind and the
    # dihydro locants disagreed with the suffix numbering the caller appends
    # (`3,6-dihydro-2H-1,4-thiazine`+`-3-carboxylic acid` vs the correct
    # `5,6-dihydro-2H`). `principal_group_atoms` may be exocyclic (a `-carboxylic
    # acid` carbon), so map to the ring atom that bears it. Default None ->
    # `pcg_pos` empty -> the key term is `` and every bare-ring name is
    # byte-identical (additive). Mirrors the carbocyclic sibling
    # `partial_saturation.py:546`, which already carries a pcg term.
    pcg_atoms = set(principal_group_atoms or ())
    # The neighbor clause counts ONLY pg atoms OUTSIDE the ring (a `-carboxylic
    # acid` carbon), so a ring atom is marked as a PCG position when it either IS
    # a ring-atom suffix (`-ol`/`-amine`: the bearing C is itself in the tuple) or
    # BEARS an exocyclic suffix. Counting the whole `pcg_atoms` in the neighbor
    # clause over-included: for `-ol`/`-amine` the tuple carries the ring bearing
    # carbon, so its two RING neighbours were spuriously marked, flipping the
    # numbering of same-element-adjacent rings (pyridazine, 1,2-dithiine) and
    # regressing them (a review-dihydro BLOCKER; a project rule).
    exo_pcg = pcg_atoms - ring_set
    pcg_pos: Set[int] = set()
    if pcg_atoms:
        for p, a in enumerate(ordered):
            if a in pcg_atoms or any(
                    nb.GetIdx() in exo_pcg
                    for nb in mol.GetAtomWithIdx(a).GetNeighbors()):
                pcg_pos.add(p)
    # The molecule's own ring double bonds, as ring-edge start positions.
    mol_edges: Set[int] = set()
    unsat_pos: Set[int] = set()
    for bond in bonds_of(mol):
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in ring_set or j not in ring_set:
            continue
        pi, pj = pos_of[i], pos_of[j]
        if (pi + 1) % n == pj:
            p = pi
        elif (pj + 1) % n == pi:
            p = pj
        else:
            return None  # not a ring-cycle edge -> refuse
        if pi in unsat_pos or pj in unsat_pos:
            return None  # cumulated ring double bonds -> refuse
        mol_edges.add(p)
        unsat_pos.update((pi, pj))
    d = len(mol_edges)
    # A genuine hydro form: some ring unsaturation left, but fewer double bonds
    # than the mancude maximum. ``d == max_match`` is the mancude parent itself
    # (indicated-hydrogen namer's job); ``d == 0`` is the saturated HW stem's.
    if not (1 <= d < max_match):
        return None
    if any(not eligible[p] for p in unsat_pos):
        return None

    sp3_pos = {p for p in range(n) if eligible[p] and p not in unsat_pos}
    n_hydro = len(sp3_pos) - ih_count
    if n_hydro < 2 or n_hydro % 2:
        return None
    prefix = SATURATION_PREFIXES.get(n_hydro)
    if prefix is None:
        return None

    # Reconstruction proof: with every (indicated-H ∪ hydro) position saturated,
    # the remaining eligible positions must admit exactly ONE perfect matching,
    # and it must be the molecule's. Otherwise the name is ambiguous -> refuse.
    reconstructed = _ring_perfect_matchings(n, set(unsat_pos), eligible)
    if len(reconstructed) != 1 or reconstructed[0] != frozenset(mol_edges):
        return None

    # Candidate indicated-hydrogen sets: a subset of the saturated positions
    # whose removal still leaves a real mancude parent (a perfect matching over
    # the rest of the eligible positions).
    from itertools import combinations
    all_eligible = {p for p in range(n) if eligible[p]}
    ih_candidates: List[frozenset] = []
    for combo in combinations(sorted(sp3_pos), ih_count):
        rest = all_eligible - set(combo)
        if _ring_perfect_matchings(n, rest, eligible):
            ih_candidates.append(frozenset(combo))
    if not ih_candidates:
        return None

    het_pos = [p for p in range(n)
               if mol.GetAtomWithIdx(ordered[p]).GetSymbol() != 'C']
    if not het_pos:
        return None

    # Numbering cascade.
    #
    # (``the Blue Book``) fixes the FIRST two criteria, and
    # their order is not the intuitive one: "The locant '1' is given to a
    # heteroatom that occurs first in the seniority sequence used for citation
    # of the skeletal replacement ('a') prefixes. The numbering is THEN chosen
    # to give lowest locants to heteroatoms considered as a set in ascending
    # numerical order." Senior-heteroatom-at-1 OUTRANKS the lowest locant set,
    # which is why furazan is ``1,2,5-oxadiazole`` (BB:14717) and not
    # ``2,1,3-oxadiazole`` even though {1,2,3} is the lower set. Ranking the set
    # first silently renumbered every N-O-N / N-S-N ring.
    #
    # (``:8806``) states the whole cascade verbatim and adds the
    # third criterion: "... the locant '1' is given to the heteroatom first
    # cited in the order of seniority... The direction of numbering is then
    # chosen to give lower locants to the heteroatoms as a set without regard to
    # the kind of heteroatom, and then, if necessary, according to the order of
    # seniority above." Hence ``senior_at_one`` -> ``het_locs`` -> ``seniority``.
    #
    # (``:3219``) then continues: (b) indicated hydrogen ``:3246`` before
    # (e)(i) hydro prefixes ``:3287`` -- restated for this exact combination by
    # (``:16879``).
    # The heteroatom half of the cascade is shared with the mancude lambda
    # branch (:func:`_monocycle_numberings`); the tail below is this namer's
    # own -- (``:3219``) (b) indicated hydrogen (``:3246``) ahead of
    # (e)(i) hydro prefixes (``:3288``/``:3289``), restated for exactly this
    # combination by (``:16880``).
    # (f) `:3300` detachable prefixes (all together, lowest locant set),
    # and an input-order-invariant canonical final tie-break. These sit BELOW
    # hydro (e) in the key, so they never alter the stem string (which depends
    # only on het/ih/hydro locants, equal across any tie) — they only settle
    # WHICH numbering the shared map returns, so a decorated ring's substituent
    # gets its (f) lowest locant deterministically instead of one chosen by
    # ring-atom iteration order (a review BLOCKER 1/2: `3-methyl-1,4-dihydropyridine`
    # not `5-methyl`; representation-stable). Detachable = non-ring heavy
    # neighbour, EXCLUDING the pcg-bearing atoms (the suffix, tier c).
    sub_pos: Set[int] = set()
    for p, a in enumerate(ordered):
        if any(nb.GetIdx() not in ring_set
               for nb in mol.GetAtomWithIdx(a).GetNeighbors()):
            sub_pos.add(p)
    sub_pos -= pcg_pos
    canon = list(Chem.CanonicalRankAtoms(mol, breakTies=True))

    best_key = None
    best: Optional[Tuple[Dict[int, int], frozenset]] = None
    for het_key, loc in _monocycle_numberings(mol, ordered):
        for ih in ih_candidates:
            ih_locs = tuple(sorted(loc[p] for p in ih))
            pcg_locs = tuple(sorted(loc[p] for p in pcg_pos))
            hydro_locs = tuple(sorted(loc[p] for p in sp3_pos - set(ih)))
            sub_locs = tuple(sorted(loc[p] for p in sub_pos))
            canon_key = tuple(canon[ordered[p]]
                              for p in sorted(range(n), key=lambda q: loc[q]))
            key = (het_key, ih_locs, pcg_locs, hydro_locs, sub_locs, canon_key)
            if best_key is None or key < best_key:
                best_key = key
                best = (loc, ih)
    if best is None:
        return None
    loc, ih = best
    return {
        'mol': mol,          # NB: kekulised copy if the ring was aromatic
        'ordered': ordered,
        'loc': loc,          # ring position -> locant
        'ih': ih,            # frozenset of ring positions carrying indicated H
        'sp3_pos': sp3_pos,  # ring positions saturated in the molecule
        'het_pos': het_pos,
        'n': n,
        'prefix': prefix,    # saturation prefix (di/tetra/hexa...)
    }


def _mancude_hydro_name(mol, ring_set: Set[int],
                        principal_group_atoms=None) -> Optional[str]:
    """ 'hydro' name for a partially saturated mancude heteromonocycle.

    Thin builder over:func:`_mancude_hydro_select`, which owns the numbering
    cascade and every fail-closed guard. Emits ``<hydro-locants>-<prefix>
    <indicated-H>-<HW stem>`` from the selected numbering, unchanged from before
    the selection was factored out (the selection is byte-identical).
    """
    from .lambda_convention import nonstandard_bonding_number

    sel = _mancude_hydro_select(mol, ring_set, principal_group_atoms)
    if sel is None:
        return None
    mol = sel['mol']
    ordered = sel['ordered']
    loc = sel['loc']
    ih = sel['ih']
    sp3_pos = sel['sp3_pos']
    het_pos = sel['het_pos']
    n = sel['n']
    prefix = sel['prefix']

    het_pairs = sorted(
        (loc[p], mol.GetAtomWithIdx(ordered[p]).GetSymbol()) for p in het_pos
    )
    # (``:9158``): a nonstandard bonding number is cited as
    # lambda^n immediately after its locant. The map is EMPTY for every
    # standard-valence ring, so this is a no-op outside the lambda class.
    lambda_by_locant: Dict[int, int] = {}
    for p in het_pos:
        lam = nonstandard_bonding_number(mol, ordered[p])
        if lam is not None:
            lambda_by_locant[loc[p]] = lam
    stem = build_hw_name(het_pairs, n, is_saturated=False, is_aromatic=False,
                         lambda_by_locant=lambda_by_locant)
    if not stem:
        return None
    stem = _apply_retained_stem(stem)

    # (:3721): in a PIN the locant and the symbol 'H' must be cited.
    if ih:
        stem = ','.join(f"{loc[p]}H" for p in sorted(ih, key=lambda q: loc[q])) + '-' + stem
    # (:16879): "the 'hydro' prefixes precede them" -> hydro locants,
    # then the indicated-hydrogen term, then the stem. A hyphen is needed only
    # when what follows starts with a digit (2,3-dihydrofuran vs
    # 5,6-dihydro-4H-1,3-oxazine).
    hydro_locants = ','.join(
        str(loc[p]) for p in sorted(sp3_pos - set(ih), key=lambda q: loc[q])
    )
    sep = '-' if stem[:1].isdigit() else ''
    return f"{hydro_locants}-{prefix}{sep}{stem}"


def _mancude_hydro_numbering(mol, ring_set: Set[int],
                             principal_group_atoms=None) -> Optional[Dict[int, int]]:
    """Atom-index -> locant map for a partially-saturated mancude heteromonocycle,
    from the SAME selection:func:`_mancude_hydro_name` uses for the stem.

    Returns ``None`` when the ring is not a partially-saturated mancude
    heteromonocycle (aromatic / fully saturated / not mancude), so the caller
    keeps its own numbering unchanged. Consulted FIRST by
    ``orient_heterocycle_with_substituents`` — this is what unifies the stem and
    the suffix onto ONE numbering authority (RISK 2): the suffix locant now
    honours (b) indicated-H-outranks-suffix exactly as the stem does
    (``3,4-dihydro-2H-pyran-6-carboxylic acid``, not ``...-2-carboxylic acid``).
    """
    sel = _mancude_hydro_select(mol, ring_set, principal_group_atoms)
    if sel is None:
        return None
    ordered = sel['ordered']
    loc = sel['loc']
    n = sel['n']
    return {ordered[p]: loc[p] for p in range(n)}


def _mancude_parent_suffix_numbering(mol, ring_set: Set[int],
                                     principal_group_atoms=None) -> Optional[Dict[int, int]]:
    """Atom-index -> locant map for a FULLY-MANCUDE monocyclic heterocycle that
    carries indicated hydrogen (2H-pyran, 4H-pyran, 6H-1,3-oxazine,...) and a
    suffix — the ``d == max_match`` sibling of:func:`_mancude_hydro_numbering`
    (which handles only the hydro forms, ``1 <= d < max_match``).

    The parent STEM is numbered by ``_monocycle_indicated_h_prefix`` (which
    minimises the indicated-H locant), but the composer's suffix locant came from
    ``orient_heterocycle_with_substituents`` with no indicated-H tier, so
    ``2H-pyran`` + suffix collided at locant 2. This numbers with the indicated
    hydrogen OUTRANKING the suffix — the verbatim BB example
    ``2H-pyran-6-carboxylic acid (PIN)`` (``the Blue Book``, (b)).

    Returns ``None`` outside its tight scope (fail-closed): aromatic rings (so
    pyridine/furan/thiophene numbering is untouched), rings that are NOT at the
    mancude maximum of ring double bonds (a hydro form, owned by the sibling), a
    ring with an exocyclic ring double bond (a ketone etc.), or a ring without
    exactly ONE indicated-hydrogen atom (multi-IH rings are a follow-on). Its
    indicated-H locant matches the stem's by construction (both minimise it), so
    stem and suffix agree.
    """
    n = len(ring_set)
    if n < 4 or n > 10:
        return None
    # Aromatic mancude rings (pyridine/furan/thiophene/...) are numbered by the
    # heteroatom cascade, not here — leave them exactly as they were.
    if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_set):
        return None
    ordered = _macrocycle_ordered_ring(mol, ring_set)
    if ordered is None or len(ordered) != n:
        return None
    # An exocyclic ring double bond is a ketone / methylidene carbon, not this
    # rule added-indicated-H territory) -> fail closed.
    for i in ring_set:
        for bond in mol.GetAtomWithIdx(i).GetBonds():
            if (bond.GetBondType() == Chem.BondType.DOUBLE
                    and bond.GetOtherAtomIdx(i) not in ring_set):
                return None
    max_match, eligible = _mancude_max_matching(mol, ordered)
    if max_match < 1:
        return None
    pos_of = {a: k for k, a in enumerate(ordered)}
    db_atoms: Set[int] = set()
    d = 0
    for bond in bonds_of(mol):
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring_set and j in ring_set:
            db_atoms.update((i, j))
            d += 1
    if d != max_match:
        return None  # hydro form (or over-perceived) -> sibling / not this rule
    # The single indicated-H atom: sp3 (no ring double bond), double-bond
    # eligible, bearing hydrogen (mirrors _monocycle_indicated_h_prefix scope).
    sp3h = [idx for idx in ordered
            if idx not in db_atoms and eligible[pos_of[idx]]
            and mol.GetAtomWithIdx(idx).GetTotalNumHs() >= 1]
    if len(sp3h) != 1:
        return None
    target = sp3h[0]
    het_pos = [p for p in range(n)
               if mol.GetAtomWithIdx(ordered[p]).GetSymbol() != 'C']
    if not het_pos:
        return None
    pcg_atoms = set(principal_group_atoms or ())
    exo_pcg = pcg_atoms - ring_set
    pcg_pos: Set[int] = set()
    if pcg_atoms:
        for p, a in enumerate(ordered):
            if a in pcg_atoms or any(
                    nb.GetIdx() in exo_pcg
                    for nb in mol.GetAtomWithIdx(a).GetNeighbors()):
                pcg_pos.add(p)
    # (f) detachable prefixes + input-invariant canonical tie-break (below
    # the suffix), so a decorated mancude parent numbers its plain substituents
    # deterministically and per lowest-locant, mirroring the hydro sibling.
    sub_pos: Set[int] = set()
    for p, a in enumerate(ordered):
        if any(nb.GetIdx() not in ring_set
               for nb in mol.GetAtomWithIdx(a).GetNeighbors()):
            sub_pos.add(p)
    sub_pos -= pcg_pos
    canon = list(Chem.CanonicalRankAtoms(mol, breakTies=True))

    #: (heteroatom cascade) -> (b) indicated H -> (c) suffix -> (f) prefixes.
    best_key = None
    best_loc = None
    for het_key, loc in _monocycle_numberings(mol, ordered):
        ih_loc = loc[pos_of[target]]
        pcg_locs = tuple(sorted(loc[p] for p in pcg_pos))
        sub_locs = tuple(sorted(loc[p] for p in sub_pos))
        canon_key = tuple(canon[ordered[p]]
                          for p in sorted(range(n), key=lambda q: loc[q]))
        key = (het_key, ih_loc, pcg_locs, sub_locs, canon_key)
        if best_key is None or key < best_key:
            best_key = key
            best_loc = loc
    if best_loc is None:
        return None
    return {ordered[p]: best_loc[p] for p in range(n)}


def _aromatizable_hydro_name(mol, ring_atoms, ring_set, mol_unsat,
                             principal_group_atoms=None) -> Optional[str]:
    """Hydro name via the RDKit-AROMATIZABLE mancude parent (1,2-dihydropyridine,
    2,3-dihydro-1H-pyrrole).

    Split out of:func:`name_partially_saturated_monocyclic_heterocycle` so that
    every one of its fail-closed exits can hand off to the matching-based
    :func:`_mancude_hydro_name` instead of abstaining outright. Behaviour of
    this path is unchanged; only what happens after it declines is new.
    """
    from .partial_saturation import SATURATION_PREFIXES

    n = len(ring_set)
    parent, old_to_new = _mancude_monocycle_parent(mol, ring_atoms)
    if parent is None:
        # A 4-pi / indicated-hydrogen parent can never be RDKit-aromatized.
        # The caller falls through to the matching-based namer.
        return None
    parent_ring = parent.GetRingInfo().AtomRings()
    if len(parent_ring) != 1:
        return None

    # Atoms that hold a ring double bond in the mancude parent (kekulised).
    kp = Chem.Mol(parent)
    try:
        Chem.Kekulize(kp, clearAromaticFlags=True)
    except Exception:
        return None
    parent_db_atoms: Set[int] = set()
    for bond in kp.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            parent_db_atoms.add(bond.GetBeginAtomIdx())
            parent_db_atoms.add(bond.GetEndAtomIdx())

    # Hydro positions = ring atoms in a mancude-parent ring double bond that have
    # lost it (saturated) in the molecule.
    hydro = {
        i for i in ring_set
        if old_to_new[i] in parent_db_atoms and i not in mol_unsat
    }
    if not hydro:
        return None
    prefix = SATURATION_PREFIXES.get(len(hydro))
    if prefix is None:
        return None

    # Name the mancude parent — DEFERRED until after the hydro check so a fully
    # mancude ring (hydro == {}) never reaches this recursive name_heterocycle
    # call (the all-aromatic guard above also blocks re-entry). An intrinsic
    # indicated-H parent (1H-pyrrole, 1H-imidazole,...) is KEPT: the hydro
    # positions are the saturated C's; the indicated-H N is never in a parent
    # ring double bond so it is never counted as hydro + the parent's
    # own nH-) -> e.g. 2,3-dihydro-1H-pyrrole.
    parent_name = name_heterocycle(parent, parent_ring[0])
    if not parent_name:
        return None

    # Number the ring: lowest locants to the heteroatom set, then heteroatom
    # seniority, then the hydro set. Enumerate all 2N numberings.
    ordered = _macrocycle_ordered_ring(mol, ring_set)
    if ordered is None or len(ordered) != n:
        return None
    het = {i for i in ordered if mol.GetAtomWithIdx(i).GetSymbol() != 'C'}
    # The indicated-hydrogen atom(s) of the kept parent must take the LOWEST
    # locant, ranked ahead of the hydro set.
    #
    # "NUMBERING" (the Blue Book) fixes the order: "low locants are
    # assigned to them in the following decreasing order of seniority...
    # (b) indicated hydrogen for unsubstituted compounds" (:3246)... "(e)
    # saturation/unsaturation: (i) low locants are given to hydro/dehydro
    # prefixes... and 'ene' and 'yne' endings" (:3288). "General
    # methodology" (:16878) states the consequence outright, and — as so often
    # — in the LAST two sentences of its paragraph (:16880): "Indicated
    # hydrogen atoms have priority over 'hydro' prefixes for low locants. If
    # indicated hydrogen atoms are present in a name, the 'hydro' prefixes
    # precede them."
    #
    # WHICH atom carries the indicated hydrogen is a property of the mancude
    # PARENT, not of the molecule: per (:8320) it is the ring atom
    # "with a bonding number of three or higher connected to adjacent ring
    # atoms by single bonds only, and carrying one or more hydrogen atoms" —
    # i.e. a parent ring atom that holds no parent ring double bond and still
    # has an H. Deriving it from the parent also makes it element-agnostic
    # (2H-pyran's is a CARBON, 2,3-dihydro-1H-phosphole's a PHOSPHORUS).
    #
    # This used to be guessed from the MOLECULE instead, as "any ring N bearing
    # an H". That guess cannot tell two N-H apart, so for every 2-N parent
    # whose hydro form saturates the second nitrogen — pyrazole, imidazole and
    # all four diazonines — the tie-break went slack and the hydro criterion,
    # ranked below it, was free to put locant 1 on a HYDRO nitrogen. The result
    # cited locant 1 as a hydro position and as indicated hydrogen in the same
    # name ('1,2,6,7,8,9-hexahydro-1H-1,5-diazonine', '1,5-dihydro-1H-pyrazole').
    # A position can never be both: an indicated-hydrogen atom holds no double
    # bond in the parent, so it has none to lose. Every (PIN) example in
    # / keeps the two sets disjoint — 4,5-dihydro-3H-azepine
    # (:16888), 3,4-dihydro-2H-pyrrole (:16896), 2,7-dihydro-1H-azepine
    # (:16920), 2,3-dihydro-1H-phosphole (:16924).
    #
    # (The rule this block used to cite,, is "Bi- and polycyclic von
    # Baeyer structures with both double and triple bonds" (:16675) and governs
    # none of this.)
    new_to_old = {v: k for k, v in old_to_new.items()}
    indicated_atoms = {
        new_to_old[p]
        for p in parent_ring[0]
        if p in new_to_old
        and p not in parent_db_atoms
        and parent.GetAtomWithIdx(p).GetTotalNumHs() >= 1
    }
    best_key = None
    best_map: Optional[Dict[int, int]] = None
    _pcg_atoms = set(principal_group_atoms or ())
    # Only EXOCYCLIC pg atoms participate in the neighbor clause (see the twin
    # comment in _mancude_hydro_name): a ring-atom suffix self-marks, an exocyclic
    # suffix marks its ring-attachment atom, and a ring bearing-carbon in the
    # tuple never leaks onto its ring neighbours (a review-dihydro BLOCKER).
    _exo_pcg = _pcg_atoms - set(ring_atoms)
    for start in range(n):
        for direction in (1, -1):
            seq = [ordered[(start + k * direction) % n] for k in range(n)]
            loc = {a: idx + 1 for idx, a in enumerate(seq)}
            # (the Blue Book): "The locant '1' is given to a
            # heteroatom that occurs first in the seniority sequence... The
            # numbering is THEN chosen to give lowest locants to heteroatoms
            # considered as a set." Senior-at-1 outranks the lowest set. Without
            # it the hydro locants were numbered by the set rule while
            # ``parent_name`` came back numbered by the seniority rule, and the
            # two disagreed: C1=NONC1 was emitted as
            # '1,5-dihydro-1,2,5-oxadiazole', which hydrogenates the ring OXYGEN.
            #
            # Scoped to the Hantzsch-Widman range covers rings of 3 to
            # 10). Above it the parent is named by skeletal replacement ('a')
            # nomenclature, whose numbering rule is the lowest heteroatom SET
            #,:16697) with no senior-at-1 clause -- and those parent
            # names really do carry locant 1 on a junior heteroatom
            # (7-oxa-1,4,11-triazacyclopentadeca-...). Applying the HW rule there
            # re-created the very mismatch this fixes.
            senior_at_one = (
                get_heteroatom_priority(mol.GetAtomWithIdx(seq[0]).GetSymbol())
                if n <= 10 else 0
            )
            het_locs = tuple(sorted(loc[a] for a in het))
            seniority = tuple(sorted(
                (get_heteroatom_priority(mol.GetAtomWithIdx(a).GetSymbol()), loc[a])
                for a in het
            ))
            ih_locs = tuple(sorted(loc[a] for a in indicated_atoms))
            # (c): PCG/suffix ring atoms get low locants after indicated
            # hydrogen (b) and before hydro prefixes (e). Ring atoms bearing the
            # (possibly exocyclic) principal group; default None -> `` -> every
            # bare-ring/substituent name byte-identical (additive).
            pcg_locs = tuple(sorted(
                loc[a] for a in ring_atoms
                if a in loc and (a in _pcg_atoms or any(
                    nb.GetIdx() in _exo_pcg
                    for nb in mol.GetAtomWithIdx(a).GetNeighbors()))))
            hydro_locs = tuple(sorted(loc[a] for a in hydro))
            key = (senior_at_one, het_locs, seniority, ih_locs, pcg_locs, hydro_locs)
            if best_key is None or key < best_key:
                best_key = key
                best_map = loc
    if best_map is None:
        return None

    # Two structural post-conditions on the name we are about to concatenate.
    # Both are invariants of /, not preferences, so violating
    # either means the numbering disagrees with the parent name being appended
    # and the composite would be malformed. Fail closed (the caller falls
    # through to _mancude_hydro_name) rather than emit it.
    hydro_locants = sorted(best_map[a] for a in hydro)
    ih_locants = {best_map[a] for a in indicated_atoms}
    # (1) A hydro locant can never also be an indicated-hydrogen locant.
    if ih_locants & set(hydro_locants):
        return None
    # (2) The locants this numbering gives the parent's indicated-hydrogen
    # atoms must be exactly the ones parent_name already cites, or the
    # 'nH-' in the appended parent name points at a different ring atom
    # than the hydro prefixes do.
    cited = re.match(r'^((?:\d+H,)*\d+H)-', parent_name)
    cited_ih = {int(t) for t in re.findall(r'\d+', cited.group(1))} if cited else set()
    if cited_ih != ih_locants:
        return None

    locant_str = ','.join(str(loc) for loc in hydro_locants)
    # Hyphen before a digit-initial (intrinsic-IH) parent: 2,3-dihydro-1H-pyrrole.
    sep = '-' if parent_name[:1].isdigit() else ''
    return f"{locant_str}-{prefix}{sep}{parent_name}"


def name_partially_saturated_monocyclic_heterocycle(
        mol, ring_atoms, principal_group_atoms=None) -> Optional[str]:
    """Name a partially-saturated monocyclic mancude heterocycle (IUPAC.

    Example: ``C1C=CC=CN1`` -> ``1,2-dihydropyridine`` (the systematic HW path
    would otherwise drop the hydrogenation and emit the mancude parent
    ``pyridine``). Works by (1) building the mancude aromatic parent and naming
    it, (2) finding the ring atoms that carry a ring double bond in the mancude
    parent but have LOST it in the molecule (the hydro positions — detected by
    bond topology, NOT hybridization, so a conjugated enamine N still counts),
    (3) numbering the ring with the heteroatom set lowest then the hydro set
    lowest, and (4) emitting ``<locants>-<prefix>hydro<parent>``.

    Scope (fail closed otherwise — never a wrong name, -safe):
      * the mancude parent must aromatize and carry NO leading indicated
        hydrogen (parents like ``1H-pyrrole`` / ``2H-pyran`` whose hydro form
        interleaves added-indicated-H are deferred to a follow-on);
      * the molecule must retain >= 1 ring unsaturation (a fully saturated ring
        is named by its retained / HW saturated stem, not as a hydro prefix);
      * the hydro count must be a standard di/tetra/hexa... value.
    """

    ring_set = set(ring_atoms)
    n = len(ring_set)
    if n < 4:
        return None
    # a phase (A): scope to the Hantzsch-Widman ring-size range this
    # function's "mancude parent" premise applies to (build_hw_name / the
    # sibling _name_lambda_heteromonocycle both cap at 10 -- the Blue Book
    # 's HW stems are defined for rings of size 3-10 only).
    # Without this bound, a >10-membered partially-saturated heterocycle
    # (e.g. an unsaturated macrolactone ring) fell through to the mancude
    # "max non-cumulated double bonds + hydro" scheme below and emitted a
    # WRONG hyper-unsaturated name (measured: a 13-membered ring with ONE
    # real ring C=C came back as a 6-double-bond "...hexaene" + decahydro
    # prefix). name_heterocycle's own `ring_size > 10` branch (replacement
    # nomenclature via _orient_macrocycle_for_replacement) already numbers
    # the ACTUAL ring double bond(s) correctly and is reached once this
    # declines.
    if n > 10:
        return None

    # Ring atoms still unsaturated in the MOLECULE (ring double bond or aromatic)
    # — these are NOT hydro positions.
    mol_unsat: Set[int] = set()
    for bond in bonds_of(mol):
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in ring_set and j in ring_set:
                mol_unsat.add(i)
                mol_unsat.add(j)
    for i in ring_set:
        if mol.GetAtomWithIdx(i).GetIsAromatic():
            mol_unsat.add(i)
    if not mol_unsat:
        # Fully saturated -> retained / HW saturated stem handles it.
        return None
    # "A fully-aromatic ring is mancude, so it has NO hydro positions" was the
    # premise here, and it is FALSE -- see:func:`_kekulized_if_aromatic`.
    # Two or more pyrrole-type heteroatoms donate lone pairs instead of double
    # bonds, so ``c1c[nH][nH]1``, ``c1cc[nH]cc[nH]c1`` and ``c1ccc[nH][nH]cc1``
    # are all RDKit-aromatic while carrying one ring double bond FEWER than
    # their mancude parent. Returning None here dropped them through to the
    # plain Hantzsch-Widman path, which named each as that parent --
    # 1,2-diazete / 1,4-diazocine / 1,2-diazocine -- a DIFFERENT MOLECULE by
    # (``the Blue Book``). "Hantzsch-Widman
    # heteromonocycles" (``:24169``) gives the correct form at ``:24171``:
    # "'Hydro' prefixes added to names of fully unsaturated Hantzsch-Widman
    # rings lead to preferred IUPAC names for partially unsaturated rings."
    # Measured over 397,371 enumerated bare heteromonocycles: 133 rings, of
    # sizes 4/7/8, every one of them RDKit-aromatic.
    #
    # Hand it to the matching-based namer, NOT to _aromatizable_hydro_name --
    # that is what keeps the infinite recursion this bail used to guard
    # impossible. _mancude_hydro_name builds its stem with build_hw_name and
    # never calls name_heterocycle, whereas _aromatizable_hydro_name does, and
    # the "mancude parent" of an aromatic ring can be the ring itself.
    # _mancude_hydro_name returns None unless ``1 <= d < max_match``, so a
    # genuinely mancude aromatic ring still declines here exactly as before.
    if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_set):
        return _mancude_hydro_name(mol, ring_set, principal_group_atoms)

    # (1) the RDKit-aromatizable mancude parent (pyridine, 1H-pyrrole,...).
    named = _aromatizable_hydro_name(mol, ring_atoms, ring_set, mol_unsat,
                                     principal_group_atoms)
    if named:
        return named
    # (2) matching-based namer: parents that cannot aromatize, and
    # parents that themselves require indicated hydrogen (4H-1,3-oxazine,
    # 2H-pyran, 2H-pyrrole) or carry a multi-heteroatom locant prefix
    # (1,4-dioxine). Fail-closed; returns None rather than a mancude name.
    return _mancude_hydro_name(mol, ring_set, principal_group_atoms)


def _name_lambda_heteromonocycle(mol, oriented, heteroatom_locants, info) -> Optional[str]:
    """: heteromonocycle with nonstandard-bonding-number heteroatom(s).

    Emits '<locants-with-lambda>-<HW name>' with indicated hydrogen, and hands
    HYDRO forms to:func:`_mancude_hydro_name`.
    Fail-closed contract — returns None for: rings >10 / aromatic-perceived
    rings; lambda on a skeletal CARBON; more than one atom of the lambda
    atom's element tie-break unbuilt, ``:9515``); cumulated ring
    double bonds; a ring double bond on an atom that cannot carry one; more
    ring double bonds than the mancude parent has; >1 indicated-H position.

    MEASURED RESIDUE (Task AA re-measurement; enumerate 27,687 bare
    heteromonocycles: sizes 3-14 x N/O/S/O+N/S+N/2N at every heteroatom
    position x every independent-edge set of the ring). Of the 2,848 rings of
    size <= 10, **707 abstained at producer level and all 707 fell on ONE code
    branch** — the "hydro form of a lambda ring" refusal. All 707 are now
    named (0 lost, 0 wrong; 707/707 OPSIN round-trip to the input structure).
    It spans sizes 3-10 (2/4/13/24/53/96/185/330), so it is not a boundary
    effect: c3413078 only made sizes 7-10 *visible* by routing them here; sizes
    3-6 abstained before it and were untouched by it. (The other half of the
    residue recorded in 06093de6 — "4 nine-membered 2-N rings" — was never a
    producer abstention at all. Those emitted a malformed name that only the
    OPSIN validity gate suppressed downstream; the real class was 51 rings and
    is fixed in _aromatizable_hydro_name.)

    ⚠ The residue statement above is the shape of the class, but its claim
    that all 707 "fail on ONE branch" is true only of the CODE. In
    NOMENCLATURE they are two classes, and the single ``is any saturated ring
    atom a carbon?`` test merged them:

    * **60** are not hydro forms at all. They are mancude lambda parents
      whose indicated hydrogen happens to sit on a CARBON — which is the Blue
      Book's own ``3H-1lambda4-thiophene (PIN)`` (``:9171``), the companion of
      ``1H-1lambda4-thiophene (PIN)`` (``:9167``) whose indicated hydrogen is
      on the sulfur and which this producer already emitted. So the refusal
      was rejecting a verbatim (PIN) example. Sizes 3/5/7/9 (2/8/18/32).
    * **647** are genuine hydro forms, sizes 4-10 (4/5/24/35/96/153/330).

    Both are licensed. "Intramolecular amides of amino sulfinic
    acids." (``:33282``) gives ``3,4,5,6-tetrahydro-1<lambda>4,2-thiazin-1-ol
    (PIN)`` (``:33292``); **** "Sultims are tautomers of sultams
    and are named as described in... using the term 'sultim'"
    (``:33264``) gives ``1-hydroxy-4,5-dihydro-3H-1<lambda>6,2-thiazol-1-one
    (PIN)`` (``:33268``) — hydro prefixes on a lambda Hantzsch-Widman ring,
    one also carrying indicated hydrogen, both (PIN). (That second one was
    recorded here as; re-opened, the heading above it is
    .) A third, in another chapter, shows the same combination on
    a phosphorus ring: ``1,1,3,3-tetraphenyl-4,5-dihydro-1H-1,2,3lambda5-
    triphosphol-3-ium (PIN)`` (``:41403``).

    ⚠ Building it did NOT need (h) (``:3320``), which this docstring
    used to say it did. (h) disambiguates "the same skeletal atom in
    different valence states"; measured over the enumeration, all 707 rings
    carry exactly ONE lambda atom and it is lambda-4 in every one, so the
    criterion can never fire for this class. The guard above refuses the
    multi-same-element case outright instead. What it DID need was
    ``_mancude_max_matching`` made lambda-aware — see
    :func:`_mancude_bond_eligible`.
    """
    from .lambda_convention import nonstandard_bonding_number
    ring_size = info['ring_size']
    if ring_size > 10 or info.get('is_aromatic'):
        return None
    loc_of = {idx: pos + 1 for pos, idx in enumerate(oriented)}
    lambda_by_locant = {}
    lam_elem_values = {}
    for idx in oriented:
        lam = nonstandard_bonding_number(mol, idx)
        if lam is None:
            continue
        sym = mol.GetAtomWithIdx(idx).GetSymbol()
        if sym == 'C':
            return None  # lambda carbon skeleton: out of scope, fail closed
        lam_elem_values.setdefault(sym, set()).add(lam)
        lambda_by_locant[loc_of[idx]] = lam
    if not lambda_by_locant:
        return None
    # (same element, different bonding numbers) not implemented:
    # the lower-locant-to-higher-lambda tie-break is unbuilt -> refuse.
    for sym, vals in lam_elem_values.items():
        n_same_elem = sum(
            1 for i in oriented if mol.GetAtomWithIdx(i).GetSymbol() == sym
        )
        if n_same_elem > 1 and len(vals) >= 1:
            return None
    ih_prefix = ''
    if not info['is_saturated']:
        ring_set = set(oriented)
        in_double = set()
        n_double = 0
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in ring_set and j in ring_set and bond.GetBondTypeAsDouble() >= 2.0:
                if i in in_double or j in in_double:
                    return None  # cumulated ring double bonds -> not mancude
                in_double.update((i, j))
                n_double += 1
        sp3 = [idx for idx in oriented if idx not in in_double]
        # How many noncumulative ring double bonds the MANCUDE PARENT carries,
        # counted lambda-aware (a lambda-4 sulfur is matchable; see
        # _mancude_bond_eligible). Comparing the molecule against that number
        # is what separates the two nomenclature classes that used to be
        # merged behind a single "is any saturated ring atom a carbon?" test.
        max_match, eligible = _mancude_max_matching(mol, oriented)
        pos_of = {idx: p for p, idx in enumerate(oriented)}
        if any(not eligible[pos_of[idx]] for idx in in_double):
            return None  # a ring double bond on an atom that cannot hold one
        if n_double > max_match:
            return None  # more double bonds than a mancude parent has
        if n_double < max_match:
            # HYDRO FORM of a lambda ring. "Intramolecular
            # amides of amino sulfinic acids." (``the Blue Book``) gives
            # ``3,4,5,6-tetrahydro-1lambda4,2-thiazin-1-ol (PIN)`` (``:33292``)
            # and "Sultims are tautomers of sultams..."
            # (``:33264``) gives
            # ``1-hydroxy-4,5-dihydro-3H-1lambda6,2-thiazol-1-one (PIN)``
            # (``:33268``) -- hydro prefixes on a lambda Hantzsch-Widman ring,
            # one of them also carrying indicated hydrogen, both marked (PIN).
            # A third, in a different chapter: ``1,1,3,3-tetraphenyl-4,5-
            # dihydro-1H-1,2,3lambda5-triphosphol-3-ium (PIN)`` (``:41403``).
            #
            # Hand it to the general matching-based hydro namer, which already
            # implements the (b)-over-(e)(i) numbering and the
            # unique-reconstruction proof; it became lambda-capable when
            # _mancude_max_matching did. Delegating from HERE, after the
            # guards above, is what keeps the same-element case
            # fail-closed.
            return _mancude_hydro_name(mol, ring_set)
        # n_double == max_match: the molecule IS the mancude parent, so the
        # only thing to add is indicated hydrogen.
        #
        # RE-NUMBER first. ``oriented`` arrives from ``orient_heterocycle``,
        # which ranks the heteroatom criteria but knows nothing of indicated
        # hydrogen, so where those criteria TIE it picks arbitrarily between
        # numberings that "NUMBERING" (``:3219``) separates: "low
        # locants are assigned to them in the following decreasing order of
        # seniority... (b) indicated hydrogen for unsubstituted compounds"
        # (``:3246``). Measured on the 60 mancude lambda rings in the
        # enumeration, 5 came out non-minimal -- ``5H-1lambda4-thiophene`` for
        # C1=CC[SH]=C1 where ``2H-`` is reachable, and the same in thiepine,
        # thionine. Invisible to both available structural oracles, because
        # the two numberings describe the SAME molecule.
        ih_of = {}
        for p, idx in enumerate(oriented):
            if idx in in_double:
                continue
            if mol.GetAtomWithIdx(idx).GetTotalValence() - 2 >= 1:
                ih_of[p] = True
        best = None
        for het_key, cand in _monocycle_numberings(mol, oriented):
            key = (het_key, tuple(sorted(cand[p] for p in ih_of)))
            if best is None or key < best[0]:
                best = (key, cand)
        loc_of = {oriented[p]: L for p, L in best[1].items()}
        lambda_by_locant = {
            loc_of[idx]: nonstandard_bonding_number(mol, idx)
            for idx in oriented
            if nonstandard_bonding_number(mol, idx) is not None
        }
        heteroatom_locants = sorted(
            (loc_of[idx], mol.GetAtomWithIdx(idx).GetSymbol())
            for idx in oriented
            if mol.GetAtomWithIdx(idx).GetSymbol() != 'C'
        )
        ih_locs = []
        for idx in sp3:
            # The indicated-hydrogen position is decided by BONDING NUMBER,
            # which is exactly ``GetTotalValence`` -- the same quantity
            # ``nonstandard_bonding_number`` measures -- so it needs no lookup
            # table and is element-agnostic, as (``:8320``)
            # requires: the ring atom "with a bonding number of three or
            # higher connected to adjacent ring atoms by single bonds only,
            # and carrying one or more hydrogen atoms".
            #
            # It must NOT be read out of ``STANDARD_BONDING_NUMBER``: that
            # table deliberately omits carbon (adding 'C' there would give
            # every radical carbon a spurious lambda3), so a carbon scored 0
            # and never qualified. Harmless while a saturated ring carbon was
            # refused outright one branch above; the moment that refusal was
            # lifted it silently dropped the '3H-' from the Blue Book's own
            # ``3H-1lambda4-thiophene (PIN)`` (``:9171``).
            capacity = mol.GetAtomWithIdx(idx).GetTotalValence()
            if capacity - 2 >= 1:  # 2 ring sigma bonds; spare valence -> H position
                ih_locs.append(loc_of[idx])
        if len(ih_locs) > 1:
            return None  # multi-indicated-H lambda rings: out of scope
        if ih_locs:
            ih_prefix = f"{ih_locs[0]}H-"
    hw = build_hw_name(
        heteroatom_locants, ring_size, info['is_saturated'], False,
        lambda_by_locant=lambda_by_locant,
    )
    if not hw:
        return None
    # retained stems (BB example: 1H-1lambda4-THIOPHENE at ``:9167``,
    # not 1H-1lambda4-thiole). Shared whole-stem helper: the three rows that
    # used to be inlined here missed 'thiine' -> 'thiopyran', so
    # ``1lambda4-thiine`` was emitted for C1=CC=[SH]C=C1, and they were matched
    # with ``endswith``, which would have rewritten the Blue Book's own
    # ``1lambda4,3-dithiole (PIN)`` (``:9527``) to '...dithiophene' the moment
    # the guard above stopped refusing two-sulfur rings.
    return ih_prefix + _apply_retained_stem(hw)


def name_heterocycle(mol, ring_atoms, principal_group_atoms=None) -> Optional[str]:
    """
    Generate IUPAC name for a heterocyclic ring.

    Returns ``None`` when a ring heteroatom has no replacement prefix in the
    governing table, so no name can express it (see ``build_hw_name``).

    Naming priority:
    1. Check retained names FIRST (pyridine, furan, morpholine, etc.)
    2. For rings 3-10: Hantzsch-Widman systematic naming
    3. For rings > 10: replacement ("a") nomenclature on cycloalkane parent

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring

    Returns:
        IUPAC name for the heterocycle

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccncc1') # pyridine
        >>> ring = mol.GetRingInfo.AtomRings[0]
        >>> name_heterocycle(mol, ring)
        'pyridine'
        >>> mol2 = Chem.MolFromSmiles('C1CO1') # oxirane
        >>> ring2 = mol2.GetRingInfo.AtomRings[0]
        >>> name_heterocycle(mol2, ring2)
        'oxirane'
    """
    # Get ring canonical SMILES for retained name lookup
    ring_smiles = get_ring_canonical_smiles(mol, ring_atoms)

    # Check retained names FIRST
    retained = get_retained_name(ring_smiles)
    if retained:
        return retained

    #: lambda-convention heteromonocycle. Must run BEFORE the
    # partial-saturation and plain-HW paths so a hypervalent ring never
    # silently names as its standard-valence parent ('thiophene'/'2,3-dihydro-
    # thiophene' for [SH2]1C=CC=C1 / [SH2]1CCC=C1 — different molecules).
    from .lambda_convention import nonstandard_bonding_number as _nsbn
    _oriented_pre, _ = orient_heterocycle(mol, ring_atoms)
    if _oriented_pre and any(_nsbn(mol, i) is not None for i in _oriented_pre):
        _info_pre = classify_heterocycle(mol, ring_atoms)
        _het_pre = get_heteroatom_locants(_oriented_pre, mol)
        return _name_lambda_heteromonocycle(
            mol, _oriented_pre, _het_pre, _info_pre
        )

    # Partially-saturated mancude monocyclic heterocycle (e.g. 1,2-dihydro-
    # pyridine): the systematic HW path below drops the hydrogenation and emits
    # the mancude parent ('pyridine'). Detect + name the hydro form first
    #; fails closed for anything it cannot number correctly.
    partial = name_partially_saturated_monocyclic_heterocycle(
        mol, ring_atoms, principal_group_atoms)
    if partial:
        return partial

    # Fall back to systematic naming
    info = classify_heterocycle(mol, ring_atoms)
    oriented, _ = orient_heterocycle(mol, ring_atoms)

    # Get heteroatom locants from oriented ring
    heteroatom_locants = get_heteroatom_locants(oriented, mol)

    ring_size = info['ring_size']

    # For rings > 10: use replacement nomenclature (cycloXXXane parent).
    # V-6 /: enumerate the actual ring double bonds (kekulising a
    # conjugated macrocycle RDKit reads as aromatic) so a polyene is named with
    # ALL its bonds + locants instead of a single bare 'ene'.
    if ring_size > 10:
        macro_het, double_locants = _orient_macrocycle_for_replacement(
            mol, ring_atoms, principal_group_atoms)
        if macro_het is None:
            # Could not enumerate -> fall back to the heteroatom-only orientation.
            macro_het, double_locants = heteroatom_locants, None
        # ``bare_ring``: the molecule IS this ring, so no substituent, suffix or
        # unsaturation locant can be cited alongside the heteroatom's -- the only
        # condition under which 's omission of the sole heteroatom
        # locant is provably unambiguous. Anything exocyclic (a ring ketone, an
        # ``-ol``, a substituent) leaves this False and the locant is kept, which
        # is what the gold ``1-selenacyclotridecan-3-one`` requires.
        return _build_replacement_name(
            macro_het,
            ring_size,
            info['is_saturated'],
            double_locants=double_locants,
            bare_ring=(mol.GetNumHeavyAtoms() == ring_size),
        )

    # The degree of hydrogenation the STEM is about to assert, checked against
    # the structure. (``the Blue Book``) defines what an
    # unsaturated Hantzsch-Widman stem means -- "Unsaturated compounds are
    # those having the maximum number of noncumulative double bonds (mancude
    # compounds) and at least one double bond" -- so spelling one for a ring
    # that holds fewer than the mancude maximum names a different molecule.
    # Both flags reaching here are derived from RDKit aromaticity, which is not
    # the same question (:func:`_kekulized_if_aromatic`), so re-derive from the
    # kekulised ring.
    _d = _ring_double_bond_count(mol, set(oriented))
    if _d is not None:
        _max_match, _ = _mancude_max_matching(mol, oriented)
        if _d == 0 and not info['is_saturated']:
            # Fully saturated, but RDKit called the ring aromatic -- which it
            # does for the all-heteroatom three-membered rings (``N1NN1`` came
            # back as 'triazirine', the UNSATURATED stem, for triaziridine).
            # (``:16928``): "Preferred IUPAC names of saturated
            # heteromonocyclic compounds are either Hantzsch-Widman names
            # described in or retained names described in Table
            # 2.3." 56 such rings in the enumeration, every one of size 3:
            # 16 where a mancude double bond is possible at all, so the old
            # unsaturated stem denoted a DIFFERENT molecule (OPSIN decodes
            # 'triazirine' to N1=NN1, not to triaziridine); and 40 all-divalent
            # rings where none is, so the old '2H-...irene' spelling happened
            # to decode to the right structure and the correction is one of
            # spelling only -- invisible to OPSIN and to an InChIKey.
            # Corrected on ``info`` itself so that the indicated-hydrogen step
            # below reads the same verdict the stem does.
            info = {**info, 'is_saturated': True, 'is_aromatic': False}
        elif 1 <= _d < _max_match:
            # A hydro form that:func:`name_partially_saturated_monocyclic_heterocycle`
            # declined to name. It must NOT fall through to the mancude stem:
            # that is precisely how 1,2-diazete / 1,4-diazocine / 1,2-diazocine
            # were emitted for their dihydro molecules. Fail closed -- an
            # abstention is recoverable, a wrong molecule is not.
            return None

    hw_name = build_hw_name(
        heteroatom_locants,
        ring_size,
        info['is_saturated'],
        info['is_aromatic'],
    )
    if hw_name is None:
        # A ring heteroatom has no Table 2.4 prefix -> refuse. Returning early
        # also keeps the indicated-hydrogen step below from doing ``ih + None``.
        return None

    # Post-process: replace the Hantzsch-Widman stem with the IUPAC-preferred
    # RETAINED stem, Table 2.2). "oxine"/"thiine" (6-membered
    # unsaturated O/S-rings) are not recognized by OPSIN either; IUPAC 2013
    # uses "pyran"/"thiopyran" with the tautomer's indicated hydrogen — which
    # is computed generically below (2H-pyran vs 4H-pyran; the old hardcoded
    # '2H-' mislabelled the 4H tautomers). "azine" -> "pyridine".
    #
    # This site used to carry its own three-row copy of the table
    # (oxine/azine/thiine). The five-ring rows were missing, so a mancude
    # 5-ring emitted its HW stem: '2H-azole' for C1=CCN=C1, where
    # (``:8163``) and ``3,4-dihydro-2H-pyrrole (PIN)`` (``:16896``) require
    # '2H-pyrrole'. One shared table now, matched on the whole stem.
    hw_name = _apply_retained_stem(hw_name)

    # Wave2 /: indicated hydrogen for a
    # mancude monocyclic parent (2H-1,3-dioxole, 1H-azirine, 2H-/4H-pyran).
    # Returns '' outside its fail-closed scope (aromatic, saturated, hydro
    # forms, multi-indicated-H rings) — the bare-stem status quo.
    ih = _monocycle_indicated_h_prefix(mol, oriented, info)
    if ih:
        hw_name = ih + hw_name

    return hw_name


def _monocycle_indicated_h_prefix(mol, oriented: List[int], info) -> str:
    """Indicated-hydrogen prefix ('2H-') for a mancude monocyclic HW parent.

     /: after the maximum number of noncumulative
    double bonds is assigned, a ring atom connected only by single bonds and
    bearing hydrogen takes the indicated-hydrogen descriptor, with the lowest
    locant available once the heteroatom locants are fixed (BB examples:
    2H-1,3-dioxole, 1H-azirine, 2H-pyran / 4H-pyran).

    FAIL-CLOSED scope (returns '' -> caller keeps the bare stem, today's
    behavior): the ring must be non-aromatic and unsaturated; the molecule
    must carry EXACTLY the mancude maximum of ring double bonds (fewer = a
    hydro form, owned by the partial-saturation namer); and exactly ONE sp3
    H-bearing double-bond-eligible atom may remain (multi-indicated-H rings
    like 1,3-dioxine are a documented follow-on).
    """
    if info.get('is_saturated'):
        return ""
    n = len(oriented)
    ring_set = set(oriented)

    if info.get('is_aromatic'):
        #: an aromatic mancude ring with a single "pyrrole-type"
        # heteroatom (trivalent N/P/As/Sb bearing an explicit H, donating its
        # lone pair to the aromatic sextet, NOT in a ring double bond) carries
        # an indicated hydrogen at that atom -> 1H-phosphole (like 1H-pyrrole;
        # pyrrole itself never reaches here, it is a retained name). Pyridine-
        # type =N- (no H) and >1 such atom fall through to '' (status quo).
        _PYRROLE_TYPE = frozenset({'N', 'P', 'As', 'Sb'})
        pyrrole_atoms = [
            idx for idx in oriented
            if mol.GetAtomWithIdx(idx).GetSymbol() in _PYRROLE_TYPE
            and mol.GetAtomWithIdx(idx).GetTotalNumHs() >= 1
        ]
        if len(pyrrole_atoms) != 1:
            return ""
        target = pyrrole_atoms[0]

        def _het_map_arom(order):
            return tuple(
                (pos + 1, mol.GetAtomWithIdx(a).GetSymbol())
                for pos, a in enumerate(order)
                if mol.GetAtomWithIdx(a).GetSymbol() != 'C'
            )

        base_map = _het_map_arom(oriented)
        best_loc: Optional[int] = None
        for start in range(n):
            for step in (1, -1):
                order = [oriented[(start + step * k) % n] for k in range(n)]
                if _het_map_arom(order) != base_map:
                    continue
                loc = order.index(target) + 1
                if best_loc is None or loc < best_loc:
                    best_loc = loc
        if best_loc is None:
            return ""
        return f"{best_loc}H-"

    # Actual ring double bonds (the aromatic case was excluded above).
    db_pairs = []
    for bond in bonds_of(mol):
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in ring_set and j in ring_set:
                db_pairs.append((i, j))
    if not db_pairs:
        return ""

    # Maximum noncumulative double bonds = maximum matching on the ring cycle
    # (shared helper; divalent chalcogens are double-bond-ineligible).
    max_match, eligible = _mancude_max_matching(mol, oriented)
    if len(db_pairs) != max_match:
        return ""  # hydro form (or over-perceived) — not this rule's scope

    # The indicated-H atom: sp3 (no ring double bond), eligible, bearing H --
    # OR a fully-substituted Group-14 (Si/Ge/Sn) mancude position, whose
    # indicated hydrogen is cited even when substitution has displaced every H.
    # "Silole, germole,... rings" (``the Blue Book``):
    # ``1,1-dibutyl-1H-germole (PIN) (note the indicated hydrogen atom)``. The
    # indicated hydrogen is a property of the mancude PARENT, not of the
    # substituted molecule, so a 1,1-disubstituted silole/germole/stannole keeps
    # its ``1H`` (``1,1-diethyl-1H-silole``) exactly as the H-bearing parent does.
    # The ``len(db_pairs) == max_match`` gate above already proved this is a
    # mancude-maximum ring (not a hydro form), and ``eligible`` + ``not in
    # db_atoms`` isolate the single saturated skeletal position; a substituent
    # standing in the indicated H's place does not move it.
    db_atoms = {a for pr in db_pairs for a in pr}
    h_bearing = [
        idx for pos, idx in enumerate(oriented)
        if idx not in db_atoms and eligible[pos]
        and mol.GetAtomWithIdx(idx).GetTotalNumHs() >= 1
    ]
    if h_bearing:
        # An H-bearing saturated position exists -> this is the historical path,
        # byte-identical to before (the displaced-H branch below never runs), so
        # no ring that already cited an indicated H can change.
        sp3h = h_bearing
    else:
        # No saturated position still bears an H, i.e. a substituent has fully
        # displaced the indicated hydrogen. The indicated-H position is a
        # property of the mancude PARENT, ``the Blue Book``:
        # "any ring atom with a bonding number of three or higher connected to
        # adjacent ring atoms by single bonds only, and carrying one or more
        # hydrogen atoms"), so it is still cited when a substituent stands in the
        # H's place. ``eligible[pos]`` already proves bonding number
        # >= 3 (a standard divalent chalcogen is ineligible; a lambda atom is
        # eligible only at >= 3) and ``idx not in db_atoms`` proves the
        # single-ring-bonds-only clause, so this holds for EVERY element that can
        # occupy the position -- Group-14 ``1,1-dibutyl-1H-germole`` and equally
        # boron ``2-(methylsulfanyl)-2H-1,3,2-oxathiaborepine`` (the boron sits
        # between the divalent ring O and S). The ``len(sp3h) != 1`` guard below
        # keeps this to the unambiguous single-indicated-H ring.
        sp3h = [
            idx for pos, idx in enumerate(oriented)
            if idx not in db_atoms and eligible[pos]
        ]
    if len(sp3h) != 1:
        return ""
    target = sp3h[0]

    # Lowest indicated-H locant among the ring numberings that keep the
    # chosen heteroatom locant->element assignment IDENTICAL:
    # heteroatoms fixed first, then low locants to indicated hydrogen). This
    # settles the direction tie the single-heteroatom numbering leaves open
    # (2H-azirine, not 3H-azirine).
    def _het_map(order):
        return tuple(
            (pos + 1, mol.GetAtomWithIdx(a).GetSymbol())
            for pos, a in enumerate(order)
            if mol.GetAtomWithIdx(a).GetSymbol() != 'C'
        )

    base_map = _het_map(oriented)
    best_loc: Optional[int] = None
    for start in range(n):
        for step in (1, -1):
            order = [oriented[(start + step * k) % n] for k in range(n)]
            if _het_map(order) != base_map:
                continue
            loc = order.index(target) + 1
            if best_loc is None or loc < best_loc:
                best_loc = loc
    if best_loc is None:
        return ""
    return f"{best_loc}H-"


def _macrocycle_ordered_ring(mol, ring_set: Set[int]) -> Optional[List[int]]:
    """Return the ring atom indices in connected (cycle) order, or None if the
    atoms do not form a single simple cycle."""
    start = min(ring_set)
    order = [start]
    prev = None
    cur = start
    while len(order) < len(ring_set):
        nxt = None
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = nb.GetIdx()
            if ni in ring_set and ni != prev and ni not in order:
                nxt = ni
                break
        if nxt is None:
            return None
        order.append(nxt)
        prev = cur
        cur = nxt
    # Confirm ring closure (last atom adjacent to the start).
    if start not in {nb.GetIdx() for nb in mol.GetAtomWithIdx(cur).GetNeighbors()}:
        return None
    return order


def _macrocycle_db_locant(p: int, q: int, ring_size: int) -> int:
    """Locant cited for a ring double bond between adjacent positions p, q:
    the lower locant, except the ring-closure (1, N) bond which is cited as N."""
    if {p, q} == {1, ring_size}:
        return ring_size
    return min(p, q)


def _orient_macrocycle_for_replacement(
    mol, ring_atoms, principal_group_atoms=None
) -> Tuple[Optional[List[Tuple[int, str]]], Optional[List[int]]]:
    """Number a >10-membered heteromonocycle for replacement ("a") nomenclature.

    Returns ``(heteroatom_locants, double_bond_locants)`` where the numbering
    minimises, in order (IUPAC 2013 /: the heteroatom locant
    set, then the heteroatoms by element seniority, then the PRINCIPAL
    CHARACTERISTIC GROUP locant set (c) — the suffix, e.g. the ``-2,6-dione``
    carbons), then the ring double-bond locant set.

     M4: threading ``principal_group_atoms`` (the in-ring suffix-bearing atoms)
    into the key is the fix for the macrolactam/macrolactone class whose in-ring
    ``ene`` was numbered on a DIFFERENT direction than the composer's suffix
    locants — the two disagreed and the spliced name was internally inconsistent
    (OPSIN-unparseable → abstain). With the suffix ranked ABOVE unsaturation the
    numberer picks the same direction the composer uses, so the name round-trips.
    Empty ``principal_group_atoms`` (a bare ring / crown ether) leaves the key
    unchanged, so the previously-passing cases are byte-identical (no regression),
    and any residual mismatch still fails the OPSIN validity gate → abstain.

    The ring is *kekulised* first so a fully-conjugated macrocycle that RDKit
    perceives as aromatic (e.g. the V-6 azacyclotrideca-hexaene) yields its
    localised double bonds instead of zero (the bug that made the old path emit
    a single bare ``ene``). Returns ``(None, None)`` if the ring cannot be
    ordered/kekulised, so the caller can fall back safely.
    """
    pg_atoms = set(principal_group_atoms or ())
    ring_set = set(ring_atoms)
    ring_size = len(ring_set)
    het = [a for a in ring_set if mol.GetAtomWithIdx(a).GetSymbol() != 'C']
    if not het:
        return None, None

    kmol = Chem.Mol(mol)
    try:
        Chem.Kekulize(kmol, clearAromaticFlags=True)
    except Exception:
        return None, None

    order = _macrocycle_ordered_ring(kmol, ring_set)
    if order is None or len(order) != ring_size:
        return None, None

    ring_double_bonds: List[Tuple[int, int]] = []
    for bond in kmol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring_set and j in ring_set:
            ring_double_bonds.append((i, j))

    best = None
    for start in range(ring_size):
        for direction in (1, -1):
            seq = [order[(start + k * direction) % ring_size] for k in range(ring_size)]
            loc = {a: idx + 1 for idx, a in enumerate(seq)}
            het_set = sorted(loc[a] for a in het)
            seniority = sorted(
                (HETEROATOM_PRIORITY.get(kmol.GetAtomWithIdx(a).GetSymbol(), 999), loc[a])
                for a in het
            )
            dbl = sorted(
                _macrocycle_db_locant(loc[i], loc[j], ring_size)
                for i, j in ring_double_bonds
            )
            # (c): the principal characteristic group takes low locants
            # BEFORE unsaturation. Only in-ring suffix atoms carry a ring locant
            # (an exocyclic -carboxylic-acid carbon is filtered by ``a in loc``);
            # empty -> ```` -> no effect on the key (no regression).
            pg_key = tuple(sorted(loc[a] for a in pg_atoms if a in loc))
            key = (het_set, seniority, pg_key, dbl)
            if best is None or key < best[0]:
                best = (key, loc, dbl)

    loc = best[1]
    dbl = best[2]
    heteroatom_locants = sorted(
        ((loc[a], kmol.GetAtomWithIdx(a).GetSymbol()) for a in het),
        key=lambda t: t[0],
    )
    return heteroatom_locants, dbl


def _table_1_5_prefix(element: str) -> Optional[str]:
    """Table-1.5 skeletal replacement prefix, or None to refuse.

    The single Table-1.5 source, for the ring sizes that are NOT Hantzsch-Widman.
    """
    from .ring_replacement import HETEROATOM_PREFIXES
    entry = HETEROATOM_PREFIXES.get(element)
    return entry[0] if entry is not None else None


def _build_replacement_name(
    heteroatoms: List[Tuple[int, str]],
    ring_size: int,
    is_saturated: bool,
    double_locants: Optional[List[int]] = None,
    bare_ring: bool = False,
) -> Optional[str]:
    """
    Build replacement ("a") nomenclature name for macrocyclic heterocycles.

    For rings larger than 10 atoms, IUPAC uses "a" nomenclature:
    replacement prefixes (oxa, aza, thia) + cycloalkane parent name.

    Example: 1,4,7,10-tetraoxacyclododecane (12-crown-4)

    Args:
        heteroatoms: List of (locant, element) tuples
        ring_size: Number of atoms in the ring (> 10)
        is_saturated: True if fully saturated
        double_locants: Sorted ring double-bond locants (V-6 / polyene
            enumeration). When supplied (incl. an empty list for a saturated
            ring), the unsaturation suffix is built from the ACTUAL double bonds
            (``cyclotrideca-2,4,6,8,10,12-hexaene``) instead of the old bare
            ``ene`` that dropped every bond but one. When None, the legacy
            saturated/``ene`` behaviour is used (caller could not enumerate).

    Returns:
        IUPAC replacement name (e.g., '1,4,7,10-tetraoxacyclododecane',
        '1-azacyclotrideca-2,4,6,8,10,12-hexaene')
    """
    from ..data.chain_names import get_chain_prefix

    if not heteroatoms:
        # No heteroatoms: just cycloalkane (shouldn't reach here)
        prefix = get_chain_prefix(ring_size)
        return f"cyclo{prefix}ane"

    # Build replacement prefix (same logic as HW prefix builder)
    element_locants: Dict[str, List[int]] = {}
    for locant, elem in heteroatoms:
        if elem not in element_locants:
            element_locants[elem] = []
        element_locants[elem].append(locant)

    # Order by IUPAC priority (O > S > N >...)
    elements_by_priority = sorted(
        element_locants.keys(),
        key=lambda e: HETEROATOM_PRIORITY.get(e, 999)
    )

    #: the sole heteroatom's locant '1' is omitted -- but ONLY when NO
    # OTHER locant is cited anywhere in the final name, which is the whole reason
    # the rule is safe: with nothing else numbered, the origin cannot be ambiguous.
    # Scoped by measurement, not by reading:
    # * the Blue Book KEEPS the locant on the unsaturated forms
    # ``1-oxacycloundeca-2,4,6,8,10-pentaene`` (PIN) and
    # ``1-azacyclotetradeca-1,3,5,7,9,11,13-heptaene`` (PIN) [BBv2:8488-8490],
    # where unsaturation locants are cited -> hence ``_saturated_ring``;
    # * and the shipped gold target ``1-selenacyclotridecan-3-one``
    # (W2-RINGKET-SELENA-PROTECT) keeps it too, because a SUFFIX locant is
    # cited. That name is assembled by this builder's CALLER, which appends
    # ``-3-one`` to the ``...ane`` parent returned here -- so this function
    # cannot see it, and an omission decided from the arguments alone
    # regressed that target (measured, not hypothesised).
    # ``bare_ring`` is therefore required: the caller asserts the molecule IS the
    # ring, so there is provably no substituent, suffix or unsaturation locant to
    # collide with. That is the exact condition covering the book's
    # ``thiacyclododecane`` (PIN) and the ``thiacyclododecane`` /
    # ``azacyclotridecane`` gold targets, and nothing wider.
    _saturated_ring = (not double_locants if double_locants is not None
                       else bool(is_saturated))
    _omit_sole_heteroatom_locant = (
        len(heteroatoms) == 1 and _saturated_ring and bare_ring)

    prefix_parts = []
    for elem in elements_by_priority:
        # Rings of ELEVEN or more members are named by skeletal replacement
        # [BBv2:8482]: "For monocyclic rings with eleven and more ring
        # members, skeletal replacement ('a') nomenclature (see is
        # used"), and we spell them from the Table-1.5 source.
        #
        # NOT ESTABLISHED -- which table governs HERE is genuinely ambiguous in
        # the book, and it matters for exactly two elements:
        # * points at, i.e. Table 1.5;
        # * but [BBv2:8484] says the prefixes come from "(see
        # Table 2.4)" -- the Hantzsch-Widman table -- in the same sentence
        # that quotes Table 2.4's 22-element seniority order. (That
        # cross-reference may simply be reaching for the ORDER, which
        # Table 1.5 does not print at all.)
        # The two tables agree on 16 of the elements this path can reach, so the
        # question only bites for Al (``alumina`` vs ``aluma``) and In (``inda``
        # vs ``indiga``). Table 1.5 is kept because the Table 2.4 reading is
        # unverifiable downstream: ``alumacyclotridecane`` and
        # ``indigacyclotridecane`` are REJECTED by the round-trip parser
        # ("cyclotridecane... not parseable" after an HW prefix) while
        # ``aluminacyclotridecane`` and ``indacyclotridecane`` parse back to the
        # right structure. Switching would trade a verified name for an
        # unverifiable one, and fails OPEN on an unparseable name -- so
        # it must not be switched on the strength of a parenthetical alone.
        # Resolve against the printed Blue Book before changing this.
        hw_prefix = _table_1_5_prefix(elem)
        if not hw_prefix:
            # FAIL CLOSED -- see ``build_hw_name``: skipping here dropped the
            # atom while ``ring_size`` still counted it.
            return None

        locants = sorted(element_locants[elem])
        count = len(locants)

        locant_str = ','.join(str(loc) for loc in locants)
        if count > 1:
            multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
            prefix_parts.append(f"{locant_str}-{multiplier}{hw_prefix}")
        elif _omit_sole_heteroatom_locant:
            # [BBv2:8494]: "When a single heteroatom is present in
            # the ring, it is assigned the locant '1', which is OMITTED in the
            # name" -- the BB's own saturated example is ``thiacyclododecane
            # (PIN)`` [BBv2:8488], and ``thiacyclododecane`` /
            # ``azacyclotridecane`` are shipped gold targets carrying exactly
            # this citation. This builder emitted the locant for EVERY element;
            # O/S/N/Si never reach it (an earlier producer, which already omits
            # the locant, handles them) so the defect only surfaced for an
            # element that falls through to here -- which Al and In began doing
            # when - made them spellable. Without this the two new rows
            # would have shipped the non-PIN ``1-aluminacyclotridecane`` while
            # oxygen shipped ``oxacyclotridecane``.
            prefix_parts.append(hw_prefix)
        else:
            prefix_parts.append(f"{locant_str}-{hw_prefix}")

    replacement_prefix = '-'.join(prefix_parts)

    # Build parent ring name. 'cyclo' starts with a consonant so the replacement
    # prefix never needs terminal-'a' elision against it.
    chain_prefix = get_chain_prefix(ring_size)

    if double_locants is not None:
        # V-6 /: enumerate ALL ring double bonds with their locants
        # (e.g. 'a-2,4,6,8,10,12-hexaen' + 'e'), reusing the shared infix builder
        # so the conventions (euphonic 'a', multipliers, hyphenation) match the
        # carbocyclic path exactly. Empty list -> 'an' -> '...ane' (saturated).
        from ..assembly.composition_primitives import _build_unsaturation_infix
        infix = _build_unsaturation_infix(sorted(double_locants), [])
        return f"{replacement_prefix}cyclo{chain_prefix}{infix}e"

    # Legacy fallback (double bonds could not be enumerated): keep prior behaviour.
    if is_saturated:
        parent = f"cyclo{chain_prefix}ane"
    else:
        parent = f"cyclo{chain_prefix}ene"

    return f"{replacement_prefix}{parent}"


# ---------------------------------------------------------------------------
# Substituted heterocycle naming (HETERO-09)
# ---------------------------------------------------------------------------


def _is_monocyclic_lactam(mol):
    """Late-bound wrapper for lactams.is_monocyclic_lactam (avoids a module-load
    circular import between heterocycles and lactams)."""
    from .lactams import is_monocyclic_lactam
    return is_monocyclic_lactam(mol)


def _is_monocyclic_lactone(mol):
    """Late-bound wrapper for lactones.is_monocyclic_lactone (avoids a
    module-load circular import between heterocycles and lactones)."""
    from .lactones import is_monocyclic_lactone
    return is_monocyclic_lactone(mol)


def _ring_has_extra_heteroatom(mol, ring_set, carbonyl_idx) -> bool:
    """True if the ring containing `carbonyl_idx` carries a ring heteroatom
    besides the single amide N — i.e. it is a multi-heteroatom saturated
    heteroring (thiazolidine, oxazolidine,...) that the lactam handler
    declined. Counts a ring O/S/Se/Te, or a 2nd ring N, as the extra."""
    n_count = 0
    other_hetero = 0
    for a_idx in ring_set:
        sym = mol.GetAtomWithIdx(a_idx).GetSymbol()
        if sym == 'N':
            n_count += 1
        elif sym not in ('C', 'H'):
            other_hetero += 1
    return other_hetero >= 1 or n_count >= 2


def _is_cyclic_imide_carbonyl(mol, carbonyl_c_idx, ring_set) -> bool:
    """True iff ``carbonyl_c_idx`` is a ring carbon of a CYCLIC imide motif:
    it bears an exocyclic ``=O`` and is bonded to a ring N that is itself
    bonded to a SECOND ring carbon which also bears an exocyclic ``=O`` — the
    imide nitrogen flanked by two ring carbonyls. BOTH carbonyls of the imide
    satisfy this, so both are reclassified to the ``-one`` suffix (yielding the
    ring ``-dione`` name). Requiring the second flanking carbonyl keeps a plain
    ring lactam/amide (a single carbonyl on the N) out of this branch — those
    are handled by the secondary_amide/lactam paths."""
    c = mol.GetAtomWithIdx(carbonyl_c_idx)
    if carbonyl_c_idx not in ring_set or c.GetSymbol() != 'C':
        return False

    def _has_exocyclic_carbonyl_o(atom):
        return any(
            b.GetBondTypeAsDouble() == 2.0
            and b.GetOtherAtom(atom).GetSymbol() == 'O'
            and b.GetOtherAtom(atom).GetIdx() not in ring_set
            for b in atom.GetBonds()
        )

    if not _has_exocyclic_carbonyl_o(c):
        return False
    for nb in c.GetNeighbors():
        if nb.GetIdx() not in ring_set or nb.GetSymbol() != 'N':
            continue
        for nn in nb.GetNeighbors():
            if (nn.GetIdx() in ring_set and nn.GetIdx() != carbonyl_c_idx
                    and nn.GetSymbol() == 'C'
                    and _has_exocyclic_carbonyl_o(nn)):
                return True
    return False


def ring_principal_suffix_atoms(
    mol, ring_atoms, principal_group, functional_groups=None
) -> Set[int]:
    """RING atom indices that carry the principal characteristic group as a
    SUFFIX on this heterocycle parent, so numbering gives them the lowest
    locants.

    IUPAC NUMBERING (``the Blue Book``): "low locants are assigned
    to them in the following decreasing order of seniority" —... (c) "principal
    characteristic groups and free valences (suffixes)";... (f) "detachable
    alphabetized prefixes". Criterion (c) — the ``-dione`` / ``-sulfonic acid``
    SUFFIX — outranks (f), the ``methyl`` prefix. (For a heterocycle the ring
    heteroatoms are numbered first, change note (1) at ``the Blue Book``; the
    heteroatom set ties in both directions in these cases, so the suffix
    decides.) ``orient_heterocycle_with_substituents`` receives this set as
    ``principal_group_atoms``.

    The generic prefix-class anchor in ``namer.py`` keys on
    ``get_prefix(principal_group)`` and only anchors a CARBON-centred appended
    suffix, so it MISSES two families whose suffix decision lives in
    ``get_heterocycle_substituents`` below:

      A. Pseudoketone ring carbons named ``-one``/``-dione``/``-thione`` — a ring
         carbon bearing an exocyclic ``=O``/``=S``/``=Se``/``=Te`` whose
         molecule-level principal group is a lactam-declined amide, a cyclic
         imide, a lactone-declined ester, or a thioamide. ``get_prefix`` of
         those principal groups is ``None`` (or ``carbamoyl``), never ``oxo``, so
         the ring carbonyl carbons were never anchored and the (f)
         substituent set put a SUBSTITUTED ring atom at locant 1
         (``CN1C(=O)CNC1=O`` -> ``1-methyl...-2,5-dione`` where (c) requires
         ``3-methyl...-2,4-dione``). Mirrors the ``ring_ketone_suffix`` /
         ``ring_imide_suffix`` / ``ring_lactone_suffix`` / ``ring_thioamide_suffix``
         gates in ``get_heterocycle_substituents`` exactly, but needs no
         orientation (those gates are locant-independent).

      B. The ring atom (typically the ring N) bearing an exocyclic
         SULFUR-oxoacid / sulfonamide appended suffix (``-sulfonic acid`` /
         ``-sulfinic acid`` / ``-sulfonamide``). Its central atom is sulfur, not
         carbon, so ``namer.py``'s carbon-only anchor skipped it and the
         substituted ring N took locant 1 (``OS(=O)(=O)N1CCN(C)CC1`` ->
         ``1-methylpiperazine-4-sulfonic acid`` where (c) requires
         ``4-methylpiperazine-1-sulfonic acid``). The ring atom bonded to the
         suffix sulfur is the anchor — the sulfur analogue of the ring-N
         carboxylic-acid anchor the carbon path already feeds.
    """
    from ..perception.rings import get_containing_ring_system

    ring_set = set(get_containing_ring_system(mol, ring_atoms))
    principal_ring = set(ring_atoms)
    anchors: Set[int] = set()

    def _exocyclic_double(c_idx, symbols) -> bool:
        """Ring carbon with an exocyclic double bond to a chalcogen in symbols."""
        atom = mol.GetAtomWithIdx(c_idx)
        if atom.GetSymbol() != 'C':
            return False
        return any(
            b.GetBondTypeAsDouble() == 2.0
            and b.GetOtherAtom(atom).GetSymbol() in symbols
            and b.GetOtherAtom(atom).GetIdx() not in ring_set
            for b in atom.GetBonds()
        )

    # ---- A. pseudoketone ring carbons (-one/-dione/-thione) ----
    if principal_group == 'imide':
        for a in principal_ring:
            if _is_cyclic_imide_carbonyl(mol, a, ring_set):
                anchors.add(a)
    elif (principal_group in ('secondary_amide', 'tertiary_amide')
          and _is_monocyclic_lactam(mol) is None):
        for a in principal_ring:
            if (_exocyclic_double(a, ('O',))
                    and _ring_has_extra_heteroatom(mol, ring_set, a)):
                anchors.add(a)
    elif principal_group == 'ester' and _is_monocyclic_lactone(mol) is None:
        for a in principal_ring:
            if _exocyclic_double(a, ('O',)):
                anchors.add(a)
    elif principal_group in ('thioamide', 'selenoamide', 'telluroamide'):
        _chal = {'thioamide': 'S', 'selenoamide': 'Se',
                 'telluroamide': 'Te'}[principal_group]
        for a in principal_ring:
            if _exocyclic_double(a, (_chal,)):
                anchors.add(a)

    # ---- B. exocyclic sulfur-oxoacid / sulfonamide suffix on a ring atom ----
    _S_SUFFIX_PGS = {
        'sulfonic_acid', 'sulfinic_acid',
        'primary_sulfonamide', 'secondary_sulfonamide', 'tertiary_sulfonamide',
    }
    if principal_group in _S_SUFFIX_PGS and functional_groups:
        for match in functional_groups.get(principal_group, ()):
            for midx in match:
                if midx in ring_set:
                    continue
                s_atom = mol.GetAtomWithIdx(midx)
                if s_atom.GetSymbol() != 'S':
                    continue
                for nbr in s_atom.GetNeighbors():
                    if nbr.GetIdx() in principal_ring:
                        anchors.add(nbr.GetIdx())

    return anchors


def get_heterocycle_substituents(
    mol,
    ring_atoms,
    oriented_ring: List[int],
    atom_to_locant: Dict[int, int],
    principal_group: Optional[str] = None
) -> Dict[int, List[Dict]]:
    """
    Find substituents attached to a heterocyclic ring.

    For each ring atom that has neighbors not in the ring, identifies the
    substituent and tracks whether it's attached to a nitrogen (N-substitution).
    Also classifies substituents as ring or alkyl to correctly name ring
    substituents (piperidinyl, phenyl) instead of counting carbons (pentyl, hexyl).

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring
        oriented_ring: Ring atoms in IUPAC numbering order
        atom_to_locant: Dict mapping atom_idx to locant (1-indexed)

    Returns:
        Dict mapping locant to list of substituent info dicts:
        {
            locant: [
                {
                    'atoms': List[int], # Atom indices in substituent
                    'is_on_nitrogen': bool, # True if attached to N
                    'carbon_count': int, # Number of C atoms (for alkyl naming)
                    'connecting_atom': int, # Ring atom the sub attaches to
                    'is_ring': bool, # True if substituent is a ring
                    'ring_name': str, # Ring substituent name (if is_ring)
                }
            ]
        }

    Examples:
        >>> mol = Chem.MolFromSmiles('CN1CCCC1') # N-methylpyrrolidine
        >>> ring = mol.GetRingInfo.AtomRings[0]
        >>> oriented, atom_to_loc = orient_heterocycle(mol, ring)
        >>> subs = get_heterocycle_substituents(mol, ring, oriented, atom_to_loc)
        >>> # N-methyl should be at locant 1 (N position) with is_on_nitrogen=True
    """
    from ..perception.chains import classify_substituent
    from ..perception.rings import get_containing_ring_system
    from .seniority import get_prefix, get_suffix

    # / ring-suffix fix : when a senior characteristic group sits on the ring,
    # express it as a SUFFIX, not a detachable prefix. We recognise the
    # principal group GENERICALLY off the seniority tables: a no-carbon substituent
    # whose prefix form equals get_prefix(principal_group) AND whose class has a ring
    # suffix form (get_suffix(..., is_ring=True)) is the principal-group suffix
    # (oxo->one, hydroxy->ol, amino->amine, sulfanyl->thiol,...). Subordinate
    # same-prefix groups keep the prefix because only the principal group's prefix
    # matches. No per-FG `if`.
    pg_prefix = get_prefix(principal_group) if principal_group else None
    pg_ring_suffix = get_suffix(principal_group, is_ring=True) if principal_group else None

    # Use the complete ring system as BFS boundary (IUPAC
    # This prevents walking into fused partner rings (e.g., caffeine's
    # imidazole BFS leaking into pyrimidine)
    ring_set = set(get_containing_ring_system(mol, ring_atoms))
    substituents: Dict[int, List[Dict]] = {}

    for ring_atom_idx in oriented_ring:
        ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
        locant = atom_to_locant.get(ring_atom_idx)
        if locant is None:
            continue

        # Check if this ring atom is a nitrogen
        is_nitrogen = ring_atom.GetSymbol() == 'N'

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip atoms that are part of the ring
            if nbr_idx in ring_set:
                continue

            # BFS to find full substituent
            sub_atoms = _bfs_substituent(mol, nbr_idx, ring_set)

            # ---- free-valence morphology, decided ONCE, up front ----
            # Every branch below eventually names this fragment from its CARBON
            # COUNT (directly, or via classify_substituent's ring name), and a
            # count cannot tell -CH3 from =CH2: 'C=C1CCNCC1' was named
            # '4-methylpiperidine', which is a different molecule. The
            # attachment BOND is the only thing that settles it, and it is read
            # here -- before the specialised branches -- so that all of them
            # inherit the answer instead of each needing its own copy.
            #
            # The shared primitive DEFERS for a single bond and for any
            # non-carbon attachment, so the exocyclic =O/=S/=N machinery below
            # (hetero_name -> 'oxo', the '-one'/'-thione' ring suffixes) is
            # untouched: those multivalent heteroatom prefixes are a separate,
            # already-correct class.
            from ..assembly.substituent_enumerator import carbon_free_valence_prefix
            _fv = carbon_free_valence_prefix(mol, sub_atoms, nbr_idx)
            if not _fv.defers:
                _entry = {
                    'atoms': sub_atoms,
                    'is_on_nitrogen': is_nitrogen,
                    'carbon_count': sum(
                        1 for i in sub_atoms
                        if mol.GetAtomWithIdx(i).GetSymbol() == 'C'),
                    'connecting_atom': ring_atom_idx,
                    'is_ring': False,
                    'ring_name': None,
                }
                if _fv.prefix is not None:
                    _entry['prefix_name'] = _fv.prefix
                else:
                    # Fail closed: the whole heterocycle candidate declines
                    # (name_substituted_heterocycle returns None on this flag),
                    # so another producer may still name the molecule and none
                    # of them ships a wrong structure.
                    _entry['unnameable'] = True
                    logger.debug(
                        "heterocycle substituent at locant %s: %s",
                        locant, _fv.basis)
                substituents.setdefault(locant, []).append(_entry)
                continue

            # task 9: an ACYL group on a ring NITROGEN is an
            # AMIDE — the amide machinery names it ('-oyl' forms / parent
            # ketone), never this collector (which produced garbled
            # 'dienalyl'/alkyl forms and, once it claimed the atoms, its
            # full coverage let the wrong candidate win the pool). Skip the
            # fragment entirely: the heterocycle candidate's coverage drops
            # and the amide path wins, as it did before the chokepoint.
            if is_nitrogen:
                _nbr_atom = mol.GetAtomWithIdx(nbr_idx)
                _is_acyl = _nbr_atom.GetSymbol() == 'C' and any(
                    b.GetOtherAtom(_nbr_atom).GetSymbol() in ('O', 'S', 'Se', 'Te')
                    and b.GetBondTypeAsDouble() == 2.0
                    and b.GetOtherAtom(_nbr_atom).GetIdx() in set(sub_atoms)
                    for b in _nbr_atom.GetBonds()
                )
                if _is_acyl:
                    # R1, the Blue Book;, the Blue Book): a
                    # -C(=O)-OH / -C(=O)-NH2 / -C(=O)-NR2 directly on a ring N is
                    # NOT an N-acyl amide of some acyl parent — it is the ring-N
                    # principal-group SUFFIX (``pyrrolidine-1-carboxylic acid``,
                    # ``piperidine-1-carboxamide``), senior to the carbamic/urea
                    # (carbonic acid derivative) re-framing (the Blue Book). Let it fall
                    # through to _identify_suffix_fg so it is attached as a ring
                    # suffix. Scoped to a carboxylic-acid / carboxamide FG that is
                    # the molecule-level principal group: a genuine N-acyl group
                    # (acetyl, a longer acyl chain — captopril's N-acyl) has no
                    # -carboxylic-acid/-carboxamide suffix form, so _identify_suffix_fg
                    # returns None there and the task 9 skip still applies.
                    _ring_n_suffix = _identify_suffix_fg(
                        mol, nbr_idx, sub_atoms, ring_set)
                    if not (_ring_n_suffix
                            and _ring_n_suffix.get('suffix_name') in (
                                'carboxylic acid', 'carboxamide')
                            and principal_group in (
                                'carboxylic_acid', 'primary_amide',
                                'secondary_amide', 'tertiary_amide')):
                        continue

            # C4 /: an N-substituted exocyclic amine on a
            # ring CARBON, when the amine is the molecule-level principal group,
            # is the ring-amine SUFFIX with the N-substituents cited as italic-N
            # prefixes (pyridin-4-amine -> N-methylpyridin-4-amine /
            # N-phenylpyridin-4-amine). Bare -NH2 already routes via the
            # carbon_count==0 -> _identify_hetero_substituent -> is_principal_suffix
            # path below; this block adds the -NHR / -NR2 forms (h_count 1/0 with
            # carbon neighbours) that the carbon_count==0 gate would otherwise miss
            # (they were mis-collected as an alkyl fragment -> 'N-methylanamine').
            # Scoped strictly to: pg is amine, exocyclic N on a ring carbon, N is a
            # neutral secondary/tertiary amine whose only non-ring neighbours are
            # carbons (no =O/nitro/nitroso, no extra heteroatoms). Fail-closed: if
            # any N-substituent cannot be named, skip this block and let the
            # general path (which fails closed) handle it.
            _amine_n = mol.GetAtomWithIdx(nbr_idx)
            if (pg_ring_suffix == 'amine'
                    and not is_nitrogen                       # ring atom is a carbon
                    and _amine_n.GetSymbol() == 'N'
                    and _amine_n.GetFormalCharge() == 0
                    and _amine_n.GetTotalNumHs() < 2):        # -NHR or -NR2 (not -NH2)
                _n_nonring = [
                    nb for nb in _amine_n.GetNeighbors()
                    if nb.GetIdx() not in ring_set
                ]
                # Genuine amine: every non-ring neighbour is a carbon (excludes
                # nitroso -N=O, azo, hydrazino, N-oxide, etc.).
                if _n_nonring and all(nb.GetSymbol() == 'C' for nb in _n_nonring):
                    from ..assembly.substituent_naming import name_substituent_fragment
                    _n_sub_names = []
                    _ok = True
                    for _cn in _n_nonring:
                        _cn_atoms = _bfs_substituent(mol, _cn.GetIdx(), ring_set | {nbr_idx})
                        _nm = name_substituent_fragment(
                            mol, _cn_atoms, _cn.GetIdx(), list(ring_set | {nbr_idx})
                        )
                        if _nm is None:
                            _ok = False
                            break
                        _n_sub_names.append(_nm)
                    if _ok and _n_sub_names:
                        _n_sub_names.sort()
                        substituents.setdefault(locant, []).append({
                            'atoms': sub_atoms,
                            'is_on_nitrogen': False,
                            'carbon_count': 0,
                            'connecting_atom': ring_atom_idx,
                            'is_ring': False,
                            'ring_name': None,
                            'is_suffix': True,
                            'suffix_name': 'amine',
                            'amine_n_substituents': _n_sub_names,
                        })
                        continue

            # Count carbon atoms for alkyl naming
            carbon_count = sum(
                1 for idx in sub_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )

            # /: a heteroatom-rooted ETHER substituent on
            # the ring -- -O-R / -S-R / -Se-R / -Te-R (carbon_count>0 because R has
            # carbons) -- is an (R)oxy / (R)sulfanyl / (R)selanyl / (R)tellanyl
            # prefix. classify_substituent below counts the arm carbons and DROPS
            # the O/S/Se root (sugar-ring-oxygen-drop -> decline -> unknown; e.g. methoxy-/
            # methylsulfanyl-pyridine). Name the arm recursively and carry the
            # prefix as hetero_name. Fail-closed: only a clean single-bond ether
            # root with exactly one carbon arm; any decline falls through.
            _eth_root = mol.GetAtomWithIdx(nbr_idx)
            _ETHER_SUFFIX = {'O': 'oxy', 'S': 'sulfanyl',
                             'Se': 'selanyl', 'Te': 'tellanyl'}
            if (carbon_count > 0 and not is_nitrogen
                    and _eth_root.GetSymbol() in _ETHER_SUFFIX
                    and _eth_root.GetFormalCharge() == 0
                    and _eth_root.GetTotalNumHs() == 0
                    and not _eth_root.GetIsAromatic()
                    and all(b.GetBondTypeAsDouble() == 1.0
                            for b in _eth_root.GetBonds())):
                _eth_arm = [nb for nb in _eth_root.GetNeighbors()
                            if nb.GetIdx() not in ring_set]
                if len(_eth_arm) == 1 and _eth_arm[0].GetSymbol() == 'C':
                    from ..assembly.naming_utils import (
                        apply_enclosing_marks,
                        is_complex_substituent,
                    )
                    from ..assembly.substituent_naming import name_substituent_fragment
                    if _eth_root.GetSymbol() == 'O':
                        # O-ether: the fragment namer's O-attach path contracts the
                        # PIN (methyl+oxy -> methoxy, ethyl -> ethoxy) and encloses
                        # a complex arm itself -> pass the WHOLE -O-R fragment.
                        _eth_name = name_substituent_fragment(
                            mol, sub_atoms, nbr_idx, list(ring_set))
                    else:
                        # S/Se/Te: name the arm, append the chalcogen stem, enclose
                        # a complex arm; e.g. (prop-2-en-1-yl)sulfanyl).
                        _arm = name_substituent_fragment(
                            mol, [a for a in sub_atoms if a != nbr_idx],
                            _eth_arm[0].GetIdx(), list(ring_set | {nbr_idx}))
                        _eth_name = None
                        if _arm:
                            if is_complex_substituent(_arm):
                                _arm = apply_enclosing_marks(_arm, 0)
                            _eth_name = \
                                f"{_arm}{_ETHER_SUFFIX[_eth_root.GetSymbol()]}"
                    if _eth_name:
                        substituents.setdefault(locant, []).append({
                            'atoms': sub_atoms,
                            'is_on_nitrogen': False,
                            'carbon_count': 0,
                            'connecting_atom': ring_atom_idx,
                            'is_ring': False,
                            'ring_name': None,
                            'hetero_name': _eth_name,
                        })
                        continue

            # breadth: a ring-borne higher-oxide sulfur substituent
            # -S(=O)R (sulfinyl) / -S(=O)(=O)R (sulfonyl) is an (R)sulfinyl /
            # (R)sulfonyl PREFIX. The generic classify path below counts the R
            # carbons and DROPS the S + its =O, so the whole molecule abstained
            # (caught the atom-drop, e.g. 2-(methylsulfinyl)pyridine and the
            # omeprazole/PPI benzimidazole-sulfinyl class -> `1H-benzimidazole`).
            # The recursive name_substituent already builds these under
            # allow_mancude (the benzene path emits `(methanesulfinyl)benzene`);
            # route the whole fragment through it. best-effort only (allow_mancude
            # read from context -> PIN default byte-identical, matching the existing
            # sulfoxide/sulfone prefix conservatism); gate-INDEPENDENT coverage
            # guard so a partial name never ships even with off.
            _sx_root = mol.GetAtomWithIdx(nbr_idx)
            if (carbon_count > 0 and not is_nitrogen
                    and _sx_root.GetSymbol() in ('S', 'Se', 'Te')
                    and _sx_root.GetFormalCharge() == 0
                    and any(b.GetBondTypeAsDouble() == 2.0
                            and b.GetOtherAtom(_sx_root).GetSymbol() == 'O'
                            for b in _sx_root.GetBonds())):
                from ..metrics.provenance import best_effort_ctx
                if best_effort_ctx.get():
                    from ..assembly.naming_utils import (
                        apply_enclosing_marks,
                        is_complex_substituent,
                    )
                    from ..assembly.substituent_enumerator import name_substituent
                    _sx_name = name_substituent(
                        mol, list(sub_atoms), nbr_idx, allow_mancude=True)
                    if (_sx_name and _sx_name != 'substituent'
                            and _n_anchored_substituent_covers(
                                mol, _sx_name, sub_atoms)):
                        if is_complex_substituent(_sx_name):
                            _sx_name = apply_enclosing_marks(_sx_name, 0)
                        substituents.setdefault(locant, []).append({
                            'atoms': sub_atoms,
                            'is_on_nitrogen': False,
                            'carbon_count': 0,
                            'connecting_atom': ring_atom_idx,
                            'is_ring': False,
                            'ring_name': None,
                            'hetero_name': _sx_name,
                        })
                        continue

            # Detect non-carbon functional substituents (amino, hydroxy, nitro, etc.)
            hetero_sub_name = None
            if carbon_count == 0:
                # Check for functional groups that have no carbons
                hetero_sub_name = _identify_hetero_substituent(mol, sub_atoms, ring_set)
                if hetero_sub_name is None:
                    # An unrecognised no-carbon exocyclic group (e.g. the oxygen
                    # of an -O-SO3H sulfate or -O-PO(OH)2 phosphate ester). BFS
                    # only walks heavy atoms, so this is NEVER just implicit H —
                    # silently dropping it corrupts the structure (sugar-ring-oxygen-drop).
                    # Record it as unnameable so the assembler declines the whole
                    # heterocycle candidate rather than emit a group-dropping
                    # name. Fail-closed per accuracy-first.
                    substituents.setdefault(locant, []).append({
                        'atoms': sub_atoms,
                        'is_on_nitrogen': is_nitrogen,
                        'carbon_count': 0,
                        'connecting_atom': ring_atom_idx,
                        'is_ring': False,
                        'ring_name': None,
                        'unnameable': True,
                    })
                    continue

            # Check for suffix-type functional groups (-COOH, -CHO, -CONH2, -CN)
            # These should be expressed as suffixes, not prefix substituents
            suffix_info = _identify_suffix_fg(mol, nbr_idx, sub_atoms, ring_set)

            # Classify substituent as ring or alkyl
            classification = classify_substituent(mol, sub_atoms, ring_set)
            is_ring = classification['type'] == 'ring'
            ring_name = classification['name'] if is_ring else None

            #.2: a demoted ring substituent must carry its OWN
            # substituent prefixes ('2-oxocyclohexyl', not the FG-dropping
            # bare 'cyclohexyl'). Guarded primitive: None -> keep the bare
            # legacy form (zero regression); the coverage check requires the
            # decorated name to account for exactly the substituent atoms.
            if is_ring and ring_name:
                from .ring_substituents import decorated_ring_substituent_name
                _sub_ring = classification.get('ring_atoms')
                if _sub_ring:
                    _att = next(
                        (a for a in _sub_ring
                         if any(nb.GetIdx() == ring_atom_idx for nb in
                                mol.GetAtomWithIdx(a).GetNeighbors())),
                        None)
                    if _att is not None:
                        _dec = decorated_ring_substituent_name(
                            mol, _sub_ring, _att,
                            expected_atoms=set(sub_atoms))
                        if _dec is not None:
                            ring_name = _dec

            sub_info = {
                'atoms': sub_atoms,
                'is_on_nitrogen': is_nitrogen,
                'carbon_count': carbon_count,
                'connecting_atom': ring_atom_idx,
                'is_ring': is_ring,
                'ring_name': ring_name,
            }
            # / ring-suffix fix : is THIS no-carbon group the principal group?
            # If its prefix form matches the principal group's prefix and that class
            # has a ring suffix, emit it as the suffix (-one/-ol/-amine/...) instead
            # of a prefix. The matched ring-suffix takes priority over carbon-suffix
            # detection (suffix_info), since the principal group is, by seniority,
            # senior to or equal to those.
            is_principal_suffix = (
                pg_ring_suffix is not None
                and hetero_sub_name is not None
                and hetero_sub_name == pg_prefix
            )

            # /: a ring carbonyl whose principal group is a
            # CYCLIC secondary amide that the lactam handler DECLINED (a
            # multi-heteroatom saturated heteroring, e.g. 1,3-thiazolidin-4-one)
            # is named as the ketone '-one' suffix on the heterocycle parent,
            # NOT an 'oxo' prefix (the lactam/carboxamide name is unavailable).
            # Single-heteroatom lactams route through the lactam handler and
            # never reach here, so they are untouched.
            #
            # R1-FIX / a review-A "Pseudoketones: (a) cyclic compounds in
            # which a carbonyl group in a ring is bonded to one or two skeletal
            # heteroatoms" -> named with the '-one' SUFFIX; cf. the Blue Book
            # `imidazolidine-2,4-dione (PIN)`): a TERTIARY (N-substituted) cyclic
            # urea / ring amide is the same pseudoketone. When R1 stopped
            # perceiving cyclic ureas as urea (its `!R` on both N slots), the
            # N-substituted ones (`CN1CCN(C)C1=O`) perceived a tertiary_amide with
            # NO lactam name and fell through to the 'oxo' PREFIX
            # (`1,3-dimethyl-2-oxoimidazolidine`, non-PIN). The N-H cyclic ureas
            # already reach the '-one' suffix here via secondary_amide, so admit
            # tertiary_amide on the identical gate (lactam-declined multi-heteroatom
            # ring, ring carbonyl). Single-N tertiary lactams (N-methyl-2-
            # piperidinone) still route through the lactam handler (it accepts an
            # N-substituent) and never reach here.
            ring_ketone_suffix = (
                not is_principal_suffix
                and principal_group in ('secondary_amide', 'tertiary_amide')
                and hetero_sub_name == 'oxo'
                and _is_monocyclic_lactam(mol) is None
                and _ring_has_extra_heteroatom(mol, ring_set, ring_atom_idx)
            )

            # / "'Hidden' amides" (the Blue Book,:33847): a CYCLIC
            # imide -- a ring N flanked by two ring carbonyls -- is a ring
            # pseudoketone named with the -DIONE suffix, NOT '2,5-dioxo...'
            # detachable prefixes: succinimide -> pyrrolidine-2,5-dione (PIN),
            # and the Blue Book spells `1-bromopyrrolidine-2,5-dione (PIN)` verbatim
            # "(not N-bromosuccinimide;...)". Each imide ring carbonyl's
            # exocyclic =O becomes a '-one' on the numbered ring, exactly as the
            # secondary_amide branch above does for a multi-heteroatom ring
            # amide. The lactam handler always DECLINES an imide
            # (is_monocyclic_lactam is None), so a single-N imide ring
            # (pyrrolidine / piperidine) needs no extra-heteroatom gate; the
            # `_is_cyclic_imide_carbonyl` check confines this to a carbonyl that
            # is genuinely part of the in-ring imide motif.
            ring_imide_suffix = (
                not is_principal_suffix
                and not ring_ketone_suffix
                and principal_group == 'imide'
                and hetero_sub_name == 'oxo'
                and _is_cyclic_imide_carbonyl(mol, ring_atom_idx, ring_set)
            )

            # a phase (A fix, review finding): a ring carbonyl whose
            # principal group is an ESTER that the lactone handler DECLINED
            # (is_monocyclic_lactone returns None -- a dione, or an
            # in-ring-C=C at HW ring size, a phase Task 1) is named as
            # the ketone '-one'/'-dione' SUFFIX on the heterocycle parent,
            # NOT an 'oxo'/'dioxo' detachable PREFIX.: the ring
            # carbonyl is the senior (only) characteristic group here, so
            # requires the suffix form (oxolane-2,4-dione, not
            # 2,4-dioxooxolane). Mirrors the secondary_amide/lactam branch
            # above; unlike that branch there is no "extra heteroatom" gate
            # needed -- is_monocyclic_lactone's own all-carbon-besides-O
            # scope already means any DECLINED lactone reaching here is a
            # single-ring-O heterocycle by construction.
            ring_lactone_suffix = (
                not is_principal_suffix
                and not ring_ketone_suffix
                and principal_group == 'ester'
                and hetero_sub_name == 'oxo'
                and _is_monocyclic_lactone(mol) is None
            )

            # (the Blue Book) chalcogen ketone-analogue SUFFIX: a ring
            # carbon's C=S/C=Se/C=Te whose principal group is a CYCLIC
            # thioamide/selenoamide/telluroamide (a "thiolactam" -- the
            # chalcogen analogue of the lactam the lactam handler declines) is
            # the ketone-analogue named with the -thione/-selone/-tellone
            # SUFFIX (multiplied -> -dithione...), NOT a sulfanylidene /
            # selanylidene / tellanylidene detachable prefix. Mirrors the
            # ring_ketone_suffix (lactam) branch, but there is NO thiolactam
            # handler (unlike lactams), so ALL such rings are named here --
            # single-N monocyclic (azepane-2-thione) and multi-heteroatom
            # (1,3-thiazolidine-2,4-dithione) alike; no lactam-declined /
            # extra-heteroatom gate is applicable. Seniority C=O > C=S > C=Se >
            # C=Te (the Blue Book) is already resolved UPSTREAM: when a ring also
            # bears a senior C=O the principal group is 'secondary_amide' (not
            # a *amide chalcogen class), so this branch never fires and the C=S
            # correctly stays a sulfanylidene prefix (rhodanine
            # O=C1CSC(=S)N1 -> 2-sulfanylidene-1,3-thiazolidin-4-one, unchanged).
            # Within-family seniority (S > Se > Te) is honoured by matching only
            # the chalcogen of the principal group: in a mixed ring only that
            # chalcogen becomes the suffix, the rest stay ylidene prefixes.
            # A ring thioKETONE (no ring N) already routes via is_principal_suffix
            # (thioketone's own prefix == 'sulfanylidene'), so it is untouched.
            _PG_TO_CHALCOGEN_SUFFIX = {
                'thioamide': ('sulfanylidene', 'thione'),
                'selenoamide': ('selanylidene', 'selone'),
                'telluroamide': ('tellanylidene', 'tellone'),
            }
            _chalcogen_amide = _PG_TO_CHALCOGEN_SUFFIX.get(principal_group)
            ring_thioamide_suffix = (
                not is_principal_suffix
                and not ring_ketone_suffix
                and not ring_lactone_suffix
                and not ring_imide_suffix
                and _chalcogen_amide is not None
                and hetero_sub_name == _chalcogen_amide[0]
            )

            if is_principal_suffix:
                sub_info['is_suffix'] = True
                sub_info['suffix_name'] = pg_ring_suffix
            elif ring_ketone_suffix or ring_lactone_suffix or ring_imide_suffix:
                sub_info['is_suffix'] = True
                sub_info['suffix_name'] = 'one'
            elif ring_thioamide_suffix:
                sub_info['is_suffix'] = True
                sub_info['suffix_name'] = _chalcogen_amide[1]
            else:
                if hetero_sub_name:
                    sub_info['hetero_name'] = hetero_sub_name
                if suffix_info:
                    sub_info['is_suffix'] = True
                    sub_info['suffix_name'] = suffix_info['suffix_name']
                    if suffix_info.get('n_hydroxy'):
                        sub_info['n_hydroxy'] = True
                    # Wave2 T5d: N-substituted ring carboxamide — thread the
                    # detected N-substituent names to the suffix N-prefix
                    # builder (the C4 amine mechanism).
                    if suffix_info.get('n_substituents'):
                        sub_info['amine_n_substituents'] = [
                            _s['name'] if isinstance(_s, dict) else _s
                            for _s in suffix_info['n_substituents']
                        ]

            if locant not in substituents:
                substituents[locant] = []
            substituents[locant].append(sub_info)

    return substituents


def _identify_suffix_fg(mol, start_idx: int, sub_atoms, ring_set) -> Optional[Dict]:
    """
    Identify suffix-type functional groups attached to ring.

    Checks if the substituent starting at start_idx is a functional group
    that should be expressed as a suffix on the ring parent name.

    Returns:
        Dict with 'suffix_name' if FG found, None otherwise
    """
    start_atom = mol.GetAtomWithIdx(start_idx)
    if start_atom.GetSymbol() != 'C':
        return None

    # Check for carboxylic acid: C(=O)(OH)
    acid_pat = _compiled_smarts('[CX3](=O)[OX2H1]')
    if acid_pat:
        for match in mol.GetSubstructMatches(acid_pat):
            if match[0] == start_idx:
                return {'suffix_name': 'carboxylic acid'}

    # Check for aldehyde: C(=O)H attached to ring
    ald_pat = _compiled_smarts('[CX3H1](=O)')
    if ald_pat:
        for match in mol.GetSubstructMatches(ald_pat):
            if match[0] == start_idx:
                return {'suffix_name': 'carbaldehyde'}

    # Check for hydroxamic acid: C(=O)(NH-OH) — before primary amide because
    # N has H1 bonded to O, not H2, so the primary amide pattern would not match.
    hydroxamic_pat = _compiled_smarts('[CX3](=O)[NX3;H1][OX2H]')
    if hydroxamic_pat:
        for match in mol.GetSubstructMatches(hydroxamic_pat):
            if match[0] == start_idx:
                return {'suffix_name': 'carboxamide', 'n_hydroxy': True}

    # C1: ring-attached hydrazide C(=O)-NH-NH2 -> '-carbohydrazide'.
    # Checked BEFORE the amide pattern: the hydrazide N is bonded to another N so
    # the amide SMARTS never matches it (pyridine-4-carbohydrazide,
    # furan-2-carbohydrazide).
    hydrazide_pat = _compiled_smarts('[CX3](=O)[NX3][NX3]')
    if hydrazide_pat:
        for match in mol.GetSubstructMatches(hydrazide_pat):
            if match[0] == start_idx:
                return {'suffix_name': 'carbohydrazide'}

    # Check for primary amide: C(=O)(NH2)
    amide_pat = _compiled_smarts('[CX3](=O)[NX3H2]')
    if amide_pat:
        for match in mol.GetSubstructMatches(amide_pat):
            if match[0] == start_idx:
                return {'suffix_name': 'carboxamide'}

    # Wave2: N-SUBSTITUTED ring carboxamides — secondary
    # C(=O)NHR and tertiary C(=O)NR2. The benzene path has handled these
    # since a phase (N,N-dimethylbenzamide); the heterocycle detector only
    # matched primary amides, so every N-substituted heterocycle-carboxamide
    # fell to unknown. Reuses the benzene N-substituent extractor; FAIL-CLOSED
    # count check: _detect_n_substituents silently omits branches it cannot
    # name, so a mismatch with the N's real branch count declines the whole
    # suffix record (missing-beats-wrong) rather than dropping a substituent.
    for _pat_smarts in ('[CX3](=O)[NX3;H1][#6]', '[CX3](=O)[NX3;H0]([#6])[#6]'):
        _pat = _compiled_smarts(_pat_smarts)
        if _pat is None:
            continue
        for match in mol.GetSubstructMatches(_pat):
            if match[0] != start_idx:
                continue
            # exclude hydrazide/hydroxamic shapes (N bonded to N/O) — those
            # were claimed by the earlier blocks; defensive re-check
            _n_atom = next(
                (mol.GetAtomWithIdx(i) for i in match
                 if mol.GetAtomWithIdx(i).GetSymbol() == 'N'), None)
            if _n_atom is None or any(
                nb.GetSymbol() in ('N', 'O') for nb in _n_atom.GetNeighbors()
                if nb.GetIdx() != start_idx
                and mol.GetBondBetweenAtoms(
                    _n_atom.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 1.0
                and nb.GetIdx() != start_idx
            ):
                continue
            from .benzene import _detect_n_substituents
            n_subs = _detect_n_substituents(mol, match, set(ring_set))
            _expected = sum(
                1 for nb in _n_atom.GetNeighbors()
                if nb.GetAtomicNum() > 1 and nb.GetIdx() != start_idx
            )
            if not n_subs or len(n_subs) != _expected:
                return None  # unnameable N-branch -> decline (fail-closed)
            return {'suffix_name': 'carboxamide', 'n_substituents': n_subs}

    # Check for nitrile: C#N
    nitrile_pat = _compiled_smarts('[CX2]#[NX1]')
    if nitrile_pat:
        for match in mol.GetSubstructMatches(nitrile_pat):
            if match[0] == start_idx:
                return {'suffix_name': 'carbonitrile'}

    return None


def _identify_hetero_substituent(mol, sub_atoms, ring_set) -> Optional[str]:
    """
    Identify a non-carbon functional substituent (amino, hydroxy, nitro, etc.).

    Args:
        mol: RDKit Mol object
        sub_atoms: List of atom indices in the substituent
        ring_set: Set of ring atom indices

    Returns:
        Substituent prefix name or None if not recognized
    """
    if not sub_atoms:
        return None

    first_atom = mol.GetAtomWithIdx(sub_atoms[0])
    symbol = first_atom.GetSymbol()
    h_count = first_atom.GetTotalNumHs()

    # Halogens (F, Cl, Br, I) -- IUPAC
    _HALOGEN_PREFIX = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
    if symbol in _HALOGEN_PREFIX:
        return _HALOGEN_PREFIX[symbol]

    # Amino (-NH2)
    if symbol == 'N' and h_count == 2:
        return 'amino'

    # Hydrazinyl (-NH-NH2) -- retained substituent prefix, but ONLY when
    # the parent ring holds nitrogen. The attachment atom (sub_atoms[0], the BFS
    # root) is a NEUTRAL -NH- single-bonded to the ring and to a terminal -NH2, and
    # the whole substituent is EXACTLY those two N atoms. Without this a ring-borne
    # hydrazine group is flagged 'unnameable' and the whole heterocycle candidate
    # declines (BB 19376 `2-hydrazinylpyridine` (PIN); also
    # 2-hydrazinyl-4,5-dihydro-1H-imidazole).
    #
    # ⚠ SENIORITY SCOPE. Hydrazine's senior skeletal element is N (the top
    # of the order N >... > O > S >... > C). A ring is the senior parent
    # over the 2-N chain ONLY when it, too, holds N -- then 's
    # ring-senior-to-chain tie-break fires (pyridine/pyrimidine/pyrrole/dihydro-
    # imidazole). A ring whose senior element is JUNIOR to N (all-carbon benzene/
    # cyclohexane, or an O/S/Si heterocycle) is junior to the hydrazine chain, so
    # hydrazine is the parent and the PIN is `<ring-yl>hydrazine`, NOT
    # `hydrazinyl<ring>` -- e.g. `phenylhydrazine`, and BB 18950
    # `1-(2H-pyran-3-yl)-2-(silolan-2-yl)hydrazine` (PIN) keeps hydrazine the parent
    # even against two heterocyclic ring substituents. So gate on a nitrogen in the
    # parent ring system; the non-N rings fall through to 'unnameable' (rules/
    # polyazane owns the hydrazine-parent spelling for them). This adds no parent
    # candidate, so phenylhydrazine / methylhydrazine are untouched. A substituted
    # hydrazinyl (-NH-NHR / -N(R)-NH2) carries carbons and never reaches this
    # carbon_count==0 path.
    if (symbol == 'N' and h_count == 1 and first_atom.GetFormalCharge() == 0
            and len(sub_atoms) == 2
            and any(mol.GetAtomWithIdx(i).GetSymbol() == 'N' for i in ring_set)):
        _term = mol.GetAtomWithIdx(sub_atoms[1])
        _nn = mol.GetBondBetweenAtoms(sub_atoms[0], sub_atoms[1])
        if (_term.GetSymbol() == 'N' and _term.GetFormalCharge() == 0
                and _term.GetTotalNumHs() == 2
                and _nn is not None
                and _nn.GetBondType() == Chem.BondType.SINGLE):
            return 'hydrazinyl'

    # Hydroperoxy (-O-OH): two chained O, the attachment O bears no H and the
    # terminal O bears one.: on a ring heteroatom this is the -peroxol
    # principal group; returning the matching 'hydroperoxy' prefix name lets the
    # ring-suffix path recognise it (hetero_sub_name == pg_prefix) and attach the
    # '-peroxol' suffix (pyrrolidine-1-peroxol), rather than emit a 'hydroxyoxy' prefix.
    if (symbol == 'O' and h_count == 0 and len(sub_atoms) == 2):
        _term = mol.GetAtomWithIdx(sub_atoms[1])
        if _term.GetSymbol() == 'O' and _term.GetTotalNumHs() == 1:
            return 'hydroperoxy'

    # Hydroxy (-OH)
    if symbol == 'O' and h_count == 1:
        return 'hydroxy'

    # Nitro (-NO2)
    if symbol == 'N' and first_atom.GetFormalCharge() == 1:
        o_count = sum(1 for idx in sub_atoms if mol.GetAtomWithIdx(idx).GetSymbol() == 'O')
        if o_count == 2:
            return 'nitro'

    # Nitroso (-N=O) -- /. The attach atom is a NEUTRAL N
    # carrying exactly one terminal, doubly-bonded O (and nothing else exocyclic
    # to the ring). Mirrors the substituent_enumerator guard (BB nitroso morpheme)
    # so a nitrite -O-N=O (attach O) or an N-oxide can never match. Without this
    # branch an N-nitroso ring (e.g. N-nitrosoproline / its methyl ester) was
    # flagged 'unnameable' and the whole heterocycle candidate declined, even
    # though bare N-nitrosopyrrolidine already names — the acid/ester suffix path
    # simply never reached the nitroso morpheme.
    if (symbol == 'N' and first_atom.GetFormalCharge() == 0
            and len(sub_atoms) == 2):
        _o_idxs = [idx for idx in sub_atoms
                   if mol.GetAtomWithIdx(idx).GetSymbol() == 'O']
        if len(_o_idxs) == 1:
            _o = mol.GetAtomWithIdx(_o_idxs[0])
            _b = mol.GetBondBetweenAtoms(sub_atoms[0], _o_idxs[0])
            if (_o.GetDegree() == 1 and _o.GetFormalCharge() == 0
                    and _b is not None
                    and _b.GetBondType() == Chem.BondType.DOUBLE):
                return 'nitroso'

    # Sulfanyl (-SH)
    if symbol == 'S' and h_count == 1:
        return 'sulfanyl'

    # R2, the Blue Book + Table 6.2, the Blue Book): a sulfur-oxo-acid or a
    # primary sulfonamide whose S is directly attached to the ring is the ring
    # principal-group SUFFIX. Returning the matching preselected prefix
    # (Table 6.2: 'sulfo' / 'sulfino';: 'sulfamoyl') lets the ring-suffix
    # path recognise it (hetero_sub_name == pg_prefix) and attach the
    # '-sulfonic acid' / '-sulfinic acid' / '-sulfonamide' suffix on the ring
    # parent, exactly the hydroperoxy -> '-peroxol' mechanism above. Without it
    # this no-carbon S group is flagged 'unnameable' and the whole heterocycle
    # candidate declines (piperidine-4-sulfonic acid, and the ring-N forms
    # piperidine-1-sulfonic/sulfinic acid). Scoped to a NEUTRAL S bearing =O and
    # a terminal -OH (acid) or a terminal primary -NH2 (sulfonamide), all inside
    # this substituent; a free oxoacid never reaches here (its S is not a ring
    # substituent), and an acyclic N stays a noncarbon oxoacid upstream.
    if symbol == 'S' and h_count == 0 and first_atom.GetFormalCharge() == 0:
        _sub_set = set(sub_atoms)
        _dbl_o = sum(
            1 for b in first_atom.GetBonds()
            if b.GetBondTypeAsDouble() == 2.0
            and b.GetOtherAtom(first_atom).GetSymbol() == 'O'
            and b.GetOtherAtom(first_atom).GetIdx() in _sub_set
        )
        _oh = any(
            n.GetSymbol() == 'O' and n.GetTotalNumHs() == 1
            and n.GetIdx() in _sub_set
            and mol.GetBondBetweenAtoms(
                first_atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1.0
            for n in first_atom.GetNeighbors()
        )
        _nh2 = any(
            n.GetSymbol() == 'N' and n.GetTotalNumHs() == 2
            and n.GetFormalCharge() == 0 and n.GetDegree() == 1
            and n.GetIdx() in _sub_set
            for n in first_atom.GetNeighbors()
        )
        if _oh and _dbl_o == 2:
            return 'sulfo'        # -S(=O)(=O)-OH (sulfonic acid,
        if _oh and _dbl_o == 1:
            return 'sulfino'      # -S(=O)-OH (sulfinic acid,
        if _nh2 and not _oh and _dbl_o == 2:
            return 'sulfamoyl'    # -S(=O)(=O)-NH2 (sulfonamide,

    # Oxo (=O) - check if double-bonded to ring carbon
    if symbol == 'O' and h_count == 0 and len(sub_atoms) == 1:
        for bond in first_atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(sub_atoms[0])
            if other_idx in ring_set and bond.GetBondType() == Chem.BondType.DOUBLE:
                return 'oxo'

    # Chalcogen ylidene (=S/=Se/=Te) on a ring carbon: the heavier
    # chalcogen analogues of 'oxo'. When a ring C=O is present it is the senior
    # ('-one' suffix, chalcogen order O>S>Se>Te) and the C=S/C=Se/C=Te is cited
    # as a 'sulfanylidene'/'selanylidene'/'tellanylidene' detachable prefix at
    # its ring locant (2-sulfanylidene-1,3-thiazolidin-4-one). Without this the
    # exocyclic chalcogen was mis-flagged 'unnameable' and the whole heterocycle
    # candidate declined.
    _CHALCOGEN_YLIDENE = {'S': 'sulfanylidene', 'Se': 'selanylidene',
                          'Te': 'tellanylidene'}
    if symbol in _CHALCOGEN_YLIDENE and h_count == 0 and len(sub_atoms) == 1:
        for bond in first_atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(sub_atoms[0])
            if other_idx in ring_set and bond.GetBondType() == Chem.BondType.DOUBLE:
                return _CHALCOGEN_YLIDENE[symbol]

    # Imino (=NH) -- IUPAC
    if symbol == 'N' and h_count == 1 and len(sub_atoms) == 1:
        for bond in first_atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(sub_atoms[0])
            if other_idx in ring_set and bond.GetBondType() == Chem.BondType.DOUBLE:
                return 'imino'

    return None


def _bfs_substituent(mol, start_idx: int, excluded: Set[int]) -> List[int]:
    """
    BFS to find all atoms in a substituent.

    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index (first atom of substituent)
        excluded: Set of atom indices to exclude (ring atoms)

    Returns:
        List of atom indices in the substituent
    """
    visited = set()
    queue = deque([start_idx])
    result = []

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in excluded:
            continue

        visited.add(atom_idx)
        result.append(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in excluded:
                queue.append(nbr_idx)

    return result


# (heteroatom-locant citation for retained Hantzsch-Widman-derived
# saturated hetero-ring parents when they carry a characteristic-group suffix).
# Bare parent names ('thiazolidine') carry the numbering implicitly, but a
# suffixed PIN must cite the heteroatom locant set immediately before the parent
# stem: '1,3-thiazolidin-4-one', '1,3-oxazolidin-2-one'. Only these two retained
# stems need the explicit locant citation, because only for them is the BARE stem
# non-preferred: (``the Blue Book``) prints "oxazolidine
# 1,3-oxazolidine (PIN) thiazolidine (S instead of O) 1,3-thiazolidine (PIN)".
#
# ⚠ CORRECTED 2026-08-03. This comment used to justify the exclusions with
# "(and OPSIN rejects) an explicit heteroatom-locant prefix". **That is false.**
# Measured against OPSIN 2.9 (oracle validated on two known positives plus a
# negative control): `1,2-thiazolidine`, `1,2-oxazolidine`, `1,2-diazolidine`,
# `1,3-diazolidine`, `1,3-thiazolidine`, `1,3-oxazolidine`, `1,4-oxazinane` and
# `1,4-diazinane` **all eight parse**. OPSIN rejects none of them.
#
# The EXCLUSIONS are nonetheless correct, on the real reason -- those families
# already spell their PIN without a locant, so there is nothing to add:
# * the 1,2- forms are handled upstream and already emit their PINs
# (``C1CNSC1`` -> `1,2-thiazolidine`, ``C1CNOC1`` -> `1,2-oxazolidine`,
# both marked (PIN) at ``:8184`` against the non-preferred `isothiazolidine`
# / `isoselenazolidine`);
# * `pyrazolidine`, `imidazolidine`, `morpholine`, `piperidine`, `piperazine`
# are retained Table 2.3 names that ARE the preferred names, and are emitted
# as such (``C1CNNC1`` -> `pyrazolidine`, ``C1CNCN1`` -> `imidazolidine`,
# ``C1COCCN1`` -> `morpholine`).
# Behaviour is unchanged by this correction; only the stated rationale was wrong.
# A false rationale is how a missing check survives review, which is why it is
# corrected rather than deleted. Value = expected element-ordered heteroatom
# locant set; the fix is gated on a match against the actual ring numbering so it
# fails closed if the numbering is not deterministically the canonical one.
_RETAINED_HETERO_LOCANT_CITATION = {
    'thiazolidine': [1, 3],   # S=1, N=3
    'oxazolidine': [1, 3],    # O=1, N=3
}


def _retained_heteroatom_locant_prefix(
    mol, ring_atoms, parent_name: str, atom_to_locant: Dict[int, int]
) -> str:
    """Return the heteroatom-locant prefix ('1,3-') for a suffixed retained
    Hantzsch-Widman-derived saturated hetero-ring parent, else '' (fail-closed).

    Only fires for the curated `_RETAINED_HETERO_LOCANT_CITATION` stems, and only
    when the ring's actual heteroatom locants (from `atom_to_locant`) match the
    stem's canonical set exactly — so a nonstandard numbering never fabricates a
    wrong locant string.
    """
    expected = _RETAINED_HETERO_LOCANT_CITATION.get(parent_name)
    if expected is None:
        return ''
    ring_set = set(ring_atoms) if ring_atoms else set()
    hetero_locants = sorted(
        loc for aidx, loc in atom_to_locant.items()
        if aidx in ring_set and mol.GetAtomWithIdx(aidx).GetSymbol() != 'C'
    )
    if hetero_locants != expected:
        return ''
    return ",".join(str(loc) for loc in expected) + "-"


def _n_anchored_substituent_covers(mol, sub_name: str, sub_atoms) -> bool:
    """Gate-INDEPENDENT atom-coverage check for an N/O/S-anchored ring substituent
    named by the recursive ``name_substituent``.

     #29: the recursive namer roots N-anchored compound substituents correctly
    (acetamido / methylamino / methanesulfonamido) but can also degrade a shape it
    only partly understands to a shorter prefix (an arenesulfonamido collapsing to
    bare ``amino``), which would drop atoms. Verify the prefix accounts for exactly
    the fragment's heavy atoms by OPSIN-parsing ``<sub_name>benzene`` and comparing
    the heavy-atom count minus benzene's 6. Fail CLOSED on any parse failure /
    missing jar / mismatch, so a partial name never ships (honest without)."""
    if not sub_name:
        return False
    frag_heavy = sum(1 for i in sub_atoms
                     if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
    from ..namer import _validity_gate_name_to_smiles
    probe = _validity_gate_name_to_smiles(f"{sub_name}benzene")
    if not probe:
        return False
    pm = Chem.MolFromSmiles(probe)
    if pm is None:
        return False
    return pm.GetNumHeavyAtoms() - 6 == frag_heavy


def name_substituted_heterocycle(
    mol,
    ring_atoms,
    parent_name: str,
    substituents: Dict[int, List[Dict]],
    atom_to_locant: Dict[int, int],
    principal_group: Optional[str] = None
) -> Optional[str]:
    """
    Assemble complete name for a substituted heterocycle.

    Returns None (the heterocycle candidate declines) when an exocyclic
    substituent cannot be named correctly and completely — naming it partially
    would silently drop atoms (sugar-ring-oxygen-drop / structure loss). Callers treat a
    None return as a fail-closed decline.

    N-substituted groups use N-locant format (N-methyl, N,N-dimethyl).
    C-substituted groups use numeric locants (2-methyl, 3-ethyl).
    Ring substituents use proper ring names (piperidinyl, phenyl) not carbon counts.
    All prefixes are sorted alphabetically (ignoring N-, numbers, multipliers).
    Stereochemistry descriptors (R/S) are added as a prefix if present.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring
        parent_name: Base heterocycle name (e.g., 'pyrrolidine', 'pyridine')
        substituents: Dict from get_heterocycle_substituents
        atom_to_locant: Dict mapping atom_idx to locant (1-indexed)

    Returns:
        Complete IUPAC name (e.g., 'N-methylpyrrolidine', '3-methylpyridine',
        '(2S)-N-methyl-2-propylpiperidine')

    Examples:
        >>> mol = Chem.MolFromSmiles('CN1CCCC1') # N-methylpyrrolidine
        >>> #... get ring, orient, get substituents...
        >>> name_substituted_heterocycle(mol, ring, 'pyrrolidine', subs, atom_to_loc)
        'N-methylpyrrolidine'
    """
    from ..assembly.naming_utils import (
        _join_multiplied_suffix,  # / multiplier-'a' elision (tetraol->tetrol)
        alpha_sort_key,
        prefix_citation_sort_key,
        # a phase: the italic-N substituent of a ring-amine /
        # ring-carboxamide suffix is enclosed iff it is a COMPOUND or COMPLEX
        # prefix. _wrap_n_substituent only ESCALATES a name that already carries
        # parentheses ("Simple names (no parentheses) are returned unchanged"),
        # so a bare compound prefix such as '2-bromophenyl' was cited naked ->
        # 'N-2-bromophenylpyridin-2-amine'. enclose_if_compound is the primitive
        # that answers the question, and it subsumes the escalation
        # (-> [ -> {) that _wrap_n_substituent provided. Safe at these call
        # sites specifically because the multiplying prefix is applied OUTSIDE
        # the call (f"N,N-{_mp}{...(_subs[0])}"), so the argument is always the
        # SINGULAR substituent name -- enclose_if_compound('dimethyl') would
        # wrongly give '(dimethyl)', but 'dimethyl' is never passed here.
        enclose_if_compound,
        get_alkyl_name,
        get_multiplier_prefix,
        get_suffix_multiplier_prefix,
    )
    from ..perception.stereo import assign_stereochemistry
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    # Ensure CIP labels are assigned (idempotent guard)
    assign_stereochemistry(mol)

    # Collect stereodescriptors using the heterocycle locant mapping
    stereo_descriptors = collect_stereodescriptors(mol, atom_to_locant)

    if not substituents:
        # No substituents but may have stereo
        if stereo_descriptors:
            stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
            return f"{stereo_prefix}{parent_name}"
        return parent_name

    # Separate suffix-type FGs from prefix substituents
    suffix_fg: Dict[str, List[int]] = {}  # suffix_name -> [locants]
    suffix_n_hydroxy: set = set()  # suffix_names whose FG was a hydroxamic acid
    # C4: N-substituents carried on the amine ring-suffix (pyridin-4-amine ->
    # N-methyl/N-phenyl). Keyed by suffix_name; single-instance only in scope.
    suffix_n_substituents: Dict[str, List[str]] = {}
    # companion: for a MULTI-amine ring parent (triazine-2,4-diamine
    # etc.) the N-substituents on each amine nitrogen must carry that nitrogen's
    # RING locant as an italic-N superscript (N2-tert-butyl-N4-cyclopropyl-...),
    # so track them per ring locant (not merged under the single 'amine' key,
    # which collapsed them to one bare 'N-...' and dropped the others).
    amine_n_by_locant: Dict[int, List[str]] = {}
    prefix_substituents: Dict[int, List[Dict]] = {}

    for locant, sub_list in substituents.items():
        for sub_info in sub_list:
            if sub_info.get('unnameable'):
                # An exocyclic group we cannot name correctly and completely
                # (e.g. the oxygen of a sulfate/phosphate ester). Decline the
                # whole heterocycle candidate rather than silently drop the
                # group (sugar-ring-oxygen-drop / structure loss). Fail-closed.
                return None
            if sub_info.get('is_suffix'):
                sname = sub_info['suffix_name']
                if sname not in suffix_fg:
                    suffix_fg[sname] = []
                suffix_fg[sname].append(locant)
                if sub_info.get('n_hydroxy'):
                    suffix_n_hydroxy.add(sname)
                if sub_info.get('amine_n_substituents'):
                    suffix_n_substituents[sname] = sub_info['amine_n_substituents']
                    if sname == 'amine':
                        amine_n_by_locant[locant] = sub_info['amine_n_substituents']
            else:
                if locant not in prefix_substituents:
                    prefix_substituents[locant] = []
                prefix_substituents[locant].append(sub_info)

    # Group prefix substituents by name and N/C classification
    n_groups: Dict[str, List[int]] = {}  # N-substituents: name -> locants
    c_groups: Dict[str, List[int]] = {}  # C-substituents: name -> locants

    for locant, sub_list in prefix_substituents.items():
        for sub_info in sub_list:
            is_on_n = sub_info['is_on_nitrogen']

            # Check if this is a ring substituent
            is_ring = sub_info.get('is_ring', False)
            ring_name = sub_info.get('ring_name')

            if 'prefix_name' in sub_info:
                #: the attachment bond is double or triple, and
                # get_heterocycle_substituents already built the only prefix
                # that spells that free valence (methylidene, ethylidene,...).
                # Checked FIRST because every branch below would re-derive the
                # name from the carbon count, which is exactly the reading that
                # cannot see the bond order.
                sub_name = sub_info['prefix_name']
            elif is_ring and ring_name:
                # Use ring substituent name (piperidinyl, phenyl, etc.)
                sub_name = ring_name
            elif 'hetero_name' in sub_info:
                # Non-carbon functional substituent (amino, hydroxy, etc.)
                sub_name = sub_info['hetero_name']
            else:
                # Name the substituent fragment. A multi-atom substituent —
                # whether pure C/H (branched alkyl) OR carbon-anchored with
                # heteroatoms (hydroxymethyl, methoxymethyl, aminomethyl) — is
                # named by the recursive fragment namer. Naming a heteroatom-
                # bearing substituent from its carbon count alone silently drops
                # the heteroatoms (sugar-ring-oxygen-drop, structure loss), so the
                # carbon-count fallback below is reserved for pure C/H fragments.
                carbon_count = sub_info['carbon_count']
                sub_atoms = sub_info.get('atoms', [])
                sub_name = None
                all_c_h = bool(sub_atoms) and all(
                    mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'H')
                    for i in sub_atoms
                )
                # The fragment namer is reliable only when the substituent is
                # anchored to the ring through a CARBON; an O-/N-/S-anchored
                # substituent (e.g. alkoxy) is mis-named by it, so such a
                # fragment fails closed below instead.
                attach_is_carbon = bool(sub_atoms) and (
                    mol.GetAtomWithIdx(sub_atoms[0]).GetSymbol() == 'C'
                )
                if sub_atoms and len(sub_atoms) > 1 and (all_c_h or attach_is_carbon):
                    from ..assembly.substituent_naming import name_substituent_fragment
                    attach_idx = sub_atoms[0]
                    ring_set_local = set(ring_atoms) if ring_atoms else set()
                    sub_name = name_substituent_fragment(
                        mol, sub_atoms, attach_idx, list(ring_set_local)
                    )
                # #29: an N/O/S-anchored COMPOUND substituent (acetamido,
                # methylamino, methanesulfonamido,...). name_substituent_fragment
                # above is carbon-anchored only and mis-roots these (an N-attached
                # -NHC(=O)CH3 came back 'carbamoylmethyl'), which is why the block
                # above excludes them and the fail-closed below used to decline the
                # whole heterocycle -> the 2-<N-substituent>-1,3-thiazole-5-carboxylic
                # acid class abstained at PIN (named only at best-effort by the general
                # aromatic engine). The recursive name_substituent (C4 keystone) roots
                # them correctly; route through it, GUARDED by a gate-independent
                # atom-coverage check so a partial name (an arenesulfonamido degrading
                # to bare 'amino') fails closed instead of shipping a dropped
                # constitution at T4. Enclosing marks are applied downstream by
                # _format_c_substituent (methylamino -> (methylamino)).
                if (sub_name is None and sub_atoms and len(sub_atoms) > 1
                        and not attach_is_carbon and not all_c_h):
                    from ..assembly.substituent_enumerator import (
                        name_substituent as _rec_name_substituent,
                    )
                    _cand = _rec_name_substituent(mol, list(sub_atoms), sub_atoms[0])
                    if _cand and _n_anchored_substituent_covers(mol, _cand, sub_atoms):
                        sub_name = _cand
                if sub_name is None:
                    if not all_c_h:
                        # A heteroatom-bearing substituent the fragment namer
                        # could not name correctly and completely -> decline the
                        # whole heterocycle candidate rather than emit a
                        # heteroatom-dropping alkyl name (sugar-ring-oxygen-drop).
                        # Fail-closed per accuracy-first.
                        return None
                    try:
                        sub_name = get_alkyl_name(carbon_count)
                    except ValueError:
                        # Carbon count too large for the simple alkyl table;
                        # try recursive naming as a last resort (pure C/H here).
                        if sub_atoms and len(sub_atoms) > 1:
                            from ..assembly.substituent_naming import name_substituent_fragment
                            attach_idx = sub_atoms[0]
                            ring_set_local = set(ring_atoms) if ring_atoms else set()
                            sub_name = name_substituent_fragment(
                                mol, sub_atoms, attach_idx, list(ring_set_local)
                            )
                        if sub_name is None:
                            logger.debug(
                                "ring_fragment_declined_by_ring_engine substituent_skip: reason=large_sub_still_unnameable carbon_count=%d",
                                carbon_count,
                            )
                            continue

            if is_on_n:
                if sub_name not in n_groups:
                    n_groups[sub_name] = []
                n_groups[sub_name].append(locant)
            else:
                if sub_name not in c_groups:
                    c_groups[sub_name] = []
                c_groups[sub_name].append(locant)

    # ------------------------------------------------------------------ #
    # (the Blue Book) -- ONE multiplying prefix over the WHOLE locant set. #
    # ------------------------------------------------------------------ #
    # "The basic numerical prefixes 'di', 'tri', 'tetra', etc. are used to
    # indicate a multiplicity of:", clause (b) (the Blue Book) "simple substituent
    # prefixes...", whose own example list prints `dimethyl` (the Blue Book). The
    # multiplicity is a property of the SUBSTITUENT NAME, not of which ring atom
    # carries it -- so a `methyl` on the ring N and a `methyl` on a ring C are ONE
    # group of two, and (the Blue Book, deny-by-default) then requires the whole
    # locant set to be cited: `1,2-dimethyl-1H-imidazole`.
    #
    # ⚠ This is NOT a name-string repair. `n_groups`/`c_groups` were two name-keyed
    # dicts split by `is_on_nitrogen`, which is a RING-ATOM test (`:2060`,
    # `ring_atom.GetSymbol == 'N'`) -- not a locant-KIND test. Each dict took its
    # own `count = len(locants)`, which is the whole defect: `Cn1c(C)nc(C)c1C` gave
    # `1-methyl-2,4,5-trimethyl-...`, i.e. the C-side collapse ALREADY worked and only
    # the ring-N methyl stood outside it. The multiplier machinery was never broken.
    #
    # A ring nitrogen that is a NUMBERED SKELETAL ATOM of the parent hydride takes an
    # ARABIC NUMERAL in a PIN; the italic 'N' belongs to a nitrogen that is not itself
    # numbered. "'Hidden' amides" (the Blue Book) is decisive on the direction:
    # naming an acyl group as a substituent on a heterocyclic ring nitrogen "is allowed
    # but only in general nomenclature", because "preferred IUPAC names are constructed"
    # as pseudoketones -- i.e. on the numbered ring. "Lactams and lactims"
    # (the Blue Book) says the same for this shape: of its two methods, "(1) as heterocyclic
    # pseudoketones" is the one that "generates preferred IUPAC names".
    #
    # The `(PIN)` examples spell the ring N as a numeral and print the italic form only
    # as the general-nomenclature alternative:
    # the Blue Book `pyrrolidine-1,2-diol (PIN) 1-hydroxypyrrolidin-2-ol
    # N-hydroxypyrrolidin-2-ol` <- N is ring locant 1
    # the Blue Book `1-bromopyrrolidine-2,5-dione (PIN) (not N-bromosuccinimide;...)`
    # the Blue Book `2,5-dioxopyrrolidin-1-yl (PIN) succinimidyl`
    #
    # And the Blue Book multiplies across MIXED italic/numeral sets -- in both of these
    # the ring N is the numeral and the exocyclic amine N keeps the italic letter:
    # the Blue Book `N,1,4-triphenyl-1H-1,2,4-triazol-4-ium-3-aminide (PIN)` and
    # the Blue Book `N,N,N,1-tetramethylquinolin-1-ium-3-aminium (PIN)`.
    #
    # ⚠ Do NOT re-derive any of this from (the Blue Book), which this comment
    # cited until 2026-08-02. Two independent faults: that sentence governs SUPERSCRIPTED
    # locants (N^2, N^3) and its section sits under, "di-, tri-, tetra-, and
    # polycarbonic acids" -- not ring nitrogens at all; and it reads "nitrogen atoms that
    # are not AMIDE LINKAGES that are part of the chain...", so quoting it with "amide
    # linkages" elided inverts the sentence. Its neighbour (the Blue Book) states
    # the italic-N convention cleanly but is scoped by the same chapter heading
    # ("Replacement by NH2 and NHNH2 groups"), so it is not the general rule either.
    #
    # The sibling producers already do this and are the model: `fused_rings.py`
    # leaves its `n_substituents` bucket EMPTY and routes ring-N locants as ordinary
    # numerals (measured: `_format_n_prefix` 0 calls for `1,2-dimethyl-1H-
    # benzimidazole`), and `benzene.py` uses one flat name-keyed bucket.
    #
    # Merge only when EVERY locant of the group is a number -- that is precisely the
    # `len(numeric) == count` test the N formatter applies below to choose the
    # numeral citation over the italic-'N' fallback, so a group that would have been
    # spelled `N-`/`N,N-` is left in `n_groups` and its spelling is untouched.
    #
    # ⚠ The licence below tested `not n_groups` to mean "this scope holds
    # no substituent on a ring nitrogen". After the merge an N-substituent can sit in
    # `c_groups`, so that dict is no longer the right witness and reading it would
    # silently WIDEN the licence (`4-methylmorpholine` -> `methylmorpholine`). The
    # question the licence asks is recorded here, before the buckets move.
    _has_n_substituent = bool(n_groups)
    for _n_name in list(n_groups):
        _n_locants = n_groups[_n_name]
        if _n_locants and all(loc is not None for loc in _n_locants):
            c_groups.setdefault(_n_name, []).extend(_n_locants)
            del n_groups[_n_name]

    # Build prefix parts
    prefix_parts = []

    # Format N-substituent prefixes (numeric ring locants = PIN,
    for name, locants in n_groups.items():
        count = len(locants)
        numeric = [loc for loc in locants if loc is not None]
        prefix = _format_n_substituent(
            name, count, locants=numeric if len(numeric) == count else None
        )
        prefix_parts.append((prefix, name))

    # ------------------------------------------------------------------ #
    # (the Blue Book) "Omission of locants" -- the ring-PREFIX case. #
    # ------------------------------------------------------------------ #
    # Sibling of the ring-SUFFIX licence further down (which serves
    # `pyrazinecarboxylic acid`, the Blue Book). BOTH halves of L3's own example block
    # are printed, and the prefix half is the one this branch serves:
    #
    # the Blue Book chlorocoronene (PIN)
    # the Blue Book chloropropanedioic acid (PIN) chloromalonic acid
    #
    # `chlorocoronene` is a monosubstituted symmetrical RING parent whose single
    # substituent PREFIX cites no locant -- exactly the shape of `chloropyrazine`.
    #
    # ⚠ NOT "symmetric heterocycle => omit". The decision is delegated whole to
    # assembly.locant_omission.l3_locant_omitted_for_parent_atoms, which measures the
    # ORBITS of the parent hydride's substitutable hydrogens
    # (CanonicalRankAtoms(breakTies=False)). That is the ONLY thing separating
    # pyrazine (four ring CH, ONE orbit -> omit) from pyridine (2/6, 3/5, 4 =
    # THREE orbits -> `4-chloropyridine` and `2-chloropyridine` KEEP their locants)
    # and from piperidine (N-H, C2/C6, C3/C5, C4 = FOUR orbits, the Blue Book
    # `piperidine-1-carbonitrile`). Measured on the parents themselves:
    # pyrazine True, pyridine False, piperidine False.
    #
    # (the Blue Book) is deny-by-default, so every locant that could share this
    # scope is excluded FIRST and anything not positively established retains:
    # * exactly ONE C-substituent prefix, ONE occurrence, ONE numeric locant
    # * no N-substituent prefixes -- those carry an ESSENTIAL italic-N or ring-N
    # locant, which restores every locant in the scope
    # * no suffix FG at all: a suffix cites its own locant (and the sibling
    # licence below owns that case), so a scope with both is not "the single
    # locant this licence may omit"
    # * a parent_name that is not purely alphabetic already cites a locant -- a
    # heteroatom set (`1,4-dioxane`) or an indicated hydrogen (`1H-pyrrole`)
    # * no stereodescriptors
    # The `_het_loc_prefix` injection cannot apply here: it is computed only inside
    # the `if suffix_fg:` block below, which this branch excludes.
    #
    # ⚠ `not suffix_fg` and `not _has_n_substituent` (the pre-merge witness recorded
    # above; it was `not n_groups` before merging moved numerically-locanted
    # ring-N groups into `c_groups`) are MUTATION-SURVIVING and DELIBERATELY
    # KEPT. Measured 2026-07-30: removing either changes no name, because the
    # licence's own STRUCTURAL monosubstitution proof
    # (`locant_omission._one_substituent_removed`) already refuses any scope holding
    # two substituent components -- for `OC(=O)c1cnc(Cl)cn1` (a suffix -COOH plus a
    # chloro prefix on the ONE-orbit pyrazine ring) it measures False, and for
    # `CN1CCC(C)CC1` both it and the orbit test measure False. A DOUBLE mutation
    # (drop `not suffix_fg` AND relax `_one_substituent_removed`) does move that name,
    # while relaxing `_one_substituent_removed` alone moves nothing -- so these are
    # mutually-covering preconditions in a stack, not dead code, and they state
    # 's *"then all locants must be cited"* for the halves of the scope this
    # branch is not allowed to inspect.
    _l3_omit_prefix_locant = False
    if (not _has_n_substituent and len(c_groups) == 1 and not suffix_fg
            and parent_name and parent_name.isalpha()
            and not stereo_descriptors):
        _only_locants = next(iter(c_groups.values()))
        # The THIRD scope check (the fragment-boundary observation): this licence
        # empties the whole name of locants, so it may only fire when the scope's
        # boundary IS the molecule being named. The two ambient declarations are
        # checked inside `locant_omission` itself.
        from ..assembly.handlers._handler_shared import (
            locant_scope_is_a_name_component,
        )
        if (len(_only_locants) == 1 and _only_locants[0] is not None
                and not locant_scope_is_a_name_component()):
            from ..assembly.locant_omission import (
                l3_locant_omitted_for_parent_atoms,
            )
            _l3_omit_prefix_locant = l3_locant_omitted_for_parent_atoms(
                mol, ring_atoms,
                prefix_locants=list(_only_locants),
                suffix_locants=[],
                parent_cites_locants=False,
                stereo_text="",
            )

    # Format C-substituent prefixes
    for name, locants in c_groups.items():
        count = len(locants)
        prefix = _format_c_substituent(
            name, sorted(locants), count, omit_locants=_l3_omit_prefix_locant)
        prefix_parts.append((prefix, name))

    # Sort in alphanumerical order by the base substituent name.
    # ⚠ Must be `prefix_citation_sort_key`, NOT raw `alpha_sort_key`: the latter
    # returns a flat string that KEEPS the substituent's internal locants, so a
    # compound substituent whose name begins with a locant digit (e.g.
    # `(1,3-oxazol-5-yl)methyl` -> key `1,3-oxazol-5-ylmethyl`, leading `1`) sorts
    # ahead of every letter-initial prefix (`chloro`) -- ASCII `1` (49) < `c` (99).
    # (the Blue Book) orders "Nonitalic Roman letters... first"; digits
    # are only a tie-break. `prefix_citation_sort_key` implements that as a
    # tuple (letters-only, then own-locants, then CIP), so `chloro` < `(1,3-oxazol-
    # 5-yl)methyl` again. `x[1]` is the BARE substituent name (the parent locant lives
    # in `x[0]`), so the default `parent_locants=False` convention is correct here.
    prefix_parts.sort(key=lambda x: prefix_citation_sort_key(x[1]))

    # Join prefixes
    # Closing brackets ],), } all act as word boundaries needing hyphens
    _CLOSING_MARKS = (')', ']', '}')
    prefix_str = ""
    for i, (prefix, _) in enumerate(prefix_parts):
        if i == 0:
            prefix_str = prefix
        else:
            # Add hyphen between prefixes if needed
            last_ch = prefix_str[-1]
            if (last_ch.isalpha() or last_ch in _CLOSING_MARKS) and prefix[0].isdigit():
                prefix_str += "-"
            elif (last_ch.isalpha() or last_ch in _CLOSING_MARKS) and prefix[0] == 'N':
                prefix_str += "-"
            prefix_str += prefix

    # Join prefix to parent name with correct hyphenation
    # When parent starts with digit (e.g., "1,4-dithiane"), need hyphen after prefix
    # When parent starts with letter (e.g., "pyridine"), no extra hyphen needed
    if prefix_str and parent_name:
        last_ch = prefix_str[-1]
        if parent_name[0].isdigit() and (last_ch.isalpha() or last_ch in _CLOSING_MARKS):
            combined = f"{prefix_str}-{parent_name}"
        else:
            combined = f"{prefix_str}{parent_name}"
    else:
        combined = f"{prefix_str}{parent_name}"

    # Add suffix-type functional groups
    if suffix_fg:
        from .seniority import get_suffix as _get_ring_suffix
        # / ring-suffix fix: the PRINCIPAL group's ring suffix wins (it is, by
        # seniority, senior to any co-present carb* suffix). Otherwise fall back
        # to the fixed carb* priority list.
        pg_ring_suffix = _get_ring_suffix(principal_group, is_ring=True) if principal_group else None
        _SUFFIX_PRIORITY = [
            'carboxylic acid', 'carboxamide', 'carbohydrazide',
            'carbonitrile', 'carbaldehyde',
        ]
        chosen_suffix = None
        chosen_locants = []
        if pg_ring_suffix and pg_ring_suffix in suffix_fg:
            chosen_suffix = pg_ring_suffix
            chosen_locants = sorted(suffix_fg[pg_ring_suffix])
        if not chosen_suffix:
            for suf in _SUFFIX_PRIORITY:
                if suf in suffix_fg:
                    chosen_suffix = suf
                    chosen_locants = sorted(suffix_fg[suf])
                    break
        if not chosen_suffix:
            chosen_suffix = next(iter(suffix_fg))
            chosen_locants = sorted(suffix_fg[chosen_suffix])

        count = len(chosen_locants)
        multiplier = get_suffix_multiplier_prefix(count, chosen_suffix) if count > 1 else ""
        locant_str = ",".join(str(loc) for loc in chosen_locants)
        # /: elide the multiplier's terminal 'a' before a
        # vowel-initial suffix ('tetra'+'ol' -> 'tetrol', not 'tetraol').
        suffix_token = _join_multiplied_suffix(multiplier, chosen_suffix)
        #: a suffixed retained HW-derived saturated hetero-ring
        # parent must cite its heteroatom locant set immediately before the
        # parent stem (thiazolidine -> 1,3-thiazolidin-4-one). Inject it at the
        # parent-stem boundary within `combined` (after any detachable prefix).
        _het_loc_prefix = _retained_heteroatom_locant_prefix(
            mol, ring_atoms, parent_name, atom_to_locant
        )
        if _het_loc_prefix and combined.endswith(parent_name):
            _stem_start = len(combined) - len(parent_name)
            _head = combined[:_stem_start]
            # A preceding detachable prefix ends in a letter/closing-mark; insert
            # a hyphen so '2-methyl' + '1,3-' reads '2-methyl-1,3-thiazolidin...'.
            if _head and (_head[-1].isalpha() or _head[-1] in (')', ']', '}')):
                _head += "-"
            combined = _head + _het_loc_prefix + parent_name
        # IUPAC (a): elide the parent's terminal 'e' before a suffix token
        # that begins with a vowel (piperidine -> piperidin-4-one,
        # pyridine -> pyridin-2-ol). A consonant-initial token (multiplied
        # 'dione'/'triol', or 'carb*') keeps the 'e' (piperidine-2,6-dione,
        # pyridine-3-carbaldehyde).
        if combined and combined[-1] == 'e' and suffix_token[:1].lower() in 'aeiouy':
            combined = combined[:-1]

        # ------------------------------------------------------------------ #
        # (the Blue Book) "Omission of locants" -- the ring-SUFFIX case. #
        # ------------------------------------------------------------------ #
        # "The locant is omitted in monosubstituted symmetrical parent hydrides or
        # parent compounds where there is only one kind of substitutable hydrogen."
        #
        # Its own example block prints `pyrazinecarboxylic acid (PIN)` (the Blue Book) --
        # a SUFFIX substitution, which is why this licence lives at the suffix join
        # and not only on the prefix side. Pyrazine's four ring CH are one
        # CanonicalRankAtoms(breakTies=False) orbit, so `2` carries no information.
        #
        # ⚠ This is NOT "symmetric heterocycle => omit", and it must not be rewritten
        # into that: the decision is delegated whole to
        # assembly.locant_omission.l3_locant_omitted_for_parent_atoms, the ONE place
        # the licences live, which measures the ORBITS of the substitutable
        # hydrogens. That is the only thing separating `pyrazinecarboxylic acid` from
        # `piperidine-1-carbonitrile (PIN)` (the Blue Book) -- piperidine's N-H, C2/C6,
        # C3/C5 and C4 are FOUR orbits, so it keeps its locant -- and from
        # `pyridine-4-carboxylic acid`, whose ring is three orbits.
        #
        # (the Blue Book) is deny-by-default, so every essential locant that could
        # share this scope is excluded FIRST, and anything not positively established
        # retains the locant:
        # * no substituent prefixes at all -> "monosubstituted"
        # * exactly one suffix group, one locant -> ditto
        # * no OTHER suffix FG (those become carboxy/cyano prefixes below)
        # * a parent_name that is not purely alphabetic already cites a locant --
        # a heteroatom set (`1,4-dioxane`) or an indicated hydrogen (`1H-pyrrole`)
        # -- and then restores all of them. All five printed L3 positives
        # have locant-free parent names, so this is the conservative side of a
        # boundary the Blue Book does not settle.
        # * no retained-heteroatom locant prefix, injected above)
        # * no N-substituent / N-hydroxy prefix -- those carry ESSENTIAL italic-N
        # locants and are prepended AFTER this join, so they must be consulted
        # here or the scope would be emptied of a locant it still needs.
        # * no stereodescriptors
        _l3_omit_suffix_locant = False
        if (not prefix_str and len(suffix_fg) == 1 and len(chosen_locants) == 1
                and not multiplier
                and parent_name and parent_name.isalpha()
                and not _het_loc_prefix
                and not suffix_n_substituents and not amine_n_by_locant
                and not suffix_n_hydroxy
                and not stereo_descriptors):
            from ..assembly.locant_omission import (
                l3_locant_omitted_for_parent_atoms,
            )
            _l3_omit_suffix_locant = l3_locant_omitted_for_parent_atoms(
                mol, ring_atoms,
                prefix_locants=[],
                suffix_locants=list(chosen_locants),
                parent_cites_locants=False,
                stereo_text="",
            )

        if _l3_omit_suffix_locant:
            combined = f"{combined}{suffix_token}"
        else:
            combined = f"{combined}-{locant_str}-{suffix_token}"

        # C4: prepend the amine N-substituent prefixes
        # (N-methyl / N-phenyl / N,N-dimethyl) to the ring-amine suffix name.
        # Wave2: same mechanism serves the N-substituted
        # ring carboxamide (N,N-diethylfuran-2-carboxamide).
        if chosen_suffix == 'amine' and len(suffix_fg.get('amine', [])) > 1 \
                and amine_n_by_locant:
            # companion: MULTI-amine ring -> each amine nitrogen's
            # N-substituent(s) carry that nitrogen's RING locant as an italic-N
            # superscript (N2-tert-butyl-N4-cyclopropyl-...). Cited in ring-locant
            # order; the exocyclic order does not change the structure, so a
            # non-PIN order still round-trips (accepts).
            _parts = []
            for _loc in sorted(amine_n_by_locant):
                _subs = amine_n_by_locant[_loc]
                if len(_subs) == 2 and _subs[0] == _subs[1]:
                    _mp = get_multiplier_prefix(2, _subs[0])
                    _parts.append(
                        f"N{_loc},N{_loc}-{_mp}{enclose_if_compound(_subs[0])}")
                else:
                    for _s in _subs:
                        _parts.append(f"N{_loc}-{enclose_if_compound(_s)}")
            _n_prefix = "-".join(_parts)
            if _n_prefix:
                if combined and combined[0].isdigit():
                    combined = f"{_n_prefix}-{combined}"
                else:
                    combined = f"{_n_prefix}{combined}"
        elif chosen_suffix in ('amine', 'carboxamide') \
                and chosen_suffix in suffix_n_substituents:
            _n_subs = suffix_n_substituents[chosen_suffix]
            _n_prefix = ""
            if len(_n_subs) == 1:
                _n_prefix = f"N-{enclose_if_compound(_n_subs[0])}"
            elif len(_n_subs) == 2 and _n_subs[0] == _n_subs[1]:
                _mp = get_multiplier_prefix(2, _n_subs[0])
                _n_prefix = f"N,N-{_mp}{enclose_if_compound(_n_subs[0])}"
            else:
                _n_prefix = "-".join(
                    f"N-{enclose_if_compound(_n)}" for _n in _n_subs
                )
            if _n_prefix:
                # A hyphen is needed when the existing name already begins with a
                # ring-locant prefix (e.g. '4-fluoropyridin...'); otherwise the
                # N-prefix attaches directly (N-methylpyridin-4-amine).
                if combined and combined[0].isdigit():
                    combined = f"{_n_prefix}-{combined}"
                else:
                    combined = f"{_n_prefix}{combined}"

        # Hydroxamic acid suffix: prepend N-hydroxy to the assembled name
        #: N-hydroxy is an N-substituent on the amide parent).
        if chosen_suffix == 'carboxamide' and chosen_suffix in suffix_n_hydroxy:
            combined = f"N-hydroxy{combined}"

        # Remaining suffix FGs become prefixes (carboxy, formyl, etc.)
        _SUFFIX_TO_PREFIX = {
            'carboxylic acid': 'carboxy',
            'carbaldehyde': 'formyl',
            'carboxamide': 'carbamoyl',
            'carbohydrazide': 'hydrazinecarbonyl',  # C1
            'carbonitrile': 'cyano',
        }
        for suf_name, suf_locants in suffix_fg.items():
            if suf_name == chosen_suffix:
                continue
            prefix_name = _SUFFIX_TO_PREFIX.get(suf_name, suf_name)
            prefix = _format_c_substituent(prefix_name, sorted(suf_locants), len(suf_locants))
            if prefix_str:
                prefix_str = f"{prefix}-{prefix_str}" if prefix_str[0].isdigit() else f"{prefix}{prefix_str}"
            else:
                combined = f"{prefix}{combined}"

    # Assemble with stereo prefix if present
    if stereo_descriptors:
        stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
        return f"{stereo_prefix}{combined}"

    return combined


def _fully_enclosed(n: str) -> bool:
    """
    Return True only when a SINGLE outer pair of parentheses spans the whole
    name — '(pyridin-2-yl)' -> True; '(naphthalen-2-yl)methyl' -> False
    (the paren closes before 'methyl', so enclosure is NOT complete).

    Used by both _format_n_substituent and _format_c_substituent.
    """
    if not (n.startswith('(') and n.endswith(')')):
        return False
    depth = 0
    for i, ch in enumerate(n):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return i == len(n) - 1
    return False


def _format_n_substituent(name: str, count: int, locants=None) -> str:
    """
    Format an N-substituent prefix.

     task 9: when the ring nitrogen's NUMERIC locants are
    known they are the PIN citation ('4-cyclohexylmorpholine',
    '1-(naphthalen-2-yl)pyrrolidine'); the italic-N form ('N-methyl') is
    the fallback when no numbering is available. Complex substituent
    names get the same enclosure as C-substituents.
    """
    from ..assembly.naming_utils import (
        apply_enclosing_marks,
        enclose_if_compound,
        is_complex_substituent,
        needs_brackets,
    )
    if locants:
        from ..assembly.naming_utils import _has_stereo_prefix
        display = name
        if _has_stereo_prefix(name):
            # '(S)-sec-butyl' under a numeric locant needs
            # brackets: '1-[(S)-sec-butyl]...' (the naive startswith('(')
            # check skipped enclosure and emitted '1-(S)-sec-butyl...').
            display = f'[{name}]'
        elif not _fully_enclosed(name) and (
                is_complex_substituent(name) or needs_brackets(name)
                or any(mark in name for mark in '([{')):
            #: a name that is NOT fully enclosed by one outer pair of
            # parentheses — e.g. '(pyrimidin-5-yl)methyl' where the paren closes
            # before 'methyl' — still needs enclosure. The third disjunct (any
            # inner enclosing mark present) catches a name that already carries
            # a mark but neither is_complex_substituent nor needs_brackets flags
            # it (mirrors enclose_if_compound's third arm) — e.g.
            # '[(butan-2-yl)oxy]methyl'. Route through apply_enclosing_marks so
            # the mark ESCALATES per (-> [ -> {) instead of the old
            # raw '(' in name picker, which capped at '[' and produced a double
            # '[[...]]' when name already contained a '['. Fusion/spiro/von-
            # Baeyer brackets (e.g. '[2,3-b]') are excluded from the escalation
            # count by apply_enclosing_marks itself.
            display = apply_enclosing_marks(name, -1)
        locant_str = ",".join(str(loc) for loc in sorted(locants))
        if count == 1:
            return f"{locant_str}-{display}"
        # (a) -- same shared join as _format_c_substituent below; see the
        # comment there. A private SIMPLE_MULTIPLIERS lookup here could not emit
        # `bis` either, so `N,N-di(bromomethyl)...` had the identical defect.
        from ..assembly.naming_utils import multiplied_component
        return f"{locant_str}-{multiplied_component(count, name, display)}"
    wrapped = enclose_if_compound(name)
    if count == 1:
        return f"N-{wrapped}"
    else:
        # N,N-dimethyl, N,N,N-trimethyl, etc.
        n_locants = ",".join(["N"] * count)
        from ..assembly.naming_utils import multiplied_component
        return f"{n_locants}-{multiplied_component(count, name, wrapped)}"


def _format_c_substituent(name: str, locants: List[int], count: int,
                          omit_locants: bool = False) -> str:
    """
    Format a C-substituent prefix with numeric locants.

    Single: 3-methyl
    Multiple same: 2,4-dimethyl

    Per IUPAC, compound substituent names are parenthesized.
    Enclosing marks are required by (BB 7232); their ORDER
    (...), [...], {...} is (BB 7444), escalating per (BB 7509).
    Names already containing parentheses use square brackets.

    ``omit_locants`` is the (the Blue Book) licence, decided by the CALLER
    (``name_substituted_heterocycle``) -- the only place with the ring structure the
    orbit test needs, exactly as for the sibling suffix licence. True means this
    scope's single substituent prefix cites no locant: ``chloropyrazine``,
    ``methylpyrazine``, on the model of the Blue Book ``chlorocoronene (PIN)``.
    NOT a locant-stripping flag: it is only ever set by a positively-licensed
    structural predicate, it defaults False, and the ENCLOSING MARKS and multiplier
    below are still applied -- only the ``{locants}-`` head is withheld.
    """
    locant_str = ",".join(str(loc) for loc in locants)
    # Wrap compound names in enclosing marks to prevent locant ambiguity
    display_name = name
    from ..assembly.naming_utils import (
        _has_stereo_prefix,
        apply_enclosing_marks,
        is_complex_substituent,
        needs_brackets,
    )

    # _fully_enclosed is defined at module level (shared with _format_n_substituent)

    if _has_stereo_prefix(name):
        # Name has CIP stereo prefix like "(R)-sec-butyl":
        # use square brackets per IUPAC
        display_name = f'[{name}]'
    elif not _fully_enclosed(name) and (
            is_complex_substituent(name) or needs_brackets(name)
            or any(mark in name for mark in '([{')):
        # is_complex_substituent governs the di-/bis- multiplier choice;
        # needs_brackets is the broader enclosing test that also
        # flags compound FG-on-alkyl prefixes (hydroxymethyl, aminomethyl)
        # which take a SIMPLE multiplier but STILL require parentheses
        # (the benzene path's '1,3,5-tri(hydroxymethyl)benzene' convention).
        # The third disjunct (any inner enclosing mark present) catches a
        # name that already carries '(', '[' or '{' but is not itself fully
        # enclosed -- e.g. '[(butan-2-yl)oxy]methyl' -- which needs_brackets
        # and is_complex_substituent both miss when there is no top-level
        # digit/hyphen (mirrors enclose_if_compound's third arm, the shared
        # inner-mark test the benzene/chain path already relies on).
        #
        # Route through apply_enclosing_marks nesting ORDER, BB
        # 7444; escalation, BB 7509) under the marks
        # requirement (BB 7232) so the mark ESCALATES (-> [ -> {) instead
        # of the old raw '(' in name picker, which capped at '[' and
        # produced a double '[[...]]' when name already contained a '['.
        # Fusion/spiro/von-Baeyer brackets (e.g. '[2,3-b]') are excluded
        # from the escalation count by apply_enclosing_marks itself
        #, so a fusion-descriptor substituent still gets
        # plain parentheses.
        display_name = apply_enclosing_marks(name, -1)
    if count == 1:
        return display_name if omit_locants else f"{locant_str}-{display_name}"
    # (a) (the Blue Book): 'bis'/'tris'/'tetrakis' indicate a multiplicity of
    # "compound or complex (i.e. substituted) prefixes", with the verbatim
    # preferred prefix `bis(bromomethyl)` in that rule's own example list and the
    # assembled PIN `1,2-bis(bromomethyl)benzene (PIN)` at:25811. The private
    # SIMPLE_MULTIPLIERS lookup that used to sit here could not express that at
    # all, so EVERY multiplied prefix on a heterocyclic parent came back `di`.
    # `multiplied_component` is THE shared join of multiplier + enclosure +
    # hyphen -- the same primitive the carbocyclic producer already
    # uses, which is why benzene spelled:25811 correctly and pyridine did not.
    # It consults `is_substituted_substituent` (the MULTIPLIER question), NOT
    # `is_complex_substituent` (the ENCLOSURE question decided above); the two
    # disagree, and `di(propan-2-yl)` (:25721) is the row that proves it.
    from ..assembly.naming_utils import multiplied_component
    token = multiplied_component(count, name, display_name)
    return token if omit_locants else f"{locant_str}-{token}"


def get_saturation_prefix(
    mol,
    ring_atoms,
    is_aromatic_parent: bool
) -> str:
    """
    Get saturation prefix for partially saturated heterocycles (HETERO-08).

    For saturated versions of aromatic parents, returns prefixes like:
    - dihydro- (2 additional H atoms)
    - tetrahydro- (4 additional H atoms)

    Note: This is a placeholder - full implementation requires comparing
    to the aromatic parent structure.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring
        is_aromatic_parent: True if the parent is aromatic (e.g., furan vs THF)

    Returns:
        Saturation prefix string (e.g., 'tetrahydro-') or empty string

    Examples:
        >>> # Tetrahydrofuran is "oxolane" (retained name), but could also be
        >>> # named "tetrahydrofuran" relating it to furan
        >>> # This function helps determine the saturation prefix when needed
    """
    # For now, this is a simplified implementation
    # Full HETERO-08 support would need to compare H counts between
    # aromatic and saturated forms
    info = classify_heterocycle(mol, ring_atoms)

    if is_aromatic_parent and info['is_saturated']:
        # Calculate hydrogenation level
        # For now, return placeholder - retained names handle common cases
        # like tetrahydrofuran, tetrahydropyran, etc.
        pass

    return ""
