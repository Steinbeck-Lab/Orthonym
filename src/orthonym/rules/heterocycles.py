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
- Position 1 goes to highest-priority heteroatom (O > S > Se > Te > N > P > ...)
- Numbering direction chosen to give lowest locants to other heteroatoms
- First-point-of-difference comparison (not sum of locants)

Naming priority:
1. Check retained names FIRST (pyridine, furan, morpholine, etc.)
2. Fall back to HW systematic naming if no retained name

Reference: IUPAC 2013 Blue Book, Section P-22 (Heterocycles)
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, deque

logger = logging.getLogger(__name__)

from rdkit import Chem

from .locants import compare_locant_sets as _compare_locant_sets  # IM-02

from ..data.hw_heteroatoms import HETEROATOM_PRIORITY, get_heteroatom_priority, get_hw_prefix
from ..data.hw_stems import get_hw_stem
from ..data.retained_names import get_retained_name
from ..perception.rings import (
    get_ring_heteroatoms,
    is_aromatic_ring,
    is_saturated_ring,
)


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
        >>> mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine
        >>> info = classify_heterocycle(mol, mol.GetRingInfo().AtomRings()[0])
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
        >>> mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine
        >>> ring = mol.GetRingInfo().AtomRings()[0]
        >>> oriented = number_heterocycle_ring(mol, ring)
        >>> # N atom should be at position 1 (index 0)
        >>> mol.GetAtomWithIdx(oriented[0]).GetSymbol()
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

    # Find highest-priority heteroatom for position 1
    sorted_heteroatoms = sorted(
        heteroatoms,
        key=lambda x: get_heteroatom_priority(x[1])
    )
    start_idx = sorted_heteroatoms[0][0]  # Atom index of highest priority

    # Find position of start atom in ring list
    start_pos = ring_list.index(start_idx)

    # If only one heteroatom, direction doesn't matter
    if len(heteroatoms) == 1:
        # Rotate so start atom is first, arbitrary direction
        return _rotate_ring(ring_list, start_pos, 1)

    # Multiple heteroatoms - try both directions and pick lowest locants
    cw = _rotate_ring(ring_list, start_pos, 1)   # Clockwise
    ccw = _rotate_ring(ring_list, start_pos, -1)  # Counterclockwise

    # Get locants for OTHER heteroatoms (not position 1) in each direction
    cw_locants = _get_other_heteroatom_locants(cw, heteroatoms, start_idx)
    ccw_locants = _get_other_heteroatom_locants(ccw, heteroatoms, start_idx)

    # Compare using first-point-of-difference
    comparison = _compare_locant_sets(cw_locants, ccw_locants)

    if comparison <= 0:
        return cw
    else:
        return ccw


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
        >>> mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine
        >>> ring = mol.GetRingInfo().AtomRings()[0]
        >>> oriented, mapping = orient_heterocycle(mol, ring)
        >>> # N is at locant 1
        >>> n_idx = [i for i in oriented if mol.GetAtomWithIdx(i).GetSymbol() == 'N'][0]
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


def orient_heterocycle_with_substituents(
    mol,
    ring_atoms,
    substituent_positions: Optional[Set[int]] = None,
    principal_group_atoms: Optional[Set[int]] = None
) -> Tuple[List[int], Dict[int, int]]:
    """
    Orient heterocycle considering heteroatoms, the principal characteristic
    group, AND other substituent positions.

    IUPAC Rule (P-14.4 low-locant order, applied to a ring whose senior
    heteroatom is fixed at position 1 per P-31.1.4.3.3): after fixing the
    heteroatom at position 1, choose the numbering direction that gives lowest
    locants to, IN THIS ORDER:
    1. Other heteroatoms (if any)                          [P-31.1.4.3.3]
    2. The principal characteristic group (suffix)         [P-14.4(c)]
    3. Other substituents (if the above tie)               [P-14.4(g)]

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
        >>> mol = Chem.MolFromSmiles('Cc1ccccn1')  # 2-methylpyridine
        >>> ring = mol.GetRingInfo().AtomRings()[0]
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

    # Find highest-priority heteroatom for position 1. Among EQUAL-priority
    # heteroatoms (e.g. the two N of imidazole / pyrazole / pyrimidine) the choice
    # of which becomes position 1 MUST be both deterministic and IUPAC-correct.
    # The prior key used ONLY element priority, so Python's stable sort left the
    # tie broken by the heteroatoms' list order = ascending ATOM INDEX, which
    # flips with the SMILES spelling -> nondeterministic numbering (imidazolium
    # came out 'imidazol-1-ium' from one spelling and 'imidazol-3-ium' from
    # another). Break the tie by, in order: (a) P-31.1.4.3.4 indicated hydrogen —
    # a pyrrole-type ring atom bearing an H takes the lower locant (the NH of
    # imidazole is position 1, so the other N is 3); (b) a CANONICAL atom rank
    # (input-order INDEPENDENT) as the final deterministic discriminator. Rings
    # with a UNIQUE senior heteroatom (pyridine/furan/…) have no tie, so their
    # numbering is byte-identical.
    _canon_rank = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    sorted_heteroatoms = sorted(
        heteroatoms,
        key=lambda x: (
            get_heteroatom_priority(x[1]),
            0 if mol.GetAtomWithIdx(x[0]).GetTotalNumHs() >= 1 else 1,
            _canon_rank[x[0]],
        )
    )
    start_idx = sorted_heteroatoms[0][0]
    start_pos = ring_list.index(start_idx)

    # Try both directions
    cw = _rotate_ring(ring_list, start_pos, 1)
    ccw = _rotate_ring(ring_list, start_pos, -1)

    # Build position lookups
    cw_map = {atom_idx: pos for pos, atom_idx in enumerate(cw, 1)}
    ccw_map = {atom_idx: pos for pos, atom_idx in enumerate(ccw, 1)}

    # Compare by heteroatom locants first (excluding position 1)
    cw_hetero_locants = sorted([cw_map[idx] for idx, _ in heteroatoms if idx != start_idx])
    ccw_hetero_locants = sorted([ccw_map[idx] for idx, _ in heteroatoms if idx != start_idx])

    # Tiered lowest-locant comparison in P-14.4 order:
    #   (1) other heteroatoms  ->  (2) principal group (suffix)  ->  (3) substituents
    # First point of difference wins (compare_locant_sets), exactly mirroring the
    # carbocyclic orient_cycloalkene Path A (pg before generic substituents).
    pg_set = principal_group_atoms or set()

    def _direction_key(loc_map):
        hetero = sorted(loc_map[idx] for idx, _ in heteroatoms if idx != start_idx)
        pg = sorted(loc_map[idx] for idx in pg_set if idx in loc_map)
        sub = sorted(loc_map[idx] for idx in substituent_positions if idx in loc_map)
        return hetero, pg, sub

    cw_h, cw_pg, cw_sub = _direction_key(cw_map)
    ccw_h, ccw_pg, ccw_sub = _direction_key(ccw_map)

    cmp = _compare_locant_sets(cw_h, ccw_h)
    if cmp == 0:
        cmp = _compare_locant_sets(cw_pg, ccw_pg)
    if cmp == 0:
        cmp = _compare_locant_sets(cw_sub, ccw_sub)

    # cmp <= 0 keeps cw (preserves the prior tie-goes-to-cw behaviour).
    oriented = cw if cmp <= 0 else ccw

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
        >>> mol = Chem.MolFromSmiles('c1ncnc1')  # pyrimidine-like
        >>> ring = mol.GetRingInfo().AtomRings()[0]
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


def get_ring_canonical_smiles(mol, ring_atoms) -> str:
    """
    Extract a ring as canonical SMILES for retained name lookup.

    Creates a new molecule containing only the ring atoms and their bonds,
    then returns the canonical SMILES representation.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices defining the ring

    Returns:
        Canonical SMILES string for the ring

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine
        >>> ring = mol.GetRingInfo().AtomRings()[0]
        >>> get_ring_canonical_smiles(mol, ring)
        'c1ccncc1'
    """
    ring_set = set(ring_atoms)

    # Use MolFragmentToSmiles to extract just the ring
    # This handles aromaticity correctly
    ring_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=list(ring_atoms))

    # Canonicalize the SMILES
    ring_mol = Chem.MolFromSmiles(ring_smiles)
    if ring_mol:
        return Chem.MolToSmiles(ring_mol, canonical=True)

    return ring_smiles


def build_hw_name(
    heteroatoms: List[Tuple[int, str]],
    ring_size: int,
    is_saturated: bool,
    is_aromatic: bool
) -> str:
    """
    Build Hantzsch-Widman systematic name for a heterocycle.

    Assembles a systematic HW name from heteroatom prefixes and ring stem:
    1. Group heteroatoms by element
    2. Order by IUPAC priority (O > S > N > ...)
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

    # Build prefix parts, ordered by IUPAC priority
    prefix_parts = []
    elements_by_priority = sorted(
        element_locants.keys(),
        key=lambda e: HETEROATOM_PRIORITY.get(e, 999)
    )

    for elem in elements_by_priority:
        hw_prefix = get_hw_prefix(elem)
        if not hw_prefix:
            continue

        locants = sorted(element_locants[elem])
        count = len(locants)

        if count > 1:
            # Multiple same heteroatoms: add locants and multiplier
            locant_str = ','.join(str(loc) for loc in locants)
            multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
            prefix_parts.append(f"{locant_str}-{multiplier}{hw_prefix}")
        else:
            # Single heteroatom: just the prefix
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

    # Get the dominant heteroatom (highest priority) for stem selection
    dominant_elem = elements_by_priority[0] if elements_by_priority else 'O'

    # Get stem based on ring size, saturation, and dominant heteroatom
    stem = get_hw_stem(ring_size, saturated_for_stem, dominant_elem)
    if not stem:
        stem = ""

    # Apply 'a' elision: drop terminal 'a' before vowel stem
    if prefix.endswith('a') and stem and stem[0] in 'aeiou':
        prefix = prefix[:-1]

    return prefix + stem


def _mancude_monocycle_parent(mol, ring_atoms):
    """Build the mancude (maximum-non-cumulative-double-bond) parent of a
    monocyclic ring by forcing every ring atom + bond aromatic and re-sanitizing.

    Returns ``(parent_mol, old_to_new_idx)`` where ``parent_mol`` is the ring-only
    aromatic parent and ``old_to_new_idx`` maps each input ring atom index to its
    index in ``parent_mol``. Returns ``(None, None)`` if the ring cannot aromatize
    (the parent is then not a mancude aromatic ring, so the hydro naming declines).
    """
    ring = set(ring_atoms)
    em = Chem.RWMol()
    old_to_new = {}
    for idx in sorted(ring):
        old_to_new[idx] = em.AddAtom(Chem.Atom(mol.GetAtomWithIdx(idx).GetAtomicNum()))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring and j in ring:
            em.AddBond(old_to_new[i], old_to_new[j], Chem.BondType.AROMATIC)
            em.GetAtomWithIdx(old_to_new[i]).SetIsAromatic(True)
            em.GetAtomWithIdx(old_to_new[j]).SetIsAromatic(True)
    parent = em.GetMol()
    try:
        Chem.SanitizeMol(parent)
    except Exception:
        return None, None
    if not all(a.GetIsAromatic() for a in parent.GetAtoms()):
        return None, None
    return parent, old_to_new


def name_partially_saturated_monocyclic_heterocycle(mol, ring_atoms) -> Optional[str]:
    """Name a partially-saturated monocyclic mancude heterocycle (IUPAC P-31.1.4).

    Example: ``C1C=CC=CN1`` -> ``1,2-dihydropyridine`` (the systematic HW path
    would otherwise drop the hydrogenation and emit the mancude parent
    ``pyridine``). Works by (1) building the mancude aromatic parent and naming
    it, (2) finding the ring atoms that carry a ring double bond in the mancude
    parent but have LOST it in the molecule (the hydro positions — detected by
    bond topology, NOT hybridization, so a conjugated enamine N still counts),
    (3) numbering the ring with the heteroatom set lowest then the hydro set
    lowest (P-31.1.4.3.4), and (4) emitting ``<locants>-<prefix>hydro<parent>``.

    Scope (fail closed otherwise — never a wrong name, SELF-01-safe):
      * the mancude parent must aromatize and carry NO leading indicated
        hydrogen (parents like ``1H-pyrrole`` / ``2H-pyran`` whose hydro form
        interleaves added-indicated-H are deferred to a follow-on);
      * the molecule must retain >= 1 ring unsaturation (a fully saturated ring
        is named by its retained / HW saturated stem, not as a hydro prefix);
      * the hydro count must be a standard di/tetra/hexa... value.
    """
    from .partial_saturation import SATURATION_PREFIXES

    ring_set = set(ring_atoms)
    n = len(ring_set)
    if n < 4:
        return None

    # Ring atoms still unsaturated in the MOLECULE (ring double bond or aromatic)
    # — these are NOT hydro positions.
    mol_unsat: Set[int] = set()
    for bond in mol.GetBonds():
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
    # A fully-aromatic (mancude) ring has NO hydro positions. Bail now: the
    # mancude parent of an aromatic ring is itself, so naming it (below) would
    # re-enter name_heterocycle -> here -> infinite recursion for an aromatic
    # ring that lacks a retained name (e.g. a HW-named azine).
    if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_set):
        return None

    parent, old_to_new = _mancude_monocycle_parent(mol, ring_atoms)
    if parent is None:
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
    # call. Defer parents carrying a leading indicated hydrogen (1H-pyrrole,
    # 2H-pyran ...): their hydro form interleaves added-indicated-H (follow-on).
    parent_name = name_heterocycle(parent, parent_ring[0])
    if not parent_name or parent_name[:1].isdigit():
        return None

    # Number the ring: lowest locants to the heteroatom set, then heteroatom
    # seniority, then the hydro set (P-31.1.4.3.4). Enumerate all 2N numberings.
    ordered = _macrocycle_ordered_ring(mol, ring_set)
    if ordered is None or len(ordered) != n:
        return None
    het = {i for i in ordered if mol.GetAtomWithIdx(i).GetSymbol() != 'C'}
    best_key = None
    best_map: Optional[Dict[int, int]] = None
    for start in range(n):
        for direction in (1, -1):
            seq = [ordered[(start + k * direction) % n] for k in range(n)]
            loc = {a: idx + 1 for idx, a in enumerate(seq)}
            het_locs = tuple(sorted(loc[a] for a in het))
            seniority = tuple(sorted(
                (get_heteroatom_priority(mol.GetAtomWithIdx(a).GetSymbol()), loc[a])
                for a in het
            ))
            hydro_locs = tuple(sorted(loc[a] for a in hydro))
            key = (het_locs, seniority, hydro_locs)
            if best_key is None or key < best_key:
                best_key = key
                best_map = loc
    if best_map is None:
        return None

    locant_str = ','.join(str(loc) for loc in sorted(best_map[a] for a in hydro))
    return f"{locant_str}-{prefix}{parent_name}"


def name_heterocycle(mol, ring_atoms) -> str:
    """
    Generate IUPAC name for a heterocyclic ring.

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
        >>> mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine
        >>> ring = mol.GetRingInfo().AtomRings()[0]
        >>> name_heterocycle(mol, ring)
        'pyridine'
        >>> mol2 = Chem.MolFromSmiles('C1CO1')  # oxirane
        >>> ring2 = mol2.GetRingInfo().AtomRings()[0]
        >>> name_heterocycle(mol2, ring2)
        'oxirane'
    """
    # Get ring canonical SMILES for retained name lookup
    ring_smiles = get_ring_canonical_smiles(mol, ring_atoms)

    # Check retained names FIRST
    retained = get_retained_name(ring_smiles)
    if retained:
        return retained

    # Partially-saturated mancude monocyclic heterocycle (e.g. 1,2-dihydro-
    # pyridine): the systematic HW path below drops the hydrogenation and emits
    # the mancude parent ('pyridine'). Detect + name the hydro form first
    # (P-31.1.4); fails closed for anything it cannot number correctly.
    partial = name_partially_saturated_monocyclic_heterocycle(mol, ring_atoms)
    if partial:
        return partial

    # Fall back to systematic naming
    info = classify_heterocycle(mol, ring_atoms)
    oriented, _ = orient_heterocycle(mol, ring_atoms)

    # Get heteroatom locants from oriented ring
    heteroatom_locants = get_heteroatom_locants(oriented, mol)

    ring_size = info['ring_size']

    # For rings > 10: use replacement nomenclature (cycloXXXane parent).
    # V-6 / P-31.1.4: enumerate the actual ring double bonds (kekulising a
    # conjugated macrocycle RDKit reads as aromatic) so a polyene is named with
    # ALL its bonds + locants instead of a single bare 'ene'.
    if ring_size > 10:
        macro_het, double_locants = _orient_macrocycle_for_replacement(mol, ring_atoms)
        if macro_het is None:
            # Could not enumerate -> fall back to the heteroatom-only orientation.
            macro_het, double_locants = heteroatom_locants, None
        return _build_replacement_name(
            macro_het,
            ring_size,
            info['is_saturated'],
            double_locants=double_locants,
        )

    hw_name = build_hw_name(
        heteroatom_locants,
        ring_size,
        info['is_saturated'],
        info['is_aromatic']
    )

    # Post-process: replace OPSIN-incompatible HW names with IUPAC-preferred
    # retained names. "oxine" (6-membered unsaturated O-ring) is not recognized
    # by OPSIN; IUPAC 2013 prefers "2H-pyran" for the monocyclic system and
    # "4H-pyran" for the 4H tautomer. Similarly, "azine" should be "pyridine".
    _HW_TO_RETAINED = {
        'oxine': '2H-pyran',
        'azine': 'pyridine',
        'thiine': '2H-thiopyran',
    }
    if hw_name in _HW_TO_RETAINED:
        hw_name = _HW_TO_RETAINED[hw_name]

    return hw_name


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
    mol, ring_atoms
) -> Tuple[Optional[List[Tuple[int, str]]], Optional[List[int]]]:
    """Number a >10-membered heteromonocycle for replacement ("a") nomenclature.

    Returns ``(heteroatom_locants, double_bond_locants)`` where the numbering
    minimises, in order (IUPAC 2013 P-31.1.4.3): the heteroatom locant set, then
    the heteroatoms by element seniority, then the ring double-bond locant set.

    The ring is *kekulised* first so a fully-conjugated macrocycle that RDKit
    perceives as aromatic (e.g. the V-6 azacyclotrideca-hexaene) yields its
    localised double bonds instead of zero (the bug that made the old path emit
    a single bare ``ene``). Returns ``(None, None)`` if the ring cannot be
    ordered/kekulised, so the caller can fall back safely.
    """
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
            key = (het_set, seniority, dbl)
            if best is None or key < best[0]:
                best = (key, loc, dbl)

    loc = best[1]
    dbl = best[2]
    heteroatom_locants = sorted(
        ((loc[a], kmol.GetAtomWithIdx(a).GetSymbol()) for a in het),
        key=lambda t: t[0],
    )
    return heteroatom_locants, dbl


def _build_replacement_name(
    heteroatoms: List[Tuple[int, str]],
    ring_size: int,
    is_saturated: bool,
    double_locants: Optional[List[int]] = None,
) -> str:
    """
    Build replacement ("a") nomenclature name for macrocyclic heterocycles.

    For rings larger than 10 atoms, IUPAC uses "a" nomenclature:
    replacement prefixes (oxa, aza, thia) + cycloalkane parent name.

    Example: 1,4,7,10-tetraoxacyclododecane (12-crown-4)

    Args:
        heteroatoms: List of (locant, element) tuples
        ring_size: Number of atoms in the ring (> 10)
        is_saturated: True if fully saturated
        double_locants: Sorted ring double-bond locants (V-6 / P-31.1.4 polyene
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

    # Order by IUPAC priority (O > S > N > ...)
    elements_by_priority = sorted(
        element_locants.keys(),
        key=lambda e: HETEROATOM_PRIORITY.get(e, 999)
    )

    prefix_parts = []
    for elem in elements_by_priority:
        hw_prefix = get_hw_prefix(elem)
        if not hw_prefix:
            continue

        locants = sorted(element_locants[elem])
        count = len(locants)

        locant_str = ','.join(str(loc) for loc in locants)
        if count > 1:
            multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
            prefix_parts.append(f"{locant_str}-{multiplier}{hw_prefix}")
        else:
            prefix_parts.append(f"{locant_str}-{hw_prefix}")

    replacement_prefix = '-'.join(prefix_parts)

    # Build parent ring name. 'cyclo' starts with a consonant so the replacement
    # prefix never needs terminal-'a' elision against it.
    chain_prefix = get_chain_prefix(ring_size)

    if double_locants is not None:
        # V-6 / P-31.1.4: enumerate ALL ring double bonds with their locants
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
                    'atoms': List[int],  # Atom indices in substituent
                    'is_on_nitrogen': bool,  # True if attached to N
                    'carbon_count': int,  # Number of C atoms (for alkyl naming)
                    'connecting_atom': int,  # Ring atom the sub attaches to
                    'is_ring': bool,  # True if substituent is a ring
                    'ring_name': str,  # Ring substituent name (if is_ring)
                }
            ]
        }

    Examples:
        >>> mol = Chem.MolFromSmiles('CN1CCCC1')  # N-methylpyrrolidine
        >>> ring = mol.GetRingInfo().AtomRings()[0]
        >>> oriented, atom_to_loc = orient_heterocycle(mol, ring)
        >>> subs = get_heterocycle_substituents(mol, ring, oriented, atom_to_loc)
        >>> # N-methyl should be at locant 1 (N position) with is_on_nitrogen=True
    """
    from ..perception.chains import classify_substituent
    from ..perception.rings import get_containing_ring_system
    from .seniority import get_prefix, get_suffix

    # WS-4 / BBR-RSFX (DEF-6): when a senior characteristic group sits on the ring,
    # express it as a SUFFIX (P-33), not a detachable prefix. We recognise the
    # principal group GENERICALLY off the seniority tables: a no-carbon substituent
    # whose prefix form equals get_prefix(principal_group) AND whose class has a ring
    # suffix form (get_suffix(..., is_ring=True)) is the principal-group suffix
    # (oxo->one, hydroxy->ol, amino->amine, sulfanyl->thiol, ...). Subordinate
    # same-prefix groups keep the prefix because only the principal group's prefix
    # matches. No per-FG `if`.
    pg_prefix = get_prefix(principal_group) if principal_group else None
    pg_ring_suffix = get_suffix(principal_group, is_ring=True) if principal_group else None

    # Use the complete ring system as BFS boundary (IUPAC P-25.3)
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

            # WS-A task 9 (P-66.6.3): an ACYL group on a ring NITROGEN is an
            # AMIDE — the amide machinery names it ('-oyl' forms / parent
            # ketone), never this collector (which produced garbled
            # 'dienalyl'/alkyl forms and, once it claimed the atoms, its
            # full coverage let the wrong candidate win the pool). Skip the
            # fragment entirely: the heterocycle candidate's coverage drops
            # and the amide path wins, as it did before the WS-A chokepoint.
            if is_nitrogen:
                _nbr_atom = mol.GetAtomWithIdx(nbr_idx)
                _is_acyl = _nbr_atom.GetSymbol() == 'C' and any(
                    b.GetOtherAtom(_nbr_atom).GetSymbol() in ('O', 'S', 'Se', 'Te')
                    and b.GetBondTypeAsDouble() == 2.0
                    and b.GetOtherAtom(_nbr_atom).GetIdx() in set(sub_atoms)
                    for b in _nbr_atom.GetBonds()
                )
                if _is_acyl:
                    continue

            # Count carbon atoms for alkyl naming
            carbon_count = sum(
                1 for idx in sub_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )

            # Detect non-carbon functional substituents (amino, hydroxy, nitro, etc.)
            hetero_sub_name = None
            if carbon_count == 0:
                # Check for functional groups that have no carbons
                hetero_sub_name = _identify_hetero_substituent(mol, sub_atoms, ring_set)
                if hetero_sub_name is None:
                    continue  # Truly empty (just implicit H)

            # Check for suffix-type functional groups (-COOH, -CHO, -CONH2, -CN)
            # These should be expressed as suffixes, not prefix substituents
            suffix_info = _identify_suffix_fg(mol, nbr_idx, sub_atoms, ring_set)

            # Classify substituent as ring or alkyl
            classification = classify_substituent(mol, sub_atoms, ring_set)
            is_ring = classification['type'] == 'ring'
            ring_name = classification['name'] if is_ring else None

            # WS-A.2: a demoted ring substituent must carry its OWN
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
            # WS-4 / BBR-RSFX (DEF-6): is THIS no-carbon group the principal group?
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

            if is_principal_suffix:
                sub_info['is_suffix'] = True
                sub_info['suffix_name'] = pg_ring_suffix
            else:
                if hetero_sub_name:
                    sub_info['hetero_name'] = hetero_sub_name
                if suffix_info:
                    sub_info['is_suffix'] = True
                    sub_info['suffix_name'] = suffix_info['suffix_name']

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
    acid_pat = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    if acid_pat:
        for match in mol.GetSubstructMatches(acid_pat):
            if match[0] == start_idx:
                return {'suffix_name': 'carboxylic acid'}

    # Check for aldehyde: C(=O)H attached to ring
    ald_pat = Chem.MolFromSmarts('[CX3H1](=O)')
    if ald_pat:
        for match in mol.GetSubstructMatches(ald_pat):
            if match[0] == start_idx:
                return {'suffix_name': 'carbaldehyde'}

    # Check for primary amide: C(=O)(NH2)
    amide_pat = Chem.MolFromSmarts('[CX3](=O)[NX3H2]')
    if amide_pat:
        for match in mol.GetSubstructMatches(amide_pat):
            if match[0] == start_idx:
                return {'suffix_name': 'carboxamide'}

    # Check for nitrile: C#N
    nitrile_pat = Chem.MolFromSmarts('[CX2]#[NX1]')
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

    # Halogens (F, Cl, Br, I) -- IUPAC P-31.1.2.1
    _HALOGEN_PREFIX = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
    if symbol in _HALOGEN_PREFIX:
        return _HALOGEN_PREFIX[symbol]

    # Amino (-NH2)
    if symbol == 'N' and h_count == 2:
        return 'amino'

    # Hydroxy (-OH)
    if symbol == 'O' and h_count == 1:
        return 'hydroxy'

    # Nitro (-NO2)
    if symbol == 'N' and first_atom.GetFormalCharge() == 1:
        o_count = sum(1 for idx in sub_atoms if mol.GetAtomWithIdx(idx).GetSymbol() == 'O')
        if o_count == 2:
            return 'nitro'

    # Sulfanyl (-SH)
    if symbol == 'S' and h_count == 1:
        return 'sulfanyl'

    # Oxo (=O) - check if double-bonded to ring carbon
    if symbol == 'O' and h_count == 0 and len(sub_atoms) == 1:
        for bond in first_atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(sub_atoms[0])
            if other_idx in ring_set and bond.GetBondType() == Chem.BondType.DOUBLE:
                return 'oxo'

    # Imino (=NH) -- IUPAC P-31.1.3
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


def name_substituted_heterocycle(
    mol,
    ring_atoms,
    parent_name: str,
    substituents: Dict[int, List[Dict]],
    atom_to_locant: Dict[int, int],
    principal_group: Optional[str] = None
) -> str:
    """
    Assemble complete name for a substituted heterocycle.

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
        >>> mol = Chem.MolFromSmiles('CN1CCCC1')  # N-methylpyrrolidine
        >>> # ... get ring, orient, get substituents ...
        >>> name_substituted_heterocycle(mol, ring, 'pyrrolidine', subs, atom_to_loc)
        'N-methylpyrrolidine'
    """
    from ..assembly.naming_utils import (
        get_alkyl_name,
        alpha_sort_key,
        get_multiplier_prefix,
    )
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    from ..perception.stereo import assign_stereochemistry

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
    prefix_substituents: Dict[int, List[Dict]] = {}

    for locant, sub_list in substituents.items():
        for sub_info in sub_list:
            if sub_info.get('is_suffix'):
                sname = sub_info['suffix_name']
                if sname not in suffix_fg:
                    suffix_fg[sname] = []
                suffix_fg[sname].append(locant)
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

            if is_ring and ring_name:
                # Use ring substituent name (piperidinyl, phenyl, etc.)
                sub_name = ring_name
            elif 'hetero_name' in sub_info:
                # Non-carbon functional substituent (amino, hydroxy, etc.)
                sub_name = sub_info['hetero_name']
            else:
                # Get alkyl name from carbon count, with recursive naming for branched subs
                carbon_count = sub_info['carbon_count']
                sub_atoms = sub_info.get('atoms', [])
                sub_name = None
                # Pure C/H substituents: try recursive naming first (handles branching)
                if sub_atoms and len(sub_atoms) > 1:
                    all_c_h = all(
                        mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'H')
                        for i in sub_atoms
                    )
                    if all_c_h:
                        from ..assembly.substituent_naming import name_substituent_fragment
                        attach_idx = sub_atoms[0]
                        ring_set_local = set(ring_atoms) if ring_atoms else set()
                        sub_name = name_substituent_fragment(
                            mol, sub_atoms, attach_idx, list(ring_set_local)
                        )
                if sub_name is None:
                    try:
                        sub_name = get_alkyl_name(carbon_count)
                    except ValueError:
                        # Carbon count too large for simple alkyl name;
                        # try recursive naming as fallback (including heteroatom subs)
                        if sub_atoms and len(sub_atoms) > 1:
                            from ..assembly.substituent_naming import name_substituent_fragment
                            attach_idx = sub_atoms[0]
                            ring_set_local = set(ring_atoms) if ring_atoms else set()
                            sub_name = name_substituent_fragment(
                                mol, sub_atoms, attach_idx, list(ring_set_local)
                            )
                        if sub_name is None:
                            logger.debug(
                                "DROP-24 substituent_skip: reason=large_sub_still_unnameable carbon_count=%d",
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

    # Build prefix parts
    prefix_parts = []

    # Format N-substituent prefixes (numeric ring locants = PIN, P-14.3.2)
    for name, locants in n_groups.items():
        count = len(locants)
        numeric = [loc for loc in locants if loc is not None]
        prefix = _format_n_substituent(
            name, count, locants=numeric if len(numeric) == count else None
        )
        prefix_parts.append((prefix, name))

    # Format C-substituent prefixes
    for name, locants in c_groups.items():
        count = len(locants)
        prefix = _format_c_substituent(name, sorted(locants), count)
        prefix_parts.append((prefix, name))

    # Sort alphabetically by the base substituent name
    # (ignoring N-, locants, multipliers)
    prefix_parts.sort(key=lambda x: alpha_sort_key(x[1]))

    # Join prefixes
    # Closing brackets ], ), } all act as word boundaries needing hyphens
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
        # WS-4 / BBR-RSFX: the PRINCIPAL group's ring suffix wins (it is, by
        # seniority, senior to any co-present carb* suffix). Otherwise fall back
        # to the fixed carb* priority list.
        pg_ring_suffix = _get_ring_suffix(principal_group, is_ring=True) if principal_group else None
        _SUFFIX_PRIORITY = [
            'carboxylic acid', 'carboxamide', 'carbonitrile', 'carbaldehyde',
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
        multiplier = get_multiplier_prefix(count, chosen_suffix) if count > 1 else ""
        locant_str = ",".join(str(loc) for loc in chosen_locants)
        suffix_token = f"{multiplier}{chosen_suffix}"
        # IUPAC P-16.3.3: elide the parent's terminal 'e' before a suffix token
        # that begins with a vowel (piperidine -> piperidin-4-one,
        # pyridine -> pyridin-2-ol). A consonant-initial token (multiplied
        # 'dione'/'triol', or 'carb*') keeps the 'e' (piperidine-2,6-dione,
        # pyridine-3-carbaldehyde).
        if combined and combined[-1] == 'e' and suffix_token[:1].lower() in 'aeiouy':
            combined = combined[:-1]
        combined = f"{combined}-{locant_str}-{suffix_token}"

        # Remaining suffix FGs become prefixes (carboxy, formyl, etc.)
        _SUFFIX_TO_PREFIX = {
            'carboxylic acid': 'carboxy',
            'carbaldehyde': 'formyl',
            'carboxamide': 'carbamoyl',
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


def _format_n_substituent(name: str, count: int, locants=None) -> str:
    """
    Format an N-substituent prefix.

    WS-A task 9 (P-14.3.2): when the ring nitrogen's NUMERIC locants are
    known they are the PIN citation ('4-cyclohexylmorpholine',
    '1-(naphthalen-2-yl)pyrrolidine'); the italic-N form ('N-methyl') is
    the fallback when no numbering is available. Complex substituent
    names get the same P-16.3.3 enclosure as C-substituents.
    """
    from ..assembly.naming_utils import (
        _wrap_n_substituent, is_complex_substituent,
    )
    if locants:
        from ..assembly.naming_utils import _has_stereo_prefix
        display = name
        if _has_stereo_prefix(name):
            # '(S)-sec-butyl' under a numeric locant needs P-16.3.3
            # brackets: '1-[(S)-sec-butyl]...' (the naive startswith('(')
            # check skipped enclosure and emitted '1-(S)-sec-butyl...').
            display = f'[{name}]'
        elif is_complex_substituent(name) and not name.startswith('('):
            display = f'[{name}]' if '(' in name else f'({name})'
        locant_str = ",".join(str(loc) for loc in sorted(locants))
        if count == 1:
            return f"{locant_str}-{display}"
        multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
        return f"{locant_str}-{multiplier}{display}"
    wrapped = _wrap_n_substituent(name)
    if count == 1:
        return f"N-{wrapped}"
    else:
        # N,N-dimethyl, N,N,N-trimethyl, etc.
        n_locants = ",".join(["N"] * count)
        multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
        return f"{n_locants}-{multiplier}{wrapped}"


def _format_c_substituent(name: str, locants: List[int], count: int) -> str:
    """
    Format a C-substituent prefix with numeric locants.

    Single: 3-methyl
    Multiple same: 2,4-dimethyl

    Per IUPAC P-14.5.2, compound substituent names are parenthesized.
    Per IUPAC P-16.3.3, enclosing marks nest: (...), [...], {...}.
    Names already containing parentheses use square brackets.
    """
    locant_str = ",".join(str(loc) for loc in locants)
    # Wrap compound names in enclosing marks to prevent locant ambiguity
    display_name = name
    from ..assembly.naming_utils import is_complex_substituent, _has_stereo_prefix

    def _fully_enclosed(n: str) -> bool:
        # True only when ONE outer pair of parens spans the whole name —
        # '(pyridin-2-yl)' yes; '(naphthalen-2-yl)methyl' NO (the marks
        # close before 'methyl', so the name still needs enclosure).
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

    if _has_stereo_prefix(name):
        # Name has CIP stereo prefix like "(R)-sec-butyl":
        # use square brackets per IUPAC P-16.3.3
        display_name = f'[{name}]'
    elif not _fully_enclosed(name) and is_complex_substituent(name):
        if '(' in name:
            # P-16.3.3 nesting: a name already containing parentheses is
            # enclosed in the next mark up ([(naphthalen-2-yl)methyl]).
            display_name = f'[{name}]'
        else:
            display_name = f'({name})'
    if count == 1:
        return f"{locant_str}-{display_name}"
    else:
        multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
        return f"{locant_str}-{multiplier}{display_name}"


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
