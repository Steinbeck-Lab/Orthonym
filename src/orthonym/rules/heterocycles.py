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
from ..data.hw_stems import HW_STEMS, get_hw_stem
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

    # P-31.1.4.3.3: find the highest-priority element (lowest priority number).
    best_prio = min(get_heteroatom_priority(sym) for _, sym in heteroatoms)
    best_starts = [(idx, sym) for idx, sym in heteroatoms
                   if get_heteroatom_priority(sym) == best_prio]

    # If only one heteroatom, direction doesn't matter
    if len(heteroatoms) == 1:
        start_pos = ring_list.index(best_starts[0][0])
        return _rotate_ring(ring_list, start_pos, 1)

    # Try ALL equal-priority start atoms in both directions; pick the
    # (start, direction) that gives the lexicographically minimum sorted
    # all-heteroatom locant set (P-31.1.4.3.3 lowest-locant criterion).
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

    # Find the best (priority, H_count) tier from sorted heteroatoms.
    # P-31.1.4.3.4: heteroatom bearing indicated H (pyrrole-type) takes
    # the lower locant; (a) element priority, (b) H_count (lower=better),
    # (c) canonical rank for deterministic tiebreaking.
    # For rings where all heteroatoms are equal-priority AND equal-H_count
    # (e.g. three sp2 N in 1,2,4-triazine) the old code fixed a single
    # start atom, producing wrong locants when that atom wasn't the optimal
    # choice.  The new code collects ALL atoms sharing the best
    # (priority, h_count) tier and tries every one as position-1 candidate,
    # picking the (start, direction) pair that yields the globally minimum
    # tiered (hetero > pg > sub) locant comparison.
    _canon_rank = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    sorted_heteroatoms = sorted(
        heteroatoms,
        key=lambda x: (
            get_heteroatom_priority(x[1]),
            0 if mol.GetAtomWithIdx(x[0]).GetTotalNumHs() >= 1 else 1,
            _canon_rank[x[0]],
        )
    )
    best_prio = get_heteroatom_priority(sorted_heteroatoms[0][1])
    best_hcount = 0 if mol.GetAtomWithIdx(sorted_heteroatoms[0][0]).GetTotalNumHs() >= 1 else 1
    # Collect all atoms that share the best (priority, h_count) tier
    start_candidates = [
        idx for idx, sym in heteroatoms
        if get_heteroatom_priority(sym) == best_prio
        and (0 if mol.GetAtomWithIdx(idx).GetTotalNumHs() >= 1 else 1) == best_hcount
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
    is_aromatic: bool,
    lambda_by_locant: Optional[Dict[int, int]] = None,
) -> Optional[str]:
    """
    Build Hantzsch-Widman systematic name for a heterocycle.

    Returns ``None`` -- fail closed -- when any heteroatom has no Table 2.4
    prefix. ``""`` still means "no heteroatoms supplied" and stays distinct.

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

    # Order elements by IUPAC (Hantzsch-Widman) seniority: F > Cl > Br > I >
    # O > S > Se > Te > N > P ... (P-22.2.2.1 / Table 2.8). Prefixes are cited
    # in this order.
    elements_by_priority = sorted(
        element_locants.keys(),
        key=lambda e: HETEROATOM_PRIORITY.get(e, 999)
    )

    # Collect the FULL heteroatom locant set (P-22.2.2.1.3): for a HW name with
    # more than one heteroatom, ALL locants are cited ONCE at the front as a
    # single ascending set (e.g. 1,4-oxazepane, 1,3,5-oxadiazinane), NOT
    # distributed per prefix. A single heteroatom carries no locant (oxolane,
    # azepane). This is the fix that keeps mixed-element medium rings from
    # dropping their locants (bare 'oxazepane' is the 1,2-isomer — a different
    # molecule).
    lambda_by_locant = lambda_by_locant or {}
    all_locants = sorted(loc for locs in element_locants.values() for loc in locs)
    total_het = len(all_locants)
    locant_prefix = ""
    # P-22.2.2.1.3: a ring in which ONE heteroatom element occupies EVERY skeletal
    # position needs no heteroatom locants — the numbering is unambiguous
    # (hexasilinane, not '1,2,3,4,5,6-hexasilinane'; hexathiane). This holds only
    # when a single distinct element fills all ring positions AND no lambda is
    # present (a lambda always cites its locant, P-22.2.7.1). A same-element ring
    # NOT spanning all positions (1,2-disilinane) still needs its locants.
    _one_element_all_positions = (
        len(element_locants) == 1 and total_het == ring_size
        and not lambda_by_locant
    )
    # P-22.2.7.1: a lambda-bearing ring ALWAYS cites its heteroatom locants,
    # even for a single heteroatom (1lambda3-iodinane, not 'lambda3-iodinane'),
    # with the lambda token immediately after its locant (1,3lambda5-oxaphosphole).
    if (total_het > 1 or lambda_by_locant) and not _one_element_all_positions:
        from .lambda_convention import format_lambda_token
        locant_prefix = ','.join(
            format_lambda_token(loc, lambda_by_locant.get(loc))
            for loc in all_locants
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
            # carbocycle's name for a metallacycle. P-22.2.2 has no prefix for
            # Hg/Zn/Cd at all, so refusing is the only sound answer here.
            return None
        count = len(element_locants[elem])
        if count > 1:
            multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
            # P-22.2.2.1.2: elide the multiplier's terminal 'a' before an 'a'
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

    # Stem selection (P-22.2.2.1.2, Table 2.3): the saturated 3-6-ring stem
    # takes its N-containing form (-iridine/-etidine/-olidine/-inane) whenever
    # NITROGEN is present ANYWHERE in the ring — not merely when the senior
    # heteroatom is N. So 1,3-oxazolidine (O senior, N present) uses -olidine
    # and 1,4-oxazinane uses -inane, while N-free rings (1,3-oxathiolane,
    # 1,4-oxathiane) keep -olane/-ane. get_hw_stem selects the N-form when its
    # `heteroatom` arg is 'N', so pass 'N' iff any ring heteroatom is nitrogen;
    # otherwise pass the senior element (which drives the 6-ring O/S -ane form).
    has_nitrogen = 'N' in element_locants
    dominant_elem = elements_by_priority[0] if elements_by_priority else 'O'
    stem_heteroatom = 'N' if has_nitrogen else dominant_elem

    # Get stem based on ring size, saturation, and stem-driving heteroatom.
    # P-22.2.2.1.3 / Table 2.7 class 6C: pass the FULL ring heteroatom set so an
    # unsaturated 6-ring bearing a 6C atom (P/As/Sb/B/... e.g. 1,4-oxaphosphinine)
    # gets the '-inine' ending, not '-ine'.
    stem = get_hw_stem(ring_size, saturated_for_stem, stem_heteroatom,
                       ring_heteroatoms=set(element_locants.keys()))
    # Wave2 T2d (P-22.2.2.1.5.1): a 3-membered mancude ring with ONLY
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
    # P-54.4.1: phosphole's ring P behaves like pyrrole's ring N (1H-phosphole)
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
            continue
        if all(a.GetIsAromatic() for a in parent.GetAtoms()):
            return parent, old_to_new
    return None, None


def _mancude_max_matching(mol, oriented: List[int]) -> Tuple[int, List[bool]]:
    """Mancude maximum for a monocyclic ring given in cycle order.

    Returns ``(max_match, eligible)`` where ``max_match`` is the maximum number of
    noncumulative ring double bonds (= maximum matching on the ring cycle
    restricted to double-bond-eligible edges) and ``eligible[k]`` is the flag for
    ``oriented[k]``. Divalent chalcogens (O/S/Se/Te) never carry a mancude ring
    double bond; every other ring atom (C, N, P, ...) is eligible. Ring size <= 10
    -> exact brute force over the ring edges; deterministic.

    Factored out of :func:`_monocycle_indicated_h_prefix` so the partial-saturation
    namer can reuse the exact mancude count for non-aromatizable (4-pi) HW rings.
    """
    from itertools import combinations
    n = len(oriented)
    _DIVALENT = frozenset({'O', 'S', 'Se', 'Te'})
    eligible = [
        mol.GetAtomWithIdx(i).GetSymbol() not in _DIVALENT for i in oriented
    ]
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


def _nonaromatizable_hydro_fallback(mol, ring_set: Set[int], mol_unsat: Set[int]) -> Optional[str]:
    """P-54.4.1 pure-hydro HW name for a NON-aromatizable monocycle (1,2-dihydro-
    phosphete, 1,2-dihydroazete).

    Runs only when :func:`_mancude_monocycle_parent` declined — a 4-pi ring can
    never be RDKit-aromatized, so the aromatization-based mancude-parent path
    silently fails and the plain HW path would emit the bare over-unsaturated
    mancude stem ('phosphete'). The mancude maximum is computed here by exact
    matching instead (P-14.7.1 = max noncumulative double bonds).

    v1 scope (fail closed / return None otherwise — never a wrong name, SELF-01
    safe): a genuine hydro form (``1 <= d < max_match``); the mancude parent needs
    ZERO indicated hydrogen (``n_eligible - 2*max_match == 0``); a single
    heteroatom (multi-het parents carry a leading locant -> digit-initial gate
    declines them); no exocyclic ring double bond (lambda rings are intercepted
    upstream in ``name_heterocycle`` before this namer runs). BB P-54.4.1:24169.
    """
    from .partial_saturation import SATURATION_PREFIXES
    n = len(ring_set)
    ordered = _macrocycle_ordered_ring(mol, ring_set)
    if ordered is None or len(ordered) != n:
        return None

    # Gate (c): no exocyclic double bond from any ring atom.
    for i in ring_set:
        for bond in mol.GetAtomWithIdx(i).GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                if bond.GetOtherAtomIdx(i) not in ring_set:
                    return None

    max_match, eligible = _mancude_max_matching(mol, ordered)
    n_eligible = sum(eligible)

    # Actual ring double bonds + the atoms they touch.
    db_atoms: Set[int] = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in ring_set and b in ring_set:
                db_atoms.add(a)
                db_atoms.add(b)
    d = sum(
        1 for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetBeginAtomIdx() in ring_set
        and bond.GetEndAtomIdx() in ring_set
    )

    # Gate (a): a true hydro form (fewer ring double bonds than the mancude max);
    # d == max_match is the indicated-H namer's scope (2H-thiete), d == 0 the
    # saturated stem's.
    if not (d >= 1 and d < max_match):
        return None
    # Gate (b): v1 covers ONLY mancude parents with zero indicated hydrogen. A
    # non-aromatizable parent that WOULD need indicated H (azepine 7-ring: 7
    # eligible, max 3 -> 1 IH) stays fail-closed.
    if n_eligible - 2 * max_match != 0:
        return None

    hydro = {ordered[k] for k in range(n)
             if eligible[k] and ordered[k] not in db_atoms}
    if not hydro:
        return None
    prefix = SATURATION_PREFIXES.get(len(hydro))
    if prefix is None:
        return None

    het = {i for i in ordered if mol.GetAtomWithIdx(i).GetSymbol() != 'C'}
    if not het:
        return None

    # Number the ring: heteroatom set lowest -> heteroatom seniority -> hydro set
    # lowest (P-31.1.4.3.4). The parent carries no indicated H (gate b), so no ih
    # term participates.
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

    het_pairs = sorted(
        (best_map[a], mol.GetAtomWithIdx(a).GetSymbol()) for a in het
    )
    parent_name = build_hw_name(het_pairs, n, is_saturated=False, is_aromatic=False)
    if not parent_name:
        return None
    # v1: only bare (0-IH, single-het) parents. A digit-initial parent means either
    # an intrinsic indicated H this numbering did not place OR a multi-heteroatom
    # locant prefix (e.g. '1,2-diazete') -> out of v1 scope, fail closed.
    if parent_name[:1].isdigit():
        return None

    locant_str = ','.join(str(loc) for loc in sorted(best_map[a] for a in hydro))
    return f"{locant_str}-{prefix}{parent_name}"


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
        # P-54.4.1: a non-aromatizable (4-pi) HW ring can never be RDKit-
        # aromatized, so the aromatization-based mancude parent above declines.
        # Recover the pure-hydro PIN via matching-based mancude counting
        # (1,2-dihydrophosphete). Fail-closed for anything outside its v1 scope.
        return _nonaromatizable_hydro_fallback(mol, ring_set, mol_unsat)
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
    # indicated-H parent (1H-pyrrole, 1H-imidazole, ...) is KEPT: the hydro
    # positions are the saturated C's; the indicated-H N is never in a parent
    # ring double bond so it is never counted as hydro (P-31.1.4 + the parent's
    # own nH-) -> e.g. 2,3-dihydro-1H-pyrrole.
    parent_name = name_heterocycle(parent, parent_ring[0])
    if not parent_name:
        return None

    # Number the ring: lowest locants to the heteroatom set, then heteroatom
    # seniority, then the hydro set (P-31.1.4.3.4). Enumerate all 2N numberings.
    ordered = _macrocycle_ordered_ring(mol, ring_set)
    if ordered is None or len(ordered) != n:
        return None
    het = {i for i in ordered if mol.GetAtomWithIdx(i).GetSymbol() != 'C'}
    # The indicated-H atom(s) of the kept parent (the N-H carrying the parent's
    # nH-, when present) must take the LOWEST locant (P-31.1.4.3) — this fixes
    # the 1H position for 2-N azoles (1H-pyrazole N1 = the N-H, not the =N-),
    # ranked before the hydro set.
    indicated_nh = ({i for i in het
                     if mol.GetAtomWithIdx(i).GetSymbol() == 'N'
                     and mol.GetAtomWithIdx(i).GetTotalNumHs() >= 1}
                    if parent_name[:1].isdigit() else set())
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
            ih_locs = tuple(sorted(loc[a] for a in indicated_nh))
            hydro_locs = tuple(sorted(loc[a] for a in hydro))
            key = (het_locs, seniority, ih_locs, hydro_locs)
            if best_key is None or key < best_key:
                best_key = key
                best_map = loc
    if best_map is None:
        return None

    locant_str = ','.join(str(loc) for loc in sorted(best_map[a] for a in hydro))
    # Hyphen before a digit-initial (intrinsic-IH) parent: 2,3-dihydro-1H-pyrrole.
    sep = '-' if parent_name[:1].isdigit() else ''
    return f"{locant_str}-{prefix}{sep}{parent_name}"


def _name_lambda_heteromonocycle(mol, oriented, heteroatom_locants, info) -> Optional[str]:
    """P-22.2.7.1: heteromonocycle with nonstandard-bonding-number heteroatom(s).

    Emits '<locants-with-lambda>-<HW name>' with indicated hydrogen when a
    lambda heteroatom is the saturated skeletal position (1H-1lambda4-thiophene).
    Fail-closed contract — returns None for: rings >10 / aromatic-perceived
    rings; lambda on a skeletal CARBON; two same-element heteroatoms with
    different lambda values (P-22.2.7.2 tie-break unbuilt); any unsaturated
    ring that is not a perfect mancude matching with only HETEROATOM
    saturated positions (hydro forms); >1 indicated-H position.
    """
    from .lambda_convention import nonstandard_bonding_number, STANDARD_BONDING_NUMBER
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
    # P-22.2.7.2 (same element, different bonding numbers) not implemented:
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
                in_double.update((i, j))
                n_double += 1
        sp3 = [idx for idx in oriented if idx not in in_double]
        # a saturated ring CARBON means a hydro form, not a mancude lambda
        # parent (2,3-dihydro-... unbuilt for lambda rings) -> refuse
        if any(mol.GetAtomWithIdx(idx).GetSymbol() == 'C' for idx in sp3):
            return None
        # perfect mancude matching on the unsaturated part
        if 2 * n_double != ring_size - len(sp3):
            return None
        ih_locs = []
        for idx in sp3:
            atom = mol.GetAtomWithIdx(idx)
            capacity = lambda_by_locant.get(
                loc_of[idx], STANDARD_BONDING_NUMBER.get(atom.GetSymbol(), 0)
            )
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
    # P-22.2.1 retained stems: the mancude S/Se/Te 5-ring HW names are
    # replaced by their retained parents (BB example: 1H-1lambda4-THIOPHENE,
    # not 1H-1lambda4-thiole).
    for hw_stem, retained in (
        ('thiole', 'thiophene'),
        ('selenole', 'selenophene'),
        ('tellurole', 'tellurophene'),
    ):
        if hw.endswith(hw_stem):
            hw = hw[: -len(hw_stem)] + retained
            break
    return ih_prefix + hw


def name_heterocycle(mol, ring_atoms) -> Optional[str]:
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

    # P-22.2.7.1: lambda-convention heteromonocycle. Must run BEFORE the
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
        # ``bare_ring``: the molecule IS this ring, so no substituent, suffix or
        # unsaturation locant can be cited alongside the heteroatom's -- the only
        # condition under which P-22.2.3.2.1's omission of the sole heteroatom
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

    hw_name = build_hw_name(
        heteroatom_locants,
        ring_size,
        info['is_saturated'],
        info['is_aromatic']
    )
    if hw_name is None:
        # A ring heteroatom has no Table 2.4 prefix -> refuse. Returning early
        # also keeps the indicated-hydrogen step below from doing ``ih + None``.
        return None

    # Post-process: replace OPSIN-incompatible HW stems with the IUPAC-
    # preferred retained STEMS. "oxine"/"thiine" (6-membered unsaturated
    # O/S-rings) are not recognized by OPSIN; IUPAC 2013 uses "pyran"/
    # "thiopyran" with the tautomer's indicated hydrogen — which is computed
    # generically below (2H-pyran vs 4H-pyran; the old hardcoded '2H-'
    # mislabelled the 4H tautomers). "azine" -> "pyridine" (aromatic, no
    # indicated H).
    _HW_TO_RETAINED = {
        'oxine': 'pyran',
        'azine': 'pyridine',
        'thiine': 'thiopyran',
    }
    if hw_name in _HW_TO_RETAINED:
        hw_name = _HW_TO_RETAINED[hw_name]

    # Wave2 T2d (P-22.2.2.1.4 / P-31.1.4.2.4): indicated hydrogen for a
    # mancude monocyclic parent (2H-1,3-dioxole, 1H-azirine, 2H-/4H-pyran).
    # Returns '' outside its fail-closed scope (aromatic, saturated, hydro
    # forms, multi-indicated-H rings) — the bare-stem status quo.
    ih = _monocycle_indicated_h_prefix(mol, oriented, info)
    if ih:
        hw_name = ih + hw_name

    return hw_name


def _monocycle_indicated_h_prefix(mol, oriented: List[int], info) -> str:
    """Indicated-hydrogen prefix ('2H-') for a mancude monocyclic HW parent.

    P-22.2.2.1.4 / P-31.1.4.2.4: after the maximum number of noncumulative
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
        # P-54.4.1: an aromatic mancude ring with a single "pyrrole-type"
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
    for bond in mol.GetBonds():
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

    # The indicated-H atom: sp3 (no ring double bond), eligible, bearing H.
    db_atoms = {a for pr in db_pairs for a in pr}
    sp3h = [
        idx for pos, idx in enumerate(oriented)
        if idx not in db_atoms and eligible[pos]
        and mol.GetAtomWithIdx(idx).GetTotalNumHs() >= 1
    ]
    if len(sp3h) != 1:
        return ""
    target = sp3h[0]

    # Lowest indicated-H locant among the ring numberings that keep the
    # chosen heteroatom locant->element assignment IDENTICAL (P-31.1.4.3.4:
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


def _table_1_5_prefix(element: str) -> Optional[str]:
    """Table-1.5 skeletal replacement prefix (P-15.4.1.1), or None to refuse.

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

    # P-22.2.3.2.1: the sole heteroatom's locant '1' is omitted -- but ONLY when NO
    # OTHER locant is cited anywhere in the final name, which is the whole reason
    # the rule is safe: with nothing else numbered, the origin cannot be ambiguous.
    # Scoped by measurement, not by reading:
    #   * the Blue Book KEEPS the locant on the unsaturated forms
    #     ``1-oxacycloundeca-2,4,6,8,10-pentaene`` (PIN) and
    #     ``1-azacyclotetradeca-1,3,5,7,9,11,13-heptaene`` (PIN) [BBv2:8488-8490],
    #     where unsaturation locants are cited -> hence ``_saturated_ring``;
    #   * and the shipped gold target ``1-selenacyclotridecan-3-one``
    #     (W2-RINGKET-SELENA-PROTECT) keeps it too, because a SUFFIX locant is
    #     cited. That name is assembled by this builder's CALLER, which appends
    #     ``-3-one`` to the ``...ane`` parent returned here -- so this function
    #     cannot see it, and an omission decided from the arguments alone
    #     regressed that target (measured, not hypothesised).
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
        # (P-22.2.3 [BBv2:8482]: "For monocyclic rings with eleven and more ring
        # members, skeletal replacement ('a') nomenclature (see P-15.4) is
        # used"), and we spell them from the Table-1.5 source.
        #
        # NOT ESTABLISHED -- which table governs HERE is genuinely ambiguous in
        # the book, and it matters for exactly two elements:
        #   * P-22.2.3 points at P-15.4, i.e. Table 1.5;
        #   * but P-22.2.3.1 [BBv2:8484] says the prefixes come from "(see
        #     Table 2.4)" -- the Hantzsch-Widman table -- in the same sentence
        #     that quotes Table 2.4's 22-element seniority order. (That
        #     cross-reference may simply be reaching for the ORDER, which
        #     Table 1.5 does not print at all.)
        # The two tables agree on 16 of the elements this path can reach, so the
        # question only bites for Al (``alumina`` vs ``aluma``) and In (``inda``
        # vs ``indiga``). Table 1.5 is kept because the Table 2.4 reading is
        # unverifiable downstream: ``alumacyclotridecane`` and
        # ``indigacyclotridecane`` are REJECTED by the round-trip parser
        # ("cyclotridecane ... not parseable" after an HW prefix) while
        # ``aluminacyclotridecane`` and ``indacyclotridecane`` parse back to the
        # right structure. Switching would trade a verified name for an
        # unverifiable one, and SELF-01 fails OPEN on an unparseable name -- so
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
            # P-22.2.3.2.1 [BBv2:8494]: "When a single heteroatom is present in
            # the ring, it is assigned the locant '1', which is OMITTED in the
            # name" -- the BB's own saturated example is ``thiacyclododecane
            # (PIN)`` [BBv2:8488], and ``thiacyclododecane`` /
            # ``azacyclotridecane`` are shipped gold targets carrying exactly
            # this citation. This builder emitted the locant for EVERY element;
            # O/S/N/Si never reach it (an earlier producer, which already omits
            # the locant, handles them) so the defect only surfaced for an
            # element that falls through to here -- which Al and In began doing
            # when v29 P2-T2b made them spellable. Without this the two new rows
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


def _is_monocyclic_lactam(mol):
    """Late-bound wrapper for lactams.is_monocyclic_lactam (avoids a module-load
    circular import between heterocycles and lactams)."""
    from .lactams import is_monocyclic_lactam
    return is_monocyclic_lactam(mol)


def _ring_has_extra_heteroatom(mol, ring_set, carbonyl_idx) -> bool:
    """True if the ring containing `carbonyl_idx` carries a ring heteroatom
    besides the single amide N — i.e. it is a multi-heteroatom saturated
    heteroring (thiazolidine, oxazolidine, ...) that the lactam handler
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

            # ---- P-29.2 free-valence morphology, decided ONCE, up front ----
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
            from ..assembly.substituent_enumerator import (
                carbon_free_valence_prefix)
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

            # C4 (P-62.2.2 / P-62.2.1.1.1): an N-substituted exocyclic amine on a
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

            # v26 (P-63.2.2 / P-63.2.5): a heteroatom-rooted ETHER substituent on
            # the ring -- -O-R / -S-R / -Se-R / -Te-R (carbon_count>0 because R has
            # carbons) -- is an (R)oxy / (R)sulfanyl / (R)selanyl / (R)tellanyl
            # prefix. classify_substituent below counts the arm carbons and DROPS
            # the O/S/Se root (F-OXANE-DROP -> decline -> unknown; e.g. methoxy-/
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
                    from ..assembly.substituent_naming import name_substituent_fragment
                    from ..assembly.naming_utils import (
                        is_complex_substituent, apply_enclosing_marks)
                    if _eth_root.GetSymbol() == 'O':
                        # O-ether: the fragment namer's O-attach path contracts the
                        # PIN (methyl+oxy -> methoxy, ethyl -> ethoxy) and encloses
                        # a complex arm itself -> pass the WHOLE -O-R fragment.
                        _eth_name = name_substituent_fragment(
                            mol, sub_atoms, nbr_idx, list(ring_set))
                    else:
                        # S/Se/Te: name the arm, append the chalcogen stem, enclose
                        # a complex arm (P-63.2.5; e.g. (prop-2-en-1-yl)sulfanyl).
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

            # Detect non-carbon functional substituents (amino, hydroxy, nitro, etc.)
            hetero_sub_name = None
            if carbon_count == 0:
                # Check for functional groups that have no carbons
                hetero_sub_name = _identify_hetero_substituent(mol, sub_atoms, ring_set)
                if hetero_sub_name is None:
                    # An unrecognised no-carbon exocyclic group (e.g. the oxygen
                    # of an -O-SO3H sulfate or -O-PO(OH)2 phosphate ester). BFS
                    # only walks heavy atoms, so this is NEVER just implicit H —
                    # silently dropping it corrupts the structure (F-OXANE-DROP).
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

            # P-64.6.2 / P-66.1.6: a ring carbonyl whose principal group is a
            # CYCLIC secondary amide that the lactam handler DECLINED (a
            # multi-heteroatom saturated heteroring, e.g. 1,3-thiazolidin-4-one)
            # is named as the ketone '-one' suffix on the heterocycle parent,
            # NOT an 'oxo' prefix (the lactam/carboxamide name is unavailable).
            # Single-heteroatom lactams route through the lactam handler and
            # never reach here, so they are untouched.
            ring_ketone_suffix = (
                not is_principal_suffix
                and principal_group == 'secondary_amide'
                and hetero_sub_name == 'oxo'
                and _is_monocyclic_lactam(mol) is None
                and _ring_has_extra_heteroatom(mol, ring_set, ring_atom_idx)
            )

            if is_principal_suffix:
                sub_info['is_suffix'] = True
                sub_info['suffix_name'] = pg_ring_suffix
            elif ring_ketone_suffix:
                sub_info['is_suffix'] = True
                sub_info['suffix_name'] = 'one'
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

    # Check for hydroxamic acid: C(=O)(NH-OH) — before primary amide because
    # N has H1 bonded to O, not H2, so the primary amide pattern would not match.
    hydroxamic_pat = Chem.MolFromSmarts('[CX3](=O)[NX3;H1][OX2H]')
    if hydroxamic_pat:
        for match in mol.GetSubstructMatches(hydroxamic_pat):
            if match[0] == start_idx:
                return {'suffix_name': 'carboxamide', 'n_hydroxy': True}

    # C1 (P-66.3.1.1): ring-attached hydrazide C(=O)-NH-NH2 -> '-carbohydrazide'.
    # Checked BEFORE the amide pattern: the hydrazide N is bonded to another N so
    # the amide SMARTS never matches it (pyridine-4-carbohydrazide,
    # furan-2-carbohydrazide).
    hydrazide_pat = Chem.MolFromSmarts('[CX3](=O)[NX3][NX3]')
    if hydrazide_pat:
        for match in mol.GetSubstructMatches(hydrazide_pat):
            if match[0] == start_idx:
                return {'suffix_name': 'carbohydrazide'}

    # Check for primary amide: C(=O)(NH2)
    amide_pat = Chem.MolFromSmarts('[CX3](=O)[NX3H2]')
    if amide_pat:
        for match in mol.GetSubstructMatches(amide_pat):
            if match[0] == start_idx:
                return {'suffix_name': 'carboxamide'}

    # Wave2 T5d (P-66.1.1.3.4): N-SUBSTITUTED ring carboxamides — secondary
    # C(=O)NHR and tertiary C(=O)NR2. The benzene path has handled these
    # since Phase 85 (N,N-dimethylbenzamide); the heterocycle detector only
    # matched primary amides, so every N-substituted heterocycle-carboxamide
    # fell to unknown. Reuses the benzene N-substituent extractor; FAIL-CLOSED
    # count check: _detect_n_substituents silently omits branches it cannot
    # name, so a mismatch with the N's real branch count declines the whole
    # suffix record (missing-beats-wrong) rather than dropping a substituent.
    for _pat_smarts in ('[CX3](=O)[NX3;H1][#6]', '[CX3](=O)[NX3;H0]([#6])[#6]'):
        _pat = Chem.MolFromSmarts(_pat_smarts)
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

    # Chalcogen ylidene (=S/=Se/=Te) on a ring carbon (P-64.6.2): the heavier
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


# P-31.1.4.3.4 (heteroatom-locant citation for retained Hantzsch-Widman-derived
# saturated hetero-ring parents when they carry a characteristic-group suffix).
# Bare parent names ('thiazolidine') carry the numbering implicitly, but a
# suffixed PIN must cite the heteroatom locant set immediately before the parent
# stem: '1,3-thiazolidin-4-one', '1,3-oxazolidin-2-one'. Curated + OPSIN-verified:
# only these two retained stems BOTH require and accept the explicit locant
# citation in OPSIN 2.9. The 'iso' (1,2-) forms and the all-nitrogen
# imidazolidine/pyrazolidine and the retained morpholine/piperazine/piperidine
# families do NOT take (and OPSIN rejects) an explicit heteroatom-locant prefix,
# so they are deliberately excluded. Value = expected element-ordered heteroatom
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
    would silently drop atoms (F-OXANE-DROP / structure loss). Callers treat a
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
        >>> mol = Chem.MolFromSmiles('CN1CCCC1')  # N-methylpyrrolidine
        >>> # ... get ring, orient, get substituents ...
        >>> name_substituted_heterocycle(mol, ring, 'pyrrolidine', subs, atom_to_loc)
        'N-methylpyrrolidine'
    """
    from ..assembly.naming_utils import (
        get_alkyl_name,
        alpha_sort_key,
        get_multiplier_prefix,
        _join_multiplied_suffix,  # P-63.1.2/P-64.2.2.1 multiplier-'a' elision (tetraol->tetrol)
        _wrap_n_substituent,  # C4: italic-N substituent wrapping for amine suffix
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
    suffix_n_hydroxy: set = set()  # suffix_names whose FG was a hydroxamic acid
    # C4: N-substituents carried on the amine ring-suffix (pyridin-4-amine ->
    # N-methyl/N-phenyl). Keyed by suffix_name; single-instance only in scope.
    suffix_n_substituents: Dict[str, List[str]] = {}
    # v26 companion (P-62.2.2): for a MULTI-amine ring parent (triazine-2,4-diamine
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
                # group (F-OXANE-DROP / structure loss). Fail-closed.
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
                # P-29.2: the attachment bond is double or triple, and
                # get_heterocycle_substituents already built the only prefix
                # that spells that free valence (methylidene, ethylidene, ...).
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
                # the heteroatoms (F-OXANE-DROP, structure loss), so the
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
                if sub_name is None:
                    if not all_c_h:
                        # A heteroatom-bearing substituent the fragment namer
                        # could not name correctly and completely -> decline the
                        # whole heterocycle candidate rather than emit a
                        # heteroatom-dropping alkyl name (F-OXANE-DROP).
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
        multiplier = get_multiplier_prefix(count, chosen_suffix) if count > 1 else ""
        locant_str = ",".join(str(loc) for loc in chosen_locants)
        # P-63.1.2/P-64.2.2.1: elide the multiplier's terminal 'a' before a
        # vowel-initial suffix ('tetra'+'ol' -> 'tetrol', not 'tetraol').
        suffix_token = _join_multiplied_suffix(multiplier, chosen_suffix)
        # P-31.1.4.3.4: a suffixed retained HW-derived saturated hetero-ring
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
        # IUPAC P-16.3.3: elide the parent's terminal 'e' before a suffix token
        # that begins with a vowel (piperidine -> piperidin-4-one,
        # pyridine -> pyridin-2-ol). A consonant-initial token (multiplied
        # 'dione'/'triol', or 'carb*') keeps the 'e' (piperidine-2,6-dione,
        # pyridine-3-carbaldehyde).
        if combined and combined[-1] == 'e' and suffix_token[:1].lower() in 'aeiouy':
            combined = combined[:-1]
        combined = f"{combined}-{locant_str}-{suffix_token}"

        # C4 (P-62.2.1.1.1): prepend the amine N-substituent prefixes
        # (N-methyl / N-phenyl / N,N-dimethyl) to the ring-amine suffix name.
        # Wave2 T5d (P-66.1.1.3.4): same mechanism serves the N-substituted
        # ring carboxamide (N,N-diethylfuran-2-carboxamide).
        if chosen_suffix == 'amine' and len(suffix_fg.get('amine', [])) > 1 \
                and amine_n_by_locant:
            # v26 companion (P-62.2.2): MULTI-amine ring -> each amine nitrogen's
            # N-substituent(s) carry that nitrogen's RING locant as an italic-N
            # superscript (N2-tert-butyl-N4-cyclopropyl-...). Cited in ring-locant
            # order; the exocyclic order does not change the structure, so a
            # non-PIN order still round-trips (SELF-01 accepts).
            _parts = []
            for _loc in sorted(amine_n_by_locant):
                _subs = amine_n_by_locant[_loc]
                if len(_subs) == 2 and _subs[0] == _subs[1]:
                    _mp = get_multiplier_prefix(2, _subs[0])
                    _parts.append(
                        f"N{_loc},N{_loc}-{_mp}{_wrap_n_substituent(_subs[0])}")
                else:
                    for _s in _subs:
                        _parts.append(f"N{_loc}-{_wrap_n_substituent(_s)}")
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
                _n_prefix = f"N-{_wrap_n_substituent(_n_subs[0])}"
            elif len(_n_subs) == 2 and _n_subs[0] == _n_subs[1]:
                _mp = get_multiplier_prefix(2, _n_subs[0])
                _n_prefix = f"N,N-{_mp}{_wrap_n_substituent(_n_subs[0])}"
            else:
                _n_prefix = "-".join(
                    f"N-{_wrap_n_substituent(_n)}" for _n in _n_subs
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
        # (P-65.1.3.4: N-hydroxy is an N-substituent on the amide parent).
        if chosen_suffix == 'carboxamide' and chosen_suffix in suffix_n_hydroxy:
            combined = f"N-hydroxy{combined}"

        # Remaining suffix FGs become prefixes (carboxy, formyl, etc.)
        _SUFFIX_TO_PREFIX = {
            'carboxylic acid': 'carboxy',
            'carbaldehyde': 'formyl',
            'carboxamide': 'carbamoyl',
            'carbohydrazide': 'hydrazinecarbonyl',  # C1 (P-66.3.5)
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

    WS-A task 9 (P-14.3.2): when the ring nitrogen's NUMERIC locants are
    known they are the PIN citation ('4-cyclohexylmorpholine',
    '1-(naphthalen-2-yl)pyrrolidine'); the italic-N form ('N-methyl') is
    the fallback when no numbering is available. Complex substituent
    names get the same P-16.3.3 enclosure as C-substituents.
    """
    from ..assembly.naming_utils import (
        _wrap_n_substituent, is_complex_substituent, needs_brackets,
    )
    if locants:
        from ..assembly.naming_utils import _has_stereo_prefix
        display = name
        if _has_stereo_prefix(name):
            # '(S)-sec-butyl' under a numeric locant needs P-16.3.3
            # brackets: '1-[(S)-sec-butyl]...' (the naive startswith('(')
            # check skipped enclosure and emitted '1-(S)-sec-butyl...').
            display = f'[{name}]'
        elif not _fully_enclosed(name) and (
                is_complex_substituent(name) or needs_brackets(name)):
            # P-16.3.3: a name that is NOT fully enclosed by one outer pair of
            # parentheses — e.g. '(pyrimidin-5-yl)methyl' where the paren closes
            # before 'methyl' — still needs enclosure. Use [] when the name
            # already contains '(' (nesting rule), else use ().
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
    from ..assembly.naming_utils import (
        is_complex_substituent, needs_brackets, _has_stereo_prefix,
    )

    # _fully_enclosed is defined at module level (shared with _format_n_substituent)

    if _has_stereo_prefix(name):
        # Name has CIP stereo prefix like "(R)-sec-butyl":
        # use square brackets per IUPAC P-16.3.3
        display_name = f'[{name}]'
    elif not _fully_enclosed(name) and (
            is_complex_substituent(name) or needs_brackets(name)):
        # is_complex_substituent governs the di-/bis- multiplier choice;
        # needs_brackets is the broader P-14.5.2 enclosing test that also
        # flags compound FG-on-alkyl prefixes (hydroxymethyl, aminomethyl)
        # which take a SIMPLE multiplier but STILL require parentheses
        # (the benzene path's '1,3,5-tri(hydroxymethyl)benzene' convention).
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
