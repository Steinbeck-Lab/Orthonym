"""
Bicyclo compound naming according to IUPAC 2013 nomenclature.

Implements bicyclo[x.y.z] descriptor generation for bridged bicyclic hydrocarbons.

IUPAC Reference: Blue Book 2013, P-23.2 (Bridged bicyclic hydrocarbons)

The bicyclo descriptor format is bicyclo[x.y.z] where:
- x, y, z are the number of atoms in each bridge BETWEEN the bridgeheads
  (i.e., path length - 2, excluding the bridgehead atoms)
- Values are sorted in descending order: x >= y >= z
- Total ring atoms = x + y + z + 2 (the +2 accounts for bridgehead atoms)

Examples:
- Norbornane: bicyclo[2.2.1]heptane (bridges: 2, 2, 1; total: 7 carbons)
- Bicyclo[2.2.2]octane (bridges: 2, 2, 2; total: 8 carbons)
"""

import logging
from typing import List, Optional, Set, Tuple, Dict
from collections import deque
from functools import cmp_to_key
from itertools import permutations
from rdkit import Chem
from rdkit.Chem import BondType

from ..perception.rings import get_bridgehead_atoms, get_spiro_atoms, find_ring_bridgeheads
from .locants import compare_locant_sets

logger = logging.getLogger(__name__)


# ============================================================================
# Alkane Parent Names (shared with naming_utils but kept local for clarity)
# ============================================================================

_ALKANE_NAMES = {
    1: "methane",
    2: "ethane",
    3: "propane",
    4: "butane",
    5: "pentane",
    6: "hexane",
    7: "heptane",
    8: "octane",
    9: "nonane",
    10: "decane",
    11: "undecane",
    12: "dodecane",
}


def _get_alkane_name(carbon_count: int) -> str:
    """Get the alkane parent name for a carbon count."""
    if carbon_count in _ALKANE_NAMES:
        return _ALKANE_NAMES[carbon_count]
    from ..data.chain_names import get_chain_name
    return get_chain_name(carbon_count)


# ============================================================================
# Bridgehead Detection
# ============================================================================

def find_true_bridgeheads(mol) -> Set[int]:
    """
    Find the true bridgehead atoms in a bicyclic system.

    For a simple bicyclic system, bridgehead atoms are:
    1. In 2 or more rings
    2. Have 3 neighbors all within the ring system

    This is more specific than get_bridgehead_atoms() which returns
    all atoms in multiple rings.

    Args:
        mol: RDKit Mol object

    Returns:
        Set of atom indices that are true bridgeheads

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> find_true_bridgeheads(mol)
        {2, 5}  # or similar indices for the bridgehead carbons
    """
    # SUB-02/D-07+D-08: delegate to the SINGLE consolidated predicate
    # (perception.rings.find_ring_bridgeheads, ring_neighbours >= 3). The old
    # body required exactly 3 TOTAL neighbours all-in-ring, which wrongly
    # excluded substituted/quaternary bridgeheads (camphor's gem-dimethyl) —
    # the SUB-02 bug. The von-Baeyer-applicability guards in is_bicyclo_system
    # (cycle_rank / spiro / aromatic-fused / zero-bridge) keep fused/spiro
    # systems (naphthalene, decalin, spiro) out of von Baeyer naming.
    return find_ring_bridgeheads(mol)


# ============================================================================
# System Detection
# ============================================================================

def is_bicyclo_system(mol) -> bool:
    """
    Check if a molecule is a simple bicyclo (bridged) system.

    A bicyclo system has:
    - Exactly 2 true bridgehead atoms
    - No spiro centers
    - At least 2 rings
    - Not an aromatic fused system (naphthalene, etc.)
    - Not a zero-bridge fused system (decalin, etc.)

    Note: Aromatic fused systems like naphthalene are technically bicyclic
    (bicyclo[4.4.0]decapentaene) but IUPAC prefers their retained names.
    This function excludes aromatic fused systems.

    Systems with a zero-length bridge (bicyclo[x.y.0]) are edge-fused
    and should use fused nomenclature (e.g., decahydronaphthalene).

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule is a bicyclo system suitable for bicyclo[x.y.z] naming

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> is_bicyclo_system(mol)
        True
        >>> mol = Chem.MolFromSmiles('C1CCCCC1')  # cyclohexane
        >>> is_bicyclo_system(mol)
        False
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene (aromatic fused)
        >>> is_bicyclo_system(mol)
        False
        >>> mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')  # decalin (zero-bridge fused)
        >>> is_bicyclo_system(mol)
        False
    """
    ri = mol.GetRingInfo()

    # Must have at least 2 rings
    if ri.NumRings() < 2:
        return False

    # Ring count guard: reject tricyclo+ systems (cycle_rank >= 3)
    # Uses cycle_rank = ring_bonds - ring_atoms + 1, which is always reliable
    # regardless of SSSR issues. This prevents complex polycyclic systems
    # from being misclassified as bicyclo (find_true_bridgeheads undercounts
    # bridgeheads for such systems).
    ring_atoms_set = set()
    for ring in ri.AtomRings():
        ring_atoms_set.update(ring)

    # Filter to largest connected ring component to avoid inflated cycle rank
    # from disconnected ring systems (e.g., bicycle + sugar rings)
    from .polycyclic import _get_largest_connected_ring_component
    ring_atoms_set = _get_largest_connected_ring_component(mol, ring_atoms_set)

    ring_bond_count = 0
    for bond in mol.GetBonds():
        if bond.IsInRing() and bond.GetBeginAtomIdx() in ring_atoms_set and bond.GetEndAtomIdx() in ring_atoms_set:
            ring_bond_count += 1
    cycle_rank = ring_bond_count - len(ring_atoms_set) + 1
    if cycle_rank >= 3:
        return False

    # Must not be spiro
    if get_spiro_atoms(mol):
        return False

    # Must have exactly 2 true bridgeheads
    bridgeheads = find_true_bridgeheads(mol)
    if len(bridgeheads) != 2:
        return False

    # Check if this is an aromatic fused system (exclude these)
    # Aromatic fused systems have aromatic bridgehead atoms
    bh_list = list(bridgeheads)
    bh1, bh2 = bh_list[0], bh_list[1]
    atom1 = mol.GetAtomWithIdx(bh1)
    atom2 = mol.GetAtomWithIdx(bh2)

    if atom1.GetIsAromatic() and atom2.GetIsAromatic():
        # Both bridgeheads are aromatic - this is an aromatic fused system
        # Use retained names (naphthalene, etc.) instead of bicyclo naming
        return False

    # Check for zero-length bridge (fused system, not truly bridged).
    # bicyclo[x.y.0] means the bridgeheads share a direct bond (edge-fused).
    # These should use fused nomenclature (e.g., decahydronaphthalene)
    # rather than bicyclo[x.y.0] naming -- BUT only when both rings are
    # large enough (size >= 5) to have a known fused aromatic parent.
    # Very small ring systems like bicyclo[1.1.0]butane (two fused
    # cyclopropanes) have no fused parent name and keep the bicyclo format.
    lengths = get_bridge_lengths(mol, bh1, bh2)
    if lengths and min(lengths) == 0:
        # Check ring sizes -- only route to fused if rings are >= 5
        ri_rings = ri.AtomRings()
        ring_sizes = sorted([len(r) for r in ri_rings])
        if len(ring_sizes) >= 2 and ring_sizes[-2] >= 5:
            return False

    return True


# ============================================================================
# Bridge Path Finding
# ============================================================================

def find_bridge_paths(mol, bridgehead1: int, bridgehead2: int) -> List[List[int]]:
    """
    Find all paths between two bridgehead atoms.

    Uses BFS to find all simple paths between the bridgeheads.
    For a bicyclo system, there should be exactly 3 paths.

    Args:
        mol: RDKit Mol object
        bridgehead1: Index of first bridgehead atom
        bridgehead2: Index of second bridgehead atom

    Returns:
        List of paths, where each path is a list of atom indices
        including both bridgeheads

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> bridgeheads = list(find_true_bridgeheads(mol))
        >>> paths = find_bridge_paths(mol, bridgeheads[0], bridgeheads[1])
        >>> len(paths)
        3
    """
    # Get all ring atoms to constrain search
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    all_paths = []

    # BFS to find all paths
    # Each state is (current_atom, path_so_far)
    queue = deque([(bridgehead1, [bridgehead1])])

    while queue:
        current, path = queue.popleft()

        if current == bridgehead2 and len(path) > 1:
            all_paths.append(path)
            continue

        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()

            # Skip if already in path (except target)
            if n_idx in path and n_idx != bridgehead2:
                continue

            # Only follow ring atoms
            if n_idx not in ring_atoms:
                continue

            # If this is the target, add to path and save
            if n_idx == bridgehead2:
                queue.append((n_idx, path + [n_idx]))
            # Otherwise, continue exploring (but don't go back to start)
            elif n_idx != bridgehead1:
                queue.append((n_idx, path + [n_idx]))

    return all_paths


# ============================================================================
# Bridge Length Calculation
# ============================================================================

def get_bridge_lengths(mol, bridgehead1: int, bridgehead2: int) -> List[int]:
    """
    Calculate the bridge lengths between two bridgehead atoms.

    Bridge length = number of atoms BETWEEN bridgeheads (path length - 2).
    Returns lengths sorted in descending order: x >= y >= z.

    Args:
        mol: RDKit Mol object
        bridgehead1: Index of first bridgehead atom
        bridgehead2: Index of second bridgehead atom

    Returns:
        List of bridge lengths sorted descending

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> bridgeheads = list(find_true_bridgeheads(mol))
        >>> get_bridge_lengths(mol, bridgeheads[0], bridgeheads[1])
        [2, 2, 1]  # bicyclo[2.2.1]
    """
    paths = find_bridge_paths(mol, bridgehead1, bridgehead2)

    # Bridge length = path length - 2 (exclude both bridgeheads)
    lengths = [len(path) - 2 for path in paths]

    # Sort descending
    lengths.sort(reverse=True)

    return lengths


# ============================================================================
# Descriptor Generation
# ============================================================================

def generate_bicyclo_descriptor(mol) -> Optional[str]:
    """
    Generate the bicyclo[x.y.z] descriptor for a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Descriptor string like "bicyclo[2.2.1]", or None if not a bicyclo system

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> generate_bicyclo_descriptor(mol)
        'bicyclo[2.2.1]'
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1CC2')  # bicyclo[2.2.2]octane
        >>> generate_bicyclo_descriptor(mol)
        'bicyclo[2.2.2]'
    """
    if not is_bicyclo_system(mol):
        return None

    bridgeheads = list(find_true_bridgeheads(mol))
    if len(bridgeheads) != 2:
        return None

    lengths = get_bridge_lengths(mol, bridgeheads[0], bridgeheads[1])

    if len(lengths) != 3:
        return None

    # Verify bridge sum + 2 = total ring atoms (IUPAC invariant).
    # This catches invalid bridge calculations before generating bad names.
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)
    from .polycyclic import _get_largest_connected_ring_component
    ring_atoms = _get_largest_connected_ring_component(mol, ring_atoms)
    bridge_sum = sum(lengths)
    if bridge_sum + 2 != len(ring_atoms):
        return None

    return f"bicyclo[{lengths[0]}.{lengths[1]}.{lengths[2]}]"


# ============================================================================
# Full Naming
# ============================================================================

def name_bicyclo_system(mol) -> Optional[str]:
    """
    Generate the full IUPAC name for a bicyclo system.

    Checks for retained names first, then generates systematic name.

    Args:
        mol: RDKit Mol object

    Returns:
        Full IUPAC name like "bicyclo[2.2.1]heptane" or "norbornane",
        or None if not a bicyclo system

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> name_bicyclo_system(mol)
        'norbornane'  # or 'bicyclo[2.2.1]heptane' depending on retained name preference
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1CC2')
        >>> name_bicyclo_system(mol)
        'bicyclo[2.2.2]octane'
    """
    # Import here to avoid circular imports
    from ..data.bicyclo_systems import get_retained_bicyclo_name

    # Check for retained name first
    canonical = Chem.CanonSmiles(Chem.MolToSmiles(mol))
    retained = get_retained_bicyclo_name(canonical)
    if retained:
        return retained

    # Generate descriptor
    descriptor = generate_bicyclo_descriptor(mol)
    if not descriptor:
        return None

    # Count total ring carbons
    # For simple bicyclics, count atoms in the ring system
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    # Filter to largest connected ring component to avoid counting
    # atoms from disconnected ring systems (e.g., sugar rings)
    from .polycyclic import _get_largest_connected_ring_component
    ring_atoms = _get_largest_connected_ring_component(mol, ring_atoms)

    # Count only carbon atoms in ring
    carbon_count = sum(
        1 for idx in ring_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    )

    # Check for heteroatoms in ring
    heteroatoms = []
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            heteroatoms.append((idx, symbol))

    if heteroatoms:
        # Heterobicyclic - needs aza/oxa prefix
        # For now, return systematic name with descriptor only
        # Full hetero naming is more complex and deferred
        total_ring = len(ring_atoms)
        parent_name = _get_alkane_name(total_ring)
        return f"{descriptor}{parent_name}"

    # All-carbon bicyclic
    parent_name = _get_alkane_name(carbon_count)
    return f"{descriptor}{parent_name}"


def get_bicyclo_ring_atoms(mol) -> Optional[Set[int]]:
    """
    Get all atoms in the bicyclo ring system.

    Args:
        mol: RDKit Mol object

    Returns:
        Set of atom indices in the ring system, or None if not bicyclo

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> get_bicyclo_ring_atoms(mol)
        {0, 1, 2, 3, 4, 5, 6}
    """
    if not is_bicyclo_system(mol):
        return None

    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    # Filter to largest connected ring component
    from .polycyclic import _get_largest_connected_ring_component
    ring_atoms = _get_largest_connected_ring_component(mol, ring_atoms)

    return ring_atoms


# ============================================================================
# Bicyclo Numbering
# ============================================================================

def get_bicyclo_numbering(mol, suffix_ring_atoms: Optional[Set[int]] = None) -> Optional[Dict[int, int]]:
    """
    Generate IUPAC numbering for a bicyclo system.

    ``suffix_ring_atoms`` (WS-6 / BBR-RCON): ring atoms that bear the principal
    characteristic group (e.g. the ring carbon double-bonded to =O of a ketone, or
    the ring carbon bearing an exocyclic -OH). When given, the admissible numbering
    that gives those atoms the lowest locants (after heteroatoms) is chosen per
    P-14.4(c).

    IUPAC bicyclo numbering rules:
    1. Start at one bridgehead atom (position 1)
    2. Number along the longest bridge to the other bridgehead
    3. Continue along the second longest bridge back toward position 1
    4. Number the shortest bridge last (back to neighbors of position 1)

    For bicyclo[2.2.1]heptane (norbornane):
    - Bridgeheads are positions 1 and 4
    - Longest bridge (2 atoms): 1 -> 2 -> 3 -> 4
    - Second longest (2 atoms): 4 -> 5 -> 6 -> 1
    - Shortest (1 atom): 1 -> 7 -> 4

    Args:
        mol: RDKit Mol object

    Returns:
        Dict mapping atom_idx -> IUPAC locant (1-indexed), or None if not bicyclo

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # norbornane
        >>> numbering = get_bicyclo_numbering(mol)
        >>> len(numbering)
        7
    """
    if not is_bicyclo_system(mol):
        return None

    bridgeheads = list(find_true_bridgeheads(mol))
    if len(bridgeheads) != 2:
        return None

    ring_atoms = get_bicyclo_ring_atoms(mol) or set()
    suffix_set = {i for i in (suffix_ring_atoms or set()) if i in ring_atoms}

    # WS-6 / BBR-RCON (DEF-7): enumerate the ADMISSIBLE von Baeyer numberings
    # (P-23.2.3 keeps the descriptor: bridgeheads at 1 and 1+longest, main bridges
    # before the secondary bridge) and pick the one with the lowest locants in
    # P-14.4 order: heteroatoms -> principal-group suffix -> ene/yne -> substituents.
    # The LEGACY topology-only numbering is candidate 0, so when every P-14.4 tier
    # ties (unsubstituted / symmetric rings) it wins the stable sort and the output
    # is byte-identical to before this change (regression containment).
    candidates: List[Dict[int, int]] = []
    legacy = _legacy_bicyclo_numbering(mol)
    if legacy:
        candidates.append(legacy)
    candidates.extend(_enumerate_bicyclo_numberings(mol, bridgeheads))
    if not candidates:
        return None

    # P-14.4 feature atom-sets (structural; derived from mol, version-stable).
    hetero = {i for i in ring_atoms if mol.GetAtomWithIdx(i).GetSymbol() != 'C'}
    ring_multibonds = []
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in ring_atoms and a2 in ring_atoms and b.GetBondTypeAsDouble() >= 2.0:
            ring_multibonds.append((a1, a2))
    sub_bearing = set()
    for i in ring_atoms:
        if i in suffix_set:
            continue  # the suffix atom is ranked in its own tier, not as a substituent
        atom = mol.GetAtomWithIdx(i)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in ring_atoms and nbr.GetAtomicNum() > 1:
                sub_bearing.add(i)
                break

    def _key_lists(a2l):
        het = sorted(a2l[i] for i in hetero if i in a2l)
        suf = sorted(a2l[i] for i in suffix_set if i in a2l)
        ene = sorted(min(a2l[a], a2l[b]) for a, b in ring_multibonds if a in a2l and b in a2l)
        sub = sorted(a2l[i] for i in sub_bearing if i in a2l)
        return (het, suf, ene, sub)

    def _cmp(x, y):
        for sx, sy in zip(_key_lists(x), _key_lists(y)):
            c = compare_locant_sets(sx, sy)
            if c != 0:
                return c
        return 0

    candidates.sort(key=cmp_to_key(_cmp))
    return candidates[0]


def _legacy_bicyclo_numbering(mol) -> Optional[Dict[int, int]]:
    """Pre-WS-6 topology-only numbering (bridgeheads[0] = C1; longest -> second ->
    shortest bridge, all in bh1->bh2 path order). Preserved verbatim as the
    tie-break default so unsubstituted / symmetric bicyclics stay byte-identical."""
    bridgeheads = list(find_true_bridgeheads(mol))
    if len(bridgeheads) != 2:
        return None
    bh1, bh2 = bridgeheads[0], bridgeheads[1]
    paths = find_bridge_paths(mol, bh1, bh2)
    if len(paths) != 3:
        return None
    paths_sorted = sorted(paths, key=lambda p: len(p), reverse=True)
    atom_to_locant: Dict[int, int] = {}
    current_locant = 1
    atom_to_locant[bh1] = current_locant
    current_locant += 1
    for atom_idx in paths_sorted[0][1:-1]:
        atom_to_locant[atom_idx] = current_locant
        current_locant += 1
    atom_to_locant[bh2] = current_locant
    current_locant += 1
    for atom_idx in paths_sorted[1][1:-1]:
        if atom_idx not in atom_to_locant:
            atom_to_locant[atom_idx] = current_locant
            current_locant += 1
    for atom_idx in paths_sorted[2][1:-1]:
        if atom_idx not in atom_to_locant:
            atom_to_locant[atom_idx] = current_locant
            current_locant += 1
    return atom_to_locant


def _enumerate_bicyclo_numberings(mol, bridgeheads) -> List[Dict[int, int]]:
    """All admissible von Baeyer numberings for a simple bicyclic (P-23.2.3):
    start at either bridgehead; assign the 3 bridges to (first, second, third)
    with non-increasing interior length (permuting only EQUAL-length bridges, so
    the descriptor is preserved); number the first bridge start->other, the second
    segment from the other (higher) bridgehead back, and the secondary bridge from
    the start (bridgehead-1) side."""
    bh_a, bh_b = bridgeheads[0], bridgeheads[1]
    out: List[Dict[int, int]] = []
    for start, other in ((bh_a, bh_b), (bh_b, bh_a)):
        spaths = find_bridge_paths(mol, start, other)
        if len(spaths) != 3:
            continue
        for perm in permutations(spaths):
            lens = [len(p) - 2 for p in perm]
            if not (lens[0] >= lens[1] >= lens[2]):
                continue
            first, second, third = perm
            a2l: Dict[int, int] = {}
            loc = 1
            a2l[start] = loc
            loc += 1
            for idx in first[1:-1]:
                if idx not in a2l:
                    a2l[idx] = loc
                    loc += 1
            if other not in a2l:
                a2l[other] = loc
                loc += 1
            for idx in reversed(second[1:-1]):  # second segment: from the higher bridgehead back
                if idx not in a2l:
                    a2l[idx] = loc
                    loc += 1
            for idx in third[1:-1]:  # secondary bridge: from the bridgehead-1 side
                if idx not in a2l:
                    a2l[idx] = loc
                    loc += 1
            if a2l and len(a2l) == len(set(a2l.values())):
                out.append(a2l)
    return out


# ============================================================================
# Bicyclo Substituent Detection
# ============================================================================

def get_bicyclo_substituents(mol, ring_atoms: Set[int]) -> Dict[int, List[Dict]]:
    """
    Find all substituents attached to a bicyclo ring system.

    For each ring atom, finds atoms connected to it that are NOT part of the
    ring system. Groups substituents by their attachment point.

    Args:
        mol: RDKit Mol object
        ring_atoms: Set of atom indices in the bicyclo ring

    Returns:
        Dict mapping ring_atom_idx -> list of substituent info dicts.
        Each substituent dict contains:
        - 'atoms': list of atom indices in substituent
        - 'carbon_count': number of carbons in substituent
        - 'attachment': ring atom index where substituent attaches
        - 'first_atom': first atom of substituent (directly bonded to ring)
        and, when the attachment bond is NOT single (P-29.2), exactly one of:
        - 'prefix_name': the ``-ylidene``/``-ylidyne`` prefix to use verbatim
        - 'unnameable': True -- the consumer MUST decline the whole parent

    P-29.2 note. ``carbon_count`` alone cannot tell ``-CH3`` from ``=CH2``, so a
    consumer that builds a prefix from it names a different molecule whenever the
    attachment bond is double: ``C=C1CC2CCC1C2`` came out as
    ``2-methylbicyclo[2.2.1]heptane``, which is C8H14 for a C8H12 input. The bond
    order is therefore read HERE, once, through the shared primitive, and the
    verdict is carried in the dict so every consumer inherits it rather than
    re-deriving it (or forgetting to).

    Examples:
        >>> mol = Chem.MolFromSmiles('CC1CC2CCC1C2')  # methylnorbornane
        >>> ring_atoms = get_bicyclo_ring_atoms(mol)
        >>> subs = get_bicyclo_substituents(mol, ring_atoms)
        >>> # Should find methyl substituent
    """
    from ..assembly.substituent_enumerator import carbon_free_valence_prefix

    substituents: Dict[int, List[Dict]] = {}

    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)
        subs_for_atom = []

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip if neighbor is also in ring
            if nbr_idx in ring_atoms:
                continue

            # This is a substituent - trace the entire substituent
            sub_atoms = _trace_substituent(mol, nbr_idx, ring_atoms)

            # Count carbons
            carbon_count = sum(
                1 for idx in sub_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )

            sub_info = {
                'atoms': sub_atoms,
                'carbon_count': carbon_count,
                'attachment': ring_idx,
                'first_atom': nbr_idx,
            }

            # P-29.2 free-valence morphology. Deferral is the common case and
            # leaves the dict exactly as it has always been.
            verdict = carbon_free_valence_prefix(mol, sub_atoms, nbr_idx)
            if not verdict.defers:
                if verdict.prefix is not None:
                    sub_info['prefix_name'] = verdict.prefix
                else:
                    sub_info['unnameable'] = True
                    logger.debug(
                        "bicyclo substituent at ring atom %d: %s",
                        ring_idx, verdict.basis)

            subs_for_atom.append(sub_info)

        if subs_for_atom:
            substituents[ring_idx] = subs_for_atom

    return substituents


def _trace_substituent(mol, start_idx: int, ring_atoms: Set[int]) -> List[int]:
    """
    Trace all atoms in a substituent using BFS.

    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index (first atom of substituent)
        ring_atoms: Set of ring atom indices to exclude

    Returns:
        List of atom indices in the substituent
    """
    visited = set()
    queue = deque([start_idx])
    result = []

    while queue:
        atom_idx = queue.popleft()

        if atom_idx in visited or atom_idx in ring_atoms:
            continue

        visited.add(atom_idx)
        result.append(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_atoms:
                queue.append(nbr_idx)

    return result


# ============================================================================
# Bicyclo Unsaturation Detection
# ============================================================================

def detect_bicyclo_unsaturation(mol, ring_atoms: Set[int]) -> Dict:
    """
    Detect double and triple bonds within a bicyclo ring system.

    Scans all bonds between atoms in the ring and identifies multiple bonds.

    Args:
        mol: RDKit Mol object
        ring_atoms: Set of atom indices in the bicyclo ring

    Returns:
        Dict with:
        - 'double_bonds': list of (atom_idx1, atom_idx2) tuples for double bonds
        - 'triple_bonds': list of (atom_idx1, atom_idx2) tuples for triple bonds

    Examples:
        >>> mol = Chem.MolFromSmiles('C1=CC2CCC1C2')  # norbornene
        >>> ring_atoms = get_bicyclo_ring_atoms(mol)
        >>> unsat = detect_bicyclo_unsaturation(mol, ring_atoms)
        >>> len(unsat['double_bonds'])
        1
    """
    double_bonds: List[Tuple[int, int]] = []
    triple_bonds: List[Tuple[int, int]] = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        # Both atoms must be in ring
        if begin_idx not in ring_atoms or end_idx not in ring_atoms:
            continue

        bond_type = bond.GetBondType()

        if bond_type == BondType.DOUBLE:
            # Store with lower index first for consistency
            double_bonds.append((min(begin_idx, end_idx), max(begin_idx, end_idx)))
        elif bond_type == BondType.TRIPLE:
            triple_bonds.append((min(begin_idx, end_idx), max(begin_idx, end_idx)))

    return {
        'double_bonds': double_bonds,
        'triple_bonds': triple_bonds,
    }


# ============================================================================
# Complete Bicyclo Naming Data
# ============================================================================

def get_complete_bicyclo_data(mol, suffix_ring_atoms: Optional[Set[int]] = None) -> Optional[Dict]:
    """
    Generate complete bicyclo naming data including substituents and unsaturation.

    This is the comprehensive data structure needed for full IUPAC naming.

    Args:
        mol: RDKit Mol object

    Returns:
        Dict with all naming data, or None if not a bicyclo system:
        - 'base_name': systematic base name (e.g., 'bicyclo[2.2.1]heptane')
        - 'descriptor': bicyclo descriptor (e.g., 'bicyclo[2.2.1]')
        - 'ring_atoms': set of ring atom indices
        - 'atom_to_locant': dict mapping atom_idx -> IUPAC locant
        - 'substituents': substituent data from get_bicyclo_substituents
        - 'unsaturation': unsaturation data from detect_bicyclo_unsaturation
        - 'bridgeheads': set of bridgehead atom indices
        - 'retained_name': retained name if applicable, else None
    """
    if not is_bicyclo_system(mol):
        return None

    # Get ring atoms
    ring_atoms = get_bicyclo_ring_atoms(mol)
    if not ring_atoms:
        return None

    # Get numbering (WS-6: P-14.4-lowest among admissible numberings, given the
    # principal-group ring atoms so the suffix takes the lowest locant)
    atom_to_locant = get_bicyclo_numbering(mol, suffix_ring_atoms=suffix_ring_atoms)
    if not atom_to_locant:
        return None

    # Get descriptor
    descriptor = generate_bicyclo_descriptor(mol)

    # Get substituents
    substituents = get_bicyclo_substituents(mol, ring_atoms)

    # Get unsaturation
    unsaturation = detect_bicyclo_unsaturation(mol, ring_atoms)

    # Get bridgeheads
    bridgeheads = find_true_bridgeheads(mol)

    # Count ALL ring atoms for parent name (IUPAC: heteroatoms count toward
    # ring size in replacement nomenclature). carbon_count is kept for
    # compatibility but total_ring_atoms is what determines the parent name.
    total_ring_atoms = len(ring_atoms)
    carbon_count = sum(
        1 for idx in ring_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    )

    # Generate heteroatom replacement prefix (oxa, aza, thia) if needed
    heteroatom_prefix = ""
    if carbon_count < total_ring_atoms and atom_to_locant:
        from .polycyclic import get_heteroatom_replacement_prefix
        heteroatom_prefix = get_heteroatom_replacement_prefix(
            mol, atom_to_locant, ring_atoms
        )
        if heteroatom_prefix is None:
            # A skeletal ring atom has no replacement prefix, so no name built
            # from this descriptor can express it while ``total_ring_atoms``
            # keeps counting it. Refuse rather than emit a hydrocarbon stem.
            return None

    # Check for retained name
    from ..data.bicyclo_systems import get_retained_bicyclo_name
    canonical = Chem.CanonSmiles(Chem.MolToSmiles(mol))
    retained = get_retained_bicyclo_name(canonical)

    # Build base name: use total ring atoms for parent (includes heteroatoms)
    parent_name = _get_alkane_name(total_ring_atoms)
    base_name = f"{heteroatom_prefix}{descriptor}{parent_name}" if descriptor else parent_name

    return {
        'base_name': base_name,
        'descriptor': descriptor,
        'ring_atoms': ring_atoms,
        'atom_to_locant': atom_to_locant,
        'substituents': substituents,
        'unsaturation': unsaturation,
        'bridgeheads': bridgeheads,
        'retained_name': retained,
        'carbon_count': total_ring_atoms,  # Use total ring atoms, not just carbons
        'heteroatom_prefix': heteroatom_prefix,
    }
