"""
Bridged fused nomenclature (FR-8) - systems that are part fused and part bridged.

Implements IUPAC 2013 P-25.7 rules for naming systems like 1,4-methanonaphthalene
where a fused core (naphthalene) has additional bridges across non-adjacent positions.

Key concepts:
- Bridged fused = fused core + additional bridges across the fused system
- Different from pure von Baeyer (no fused component)
- Different from pure fused (no bridges across)
- Name format: [locants]-[bridge_prefix][fused_parent_name]
  e.g., "1,4-methanonaphthalene", "1,4:5,8-dimethanonaphthalene"

Bridge prefixes (FR-8.3):
- Carbon bridges: methano (1C), ethano (2C), propano (3C), butano (4C)
- Heteroatom bridges: epoxy (O), epithio (S), epimino (NH), epidioxy (O-O)

Reference: IUPAC 2013 Blue Book, P-25.7 (Bridged Fused Ring Systems)
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, deque
from itertools import combinations
from rdkit import Chem

from .fused_rings import classify_fused_system, get_shared_atoms


# ============================================================================
# Constants
# ============================================================================

# Bridge prefix mapping: (length, element) -> prefix name
# For carbon bridges, length is the number of carbon atoms in the bridge
# For heteroatom bridges, element determines the prefix
BRIDGE_PREFIXES: Dict[Tuple[int, str], str] = {
    # Carbon bridges by length
    (1, 'C'): 'methano',
    (2, 'C'): 'ethano',
    (3, 'C'): 'propano',
    (4, 'C'): 'butano',
    (5, 'C'): 'pentano',
    (6, 'C'): 'hexano',
    # Zero-length heteroatom bridges (direct connection)
    (1, 'O'): 'epoxy',
    (1, 'S'): 'epithio',
    (1, 'N'): 'epimino',
    # Extended heteroatom bridges
    (2, 'O'): 'epidioxy',  # -O-O- peroxide bridge
    (2, 'N'): 'diazeno',   # -N=N- diazene bridge
}

# Alternate lookup by element for heteroatom bridges
HETEROATOM_BRIDGE_PREFIXES: Dict[str, str] = {
    'O': 'epoxy',
    'S': 'epithio',
    'N': 'epimino',
}

# Known fused ring parent names for lookup
# Minimal set - extend as needed
FUSED_PARENT_NAMES: Dict[str, str] = {
    # Bicyclic aromatics
    'naphthalene': 'naphthalene',
    'azulene': 'azulene',
    # Tricyclic aromatics
    'anthracene': 'anthracene',
    'phenanthrene': 'phenanthrene',
    'acenaphthene': 'acenaphthene',
    # Heterocyclic fused
    'quinoline': 'quinoline',
    'isoquinoline': 'isoquinoline',
    'indole': 'indole',
    'benzofuran': 'benzofuran',
    'carbazole': 'carbazole',
}

# Multiplier prefixes for multiple identical bridges
MULTIPLIERS: Dict[int, str] = {
    2: 'di',
    3: 'tri',
    4: 'tetra',
    5: 'penta',
    6: 'hexa',
}


# ============================================================================
# Core Detection Functions
# ============================================================================

def detect_bridged_fused(mol) -> bool:
    """
    Detect whether a molecule is a bridged fused system.

    A bridged fused system has BOTH:
    1. A fused ring core (two or more rings sharing edges)
    2. Additional bridges connecting non-adjacent atoms of the fused core

    This distinguishes bridged fused from:
    - Pure fused systems (like naphthalene) - no bridges
    - Pure bridged systems (like norbornane) - no fused core

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule is a bridged fused system, False otherwise

    Examples:
        >>> mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene
        >>> detect_bridged_fused(mol)
        False  # pure fused, no bridges
        >>> mol = Chem.MolFromSmiles("C1CC2CCC1C2")  # norbornane
        >>> detect_bridged_fused(mol)
        False  # pure bridged, no fused core
    """
    if mol is None:
        return False

    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    # Need at least 2 rings
    if len(atom_rings) < 2:
        return False

    # Step 1: Check for fused component (rings sharing edges)
    has_fused_component = False
    fused_ring_atoms = set()

    for i, ring1 in enumerate(atom_rings):
        for j, ring2 in enumerate(atom_rings):
            if i >= j:
                continue
            shared = get_shared_atoms(mol, ring1, ring2)
            if len(shared) >= 2:
                # Fused pair found (shared edge)
                has_fused_component = True
                fused_ring_atoms.update(ring1)
                fused_ring_atoms.update(ring2)

    if not has_fused_component:
        return False

    # Step 2: Check for bridges across the fused core
    # A bridge connects two atoms of the fused core via a path
    # that goes through atoms NOT in any fused ring

    # Get all ring atoms
    all_ring_atoms = set()
    for ring in atom_rings:
        all_ring_atoms.update(ring)

    # Find atoms that could be bridge atoms (in rings but not in fused core,
    # or in additional small rings bridging across)
    # Build adjacency for analysis
    bridging_found = _find_bridging_connections(mol, fused_ring_atoms, atom_rings)

    return bridging_found


def _find_bridging_connections(
    mol,
    fused_core_atoms: Set[int],
    atom_rings: List[Tuple[int, ...]]
) -> bool:
    """
    Check if there are bridging connections across the fused core.

    A bridging connection exists when:
    - Two non-adjacent atoms in the fused core are connected by a path
      that includes atoms from a different ring or non-ring atoms

    Args:
        mol: RDKit Mol object
        fused_core_atoms: Set of atom indices in the fused core
        atom_rings: List of ring atom tuples

    Returns:
        True if bridging connections exist
    """
    if not fused_core_atoms:
        return False

    # Build adjacency within fused core
    fused_adj = defaultdict(set)
    for idx in fused_core_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in fused_core_atoms:
                fused_adj[idx].add(nbr_idx)

    # Check each ring to see if it forms a bridge across the fused core
    for ring in atom_rings:
        ring_set = set(ring)

        # Skip if this ring is entirely part of fused core (no bridging)
        if ring_set.issubset(fused_core_atoms):
            continue

        # Check if this ring connects two atoms of the fused core
        # via atoms NOT in the fused core
        fused_in_ring = ring_set & fused_core_atoms
        non_fused_in_ring = ring_set - fused_core_atoms

        if len(fused_in_ring) >= 2 and len(non_fused_in_ring) >= 1:
            # This ring has atoms in fused core and atoms outside
            # Check if the non-fused atoms form a bridge
            fused_list = list(fused_in_ring)

            for i, atom1 in enumerate(fused_list):
                for atom2 in fused_list[i+1:]:
                    # Check if these atoms are NOT adjacent in the fused core
                    # but are connected via non-fused atoms in this ring
                    if atom2 not in fused_adj[atom1]:
                        # Non-adjacent in fused core - check if connected via non-fused
                        if _connected_via_set(mol, atom1, atom2, non_fused_in_ring):
                            return True

    # Also check for bridging atoms not in any SSSR ring
    # (e.g., a methano bridge with 1 CH2 connecting two fused core atoms)
    all_ring_atoms = set()
    for ring in atom_rings:
        all_ring_atoms.update(ring)

    # Find atoms bonded to fused core but not in any ring
    for idx in fused_core_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in all_ring_atoms:
                # Non-ring atom connected to fused core
                # Check if it connects to another fused core atom (forming a bridge)
                nbr_atom = mol.GetAtomWithIdx(nbr_idx)
                for nbr_neighbor in nbr_atom.GetNeighbors():
                    nbr_nbr_idx = nbr_neighbor.GetIdx()
                    if nbr_nbr_idx != idx and nbr_nbr_idx in fused_core_atoms:
                        # Found a bridge: idx -> nbr_idx -> nbr_nbr_idx
                        # where nbr_idx is not in any ring
                        if nbr_nbr_idx not in fused_adj[idx]:
                            return True

    return False


def _connected_via_set(mol, atom1: int, atom2: int, via_atoms: Set[int]) -> bool:
    """
    Check if atom1 and atom2 are connected via atoms in via_atoms.

    Uses BFS to find path from atom1 to atom2 through via_atoms.

    Args:
        mol: RDKit Mol object
        atom1: Starting atom index
        atom2: Target atom index
        via_atoms: Set of atom indices the path must go through

    Returns:
        True if a path exists from atom1 to atom2 through via_atoms
    """
    if not via_atoms:
        return False

    # Check if atom1 is adjacent to any via_atom
    visited = set()
    queue = deque()

    atom1_obj = mol.GetAtomWithIdx(atom1)
    for neighbor in atom1_obj.GetNeighbors():
        nbr_idx = neighbor.GetIdx()
        if nbr_idx in via_atoms:
            queue.append(nbr_idx)
            visited.add(nbr_idx)

    while queue:
        current = queue.popleft()

        # Check if we can reach atom2 from current
        current_atom = mol.GetAtomWithIdx(current)
        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx == atom2:
                return True
            if nbr_idx in via_atoms and nbr_idx not in visited:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    return False


# ============================================================================
# Fused Core Identification
# ============================================================================

def identify_fused_core(mol) -> Optional[Dict[str, Any]]:
    """
    Identify the maximal fused ring component (FR-8.2 algorithm).

    For a bridged fused system, the fused core is the largest set of
    rings that share edges (ortho-fused or ortho-peri-fused).

    Maximization priority (FR-8.2):
    1. Maximum number of fused rings
    2. Maximum number of skeletal atoms

    Args:
        mol: RDKit Mol object

    Returns:
        Dict with:
        - 'core_atoms': Set of atom indices in the fused core
        - 'core_name': Name of the fused parent if recognized
        - 'core_numbering': Dict mapping atom index to IUPAC locant
        - 'ring_count': Number of rings in the fused core
        Or None if no fused core found

    Examples:
        >>> mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene
        >>> core = identify_fused_core(mol)
        >>> len(core['core_atoms'])
        10
    """
    if mol is None:
        return None

    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    if len(atom_rings) < 2:
        return None

    # Build fusion graph: which rings are fused to which
    fusion_graph = defaultdict(set)
    for i, ring1 in enumerate(atom_rings):
        for j, ring2 in enumerate(atom_rings):
            if i >= j:
                continue
            shared = get_shared_atoms(mol, ring1, ring2)
            if len(shared) >= 2:  # Fused (shared edge)
                fusion_graph[i].add(j)
                fusion_graph[j].add(i)

    if not fusion_graph:
        return None

    # Find connected components of fused rings
    visited = set()
    components = []

    for ring_idx in range(len(atom_rings)):
        if ring_idx not in fusion_graph:
            continue
        if ring_idx in visited:
            continue

        # BFS to find all rings in this fused component
        component = set()
        queue = deque([ring_idx])
        while queue:
            current = queue.popleft()
            if current in component:
                continue
            component.add(current)
            visited.add(current)
            for neighbor in fusion_graph.get(current, []):
                if neighbor not in component:
                    queue.append(neighbor)

        components.append(component)

    if not components:
        return None

    # Select largest component (by ring count, then atom count)
    best_component = None
    best_score = (-1, -1)

    for component in components:
        ring_count = len(component)
        atom_count = len(set().union(*[set(atom_rings[i]) for i in component]))
        score = (ring_count, atom_count)
        if score > best_score:
            best_score = score
            best_component = component

    if best_component is None:
        return None

    # Collect atoms in the fused core
    core_atoms = set()
    for ring_idx in best_component:
        core_atoms.update(atom_rings[ring_idx])

    # Try to identify the fused parent name
    core_name = _identify_fused_parent_name(mol, core_atoms)

    # Generate basic numbering (placeholder - proper numbering is complex)
    core_numbering = {idx: i + 1 for i, idx in enumerate(sorted(core_atoms))}

    return {
        'core_atoms': core_atoms,
        'core_name': core_name,
        'core_numbering': core_numbering,
        'ring_count': len(best_component),
        'atoms': core_atoms,  # Alias for compatibility
    }


def _identify_fused_parent_name(mol, core_atoms: Set[int]) -> Optional[str]:
    """
    Try to identify the name of the fused parent from core atoms.

    Uses SMILES pattern matching for known fused systems.

    Args:
        mol: RDKit Mol object
        core_atoms: Set of atom indices in the fused core

    Returns:
        Parent name string or None if not recognized
    """
    # Create a submolecule from core atoms
    if len(core_atoms) == 10:
        # Could be naphthalene (10 atoms, 2 fused benzenes)
        # Check aromaticity pattern
        aromatic_count = sum(
            1 for idx in core_atoms
            if mol.GetAtomWithIdx(idx).GetIsAromatic()
        )
        if aromatic_count == 10:
            return 'naphthalene'

    if len(core_atoms) == 14:
        # Could be anthracene or phenanthrene
        aromatic_count = sum(
            1 for idx in core_atoms
            if mol.GetAtomWithIdx(idx).GetIsAromatic()
        )
        if aromatic_count == 14:
            # Both anthracene and phenanthrene have 14 aromatic atoms
            # Would need more sophisticated analysis to distinguish
            return 'anthracene'  # Default to linear

    # For other sizes, return generic description
    return None


# ============================================================================
# Bridge Identification
# ============================================================================

def identify_bridges(mol, fused_core_atoms: Set[int]) -> List[Dict[str, Any]]:
    """
    Find bridges across the fused core.

    A bridge connects two atoms of the fused core via atoms
    NOT in the fused core.

    Args:
        mol: RDKit Mol object
        fused_core_atoms: Set of atom indices in the fused core

    Returns:
        List of bridge info dicts, each containing:
        - 'atoms': List of atom indices in the bridge (excluding endpoints)
        - 'length': Number of atoms in the bridge
        - 'start_locant': Locant of starting endpoint on fused core
        - 'end_locant': Locant of ending endpoint on fused core
        - 'element': Primary element type ('C', 'O', 'S', 'N')
        - 'heteroatom': Heteroatom type if not all carbon

    Examples:
        >>> mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene
        >>> bridges = identify_bridges(mol, set(range(10)))
        >>> len(bridges)
        0  # No bridges in pure naphthalene
    """
    if mol is None or not fused_core_atoms:
        return []

    bridges = []

    # Get all ring atoms for context
    ri = mol.GetRingInfo()
    all_ring_atoms = set()
    for ring in ri.AtomRings():
        all_ring_atoms.update(ring)

    # Strategy 1: Find non-ring atoms that connect two fused core atoms
    non_ring_atoms = set(range(mol.GetNumAtoms())) - all_ring_atoms

    for bridge_atom in non_ring_atoms:
        atom = mol.GetAtomWithIdx(bridge_atom)
        # Find fused core atoms this connects to
        core_connections = []
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in fused_core_atoms:
                core_connections.append(nbr_idx)

        if len(core_connections) >= 2:
            # This atom bridges two core atoms
            start = min(core_connections)
            end = max(core_connections)
            element = atom.GetSymbol()

            bridge_info = {
                'atoms': [bridge_atom],
                'length': 1,
                'start_locant': start,
                'end_locant': end,
                'element': element,
                'heteroatom': element if element != 'C' else None,
            }
            bridges.append(bridge_info)

    # Strategy 2: Find ring atoms outside fused core that form bridges
    for ring in ri.AtomRings():
        ring_set = set(ring)
        if ring_set.issubset(fused_core_atoms):
            continue  # Entirely in fused core

        # Check for bridging pattern
        core_in_ring = ring_set & fused_core_atoms
        non_core_in_ring = ring_set - fused_core_atoms

        if len(core_in_ring) >= 2 and len(non_core_in_ring) >= 1:
            # Find the endpoints in core and bridge atoms
            endpoints = list(core_in_ring)
            bridge_atoms_list = list(non_core_in_ring)

            # Simple case: 2 endpoints connected by non-core atoms
            if len(endpoints) == 2:
                start, end = min(endpoints), max(endpoints)
                element = 'C'  # Default

                # Check element composition
                for ba in bridge_atoms_list:
                    sym = mol.GetAtomWithIdx(ba).GetSymbol()
                    if sym != 'C':
                        element = sym
                        break

                bridge_info = {
                    'atoms': bridge_atoms_list,
                    'length': len(bridge_atoms_list),
                    'start_locant': start,
                    'end_locant': end,
                    'element': element,
                    'heteroatom': element if element != 'C' else None,
                }
                bridges.append(bridge_info)

    return bridges


# ============================================================================
# Bridge Prefix Generation
# ============================================================================

def get_bridge_prefix(bridge_info: Dict[str, Any]) -> str:
    """
    Get the IUPAC bridge prefix for a bridge.

    Carbon bridges: methano (1), ethano (2), propano (3), etc.
    Heteroatom bridges: epoxy (O), epithio (S), epimino (NH)

    Args:
        bridge_info: Dict with 'length', 'element', and optionally 'heteroatom'

    Returns:
        Bridge prefix string

    Examples:
        >>> get_bridge_prefix({'length': 1, 'element': 'C', 'atoms': [1]})
        'methano'
        >>> get_bridge_prefix({'length': 1, 'element': 'O', 'heteroatom': 'O'})
        'epoxy'
    """
    length = bridge_info.get('length', 0)
    element = bridge_info.get('element', 'C')
    heteroatom = bridge_info.get('heteroatom')

    # Check for heteroatom bridge first
    if heteroatom or element in ('O', 'S', 'N'):
        het = heteroatom or element
        if het in HETEROATOM_BRIDGE_PREFIXES:
            return HETEROATOM_BRIDGE_PREFIXES[het]
        # Fallback to length-based lookup
        key = (length, het)
        if key in BRIDGE_PREFIXES:
            return BRIDGE_PREFIXES[key]

    # Carbon bridge by length
    key = (length, 'C')
    if key in BRIDGE_PREFIXES:
        return BRIDGE_PREFIXES[key]

    # Fallback for longer carbon bridges
    carbon_prefix_base = {
        1: 'meth', 2: 'eth', 3: 'prop', 4: 'but',
        5: 'pent', 6: 'hex', 7: 'hept', 8: 'oct',
        9: 'non', 10: 'dec'
    }
    if length in carbon_prefix_base:
        return f"{carbon_prefix_base[length]}ano"

    return f"{length}carbano"  # Generic fallback


# ============================================================================
# Name Assembly
# ============================================================================

def name_bridged_fused_system(mol):
    """
    Generate the IUPAC name for a bridged fused system.

    Name format (FR-8): [locants]-[bridge_prefix][fused_parent_name]
    Examples:
    - 1,4-methanonaphthalene
    - 1,4:5,8-dimethanonaphthalene

    For multiple identical bridges, use multiplicative prefix (di-, tri-).
    For different bridges, alphabetize.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (name, ring_atoms, atom_to_locant, substituents_included)
        where substituents_included is False (bridged-fused handler does not
        discover substituents), or None if not a bridged fused system.

    Examples:
        >>> mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene
        >>> name_bridged_fused_system(mol)
        None  # Not bridged fused
    """
    if mol is None:
        return None

    # Check if this is a bridged fused system
    if not detect_bridged_fused(mol):
        return None

    # Identify the fused core
    core_info = identify_fused_core(mol)
    if core_info is None:
        return None

    core_atoms = core_info['core_atoms']
    core_name = core_info.get('core_name')
    core_numbering = core_info.get('core_numbering', {})

    # Identify bridges
    bridges = identify_bridges(mol, core_atoms)
    if not bridges:
        return None  # Should have bridges if detect_bridged_fused returned True

    # Collect ALL ring system atoms (core + bridge atoms) for substituent discovery
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for r in ri.AtomRings():
        ring_atoms.update(r)

    # Generate parent name
    if core_name:
        parent_name = core_name
    else:
        # Generate systematic name for fused core
        parent_name = f"fused[{len(core_atoms)}]system"

    # Group bridges by prefix for multiplicative naming
    bridge_groups = defaultdict(list)
    for bridge in bridges:
        prefix = get_bridge_prefix(bridge)
        bridge_groups[prefix].append(bridge)

    # Build name parts
    name_parts = []

    for prefix in sorted(bridge_groups.keys()):  # Alphabetize
        bridge_list = bridge_groups[prefix]
        count = len(bridge_list)

        # Collect locants for all bridges with this prefix
        locant_pairs = []
        for bridge in bridge_list:
            start = bridge.get('start_locant', 0)
            end = bridge.get('end_locant', 0)
            # Convert to IUPAC locants using core numbering
            start_loc = core_numbering.get(start, start + 1)
            end_loc = core_numbering.get(end, end + 1)
            locant_pairs.append((min(start_loc, end_loc), max(start_loc, end_loc)))

        # Sort locant pairs
        locant_pairs.sort()

        # Format locants
        if count == 1:
            loc_str = f"{locant_pairs[0][0]},{locant_pairs[0][1]}"
            bridge_str = f"{loc_str}-{prefix}"
        else:
            # Multiple bridges with same prefix
            loc_strs = [f"{p[0]},{p[1]}" for p in locant_pairs]
            multiplier = MULTIPLIERS.get(count, str(count))
            bridge_str = f"{':'.join(loc_strs)}-{multiplier}{prefix}"

        name_parts.append(bridge_str)

    # Combine parts
    if name_parts:
        prefix_str = '-'.join(name_parts)
        name = f"{prefix_str}{parent_name}"
    else:
        name = parent_name

    return (name, ring_atoms, core_numbering, False)


# ============================================================================
# Utility Functions
# ============================================================================

def is_bridged_fused(mol) -> bool:
    """
    Alias for detect_bridged_fused for API consistency.

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule is a bridged fused system
    """
    return detect_bridged_fused(mol)


# ============================================================================
# Bridged-fused PIN constructor (v22 Phase G1, DD7 COV-01)
# ============================================================================
#
# A bridged fused ring system (P-25.4.1.1) = a recognised fused parent (the
# "main ring system", e.g. naphthalene) + one or more bridges across it.
# The G0 fail-closed safety (DD7 S1) currently refuses these (von Baeyer would
# drop the benzo aromaticity). This constructor names the dominant, well-defined
# sub-class CORRECTLY and returns None (-> caller stays fail-closed) for anything
# outside it, so we never emit a wrong bridged-fused name.
#
# Handled class (this phase):
#   * a SINGLE divalent bridge (1-2 skeletal atoms; C or a single O/S/NH),
#   * across a NAPHTHALENE residual (10 C, two ortho-fused 6-rings, one benzo),
#   * with the parent ring system carrying NO substituents (bare ring systems).
# Anything else (anthracene+ residual, multi-bridge, polyvalent/composite/cyclic
# bridge, substituted, heteroaromatic parent) -> None -> G0 fail-closed (G1b/G2+).


def name_bridged_fused_pin(mol):
    """Name a bridged-fused ring system by the P-25.4 cascade (DD7 COV-01).

    Returns the standard complex-ring tuple ``(name, ring_atoms, atom_to_locant,
    substituents_included)`` for the handled class, else ``None`` (the caller then
    falls through to G0 fail-closed). The name is assembled as:
    bridge + hydro prefixes cited TOGETHER in alphanumerical order, ignoring
    multiplying prefixes ('epoxy' < 'ethano' < 'hydro' < 'methano'), each with its
    own locant set, then the parent (Blue Book P-25.4.3.4 / P-31.1.4.3.4):
    '1,4-dihydro-1,4-methanonaphthalene', '1,4-epoxy-1,4-dihydronaphthalene',
    '1,4-ethano-1,2,3,4-tetrahydronaphthalene'.
    """
    if mol is None:
        return None

    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    # Need fused parent (>=2 rings) + at least one extra ring created by a bridge.
    if len(rings) < 3:
        return None
    all_ring_atoms: Set[int] = set().union(*rings)

    # Bare ring systems only: a substituent would be dropped (a wrong name), so
    # fail closed if any heavy atom is outside the ring system. (Substituted
    # bridged-fused is a documented G1 follow-on.)
    heavy = {a.GetIdx() for a in mol.GetAtoms()}
    if heavy != all_ring_atoms:
        return None

    aromatic = {i for i in all_ring_atoms if mol.GetAtomWithIdx(i).GetIsAromatic()}
    if not aromatic:
        return None  # the fused parent must be (at least partly) aromatic (benzo)

    # Enumerate ALL candidate bridge excisions: connected subsets of NON-aromatic
    # ring atoms whose excision leaves a clean naphthalene residual ("clean" = the
    # residual ring atoms plus the bridge atoms PARTITION every original ring atom,
    # so no leftover dangling atom is mis-read as a phantom substituent — the size-1
    # false positive on a 2-atom ethano bridge). Name each candidate and accept ONLY
    # if every valid excision yields the SAME name; otherwise the choice would be
    # SMILES-spelling dependent (atom-index order), so fail closed rather than risk
    # an order-dependent wrong name.
    non_aromatic = sorted(all_ring_atoms - aromatic)
    results = []
    for k in (1, 2, 3):
        for combo in combinations(non_aromatic, k):
            bridge = set(combo)
            if not _bridge_is_connected(mol, bridge):
                continue
            residual = _excise_to_naphthalene_residual(mol, bridge, all_ring_atoms)
            if residual is None:
                continue
            res = _name_bridged_fused_excision(mol, bridge, residual, all_ring_atoms)
            if res is not None:
                results.append(res)

    distinct = {r[0] for r in results}
    if len(distinct) == 1:
        return results[0]
    if results:
        return None  # ambiguous (>1 distinct) name — fail closed

    # Wave-2 completion (P-25.4.3.4.1): the MANCUDE bridged classes — the
    # residual keeps its full aromatic system and the bridgeheads stay sp2
    # (0 H, part of a ring double bond), so there is NO hydro prefix:
    # 1,4-epoxynaphthalene / 1,4-ethanonaphthalene / 9,10-ethanoanthracene /
    # 1,4-ethano-5,8-methanoanthracene (all BB verbatim). The dihydro path
    # above cannot see these (its bridge candidates are non-aromatic atoms
    # with sp3 bridgeheads); this path fires only when it found nothing.
    return _try_mancude_bridged(mol, all_ring_atoms)


def _name_bridged_fused_excision(mol, bridge_atoms: Set[int], residual, all_ring_atoms: Set[int]):
    """Build the P-25.4 name for one (bridge, residual) excision — naphthalene,
    anthracene, or acridine residual — or None if the excision is not an
    in-scope, correctly-nameable bridged-fused system. Every guard here is a
    no-wrong-name guard: a None return cascades to G0 fail-closed."""
    res_mol, orig_to_res, _res_ring_atoms, base_name, ring_map = residual

    # Bridgeheads = original-mol neighbours of bridge atoms that are not themselves
    # bridge atoms. A single divalent bridge has exactly two.
    bridgeheads: Set[int] = set()
    for b in bridge_atoms:
        for nb in mol.GetAtomWithIdx(b).GetNeighbors():
            if nb.GetIdx() not in bridge_atoms:
                bridgeheads.add(nb.GetIdx())
    # P-25.4.1.7/.2.2.1/.2.2.2 fail-closed backstop: a POLYVALENT (tripodal,
    # 3+-attachment) bridge has >2 bridgeheads and no OPSIN-2.9-parseable oracle
    # (metheno==methano for the monocyclic case; [1,1,2]triyl forms unparseable).
    # Declining here guarantees a divalent bridge name is never emitted for a
    # trivalent attachment — documented follow-up when a parseable oracle exists.
    if len(bridgeheads) != 2:
        return None
    if any(bh not in orig_to_res for bh in bridgeheads):
        return None

    # A P-25.4 bridge spans NON-adjacent atoms whose attachment positions are
    # SATURATED. If the two bridgeheads are directly bonded, or either remains
    # aromatic, this is an ORTHO-FUSED ring (fusion nomenclature, e.g.
    # cyclopropa[b]naphthalene), NOT a bridge — fail closed (a 2,3-methano "bridge"
    # across the adjacent aromatic 2,3-bond is a non-PIN mis-description).
    bh = list(bridgeheads)
    if mol.GetBondBetweenAtoms(bh[0], bh[1]) is not None:
        return None
    if any(mol.GetAtomWithIdx(b).GetIsAromatic() for b in bridgeheads):
        return None

    # P-25.4.2.1.2/.1.3 CYCLIC (ring) bridge guard. A ring bridge (the bridge
    # atoms close a ring with the two bridgeheads: [1,2]benzeno, [1,2]epicyclopenta)
    # has its own free-valence-bracket grammar and OPSIN 2.9 cannot round-trip the
    # BB PIN (9,10-[1,2]benzenoanthracene) — an INTERNAL-ORACLE-only class. The
    # recognizer below identifies the two named ring bridges; the full residual +
    # bracket assembly is a documented follow-up, so we fail closed (return None)
    # rather than emit an unverifiable name for anything not already handled. A
    # heterocyclic ring bridge (P-25.4.2.1.5) returns None here too (all-carbon
    # check), which is the correct fail-closed for its no-OPSIN-oracle class.
    if _cyclic_bridge_prefix(mol, bridge_atoms, bridgeheads) is not None:
        return None

    # Bridge composition: only an all-carbon bridge (methano/ethano/propano) or a
    # SINGLE-atom heteroatom bridge (epoxy/epithio/epimino) is named correctly here.
    # A multi-atom or multi-heteroatom bridge (epidioxy -O-O-, composite -CH2-O-,
    # -CH2-O-CH2-) needs the composite-bridge grammar (P-25.4.1.5); get_bridge_prefix's
    # length-blind heteroatom shortcut would otherwise silently DROP atoms — fail closed.
    elements = [mol.GetAtomWithIdx(b).GetSymbol() for b in bridge_atoms]
    n_hetero = sum(1 for e in elements if e != 'C')
    if n_hetero > 1 or (n_hetero == 1 and len(bridge_atoms) != 1):
        return None

    # Bridge unsaturation (Wave-2 completion C, P-25.4.2.1.1): a single inner
    # C=C in a 2-carbon all-C bridge is the 'etheno' bridge (9,10-dihydro-
    # 9,10-ethenoanthracene = dibenzobarrelene). Any bond from a bridge atom
    # to a BRIDGEHEAD must stay single, and any other unsaturation pattern
    # (3-atom bridges needing prop[x]eno bracket grammar, cumulated,
    # hetero-unsaturated) fails closed — the length-keyed prefix table would
    # otherwise mis-name it 'ethano'/'propano'.
    n_inner_double = 0
    for b in bridge_atoms:
        for bond in mol.GetAtomWithIdx(b).GetBonds():
            other = bond.GetOtherAtomIdx(b)
            if bond.GetBondType() == Chem.BondType.SINGLE:
                continue
            if other in bridgeheads:
                return None
            if other in bridge_atoms:
                n_inner_double += 1  # counted twice (once per endpoint)
    n_inner_double //= 2
    if n_inner_double > 0 and not (
            n_inner_double == 1 and len(bridge_atoms) == 2 and n_hetero == 0):
        return None

    # Number the bridge attachment pair on the residual's fixed numbering.
    bh_res = {orig_to_res[x] for x in bridgeheads}
    if base_name == 'naphthalene':
        numbering = _number_naphthalene_bridged_ring(res_mol, bh_res)
        if numbering is None:
            return None  # bridge not across a numberable (alpha) position pair
        bridge_locants = sorted(numbering[orig_to_res[x]] for x in bridgeheads)
        hydro_locants = sorted(
            loc for res_idx, loc in numbering.items()
            if res_mol.GetAtomWithIdx(res_idx).GetHybridization()
            == Chem.HybridizationType.SP3
        )
    elif bh_res == ring_map['meso']:
        # Meso pair of anthracene/acridine: fixed retained locants 9,10
        # (acridine: N is 10 — verified a bridgehead by the residual
        # detector via n_meso ∈ meso).
        bridge_locants = [9, 10]
        sp3 = {i for i in _res_ring_atoms
               if res_mol.GetAtomWithIdx(i).GetHybridization()
               == Chem.HybridizationType.SP3}
        if base_name == 'acridine':
            # P-25.4.3.4.1: bridging N10 forbids the quinoid mancude form,
            # so the compound carries INDICATED hydrogen at C9, not hydro.
            # Exactly the C-bridgehead may be sp3; H counts are checked on
            # the ORIGINAL molecule (the residual's cut bridgeheads gain H).
            n_idx = ring_map['n_meso']
            c_bh = next(iter(bh_res - {n_idx}), None)
            if c_bh is None or sp3 - {c_bh}:
                return None
            orig_n = next(o for o, r in orig_to_res.items() if r == n_idx)
            orig_c = next(o for o, r in orig_to_res.items() if r == c_bh)
            if (mol.GetAtomWithIdx(orig_n).GetTotalNumHs() != 0
                    or mol.GetAtomWithIdx(orig_c).GetTotalNumHs() != 1):
                return None
            hydro_locants = []  # indicated-H handled at assembly
        else:
            # X,Y-dihydro-X,Y-(bridge)anthracene: the sp3 set must be
            # EXACTLY the bridgeheads (wider hydro sets not built).
            if sp3 != bh_res:
                return None
            hydro_locants = [9, 10]
    elif base_name == 'anthracene':
        # Terminal-ring alpha,alpha pair -> 1,4 (host-ring adjacency check
        # mirrors the mancude helper).
        locs = _mancude_bridge_locants(res_mol, bh_res, 'anthracene', ring_map)
        if locs is None:
            return None
        bridge_locants = locs
        sp3 = {i for i in _res_ring_atoms
               if res_mol.GetAtomWithIdx(i).GetHybridization()
               == Chem.HybridizationType.SP3}
        if sp3 != bh_res:
            return None
        hydro_locants = [1, 4]
    else:
        return None

    # Bridge prefix (methano/ethano/propano | epoxy/epithio/epimino | etheno).
    hetero = next((e for e in elements if e != 'C'), None)
    if n_inner_double == 1:
        bridge_prefix = 'etheno'
    else:
        bridge_prefix = get_bridge_prefix({
            'length': len(bridge_atoms),
            'element': hetero or 'C',
            'heteroatom': hetero,
        })
    if not bridge_prefix:
        return None

    # Assembly (P-31.1.4.2.4, Wave-2 completion C ordering fix): hydro
    # prefixes sit between detachable and NONDETACHABLE prefixes; the bridge
    # prefix is nondetachable and abuts the parent — so hydro is cited
    # BEFORE the bridge in every BB example (1,4-dihydro-1,4-ethano-
    # anthracene BB:14399, octahydro-9,10-ethanoanthracene BB:19964). The
    # old alphanumeric mixing emitted '1,4-ethano-1,2,3,4-tetrahydro-'
    # (BB-nonconformant; OPSIN accepts both, so RT never caught it).
    parts = []
    if hydro_locants:
        hydro_prefix = _hydro_prefix(len(hydro_locants))
        if hydro_prefix is None:
            return None
        parts.append(f"{_format_locants(hydro_locants)}-{hydro_prefix}")
    parts.append(f"{_format_locants(bridge_locants)}-{bridge_prefix}")
    name = '-'.join(parts) + base_name
    if base_name == 'acridine':
        name = f"9H-{name}"

    # atom_to_locant is a ring-MEMBERSHIP map only (not real IUPAC locants); harmless
    # because substituents_included=True makes the caller skip substituent enrichment.
    atom_to_locant = {i: i + 1 for i in sorted(all_ring_atoms)}
    return (name, set(all_ring_atoms), atom_to_locant, True)


def _try_mancude_bridged(mol, all_ring_atoms: Set[int]):
    """P-25.4.3.4.1 mancude bridged-fused constructor (Wave-2 completion).

    Handles bare ring systems where excising every saturated (all-single-bond)
    ring atom leaves a fully AROMATIC naphthalene or anthracene residual and
    every bridgehead keeps its ring double bond (0 H) — so the name carries no
    hydro prefix: '1,4-epoxynaphthalene', '1,4-ethano-5,8-methanoanthracene'.

    Deterministic (no subset enumeration): for the residual to be mancude,
    the removed set must be EXACTLY the saturated atoms; each connected
    component of that set is one bridge. Every guard is a no-wrong-name
    guard — any deviation returns None and cascades to G0 fail-closed.
    Analysis runs on a kekulized copy because RDKit marks the epoxy oxygen
    of 1,4-epoxynaphthalene aromatic (extended-ring perception), which
    hides it from aromatic-flag-based candidate selection.
    """
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return None

    # Candidate bridge atoms: every incident bond single on the kekulized
    # copy. In a mancude system every non-bridge ring atom carries a double
    # bond, so this is exactly the removed set for SATURATED bridges.
    def _all_single(idx: int) -> bool:
        return all(b.GetBondType() == Chem.BondType.SINGLE
                   for b in kek.GetAtomWithIdx(idx).GetBonds())

    # P-25.4.2.1.1 (Wave-2 P5): an UNSATURATED acyclic bridge (-CH=CH-, etc.)
    # carries an internal C=C, so its atoms are not all-single and are invisible
    # to the saturated seed above. Grow bridge components on the kekulized copy
    # by absorbing double-bonded ring neighbours, then keep only components whose
    # every EXTERNAL bond (to an atom outside the component) is single. That is
    # the exact bridge criterion: a genuine bridge attaches to the mancude parent
    # via single bonds; a parent ring atom (e.g. the naphthalene 1,2 edge) leaves
    # via a ring double bond and is correctly rejected.
    def _external_bonds_all_single(comp: Set[int]) -> bool:
        for a in comp:
            for b in kek.GetAtomWithIdx(a).GetBonds():
                o = b.GetOtherAtomIdx(a)
                if o not in comp and b.GetBondType() != Chem.BondType.SINGLE:
                    return False
        return True

    def _external_bonds_all_single(comp: Set[int]) -> bool:
        for a in comp:
            for b in kek.GetAtomWithIdx(a).GetBonds():
                o = b.GetOtherAtomIdx(a)
                if o not in comp and b.GetBondType() != Chem.BondType.SINGLE:
                    return False
        return True

    saturated = {i for i in all_ring_atoms if _all_single(i)}

    # Discover bridge components. A SATURATED bridge is a connected component of
    # the all-single atoms (the fast, common path — 1,4-methano/ethano/epoxy).
    # An UNSATURATED acyclic bridge (P-25.4.2.1.1: -CH=CH-, -CH=CH-CH=CH-) has
    # NO all-single atom, so it is discovered by enumerating short connected
    # subsets of the ring atoms whose EXTERNAL bonds are all single (the exact
    # bridge attachment criterion — a parent aromatic edge like naphthalene's
    # 1,2 leaves via a ring double bond and is rejected). Both classes reduce to
    # the same downstream validation.
    components: List[Set[int]] = []
    if saturated:
        remaining = set(saturated)
        while remaining:
            start = min(remaining)
            comp = {start}
            stack = [start]
            while stack:
                cur = stack.pop()
                for nb in kek.GetAtomWithIdx(cur).GetNeighbors():
                    j = nb.GetIdx()
                    if j in remaining and j not in comp:
                        comp.add(j)
                        stack.append(j)
            components.append(comp)
            remaining -= comp
    else:
        # Pure-unsaturated-bridge case: enumerate connected subsets (size 2..4)
        # of ring atoms whose external bonds are all single; each such subset is
        # a candidate single-bridge excision. Try each independently through the
        # shared validator and require a UNIQUE resulting name (determinism guard
        # — atom-index / SMILES-spelling order must not change the answer).
        cand_sets: List[frozenset] = []
        ring_list = sorted(all_ring_atoms)
        for k in (2, 3, 4):
            for combo in combinations(ring_list, k):
                s = set(combo)
                if not _bridge_is_connected(kek, s):
                    continue
                if not _external_bonds_all_single(s):
                    continue
                cand_sets.append(frozenset(s))
        if not cand_sets:
            return None
        by_name: Dict[str, Any] = {}
        for cs in cand_sets:
            res = _validate_and_name_mancude(
                mol, [set(cs)], all_ring_atoms, _all_single, kek)
            if res is not None:
                by_name[res[0]] = res
        if len(by_name) == 1:
            return next(iter(by_name.values()))
        return None  # 0 or ambiguous (>1 distinct) — fail closed

    if not components:
        return None
    if len(components) > 2 or any(len(c) > 4 for c in components):
        return None
    return _validate_and_name_mancude(mol, components, all_ring_atoms,
                                      _all_single, kek)


def _validate_and_name_mancude(mol, components: List[Set[int]],
                               all_ring_atoms: Set[int], _all_single, kek):
    """Shared validator/assembler for the mancude bridged path: given the bridge
    *components* (each a connected set of removed ring atoms), excise them, verify
    a clean naphthalene/anthracene residual, validate every bridge, and assemble
    the P-25.4.3.4.1 name. Returns the complex-ring tuple or None (fail closed).
    """
    removed = set().union(*components)
    # Excise ALL bridges at once; the residual must be a clean fully-aromatic
    # naphthalene or anthracene whose ring atoms partition the original ring
    # atoms with the bridges (same no-phantom-substituent rule as the
    # dihydro path).
    residual = _excise_to_mancude_residual(mol, removed, all_ring_atoms)
    if residual is None:
        return None
    res_mol, orig_to_res, base_name, ring_map = residual

    # Validate each bridge and collect (prefix, sorted locant pair).
    entries: List[Tuple[str, List[int]]] = []
    for comp in components:
        bridgeheads: Set[int] = set()
        for b in comp:
            for nb in mol.GetAtomWithIdx(b).GetNeighbors():
                if nb.GetIdx() not in comp:
                    bridgeheads.add(nb.GetIdx())
        if len(bridgeheads) != 2:
            return None
        bh = sorted(bridgeheads)
        if mol.GetBondBetweenAtoms(bh[0], bh[1]) is not None:
            return None  # adjacent attachment = ortho-fusion, not a bridge
        if any(bh_i not in orig_to_res for bh_i in bh):
            return None
        # Mancude form: the bridgehead keeps its ring double bond -> 0 H.
        if any(mol.GetAtomWithIdx(b).GetTotalNumHs() != 0 for b in bh):
            return None
        # P-25.4.2.1.1: classify saturated vs unsaturated-acyclic bridge. A
        # component with an internal double bond (not all-single on the
        # kekulized copy) is an etheno/buta[1,3]dieno bridge; the length-keyed
        # 'ano' table would silently drop the double bond, so route it through
        # the unsaturated recognizer (which fails closed on hetero/branched/
        # triple/non-tabulated).
        comp_all_single = all(_all_single(a) for a in comp)
        if not comp_all_single:
            # Bond-order analysis on the KEKULIZED copy: RDKit's extended
            # aromaticity marks a -CH=CH- bridge aromatic, so the double bond is
            # only visible after kekulization.
            prefix = _unsaturated_bridge_prefix(kek, comp, bridgeheads)
            if prefix is None:
                return None
        else:
            # P-15.3.1.2.2.1 composite bridge (-O-CH2-, -CH2-O-CH2-): a
            # multi-atom bridge with a heteroatom that the single-simple-bridge
            # composition guard would reject. Recognize it FIRST; composite
            # prefixes are parenthesized (P-25.4.2.3.2).
            comp_bridge = _composite_bridge_prefix(mol, comp, bridgeheads)
            if comp_bridge is not None:
                prefix = f"({comp_bridge[0]})"
            else:
                # Composition: all-C saturated bridge or a single O (epoxy) —
                # same scope rule as the dihydro path.
                elements = [mol.GetAtomWithIdx(b).GetSymbol() for b in comp]
                n_hetero = sum(1 for e in elements if e != 'C')
                if n_hetero > 1 or (n_hetero == 1 and (len(comp) != 1
                                                       or elements[0] != 'O')):
                    return None
                hetero = next((e for e in elements if e != 'C'), None)
                prefix = get_bridge_prefix({
                    'length': len(comp), 'element': hetero or 'C',
                    'heteroatom': hetero,
                })
        if not prefix:
            return None
        locants = _mancude_bridge_locants(
            res_mol, {orig_to_res[x] for x in bh}, base_name, ring_map)
        if locants is None:
            return None
        entries.append((prefix, locants))

    if len(entries) == 2:
        if base_name != 'anthracene':
            return None
        # Both bridges must sit on DISTINCT terminal rings at the alpha,alpha
        # positions ((1,4)-type); _mancude_bridge_locants returned the
        # ring-local pattern [1, 4] for each. P-25.4.3.3(b): the first-cited
        # (alphabetical) bridge takes the lower pair (1,4), the other (5,8) —
        # legitimate because bare anthracene's terminal rings are equivalent.
        if any(loc != [1, 4] for _, loc in entries):
            return None
        # Each component's bridgeheads must belong to different terminal rings.
        ring_ids = []
        for comp in components:
            bh_rings = set()
            for b in comp:
                for nb in mol.GetAtomWithIdx(b).GetNeighbors():
                    if nb.GetIdx() in orig_to_res:
                        t = ring_map['terminal_ring_of'].get(
                            orig_to_res[nb.GetIdx()])
                        if t is not None:
                            bh_rings.add(t)
            if len(bh_rings) != 1:
                return None
            ring_ids.append(bh_rings.pop())
        if ring_ids[0] == ring_ids[1]:
            return None
        entries.sort(key=lambda e: e[0])
        if entries[0][0] == entries[1][0]:
            # identical bridges: 1,4:5,8-di<prefix> (P-25.4.3.2.1)
            name = (f"1,4:5,8-di{entries[0][0]}{base_name}")
        else:
            name = (f"1,4-{entries[0][0]}-5,8-{entries[1][0]}{base_name}")
    else:
        prefix, locants = entries[0]
        name = f"{_format_locants(locants)}-{prefix}{base_name}"

    atom_to_locant = {i: i + 1 for i in sorted(all_ring_atoms)}
    return (name, set(all_ring_atoms), atom_to_locant, True)


def has_aromatic_mancude_bridge(mol) -> bool:
    """True when a FULLY-AROMATIC-perceived ring system is actually a mancude
    bridged-fused hydrocarbon that ``name_bridged_fused_pin`` can name
    (P-25.4.2.1.1 etheno/buta[1,3]dieno on naphthalene: RDKit's extended
    aromaticity marks the -CH=CH- bridge aromatic, so the von-Baeyer gate's
    all-aromatic skip would otherwise strand it). Used only to EXEMPT this class
    from that skip; returns False for a plain fused PAH (no bridge) so the
    retained-name / fusion path keeps it."""
    if mol is None:
        return False
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    if len(rings) < 3:
        return False
    all_ring_atoms: Set[int] = set().union(*rings)
    heavy = {a.GetIdx() for a in mol.GetAtoms()}
    if heavy != all_ring_atoms:
        return False  # bare ring systems only (substituted -> other paths)
    return _try_mancude_bridged(mol, all_ring_atoms) is not None


def has_aromatic_chalcogen_bridge(mol) -> bool:
    """True when a divalent O/S ring atom is a genuine BRIDGE that RDKit's
    extended aromaticity model hides (Wave-2 completion, P-25.4.3.3(a)):
    its two neighbours share a ring that does NOT contain the chalcogen, at
    NON-ADJACENT positions of that ring (1,4-epoxynaphthalene). A fusion
    chalcogen (dibenzofuran O) shares only its own ring with its neighbours
    and returns False. Used to exempt this class from the all-aromatic
    routing skips that would otherwise strand it on a partial-parent namer.
    """
    if mol is None:
        return False
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for r in ri.AtomRings():
        ring_atoms.update(r)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ('O', 'S'):
            continue
        if atom.GetDegree() != 2 or atom.GetFormalCharge() != 0:
            continue
        idx = atom.GetIdx()
        if idx not in ring_atoms:
            continue
        n1, n2 = (n.GetIdx() for n in atom.GetNeighbors())
        if mol.GetBondBetweenAtoms(n1, n2) is not None:
            continue
        # Bridge test robust to SSSR ring choice: delete the chalcogen; if
        # its two neighbours still share a ring of the residual, the atom
        # closed a SHORTCUT across that ring (a bridge). A fusion chalcogen's
        # neighbours fall into separate rings (dibenzofuran -> biphenyl).
        rw = Chem.RWMol(mol)
        for a in rw.GetAtoms():
            a.SetIntProp('__bf_chal_orig', a.GetIdx())
        rw.RemoveAtom(idx)
        res = rw.GetMol()
        try:
            Chem.SanitizeMol(res)
        except Exception:
            continue
        orig_of = {i: res.GetAtomWithIdx(i).GetIntProp('__bf_chal_orig')
                   for i in range(res.GetNumAtoms())}
        for r in res.GetRingInfo().AtomRings():
            r_orig = {orig_of[i] for i in r}
            if n1 in r_orig and n2 in r_orig:
                return True
    return False


def _excise_to_mancude_residual(mol, removed: Set[int], all_ring_atoms: Set[int]):
    """Excise *removed*; if the residual is a clean FULLY-AROMATIC naphthalene
    (2 ortho-fused 6-rings, 10 C) or linear anthracene (3 six-rings, 14 C),
    return ``(res_mol, orig_to_res, base_name, ring_map)``; else None.
    ``ring_map`` carries the ring topology used for locant assignment."""
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetIntProp('__bf_orig', a.GetIdx())
    for idx in sorted(removed, reverse=True):
        rw.RemoveAtom(idx)
    res = rw.GetMol()
    try:
        Chem.SanitizeMol(res)
    except Exception:
        return None

    ri = res.GetRingInfo()
    res_rings = [set(r) for r in ri.AtomRings()]
    six = [r for r in res_rings if len(r) == 6]
    res_ring_atoms = set().union(*res_rings) if res_rings else set()
    if not all(res.GetAtomWithIdx(i).GetSymbol() == 'C'
               and res.GetAtomWithIdx(i).GetIsAromatic()
               for i in res_ring_atoms):
        return None  # not mancude (leftover sp3 / heteroatom in the parent)

    base_name = None
    ring_map: Dict[str, Any] = {}
    if (len(res_rings) == 2 and len(six) == 2
            and len(six[0] & six[1]) == 2 and len(res_ring_atoms) == 10):
        base_name = 'naphthalene'
        ring_map['rings'] = six
        ring_map['fusion'] = six[0] & six[1]
    elif (len(res_rings) == 3 and len(six) == 3
            and len(res_ring_atoms) == 14):
        # Linear anthracene: the middle ring shares 2 atoms with EACH
        # terminal ring; the terminal rings share none (angular phenanthrene
        # has terminal rings meeting the same criteria? no — in phenanthrene
        # every ring pair shares atoms only along the fusion chain, but the
        # two OUTER rings still share 0 atoms; distinguish by the fusion
        # atoms: anthracene's middle ring carries 4 fusion atoms in two
        # OPPOSITE (para-type) pairs, phenanthrene's in adjacent pairs —
        # check: the two fusion BONDS of the middle ring do not share atoms
        # AND are separated by 2 atoms on both sides (meso carbons).
        shared = [(i, j, six[i] & six[j])
                  for i in range(3) for j in range(i + 1, 3)]
        pairs = [(i, j, s) for i, j, s in shared if len(s) == 2]
        if len(pairs) != 2:
            return None
        mid_candidates = set([pairs[0][0], pairs[0][1]]) & set(
            [pairs[1][0], pairs[1][1]])
        if len(mid_candidates) != 1:
            return None
        mid = mid_candidates.pop()
        terminals = [k for k in range(3) if k != mid]
        fusion_atoms = (six[mid] & six[terminals[0]]) | (
            six[mid] & six[terminals[1]])
        meso = six[mid] - fusion_atoms
        if len(meso) != 2:
            return None
        # Linear check: the meso carbons are non-adjacent and each bonds to
        # one fusion atom of EACH terminal pair (true for anthracene; in
        # phenanthrene the middle ring has 2 non-fusion atoms ADJACENT to
        # each other).
        meso_l = sorted(meso)
        if res.GetBondBetweenAtoms(meso_l[0], meso_l[1]) is not None:
            return None
        base_name = 'anthracene'
        ring_map['rings'] = six
        ring_map['mid'] = mid
        ring_map['terminals'] = terminals
        ring_map['meso'] = meso
        ring_map['fusion'] = fusion_atoms
        ring_map['terminal_ring_of'] = {
            a: t for t in terminals for a in six[t] - fusion_atoms
        }
    else:
        return None

    orig_to_res = {
        res.GetAtomWithIdx(i).GetIntProp('__bf_orig'): i
        for i in range(res.GetNumAtoms())
    }
    res_ring_orig = {res.GetAtomWithIdx(i).GetIntProp('__bf_orig')
                     for i in res_ring_atoms}
    if res_ring_orig | removed != all_ring_atoms:
        return None  # unclean partition (phantom substituent)
    return (res, orig_to_res, base_name, ring_map)


def _mancude_bridge_locants(res_mol, bh_res: Set[int], base_name: str,
                            ring_map: Dict[str, Any]):
    """Locants for one bridge's attachment pair on the mancude residual.

    naphthalene: both bridgeheads must be the alpha,alpha pair (1,4) of ONE
    ring — non-fusion, each adjacent to a fusion atom, not adjacent to each
    other. (1,3/2,3/peri patterns fail closed: fusion nomenclature or the
    unbuilt P-25.4.3.3 locant cascade owns them.)
    anthracene: the meso pair -> [9, 10]; a terminal alpha,alpha pair ->
    [1, 4] (ring-local; the caller maps the second bridge to 5,8)."""
    if base_name == 'anthracene' and bh_res == ring_map['meso']:
        return [9, 10]

    rings = ring_map['rings']
    if base_name == 'anthracene':
        rings = [rings[t] for t in ring_map['terminals']]
    host = next((r for r in rings if bh_res <= r), None)
    if host is None:
        return None
    fusion_in_host = host & ring_map['fusion']
    non_fusion = host - ring_map['fusion']
    if not bh_res <= non_fusion:
        return None
    bh = sorted(bh_res)
    if res_mol.GetBondBetweenAtoms(bh[0], bh[1]) is not None:
        return None
    # alpha positions: adjacent to a fusion atom of the host ring.
    for b in bh:
        nbrs = {n.GetIdx() for n in res_mol.GetAtomWithIdx(b).GetNeighbors()}
        if not (nbrs & fusion_in_host):
            return None
    return [1, 4]


# Multiplying prefixes only matter for the hydro term, whose multiplier is always
# a true multiplying prefix; the bridge prefix in the handled class never carries
# one, so we key it as-is (avoids mis-stripping e.g. 'diazeno').
def _format_locants(locants: List[int]) -> str:
    return ','.join(str(x) for x in locants)


_HYDRO_PREFIXES = {
    2: 'dihydro', 4: 'tetrahydro', 6: 'hexahydro', 8: 'octahydro',
    10: 'decahydro', 12: 'dodecahydro',
}


def _hydro_prefix(n_sp3: int) -> Optional[str]:
    """Hydro prefix for *n_sp3* saturated ring positions (each adds one H)."""
    return _HYDRO_PREFIXES.get(n_sp3)


# Prefix table for acyclic unsaturated bridges (P-25.4.2.1.1). The double-bond
# locant is the bridge-INTERNAL numbering (prop[1]eno etc.), NOT the final
# ring-system locant. Keyed on (n_carbons, tuple(sorted internal db positions)).
_UNSATURATED_ACYCLIC_BRIDGE: Dict[Tuple[int, Tuple[int, ...]], str] = {
    (2, (1,)): 'etheno',            # -CH=CH-
    (3, (1,)): 'prop[1]eno',        # -CH=CH-CH2-
    (4, (1, 3)): 'buta[1,3]dieno',  # -CH=CH-CH=CH-
}


def _linear_bridge_order(mol, comp: Set[int], bridgeheads: Set[int]):
    """Return the bridge atoms as a path (list) whose ends each bond a distinct
    bridgehead, or None if *comp* is not a simple path between the two
    bridgeheads (branched / cyclic bridge)."""
    ends = [a for a in comp
            if any(nb.GetIdx() in bridgeheads
                   for nb in mol.GetAtomWithIdx(a).GetNeighbors())]
    if len(comp) == 1:
        return list(comp) if len(ends) == 1 else None
    if len(ends) != 2:
        return None
    start = ends[0]
    path, prev, cur = [start], None, start
    while len(path) < len(comp):
        nxt = None
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if j in comp and j != prev and j not in path:
                nxt = j
                break
        if nxt is None:
            return None
        path.append(nxt)
        prev, cur = cur, nxt
    return path if path[-1] in ends else None


# Composite two/three-atom bridge sequences (P-15.3.1.2.2.1 + P-25.4.2.3.1).
# Key = tuple of element symbols in atom order from one bridgehead to the other;
# value = the concatenated prefix (senior simple bridge first, 'epi' elided when
# not first). Heteroatom seniority O > S > Se > N (P-25.4.2.3.1).
_COMPOSITE_BRIDGE_SEQUENCES: Dict[Tuple[str, ...], str] = {
    ('O', 'C'): 'epoxymethano',            # -O-CH2-
    ('C', 'O', 'C'): 'methanooxymethano',  # -CH2-O-CH2-
}


def _composite_bridge_prefix(mol, bridge_atoms: Set[int], bridgeheads: Set[int]):
    """P-15.3.1.2.2.1/.2.2.4 + P-25.4.2.3.1: a composite (multi-simple-bridge)
    acyclic bridge. Order the bridge atoms into a chain between the two
    bridgeheads, look up the element sequence (tried both directions). Returns
    ('epoxymethano', sorted([bh_lo, bh_hi])) or None (fail closed). Only
    OPSIN-verifiable composite sequences are tabulated."""
    chain = _linear_bridge_order(mol, set(bridge_atoms), set(bridgeheads))
    if chain is None or len(chain) < 2:
        return None
    # composite bridges are all-single-bonded internally (no C=C/C#C here).
    for i in range(len(chain) - 1):
        b = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
        if b.GetBondType() != Chem.BondType.SINGLE:
            return None
    elems = tuple(mol.GetAtomWithIdx(a).GetSymbol() for a in chain)
    name = _COMPOSITE_BRIDGE_SEQUENCES.get(elems)
    if name is None:
        name = _COMPOSITE_BRIDGE_SEQUENCES.get(tuple(reversed(elems)))
    if name is None:
        return None
    return (name, sorted(bridgeheads))


def _cyclic_bridge_prefix(mol, bridge_atoms: Set[int], bridgeheads: Set[int]):
    """P-25.4.2.1.2/.1.3 divalent monocyclic hydrocarbon bridge prefix.
    A ring bridge is the bridge atoms forming, WITH the two bridgeheads, a single
    ring. Return '[1,2]benzeno' for a benzene ring bridge, '[1,2]epicyclopenta'
    for a cyclopentane ring bridge, else None (fail closed). All-carbon only — a
    heteroatom ring bridge (P-25.4.2.1.5 furano/epipyrrolo) returns None (no
    OPSIN 2.9 oracle for that class)."""
    if any(mol.GetAtomWithIdx(a).GetSymbol() != 'C' for a in bridge_atoms):
        return None
    ring_atoms = set(bridge_atoms) | set(bridgeheads)
    ri = mol.GetRingInfo()
    # the bridge + the two bridgeheads must be exactly one SSSR ring.
    if not any(set(r) == ring_atoms for r in ri.AtomRings()):
        return None
    size = len(ring_atoms)
    aromatic = all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in bridge_atoms)
    if size == 6 and aromatic:
        return '[1,2]benzeno'
    if size == 5 and not aromatic:
        return '[1,2]epicyclopenta'
    return None


def _unsaturated_bridge_prefix(mol, comp: Set[int], bridgeheads: Set[int]):
    """P-25.4.2.1.1 prefix for an all-carbon acyclic bridge that carries one or
    more internal C=C. Returns None (fail closed) for any hetero, branched,
    cumulated, triple-bond (P-31.1.4.3), or non-tabulated pattern — the
    length-keyed 'ano' table would otherwise mis-name it. Bonds from a bridge
    atom to a bridgehead MUST be single (a double bond to the aromatic ring is
    fusion, not a bridge)."""
    atoms = list(comp)
    if any(mol.GetAtomWithIdx(a).GetSymbol() != 'C' for a in atoms):
        return None
    # Order the bridge into a linear chain from one bridgehead-neighbour to the
    # other; refuse if it is not a simple path (branched/cyclic bridge).
    chain = _linear_bridge_order(mol, comp, bridgeheads)
    if chain is None:
        return None
    # P-31.1.4.3: a triple bond anywhere in the bridge has no verified
    # bridged-fused oracle — decline rather than mis-name it 'ano'/'eno'.
    for a in atoms:
        for bond in mol.GetAtomWithIdx(a).GetBonds():
            if bond.GetBondType() == Chem.BondType.TRIPLE:
                return None
    db_positions = []
    for i in range(len(chain) - 1):
        b = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
        if b.GetBondType() == Chem.BondType.DOUBLE:
            db_positions.append(i + 1)  # 1-based internal position
    # any double bond to a bridgehead => fusion, not a bridge
    for a in atoms:
        for bond in mol.GetAtomWithIdx(a).GetBonds():
            o = bond.GetOtherAtomIdx(a)
            if o in bridgeheads and bond.GetBondType() != Chem.BondType.SINGLE:
                return None
    key = (len(atoms), tuple(sorted(db_positions)))
    return _UNSATURATED_ACYCLIC_BRIDGE.get(key)


def _bridge_is_connected(mol, atoms: Set[int]) -> bool:
    """True if *atoms* form a connected subgraph (a single bridge unit)."""
    if not atoms:
        return False
    if len(atoms) == 1:
        return True
    seen = set()
    stack = [next(iter(atoms))]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            if nb.GetIdx() in atoms and nb.GetIdx() not in seen:
                stack.append(nb.GetIdx())
    return seen == atoms


def _excise_to_naphthalene_residual(mol, bridge_atoms: Set[int], all_ring_atoms: Set[int]):
    """Excise *bridge_atoms*; if the residual ring system is a clean naphthalene
    (2 ortho-fused 6-rings, 10 C), linear anthracene (3 six-rings, 14 C), or
    acridine (3 six-rings, 13 C + 1 N at a MESO position) whose atoms PARTITION
    the original ring atoms with the bridge (no dangling leftover), return
    ``(residual_mol, orig_to_res, residual_ring_atoms, base_name, ring_map)``;
    else None. ``ring_map`` carries the topology needed for locant assignment
    (empty for naphthalene; meso/terminals/fusion for the 3-ring bases).
    """
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetIntProp('__bf_orig', a.GetIdx())
    for idx in sorted(bridge_atoms, reverse=True):
        rw.RemoveAtom(idx)
    res = rw.GetMol()
    try:
        Chem.SanitizeMol(res)
    except Exception:
        return None

    ri = res.GetRingInfo()
    res_rings = [set(r) for r in ri.AtomRings()]
    six = [r for r in res_rings if len(r) == 6]
    res_ring_atoms = set().union(*res_rings) if res_rings else set()

    base_name = None
    ring_map: Dict[str, Any] = {}
    if (len(res_rings) == 2 and len(six) == 2
            and len(six[0] & six[1]) == 2
            and len(res_ring_atoms) == 10
            and all(res.GetAtomWithIdx(i).GetSymbol() == 'C'
                    for i in res_ring_atoms)):
        base_name = 'naphthalene'
    elif len(res_rings) == 3 and len(six) == 3 and len(res_ring_atoms) == 14:
        # Wave-2 completion C: linear 3-ring residual — anthracene (all-C) or
        # acridine (one N at a meso position). Terminal benzo rings must stay
        # fully aromatic (fail-closed on anything partly reduced there).
        syms = [res.GetAtomWithIdx(i).GetSymbol() for i in res_ring_atoms]
        n_n = syms.count('N')
        if n_n > 1 or any(s not in ('C', 'N') for s in syms):
            return None
        shared = [(i, j, six[i] & six[j])
                  for i in range(3) for j in range(i + 1, 3)]
        pairs = [(i, j, s) for i, j, s in shared if len(s) == 2]
        if len(pairs) != 2:
            return None
        mids = set([pairs[0][0], pairs[0][1]]) & set([pairs[1][0], pairs[1][1]])
        if len(mids) != 1:
            return None
        mid = mids.pop()
        terminals = [k for k in range(3) if k != mid]
        fusion = (six[mid] & six[terminals[0]]) | (six[mid] & six[terminals[1]])
        meso = six[mid] - fusion
        if len(meso) != 2:
            return None
        meso_l = sorted(meso)
        if res.GetBondBetweenAtoms(meso_l[0], meso_l[1]) is not None:
            return None  # angular (phenanthrene-type) — not built
        # Exactly ONE ring may carry the saturated (bridgehead) positions —
        # the middle ring for a 9,10-bridge, one terminal for a 1,4-bridge;
        # the other two rings must be fully aromatic (fail-closed).
        non_arom_rings = [k for k in range(3)
                          if not all(res.GetAtomWithIdx(i).GetIsAromatic()
                                     for i in six[k])]
        if len(non_arom_rings) > 1:
            return None
        if n_n == 1:
            n_idx = next(i for i in res_ring_atoms
                         if res.GetAtomWithIdx(i).GetSymbol() == 'N')
            if n_idx not in meso:
                return None  # N elsewhere = a different fused base — closed
            base_name = 'acridine'
            ring_map['n_meso'] = n_idx
        else:
            base_name = 'anthracene'
        ring_map['meso'] = meso
        ring_map['fusion'] = fusion
        ring_map['rings'] = six
        ring_map['terminals'] = terminals
    else:
        return None

    orig_to_res = {
        res.GetAtomWithIdx(i).GetIntProp('__bf_orig'): i
        for i in range(res.GetNumAtoms())
    }
    res_ring_orig = {res.GetAtomWithIdx(i).GetIntProp('__bf_orig') for i in res_ring_atoms}
    # Clean partition: every original ring atom is either in the residual ring
    # system or in the bridge (no atom orphaned into a phantom substituent).
    if res_ring_orig | bridge_atoms != all_ring_atoms:
        return None

    return (res, orig_to_res, res_ring_atoms, base_name, ring_map)


def _number_naphthalene_bridged_ring(res_mol, bridgeheads_res: Set[int]) -> Optional[Dict[int, int]]:
    """Number the bridged (non-benzo) ring of a naphthalene residual 1..4 around
    its four non-fusion atoms, choosing the direction that gives the bridgeheads
    the lowest locant set (P-25.4.4). Returns ``{res_idx: locant}`` for those four
    atoms, or None if the bridge is not across a numberable position pair.
    """
    ri = res_mol.GetRingInfo()
    res_rings = [set(r) for r in ri.AtomRings() if len(r) == 6]
    if len(res_rings) != 2:
        return None
    fusion = res_rings[0] & res_rings[1]
    if len(fusion) != 2:
        return None

    # Bridged ring = the 6-ring containing both bridgeheads.
    bridged_ring = next((r for r in res_rings if bridgeheads_res <= r), None)
    if bridged_ring is None:
        return None

    non_fusion = list(bridged_ring - fusion)  # 4 atoms
    if len(non_fusion) != 4:
        return None

    # The four non-fusion atoms form a path between the two fusion atoms (the
    # ring minus its fusion edge). Endpoints are adjacent to a fusion atom.
    nf_set = set(non_fusion)
    endpoints = [
        a for a in non_fusion
        if any(nb.GetIdx() in fusion for nb in res_mol.GetAtomWithIdx(a).GetNeighbors())
    ]
    if len(endpoints) != 2:
        return None

    # Walk the path from one endpoint through non-fusion atoms.
    path = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while len(path) < 4:
        nxt = None
        for nb in res_mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if j in nf_set and j != prev and j not in path:
                nxt = j
                break
        if nxt is None:
            return None
        path.append(nxt)
        prev, cur = cur, nxt

    # Two candidate numberings (path and its reverse); choose lowest bridgehead set.
    def numbering_for(order):
        return {idx: i + 1 for i, idx in enumerate(order)}

    cand = []
    for order in (path, list(reversed(path))):
        numb = numbering_for(order)
        bh_set = tuple(sorted(numb[b] for b in bridgeheads_res))
        cand.append((bh_set, numb))
    cand.sort(key=lambda c: c[0])
    return cand[0][1]


def get_bridged_fused_info(mol) -> Optional[Dict[str, Any]]:
    """
    Get complete information about a bridged fused system.

    Combines detection, core identification, and bridge analysis
    into a single comprehensive dict.

    Args:
        mol: RDKit Mol object

    Returns:
        Dict with all bridged fused information, or None if not applicable:
        - 'is_bridged_fused': True
        - 'core': Result from identify_fused_core()
        - 'bridges': Result from identify_bridges()
        - 'name': Generated IUPAC name

    Examples:
        >>> mol = Chem.MolFromSmiles("...")  # bridged fused system
        >>> info = get_bridged_fused_info(mol)
        >>> info['is_bridged_fused']
        True
    """
    if mol is None:
        return None

    if not detect_bridged_fused(mol):
        return None

    core = identify_fused_core(mol)
    if core is None:
        return None

    bridges = identify_bridges(mol, core['core_atoms'])
    result = name_bridged_fused_system(mol)
    # name_bridged_fused_system returns a tuple (name, ring_atoms, atom_to_locant, subs_included) or None
    name = result[0] if isinstance(result, tuple) else result

    return {
        'is_bridged_fused': True,
        'core': core,
        'bridges': bridges,
        'name': name,
    }
