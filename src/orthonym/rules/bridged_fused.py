"""
Bridged fused nomenclature  - systems that are part fused and part bridged.

Implements IUPAC 2013 rules for naming systems like 1,4-methanonaphthalene
where a fused core (naphthalene) has additional bridges across non-adjacent positions.

Key concepts:
- Bridged fused = fused core + additional bridges across the fused system
- Different from pure von Baeyer (no fused component)
- Different from pure fused (no bridges across)
- Name format: [locants]-[bridge_prefix][fused_parent_name]
  e.g., "1,4-methanonaphthalene", "1,4:5,8-dimethanonaphthalene"

Bridge prefixes (.3):
- Carbon bridges: methano (1C), ethano (2C), propano (3C), butano (4C)
- Heteroatom bridges: epoxy (O), epithio (S), epimino (NH), epidioxy (O-O)

Reference: IUPAC 2013 Blue Book, (Bridged Fused Ring Systems)
"""

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from rdkit import Chem

from .fused_rings import get_shared_atoms

logger = logging.getLogger(__name__)

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

#: (the Blue Book-14110): these bridge prefixes are general nomenclature;
#: the preselected (PIN) prefixes are 'sulfano', 'disulfano', 'selano', 'tellano' and 'azano',
#: which OPSIN 2.9.0 cannot read, so a name that uses one of these is never certified a PIN.
GENERAL_ONLY_BRIDGE_PREFIXES = frozenset({'epithio', 'epidithio', 'episeleno', 'epitelluro', 'epimino'})


def _record_general_bridge(prefix: str) -> None:
    """Record a general-nomenclature bridge prefix as a part that is never a PIN.

    The record is name-scoped (``metrics.provenance.record_non_pin_fragment``): it
    lowers the label of every shipped name that carries the prefix -- the bare
    bridged parent and any larger name built on it -- and of no other name."""
    if prefix in GENERAL_ONLY_BRIDGE_PREFIXES:
        try:
            from ..metrics.provenance import record_non_pin_fragment
            record_non_pin_fragment(prefix)
        except Exception:  # a label record must never break naming
            pass

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
        >>> mol = Chem.MolFromSmiles("c1ccc2ccccc2c1") # naphthalene
        >>> detect_bridged_fused(mol)
        False # pure fused, no bridges
        >>> mol = Chem.MolFromSmiles("C1CC2CCC1C2") # norbornane
        >>> detect_bridged_fused(mol)
        False # pure bridged, no fused core
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
    Identify the maximal fused ring component (.2 algorithm).

    For a bridged fused system, the fused core is the largest set of
    rings that share edges (ortho-fused or ortho-peri-fused).

    Maximization priority (.2):
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
        >>> mol = Chem.MolFromSmiles("c1ccc2ccccc2c1") # naphthalene
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
        >>> mol = Chem.MolFromSmiles("c1ccc2ccccc2c1") # naphthalene
        >>> bridges = identify_bridges(mol, set(range(10)))
        >>> len(bridges)
        0 # No bridges in pure naphthalene
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
        # A heteroatom bridge without a tabulated prefix (-SiH2-, -Se-, -PH-,...)
        # has no name here; the carbon table below would name it 'methano', a
        # different molecule, so return no prefix and let the caller decline.
        return ''

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

    Name format : [locants]-[bridge_prefix][fused_parent_name]
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
        >>> mol = Chem.MolFromSmiles("c1ccc2ccccc2c1") # naphthalene
        >>> name_bridged_fused_system(mol)
        None # Not bridged fused
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
        if not prefix:
            return None
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

    for prefix in bridge_groups:
        _record_general_bridge(prefix)
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
# Bridged fused PIN entry points
# ============================================================================
#
# A bridged fused ring system, the Blue Book) is a fused parent ring
# system plus one or more bridges across it. The bridged fused PIN builder (``rules/bridged_fused_pin``) selects
# the parent and the bridges by (the Blue Book-:14395), numbers the
# system by and, and spells substituents, suffixes and
# stereodescriptors on that numbering. It returns None for anything it cannot name; the
# caller then keeps the von Baeyer name or fails closed (G0).


def name_bridged_fused_pin(mol):
    """The bridged fused name of ``mol`` from the bridged fused PIN builder: the
    complex-ring tuple ``(name, ring_atoms, atom_to_locant, substituents_included)``, or
    None when the builder declines. ``substituents_included`` is True: the builder spells
    every prefix, suffix and stereodescriptor on its own numbering.

    A dependent secondary bridge (``tricyclo[...0^2,7]...``), a polyvalent bridge and a
    ring bridge are bridge shapes the builder only ranks as competitors
    (``selection._competitor_sets``); a reading that one of them outranks declines, and
    the caller keeps the von Baeyer name.
    """
    if mol is None:
        return None

    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    # A bridged fused ring system has at least three rings: a fused ring system of two or
    # more rings plus the rings its bridges create, the Blue Book).
    if len(rings) < 3:
        return None
    # Every bridged fused system goes to the bridged fused PIN builder, which ranks every
    # reading of the ring system by (a)-(j) (the Blue Book-:14395). The
    # bare-system excisions this function ran before slice S2 named a naphthalene parent
    # wherever one existed and so missed (b) "include the maximum number of skeletal atoms"
    # (:14271): '1,2,3,4-tetrahydro-1,4-propanonaphthalene' is
    # '6,7,8,9-tetrahydro-5H-5,9-ethanobenzo[7]annulene'. They were removed.
    return _package_pin(mol)


def _package_pin(mol):
    """The complex-ring tuple of the bridged fused PIN builder (``rules/bridged_fused_pin``),
    or None when it declines.

    ``name_bridged_fused_pin`` and the all-aromatic routing predicate
    ``has_aromatic_mancude_bridge`` both call it, so an internal error of the builder
    declines here (``builder_declining_on_error``) instead of escaping into routing."""
    from .bridged_fused_pin import build
    return builder_declining_on_error(build, mol)


def builder_declining_on_error(entry, *args):
    """``entry(*args)`` for an entry point of the bridged fused PIN builder (``build``,
    ``build_substituent``); None when the builder raises: no name, never a different one.

    The error is logged at ERROR level with its traceback, so a defect inside the builder
    shows in the eval and BB measure logs (the eval harness workers switch off logging up
    to WARNING) instead of looking like an ordinary decline. A deliberate refusal
    (``OrthonymLimitError``) still propagates, and a wall-clock limit
    (``WallClockTimeout``, a ``BaseException``) is not caught."""
    from ..errors import OrthonymLimitError
    try:
        return entry(*args)
    except OrthonymLimitError:
        raise
    except Exception:
        logger.exception("bridged fused PIN builder raised in %s; declined",
                         getattr(entry, "__name__", entry))
        return None


def has_aromatic_mancude_bridge(mol) -> bool:
    """True when a FULLY-AROMATIC-perceived ring system is actually a mancude
    bridged fused ring system that the bridged fused PIN builder names
    etheno bridges: RDKit's extended aromaticity marks the -CH=CH- bridge aromatic,
    so the von-Baeyer gate's all-aromatic skip would otherwise strand it, as in
    1,4-ethenonaphthalene and 4,7-ethenoazulene). Used only to EXEMPT this class
    from that skip; returns False for a plain fused PAH (no bridge) so the
    retained-name / fusion path keeps it."""
    if mol is None:
        return False
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    if len(rings) < 3:
        return False
    # exempt the system only when the bridged fused PIN builder names it (bare or
    # substituted: 1,4-ethenonaphthalene, 4,7-ethenoazulene, 6-methyl-1,4-ethenonaphthalene)
    return _package_pin(mol) is not None


def has_aromatic_chalcogen_bridge(mol) -> bool:
    """True when a divalent O/S ring atom is a genuine BRIDGE that RDKit's
    extended aromaticity model hides (Wave-2 completion, (a)):
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
