"""
IUPAC Ortho-Fused Ring System Naming

This module implements comprehensive IUPAC naming for ortho-fused heterocyclic systems,
including bicyclic lactones and other complex fused ring systems following Blue Book rules.

References:
- IUPAC 2013: ortho-fused polycyclic hydrocarbons
- IUPAC 2013: numbering of fused polycyclic hydrocarbons
- IUPAC 2013: indicated hydrogen in fused ring systems
- IUPAC 2013: von Baeyer nomenclature for bicyclic systems
"""

from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Mol


def detect_lactone(ring_atoms: List[int], mol: Mol) -> Optional[Dict]:
    """
    Detect if a ring is a lactone (cyclic ester).
    
    A lactone contains the -C(=O)O- pattern within the ring structure.
    
    IUPAC lactone classification:
    - γ-lactone (5-membered): oxolan-2-one
    - δ-lactone (6-membered): oxan-2-one
    - ε-lactone (7-membered): oxepan-2-one
    
    Args:
        ring_atoms: List of atom indices forming the ring
        mol: RDKit Mol object
    
    Returns:
        Dictionary with keys:
        - 'carbonyl_idx': Index of the carbonyl carbon
        - 'ester_O_idx': Index of the ester oxygen in the ring
        - 'ring_size': Number of atoms in the ring
        - 'type': Lactone type ('gamma', 'delta', 'epsilon', etc.)
        
        Returns None if not a lactone
    """
    if len(ring_atoms) < 4:  # Lactones need at least 4 atoms (β-lactone minimum)
        return None

    ring_set = set(ring_atoms)
    carbonyl_idx = None
    ester_O_idx = None

    # Iterate through ring atoms to find C=O pattern
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)

        # Look for carbon atoms
        if atom.GetSymbol() != 'C':
            continue

        # Check if this carbon has a double-bonded oxygen
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)

            # Check for C=O (double bond to oxygen)
            if (neighbor.GetSymbol() == 'O' and
                bond.GetBondType() == Chem.BondType.DOUBLE):

                # Carbonyl oxygen should NOT be in the ring
                if neighbor_idx in ring_set:
                    continue

                # This carbon should also be bonded to an oxygen IN the ring
                for ring_neighbor in atom.GetNeighbors():
                    ring_neighbor_idx = ring_neighbor.GetIdx()

                    if (ring_neighbor_idx in ring_set and
                        ring_neighbor_idx != neighbor_idx and
                        ring_neighbor.GetSymbol() == 'O' and
                        mol.GetBondBetweenAtoms(atom_idx, ring_neighbor_idx).GetBondType() == Chem.BondType.SINGLE):

                        # Found lactone pattern: C(=O)O where both C and O are in ring
                        carbonyl_idx = atom_idx
                        ester_O_idx = ring_neighbor_idx
                        break

            if carbonyl_idx is not None:
                break

        if carbonyl_idx is not None:
            break

    if carbonyl_idx is None or ester_O_idx is None:
        return None

    # Classify lactone type based on ring size
    ring_size = len(ring_atoms)
    lactone_type_map = {
        4: 'beta',      # β-lactone (rare, strained)
        5: 'gamma',     # γ-lactone
        6: 'delta',     # δ-lactone
        7: 'epsilon'    # ε-lactone
    }

    lactone_type = lactone_type_map.get(ring_size, 'large')

    return {
        'carbonyl_idx': carbonyl_idx,
        'ester_O_idx': ester_O_idx,
        'ring_size': ring_size,
        'type': lactone_type
    }




def identify_fused_rings(mol: Mol) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    Identify pairs of ortho-fused rings (sharing 2 adjacent atoms and 1 bond).
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        List of tuples: (ring1_atoms, ring2_atoms, shared_atoms)
    """
    ring_info = mol.GetRingInfo()
    rings = [list(ring) for ring in ring_info.AtomRings()]
    fused_pairs = []

    # Check all pairs of rings
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            ring1 = set(rings[i])
            ring2 = set(rings[j])

            # Find shared atoms
            shared = list(ring1 & ring2)

            # IUPAC: ortho-fused = two rings sharing exactly two
            # adjacent atoms (one bond)
            if len(shared) == 2:
                idx1, idx2 = shared
                if mol.GetBondBetweenAtoms(idx1, idx2) is not None:
                    fused_pairs.append((rings[i], rings[j], shared))

    return fused_pairs


def classify_heterocycle_priority(ring_atoms: List[int], mol: Mol) -> int:
    """
    Classify heterocycle priority according to IUPAC rules.
    
    IUPAC Priority (highest to lowest):
    1. Nitrogen-containing heterocycles
    2. Oxygen-containing heterocycles
    3. Sulfur-containing heterocycles
    4. Carbocycles (all carbon)
    
    Within each class, larger rings have priority.
    
    Args:
        ring_atoms: List of atom indices in the ring
        mol: RDKit Mol object
        
    Returns:
        Priority score (higher = more preferred as base component)
    """
    heteroatom_priority = {'N': 300, 'O': 200, 'S': 100, 'C': 0}

    # Find highest priority heteroatom in ring
    max_hetero_priority = 0
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        priority = heteroatom_priority.get(symbol, 0)
        max_hetero_priority = max(max_hetero_priority, priority)

    # Priority = heteroatom_priority * 1000 + ring_size
    # This ensures N > O > S > C, with larger rings preferred within each class
    return max_hetero_priority * 1000 + len(ring_atoms)


def determine_base_component(rings: List[Tuple[List[int], List[int], List[int]]],
                             mol: Mol) -> Optional[Tuple[List[int], List[int]]]:
    """
    Determine the base component (parent ring) for a fused system.
    
    IUPAC rules for base component selection:
    1. Heterocyclic ring preferred over carbocyclic
    2. Among heterocycles: N > O > S
    3. Larger ring preferred
    4. Lactone rings get special consideration
    
    Args:
        rings: List of fused ring pairs from identify_fused_rings
        mol: RDKit Mol object
        
    Returns:
        Tuple of (base_ring_atoms, attached_ring_atoms) or None
    """
    if not rings:
        return None

    # Take the first fused pair (we'll handle multiple fusions later)
    ring1, ring2, shared = rings[0]

    # Check if either is a lactone (lactones are typically base component)
    lactone1 = detect_lactone(ring1, mol)
    lactone2 = detect_lactone(ring2, mol)

    if lactone1 and not lactone2:
        return (ring1, ring2)
    elif lactone2 and not lactone1:
        return (ring2, ring1)

    # Otherwise, use heterocycle priority
    priority1 = classify_heterocycle_priority(ring1, mol)
    priority2 = classify_heterocycle_priority(ring2, mol)

    if priority1 >= priority2:
        return (ring1, ring2)
    else:
        return (ring2, ring1)


def get_bicyclo_descriptor(base_ring: List[int], attached_ring: List[int],
                           shared_atoms: List[int], mol: Mol) -> str:
    """
    Generate bicyclo descriptor [x.y.z] for fused ring system.
    
    For ortho-fused systems, one bridge has 0 atoms (the shared bond).
    Numbers are listed in descending order.
    
    Args:
        base_ring: Atoms in the base ring
        attached_ring: Atoms in the attached ring
        shared_atoms: The 2 atoms shared between rings
        mol: RDKit Mol object
        
    Returns:
        Bicyclo descriptor string like "[4.3.0]"
    """
    # Total unique atoms in the bicyclic system
    all_atoms = set(base_ring) | set(attached_ring)
    shared_set = set(shared_atoms)

    # Bridge lengths are the paths between bridgehead atoms
    # For ortho-fused: two bridges with atoms, one with 0
    base_only = [a for a in base_ring if a not in shared_set]
    attached_only = [a for a in attached_ring if a not in shared_set]

    # Bridge sizes (excluding bridgehead atoms)
    bridge1 = len(base_only)
    bridge2 = len(attached_only)
    bridge3 = 0  # The fused bond has 0 atoms between bridgeheads

    # Sort in descending order
    bridges = sorted([bridge1, bridge2, bridge3], reverse=True)

    return f"[{bridges[0]}.{bridges[1]}.{bridges[2]}]"


def get_parent_name(total_carbons: int) -> str:
    """Get parent alkane name for carbon count."""
    from ..data.chain_names import get_chain_prefix
    return get_chain_prefix(total_carbons)


def number_bicyclic_system(base_ring: List[int], attached_ring: List[int],
                           shared_atoms: List[int], mol: Mol,
                           lactone_info: Optional[Dict] = None) -> Dict[int, int]:
    """
    Number bicyclic system according to IUPAC bicyclo rules.
    
    IUPAC bicyclo numbering:
    1. Start at one bridgehead atom
    2. Number along the longest bridge first
    3. Then the second longest bridge
    4. The fused bond (0-length bridge) is between the two bridgeheads
    
    For bicyclo[4.3.0]nonane:
    - Position 1: first bridgehead
    - Positions 2-5: longest bridge (4 atoms)
    - Position 6: second bridgehead
    - Positions 7-9: second bridge (3 atoms back to pos 1)
    
    Args:
        base_ring: Atoms in base ring
        attached_ring: Atoms in attached ring
        shared_atoms: Bridgehead atoms
        mol: RDKit Mol object
        lactone_info: Lactone detection info if present
        
    Returns:
        Dict mapping atom index to IUPAC locant
    """
    atom_to_locant = {}
    all_atoms = set(base_ring) | set(attached_ring)
    shared_set = set(shared_atoms)

    if len(shared_atoms) != 2:
        # Not a proper ortho-fused system
        for i, idx in enumerate(all_atoms):
            atom_to_locant[idx] = i + 1
        return atom_to_locant

    bridgehead1, bridgehead2 = shared_atoms[0], shared_atoms[1]

    # Separate bridges (paths from bridgehead1 to bridgehead2)
    base_only = [a for a in base_ring if a not in shared_set]
    attached_only = [a for a in attached_ring if a not in shared_set]

    # Determine which bridge is longer
    if len(base_only) >= len(attached_only):
        longer_bridge = base_only
        shorter_bridge = attached_only
        longer_ring = base_ring
        shorter_ring = attached_ring
    else:
        longer_bridge = attached_only
        shorter_bridge = base_only
        longer_ring = attached_ring
        shorter_ring = base_ring

    # Build ordered path for a bridge
    def get_ordered_path(ring: List[int], start: int, end: int, exclude: set) -> List[int]:
        """Get ordered path from start to end through ring, excluding certain atoms."""
        path = []
        current = start
        visited = {start}

        while True:
            atom = mol.GetAtomWithIdx(current)
            next_atom = None

            for neighbor in atom.GetNeighbors():
                n_idx = neighbor.GetIdx()
                if n_idx in ring and n_idx not in visited and n_idx not in exclude:
                    next_atom = n_idx
                    break
                elif n_idx == end and len(path) > 0:
                    return path

            if next_atom is None:
                break

            path.append(next_atom)
            visited.add(next_atom)
            current = next_atom

            if current == end:
                path.pop()
                break

        return path

    # Get ordered atoms in each bridge
    # Longer bridge: bh1 → bh2
    # Shorter bridge: bh1 → bh2 (same direction for consistency)
    longer_path = get_ordered_path(longer_ring, bridgehead1, bridgehead2, shared_set - {bridgehead1, bridgehead2})
    shorter_path = get_ordered_path(shorter_ring, bridgehead1, bridgehead2, shared_set - {bridgehead1, bridgehead2})

    # If paths didn't work, use simple ordering
    if len(longer_path) != len(longer_bridge) or len(shorter_path) != len(shorter_bridge):
        longer_path = longer_bridge
        shorter_path = shorter_bridge

    # IUPAC lactone numbering: carbonyl should get highest locant in shorter bridge
    # For correct OPSIN recognition (8-oxa-9-one): furanyl-C → O → carbonyl
    # So we need to ensure carbonyl carbon is LAST in shorter_path

    # Find carbonyl position in shorter path
    carbonyl_idx_in_path = None
    if lactone_info:
        for i, idx in enumerate(shorter_path):
            if idx == lactone_info.get('carbonyl_idx'):
                carbonyl_idx_in_path = i
                break

    # If carbonyl is first, reverse to put it last
    if carbonyl_idx_in_path is not None and carbonyl_idx_in_path == 0:
        shorter_path = list(reversed(shorter_path))

    # Build the complete numbering
    # 1 = bridgehead1
    # 2 to len(longer)+1 = longer bridge
    # len(longer)+2 = bridgehead2 (position 6 for bicyclo[4.3.0])
    # len(longer)+3 to total = shorter bridge (positions 7,8,9)

    locant = 1
    atom_to_locant[bridgehead1] = locant
    locant += 1

    for idx in longer_path:
        atom_to_locant[idx] = locant
        locant += 1

    atom_to_locant[bridgehead2] = locant
    locant += 1

    for idx in shorter_path:
        atom_to_locant[idx] = locant
        locant += 1

    return atom_to_locant


def get_substituents(mol: Mol, ring_atoms: set, atom_to_locant: Dict[int, int]) -> List[Tuple[int, str]]:
    """
    Identify substituents on the bicyclic system.
    
    Args:
        mol: RDKit Mol object
        ring_atoms: Set of atoms in the bicyclic system
        atom_to_locant: Mapping of atom idx to IUPAC locant
        
    Returns:
        List of (locant, substituent_name) tuples
    """
    substituents = []
    alkyl_names = {1: "methyl", 2: "ethyl", 3: "propyl", 4: "butyl"}

    # Get all rings in molecule to check for heterocyclic substituents
    ring_info = mol.GetRingInfo()
    all_rings = [set(ring) for ring in ring_info.AtomRings()]

    for atom_idx in ring_atoms:
        if atom_idx not in atom_to_locant:
            continue

        atom = mol.GetAtomWithIdx(atom_idx)
        locant = atom_to_locant[atom_idx]

        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()

            # Skip ring atoms (in the bicyclic system)
            if n_idx in ring_atoms:
                continue

            # Skip carbonyl oxygen (part of lactone)
            bond = mol.GetBondBetweenAtoms(atom_idx, n_idx)
            if neighbor.GetSymbol() == 'O' and bond.GetBondType() == Chem.BondType.DOUBLE:
                continue

            # Check if this substituent is part of a separate ring (heterocycle)
            is_in_other_ring = False
            for ring in all_rings:
                if n_idx in ring and not ring.issubset(ring_atoms):
                    # Substituent is in a ring not part of the bicyclic system
                    is_in_other_ring = True

                    # Identify the heterocycle type
                    ring_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in ring]
                    ring_size = len(ring)
                    has_O = 'O' in ring_symbols
                    has_N = 'N' in ring_symbols
                    has_S = 'S' in ring_symbols

                    # Check aromaticity
                    is_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)

                    if ring_size == 5 and has_O and not has_N and not has_S and is_aromatic:
                        substituents.append((locant, "(furan-3-yl)"))
                    elif ring_size == 5 and has_N and is_aromatic:
                        substituents.append((locant, "(pyrrol-2-yl)"))
                    elif ring_size == 6 and has_N and is_aromatic:
                        substituents.append((locant, "(pyridin-2-yl)"))
                    else:
                        # Generic heterocycle
                        substituents.append((locant, f"({ring_size}-membered-heterocyclyl)"))
                    break

            if is_in_other_ring:
                continue

            # Identify simple substituents
            if neighbor.GetSymbol() == 'C':
                # Check for hydroxymethyl (-CH2OH)
                for n2 in neighbor.GetNeighbors():
                    if n2.GetIdx() != atom_idx:
                        if n2.GetSymbol() == 'O' and n2.GetTotalNumHs() > 0:
                            substituents.append((locant, "hydroxymethyl"))
                            break
                else:
                    # Count carbons in alkyl chain (but avoid counting ring atoms)
                    sub_carbons = 1
                    for n2 in neighbor.GetNeighbors():
                        if n2.GetIdx() != atom_idx and n2.GetSymbol() == 'C':
                            # Check it's not in any ring
                            if not any(n2.GetIdx() in r for r in all_rings):
                                sub_carbons += 1
                    if sub_carbons in alkyl_names:
                        name = alkyl_names[sub_carbons]
                    else:
                        from ..data.chain_names import get_alkyl_name as _get_alkyl
                        try:
                            name = _get_alkyl(sub_carbons)
                        except (ValueError, KeyError):
                            name = f"{sub_carbons}C-alkyl"
                    substituents.append((locant, name))

    return substituents


def format_substituent_prefix(substituents: List[Tuple[int, str]]) -> str:
    """Format substituents as IUPAC prefix."""
    if not substituents:
        return ""

    # Group by name
    from collections import defaultdict
    grouped = defaultdict(list)
    for locant, name in substituents:
        grouped[name].append(locant)


    parts = []
    for name, locants in sorted(grouped.items()):
        locants_str = ",".join(str(l) for l in sorted(locants))
        from ..assembly.naming_utils import multiplied_component as _mc
        parts.append(f"{locants_str}-{_mc(len(locants), name, name)}")

    return "-".join(parts) + "-" if parts else ""


def name_fused_lactone_system(mol: Mol, base_ring: List[int], attached_ring: List[int],
                              shared_atoms: List[int], lactone_info: Dict) -> str:
    """
    Generate IUPAC name for a fused lactone system.
    
    Format: [substituents]-bicyclo[x.y.z]parent-2-one
    
    Args:
        mol: RDKit Mol object
        base_ring: Atoms in lactone ring
        attached_ring: Atoms in attached ring
        shared_atoms: Bridgehead atoms
        lactone_info: Lactone detection info
        
    Returns:
        IUPAC systematic name
    """
    # Get bicyclo descriptor
    descriptor = get_bicyclo_descriptor(base_ring, attached_ring, shared_atoms, mol)

    # Count total atoms in bicyclic system (IUPAC uses total atoms, not just carbons)
    all_atoms = set(base_ring) | set(attached_ring)
    total_atoms = len(all_atoms)

    # Find heteroatoms and their positions for "oxa", "aza" prefixes
    # Number the system first to get locants
    atom_to_locant = number_bicyclic_system(base_ring, attached_ring, shared_atoms, mol, lactone_info)

    # Identify heteroatoms (oxygen in ring, not carbonyl oxygen)
    heteroatom_prefixes = []
    for idx in all_atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol == 'O' and idx == lactone_info['ester_O_idx']:
            locant = atom_to_locant.get(idx, 1)
            heteroatom_prefixes.append((locant, 'oxa'))
        elif symbol == 'N':
            locant = atom_to_locant.get(idx, 1)
            heteroatom_prefixes.append((locant, 'aza'))
        elif symbol == 'S':
            locant = atom_to_locant.get(idx, 1)
            heteroatom_prefixes.append((locant, 'thia'))

    # Sort by locant
    heteroatom_prefixes.sort(key=lambda x: x[0])
    heteroatom_str = "".join(f"{loc}-{prefix}" for loc, prefix in heteroatom_prefixes)

    # Get parent name based on total atom count
    parent = get_parent_name(total_atoms)

    # Get substituents (excluding ring atoms)
    substituents = get_substituents(mol, all_atoms, atom_to_locant)

    # Find carbonyl locant
    carbonyl_locant = atom_to_locant.get(lactone_info['carbonyl_idx'], 2)

    # Check for unsaturation (double bonds other than carbonyl)
    unsaturation = ""
    for atom_idx in all_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in all_atoms:
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    # Skip carbonyl (C=O)
                    if atom_idx == lactone_info['carbonyl_idx']:
                        continue
                    if neighbor.GetSymbol() == 'O':
                        continue
                    # Found C=C double bond - use lower locant per IUPAC
                    loc1 = atom_to_locant.get(atom_idx, 99)
                    loc2 = atom_to_locant.get(neighbor_idx, 99)
                    db_locant = min(loc1, loc2)
                    unsaturation = f"{db_locant}-en-"
                    break
        if unsaturation:
            break

    # Collect stereochemistry from bicyclic system
    from ..perception.stereo import assign_stereochemistry
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    # Ensure CIP labels are assigned (idempotent guard)
    assign_stereochemistry(mol)

    # Collect stereodescriptors for atoms in the bicyclic system
    stereo_descriptors = collect_stereodescriptors(mol, atom_to_locant)

    # Format as IUPAC prefix "(6S,8R)-"
    stereo_prefix = format_stereodescriptor_string(stereo_descriptors)

    # Build name components
    prefix = format_substituent_prefix(substituents)

    # IUPAC format: stereo-prefix-substituent-bicyclo[x.y.z]heteroatom-parent-unsaturation-locant-one
    # Example: (6S,8R)-5-hydroxymethyl-6-methyl-9-oxabicyclo[4.3.0]non-1-en-7-one
    if unsaturation:
        name = f"{stereo_prefix}{prefix}{heteroatom_str}bicyclo{descriptor}{parent}-{unsaturation}{carbonyl_locant}-one"
    else:
        name = f"{stereo_prefix}{prefix}{heteroatom_str}bicyclo{descriptor}{parent}an-{carbonyl_locant}-one"

    return name


def name_ortho_fused_system(mol: Mol) -> Optional[str]:
    """
    Name ortho-fused ring systems using IUPAC Blue Book rules.
    
    Priority:
    1. Check for retained names (handled by existing fused_heterocycles)
    2. Identify lactone systems
    3. Apply systematic bicyclo/fusion naming
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        IUPAC systematic name or None if naming fails
    """
    # Identify all ortho-fused ring pairs
    fused_pairs = identify_fused_rings(mol)

    if not fused_pairs:
        return None

    # Determine base and attached components
    components = determine_base_component(fused_pairs, mol)

    if not components:
        return None

    base_ring, attached_ring = components

    # Get shared atoms
    shared_atoms = list(set(base_ring) & set(attached_ring))

    # Check if base ring is a lactone
    lactone_info = detect_lactone(base_ring, mol)

    if lactone_info:
        # Generate fused lactone name
        return name_fused_lactone_system(mol, base_ring, attached_ring, shared_atoms, lactone_info)

    # For non-lactone fused systems, return None to use existing fallback
    return None
