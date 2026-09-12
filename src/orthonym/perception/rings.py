"""
Ring system detection and analysis.

Handles detection of:
- Simple rings
- Fused ring systems
- Spiro systems
- Aromatic rings
- Heterocyclic rings
"""

from typing import Dict, List, Set, Tuple

from rdkit import Chem
from .molcache import bonds_of


def get_ring_info(mol) -> Dict:
    """
    Get basic ring information from RDKit.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Dictionary with ring information:
        - atom_rings: tuple of tuples of atom indices
        - bond_rings: tuple of tuples of bond indices
        - num_rings: number of rings
    """
    ri = mol.GetRingInfo()
    return {
        "atom_rings": ri.AtomRings(),
        "bond_rings": ri.BondRings(),
        "num_rings": ri.NumRings(),
    }


def get_ring_systems(mol, include_spiro: bool = False) -> List[Set[int]]:
    """
    Find connected ring systems.
    
    Groups rings that share atoms into ring systems:
    - Fused rings share >1 atom
    - Spiro rings share exactly 1 atom
    
    Args:
        mol: RDKit Mol object
        include_spiro: If True, spiro-connected rings are in same system
        
    Returns:
        List of sets, each set contains atom indices in one ring system
    """
    ri = mol.GetRingInfo()
    systems = []

    for ring in ri.AtomRings():  #: merge is order-independent (produces same sets)
        ring_atoms = set(ring)

        # Threshold: >1 for fused only, >0 to include spiro
        threshold = 0 if include_spiro else 1

        # Find systems sharing atoms with this ring
        merged = [s for s in systems if len(ring_atoms & s) > threshold]

        if merged:
            # Merge all overlapping systems
            new_system = ring_atoms.union(*merged)
            systems = [s for s in systems if s not in merged] + [new_system]
        else:
            systems.append(ring_atoms)

    return systems


def get_complete_ring_atom_set(mol) -> frozenset:
    """Return ALL atoms in ALL ring systems (fused, bridged, spiro merged).

    IUPAC: Ring system = all atoms in connected ring components.
    Includes bridgehead atoms, bridge atoms, spiro atoms.
    Does NOT include exocyclic atoms (=O, -OH, etc.) per.
    RDKit's AtomRings correctly reports only ring-member atoms.

    Args:
        mol: RDKit Mol object

    Returns:
        frozenset of all ring atom indices across all ring systems
    """
    systems = get_ring_systems(mol, include_spiro=True)
    all_atoms = set()
    for system in systems:
        all_atoms.update(system)
    return frozenset(all_atoms)


def get_containing_ring_system(mol, ring_atoms) -> frozenset:
    """Return all atoms in the ring system(s) containing the given ring atoms.

    For a single SSSR ring that is part of a fused/bridged/spiro system,
    this returns the complete merged system. Used by Type A callers
    (ring-parent substituent detection) to prevent BFS from walking
    into fused partner rings.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices (e.g., one SSSR ring)

    Returns:
        frozenset of all atom indices in the containing ring system(s)
    """
    target = set(ring_atoms)
    systems = get_ring_systems(mol, include_spiro=True)
    result = set()
    for system in systems:
        if target & system:
            result.update(system)
    return frozenset(result) if result else frozenset(target)


def is_aromatic_ring(mol, ring_atoms) -> bool:
    """
    Check if all atoms in a ring are aromatic.
    
    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices
        
    Returns:
        True if all atoms in the ring are aromatic
    """
    return all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_atoms)


def get_aromatic_rings(mol) -> List[Tuple[int, ...]]:
    """
    Get all aromatic rings in the molecule.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        List of tuples, each tuple contains atom indices of an aromatic ring
    """
    ri = mol.GetRingInfo()
    return [ring for ring in ri.AtomRings() if is_aromatic_ring(mol, ring)]


def get_ring_size(mol, atom_idx: int) -> int:
    """
    Get the smallest ring size containing an atom.
    
    Args:
        mol: RDKit Mol object
        atom_idx: Index of atom
        
    Returns:
        Smallest ring size, or 0 if atom is not in any ring
    """
    ri = mol.GetRingInfo()
    sizes = [len(ring) for ring in ri.AtomRings() if atom_idx in ring]
    return min(sizes) if sizes else 0


def is_in_ring(mol, atom_idx: int) -> bool:
    """
    Check if an atom is in any ring.
    
    Args:
        mol: RDKit Mol object
        atom_idx: Index of atom
        
    Returns:
        True if atom is in a ring
    """
    return mol.GetRingInfo().NumAtomRings(atom_idx) > 0


def atoms_in_same_ring(mol, idx1: int, idx2: int) -> bool:
    """
    Check if two atoms are in the same ring.
    
    Args:
        mol: RDKit Mol object
        idx1, idx2: Atom indices
        
    Returns:
        True if atoms share at least one ring
    """
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if idx1 in ring and idx2 in ring:
            return True
    return False


def get_ring_heteroatoms(mol, ring_atoms) -> List[Tuple[int, str]]:
    """
    Get heteroatoms (non-carbon) in a ring.
    
    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices
        
    Returns:
        List of (position_in_ring, element_symbol) tuples
        Position is 0-indexed within the ring
    """
    heteroatoms = []
    ring_list = list(ring_atoms)

    for pos, idx in enumerate(ring_list):
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            heteroatoms.append((pos, symbol))

    return heteroatoms


def count_ring_double_bonds(mol, ring_atoms) -> int:
    """
    Count double bonds within a ring.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices

    Returns:
        Number of double bonds in the ring
    """
    ring_set = set(ring_atoms)
    count = 0

    for bond in bonds_of(mol):
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        if begin_idx in ring_set and end_idx in ring_set:
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                count += 1

    return count


def get_ring_double_bond_atoms(mol, ring_atoms) -> List[Tuple[int, int]]:
    """
    Get all double bonds within a ring as atom pairs.

    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices

    Returns:
        List of (atom_idx1, atom_idx2) tuples for each double bond
    """
    ring_set = set(ring_atoms)
    double_bonds = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        if begin_idx in ring_set and end_idx in ring_set:
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                double_bonds.append((begin_idx, end_idx))

    return double_bonds


def is_saturated_ring(mol, ring_atoms) -> bool:
    """
    Check if a ring is fully saturated (no double bonds).
    
    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices
        
    Returns:
        True if ring has no double bonds
    """
    return count_ring_double_bonds(mol, ring_atoms) == 0


def is_heterocyclic(mol, ring_atoms) -> bool:
    """
    Check if a ring contains heteroatoms.
    
    Args:
        mol: RDKit Mol object
        ring_atoms: Iterable of atom indices
        
    Returns:
        True if ring contains non-carbon atoms
    """
    for idx in ring_atoms:
        if mol.GetAtomWithIdx(idx).GetSymbol() != 'C':
            return True
    return False


def get_spiro_atoms(mol) -> Set[int]:
    """
    Find atoms that are spiro centers (shared by exactly 2 rings).
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Set of atom indices that are spiro centers
    """
    ri = mol.GetRingInfo()
    atom_ring_count = {}

    for ring in ri.AtomRings():
        for atom_idx in ring:
            atom_ring_count[atom_idx] = atom_ring_count.get(atom_idx, 0) + 1

    # Spiro atoms are in exactly 2 rings and only share with themselves
    spiro_atoms = set()
    for atom_idx, count in atom_ring_count.items():
        if count == 2:
            # Check if this is the only shared atom between two rings
            rings_containing = [ring for ring in ri.AtomRings() if atom_idx in ring]
            if len(rings_containing) == 2:
                shared = set(rings_containing[0]) & set(rings_containing[1])
                if shared == {atom_idx}:
                    spiro_atoms.add(atom_idx)

    return spiro_atoms


def get_bridgehead_atoms(mol) -> Set[int]:
    """
    Find bridgehead atoms in bridged ring systems.

    Args:
        mol: RDKit Mol object

    Returns:
        Set of atom indices that are bridgeheads
    """
    ri = mol.GetRingInfo()
    atom_ring_count = {}

    for ring in ri.AtomRings():
        for atom_idx in ring:
            atom_ring_count[atom_idx] = atom_ring_count.get(atom_idx, 0) + 1

    # Bridgehead atoms are in 2+ rings (but not spiro centers)
    spiro = get_spiro_atoms(mol)
    bridgeheads = set()

    for atom_idx, count in atom_ring_count.items():
        if count >= 2 and atom_idx not in spiro:
            bridgeheads.add(atom_idx)

    return bridgeheads


def find_ring_bridgeheads(mol, ring_atoms: Set[int] = None) -> Set[int]:
    """Find von-Baeyer bridgehead atoms: ring-skeletal atoms bonded to >=3
    other ring-skeletal atoms.

    /: the SINGLE consolidated bridgehead predicate (seeded from
    VonBaeyerAnalyzer._find_all_bridgeheads, polycyclic.py:425-433). Counts
    only ring-member neighbours, so an exocyclic substituent (camphor's
    gem-dimethyl bridgehead) does NOT disqualify a bridgehead — exactly the
     fix. This is distinct from get_bridgehead_atoms (count>=2 ring
    membership, too loose) which is left unchanged because it is widely
    imported.

    IUPAC: a bridgehead is a skeletal atom bonded to three or more
    other skeletal atoms (excluding H); bridgeheads MAY be quaternary/
    substituted.

    Args:
        mol: RDKit Mol object

    Returns:
        Set of atom indices that are bridgeheads (>=3 ring-member neighbours).
    """
    if ring_atoms is None:
        ring_atoms = set()
        for ring in mol.GetRingInfo().AtomRings():
            ring_atoms.update(ring)

    bridgeheads = set()
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        ring_neighbors = sum(
            1 for n in atom.GetNeighbors() if n.GetIdx() in ring_atoms
        )
        if ring_neighbors >= 3:
            bridgeheads.add(idx)
    return bridgeheads


def classify_ring(mol, ring_atoms: Tuple[int, ...]) -> str:
    """
    Classify a ring by its chemical type.

    Classification order (check in this order):
    1. Heterocyclic aromatic - contains non-carbon atoms AND all atoms aromatic
    2. Heterocyclic saturated - contains non-carbon atoms AND not all aromatic
    3. Aromatic - all atoms are aromatic (RDKit detection), carbocyclic
    4. Cycloalkane - saturated, all carbon, no double bonds
    5. Cycloalkene - unsaturated, all carbon, has double bonds but not aromatic

    All heterocyclic return values start with 'heterocyclic' so callers can use
    ``ring_type.startswith('heterocyclic')`` for backward-compatible matching.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices defining the ring

    Returns:
        Classification string: 'heterocyclic_aromatic', 'heterocyclic_saturated',
        'aromatic', 'cycloalkane', or 'cycloalkene'

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCCCC1') # cyclohexane
        >>> classify_ring(mol, mol.GetRingInfo.AtomRings[0])
        'cycloalkane'
        >>> mol = Chem.MolFromSmiles('c1ccccc1') # benzene
        >>> classify_ring(mol, mol.GetRingInfo.AtomRings[0])
        'aromatic'
        >>> mol = Chem.MolFromSmiles('c1ccncc1') # pyridine
        >>> classify_ring(mol, mol.GetRingInfo.AtomRings[0])
        'heterocyclic_aromatic'
        >>> mol = Chem.MolFromSmiles('C1CCNCC1') # piperidine
        >>> classify_ring(mol, mol.GetRingInfo.AtomRings[0])
        'heterocyclic_saturated'
    """
    # Check heterocyclic first (highest priority)
    if is_heterocyclic(mol, ring_atoms):
        if is_aromatic_ring(mol, ring_atoms):
            return 'heterocyclic_aromatic'
        return 'heterocyclic_saturated'

    # Check aromatic (carbocyclic)
    if is_aromatic_ring(mol, ring_atoms):
        return 'aromatic'

    # Check saturated (cycloalkane)
    if is_saturated_ring(mol, ring_atoms):
        return 'cycloalkane'

    # Must be cycloalkene (unsaturated carbocyclic, non-aromatic)
    return 'cycloalkene'
