"""
Ring system type classification, scoring, and selection per IUPAC P-44.2.

Provides the foundation for correct principal ring system selection.
Implements:
- P-44.2.2 type hierarchy (RingSystemType enum)
- P-44.2.1 general criteria (ring_system_score tuple)
- Principal ring system selection (select_principal_ring_system)

These are pure functions that can be tested independently before integration.

Reference: IUPAC 2013 Blue Book, P-44.2 (Selection of Preferred Ring System)
"""

from enum import IntEnum
from typing import List, Optional, Set, Tuple

from rdkit import Chem

from ..perception.rings import (
    get_ring_info,
    get_ring_systems,
    get_spiro_atoms,
    is_heterocyclic,
)


# ============================================================================
# P-44.2.2 Ring System Type Hierarchy
# ============================================================================


class RingSystemType(IntEnum):
    """P-44.2.2 type hierarchy. Lower value = more senior.

    IUPAC 2013 P-44.2.2.2:
    1. Spiro ring systems (P-24)
    2. Cyclic phane parent hydrides (P-26.4)
    3. Fused ring systems (P-25)
    4. Bridged fused ring systems (P-25.7)
    5. Von Baeyer ring systems (P-23)
    6. Linear phane parent hydrides (P-26)
    7. Ring assemblies (P-28)

    MONOCYCLIC is not in P-44.2.2 but needed as fallback for simple rings.
    """
    SPIRO = 1           # P-44.2.2.2.1
    CYCLIC_PHANE = 2    # P-44.2.2.2.2 (stub)
    FUSED = 3           # P-44.2.2.2.3
    BRIDGED_FUSED = 4   # P-44.2.2.2.4
    VON_BAEYER = 5      # P-44.2.2.2.5
    LINEAR_PHANE = 6    # P-44.2.2.2.6 (stub)
    RING_ASSEMBLY = 7   # P-44.2.2.2.7
    MONOCYCLIC = 8      # Simple monocyclic (not in P-44.2.2 hierarchy)


# ============================================================================
# Heteroatom Seniority for P-44.2.1
# ============================================================================

# Higher value = more senior. Used negated in scoring tuple so min() wins.
_HETEROATOM_SENIORITY = {
    'N': 10,
    'F': 9,
    'Cl': 8,
    'Br': 7,
    'I': 6,
    'O': 5,
    'S': 4,
    'Se': 3,
    'Te': 2,
    'P': 1,
}


# ============================================================================
# Ring System Type Classification
# ============================================================================


def classify_ring_system_type(
    mol: Chem.Mol,
    ring_system_atoms: Set[int]
) -> RingSystemType:
    """Classify a ring system into its P-44.2.2 type.

    Reuses existing detection functions from the codebase, operating on a
    sub-molecule built from the ring system atoms when necessary.

    Classification priority (same as _classify_complex_ring in composer.py):
    1. Spiro junction detected -> SPIRO
    2. Fused core + extra bridges -> BRIDGED_FUSED
    3. Ortho-fused or ortho-peri-fused -> FUSED
    4. Bicyclo or polycyclic bridged -> VON_BAEYER
    5. Fallback -> MONOCYCLIC

    Args:
        mol: RDKit Mol object (full molecule)
        ring_system_atoms: Set of atom indices belonging to this ring system

    Returns:
        RingSystemType enum value
    """
    if not ring_system_atoms:
        return RingSystemType.MONOCYCLIC

    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    # Count how many SSSR rings are fully contained in this system
    contained_rings = [
        ring for ring in atom_rings
        if set(ring).issubset(ring_system_atoms)
    ]
    num_rings = len(contained_rings)

    if num_rings <= 1:
        # Single ring or no rings -> check for spiro (spiro connects 2 rings
        # but with include_spiro=True they become one system)
        # A spiro system has a spiro atom shared by exactly 2 rings
        spiro_atoms = get_spiro_atoms(mol)
        spiro_in_system = spiro_atoms & ring_system_atoms
        if spiro_in_system:
            return RingSystemType.SPIRO
        return RingSystemType.MONOCYCLIC

    # Multiple rings in this system -- build a sub-molecule for detection
    sub_mol = _build_submol(mol, ring_system_atoms)
    if sub_mol is None:
        return RingSystemType.MONOCYCLIC

    # Check for spiro first (shared single atom between two rings)
    spiro_atoms_in_system = get_spiro_atoms(mol)
    spiro_in_system = spiro_atoms_in_system & ring_system_atoms
    if spiro_in_system:
        return RingSystemType.SPIRO

    # Classification priority (adapted from _classify_complex_ring in composer.py):
    # 1. bridged-fused (detect_bridged_fused) FIRST
    # 2. bicyclo (is_bicyclo_system) -- pure 2-ring bridged
    # 3. classify_fused_system 'bridged-fused' -- catches cases missed by
    #    detect_bridged_fused (e.g., benzonorbornadiene). Must come BEFORE
    #    is_polycyclic_system because that function would catch these as VB.
    # 4. polycyclic-bridged (is_polycyclic_system) -- tricyclo+ pure VB
    #    Must come after bridged-fused checks to avoid misclassification.
    # 5. fused (ortho-fused/ortho-peri-fused)

    from .bridged_fused import detect_bridged_fused
    from .fused_rings import classify_fused_system
    from .bicyclo import is_bicyclo_system
    from .polycyclic import is_polycyclic_system

    # Step 1: Check bridged-fused via the dedicated detector
    try:
        if detect_bridged_fused(sub_mol):
            return RingSystemType.BRIDGED_FUSED
    except Exception:
        pass

    # Step 2: Check bicyclo (2-ring bridged)
    # This catches norbornane, bicyclo[2.2.2]octane, etc. BEFORE the
    # classify_fused_system check which incorrectly flags them.
    try:
        if is_bicyclo_system(sub_mol):
            return RingSystemType.VON_BAEYER
    except Exception:
        pass

    # Step 3: Get classify_fused_system result
    fused_type = 'not-fused'
    try:
        fused_type = classify_fused_system(sub_mol)
    except Exception:
        pass

    # Step 3b: Check for bridged-fused via classify_fused_system
    # This catches systems like benzonorbornadiene where detect_bridged_fused
    # misses but classify_fused_system correctly identifies 'bridged-fused'.
    # Pure VB systems (norbornane) are already caught at step 2.
    if fused_type == 'bridged-fused':
        return RingSystemType.BRIDGED_FUSED

    # Step 4: Check polycyclic bridged (tricyclo+)
    try:
        if is_polycyclic_system(sub_mol):
            return RingSystemType.VON_BAEYER
    except Exception:
        pass

    # Step 5: Check fused (ortho-fused or ortho-peri-fused)
    if fused_type in ('ortho-fused', 'ortho-peri-fused'):
        return RingSystemType.FUSED

    # Multi-ring but doesn't match any specific type
    # Could be ring assembly or monocyclic fallback
    return RingSystemType.MONOCYCLIC


def _build_submol(mol: Chem.Mol, atom_indices: Set[int]) -> Optional[Chem.Mol]:
    """Build an RWMol sub-molecule from a subset of atoms.

    Creates a new molecule containing only the specified atoms and the
    bonds between them. Preserves atom properties (element, aromaticity,
    formal charge, etc.) and bond properties (type, aromaticity).

    Args:
        mol: Source RDKit Mol object
        atom_indices: Set of atom indices to include

    Returns:
        New RDKit Mol object, or None on failure
    """
    if not atom_indices:
        return None

    try:
        rw = Chem.RWMol()
        # Map old atom index -> new atom index
        old_to_new = {}

        # Add atoms
        for old_idx in sorted(atom_indices):
            old_atom = mol.GetAtomWithIdx(old_idx)
            new_idx = rw.AddAtom(Chem.Atom(old_atom.GetAtomicNum()))
            new_atom = rw.GetAtomWithIdx(new_idx)
            new_atom.SetIsAromatic(old_atom.GetIsAromatic())
            new_atom.SetFormalCharge(old_atom.GetFormalCharge())
            new_atom.SetNoImplicit(old_atom.GetNoImplicit())
            new_atom.SetNumExplicitHs(old_atom.GetNumExplicitHs())
            old_to_new[old_idx] = new_idx

        # Add bonds
        seen_bonds = set()
        for old_idx in atom_indices:
            for bond in mol.GetAtomWithIdx(old_idx).GetBonds():
                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                if begin in atom_indices and end in atom_indices:
                    bond_key = (min(begin, end), max(begin, end))
                    if bond_key not in seen_bonds:
                        seen_bonds.add(bond_key)
                        new_begin = old_to_new[begin]
                        new_end = old_to_new[end]
                        rw.AddBond(new_begin, new_end, bond.GetBondType())
                        new_bond = rw.GetBondBetweenAtoms(new_begin, new_end)
                        if new_bond is not None:
                            new_bond.SetIsAromatic(bond.GetIsAromatic())

        # Sanitize
        try:
            Chem.SanitizeMol(rw)
        except Exception:
            # If sanitization fails, try without kekulization
            try:
                Chem.SanitizeMol(
                    rw,
                    Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                )
            except Exception:
                pass

        return rw.GetMol()
    except Exception:
        return None


# ============================================================================
# Ring System Scoring (P-44.2.1 General Criteria)
# ============================================================================


def ring_system_score(
    mol: Chem.Mol,
    system_atoms: Set[int]
) -> tuple:
    """Score a ring system for principal ring system selection.

    Returns a scoring tuple where ALL values are arranged so that
    ``min()`` selects the most senior ring system.

    IUPAC P-44.2: General criteria (P-44.2.1) are applied BEFORE type
    hierarchy (P-44.2.2). Type hierarchy is a tiebreaker within the
    same general criteria class.

    Tuple ordering:
    - -has_heteroatom: P-44.2.1(a) heterocyclic preferred (negated)
    - -has_nitrogen: P-44.2.1(b) N-containing preferred (negated)
    - -senior_heteroatom_rank: P-44.2.1(c) most senior heteroatom (negated)
    - -num_rings: P-44.2.1(d) more rings = senior (negated)
    - -num_skeletal_atoms: P-44.2.1(e) more atoms = senior (negated)
    - -num_heteroatoms: P-44.2.1(f) more heteroatoms = senior (negated)
    - -heteroatom_variety_score: P-44.2.1(g) more of senior type (negated)
    - type_rank: P-44.2.2 type hierarchy (tiebreaker, lower = senior)

    Args:
        mol: RDKit Mol object
        system_atoms: Set of atom indices in this ring system

    Returns:
        Tuple suitable for comparison with min() to select most senior
    """
    if not system_atoms:
        return (0, 0, 0, 0, 0, 0, 0, 999)

    # 1. Type rank
    type_rank = int(classify_ring_system_type(mol, system_atoms))

    # 2-7. Analyze atoms in the system
    has_heteroatom = False
    has_nitrogen = False
    senior_heteroatom_rank = 0
    num_heteroatoms = 0
    heteroatom_counts = {}  # element -> count

    for idx in system_atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            has_heteroatom = True
            num_heteroatoms += 1
            heteroatom_counts[symbol] = heteroatom_counts.get(symbol, 0) + 1
            if symbol == 'N':
                has_nitrogen = True
            rank = _HETEROATOM_SENIORITY.get(symbol, 0)
            if rank > senior_heteroatom_rank:
                senior_heteroatom_rank = rank

    # Number of SSSR rings fully contained in this system
    ri = mol.GetRingInfo()
    num_rings = sum(
        1 for ring in ri.AtomRings()
        if set(ring).issubset(system_atoms)
    )

    # Number of skeletal atoms
    num_skeletal_atoms = len(system_atoms)

    # Heteroatom variety score: sum of (count * seniority) for each element
    heteroatom_variety_score = sum(
        count * _HETEROATOM_SENIORITY.get(elem, 0)
        for elem, count in heteroatom_counts.items()
    )

    # P-44.2: General criteria (P-44.2.1) applied BEFORE type hierarchy (P-44.2.2)
    return (
        -int(has_heteroatom),               # P-44.2.1(a): heterocyclic preferred
        -int(has_nitrogen),                 # P-44.2.1(b): N-containing preferred
        -senior_heteroatom_rank,            # P-44.2.1(c): most senior heteroatom
        -num_rings,                         # P-44.2.1(d): more rings = senior
        -num_skeletal_atoms,                # P-44.2.1(e): more atoms = senior
        -num_heteroatoms,                   # P-44.2.1(f): more heteroatoms
        -heteroatom_variety_score,          # P-44.2.1(g): more of senior type
        type_rank,                          # P-44.2.2: type hierarchy (tiebreaker)
    )


# ============================================================================
# Principal Ring System Selection
# ============================================================================


def select_principal_ring_system(
    mol: Chem.Mol,
    ring_systems: List[Set[int]]
) -> Tuple[int, ...]:
    """Select the most senior ring system from a list of candidates.

    Uses ring_system_score() to compare candidates. The system with the
    minimum score tuple is the most senior (P-44.2 hierarchy).

    Args:
        mol: RDKit Mol object
        ring_systems: List of sets, each set contains atom indices in
                      one ring system (from get_ring_systems())

    Returns:
        Tuple of sorted atom indices of the most senior ring system,
        or empty tuple if no ring systems provided
    """
    if not ring_systems:
        return ()

    if len(ring_systems) == 1:
        return tuple(sorted(ring_systems[0]))

    # Score each ring system and select the one with minimum score
    best_idx = 0
    best_score = ring_system_score(mol, ring_systems[0])

    for i in range(1, len(ring_systems)):
        score = ring_system_score(mol, ring_systems[i])
        if score < best_score:
            best_score = score
            best_idx = i

    return tuple(sorted(ring_systems[best_idx]))
