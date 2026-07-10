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
# P-44.2.1 ring selection uses P-18(b) heteroatom order plus halogens.
# Expanded per IUPAC 2013 errata (BBerrors.html) to include all 20 elements
# that can appear as heteroatoms in ring systems.
_HETEROATOM_SENIORITY = {
    'N': 20,     # Most senior heteroatom per P-18(b)
    'F': 19,     # Halogen (P-44.2.1 ring comparison)
    'Cl': 18,    # Halogen
    'Br': 17,    # Halogen
    'I': 16,     # Halogen
    'O': 15,     # P-18(b) Group 16
    'S': 14,
    'Se': 13,
    'Te': 12,
    'P': 11,     # P-18(b) Group 15
    'As': 10,    # Per P-18(b) errata
    'Sb': 9,
    'Bi': 8,
    'Si': 7,     # Group 14
    'Ge': 6,
    'Sn': 5,
    'Pb': 4,
    'B': 3,      # Group 13
    'Al': 2,
    'Ga': 1,
}

# Seniority order for P-44.2.1(g) term-by-term variety comparison.
# Tuple position i represents the count of element _HETEROATOM_VARIETY_ORDER[i].
# Negated counts so min() selects ring with MORE of senior element.
_HETEROATOM_VARIETY_ORDER = [
    'F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'N',
    'P', 'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga'
]


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

    # Phase 155.A D-03 + D-26: Cyclophane classification fires after spiro and
    # before bridged-fused (P-44.2.2 hierarchy: SPIRO=1 < CYCLIC_PHANE=2 < FUSED=3).
    # Source: 155-CONTEXT.md D-03, D-20, D-26; ring_selection.py:48 enum.
    # NOTE (D-20 root-cause-only, ISS-005): narrow exception scope to ImportError
    # only -- circular-import-safe lazy import idiom (matches multiplicative.py
    # lazy-import pattern). Runtime errors from is_cyclophane MUST bubble up;
    # do NOT swallow them. is_cyclophane already returns False (not raises) for
    # non-cyclophane mol per D-03 topology gate, so the try/except handles
    # ONLY the bootstrap ImportError case.
    try:
        from .phane import is_cyclophane
    except ImportError:
        pass
    else:
        if is_cyclophane(mol):
            all_systems = get_ring_systems(mol)
            if any(ring_system_atoms == s for s in all_systems):
                return RingSystemType.CYCLIC_PHANE

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


def _spiro_fusion_count(mol: Chem.Mol, system_atoms: Set[int]) -> int:
    """P-44.2.2.2.1.1: number of spiro fusions in this ring system (spiro atoms
    that lie within the system). 0 for a non-spiro system. Deterministic
    (depends only on the atom set, not SMILES order)."""
    return len(get_spiro_atoms(mol) & set(system_atoms))


def ring_system_score(
    mol: Chem.Mol,
    system_atoms: Set[int]
) -> tuple:
    """Score a ring system for principal ring system selection.

    Returns a scoring tuple where ALL values are arranged so that
    ``min()`` selects the most senior ring system.

    IUPAC P-44.2: General criteria (P-44.2.1) are applied BEFORE type
    hierarchy (P-44.2.2). Type hierarchy is a tiebreaker within the
    same general criteria class. P-44.4.1 unsaturation is a FURTHER
    tiebreaker, applied only after P-44.2.2 type seniority (so e.g.
    spiro > phane > fused stays senior to a mere double-bond difference).

    Tuple ordering (30 elements; Tasks 12-17 append P-44.2.2.2.x tiebreakers
    AFTER the P-44.4.1 tier so they only break within-type ties):
    - [0]  -has_heteroatom: P-44.2.1(a) heterocyclic preferred (negated)
    - [1]  -has_nitrogen: P-44.2.1(b) N-containing preferred (negated)
    - [2]  -senior_heteroatom_rank: P-44.2.1(c) most senior heteroatom (negated)
    - [3]  -num_rings: P-44.2.1(d) more rings = senior (negated)
    - [4]  -num_skeletal_atoms: P-44.2.1(e) more atoms = senior (negated)
    - [5]  -num_heteroatoms: P-44.2.1(f) more heteroatoms = senior (negated)
    - [6..25] heteroatom_variety_tuple: P-44.2.1(g) term-by-term comparison
              (20 elements: -count_N, -count_F, -count_Cl, -count_Br, -count_I,
               -count_O, -count_S, -count_Se, -count_Te, -count_P, ...)
    - [26] type_rank: P-44.2.2 type hierarchy (tiebreaker, lower = senior)
    - [27] -num_multiple_bonds: P-44.4.1.1 max ring multiple bonds (negated)
    - [28] -num_double_bonds: P-44.4.1.2 then max double bonds (negated)
    - [29] -spiro_fusions: P-44.2.2.2.1.1 more spiro fusions = senior (negated)

    The unsaturation tier (S1, V21 WS-A.1) breaks the among-equal-carbocycle
    tie that previously made ``C1CCCCC1c1ccccc1`` resolve to the arbitrary
    list-order winner ``phenylcyclohexane``; benzene now wins on unsaturation
    (P-44.4.1.1) -> ``cyclohexylbenzene``. Appended AFTER type_rank so it can
    never override P-44.2.2 type seniority. RDKit reports benzene bonds as
    AROMATIC, so the counter must treat AROMATIC as multiple (a naive
    DOUBLE-only count gives benzene zero).

    Args:
        mol: RDKit Mol object
        system_atoms: Set of atom indices in this ring system

    Returns:
        Tuple suitable for comparison with min() to select most senior
    """
    if not system_atoms:
        return (0, 0, 0, 0, 0, 0) + (0,) * len(_HETEROATOM_VARIETY_ORDER) + (999, 0, 0, 0)

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

    # P-44.4.1 unsaturation (aromatic-aware): count ring bonds that are
    # DOUBLE / TRIPLE / AROMATIC. RDKit kekulizes benzene to AROMATIC bonds,
    # so AROMATIC must count as multiple or an aromatic ring scores zero.
    num_multiple_bonds = 0
    num_double_bonds = 0
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in system_atoms or b not in system_atoms:
            continue
        bt = bond.GetBondType()
        if bond.GetIsAromatic() or bt in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            num_multiple_bonds += 1
            if bond.GetIsAromatic() or bt == Chem.BondType.DOUBLE:
                num_double_bonds += 1

    # P-44.2.1(g): heteroatom variety -- term-by-term comparison by seniority
    # Build tuple: (-count_of_N, -count_of_F, ..., -count_of_P)
    # Negated so min() selects ring with MORE of the most-senior element
    heteroatom_variety_tuple = tuple(
        -heteroatom_counts.get(elem, 0)
        for elem in _HETEROATOM_VARIETY_ORDER
    )

    # P-44.2.2.2.1.1 (Task 12): number of spiro fusions (more = senior). Appended
    # AFTER the P-44.4.1 unsaturation tier so it only breaks a WITHIN-spiro tie
    # and never overrides type/unsaturation seniority. Deterministic (atom-set
    # only), so it introduces no spelling dependence.
    spiro_fusions = _spiro_fusion_count(mol, system_atoms)

    # P-44.2: General criteria (P-44.2.1) applied BEFORE type hierarchy (P-44.2.2)
    return (
        -int(has_heteroatom),               # P-44.2.1(a): heterocyclic preferred
        -int(has_nitrogen),                 # P-44.2.1(b): N-containing preferred
        -senior_heteroatom_rank,            # P-44.2.1(c): most senior heteroatom
        -num_rings,                         # P-44.2.1(d): more rings = senior
        -num_skeletal_atoms,                # P-44.2.1(e): more atoms = senior
        -num_heteroatoms,                   # P-44.2.1(f): more heteroatoms
        *heteroatom_variety_tuple,          # P-44.2.1(g): 20 elements, term-by-term
        type_rank,                          # P-44.2.2: type hierarchy (tiebreaker)
        -num_multiple_bonds,                # P-44.4.1.1: max ring multiple bonds
        -num_double_bonds,                  # P-44.4.1.2: then max double bonds
        -spiro_fusions,                     # P-44.2.2.2.1.1: more spiro fusions
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
