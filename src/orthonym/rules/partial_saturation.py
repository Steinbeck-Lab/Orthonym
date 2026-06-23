"""
Partial saturation detection and prefix generation for fused ring systems.

This module handles IUPAC 2013 hydro prefixes for partially saturated
fused heterocycles and polycyclic compounds:
- dihydro- (2 H added)
- tetrahydro- (4 H added)
- hexahydro- (6 H added)
- octahydro- (8 H added)
- decahydro- (10 H added)
- dodecahydro- (12 H added)
- perhydro- (fully saturated, no locants needed)

IUPAC 2013 Blue Book P-31.1.1:
"Prefixes 'dihydro', 'tetrahydro', etc. indicate the addition of hydrogen
to specified positions of an otherwise unsaturated parent structure."

Key Rules:
1. Always include locants for partial saturation (2,3-dihydro, not just dihydro)
2. Perhydro means ALL ring atoms saturated - no locants used
3. Saturation prefix comes LAST before parent name, AFTER substituents
4. Order: [substituents]-[saturation prefix]-[indicated H]-[parent]

Reference: IUPAC 2013 Blue Book P-31.1.1
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from rdkit import Chem


# Saturation prefix mapping based on number of added hydrogens
# Each sp3 carbon in ring adds 2 hydrogens vs aromatic parent
SATURATION_PREFIXES: Dict[int, str] = {
    2: 'dihydro',
    4: 'tetrahydro',
    6: 'hexahydro',
    8: 'octahydro',
    10: 'decahydro',
    12: 'dodecahydro',
    14: 'tetradecahydro',
    16: 'hexadecahydro',
}


def detect_partial_saturation(
    mol: Chem.Mol,
    aromatic_parent_smiles: str
) -> Optional[Dict[str, Any]]:
    """
    Detect partial saturation by comparing molecule to aromatic parent.

    Counts sp3-hybridized atoms in the ring system that would be sp2/aromatic
    in the parent structure. The IUPAC hydro prefix (dihydro-, tetrahydro-, etc.)
    indicates the number of hydrogen atoms added, which corresponds to
    2 * (number of sp3 ring atoms).

    For fused ring systems:
    - tetrahydroquinoline: 4 positions saturated = 4 sp3 atoms in reduced ring
    - indoline (2,3-dihydroindole): 2 sp3 atoms at positions 2,3
    - decahydronaphthalene (decalin): all 10 ring atoms sp3 = perhydro

    Args:
        mol: RDKit molecule to analyze
        aromatic_parent_smiles: SMILES of the fully aromatic parent structure

    Returns:
        Dict with saturation info, or None if no saturation detected:
        - 'sp3_count': Number of sp3 atoms in ring system
        - 'hydrogen_count': Number of added hydrogens (sp3_count * 2)
        - 'prefix': Saturation prefix string ('dihydro', 'tetrahydro', etc.)
        - 'saturated_indices': List of atom indices that are sp3
        - 'is_perhydro': True if fully saturated

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')  # tetrahydroquinoline
        >>> result = detect_partial_saturation(mol, 'c1ccc2ncccc2c1')
        >>> result['prefix']
        'hexahydro'  # 3 sp3 carbons * 2 = 6H
    """
    if mol is None:
        return None

    if aromatic_parent_smiles is None:
        return None

    parent = Chem.MolFromSmiles(aromatic_parent_smiles)
    if parent is None:
        return None

    # Get ring atoms in the molecule
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    # Count aromatic atoms in the parent
    parent_aromatic_count = sum(1 for atom in parent.GetAtoms() if atom.GetIsAromatic())

    # Find sp3-hybridized atoms in the ring system
    # These are the "saturated positions" where hydrogen was added
    sp3_indices = []

    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)

        # Count sp3 atoms (saturated positions)
        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            sp3_indices.append(idx)

    sp3_count = len(sp3_indices)

    # No saturation detected
    if sp3_count == 0:
        return None

    # Calculate hydrogen count based on actual atom valence
    # Carbon sp3 adds 2H vs aromatic, but N adds only 1H (trivalent)
    hydrogen_count = 0
    for idx in sp3_indices:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 7:  # Nitrogen
            hydrogen_count += 1  # N-H (trivalent N in ring)
        else:
            hydrogen_count += 2  # C adds 2H when saturated

    # Check for perhydro (fully saturated)
    # When all ring atoms are sp3
    is_perhydro = sp3_count >= parent_aromatic_count

    # Get prefix
    if is_perhydro:
        prefix = 'perhydro'
    else:
        prefix = get_saturation_prefix(hydrogen_count)

    if prefix is None:
        return None

    return {
        'sp3_count': sp3_count,
        'hydrogen_count': hydrogen_count,
        'prefix': prefix,
        'saturated_indices': sp3_indices,
        'is_perhydro': is_perhydro,
        'parent_aromatic_count': parent_aromatic_count,
    }


def get_saturation_prefix(hydrogen_count: int) -> Optional[str]:
    """
    Get the IUPAC saturation prefix for a given hydrogen count.

    Args:
        hydrogen_count: Number of added hydrogens (typically 2, 4, 6, 8, 10, 12)

    Returns:
        Saturation prefix string, or None if not a standard count

    Examples:
        >>> get_saturation_prefix(2)
        'dihydro'
        >>> get_saturation_prefix(4)
        'tetrahydro'
        >>> get_saturation_prefix(10)
        'decahydro'
    """
    return SATURATION_PREFIXES.get(hydrogen_count)


def get_saturation_locants(
    mol: Chem.Mol,
    sp3_atom_indices: List[int],
    atom_to_locant: Dict[int, Union[int, str]]
) -> List[Union[int, str]]:
    """
    Get IUPAC locants for saturated positions.

    Converts atom indices to IUPAC locants using the provided mapping,
    then sorts them according to IUPAC rules (lowest locant set).

    Args:
        mol: RDKit molecule
        sp3_atom_indices: List of atom indices that are sp3 (saturated)
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Sorted list of locants for saturation prefix

    Examples:
        >>> atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4, ...}
        >>> get_saturation_locants(mol, [0, 1, 2, 3], atom_to_locant)
        [1, 2, 3, 4]
    """
    locants = []

    for idx in sp3_atom_indices:
        locant = atom_to_locant.get(idx)
        if locant is not None:
            locants.append(locant)

    # Sort locants using IUPAC rules
    return _sort_locants(locants)


def _sort_locants(locants: List[Union[int, str]]) -> List[Union[int, str]]:
    """
    Sort locants according to IUPAC rules.

    Handles mixed int/str locants (e.g., 1, 2, '3a', '4a').
    Numeric locants come first in ascending order, then fusion locants.

    Args:
        locants: List of locants (int or str like '3a', '7a')

    Returns:
        Sorted list of locants
    """
    def _locant_sort_key(loc: Union[int, str]) -> Tuple[int, str]:
        """Sort key: (numeric_part, alpha_suffix)."""
        if isinstance(loc, int):
            return (loc, '')
        elif isinstance(loc, str):
            # Parse '3a' -> (3, 'a'), '7a' -> (7, 'a')
            if loc and loc[-1].isalpha():
                try:
                    return (int(loc[:-1]), loc[-1])
                except ValueError:
                    return (999, loc)
            try:
                return (int(loc), '')
            except ValueError:
                return (999, loc)
        return (999, str(loc))

    return sorted(locants, key=_locant_sort_key)


def format_saturation_prefix(
    prefix: str,
    locants: Optional[List[Union[int, str]]] = None
) -> str:
    """
    Format saturation prefix with locants for IUPAC name.

    IUPAC rules:
    - Perhydro: no locants (perhydro-)
    - All others: locants required (2,3-dihydro-, 1,2,3,4-tetrahydro-)

    Args:
        prefix: Saturation prefix ('dihydro', 'tetrahydro', 'perhydro', etc.)
        locants: Optional list of locants (not used for perhydro)

    Returns:
        Formatted prefix string ready for name assembly

    Examples:
        >>> format_saturation_prefix('perhydro')
        'perhydro'
        >>> format_saturation_prefix('tetrahydro', [1, 2, 3, 4])
        '1,2,3,4-tetrahydro'
        >>> format_saturation_prefix('dihydro', [2, 3])
        '2,3-dihydro'
    """
    # Perhydro never has locants
    if prefix == 'perhydro':
        return 'perhydro'

    # All other saturation prefixes MUST have locants per IUPAC
    if locants:
        sorted_locants = _sort_locants(locants)
        locant_str = ','.join(str(loc) for loc in sorted_locants)
        return f"{locant_str}-{prefix}"

    # Fallback: return prefix without locants (not ideal, but better than nothing)
    return prefix


def get_saturated_position_locants(
    mol: Chem.Mol,
    saturated_indices: List[int],
    atom_to_locant: Dict[int, Union[int, str]]
) -> List[Union[int, str]]:
    """
    Get IUPAC locants for saturated positions.

    This is an alias for get_saturation_locants for clearer naming.

    Args:
        mol: RDKit molecule
        saturated_indices: List of atom indices that are saturated
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Sorted list of locants for saturation prefix
    """
    return get_saturation_locants(mol, saturated_indices, atom_to_locant)


def analyze_saturation_for_naming(
    mol: Chem.Mol,
    aromatic_parent_smiles: str,
    atom_to_locant: Dict[int, Union[int, str]]
) -> Optional[str]:
    """
    Complete analysis for saturation prefix generation in naming.

    This is the main entry point for the composer module. It combines
    detection, locant assignment, and formatting into a single call.

    Args:
        mol: RDKit molecule to analyze
        aromatic_parent_smiles: SMILES of the aromatic parent structure
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Formatted saturation prefix string, or None if no saturation

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')  # tetrahydroquinoline
        >>> parent = 'c1ccc2ncccc2c1'  # quinoline
        >>> atom_to_locant = {...}  # IUPAC locant mapping
        >>> analyze_saturation_for_naming(mol, parent, atom_to_locant)
        '1,2,3,4-tetrahydro'
    """
    # Detect saturation
    result = detect_partial_saturation(mol, aromatic_parent_smiles)
    if result is None:
        return None

    prefix = result['prefix']
    saturated_indices = result['saturated_indices']
    is_perhydro = result['is_perhydro']

    if is_perhydro:
        # Perhydro: no locants needed
        return 'perhydro'

    # Get locants for saturated positions
    locants = get_saturated_position_locants(mol, saturated_indices, atom_to_locant)

    if not locants:
        # No locants available - return prefix only (suboptimal)
        return prefix

    return format_saturation_prefix(prefix, locants)


def is_fully_saturated(mol: Chem.Mol, aromatic_parent_smiles: str) -> bool:
    """
    Check if a molecule is fully saturated (perhydro) relative to parent.

    Args:
        mol: RDKit molecule to check
        aromatic_parent_smiles: SMILES of the aromatic parent

    Returns:
        True if the molecule is fully saturated (perhydro)

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCCC2CCCCC12')  # decalin (perhydronaphthalene)
        >>> is_fully_saturated(mol, 'c1ccc2ccccc2c1')  # naphthalene
        True
    """
    result = detect_partial_saturation(mol, aromatic_parent_smiles)
    if result is None:
        # No saturation info - could be already fully aromatic or no match
        return False
    return result.get('is_perhydro', False)


def count_ring_sp3_atoms(mol: Chem.Mol) -> int:
    """
    Count sp3-hybridized atoms in ring systems.

    Utility function for saturation analysis.

    Args:
        mol: RDKit molecule

    Returns:
        Number of sp3 atoms in rings
    """
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    count = 0
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            count += 1

    return count


def get_ring_saturation_level(
    mol: Chem.Mol,
    aromatic_parent_smiles: str
) -> str:
    """
    Get a human-readable saturation level description.

    Args:
        mol: RDKit molecule
        aromatic_parent_smiles: SMILES of aromatic parent

    Returns:
        Description string: 'aromatic', 'partially saturated', or 'fully saturated'
    """
    result = detect_partial_saturation(mol, aromatic_parent_smiles)

    if result is None:
        return 'aromatic'
    elif result.get('is_perhydro', False):
        return 'fully saturated'
    else:
        return 'partially saturated'


# =============================================================================
# Carbocyclic Partial Saturation (PAH systems)
# =============================================================================


def _carbocyclic_mancude_parents_by_size(n_ring_atoms: int) -> List[str]:
    """All-carbon mancude parents in ``POLYCYCLIC_DATA`` that carry a populated
    ``iupac_numbering`` and are a *pure* ring system of exactly ``n_ring_atoms``
    atoms.

    Returning every same-size candidate (e.g. naphthalene AND azulene at 10;
    anthracene AND phenanthrene at 14) is safe: the bond-order-agnostic
    automorphism match in :func:`_match_mancude_parent_numbering` only succeeds
    for the parent whose *connectivity* matches, so a mismatched same-size
    parent is silently skipped (fail-closed — never a wrong parent guess).
    """
    from ..data.polycyclic_data import POLYCYCLIC_DATA

    out: List[str] = []
    for name, entry in POLYCYCLIC_DATA.items():
        numbering = entry.get('iupac_numbering')
        csmiles = entry.get('canonical_smiles')
        if not numbering or not csmiles:
            continue
        cmol = Chem.MolFromSmiles(csmiles)
        if cmol is None or cmol.GetNumAtoms() != n_ring_atoms:
            continue
        if any(a.GetSymbol() != 'C' for a in cmol.GetAtoms()):
            continue
        out.append(name)
    return out


def _match_mancude_parent_numbering(
    mol: Chem.Mol,
    ring_atoms: Set[int],
    sp3_atoms: Set[int],
    ring_double_bonds: List[Tuple[int, int]],
    candidate_parents: List[str],
) -> Optional[Tuple[str, Dict[int, Any]]]:
    """Number a (partly) saturated fused carbocycle by its mancude parent.

    For each candidate parent, do a bond-order-agnostic automorphism match of
    the stored aromatic parent skeleton onto ``mol`` and read off the parent's
    fixed ``iupac_numbering`` (incl. the lettered fusion locants 4a/8a/...).
    Among ALL matching numberings (across all candidates) pick the one giving
    the LOWEST locant set to the hydro positions (the sp3 ring atoms), then to
    the residual ring double bonds — IUPAC 2013 P-31.1.4.3.4.

    Returns ``(parent_name, {atom_idx: locant})`` for the winning numbering, or
    ``None`` if no candidate's skeleton matches the ring system exactly (fail
    closed — never emit a wrong parent/locant).

    This is the shared primitive behind both the aromatic-bearing partial-PAH
    path (:func:`detect_carbocyclic_partial_saturation`) and the all-saturated
    residual-ene path (:func:`name_hydrogenated_fused_carbocycle`); it replaces
    the old naive sp3-walk that mis-numbered non-adjacent hydro positions (e.g.
    1,4-dihydronaphthalene → wrong "1,2-").
    """
    from ..data.polycyclic_data import POLYCYCLIC_DATA

    ring_atom_set = set(ring_atoms)
    best_key = None
    best: Optional[Tuple[str, Dict[int, Any]]] = None

    for parent in candidate_parents:
        entry = POLYCYCLIC_DATA.get(parent)
        if not entry:
            continue
        numbering = entry.get('iupac_numbering') or {}
        csmiles = entry.get('canonical_smiles')
        if not numbering or not csmiles:
            continue
        cmol = Chem.MolFromSmiles(csmiles)
        if cmol is None:
            continue
        n_canonical = cmol.GetNumAtoms()
        if n_canonical != len(ring_atom_set):
            continue

        params = Chem.AdjustQueryParameters.NoAdjustments()
        params.makeBondsGeneric = True
        params.aromatizeIfPossible = False
        params.adjustDegree = False
        query = Chem.AdjustQueryProperties(cmol, params)

        for match in mol.GetSubstructMatches(query, uniquify=False):
            if len(match) != n_canonical:
                continue
            atom_to_locant = {
                match[c_idx]: loc
                for c_idx, loc in numbering.items()
                if c_idx < len(match)
            }
            # The match must cover EXACTLY the ring system (not some other subset).
            if set(atom_to_locant) != ring_atom_set:
                continue
            hydro_locs = sorted(_locant_key(atom_to_locant[a]) for a in sp3_atoms)
            ene_locs = sorted(
                min(_locant_key(atom_to_locant[i]), _locant_key(atom_to_locant[j]))
                for i, j in ring_double_bonds
            )
            key = (hydro_locs, ene_locs)
            if best_key is None or key < best_key:
                best_key = key
                best = (parent, atom_to_locant)

    return best


def detect_carbocyclic_partial_saturation(
    mol: Chem.Mol,
    fused_ring_atoms: Set[int]
) -> Optional[Dict[str, Any]]:
    """
    Detect partial saturation in carbocyclic fused systems.

    This function identifies partially saturated polycyclic aromatic hydrocarbons
    like tetrahydronaphthalene, dihydroanthracene, etc. It analyzes the fused
    ring system to detect if it's a partially saturated version of a known
    aromatic parent (naphthalene, anthracene, phenanthrene).

    The detection works by:
    1. Checking all ring atoms are carbons (pure carbocycle)
    2. Counting aromatic vs sp3 atoms
    3. Matching the ring system size and structure to known parents
    4. Computing the saturation prefix based on sp3 count

    IUPAC 2013 Blue Book P-31.1.1:
    - tetrahydronaphthalene: 4 sp3 atoms = tetrahydro prefix
    - dihydronaphthalene: 2 sp3 atoms = dihydro prefix
    - decahydronaphthalene (decalin): all sp3 = perhydro or decahydro

    Args:
        mol: RDKit molecule
        fused_ring_atoms: Set of atom indices in the fused ring system

    Returns:
        Dict with saturation info, or None if not a recognized partially
        saturated carbocycle:
        - 'parent_name': Name of aromatic parent ('naphthalene', etc.)
        - 'parent_smiles': SMILES of aromatic parent
        - 'prefix': Saturation prefix string ('tetrahydro', etc.)
        - 'saturated_indices': List of atom indices that are sp3
        - 'sp3_count': Number of sp3 atoms
        - 'hydrogen_count': Number of added hydrogens (sp3_count * 2)
        - 'is_perhydro': True if fully saturated
        - 'atom_to_locant': Mapping from atom index to IUPAC locant (if available)

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCC2')  # tetrahydronaphthalene
        >>> ri = mol.GetRingInfo()
        >>> ring_atoms = set()
        >>> for ring in ri.AtomRings():
        ...     ring_atoms.update(ring)
        >>> result = detect_carbocyclic_partial_saturation(mol, ring_atoms)
        >>> result['prefix']
        'tetrahydro'
        >>> result['parent_name']
        'naphthalene'
    """
    if mol is None or not fused_ring_atoms:
        return None

    # Check if all ring atoms are carbons (carbocyclic)
    for idx in fused_ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            # Has heteroatom - not a pure carbocycle
            return None

    # Count sp3 and aromatic atoms in the fused ring system
    sp3_indices = []
    aromatic_indices = []

    for idx in fused_ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            sp3_indices.append(idx)
        elif atom.GetIsAromatic():
            aromatic_indices.append(idx)

    sp3_count = len(sp3_indices)
    aromatic_count = len(aromatic_indices)
    total_ring_atoms = len(fused_ring_atoms)

    # No sp3 atoms = fully aromatic, not partially saturated
    if sp3_count == 0:
        return None

    # For fully saturated systems (aromatic_count == 0), we need to distinguish:
    # - Fused bicyclic systems (decalin = 2 fused 6-rings = perhydronaphthalene)
    # - Simple large rings (cyclodecane = 1 ring with 10 atoms)
    #
    # Only fused systems should be named as perhydro-aromatics.
    if aromatic_count == 0:
        # Check if this is a fused ring system (multiple rings sharing atoms)
        ri = mol.GetRingInfo()
        atom_rings = ri.AtomRings()

        # Count rings that overlap with fused_ring_atoms
        rings_in_system = []
        for ring in atom_rings:
            ring_set = set(ring)
            if ring_set & fused_ring_atoms:
                rings_in_system.append(ring_set)

        # If there's only 1 ring, it's a simple cycloalkane, not a fused system
        if len(rings_in_system) <= 1:
            return None

        # Check if rings are actually fused (share atoms)
        is_fused = False
        for i, ring1 in enumerate(rings_in_system):
            for j, ring2 in enumerate(rings_in_system):
                if i < j and len(ring1 & ring2) >= 2:
                    is_fused = True
                    break
            if is_fused:
                break

        if not is_fused:
            return None

    # Residual non-aromatic ring C=C double bonds — needed both to choose the
    # lowest-locant numbering (P-31.1.4.3.4) and to recognise full saturation.
    ring_double_bonds: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetIsAromatic():
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in fused_ring_atoms and b in fused_ring_atoms:
            ring_double_bonds.append((a, b))

    # Identify the mancude parent AND its IUPAC numbering by a bond-order-agnostic
    # automorphism match (P-31.1.4): the all-carbon parent skeleton whose
    # connectivity matches this ring system, numbered to give the lowest locants
    # first to the hydro (sp3) positions, then to any residual ring double bond.
    # Fail closed if no all-carbon parent of this exact ring size matches — never
    # guess the parent by ring-atom count (the old size->name guess mislabelled
    # phenanthrene as anthracene) and never number by a naive sp3 walk (the old
    # _build_naphthalene_type_locants mis-numbered non-adjacent hydro positions,
    # e.g. 1,4-dihydronaphthalene -> wrong "1,2-").
    sp3_set = set(sp3_indices)
    candidates = _carbocyclic_mancude_parents_by_size(total_ring_atoms)
    matched = _match_mancude_parent_numbering(
        mol, fused_ring_atoms, sp3_set, ring_double_bonds, candidates,
    )
    if matched is None:
        return None
    parent_name, atom_to_locant = matched

    from ..data.polycyclic_data import POLYCYCLIC_DATA
    parent_smiles = POLYCYCLIC_DATA[parent_name].get('canonical_smiles')

    # In a mancude carbocycle every ring atom is sp2; each sp3 ring atom is one
    # added hydrogen, so hydrogen_count == sp3_count.
    hydrogen_count = sp3_count
    is_perhydro = (sp3_count == total_ring_atoms)

    if is_perhydro:
        prefix = 'perhydro'
    else:
        prefix = get_saturation_prefix(hydrogen_count)

    if prefix is None:
        return None

    return {
        'parent_name': parent_name,
        'parent_smiles': parent_smiles,
        'prefix': prefix,
        'saturated_indices': sp3_indices,
        'sp3_count': sp3_count,
        'hydrogen_count': hydrogen_count,
        'is_perhydro': is_perhydro,
        'atom_to_locant': atom_to_locant,
    }


def _build_naphthalene_type_locants(
    mol: Chem.Mol,
    ring_atoms: Set[int],
    sp3_indices: List[int]
) -> Dict[int, int]:
    """
    Build IUPAC locant mapping for naphthalene-type fused systems.

    For tetrahydronaphthalene, IUPAC numbering is:
    - Positions 1-4: saturated ring (the sp3 atoms)
    - Positions 4a, 5-8, 8a: aromatic ring

    For simplicity, we assign:
    - sp3 atoms get locants 1, 2, 3, 4 (in order around the saturated ring)
    - Aromatic atoms get locants 5, 6, 7, 8 (in order)

    Args:
        mol: RDKit molecule
        ring_atoms: Set of atom indices in the fused system
        sp3_indices: List of sp3 atom indices

    Returns:
        Dict mapping atom index to IUPAC locant (1-indexed)
    """
    from collections import defaultdict

    if not sp3_indices:
        return {idx: i + 1 for i, idx in enumerate(sorted(ring_atoms))}

    # Get ring info
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    # Find the ring containing sp3 atoms (the saturated ring)
    saturated_ring = None
    for ring in atom_rings:
        ring_set = set(ring)
        if not (ring_set & ring_atoms):
            continue
        ring_sp3 = ring_set & set(sp3_indices)
        if len(ring_sp3) >= 2:
            saturated_ring = ring
            break

    if saturated_ring is None:
        return {idx: i + 1 for i, idx in enumerate(sorted(ring_atoms))}

    # Find fusion atoms (shared between rings)
    atom_ring_count = defaultdict(int)
    for ring in atom_rings:
        ring_set = set(ring)
        if ring_set & ring_atoms:
            for idx in ring:
                if idx in ring_atoms:
                    atom_ring_count[idx] += 1

    fusion_atoms = {idx for idx, count in atom_ring_count.items() if count > 1}

    # Build ordered traversal around the saturated ring
    # Start from an sp3 atom that's adjacent to a fusion atom
    sp3_set = set(sp3_indices)

    # Find starting sp3 atom adjacent to fusion
    start_atom = None
    for idx in sp3_indices:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in fusion_atoms:
                start_atom = idx
                break
        if start_atom:
            break

    if start_atom is None:
        start_atom = sp3_indices[0]

    # Traverse the sp3 atoms in order (simple chain traversal)
    visited = set()
    sp3_order = []
    current = start_atom

    while len(sp3_order) < len(sp3_indices):
        if current in visited:
            break
        visited.add(current)
        sp3_order.append(current)

        # Find next unvisited sp3 neighbor
        atom = mol.GetAtomWithIdx(current)
        found_next = False
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in sp3_set and nbr_idx not in visited:
                current = nbr_idx
                found_next = True
                break

        if not found_next:
            # No more sp3 neighbors, break
            break

    # Add any missed sp3 atoms
    for idx in sp3_indices:
        if idx not in visited:
            sp3_order.append(idx)

    # Build locant mapping: sp3 atoms get locants 1,2,3,4
    atom_to_locant = {}
    for i, idx in enumerate(sp3_order):
        atom_to_locant[idx] = i + 1

    # Non-sp3 atoms get higher locants (5, 6, 7, 8, ...)
    non_sp3 = [idx for idx in ring_atoms if idx not in sp3_set]
    next_locant = len(sp3_order) + 1
    for idx in sorted(non_sp3):  # Simple ordering for now
        atom_to_locant[idx] = next_locant
        next_locant += 1

    return atom_to_locant


def is_tetrahydronaphthalene(mol: Chem.Mol, fused_ring_atoms: Set[int]) -> bool:
    """
    Check if fused system is tetrahydronaphthalene-type.

    Tetrahydronaphthalene has:
    - 10 ring atoms total
    - 4 sp3 carbons (saturated ring)
    - 6 aromatic carbons (benzene ring)
    - All carbons (no heteroatoms)

    Args:
        mol: RDKit molecule
        fused_ring_atoms: Set of atom indices in the fused ring system

    Returns:
        True if the system is tetrahydronaphthalene-type

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCC2')
        >>> ri = mol.GetRingInfo()
        >>> ring_atoms = set()
        >>> for ring in ri.AtomRings():
        ...     ring_atoms.update(ring)
        >>> is_tetrahydronaphthalene(mol, ring_atoms)
        True
    """
    if len(fused_ring_atoms) != 10:
        return False

    sp3_count = 0
    aromatic_count = 0

    for idx in fused_ring_atoms:
        atom = mol.GetAtomWithIdx(idx)

        # Must be carbon
        if atom.GetSymbol() != 'C':
            return False

        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            sp3_count += 1
        elif atom.GetIsAromatic():
            aromatic_count += 1

    # Tetrahydronaphthalene: 4 sp3 + 6 aromatic
    return sp3_count == 4 and aromatic_count == 6


# =============================================================================
# Hydrogenated fused carbocycles with residual ring double bonds (V-5 / V2 theme)
# =============================================================================

# Number of ring atoms -> mancude parent that has a populated ``iupac_numbering``
# map in POLYCYCLIC_DATA. Only parents whose numbering (incl. the lettered fusion
# locants 4a/8a/...) is stored can be numbered correctly; everything else fails
# closed (no wrong locants). 10 = naphthalene covers the common V-5 class.
_HYDRO_FUSED_PARENT_BY_SIZE: Dict[int, str] = {
    10: 'naphthalene',
}


def _locant_key(loc: Union[int, str, Tuple[int, str]]) -> Tuple[int, str]:
    """Sortable key for a locant that may be an int, a ``(int, str)`` tuple, or a
    string like ``'4a'`` — so that ``4 < 4a < 5`` and integers/tuples mix cleanly."""
    if isinstance(loc, tuple):
        return (int(loc[0]), str(loc[1]))
    if isinstance(loc, int):
        return (loc, '')
    if isinstance(loc, str):
        i = 0
        while i < len(loc) and loc[i].isdigit():
            i += 1
        if i == 0:
            return (999, loc)
        return (int(loc[:i]), loc[i:])
    return (999, str(loc))


def _locant_display(loc: Union[int, str, Tuple[int, str]]) -> str:
    """Render a locant (int / ``(int, str)`` tuple / ``'4a'`` string) as the
    string used in a name (``4a``, ``8a``, ``1`` ...)."""
    if isinstance(loc, tuple):
        return f"{loc[0]}{loc[1]}"
    return str(loc)


def name_hydrogenated_fused_carbocycle(mol: Chem.Mol) -> Optional[str]:
    """Name a *partially* saturated fused bicyclic carbocycle (naphthalene-type).

    This covers the V-5 / V2-theme defect: a fused carbocycle that is the
    hydrogenated form of a mancude parent (e.g. naphthalene) but still retains
    one or more *isolated* (non-aromatic) ring C=C double bonds. The legacy
    ``_name_saturated_fused_carbocyclic`` blindly emits the fully-saturated
    (``decahydro``) name for any non-aromatic two-ring carbocycle, dropping the
    residual double bond and naming a different molecule.

    Algorithm (IUPAC 2013 P-31.1.4):
      1. Require a two-ring, all-carbon, non-aromatic system with >= 1 residual
         ring double bond (fully-saturated systems are handled as ``perhydro``
         elsewhere -> return None here).
      2. Identify the mancude parent by ring-atom count; require a populated
         ``iupac_numbering`` map (with lettered fusion locants).
      3. Enumerate every automorphic numbering of the parent skeleton onto the
         molecule (bond-order-agnostic substructure match), and pick the one
         giving the LOWEST locant set to the ``hydro`` prefixes (the sp3 ring
         atoms), then to the residual double bonds (P-31.1.4.3.4).
      4. Emit ``<locants>-<count>hydro<parent>`` (e.g.
         ``1,2,3,4,4a,5,6,8a-octahydronaphthalene``).

    Fails closed (returns None) for any system it cannot number correctly so it
    never emits a wrong name.

    Args:
        mol: RDKit molecule (a two-ring carbocyclic, non-aromatic fused system).

    Returns:
        The hydro-prefixed parent name, or None if not handled (fail closed).
    """
    from ..data.polycyclic_data import POLYCYCLIC_DATA

    if mol is None:
        return None

    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if len(atom_rings) != 2:
        return None

    ring_atoms: Set[int] = set()
    for ring in atom_rings:
        ring_atoms.update(ring)

    # Carbocyclic only; no aromatic ring atoms (the aromatic-bearing partial
    # case is handled by the polycyclics PAH path).
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            return None
        if atom.GetIsAromatic():
            return None

    # Residual ring double bonds (non-aromatic C=C with both atoms in the ring).
    ring_double_bonds: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetIsAromatic():
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in ring_atoms and b in ring_atoms:
            ring_double_bonds.append((a, b))

    # No residual double bond -> fully saturated; perhydro is handled elsewhere.
    if not ring_double_bonds:
        return None

    # The sp3 (saturated / "hydro") ring atoms.
    sp3_atoms = {
        idx for idx in ring_atoms
        if mol.GetAtomWithIdx(idx).GetHybridization() == Chem.HybridizationType.SP3
    }
    sp2_atoms = ring_atoms - sp3_atoms

    # Every sp2 ring atom must be accounted for by a ring double bond (no
    # exocyclic unsaturation such as an exocyclic =CH2 or =O on the ring) — else
    # the hydro count would be wrong. Fail closed otherwise.
    if len(sp2_atoms) != 2 * len(ring_double_bonds):
        return None

    parent = _HYDRO_FUSED_PARENT_BY_SIZE.get(len(ring_atoms))
    if parent is None:
        return None
    entry = POLYCYCLIC_DATA.get(parent)
    if entry is None:
        return None
    numbering = entry.get('iupac_numbering') or {}
    canonical_smiles = entry.get('canonical_smiles')
    if not numbering or not canonical_smiles:
        return None

    canonical_mol = Chem.MolFromSmiles(canonical_smiles)
    if canonical_mol is None:
        return None

    # Bond-order-agnostic query: the stored parent SMILES is aromatic, but the
    # molecule is (partly) saturated, so match on connectivity only while keeping
    # the canonical atom order so ``numbering`` (keyed by canonical index) applies.
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    params.aromatizeIfPossible = False
    params.adjustDegree = False
    query = Chem.AdjustQueryProperties(canonical_mol, params)

    matches = mol.GetSubstructMatches(query, uniquify=False)
    if not matches:
        return None

    n_hydro = len(sp3_atoms)
    prefix = SATURATION_PREFIXES.get(n_hydro)
    if prefix is None:
        return None

    n_canonical = canonical_mol.GetNumAtoms()

    # Choose the numbering minimizing (hydro-locant set, then residual-double-bond
    # locant set) per P-31.1.4.3.4 (lowest locants to hydro prefixes + ene).
    best_key = None
    best_map: Optional[Dict[int, Any]] = None
    for match in matches:
        if len(match) != n_canonical:
            continue
        atom_to_locant = {
            match[c_idx]: loc
            for c_idx, loc in numbering.items()
            if c_idx < len(match)
        }
        if len(atom_to_locant) != len(ring_atoms):
            continue
        hydro_locs = sorted((_locant_key(atom_to_locant[a]) for a in sp3_atoms))
        ene_locs = sorted(
            min(_locant_key(atom_to_locant[i]), _locant_key(atom_to_locant[j]))
            for i, j in ring_double_bonds
        )
        key = (hydro_locs, ene_locs)
        if best_key is None or key < best_key:
            best_key = key
            best_map = atom_to_locant

    if best_map is None:
        return None

    hydro_display = sorted((best_map[a] for a in sp3_atoms), key=_locant_key)
    locant_str = ','.join(_locant_display(loc) for loc in hydro_display)
    return f"{locant_str}-{prefix}{parent}"


# =============================================================================
# Added indicated hydrogen + ring-ketone suffix (P-31.1.4.2.4 / P-58.2)
# =============================================================================


def _ring_carbonyl_carbons(mol, ring_set: Set[int]) -> List[int]:
    """Ring carbons bearing an exocyclic ketone-type =O (suitable for an -one /
    -dione suffix). Covers ketone, lactam, and lactone carbonyls on mancude
    rings — all named with the -one suffix per P-64.2.2.2."""
    out: List[int] = []
    for idx in ring_set:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        for bond in atom.GetBonds():
            o = bond.GetOtherAtom(atom)
            if (bond.GetBondType() == Chem.BondType.DOUBLE
                    and o.GetSymbol() == 'O'
                    and o.GetIdx() not in ring_set
                    and o.GetDegree() == 1):
                out.append(idx)
                break
    return out


def _elide_terminal_e(stem: str) -> str:
    """Elide a terminal 'e' before a vowel-initial suffix (P-16.3.3): naphthalene
    -> naphthalen (before -one); kept before -dione/-trione (consonant)."""
    return stem[:-1] if stem.endswith('e') else stem


def _max_oxo_matching(adj, nodes, locant):
    """Max-cardinality matching over ``nodes`` (ring sub-graph ``adj``); among
    maximum matchings, minimise the sorted unmatched-locant tuple. Returns
    ``(matched_set, unmatched_set)``. The matched atoms pair into reduced ring
    double bonds (hydro); the unmatched atoms are the indicated-H positions."""
    from functools import lru_cache
    nodes = frozenset(nodes)

    @lru_cache(maxsize=None)
    def rec(avail):
        if not avail:
            return (0, (), frozenset())
        a = min(avail)
        rest = avail - {a}
        best = rec(rest)
        best = (best[0], tuple(sorted(best[1] + (_locant_key(locant[a]),))), best[2])
        for b in adj[a]:
            if b in rest:
                s, uk, mset = rec(rest - {b})
                cand = (s + 1, uk, mset | {a, b})
                if (cand[0] > best[0]) or (cand[0] == best[0] and cand[1] < best[1]):
                    best = cand
        return best

    _, _, matched = rec(nodes)
    return set(matched), set(nodes) - set(matched)


def _mancude_ring_parent(mol, ring_atoms):
    """Aromatic mancude parent of the whole ring system (drop every exocyclic
    bond, force all ring atoms+bonds aromatic, re-sanitise). Returns
    ``(parent_mol, old_to_new)`` or ``(None, None)`` if it cannot aromatize."""
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


def _resolve_oxo_parent(mol, ring_atoms):
    """Resolve the mancude parent of a ring-ketone's ring system.

    Returns ``(parent_name, [(locant_map, parentH_map), ...])`` or ``(None, None)``.
    ``locant_map``: mol-atom -> IUPAC locant; ``parentH_map``: mol-atom -> the H
    count of the corresponding mancude-parent atom. parentH is paired PER
    candidate numbering because for an intrinsic-indicated-H parent (1H-indene,
    9H-fluorene) the indicated-H carbon's H count (2) is alignment-dependent —
    the carbonyl must align with that >CH2 (P-64.2.2.2.2).
    """
    from .heterocycles import (
        name_heterocycle, _macrocycle_ordered_ring, get_heteroatom_priority,
    )
    from ..data.polycyclic_data import POLYCYCLIC_DATA
    from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

    ring_set = set(ring_atoms)
    n = len(ring_set)
    rings = mol.GetRingInfo().AtomRings()

    # --- monocyclic: build the mancude (aromatic) parent + free numbering ---
    if len(rings) == 1:
        if all(mol.GetAtomWithIdx(i).GetSymbol() == 'C' for i in ring_set):
            return None, None  # carbocyclic monocycle -> cycloalkanone path
        parent, old_to_new = _mancude_ring_parent(mol, ring_set)
        if parent is None:
            return None, None
        pr = parent.GetRingInfo().AtomRings()
        if len(pr) != 1:
            return None, None
        pname = name_heterocycle(parent, pr[0])
        if not pname:
            return None, None
        parentH = {a: parent.GetAtomWithIdx(old_to_new[a]).GetTotalNumHs() for a in ring_set}
        ordered = _macrocycle_ordered_ring(mol, ring_set)
        if ordered is None or len(ordered) != n:
            return None, None
        het = {i for i in ring_set if mol.GetAtomWithIdx(i).GetSymbol() != 'C'}
        nums = []
        for start in range(n):
            for direction in (1, -1):
                seq = [ordered[(start + k * direction) % n] for k in range(n)]
                nums.append({a: idx + 1 for idx, a in enumerate(seq)})

        def _hk(loc):
            return (tuple(sorted(loc[a] for a in het)),
                    tuple(sorted((get_heteroatom_priority(mol.GetAtomWithIdx(a).GetSymbol()), loc[a]) for a in het)))
        if het:
            best_h = min(_hk(l) for l in nums)
            nums = [l for l in nums if _hk(l) == best_h]
        return pname, [(l, parentH) for l in nums]

    # --- fused: bond-generic skeleton match vs PAH then fused-heterocycle data ---
    cands = []
    for nm, e in POLYCYCLIC_DATA.items():
        if e.get('iupac_numbering') and e.get('canonical_smiles'):
            cands.append((nm, e['canonical_smiles'], e['iupac_numbering']))
    for cs, e in FUSED_HETEROCYCLE_DATA.items():
        if e.get('iupac_locants'):
            cands.append((e['name'], cs, e['iupac_locants']))
    for nm, cs, numbering in cands:
        cmol = Chem.MolFromSmiles(cs)
        if cmol is None or cmol.GetNumAtoms() != n:
            continue
        params = Chem.AdjustQueryParameters.NoAdjustments()
        params.makeBondsGeneric = True
        params.aromatizeIfPossible = False
        params.adjustDegree = False
        q = Chem.AdjustQueryProperties(cmol, params)
        pairs = []
        for match in mol.GetSubstructMatches(q, uniquify=False):
            a2l = {match[c]: loc for c, loc in numbering.items() if c < len(match)}
            if set(a2l) != ring_set:
                continue
            if not all(mol.GetAtomWithIdx(match[c]).GetAtomicNum() == cmol.GetAtomWithIdx(c).GetAtomicNum()
                       for c in numbering if c < len(match)):
                continue
            pH = {match[c]: cmol.GetAtomWithIdx(c).GetTotalNumHs() for c in numbering if c < len(match)}
            pairs.append((a2l, pH))
        if pairs:
            return nm, pairs
    return None, None


def name_cyclic_oxo_compound(mol: Chem.Mol) -> Optional[str]:
    """Name an UNSUBSTITUTED cyclic ketone / dione on a mancude ring system,
    emitting the preferred IUPAC name with added indicated hydrogen and/or hydro
    prefixes (IUPAC P-31.1.4 / P-58.2 / P-64.2.2.2 / P-14.7.2).

    Worked examples (all OPSIN-2.9.0 round-trip + Blue-Book verified):
        ``O=c1cccc[nH]1``              -> ``pyridin-2(1H)-one``
        ``O=c1ccc2ccccc2[nH]1``        -> ``quinolin-2(1H)-one``
        ``O=c1c2ccccc2[nH]c2ccccc12``  -> ``acridin-9(10H)-one``
        ``O=C1CC=Cc2ccccc21``          -> ``naphthalen-1(2H)-one``
        ``O=C1CCCc2ccccc21``           -> ``3,4-dihydronaphthalen-1(2H)-one``
        ``O=C1C=CC(=O)c2ccccc21``      -> ``naphthalene-1,4-dione``     (no added-H, P-58.2.2.3)
        ``O=c1[nH]c(=O)c2ccccc2[nH]1`` -> ``quinazoline-2,4(1H,3H)-dione``
        ``O=C1C=Cc2ccccc21``           -> ``1H-inden-1-one``           (intrinsic-IH parent)
        ``O=C1CCc2ccccc21``            -> ``2,3-dihydro-1H-inden-1-one``

    Method: identify the mancude parent + numbering; the carbonyl C(s) take the
    -one/-dione suffix; ring atoms carrying an EXTRA hydrogen vs the mancude
    parent (an N-H or a >CH2) split, by a maximum matching of the ring graph,
    into hydro positions (matched pairs = reduced ring C=C) and added-indicated-H
    positions (unmatched, cited as ``(nH)`` after the suffix locant). Lowest
    locants go to the suffix, then added-IH, then hydro (P-58.2.2.2 / P-31.1).

    Tightly fail-closed (returns None — never a wrong name):
      * >= 1 ring carbonyl; UNSUBSTITUTED (ring + carbonyl O's only);
      * the ring must retain residual unsaturation (an aromatic ring atom OR a
        non-carbonyl ring C=C) — a fully-saturated ring carbonyl is a saturated
        lactam/lactone/ketone named on the saturated parent (piperidin-2-one,
        oxolan-2-one), NOT here;
      * the whole molecule has no retained PIN name (so uracil / maleimide are
        left to their retained entries);
      * the mancude parent must resolve (aromatic monocycle, or a stored PAH /
        fused-heterocycle skeleton).
    """
    from ..data.retained_names import get_retained_name

    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        return None
    ring_set: Set[int] = set()
    for r in rings:
        ring_set.update(r)

    carbonyls = set(_ring_carbonyl_carbons(mol, ring_set))
    if not carbonyls:
        return None
    # The carbonyl O's belong to the suffix, not to substituents.
    carbonyl_oxygens: Set[int] = set()
    for c in carbonyls:
        ca = mol.GetAtomWithIdx(c)
        for bond in ca.GetBonds():
            o = bond.GetOtherAtom(ca)
            if (bond.GetBondType() == Chem.BondType.DOUBLE and o.GetSymbol() == 'O'
                    and o.GetIdx() not in ring_set and o.GetDegree() == 1):
                carbonyl_oxygens.add(o.GetIdx())
    substituent_atoms = {a.GetIdx() for a in mol.GetAtoms()
                         if a.GetIdx() not in ring_set and a.GetIdx() not in carbonyl_oxygens}
    substituted = bool(substituent_atoms)
    if substituted:
        # Substituted ring-ketones are SCOPED to a MONOCYCLIC ring whose
        # substituents are hydrocarbon/halogen only: then the ring ketone is
        # unambiguously the principal group (no heteroatom substituent can be
        # senior to or rival the ketone), and the existing heterocycle
        # substituent assembler can render the prefixes against this numbering.
        # Fused-substituted + heteroatom-substituent ring-ketones fail closed.
        if len(rings) != 1:
            return None
        _ALKYL_HALO = {'C', 'F', 'Cl', 'Br', 'I'}
        if any(mol.GetAtomWithIdx(a).GetSymbol() not in _ALKYL_HALO for a in substituent_atoms):
            return None
    if get_retained_name(Chem.MolToSmiles(mol)):
        return None  # a retained PIN (uracil, maleimide) owns the name

    # Residual unsaturation required (else fully-saturated lactam/lactone/ketone).
    has_residual = any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_set)
    if not has_residual:
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if (i in ring_set and j in ring_set
                        and i not in carbonyls and j not in carbonyls):
                    has_residual = True
                    break
    if not has_residual:
        return None

    parent_name, pairs = _resolve_oxo_parent(mol, ring_set)
    if not parent_name:
        return None

    # SAT = the ring atoms that are SATURATED relative to the mancude parent
    # (added-IH + hydro), detected STRUCTURALLY: non-carbonyl ring C/N that are
    # NOT in a ring double bond in the KEKULISED molecule. Structural (not H-count)
    # so a SUBSTITUTED indicated-H position still counts (1-methylpyridin-2(1H)-one
    # keeps its (1H); the parent O/S never enters — only C/N can be a hydro/IH
    # position). Independent of numbering, so computed once.
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return None
    mol_ring_db: Set[int] = set()
    for bond in kek.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in ring_set and j in ring_set:
                mol_ring_db.add(i)
                mol_ring_db.add(j)
    sat = {i for i in ring_set
           if i not in carbonyls and i not in mol_ring_db
           and mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'N')}
    adj = {i: set() for i in sat}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in sat and j in sat:
            adj[i].add(j)
            adj[j].add(i)

    best = None
    for loc, parentH in pairs:
        matched, unmatched = _max_oxo_matching(adj, sat, loc)
        added_ih, hydro = unmatched, matched
        # Substituent locants are the LAST numbering criterion (P-14.4: after the
        # suffix, added-IH and hydro).
        sub_ring = {i for i in ring_set
                    if any(nb.GetIdx() in substituent_atoms
                           for nb in mol.GetAtomWithIdx(i).GetNeighbors())}
        key = (sorted(_locant_key(loc[c]) for c in carbonyls),
               sorted(_locant_key(loc[a]) for a in added_ih),
               sorted(_locant_key(loc[a]) for a in hydro),
               sorted(_locant_key(loc[a]) for a in sub_ring))
        if best is None or key < best[0]:
            best = (key, loc, added_ih, hydro)
    if best is None:
        return None
    _, loc, added_ih, hydro = best

    carb_sorted = sorted(carbonyls, key=lambda c: _locant_key(loc[c]))
    carb_str = ','.join(_locant_display(loc[c]) for c in carb_sorted)
    suffix = {1: 'one', 2: 'dione', 3: 'trione', 4: 'tetrone'}.get(len(carbonyls))
    if suffix is None:
        return None
    if added_ih:
        ih_sorted = sorted(added_ih, key=lambda a: _locant_key(loc[a]))
        ih_str = '(' + ','.join(f"{_locant_display(loc[a])}H" for a in ih_sorted) + ')'
    else:
        ih_str = ''
    # P-16.3.3: elide terminal 'e' only before the vowel-initial '-one' suffix.
    elide = parent_name.endswith('e') and len(carbonyls) == 1
    stem = parent_name[:-1] if elide else parent_name
    name = f"{stem}-{carb_str}{ih_str}-{suffix}"
    if hydro:
        prefix = SATURATION_PREFIXES.get(len(hydro))
        if prefix is None:
            return None
        hy_sorted = sorted(hydro, key=lambda a: _locant_key(loc[a]))
        sep = '-' if name[:1].isdigit() else ''  # hyphen before a digit-initial parent (1H-inden...)
        name = f"{','.join(_locant_display(loc[a]) for a in hy_sorted)}-{prefix}{sep}{name}"

    # Substituents (monocyclic, hydrocarbon/halogen): render them as alphabetized
    # prefixes against THIS numbering, with the carbonyl 'oxo' excluded (it is the
    # -one/-dione suffix), reusing the heterocycle substituent assembler. The
    # base `name` (parent + added-IH + suffix, possibly hydro-prefixed) is passed
    # as the parent so prefixes are prepended -> e.g. 5-methylpyridin-2(1H)-one.
    if substituted:
        from .heterocycles import (
            get_heterocycle_substituents, name_substituted_heterocycle,
        )
        oriented = [a for a, _ in sorted(loc.items(), key=lambda kv: _locant_key(kv[1]))]
        subs = get_heterocycle_substituents(mol, list(ring_set), oriented, loc)
        filtered = {}
        for locant, slist in subs.items():
            keep = [s for s in slist
                    if s.get('hetero_name') != 'oxo'
                    and not any(a in carbonyl_oxygens for a in s.get('atoms', []))]
            if keep:
                filtered[locant] = keep
        if filtered:
            name = name_substituted_heterocycle(mol, list(ring_set), name, filtered, loc)
    return name


# Back-compat alias: the original (narrower) Phase-2 entry point, now backed by
# the general cyclic-oxo engine.
def name_ring_ketone_with_added_indicated_h(mol: Chem.Mol) -> Optional[str]:
    return name_cyclic_oxo_compound(mol)
