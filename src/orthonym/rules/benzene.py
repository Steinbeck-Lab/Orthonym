"""
Benzene naming rules according to IUPAC 2013 (Blue Book).

Handles:
- Monosubstituted benzenes (chlorobenzene, nitrobenzene)
- Polysubstituted benzenes with numeric locants (1,4-dimethylbenzene)
- Benzene ring orientation for lowest locants
- Alphabetical ordering of substituents
- Suffix functional groups on benzene (carboxylic acid, amide, sulfonamide, etc.)

IUPAC 2013 PIN Rules:
- Numeric locants are REQUIRED (not ortho/meta/para)
- Toluene is retained ONLY for unsubstituted methylbenzene
- Substituted methylbenzene uses "methylbenzene" (not "toluene")
- Position 1 assigned to give lowest locants via first-point-of-difference
- Ring-attached principal groups use suffix form (P-65.1.2)
- benzamide = retained name for C6H5CONH2 (P-66.1.1.1)
"""

from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from rdkit import Chem

from .locants import compare_locant_sets as _compare_locant_sets  # IM-02
from ..assembly.naming_utils import (
    get_alkyl_name,
    get_multiplier_prefix,
    format_substituent_prefix,
    alpha_sort_key,
    is_complex_substituent,
    should_omit_locant_one,
)


# Mapping from substituent atom symbol/pattern to prefix name
# Key: (symbol, hybridization/bond_info) or simple symbol
# Value: prefix name
SUBSTITUENT_PREFIXES = {
    # Halogens
    "F": "fluoro",
    "Cl": "chloro",
    "Br": "bromo",
    "I": "iodo",
    # Common groups - these are detected by the get_benzene_substituents function
    # based on the substituent structure
}

# Priority order for suffix functional groups on benzene
# Highest priority first (carboxylic acid > sulfonamide > amide > nitrile > aldehyde)
_SUFFIX_PRIORITY = [
    'carboxylic acid',
    'sulfonic acid',
    'sulfonamide',
    'carbonyl chloride',
    'carboxamide',
    'carbonitrile',
    'carbaldehyde',
    'ol',  # ASML-13: hydroxyl as suffix when principal group on benzene
]

# Prefix forms for suffix FGs when they are NOT the principal group
_SUFFIX_TO_PREFIX = {
    'carboxylic acid': 'carboxy',
    'sulfonic acid': 'sulfo',
    'sulfonamide': 'sulfamoyl',
    'carbonyl chloride': 'carbonochloridoyl',
    'carboxamide': 'carbamoyl',
    'carbonitrile': 'cyano',
    'carbaldehyde': 'formyl',
    'ol': 'hydroxy',  # ASML-13: when OH is not principal, use prefix form
}


# Pre-compiled SMARTS for suffix FG identification (avoid per-call recompilation)
_BENZENE_FG_SMARTS = {
    'acid': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),
    'amide': Chem.MolFromSmarts('[CX3](=O)[NX3H2]'),
    'sec_amide': Chem.MolFromSmarts('[CX3](=O)[NX3H1][#6]'),
    'tert_amide': Chem.MolFromSmarts('[CX3](=O)[NX3]([#6])[#6]'),
    'aldehyde': Chem.MolFromSmarts('[CX3H1](=O)'),
    'nitrile': Chem.MolFromSmarts('[CX2]#[NX1]'),
    'acid_cl': Chem.MolFromSmarts('[CX3](=O)[Cl]'),
    'thio_acid': Chem.MolFromSmarts('[CX3](=O)[SX2H1]'),
    'sulfonamide': Chem.MolFromSmarts('[SX4](=O)(=O)[NX3H2]'),
    'sulfonic': Chem.MolFromSmarts('[SX4](=O)(=O)[OX2H1]'),
}


def is_benzene_ring(mol, ring_atoms: Tuple[int, ...]) -> bool:
    """
    Check if a ring is a benzene ring (6-membered aromatic carbocycle).

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring

    Returns:
        True if ring is benzene (6 aromatic carbons)
    """
    if len(ring_atoms) != 6:
        return False

    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        # Must be carbon
        if atom.GetSymbol() != 'C':
            return False
        # Must be aromatic
        if not atom.GetIsAromatic():
            return False

    return True


def get_benzene_ring(mol) -> Optional[Tuple[int, ...]]:
    """
    Find the benzene ring in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of atom indices in the benzene ring, or None if not found
    """
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if is_benzene_ring(mol, ring):
            return ring
    return None


def get_benzene_substituents(mol, ring_atoms: Tuple[int, ...]) -> Dict[int, List[Dict]]:
    """
    Find substituents attached to each benzene carbon.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the benzene ring

    Returns:
        Dict mapping ring atom index to list of substituent info dicts.
        Each dict has keys: 'name' (str), 'atoms' (list of atom indices)
    """
    from ..perception.rings import get_containing_ring_system

    # Use the complete ring system as BFS boundary (IUPAC P-25.3)
    # Prevents walking into fused partner rings
    ring_set = set(get_containing_ring_system(mol, ring_atoms))
    substituents: Dict[int, List[Dict]] = defaultdict(list)

    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip ring atoms
            if nbr_idx in ring_set:
                continue

            # Identify the substituent
            sub_info = _identify_substituent(mol, nbr_idx, ring_set)
            if sub_info:
                substituents[ring_idx].append(sub_info)

    return dict(substituents)


def _bfs_substituent_atoms(mol, start_idx: int, ring_atoms: Set[int]) -> List[int]:
    """
    BFS to collect all atom indices in a substituent starting from start_idx.

    Does not constrain to pure alkyl -- collects all atoms outside ring_atoms.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the first atom in the substituent
        ring_atoms: Set of ring atom indices to exclude

    Returns:
        List of all atom indices in the substituent (including start_idx)
    """
    visited = {start_idx}
    queue = deque([start_idx])
    all_atoms = []

    while queue:
        current_idx = queue.popleft()
        all_atoms.append(current_idx)

        current_atom = mol.GetAtomWithIdx(current_idx)
        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_atoms:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    return all_atoms


def _identify_suffix_fg_on_benzene(
    mol, start_idx: int, sub_atoms: List[int], ring_atoms: Set[int]
) -> Optional[Dict]:
    """
    Identify suffix-type functional groups attached to benzene ring.

    Checks if the substituent starting at start_idx is a functional group
    that should be expressed as a suffix on the benzene parent name.

    Handles both C-based FGs (carboxylic acid, amide, aldehyde, nitrile, acid chloride)
    and S-based FGs (sulfonamide, sulfonic acid).

    Args:
        mol: RDKit Mol object
        start_idx: Index of the atom directly attached to ring
        sub_atoms: All atom indices in the substituent
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name', 'suffix_name', 'is_suffix', 'atoms', and optionally
        'n_substituents' for N-substituted amides; or None if not a suffix FG.
    """
    start_atom = mol.GetAtomWithIdx(start_idx)
    symbol = start_atom.GetSymbol()

    # Carbon-based suffix FGs
    if symbol == 'C':
        # Carboxylic acid: C(=O)(OH) -- check before amide!
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['acid']):
            if match[0] == start_idx:
                return {
                    'name': 'carboxylic acid', 'suffix_name': 'carboxylic acid',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Primary amide: C(=O)(NH2)
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['amide']):
            if match[0] == start_idx:
                return {
                    'name': 'carboxamide', 'suffix_name': 'carboxamide',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Secondary amide: C(=O)(NHR)
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['sec_amide']):
            if match[0] == start_idx:
                n_subs = _detect_n_substituents(mol, match, ring_atoms)
                return {
                    'name': 'carboxamide', 'suffix_name': 'carboxamide',
                    'is_suffix': True, 'atoms': sub_atoms,
                    'n_substituents': n_subs,
                }

        # Tertiary amide: C(=O)(NR2)
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['tert_amide']):
            if match[0] == start_idx:
                n_subs = _detect_n_substituents(mol, match, ring_atoms)
                return {
                    'name': 'carboxamide', 'suffix_name': 'carboxamide',
                    'is_suffix': True, 'atoms': sub_atoms,
                    'n_substituents': n_subs,
                }

        # Aldehyde: C(=O)H
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['aldehyde']):
            if match[0] == start_idx:
                return {
                    'name': 'carbaldehyde', 'suffix_name': 'carbaldehyde',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Nitrile: C#N
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['nitrile']):
            if match[0] == start_idx:
                return {
                    'name': 'carbonitrile', 'suffix_name': 'carbonitrile',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Acid chloride: C(=O)Cl
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['acid_cl']):
            if match[0] == start_idx:
                return {
                    'name': 'carbonyl chloride', 'suffix_name': 'carbonyl chloride',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Thiocarboxylic S-acid: C(=O)(SH)
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['thio_acid']):
            if match[0] == start_idx:
                return {
                    'name': 'carbothioic S-acid', 'suffix_name': 'carbothioic S-acid',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

    # Sulfur-based suffix FGs
    if symbol == 'S':
        # Sulfonamide: S(=O)(=O)(NH2)
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['sulfonamide']):
            if match[0] == start_idx:
                return {
                    'name': 'sulfonamide', 'suffix_name': 'sulfonamide',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

        # Sulfonic acid: S(=O)(=O)(OH)
        for match in mol.GetSubstructMatches(_BENZENE_FG_SMARTS['sulfonic']):
            if match[0] == start_idx:
                return {
                    'name': 'sulfonic acid', 'suffix_name': 'sulfonic acid',
                    'is_suffix': True, 'atoms': sub_atoms,
                }

    return None


def _detect_n_substituents(mol, amide_match: Tuple[int, ...], ring_atoms: Set[int]) -> List[str]:
    """
    Detect N-alkyl substituents on an amide nitrogen.

    Args:
        mol: RDKit Mol object
        amide_match: SMARTS match tuple (C, O/N indices depending on pattern)
        ring_atoms: Set of ring atom indices

    Returns:
        List of alkyl names attached to nitrogen (e.g., ['methyl'] or ['methyl', 'methyl'])
    """
    # Find the nitrogen atom in the amide
    c_idx = amide_match[0]
    c_atom = mol.GetAtomWithIdx(c_idx)

    n_atom = None
    for nbr in c_atom.GetNeighbors():
        if nbr.GetSymbol() == 'N' and nbr.GetIdx() not in ring_atoms:
            n_atom = nbr
            break

    if n_atom is None:
        return []

    n_idx = n_atom.GetIdx()
    alkyl_names = []

    for nbr in n_atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == c_idx or nbr_idx in ring_atoms:
            continue
        if nbr.GetSymbol() == 'C':
            # Collect alkyl chain
            alkyl_atoms, carbon_count = _collect_pure_alkyl(
                mol, nbr_idx, ring_atoms | {n_idx, c_idx}
            )
            if alkyl_atoms is not None and carbon_count > 0:
                from ..assembly.substituent_naming import name_substituent_fragment
                rec_name = name_substituent_fragment(
                    mol, alkyl_atoms, nbr_idx, list(ring_atoms | {n_idx, c_idx})
                )
                if rec_name:
                    alkyl_names.append(rec_name)
                else:
                    try:
                        alkyl_names.append(get_alkyl_name(carbon_count))
                    except (ValueError, KeyError):
                        pass

    alkyl_names.sort()
    return alkyl_names


def _identify_substituent(mol, start_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify a substituent starting from an atom attached to the ring.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the atom attached to the ring
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name' and 'atoms', or None if unknown
    """
    start_atom = mol.GetAtomWithIdx(start_idx)
    symbol = start_atom.GetSymbol()

    # Halogens - single atom substituents
    if symbol in SUBSTITUENT_PREFIXES:
        return {
            'name': SUBSTITUENT_PREFIXES[symbol],
            'atoms': [start_idx]
        }

    # For C and S atoms, check suffix FGs first
    if symbol in ('C', 'S'):
        sub_atoms = _bfs_substituent_atoms(mol, start_idx, ring_atoms)
        suffix_fg = _identify_suffix_fg_on_benzene(mol, start_idx, sub_atoms, ring_atoms)
        if suffix_fg:
            return suffix_fg

    # Nitrogen-based groups
    if symbol == 'N':
        return _identify_nitrogen_group(mol, start_idx, ring_atoms)

    # Oxygen-based groups
    if symbol == 'O':
        return _identify_oxygen_group(mol, start_idx, ring_atoms)

    # Sulfur-based groups (non-suffix fallback)
    if symbol == 'S':
        return _identify_sulfur_group(mol, start_idx, ring_atoms)

    # Phosphorus-based groups: generate phosphanyl prefix (IUPAC P-68)
    if symbol == 'P':
        from ..rules.phosphorus import get_phosphanyl_prefix
        sub_atoms = _bfs_substituent_atoms(mol, start_idx, ring_atoms)
        # Exclude parent ring atoms from P's substituent count so the
        # attachment carbon is not counted (e.g., PPh2 on benzene -> diphenylphosphanyl)
        prefix = get_phosphanyl_prefix(mol, start_idx, exclude_atoms=ring_atoms)
        if prefix:
            return {'name': prefix, 'atoms': sub_atoms}

    # Carbon-based groups (alkyl or functionalized chain) - fallback for non-suffix C
    if symbol == 'C':
        # Check for nitrile C#N pattern FIRST (BUG-2 fix)
        nitrile_result = _identify_nitrile_group(mol, start_idx, ring_atoms)
        if nitrile_result:
            return nitrile_result

        # Try simple alkyl
        alkyl_result = _identify_alkyl_group(mol, start_idx, ring_atoms)
        if alkyl_result:
            return alkyl_result

        # Try functionalized chain (chains with FG like -CCCC(=O)O)
        func_chain = _identify_functionalized_chain(mol, start_idx, ring_atoms)
        if func_chain:
            return func_chain

        # Fallback: collect all atoms and produce a generic substituent name
        # Handles haloalkyl groups like CF3 (trifluoromethyl)
        sub_atoms = _bfs_substituent_atoms(mol, start_idx, ring_atoms)
        generic = _identify_generic_carbon_substituent(mol, start_idx, sub_atoms, ring_atoms)
        if generic:
            return generic

        # Phase 86: Universal pipeline fallback for complex C-substituents.
        # By this point, suffix FG detection (line 387-389) has already returned
        # for recognized suffix patterns (acid, amide, nitrile, aldehyde, etc.).
        # We only reach here for C-substituents that are NOT suffix FGs and NOT
        # simple alkyls/haloalkyls -- e.g., cyanomethyl, carbamoylmethyl, etc.
        #
        # Guards:
        # 1. Carbonyl guard: if start C has C=O, it's a carbonyl carbon
        #    (ester C(=O)OR, ketone C(=O)R). These are FG features handled
        #    by dedicated handlers; naming them as benzene substituents causes
        #    routing regressions. All suffix-type carbonyls (acid, amide,
        #    aldehyde, acid chloride) are already caught by suffix FG check.
        # 2. Size guard: fragments larger than 10 atoms are major structural
        #    features (fused systems, long chains), not genuine substituents.
        if sub_atoms:
            # Check if start carbon is a carbonyl (has C=O bond)
            _is_carbonyl = False
            for _nbr in start_atom.GetNeighbors():
                if _nbr.GetIdx() in ring_atoms:
                    continue
                if _nbr.GetSymbol() == 'O':
                    _bond = mol.GetBondBetweenAtoms(start_idx, _nbr.GetIdx())
                    if _bond and _bond.GetBondType() == Chem.BondType.DOUBLE:
                        _is_carbonyl = True
                        break

            if not _is_carbonyl:
                _MAX_FALLBACK_ATOMS = 10
                if len(sub_atoms) <= _MAX_FALLBACK_ATOMS:
                    from ..assembly.substituent_enumerator import name_substituent
                    from ..assembly.naming_utils import needs_brackets
                    prefix_name = name_substituent(mol, set(sub_atoms), start_idx)
                    if prefix_name and prefix_name != "substituent":
                        # Wrap in parentheses if compound name per IUPAC P-14.5.2
                        is_compound = needs_brackets(prefix_name)
                        if is_compound and not (prefix_name.startswith('(') and prefix_name.endswith(')')):
                            prefix_name = f'({prefix_name})'
                        return {
                            'name': prefix_name,
                            'atoms': sub_atoms,
                            'is_complex': is_compound,
                        }

    return None


def _identify_nitrile_group(mol, c_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify nitrile (C#N) substituent attached to benzene.

    The nitrile carbon is directly attached to the ring. We check if this carbon
    has a triple bond to nitrogen and no other heavy atom neighbors (besides the ring).

    Args:
        mol: RDKit Mol object
        c_idx: Index of the carbon atom attached to the ring
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name': 'nitrile', 'type': 'functional', 'atoms': [...] or None
    """
    c_atom = mol.GetAtomWithIdx(c_idx)

    # Check neighbors of the nitrile carbon
    for neighbor in c_atom.GetNeighbors():
        if neighbor.GetIdx() in ring_atoms:
            continue

        # Check for triple bond to nitrogen
        bond = mol.GetBondBetweenAtoms(c_idx, neighbor.GetIdx())
        if (neighbor.GetSymbol() == 'N' and
                bond and bond.GetBondType() == Chem.BondType.TRIPLE):
            # Verify the nitrogen has no other heavy atom neighbors (just the C#N)
            n_atom = neighbor
            n_heavy_neighbors = [n for n in n_atom.GetNeighbors()
                                 if n.GetSymbol() != 'H' and n.GetIdx() != c_idx]
            if not n_heavy_neighbors:
                return {
                    'name': 'nitrile',
                    'type': 'functional',
                    'atoms': [c_idx, neighbor.GetIdx()]
                }

    return None


def _detect_benzene_nitrile(mol, ring_atoms: Tuple[int, ...]) -> Dict:
    """
    Detect if benzene ring has a nitrile substituent.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the benzene ring

    Returns:
        Dict with 'is_nitrile': bool, 'nitrile_positions': list of ring atom indices
        that have nitrile attached
    """
    ring_set = set(ring_atoms)
    nitrile_positions = []

    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in ring_set:
                continue

            # Check if this neighbor is a nitrile carbon
            if neighbor.GetSymbol() == 'C':
                result = _identify_nitrile_group(mol, nbr_idx, ring_set)
                if result and result.get('name') == 'nitrile':
                    nitrile_positions.append(ring_idx)
                    break  # Only count once per ring position

    return {
        'is_nitrile': len(nitrile_positions) > 0,
        'nitrile_positions': nitrile_positions
    }


def _identify_nitrogen_group(mol, n_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """Identify nitrogen-based substituent (amino, nitro, N-alkylamino, etc.)."""
    n_atom = mol.GetAtomWithIdx(n_idx)

    # Count neighbors (excluding ring)
    neighbors = [n for n in n_atom.GetNeighbors() if n.GetIdx() not in ring_atoms]

    # Check for nitro group: N with 2 oxygens, positive charge
    if n_atom.GetFormalCharge() == 1:
        o_count = sum(1 for n in neighbors if n.GetSymbol() == 'O')
        if o_count == 2:
            # Nitro group
            atoms = [n_idx] + [n.GetIdx() for n in neighbors if n.GetSymbol() == 'O']
            return {'name': 'nitro', 'atoms': atoms}

    h_count = n_atom.GetTotalNumHs()

    # Simple amino (-NH2)
    if h_count == 2 and len(neighbors) == 0:
        return {'name': 'amino', 'atoms': [n_idx]}

    # N-monoalkyl amino (-NHR): 1 H, 1 carbon neighbor
    # IUPAC 2013: N-alkylamino (e.g., N-methylamino, N-ethylamino)
    if h_count == 1 and len(neighbors) == 1 and neighbors[0].GetSymbol() == 'C':
        alkyl_atoms, carbon_count = _collect_pure_alkyl(mol, neighbors[0].GetIdx(), ring_atoms | {n_idx})
        if alkyl_atoms is not None and carbon_count > 0:
            from ..assembly.substituent_naming import name_substituent_fragment
            alkyl_name = name_substituent_fragment(
                mol, alkyl_atoms, neighbors[0].GetIdx(), list(ring_atoms | {n_idx})
            )
            if alkyl_name is None:
                try:
                    alkyl_name = get_alkyl_name(carbon_count)
                except (ValueError, KeyError):
                    alkyl_name = None
            if alkyl_name:
                # OPSIN treats "anilino" as a simple substituent
                if alkyl_name == 'phenyl':
                    return {
                        'name': 'anilino',
                        'atoms': [n_idx] + alkyl_atoms,
                        'is_complex': False,
                    }
                return {
                    'name': f'(N-{alkyl_name}amino)',
                    'atoms': [n_idx] + alkyl_atoms,
                    'is_complex': True,
                }

    # N,N-dialkyl amino (-NR2): 0 H, 2 carbon neighbors
    # IUPAC 2013: (N,N-dialkylamino) (e.g., (N,N-dimethylamino))
    if h_count == 0 and len(neighbors) == 2:
        c_neighbors = [n for n in neighbors if n.GetSymbol() == 'C']
        if len(c_neighbors) == 2:
            alkyl_names_list = []
            all_sub_atoms = [n_idx]
            for cn in c_neighbors:
                alkyl_atoms, carbon_count = _collect_pure_alkyl(mol, cn.GetIdx(), ring_atoms | {n_idx})
                if alkyl_atoms is None or carbon_count == 0:
                    break
                from ..assembly.substituent_naming import name_substituent_fragment
                aname = name_substituent_fragment(
                    mol, alkyl_atoms, cn.GetIdx(), list(ring_atoms | {n_idx})
                )
                if aname is None:
                    try:
                        aname = get_alkyl_name(carbon_count)
                    except (ValueError, KeyError):
                        break
                alkyl_names_list.append(aname)
                all_sub_atoms.extend(alkyl_atoms)
            else:
                # Both identified - build name with N,N- locants
                alkyl_names_list.sort()
                if alkyl_names_list[0] == alkyl_names_list[1]:
                    from ..assembly.naming_utils import get_multiplier_prefix
                    mp = get_multiplier_prefix(2, alkyl_names_list[0])
                    prefix_name = f'(N,N-{mp}{alkyl_names_list[0]}amino)'
                else:
                    prefix_name = f'(N-{alkyl_names_list[0]}-N-{alkyl_names_list[1]}amino)'
                return {
                    'name': prefix_name,
                    'atoms': all_sub_atoms,
                    'is_complex': True,
                }

    # Nitroso (-NO)
    if h_count == 0 and len(neighbors) == 1 and neighbors[0].GetSymbol() == 'O':
        bond = mol.GetBondBetweenAtoms(n_idx, neighbors[0].GetIdx())
        if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
            return {
                'name': 'nitroso',
                'atoms': [n_idx, neighbors[0].GetIdx()]
            }

    # Fallback: complex N-substituent (non-alkyl chains, heteroatom-containing
    # groups like guanidino, ureido, etc.).  Collect all atoms via BFS and try
    # recursive naming.
    if neighbors:
        sub_atoms = _bfs_substituent_atoms(mol, n_idx, ring_atoms)
        if len(sub_atoms) > 1 and len(sub_atoms) <= 25:
            from ..assembly.substituent_naming import name_substituent_fragment
            frag_name = name_substituent_fragment(
                mol, sub_atoms, n_idx, list(ring_atoms)
            )
            if frag_name:
                return {
                    'name': frag_name,
                    'atoms': sub_atoms,
                    'is_complex': True,
                }

    return None


def _collect_pure_alkyl(mol, start_idx: int, excluded: Set[int]):
    """
    BFS to collect a pure alkyl group (only C/H atoms).

    Returns:
        Tuple of (list of atom indices, carbon_count) or (None, 0) if not pure alkyl.
    """
    visited = {start_idx}
    queue = deque([start_idx])
    all_atoms = []
    carbon_count = 0

    while queue:
        current_idx = queue.popleft()
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        if current_atom.GetSymbol() == 'C':
            carbon_count += 1
        elif current_atom.GetSymbol() != 'H':
            return None, 0  # Not pure alkyl

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in excluded:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    return all_atoms, carbon_count


def _identify_oxygen_group(mol, o_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """Identify oxygen-based substituent (hydroxy, methoxy, alkoxy, hydroperoxy, etc.)."""
    o_atom = mol.GetAtomWithIdx(o_idx)

    # Count neighbors (excluding ring)
    neighbors = [n for n in o_atom.GetNeighbors() if n.GetIdx() not in ring_atoms]

    # Simple hydroxy (-OH)
    h_count = o_atom.GetTotalNumHs()
    if h_count == 1 and len(neighbors) == 0:
        return {'name': 'hydroxy', 'atoms': [o_idx]}

    if len(neighbors) == 1:
        nbr = neighbors[0]
        nbr_symbol = nbr.GetSymbol()

        # Hydroperoxy (-O-OH): oxygen connected to another oxygen with H
        if nbr_symbol == 'O' and nbr.GetTotalNumHs() >= 1:
            return {'name': 'hydroperoxy', 'atoms': [o_idx, nbr.GetIdx()]}

        # Alkoxy (-O-C...): methoxy, ethoxy, propoxy, etc.
        if nbr_symbol == 'C':
            c_atom = nbr
            # Collect the alkyl part
            alkyl_atoms, carbon_count = _collect_pure_alkyl(mol, c_atom.GetIdx(), ring_atoms | {o_idx})
            if alkyl_atoms is not None and carbon_count > 0:
                from ..assembly.substituent_naming import name_substituent_fragment
                alkyl_name = name_substituent_fragment(
                    mol, alkyl_atoms, c_atom.GetIdx(), list(ring_atoms | {o_idx})
                )
                if alkyl_name is None:
                    try:
                        alkyl_name = get_alkyl_name(carbon_count)
                    except (ValueError, KeyError):
                        alkyl_name = None
                if alkyl_name:
                    # methyl -> methoxy, ethyl -> ethoxy, propyl -> propoxy, etc.
                    if alkyl_name.endswith('yl'):
                        oxy_name = alkyl_name[:-2] + 'oxy'
                    else:
                        oxy_name = alkyl_name + 'oxy'
                    return {'name': oxy_name, 'atoms': [o_idx] + alkyl_atoms}

        # Alkoxy fallback for O-C where C is not pure alkyl
        # (aromatic carbon, sugar ring carbon, carbonyl carbon, etc.)
        if nbr_symbol == 'C':
            c_atom = nbr
            sub_atoms = _bfs_substituent_atoms(mol, o_idx, ring_atoms)

            # Case 1: O -> aromatic C in benzene ring -> "phenoxy"
            if c_atom.GetIsAromatic() and c_atom.IsInRing():
                ring_info = mol.GetRingInfo()
                for ring in ring_info.AtomRings():
                    if c_atom.GetIdx() in ring and len(ring) == 6:
                        all_arom_c = all(
                            mol.GetAtomWithIdx(r).GetIsAromatic() and
                            mol.GetAtomWithIdx(r).GetSymbol() == 'C'
                            for r in ring
                        )
                        if all_arom_c:
                            return {'name': 'phenoxy', 'atoms': sub_atoms}

            # Case 2: O -> C(=O)R -> acyloxy group (ester linkage on benzene)
            for cn in c_atom.GetNeighbors():
                if cn.GetIdx() == o_idx:
                    continue
                if cn.GetSymbol() == 'O':
                    bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), cn.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                        # This is an ester: -O-C(=O)-R (acyloxy)
                        # Count carbons in the acyl chain (including the carbonyl C)
                        acyl_excluded = ring_atoms | {o_idx, cn.GetIdx()}
                        acyl_atoms_list, acyl_c = _collect_pure_alkyl(
                            mol, c_atom.GetIdx(), acyl_excluded
                        )
                        if acyl_atoms_list is not None and acyl_c > 0:
                            from ..data.chain_names import get_chain_prefix
                            try:
                                acyl_prefix = get_chain_prefix(acyl_c)
                                acyloxy_name = f"({acyl_prefix}anoyloxy)"
                                return {
                                    'name': acyloxy_name,
                                    'atoms': sub_atoms,
                                    'is_complex': True,
                                }
                            except (ValueError, KeyError):
                                pass
                        break

            # Case 3: O -> non-aromatic ring C (glycoside/sugar etc.)
            if c_atom.IsInRing():
                ring_info = mol.GetRingInfo()
                for ring in ring_info.AtomRings():
                    if c_atom.GetIdx() in ring:
                        ring_size = len(ring)
                        # 5- or 6-membered ring with O in ring -> likely glycoside
                        ring_has_o = any(
                            mol.GetAtomWithIdx(r).GetSymbol() == 'O'
                            for r in ring
                        )
                        if ring_has_o and ring_size in (5, 6):
                            # Use systematic heterocyclic names:
                            # 5-membered with O = oxolane (tetrahydrofuran)
                            # 6-membered with O = oxane (tetrahydropyran)
                            glyco_prefix = "(oxolan-2-yl)oxy" if ring_size == 5 else "(oxan-2-yl)oxy"
                            return {
                                'name': glyco_prefix,
                                'atoms': sub_atoms,
                                'is_complex': True,
                            }
                        break

            # Case 4: O-C that's not pure alkyl but not ring/aromatic
            # Try to count total carbons and make a best-effort alkoxy name
            all_sub = _bfs_substituent_atoms(mol, nbr.GetIdx(), ring_atoms | {o_idx})
            total_c = sum(
                1 for a in all_sub
                if mol.GetAtomWithIdx(a).GetSymbol() == 'C'
            )
            if total_c > 0:
                try:
                    alkyl_name = get_alkyl_name(total_c)
                    if alkyl_name.endswith('yl'):
                        oxy_name = alkyl_name[:-2] + 'oxy'
                    else:
                        oxy_name = alkyl_name + 'oxy'
                    return {'name': oxy_name, 'atoms': [o_idx] + all_sub}
                except (ValueError, KeyError):
                    from ..data.chain_names import get_chain_prefix
                    try:
                        cp = get_chain_prefix(total_c)
                        return {'name': cp + 'yloxy', 'atoms': [o_idx] + all_sub}
                    except (ValueError, KeyError):
                        pass

            # Last resort for O-C where all naming attempts failed.
            # If total_c > 0, use get_chain_prefix to build a systematic name.
            # Bare 'oxy' only for total_c == 0 (no carbon atoms, e.g. O-O or O-N
            # that somehow reached here -- normally handled by earlier branches).
            if total_c > 0:
                from ..data.chain_names import get_chain_prefix
                try:
                    cp = get_chain_prefix(total_c)
                    return {'name': cp + 'yloxy', 'atoms': [o_idx] + all_sub}
                except (ValueError, KeyError):
                    pass
            # total_c == 0: genuinely no carbon in substituent (O-O/O-N cases
            # normally handled by hydroperoxy/nitrooxy above).
            # Return None rather than bare 'oxy' -- OPSIN cannot parse standalone
            # 'oxy' and the caller skips None substituents gracefully.
            return None

        # Fallback for non-C neighbors (e.g., O-N in nitrate esters, O-S, O-P)
        sub_atoms = _bfs_substituent_atoms(mol, o_idx, ring_atoms)
        if len(sub_atoms) > 1:
            # For O-N(=O)=O: (nitrooxy) per IUPAC
            if nbr_symbol == 'N':
                n_atom = nbr
                o_neighbors_of_n = [
                    nb for nb in n_atom.GetNeighbors()
                    if nb.GetSymbol() == 'O' and nb.GetIdx() != o_idx
                ]
                if len(o_neighbors_of_n) >= 2:
                    return {'name': '(nitrooxy)', 'atoms': sub_atoms, 'is_complex': True}
                # O-N without multiple O on N: rare N-oxide-like linkage.
                # No standard IUPAC prefix; return None to avoid bare 'oxy'
                # which OPSIN cannot parse. Caller skips None gracefully.
                return None

            # O-S neighbors: sulfanyloxy / sulfinyloxy / sulfonyloxy
            if nbr_symbol == 'S':
                s_atom = nbr
                # Check oxidation state of sulfur (count =O bonds)
                s_double_o = sum(
                    1 for nb in s_atom.GetNeighbors()
                    if nb.GetSymbol() == 'O' and nb.GetIdx() != o_idx
                    and mol.GetBondBetweenAtoms(s_atom.GetIdx(), nb.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                )
                if s_double_o == 0:
                    # -O-S- : sulfanyloxy (IUPAC P-63.6)
                    return {'name': 'sulfanyloxy', 'atoms': sub_atoms}
                elif s_double_o == 1:
                    # -O-S(=O)- : sulfinyloxy
                    return {'name': 'sulfinyloxy', 'atoms': sub_atoms}
                elif s_double_o >= 2:
                    # -O-S(=O)(=O)- : sulfonyloxy
                    return {'name': 'sulfonyloxy', 'atoms': sub_atoms}

            # O-P neighbors: phosphonooxy (IUPAC P-67.1.3)
            if nbr_symbol == 'P':
                return {'name': 'phosphonooxy', 'atoms': sub_atoms}

            # Other non-C neighbors: return None to avoid bare 'oxy'
            # which OPSIN cannot parse. Caller skips None gracefully.
            return None

    return None


def _identify_sulfur_group(mol, s_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify sulfur-based substituent (thiol/sulfanyl, thioether, etc.).

    Args:
        mol: RDKit Mol object
        s_idx: Index of the sulfur atom attached to the ring
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name' and 'atoms', or None if unknown
    """
    s_atom = mol.GetAtomWithIdx(s_idx)
    neighbors = [n for n in s_atom.GetNeighbors() if n.GetIdx() not in ring_atoms]
    h_count = s_atom.GetTotalNumHs()

    # Simple thiol (-SH): IUPAC prefix = "sulfanyl"
    if h_count >= 1 and len(neighbors) == 0:
        return {'name': 'sulfanyl', 'atoms': [s_idx]}

    # Classify sulfur oxidation state by counting =O neighbors (non-ring)
    # Sulfoxide: S with 1 =O; Sulfone: S with 2 =O; Thioether: S with 0 =O
    o_double_neighbors = []
    c_neighbors = []
    for n in neighbors:
        if n.GetSymbol() == 'O':
            bond = mol.GetBondBetweenAtoms(s_idx, n.GetIdx())
            if bond and bond.GetBondTypeAsDouble() == 2.0:
                o_double_neighbors.append(n)
        elif n.GetSymbol() == 'C':
            c_neighbors.append(n)

    # Sulfoxide (-S(=O)-R) or Sulfone (-S(=O)(=O)-R): one C neighbor + =O neighbors
    if len(c_neighbors) == 1 and len(o_double_neighbors) in (1, 2):
        c_nbr = c_neighbors[0]
        alkyl_atoms, carbon_count = _collect_pure_alkyl(
            mol, c_nbr.GetIdx(), ring_atoms | {s_idx} | {o.GetIdx() for o in o_double_neighbors}
        )
        alkyl_name = None
        if alkyl_atoms is not None and carbon_count > 0:
            from ..assembly.substituent_naming import name_substituent_fragment
            alkyl_name = name_substituent_fragment(
                mol, alkyl_atoms, c_nbr.GetIdx(),
                list(ring_atoms | {s_idx} | {o.GetIdx() for o in o_double_neighbors})
            )
            if alkyl_name is None:
                try:
                    alkyl_name = get_alkyl_name(carbon_count)
                except (ValueError, KeyError):
                    alkyl_name = None
        if alkyl_name:
            o_atom_idxs = [o.GetIdx() for o in o_double_neighbors]
            all_sub_atoms = [s_idx] + o_atom_idxs + alkyl_atoms
            if len(o_double_neighbors) == 2:
                # Sulfone: alkylsulfonyl
                return {'name': f'{alkyl_name}sulfonyl', 'atoms': all_sub_atoms}
            else:
                # Sulfoxide: alkylsulfinyl
                return {'name': f'{alkyl_name}sulfinyl', 'atoms': all_sub_atoms}

    # Thioether (-S-R): named as alkylsulfanyl (methylsulfanyl, etc.)
    if len(c_neighbors) == 1 and len(o_double_neighbors) == 0:
        alkyl_atoms, carbon_count = _collect_pure_alkyl(
            mol, c_neighbors[0].GetIdx(), ring_atoms | {s_idx}
        )
        if alkyl_atoms is not None and carbon_count > 0:
            from ..assembly.substituent_naming import name_substituent_fragment
            alkyl_name = name_substituent_fragment(
                mol, alkyl_atoms, c_neighbors[0].GetIdx(), list(ring_atoms | {s_idx})
            )
            if alkyl_name is None:
                try:
                    alkyl_name = get_alkyl_name(carbon_count)
                except (ValueError, KeyError):
                    alkyl_name = None
            if alkyl_name:
                return {
                    'name': f'{alkyl_name}sulfanyl',
                    'atoms': [s_idx] + alkyl_atoms,
                }

    # Disulfanyl (-S-SH) or dithio linkages
    if len(neighbors) == 1 and neighbors[0].GetSymbol() == 'S':
        other_s = neighbors[0]
        if other_s.GetTotalNumHs() >= 1:
            return {'name': 'disulfanyl', 'atoms': [s_idx, other_s.GetIdx()]}

    # Generic fallback for other S-based substituents
    sub_atoms = _bfs_substituent_atoms(mol, s_idx, ring_atoms)
    if sub_atoms:
        return {'name': 'sulfanyl', 'atoms': sub_atoms}

    return None


def _identify_generic_carbon_substituent(
    mol, start_idx: int, sub_atoms: List[int], ring_atoms: Set[int]
) -> Optional[Dict]:
    """
    Fallback identification for carbon-based substituents that are not
    simple alkyl or known functionalized chains.

    Handles haloalkyl groups (trifluoromethyl, dichloromethyl, etc.)
    and other mixed-atom carbon substituents.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the carbon attached to the ring
        sub_atoms: All atom indices in the substituent
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name' and 'atoms', or None
    """
    start_atom = mol.GetAtomWithIdx(start_idx)

    # Count halogens and carbons
    halogen_counts = {'F': 0, 'Cl': 0, 'Br': 0, 'I': 0}
    carbon_count = 0
    other_count = 0

    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym == 'C':
            carbon_count += 1
        elif sym in halogen_counts:
            halogen_counts[sym] += 1
        elif sym != 'H':
            other_count += 1

    # Pure haloalkyl: only C and halogens (no other heteroatoms)
    if other_count == 0 and any(halogen_counts.values()):
        total_halogens = sum(halogen_counts.values())

        if carbon_count == 1:
            # Single C with halogens: trifluoromethyl, dichloromethyl, etc.
            halogen_parts = []
            halogen_prefix_map = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
            for hal in ['Br', 'Cl', 'F', 'I']:  # alphabetical by prefix
                count = halogen_counts[hal]
                if count > 0:
                    mp = get_multiplier_prefix(count, halogen_prefix_map[hal]) if count > 1 else ''
                    halogen_parts.append(f'{mp}{halogen_prefix_map[hal]}')

            name = '(' + ''.join(halogen_parts) + 'methyl)'
            return {'name': name, 'atoms': sub_atoms, 'is_complex': True}

    return None


def _identify_alkyl_group(mol, start_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify alkyl substituent.

    Uses BFS to find all connected carbons and counts total carbons
    to determine alkyl name.
    """
    # BFS to find all atoms in the substituent
    visited = {start_idx}
    queue = deque([start_idx])
    carbon_count = 0
    all_atoms = []

    while queue:
        current_idx = queue.popleft()
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        if current_atom.GetSymbol() == 'C':
            carbon_count += 1
        else:
            # Non-carbon in the chain - this is not a simple alkyl
            # For Phase 2, we'll handle more complex substituents
            pass

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_atoms:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    # Check if this is a pure alkyl (only carbons and hydrogens)
    for idx in all_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() not in ('C', 'H'):
            # Contains heteroatom - not a simple alkyl
            # Check for vinyl (ethenyl) - C=C attached to ring
            if _is_vinyl_group(mol, start_idx, ring_atoms):
                return {'name': 'ethenyl', 'atoms': all_atoms}
            return None

    # Check for branched alkyl (isopropyl, tert-butyl, etc.)
    name = _get_alkyl_name(mol, start_idx, carbon_count, ring_atoms, sub_atoms=all_atoms)
    if name:
        return {'name': name, 'atoms': all_atoms}

    return None


def _is_vinyl_group(mol, start_idx: int, ring_atoms: Set[int]) -> bool:
    """Check if the group is a vinyl (ethenyl) group."""
    start_atom = mol.GetAtomWithIdx(start_idx)

    # Check for C=C where start is attached to ring
    for neighbor in start_atom.GetNeighbors():
        if neighbor.GetIdx() in ring_atoms:
            continue

        bond = mol.GetBondBetweenAtoms(start_idx, neighbor.GetIdx())
        if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
            if neighbor.GetSymbol() == 'C':
                return True

    return False


def _get_alkyl_name(mol, start_idx: int, carbon_count: int, ring_atoms: Set[int],
                    sub_atoms: list = None) -> Optional[str]:
    """
    Get the name for an alkyl substituent.

    Uses name_substituent_fragment() for branched/complex substituents,
    falls back to get_alkyl_name(carbon_count) for linear alkyls.

    Args:
        mol: RDKit Mol object
        start_idx: First carbon of the substituent (bonded to ring)
        carbon_count: Number of carbons in the substituent
        ring_atoms: Set of ring atom indices
        sub_atoms: Optional list of all atom indices in the substituent
    """
    from ..assembly.substituent_naming import name_substituent_fragment

    # If we have atom context, try name_substituent_fragment (handles retained
    # names like sec-butyl/isopropyl and branched substituents)
    if sub_atoms is not None and len(sub_atoms) > 0:
        recursive_name = name_substituent_fragment(
            mol, sub_atoms, start_idx, list(ring_atoms)
        )
        if recursive_name:
            return recursive_name

    # Fallback: simple carbon-count naming
    try:
        return get_alkyl_name(carbon_count)
    except (ValueError, KeyError):
        return None


def _identify_functionalized_chain(mol, start_idx: int, ring_atoms: Set[int]) -> Optional[Dict]:
    """
    Identify a chain with functional group attached to benzene ring.

    This handles cases like `-CCCC(=O)O` (butanoic acid chain) that the
    simple alkyl detection misses because they contain heteroatoms.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the first atom attached to the ring
        ring_atoms: Set of ring atom indices

    Returns:
        Dict with 'name' (a proper substituent name), 'atoms', 'chain_length',
        'functional_group', or None if not a functionalized chain or cannot be named.
    """
    # BFS to find all atoms in the substituent
    visited = {start_idx}
    queue = deque([start_idx])
    all_atoms = []
    carbon_count = 0
    has_heteroatom = False

    while queue:
        current_idx = queue.popleft()
        current_atom = mol.GetAtomWithIdx(current_idx)
        all_atoms.append(current_idx)

        symbol = current_atom.GetSymbol()
        if symbol == 'C':
            carbon_count += 1
        elif symbol != 'H':
            has_heteroatom = True

        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_atoms:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    # Only consider if there's a heteroatom (indicating functional group)
    if not has_heteroatom:
        return None

    # Check for common functional groups
    functional_group = _detect_chain_functional_group(mol, all_atoms)
    if not functional_group:
        return None

    # Generate a proper substituent name based on FG type and chain length
    sub_name = _name_functionalized_chain_substituent(carbon_count, functional_group)
    if not sub_name:
        # Cannot produce a valid name; return None so caller can handle gracefully
        return None

    # Wrap compound substituent names in parentheses per IUPAC P-14.5.2
    # e.g., "hydroxymethyl" -> "(hydroxymethyl)", "carboxymethyl" -> "(carboxymethyl)"
    from ..assembly.naming_utils import needs_brackets
    is_compound = needs_brackets(sub_name)
    if is_compound and not (sub_name.startswith('(') and sub_name.endswith(')')):
        sub_name = f'({sub_name})'

    return {
        'name': sub_name,
        'atoms': all_atoms,
        'chain_length': carbon_count,
        'functional_group': functional_group,
        'is_complex': is_compound,
    }


# Mapping from chain length (carbons) to substituent prefix stem
_CHAIN_SUB_STEMS = {
    1: "methyl", 2: "ethyl", 3: "propyl", 4: "butyl", 5: "pentyl",
    6: "hexyl", 7: "heptyl", 8: "octyl", 9: "nonyl", 10: "decyl",
}

# Mapping from functional group type to substituent prefix modifier
# These convert a chain with FG into a proper IUPAC prefix substituent name
_FG_SUB_PREFIX = {
    'carboxylic_acid': {
        # -C(=O)OH chain: named as "carboxylalkyl" (e.g., 2-carboxyethyl for -CH2CH2COOH)
        # or simply use the acyl prefix approach
        1: "carboxy",            # just -COOH
        2: "carboxymethyl",      # -CH2COOH
        3: "2-carboxyethyl",     # -CH2CH2COOH
        4: "3-carboxypropyl",    # -(CH2)3COOH
        5: "4-carboxybutyl",     # -(CH2)4COOH
    },
    'aldehyde': {
        # -CHO on the ring carbon (carbon NOT absorbable) -> formyl (P-66.6.1.1.3).
        1: "formyl",             # -CHO
        # Moving-base-atom: the -CHO carbon is absorbed into the chain and expressed
        # as 'oxo' at the terminal carbon (= chain length, opposite the attachment).
        # '1-oxo...yl' (oxo at the acyl carbon) is the disfavoured CAS form (P-66 note m).
        2: "2-oxoethyl",         # -CH2CHO
        3: "3-oxopropyl",        # -(CH2)2CHO
    },
    'alcohol': {
        # NOTE (Phase 172): the locant-drop here (hydroxyethyl vs 2-hydroxyethyl) is a
        # real DEF-4-family defect, but alcohol is NOT a moving-base-atom FG (no central
        # carbon migrates) and the sibling parent_to_prefix alcohol branch drops the
        # locant too -- fixing only this copy creates an inconsistency. Deferred to a
        # uniform DEF-4 alcohol-locant pass (out of Phase-172 MBA scope). Left as-is.
        1: "hydroxymethyl",      # -CH2OH
        2: "hydroxyethyl",       # -CH2CH2OH
        3: "hydroxypropyl",      # -(CH2)2CH2OH
    },
}


def _name_functionalized_chain_substituent(carbon_count: int, functional_group: str) -> Optional[str]:
    """
    Generate a proper IUPAC substituent name for a functionalized chain.

    Args:
        carbon_count: Number of carbon atoms in the chain
        functional_group: Type of functional group ('carboxylic_acid', 'aldehyde', 'alcohol')

    Returns:
        Substituent prefix name string, or None if cannot be named
    """
    # Try specific FG + chain length lookup
    fg_map = _FG_SUB_PREFIX.get(functional_group, {})
    if carbon_count in fg_map:
        return fg_map[carbon_count]

    # Fallback: generic naming based on FG type
    if functional_group == 'carboxylic_acid' and carbon_count > 0:
        if carbon_count == 1:
            return "carboxy"
        # For longer chains: (N-1)-carboxyalkyl
        alkyl = _CHAIN_SUB_STEMS.get(carbon_count - 1)
        if alkyl:
            return f"carboxy{alkyl}"

    if functional_group == 'alcohol' and carbon_count > 0:
        # NOTE (Phase 172): alcohol locant-drop left as-is (out of MBA scope; see the
        # _FG_SUB_PREFIX['alcohol'] note). Uniform DEF-4 alcohol fix deferred.
        alkyl = _CHAIN_SUB_STEMS.get(carbon_count)
        if alkyl:
            return f"hydroxy{alkyl}"

    if functional_group == 'aldehyde' and carbon_count > 0:
        if carbon_count == 1:
            return "formyl"  # -CHO on the ring carbon (P-66.6.1.1.3); not absorbable
        alkyl = _CHAIN_SUB_STEMS.get(carbon_count)
        if alkyl:
            # Moving-base-atom: -CHO carbon absorbed -> 'oxo' at the terminal carbon
            # (= carbon_count). '1-oxo...yl' acyl form is non-PIN (P-66 note m).
            return f"{carbon_count}-oxo{alkyl}"

    # Cannot name this chain
    return None


def _detect_chain_functional_group(mol, chain_atoms: List[int]) -> Optional[str]:
    """
    Detect what functional group is on a chain.

    Checks for carboxylic acid, alcohol, and aldehyde patterns.

    Args:
        mol: RDKit Mol object
        chain_atoms: List of atom indices in the chain

    Returns:
        Functional group name ('carboxylic_acid', 'alcohol', 'aldehyde') or None
    """
    chain_set = set(chain_atoms)

    # Check for carboxylic acid pattern: C(=O)O with O having H
    for idx in chain_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue

        neighbors = list(atom.GetNeighbors())
        o_double = None
        o_single = None

        for nbr in neighbors:
            if nbr.GetSymbol() == 'O':
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    o_double = nbr
                elif bond.GetBondType() == Chem.BondType.SINGLE:
                    if nbr.GetTotalNumHs() >= 1:  # -OH
                        o_single = nbr

        if o_double and o_single:
            return 'carboxylic_acid'

    # Check for aldehyde pattern: C(=O)H (must check before alcohol)
    for idx in chain_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'C' and atom.GetTotalNumHs() >= 1:
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'O':
                    bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        return 'aldehyde'

    # Check for alcohol pattern: C-O-H
    for idx in chain_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() >= 1:
            # Check it's not part of carboxylic acid (already checked above)
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'C':
                    c_atom = nbr
                    has_double_o = False
                    for c_nbr in c_atom.GetNeighbors():
                        if c_nbr.GetSymbol() == 'O' and c_nbr.GetIdx() != atom.GetIdx():
                            bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), c_nbr.GetIdx())
                            if bond.GetBondType() == Chem.BondType.DOUBLE:
                                has_double_o = True
                    if not has_double_o:
                        return 'alcohol'

    return None


def orient_benzene(
    mol,
    ring_atoms: Tuple[int, ...],
    substituents: Dict[int, List[Dict]]
) -> List[int]:
    """
    Orient benzene ring to give lowest locants to substituents.

    IUPAC 2013 Rules:
    1. For monosubstituted: substituent position is 1
    2. For polysubstituted: use first-point-of-difference for locants
    3. For identical substituents: minimize locant set
    4. When locant sets are equal: position 1 goes to alphabetically first substituent

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the benzene ring (ordered)
        substituents: Dict from get_benzene_substituents

    Returns:
        List of ring atom indices reordered so position 1 is first
    """
    ring_list = list(ring_atoms)
    n = len(ring_list)  # Should be 6

    # Get substituted positions (indices in ring_list)
    substituted_indices = []
    for i, atom_idx in enumerate(ring_list):
        if atom_idx in substituents:
            substituted_indices.append(i)

    if not substituted_indices:
        # Unsubstituted benzene - any orientation is fine
        return ring_list

    if len(substituted_indices) == 1:
        # Monosubstituted - that position becomes 1
        start_pos = substituted_indices[0]
        return _rotate_list(ring_list, start_pos)

    # Polysubstituted - try all starting positions and both directions
    # Collect all candidates with their locant sets and alphabetic scores
    candidates = []

    for start_pos in range(n):
        for direction in [1, -1]:  # 1 = clockwise, -1 = counterclockwise
            # Build oriented ring
            oriented = _build_oriented_ring(ring_list, start_pos, direction)

            # Calculate locants for this orientation
            locants = _calculate_locants(oriented, substituents)

            # Get the substituent at position 1 for alphabetical tie-breaking
            pos1_atom = oriented[0]
            pos1_sub_name = None
            if pos1_atom in substituents and substituents[pos1_atom]:
                pos1_sub_name = substituents[pos1_atom][0]['name']

            candidates.append((oriented, locants, pos1_sub_name))

    # Find the best locant set
    best_locants = None
    for _, locants, _ in candidates:
        if best_locants is None or _compare_locant_sets(locants, best_locants) < 0:
            best_locants = locants

    # Filter to only candidates with the best locant set
    best_candidates = [
        (oriented, pos1_sub) for oriented, locants, pos1_sub in candidates
        if locants == best_locants
    ]

    # If multiple candidates with same locant set, pick one where alphabetically
    # first substituent is at position 1
    if len(best_candidates) == 1:
        return best_candidates[0][0]

    # Sort by alphabetical order of position 1 substituent
    # None should sort last (no substituent at position 1)
    def sort_key(item):
        oriented, pos1_sub = item
        if pos1_sub is None:
            return 'zzzzz'  # Sort last
        return alpha_sort_key(pos1_sub)

    best_candidates.sort(key=sort_key)
    return best_candidates[0][0]


def _rotate_list(lst: List, start: int) -> List:
    """Rotate list so element at index start becomes first."""
    return lst[start:] + lst[:start]


def _build_oriented_ring(
    ring_list: List[int],
    start_pos: int,
    direction: int
) -> List[int]:
    """
    Build an oriented version of the ring.

    Args:
        ring_list: Original list of ring atom indices
        start_pos: Index to start from
        direction: 1 for clockwise, -1 for counterclockwise

    Returns:
        Reordered list with start_pos first, going in specified direction
    """
    n = len(ring_list)
    oriented = []

    for i in range(n):
        idx = (start_pos + i * direction) % n
        oriented.append(ring_list[idx])

    return oriented


def _calculate_locants(
    oriented_ring: List[int],
    substituents: Dict[int, List[Dict]]
) -> List[int]:
    """
    Calculate locant set for a given orientation.

    Args:
        oriented_ring: Ring atoms in order (position 0 = locant 1)
        substituents: Dict from get_benzene_substituents

    Returns:
        Sorted list of locants (1-indexed positions)
    """
    locants = []
    for i, atom_idx in enumerate(oriented_ring):
        if atom_idx in substituents:
            locants.append(i + 1)  # Locants are 1-indexed

    return sorted(locants)


def name_substituted_benzene(
    mol,
    ring_atoms: Tuple[int, ...],
    oriented_ring: List[int],
    substituents: Dict[int, List[Dict]],
    detected_fgs: Optional[Dict] = None,
) -> str:
    """
    Generate systematic name for substituted benzene.

    IUPAC 2013 PIN Rules:
    - Use numeric locants (not ortho/meta/para) for polysubstituted
    - Monosubstituted benzenes do NOT include locant (it's always 1)
    - Alphabetize substituent prefixes
    - Use multiplicative prefixes (di-, tri-) for repeated substituents
    - Format: locants-substituent-benzene (or just substituent-benzene for mono)
    - Special case: benzonitrile (C6H5CN) uses suffix-style naming per P-66.1.1.1
    - Ring-attached principal groups use suffix form (P-65.1.2)

    Args:
        mol: RDKit Mol object
        ring_atoms: Original ring atom tuple
        oriented_ring: Oriented ring from orient_benzene
        substituents: Dict from get_benzene_substituents

    Returns:
        IUPAC name string (e.g., "chlorobenzene" or "1,4-dimethylbenzene")
    """
    # Build locant-to-substituent mapping
    # oriented_ring[0] = position 1, oriented_ring[1] = position 2, etc.
    atom_to_locant = {atom_idx: i + 1 for i, atom_idx in enumerate(oriented_ring)}

    # Collect stereodescriptors for ring atoms (benzene carbons are sp2 so
    # typically no R/S, but substituents attached at ring positions may carry
    # E/Z on bonds to ring atoms). We pass the ring atom_to_locant mapping.
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    from ..perception.stereo import assign_stereochemistry

    assign_stereochemistry(mol)
    stereo_descriptors = collect_stereodescriptors(mol, atom_to_locant)

    # Separate suffix-type FGs from prefix-type substituents
    suffix_groups: Dict[str, List[int]] = defaultdict(list)
    prefix_groups: Dict[str, List[int]] = defaultdict(list)
    # Track N-substituents for amides keyed by suffix_name
    n_substituents_map: Dict[str, List[str]] = {}

    for atom_idx in oriented_ring:
        if atom_idx not in substituents:
            continue

        for sub_info in substituents[atom_idx]:
            locant = atom_to_locant[atom_idx]
            if sub_info.get('is_suffix'):
                suffix_groups[sub_info['suffix_name']].append(locant)
                # Track N-substituents if present
                if 'n_substituents' in sub_info and sub_info['n_substituents']:
                    n_substituents_map[sub_info['suffix_name']] = sub_info['n_substituents']
            else:
                prefix_groups[sub_info['name']].append(locant)

    # Sort locants within each group
    for name in suffix_groups:
        suffix_groups[name].sort()
    for name in prefix_groups:
        prefix_groups[name].sort()

    # --- ASML-13: Reclassify hydroxyl as suffix when it IS the principal group ---
    # _identify_oxygen_group always returns hydroxy as prefix. When hydroxyl is
    # the principal group (no higher-seniority suffix FG detected), move it to
    # suffix_groups so it routes to the phenol/ol naming path.
    # Guard: also check detected_fgs for FGs that use functional class naming
    # (isocyanate, azide, etc.). These FGs are not in _SUFFIX_PRIORITY, so
    # suffix_groups would be empty even though OH may not be the true principal
    # group. Blacklist approach: only block known functional-class-naming FGs.
    _FUNCTIONAL_CLASS_FGS = {'isocyanate', 'isothiocyanate', 'azide', 'diazo',
                              'cyanate', 'thiocyanate', 'selenocyanate'}
    has_competing_fg = False
    if detected_fgs:
        for fg_name in detected_fgs:
            if fg_name in _FUNCTIONAL_CLASS_FGS:
                has_competing_fg = True
                break
    if 'hydroxy' in prefix_groups and not suffix_groups and not has_competing_fg:
        # No other suffix FGs detected and no competing FGs -- hydroxyl IS
        # the principal group. Move from prefix to suffix.
        suffix_groups['ol'] = prefix_groups.pop('hydroxy')
    # When suffix_groups is non-empty (acid, aldehyde, etc.), hydroxyl stays
    # as prefix "hydroxy" -- correct per IUPAC seniority rules.

    # If suffix groups exist, use suffix naming path
    if suffix_groups:
        name = _assemble_benzene_with_suffix(
            mol, suffix_groups, prefix_groups, n_substituents_map,
            atom_to_locant, oriented_ring
        )
        if stereo_descriptors:
            stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
            name = f"{stereo_prefix}{name}"
        return name

    # === PREFIX-ONLY PATH (existing logic) ===

    # Check for nitrile - special handling for benzonitrile naming (BUG-2 fix)
    # IUPAC 2013 PIN: benzonitrile (not cyanobenzene) per P-66.1.1.1
    if 'nitrile' in prefix_groups:
        nitrile_locants = prefix_groups['nitrile']
        if len(nitrile_locants) == 1:
            # Single nitrile: use benzonitrile as parent
            # Remove nitrile from substituent groups since it becomes the parent
            del prefix_groups['nitrile']

            # Re-orient so nitrile is at position 1 for locant calculation
            nitrile_locant = nitrile_locants[0]

            # Other substituents become prefixes relative to benzonitrile
            if not prefix_groups:
                # Pure benzonitrile
                name = "benzonitrile"
                if stereo_descriptors:
                    stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
                    name = f"{stereo_prefix}{name}"
                return name

            # Build prefixes for other substituents
            # Need to recalculate locants relative to nitrile at position 1
            name = _name_substituted_benzonitrile(
                prefix_groups, nitrile_locant, atom_to_locant, oriented_ring
            )
            if stereo_descriptors:
                stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
                name = f"{stereo_prefix}{name}"
            return name

    # Count total number of substituents
    total_substituents = sum(len(locs) for locs in prefix_groups.values())

    # For monosubstituted benzenes, omit the locant (it's always 1)
    is_monosubstituted = total_substituents == 1
    _omit = should_omit_locant_one(
        context="prefix",
        is_ring=True,
        is_heterocyclic=False,  # Benzene is always carbocyclic
        is_monosubstituted=is_monosubstituted,
    )

    # Build prefix strings, sorted alphabetically by substituent name
    prefixes = []
    for name in sorted(prefix_groups.keys(), key=alpha_sort_key):
        locants = prefix_groups[name]
        count = len(locants)

        if _omit:
            # Monosubstituted: just "chloro", "methyl", etc. - no locant
            if name.startswith('(') or name.startswith('['):
                # Already has enclosing marks (possibly internal, e.g.
                # "(oxan-2-yl)oxy") -- keep as-is to avoid double-wrapping
                prefix_str = name
            elif is_complex_substituent(name):
                # Complex substituent needs enclosing marks per IUPAC P-14.5.2
                # e.g., "(2-methylbut-2-en-1-yl)benzene"
                prefix_str = f"({name})"
            else:
                prefix_str = name
        else:
            # Polysubstituted: include locants
            prefix_str = format_substituent_prefix(name, locants, count)

        prefixes.append(prefix_str)

    # Join prefixes with proper hyphenation
    prefix_part = _join_benzene_prefixes(prefixes)

    # Build final name
    name = f"{prefix_part}benzene"

    # Prepend stereo prefix if descriptors exist
    if stereo_descriptors:
        stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
        name = f"{stereo_prefix}{name}"

    return name


def _assemble_benzene_with_suffix(
    mol,
    suffix_groups: Dict[str, List[int]],
    prefix_groups: Dict[str, List[int]],
    n_substituents_map: Dict[str, List[str]],
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
) -> str:
    """
    Assemble benzene name with suffix functional groups.

    Picks the highest-priority suffix FG, builds locants + multiplier + suffix,
    then adds remaining FGs as prefixes.

    Args:
        mol: RDKit Mol object
        suffix_groups: Dict of suffix_name -> list of locants
        prefix_groups: Dict of prefix_name -> list of locants
        n_substituents_map: Dict of suffix_name -> list of N-alkyl names
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring

    Returns:
        IUPAC name string with suffix FG
    """
    # Pick highest-priority suffix
    chosen_suffix = None
    for sfx in _SUFFIX_PRIORITY:
        if sfx in suffix_groups:
            chosen_suffix = sfx
            break

    if chosen_suffix is None:
        # Shouldn't happen, but fallback to first suffix
        chosen_suffix = next(iter(suffix_groups))

    chosen_locants = suffix_groups[chosen_suffix]
    chosen_count = len(chosen_locants)

    # Single carbonitrile: delegate to existing benzonitrile path
    # (which uses the retained name "benzonitrile" and proper renumbering)
    if chosen_suffix == 'carbonitrile' and chosen_count == 1:
        # Convert suffix+prefix groups back to prefix-only for benzonitrile path
        nitrile_locant = chosen_locants[0]
        all_prefix = dict(prefix_groups)
        # Add any other suffix FGs as prefixes
        for sfx_name, sfx_locants in suffix_groups.items():
            if sfx_name == 'carbonitrile':
                continue
            prefix_form = _SUFFIX_TO_PREFIX.get(sfx_name, sfx_name)
            if prefix_form:
                all_prefix[prefix_form] = sfx_locants
        if not all_prefix:
            return "benzonitrile"
        return _name_substituted_benzonitrile(
            all_prefix, nitrile_locant, atom_to_locant, oriented_ring
        )

    # Remaining suffix FGs become prefixes
    remaining_prefix_groups = dict(prefix_groups)
    for sfx_name, sfx_locants in suffix_groups.items():
        if sfx_name == chosen_suffix:
            continue
        # Convert to prefix form
        prefix_form = _SUFFIX_TO_PREFIX.get(sfx_name, sfx_name)
        if prefix_form:
            remaining_prefix_groups[prefix_form] = sfx_locants

    # ASML-13: Handle hydroxyl suffix -- phenol retained name for benzene
    if chosen_suffix == 'ol':
        ol_locants = chosen_locants
        if len(ol_locants) == 1:
            # Single OH on benzene: use "phenol" retained name
            return _name_substituted_phenol(
                remaining_prefix_groups, ol_locants[0],
                atom_to_locant, oriented_ring
            )
        else:
            # Multiple OH on benzene: benzenediol, benzenetriol
            # Use systematic naming with multiplied -ol suffix.
            from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
            mult = SIMPLE_MULTIPLIERS.get(len(ol_locants), str(len(ol_locants)))
            loc_str = ','.join(str(l) for l in sorted(ol_locants))
            suffix_part = f"benzene-{loc_str}-{mult}ol"
            if remaining_prefix_groups:
                prefix_part = _build_prefix_string_with_locants(
                    remaining_prefix_groups, mono_needs_locant=True
                )
                return f"{prefix_part}{suffix_part}"
            return suffix_part

    # Check for special "benzoic acid" retained base name:
    # Single carboxylic acid -> "benzoic acid" base (P-65.1.2.1)
    if chosen_suffix == 'carboxylic acid' and chosen_count == 1:
        return _name_substituted_benzoic_acid(
            remaining_prefix_groups, chosen_locants[0],
            atom_to_locant, oriented_ring
        )

    # Check for benzamide-based naming:
    # Single carboxamide -> use "benzamide" as retained base
    if chosen_suffix == 'carboxamide' and chosen_count == 1:
        n_subs = n_substituents_map.get('carboxamide', [])
        return _name_substituted_benzamide(
            remaining_prefix_groups, chosen_locants[0],
            atom_to_locant, oriented_ring, n_subs
        )

    # Check for benzenesulfonamide-based naming:
    # Single sulfonamide -> "benzenesulfonamide"
    if chosen_suffix == 'sulfonamide' and chosen_count == 1:
        return _name_substituted_benzenesulfonamide(
            remaining_prefix_groups, chosen_locants[0],
            atom_to_locant, oriented_ring
        )

    # Single carbaldehyde: delegate to benzaldehyde retained name path
    # "benzaldehyde" is an IUPAC retained name (P-66.6.3.1.1)
    if chosen_suffix == 'carbaldehyde' and chosen_count == 1:
        return _name_substituted_benzaldehyde(
            remaining_prefix_groups, chosen_locants[0],
            atom_to_locant, oriented_ring
        )

    # General suffix assembly for multi-suffix or non-retained cases
    # Build suffix part: benzene-{locants}-{multiplier}{suffix}
    multiplier = get_multiplier_prefix(chosen_count, chosen_suffix) if chosen_count > 1 else ""
    locant_str = ",".join(str(loc) for loc in chosen_locants)

    # Build prefix part from remaining groups
    prefix_part = _build_prefix_string(remaining_prefix_groups)

    # Assemble: {prefix}benzene-{locants}-{multiplier}{suffix}
    if chosen_count > 1:
        return f"{prefix_part}benzene-{locant_str}-{multiplier}{chosen_suffix}"
    else:
        # Monosubstituted: benzene{suffix} (no locant, no hyphen)
        return f"{prefix_part}benzene{chosen_suffix}"


def _name_substituted_benzoic_acid(
    prefix_groups: Dict[str, List[int]],
    acid_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
) -> str:
    """
    Name substituted benzoic acid derivatives.

    Uses "benzoic acid" as the retained base name. Position 1 is the
    carboxylic acid position. Other substituents get locants relative to it.

    Args:
        prefix_groups: Dict of prefix name -> locants (non-acid substituents)
        acid_locant: Locant of the carboxylic acid in original numbering
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring

    Returns:
        IUPAC name like "2-hydroxybenzoic acid"
    """
    if not prefix_groups:
        return "benzoic acid"

    # Renumber relative to acid position (acid = position 1)
    renumbered_groups = _renumber_relative_to(prefix_groups, acid_locant)

    # Build prefixes
    prefix_part = _build_prefix_string_with_locants(renumbered_groups, mono_needs_locant=True)

    return f"{prefix_part}benzoic acid"


def _name_substituted_phenol(
    prefix_groups: Dict[str, List[int]],
    ol_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
) -> str:
    """
    Name substituted phenol derivatives.

    Uses "phenol" as the retained base name per IUPAC P-63.1.1.1.
    Position 1 is the hydroxyl position. Other substituents get locants
    relative to it.

    Args:
        prefix_groups: Dict of prefix name -> locants (non-OH substituents)
        ol_locant: Locant of the hydroxyl in original numbering
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring

    Returns:
        IUPAC name like "2-methylphenol" or "4-chlorophenol"
    """
    if not prefix_groups:
        return "phenol"

    # Renumber relative to OH position (OH = position 1)
    renumbered_groups = _renumber_relative_to(prefix_groups, ol_locant)

    # Build prefixes
    prefix_part = _build_prefix_string_with_locants(renumbered_groups, mono_needs_locant=True)

    return f"{prefix_part}phenol"


def _name_substituted_benzamide(
    prefix_groups: Dict[str, List[int]],
    amide_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
    n_substituents: List[str],
) -> str:
    """
    Name substituted benzamide derivatives.

    Uses "benzamide" as the retained base name. Position 1 is the
    carboxamide position. Other substituents get locants relative to it.

    Args:
        prefix_groups: Non-amide substituent groups
        amide_locant: Locant of the amide in original numbering
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring
        n_substituents: List of N-alkyl names (e.g., ['methyl'] for N-methylbenzamide)

    Returns:
        IUPAC name like "4-methylbenzamide" or "N-methylbenzamide"
    """
    # Build N-substituent prefix part
    from ..assembly.naming_utils import _wrap_n_substituent
    n_prefix = ""
    if n_substituents:
        if len(n_substituents) == 1:
            n_prefix = f"N-{_wrap_n_substituent(n_substituents[0])}"
        elif len(n_substituents) == 2 and n_substituents[0] == n_substituents[1]:
            mp = get_multiplier_prefix(2, n_substituents[0])
            n_prefix = f"N,N-{mp}{_wrap_n_substituent(n_substituents[0])}"
        else:
            # Different N-substituents
            parts = [f"N-{_wrap_n_substituent(name)}" for name in n_substituents]
            n_prefix = "-".join(parts)

    if not prefix_groups and not n_prefix:
        return "benzamide"

    # Renumber relative to amide position (amide = position 1)
    renumbered_groups = _renumber_relative_to(prefix_groups, amide_locant)

    # Build ring-substituent prefixes
    ring_prefix_part = _build_prefix_string_with_locants(renumbered_groups, mono_needs_locant=True)

    # Combine N-prefix and ring-prefix
    if n_prefix and ring_prefix_part:
        return f"{n_prefix}-{ring_prefix_part}benzamide"
    elif n_prefix:
        return f"{n_prefix}benzamide"
    else:
        return f"{ring_prefix_part}benzamide"


def _name_substituted_benzenesulfonamide(
    prefix_groups: Dict[str, List[int]],
    sulfonamide_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
) -> str:
    """
    Name substituted benzenesulfonamide derivatives.

    Uses "benzenesulfonamide" as the base name. Position 1 is the
    sulfonamide position.

    Args:
        prefix_groups: Non-sulfonamide substituent groups
        sulfonamide_locant: Locant of the sulfonamide in original numbering
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring

    Returns:
        IUPAC name like "4-methylbenzenesulfonamide"
    """
    if not prefix_groups:
        return "benzenesulfonamide"

    # Renumber relative to sulfonamide position
    renumbered_groups = _renumber_relative_to(prefix_groups, sulfonamide_locant)

    # Build prefixes
    prefix_part = _build_prefix_string_with_locants(renumbered_groups, mono_needs_locant=True)

    return f"{prefix_part}benzenesulfonamide"


def _name_substituted_benzaldehyde(
    prefix_groups: Dict[str, List[int]],
    aldehyde_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int],
) -> str:
    """
    Name substituted benzaldehyde derivatives.

    Uses 'benzaldehyde' as retained base per IUPAC P-66.6.3.1.1.
    Position 1 = CHO-bearing carbon.

    Pattern follows _name_substituted_benzoic_acid().

    Args:
        prefix_groups: Non-aldehyde substituent groups
        aldehyde_locant: Locant of the aldehyde in original numbering
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring

    Returns:
        IUPAC name like "4-chlorobenzaldehyde"
    """
    if not prefix_groups:
        return "benzaldehyde"

    # Renumber relative to aldehyde position (aldehyde = position 1)
    renumbered_groups = _renumber_relative_to(prefix_groups, aldehyde_locant)

    # Build prefixes
    prefix_part = _build_prefix_string_with_locants(renumbered_groups, mono_needs_locant=True)

    return f"{prefix_part}benzaldehyde"


def _renumber_relative_to(
    groups: Dict[str, List[int]],
    reference_locant: int,
) -> Dict[str, List[int]]:
    """
    Renumber substituent locants relative to a reference position.

    The reference position becomes position 1. Tries both clockwise and
    counterclockwise numbering and picks the one giving lowest locants.

    Args:
        groups: Dict of name -> list of locants in original numbering
        reference_locant: The locant that should become position 1

    Returns:
        Dict of name -> list of renumbered locants
    """
    best_groups = None
    best_locant_set = None

    for direction in [1, -1]:
        converted: Dict[str, List[int]] = defaultdict(list)

        for name, locants in groups.items():
            for old_loc in locants:
                diff = (old_loc - reference_locant) * direction
                new_loc = (diff % 6) + 1
                if new_loc == 1:
                    # Position 1 is reserved for the principal group
                    new_loc = 6
                converted[name].append(new_loc)

        # Sort locants within each group
        for name in converted:
            converted[name].sort()

        # Calculate overall locant set for comparison
        all_locants = sorted([loc for locs in converted.values() for loc in locs])

        if best_locant_set is None or all_locants < best_locant_set:
            best_locant_set = all_locants
            best_groups = dict(converted)

    return best_groups if best_groups else {}


def _build_prefix_string(prefix_groups: Dict[str, List[int]]) -> str:
    """
    Build prefix part string from prefix groups.

    For groups where locants are all 1 and single, omits locants.
    Handles monosubstituted (no locant needed).

    Args:
        prefix_groups: Dict of prefix name -> list of locants

    Returns:
        Prefix string to prepend to parent name
    """
    if not prefix_groups:
        return ""

    total = sum(len(locs) for locs in prefix_groups.values())
    is_mono = total == 1
    _omit = should_omit_locant_one(
        context="prefix",
        is_ring=True,
        is_heterocyclic=False,  # Benzene is always carbocyclic
        is_monosubstituted=is_mono,
    )

    prefixes = []
    for name in sorted(prefix_groups.keys(), key=alpha_sort_key):
        locants = prefix_groups[name]
        count = len(locants)

        if _omit:
            if name.startswith('(') or name.startswith('['):
                # Already has enclosing marks -- keep as-is
                prefix_str = name
            elif is_complex_substituent(name):
                # Complex substituent needs enclosing marks per IUPAC P-14.5.2
                prefix_str = f"({name})"
            else:
                prefix_str = name
        else:
            prefix_str = format_substituent_prefix(name, locants, count)

        prefixes.append(prefix_str)

    return _join_benzene_prefixes(prefixes)


def _build_prefix_string_with_locants(
    prefix_groups: Dict[str, List[int]],
    mono_needs_locant: bool = True,
) -> str:
    """
    Build prefix string where locants are always included (for substituted retained names).

    For "4-methylbenzamide", the locant 4 is needed even for mono-substitution.

    Args:
        prefix_groups: Dict of prefix name -> list of locants
        mono_needs_locant: If True, even single substituents include locant

    Returns:
        Prefix string like "4-methyl" or "2-hydroxy"
    """
    if not prefix_groups:
        return ""

    prefixes = []
    for name in sorted(prefix_groups.keys(), key=alpha_sort_key):
        locants = prefix_groups[name]
        count = len(locants)

        prefix_str = format_substituent_prefix(name, locants, count)
        prefixes.append(prefix_str)

    result = _join_benzene_prefixes(prefixes)

    return result


def _name_substituted_benzonitrile(
    substituent_groups: Dict[str, List[int]],
    nitrile_locant: int,
    atom_to_locant: Dict[int, int],
    oriented_ring: List[int]
) -> str:
    """
    Name a substituted benzonitrile.

    IUPAC 2013: substituents are numbered relative to the nitrile position (position 1).
    Example: 4-chlorobenzonitrile, 4-methylbenzonitrile

    The nitrile carbon position becomes position 1 in the benzonitrile numbering.
    For a 6-membered ring with nitrile at old position N:
    - Position N becomes 1
    - Other positions are renumbered going clockwise (or counterclockwise for lowest locants)

    Args:
        substituent_groups: Dict of substituent name -> list of locants (in original numbering)
        nitrile_locant: The locant of the nitrile in the original numbering
        atom_to_locant: Mapping from atom index to locant
        oriented_ring: The oriented ring

    Returns:
        IUPAC name string (e.g., "4-chlorobenzonitrile")
    """
    # Try both directions (clockwise and counterclockwise) and pick lowest locants
    best_groups = None
    best_locant_set = None

    for direction in [1, -1]:
        converted_groups: Dict[str, List[int]] = defaultdict(list)

        for name, locants in substituent_groups.items():
            for old_loc in locants:
                # Calculate new position relative to nitrile at position 1
                # direction = 1: clockwise numbering from nitrile
                # direction = -1: counterclockwise numbering from nitrile
                # Formula: new_pos = ((old_pos - nitrile_pos) * direction % 6) + 1
                # This ensures nitrile_pos -> 1, and other positions follow in order
                diff = (old_loc - nitrile_locant) * direction
                new_loc = (diff % 6) + 1
                if new_loc == 1:
                    # Position 1 is reserved for nitrile; this shouldn't happen
                    # for other substituents, but handle edge case
                    new_loc = 7 - new_loc  # Map to position 6 (opposite direction)
                converted_groups[name].append(new_loc)

        # Sort locants within each group
        for name in converted_groups:
            converted_groups[name].sort()

        # Calculate locant set for comparison
        all_locants = sorted([loc for locs in converted_groups.values() for loc in locs])

        if best_locant_set is None or all_locants < best_locant_set:
            best_locant_set = all_locants
            best_groups = dict(converted_groups)

    # Build prefix strings
    total_substituents = sum(len(locs) for locs in best_groups.values())
    is_monosubstituted = total_substituents == 1

    prefixes = []
    for name in sorted(best_groups.keys(), key=alpha_sort_key):
        locants = best_groups[name]
        count = len(locants)

        if is_monosubstituted:
            # Single other substituent: include locant (e.g., "4-chloro")
            prefix_str = f"{locants[0]}-{name}"
        else:
            # Multiple substituents
            prefix_str = format_substituent_prefix(name, locants, count)

        prefixes.append(prefix_str)

    # Join prefixes
    prefix_part = _join_benzene_prefixes(prefixes)

    return f"{prefix_part}benzonitrile"


def _join_benzene_prefixes(prefixes: List[str]) -> str:
    """
    Join benzene substituent prefixes with proper hyphenation.

    When one prefix ends with a letter and the next starts with a digit,
    a hyphen is needed.

    Args:
        prefixes: List of formatted prefix strings

    Returns:
        Joined prefix string
    """
    if not prefixes:
        return ""

    if len(prefixes) == 1:
        return prefixes[0]

    result = prefixes[0]
    for i in range(1, len(prefixes)):
        current = prefixes[i]

        # Check if we need a hyphen
        if result and current:
            last_char = result[-1]
            first_char = current[0]

            # Hyphen needed between alpha/paren and digit
            # e.g., "1-(N,N-dimethylamino)" + "4-amino" needs hyphen after ")"
            if (last_char.isalpha() or last_char == ')') and first_char.isdigit():
                result += "-"

        result += current

    return result


def name_benzene_derivative(mol) -> Optional[str]:
    """
    Generate name for a benzene derivative.

    This is the main entry point for benzene naming.

    Args:
        mol: RDKit Mol object

    Returns:
        IUPAC name string, or None if not a benzene derivative
    """
    # Find benzene ring
    ring_atoms = get_benzene_ring(mol)
    if ring_atoms is None:
        return None

    # Get substituents
    substituents = get_benzene_substituents(mol, ring_atoms)

    if not substituents:
        # Unsubstituted benzene - should be caught by retained names
        return "benzene"

    # Orient the ring for lowest locants
    oriented_ring = orient_benzene(mol, ring_atoms, substituents)

    # Generate systematic name
    return name_substituted_benzene(mol, ring_atoms, oriented_ring, substituents)
