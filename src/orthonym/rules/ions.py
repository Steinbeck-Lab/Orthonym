"""
Ion naming rules per IUPAC 2013.

Handles naming of:
- Anions: carboxylates (-ate), alkoxides (-olate), phenolates, aminides, carbanions (-ide)
- Cations: aminium (-aminium), carbenium/ylium (-ylium), onium, diazonium

IUPAC 2013 References:
- P-72: Anion nomenclature
- P-73: Cation nomenclature

Key naming patterns:
- Carboxylate anions: acetic acid -> acetate
- Alkoxide anions: methanol -> methanolate (PIN) or methoxide (acceptable)
- Aminium cations: amine -> aminium (protonated amine)
- Carbenium cations: alkane -> ylium (loss of H-)
"""

from typing import Dict, List, Optional, Any
from rdkit import Chem

from ..data.ion_retained_names import get_anion_name, get_cation_name
from ..perception.ions import get_ion_sites


# === SUFFIX MAPPINGS ===

ANION_SUFFIXES = {
    'carboxylate': 'ate',       # -COOH -> -COO-
    'alkoxide': 'olate',        # -OH -> -O-
    'phenolate': 'olate',       # PhOH -> PhO-
    'aminide': 'aminide',       # -NH2 -> -NH-
    'carbanion': 'ide',         # C-H -> C-
    'thiolate': 'thiolate',     # -SH -> -S-
}

CATION_SUFFIXES = {
    'aminium': 'aminium',       # -NH2 + H+ -> -NH3+
    'ylium': 'ylium',           # CH4 - H- -> CH3+
    'ium': 'ium',               # add H+
    'onium': 'onium',           # O/S/P cations
    'diazonium': 'diazonium',   # -N2+
}


# === ANION CLASSIFICATION ===

def classify_anion(mol, anion_site: Dict[str, Any]) -> str:
    """
    Classify anion type based on the anionic atom environment.

    Examines the local chemical environment of the anionic atom to
    determine the appropriate naming suffix.

    Args:
        mol: RDKit Mol object
        anion_site: Dictionary from get_ion_sites containing:
            - atom_idx: int
            - charge: int
            - element: str
            - hybridization: str
            - n_hydrogens: int

    Returns:
        One of: 'carboxylate', 'alkoxide', 'phenolate', 'aminide',
                'carbanion', 'thiolate', or 'unknown'

    Example:
        >>> mol = Chem.MolFromSmiles('CC(=O)[O-]')
        >>> sites = get_ion_sites(mol)
        >>> classify_anion(mol, sites['anions'][0])
        'carboxylate'
    """
    atom_idx = anion_site['atom_idx']
    element = anion_site['element']
    atom = mol.GetAtomWithIdx(atom_idx)

    # Classify based on element
    if element == 'O':
        # Check if part of carboxyl group (carboxylate)
        # Carboxylate: O- connected to C which has double bond to another O
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C':
                # Check for carbonyl oxygen (C=O)
                for second_neighbor in neighbor.GetNeighbors():
                    if second_neighbor.GetIdx() != atom_idx:
                        if second_neighbor.GetSymbol() == 'O':
                            bond = mol.GetBondBetweenAtoms(
                                neighbor.GetIdx(), second_neighbor.GetIdx()
                            )
                            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                                return 'carboxylate'

                # Check if attached to aromatic carbon (phenolate)
                if neighbor.GetIsAromatic():
                    return 'phenolate'

        # Default to alkoxide (O- attached to alkyl)
        return 'alkoxide'

    elif element == 'N':
        # Nitrogen anion (aminide)
        return 'aminide'

    elif element == 'C':
        # Carbon anion (carbanion)
        return 'carbanion'

    elif element == 'S':
        # Sulfur anion (thiolate)
        return 'thiolate'

    return 'unknown'


# === CATION CLASSIFICATION ===

def classify_cation(mol, cation_site: Dict[str, Any]) -> str:
    """
    Classify cation type based on the cationic atom environment.

    Examines the local chemical environment of the cationic atom to
    determine the appropriate naming suffix.

    Args:
        mol: RDKit Mol object
        cation_site: Dictionary from get_ion_sites containing:
            - atom_idx: int
            - charge: int
            - element: str
            - hybridization: str
            - n_hydrogens: int

    Returns:
        One of: 'aminium', 'ylium', 'onium', 'diazonium', or 'unknown'

    Example:
        >>> mol = Chem.MolFromSmiles('[NH4+]')
        >>> sites = get_ion_sites(mol)
        >>> classify_cation(mol, sites['cations'][0])
        'aminium'
    """
    atom_idx = cation_site['atom_idx']
    element = cation_site['element']
    atom = mol.GetAtomWithIdx(atom_idx)

    if element == 'N':
        # Check for diazonium (N2+ pattern: N connected to another N)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'N':
                # Diazonium: R-N+=N or R-N=N+
                return 'diazonium'

        # Protonated nitrogen -> aminium
        # Includes primary (NH3+), secondary (NH2+), tertiary (NH+), quaternary (N+)
        return 'aminium'

    elif element == 'C':
        # Carbocation -> ylium (carbenium ion)
        return 'ylium'

    elif element in ('O', 'S', 'P', 'Se'):
        # Oxonium, sulfonium, phosphonium, selenonium
        return 'onium'

    return 'unknown'


# === SUFFIX TRANSFORMATIONS ===

def name_carboxylate_anion(parent_name: str) -> str:
    """
    Convert carboxylic acid name to carboxylate anion name.

    Transformation rules:
    - 'ic acid' -> 'ate' (e.g., acetic acid -> acetate)
    - 'oic acid' -> 'oate' (e.g., propanoic acid -> propanoate)

    Args:
        parent_name: Parent acid name (e.g., 'acetic acid', 'propanoic acid')

    Returns:
        Anion name (e.g., 'acetate', 'propanoate')

    Example:
        >>> name_carboxylate_anion('acetic acid')
        'acetate'
        >>> name_carboxylate_anion('propanoic acid')
        'propanoate'
    """
    name = parent_name.lower().strip()

    # Handle 'oic acid' ending (systematic names)
    if name.endswith('oic acid'):
        return name[:-8] + 'oate'

    # Handle 'ic acid' ending (trivial names like acetic, formic)
    if name.endswith('ic acid'):
        return name[:-7] + 'ate'

    # Fallback: just add -ate
    return name.replace(' acid', 'ate')


def name_alkoxide_anion(parent_name: str, style: str = 'pin') -> str:
    """
    Convert alcohol name to alkoxide anion name.

    IUPAC 2013 PIN style uses -olate (methanolate, ethanolate).
    Common style uses -oxide (methoxide, ethoxide).

    Args:
        parent_name: Parent alcohol name (e.g., 'methanol', 'ethanol')
        style: 'pin' for -olate, 'common' for -oxide

    Returns:
        Anion name

    Example:
        >>> name_alkoxide_anion('methanol')
        'methanolate'
        >>> name_alkoxide_anion('methanol', style='common')
        'methoxide'
    """
    name = parent_name.lower().strip()

    if style == 'pin':
        # PIN style: methanol -> methanolate
        if name.endswith('ol'):
            return name + 'ate'
        elif name.endswith('anol'):
            return name + 'ate'
        else:
            return name + 'olate'
    else:
        # Common style: methanol -> methoxide
        if name.endswith('anol'):
            return name[:-4] + 'oxide'
        elif name.endswith('ol'):
            return name[:-2] + 'oxide'
        else:
            return name + 'oxide'


def name_phenolate_anion(parent_name: str) -> str:
    """
    Convert phenol name to phenolate anion name.

    Args:
        parent_name: Parent phenol name (e.g., 'phenol')

    Returns:
        Anion name (e.g., 'phenolate')

    Example:
        >>> name_phenolate_anion('phenol')
        'phenolate'
    """
    name = parent_name.lower().strip()

    if name.endswith('ol'):
        return name + 'ate'
    else:
        return name + 'olate'


def name_aminium_cation(parent_name: str) -> str:
    """
    Convert amine name to aminium cation name.

    Args:
        parent_name: Parent amine name (e.g., 'methanamine', 'ethylamine')

    Returns:
        Cation name (e.g., 'methanaminium', 'ethylaminium')

    Example:
        >>> name_aminium_cation('methanamine')
        'methanaminium'
        >>> name_aminium_cation('ammonia')
        'ammonium'
    """
    name = parent_name.lower().strip()

    # Special case: ammonia -> ammonium
    if name == 'ammonia':
        return 'ammonium'

    # amine -> aminium
    if name.endswith('amine'):
        return name[:-1] + 'ium'
    elif name.endswith('ane'):
        # Handle alkane-based names
        return name[:-1] + 'ium'
    else:
        return name + 'ium'


def name_carbenium_cation(parent_name: str) -> str:
    """
    Convert alkane/alkyl name to carbenium (ylium) cation name.

    The ylium suffix indicates loss of hydride (H-) from the parent.

    Args:
        parent_name: Parent name (e.g., 'methane', 'methyl')

    Returns:
        Cation name (e.g., 'methylium')

    Example:
        >>> name_carbenium_cation('methane')
        'methylium'
        >>> name_carbenium_cation('ethane')
        'ethylium'
    """
    name = parent_name.lower().strip()

    # Handle -ane suffix (alkanes)
    if name.endswith('ane'):
        return name[:-3] + 'ylium'

    # Handle -yl suffix (already a radical/substituent form)
    if name.endswith('yl'):
        return name + 'ium'

    # Default: add -ylium
    return name + 'ylium'


# === ANION SUFFIX GETTERS ===

def get_anion_suffix(anion_type: str) -> str:
    """
    Get the appropriate suffix for an anion type.

    Args:
        anion_type: Type from classify_anion()

    Returns:
        Suffix string (e.g., 'ate', 'olate', 'ide')
    """
    return ANION_SUFFIXES.get(anion_type, 'ide')


def get_cation_suffix(cation_type: str) -> str:
    """
    Get the appropriate suffix for a cation type.

    Args:
        cation_type: Type from classify_cation()

    Returns:
        Suffix string (e.g., 'ium', 'ylium', 'aminium')
    """
    return CATION_SUFFIXES.get(cation_type, 'ium')


# === MAIN NAMING FUNCTIONS ===

def name_anion(mol, style: str = 'pin') -> str:
    """
    Generate IUPAC name for an anionic molecule.

    Workflow:
    1. Get canonical SMILES
    2. Check retained names (unless systematic style)
    3. Detect anion sites
    4. Classify anion type
    5. Generate systematic name

    Args:
        mol: RDKit Mol object (must have negative charge)
        style: Naming style ('pin', 'systematic', 'common')

    Returns:
        Anion name (e.g., 'acetate', 'methoxide', 'phenolate')

    Example:
        >>> mol = Chem.MolFromSmiles('CC(=O)[O-]')
        >>> name_anion(mol)
        'acetate'
    """
    if mol is None:
        return ''

    # Get canonical SMILES for lookup
    canonical = Chem.MolToSmiles(mol, canonical=True)

    # Check retained names first (unless systematic requested)
    if style != 'systematic':
        retained = get_anion_name(canonical)
        if retained:
            return retained

    # Get anion sites
    sites = get_ion_sites(mol)
    anions = sites.get('anions', [])

    if not anions:
        return ''

    # For single anion, classify and name
    if len(anions) == 1:
        anion_site = anions[0]
        anion_type = classify_anion(mol, anion_site)

        # Generate systematic name based on type
        if anion_type == 'carboxylate':
            return _name_carboxylate_systematic(mol, anion_site)
        elif anion_type == 'alkoxide':
            return _name_alkoxide_systematic(mol, anion_site, style)
        elif anion_type == 'phenolate':
            return _name_phenolate_systematic(mol, anion_site)
        elif anion_type == 'carbanion':
            return _name_carbanion_systematic(mol, anion_site)
        elif anion_type == 'thiolate':
            return _name_thiolate_systematic(mol, anion_site)
        else:
            # Generic anion - use parent name + suffix
            suffix = get_anion_suffix(anion_type)
            return f'anion-{suffix}'

    # Multiple anions - more complex naming
    # For now, return first anion name
    return name_anion(mol, style)


def name_cation(mol, style: str = 'pin') -> str:
    """
    Generate IUPAC name for a cationic molecule.

    Workflow:
    1. Get canonical SMILES
    2. Check retained names (unless systematic style)
    3. Detect cation sites
    4. Classify cation type
    5. Generate systematic name

    Args:
        mol: RDKit Mol object (must have positive charge)
        style: Naming style ('pin', 'systematic', 'common')

    Returns:
        Cation name (e.g., 'ammonium', 'methylammonium', 'methylium')

    Example:
        >>> mol = Chem.MolFromSmiles('[NH4+]')
        >>> name_cation(mol)
        'ammonium'
    """
    if mol is None:
        return ''

    # Get canonical SMILES for lookup
    canonical = Chem.MolToSmiles(mol, canonical=True)

    # Check retained names first (unless systematic requested)
    if style != 'systematic':
        retained = get_cation_name(canonical)
        if retained:
            return retained

    # Get cation sites
    sites = get_ion_sites(mol)
    cations = sites.get('cations', [])

    if not cations:
        return ''

    # For single cation, classify and name
    if len(cations) == 1:
        cation_site = cations[0]
        cation_type = classify_cation(mol, cation_site)

        # Generate systematic name based on type
        if cation_type == 'aminium':
            return _name_aminium_systematic(mol, cation_site)
        elif cation_type == 'ylium':
            return _name_carbenium_systematic(mol, cation_site)
        elif cation_type == 'onium':
            return _name_onium_systematic(mol, cation_site)
        elif cation_type == 'diazonium':
            return _name_diazonium_systematic(mol, cation_site)
        else:
            # Generic cation
            suffix = get_cation_suffix(cation_type)
            return f'cation-{suffix}'

    # Multiple cations - more complex naming
    return name_cation(mol, style)


# === SYSTEMATIC NAMING HELPERS ===

def _name_carboxylate_systematic(mol, anion_site: Dict) -> str:
    """Generate systematic name for carboxylate anion."""
    # Count carbons to determine chain length
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    # Map chain length to name prefix
    CHAIN_PREFIXES = {
        1: 'form', 2: 'acet', 3: 'propano', 4: 'butano',
        5: 'pentano', 6: 'hexano', 7: 'heptano', 8: 'octano',
        9: 'nonano', 10: 'decano'
    }

    prefix = CHAIN_PREFIXES.get(carbon_count, f'{carbon_count}C-')

    # Special cases for common names
    if carbon_count == 1:
        return 'formate'
    elif carbon_count == 2:
        return 'acetate'
    else:
        return prefix + 'ate'


def _name_alkoxide_systematic(mol, anion_site: Dict, style: str) -> str:
    """Generate systematic name for alkoxide anion."""
    # Count carbons
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    CHAIN_NAMES = {
        1: 'methan', 2: 'ethan', 3: 'propan', 4: 'butan',
        5: 'pentan', 6: 'hexan', 7: 'heptan', 8: 'octan',
        9: 'nonan', 10: 'decan'
    }

    base = CHAIN_NAMES.get(carbon_count, f'{carbon_count}C-')

    if style == 'pin':
        return base + 'olate'
    else:
        # Common names: methoxide, ethoxide, etc.
        COMMON_NAMES = {1: 'methoxide', 2: 'ethoxide', 3: 'propoxide', 4: 'butoxide'}
        return COMMON_NAMES.get(carbon_count, base + 'oxide')


def _name_phenolate_systematic(mol, anion_site: Dict) -> str:
    """Generate systematic name for phenolate anion."""
    # Check for substituents on the benzene ring
    # For unsubstituted phenol: phenolate
    return 'phenolate'


def _name_carbanion_systematic(mol, anion_site: Dict) -> str:
    """Generate systematic name for carbanion."""
    # Count carbons
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    CHAIN_NAMES = {
        1: 'methan', 2: 'ethan', 3: 'propan', 4: 'butan',
        5: 'pentan', 6: 'hexan', 7: 'heptan', 8: 'octan'
    }

    base = CHAIN_NAMES.get(carbon_count, f'{carbon_count}C-')
    return base + 'ide'


def _name_thiolate_systematic(mol, anion_site: Dict) -> str:
    """Generate systematic name for thiolate anion."""
    # Count carbons
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    CHAIN_NAMES = {
        1: 'methane', 2: 'ethane', 3: 'propane', 4: 'butane'
    }

    base = CHAIN_NAMES.get(carbon_count, f'{carbon_count}C-')
    return base + 'thiolate'


def _name_aminium_systematic(mol, cation_site: Dict) -> str:
    """Generate systematic name for aminium cation."""
    # Count carbons attached to nitrogen
    atom = mol.GetAtomWithIdx(cation_site['atom_idx'])
    carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']

    if len(carbon_neighbors) == 0:
        # NH4+ -> ammonium
        return 'ammonium'
    elif len(carbon_neighbors) == 1:
        # Count carbons in substituent
        carbon_count = _count_alkyl_carbons(mol, carbon_neighbors[0].GetIdx(), {cation_site['atom_idx']})
        ALKYL_NAMES = {1: 'methyl', 2: 'ethyl', 3: 'propyl', 4: 'butyl'}
        alkyl = ALKYL_NAMES.get(carbon_count, f'{carbon_count}C-')
        return alkyl + 'ammonium'
    elif len(carbon_neighbors) == 4:
        # Quaternary: tetra-alkyl-ammonium
        alkyl_names = []
        for neighbor in carbon_neighbors:
            count = _count_alkyl_carbons(mol, neighbor.GetIdx(), {cation_site['atom_idx']})
            ALKYL_NAMES = {1: 'methyl', 2: 'ethyl', 3: 'propyl', 4: 'butyl'}
            alkyl_names.append(ALKYL_NAMES.get(count, f'{count}C-'))

        # Check if all same
        if len(set(alkyl_names)) == 1:
            return 'tetra' + alkyl_names[0] + 'ammonium'
        else:
            # Sort alphabetically
            alkyl_names.sort()
            return ''.join(alkyl_names) + 'ammonium'
    else:
        return 'aminium'


def _name_carbenium_systematic(mol, cation_site: Dict) -> str:
    """Generate systematic name for carbenium (ylium) cation."""
    # Count total carbons
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    NAMES = {1: 'methylium', 2: 'ethylium', 3: 'propylium', 4: 'butylium'}
    return NAMES.get(carbon_count, f'{carbon_count}C-ylium')


def _name_onium_systematic(mol, cation_site: Dict) -> str:
    """Generate systematic name for onium cation (oxonium, sulfonium, phosphonium)."""
    element = cation_site['element']

    ONIUM_NAMES = {
        'O': 'oxonium',
        'S': 'sulfonium',
        'P': 'phosphonium',
        'Se': 'selenonium',
        'Te': 'telluronium'
    }

    base = ONIUM_NAMES.get(element, 'onium')

    # Check for substituents
    atom = mol.GetAtomWithIdx(cation_site['atom_idx'])
    carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']

    if len(carbon_neighbors) == 0:
        return base
    elif len(carbon_neighbors) == 1:
        count = _count_alkyl_carbons(mol, carbon_neighbors[0].GetIdx(), {cation_site['atom_idx']})
        ALKYL_NAMES = {1: 'methyl', 2: 'ethyl', 3: 'propyl', 4: 'butyl'}
        alkyl = ALKYL_NAMES.get(count, '')
        return alkyl + base
    else:
        # Multiple substituents
        return 'tri' + base if len(carbon_neighbors) == 3 else base


def _name_diazonium_systematic(mol, cation_site: Dict) -> str:
    """Generate systematic name for diazonium cation."""
    # Find the attached group
    atom = mol.GetAtomWithIdx(cation_site['atom_idx'])

    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() == 'C':
            if neighbor.GetIsAromatic():
                return 'benzenediazonium'
            else:
                # Aliphatic diazonium
                return 'diazonium'

    return 'diazonium'


def _count_alkyl_carbons(mol, start_idx: int, exclude: set) -> int:
    """Count carbon atoms in an alkyl group via BFS."""
    from collections import deque

    visited = set()
    queue = deque([start_idx])
    count = 0

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'C':
            count += 1
            for neighbor in atom.GetNeighbors():
                nbr_idx = neighbor.GetIdx()
                if nbr_idx not in visited and nbr_idx not in exclude:
                    if neighbor.GetSymbol() == 'C':
                        queue.append(nbr_idx)

    return count
