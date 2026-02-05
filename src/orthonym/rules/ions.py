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

def name_anion(mol, style: str = 'pin', _depth: int = 0) -> str:
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
        _depth: Internal recursion depth guard (do not set manually)

    Returns:
        Anion name (e.g., 'acetate', 'methoxide', 'phenolate'),
        or empty string if naming fails

    Example:
        >>> mol = Chem.MolFromSmiles('CC(=O)[O-]')
        >>> name_anion(mol)
        'acetate'
    """
    if mol is None:
        return ''

    # Guard against infinite recursion
    if _depth > 2:
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

    # Multiple anions - try neutralize-then-name approach
    neutral_name = _try_neutralize_and_name(mol)
    if neutral_name:
        return neutral_name

    # Fallback: name first anion site only
    anion_site = anions[0]
    anion_type = classify_anion(mol, anion_site)
    if anion_type == 'carboxylate':
        return _name_carboxylate_systematic(mol, anion_site)
    return ''


def name_cation(mol, style: str = 'pin', _depth: int = 0) -> str:
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
        _depth: Internal recursion depth guard (do not set manually)

    Returns:
        Cation name (e.g., 'ammonium', 'methylammonium', 'methylium'),
        or empty string if naming fails

    Example:
        >>> mol = Chem.MolFromSmiles('[NH4+]')
        >>> name_cation(mol)
        'ammonium'
    """
    if mol is None:
        return ''

    # Guard against infinite recursion
    if _depth > 2:
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

    # Multiple cations - try neutralize-then-name approach
    neutral_name = _try_neutralize_and_name(mol)
    if neutral_name:
        return neutral_name

    # Fallback: name first cation site only
    cation_site = cations[0]
    cation_type = classify_cation(mol, cation_site)
    if cation_type == 'aminium':
        return _name_aminium_systematic(mol, cation_site)
    return ''


# === NEUTRALIZATION HELPER ===

# Metallic elements that are out of scope for organic naming
_INORGANIC_ELEMENTS = frozenset({
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
    'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
    'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Hf', 'Ta',
    'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
})


def _has_metal(mol) -> bool:
    """Check if molecule contains metallic/inorganic elements."""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in _INORGANIC_ELEMENTS:
            return True
    return False


def _try_neutralize_and_name(mol) -> str:
    """
    Neutralize a multi-charged ion and name the organic skeleton.

    For multi-charged species, strips all formal charges and names the
    resulting neutral molecule using the standard pipeline.

    Returns:
        Name of the neutralized skeleton, or empty string on failure.
    """
    if mol is None:
        return ''

    # Bail on inorganic/metallic species (out of scope)
    if _has_metal(mol):
        return ''

    try:
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            charge = atom.GetFormalCharge()
            if charge != 0:
                atom.SetFormalCharge(0)
                # Adjust hydrogen count to compensate
                if charge > 0:
                    # Cation: had extra H from protonation, remove them
                    cur_h = atom.GetNumExplicitHs()
                    atom.SetNumExplicitHs(max(0, cur_h - charge))
                elif charge < 0:
                    # Anion: was deprotonated, add H back
                    cur_h = atom.GetNumExplicitHs()
                    atom.SetNumExplicitHs(cur_h + abs(charge))

        try:
            Chem.SanitizeMol(rw)
        except Exception:
            return ''

        neutral_smiles = Chem.MolToSmiles(rw, canonical=True)
        if not neutral_smiles:
            return ''

        # Use namer to name the neutral form (deferred import to avoid circular)
        from ..namer import Orthonym
        namer = Orthonym(style='pin')
        neutral_name = namer.name(neutral_smiles)
        if neutral_name:
            return neutral_name
    except (RecursionError, ValueError, RuntimeError):
        pass

    return ''


# === SYSTEMATIC NAMING HELPERS ===

def _find_carboxyl_carbon(mol, anion_site: Dict) -> Optional[int]:
    """
    Find the carbon atom of the carboxyl group from the anionic oxygen.

    Args:
        mol: RDKit Mol object
        anion_site: Dictionary containing 'atom_idx' of the anionic oxygen

    Returns:
        Index of the carboxyl carbon, or None if not found
    """
    o_idx = anion_site['atom_idx']
    o_atom = mol.GetAtomWithIdx(o_idx)

    # Find the carbon attached to the anionic oxygen
    for neighbor in o_atom.GetNeighbors():
        if neighbor.GetSymbol() == 'C':
            # Verify this is a carboxyl carbon (has C=O double bond to another oxygen)
            for second_neighbor in neighbor.GetNeighbors():
                if second_neighbor.GetIdx() != o_idx and second_neighbor.GetSymbol() == 'O':
                    bond = mol.GetBondBetweenAtoms(neighbor.GetIdx(), second_neighbor.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                        return neighbor.GetIdx()
    return None


def _detect_aromatic_carboxylate(mol, carboxyl_carbon_idx: int) -> Optional[str]:
    """
    Detect if a carboxylate is attached to an aromatic ring.

    Args:
        mol: RDKit Mol object
        carboxyl_carbon_idx: Index of the carboxyl carbon

    Returns:
        'benzoate' for benzene-attached carboxylates,
        'naphthoate' for naphthalene-attached carboxylates,
        None for acyclic or other structures
    """
    carboxyl_carbon = mol.GetAtomWithIdx(carboxyl_carbon_idx)

    # Check neighbors for aromatic carbon
    for neighbor in carboxyl_carbon.GetNeighbors():
        if neighbor.GetSymbol() == 'C' and neighbor.GetIsAromatic():
            # Found aromatic carbon attached to carboxyl
            # Determine ring type
            ri = mol.GetRingInfo()
            atom_rings = ri.AtomRings()

            aromatic_neighbor_idx = neighbor.GetIdx()

            # Find which ring(s) contain this aromatic carbon
            for ring in atom_rings:
                if aromatic_neighbor_idx in ring:
                    ring_size = len(ring)
                    # Check if all atoms in ring are aromatic carbons (carbocycle)
                    all_aromatic_c = all(
                        mol.GetAtomWithIdx(idx).GetSymbol() == 'C' and
                        mol.GetAtomWithIdx(idx).GetIsAromatic()
                        for idx in ring
                    )

                    if ring_size == 6 and all_aromatic_c:
                        # Check if this is part of naphthalene (fused bicyclic)
                        if _is_naphthalene_system(mol, aromatic_neighbor_idx, atom_rings):
                            return 'naphthoate'
                        return 'benzoate'

    return None


def _is_naphthalene_system(mol, aromatic_idx: int, atom_rings) -> bool:
    """
    Check if an aromatic atom is part of a naphthalene (fused bicyclic) system.

    Args:
        mol: RDKit Mol object
        aromatic_idx: Index of an aromatic atom
        atom_rings: Ring information from RDKit

    Returns:
        True if part of a naphthalene system
    """
    # Find all 6-membered aromatic carbocyclic rings
    aromatic_6_rings = []
    for ring in atom_rings:
        if len(ring) == 6:
            all_aromatic_c = all(
                mol.GetAtomWithIdx(idx).GetSymbol() == 'C' and
                mol.GetAtomWithIdx(idx).GetIsAromatic()
                for idx in ring
            )
            if all_aromatic_c:
                aromatic_6_rings.append(set(ring))

    # Check if there are 2 fused 6-membered rings (sharing 2 atoms = naphthalene)
    if len(aromatic_6_rings) >= 2:
        for i, ring1 in enumerate(aromatic_6_rings):
            for ring2 in aromatic_6_rings[i+1:]:
                shared = ring1 & ring2
                if len(shared) == 2:  # Two shared atoms = fused rings
                    # Check if our aromatic atom is in either ring
                    if aromatic_idx in ring1 or aromatic_idx in ring2:
                        return True
    return False


def _name_aromatic_carboxylate_with_substituents(mol, carboxyl_carbon_idx: int, base_name: str) -> str:
    """
    Name an aromatic carboxylate with any substituents on the ring.

    Args:
        mol: RDKit Mol object
        carboxyl_carbon_idx: Index of the carboxyl carbon
        base_name: Base name ('benzoate' or 'naphthoate')

    Returns:
        Full name with substituent prefixes (e.g., '4-chlorobenzoate')
    """
    carboxyl_carbon = mol.GetAtomWithIdx(carboxyl_carbon_idx)

    # Find the aromatic ring attached to carboxyl
    aromatic_ring_atom = None
    for neighbor in carboxyl_carbon.GetNeighbors():
        if neighbor.GetSymbol() == 'C' and neighbor.GetIsAromatic():
            aromatic_ring_atom = neighbor
            break

    if aromatic_ring_atom is None:
        return base_name

    # Get the benzene ring
    ri = mol.GetRingInfo()
    benzene_ring = None
    for ring in ri.AtomRings():
        if aromatic_ring_atom.GetIdx() in ring and len(ring) == 6:
            # Check all atoms are aromatic carbons
            all_aromatic_c = all(
                mol.GetAtomWithIdx(idx).GetSymbol() == 'C' and
                mol.GetAtomWithIdx(idx).GetIsAromatic()
                for idx in ring
            )
            if all_aromatic_c:
                benzene_ring = ring
                break

    if benzene_ring is None:
        return base_name

    # Find substituents on the ring (excluding the carboxyl attachment point)
    ring_set = set(benzene_ring)
    carboxyl_attachment_idx = aromatic_ring_atom.GetIdx()

    # Map substituent type to name
    SUBSTITUENT_NAMES = {
        'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo',
        'N': 'amino', 'O': 'hydroxy'
    }

    # Alkyl group names by carbon count
    from ..assembly.naming_utils import get_alkyl_name as _get_alkyl_name
    ALKYL_NAMES = {i: _get_alkyl_name(i) for i in range(1, 11)}

    # Collect substituents: {position: [(name, sort_key), ...]}
    # Position 1 is the carboxyl attachment point
    substituents_by_position = {}

    # Orient ring: carboxyl attachment is position 1
    # Need to find ring order and direction for lowest locants
    ring_list = list(benzene_ring)

    # Find index of carboxyl attachment in ring
    carboxyl_pos = ring_list.index(carboxyl_attachment_idx)

    # Try both directions and all starting positions to get lowest locants
    best_orientation = None
    best_locants = None

    for direction in [1, -1]:
        oriented = []
        for i in range(6):
            idx = (carboxyl_pos + i * direction) % 6
            oriented.append(ring_list[idx])

        # Collect substituents for this orientation
        subs = {}
        for pos, atom_idx in enumerate(oriented, start=1):
            if pos == 1:
                continue  # Skip carboxyl attachment position

            ring_atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in ring_atom.GetNeighbors():
                nbr_idx = neighbor.GetIdx()
                if nbr_idx in ring_set:
                    continue  # Skip ring atoms
                if nbr_idx == carboxyl_carbon_idx:
                    continue  # Skip carboxyl carbon

                # Identify substituent
                symbol = neighbor.GetSymbol()
                if symbol in SUBSTITUENT_NAMES:
                    sub_name = SUBSTITUENT_NAMES[symbol]
                    if pos not in subs:
                        subs[pos] = []
                    subs[pos].append((sub_name, sub_name))  # (name, sort_key)
                elif symbol == 'C' and not neighbor.GetIsAromatic():
                    # Alkyl group - count carbons
                    carbon_count = _count_alkyl_carbons(mol, nbr_idx, ring_set | {carboxyl_carbon_idx})
                    if carbon_count in ALKYL_NAMES:
                        sub_name = ALKYL_NAMES[carbon_count]
                        if pos not in subs:
                            subs[pos] = []
                        subs[pos].append((sub_name, sub_name))

        # Calculate locant set for this orientation
        locants = sorted(subs.keys()) if subs else []

        if best_locants is None or _compare_locant_lists(locants, best_locants) < 0:
            best_locants = locants
            best_orientation = subs

    if not best_orientation:
        return base_name

    # Group substituents by name
    from collections import defaultdict
    grouped = defaultdict(list)
    for pos, sub_list in best_orientation.items():
        for sub_name, _ in sub_list:
            grouped[sub_name].append(pos)

    # Sort locants within each group
    for name in grouped:
        grouped[name].sort()

    # Build prefix: alphabetically sorted, with multipliers
    MULTIPLIERS = {1: '', 2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa'}

    prefixes = []
    for sub_name in sorted(grouped.keys()):
        locants = grouped[sub_name]
        count = len(locants)
        multiplier = MULTIPLIERS.get(count, f'{count}-')

        locant_str = ','.join(str(loc) for loc in locants)
        prefix = f"{locant_str}-{multiplier}{sub_name}"
        prefixes.append(prefix)

    if not prefixes:
        return base_name

    prefix_str = ''.join(prefixes)
    return f"{prefix_str}{base_name}"


def _compare_locant_lists(a: list, b: list) -> int:
    """Compare two locant lists by first-point-of-difference."""
    for i in range(max(len(a), len(b))):
        val_a = a[i] if i < len(a) else float('inf')
        val_b = b[i] if i < len(b) else float('inf')
        if val_a < val_b:
            return -1
        if val_a > val_b:
            return 1
    return 0


def _name_carboxylate_systematic(mol, anion_site: Dict) -> str:
    """Generate systematic name for carboxylate anion.

    Strategy: Neutralize the carboxylate ([O-] -> OH) to form the parent
    carboxylic acid, name it with the full naming pipeline (which handles
    substituents, stereo, unsaturation), then convert '-oic acid' to '-oate'.
    This ensures all substituents are properly detected and included.
    """
    # NEW: Check for aromatic parent FIRST
    carboxyl_carbon = _find_carboxyl_carbon(mol, anion_site)
    if carboxyl_carbon is not None:
        aromatic_name = _detect_aromatic_carboxylate(mol, carboxyl_carbon)
        if aromatic_name:
            return _name_aromatic_carboxylate_with_substituents(mol, carboxyl_carbon, aromatic_name)

    # Try neutralize-then-name approach for full substituent detection
    acid_name = _neutralize_carboxylate_to_acid(mol, anion_site)
    if acid_name:
        oate_name = _acid_to_oate(acid_name)
        if oate_name:
            return oate_name

    # Fallback: simple chain naming (no substituent detection)
    chain, double_bond_locs = _find_carboxylate_chain(mol, anion_site)
    carbon_count = len(chain) if chain else sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C')

    if carbon_count == 1:
        return 'formate'
    elif carbon_count == 2 and not double_bond_locs:
        return 'acetate'

    from ..data.chain_names import get_chain_prefix
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
    prefix = get_chain_prefix(carbon_count)

    if double_bond_locs:
        n_double = len(double_bond_locs)
        loc_str = ",".join(str(loc) for loc in sorted(double_bond_locs))
        if n_double == 1:
            return f"{prefix}-{loc_str}-enoate"
        else:
            mult = SIMPLE_MULTIPLIERS.get(n_double, str(n_double))
            return f"{prefix}a-{loc_str}-{mult}enoate"
    else:
        return prefix + "anoate"


def _neutralize_carboxylate_to_acid(mol, anion_site: Dict) -> str:
    """Neutralize carboxylate [O-] to OH and name as carboxylic acid.

    Returns the acid name or empty string on failure.
    """
    try:
        rw = Chem.RWMol(mol)
        o_idx = anion_site['atom_idx']
        o_atom = rw.GetAtomWithIdx(o_idx)
        o_atom.SetFormalCharge(0)
        o_atom.SetNumExplicitHs(o_atom.GetNumExplicitHs() + 1)

        try:
            Chem.SanitizeMol(rw)
        except Exception:
            return ''

        neutral_smiles = Chem.MolToSmiles(rw, canonical=True)
        if not neutral_smiles:
            return ''

        from ..namer import Orthonym
        namer = Orthonym(style='pin')
        acid_name = namer.name(neutral_smiles)
        if acid_name and ('oic acid' in acid_name or 'ic acid' in acid_name):
            return acid_name
    except (RecursionError, ValueError, RuntimeError):
        pass
    return ''


def _acid_to_oate(acid_name: str) -> str:
    """Convert a carboxylic acid name to its carboxylate (-oate) form.

    Handles:
    - 'X-oic acid' -> 'X-oate'
    - 'Xanoic acid' -> 'Xanoate'
    - Retained names: 'acetic acid' -> 'acetate', 'formic acid' -> 'formate'
    """
    if not acid_name:
        return ''

    # Retained acid -> retained oate
    retained_map = {
        'formic acid': 'formate',
        'acetic acid': 'acetate',
        'propionic acid': 'propanoate',
        'butyric acid': 'butanoate',
        'valeric acid': 'pentanoate',
        'isovaleric acid': '3-methylbutanoate',
    }
    if acid_name in retained_map:
        return retained_map[acid_name]

    # Standard conversion: '-oic acid' -> '-oate'
    if acid_name.endswith('oic acid'):
        return acid_name[:-len('oic acid')] + 'oate'
    # '-ic acid' (benzoic acid -> benzoate)
    if acid_name.endswith('ic acid'):
        return acid_name[:-len('ic acid')] + 'ate'
    return ''


def _find_carboxylate_chain(mol, anion_site: Dict):
    """Find the longest carbon chain from the carboxylate group.

    Returns (chain_atoms, double_bond_locants) where chain_atoms is a list
    of atom indices starting from the carboxylate carbon, and double_bond_locants
    is a list of IUPAC locants for C=C double bonds along the chain.
    """
    anion_idx = anion_site['atom_idx']
    atom = mol.GetAtomWithIdx(anion_idx)

    # Find the carboxyl carbon (C attached to charged O)
    carboxyl_c = None
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() == 'C':
            carboxyl_c = nbr.GetIdx()
            break
    if carboxyl_c is None:
        return None, []

    # BFS to find longest carbon chain from carboxyl C
    # Exclude the carboxylate oxygens from traversal
    carboxylate_os = set()
    for nbr in mol.GetAtomWithIdx(carboxyl_c).GetNeighbors():
        if nbr.GetSymbol() == 'O':
            carboxylate_os.add(nbr.GetIdx())

    def _longest_chain(start_idx, visited):
        """DFS to find longest carbon chain."""
        best = [start_idx]
        a = mol.GetAtomWithIdx(start_idx)
        for nbr in a.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in visited or nidx in carboxylate_os:
                continue
            if nbr.GetSymbol() != 'C':
                continue
            visited.add(nidx)
            sub = _longest_chain(nidx, visited)
            candidate = [start_idx] + sub
            if len(candidate) > len(best):
                best = candidate
            visited.discard(nidx)
        return best

    chain = _longest_chain(carboxyl_c, {carboxyl_c})

    # Find double bonds along the chain and their locants
    double_bond_locs = []
    for i in range(len(chain) - 1):
        bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
        if bond and bond.GetBondTypeAsDouble() == 2.0:
            double_bond_locs.append(i + 1)  # 1-indexed: C1 is carboxylate carbon

    return chain, double_bond_locs


def _name_alkoxide_systematic(mol, anion_site: Dict, style: str) -> str:
    """Generate systematic name for alkoxide anion."""
    # Count carbons
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    if carbon_count < 1:
        return ''  # No carbon chain - inorganic anion, not an alkoxide

    from ..data.chain_names import get_chain_prefix as _gcp
    base = _gcp(carbon_count) + 'an'

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
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    if carbon_count < 1:
        return ''

    from ..data.chain_names import get_chain_prefix as _gcp
    base = _gcp(carbon_count) + 'an'
    return base + 'ide'


def _name_thiolate_systematic(mol, anion_site: Dict) -> str:
    """Generate systematic name for thiolate anion."""
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    if carbon_count < 1:
        return ''

    from ..data.chain_names import get_chain_name
    base = get_chain_name(carbon_count)
    return base + 'thiolate'


def _name_aminium_systematic(mol, cation_site: Dict) -> str:
    """Generate systematic name for aminium cation."""
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
    # Count carbons attached to nitrogen
    atom = mol.GetAtomWithIdx(cation_site['atom_idx'])
    carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']

    if len(carbon_neighbors) == 0:
        # NH4+ -> ammonium
        return 'ammonium'

    from ..data.chain_names import get_alkyl_name as _gal

    if len(carbon_neighbors) == 1:
        # Count carbons in substituent
        carbon_count = _count_alkyl_carbons(mol, carbon_neighbors[0].GetIdx(), {cation_site['atom_idx']})
        alkyl = _gal(carbon_count)
        return alkyl + 'ammonium'

    # 2, 3, or 4 carbon neighbors
    alkyl_names = []
    for neighbor in carbon_neighbors:
        count = _count_alkyl_carbons(mol, neighbor.GetIdx(), {cation_site['atom_idx']})
        alkyl_names.append(_gal(count))

    # Check if all same
    if len(set(alkyl_names)) == 1:
        n = len(alkyl_names)
        mult = SIMPLE_MULTIPLIERS.get(n, str(n))
        return mult + alkyl_names[0] + 'ammonium'
    else:
        # Sort alphabetically and concatenate
        alkyl_names.sort()
        return ''.join(alkyl_names) + 'ammonium'


def _name_carbenium_systematic(mol, cation_site: Dict) -> str:
    """Generate systematic name for carbenium (ylium) cation."""
    # Count total carbons
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    NAMES = {1: 'methylium', 2: 'ethylium', 3: 'propylium', 4: 'butylium'}
    if carbon_count in NAMES:
        return NAMES[carbon_count]
    # Use centralized chain naming for longer chains
    from ..data.chain_names import get_chain_prefix
    return get_chain_prefix(carbon_count) + 'ylium'


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
        from ..assembly.naming_utils import get_alkyl_name as _gname
        try:
            alkyl = _gname(count)
        except (ValueError, KeyError):
            alkyl = ''
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
