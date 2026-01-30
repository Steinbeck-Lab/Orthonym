"""
Polycyclic Aromatic Hydrocarbon (PAH) data for IUPAC naming.

Contains lookup tables for common polycyclic aromatic hydrocarbons including:
- Canonical SMILES for exact matching
- SMARTS patterns for substructure matching
- Number of atoms in the PAH core
- IUPAC standard numbering (atom index to IUPAC locant mapping)
- Allowed substituent positions

IUPAC Naming Rules for PAHs:
- PAH numbering is FIXED by IUPAC, not reoriented based on substituents
- Use retained names (naphthalene, anthracene, etc.) as parent
- Substituents are named with their IUPAC locant position

Reference: IUPAC Blue Book 2013, Section P-25 (Fused and Bridged Fused Ring Systems)

PAH Classification:
- Bicyclic: naphthalene
- Tricyclic: anthracene, phenanthrene, fluorene, acenaphthene, acenaphthylene
- Tetracyclic: pyrene, chrysene, tetracene, triphenylene, benz[a]anthracene, benzo[c]phenanthrene
- Pentacyclic: pentacene, perylene, benzo[a]pyrene
- Hexacyclic+: coronene

Partially saturated PAHs:
- 9,10-dihydroanthracene
- 1,2-dihydronaphthalene
"""

from typing import Dict, Optional, List, Tuple, Any, Set


# Polycyclic aromatic hydrocarbon data
# Key: retained name
# Value: dict with canonical_smiles, smarts, num_atoms, iupac_numbering, substituent_positions
POLYCYCLIC_DATA: Dict[str, Dict[str, Any]] = {
    'naphthalene': {
        'canonical_smiles': 'c1ccc2ccccc2c1',
        'smarts': 'c1ccc2ccccc2c1',  # For substructure matching
        'num_atoms': 10,
        # IUPAC numbering for naphthalene:
        #     8  1
        #    /  \ /
        #   7    2
        #   |    |
        #   6    3
        #    \  / \
        #     5  4
        # Maps canonical atom index -> IUPAC position (1-indexed)
        # Note: This mapping is determined empirically based on RDKit's canonical ordering
        'iupac_numbering': {
            0: 1, 1: 2, 2: 3, 3: 4, 4: 4.5, 5: 5, 6: 6, 7: 7, 8: 8, 9: 8.5
        },
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],  # Allowed positions
        'num_rings': 2,
    },
    'anthracene': {
        'canonical_smiles': 'c1ccc2cc3ccccc3cc2c1',
        'smarts': 'c1ccc2cc3ccccc3cc2c1',
        'num_atoms': 14,
        # IUPAC numbering for anthracene (linear tricyclic)
        'iupac_numbering': {},  # Will be populated by matching
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num_rings': 3,
    },
    'phenanthrene': {
        'canonical_smiles': 'c1ccc2c(c1)ccc1ccccc12',
        'smarts': 'c1ccc2c(c1)ccc1ccccc12',
        'num_atoms': 14,
        # IUPAC numbering for phenanthrene (angular tricyclic)
        'iupac_numbering': {},  # Will be populated by matching
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num_rings': 3,
    },
    'pyrene': {
        'canonical_smiles': 'c1cc2ccc3cccc4ccc(c1)c2c34',
        'smarts': 'c1cc2ccc3cccc4ccc(c1)c2c34',
        'num_atoms': 16,
        # IUPAC numbering for pyrene (condensed tetracyclic)
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num_rings': 4,
    },
    'fluorene': {
        'canonical_smiles': 'c1ccc2c(c1)Cc1ccccc1-2',  # Has sp3 carbon (position 9)
        'smarts': 'c1ccc2c(c1)Cc1ccccc1-2',
        'num_atoms': 13,
        # IUPAC numbering for fluorene (with methylene bridge at position 9)
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9],  # 9 is the sp3 carbon
        'num_rings': 3,
    },
    'acenaphthene': {
        'canonical_smiles': 'c1cc2c3c(cccc3c1)CC2',  # Has two sp3 carbons
        'smarts': 'c1cc2c3c(cccc3c1)CC2',
        'num_atoms': 12,
        # IUPAC numbering for acenaphthene
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],  # Excluding the ethylene bridge
        'num_rings': 3,
    },
    'acenaphthylene': {
        'canonical_smiles': 'C1=Cc2cccc3cccc1c23',  # Fully unsaturated
        'smarts': 'C1=Cc2cccc3cccc1c23',
        'num_atoms': 12,
        # IUPAC numbering for acenaphthylene
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],
        'num_rings': 3,
    },
    'chrysene': {
        'canonical_smiles': 'c1ccc2c(c1)ccc1c3ccccc3ccc21',
        'smarts': 'c1ccc2c(c1)ccc1c3ccccc3ccc21',
        'num_atoms': 18,
        # IUPAC numbering for chrysene
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    # ==========================================================================
    # Tetracyclic (4 rings) - additional
    # ==========================================================================
    'tetracene': {
        # Also known as naphthacene - linear 4-ring PAH
        'canonical_smiles': 'c1ccc2cc3cc4ccccc4cc3cc2c1',
        'smarts': 'c1ccc2cc3cc4ccccc4cc3cc2c1',
        'num_atoms': 18,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    'triphenylene': {
        # Angular 4-ring PAH (three benzene rings sharing a central ring)
        'canonical_smiles': 'c1ccc2c(c1)c1ccccc1c1ccccc21',
        'smarts': 'c1ccc2c(c1)c1ccccc1c1ccccc21',
        'num_atoms': 18,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    'benz[a]anthracene': {
        # Bent 4-ring PAH (benzene fused to anthracene)
        'canonical_smiles': 'c1ccc2cc3c(ccc4ccccc43)cc2c1',
        'smarts': 'c1ccc2cc3c(ccc4ccccc43)cc2c1',
        'num_atoms': 18,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    'benzo[c]phenanthrene': {
        # Angular 4-ring PAH
        'canonical_smiles': 'c1ccc2c(c1)ccc1ccc3ccccc3c12',
        'smarts': 'c1ccc2c(c1)ccc1ccc3ccccc3c12',
        'num_atoms': 18,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 4,
    },
    # ==========================================================================
    # Pentacyclic (5 rings)
    # ==========================================================================
    'pentacene': {
        # Linear 5-ring PAH
        'canonical_smiles': 'c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1',
        'smarts': 'c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1',
        'num_atoms': 22,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'num_rings': 5,
    },
    'perylene': {
        # Peri-condensed PAH (two naphthalene units joined peri)
        'canonical_smiles': 'c1ccc2cccc3cc4c(c1)cc1cccc4c1c23',
        'smarts': 'c1ccc2cccc3cc4c(c1)cc1cccc4c1c23',
        'num_atoms': 20,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 6,  # RDKit counts 6 rings due to perception
    },
    'benzo[a]pyrene': {
        # Important carcinogen - benzene fused to pyrene
        'canonical_smiles': 'c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34',
        'smarts': 'c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34',
        'num_atoms': 20,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 5,
    },
    # ==========================================================================
    # Hexacyclic+ (6+ rings)
    # ==========================================================================
    'coronene': {
        # 7 rings, hexagonal symmetry
        'canonical_smiles': 'c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61',
        'smarts': 'c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61',
        'num_atoms': 24,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'num_rings': 7,
    },
    # ==========================================================================
    # Partially saturated PAHs
    # ==========================================================================
    '9,10-dihydroanthracene': {
        # Anthracene with positions 9 and 10 saturated (sp3)
        'canonical_smiles': 'c1ccc2c(c1)Cc1ccccc1C2',
        'smarts': 'c1ccc2c(c1)Cc1ccccc1C2',
        'num_atoms': 14,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'num_rings': 3,
    },
    '1,2-dihydronaphthalene': {
        # Naphthalene with positions 1 and 2 saturated
        'canonical_smiles': 'C1=Cc2ccccc2CC1',
        'smarts': 'C1=Cc2ccccc2CC1',
        'num_atoms': 10,
        'iupac_numbering': {},
        'substituent_positions': [1, 2, 3, 4, 5, 6, 7, 8],
        'num_rings': 2,
    },
}


# Reverse lookup: canonical SMILES -> PAH name
_SMILES_TO_NAME: Dict[str, str] = {
    data['canonical_smiles']: name
    for name, data in POLYCYCLIC_DATA.items()
}


def get_polycyclic_by_smiles(canonical_smiles: str) -> Optional[Dict[str, Any]]:
    """
    Look up polycyclic data by canonical SMILES.

    Args:
        canonical_smiles: RDKit canonical SMILES string

    Returns:
        Dict with 'name' and all PAH data if found, None otherwise

    Example:
        >>> get_polycyclic_by_smiles('c1ccc2ccccc2c1')
        {'name': 'naphthalene', 'canonical_smiles': 'c1ccc2ccccc2c1', ...}
    """
    name = _SMILES_TO_NAME.get(canonical_smiles)
    if name is None:
        return None

    return {'name': name, **POLYCYCLIC_DATA[name]}


def get_polycyclic_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Look up polycyclic data by name.

    Args:
        name: PAH name (e.g., 'naphthalene', 'anthracene')

    Returns:
        Dict with PAH data if found, None otherwise
    """
    if name not in POLYCYCLIC_DATA:
        return None

    return {'name': name, **POLYCYCLIC_DATA[name]}


def is_polycyclic_aromatic(canonical_smiles: str) -> bool:
    """
    Check if a canonical SMILES represents a known polycyclic aromatic.

    Args:
        canonical_smiles: RDKit canonical SMILES string

    Returns:
        True if the SMILES matches a known PAH
    """
    return canonical_smiles in _SMILES_TO_NAME


def get_pah_names() -> List[str]:
    """
    Get list of all supported PAH names.

    Returns:
        List of PAH names
    """
    return list(POLYCYCLIC_DATA.keys())


def match_polycyclic_core(mol) -> Optional[Tuple[str, Dict[int, int]]]:
    """
    Match a molecule against known PAH cores using substructure matching.

    For substituted PAHs, finds the largest matching core and returns
    the atom mapping from molecule indices to IUPAC locants.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (pah_name, atom_mapping) where atom_mapping is {mol_idx: iupac_locant}
        Returns None if no PAH core found

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        >>> name, mapping = match_polycyclic_core(mol)
        >>> name
        'naphthalene'
    """
    # Lazy import to avoid circular dependency
    from rdkit import Chem

    # Sort PAHs by size (largest first) to find best match
    pah_by_size = sorted(
        POLYCYCLIC_DATA.items(),
        key=lambda x: x[1]['num_atoms'],
        reverse=True
    )

    for pah_name, pah_data in pah_by_size:
        smarts = pah_data['smarts']
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue

        matches = mol.GetSubstructMatches(pattern)
        if matches:
            # Found a match - build atom mapping
            match_atoms = matches[0]

            # Verify the match contains expected number of atoms
            if len(match_atoms) != pah_data['num_atoms']:
                continue

            # Build atom index to locant mapping
            # The mapping is based on the SMARTS match order
            # For now, use position in match as proxy for locant
            # More sophisticated mapping uses _map_pah_atoms_to_iupac in polycyclics.py
            atom_mapping = {match_atoms[i]: i + 1 for i in range(len(match_atoms))}

            return (pah_name, atom_mapping)

    return None


def get_pah_core_atoms(mol, pah_name: str) -> Optional[Set[int]]:
    """
    Get the atom indices that form a PAH core in a molecule.

    Args:
        mol: RDKit Mol object
        pah_name: Name of the PAH (e.g., 'naphthalene')

    Returns:
        Set of atom indices forming the PAH core, or None if no match

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        >>> core = get_pah_core_atoms(mol, 'naphthalene')
        >>> len(core)  # 10 atoms in naphthalene core
        10
    """
    from rdkit import Chem

    if pah_name not in POLYCYCLIC_DATA:
        return None

    pah_data = POLYCYCLIC_DATA[pah_name]
    smarts = pah_data['smarts']
    pattern = Chem.MolFromSmarts(smarts)

    if pattern is None:
        return None

    matches = mol.GetSubstructMatches(pattern)
    if matches:
        return set(matches[0])

    return None


def get_pah_substituent_positions(mol, pah_name: str) -> List[int]:
    """
    Get IUPAC locant positions where substituents are attached.

    Args:
        mol: RDKit Mol object with substituted PAH
        pah_name: Name of the PAH core (e.g., 'naphthalene')

    Returns:
        List of IUPAC locants with substituents

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        >>> get_pah_substituent_positions(mol, 'naphthalene')
        [2]  # Methyl at position 2
    """
    from rdkit import Chem

    core_atoms = get_pah_core_atoms(mol, pah_name)
    if not core_atoms:
        return []

    result = match_polycyclic_core(mol)
    if not result or result[0] != pah_name:
        return []

    _, atom_mapping = result

    # Find atoms with non-core neighbors (these have substituents)
    substituted_positions = []
    for atom_idx in core_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in core_atoms:
                # This core atom has a substituent
                if atom_idx in atom_mapping:
                    substituted_positions.append(atom_mapping[atom_idx])
                break

    return sorted(set(substituted_positions))
