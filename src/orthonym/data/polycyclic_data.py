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
"""

from typing import Dict, Optional, List, Tuple, Any


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
