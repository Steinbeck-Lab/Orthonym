"""
Aromatic reference templates for partial saturation detection.

This module provides aromatic parent SMILES for comparing against partially
saturated fused heterocycles. By comparing a molecule's ring atoms against
its aromatic parent, we can detect the saturation level and generate
appropriate prefixes (dihydro-, tetrahydro-, hexahydro-, perhydro-).

IUPAC 2013 Blue Book P-31.1.1: Hydro prefixes indicate the addition of
hydrogen to an otherwise unsaturated parent structure.

Key aromatic systems covered:
- Bicyclic: quinoline, isoquinoline, indole, benzofuran, naphthalene
- Tricyclic: acridine, carbazole, phenazine, purine, pteridine
- Monocyclic: furan, pyrrole, pyridine (for simple dihydro cases)
"""

from typing import Dict, Optional, Tuple, Any
from rdkit import Chem


# Aromatic reference SMILES for common fused heterocycles
# Maps ring system family names to canonical SMILES of the aromatic parent
AROMATIC_REFERENCES: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # BENZO-FUSED 6-MEMBERED HETEROCYCLES
    # =========================================================================

    'quinoline': {
        'smiles': 'c1ccc2ncccc2c1',
        'ring_atoms': 10,
        'description': 'benzo[b]pyridine',
    },

    'isoquinoline': {
        'smiles': 'c1ccc2cnccc2c1',
        'ring_atoms': 10,
        'description': 'benzo[c]pyridine',
    },

    'quinazoline': {
        'smiles': 'c1ccc2ncncc2c1',
        'ring_atoms': 10,
        'description': 'benzo[d]pyrimidine',
    },

    'quinoxaline': {
        'smiles': 'c1ccc2nccnc2c1',
        'ring_atoms': 10,
        'description': 'benzo[e]pyrazine',
    },

    'cinnoline': {
        'smiles': 'c1ccc2cnncc2c1',
        'ring_atoms': 10,
        'description': 'benzo[c]pyridazine',
    },

    'phthalazine': {
        'smiles': 'c1ccc2nnccc2c1',
        'ring_atoms': 10,
        'description': 'benzo[d]pyridazine',
    },

    # =========================================================================
    # BENZO-FUSED 5-MEMBERED HETEROCYCLES
    # =========================================================================

    'indole': {
        'smiles': 'c1ccc2[nH]ccc2c1',
        'ring_atoms': 9,
        'description': 'benzo[b]pyrrole',
    },

    'isoindole': {
        'smiles': 'c1ccc2c[nH]cc2c1',
        'ring_atoms': 9,
        'description': 'benzo[c]pyrrole',
    },

    'benzofuran': {
        'smiles': 'c1ccc2occc2c1',
        'ring_atoms': 9,
        'description': 'benzo[b]furan',
    },

    'benzothiophene': {
        'smiles': 'c1ccc2sccc2c1',
        'ring_atoms': 9,
        'description': 'benzo[b]thiophene',
    },

    'indazole': {
        'smiles': 'c1ccc2[nH]ncc2c1',
        'ring_atoms': 9,
        'description': 'benzo[c]pyrazole',
    },

    'benzimidazole': {
        'smiles': 'c1ccc2[nH]cnc2c1',
        'ring_atoms': 9,
        'description': 'benzo[d]imidazole',
    },

    'benzoxazole': {
        'smiles': 'c1ccc2ocnc2c1',
        'ring_atoms': 9,
        'description': 'benzo[d]oxazole',
    },

    'benzothiazole': {
        'smiles': 'c1ccc2scnc2c1',
        'ring_atoms': 9,
        'description': 'benzo[d]thiazole',
    },

    # =========================================================================
    # POLYCYCLIC AROMATICS (carbocyclic)
    # =========================================================================

    'naphthalene': {
        'smiles': 'c1ccc2ccccc2c1',
        'ring_atoms': 10,
        'description': 'bicyclic aromatic hydrocarbon',
        'is_carbocycle': True,
    },

    'anthracene': {
        'smiles': 'c1ccc2cc3ccccc3cc2c1',
        'ring_atoms': 14,
        'description': 'tricyclic aromatic hydrocarbon',
        'is_carbocycle': True,
    },

    'phenanthrene': {
        'smiles': 'c1ccc2c(c1)ccc1ccccc12',
        'ring_atoms': 14,
        'description': 'tricyclic angular aromatic hydrocarbon',
        'is_carbocycle': True,
    },

    # =========================================================================
    # TRICYCLIC HETEROCYCLES
    # =========================================================================

    'acridine': {
        'smiles': 'c1ccc2nc3ccccc3cc2c1',
        'ring_atoms': 14,
        'description': 'dibenzo[b,e]pyridine',
    },

    'carbazole': {
        'smiles': 'c1ccc2c(c1)[nH]c1ccccc12',
        'ring_atoms': 13,
        'description': 'dibenzo[b,d]pyrrole',
    },

    'phenazine': {
        'smiles': 'c1ccc2nc3ccccc3nc2c1',
        'ring_atoms': 14,
        'description': 'dibenzo[b,e]pyrazine',
    },

    # RING-05: Xanthene (9H-xanthene, dibenzo[b,e]pyran)
    'xanthene': {
        'smiles': 'c1ccc2c(c1)oc1ccccc1c2',
        'ring_atoms': 13,
        'description': 'dibenzo[b,e]pyran (9H-xanthene)',
    },

    # =========================================================================
    # PURINES AND PTERIDINES (bicyclic N-heterocycles)
    # =========================================================================

    'purine': {
        'smiles': 'c1ncc2nc[nH]c2n1',
        'ring_atoms': 9,
        'description': 'imidazo[4,5-d]pyrimidine',
    },

    'pteridine': {
        'smiles': 'c1cnc2ncncc2n1',
        'ring_atoms': 10,
        'description': 'pyrimido[4,5-b]pyrazine',
    },

    # =========================================================================
    # SIMPLE MONOCYCLIC REFERENCES
    # For partial saturation of simple heterocycles (dihydrofuran, etc.)
    # =========================================================================

    'furan': {
        'smiles': 'c1ccoc1',
        'ring_atoms': 5,
        'description': '5-membered O-heterocycle',
    },

    'pyrrole': {
        'smiles': 'c1cc[nH]c1',
        'ring_atoms': 5,
        'description': '5-membered N-heterocycle',
    },

    'thiophene': {
        'smiles': 'c1ccsc1',
        'ring_atoms': 5,
        'description': '5-membered S-heterocycle',
    },

    'pyridine': {
        'smiles': 'c1ccncc1',
        'ring_atoms': 6,
        'description': '6-membered N-heterocycle',
    },

    'pyrimidine': {
        'smiles': 'c1cncnc1',
        'ring_atoms': 6,
        'description': '6-membered diN-heterocycle',
    },

    'imidazole': {
        'smiles': 'c1c[nH]cn1',
        'ring_atoms': 5,
        'description': '5-membered 1,3-diN-heterocycle',
    },

    'pyrazole': {
        'smiles': 'c1cc[nH]n1',
        'ring_atoms': 5,
        'description': '5-membered 1,2-diN-heterocycle',
    },

    'oxazole': {
        'smiles': 'c1cocn1',
        'ring_atoms': 5,
        'description': '5-membered 1,3-oxazole',
    },

    'thiazole': {
        'smiles': 'c1cscn1',
        'ring_atoms': 5,
        'description': '5-membered 1,3-thiazole',
    },

    # =========================================================================
    # ADDITIONAL MONOCYCLIC AND POLYCYCLIC REFERENCES (Phase 89)
    # =========================================================================

    'pyrazine': {
        'smiles': 'c1cnccn1',
        'ring_atoms': 6,
        'description': '6-membered 1,4-diN-heterocycle',
    },

    'pyridazine': {
        'smiles': 'c1ccnnc1',
        'ring_atoms': 6,
        'description': '6-membered 1,2-diN-heterocycle',
    },

    'fluorene': {
        'smiles': 'c1ccc2c(c1)Cc1ccccc1-2',
        'ring_atoms': 13,
        'description': 'tricyclic carbocyclic PAH',
        'is_carbocycle': True,
    },
}


# Cache for RDKit Mol objects created from reference SMILES
_REFERENCE_MOLS: Dict[str, Chem.Mol] = {}


def _get_reference_mol(name: str) -> Optional[Chem.Mol]:
    """
    Get cached RDKit Mol object for a reference compound.

    Args:
        name: Reference compound name (key in AROMATIC_REFERENCES)

    Returns:
        RDKit Mol object, or None if not found
    """
    global _REFERENCE_MOLS

    if name not in _REFERENCE_MOLS:
        ref_data = AROMATIC_REFERENCES.get(name)
        if ref_data:
            mol = Chem.MolFromSmiles(ref_data['smiles'])
            if mol:
                _REFERENCE_MOLS[name] = mol

    return _REFERENCE_MOLS.get(name)


def get_aromatic_reference(mol: Chem.Mol) -> Optional[Tuple[str, str]]:
    """
    Find the aromatic parent structure for a molecule.

    Performs substructure matching against known aromatic reference structures
    to identify which aromatic parent the molecule is derived from. This is
    essential for determining partial saturation (dihydro-, tetrahydro-, etc.).

    The function returns the largest matching reference (by ring atoms) to
    handle nested systems correctly (e.g., naphthalene subsumes benzene).

    Args:
        mol: RDKit molecule object

    Returns:
        Tuple of (reference_name, reference_smiles) if a match is found,
        None otherwise.

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')  # tetrahydroquinoline
        >>> get_aromatic_reference(mol)
        ('quinoline', 'c1ccc2ncccc2c1')

        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCN2')  # dihydroindole
        >>> get_aromatic_reference(mol)
        ('indole', 'c1ccc2[nH]ccc2c1')
    """
    if mol is None:
        return None

    best_match: Optional[Tuple[str, str, int]] = None

    for name, ref_data in AROMATIC_REFERENCES.items():
        ref_mol = _get_reference_mol(name)
        if ref_mol is None:
            continue

        # Try substructure match
        if mol.HasSubstructMatch(ref_mol):
            ring_atoms = ref_data['ring_atoms']

            # Keep the largest matching reference
            if best_match is None or ring_atoms > best_match[2]:
                best_match = (name, ref_data['smiles'], ring_atoms)

    if best_match:
        return (best_match[0], best_match[1])

    return None


def get_reference_smiles(name: str) -> Optional[str]:
    """
    Get the aromatic SMILES for a named reference compound.

    Args:
        name: Reference compound name (e.g., 'quinoline', 'indole')

    Returns:
        Canonical SMILES string, or None if not found
    """
    ref_data = AROMATIC_REFERENCES.get(name)
    if ref_data:
        return ref_data['smiles']
    return None


def get_reference_ring_atoms(name: str) -> int:
    """
    Get the number of ring atoms for a named reference compound.

    Args:
        name: Reference compound name

    Returns:
        Number of ring atoms, or 0 if not found
    """
    ref_data = AROMATIC_REFERENCES.get(name)
    if ref_data:
        return ref_data['ring_atoms']
    return 0


def list_reference_names() -> list:
    """Return list of all available reference compound names."""
    return list(AROMATIC_REFERENCES.keys())


def get_carbocyclic_aromatic_reference(mol: Chem.Mol) -> Optional[Tuple[str, str]]:
    """
    Find carbocyclic aromatic parent structure for a molecule.

    This is similar to get_aromatic_reference but only matches
    carbocyclic systems (naphthalene, anthracene, phenanthrene).
    Used for partial saturation detection in PAH systems.

    Args:
        mol: RDKit molecule object

    Returns:
        Tuple of (reference_name, reference_smiles) if a carbocyclic match found,
        None otherwise.

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCC2')  # tetrahydronaphthalene
        >>> get_carbocyclic_aromatic_reference(mol)
        ('naphthalene', 'c1ccc2ccccc2c1')
    """
    if mol is None:
        return None

    best_match: Optional[Tuple[str, str, int]] = None

    for name, ref_data in AROMATIC_REFERENCES.items():
        # Only consider carbocyclic references
        if not ref_data.get('is_carbocycle', False):
            continue

        ref_mol = _get_reference_mol(name)
        if ref_mol is None:
            continue

        # Try substructure match
        if mol.HasSubstructMatch(ref_mol):
            ring_atoms = ref_data['ring_atoms']

            # Keep the largest matching reference
            if best_match is None or ring_atoms > best_match[2]:
                best_match = (name, ref_data['smiles'], ring_atoms)

    if best_match:
        return (best_match[0], best_match[1])

    return None


def is_carbocyclic_reference(name: str) -> bool:
    """
    Check if a reference compound is a carbocycle.

    Args:
        name: Reference compound name

    Returns:
        True if the reference is a carbocyclic aromatic (no heteroatoms)
    """
    ref_data = AROMATIC_REFERENCES.get(name)
    if ref_data:
        return ref_data.get('is_carbocycle', False)
    return False
