"""
Amino acid detection and naming.

Alpha-amino acids have the general structure: H2N-CHR-COOH
SMARTS pattern: [NX3;H2,H1][CX4][CX3](=O)[OX2H1]

For standard amino acids, trivial names are used.
For non-standard, systematic naming applies:
- Carboxylic acid is principal group
- Amine becomes "amino" prefix
"""

from typing import Optional, List, Tuple
from rdkit import Chem

from ..data.amino_acids import get_amino_acid_name, is_standard_amino_acid


# SMARTS for alpha-amino acid pattern
# [NX3;H2,H1] - primary or secondary amine nitrogen
# [CX4] - sp3 carbon (alpha carbon)
# [CX3](=O)[OX2H1] - carboxylic acid
ALPHA_AMINO_ACID_SMARTS = "[NX3;H2,H1][CX4][CX3](=O)[OX2H1]"


def detect_amino_acid(mol) -> bool:
    """
    Check if molecule is an amino acid.

    Detects alpha-amino acid pattern: NH2-CH(R)-COOH

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule contains alpha-amino acid pattern
    """
    pattern = Chem.MolFromSmarts(ALPHA_AMINO_ACID_SMARTS)
    if pattern is None:
        return False
    return mol.HasSubstructMatch(pattern)


def get_amino_acid_atoms(mol) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Get atom indices of alpha-amino acid core.

    The SMARTS pattern [NX3;H2,H1][CX4][CX3](=O)[OX2H1] matches:
    - N (amino nitrogen)
    - alpha_C (alpha carbon attached to N)
    - carbonyl_C (carboxylic acid carbon)
    - carbonyl_O (=O oxygen)
    - acid_O (OH oxygen)

    Returns:
        Tuple of (N, alpha_C, carbonyl_C, carbonyl_O, acid_O) atom indices, or None
    """
    pattern = Chem.MolFromSmarts(ALPHA_AMINO_ACID_SMARTS)
    if pattern is None:
        return None

    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return None

    # Return first match (5 atoms)
    return matches[0]


def name_amino_acid(mol, canonical_smiles: str) -> Optional[str]:
    """
    Generate name for an amino acid.

    For standard amino acids, returns trivial name.
    For non-standard, returns systematic name.

    Args:
        mol: RDKit Mol object
        canonical_smiles: Canonical SMILES of the molecule

    Returns:
        Amino acid name, or None if not an amino acid
    """
    # Check if it's an amino acid at all
    if not detect_amino_acid(mol):
        return None

    # Try trivial name lookup first
    trivial = get_amino_acid_name(canonical_smiles)
    if trivial:
        return trivial

    # Generate systematic name
    return _name_amino_acid_systematic(mol)


def _name_amino_acid_systematic(mol) -> str:
    """
    Generate systematic IUPAC name for non-standard amino acid.

    Uses: "amino" prefix + acid name
    Example: 2-aminopropanoic acid (systematic for alanine)
    """
    from ..assembly.composer import CHAIN_PREFIXES

    # Get amino acid core atoms
    aa_atoms = get_amino_acid_atoms(mol)
    if not aa_atoms:
        return None

    # Unpack 5 atoms: N, alpha_C, carbonyl_C, carbonyl_O (=O), acid_O (OH)
    n_atom, alpha_c, carbonyl_c, carbonyl_o, acid_o = aa_atoms

    # Count carbons in the backbone (acid chain)
    # The acid chain starts at carbonyl carbon
    # For alpha-amino acids, alpha-C is position 2

    # Simple approach: count total carbons
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    # Get stem from carbon count
    if carbon_count in CHAIN_PREFIXES:
        stem = CHAIN_PREFIXES[carbon_count]
    else:
        stem = f"{carbon_count}C"

    # For alpha-amino acids, the amino group is at position 2
    # (position 1 is the acid carbon)
    return f"2-amino{stem}anoic acid"


def is_n_substituted_amino_acid(mol) -> bool:
    """
    Check if amino acid has N-substituents (like sarcosine = N-methylglycine).
    """
    # Look for secondary amine in amino acid context
    pattern = Chem.MolFromSmarts("[NX3;H1]([CX4])[CX4][CX3](=O)[OX2H1]")
    if pattern is None:
        return False
    return mol.HasSubstructMatch(pattern)
