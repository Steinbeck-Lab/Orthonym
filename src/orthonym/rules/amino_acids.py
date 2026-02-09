"""
Amino acid detection and naming.

Alpha-amino acids have the general structure: H2N-CHR-COOH
SMARTS pattern: [NX3;H2,H1][CX4][CX3](=O)[OX2H1]

For standard amino acids, trivial names are used.
For non-standard, systematic naming applies:
- Carboxylic acid is principal group
- Amine becomes "amino" prefix

Peptide guard: molecules with peptide bonds (-C(=O)-NH-) are NOT simple amino
acids and must NOT be flattened to "2-aminoXXXanoic acid".
"""

from typing import Optional, List, Tuple
from rdkit import Chem

from ..data.amino_acids import get_amino_acid_name, is_standard_amino_acid


# SMARTS for alpha-amino acid pattern
# [NX3;H2,H1] - primary or secondary amine nitrogen
# [CX4] - sp3 carbon (alpha carbon)
# [CX3](=O)[OX2H1] - carboxylic acid
ALPHA_AMINO_ACID_SMARTS = "[NX3;H2,H1][CX4][CX3](=O)[OX2H1]"

# SMARTS for peptide bond (secondary amide linkage between amino acids)
# [NX3;H1] - secondary amide nitrogen (NH, not NH2)
# [CX3](=O) - carbonyl carbon
# [CX4] - alpha carbon on the acid side
# This detects -C(=O)-NH-CH- linkages typical of peptide bonds
PEPTIDE_BOND_SMARTS = "[CX3](=O)[NX3;H1][CX4]"


def count_peptide_bonds(mol) -> int:
    """
    Count the number of peptide bonds (-C(=O)-NH-CH-) in a molecule.

    Peptide bonds link amino acid residues. A molecule with >= 1 peptide bond
    is a peptide, not a simple amino acid.

    Args:
        mol: RDKit Mol object

    Returns:
        Number of peptide bond matches found
    """
    pattern = Chem.MolFromSmarts(PEPTIDE_BOND_SMARTS)
    if pattern is None:
        return 0
    matches = mol.GetSubstructMatches(pattern)
    return len(matches)


def is_peptide(mol) -> bool:
    """
    Check if molecule contains peptide bonds (is a di/tri/polypeptide).

    A molecule with one or more -C(=O)-NH-CH- linkages is a peptide,
    not a simple amino acid. Must also have the amino acid pattern
    (terminal NH2 + COOH) to distinguish from random amides.

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule is a peptide (has amino acid pattern AND peptide bonds)
    """
    # Must have amino acid pattern AND peptide bonds
    if not detect_amino_acid(mol):
        return False
    return count_peptide_bonds(mol) >= 1


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

    Peptide guard: if the molecule contains peptide bonds (-C(=O)-NH-),
    it is NOT a simple amino acid. Returns None so the molecule falls
    through to the general naming pipeline (or returns None for
    complex peptides beyond current scope).

    Args:
        mol: RDKit Mol object
        canonical_smiles: Canonical SMILES of the molecule

    Returns:
        Amino acid name, or None if not an amino acid or is a peptide
    """
    # Check if it's an amino acid at all
    if not detect_amino_acid(mol):
        return None

    # PEPTIDE GUARD: check for peptide bonds before naming as amino acid
    # Molecules with peptide bonds are peptides, not simple amino acids
    n_peptide_bonds = count_peptide_bonds(mol)
    if n_peptide_bonds >= 1:
        # This is a peptide (di-, tri-, or polypeptide)
        # Return None to let it fall through to general naming pipeline
        # For 2+ peptide bonds (tripeptide+), these are beyond current scope
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

    For amino acids with multiple amino groups (e.g., lysine),
    returns None to let the general pipeline handle it.
    """
    from ..data.chain_names import get_chain_prefix

    # Get amino acid core atoms
    aa_atoms = get_amino_acid_atoms(mol)
    if not aa_atoms:
        return None

    # Count primary amine groups - if more than one, let general pipeline handle
    amine_pattern = Chem.MolFromSmarts('[NX3;H2;!$([NX3][CX3]=O)]')
    if amine_pattern:
        amine_matches = mol.GetSubstructMatches(amine_pattern)
        if len(amine_matches) > 1:
            return None  # Multiple amines - use general naming pipeline

    # If the molecule has rings, the simple carbon-count approach would
    # include ring carbons in the chain length (e.g., tyrosine would give
    # "2-aminononanoic acid" instead of falling through to the general
    # pipeline which correctly uses parent selection with ring-atom exclusion).
    ri = mol.GetRingInfo()
    if ri.NumRings() > 0:
        return None  # Let general pipeline handle ring-containing amino acids

    # Count carbons in the backbone (acid chain) -- safe for acyclic molecules
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    # Get stem from carbon count using centralized module
    stem = get_chain_prefix(carbon_count)

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
