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

# SMARTS patterns for additional functional groups (PEP-02 bailout)
# These detect FGs beyond amino + acid that _name_amino_acid_systematic
# cannot handle. When found, we bail out to the general polyfunctional pipeline.
_EXTRA_FG_SMARTS = [
    '[OX2H1;!$([OX2H1]C=O)]',  # Hydroxy OH (not in COOH)
    '[SX2H1]',                   # Thiol SH
    '[F,Cl,Br,I]',               # Halogen
    '[N+](=O)[O-]',              # Nitro
    '[CX3;!$([CX3](=O)[OX2H1]);!$([CX3](=O)[NX3])](=O)',  # Ketone C=O (excludes acid and amide C=O)
    '[CX3](=O)[NX3H2]',         # Primary amide -C(=O)NH2 (e.g., glutamine side chain)
    '[CX2]#[NX1]',              # Nitrile -C#N (e.g., cyano-amino acids)
]


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


def _has_extra_functional_groups(mol) -> bool:
    """
    Check if molecule has functional groups beyond amino + carboxylic acid.

    PEP-02: When additional FGs are detected (hydroxy, thiol, halogen, nitro,
    ketone), _name_amino_acid_systematic should bail out and let the general
    polyfunctional pipeline handle the molecule, since it correctly handles
    multi-FG chains with proper prefix ordering, locant assignment, and
    alphabetization.

    Args:
        mol: RDKit Mol object

    Returns:
        True if extra functional groups found
    """
    for smarts in _EXTRA_FG_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None and mol.HasSubstructMatch(pat):
            return True
    return False


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
        # STER-09 / D-03: Inject stereo for trivial names (IUPAC P-91)
        from ..perception.stereo import assign_stereochemistry
        from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
        assign_stereochemistry(mol)
        locant_map = _build_amino_acid_locant_map(mol)
        descriptors = collect_stereodescriptors(mol, locant_map)
        if descriptors:
            stereo_prefix = format_stereodescriptor_string(descriptors)
            return f"{stereo_prefix}{trivial}"
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

    PEP-02: Also bails out when additional functional groups are detected
    (hydroxy, thiol, halogen, nitro, ketone) since the general polyfunctional
    pipeline handles multi-FG chains correctly.
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

    # AMAC-01: Multiple COOH groups -> polyfunctional pipeline
    # Dicarboxylic amino acids (aspartic, glutamic) need "dioic acid" suffix
    # which the specialized handler can't produce (it hardcodes mono-acid).
    cooh_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    if cooh_pattern:
        cooh_matches = mol.GetSubstructMatches(cooh_pattern)
        if len(cooh_matches) >= 2:
            return None  # Dicarboxylic - use polyfunctional pipeline

    # If the molecule has rings, the simple carbon-count approach would
    # include ring carbons in the chain length (e.g., tyrosine would give
    # "2-aminononanoic acid" instead of falling through to the general
    # pipeline which correctly uses parent selection with ring-atom exclusion).
    ri = mol.GetRingInfo()
    if ri.NumRings() > 0:
        return None  # Let general pipeline handle ring-containing amino acids

    # PEP-02: Check for additional functional groups beyond amino and acid.
    # If present, bail out to general polyfunctional pipeline which handles
    # multi-FG chains correctly with proper prefix ordering and locants.
    if _has_extra_functional_groups(mol):
        return None

    # Count carbons in the backbone (acid chain) -- safe for acyclic molecules
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    # Get stem from carbon count using centralized module
    stem = get_chain_prefix(carbon_count)

    # For alpha-amino acids, the amino group is at position 2
    # (position 1 is the acid carbon)
    name = f"2-amino{stem}anoic acid"

    # STER-09: Inject CIP stereodescriptors (self-contained, runs before perception)
    from ..perception.stereo import assign_stereochemistry
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    assign_stereochemistry(mol)
    locant_map = _build_amino_acid_locant_map(mol)
    descriptors = collect_stereodescriptors(mol, locant_map)
    if descriptors:
        stereo_prefix = format_stereodescriptor_string(descriptors)
        name = f"{stereo_prefix}{name}"

    return name


def _build_amino_acid_locant_map(mol) -> dict:
    """Build atom_to_locant map for amino acid backbone.

    Locant 1 = acid carbon (C(=O)O), then walk along the carbon backbone.
    For alpha-amino acids, the stereocenter is at locant 2.

    Args:
        mol: RDKit Mol object.

    Returns:
        Dict mapping atom index to IUPAC locant number.
    """
    from collections import deque

    # Find the acid carbon: a carbon with a C=O double bond and C-OH single bond
    acid_c = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C':
            continue
        has_double_o = False
        has_single_o = False
        for bond in atom.GetBonds():
            nbr = bond.GetOtherAtom(atom)
            if nbr.GetSymbol() == 'O' and bond.GetBondTypeAsDouble() == 2.0:
                has_double_o = True
            elif nbr.GetSymbol() == 'O' and bond.GetBondTypeAsDouble() == 1.0:
                has_single_o = True
        if has_double_o and has_single_o:
            acid_c = atom.GetIdx()
            break

    if acid_c is None:
        return {}

    # BFS from acid carbon through the carbon backbone
    atom_to_locant = {acid_c: 1}
    visited = {acid_c}
    queue = deque([(acid_c, 1)])

    while queue:
        current, locant = queue.popleft()
        atom = mol.GetAtomWithIdx(current)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in visited:
                continue
            if nbr.GetSymbol() == 'C':
                next_locant = locant + 1
                atom_to_locant[nbr_idx] = next_locant
                visited.add(nbr_idx)
                queue.append((nbr_idx, next_locant))
            else:
                visited.add(nbr_idx)

    return atom_to_locant


def is_n_substituted_amino_acid(mol) -> bool:
    """
    Check if amino acid has N-substituents (like sarcosine = N-methylglycine).
    """
    # Look for secondary amine in amino acid context
    pattern = Chem.MolFromSmarts("[NX3;H1]([CX4])[CX4][CX3](=O)[OX2H1]")
    if pattern is None:
        return False
    return mol.HasSubstructMatch(pattern)
