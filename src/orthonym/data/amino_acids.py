"""
Amino acid data for trivial name lookups.

Standard amino acids (the 20 proteinogenic amino acids) have trivial names
that are preferred as IUPAC PINs over systematic names.

Stereochemistry note: Most natural amino acids are L-configured (S in CIP),
but IUPAC prefers R/S notation over D/L for PINs.
"""

from typing import Optional, Dict

# Standard amino acids: canonical SMILES -> trivial name
# Note: These are for the L-enantiomers (naturally occurring form)
# SMILES without explicit stereo will match
STANDARD_AMINO_ACIDS: Dict[str, str] = {
    # Aliphatic
    "NCC(=O)O": "glycine",                    # Gly - achiral
    "CC(N)C(=O)O": "alanine",                 # Ala
    "CC(C)C(N)C(=O)O": "valine",              # Val
    "CC(C)CC(N)C(=O)O": "leucine",            # Leu
    "CCC(C)C(N)C(=O)O": "isoleucine",         # Ile

    # Hydroxyl-containing
    "CC(O)C(N)C(=O)O": "threonine",           # Thr
    "NC(CO)C(=O)O": "serine",                 # Ser
    "NC(C(=O)O)CO": "serine",                 # Ser - alternate canonical

    # Sulfur-containing
    "CSCC(N)C(=O)O": "methionine",            # Met
    "NC(CS)C(=O)O": "cysteine",               # Cys

    # Acidic and amides
    "NC(CC(=O)O)C(=O)O": "aspartic acid",     # Asp
    "NC(CCC(=O)O)C(=O)O": "glutamic acid",    # Glu
    "NC(CC(N)=O)C(=O)O": "asparagine",        # Asn
    "NC(=O)CC(N)C(=O)O": "asparagine",        # Asn - alternate
    "NC(CCC(N)=O)C(=O)O": "glutamine",        # Gln
    "NC(=O)CCC(N)C(=O)O": "glutamine",        # Gln - alternate

    # Basic
    "NCCCCC(N)C(=O)O": "lysine",              # Lys
    "NC(CCCNC(N)=N)C(=O)O": "arginine",       # Arg (input form)
    "NC(=N)NCCCC(N)C(=O)O": "arginine",       # Arg - alternate
    "N=C(N)NCCCC(N)C(=O)O": "arginine",       # Arg - canonical form
    "NC(Cc1cnc[nH]1)C(=O)O": "histidine",     # His
    "NC(Cc1c[nH]cn1)C(=O)O": "histidine",     # His - alternate
    "NC(Cc1[nH]cnc1)C(=O)O": "histidine",     # His - alternate 2

    # Aromatic
    "NC(Cc1ccccc1)C(=O)O": "phenylalanine",   # Phe
    "NC(Cc1ccc(O)cc1)C(=O)O": "tyrosine",     # Tyr
    "NC(Cc1c[nH]c2ccccc12)C(=O)O": "tryptophan",  # Trp

    # Imino acid (proline)
    "OC(=O)C1CCCN1": "proline",               # Pro
    "O=C(O)C1CCCN1": "proline",               # Pro - alternate canonical
}

# Non-standard amino acids with trivial names
NON_STANDARD_AMINO_ACIDS: Dict[str, str] = {
    "NCCCC(N)C(=O)O": "ornithine",            # Orn - not proteinogenic
    "NC(CCCN)C(=O)O": "ornithine",            # Orn - alternate
    "NCCC(N)C(=O)O": "2,4-diaminobutanoic acid",  # Dab
    "NCCCC(=O)O": "4-aminobutanoic acid",     # GABA - gamma-aminobutyric acid
    "CNCC(=O)O": "sarcosine",                 # N-methylglycine
}


def get_amino_acid_name(canonical_smiles: str) -> Optional[str]:
    """
    Get the trivial name for an amino acid if it's a standard one.

    Args:
        canonical_smiles: Canonical SMILES of the amino acid

    Returns:
        Trivial name if found, None otherwise
    """
    # Check standard amino acids first
    if canonical_smiles in STANDARD_AMINO_ACIDS:
        return STANDARD_AMINO_ACIDS[canonical_smiles]

    # Check non-standard
    if canonical_smiles in NON_STANDARD_AMINO_ACIDS:
        return NON_STANDARD_AMINO_ACIDS[canonical_smiles]

    return None


def is_standard_amino_acid(canonical_smiles: str) -> bool:
    """Check if SMILES matches a standard amino acid."""
    return canonical_smiles in STANDARD_AMINO_ACIDS
