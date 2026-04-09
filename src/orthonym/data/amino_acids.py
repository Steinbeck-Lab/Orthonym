"""
Amino acid data for trivial name lookups.

Standard amino acids (the 20 proteinogenic amino acids) have trivial names
that are preferred as IUPAC PINs over systematic names.

Expanded in Phase 141-02 with 88 OPSIN simpleGroup amino acid entries.
These are complete molecules with canonical SMILES directly from OPSIN's
aminoAcids.xml vocabulary.

Stereochemistry note: Most natural amino acids are L-configured (S in CIP),
but IUPAC prefers R/S notation over D/L for PINs.
"""

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

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


def _integrate_opsin_simplegroup() -> Dict[str, str]:
    """Integrate OPSIN simpleGroup amino acid entries into NON_STANDARD_AMINO_ACIDS.

    These are complete molecules (full SMILES with all atoms) from OPSIN's
    aminoAcids.xml vocabulary. Their canonical SMILES can be directly used
    as lookup keys.

    Returns:
        Dict mapping canonical SMILES to trivial names for entries not already
        in STANDARD_AMINO_ACIDS or NON_STANDARD_AMINO_ACIDS.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return {}

    try:
        from .opsin_imports.amino_acids_opsin import OPSIN_AMINO_ACIDS
    except ImportError:
        return {}

    integrated = {}
    for _key, entry in OPSIN_AMINO_ACIDS.items():
        if entry.get('subType') != 'simpleGroup':
            continue

        smiles = entry.get('smiles', '')
        names = entry.get('names', [])
        if not smiles or not names:
            continue

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            can_smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            continue

        # Hand-curated entries take precedence
        if can_smiles in STANDARD_AMINO_ACIDS:
            continue
        if can_smiles in NON_STANDARD_AMINO_ACIDS:
            continue
        if can_smiles in integrated:
            continue

        # Use the first name from OPSIN's name list
        integrated[can_smiles] = names[0]

    return integrated


def _build_acyl_names(all_amino_acids: Dict[str, str],
                      existing_acyl: Dict[str, str]) -> Dict[str, str]:
    """Build acyl (peptide linkage) names for all known amino acids.

    Rules per IUPAC 3AA-13:
    - Names ending in 'ine': replace with 'yl' (e.g., glycine -> glycyl)
    - Names ending in 'ic acid': replace with 'yl' (e.g., aspartic acid -> aspartyl)
    - Names ending in 'an': add 'yl' (e.g., tryptophan -> tryptophanyl)
    - Names ending in 'ate': replace with 'yl' (e.g., dimethyltaurate -> dimethyltauryl)
    - Default: add 'yl'

    Hand-curated entries in existing_acyl take precedence.

    Args:
        all_amino_acids: Dict of canonical SMILES -> trivial name for all amino acids.
        existing_acyl: Dict of existing hand-curated acyl names.

    Returns:
        Dict of generated trivial name -> acyl name (excludes hand-curated entries).
    """
    generated = {}
    for _smiles, name in all_amino_acids.items():
        if name in existing_acyl:
            # Hand-curated entry takes precedence
            continue
        if name in generated:
            continue

        # Generate acyl name based on suffix rules
        if name.endswith('ine'):
            acyl = name[:-3] + 'yl'
        elif name.endswith('ic acid'):
            acyl = name[:-7] + 'yl'
        elif name.endswith('an'):
            acyl = name + 'yl'
        elif name.endswith('ate'):
            acyl = name[:-3] + 'yl'
        elif name.endswith('ol'):
            acyl = name[:-2] + 'yl'
        elif name.endswith('amine'):
            acyl = name[:-5] + 'aminyl'
        else:
            acyl = name + 'yl'

        generated[name] = acyl

    return generated


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


# Amino acid acyl (peptide linkage) names for all 20 standard amino acids.
# Used when assembling peptide names: N-terminal residues use the acyl form.
# IUPAC 3AA-13: the acyl form replaces the terminal -ine/-ic acid/-an with -yl.
AMINO_ACID_ACYL_NAMES: Dict[str, str] = {
    # Aliphatic
    "glycine": "glycyl",
    "alanine": "alanyl",
    "valine": "valyl",
    "leucine": "leucyl",
    "isoleucine": "isoleucyl",
    # Hydroxyl-containing
    "serine": "seryl",
    "threonine": "threonyl",
    # Sulfur-containing
    "cysteine": "cysteinyl",
    "methionine": "methionyl",
    # Acidic and amides
    "aspartic acid": "aspartyl",
    "glutamic acid": "glutamyl",
    "asparagine": "asparaginyl",
    "glutamine": "glutaminyl",
    # Basic
    "lysine": "lysyl",
    "arginine": "arginyl",
    "histidine": "histidyl",
    # Aromatic
    "phenylalanine": "phenylalanyl",
    "tyrosine": "tyrosyl",
    "tryptophan": "tryptophyl",
    # Imino acid
    "proline": "prolyl",
}


def get_amino_acid_acyl_name(trivial_name: str) -> Optional[str]:
    """
    Get the acyl (peptide linkage) form of an amino acid name.

    Args:
        trivial_name: Trivial name of the amino acid (e.g., "glycine")

    Returns:
        Acyl name (e.g., "glycyl") if found, None otherwise
    """
    return AMINO_ACID_ACYL_NAMES.get(trivial_name)


# --- Module-level integration (must run after all dicts are defined) ---
# Integrate OPSIN simpleGroup entries at import time
_opsin_simple = _integrate_opsin_simplegroup()
NON_STANDARD_AMINO_ACIDS.update(_opsin_simple)
logger.debug(
    "Integrated %d OPSIN simpleGroup amino acids (total non-standard: %d)",
    len(_opsin_simple), len(NON_STANDARD_AMINO_ACIDS),
)

# Build expanded acyl names from all amino acid entries
_all_aa = {}
_all_aa.update(STANDARD_AMINO_ACIDS)
_all_aa.update(NON_STANDARD_AMINO_ACIDS)
_generated_acyl = _build_acyl_names(_all_aa, AMINO_ACID_ACYL_NAMES)
AMINO_ACID_ACYL_NAMES.update(_generated_acyl)
logger.debug(
    "Expanded acyl names to %d entries (+%d generated)",
    len(AMINO_ACID_ACYL_NAMES), len(_generated_acyl),
)
