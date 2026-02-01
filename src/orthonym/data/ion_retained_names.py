"""
Retained names for common ions used in IUPAC nomenclature.

This module provides lookup tables for common organic and inorganic ions
that have retained (trivial) names preferred over systematic names.

For salts, the naming convention is: cation name + anion name
Example: sodium acetate, potassium chloride

Keys are canonical SMILES (ensure consistency with RDKit canonicalization).
"""

from typing import Optional
from rdkit import Chem


# === ORGANIC ANIONS (carboxylates, alkoxides, etc.) ===
# Named with -ate, -ide suffixes
RETAINED_ANIONS = {
    # Carboxylate anions
    'CC([O-])=O': 'acetate',
    '[O-]C=O': 'formate',
    'CCC([O-])=O': 'propanoate',
    'CCCC([O-])=O': 'butanoate',
    'O=C([O-])c1ccccc1': 'benzoate',
    'O=C([O-])C([O-])=O': 'oxalate',
    'O=C([O-])CC([O-])=O': 'malonate',
    'O=C([O-])CCC([O-])=O': 'succinate',

    # Alkoxide anions
    '[CH3][O-]': 'methoxide',
    'CC[O-]': 'ethoxide',
    'CC(C)[O-]': 'isopropoxide',
    'CC(C)(C)[O-]': 'tert-butoxide',
    '[O-]c1ccccc1': 'phenoxide',  # Also called phenolate

    # Carbanions
    '[CH3-]': 'methanide',
    '[c-]1ccccc1': 'phenide',  # Also called benzenide

    # Other organic anions
    'CC#[N-]': 'acetylide',  # Terminal alkynide
}

# === ORGANIC CATIONS (ammonium, carbocations) ===
# Named with -ium, -ylium suffixes
RETAINED_CATIONS = {
    # Ammonium cations
    '[NH4+]': 'ammonium',
    'C[NH3+]': 'methylammonium',
    'CC[NH3+]': 'ethylammonium',
    'CCC[NH3+]': 'propylammonium',
    'C[NH2+]C': 'dimethylammonium',
    'CC[NH2+]CC': 'diethylammonium',
    'C[NH+](C)C': 'trimethylammonium',
    'C[N+](C)(C)C': 'tetramethylammonium',
    'CC[N+](CC)(CC)CC': 'tetraethylammonium',

    # Carbocations (carbonium/carbenium ions)
    '[CH3+]': 'methylium',
    'C[CH2+]': 'ethylium',
    'CC[CH2+]': 'propylium',
    'CC(C)[CH2+]': 'isobutylium',
    'CC([CH3+])C': 'isopropylium',  # Secondary carbocation
    'CC([CH2+])(C)C': 'tert-butylium',  # Tertiary carbocation

    # Aromatic cations
    '[cH+]1ccccc1': 'phenylium',

    # Oxonium cations
    '[OH3+]': 'oxonium',
    'C[OH2+]': 'methyloxonium',
    'C[OH+]C': 'dimethyloxonium',

    # Sulfonium cations
    'C[SH2+]': 'methylsulfonium',
    'C[SH+]C': 'dimethylsulfonium',
    'C[S+](C)C': 'trimethylsulfonium',

    # Phosphonium cations
    '[PH4+]': 'phosphonium',
    'C[PH3+]': 'methylphosphonium',
    'C[P+](C)(C)C': 'tetramethylphosphonium',
}

# === INORGANIC CATIONS (metal ions) ===
# Used for naming salts: sodium chloride, calcium acetate
INORGANIC_CATIONS = {
    # Alkali metals (Group 1)
    '[Li+]': 'lithium',
    '[Na+]': 'sodium',
    '[K+]': 'potassium',
    '[Rb+]': 'rubidium',
    '[Cs+]': 'cesium',

    # Alkaline earth metals (Group 2)
    '[Mg+2]': 'magnesium',
    '[Ca+2]': 'calcium',
    '[Sr+2]': 'strontium',
    '[Ba+2]': 'barium',

    # Transition metals (common oxidation states)
    '[Fe+2]': 'iron(II)',
    '[Fe+3]': 'iron(III)',
    '[Cu+]': 'copper(I)',
    '[Cu+2]': 'copper(II)',
    '[Zn+2]': 'zinc',
    '[Ag+]': 'silver',
    '[Au+]': 'gold(I)',
    '[Au+3]': 'gold(III)',
    '[Mn+2]': 'manganese(II)',
    '[Co+2]': 'cobalt(II)',
    '[Ni+2]': 'nickel(II)',
    '[Cr+3]': 'chromium(III)',

    # Other metals
    '[Al+3]': 'aluminium',
    '[Pb+2]': 'lead(II)',
    '[Sn+2]': 'tin(II)',
    '[Sn+4]': 'tin(IV)',
}

# === INORGANIC ANIONS (halides, hydroxide, etc.) ===
# Used for naming salts
INORGANIC_ANIONS = {
    # Halides
    '[F-]': 'fluoride',
    '[Cl-]': 'chloride',
    '[Br-]': 'bromide',
    '[I-]': 'iodide',

    # Hydroxide and oxide
    '[OH-]': 'hydroxide',
    '[O-2]': 'oxide',

    # Chalcogenides
    '[S-2]': 'sulfide',
    '[HS-]': 'hydrosulfide',
    '[Se-2]': 'selenide',

    # Nitrogen anions
    '[N-3]': 'nitride',
    '[NH2-]': 'amide',  # Metal amides (not organic amides)
    '[N3-]': 'azide',

    # Carbon anions
    '[C-4]': 'carbide',
    '[CN-]': 'cyanide',

    # Oxygen-containing anions
    '[NO3-]': 'nitrate',
    '[NO2-]': 'nitrite',
    'O=S([O-])=O': 'sulfate',  # Note: simplified SMILES
    '[O-]S([O-])=O': 'sulfite',
    'O=P([O-])([O-])[O-]': 'phosphate',
    '[O-]C([O-])=O': 'carbonate',
    '[O-]Cl=O': 'chlorate',
}


def get_anion_name(smiles: str) -> Optional[str]:
    """
    Look up retained name for an anion by its SMILES.

    The input SMILES is canonicalized before lookup to ensure
    consistent matching regardless of input format.

    Args:
        smiles: SMILES string (will be canonicalized)

    Returns:
        Retained name if found, None otherwise

    Example:
        >>> get_anion_name('CC(=O)[O-]')
        'acetate'
        >>> get_anion_name('[Cl-]')
        'chloride'
    """
    # Canonicalize input SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canonical = Chem.MolToSmiles(mol)

    # Check organic anions first
    if canonical in RETAINED_ANIONS:
        return RETAINED_ANIONS[canonical]

    # Check inorganic anions
    if canonical in INORGANIC_ANIONS:
        return INORGANIC_ANIONS[canonical]

    return None


def get_cation_name(smiles: str) -> Optional[str]:
    """
    Look up retained name for a cation by its SMILES.

    The input SMILES is canonicalized before lookup to ensure
    consistent matching regardless of input format.

    Args:
        smiles: SMILES string (will be canonicalized)

    Returns:
        Retained name if found, None otherwise

    Example:
        >>> get_cation_name('[NH4+]')
        'ammonium'
        >>> get_cation_name('[Na+]')
        'sodium'
    """
    # Canonicalize input SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canonical = Chem.MolToSmiles(mol)

    # Check organic cations first
    if canonical in RETAINED_CATIONS:
        return RETAINED_CATIONS[canonical]

    # Check inorganic cations
    if canonical in INORGANIC_CATIONS:
        return INORGANIC_CATIONS[canonical]

    return None


def get_ion_name(smiles: str) -> Optional[str]:
    """
    Look up retained name for any ion (cation or anion).

    Convenience function that checks both cation and anion lookups.

    Args:
        smiles: SMILES string (will be canonicalized)

    Returns:
        Retained name if found, None otherwise

    Example:
        >>> get_ion_name('[Na+]')
        'sodium'
        >>> get_ion_name('[Cl-]')
        'chloride'
    """
    # Try cation lookup
    name = get_cation_name(smiles)
    if name:
        return name

    # Try anion lookup
    return get_anion_name(smiles)
