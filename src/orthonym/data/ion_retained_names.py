"""
Retained names for common ions used in IUPAC nomenclature.

This module provides lookup tables for common organic and inorganic ions
that have retained (trivial) names preferred over systematic names.

For salts, the naming convention is: cation name + anion name
Example: sodium acetate, potassium chloride

Keys are canonical SMILES (verified with RDKit canonicalization).
"""

from typing import Optional
from rdkit import Chem


# === ORGANIC ANIONS (carboxylates, alkoxides, etc.) ===
# Named with -ate, -ide suffixes
# All keys are in RDKit canonical SMILES form
RETAINED_ANIONS = {
    # Carboxylate anions
    'CC(=O)[O-]': 'acetate',
    'O=C[O-]': 'formate',
    'CCC(=O)[O-]': 'propanoate',
    'CCCC(=O)[O-]': 'butanoate',
    'O=C([O-])c1ccccc1': 'benzoate',
    'O=C([O-])C(=O)[O-]': 'oxalate',
    'O=C([O-])CC(=O)[O-]': 'malonate',
    'O=C([O-])CCC(=O)[O-]': 'succinate',

    # Alkoxide anions
    'C[O-]': 'methoxide',
    'CC[O-]': 'ethoxide',
    'CC(C)[O-]': 'isopropoxide',
    'CC(C)(C)[O-]': 'tert-butoxide',
    '[O-]c1ccccc1': 'phenoxide',  # Also called phenolate

    # Carbanions
    # D-09 (Plan 184-01): the '[CH3-]' -> 'methanide' entry was SUBSUMED by the
    # systematic emit_parent_hydride_cumulative_suffix primitive (ions.py). The
    # equivalence was proven byte-identical (the primitive emits 'methanide' for
    # the single-carbon chain, locant omitted) AND route_charged is confirmed to
    # reach the carbanion branch for [CH3-], so the primitive is now the single
    # source of truth (no per-molecule retained band-aid). [c-]1ccccc1 (phenide)
    # stays — it is a ring carbanion outside the acyclic parent-hydride path.
    '[c-]1ccccc1': 'phenide',  # Also called benzenide

    # Azanide (NH2-) — the conjugate base of azane (NH3). F-T6 (DD3, P-72.2.2.2):
    # the preselected name of the bare nitrogen-hydride anion. RETAINED_ANIONS is
    # consulted before INORGANIC_ANIONS, so this shadows the legacy inorganic
    # 'amide' (a deprecated name) — azanide is the IUPAC 2013 PIN and OPSIN
    # round-trips it to [NH2-]. (Bare [NH2-] previously emitted 'amide', which the
    # OPSIN validity gate suppressed to 'unknown organic compound'.)
    '[NH2-]': 'azanide',

    # Alkynide anions
    '[C-]#C': 'ethynide',  # Terminal alkynide

    # BBR-CHG-169.6-caveats (Phase 169.7): retained charged-species names recovered
    # from the 169.6 route_charged regression (neutralize-first produced OPSIN-
    # unparseable forms -> suppressed). All RT-verified; sanctioned by P-72/P-74.
    'N[O-]': 'aminoxide',  # H2N-O- conjugate base of hydroxylamine (P-74); was -> unknown
    'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F': 'bistriflimide',  # was -> 'triflimidic acid'
}

# === ORGANIC CATIONS (ammonium, carbocations) ===
# Named with -ium, -ylium suffixes
# All keys are in RDKit canonical SMILES form
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

    # Guanidinium cation (P-73, Table 7.3)
    'NC(N)=[NH2+]': 'guanidinium',

    # Carbocations (carbonium/carbenium ions)
    '[CH3+]': 'methylium',
    '[CH2+]C': 'ethylium',
    '[CH2+]CC': 'propylium',
    '[CH2+]C(C)C': 'isobutylium',
    'C[C+](C)C': 'isopropylium',  # Secondary carbocation (tert-butyl cation)
    '[CH2+]C(C)(C)C': 'neopentylium',  # Primary carbocation adjacent to tert-butyl

    # Aromatic cations
    '[C+]1=CC=CC=C1': 'phenylium',

    # Oxonium cations
    '[OH3+]': 'oxonium',
    'C[OH2+]': 'methyloxonium',
    'C[OH+]C': 'dimethyloxonium',

    # Sulfonium cations
    '[SH3+]': 'sulfonium',  # BBR-CHG-169.6-caveat: parent sulfonium (P-73.1.1.1 Table 7.3); RT-verified
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
# All keys are in RDKit canonical SMILES form
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
    '[S-]': 'hydrosulfide',  # HS- as [S-] (H implicit)
    '[Se-2]': 'selenide',

    # Nitrogen anions
    '[N-3]': 'nitride',
    '[NH2-]': 'amide',  # Metal amides (not organic amides)
    '[N-]=[N+]=[N-]': 'azide',

    # Carbon anions
    '[C-4]': 'carbide',
    '[C-]#N': 'cyanide',

    # Oxygen-containing anions (canonical forms)
    'O=[N+]([O-])[O-]': 'nitrate',
    'O=[N+][O-]': 'nitrite',
    'O=[SH](=O)[O-]': 'sulfate',  # Note: RDKit canonical form
    # SO4(2-) bare dianion of sulfuric acid (P-12.2 / BlueBookV2.md:35449); the
    # legacy 'O=[SH](=O)[O-]' key above was a -1 [SH] form that never matched the
    # real fully-deprotonated dianion '[O-]S(=O)(=O)[O-]' (canonical O=S(=O)([O-])[O-]).
    'O=S(=O)([O-])[O-]': 'sulfate',
    # HSO4- mono-anion (P-12.2 / BlueBookV2.md:7150); MUST stay distinct from
    # 'sulfate' (different protonation state). OPSIN round-trips 'hydrogensulfate'
    # -> S(=O)(=O)(O)[O-] (the correct mono-anion).
    'O=S(=O)([O-])O': 'hydrogensulfate',
    'O=S([O-])[O-]': 'sulfite',
    'O=P([O-])([O-])[O-]': 'phosphate',
    'O=P([O-])([O-])O': 'hydrogen phosphate',
    'O=P([O-])(O)O': 'dihydrogen phosphate',
    'O=C([O-])[O-]': 'carbonate',
    'O=C([O-])O': 'hydrogen carbonate',
    '[O-][Cl+][O-]': 'chlorate',  # RDKit canonical form
}


def _canonicalize_anion_table(table):
    # Re-key an anion lookup table by RDKit-canonical SMILES so a hand-written
    # non-canonical key can never silently fail to match a canonicalized input
    # (the WS-E.3 sulfate-key bug: 'O=[SH](=O)[O-]' never matched the bare -2
    # dianion 'O=S(=O)([O-])[O-]'). Keys that fail to parse are kept verbatim.
    out = {}
    for smi, word in table.items():
        m = Chem.MolFromSmiles(smi)
        out[Chem.MolToSmiles(m) if m is not None else smi] = word
    return out


INORGANIC_ANIONS = _canonicalize_anion_table(INORGANIC_ANIONS)
RETAINED_ANIONS = _canonicalize_anion_table(RETAINED_ANIONS)


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
