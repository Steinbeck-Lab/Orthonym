"""
Salt and zwitterion naming rules per IUPAC 2013.

Handles naming of:
- Salts: Compositional nomenclature (cation + anion as separate words)
- Zwitterions: Internal ion pairs with combined suffixes

IUPAC 2013 References:
- P-72: Anion nomenclature
- P-73: Cation nomenclature
- P-74: Zwitterion nomenclature

Key naming patterns:
- Salts: "cation anion" format (sodium acetate, ammonium chloride)
- Zwitterions: base name with ionic suffixes (2-azaniumylacetate)
- Multiple ions: alphabetized cations before alphabetized anions
- Stoichiometry: multiplicative prefixes for repeated ions (diacetate)
"""

from typing import Dict, List, Optional, Any
from collections import Counter
from rdkit import Chem

from ..perception.ions import parse_salt_fragments, get_ion_sites
from .ions import name_anion, name_cation
from ..data.ion_retained_names import INORGANIC_CATIONS, INORGANIC_ANIONS


# === STOICHIOMETRIC PREFIXES ===

STOICHIOMETRIC_PREFIXES = {
    2: 'di',
    3: 'tri',
    4: 'tetra',
    5: 'penta',
    6: 'hexa',
    7: 'hepta',
    8: 'octa',
    9: 'nona',
    10: 'deca',
}


# === AMINO ACID ZWITTERION PATTERNS ===

# SMARTS for alpha-amino acid zwitterion pattern
ALPHA_AA_ZWITTERION = '[NX4+;H3][CX4][CX3](=[OX1])[OX1-]'


# === RETAINED AMINO ACID NAMES ===

# Map canonical SMILES of zwitterion form to trivial name
RETAINED_AMINO_ACID_ZWITTERIONS = {
    # Glycine zwitterion
    '[NH3+]CC([O-])=O': 'glycine',
    # Alanine zwitterion
    'C[C@H]([NH3+])C([O-])=O': 'L-alanine',
    'C[C@@H]([NH3+])C([O-])=O': 'D-alanine',
    'CC([NH3+])C([O-])=O': 'alanine',
    # Add more as needed
}


# === SALT NAMING ===

# Inorganic acid anion to hydroacid salt name mapping
_HYDROACID_SALT_NAMES = {
    '[Cl-]': 'hydrochloride',
    '[Br-]': 'hydrobromide',
    '[I-]': 'hydroiodide',
    '[F-]': 'hydrofluoride',
}

# Hydrogen prefix multipliers for partial salts
_HYDROGEN_PREFIXES = {
    1: 'hydrogen',
    2: 'dihydrogen',
    3: 'trihydrogen',
}


def _count_protonated_acid_sites(frag_mol) -> int:
    """Count the number of still-protonated carboxylic acid sites (-COOH).

    Only counts protonated acid groups, NOT deprotonated carboxylates.
    Used to detect partial deprotonation for "hydrogen" prefix in salt names.
    E.g., sodium hydrogen fumarate has 1 COOH + 1 COO-.
    """
    from rdkit.Chem import MolFromSmarts
    acid_pat = MolFromSmarts('[CX3](=O)[OX2H1]')
    if acid_pat:
        return len(frag_mol.GetSubstructMatches(acid_pat))
    return 0


def name_salt(mol, style: str = 'pin') -> str:
    """
    Name a salt using compositional nomenclature.

    Format: cation_name + space + anion_name
    Example: "sodium acetate", "ammonium chloride"

    Also handles:
    - Neutral organic fragments with inorganic counter-ions (Drug.HCl pattern)
    - H+ fragments merged with anions for hydroacid salt naming
    - Partial salts with "hydrogen" prefix (sodium hydrogen fumarate)

    For multiple cations/anions, order alphabetically.
    For stoichiometry > 1, use multiplier prefixes.

    Args:
        mol: RDKit Mol object (contains disconnected fragments)
        style: 'pin' for preferred names

    Returns:
        Salt name as "cation anion" (separate words)

    Example:
        >>> mol = Chem.MolFromSmiles('[Na+].[O-]C(C)=O')
        >>> name_salt(mol)
        'sodium acetate'
    """
    if mol is None:
        return ''

    frags = parse_salt_fragments(mol)

    # --- Handle H+ fragments: merge with Cl-/Br- for hydroacid salt naming ---
    # H+ is a bare proton fragment (canonical SMILES: '[H+]')
    h_plus_frags = [f for f in frags['cations'] if f['smiles'] == '[H+]']
    other_cation_frags = [f for f in frags['cations'] if f['smiles'] != '[H+]']
    neutrals = frags.get('neutrals', [])

    # Pattern: Organic_neutral.[H+].[Cl-] -> "organic_name hydrochloride"
    # The H+ merges with Cl- to form HCl, and the neutral organic fragment
    # is the main compound being named as a hydrochloride salt.
    if h_plus_frags and not other_cation_frags and neutrals:
        # Check if all anions are simple halide-type
        hydroacid_names = []
        for anion_frag in frags['anions']:
            hydroacid_name = _HYDROACID_SALT_NAMES.get(anion_frag['smiles'])
            if hydroacid_name:
                hydroacid_names.append(hydroacid_name)

        if hydroacid_names and len(hydroacid_names) == len(frags['anions']):
            # All anions are halides -- name as "organic hydrochloride"
            # Pick the largest neutral organic fragment as the main compound
            organic_neutrals = [
                f for f in neutrals if f['mol'].GetNumHeavyAtoms() > 1
            ]
            if organic_neutrals:
                main_frag = max(organic_neutrals,
                                key=lambda f: f['mol'].GetNumHeavyAtoms())
                try:
                    from ..namer import Orthonym
                    namer = Orthonym(style=style)
                    organic_name = namer.name(main_frag['smiles'])
                    if organic_name:
                        salt_suffix = ' '.join(sorted(hydroacid_names))
                        return f"{organic_name} {salt_suffix}"
                except (RecursionError, ValueError, RuntimeError):
                    pass

    cation_names = []
    anion_names = []

    # Process cations (excluding H+ fragments already handled above)
    cation_list = other_cation_frags if h_plus_frags else frags['cations']
    for cation_frag in cation_list:
        frag_mol = cation_frag['mol']
        smiles = cation_frag['smiles']

        # Check inorganic cations first
        if smiles in INORGANIC_CATIONS:
            cation_names.append(INORGANIC_CATIONS[smiles])
        else:
            # Try organic cation naming
            name = name_cation(frag_mol, style)
            if name:
                cation_names.append(name)
            # Skip unnamed cations rather than using generic 'cation'

    # Process anions
    for anion_frag in frags['anions']:
        frag_mol = anion_frag['mol']
        smiles = anion_frag['smiles']

        # Check inorganic anions first
        if smiles in INORGANIC_ANIONS:
            anion_names.append(INORGANIC_ANIONS[smiles])
        else:
            # Try organic anion naming
            name = name_anion(frag_mol, style)
            if name:
                anion_names.append(name)
            # Skip unnamed anions rather than using generic 'anion'

    # --- Hydrogen prefix for partial salts (IUPAC P-72.2.1) ---
    # When an anion fragment still has protonated carboxylic acid groups
    # (-COOH), it is only partially deprotonated. Insert "hydrogen"
    # between cation and anion names.
    # E.g., "sodium hydrogen fumarate" = one Na+ + one COOH + one COO-.
    hydrogen_prefix = ''
    if len(frags['anions']) == 1 and cation_names:
        anion_frag = frags['anions'][0]
        protonated_acids = _count_protonated_acid_sites(anion_frag['mol'])
        if protonated_acids > 0:
            hydrogen_prefix = _HYDROGEN_PREFIXES.get(
                protonated_acids, 'hydrogen'
            )

    # Handle stoichiometry - count duplicates
    cation_counts = Counter(cation_names)
    anion_counts = Counter(anion_names)

    # Format cation part with multipliers
    formatted_cations = []
    for name in sorted(cation_counts.keys()):
        count = cation_counts[name]
        formatted_cations.append(_apply_stoichiometric_prefix(name, count))

    # Format anion part with multipliers
    formatted_anions = []
    for name in sorted(anion_counts.keys()):
        count = anion_counts[name]
        formatted_anions.append(_apply_stoichiometric_prefix(name, count))

    # Combine: cations first, then hydrogen prefix (if any), then anions
    if hydrogen_prefix and formatted_anions:
        result_parts = formatted_cations + [hydrogen_prefix] + formatted_anions
    else:
        result_parts = formatted_cations + formatted_anions

    return ' '.join(result_parts)


def _apply_stoichiometric_prefix(name: str, count: int) -> str:
    """
    Apply di-, tri-, tetra- prefix for stoichiometry.

    Args:
        name: Base ion name
        count: Number of occurrences

    Returns:
        Name with stoichiometric prefix if count > 1

    Example:
        >>> _apply_stoichiometric_prefix('acetate', 2)
        'diacetate'
        >>> _apply_stoichiometric_prefix('sodium', 1)
        'sodium'
    """
    if count == 1:
        return name

    prefix = STOICHIOMETRIC_PREFIXES.get(count, str(count))
    return f"{prefix}{name}"


# === ZWITTERION NAMING ===

def name_zwitterion(mol, style: str = 'pin') -> str:
    """
    Name a zwitterionic compound.

    IUPAC P-74 rules:
    - Anionic centers get lower locants (higher seniority)
    - Cationic suffixes cited BEFORE anionic suffixes
    - Format: base-name-cation_suffix-anion_suffix

    Common zwitterions:
    - Amino acid zwitterions: glycine = 2-ammonioacetate (PIN) or glycine (trivial)

    Args:
        mol: RDKit Mol object with internal positive and negative charges
        style: 'pin' for preferred names

    Returns:
        Zwitterion IUPAC name

    Example:
        >>> mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        >>> name_zwitterion(mol)
        '2-azaniumylacetate'
    """
    if mol is None:
        return ''

    # Check for retained amino acid names first (unless systematic requested)
    if style != 'systematic':
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in RETAINED_AMINO_ACID_ZWITTERIONS:
            return RETAINED_AMINO_ACID_ZWITTERIONS[canonical]

    # Check for amino acid zwitterion pattern
    if _is_amino_acid_zwitterion(mol):
        return _name_amino_acid_zwitterion(mol, style)

    # General zwitterion naming
    return _name_general_zwitterion(mol, style)


def _is_amino_acid_zwitterion(mol) -> bool:
    """
    Check if molecule is amino acid zwitterion [NH3+]-C-[COO-].

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule matches alpha-amino acid zwitterion pattern

    Example:
        >>> mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        >>> _is_amino_acid_zwitterion(mol)
        True
    """
    pattern = Chem.MolFromSmarts(ALPHA_AA_ZWITTERION)
    if pattern is None:
        return False

    return mol.HasSubstructMatch(pattern)


def _name_amino_acid_zwitterion(mol, style: str) -> str:
    """
    Name amino acid zwitterion (e.g., glycine zwitterion).

    For systematic naming:
    - 2-azaniumylacetate (glycine)
    - 2-azaniumylpropanoate (alanine)

    The -azaniumyl prefix denotes -NH3+ group.
    The -ate suffix denotes -COO- group.

    Args:
        mol: RDKit Mol object
        style: 'pin' for systematic, others may use trivial

    Returns:
        Systematic amino acid zwitterion name
    """
    # Count carbons for chain naming
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    # Map chain length to carboxylate base name
    CHAIN_TO_CARBOXYLATE = {
        2: 'acetate',
        3: 'propanoate',
        4: 'butanoate',
        5: 'pentanoate',
        6: 'hexanoate',
    }

    if carbon_count in CHAIN_TO_CARBOXYLATE:
        base = CHAIN_TO_CARBOXYLATE[carbon_count]
    else:
        from ..data.chain_names import get_anoate_name
        base = get_anoate_name(carbon_count)

    # Simple alpha amino acid: 2-azaniumyl-base
    return f'2-azaniumyl{base}'


# === ZWITTERION NEUTRALIZATION HELPERS ===


def _neutralize_zwitterion(mol):
    """
    Neutralize a zwitterion by removing internal charges.

    Handles:
    - Protonated amines ([NH3+] -> NH2): reduce explicit H by charge
    - Quaternary ammonium ([N+](C)(C)(C)C): skip (can't neutralize without
      breaking a bond - not chemically meaningful as neutral)
    - Deprotonated acids ([COO-] -> COOH): increase explicit H by abs(charge)

    Args:
        mol: RDKit Mol object with internal charges

    Returns:
        Neutralized RDKit Mol object, or None on failure
    """
    try:
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            charge = atom.GetFormalCharge()
            if charge > 0:
                cur_h = atom.GetNumExplicitHs()
                total_h = atom.GetTotalNumHs()
                if total_h >= charge:
                    # Protonated: remove H to compensate
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(max(0, cur_h - charge))
                else:
                    # Quaternary (no H to remove): just drop charge
                    # This may create an invalid valence; will be caught by sanitize
                    atom.SetFormalCharge(0)
                    atom.SetNoImplicit(True)
            elif charge < 0:
                atom.SetFormalCharge(0)
                cur_h = atom.GetNumExplicitHs()
                atom.SetNumExplicitHs(cur_h + abs(charge))

        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


def _name_as_neutral(mol, style: str) -> str:
    """
    Try to name a zwitterion by neutralizing it first.

    Strips all internal charges, names the neutral form using the
    standard naming pipeline. This is an acceptable approximation
    per IUPAC for complex zwitterions.

    Args:
        mol: RDKit Mol with zwitterionic charges
        style: Naming style

    Returns:
        Name of the neutral form, or empty string on failure
    """
    neutral = _neutralize_zwitterion(mol)
    if neutral is None:
        return ''

    try:
        neutral_smiles = Chem.MolToSmiles(neutral, canonical=True)
        if not neutral_smiles:
            return ''

        from ..namer import Orthonym
        namer = Orthonym(style=style)
        neutral_name = namer.name(neutral_smiles)
        # Guard: never return 'zwitterion' from the neutral naming path
        if neutral_name and neutral_name != 'zwitterion':
            return neutral_name
    except (RecursionError, ValueError, RuntimeError):
        pass

    return ''


# === GENERAL ZWITTERION NAMING ===


def _name_general_zwitterion(mol, style: str) -> str:
    """
    Name a general zwitterion (not amino acid pattern).

    For zwitterions with various functional groups, combines
    the cationic and anionic descriptors. Falls back to naming
    the neutralized form if specific pattern matching fails.

    Args:
        mol: RDKit Mol object
        style: Naming style

    Returns:
        Zwitterion name, or empty string if naming fails.
        Never returns the literal 'zwitterion'.
    """
    sites = get_ion_sites(mol)

    cation_sites = sites.get('cations', [])
    anion_sites = sites.get('anions', [])

    if not cation_sites or not anion_sites:
        # No ionic sites found - cannot name as zwitterion
        return ''

    # Determine the type of cation and anion
    cation_element = cation_sites[0]['element'] if cation_sites else ''
    anion_element = anion_sites[0]['element'] if anion_sites else ''

    # Build name based on ionic sites
    if cation_element == 'N' and anion_element == 'O':
        # Likely amino acid-like or betaine-like
        result = _infer_zwitterion_name(mol, cation_sites, anion_sites)
        if result:
            return result

    # Fallback: neutralize and name the skeleton
    neutral_name = _name_as_neutral(mol, style)
    if neutral_name:
        return neutral_name

    # Honest failure instead of placeholder literal
    return ''


def _infer_zwitterion_name(
    mol,
    cation_sites: List[Dict[str, Any]],
    anion_sites: List[Dict[str, Any]]
) -> str:
    """
    Infer zwitterion name from ion site positions.

    Analyzes the molecular structure to determine appropriate naming.

    Args:
        mol: RDKit Mol object
        cation_sites: List of cation site dictionaries
        anion_sites: List of anion site dictionaries

    Returns:
        Inferred zwitterion name, or empty string on failure.
    """
    # Count total atoms for structural inference
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    # Check for common zwitterion patterns

    # Betaine pattern: R3N+-CH2-COO-
    betaine_pattern = Chem.MolFromSmarts('[NX4+](C)(C)(C)CC([O-])=O')
    if betaine_pattern and mol.HasSubstructMatch(betaine_pattern):
        return 'betaine'

    # For N+/O- zwitterions, best approach is to neutralize and name
    # as the parent amino compound (IUPAC recommendation for amino acids)
    neutral_name = _name_as_neutral(mol, 'pin')
    if neutral_name:
        return neutral_name

    # Fallback: systematic ammonium + carboxylate (with space!)
    if carbon_count > 0:
        from ..data.chain_names import get_anoate_name
        CHAIN_TO_CARBOXYLATE = {
            2: 'acetate', 3: 'propanoate', 4: 'butanoate',
            5: 'pentanoate', 6: 'hexanoate',
        }
        base = CHAIN_TO_CARBOXYLATE.get(carbon_count)
        if not base:
            try:
                base = get_anoate_name(carbon_count)
            except ValueError:
                return ''
        return f'ammonium {base}'

    return ''


# === SALT DETECTION HELPERS ===

def is_salt(mol) -> bool:
    """
    Check if a molecule is a salt (has separate cation and anion fragments).

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule is a salt

    Example:
        >>> mol = Chem.MolFromSmiles('[Na+].[Cl-]')
        >>> is_salt(mol)
        True
    """
    if mol is None:
        return False

    frags = parse_salt_fragments(mol)
    return bool(frags['cations'] and frags['anions'])


def is_zwitterion(mol) -> bool:
    """
    Check if a molecule is a zwitterion (internal + and - charges).

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule is a zwitterion

    Example:
        >>> mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        >>> is_zwitterion(mol)
        True
    """
    if mol is None:
        return False

    # Zwitterion: single fragment with both + and - charges that cancel
    frags = Chem.GetMolFrags(mol, asMols=True)

    if len(frags) != 1:
        return False

    # Check for both positive and negative atoms
    has_positive = False
    has_negative = False

    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge > 0:
            has_positive = True
        elif charge < 0:
            has_negative = True

    if not (has_positive and has_negative):
        return False

    # Net charge should be zero
    net_charge = Chem.GetFormalCharge(mol)
    return net_charge == 0
