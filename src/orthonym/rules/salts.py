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


# === VARIABLE-VALENCE METALS (BBR-CHG-169.6-caveat / D-13) ===
# Metals that exhibit more than one common oxidation state and therefore carry a
# Stock oxidation-state numeral in their salt cation word (IR-5.4.2.2 / P-65.6.2.1):
# e.g. gold(I) chloride, iron(II/III). FIXED-valence metals (group 1/2, Al, Zn, Ag,
# Sc, Ge, ...) do NOT carry a Stock numeral (sodium chloride, calcium dichloride).
# This restores the 169.6-pre 'gold(I) chloride' that the salt path regressed to
# 'gold chloride' (audit Dim-08 §B Cause 2, with the framing correction).
_VARIABLE_VALENCE_METALS = frozenset({
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",          # 3d transition (variable)
    "Mo", "W", "Tc", "Re", "Ru", "Os", "Rh", "Ir", "Pd", "Pt",  # 4d/5d transition
    "Au", "Hg", "Sn", "Pb", "Tl", "Sb", "Bi", "Ce", "Eu", "Sm", "Yb", "U",
})


def _with_stock_if_variable_valence(frag_mol, word: str) -> str:
    """Append the Stock oxidation-state numeral to a salt cation word IFF the cation
    is a MONATOMIC variable-valence metal (IR-5.4.2.2). For a monatomic metal cation
    the oxidation state equals the formal charge. Fixed-valence metals are unchanged."""
    if frag_mol.GetNumHeavyAtoms() != 1:
        return word
    atom = frag_mol.GetAtomWithIdx(0)
    if atom.GetSymbol() not in _VARIABLE_VALENCE_METALS:
        return word
    ox = atom.GetFormalCharge()
    if ox <= 0:
        return word
    try:
        from .organometallics import _to_roman
        return f"{word}({_to_roman(ox)})"
    except (ValueError, ImportError):
        return word


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

# SMARTS for beta-amino acid zwitterion pattern (2-carbon gap)
BETA_AA_ZWITTERION = '[NX4+;H3][CX4][CX4][CX3](=[OX1])[OX1-]'

# SMARTS for gamma-amino acid zwitterion pattern (3-carbon gap)
GAMMA_AA_ZWITTERION = '[NX4+;H3][CX4][CX4][CX4][CX3](=[OX1])[OX1-]'


# === RETAINED AMINO ACID NAMES ===

# Map canonical SMILES of zwitterion form to trivial name.
# Both chirality variants are included where the canonical SMILES
# differs depending on input notation (e.g., @@ vs @).
RETAINED_AMINO_ACID_ZWITTERIONS = {
    # Glycine zwitterion
    '[NH3+]CC(=O)[O-]': 'glycine',
    # Alanine zwitterion
    'C[C@H]([NH3+])C(=O)[O-]': 'L-alanine',
    'C[C@@H]([NH3+])C(=O)[O-]': 'D-alanine',
    'CC([NH3+])C(=O)[O-]': 'alanine',
    # Valine zwitterion
    'CC(C)[C@@H]([NH3+])C(=O)[O-]': 'L-valine',
    'CC(C)[C@H]([NH3+])C(=O)[O-]': 'L-valine',
    # Leucine zwitterion
    'CC(C)C[C@@H]([NH3+])C(=O)[O-]': 'L-leucine',
    'CC(C)C[C@H]([NH3+])C(=O)[O-]': 'L-leucine',
    # Isoleucine zwitterion
    'CC[C@H](C)[C@@H]([NH3+])C(=O)[O-]': 'L-isoleucine',
    'CC[C@H](C)[C@H]([NH3+])C(=O)[O-]': 'L-isoleucine',
    # Serine zwitterion
    '[NH3+][C@@H](CO)C(=O)[O-]': 'L-serine',
    '[NH3+][C@H](CO)C(=O)[O-]': 'L-serine',
    # Threonine zwitterion
    'C[C@@H](O)[C@@H]([NH3+])C(=O)[O-]': 'L-threonine',
    'C[C@H](O)[C@@H]([NH3+])C(=O)[O-]': 'L-threonine',
    # Proline zwitterion
    'O=C([O-])[C@@H]1CCC[NH2+]1': 'L-proline',
    # Phenylalanine zwitterion
    '[NH3+][C@@H](Cc1ccccc1)C(=O)[O-]': 'L-phenylalanine',
    # Tyrosine zwitterion
    '[NH3+][C@@H](Cc1ccc(O)cc1)C(=O)[O-]': 'L-tyrosine',
    # Tryptophan zwitterion
    '[NH3+][C@@H](Cc1c[nH]c2ccccc12)C(=O)[O-]': 'L-tryptophan',
    # Methionine zwitterion
    'CSCC[C@@H]([NH3+])C(=O)[O-]': 'L-methionine',
    # Histidine zwitterion
    '[NH3+][C@@H](Cc1c[nH]cn1)C(=O)[O-]': 'L-histidine',
    # Glutamic acid zwitterion (one COOH protonated)
    '[NH3+][C@@H](CCC(=O)O)C(=O)[O-]': 'L-glutamic acid',
    # Aspartic acid zwitterion (one COOH protonated)
    '[NH3+][C@@H](CC(=O)O)C(=O)[O-]': 'L-aspartic acid',
    # Beta-alanine zwitterion (beta-amino acid)
    '[NH3+]CCC(=O)[O-]': 'beta-alanine',
    # GABA zwitterion (gamma-aminobutyric acid)
    '[NH3+]CCCC(=O)[O-]': '4-aminobutanoic acid',
    # NOTE (169.6-04): the hardcoded betaine literal entry was DELETED. 'betaine'
    # is NOT OPSIN-parseable (the validity gate suppressed it to 'unknown organic
    # compound'); the route_charged GUARD-4 structured producer now emits the
    # RT-correct (trimethylazaniumyl)acetate (P-74.1.3). No per-molecule literal.
    # L-Carnitine zwitterion
    'C[N+](C)(C)C[C@H](O)CC(=O)[O-]': 'L-carnitine',
    # DL-Carnitine (racemic)
    'C[N+](C)(C)CC(O)CC(=O)[O-]': 'carnitine',
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
                    from ..assembly.fragment_naming import name_fragment_recursively
                    organic_name = name_fragment_recursively(main_frag['smiles'])
                    if organic_name:
                        salt_suffix = ' '.join(sorted(hydroacid_names))
                        return f"{organic_name} {salt_suffix}"
                except (RecursionError, ValueError, RuntimeError):
                    pass

    cation_names = []
    anion_names = []

    # Process cations (excluding H+ fragments already handled above).
    # P-65.6.2.1: the cation word is the element name (metal) or 'ammonium'
    # (NH4+). The CATION_WORDS table (data/cation_words.py) is the single source
    # of truth, reusing namer._METAL_NAMES; fall back to the INORGANIC_CATIONS
    # retained-name table and then to organic cation naming for substituted
    # ammoniums / carbenium counter-cations.
    from ..data.cation_words import get_cation_word
    cation_list = other_cation_frags if h_plus_frags else frags['cations']
    for cation_frag in cation_list:
        frag_mol = cation_frag['mol']
        smiles = cation_frag['smiles']

        word = get_cation_word(frag_mol)
        if word:
            # D-13: variable-valence metal cations carry the Stock oxidation state
            # (gold(I) chloride); fixed-valence metals (Na/K/Ca/...) do not.
            cation_names.append(_with_stock_if_variable_valence(frag_mol, word))
        elif smiles in INORGANIC_CATIONS:
            cation_names.append(INORGANIC_CATIONS[smiles])
        else:
            # Substituted organic cation (e.g. tetramethylammonium) -> name_cation.
            name = name_cation(frag_mol, style)
            if name:
                cation_names.append(name)
            # Skip unnamed cations rather than using generic 'cation'

    # Process anions. P-65.6.2.1 / P-63.8.1: name the ORGANIC anion via the
    # route_charged chokepoint (it owns the parent decision: -oate / -olate /
    # -sulfonate / -ide), falling back to name_anion and the INORGANIC_ANIONS
    # retained table (chloride / sulfate / phosphate — which route_charged does
    # not name). The cation is NOT substituted in (salt = functionalization).
    from .charged_router import route_charged
    for anion_frag in frags['anions']:
        frag_mol = anion_frag['mol']
        smiles = anion_frag['smiles']

        if smiles in INORGANIC_ANIONS:
            anion_names.append(INORGANIC_ANIONS[smiles])
            continue
        # Organic anion: chokepoint first (the sound parent decision), then the
        # legacy name_anion path on '' (retained carboxylate names etc.).
        name = route_charged(frag_mol, style) or name_anion(frag_mol, style)
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
        >>> mol = Chem.MolFromSmiles('C[N+](C)(C)CC(=O)[O-]')
        >>> name_zwitterion(mol)
        '(trimethylazaniumyl)acetate'
    """
    if mol is None:
        return ''

    # Check for retained amino acid names first (D-06: amino-acid zwitterions +
    # betaines sequenced first). These (glycine / L-alanine / ...) are valid
    # OPSIN-parseable retained names per P-74 (which allows retained names).
    if style != 'systematic':
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in RETAINED_AMINO_ACID_ZWITTERIONS:
            return RETAINED_AMINO_ACID_ZWITTERIONS[canonical]

    # GUARD 4 (P-74.0): the route_charged chokepoint owns the anion-is-parent
    # override + the structured (…azaniumyl) cation prefix (P-74.1.3). This
    # REPLACES the deleted hardcoded "2-azaniumyl{base}" / "betaine" / "ammonium
    # {base}" f-string band-aids (fix-methodology.md: structured, not literal).
    from .charged_router import route_charged
    routed = route_charged(mol, style)
    if routed:
        return routed

    # Check for amino acid zwitterion pattern (the neutral-form path: GABA ->
    # 4-aminobutanoic acid, which is RT-correct at connectivity since InChI-L1
    # ignores charge — the deferred D-03 precision path).
    if _is_amino_acid_zwitterion(mol):
        return _name_amino_acid_zwitterion(mol, style)

    # General zwitterion naming
    return _name_general_zwitterion(mol, style)


def _is_amino_acid_zwitterion(mol) -> bool:
    """
    Check if molecule is amino acid zwitterion [NH3+]-Cn-[COO-].

    Detects alpha, beta, and gamma amino acid zwitterion patterns
    per IUPAC P-74.1.1.

    Args:
        mol: RDKit Mol object

    Returns:
        True if molecule matches alpha/beta/gamma amino acid zwitterion pattern

    Example:
        >>> mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        >>> _is_amino_acid_zwitterion(mol)
        True
        >>> mol = Chem.MolFromSmiles('[NH3+]CCC([O-])=O')
        >>> _is_amino_acid_zwitterion(mol)
        True
    """
    # Check alpha, beta, and gamma patterns
    for smarts in (ALPHA_AA_ZWITTERION, BETA_AA_ZWITTERION, GAMMA_AA_ZWITTERION):
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return True

    return False


def _name_amino_acid_zwitterion(mol, style: str) -> str:
    """
    Name amino acid zwitterion (e.g., glycine zwitterion).

    Per IUPAC P-74 recommendation, amino acid zwitterions may be named as their
    neutral form (e.g., "2-aminoacetic acid" for glycine zwitterion). OPSIN
    parses these neutral-form names correctly, and they are RT-correct at
    connectivity (InChI-L1 ignores charge).

    The structured P-74.1.3 ionic form ((azaniumyl)…oate) is produced UPSTREAM
    by route_charged GUARD 4 (called first in name_zwitterion); this function is
    only reached when the chokepoint declined, so it returns the neutral form or
    '' (honest-fail) — the carbon-counting "2-azaniumyl{base}" f-string band-aid
    was DELETED (169.6-04, fix-methodology.md).

    Args:
        mol: RDKit Mol object
        style: 'pin' for systematic, others may use trivial

    Returns:
        Neutral amino acid name, or '' on failure (NO carbon-counting fallback).
    """
    # Neutralize and name the neutral form (the P-74 neutral-form recommendation).
    neutral_mol = _neutralize_zwitterion(mol)
    if neutral_mol is not None:
        try:
            neutral_smiles = Chem.MolToSmiles(neutral_mol, canonical=True)
            if neutral_smiles:
                from ..assembly.fragment_naming import name_fragment_recursively
                neutral_name = name_fragment_recursively(neutral_smiles)
                if neutral_name and neutral_name != 'zwitterion':
                    return neutral_name
        except (RecursionError, ValueError, RuntimeError):
            pass

    # Honest-fail (no carbon-counting band-aid): the structured route_charged
    # GUARD-4 path ran first; if both declined there is no valid name here.
    return ''


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

        from ..assembly.fragment_naming import name_fragment_recursively
        neutral_name = name_fragment_recursively(neutral_smiles)
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

    169.6-04: the hardcoded ``betaine`` SMARTS literal and the carbon-counting
    ``ammonium {base}`` f-string band-aids were DELETED. The structured
    P-74.1.3 form is produced UPSTREAM by route_charged GUARD 4 (called first in
    name_zwitterion); this function is only reached when the chokepoint declined,
    so it returns the neutral-form name or '' (honest-fail, fix-methodology.md).
    """
    # Neutralize and name the parent amino compound (the P-74 neutral-form
    # recommendation for amino-acid-shaped zwitterions). No literal, no
    # carbon-counting fallback.
    neutral_name = _name_as_neutral(mol, 'pin')
    if neutral_name:
        return neutral_name

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
