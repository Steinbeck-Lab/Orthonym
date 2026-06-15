"""
Ion naming rules per IUPAC 2013.

Handles naming of:
- Anions: carboxylates (-ate), alkoxides (-olate), phenolates, aminides, carbanions (-ide)
- Cations: aminium (-aminium), carbenium/ylium (-ylium), onium, diazonium

IUPAC 2013 References:
- P-72: Anion nomenclature
- P-73: Cation nomenclature

Key naming patterns:
- Carboxylate anions: acetic acid -> acetate
- Alkoxide anions: methanol -> methanolate (PIN) or methoxide (acceptable)
- Aminium cations: amine -> aminium (protonated amine)
- Carbenium cations: alkane -> ylium (loss of H-)
"""

import re
from typing import Dict, List, Optional, Any
from rdkit import Chem

from ..assembly.naming_utils import get_alkyl_name
from ..data.ion_retained_names import get_anion_name, get_cation_name
from ..perception.ions import get_ion_sites


# === SUFFIX MAPPINGS ===

ANION_SUFFIXES = {
    'carboxylate': 'ate',       # -COOH -> -COO-
    'alkoxide': 'olate',        # -OH -> -O-
    'phenolate': 'olate',       # PhOH -> PhO-
    'aminide': 'aminide',       # -NH2 -> -NH-
    'carbanion': 'ide',         # C-H -> C-
    'thiolate': 'thiolate',     # -SH -> -S-
}

CATION_SUFFIXES = {
    'aminium': 'aminium',       # -NH2 + H+ -> -NH3+
    'ylium': 'ylium',           # CH4 - H- -> CH3+
    'ium': 'ium',               # add H+
    'onium': 'onium',           # O/S/P cations
    'diazonium': 'diazonium',   # -N2+
}


# === ANION CLASSIFICATION ===

def classify_anion(mol, anion_site: Dict[str, Any]) -> str:
    """
    Classify anion type based on the anionic atom environment.

    Examines the local chemical environment of the anionic atom to
    determine the appropriate naming suffix.

    Args:
        mol: RDKit Mol object
        anion_site: Dictionary from get_ion_sites containing:
            - atom_idx: int
            - charge: int
            - element: str
            - hybridization: str
            - n_hydrogens: int

    Returns:
        One of: 'carboxylate', 'alkoxide', 'phenolate', 'aminide',
                'carbanion', 'thiolate', or 'unknown'

    Example:
        >>> mol = Chem.MolFromSmiles('CC(=O)[O-]')
        >>> sites = get_ion_sites(mol)
        >>> classify_anion(mol, sites['anions'][0])
        'carboxylate'
    """
    atom_idx = anion_site['atom_idx']
    element = anion_site['element']
    atom = mol.GetAtomWithIdx(atom_idx)

    # Classify based on element
    if element == 'O':
        # Check if part of carboxyl group (carboxylate)
        # Carboxylate: O- connected to C which has double bond to another O
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C':
                # Check for carbonyl oxygen (C=O)
                for second_neighbor in neighbor.GetNeighbors():
                    if second_neighbor.GetIdx() != atom_idx:
                        if second_neighbor.GetSymbol() == 'O':
                            bond = mol.GetBondBetweenAtoms(
                                neighbor.GetIdx(), second_neighbor.GetIdx()
                            )
                            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                                return 'carboxylate'

                # Check if attached to aromatic carbon (phenolate)
                if neighbor.GetIsAromatic():
                    return 'phenolate'

        # SUB-01/D-02: heteroatom-oxoacid anion (S/P-bound [O-]) recognition,
        # BEFORE the alkoxide fallthrough. Previously these fell to 'alkoxide'
        # -> the heptanolate carbon-counting stub. Recognized here ONLY for
        # routing (name_anion sends them to the general parent-selection
        # pipeline + the structured ionic suffix); replicates the carboxylate
        # bond-walk shape (count Chem.BondType.DOUBLE bonds to O).
        heavy = [n for n in atom.GetNeighbors() if n.GetSymbol() != 'H']
        if len(heavy) == 1:
            nb = heavy[0]
            # TRUE C-bonded sulfonate/sulfinate/phosphonate ONLY (a C-S / C-P
            # bond on the heteroatom). Sulfate / phosphate ESTERS (R-O-S /
            # R-O-P, no carbon on the heteroatom) are a SEPARATE class — leave
            # them on the existing path so the sulfated-glycolipid / phospho-
            # lipid canaries do not regress. SUB-01/D-02.
            nb_has_carbon = any(nn.GetSymbol() == 'C' for nn in nb.GetNeighbors())
            if nb.GetSymbol() == 'S' and nb_has_carbon:
                double_o = sum(
                    1 for nn in nb.GetNeighbors()
                    if nn.GetSymbol() == 'O' and nn.GetIdx() != atom_idx
                    and mol.GetBondBetweenAtoms(nb.GetIdx(), nn.GetIdx())
                        .GetBondType() == Chem.BondType.DOUBLE
                )
                if double_o >= 2:
                    return 'sulfonate'
                if double_o == 1:
                    return 'sulfinate'
            elif nb.GetSymbol() == 'P' and nb_has_carbon:
                return 'phosphonate'

        # Default to alkoxide (O- attached to alkyl)
        return 'alkoxide'

    elif element == 'N':
        # Nitrogen anion (aminide)
        return 'aminide'

    elif element == 'C':
        # Carbon anion (carbanion)
        return 'carbanion'

    elif element == 'S':
        # Sulfur anion (thiolate)
        return 'thiolate'

    return 'unknown'


# === CATION CLASSIFICATION ===

def classify_cation(mol, cation_site: Dict[str, Any]) -> str:
    """
    Classify cation type based on the cationic atom environment.

    Examines the local chemical environment of the cationic atom to
    determine the appropriate naming suffix.

    Args:
        mol: RDKit Mol object
        cation_site: Dictionary from get_ion_sites containing:
            - atom_idx: int
            - charge: int
            - element: str
            - hybridization: str
            - n_hydrogens: int

    Returns:
        One of: 'aminium', 'ylium', 'onium', 'diazonium', or 'unknown'

    Example:
        >>> mol = Chem.MolFromSmiles('[NH4+]')
        >>> sites = get_ion_sites(mol)
        >>> classify_cation(mol, sites['cations'][0])
        'aminium'
    """
    atom_idx = cation_site['atom_idx']
    element = cation_site['element']
    atom = mol.GetAtomWithIdx(atom_idx)

    if element == 'N':
        # Check for diazonium (N2+ pattern: N connected to another N)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'N':
                # Diazonium: R-N+=N or R-N=N+
                return 'diazonium'

        # Phase 184 WS-E.1 (P-73.1.2.1): a QUATERNARY ammonium N (formal charge
        # +1, NO hydrogens to remove, degree >= 4) cannot be neutralized by the
        # proton-removal path the protonated-amine cascade uses — removing a
        # non-existent H leaves an over-valent neutral N and SanitizeMol raises
        # (RESEARCH Pitfall 2). It is named systematically by DEMOTING N -> a
        # carbon-context amine parent and appending '-ium' to the '-amine' suffix
        # (name_quaternary_aminium), so it needs a discriminator distinct from the
        # protonated-amine 'aminium' value.
        if (atom.GetFormalCharge() == 1
                and atom.GetTotalNumHs() == 0
                and atom.GetDegree() >= 4):
            return 'quaternary'

        # Protonated nitrogen -> aminium
        # Includes primary (NH3+), secondary (NH2+), tertiary (NH+)
        return 'aminium'

    elif element == 'C':
        # Carbocation -> ylium (carbenium ion)
        return 'ylium'

    elif element in ('O', 'S', 'P', 'Se'):
        # Oxonium, sulfonium, phosphonium, selenonium
        return 'onium'

    return 'unknown'


# === SUFFIX TRANSFORMATIONS ===

def name_carboxylate_anion(parent_name: str) -> str:
    """
    Convert carboxylic acid name to carboxylate anion name.

    Transformation rules:
    - 'ic acid' -> 'ate' (e.g., acetic acid -> acetate)
    - 'oic acid' -> 'oate' (e.g., propanoic acid -> propanoate)

    Args:
        parent_name: Parent acid name (e.g., 'acetic acid', 'propanoic acid')

    Returns:
        Anion name (e.g., 'acetate', 'propanoate')

    Example:
        >>> name_carboxylate_anion('acetic acid')
        'acetate'
        >>> name_carboxylate_anion('propanoic acid')
        'propanoate'
    """
    name = parent_name.strip()
    name_lower = name.lower()

    # Handle 'oic acid' ending (systematic names)
    if name_lower.endswith('oic acid'):
        return name[:-8] + 'oate'

    # Handle 'ic acid' ending (trivial names like acetic, formic)
    if name_lower.endswith('ic acid'):
        return name[:-7] + 'ate'

    # Fallback: just add -ate
    if ' acid' in name_lower:
        idx = name_lower.index(' acid')
        return name[:idx] + 'ate'
    return name + 'ate'


def name_alkoxide_anion(parent_name: str, style: str = 'pin') -> str:
    """
    Convert alcohol name to alkoxide anion name.

    IUPAC 2013 PIN style uses -olate (methanolate, ethanolate).
    Common style uses -oxide (methoxide, ethoxide).

    Args:
        parent_name: Parent alcohol name (e.g., 'methanol', 'ethanol')
        style: 'pin' for -olate, 'common' for -oxide

    Returns:
        Anion name

    Example:
        >>> name_alkoxide_anion('methanol')
        'methanolate'
        >>> name_alkoxide_anion('methanol', style='common')
        'methoxide'
    """
    name = parent_name.strip()
    name_lower = name.lower()

    if style == 'pin':
        # PIN style: methanol -> methanolate
        if name_lower.endswith('ol'):
            return name + 'ate'
        elif name_lower.endswith('anol'):
            return name + 'ate'
        else:
            return name + 'olate'
    else:
        # Common style: methanol -> methoxide
        if name_lower.endswith('anol'):
            return name[:-4] + 'oxide'
        elif name_lower.endswith('ol'):
            return name[:-2] + 'oxide'
        else:
            return name + 'oxide'


def name_phenolate_anion(parent_name: str) -> str:
    """
    Convert phenol name to phenolate anion name.

    Args:
        parent_name: Parent phenol name (e.g., 'phenol')

    Returns:
        Anion name (e.g., 'phenolate')

    Example:
        >>> name_phenolate_anion('phenol')
        'phenolate'
    """
    name = parent_name.strip()
    name_lower = name.lower()

    if name_lower.endswith('ol'):
        return name + 'ate'
    else:
        return name + 'olate'


def name_aminium_cation(parent_name: str) -> str:
    """
    Convert amine name to aminium cation name.

    Args:
        parent_name: Parent amine name (e.g., 'methanamine', 'ethylamine')

    Returns:
        Cation name (e.g., 'methanaminium', 'ethylaminium')

    Example:
        >>> name_aminium_cation('methanamine')
        'methanaminium'
        >>> name_aminium_cation('ammonia')
        'ammonium'
    """
    name = parent_name.strip()
    name_lower = name.lower()

    # Special case: ammonia -> ammonium
    if name_lower == 'ammonia':
        return 'ammonium'

    # amine -> aminium
    if name_lower.endswith('amine'):
        return name[:-1] + 'ium'
    elif name_lower.endswith('ane'):
        # Handle alkane-based names
        return name[:-1] + 'ium'
    else:
        return name + 'ium'


def name_carbenium_cation(parent_name: str) -> str:
    """
    Convert alkane/alkyl name to carbenium (ylium) cation name.

    The ylium suffix indicates loss of hydride (H-) from the parent.

    Args:
        parent_name: Parent name (e.g., 'methane', 'methyl')

    Returns:
        Cation name (e.g., 'methylium')

    Example:
        >>> name_carbenium_cation('methane')
        'methylium'
        >>> name_carbenium_cation('ethane')
        'ethylium'
    """
    name = parent_name.strip()
    name_lower = name.lower()

    # Handle -ane suffix (alkanes)
    if name_lower.endswith('ane'):
        return name[:-3] + 'ylium'

    # Handle -yl suffix (already a radical/substituent form)
    if name_lower.endswith('yl'):
        return name + 'ium'

    # Default: add -ylium
    return name + 'ylium'


# === ANION SUFFIX GETTERS ===

def get_anion_suffix(anion_type: str) -> str:
    """
    Get the appropriate suffix for an anion type.

    Args:
        anion_type: Type from classify_anion()

    Returns:
        Suffix string (e.g., 'ate', 'olate', 'ide')
    """
    return ANION_SUFFIXES.get(anion_type, 'ide')


def get_cation_suffix(cation_type: str) -> str:
    """
    Get the appropriate suffix for a cation type.

    Args:
        cation_type: Type from classify_cation()

    Returns:
        Suffix string (e.g., 'ium', 'ylium', 'aminium')
    """
    return CATION_SUFFIXES.get(cation_type, 'ium')


# === MAIN NAMING FUNCTIONS ===

def name_anion(mol, style: str = 'pin', _depth: int = 0, retained_only: bool = False) -> str:
    """
    Generate IUPAC name for an anionic molecule.

    Workflow:
    1. Get canonical SMILES
    2. Check retained names (unless systematic style)
    3. Detect anion sites
    4. Classify anion type
    5. Generate systematic name

    Args:
        mol: RDKit Mol object (must have negative charge)
        style: Naming style ('pin', 'systematic', 'common')
        _depth: Internal recursion depth guard (do not set manually)
        retained_only: If True, only check retained names and return None
            if no retained name found (used for namer.py fall-through).

    Returns:
        Anion name (e.g., 'acetate', 'methoxide', 'phenolate'),
        or empty string if naming fails.
        When retained_only=True, returns None if no retained name found.

    Example:
        >>> mol = Chem.MolFromSmiles('CC(=O)[O-]')
        >>> name_anion(mol)
        'acetate'
    """
    if mol is None:
        return '' if not retained_only else None

    # Guard against infinite recursion
    if _depth > 2:
        return '' if not retained_only else None

    # Get canonical SMILES for lookup
    canonical = Chem.MolToSmiles(mol, canonical=True)

    # Check retained names first (unless systematic requested)
    if style != 'systematic':
        retained = get_anion_name(canonical)
        if retained:
            return retained

    # If only checking retained names, return None to signal fall-through
    if retained_only:
        return None

    # WS-E.3-dianion (D-13, P-12.2 / P-34.1.4): a BARE carbon-free inorganic
    # oxoacid anion ([O-]S(=O)(=O)[O-], [O-]P(=O)([O-])[O-], [O-]C(=O)[O-]) is
    # its preselected functional-parent anion word. Run BEFORE classify_anion,
    # which mis-routes both sulfate [O-] to 'alkoxide' (ions.py:114 needs
    # nb_has_carbon) -> _try_neutralize_and_name -> carbon-free 'unknown'.
    # Carbon-free guard => cannot intercept a sulfate/phosphate ESTER or the
    # NP-conjugate path (steroid sulfate, gold WSC-03).
    inorganic = _name_inorganic_oxoacid_anion(mol)
    if inorganic:
        return _validate_anion_name(mol, inorganic)

    # Get anion sites
    sites = get_ion_sites(mol)
    anions = sites.get('anions', [])

    if not anions:
        return ''

    # For single anion, classify and name
    if len(anions) == 1:
        anion_site = anions[0]
        anion_type = classify_anion(mol, anion_site)

        # CARBOXYLATE keeps its existing proven path FIRST (byte-identical: it
        # carries the retained aromatic-carboxylate detection — benzoate /
        # 2-naphthoate — that the systematic chokepoint name would not reproduce;
        # Pitfall 4 / "the anion path must stay byte-identical"). route_charged
        # also names carboxylates correctly, but routing them here would flip
        # 2-naphthoate -> naphthalene-2-carboxylate (a style change, not a fix).
        if anion_type == 'carboxylate':
            return _validate_anion_name(
                mol, _name_carboxylate_systematic(mol, anion_site))

        # 169.6-03 (CHOKE-01): for every OTHER single-anion class delegate to the
        # SINGLE route_charged chokepoint. It generalizes _name_oxoacid_anion
        # (neutralize -> re-enter the full pipeline -> re-apply the class-correct
        # ionic suffix), so the alkoxide/phenolate/carbanion/thiolate
        # carbon-counting stubs (the heptanolate bug class) are DELETED. '' on
        # failure -> fall through to the legacy cascade below.
        from .charged_router import route_charged
        routed = route_charged(mol, style)
        if routed:
            return _validate_anion_name(mol, routed)

        # Phase 182 (WSC-03, D-04/D-09/D-10): natural-product conjugate ANION. route_charged
        # deliberately declines a sulfate/phosphate ESTER [O-] (charged_router.py:386-389:
        # "out of scope -> neutral-form names it"), and the neutralize-recurse below would
        # strip the charge BEFORE the NP path sees it, emitting the free-acid word
        # ("hydrogen sulfate") which does NOT OPSIN-round-trip to the anion. To derive the
        # correct anion word ("sulfate") per D-04 (classify the protonation state IN PLACE,
        # never neutralize-then-rename), let the NP conjugate path see the CHARGED mol here.
        # NARROW gate (D-09): only when an unclaimed sulfate/phosphate/glycosyl conjugate is
        # present on a recognized NP scaffold; the NP path internally RT-gates the conjugate
        # name (D-10), so a returned name has already round-tripped. Non-conjugate anions are
        # untouched. The NP path returns None (honest-fail) for anything it cannot name.
        np_conj = _name_np_conjugate_anion(mol)
        if np_conj:
            return _validate_anion_name(mol, np_conj)

        # Fall-through cascade (route_charged declined, e.g. metal/fragment).
        if anion_type in ('sulfonate', 'sulfinate', 'phosphonate'):
            result = _name_oxoacid_anion(mol, style)
        else:
            result = _try_neutralize_and_name(mol)

        return _validate_anion_name(mol, result)

    # Multiple anions - try neutralize-then-name approach.
    # CR-02 (code review 2026-06-02): a fully-deprotonated S/P-oxoacid (e.g. the
    # methylphosphonate DIANION, CP(=O)([O-])[O-]) reaches this branch with
    # len(anions) >= 2 and MUST NOT ship the neutral acid name. _name_oxoacid_anion
    # already neutralizes ALL non-internal [O-] and applies the canonical ionic
    # suffix, so it handles the multi-charge oxoacid exactly as the single-anion
    # path does ("methanephosphonic acid" -> "methanephosphonate", which OPSIN
    # round-trips to the -2 dianion). Routed only when EVERY anion is an S/P-oxoacid
    # site; a carboxylate or mixed case falls through to the handling below.
    oxo_anion_types = ('sulfonate', 'sulfinate', 'phosphonate')
    if all(classify_anion(mol, a) in oxo_anion_types for a in anions):
        oxo_name = _name_oxoacid_anion(mol, style)
        if oxo_name:
            return _validate_anion_name(mol, oxo_name)

    neutral_name = _try_neutralize_and_name(mol)
    if neutral_name:
        # Convert acid suffixes to carboxylate suffixes for deprotonated
        # carboxylate sites (IUPAC P-72.2.1: -oic acid -> -oate)
        carboxylate_count = sum(
            1 for a in anions if classify_anion(mol, a) == 'carboxylate'
        )
        if carboxylate_count > 0:
            anion_name = _acid_name_to_carboxylate(neutral_name, carboxylate_count)
            if anion_name:
                return _validate_anion_name(mol, anion_name)
        return _validate_anion_name(mol, neutral_name)

    # Fallback: name first anion site only
    anion_site = anions[0]
    anion_type = classify_anion(mol, anion_site)
    if anion_type == 'carboxylate':
        result = _name_carboxylate_systematic(mol, anion_site)
        return _validate_anion_name(mol, result)
    return ''


def name_cation(mol, style: str = 'pin', _depth: int = 0, retained_only: bool = False) -> str:
    """
    Generate IUPAC name for a cationic molecule.

    Workflow:
    1. Get canonical SMILES
    2. Check retained names (unless systematic style)
    3. Detect cation sites
    4. Classify cation type
    5. Generate systematic name

    Args:
        mol: RDKit Mol object (must have positive charge)
        style: Naming style ('pin', 'systematic', 'common')
        _depth: Internal recursion depth guard (do not set manually)
        retained_only: If True, only check retained names and return None
            if no retained name found (used for namer.py fall-through).

    Returns:
        Cation name (e.g., 'ammonium', 'methylammonium', 'methylium'),
        or empty string if naming fails.
        When retained_only=True, returns None if no retained name found.

    Example:
        >>> mol = Chem.MolFromSmiles('[NH4+]')
        >>> name_cation(mol)
        'ammonium'
    """
    if mol is None:
        return '' if not retained_only else None

    # Guard against infinite recursion
    if _depth > 2:
        return '' if not retained_only else None

    # Get canonical SMILES for lookup
    canonical = Chem.MolToSmiles(mol, canonical=True)

    # Check retained names first (unless systematic requested)
    if style != 'systematic':
        retained = get_cation_name(canonical)
        if retained:
            return retained

    # If only checking retained names, return None to signal fall-through
    if retained_only:
        return None

    # Get cation sites
    sites = get_ion_sites(mol)
    cations = sites.get('cations', [])

    if not cations:
        return ''

    # For single cation, classify and name
    if len(cations) == 1:
        cation_site = cations[0]
        cation_type = classify_cation(mol, cation_site)

        # 169.6-03 (CHOKE-01): delegate to the SINGLE route_charged chokepoint
        # FIRST (replacing the carbenium/onium/diazonium carbon-counting stubs,
        # which are DELETED). route_charged neutralizes -> re-enters -> re-applies
        # the class-keyed cation suffix (carbenium ane->ylium, diazonium append,
        # aminium amine->aminium). On '' we fall through to the kept aminium
        # primary path (neutralize-recurse) so aminium stays byte-identical.
        from .charged_router import route_charged
        routed = route_charged(mol, style)
        if routed:
            return _validate_cation_name(mol, routed)

        # Fall-through cascade.
        if cation_type == 'aminium':
            result = _name_aminium_systematic(mol, cation_site)
        else:
            # Generic cation - try to name the neutral skeleton
            result = _try_neutralize_and_name(mol)

        return _validate_cation_name(mol, result)

    # Multiple cations - try neutralize-then-name approach
    neutral_name = _try_neutralize_and_name(mol)
    if neutral_name:
        return _validate_cation_name(mol, neutral_name)

    # Fallback: name first cation site only
    cation_site = cations[0]
    cation_type = classify_cation(mol, cation_site)
    if cation_type == 'aminium':
        result = _name_aminium_systematic(mol, cation_site)
        return _validate_cation_name(mol, result)
    return ''


# === NAME VALIDATION GUARDS ===

def _name_np_conjugate_anion(mol) -> str:
    """Name a natural-product conjugate ANION via the NP conjugate path (Phase 182, WSC-03).

    Fires ONLY when the charged molecule is a recognized NP scaffold carrying an unclaimed
    sulfate / mono-phosphate / glycosyl conjugate (D-09 narrow gate). Naming the CHARGED mol
    lets the conjugate controller derive the anion word IN PLACE (D-04 — `sulfate`, not
    `hydrogen sulfate`), and the NP path internally RT-gates the conjugate name (D-10). Returns
    '' (no match / not a conjugate / NP path declined) so the caller falls through to the
    legacy neutralize-recurse cascade unchanged.

    This is the seam that keeps the original charge visible to the NP subsystem; the anion
    pipeline otherwise neutralizes before the NP path runs, which would lose the anion word.
    """
    try:
        from ..perception.natural_products import detect_natural_product
        from .natural_products import (
            _build_target_to_iupac,
            _find_glycosyl_conjugates,
            _find_phosphate_conjugates,
            _find_sulfate_conjugates,
            name_natural_product,
        )
        scaffold_info = detect_natural_product(mol)
        if scaffold_info is None:
            return ''
        numbering = _build_target_to_iupac(scaffold_info)
        if numbering is None:
            return ''
        has_conjugate = (
            _find_sulfate_conjugates(mol, scaffold_info, numbering)
            or _find_phosphate_conjugates(mol, scaffold_info, numbering)
            or _find_glycosyl_conjugates(mol, scaffold_info, numbering)
        )
        if not has_conjugate:
            return ''
        name = name_natural_product(mol)
        return name or ''
    except (ValueError, RuntimeError, KeyError, IndexError, RecursionError):
        return ''


def _validate_anion_name(mol, result: str) -> str:
    """Validate an anion name and guard against misapplied suffixes.

    Guards against:
    - 'oic acid' suffix on a molecule without a carboxylic acid group
    - 'methylidene' appearing in an ion name (radical name leaking)

    Args:
        mol: RDKit Mol object
        result: The proposed anion name

    Returns:
        The validated name, or empty string if invalid.
    """
    if not result:
        return ''

    # Guard: 'oic acid' suffix on non-acid molecule
    if 'oic acid' in result:
        from rdkit.Chem import MolFromSmarts
        acid_pattern = MolFromSmarts('[CX3](=O)[OX2H1]')
        if acid_pattern and not mol.HasSubstructMatch(acid_pattern):
            # Molecule doesn't actually have a carboxylic acid - suffix is wrong
            return ''

    # Guard: methylidene false positive (radical name leaking into ion naming)
    if 'methylidene' in result:
        return ''

    # D-10 (Table 3.4 / P-72.2.2.1): a carbanion '-ide' name is a valid anion
    # ending. Explicitly accept it (it passes today via no rejecting guard, but a
    # future guard must not silently suppress it). The single existing risk is the
    # 'methylidene' radical-leak guard above — '-ide' (preceded by 'an-N-') is a
    # distinct, valid suffix and must reach this return.
    if result.endswith('ide') and 'methylidene' not in result:
        return result

    return result


def _validate_cation_name(mol, result: str) -> str:
    """Validate a cation name and guard against misapplied suffixes.

    Guards against:
    - 'aminium' suffix on a molecule without an amine group
    - 'methylidene' appearing in a cation name (radical name leaking)

    Args:
        mol: RDKit Mol object
        result: The proposed cation name

    Returns:
        The validated name, or empty string if invalid.
    """
    if not result:
        return ''

    # Guard: 'aminium' suffix on non-amine molecule
    if 'aminium' in result:
        from rdkit.Chem import MolFromSmarts
        # Match protonated or neutral amines: NH3+, NH2+R, NH+R2, N+R3, NH2, NHR, NR2
        # Phase 184 WS-E.1 D-10: the H0 term of [NX4;H1,H2,H3,H0] ALREADY matches a
        # QUATERNARY [NX4;H0] N+, so a systematic quaternary '-aminium' PIN (e.g.
        # 'N,N,N-trimethylmethanaminium' from name_quaternary_aminium) is accepted
        # here unchanged. This comment is defense-in-depth: a future tightening of
        # this SMARTS MUST keep the H0 term or it will silently suppress the
        # quaternary aminium suffix.
        amine_pattern = MolFromSmarts('[NX4;H1,H2,H3,H0]')
        has_protonated_amine = mol.HasSubstructMatch(amine_pattern) if amine_pattern else False
        neutral_amine = MolFromSmarts('[NX3;H1,H2,H3]')
        has_neutral_amine = mol.HasSubstructMatch(neutral_amine) if neutral_amine else False
        if not has_protonated_amine and not has_neutral_amine:
            # Molecule doesn't actually have an amine group - suffix is wrong
            return ''

    # Guard: methylidene false positive (radical name leaking into cation naming)
    if 'methylidene' in result:
        return ''

    return result


# === NEUTRALIZATION HELPER ===

# Metallic elements that are out of scope for organic naming
_INORGANIC_ELEMENTS = frozenset({
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
    'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
    'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Hf', 'Ta',
    'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
})


# Allowlist of elements Orthonym names inside an organic parent: the core
# non-metals + the metalloids handled by skeletal-replacement ('a') nomenclature
# (sila/germa/arsa/stiba/sela/tellura/bora). Anything OUTSIDE this set is
# inorganic/metallic and out of scope for charged-species naming. Defined as an
# ALLOWLIST so it is COMPLETE -- the legacy _INORGANIC_ELEMENTS frozenset (a
# hardcoded 46-metal include-list) silently OMITTED Tc, U and the entire f-block,
# which let a lone [99Tc] radical reach route_charged and infinite-loop. This
# allowlist preserves the exact prior boundary (Sn/Pb/Tl/Al/Ga/In/Bi remain
# metals; Si/Ge/As/Sb/Se/Te/B remain organic) while catching every missing metal.
_ORGANIC_NONMETALS = frozenset({
    'H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl',
    'Ge', 'As', 'Se', 'Br', 'Sb', 'Te', 'I',
})


def _has_metal(mol) -> bool:
    """True if the molecule contains any metallic/inorganic element -- i.e. any
    atom outside the organic non-metal/metalloid allowlist (_ORGANIC_NONMETALS).
    Complete by construction (catches Tc/U/f-block, unlike the legacy metal list)."""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in _ORGANIC_NONMETALS:
            return True
    return False


def _try_neutralize_and_name(mol) -> str:
    """
    Neutralize a multi-charged ion and name the organic skeleton.

    For multi-charged species, strips all formal charges and names the
    resulting neutral molecule using the standard pipeline.

    Returns:
        Name of the neutralized skeleton, or empty string on failure.
    """
    if mol is None:
        return ''

    # Bail on inorganic/metallic species (out of scope)
    if _has_metal(mol):
        return ''

    try:
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            charge = atom.GetFormalCharge()
            if charge != 0:
                atom.SetFormalCharge(0)
                # Adjust hydrogen count to compensate
                if charge > 0:
                    # Cation: had extra H from protonation, remove them
                    cur_h = atom.GetNumExplicitHs()
                    atom.SetNumExplicitHs(max(0, cur_h - charge))
                elif charge < 0:
                    # Anion: was deprotonated, add H back
                    cur_h = atom.GetNumExplicitHs()
                    atom.SetNumExplicitHs(cur_h + abs(charge))

        try:
            Chem.SanitizeMol(rw)
        except Exception:
            return ''

        neutral_smiles = Chem.MolToSmiles(rw, canonical=True)
        if not neutral_smiles:
            return ''

        # Use unified fragment naming (cycle-detected, session-cached)
        from ..assembly.fragment_naming import name_fragment_recursively
        neutral_name = name_fragment_recursively(neutral_smiles)
        if neutral_name:
            return neutral_name
    except (RecursionError, ValueError, RuntimeError):
        pass

    return ''


# WR-05 (code review 2026-06-02): the exact set of neutral-acid suffixes a
# routed S/P-oxoacid anion can carry. Passed by _name_oxoacid_anion to restrict
# _ionize_acid_name's trailing-suffix match to GENUINE oxoacid suffixes, so the
# short bare entries of _ANION_SUFFIX_MAP ("ol", "amine", "thiol", "amide") can
# never mis-fire on a parent stem that merely happens to end in those letters.
_OXOACID_NEUTRAL_SUFFIXES = frozenset({
    'sulfonic acid', 'sulfinic acid',
    'phosphonic acid', 'phosphinic acid', 'phosphoric acid',
})


# =============================================================================
# 169.6-02 (CHOKE-02, D-10/D-11): class-keyed cation re-application transforms.
#
# Three cation classes are NOT plain neutral->ionic suffix swaps (they cannot be
# expressed in the flat _CATION_SUFFIX_MAP because the transform depends on the
# CATION CLASS, and on WHICH neutral form was neutralized to — acid vs hydride;
# CONTEXT D-02). They are encoded here as a STRUCTURED, declarative, class-keyed
# table (the same single-source-of-truth pattern as _CATION_SUFFIX_MAP, NOT a
# growing list of molecule-specific .replace calls — fix-methodology.md). Each
# row carries its verbatim Blue Book authority; the longest matching neutral
# suffix wins so 'carboxylic acid' is tried before 'ic acid'.
#
#   ylium     (carbenium)  parent-hydride 'ane'->'ylium'  methane->methylium
#                          [P-73.2.2.1.1] — methylium is the PIN; "carbenium" is
#                          NOT a substitutive PIN. Empty->'ium' is WRONG here.
#   acylium                operates on the ACID name: 'oic acid'->'oylium',
#                          'carboxylic acid'->'carbonylium'  [P-73.2.3.1]
#   diazonium              append 'diazonium' to the hydride name (no elision,
#                          consonant-initial)  [P-73.2.2.3]
#
# A trailing ('', suffix) row means "append <suffix> to the whole name" (no stem
# removal) — used by diazonium, which never strips a neutral ending.
_CLASS_KEYED_CATION_TRANSFORMS = {
    # P-73.2.2.1.1: 'ane'->'ylium' (methane->methylium, ethane->ethylium,
    # propane->propylium, cyclobutane->cyclobutylium). The general 'append ylium
    # eliding final e' case is reached via the ('e','ylium') / ('','ylium') rows.
    'ylium': (
        ('ane', 'ylium'),     # P-73.2.2.1.1 (saturated terminal / Group-14)
        ('ene', 'enylium'),   # P-73.2.2.1 general: keep unsaturation, append ylium with e-elision
        ('yne', 'ynylium'),   # P-73.2.2.1 general
        ('e', 'ylium'),       # P-73.2.2.1 general: elide a trailing 'e' then append ylium
        ('', 'ylium'),        # P-73.2.2.1 general: no final 'e' -> append ylium
    ),
    # P-73.2.3.1: acid name -> acylium. 'oic acid'->'oylium',
    # 'carboxylic acid'->'carbonylium' (butanoic acid->butanoylium,
    # cyclohexanecarboxylic acid->cyclohexanecarbonylium).
    'acylium': (
        ('carboxylic acid', 'carbonylium'),  # P-73.2.3.1
        ('oic acid', 'oylium'),              # P-73.2.3.1
        ('ic acid', 'ylium'),                # P-73.2.3.1
    ),
    # P-73.2.2.3: append 'diazonium' to the hydride name (no e-elision; the
    # suffix is consonant-initial). benzene->benzenediazonium,
    # methane->methanediazonium.
    'diazonium': (
        ('', 'diazonium'),  # P-73.2.2.3
    ),
}


def apply_ion_suffix_to_name(name: str, total_charge: int,
                             allowed_suffixes: Optional[frozenset] = None,
                             cation_class: Optional[str] = None) -> str:
    """The SINGLE ionic-suffix re-application primitive for every charged class.

    Generalized from ``_ionize_acid_name`` (169.6-02 / CHOKE-02). Re-applies the
    class-correct ionic suffix to a re-entered NEUTRAL name (the output of
    ``Orthonym(style).name(neutral_smiles)``). Drives the structured resolvers
    seam (``_apply_anion_modification`` / ``_apply_cation_modification`` off the
    canonical ``_ANION_SUFFIX_MAP`` / ``_CATION_SUFFIX_MAP``) — single source of
    truth, NEVER a per-molecule ``.replace`` band-aid (fix-methodology.md).

    Two paths:

    1. ``cation_class`` in {ylium, acylium, diazonium} -> the class-keyed
       transform table ``_CLASS_KEYED_CATION_TRANSFORMS`` (these are NOT plain
       suffix swaps and depend on the neutralized form, CONTEXT D-02):
         - ylium (carbenium): parent-hydride ``ane``->``ylium``
           (``methane``->``methylium``; P-73.2.2.1.1; ``methylium`` is the PIN).
         - acylium: on the ACID name, ``oic acid``->``oylium`` /
           ``carboxylic acid``->``carbonylium`` (P-73.2.3.1).
         - diazonium: append ``diazonium`` to the hydride name (P-73.2.2.3).

    2. Otherwise the generic map-driven path (anion/protonated-amine etc.):
       longest trailing neutral suffix in the relevant SUFFIX_MAP wins.

    ``allowed_suffixes`` (GUARD 1 — the ``heptanolate`` fix): when given, ONLY
    those neutral suffixes are eligible for the trailing match, so a name ending
    "...sulfonic acid" with ``allowed_suffixes={"ol"}`` returns '' (it cannot
    mis-fire to ``-olate``). ``classify_anion`` passes the per-class subset so a
    sulfonate stem can never be mistaken for an alkoxide. Applies to the generic
    (map) path only; the class-keyed cation path is already class-gated.

    Returns the ionized name, or '' if no canonical transform applies (the caller
    then falls through to the existing cascade — v18 byte-identical contract).
    """
    if not name:
        return ''

    # --- Path 1: class-keyed cation transforms (NOT plain suffix swaps) -------
    if total_charge > 0 and cation_class in _CLASS_KEYED_CATION_TRANSFORMS:
        for neutral_suffix, ionic_suffix in _CLASS_KEYED_CATION_TRANSFORMS[cation_class]:
            if neutral_suffix == '':
                # Append-only row (e.g. diazonium): no stem removal, no elision.
                return name + ionic_suffix
            if name.endswith(neutral_suffix):
                return name[:-len(neutral_suffix)] + ionic_suffix
        return ''

    # --- Path 2: generic structured-seam map (anion / protonated amine / ...) -
    from ..assembly.resolvers import (
        _ANION_SUFFIX_MAP, _CATION_SUFFIX_MAP, SuffixInfo,
        _apply_anion_modification, _apply_cation_modification,
    )
    smap = _ANION_SUFFIX_MAP if total_charge < 0 else _CATION_SUFFIX_MAP
    apply = _apply_anion_modification if total_charge < 0 else _apply_cation_modification
    candidates = (
        smap if allowed_suffixes is None
        else {k: v for k, v in smap.items() if k in allowed_suffixes}
    )
    # Longest trailing suffix wins ("sulfonic acid" before "ol").
    for neutral_suffix in sorted(candidates, key=len, reverse=True):
        if name.endswith(neutral_suffix):
            modified = apply(SuffixInfo(text=neutral_suffix, is_terminal=True))
            if modified.text and modified.text != neutral_suffix:
                return name[:-len(neutral_suffix)] + modified.text
    return ''


# === CUMULATIVE PARENT-HYDRIDE SUFFIX (anion -ide / cation -ium / radical -yl) ==

# P-72.2.2.1 (anion, loss of H+ -> -ide) / P-71 + Table 3.4 (radical, loss of
# H. -> -yl/-ylidene/-ylidyne) / P-73.1.2 (cation -ium): the number of hydrogens
# the centre gains (anion) or loses (radical/cation) to recover the SATURATED
# neutral parent hydride. For an anion centre (e.g. [CH-]) the parent gains 1 H
# back ([CH2]); for a monovalent radical it gains 1 H, divalent 2, trivalent 3;
# for a cation it has 1 H too few, so the parent gains... no: a -ium centre is
# protonated, so the parent loses 1 H. (BlueBookV2/BlueBookV2.md Table 3.4,
# lines 17580-17608.)
_CUMULATIVE_SUFFIX_H_DELTA = {
    'ide': +1,        # P-72.2.2.1: anion gained an electron pair losing H+; parent re-adds 1 H
    'yl': +1,         # P-71 / Table 3.4: monovalent radical; parent re-adds 1 H
    'ylidene': +2,    # P-71: divalent radical; parent re-adds 2 H
    'ylidyne': +3,    # P-71: trivalent radical; parent re-adds 3 H
    'ium': -1,        # P-73.1.2: cation is protonated; parent removes 1 H
}


def emit_parent_hydride_cumulative_suffix(mol, center_idx: int, suffix: str) -> str:
    """Name a parent hydride with a cumulative anionic / radical / cationic suffix
    on a single carbon centre, with a FIRST-CLASS locant.

    suffix in {'ide', 'ium', 'yl', 'ylidene', 'ylidyne'}.
      'ide'      -> P-72.2.2.1 / Table 3.4 (anion, loss of H+)
      'yl'/'ylidene'/'ylidyne' -> P-71 / Table 3.4 (radical, loss of H.)
      'ium'      -> P-73.1.2 (cation; reserved for later reuse)

    center_idx is the atom index OF THE CHARGED/RADICAL CARBON ON ``mol`` (the
    index-preserving mol passed in — NOT a freshly canonicalized copy; canonical
    SMILES reorders atoms, RESEARCH Pitfall 1). The centre gets the LOWEST locant
    over the re-found chain, competing with unsaturation and substituents per
    P-31.1.4 / P-14.4 (``orient_chain`` criterion (a), principal_group_atoms =
    {center_idx}). This is a NEW P-72.2.2.1 numbering call ("locants identify
    positions of the negative charges", BlueBookV2 lines 40876-40902), NOT a
    reused -ol / FG anchor and NOT a hand-rolled carbon-counting scan.

    Returns '' on any failure (caller falls through to legacy).
    """
    # --- 1. Defensive guards --------------------------------------------------
    if mol is None or suffix not in _CUMULATIVE_SUFFIX_H_DELTA:
        return ''
    try:
        center = mol.GetAtomWithIdx(center_idx)
    except (RuntimeError, IndexError, OverflowError):
        return ''
    # This primitive is the CARBON-centred parent-hydride path; heteroatom anions
    # (alkoxide/thiolate/aminide) stay on their existing class paths.
    if center.GetSymbol() != 'C':
        return ''

    # --- 2. Saturate the centre on an INDEX-PRESERVING copy (Pitfall 1) -------
    # Work on an RWMol whose atom indices match ``mol`` exactly — do NOT round-trip
    # through Chem.MolToSmiles/MolFromSmiles, which canonicalizes and reorders.
    work = Chem.RWMol(mol)
    try:
        a = work.GetAtomWithIdx(center_idx)
        a.SetFormalCharge(0)
        a.SetNumRadicalElectrons(0)
        # Re-add (anion/radical: +delta) or remove (cation: -delta) the hydrogens
        # the centre lost/gained relative to the saturated neutral parent hydride.
        h_delta = _CUMULATIVE_SUFFIX_H_DELTA[suffix]
        new_h = a.GetTotalNumHs() + h_delta
        if new_h < 0:
            return ''
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(new_h)
        work_mol = work.GetMol()
        Chem.SanitizeMol(work_mol)
    except (RuntimeError, ValueError):
        return ''

    # --- 3. Find the principal chain on the saturated hydride -----------------
    from ..perception.chains import find_principal_chain, get_substituents, classify_substituent
    from ..perception.functional_groups import detect_functional_groups
    try:
        fgs = detect_functional_groups(work_mol)
        # A bare parent hydride has NO principal characteristic group; the centre
        # is located via the NEW P-72.2.2.1 numbering call below, not an FG.
        chain = find_principal_chain(work_mol, fgs, principal_group=None)
    except (RuntimeError, ValueError, KeyError):
        return ''
    if not chain or center_idx not in chain:
        # Out of scope (e.g. centre is off the principal carbon chain, or a
        # ring carbanion): caller falls through to legacy.
        return ''

    # --- 4. Orient the chain so the centre gets the lowest locant (P-72.2.2.1) -
    from .locants import orient_chain, build_atom_to_locant
    chain_set = set(chain)
    double_bonds: List[tuple] = []
    triple_bonds: List[tuple] = []
    for i in range(len(chain) - 1):
        bond = work_mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
        if bond is None:
            continue
        bt = bond.GetBondType()
        if bt == Chem.BondType.DOUBLE:
            double_bonds.append((chain[i], chain[i + 1]))
        elif bt == Chem.BondType.TRIPLE:
            triple_bonds.append((chain[i], chain[i + 1]))

    # Substituent positions keyed by chain atom index drive orient_chain
    # criterion (d) (lowest locants for substituents) as a tie-break.
    subs_by_position = get_substituents(work_mol, chain)  # 1-based position -> [atoms]
    substituent_positions: Dict[int, list] = {}
    for position, sub_lists in subs_by_position.items():
        substituent_positions[chain[position - 1]] = sub_lists

    oriented = orient_chain(
        chain, work_mol,
        principal_group_atoms={center_idx},   # criterion (a): lowest locant for the -ide centre
        double_bonds=double_bonds,
        triple_bonds=triple_bonds,
        substituent_positions=substituent_positions,
    )
    loc_map = build_atom_to_locant(oriented)
    center_locant = loc_map[center_idx]
    chain_len = len(oriented)

    # --- 5. Assemble the name -------------------------------------------------
    from ..data.chain_names import get_chain_prefix
    from ..assembly.naming_utils import get_multiplier_prefix, alpha_sort_key
    try:
        stem = get_chain_prefix(chain_len)
    except ValueError:
        return ''

    # 5a. Substituent prefixes with their ORIENTED locants. Each substituent is
    # named via the shared classify_substituent (alkyl/ring), grouped by name,
    # given its chain-carbon locant, alphabetized (P-14.5.2) and multiplied
    # (P-16.3.3) — the same machinery the neutral acyclic namer uses.
    # subs_by_position keys are 1-based positions on the ORIGINAL ``chain`` order;
    # map each back to its atom index, then to its oriented locant via loc_map.
    parent_atoms = chain_set
    name_to_locants: Dict[str, list] = {}
    for position, sub_lists in subs_by_position.items():
        chain_idx = chain[position - 1]
        for sub_atoms in sub_lists:
            info = classify_substituent(work_mol, sub_atoms, parent_atoms)
            sub_name = info.get('name')
            if not sub_name:
                return ''  # unnameable substituent -> out of scope, fall through
            name_to_locants.setdefault(sub_name, []).append(loc_map[chain_idx])

    prefix_parts = []  # (sort_key, locant_min, text)
    for sub_name, locants in name_to_locants.items():
        locants_sorted = sorted(locants)
        count = len(locants_sorted)
        mult = get_multiplier_prefix(count, sub_name)
        loc_str = ','.join(str(l) for l in locants_sorted)
        from ..assembly.naming_utils import is_complex_substituent
        if is_complex_substituent(sub_name) and count > 1:
            body = f"{mult}({sub_name})"
        else:
            body = f"{mult}{sub_name}"
        prefix_parts.append((alpha_sort_key(sub_name), locants_sorted[0],
                             f"{loc_str}-{body}"))
    prefix_parts.sort(key=lambda t: (t[0], t[1]))
    # P-14.5.2: detachable prefixes join one another and the parent stem directly
    # (the locant-hyphen is INTERNAL to each prefix block); '3-methyl' + 'butan-2-ide'
    # -> '3-methylbutan-2-ide', NEVER '3-methyl-butan-2-ide'.
    prefix_str = ''.join(p[2] for p in prefix_parts)

    # 5b. Unsaturation endings (en/yn) come BEFORE the cumulative -ide/-yl ending
    # (Table 3.4 cumulative-ending order; RESEARCH (d): but-3-en-2-ide, not
    # 2-id-3-ene). Build the saturated/unsaturated parent stem 'hexan'/'but-3-en'.
    unsat_endings = ''
    if double_bonds or triple_bonds:
        def _bond_locant(pair):
            return min(loc_map[pair[0]], loc_map[pair[1]])
        d_locs = sorted(_bond_locant(b) for b in double_bonds)
        t_locs = sorted(_bond_locant(b) for b in triple_bonds)
        if d_locs:
            dmult = get_multiplier_prefix(len(d_locs), 'ene') or ''
            joiner = 'a' if dmult else ''
            unsat_endings += f"{joiner}-{','.join(map(str, d_locs))}-{dmult}en"
        if t_locs:
            tmult = get_multiplier_prefix(len(t_locs), 'yne') or ''
            joiner = 'a' if tmult else ''
            unsat_endings += f"{joiner}-{','.join(map(str, t_locs))}-{tmult}yn"

    # 5c. Stem + 'an' + (unsaturation) + cumulative-suffix locant + suffix.
    # Single-carbon special case (D-09 methanide): one-atom chain -> NO locant
    # ('methane' -> 'methanide'). Otherwise the centre locant is first-class.
    if chain_len == 1:
        # methane -> methanide / methyl / methylidene ...
        core = f"{stem}an{suffix}"
    elif unsat_endings:
        # 'but-3-en' already carries its own trailing locant on the unsaturation;
        # append '-{center_locant}-{suffix}' (but-3-en-2-ide).
        core = f"{stem}{unsat_endings}-{center_locant}-{suffix}"
    else:
        # 'hexane' -> 'hexan-{locant}-ide'; elide the trailing 'e' of 'ane'.
        core = f"{stem}an-{center_locant}-{suffix}"

    return f"{prefix_str}{core}"


def _ionize_acid_name(neutral_name: str, total_charge: int,
                      allowed_suffixes: Optional[frozenset] = None) -> str:
    """Thin back-compat wrapper over ``apply_ion_suffix_to_name`` (169.6-02).

    Kept so ``_name_oxoacid_anion`` (and any other existing caller) keeps working
    byte-identical. The anion/generic map path is unchanged; this wrapper simply
    forwards to the generalized primitive with no ``cation_class``.

    SUB-01/D-01: single source of truth = the canonical
    ``_ANION_SUFFIX_MAP``/``_CATION_SUFFIX_MAP`` driven through the actual
    ``_apply_anion_modification``/``_apply_cation_modification`` seam — NOT a
    per-molecule ``.replace`` band-aid. ``allowed_suffixes`` (WR-05) restricts the
    trailing match to the caller's known chemical class.
    """
    return apply_ion_suffix_to_name(neutral_name, total_charge,
                                    allowed_suffixes=allowed_suffixes)


# WR-01 (code review 2026-06-02): a leading unsaturation marker (ane/ene/yne)
# with NO chain/ring stem in front, glued directly to an oxoacid suffix stem
# (sulf/phosph/arso/boro) or a locant ('-'/digit/'('). This is the degenerate
# fused-ring-parent failure shape (D-10), NOT a legitimate parent — a real
# parent always carries a stem (meth/eth/.../benzene/cyclo...) before ane/ene/yne.
_DEGENERATE_OXOACID_PARENT_RE = re.compile(r'^(?:ane|ene|yne)(?:sulf|phosph|arso|boro|[0-9(-])')


def _name_inorganic_oxoacid_anion(mol) -> str:
    """WS-E.3-dianion (D-13): name a BARE (carbon-free) inorganic oxoacid anion as
    its preselected functional-parent anion word (P-12.2 / P-34.1.4):
    sulfate/sulfite/phosphate/hydrogenphosphate/nitrate/carbonate. There is no
    carbon to name, so this is a DIRECT functional-parent lookup, NOT a
    substitutive parent (BlueBookV2.md:2076 phosphoric acid; 35449 sulfuric acid).

    CONSERVATIVE GUARD (D-13 risk): fires ONLY on a CARBON-FREE, non-metal fragment
    matched by EXACT canonical SMILES in INORGANIC_ANIONS. A sulfate/phosphate
    ESTER (R-O-SO3-, R-O-PO3-) has carbon -> declined here (and at
    charged_router.py:382-389), so the steroid-sulfate NP-conjugate (gold WSC-03
    'cholest-5-en-3beta-yl sulfate', '...yl hydrogen sulfate') is NEVER intercepted.
    The mono-anion (OS(=O)(=O)[O-] -> 'hydrogensulfate') stays DISTINCT from the
    dianion ('sulfate') because the table keys on protonation state
    (BlueBookV2.md:7150). Returns '' if not a bare inorganic anion (caller falls
    through to the existing cascade -- byte-identical contract).
    """
    if mol is None or _has_metal(mol):
        return ''
    # Carbon-free guard: any carbon -> organic anion / ester / conjugate,
    # NOT a bare inorganic oxoacid anion. Decline so the existing paths handle it.
    if any(atom.GetSymbol() == 'C' for atom in mol.GetAtoms()):
        return ''
    canonical = Chem.MolToSmiles(mol, canonical=True)
    return get_anion_name(canonical) or ''


def _name_oxoacid_anion(mol, style: str) -> str:
    """SUB-01/D-01: name an S/P-oxoacid anion (sulfonate/sulfinate/phosphonate/
    phosphate) via the GENERAL parent-selection pipeline (the chokepoint) plus
    the structured ionic suffix.

    Mechanism (the C1 anchor — ROUTING, not string surgery): neutralize the
    [O-] site(s), re-enter ``Orthonym(style).name(neutral_smi)`` (which runs
    ``_classify`` -> ``select_parent``, naming e.g. ``propane-1-sulfonic acid``),
    then apply the canonical ionic suffix via ``_ionize_acid_name``
    (``sulfonic acid`` -> ``sulfonate``). Returns '' on any failure (caller
    falls through to the existing cascade — v18 byte-identical contract).
    """
    if mol is None or _has_metal(mol):
        return ''
    try:
        from ..perception.ions import _get_internal_charge_atoms
        internal = _get_internal_charge_atoms(mol)
        rw = Chem.RWMol(mol)
        neutralized = False
        for atom in rw.GetAtoms():
            if (atom.GetSymbol() == 'O' and atom.GetFormalCharge() == -1
                    and atom.GetIdx() not in internal):
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
                neutralized = True
        if not neutralized:
            return ''
        Chem.SanitizeMol(rw)
        neutral_smi = Chem.MolToSmiles(rw.GetMol(), canonical=True)
        if not neutral_smi:
            return ''
        from ..namer import Orthonym
        # _disable_opsin_validity_gate: the neutral acid name is an INTERMEDIATE
        # (ionized below), not a final output — it must NOT be SUB-03-gated, else
        # a malformed fused-ring parent ('anesulfonic acid') would be suppressed
        # to a descriptive string and break the ionize step (169.5 SUB-01/SUB-03
        # interaction).
        neutral_name = Orthonym(style=style, _disable_opsin_validity_gate=True).name(neutral_smi)
        if not neutral_name:
            return ''
        # Robustness: refuse to propagate a MALFORMED upstream parent name.
        # A failed fused-ring parent (the deferred P-25 limitation, D-10) drops
        # the chain/ring stem, leaving the bare unsaturation marker glued onto
        # the oxoacid suffix or a locant — "anesulfonic acid", "ene-1-phosphonic
        # acid". WR-01 (code review 2026-06-02): anchor on EXACTLY that shape (a
        # leading ane/ene/yne immediately followed by an oxoacid-suffix stem or a
        # locant) instead of a bare startswith(("ane","ene","yne")), which would
        # also discard any legitimate parent merely beginning with those three
        # letters. A descriptive fallback is likewise not a valid parent. Fall
        # through to the existing path rather than emit garbage; SUB-03 gates
        # residual malformed outputs globally.
        low = neutral_name.lstrip().lower()
        if (_DEGENERATE_OXOACID_PARENT_RE.match(low)
                or 'unknown' in low or 'not supported' in low):
            return ''
        # WR-05: restrict the suffix match to genuine oxoacid suffixes — this
        # path only ever sees a re-entered S/P oxoacid parent, so the bare
        # "ol"/"amine"/"thiol" entries must not be eligible to mis-fire.
        return _ionize_acid_name(neutral_name, total_charge=-1,
                                 allowed_suffixes=_OXOACID_NEUTRAL_SUFFIXES)
    except (RecursionError, ValueError, RuntimeError):
        return ''


# === SYSTEMATIC NAMING HELPERS ===

def _find_carboxyl_carbon(mol, anion_site: Dict) -> Optional[int]:
    """
    Find the carbon atom of the carboxyl group from the anionic oxygen.

    Args:
        mol: RDKit Mol object
        anion_site: Dictionary containing 'atom_idx' of the anionic oxygen

    Returns:
        Index of the carboxyl carbon, or None if not found
    """
    o_idx = anion_site['atom_idx']
    o_atom = mol.GetAtomWithIdx(o_idx)

    # Find the carbon attached to the anionic oxygen
    for neighbor in o_atom.GetNeighbors():
        if neighbor.GetSymbol() == 'C':
            # Verify this is a carboxyl carbon (has C=O double bond to another oxygen)
            for second_neighbor in neighbor.GetNeighbors():
                if second_neighbor.GetIdx() != o_idx and second_neighbor.GetSymbol() == 'O':
                    bond = mol.GetBondBetweenAtoms(neighbor.GetIdx(), second_neighbor.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                        return neighbor.GetIdx()
    return None


def _detect_aromatic_carboxylate(mol, carboxyl_carbon_idx: int) -> Optional[str]:
    """
    Detect if a carboxylate is attached to an aromatic ring.

    Args:
        mol: RDKit Mol object
        carboxyl_carbon_idx: Index of the carboxyl carbon

    Returns:
        'benzoate' for benzene-attached carboxylates,
        'naphthoate' for naphthalene-attached carboxylates,
        None for acyclic or other structures
    """
    carboxyl_carbon = mol.GetAtomWithIdx(carboxyl_carbon_idx)

    # Check neighbors for aromatic carbon
    for neighbor in carboxyl_carbon.GetNeighbors():
        if neighbor.GetSymbol() == 'C' and neighbor.GetIsAromatic():
            # Found aromatic carbon attached to carboxyl
            # Determine ring type
            ri = mol.GetRingInfo()
            atom_rings = ri.AtomRings()

            aromatic_neighbor_idx = neighbor.GetIdx()

            # Find which ring(s) contain this aromatic carbon
            for ring in atom_rings:
                if aromatic_neighbor_idx in ring:
                    ring_size = len(ring)
                    # Check if all atoms in ring are aromatic carbons (carbocycle)
                    all_aromatic_c = all(
                        mol.GetAtomWithIdx(idx).GetSymbol() == 'C' and
                        mol.GetAtomWithIdx(idx).GetIsAromatic()
                        for idx in ring
                    )

                    if ring_size == 6 and all_aromatic_c:
                        # Check if this is part of naphthalene (fused bicyclic)
                        if _is_naphthalene_system(mol, aromatic_neighbor_idx, atom_rings):
                            return 'naphthoate'
                        return 'benzoate'

    return None


def _is_naphthalene_system(mol, aromatic_idx: int, atom_rings) -> bool:
    """
    Check if an aromatic atom is part of a naphthalene (fused bicyclic) system.

    Args:
        mol: RDKit Mol object
        aromatic_idx: Index of an aromatic atom
        atom_rings: Ring information from RDKit

    Returns:
        True if part of a naphthalene system
    """
    # Find all 6-membered aromatic carbocyclic rings
    aromatic_6_rings = []
    for ring in atom_rings:
        if len(ring) == 6:
            all_aromatic_c = all(
                mol.GetAtomWithIdx(idx).GetSymbol() == 'C' and
                mol.GetAtomWithIdx(idx).GetIsAromatic()
                for idx in ring
            )
            if all_aromatic_c:
                aromatic_6_rings.append(set(ring))

    # Check if there are 2 fused 6-membered rings (sharing 2 atoms = naphthalene)
    if len(aromatic_6_rings) >= 2:
        for i, ring1 in enumerate(aromatic_6_rings):
            for ring2 in aromatic_6_rings[i+1:]:
                shared = ring1 & ring2
                if len(shared) == 2:  # Two shared atoms = fused rings
                    # Check if our aromatic atom is in either ring
                    if aromatic_idx in ring1 or aromatic_idx in ring2:
                        return True
    return False


def _name_aromatic_carboxylate_with_substituents(mol, carboxyl_carbon_idx: int, base_name: str) -> str:
    """
    Name an aromatic carboxylate with any substituents on the ring.

    Args:
        mol: RDKit Mol object
        carboxyl_carbon_idx: Index of the carboxyl carbon
        base_name: Base name ('benzoate' or 'naphthoate')

    Returns:
        Full name with substituent prefixes (e.g., '4-chlorobenzoate')
    """
    carboxyl_carbon = mol.GetAtomWithIdx(carboxyl_carbon_idx)

    # Find the aromatic ring attached to carboxyl
    aromatic_ring_atom = None
    for neighbor in carboxyl_carbon.GetNeighbors():
        if neighbor.GetSymbol() == 'C' and neighbor.GetIsAromatic():
            aromatic_ring_atom = neighbor
            break

    if aromatic_ring_atom is None:
        return base_name

    # Get the benzene ring
    ri = mol.GetRingInfo()
    benzene_ring = None
    for ring in ri.AtomRings():
        if aromatic_ring_atom.GetIdx() in ring and len(ring) == 6:
            # Check all atoms are aromatic carbons
            all_aromatic_c = all(
                mol.GetAtomWithIdx(idx).GetSymbol() == 'C' and
                mol.GetAtomWithIdx(idx).GetIsAromatic()
                for idx in ring
            )
            if all_aromatic_c:
                benzene_ring = ring
                break

    if benzene_ring is None:
        return base_name

    # Find substituents on the ring (excluding the carboxyl attachment point)
    ring_set = set(benzene_ring)
    carboxyl_attachment_idx = aromatic_ring_atom.GetIdx()

    # Map substituent type to name
    SUBSTITUENT_NAMES = {
        'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo',
        'N': 'amino', 'O': 'hydroxy'
    }

    # Collect substituents: {position: [(name, sort_key), ...]}
    # Position 1 is the carboxyl attachment point
    substituents_by_position = {}

    # Orient ring: carboxyl attachment is position 1
    # Need to find ring order and direction for lowest locants
    ring_list = list(benzene_ring)

    # Find index of carboxyl attachment in ring
    carboxyl_pos = ring_list.index(carboxyl_attachment_idx)

    # Try both directions and all starting positions to get lowest locants
    best_orientation = None
    best_locants = None

    for direction in [1, -1]:
        oriented = []
        for i in range(6):
            idx = (carboxyl_pos + i * direction) % 6
            oriented.append(ring_list[idx])

        # Collect substituents for this orientation
        subs = {}
        for pos, atom_idx in enumerate(oriented, start=1):
            if pos == 1:
                continue  # Skip carboxyl attachment position

            ring_atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in ring_atom.GetNeighbors():
                nbr_idx = neighbor.GetIdx()
                if nbr_idx in ring_set:
                    continue  # Skip ring atoms
                if nbr_idx == carboxyl_carbon_idx:
                    continue  # Skip carboxyl carbon

                # Identify substituent
                symbol = neighbor.GetSymbol()
                sub_name = None
                # Nitro group: N+(=O)[O-] — must check before simple N→amino
                if (symbol == 'N' and neighbor.GetFormalCharge() == 1
                        and sum(1 for n in neighbor.GetNeighbors()
                                if n.GetSymbol() == 'O') >= 2):
                    sub_name = 'nitro'
                elif symbol in SUBSTITUENT_NAMES:
                    sub_name = SUBSTITUENT_NAMES[symbol]
                if sub_name is not None:
                    if pos not in subs:
                        subs[pos] = []
                    subs[pos].append((sub_name, sub_name))  # (name, sort_key)
                elif symbol == 'C' and not neighbor.GetIsAromatic():
                    # Alkyl group - count carbons
                    carbon_count = _count_alkyl_carbons(mol, nbr_idx, ring_set | {carboxyl_carbon_idx})
                    try:
                        sub_name = get_alkyl_name(carbon_count)
                    except (ValueError, KeyError):
                        sub_name = None
                    if sub_name:
                        if pos not in subs:
                            subs[pos] = []
                        subs[pos].append((sub_name, sub_name))

        # Calculate locant set for this orientation
        locants = sorted(subs.keys()) if subs else []

        if best_locants is None or _compare_locant_lists(locants, best_locants) < 0:
            best_locants = locants
            best_orientation = subs

    if not best_orientation:
        return base_name

    # Group substituents by name
    from collections import defaultdict
    grouped = defaultdict(list)
    for pos, sub_list in best_orientation.items():
        for sub_name, _ in sub_list:
            grouped[sub_name].append(pos)

    # Sort locants within each group
    for name in grouped:
        grouped[name].sort()

    # Build prefix: alphabetically sorted, with multipliers
    MULTIPLIERS = {1: '', 2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa'}

    prefixes = []
    for sub_name in sorted(grouped.keys()):
        locants = grouped[sub_name]
        count = len(locants)
        multiplier = MULTIPLIERS.get(count, f'{count}-')

        locant_str = ','.join(str(loc) for loc in locants)
        prefix = f"{locant_str}-{multiplier}{sub_name}"
        prefixes.append(prefix)

    if not prefixes:
        return base_name

    prefix_str = ''.join(prefixes)
    return f"{prefix_str}{base_name}"


def _compare_locant_lists(a: list, b: list) -> int:
    """Compare two locant lists by first-point-of-difference (-1/0/1).

    IN-04 (code review 2026-06-02): delegates to the canonical
    ``rules.locants.compare_locant_sets`` so the first-point-of-difference
    semantics have ONE source and cannot desync. The sole caller passes
    ``sorted(subs.keys())`` for both lists (always sorted, and same length
    across ring orientations), for which the two are provably equivalent
    (verified exhaustively over small sorted equal-length int lists).
    """
    from .locants import compare_locant_sets
    return compare_locant_sets(a, b)


def _name_carboxylate_systematic(mol, anion_site: Dict) -> str:
    """Generate systematic name for carboxylate anion.

    Strategy: Neutralize the carboxylate ([O-] -> OH) to form the parent
    carboxylic acid, name it with the full naming pipeline (which handles
    substituents, stereo, unsaturation), then convert '-oic acid' to '-oate'.
    This ensures all substituents are properly detected and included.
    """
    # NEW: Check for aromatic parent FIRST
    carboxyl_carbon = _find_carboxyl_carbon(mol, anion_site)
    if carboxyl_carbon is not None:
        aromatic_name = _detect_aromatic_carboxylate(mol, carboxyl_carbon)
        if aromatic_name:
            return _name_aromatic_carboxylate_with_substituents(mol, carboxyl_carbon, aromatic_name)

    # Try neutralize-then-name approach for full substituent detection
    acid_name = _neutralize_carboxylate_to_acid(mol, anion_site)
    if acid_name:
        oate_name = _acid_to_oate(acid_name)
        if oate_name:
            return oate_name

    # Fallback: simple chain naming (no substituent detection)
    chain, double_bond_locs = _find_carboxylate_chain(mol, anion_site)
    carbon_count = len(chain) if chain else sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C')

    if carbon_count == 1:
        return 'formate'
    elif carbon_count == 2 and not double_bond_locs:
        return 'acetate'

    from ..data.chain_names import get_chain_prefix
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
    prefix = get_chain_prefix(carbon_count)

    if double_bond_locs:
        n_double = len(double_bond_locs)
        loc_str = ",".join(str(loc) for loc in sorted(double_bond_locs))
        if n_double == 1:
            return f"{prefix}-{loc_str}-enoate"
        else:
            mult = SIMPLE_MULTIPLIERS.get(n_double, str(n_double))
            return f"{prefix}a-{loc_str}-{mult}enoate"
    else:
        return prefix + "anoate"


def _neutralize_carboxylate_to_acid(mol, anion_site: Dict) -> str:
    """Neutralize carboxylate [O-] to OH and name as carboxylic acid.

    Returns the acid name or empty string on failure.
    """
    try:
        rw = Chem.RWMol(mol)
        o_idx = anion_site['atom_idx']
        o_atom = rw.GetAtomWithIdx(o_idx)
        o_atom.SetFormalCharge(0)
        o_atom.SetNumExplicitHs(o_atom.GetNumExplicitHs() + 1)

        try:
            Chem.SanitizeMol(rw)
        except Exception:
            return ''

        neutral_smiles = Chem.MolToSmiles(rw, canonical=True)
        if not neutral_smiles:
            return ''

        from ..assembly.fragment_naming import name_fragment_recursively
        acid_name = name_fragment_recursively(neutral_smiles)
        if acid_name and ('oic acid' in acid_name or 'ic acid' in acid_name):
            return acid_name
    except (RecursionError, ValueError, RuntimeError):
        pass
    return ''


def _acid_to_oate(acid_name: str) -> str:
    """Convert a carboxylic acid name to its carboxylate (-oate) form.

    Handles:
    - 'X-oic acid' -> 'X-oate'
    - 'Xanoic acid' -> 'Xanoate'
    - Retained names: 'acetic acid' -> 'acetate', 'formic acid' -> 'formate'
    """
    if not acid_name:
        return ''

    # Retained acid -> retained oate
    retained_map = {
        'formic acid': 'formate',
        'acetic acid': 'acetate',
        'propionic acid': 'propanoate',
        'butyric acid': 'butanoate',
        'valeric acid': 'pentanoate',
        'isovaleric acid': '3-methylbutanoate',
    }
    if acid_name in retained_map:
        return retained_map[acid_name]

    # Standard conversion: '-oic acid' -> '-oate'
    if acid_name.endswith('oic acid'):
        return acid_name[:-len('oic acid')] + 'oate'
    # '-ic acid' (benzoic acid -> benzoate)
    if acid_name.endswith('ic acid'):
        return acid_name[:-len('ic acid')] + 'ate'
    return ''


def _acid_name_to_carboxylate(acid_name: str, carboxylate_count: int) -> str:
    """Convert acid name to carboxylate form for deprotonated carboxylates.

    Handles mono- and poly-carboxylates:
    - 'pentanedioic acid' (2 COO-) -> 'pentanedioate'
    - 'hexanedioic acid' (2 COO-) -> 'hexanedioate'
    - 'propanoic acid' (1 COO-) -> 'propanoate'
    - 'glutamic acid' -> 'glutamate' (retained)

    IUPAC P-72.2.1: Replace '-ic acid' with '-ate' for full deprotonation.
    """
    if not acid_name:
        return ''

    # Use _acid_to_oate for simple cases (includes retained name mapping)
    result = _acid_to_oate(acid_name)
    if result:
        return result

    # Handle 'carboxylic acid' suffix (polyfunctional: -dicarboxylic acid etc.)
    if acid_name.endswith('carboxylic acid'):
        return acid_name[:-len('carboxylic acid')] + 'carboxylate'

    # Handle retained diacid names ending in 'ic acid'
    # e.g., "glutamic acid" -> "glutamate", "succinic acid" -> "succinate"
    if acid_name.endswith('ic acid'):
        return acid_name[:-len('ic acid')] + 'ate'

    return ''


def _find_carboxylate_chain(mol, anion_site: Dict):
    """Find the longest carbon chain from the carboxylate group.

    Returns (chain_atoms, double_bond_locants) where chain_atoms is a list
    of atom indices starting from the carboxylate carbon, and double_bond_locants
    is a list of IUPAC locants for C=C double bonds along the chain.
    """
    anion_idx = anion_site['atom_idx']
    atom = mol.GetAtomWithIdx(anion_idx)

    # Find the carboxyl carbon (C attached to charged O)
    carboxyl_c = None
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() == 'C':
            carboxyl_c = nbr.GetIdx()
            break
    if carboxyl_c is None:
        return None, []

    # BFS to find longest carbon chain from carboxyl C
    # Exclude the carboxylate oxygens from traversal
    carboxylate_os = set()
    for nbr in mol.GetAtomWithIdx(carboxyl_c).GetNeighbors():
        if nbr.GetSymbol() == 'O':
            carboxylate_os.add(nbr.GetIdx())

    def _longest_chain(start_idx, visited):
        """DFS to find longest carbon chain."""
        best = [start_idx]
        a = mol.GetAtomWithIdx(start_idx)
        for nbr in a.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in visited or nidx in carboxylate_os:
                continue
            if nbr.GetSymbol() != 'C':
                continue
            visited.add(nidx)
            sub = _longest_chain(nidx, visited)
            candidate = [start_idx] + sub
            if len(candidate) > len(best):
                best = candidate
            visited.discard(nidx)
        return best

    chain = _longest_chain(carboxyl_c, {carboxyl_c})

    # Find double bonds along the chain and their locants
    double_bond_locs = []
    for i in range(len(chain) - 1):
        bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
        if bond and bond.GetBondTypeAsDouble() == 2.0:
            double_bond_locs.append(i + 1)  # 1-indexed: C1 is carboxylate carbon

    return chain, double_bond_locs


# 169.6-03 (CHOKE-01, kill-list §2.1): the four carbon-counting ANION naming
# stubs (alkoxide = the canonical `heptanolate` bug; phenolate = hardcoded
# `return 'phenolate'`; carbanion; thiolate) were DELETED. They counted carbons
# -> prefix+suffix and dropped connectivity / substituents / unsaturation /
# locants. Their single caller (name_anion's single-anion dispatch) now delegates
# to route_charged, which neutralizes -> re-enters the full pipeline -> re-applies
# the class-correct ionic suffix (so a branched alkoxide names the FULL
# substituted -olate; the heptanolate bug class is gone — not masked).
# fix-methodology.md: the stubs ARE the band-aids; deleting them IS the fix.


def _name_aminium_systematic(mol, cation_site: Dict) -> str:
    """Generate systematic name for aminium cation.

    IUPAC P-73.1.2: Protonated amines use '-aminium' suffix.
    - NH4+ -> 'ammonium' (retained)
    - R-NH3+ -> neutralize to R-NH2, name as amine, convert amine -> aminium
    - R2NH2+ -> neutralize, name, convert
    - R3NH+ -> neutralize, name, convert
    - R4N+ -> quaternary, use azanium naming
    """
    atom = mol.GetAtomWithIdx(cation_site['atom_idx'])
    carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']

    if len(carbon_neighbors) == 0:
        # NH4+ -> ammonium
        return 'ammonium'

    # Strategy: neutralize N+ to N, name the neutral amine, then convert
    # amine -> aminium (IUPAC P-73.1.2)
    try:
        rw = Chem.RWMol(mol)
        n_atom = rw.GetAtomWithIdx(cation_site['atom_idx'])
        n_atom.SetFormalCharge(0)
        # Remove one H (deprotonation)
        cur_h = n_atom.GetNumExplicitHs()
        if cur_h > 0:
            n_atom.SetNumExplicitHs(cur_h - 1)
        else:
            # Implicit H - let sanitization handle
            pass
        try:
            Chem.SanitizeMol(rw)
        except Exception:
            pass

        neutral_smiles = Chem.MolToSmiles(rw, canonical=True)
        if neutral_smiles:
            from ..assembly.fragment_naming import name_fragment_recursively
            amine_name = name_fragment_recursively(neutral_smiles)
            if amine_name:
                # Convert amine suffix to aminium
                aminium_name = name_aminium_cation(amine_name)
                if aminium_name:
                    return aminium_name
    except (RecursionError, ValueError, RuntimeError):
        pass

    # 169.6-03 (CHOKE-01, kill-list §2.1): the carbon-counting FALLBACK ("old
    # approach using simple alkyl counting") was DELETED. Only the neutralize ->
    # name_fragment_recursively -> name_aminium_cation PRIMARY path above is kept
    # (it handles substituents/rings correctly: methylamine->methylaminium,
    # pyrrolidine->pyrrolidineium). On failure return '' so the caller falls
    # through to the legacy cascade (v18 byte-identical no-crash contract) —
    # NOT a carbon-counted aminium misname.
    return ''


def name_quaternary_aminium(mol, cation_site: Dict) -> str:
    """Systematic ``-aminium`` PIN for a STANDALONE quaternary ammonium cation.

    Phase 184 WS-E.1 (P-73.1.2.1 method (1) + Table 7.4; BlueBookV2:41354,
    41429-41438): a quaternary N (formal charge +1, 0 H, degree >= 4) is named by
    treating the SENIOR carbon chain through N as the amine parent (``-amine``
    suffix), the other three N-branches as ``N,N,N-``/``N,N-``/``N-`` substituent
    prefixes, then appending ``-ium`` (``methanamine`` -> ``methanaminium``):

      C[N+](C)(C)C   -> 'N,N,N-trimethylmethanaminium'
      OCC[N+](C)(C)C -> '2-hydroxy-N,N,N-trimethylethan-1-aminium'
      CC[N+](C)(C)CC -> 'N-ethyl-N,N-dimethylethanaminium'

    Mechanism (D-05): the quaternary N CANNOT be neutralized — removing a
    (non-existent) proton leaves an over-valent neutral N and SanitizeMol raises
    (RESEARCH Pitfall 2). Instead the N is DEMOTED to a neutral carbon-context
    amine parent: a ``tertiary_amine`` functional-group match is injected on the
    ORIGINAL (still-charged) mol and forced as the principal group, so
    ``find_principal_chain`` selects the senior chain THROUGH N as the parent and
    the remaining N-branches become N-locant prefixes (NOT ``select_parent``, NOT
    the ``azaniumyl`` prefix). The well-formed neutral amine name is then converted
    to ``-aminium`` via ``name_aminium_cation`` (the existing amine->aminium text
    transform). NO postprocessor, NO regex band-aid, NO per-molecule .replace.

    Returns '' (fail-closed) when the cation is NOT a quaternary N or any step
    fails, so callers fall through to the proven aminium / legacy cascade
    (the v18 no-crash byte-identical contract).
    """
    if mol is None:
        return ''
    try:
        atom = mol.GetAtomWithIdx(cation_site['atom_idx'])
    except (RuntimeError, IndexError, KeyError, OverflowError):
        return ''
    # Guard: only a genuine quaternary N (degree-4, 0-H, +1) — otherwise return ''
    # so the caller uses the proven protonated-amine path.
    if not (atom.GetSymbol() == 'N'
            and atom.GetFormalCharge() == 1
            and atom.GetTotalNumHs() == 0
            and atom.GetDegree() >= 4):
        return ''
    n_idx = cation_site['atom_idx']
    carbon_neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()
                        if nb.GetSymbol() == 'C']
    # A quaternary AMMONIUM PIN needs >= one carbon branch to form an amine parent
    # (a heteroatom-only quaternary N is out of scope for this emitter).
    if not carbon_neighbors:
        return ''

    try:
        from ..namer import Orthonym
        from ..assembly.composer import _assemble_amine_name

        # Build features on the ORIGINAL (charged) mol so the N-substituent
        # enumeration sees all branches; inject a tertiary_amine FG so the amine N
        # becomes the principal group and find_principal_chain orients the parent
        # chain with the N-bearing carbon as C1 (any senior neutral group such as
        # an -OH is demoted to a detachable prefix per P-73.1.2.1).
        namer = Orthonym(
            style='pin', _disable_opsin_validity_gate=True,
            _principal_group_override='tertiary_amine',
        )
        canonical = Chem.MolToSmiles(mol, canonical=True)
        features = namer._perceive(mol, canonical, canonical)
        features.functional_groups = dict(features.functional_groups)
        features.functional_groups['tertiary_amine'] = [
            tuple([n_idx] + carbon_neighbors)
        ]
        # Drop the cationic 'ammonium' FG so the override resolves cleanly to the
        # injected amine.
        features.functional_groups.pop('ammonium', None)
        namer._classify(features)

        # Re-point the principal-group N-substituent set to EXCLUDE the carbon that
        # find_principal_chain chose as the parent chain anchor, so the N-prefix
        # multiplier counts exactly the off-chain branches (P-73.1.2.1 N-locants).
        chain_set = set(features.principal_chain or [])
        nsub_carbons = [c for c in carbon_neighbors if c not in chain_set]
        features.principal_group_atoms = [tuple([n_idx] + nsub_carbons)]

        amine_name = _assemble_amine_name(features, 'pin')
        if not amine_name:
            return ''
        # amine -> aminium (P-73.1.2.1): methanamine -> methanaminium,
        # ethan-1-amine -> ethan-1-aminium.
        aminium = name_aminium_cation(amine_name)
        if not aminium or 'aminium' not in aminium:
            return ''
        return aminium
    except (RecursionError, ValueError, RuntimeError, KeyError, AttributeError):
        return ''


# 169.6-03 (CHOKE-01, kill-list §2.1): the three carbon-counting CATION naming
# stubs (carbenium = carbon-count -> ylium; onium = carbon-count + hardcoded
# 'tri'; diazonium = literal 'benzenediazonium') were DELETED. Their single
# caller (name_cation's single-cation dispatch) now delegates to route_charged,
# which neutralizes -> re-enters -> re-applies the class-keyed cation suffix
# (carbenium ane->ylium via the parent hydride; diazonium append;
# P-73.2.2.1.1/.3). fix-methodology.md: the stubs ARE the band-aids; deletion IS
# the fix.


def _count_alkyl_carbons(mol, start_idx: int, exclude: set) -> int:
    """Count carbon atoms in an alkyl group via BFS."""
    from collections import deque

    visited = set()
    queue = deque([start_idx])
    count = 0

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'C':
            count += 1
            for neighbor in atom.GetNeighbors():
                nbr_idx = neighbor.GetIdx()
                if nbr_idx not in visited and nbr_idx not in exclude:
                    if neighbor.GetSymbol() == 'C':
                        queue.append(nbr_idx)

    return count
