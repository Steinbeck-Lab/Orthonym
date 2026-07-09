"""
Radical naming rules per IUPAC 2013 P-71.

Handles naming of:
- Monovalent radicals: -yl suffix (methyl, ethyl, phenyl)
- Divalent radicals: -ylidene suffix (methylidene, ethylidene)
- Trivalent radicals: -ylidyne suffix (methylidyne)
- Acyl radicals: -oyl suffix (acetyl, benzoyl)
- Oxyl radicals: -oxyl suffix (methoxyl, phenoxyl)

IUPAC 2013 References:
- P-71: Radical nomenclature

Key naming patterns:
- Monovalent alkyl radicals: alkane - H -> alkyl (methyl, ethyl)
- Divalent alkylidene radicals: methane -> methylidene
- Trivalent alkylidyne radicals: methane -> methylidyne
- Acyl radicals: acid - OH -> -oyl (acetyl from acetic)
- Oxyl radicals: R-O. -> R-oxyl (methoxyl from methanol)
"""

from typing import Any, Dict, List, Optional
from rdkit import Chem

from ..perception.ions import get_radical_sites
from ..assembly.naming_utils import ALKYL_NAMES


# === SUFFIX MAPPINGS ===

RADICAL_SUFFIXES = {
    1: 'yl',        # Monovalent: methyl, ethyl
    2: 'ylidene',   # Divalent: methylidene, ethylidene
    3: 'ylidyne',   # Trivalent: methylidyne
}

# Radical type names
RADICAL_TYPE_NAMES = {
    1: 'monovalent',
    2: 'divalent',
    3: 'trivalent',
}


# === CHAIN PREFIXES ===
# Delegated to centralized chain_names module
from ..data.chain_names import get_chain_prefix as _get_chain_prefix

# Retained radical names (canonical SMILES -> name)
# These are looked up first before systematic naming
RETAINED_RADICALS = {
    # Monovalent alkyl radicals
    '[CH3]': 'methyl',
    'C[CH2]': 'ethyl',
    'CC[CH2]': 'propyl',
    'CCC[CH2]': 'butyl',

    # Divalent radicals
    '[CH2]': 'methylidene',
    'C[CH]': 'ethylidene',

    # Trivalent radicals
    '[CH]': 'methylidyne',

    # Aryl radicals
    '[c]1ccccc1': 'phenyl',
    'c1ccc([CH2])cc1': 'benzyl',

    # Acyl radicals (carbonyl radicals)
    '[CH]=O': 'formyl',
    'C[C]=O': 'acetyl',
    'CC[C]=O': 'propanoyl',
    'c1ccc([C]=O)cc1': 'benzoyl',

    # Oxyl radicals (oxygen-centered)
    '[O]C': 'methoxyl',
    '[O]CC': 'ethoxyl',
    '[O]c1ccccc1': 'phenoxyl',
}


# === RADICAL CLASSIFICATION ===

def classify_radical(mol, radical_site: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify radical type based on the radical center environment.

    Examines the local chemical environment of the radical atom to
    determine the appropriate naming approach.

    Args:
        mol: RDKit Mol object
        radical_site: Dictionary from get_radical_sites containing:
            - atom_idx: int
            - n_electrons: int (1, 2, or 3)
            - element: str
            - radical_type: str ('monovalent', 'divalent', 'trivalent')
            - hybridization: str

    Returns:
        Dictionary with:
        - 'n_electrons': 1, 2, or 3
        - 'radical_type': 'monovalent', 'divalent', 'trivalent'
        - 'subtype': 'alkyl', 'acyl', 'oxyl', 'aryl', or 'generic'

    Example:
        >>> mol = Chem.MolFromSmiles('[CH3]')
        >>> sites = get_radical_sites(mol)
        >>> classify_radical(mol, sites[0])
        {'n_electrons': 1, 'radical_type': 'monovalent', 'subtype': 'alkyl'}
    """
    n_electrons = radical_site['n_electrons']
    element = radical_site['element']
    atom_idx = radical_site['atom_idx']
    radical_type = radical_site['radical_type']

    atom = mol.GetAtomWithIdx(atom_idx)

    result = {
        'n_electrons': n_electrons,
        'radical_type': radical_type,
        'subtype': 'generic',
    }

    # Classify based on element
    if element == 'O':
        # Oxygen-centered radical
        # Check if attached to carbon (oxyl radical: R-O.)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C':
                result['subtype'] = 'oxyl'
                return result
        result['subtype'] = 'oxyl'

    elif element == 'C':
        # Carbon-centered radical
        # Check for acyl radical (C. attached to C=O)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'O':
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    result['subtype'] = 'acyl'
                    return result

        # Check if aromatic (aryl radical)
        if atom.GetIsAromatic():
            result['subtype'] = 'aryl'
            return result

        # Check if attached to aromatic ring (benzylic)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIsAromatic():
                result['subtype'] = 'benzylic'
                return result

        # Default to alkyl
        result['subtype'] = 'alkyl'

    elif element == 'N':
        # Nitrogen-centered radical (aminyl)
        result['subtype'] = 'aminyl'

    elif element == 'S':
        # Sulfur-centered radical (thiyl)
        result['subtype'] = 'thiyl'

    return result


# === SUFFIX HELPERS ===

def get_radical_suffix(n_electrons: int) -> str:
    """
    Return suffix based on radical electron count.

    Args:
        n_electrons: Number of unpaired electrons (1, 2, or 3)

    Returns:
        Suffix string: 'yl' for 1, 'ylidene' for 2, 'ylidyne' for 3

    Example:
        >>> get_radical_suffix(1)
        'yl'
        >>> get_radical_suffix(2)
        'ylidene'
        >>> get_radical_suffix(3)
        'ylidyne'
    """
    return RADICAL_SUFFIXES.get(n_electrons, 'yl')


def _count_chain_carbons(mol, start_idx: int, exclude: set) -> int:
    """Count carbon atoms in a chain via BFS from a starting point."""
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
                    if neighbor.GetSymbol() in ('C', 'H'):
                        queue.append(nbr_idx)

    return count


def _is_carboxyl_carbon(mol, idx: int, from_idx: int) -> bool:
    """True if atom `idx` is a carboxylic-acid carbon -C(=O)OH reached from
    `from_idx` (a chain carbon) — i.e. C bonded to =O and -OH and nothing else
    heavy but `from_idx`."""
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != 'C' or atom.GetIsAromatic():
        return False
    dbl_o = single_oh = 0
    other_heavy = 0
    for nbr in atom.GetNeighbors():
        n_idx = nbr.GetIdx()
        if n_idx == from_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, n_idx)
        if nbr.GetSymbol() == 'O':
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                dbl_o += 1
            elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetTotalNumHs() >= 1:
                single_oh += 1
            else:
                other_heavy += 1
        else:
            other_heavy += 1
    return dbl_o == 1 and single_oh == 1 and other_heavy == 0


def _name_carboxy_alkyl_radical(mol, radical_idx: int) -> Optional[str]:
    """P-41 Table 4.1 cls 1: the radical (free valence) is the MOST senior
    class, senior to a carboxylic acid. A monovalent alkyl radical whose linear
    carbon chain (starting at the free valence = C-1) terminates in a
    carboxylic-acid carbon names the chain as the -yl parent and cites the acid
    as a 'carboxy' prefix, e.g. ``HOOC-CH2-CH2.`` -> ``2-carboxyethyl``.

    Fail-closed (returns None) for anything but a single unbranched all-carbon
    chain from the free valence bearing exactly one terminal -COOH and no other
    substituent/heteroatom/ring/unsaturation.
    """
    if mol.GetRingInfo().NumRings():
        return None
    # Walk the chain from the radical carbon. Each step: the current C must have
    # exactly one onward C neighbour (unbranched), until we reach a carboxyl C.
    chain: List[int] = [radical_idx]
    prev = -1
    cur = radical_idx
    acid_locant: Optional[int] = None
    while True:
        atom = mol.GetAtomWithIdx(cur)
        if atom.GetSymbol() != 'C' or atom.GetIsAromatic():
            return None
        # onward heavy neighbours (excluding where we came from)
        onward = [n.GetIdx() for n in atom.GetNeighbors()
                  if n.GetIdx() != prev]
        # any onward atom must be C (no ethers/amines etc. in scope)
        carboxyls = [i for i in onward if _is_carboxyl_carbon(mol, i, cur)]
        carbons = [i for i in onward
                   if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                   and i not in carboxyls]
        others = [i for i in onward
                  if i not in carboxyls and i not in carbons]
        if others:
            return None
        if carboxyls:
            if len(carboxyls) != 1 or carbons:
                return None  # branch or >1 acid: out of scope
            acid_locant = len(chain)  # current carbon's chain locant
            break
        if len(carbons) != 1:
            return None  # branch or chain terminus without an acid
        prev, cur = cur, carbons[0]
        chain.append(cur)
        if len(chain) > 30:
            return None
    if acid_locant is None:
        return None
    # any ring/unsaturation on the chain is out of scope (keep it simple)
    for a_idx in chain:
        a = mol.GetAtomWithIdx(a_idx)
        for b in a.GetBonds():
            if b.GetBondType() != Chem.BondType.SINGLE:
                return None
    stem = _get_chain_prefix(len(chain))
    # A single-carbon stem carries no locant ambiguity: 'carboxymethyl'
    # (not '1-carboxymethyl').
    if len(chain) == 1:
        return f"carboxy{stem}yl"
    return f"{acid_locant}-carboxy{stem}yl"


# 169.6-03 (CHOKE-01, kill-list §2.2): the carbon-counting chain-counter that
# fed the alkyl / -ylidene / -ylidyne radical naming was DELETED. The three
# helpers below are now thin shims over route_charged (the single chokepoint),
# which H-saturates the radical center, re-enters the FULL pipeline, and
# re-applies the P-71 suffix — so a SUBSTITUTED/branched/unsaturated alkyl
# radical is named correctly instead of by a bare carbon count.


def name_alkyl_radical(mol, radical_site: Dict[str, Any]) -> str:
    """
    Name an alkyl radical (R.) via the route_charged chokepoint (P-71.1.1).

    169.6-03: delegates to route_charged (neutralize the radical center ->
    re-enter the full pipeline -> re-apply the -yl/-ylidene/-ylidyne suffix),
    replacing the old carbon-counting body. Returns the radical name (e.g.
    'methyl', 'propyl') or '' on fall-through.

    Args:
        mol: RDKit Mol object
        radical_site: Dictionary from get_radical_sites (kept for API stability)

    Example:
        >>> mol = Chem.MolFromSmiles('[CH3]')
        >>> sites = get_radical_sites(mol)
        >>> name_alkyl_radical(mol, sites[0])
        'methyl'
    """
    from .charged_router import route_charged
    return route_charged(mol, 'pin')


def name_acyl_radical(mol, radical_site: Dict[str, Any]) -> str:
    """
    Name an acyl radical (RC.=O).

    Acyl radicals are derived from carboxylic acids by loss of -OH.
    Named with -oyl suffix (e.g., acetyl from acetic acid).

    Args:
        mol: RDKit Mol object
        radical_site: Dictionary from get_radical_sites

    Returns:
        Acyl radical name (e.g., 'acetyl', 'propanoyl', 'benzoyl')

    Example:
        >>> mol = Chem.MolFromSmiles('C[C]=O')
        >>> sites = get_radical_sites(mol)
        >>> name_acyl_radical(mol, sites[0])
        'acetyl'
    """
    radical_idx = radical_site['atom_idx']
    radical_atom = mol.GetAtomWithIdx(radical_idx)

    # Count carbons (excluding the carbonyl oxygen)
    carbon_count = 1  # Start with the carbonyl carbon

    for neighbor in radical_atom.GetNeighbors():
        if neighbor.GetSymbol() == 'C':
            carbon_count += _count_chain_carbons(mol, neighbor.GetIdx(), {radical_idx})

    # Map to acyl name
    ACYL_NAMES = {
        1: 'formyl',      # H-C(=O).
        2: 'acetyl',      # CH3-C(=O).
        3: 'propanoyl',   # C2H5-C(=O).
        4: 'butanoyl',
        5: 'pentanoyl',
        6: 'hexanoyl',
        7: 'heptanoyl',
        8: 'octanoyl',
    }

    # Check for aromatic acyl (benzoyl)
    for neighbor in radical_atom.GetNeighbors():
        if neighbor.GetIsAromatic():
            return 'benzoyl'

    if carbon_count in ACYL_NAMES:
        return ACYL_NAMES[carbon_count]
    return _get_chain_prefix(carbon_count) + 'anoyl'


def name_oxyl_radical(mol, radical_site: Dict[str, Any]) -> str:
    """
    Name an oxyl radical (RO.).

    Oxygen-centered radicals named as R-oxyl.

    Args:
        mol: RDKit Mol object
        radical_site: Dictionary from get_radical_sites

    Returns:
        Oxyl radical name (e.g., 'methoxyl', 'ethoxyl', 'phenoxyl')

    Example:
        >>> mol = Chem.MolFromSmiles('[O]C')
        >>> sites = get_radical_sites(mol)
        >>> name_oxyl_radical(mol, sites[0])
        'methoxyl'
    """
    radical_idx = radical_site['atom_idx']
    radical_atom = mol.GetAtomWithIdx(radical_idx)

    # Find what's attached to the oxygen
    for neighbor in radical_atom.GetNeighbors():
        if neighbor.GetSymbol() == 'C':
            # Check if aromatic (phenoxyl)
            if neighbor.GetIsAromatic():
                return 'phenoxyl'

            # Count carbons in the alkyl chain
            carbon_count = _count_chain_carbons(mol, neighbor.GetIdx(), {radical_idx})

            # Map to alkoxy name
            ALKOXY_NAMES = {
                1: 'methoxyl',
                2: 'ethoxyl',
                3: 'propoxyl',
                4: 'butoxyl',
                5: 'pentoxyl',
                6: 'hexoxyl',
            }

            if carbon_count in ALKOXY_NAMES:
                return ALKOXY_NAMES[carbon_count]
            return _get_chain_prefix(carbon_count) + 'oxyl'

    return 'oxyl'


def name_divalent_radical(mol, radical_site: Dict[str, Any]) -> str:
    """
    Name a divalent (carbene-like) radical -> -ylidene, via route_charged (P-71).

    169.6-03: delegates to the chokepoint (replacing the carbon-counting body).

    Example:
        >>> mol = Chem.MolFromSmiles('[CH2]')
        >>> sites = get_radical_sites(mol)
        >>> name_divalent_radical(mol, sites[0])
        'methylidene'
    """
    from .charged_router import route_charged
    return route_charged(mol, 'pin')


def name_trivalent_radical(mol, radical_site: Dict[str, Any]) -> str:
    """
    Name a trivalent (carbyne-like) radical -> -ylidyne, via route_charged (P-71).

    169.6-03: delegates to the chokepoint (replacing the carbon-counting body).

    Example:
        >>> mol = Chem.MolFromSmiles('[CH]')
        >>> sites = get_radical_sites(mol)
        >>> name_trivalent_radical(mol, sites[0])
        'methylidyne'
    """
    from .charged_router import route_charged
    return route_charged(mol, 'pin')


def name_aryl_radical(mol, radical_site: Dict[str, Any]) -> str:
    """
    Name an aryl radical (Ar.).

    Aromatic carbon-centered radicals.

    Args:
        mol: RDKit Mol object
        radical_site: Dictionary from get_radical_sites

    Returns:
        Aryl radical name (e.g., 'phenyl', 'naphthyl')

    Example:
        >>> mol = Chem.MolFromSmiles('[c]1ccccc1')
        >>> sites = get_radical_sites(mol)
        >>> name_aryl_radical(mol, sites[0])
        'phenyl'
    """
    # Count aromatic carbons to determine ring system
    aromatic_count = sum(1 for atom in mol.GetAtoms()
                         if atom.GetIsAromatic() and atom.GetSymbol() == 'C')

    if aromatic_count == 6:
        return 'phenyl'
    elif aromatic_count == 10:
        return 'naphthyl'
    elif aromatic_count == 14:
        return 'anthryl'
    else:
        return 'aryl'


# === MAIN NAMING FUNCTION ===

def name_radical(mol, style: str = 'pin') -> str:
    """
    Generate IUPAC name for a radical species.

    Workflow:
    1. Get canonical SMILES
    2. Check retained names (unless systematic style)
    3. Detect radical sites
    4. Classify radical type
    5. Generate systematic name

    Args:
        mol: RDKit Mol object with radical center(s)
        style: Naming style ('pin', 'systematic', 'common')

    Returns:
        IUPAC name for the radical (e.g., 'methyl', 'methylidene')

    Example:
        >>> mol = Chem.MolFromSmiles('[CH3]')
        >>> name_radical(mol)
        'methyl'
        >>> mol = Chem.MolFromSmiles('[CH2]')
        >>> name_radical(mol)
        'methylidene'
    """
    if mol is None:
        return ''

    # Get canonical SMILES for lookup
    canonical = Chem.MolToSmiles(mol, canonical=True)

    # Check retained names first (unless systematic requested)
    if style != 'systematic':
        if canonical in RETAINED_RADICALS:
            return RETAINED_RADICALS[canonical]

    # Get radical sites
    sites = get_radical_sites(mol)

    if not sites:
        return ''

    # For single radical, classify and name
    if len(sites) == 1:
        site = sites[0]
        info = classify_radical(mol, site)
        subtype = info['subtype']

        # Route to appropriate naming function. The acyl/oxyl/aryl heteroatom-
        # context subtypes keep their structured helpers (they are NOT plain
        # parent-hydride -yl/-ylidene/-ylidyne loss and are not route_charged's
        # scope).
        if subtype == 'oxyl':
            return name_oxyl_radical(mol, site)
        elif subtype == 'acyl':
            return name_acyl_radical(mol, site)
        elif subtype == 'aryl':
            return name_aryl_radical(mol, site)

        # P-41 Table 4.1 cls 1: the free valence is the MOST senior class,
        # senior to a carboxylic acid — a chain-terminal -COOH on an alkyl
        # radical is demoted to a 'carboxy' prefix (2-carboxyethyl). Try this
        # before route_charged (which H-saturates the acid and mis-drops it).
        if info['radical_type'] == 'monovalent':
            carboxy_name = _name_carboxy_alkyl_radical(mol, site['atom_idx'])
            if carboxy_name:
                return carboxy_name

        # 169.6-03 (CHOKE-01, kill-list §2.2): the alkyl / -ylidene / -ylidyne
        # carbon-counting paths are DELETED. Delegate to route_charged, which
        # H-saturates the radical center, re-enters the FULL pipeline (so
        # substituents/unsaturation/branching are named correctly), and re-applies
        # the P-71 -yl/-ylidene/-ylidyne suffix. On '' fall through to '' (no
        # carbon-counted misname).
        from .charged_router import route_charged
        return route_charged(mol, style)

    # Multiple radicals (diradicals/polyradicals): the oxyl/acyl helpers still
    # apply to the first site; otherwise out of scope -> route_charged ('' on
    # multi-radical, the v18 fall-through).
    first_site = sites[0]
    info = classify_radical(mol, first_site)
    if info['subtype'] == 'oxyl':
        return name_oxyl_radical(mol, first_site)
    elif info['subtype'] == 'acyl':
        return name_acyl_radical(mol, first_site)
    from .charged_router import route_charged
    return route_charged(mol, style)
