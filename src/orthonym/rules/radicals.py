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
    '[CH2]C': 'ethyl',
    '[CH2]CC': 'propyl',
    '[CH2]CCC': 'butyl',

    # Divalent radicals
    '[CH2]': 'methylidene',
    '[CH]C': 'ethylidene',

    # Trivalent radicals
    '[CH]': 'methylidyne',

    # Aryl radicals
    '[c]1ccccc1': 'phenyl',
    '[CH2]c1ccccc1': 'benzyl',

    # Acyl radicals (carbonyl radicals)
    '[CH]=O': 'formyl',
    'C[C]=O': 'acetyl',
    'CC[C]=O': 'propanoyl',
    'O=[C]c1ccccc1': 'benzoyl',

    # Oxyl radicals (oxygen-centered)
    'C[O]': 'methoxyl',
    'CC[O]': 'ethoxyl',
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


def _collect_oxyl_parent_fragment(mol, radical_idx: int, attach_idx: int) -> List[int]:
    """BFS the atoms of the R group hanging off an R-O. oxyl radical's
    oxygen, starting at ``attach_idx`` (the O's sole heavy neighbour) and
    never crossing back through ``radical_idx`` (the radical oxygen itself).
    Mirrors the R-substituent-fragment collection already used for the
    ester-alkyl side elsewhere in this codebase (``acid_halides.py``'s
    ``r_atoms``/``r_root`` walk)."""
    from collections import deque
    seen = {radical_idx}
    frag: List[int] = []
    queue = deque([attach_idx])
    while queue:
        cur = queue.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        frag.append(cur)
        for nbr in mol.GetAtomWithIdx(cur).GetNeighbors():
            if nbr.GetIdx() not in seen:
                queue.append(nbr.GetIdx())
    return frag


def _is_plain_alkyl_radical_fragment(mol, radical_idx: int, attach_idx: int) -> bool:
    """True iff the R group of an aliphatic R-O. radical is a BARE
    hydrocarbon (every atom reachable without crossing the radical O is a
    non-aromatic, non-ring carbon) -- the exact shape the existing retained
    '-oxyl' contraction (methoxyl/ethoxyl/propoxyl/.../hexoxyl) was already
    built for. False for anything carrying a ring or any heteroatom
    substituent, which must go through the general substituent pipeline
    instead so a substituent is never silently dropped."""
    for idx in _collect_oxyl_parent_fragment(mol, radical_idx, attach_idx):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C' or atom.GetIsAromatic() or atom.IsInRing():
            return False
    return True


def _name_oxyl_parent_group(mol, radical_idx: int, attach_idx: int) -> str:
    """Name the R group of an R-O. oxyl radical as a substituent (P-29.2 -yl
    prefix), via the project's existing recursive substituent-naming
    chokepoint ``name_substituent_fragment`` -- the SAME pipeline other rule
    modules use to name an R group hanging off an excluded heteroatom (e.g.
    the ester-alkyl side in ``acid_halides.py``/``esters.py``). '' on failure
    (caller falls through / abstains -- never a wrong name)."""
    frag = _collect_oxyl_parent_fragment(mol, radical_idx, attach_idx)
    if not frag:
        return ''
    from ..assembly.substituent_naming import name_substituent_fragment
    return name_substituent_fragment(mol, frag, attach_idx, [radical_idx]) or ''


def _compose_oxidanyl(parent: str) -> str:
    """P-63.2.1 systematic oxyl-radical composition: 'oxidane' (H2O's parent
    hydride name) elides its final 'e' for the -yl radical suffix
    ('oxidanyl' == HO.), and a substituent R on that oxygen is cited as its
    ordinary P-29.2 prefix in front -- '(R)oxidanyl'. A compound/locanted R
    takes enclosing marks (P-16.5.1.1) via the same ``enclose_if_compound``
    gate every other compound substituent prefix in this codebase uses."""
    from ..assembly.naming_utils import enclose_if_compound
    return f"{enclose_if_compound(parent)}oxidanyl"


def name_oxyl_radical(mol, radical_site: Dict[str, Any]) -> str:
    """
    Name an oxyl radical (RO.).

    Oxygen-centered radicals named as R-oxyl for the small retained set of
    UNSUBSTITUTED R groups (methoxyl/ethoxyl/phenoxyl/propoxyl/.../hexoxyl),
    or systematically as ``(R)oxidanyl`` (P-63.2.1) when R itself carries a
    further substituent -- v36 A2 generalisation: the R group is named
    through the existing substituent-naming pipeline instead of being
    collapsed to the unsubstituted retained form (the pre-v36 bug: any
    aromatic-O radical mapped unconditionally to 'phenoxyl', dropping every
    ring substituent).

    Args:
        mol: RDKit Mol object
        radical_site: Dictionary from get_radical_sites

    Returns:
        Oxyl radical name (e.g., 'methoxyl', 'phenoxyl',
        '(4-hydroxyphenyl)oxidanyl'), or '' if the R group cannot be named
        (fail-closed -- never a wrong name).

    Example:
        >>> mol = Chem.MolFromSmiles('[O]C')
        >>> sites = get_radical_sites(mol)
        >>> name_oxyl_radical(mol, sites[0])
        'methoxyl'
    """
    radical_idx = radical_site['atom_idx']
    radical_atom = mol.GetAtomWithIdx(radical_idx)

    heavy_neighbors = list(radical_atom.GetNeighbors())
    if len(heavy_neighbors) != 1 or heavy_neighbors[0].GetSymbol() != 'C':
        return 'oxyl'

    neighbor = heavy_neighbors[0]
    attach_idx = neighbor.GetIdx()

    if neighbor.GetIsAromatic():
        # name_radical's RETAINED_RADICALS canonical-SMILES lookup already
        # returns 'phenoxyl' for the bare unsubstituted ring ('[O]c1ccccc1')
        # BEFORE classify_radical/this function ever run (name_radical.py:
        # the `canonical in RETAINED_RADICALS` check precedes the
        # classify_radical dispatch), so any aromatic neighbour reaching here
        # carries a real substituent (or is a larger fused ring system) --
        # generalise via the substituent pipeline.
        parent = _name_oxyl_parent_group(mol, radical_idx, attach_idx)
        if not parent:
            return ''
        if parent == 'phenyl':
            return 'phenoxyl'  # defensive; unreachable given the guard above
        return _compose_oxidanyl(parent)

    # Aliphatic R: keep the existing retained '-oxyl' contraction for a PLAIN
    # (unsubstituted) hydrocarbon chain -- unchanged PINs -- and generalise
    # via the same substituent pipeline for anything else (a ring or any
    # heteroatom substituent the old carbon count could not see).
    if _is_plain_alkyl_radical_fragment(mol, radical_idx, attach_idx):
        carbon_count = _count_chain_carbons(mol, attach_idx, {radical_idx})

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

    parent = _name_oxyl_parent_group(mol, radical_idx, attach_idx)
    if not parent:
        return ''
    return _compose_oxidanyl(parent)


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


# === HETEROATOM-CENTRED RADICALS (P-71.2.1.2 / P-71.2.2.2 / P-71.3.2 / P-71.3.3) ===
#
# The carbon chokepoint (charged_router.route_charged) names a parent hydride by
# neutralize -> re-enter -> _apply_radical_suffix, which drops the whole 'ane'
# ending (methane->methyl, silane->silyl). That contraction is the P-71.2.1.1 rule
# and is correct ONLY for Group-14 mononuclear hydrides / acyclic-hydrocarbon
# termini / monocyclic saturated hydrocarbon rings. For a NON-Group-14 heteroatom
# parent hydride (azane, sulfane, borane, ...) the P-71.2.1.2 general method elides
# ONLY the final 'e' (azane->azanyl, sulfane->sulfanyl, borane->boranyl) — so the
# chokepoint produced the WRONG 'azyl'/'sulfyl'/'boryl'. These namers own the
# heteroatom cases with the correct, element-keyed contraction and fail closed
# ('' -> the caller's carbon path) for everything else.

# P-71.2.1.1 / P-71.2.2.1: Group-14 mononuclear parent hydrides -> drop 'ane'.
_GROUP14_HYDRIDE_RADICAL = {
    'C': 'methane', 'Si': 'silane', 'Ge': 'germane', 'Sn': 'stannane', 'Pb': 'plumbane',
}
# P-71.2.1.2 / P-71.2.2.2: non-Group-14 heteroatom parent hydrides -> elide 'e' only.
_ELIDE_E_HYDRIDE_RADICAL = {
    'N': 'azane', 'P': 'phosphane', 'As': 'arsane', 'Sb': 'stibane', 'Bi': 'bismuthane',
    'S': 'sulfane', 'Se': 'selane', 'Te': 'tellane',
    'B': 'borane', 'Al': 'alumane', 'Ga': 'gallane', 'In': 'indigane', 'Tl': 'thallane',
}
_RADICAL_SUFFIX_BY_NE = {1: 'yl', 2: 'ylidene', 3: 'ylidyne'}
# P-70.3.2: multiplying prefixes 'bis'/'tris'/... precede compound suffixes.
_COMPOUND_MULTIPLIER = {2: 'bis', 3: 'tris', 4: 'tetrakis'}


def _parent_hydride_radical_name(element: str, n_electrons: int) -> str:
    """P-71.2.1.1/.2 + P-71.2.2.1/.2: name a MONONUCLEAR heteroatom radical from its
    element's parent-hydride name. Group-14 (silane/germane) drop the whole 'ane';
    non-Group-14 (azane/sulfane/borane) elide only the final 'e'.
    -> azanyl, azanylidene, sulfanyl, boranyl, silyl, germyl. '' if unmapped.

    BB (PIN/preselected): HS. sulfanyl (40430); H2N. azanyl (40434); H2B. boranyl,
    'not boryl' (40438); HN: azanylidene (40506); H2Si: silylidene (40484)."""
    suffix = _RADICAL_SUFFIX_BY_NE.get(n_electrons)
    if suffix is None:
        return ''
    if element in _GROUP14_HYDRIDE_RADICAL:
        return _GROUP14_HYDRIDE_RADICAL[element][:-3] + suffix
    if element in _ELIDE_E_HYDRIDE_RADICAL:
        return _ELIDE_E_HYDRIDE_RADICAL[element][:-1] + suffix
    return ''


def _reenter_neutral_name(mol) -> str:
    """Name the neutral parent by re-entering the full pipeline with the OPSIN
    validity gate disabled (radical intermediates are correct-by-construction; the
    outer gate re-checks the final radical name). '' on failure/unknown."""
    smi = Chem.MolToSmiles(mol)
    if not smi:
        return ''
    try:
        from ..namer import Orthonym
        name = Orthonym(style='pin', _disable_opsin_validity_gate=True).name(smi)
    except (RecursionError, ValueError, RuntimeError):
        return ''
    if not name or 'unknown' in name.lower():
        return ''
    return name


def _name_amine_family_radical(mol, radical_site: Dict[str, Any]) -> str:
    """P-71.3.2 (Table 7.2): a radical on a substituted amine / imine / amide N is
    named with the compound suffix -aminyl / -iminyl / -amidyl on the NEUTRAL parent
    name (the N as its principal characteristic group). The cumulative radical
    suffix is appended by eliding the parent name's final 'e' (P-70.3.3.2.1):
      methanamine -> methanaminyl, propan-1-imine -> propan-1-iminyl,
      formamide -> formamidyl.
    Fail-closed ('') when the neutral parent does NOT name the N as a suffix
    (e.g. the retained 'aniline', where the systematic 'benzenamine' would be
    needed for 'benzenaminyl') — never a wrong name."""
    idx = radical_site['atom_idx']
    n_e = radical_site['n_electrons']
    cum = {1: 'yl', 2: 'ylidene'}.get(n_e)
    if cum is None:
        return ''
    work = Chem.RWMol(mol)
    try:
        a = work.GetAtomWithIdx(idx)
        a.SetNumRadicalElectrons(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(a.GetTotalNumHs() + n_e)
        wm = work.GetMol()
        Chem.SanitizeMol(wm)
    except (RuntimeError, ValueError):
        return ''
    neutral = _reenter_neutral_name(wm)
    if not neutral:
        return ''
    # '-carboxamide'/'-amide'/'-imine'/'-amine' -> elide final 'e' + cumulative suffix.
    for base in ('carboxamide', 'amide', 'imine', 'amine'):
        if neutral.endswith(base):
            return neutral[:-1] + cum
    return ''


def _name_multiplicative_amine_radical(mol, radical_sites: List[Dict[str, Any]]) -> str:
    """P-71.3.3 method (1): poly-amine radicals named as
    (central-multivalent-group)bis/tris(aminyl). Scope (tight, fail-closed): >=2
    monovalent radical centres, each a nitrogen bearing exactly one carbon neighbour
    (a primary-amine -NH.) on a shared acyclic all-carbon central skeleton, all
    identically derived. -> [NH]CC[NH] (ethane-1,2-diyl)bis(aminyl) (BB 40660, PIN).

    Per P-71.3.3 the parent radical in multiplicative nomenclature is 'azanyl'
    (method 2); method (1) — the PIN — uses the reserved suffix 'aminyl'."""
    centers_c: List[int] = []
    for s in radical_sites:
        a = mol.GetAtomWithIdx(s['atom_idx'])
        if a.GetSymbol() != 'N' or s['n_electrons'] != 1:
            return ''
        heavy = [nb for nb in a.GetNeighbors() if nb.GetSymbol() != 'H']
        if len(heavy) != 1 or heavy[0].GetSymbol() != 'C':
            return ''
        centers_c.append(heavy[0].GetIdx())
    if len(set(centers_c)) != len(centers_c):
        return ''  # two amine N on one carbon -> out of scope
    mult = _COMPOUND_MULTIPLIER.get(len(radical_sites))
    if mult is None:
        return ''
    # Build the central multivalent group: mark the attachment carbons, sever the
    # amine nitrogens, set the carbons as free-valence centres, name via the
    # polyvalent parent-hydride primitive (-> 'ethane-1,2-diyl').
    n_idxs = [s['atom_idx'] for s in radical_sites]
    work = Chem.RWMol(mol)
    for c_idx in centers_c:
        work.GetAtomWithIdx(c_idx).SetIntProp('_mult_center', 1)
    for nidx in sorted(n_idxs, reverse=True):
        work.RemoveAtom(nidx)
    frag = work.GetMol()
    centers: List[tuple] = []
    for a in frag.GetAtoms():
        if a.HasProp('_mult_center'):
            a.SetNumRadicalElectrons(1)
            a.SetNoImplicit(False)
            a.ClearProp('_mult_center')
            centers.append((a.GetIdx(), 1))
    if len(centers) != len(radical_sites):
        return ''
    try:
        Chem.SanitizeMol(frag)
    except (RuntimeError, ValueError):
        return ''
    from .ions import emit_parent_hydride_polyvalent_suffixes
    central = emit_parent_hydride_polyvalent_suffixes(frag, centers)
    if not central:
        return ''
    return f"({central}){mult}(aminyl)"


def name_heteroatom_radical(mol, radical_sites: List[Dict[str, Any]],
                            style: str = 'pin') -> str:
    """Heteroatom-centred radical PINs (called from the route_charged chokepoint
    BEFORE its carbon-centric path so a heteroatom never gets a wrong Group-14
    contraction):

      * P-71.2.1.2 / P-71.2.2.2 mononuclear parent-hydride radicals — azanyl,
        azanylidene, sulfanyl, boranyl, silyl, germyl, ...
      * P-71.3.2 amine / imine / amide compound-suffix radicals — methanaminyl,
        propan-1-iminyl, formamidyl.
      * P-71.3.3 multiplicative poly-amine radicals — (ethane-1,2-diyl)bis(aminyl).

    Returns '' (fail-closed) for any carbon-centred or out-of-scope radical so the
    caller's existing carbon path / name_radical helpers run unchanged."""
    if not radical_sites:
        return ''
    # Multi-centre: only the P-71.3.3 multiplicative poly-amine class is in scope
    # here; every other multi-radical is the carbon polyvalent primitive's job.
    if len(radical_sites) >= 2:
        return _name_multiplicative_amine_radical(mol, radical_sites)

    site = radical_sites[0]
    atom = mol.GetAtomWithIdx(site['atom_idx'])
    element = atom.GetSymbol()
    if element in ('C', 'O'):
        # Carbon -> caller's chokepoint; oxygen -> the oxyl/hydroxyl handler.
        return ''
    heavy_nbrs = [nb for nb in atom.GetNeighbors() if nb.GetSymbol() != 'H']
    # (A) Mononuclear heteroatom parent-hydride radical (only H neighbours).
    if not heavy_nbrs and mol.GetNumHeavyAtoms() == 1:
        return _parent_hydride_radical_name(element, site['n_electrons'])
    # (B) Substituted nitrogen -> amine / imine / amide compound suffix.
    if element == 'N':
        return _name_amine_family_radical(mol, site)
    # Substituted non-N heteroatom radical -> out of scope, fail closed.
    return ''


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

    # Multiple radicals: delegate to route_charged (P-71.2.3 multi-site free-valence
    # namer). No first-site oxyl/acyl shortcut — it produced structure-dropping names
    # (e.g. [O]CC[O] -> 'ethoxyl') that SELF-01 only suppressed. Fail closed on ''.
    from .charged_router import route_charged
    return route_charged(mol, style)
