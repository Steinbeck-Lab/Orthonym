"""
Polyfunctional compound naming coordinator.

Handles naming of compounds with multiple functional groups:
- Principal group (highest seniority) becomes the suffix
- Lower-seniority groups become prefixes with locants
- Ethers (no seniority) are always named as alkoxy prefixes

Based on IUPAC 2013 Blue Book P-41 to P-43.
"""

import logging
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Any, Set

logger = logging.getLogger(__name__)

from ..assembly.naming_utils import (
    alpha_sort_key,
    get_multiplier_prefix,
    format_suffix_with_locants,
    get_alkyl_name,
    ALKYL_NAMES,
)
from .seniority import (
    get_principal_group,
    get_prefix,
    SENIORITY_ORDER,
    PREFIX_FORMS,
)


# Functional groups that can be detected but have no seniority
# These are always named as prefixes (never suffix)
NO_SENIORITY_GROUPS = {
    "ether",
    "vinyl_ether",
    "aromatic_ether",
    "fluoro",
    "chloro",
    "bromo",
    "iodo",
    "nitro",
    "nitroso",
    "azido",
    "alkene",  # Handled separately via unsaturation
    "alkyne",  # Handled separately via unsaturation
}

# Chain prefixes for ether alkoxy naming (match ALKYL_NAMES pattern)
ALKOXY_NAMES = {
    1: "methoxy",
    2: "ethoxy",
    3: "propoxy",
    4: "butoxy",
    5: "pentyloxy",
    6: "hexyloxy",
    7: "heptyloxy",
    8: "octyloxy",
    9: "nonyloxy",
    10: "decyloxy",
}


def detect_polyfunctional(mol, functional_groups: Dict[str, List[tuple]]) -> bool:
    """
    Determine if a molecule has multiple distinct functional groups.

    Returns True if the molecule has 2+ distinct functional groups that
    participate in naming (excludes unsaturation markers and halogens).

    A compound with multiple instances of the SAME group (e.g., diol)
    is NOT considered polyfunctional by this function - that's multiplicity.

    Args:
        mol: RDKit Mol object
        functional_groups: Dict from detect_functional_groups()

    Returns:
        True if molecule is polyfunctional (multiple distinct FGs)
    """
    # Groups that count as "functional" for polyfunctional detection
    # Excludes: alkene/alkyne (unsaturation), halogens (always prefix anyway)
    exclude_groups = {
        "alkene", "alkyne",
        "fluoro", "chloro", "bromo", "iodo",
    }

    # Count distinct functional group types with seniority or special handling
    distinct_groups = set()

    for fg_name, matches in functional_groups.items():
        if not matches:
            continue
        if fg_name in exclude_groups:
            continue

        # Count this as a distinct group
        distinct_groups.add(fg_name)

    return len(distinct_groups) >= 2


def get_fg_prefix_form(
    fg_name: str,
    mol,
    atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """
    Get the prefix form for a functional group.

    For most groups, uses the standard prefix from seniority.py.
    For ethers, determines the alkoxy prefix based on substituent size.

    Args:
        fg_name: Name of the functional group
        mol: RDKit Mol object
        atoms: Atom indices matching this FG instance
        principal_chain: Atom indices of the principal chain

    Returns:
        Prefix string (e.g., "hydroxy", "oxo", "methoxy") or None if no prefix
    """
    # Handle ethers specially - determine alkoxy prefix
    if fg_name in ("ether", "vinyl_ether", "aromatic_ether"):
        return _get_alkoxy_prefix(mol, atoms, principal_chain)

    # Handle esters specially - generate alkoxycarbonyl prefix (IUPAC P-65.6.3)
    if fg_name == "ester":
        return _get_alkoxycarbonyl_prefix(mol, atoms, principal_chain)

    # Handle sulfoxide -> (alkyl)sulfinyl compound prefix (IUPAC P-63.6)
    if fg_name == "sulfoxide":
        return _get_sulfinyl_prefix(mol, atoms, principal_chain)

    # Handle sulfone -> (alkyl)sulfonyl compound prefix (IUPAC P-63.6)
    if fg_name == "sulfone":
        return _get_sulfonyl_prefix(mol, atoms, principal_chain)

    # Handle thioether -> (alkyl)sulfanyl compound prefix (IUPAC P-63.2.5)
    if fg_name == "thioether":
        return _get_sulfanyl_prefix(mol, atoms, principal_chain)

    # For other groups, use standard prefix form
    return get_prefix(fg_name)


def _get_alkoxy_prefix(
    mol,
    ether_atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """
    Determine the alkoxy prefix for an ether.

    The ether SMARTS "[OX2]([CX4])[CX4]" matches both carbons.
    We need to determine which side is the substituent (smaller/not in chain)
    and name it as alkoxy.

    Args:
        mol: RDKit Mol object
        ether_atoms: Atom indices from ether SMARTS match (O, C, C)
        principal_chain: Atom indices of the principal chain

    Returns:
        Alkoxy prefix (e.g., "methoxy", "ethoxy"), or None if naming fails.
        Never returns the literal string "alkoxy" (not valid IUPAC).
    """
    # ether_atoms from SMARTS "[OX2]([CX4])[CX4]" = (O, C1, C2)
    if len(ether_atoms) < 3:
        return None  # Defensive guard: let caller skip this ether

    oxygen_idx = ether_atoms[0]
    carbon1_idx = ether_atoms[1]
    carbon2_idx = ether_atoms[2]

    chain_set = set(principal_chain)

    # Determine which carbon is the substituent (not in principal chain)
    # or the smaller fragment if both/neither in chain
    carbon1_in_chain = carbon1_idx in chain_set
    carbon2_in_chain = carbon2_idx in chain_set

    if carbon1_in_chain and not carbon2_in_chain:
        # carbon2 is the substituent
        sub_carbon = carbon2_idx
    elif carbon2_in_chain and not carbon1_in_chain:
        # carbon1 is the substituent
        sub_carbon = carbon1_idx
    else:
        # Neither or both in chain - use smaller fragment
        # Count atoms on each side via BFS
        frag1_size = _count_fragment_atoms(mol, carbon1_idx, {oxygen_idx})
        frag2_size = _count_fragment_atoms(mol, carbon2_idx, {oxygen_idx})
        sub_carbon = carbon1_idx if frag1_size <= frag2_size else carbon2_idx

    # Check if the substituent is aromatic (phenoxy, benzyloxy)
    sub_atom = mol.GetAtomWithIdx(sub_carbon)

    # Case A: O -> aromatic C in 6-membered all-carbon ring -> "phenoxy"
    if sub_atom.GetIsAromatic():
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if sub_carbon in ring and len(ring) == 6:
                if all(mol.GetAtomWithIdx(r).GetIsAromatic()
                       and mol.GetAtomWithIdx(r).GetSymbol() == 'C'
                       for r in ring):
                    return "phenoxy"
        # Fallback for other aromatic ethers
        return "phenoxy"

    # Case B: O -> CH2 -> aromatic ring -> "benzyloxy"
    if (not sub_atom.GetIsAromatic()
            and sub_atom.GetSymbol() == 'C'
            and sub_atom.GetTotalNumHs() >= 1):
        arom_nbrs = [n for n in sub_atom.GetNeighbors()
                     if n.GetIdx() != oxygen_idx and n.GetIsAromatic()]
        non_h_non_arom = [n for n in sub_atom.GetNeighbors()
                          if n.GetIdx() != oxygen_idx
                          and not n.GetIsAromatic()
                          and n.GetSymbol() != 'H']
        if arom_nbrs and not non_h_non_arom:
            return "benzyloxy"

    # Count carbons in the substituent fragment
    carbon_count = _count_fragment_atoms(mol, sub_carbon, {oxygen_idx}, carbons_only=True)

    # Get alkoxy name
    if carbon_count in ALKOXY_NAMES:
        return ALKOXY_NAMES[carbon_count]
    else:
        # For larger groups, build name from alkyl
        try:
            alkyl = get_alkyl_name(carbon_count)
            # Remove 'yl' and add 'yloxy' for larger alkoxy groups
            if alkyl.endswith("yl"):
                return alkyl[:-2] + "yloxy"
            return alkyl + "oxy"
        except (ValueError, KeyError):
            # get_alkyl_name failed; try get_chain_prefix for arbitrary counts
            try:
                from ..data.chain_names import get_chain_prefix
                prefix = get_chain_prefix(carbon_count)
                return f"{prefix}yloxy"
            except (ValueError, KeyError):
                return None  # Not "alkoxy" -- let caller handle absence


def _get_alkoxycarbonyl_prefix(
    mol,
    ester_atoms: tuple,
    principal_chain: List[int],
) -> Optional[str]:
    """Generate alkoxycarbonyl prefix for ester-as-non-principal-group.

    Per IUPAC P-65.6.3, when an ester group -C(=O)-O-R is not the principal
    characteristic group, it is expressed as an alkoxycarbonyl prefix:
      -COOCH3   -> methoxycarbonyl
      -COOC2H5  -> ethoxycarbonyl
      -COOPh    -> phenoxycarbonyl

    Args:
        mol: RDKit Mol object.
        ester_atoms: Tuple from ester SMARTS "[CX3](=O)[OX2][#6]":
                     (carbonyl_C, carbonyl_O, ester_O, alkyl_C).
        principal_chain: Atom indices of the principal chain.

    Returns:
        Alkoxycarbonyl prefix string, or None if this ester should not
        be named as alkoxycarbonyl (e.g., lactones).
    """
    from .esters import parse_ester_fragments, is_lactone

    # Guard 1: lactones are named differently, not as alkoxycarbonyl
    if is_lactone(mol, ester_atoms):
        return None

    # Guard 2: orientation check — alkoxycarbonyl only applies when the
    # carbonyl C is on or bonded to the principal chain, meaning the ester
    # extends as -C(=O)-O-R away from the chain.  When the ester O is on
    # the chain instead (chain-O-C(=O)-R), the ester is an acyloxy
    # substituent, handled by _check_for_acyloxy() in composer.py.
    carbonyl_c = ester_atoms[0]
    ester_o = ester_atoms[2] if len(ester_atoms) > 2 else None
    chain_set = set(principal_chain)
    if chain_set and ester_o is not None:
        c_on_chain = carbonyl_c in chain_set
        o_on_chain = ester_o in chain_set
        c_adj_chain = c_on_chain or any(
            nbr.GetIdx() in chain_set
            for nbr in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors()
            if nbr.GetIdx() != ester_atoms[1]  # exclude carbonyl O
            and nbr.GetIdx() != ester_o
        )
        if not c_adj_chain and o_on_chain:
            # O faces the chain, C(=O) faces away -> acyloxy, not alkoxycarbonyl
            return None
        if not c_adj_chain and not o_on_chain:
            # Neither end touches the chain; ester is in an isolated branch
            return None

    # Split ester into acid and alkyl (OR) fragments
    acid_atoms, alkyl_atoms = parse_ester_fragments(mol, ester_atoms)
    if not alkyl_atoms:
        return None

    if ester_o is None:
        return None

    # Early return for aromatic alkyl: phenoxycarbonyl, (benzyloxy)carbonyl
    first_alkyl = alkyl_atoms[0]
    first_atom = mol.GetAtomWithIdx(first_alkyl)

    if first_atom.GetIsAromatic():
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if first_alkyl in ring and len(ring) == 6:
                if all(mol.GetAtomWithIdx(r).GetIsAromatic()
                       and mol.GetAtomWithIdx(r).GetSymbol() == 'C'
                       for r in ring):
                    return "phenoxycarbonyl"
        return "phenoxycarbonyl"

    if (not first_atom.GetIsAromatic()
            and first_atom.GetSymbol() == 'C'
            and first_atom.GetTotalNumHs() >= 1):
        arom_nbrs = [n for n in first_atom.GetNeighbors()
                     if n.GetIdx() != ester_o and n.GetIsAromatic()]
        non_h_non_arom = [n for n in first_atom.GetNeighbors()
                          if n.GetIdx() != ester_o
                          and not n.GetIsAromatic()
                          and n.GetSymbol() != 'H']
        if arom_nbrs and not non_h_non_arom:
            return "(benzyloxy)carbonyl"

    # Guard 3: the alkyl (OR) fragment must be pure carbon (no heteroatoms)
    # for simple alkoxycarbonyl naming.  Heteroatom-containing fragments
    # (e.g., amino acid side chains) need complex naming beyond the scope
    # of this prefix generator.
    has_heteroatom = any(
        mol.GetAtomWithIdx(a).GetSymbol() not in ('C', 'H')
        for a in alkyl_atoms
    )
    if has_heteroatom:
        return None

    # Guard 4: size sanity — if the OR fragment is larger than the principal
    # chain, this ester should be handled by functional-class naming.
    carbon_count = sum(
        1 for a in alkyl_atoms if mol.GetAtomWithIdx(a).GetSymbol() == 'C'
    )
    chain_len = len(principal_chain) if principal_chain else 0
    if chain_len > 0 and carbon_count > chain_len:
        return None
    # Reject non-aromatic ring-containing alkyl fragments (macrocyclic esters).
    # Aromatic rings (phenyl) are already handled above.
    ring_info = mol.GetRingInfo()
    has_non_aromatic_ring = any(
        ring_info.NumAtomRings(a) > 0 and not mol.GetAtomWithIdx(a).GetIsAromatic()
        for a in alkyl_atoms
    )
    if has_non_aromatic_ring:
        return None

    if carbon_count == 0:
        return None

    # Build the alkoxycarbonyl name from ALKOXY_NAMES (same table as ethers)
    if carbon_count in ALKOXY_NAMES:
        alkoxy = ALKOXY_NAMES[carbon_count]
        return f"{alkoxy}carbonyl"

    # For larger/unknown sizes, build from alkyl name
    try:
        alkyl_name = get_alkyl_name(carbon_count)
        if alkyl_name.endswith("yl"):
            return f"{alkyl_name[:-2]}yloxy" + "carbonyl"
        return f"{alkyl_name}oxy" + "carbonyl"
    except (ValueError, KeyError):
        try:
            from ..data.chain_names import get_chain_prefix
            prefix = get_chain_prefix(carbon_count)
            return f"{prefix}yloxycarbonyl"
        except (ValueError, KeyError):
            return None


def _get_sulfinyl_prefix(
    mol,
    sulfoxide_atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """Generate (alkyl)sulfinyl prefix for sulfoxide as non-principal group.

    IUPAC P-63.6: R-S(=O)-R' when not the principal group is expressed as
    an (alkyl)sulfinyl prefix on the parent chain.

    SMARTS "[SX3](=[OX1])([#6])[#6]" matches (S, O, C1, C2).

    Args:
        mol: RDKit Mol object.
        sulfoxide_atoms: Atom indices from sulfoxide SMARTS match.
        principal_chain: Atom indices of the principal chain.

    Returns:
        Compound prefix string (e.g., "methylsulfinyl"), or None on failure.
    """
    if len(sulfoxide_atoms) < 3:
        return None

    sulfur_idx = sulfoxide_atoms[0]
    chain_set = set(principal_chain)

    # Find the two C neighbors of S (skip O neighbors)
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    c_neighbors = [n for n in sulfur.GetNeighbors()
                   if n.GetSymbol() == 'C']
    if len(c_neighbors) < 2:
        return None

    # Determine which C is on the chain vs substituent
    c1, c2 = c_neighbors[0].GetIdx(), c_neighbors[1].GetIdx()
    c1_on_chain = c1 in chain_set
    c2_on_chain = c2 in chain_set

    if c1_on_chain and not c2_on_chain:
        sub_carbon = c2
    elif c2_on_chain and not c1_on_chain:
        sub_carbon = c1
    else:
        # Neither or both on chain -- use smaller fragment
        frag1 = _count_fragment_atoms(mol, c1, {sulfur_idx})
        frag2 = _count_fragment_atoms(mol, c2, {sulfur_idx})
        sub_carbon = c1 if frag1 <= frag2 else c2

    # Count carbons in substituent fragment
    carbon_count = _count_fragment_atoms(
        mol, sub_carbon, {sulfur_idx}, carbons_only=True
    )
    if carbon_count == 0:
        return None

    # Build compound prefix: methylsulfinyl, ethylsulfinyl, etc.
    try:
        alkyl = get_alkyl_name(carbon_count)
        return f"{alkyl}sulfinyl"
    except (ValueError, KeyError):
        return None


def _get_sulfonyl_prefix(
    mol,
    sulfone_atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """Generate (alkyl)sulfonyl prefix for sulfone as non-principal group.

    IUPAC P-63.6: R-S(=O)(=O)-R' when not the principal group is expressed
    as an (alkyl)sulfonyl prefix on the parent chain.

    SMARTS "[SX4](=[OX1])(=[OX1])([#6])[#6]" matches (S, O1, O2, C1, C2).

    Args:
        mol: RDKit Mol object.
        sulfone_atoms: Atom indices from sulfone SMARTS match.
        principal_chain: Atom indices of the principal chain.

    Returns:
        Compound prefix string (e.g., "methylsulfonyl"), or None on failure.
    """
    if len(sulfone_atoms) < 3:
        return None

    sulfur_idx = sulfone_atoms[0]
    chain_set = set(principal_chain)

    # Find the two C neighbors of S (skip O neighbors)
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    c_neighbors = [n for n in sulfur.GetNeighbors()
                   if n.GetSymbol() == 'C']
    if len(c_neighbors) < 2:
        return None

    # Determine which C is on the chain vs substituent
    c1, c2 = c_neighbors[0].GetIdx(), c_neighbors[1].GetIdx()
    c1_on_chain = c1 in chain_set
    c2_on_chain = c2 in chain_set

    if c1_on_chain and not c2_on_chain:
        sub_carbon = c2
    elif c2_on_chain and not c1_on_chain:
        sub_carbon = c1
    else:
        # Neither or both on chain -- use smaller fragment
        frag1 = _count_fragment_atoms(mol, c1, {sulfur_idx})
        frag2 = _count_fragment_atoms(mol, c2, {sulfur_idx})
        sub_carbon = c1 if frag1 <= frag2 else c2

    # Count carbons in substituent fragment
    carbon_count = _count_fragment_atoms(
        mol, sub_carbon, {sulfur_idx}, carbons_only=True
    )
    if carbon_count == 0:
        return None

    # Build compound prefix: methylsulfonyl, ethylsulfonyl, etc.
    try:
        alkyl = get_alkyl_name(carbon_count)
        return f"{alkyl}sulfonyl"
    except (ValueError, KeyError):
        return None


def _get_sulfanyl_prefix(
    mol,
    thioether_atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """Generate (alkyl)sulfanyl prefix for thioether as non-principal group.

    IUPAC P-63.2.5: R-S-R' when not the principal group is expressed as
    an (alkyl)sulfanyl prefix on the parent chain.

    SMARTS "[SX2]([#6])[#6]" matches (S, C1, C2).

    Args:
        mol: RDKit Mol object.
        thioether_atoms: Atom indices from thioether SMARTS match.
        principal_chain: Atom indices of the principal chain.

    Returns:
        Compound prefix string (e.g., "methylsulfanyl"), or None on failure.
    """
    if len(thioether_atoms) < 3:
        return None

    sulfur_idx = thioether_atoms[0]
    chain_set = set(principal_chain)

    # Find the two C neighbors of S
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    c_neighbors = [n for n in sulfur.GetNeighbors()
                   if n.GetSymbol() == 'C']
    if len(c_neighbors) < 2:
        return None

    # Determine which C is on the chain vs substituent
    c1, c2 = c_neighbors[0].GetIdx(), c_neighbors[1].GetIdx()
    c1_on_chain = c1 in chain_set
    c2_on_chain = c2 in chain_set

    if c1_on_chain and not c2_on_chain:
        sub_carbon = c2
    elif c2_on_chain and not c1_on_chain:
        sub_carbon = c1
    else:
        # Neither or both on chain -- use smaller fragment
        frag1 = _count_fragment_atoms(mol, c1, {sulfur_idx})
        frag2 = _count_fragment_atoms(mol, c2, {sulfur_idx})
        sub_carbon = c1 if frag1 <= frag2 else c2

    # Count carbons in substituent fragment
    carbon_count = _count_fragment_atoms(
        mol, sub_carbon, {sulfur_idx}, carbons_only=True
    )
    if carbon_count == 0:
        return None

    # Build compound prefix: methylsulfanyl, ethylsulfanyl, etc.
    try:
        alkyl = get_alkyl_name(carbon_count)
        return f"{alkyl}sulfanyl"
    except (ValueError, KeyError):
        return None


def _count_fragment_atoms(
    mol,
    start_atom: int,
    exclude: Set[int],
    carbons_only: bool = False
) -> int:
    """
    Count atoms in a fragment via BFS.

    Args:
        mol: RDKit Mol object
        start_atom: Starting atom index
        exclude: Atoms to not cross (boundary)
        carbons_only: If True, only count carbon atoms

    Returns:
        Number of atoms (or carbons if carbons_only=True)
    """
    from collections import deque

    visited = set()
    queue = deque([start_atom])
    count = 0

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        if carbons_only:
            if atom.GetSymbol() == 'C':
                count += 1
        else:
            count += 1

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude:
                queue.append(nbr_idx)

    return count


def format_fg_prefix(prefix_form: str, locants: List[int], count: int) -> str:
    """
    Format functional group prefix with locants and multiplier.

    Per IUPAC P-16.3.3, compound substituent prefixes (e.g., methylsulfinyl,
    methylsulfonyl) are enclosed in parentheses when used with locants.
    Simple prefixes (hydroxy, oxo, amino) are not parenthesized.

    Args:
        prefix_form: Base prefix name (e.g., "hydroxy", "oxo", "methoxy")
        locants: List of locant positions
        count: Number of instances

    Returns:
        Formatted prefix string (e.g., "2-hydroxy", "3-oxo", "2-(methylsulfinyl)")
    """
    from ..assembly.naming_utils import needs_brackets

    if not locants:
        # No locants - just return prefix with multiplier if needed
        if count > 1:
            multiplier = get_multiplier_prefix(count, prefix_form)
            return f"{multiplier}{prefix_form}"
        return prefix_form

    # Format locants
    locant_str = ",".join(str(loc) for loc in sorted(locants))

    # Check if prefix is a compound substituent needing parentheses (P-16.3.3)
    compound = needs_brackets(prefix_form)

    # Get multiplier if multiple instances
    if count > 1:
        multiplier = get_multiplier_prefix(count, prefix_form)
        if compound:
            return f"{locant_str}-{multiplier}({prefix_form})"
        return f"{locant_str}-{multiplier}{prefix_form}"

    if compound:
        return f"{locant_str}-({prefix_form})"
    return f"{locant_str}-{prefix_form}"


def _find_fg_center_atom(mol, match: tuple, fg_name: str) -> Optional[int]:
    """
    Find the functional group CENTER atom in a SMARTS match.

    Different FGs have different definitions of "center":
    - Alcohols/thiols: The carbon attached to -OH/-SH (typically index 1 in SMARTS)
    - Ketones: The carbonyl carbon (typically index 1 in SMARTS)
    - Aldehydes: The aldehyde carbon (typically index 0 in SMARTS)

    For most FGs, the center is the FIRST carbon in the match that is
    directly bonded to a heteroatom that is ALSO in the match.

    Args:
        mol: RDKit Mol object
        match: Tuple of atom indices from SMARTS match
        fg_name: Name of the functional group

    Returns:
        Atom index of the FG center, or None if not found
    """
    match_set = set(match)

    # For each carbon in match, check if it's bonded to a heteroatom in match
    for atom_idx in match:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != 'C':
            continue

        # Check if this carbon is bonded to a heteroatom that's also in the match
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in match_set and neighbor.GetSymbol() not in ('C', 'H'):
                # This carbon is directly bonded to a heteroatom in the FG match
                return atom_idx

    # Fallback: return first carbon in match
    for atom_idx in match:
        if mol.GetAtomWithIdx(atom_idx).GetSymbol() == 'C':
            return atom_idx

    return None


def get_non_principal_fg_locants(
    mol,
    fg_atoms: List[tuple],
    principal_chain: List[int],
    atom_to_locant: Dict[int, int],
    fg_name: str = ""
) -> List[int]:
    """
    Get locants for non-principal functional groups.

    For each FG match, find the functional group CENTER atom (the carbon
    bearing the heteroatom) on the principal chain and return its locant.

    Args:
        mol: RDKit Mol object
        fg_atoms: List of atom index tuples for each FG match
        principal_chain: Atom indices of the principal chain
        atom_to_locant: Mapping from atom index to locant
        fg_name: Name of the functional group (for specialized handling)

    Returns:
        Sorted list of locants for the functional group positions
    """
    locants = []

    for match in fg_atoms:
        # Find the FG center atom
        center_idx = _find_fg_center_atom(mol, match, fg_name)

        if center_idx is not None and center_idx in atom_to_locant:
            locants.append(atom_to_locant[center_idx])
        else:
            # Fallback: find any atom in match that's on chain
            for atom_idx in match:
                if atom_idx in atom_to_locant:
                    locants.append(atom_to_locant[atom_idx])
                    break
            else:
                # Check if any atom's neighbor is on the chain
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for neighbor in atom.GetNeighbors():
                        nbr_idx = neighbor.GetIdx()
                        if nbr_idx in atom_to_locant:
                            locants.append(atom_to_locant[nbr_idx])
                            break
                    else:
                        continue
                    break

    # Do NOT deduplicate locants: two groups at the same position need
    # repeated locants (e.g., 3,3-diamino). The count and locant list
    # must agree for OPSIN compatibility.
    return sorted(locants)


def get_non_principal_groups(
    functional_groups: Dict[str, List[tuple]],
    principal_group: Optional[str]
) -> Dict[str, List[tuple]]:
    """
    Get all functional groups that are not the principal group.

    Filters out the principal group and unsaturation markers.

    Args:
        functional_groups: Dict from detect_functional_groups()
        principal_group: Name of the principal group (or None)

    Returns:
        Dict of non-principal functional group names to their atom indices
    """
    exclude_groups = {"alkene", "alkyne"}  # Handled by unsaturation

    result = {}
    for fg_name, matches in functional_groups.items():
        if fg_name == principal_group:
            continue
        if fg_name in exclude_groups:
            continue
        if matches:
            result[fg_name] = matches

    return result


def _name_ring_as_parent_polyfunctional(features: Any) -> Optional[str]:
    """Name a polyfunctional compound where the ring is the parent structure.

    Handles the case where ``principal_chain`` is None because the molecule
    uses a ring as its parent hydride. Builds the name from:
    1. Ring parent name (from oriented ring or ring detection)
    2. Principal group suffix with locants on the ring
    3. Non-principal group prefixes with ring locants
    4. Substituent prefixes via the universal pipeline

    IUPAC 2013 P-31 through P-43.

    Args:
        features: MolecularFeatures object (must have is_cyclic=True,
            principal_group set).

    Returns:
        Complete IUPAC name, or None if ring parent cannot be determined.
    """
    mol = features.mol
    principal_group = features.principal_group
    non_principal = getattr(features, 'non_principal_groups', {})

    if not principal_group:
        return None

    # --- Step 1: Identify ring parent ---
    ring_info = mol.GetRingInfo()
    all_rings = ring_info.AtomRings()
    if not all_rings:
        return None

    # Use oriented_ring if available, else find the largest ring
    oriented_ring = getattr(features, 'oriented_ring', None)
    if oriented_ring:
        ring_atoms = list(oriented_ring)
    else:
        # Pick the largest ring as parent
        ring_atoms = list(max(all_rings, key=len))

    ring_set = set(ring_atoms)
    ring_size = len(ring_atoms)

    # --- Step 2: Build ring parent name ---
    ring_parent_name = _get_ring_parent_name(mol, ring_atoms)
    if not ring_parent_name:
        return None

    # --- Step 3: Build locant mapping for ring atoms ---
    # Use features.atom_to_locant if available, else build from ring order
    atom_to_locant = features.atom_to_locant
    if not atom_to_locant:
        atom_to_locant = {}
        for i, idx in enumerate(ring_atoms):
            atom_to_locant[idx] = i + 1

    # --- Step 4: Generate principal group suffix with locants ---
    from .seniority import get_suffix
    suffix = get_suffix(principal_group, is_ring=True)
    if not suffix:
        # Try chain suffix as fallback
        suffix = get_suffix(principal_group, is_ring=False)
    if not suffix:
        return None

    suffix_locants = []
    if features.principal_group_atoms:
        for match in features.principal_group_atoms:
            for atom_idx in match:
                if atom_idx in atom_to_locant:
                    suffix_locants.append(atom_to_locant[atom_idx])
                    break
                # For FGs attached TO the ring (e.g., -COOH on cyclohexane),
                # the center atom may not be in the ring; check neighbors
                atom = mol.GetAtomWithIdx(atom_idx)
                for nbr in atom.GetNeighbors():
                    nidx = nbr.GetIdx()
                    if nidx in atom_to_locant and nidx in ring_set:
                        suffix_locants.append(atom_to_locant[nidx])
                        break
    suffix_locants = sorted(set(suffix_locants))

    # Determine multiplier for multiple principal groups
    count = len(suffix_locants) if suffix_locants else 1
    multiplier = get_multiplier_prefix(count, suffix) if count > 1 else ""

    # --- Step 5: Generate non-principal FG prefixes with ring locants ---
    all_prefixes = []
    for fg_name, matches in non_principal.items():
        if not matches:
            continue
        if fg_name in NO_SENIORITY_GROUPS and fg_name not in ('alkene', 'alkyne'):
            # Named as prefix if it has a prefix form
            prefix_form = get_fg_prefix_form(fg_name, mol, matches[0], None)
            if prefix_form:
                fg_locants = []
                for match in matches:
                    for atom_idx in match:
                        if atom_idx in atom_to_locant:
                            fg_locants.append(atom_to_locant[atom_idx])
                            break
                        atom = mol.GetAtomWithIdx(atom_idx)
                        for nbr in atom.GetNeighbors():
                            nidx = nbr.GetIdx()
                            if nidx in atom_to_locant and nidx in ring_set:
                                fg_locants.append(atom_to_locant[nidx])
                                break
                fg_locants = sorted(set(fg_locants))
                fg_count = len(fg_locants) if fg_locants else len(matches)
                formatted = format_fg_prefix(prefix_form, fg_locants, fg_count)
                all_prefixes.append(formatted)
            continue

        # Seniority-bearing non-principal groups
        prefix_form = get_prefix(fg_name)
        if prefix_form:
            fg_locants = []
            for match in matches:
                for atom_idx in match:
                    if atom_idx in atom_to_locant:
                        fg_locants.append(atom_to_locant[atom_idx])
                        break
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for nbr in atom.GetNeighbors():
                        nidx = nbr.GetIdx()
                        if nidx in atom_to_locant and nidx in ring_set:
                            fg_locants.append(atom_to_locant[nidx])
                            break
            fg_locants = sorted(set(fg_locants))
            fg_count = len(fg_locants) if fg_locants else len(matches)
            formatted = format_fg_prefix(prefix_form, fg_locants, fg_count)
            all_prefixes.append(formatted)

    # --- Step 6: Discover substituents on the ring via universal pipeline ---
    # Collect atoms consumed by the principal group and non-principal FGs
    consumed_atoms = set()
    if features.principal_group_atoms:
        for match in features.principal_group_atoms:
            for idx in match:
                if idx not in ring_set:
                    consumed_atoms.add(idx)
    for fg_name, matches in non_principal.items():
        for match in matches:
            for idx in match:
                if idx not in ring_set:
                    consumed_atoms.add(idx)

    try:
        from ..assembly.composer import _integrate_universal_prefixes
        sub_prefix_str = _integrate_universal_prefixes(
            mol, ring_set,
            parent_type="ring",
            oriented_ring=ring_atoms,
            atom_to_locant=atom_to_locant,
            exclude_atoms=consumed_atoms,
        )
        if sub_prefix_str:
            all_prefixes.append(sub_prefix_str)
    except Exception as exc:
        logger.debug("Ring-as-parent universal prefix failed: %s", exc)

    # Sort prefixes alphabetically
    all_prefixes.sort(key=alpha_sort_key)

    # --- Step 7: Assemble the complete name ---
    # Format: [prefixes]-[ring_parent]-[suffix_locants]-[multiplier][suffix]
    # e.g., "4-hydroxy-cyclohexan-1-one"
    formatted_suffix = format_suffix_with_locants(
        ring_parent_name, "", suffix, suffix_locants, multiplier
    )

    if all_prefixes:
        prefix_str = _join_prefixes(all_prefixes)
        if prefix_str and formatted_suffix and prefix_str[-1].isalpha() and formatted_suffix[0].isdigit():
            name = f"{prefix_str}-{formatted_suffix}"
        else:
            name = f"{prefix_str}{formatted_suffix}"
    else:
        name = formatted_suffix

    # Add stereodescriptors if present
    if features.stereocenters or getattr(features, 'double_bond_stereo', None):
        from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        if descriptors:
            stereo_prefix = format_stereodescriptor_string(descriptors)
            name = f"{stereo_prefix}{name}"

    return name


def _get_ring_parent_name(mol, ring_atoms: list) -> Optional[str]:
    """Get the parent hydride name for a ring.

    Checks retained names first (benzene, pyridine, etc.), then falls
    back to systematic cyclo- naming.

    Args:
        mol: RDKit Mol object.
        ring_atoms: List of atom indices forming the ring.

    Returns:
        Ring parent stem (e.g., ``"cyclohexan"``, ``"benzene"``),
        or None if not determinable.
    """
    ring_set = set(ring_atoms)
    ring_size = len(ring_atoms)

    # Check if ring is all-carbon aromatic (benzene)
    all_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_atoms)
    all_carbon = all(mol.GetAtomWithIdx(idx).GetSymbol() == 'C' for idx in ring_atoms)

    if ring_size == 6 and all_aromatic and all_carbon:
        return "benzene"

    # Check for heterocyclic retained names
    has_heteroatom = any(
        mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
        for idx in ring_atoms
    )
    if has_heteroatom:
        try:
            from ..rules.heterocycles import classify_heterocycle
            het_info = classify_heterocycle(mol, tuple(ring_atoms))
            if het_info and het_info.get('name'):
                return het_info['name']
        except Exception:
            pass

    # Systematic cyclo- naming for carbocyclic rings
    if all_carbon:
        from ..data.chain_names import get_chain_prefix
        stem = get_chain_prefix(ring_size)
        if stem:
            # Return stem without "ane" suffix -- format_suffix_with_locants
            # will add the appropriate suffix. For cycloalkanes with
            # functional groups, we need "cyclohexan" not "cyclohexane".
            return f"cyclo{stem}an"

    return None


def name_polyfunctional(features: Any) -> Optional[str]:
    """
    Generate IUPAC name for a polyfunctional compound.

    This is the main coordinator function for multi-FG naming.
    If the compound is not polyfunctional, returns None.

    Args:
        features: MolecularFeatures object with extracted features

    Returns:
        Complete IUPAC name string, or None if not polyfunctional
    """
    # Check if this is a polyfunctional compound
    if not getattr(features, 'is_polyfunctional', False):
        return None

    mol = features.mol
    principal_chain = features.principal_chain
    atom_to_locant = features.atom_to_locant
    principal_group = features.principal_group
    non_principal = getattr(features, 'non_principal_groups', {})

    if not principal_chain or not atom_to_locant:
        # --- Phase 86-02: Ring-as-parent polyfunctional path ---
        # When principal_chain is None but the molecule is cyclic, attempt
        # ring-as-parent naming. Conservative guards prevent intercepting
        # compounds better handled by specialized fallthrough handlers:
        # - Only monocyclic (1 ring) -- fused/polycyclic have specialized handlers
        # - Only fully saturated rings -- unsaturated rings need -ene/-yne handling
        # - No aromatic rings -- benzene/pyridine/etc. have retained name handlers
        if getattr(features, 'is_cyclic', False) and principal_group:
            ring_info = mol.GetRingInfo()
            n_rings = ring_info.NumRings()
            if n_rings == 1:
                ring_atoms = list(ring_info.AtomRings()[0])
                ring_set = set(ring_atoms)
                ring_size = len(ring_atoms)
                total_heavy = mol.GetNumHeavyAtoms()

                # Guard: ring must be a significant portion of the molecule.
                # If the ring is < 40% of heavy atoms, it's likely a
                # substituent on a chain parent, not the parent itself.
                # E.g., sphingolipid with cyclohexane ring but 50+ chain atoms.
                if total_heavy > 0 and ring_size / total_heavy < 0.35:
                    pass  # Fall through to return None
                else:
                    # Check ring is fully saturated and non-aromatic
                    has_ring_double = False
                    is_aromatic = False
                    for idx in ring_atoms:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetIsAromatic():
                            is_aromatic = True
                            break
                        for bond in atom.GetBonds():
                            other = bond.GetOtherAtomIdx(idx)
                            if (other in ring_set
                                    and bond.GetBondTypeAsDouble() == 2.0):
                                has_ring_double = True
                                break
                        if has_ring_double:
                            break

                    if not is_aromatic and not has_ring_double:
                        ring_name = _name_ring_as_parent_polyfunctional(
                            features,
                        )
                        if ring_name:
                            return ring_name
        return None

    # --- EL-02: Ester demotion in polyfunctional context ---
    # When ester is the principal group in a polyfunctional compound, it should
    # NOT use "-oate" suffix. Instead, demote esters to acyloxy prefixes and
    # re-select the next-highest seniority group as the principal group.
    ester_acyloxy_prefixes = []
    _esters_demoted = False
    if principal_group == "ester":
        from ..rules.esters import name_ester_as_prefix
        ester_matches = features.functional_groups.get("ester", [])
        for match in ester_matches:
            acyloxy = name_ester_as_prefix(mol, match)
            if acyloxy:
                # Find the alkyl carbon (last in match) on the principal chain
                alkyl_c = match[-1] if len(match) >= 4 else match[3] if len(match) > 3 else None
                locant = atom_to_locant.get(alkyl_c) if alkyl_c is not None else None
                ester_acyloxy_prefixes.append((acyloxy, locant))

        if ester_acyloxy_prefixes:
            _esters_demoted = True

        # Re-select principal group excluding esters
        filtered_fgs = {k: v for k, v in features.functional_groups.items() if k != "ester"}
        new_principal, new_atoms = get_principal_group(mol, filtered_fgs)
        if new_principal:
            principal_group = new_principal
            features.principal_group_atoms = new_atoms
            # Update non_principal to exclude the new principal group
            non_principal = {k: v for k, v in filtered_fgs.items()
                            if k != new_principal and k not in ("alkene", "alkyne")}

    # Collect all prefixes (FG prefixes + alkyl substituents)
    all_prefixes = []

    # Add ester acyloxy prefixes if esters were demoted
    if ester_acyloxy_prefixes:
        # Group identical acyloxy prefixes for multipliers
        from collections import Counter as _Ctr
        acyloxy_groups: Dict[str, List] = defaultdict(list)
        for acyloxy_name, locant in ester_acyloxy_prefixes:
            acyloxy_groups[acyloxy_name].append(locant)

        for acyloxy_name, locants in sorted(acyloxy_groups.items()):
            valid_locants = sorted(loc for loc in locants if loc is not None)
            count = len(locants)
            if count > 1 and valid_locants:
                locant_str = ",".join(str(loc) for loc in valid_locants)
                multiplier = get_multiplier_prefix(count, acyloxy_name)
                all_prefixes.append(f"{locant_str}-{multiplier}({acyloxy_name})")
            elif valid_locants:
                all_prefixes.append(f"{valid_locants[0]}-({acyloxy_name})")
            else:
                if count > 1:
                    multiplier = get_multiplier_prefix(count, acyloxy_name)
                    all_prefixes.append(f"{multiplier}({acyloxy_name})")
                else:
                    all_prefixes.append(f"({acyloxy_name})")

    # --- Generate FG prefixes from non-principal groups ---
    chain_set = set(principal_chain)
    ring_fg_groups = defaultdict(list)  # FGs on ring atoms, for ring substituent naming

    # Collect substituent branch atoms for FG-on-branch filtering (BUG-B).
    # FGs located entirely on a substituent branch are already named by the
    # substituent naming path (e.g., hydroxymethyl), so skip them here.
    _branch_atoms = set()
    if features.substituents:
        for _pos, sub_list in features.substituents.items():
            for sub_atoms in sub_list:
                _branch_atoms.update(sub_atoms)

    for fg_name, matches in non_principal.items():
        if not matches:
            continue

        # When chain is parent, separate FGs on ring from FGs on chain
        if getattr(features, 'chain_is_parent', False):
            ring_atom_set = set()
            for rg in getattr(features, 'ring_substituents_as_groups', []):
                ring_atom_set.update(rg)

            chain_matches = []
            for match in matches:
                # Check if this FG's center atom is on the chain
                center = _find_fg_center_atom(mol, match, fg_name)
                if center is not None and center in chain_set and center not in ring_atom_set:
                    chain_matches.append(match)
                elif any(a in chain_set and a not in ring_atom_set for a in match):
                    chain_matches.append(match)
                else:
                    # FG is on a ring - track for ring substituent naming
                    ring_fg_groups[fg_name].append(match)
            matches = chain_matches
            if not matches:
                continue

        # BUG-B guard: Skip FG matches located entirely on a small substituent
        # branch (<=3 carbons) when the substituent naming path demonstrably
        # handles them (producing e.g. "hydroxymethyl", "chloromethyl").
        # IUPAC P-59.1(a): all non-principal FGs must appear as prefixes.
        # This guard prevents DOUBLE-naming (both substituent prefix AND standalone
        # FG prefix for the same group).
        #
        # Verified empirically (Phase 105-01): each FG type below produces the
        # correct prefix via substituent naming on 1-3C branches:
        #   - Halogens: "fluoromethyl", "chloromethyl", "bromomethyl", "iodomethyl"
        #   - primary_alcohol: "hydroxymethyl" on -CH2OH
        #   - secondary_alcohol: "hydroxy" included in branch name
        #   - primary_amine: "aminomethyl" on -CH2NH2
        # FG types NOT verified safe must NOT be added to this set.
        _BRANCH_HANDLED_FGS = {
            'primary_alcohol', 'secondary_alcohol', 'primary_amine',
            'fluoro', 'chloro', 'bromo', 'iodo',
        }
        if _branch_atoms and fg_name in _BRANCH_HANDLED_FGS:
            filtered_matches = []
            for match in matches:
                if not all(a in _branch_atoms for a in match):
                    filtered_matches.append(match)
                    continue
                # Check if this FG is on a small branch with carbons
                # (i.e., a branch where _name_heteroatom_substituent handles the FG)
                on_small_branch = False
                for _pos, sub_list in features.substituents.items():
                    for sub_atoms in sub_list:
                        sub_set = set(sub_atoms)
                        if all(a in sub_set for a in match):
                            # This branch contains the entire FG match
                            c_count = sum(1 for a in sub_atoms
                                          if mol.GetAtomWithIdx(a).GetSymbol() == 'C')
                            if 1 <= c_count <= 3:
                                on_small_branch = True
                                break
                    if on_small_branch:
                        break
                if not on_small_branch:
                    filtered_matches.append(match)
            original_count = len(non_principal[fg_name])
            matches = filtered_matches
            if not matches:
                if original_count > 0:
                    logger.debug(
                        "DROP-17 polyfunc_bugb: fg_name=%s filtered=%d "
                        "(substituent naming handles these on small branches)",
                        fg_name, original_count,
                    )
                continue

        # Get prefix form for this FG
        prefix_form = get_fg_prefix_form(
            fg_name, mol, matches[0], principal_chain
        )
        if not prefix_form:
            # By-design: FGs using functional class naming (ester→alkoxycarbonyl,
            # secondary_amide→acylamino, thioether, etc.) are handled by
            # specialized naming paths, not as simple prefixes
            logger.debug(
                "DROP-23 substituent_skip: reason=no_fg_prefix_form fg_name=%s",
                fg_name,
            )
            continue

        # Get locants for this FG
        locants = get_non_principal_fg_locants(
            mol, matches, principal_chain, atom_to_locant, fg_name
        )

        # Reconcile count with locants: if we found fewer locants than
        # matches, some FG instances are off-chain (inside substituent branches)
        # and should not inflate the multiplier. Use locant count as truth.
        count = len(matches)
        if locants:
            count = len(locants)

        # Format the prefix
        formatted = format_fg_prefix(prefix_form, locants, count)
        all_prefixes.append(formatted)

    # --- Generate ring substituent prefixes (when chain is parent) ---
    if getattr(features, 'chain_is_parent', False):
        from ..assembly.composer import _generate_ring_substituent_prefixes
        ring_prefixes = _generate_ring_substituent_prefixes(features)
        for ring_prefix in ring_prefixes:
            all_prefixes.append(ring_prefix.text)

    # --- Generate alkyl substituent prefixes ---
    if features.substituents:
        alkyl_prefixes = _generate_alkyl_prefixes_for_polyfunctional(
            features, skip_acyloxy=_esters_demoted
        )
        all_prefixes.extend(alkyl_prefixes)

    # --- Merge duplicate bare prefix names ---
    # When two FG detection paths (e.g., primary_alcohol and secondary_alcohol)
    # both produce the same bare prefix form ("hydroxy"), merge into one entry
    # with the correct multiplier ("dihydroxy"). Only bare (no-locant) prefixes
    # are merged; locanted prefixes represent distinct chain positions.
    all_prefixes = _merge_bare_duplicate_prefixes(all_prefixes)

    # Sort all prefixes alphabetically
    all_prefixes.sort(key=alpha_sort_key)

    # --- Build the name ---
    from ..data.chain_names import get_chain_prefix

    chain_length = len(principal_chain)
    stem = get_chain_prefix(chain_length)

    # Get suffix for principal group
    from .seniority import get_suffix
    from ..rules.locants import get_functional_group_locants

    suffix = get_suffix(principal_group, is_ring=False)
    if not suffix:
        return None

    # Get locants for principal group
    suffix_locants = []
    if features.principal_group_atoms:
        suffix_locants = get_functional_group_locants(
            principal_chain,
            features.principal_group_atoms,
            atom_to_locant,
            mol=mol
        )
        # Deduplicate locants (overlapping SMARTS can produce duplicates)
        suffix_locants = sorted(set(suffix_locants))

    # For terminal groups (acid, aldehyde), locant is implicit
    from ..assembly.composer import TERMINAL_GROUPS
    if principal_group in TERMINAL_GROUPS:
        suffix_locants = []

    # Determine multiplier for multiple principal groups
    # Validate: suffix count cannot exceed parent chain/ring capacity
    count = len(features.principal_group_atoms)
    max_capacity = chain_length
    if count > max_capacity:
        count = max_capacity
    # If we have unique locants, use those as the count (more reliable)
    if suffix_locants:
        count = len(suffix_locants)
    multiplier = get_multiplier_prefix(count, suffix) if count > 1 else ""

    # Build unsaturation infix
    from ..assembly.composer import _build_unsaturation_infix
    from ..rules.locants import get_bond_locants

    double_locants = get_bond_locants(
        principal_chain, features.double_bonds, atom_to_locant
    )
    triple_locants = get_bond_locants(
        principal_chain, features.triple_bonds, atom_to_locant
    )
    unsaturation = _build_unsaturation_infix(double_locants, triple_locants)

    # Detect suffix-prefix locant collisions (safety net for ring parents)
    if suffix_locants and all_prefixes:
        import re as _re
        from .locant_validation import detect_locant_collisions

        # Extract prefix locants from formatted prefix strings
        prefix_locant_groups = []
        for ptext in all_prefixes:
            match = _re.match(r'^([\d,]+)-', ptext)
            if match:
                try:
                    locs = [int(x) for x in match.group(1).split(',')]
                    prefix_locant_groups.append(locs)
                except ValueError:
                    pass

        is_ring = getattr(features, 'is_cyclic', False)
        if prefix_locant_groups:
            collisions = detect_locant_collisions(
                suffix_locants,
                prefix_locant_groups,
                parent_type="ring" if is_ring else "chain",
            )
            if collisions:
                collision_set = set(loc for _, loc in collisions)
                # Remove colliding prefix locants from affected prefix strings
                cleaned = []
                for ptext in all_prefixes:
                    match = _re.match(r'^([\d,]+)-(.+)$', ptext)
                    if match:
                        locs = [int(x) for x in match.group(1).split(',') if int(x) not in collision_set]
                        name_part = match.group(2)
                        if locs:
                            cleaned.append(f"{','.join(str(l) for l in locs)}-{name_part}")
                        else:
                            cleaned.append(name_part)
                    else:
                        cleaned.append(ptext)
                all_prefixes = cleaned

    # Assemble the name
    name = format_suffix_with_locants(
        stem, unsaturation, suffix, suffix_locants, multiplier
    )

    # Add prefixes with proper hyphenation at boundary
    if all_prefixes:
        prefix_str = _join_prefixes(all_prefixes)
        # Ensure hyphen between prefix ending with letter and name starting with digit
        if prefix_str and name and prefix_str[-1].isalpha() and name[0].isdigit():
            name = f"{prefix_str}-{name}"
        else:
            name = f"{prefix_str}{name}"

    # Add stereodescriptors if present
    if features.stereocenters or getattr(features, 'double_bond_stereo', None):
        from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        if descriptors:
            stereo_prefix = format_stereodescriptor_string(descriptors)
            name = f"{stereo_prefix}{name}"

    return name


def _generate_alkyl_prefixes_for_polyfunctional(
    features: Any, skip_acyloxy: bool = False
) -> List[str]:
    """
    Generate alkyl substituent prefixes for polyfunctional compounds.

    Similar to _generate_alkyl_prefixes in composer.py but returns
    just the formatted strings for combination with FG prefixes.

    Args:
        features: MolecularFeatures object
        skip_acyloxy: If True, skip acyloxy detection (esters already handled
                      by ester demotion in name_polyfunctional)
    """
    from collections import defaultdict

    mol = features.mol
    substituent_groups: Dict[str, List[int]] = defaultdict(list)

    # Collect ring atoms that should be skipped (handled by ring substituent prefixes)
    ring_atoms_to_skip: set = set()
    if getattr(features, 'chain_is_parent', False):
        ring_groups = getattr(features, 'ring_substituents_as_groups', [])
        for ring_atoms in ring_groups:
            ring_atoms_to_skip.update(ring_atoms)

    for position, sub_list in features.substituents.items():
        for sub_atoms in sub_list:
            # Skip substituents whose ring atoms are handled by
            # _generate_ring_substituent_prefixes (ring + its own substituents).
            # A substituent is a "ring substituent" if its attachment atom
            # (the atom bonded to the principal chain) is IN the ring.
            # Substituents that contain rings deeper in the branch (e.g.,
            # benzoylamino where N attaches to chain) are NOT skipped.
            if ring_atoms_to_skip:
                _chain_set_local = set(features.principal_chain) if features.principal_chain else set()
                _attach_in_ring = False
                for _si in sub_atoms:
                    if _si in ring_atoms_to_skip:
                        _sa = mol.GetAtomWithIdx(_si)
                        for _nb in _sa.GetNeighbors():
                            if _nb.GetIdx() in _chain_set_local:
                                _attach_in_ring = True
                                break
                    if _attach_in_ring:
                        break
                if _attach_in_ring:
                    continue

            # Skip substituent branches entirely covered by a single non-principal
            # FG match ONLY for FG types whose prefix form includes the carbon
            # (carbamoyl, carboxy, chlorocarbonyl, etc.).  These are the FG itself
            # (e.g., -C(=O)NH2 for amide) and are emitted as FG prefixes by the
            # FG prefix loop above.  Processing them here would produce incorrect
            # compound substituent names like "(aminomethyl)".
            # IUPAC P-66.1(c): non-principal amide = carbamoyl prefix.
            # NOTE: Only applies to specific terminal-C FG types. Other FGs
            # (amine, ketone, secondary_amide, etc.) must NOT trigger this guard.
            _POLY_GUARD_FG_TYPES = {
                'primary_amide', 'carboxylic_acid',
                'acid_chloride', 'acid_bromide', 'acid_fluoride',
            }
            _sub_set = set(sub_atoms)
            _skip_fg_branch = False
            _non_principal = getattr(features, 'non_principal_groups', {})
            for _fg_nm, _fg_ms in _non_principal.items():
                if _fg_nm not in _POLY_GUARD_FG_TYPES:
                    continue
                if not _fg_ms:
                    continue
                for _fg_m in _fg_ms:
                    if set(_fg_m) and _sub_set.issubset(set(_fg_m)):
                        _skip_fg_branch = True
                        break
                if _skip_fg_branch:
                    break
            if _skip_fg_branch:
                continue

            # Count only carbon atoms
            carbon_count = sum(
                1 for idx in sub_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )

            # Skip non-alkyl substituents
            if carbon_count == 0:
                continue

            # Check for heteroatoms
            has_heteroatom = any(
                mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
                for idx in sub_atoms
            )

            if has_heteroatom:
                from ..assembly.composer import (
                    _check_for_acylamino,
                    _check_for_acyloxy,
                    _name_heteroatom_substituent,
                )
                # NOTE: Do NOT check alkoxy here - ethers are already handled
                # by the FG prefix system in name_polyfunctional
                het_name = _check_for_acylamino(mol, sub_atoms, features.principal_chain)
                if not het_name and not skip_acyloxy:
                    het_name = _check_for_acyloxy(mol, sub_atoms, features.principal_chain)
                if not het_name:
                    het_name = _name_heteroatom_substituent(mol, sub_atoms, features.principal_chain)
                if not het_name:
                    # Enumerator fallback for simple non-ring, non-ether branches.
                    # Ethers (O-attached) are handled by the FG prefix system.
                    ring_info = mol.GetRingInfo()
                    sub_has_ring = any(ring_info.NumAtomRings(idx) > 0 for idx in sub_atoms)
                    # Find attachment atom
                    _attach = None
                    _chain_set_tmp = set(features.principal_chain) if features.principal_chain else set()
                    for _si in sub_atoms:
                        for _nb in mol.GetAtomWithIdx(_si).GetNeighbors():
                            if _nb.GetIdx() in _chain_set_tmp:
                                _attach = _si
                                break
                        if _attach is not None:
                            break
                    _attach_sym = mol.GetAtomWithIdx(_attach).GetSymbol() if _attach is not None else ''
                    # Skip S-attached branches if S is part of a named FG
                    # (sulfoxide, sulfone, thioether) -- already named by FG prefix system
                    _S_FG_NAMES = {'sulfoxide', 'sulfone', 'thioether'}
                    _skip_s_branch = False
                    if _attach is not None and _attach_sym == 'S':
                        _all_fgs = features.functional_groups if hasattr(features, 'functional_groups') else {}
                        for _fg_n in _S_FG_NAMES:
                            for _fg_match in _all_fgs.get(_fg_n, []):
                                if _attach in _fg_match:
                                    _skip_s_branch = True
                                    break
                            if _skip_s_branch:
                                break
                    if not sub_has_ring and _attach_sym != 'O' and not _skip_s_branch:
                        from ..assembly.substituent_enumerator import (
                            SubstituentInfo,
                            classify_and_name_fragment,
                        )
                        from ..assembly.naming_utils import needs_brackets
                        chain_set = set(features.principal_chain) if features.principal_chain else set()
                        frag_info = SubstituentInfo(
                            frag_mol=None,
                            locant=position,
                            attach_mol_idx=features.principal_chain[position - 1] if features.principal_chain and position <= len(features.principal_chain) else sub_atoms[0],
                            frag_atoms=frozenset(sub_atoms),
                        )
                        het_name = classify_and_name_fragment(mol, frag_info, chain_set, features)
                        if het_name and needs_brackets(het_name):
                            het_name = f"({het_name})"
                if het_name:
                    substituent_groups[het_name].append(position)
                continue

            try:
                alkyl_name = get_alkyl_name(carbon_count)
                substituent_groups[alkyl_name].append(position)
            except ValueError:
                continue

    # Build formatted prefixes using format_substituent_prefix for proper
    # complex substituent handling (e.g., bis(acetyloxy) not bisacetyloxy)
    from ..assembly.naming_utils import format_substituent_prefix as _fmt_sub
    prefixes = []
    for name, locants in substituent_groups.items():
        count = len(locants)
        sorted_locs = sorted(locants)
        formatted = _fmt_sub(name, sorted_locs, count)
        prefixes.append(formatted)

    return prefixes


def _merge_bare_duplicate_prefixes(prefixes: List[str]) -> List[str]:
    """Merge duplicate bare (no-locant) prefix strings into single entries.

    Only merges prefixes that have NO locants. Two bare "hydroxy" from
    primary_alcohol + secondary_alcohol become "dihydroxy". Prefixes with
    locants, parentheses, or N- are never merged.

    Args:
        prefixes: Formatted prefix strings.

    Returns:
        De-duplicated prefix list.
    """
    import re as _re

    if len(prefixes) <= 1:
        return prefixes

    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS

    # Group bare (no-locant, no-paren, no-N-) prefixes by base name.
    bare_groups: Dict[str, List[int]] = {}
    for i, ptext in enumerate(prefixes):
        if _re.match(r'^[\d,]+-', ptext):
            continue
        if ptext.startswith('(') or ptext.startswith('N-') or ptext.startswith('N,'):
            continue
        base = ptext
        for _cnt, mult in sorted(SIMPLE_MULTIPLIERS.items(),
                                 key=lambda x: len(x[1]), reverse=True):
            if ptext.startswith(mult):
                cand = ptext[len(mult):]
                if cand and cand[0].islower():
                    base = cand
                    break
        if base not in bare_groups:
            bare_groups[base] = []
        bare_groups[base].append(i)

    merged_indices: set = set()
    extra: List[str] = []
    for base, indices in bare_groups.items():
        if len(indices) <= 1:
            continue
        total = 0
        for idx in indices:
            ptext = prefixes[idx]
            cnt = 1
            for c, mult in sorted(SIMPLE_MULTIPLIERS.items(),
                                   key=lambda x: len(x[1]), reverse=True):
                if ptext.startswith(mult):
                    cand = ptext[len(mult):]
                    if cand and cand[0].islower():
                        cnt = c
                        break
            total += cnt
            merged_indices.add(idx)
        if total <= 0:
            total = len(indices)
        if total > 1:
            mult = SIMPLE_MULTIPLIERS.get(total, str(total))
            extra.append(f"{mult}{base}")
        else:
            extra.append(base)

    if not merged_indices:
        return prefixes
    result = [prefixes[i] for i in range(len(prefixes)) if i not in merged_indices]
    result.extend(extra)
    return result


def _join_prefixes(prefix_texts: List[str]) -> str:
    """
    Join multiple prefix strings with proper IUPAC hyphenation.

    When concatenating prefixes like "3-ethyl" and "2-hydroxy", the result
    should be "3-ethyl-2-hydroxy" (with hyphen between letter and digit).
    """
    if not prefix_texts:
        return ""

    if len(prefix_texts) == 1:
        return prefix_texts[0]

    result = prefix_texts[0]
    for i in range(1, len(prefix_texts)):
        current = prefix_texts[i]
        if result and current:
            last_char = result[-1]
            first_char = current[0]
            # Insert hyphen between letter/paren and digit
            # e.g., "amino" + "4-methyl" → "amino-4-methyl"
            # e.g., "(ethanoyl)amino" + "4-methyl" → "(ethanoyl)amino-4-methyl"
            if first_char.isdigit() and (last_char.isalpha() or last_char == ')'):
                result += "-"
            # Also between ')' and letter for clarity
            elif last_char == ')' and first_char.isalpha():
                result += "-"
        result += current

    return result
