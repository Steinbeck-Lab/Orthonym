"""
Ester naming following IUPAC two-component format.

Esters are named as "alkyl alkanoate" where:
- alkyl: derived from the alcohol portion (attached to ester oxygen)
- alkanoate: derived from the acid portion (contains C=O)

Example: CH3-COO-CH2-CH3 -> "ethyl acetate"
         (acetate from acetic acid, ethyl from ethanol)

SMARTS: "[CX3](=O)[OX2][#6]"
        - Position 0: carbonyl carbon (acid side)
        - Position 1: carbonyl oxygen (=O)
        - Position 2: ester oxygen (-O-)
        - Position 3: first alkyl carbon (alcohol side)
"""

from typing import Tuple, List, Optional
from rdkit import Chem

from ..data.trivial_acids import get_acylate_name
from ..assembly.naming_utils import get_alkyl_name
from ..assembly.composer import CHAIN_PREFIXES


def parse_ester_fragments(mol, ester_match: tuple) -> Tuple[List[int], List[int]]:
    """
    Split ester into acid and alkyl fragments.

    Args:
        mol: RDKit Mol object
        ester_match: Tuple of atom indices from ester SMARTS match
                     (carbonyl_c, carbonyl_o, ester_o, alkyl_c)

    Returns:
        Tuple of (acid_atoms, alkyl_atoms) where each is a list of atom indices

    The acid fragment includes the carbonyl carbon and everything attached
    to it except via the ester oxygen.
    The alkyl fragment includes everything attached to the ester oxygen
    except the carbonyl carbon.
    """
    # SMARTS "[CX3](=O)[OX2][#6]" gives us:
    # match[0] = carbonyl carbon
    # match[1] = carbonyl oxygen (=O)
    # match[2] = ester oxygen
    # match[3] = first alkyl carbon (may not be present depending on SMARTS)

    carbonyl_c = ester_match[0]
    ester_o = ester_match[2] if len(ester_match) > 2 else None

    if ester_o is None:
        # Handle different SMARTS match structures
        # Look for the ester oxygen by finding O neighbor of carbonyl C that isn't =O
        carbonyl_atom = mol.GetAtomWithIdx(carbonyl_c)
        for neighbor in carbonyl_atom.GetNeighbors():
            if neighbor.GetSymbol() == 'O':
                bond = mol.GetBondBetweenAtoms(carbonyl_c, neighbor.GetIdx())
                if bond.GetBondType() == Chem.BondType.SINGLE:
                    ester_o = neighbor.GetIdx()
                    break

    if ester_o is None:
        return [], []

    # BFS to find acid fragment (from carbonyl_c, excluding ester_o)
    acid_atoms = _bfs_fragment(mol, carbonyl_c, exclude_atom=ester_o)

    # BFS to find alkyl fragment (from ester_o, excluding carbonyl_c)
    # Start from neighbors of ester_o that aren't carbonyl_c
    ester_o_atom = mol.GetAtomWithIdx(ester_o)
    alkyl_start = None
    for neighbor in ester_o_atom.GetNeighbors():
        if neighbor.GetIdx() != carbonyl_c:
            alkyl_start = neighbor.GetIdx()
            break

    if alkyl_start is None:
        return acid_atoms, []

    alkyl_atoms = _bfs_fragment(mol, alkyl_start, exclude_atom=ester_o)

    return acid_atoms, alkyl_atoms


def _bfs_fragment(mol, start_atom: int, exclude_atom: int) -> List[int]:
    """BFS to collect all atoms in a fragment, excluding one connection."""
    visited = {start_atom}
    queue = [start_atom]

    while queue:
        current = queue.pop(0)
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx != exclude_atom:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    return list(visited)


def get_acid_fragment_name(mol, acid_atoms: List[int]) -> str:
    """
    Get the acid name from the acid fragment.

    For simple chain acids, determines chain length and returns systematic name.
    For recognized trivial acids, returns trivial name.

    Args:
        mol: RDKit Mol object
        acid_atoms: Atom indices of the acid fragment

    Returns:
        Acid stem name (e.g., "acetic", "propanoic")
    """
    # Count carbons in acid fragment
    carbon_count = sum(
        1 for idx in acid_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    )

    # Check for common trivial acids by structure
    # For now, use systematic naming based on carbon count
    # 1 carbon = formic (methanoic)
    # 2 carbons = acetic (ethanoic)
    # 3 carbons = propanoic
    # etc.

    if carbon_count == 1:
        return "formic"  # Trivial name preferred
    elif carbon_count == 2:
        return "acetic"  # Trivial name preferred

    # Systematic for others
    if carbon_count in CHAIN_PREFIXES:
        return CHAIN_PREFIXES[carbon_count] + "anoic"

    return f"{carbon_count}Canoic"


def get_alkyl_fragment_name(mol, alkyl_atoms: List[int]) -> str:
    """
    Get the alkyl name from the alkyl fragment.

    For simple chains, returns methyl, ethyl, propyl, etc.
    For branched alkyl groups, may need more complex naming.

    Args:
        mol: RDKit Mol object
        alkyl_atoms: Atom indices of the alkyl fragment

    Returns:
        Alkyl name (e.g., "methyl", "ethyl")
    """
    # Count carbons
    carbon_count = sum(
        1 for idx in alkyl_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    )

    if carbon_count == 0:
        return ""

    try:
        return get_alkyl_name(carbon_count)
    except ValueError:
        # Complex or long alkyl group
        return f"{carbon_count}C-yl"


def is_lactone(mol, ester_match: tuple) -> bool:
    """
    Check if the ester is a lactone (cyclic ester).

    Lactones have the carbonyl carbon and an alkyl carbon in the same ring.
    """
    carbonyl_c = ester_match[0]

    # Get ester oxygen and find alkyl carbon
    ester_o = ester_match[2] if len(ester_match) > 2 else None
    if ester_o is None:
        # Find ester oxygen manually
        carbonyl_atom = mol.GetAtomWithIdx(carbonyl_c)
        for neighbor in carbonyl_atom.GetNeighbors():
            if neighbor.GetSymbol() == 'O':
                bond = mol.GetBondBetweenAtoms(carbonyl_c, neighbor.GetIdx())
                if bond.GetBondType() == Chem.BondType.SINGLE:
                    ester_o = neighbor.GetIdx()
                    break

    if ester_o is None:
        return False

    ester_o_atom = mol.GetAtomWithIdx(ester_o)
    alkyl_c = None
    for neighbor in ester_o_atom.GetNeighbors():
        if neighbor.GetIdx() != carbonyl_c:
            alkyl_c = neighbor.GetIdx()
            break

    if alkyl_c is None:
        return False

    # Check if carbonyl_c and alkyl_c are in the same ring
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if carbonyl_c in ring and alkyl_c in ring:
            return True

    return False


def name_ester(mol, ester_match: tuple) -> Optional[str]:
    """
    Generate IUPAC name for an ester.

    Format: "alkyl alkanoate" (e.g., "methyl acetate", "ethyl propanoate")

    Args:
        mol: RDKit Mol object
        ester_match: Tuple from ester SMARTS match

    Returns:
        Ester name string, or None if cannot be named (e.g., lactone)
    """
    # Check for lactone (cyclic ester) - handle differently
    if is_lactone(mol, ester_match):
        # Lactones need special handling - defer to later implementation
        return None  # Signal to use different naming path

    # Parse into fragments
    acid_atoms, alkyl_atoms = parse_ester_fragments(mol, ester_match)

    if not acid_atoms or not alkyl_atoms:
        return None  # Cannot determine fragments

    # Get acid name and convert to acylate
    acid_name = get_acid_fragment_name(mol, acid_atoms)
    acylate_name = get_acylate_name(acid_name)

    # Get alkyl name
    alkyl_name = get_alkyl_fragment_name(mol, alkyl_atoms)

    if not alkyl_name:
        return None

    # Combine: "alkyl acylate"
    return f"{alkyl_name} {acylate_name}"


def find_ester_match(mol) -> Optional[tuple]:
    """
    Find the first ester match in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of atom indices for the ester, or None if no ester found
    """
    pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
    matches = mol.GetSubstructMatches(pattern)
    if matches:
        return matches[0]
    return None
