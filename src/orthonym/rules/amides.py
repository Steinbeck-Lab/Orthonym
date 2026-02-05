"""
Amide naming rules for IUPAC nomenclature.

Amide naming follows these patterns:
- Primary amides: stem + 'amide' (acetamide, propanamide)
- Secondary amides: N-substituent + stem + 'amide' (N-methylacetamide)
- Tertiary amides: N,N-disubstituent + stem + 'amide' (N,N-dimethylacetamide)
- Ring-attached amides: parent + 'carboxamide' (cyclohexanecarboxamide)

Based on IUPAC 2013 Blue Book P-66.1.
"""

from typing import List, Optional, Dict, Any
from collections import defaultdict

from rdkit import Chem

from ..assembly.naming_utils import (
    get_alkyl_name,
    get_multiplier_prefix,
    alpha_sort_key,
    ALKYL_NAMES,
)


# Chain length prefixes for amide naming (trivial + systematic)
CHAIN_PREFIXES = {
    1: "form",  # formamide (special for 1 carbon)
    2: "acet",  # acetamide (special for 2 carbons)
    3: "propan",
    4: "butan",
    5: "pentan",
    6: "hexan",
    7: "heptan",
    8: "octan",
    9: "nonan",
    10: "decan",
}

# Standard stems - delegated to centralized chain_names module
from ..data.chain_names import get_chain_prefix as _get_chain_prefix
STEM_PREFIXES = {i: _get_chain_prefix(i) for i in range(1, 21)}


def get_amide_type(mol, amide_atoms: tuple) -> str:
    """
    Determine if an amide is primary, secondary, or tertiary.

    The amide nitrogen determines the type:
    - Primary: -C(=O)NH2 (N has 2 hydrogens)
    - Secondary: -C(=O)NHR (N has 1 hydrogen, 1 substituent)
    - Tertiary: -C(=O)NR2 (N has 0 hydrogens, 2 substituents)

    Args:
        mol: RDKit Mol object
        amide_atoms: Atom indices from amide SMARTS match

    Returns:
        "primary", "secondary", or "tertiary"
    """
    # Find the nitrogen atom in the amide
    nitrogen_idx = None
    for idx in amide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'N':
            nitrogen_idx = idx
            break

    if nitrogen_idx is None:
        return "primary"

    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)

    # Count hydrogens (explicit + implicit)
    num_h = nitrogen.GetTotalNumHs()

    if num_h >= 2:
        return "primary"
    elif num_h == 1:
        return "secondary"
    else:
        return "tertiary"


def get_n_substituents(mol, amide_atoms: tuple) -> List[Dict]:
    """
    Get substituents attached to the amide nitrogen.

    For secondary/tertiary amides, find the carbon substituents
    attached to nitrogen (not the carbonyl carbon).

    Args:
        mol: RDKit Mol object
        amide_atoms: Atom indices from amide SMARTS match

    Returns:
        List of dicts: [{"atoms": [atom_indices], "name": "methyl"}, ...]
    """
    substituents = []

    # Find the nitrogen and carbonyl carbon
    nitrogen_idx = None
    carbonyl_carbon_idx = None

    for idx in amide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'N':
            nitrogen_idx = idx
        elif atom.GetSymbol() == 'C':
            # Check if this carbon has a double bond to oxygen
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'O':
                    bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                        carbonyl_carbon_idx = idx
                        break

    if nitrogen_idx is None:
        return substituents

    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)

    # Find carbon substituents on nitrogen (excluding the carbonyl carbon)
    for neighbor in nitrogen.GetNeighbors():
        nbr_idx = neighbor.GetIdx()
        if neighbor.GetSymbol() != 'C':
            continue
        if nbr_idx == carbonyl_carbon_idx:
            continue

        # This is a substituent on nitrogen
        sub_atoms = _bfs_substituent(mol, nbr_idx, {nitrogen_idx, carbonyl_carbon_idx})
        carbon_count = sum(1 for idx in sub_atoms if mol.GetAtomWithIdx(idx).GetSymbol() == 'C')

        if carbon_count in ALKYL_NAMES:
            sub_name = ALKYL_NAMES[carbon_count]
            substituents.append({
                "atoms": sub_atoms,
                "name": sub_name,
                "carbon_count": carbon_count,
            })

    return substituents


def _bfs_substituent(mol, start_idx: int, exclude: set) -> List[int]:
    """BFS to find all atoms in a substituent."""
    from collections import deque

    visited = set()
    queue = deque([start_idx])
    atoms = []

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)
        atoms.append(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude:
                queue.append(nbr_idx)

    return atoms


def format_n_substitution(substituents: List[Dict]) -> str:
    """
    Format N-substituents as IUPAC prefix.

    Rules:
    - Single substituent: "N-methyl"
    - Two identical: "N,N-dimethyl"
    - Two different: "N-ethyl-N-methyl" (alphabetized)

    Args:
        substituents: List of dicts from get_n_substituents()

    Returns:
        Formatted N-substitution prefix (e.g., "N-methyl", "N,N-dimethyl")
    """
    if not substituents:
        return ""

    # Group by name
    groups: Dict[str, int] = defaultdict(int)
    for sub in substituents:
        groups[sub["name"]] += 1

    # Build prefix parts
    parts = []
    for name in sorted(groups.keys(), key=alpha_sort_key):
        count = groups[name]
        if count == 1:
            parts.append(f"N-{name}")
        else:
            # Multiple of same: N,N-di...
            multiplier = get_multiplier_prefix(count, name)
            n_locants = ",".join(["N"] * count)
            parts.append(f"{n_locants}-{multiplier}{name}")

    # Join parts with hyphen
    return "-".join(parts)


def is_ring_attached_amide(mol, amide_atoms: tuple) -> bool:
    """
    Check if an amide is attached to a ring.

    A ring-attached amide has its carbonyl carbon directly
    bonded to a ring carbon. These use -carboxamide suffix.

    Args:
        mol: RDKit Mol object
        amide_atoms: Atom indices from amide SMARTS match

    Returns:
        True if the amide is ring-attached
    """
    # Find the carbonyl carbon
    carbonyl_carbon_idx = None
    for idx in amide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'C':
            # Check if this carbon has a double bond to oxygen
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'O':
                    bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                        carbonyl_carbon_idx = idx
                        break
            if carbonyl_carbon_idx:
                break

    if carbonyl_carbon_idx is None:
        return False

    carbonyl = mol.GetAtomWithIdx(carbonyl_carbon_idx)

    # Check if carbonyl carbon has a neighbor in a ring
    for neighbor in carbonyl.GetNeighbors():
        if neighbor.GetSymbol() == 'N':
            continue  # Skip the nitrogen
        if neighbor.GetSymbol() == 'O':
            continue  # Skip the oxygen
        if neighbor.IsInRing():
            return True

    return False


def get_amide_chain_length(mol, amide_atoms: tuple) -> int:
    """
    Get the chain length for an amide (including carbonyl carbon).

    The chain length determines the parent name:
    - 1: formamide
    - 2: acetamide
    - 3: propanamide
    etc.

    Args:
        mol: RDKit Mol object
        amide_atoms: Atom indices from amide SMARTS match

    Returns:
        Chain length (number of carbons in parent chain)
    """
    # Find the carbonyl carbon and nitrogen
    carbonyl_carbon_idx = None
    nitrogen_idx = None

    for idx in amide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'N':
            nitrogen_idx = idx
        elif atom.GetSymbol() == 'C':
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'O':
                    bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                        carbonyl_carbon_idx = idx
                        break

    if carbonyl_carbon_idx is None:
        return 1

    # BFS to find longest chain from carbonyl carbon (excluding nitrogen direction)
    exclude = {nitrogen_idx} if nitrogen_idx else set()
    for idx in amide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'O':
            exclude.add(idx)

    chain = _find_longest_carbon_chain(mol, carbonyl_carbon_idx, exclude)
    return len(chain)


def _find_longest_carbon_chain(mol, start_idx: int, exclude: set) -> List[int]:
    """Find longest carbon chain from a starting atom."""
    from collections import deque

    best_chain = [start_idx]

    def dfs(current_idx: int, path: List[int], visited: set):
        nonlocal best_chain

        if len(path) > len(best_chain):
            best_chain = path.copy()

        atom = mol.GetAtomWithIdx(current_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in visited or nbr_idx in exclude:
                continue
            if neighbor.GetSymbol() != 'C':
                continue

            visited.add(nbr_idx)
            path.append(nbr_idx)
            dfs(nbr_idx, path, visited)
            path.pop()
            visited.discard(nbr_idx)

    visited = {start_idx}
    dfs(start_idx, [start_idx], visited)

    return best_chain


def get_amide_parent_name(chain_length: int, is_ring: bool = False) -> str:
    """
    Get the parent amide name based on chain length.

    Args:
        chain_length: Number of carbons in parent chain
        is_ring: If True, use -carboxamide form

    Returns:
        Parent amide name (e.g., "formamide", "acetamide", "propanamide")
    """
    if is_ring:
        # Ring-attached amides: parent + carboxamide
        stem = _get_chain_prefix(chain_length)
        return f"cyclo{stem}anecarboxamide"

    # Chain amides
    if chain_length == 1:
        return "formamide"
    elif chain_length == 2:
        return "acetamide"
    else:
        stem = _get_chain_prefix(chain_length)
        return f"{stem}anamide"


def name_amide(mol, amide_atoms: tuple) -> str:
    """
    Generate IUPAC name for an amide compound.

    Handles:
    - Primary amides: acetamide
    - Secondary amides: N-methylacetamide
    - Tertiary amides: N,N-dimethylformamide
    - Ring-attached amides: cyclohexanecarboxamide

    Args:
        mol: RDKit Mol object
        amide_atoms: Atom indices from amide SMARTS match

    Returns:
        IUPAC name for the amide
    """
    # Check if ring-attached
    is_ring = is_ring_attached_amide(mol, amide_atoms)

    if is_ring:
        # Find the ring and get its size
        carbonyl_carbon_idx = None
        for idx in amide_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'C':
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'O':
                        bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
                        if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                            carbonyl_carbon_idx = idx
                            break

        ring_size = 6  # Default
        if carbonyl_carbon_idx is not None:
            carbonyl = mol.GetAtomWithIdx(carbonyl_carbon_idx)
            for neighbor in carbonyl.GetNeighbors():
                if neighbor.IsInRing():
                    ring_info = mol.GetRingInfo()
                    for ring in ring_info.AtomRings():
                        if neighbor.GetIdx() in ring:
                            ring_size = len(ring)
                            break
                    break

        stem = _get_chain_prefix(ring_size)

        parent_name = f"cyclo{stem}anecarboxamide"

        # Check for N-substitution
        amide_type = get_amide_type(mol, amide_atoms)
        if amide_type in ("secondary", "tertiary"):
            n_subs = get_n_substituents(mol, amide_atoms)
            n_prefix = format_n_substitution(n_subs)
            if n_prefix:
                return f"{n_prefix}{parent_name}"

        return parent_name

    # Chain amide
    chain_length = get_amide_chain_length(mol, amide_atoms)
    parent_name = get_amide_parent_name(chain_length)

    # Check for N-substitution
    amide_type = get_amide_type(mol, amide_atoms)
    if amide_type in ("secondary", "tertiary"):
        n_subs = get_n_substituents(mol, amide_atoms)
        n_prefix = format_n_substitution(n_subs)
        if n_prefix:
            return f"{n_prefix}{parent_name}"

    return parent_name
