"""
Sulfur compound naming rules per IUPAC 2013.

Handles:
- Thiols (-SH): suffix -thiol, prefix sulfanyl-
- Sulfides (R-S-R'): functional class naming (dimethyl sulfide)
- Sulfoxides (R-SO-R'): functional class naming (dimethyl sulfoxide)
- Sulfones (R-SO2-R'): functional class naming (dimethyl sulfone)
- Sulfonic acids (-SO3H): suffix -sulfonic acid, prefix sulfo-
"""

from typing import Optional, Tuple, List
from collections import deque
from rdkit import Chem

from ..assembly.naming_utils import get_alkyl_name


def name_thiol(mol, thiol_atoms: Tuple[int, ...], parent_name: str, locant: Optional[int] = None) -> str:
    """
    Name a thiol compound with -thiol suffix.

    Args:
        mol: RDKit Mol object
        thiol_atoms: Atom indices from SMARTS match (S, C)
        parent_name: Parent chain/ring name without suffix
        locant: Position of thiol group (None if implied)

    Returns:
        Name like "methanethiol", "propane-1-thiol"
    """
    # Terminal thiols: locant is 1, often omitted for 1-2 carbon chains
    if locant is None or locant == 1:
        # For methane/ethane, no locant needed
        if parent_name in ("methan", "ethan"):
            return f"{parent_name}ethiol"
        # For longer chains, include locant in PIN style
        return f"{parent_name}e-1-thiol"
    else:
        return f"{parent_name}e-{locant}-thiol"


def name_sulfide(mol, sulfur_idx: int) -> Optional[str]:
    """
    Name a sulfide (thioether) using functional class nomenclature.

    IUPAC prefers functional class for simple sulfides:
    - Symmetric: "dimethyl sulfide", "diethyl sulfide"
    - Asymmetric: "ethyl methyl sulfide" (alphabetical order)

    Args:
        mol: RDKit Mol object
        sulfur_idx: Index of sulfur atom

    Returns:
        Functional class name, or None if not a simple sulfide
    """
    sulfur = mol.GetAtomWithIdx(sulfur_idx)

    # Get carbon neighbors
    neighbors = [n for n in sulfur.GetNeighbors() if n.GetSymbol() == 'C']
    if len(neighbors) != 2:
        return None

    # Count carbons in each alkyl group
    alkyl_names = []
    for neighbor in neighbors:
        carbon_count = _count_alkyl_carbons(mol, neighbor.GetIdx(), {sulfur_idx})
        if carbon_count == 0 or carbon_count > 10:
            return None  # Not a simple alkyl
        alkyl_names.append(get_alkyl_name(carbon_count))

    # Sort alphabetically
    alkyl_names.sort()

    # Check for symmetry
    if alkyl_names[0] == alkyl_names[1]:
        return f"di{alkyl_names[0]} sulfide"
    else:
        return f"{alkyl_names[0]} {alkyl_names[1]} sulfide"


def name_sulfoxide(mol, sulfoxide_atoms: Tuple[int, ...]) -> Optional[str]:
    """
    Name a sulfoxide using functional class nomenclature.

    IUPAC prefers functional class for simple sulfoxides:
    - Symmetric: "dimethyl sulfoxide"
    - Asymmetric: "ethyl methyl sulfoxide" (alphabetical order)

    Args:
        mol: RDKit Mol object
        sulfoxide_atoms: Atom indices from SMARTS match

    Returns:
        Functional class name, or None if not a simple sulfoxide
    """
    # Find the sulfur atom (has =O and 2 C neighbors)
    sulfur_idx = None
    for idx in sulfoxide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'S':
            sulfur_idx = idx
            break

    if sulfur_idx is None:
        return None

    sulfur = mol.GetAtomWithIdx(sulfur_idx)

    # Get carbon neighbors (exclude oxygen)
    neighbors = [n for n in sulfur.GetNeighbors() if n.GetSymbol() == 'C']
    if len(neighbors) != 2:
        return None

    # Count carbons in each alkyl group
    alkyl_names = []
    for neighbor in neighbors:
        carbon_count = _count_alkyl_carbons(mol, neighbor.GetIdx(), {sulfur_idx})
        if carbon_count == 0 or carbon_count > 10:
            return None
        alkyl_names.append(get_alkyl_name(carbon_count))

    # Sort alphabetically
    alkyl_names.sort()

    # Check for symmetry
    if alkyl_names[0] == alkyl_names[1]:
        return f"di{alkyl_names[0]} sulfoxide"
    else:
        return f"{alkyl_names[0]} {alkyl_names[1]} sulfoxide"


def name_sulfone(mol, sulfone_atoms: Tuple[int, ...]) -> Optional[str]:
    """
    Name a sulfone using functional class nomenclature.

    IUPAC prefers functional class for simple sulfones:
    - Symmetric: "dimethyl sulfone"
    - Asymmetric: "ethyl methyl sulfone" (alphabetical order)

    Args:
        mol: RDKit Mol object
        sulfone_atoms: Atom indices from SMARTS match

    Returns:
        Functional class name, or None if not a simple sulfone
    """
    # Find the sulfur atom
    sulfur_idx = None
    for idx in sulfone_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'S':
            sulfur_idx = idx
            break

    if sulfur_idx is None:
        return None

    sulfur = mol.GetAtomWithIdx(sulfur_idx)

    # Get carbon neighbors (exclude oxygens)
    neighbors = [n for n in sulfur.GetNeighbors() if n.GetSymbol() == 'C']
    if len(neighbors) != 2:
        return None

    # Count carbons in each alkyl group
    alkyl_names = []
    for neighbor in neighbors:
        carbon_count = _count_alkyl_carbons(mol, neighbor.GetIdx(), {sulfur_idx})
        if carbon_count == 0 or carbon_count > 10:
            return None
        alkyl_names.append(get_alkyl_name(carbon_count))

    # Sort alphabetically
    alkyl_names.sort()

    # Check for symmetry
    if alkyl_names[0] == alkyl_names[1]:
        return f"di{alkyl_names[0]} sulfone"
    else:
        return f"{alkyl_names[0]} {alkyl_names[1]} sulfone"


def name_sulfonic_acid(mol, sulfonic_atoms: Tuple[int, ...], parent_name: str) -> str:
    """
    Name a sulfonic acid with -sulfonic acid suffix.

    Args:
        mol: RDKit Mol object
        sulfonic_atoms: Atom indices from SMARTS match
        parent_name: Parent chain/ring name

    Returns:
        Name like "methanesulfonic acid", "benzenesulfonic acid"
    """
    # Sulfonic acid always at chain/ring end, no locant needed
    return f"{parent_name}sulfonic acid"


def _count_alkyl_carbons(mol, start_idx: int, exclude: set) -> int:
    """Count carbon atoms in an alkyl group via BFS."""
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
                    # Only follow C-C bonds for simple alkyls
                    if neighbor.GetSymbol() == 'C':
                        queue.append(nbr_idx)

    return count


def get_sulfur_prefix(fg_name: str) -> Optional[str]:
    """
    Get prefix form for sulfur functional groups.

    Returns:
        Prefix string, or None if group uses functional class naming
    """
    SULFUR_PREFIXES = {
        "thiol": "sulfanyl",  # IUPAC 2013, not "mercapto"
        "sulfonic_acid": "sulfo",
        "sulfinic_acid": "sulfino",
        # These use functional class naming, no prefix:
        "thioether": None,
        "sulfide": None,
        "sulfoxide": None,
        "sulfone": None,
    }
    return SULFUR_PREFIXES.get(fg_name)
