"""
Phosphorus compound naming rules per IUPAC 2013.

Handles:
- Phosphines (R3P): substitutive naming with "phosphane" (PIN, not "phosphine")
- Phosphine oxides (R3P=O): functional class naming
- Phosphonic acids (RP(O)(OH)2): suffix -phosphonic acid
- Phosphinic acids (R2P(O)OH): suffix -phosphinic acid
- Phosphate esters: functional class naming (methyl phosphate)
"""

from typing import Optional, Tuple, List
from collections import deque
from rdkit import Chem

from ..assembly.naming_utils import get_alkyl_name


def name_phosphine(mol, phosphorus_idx: int) -> Optional[str]:
    """
    Name a phosphine using substitutive nomenclature.

    IUPAC 2013 PIN: "phosphane" (not "phosphine")
    - methylphosphane, dimethylphosphane, trimethylphosphane

    Args:
        mol: RDKit Mol object
        phosphorus_idx: Index of phosphorus atom

    Returns:
        Substitutive name like "trimethylphosphane", or None if not a simple phosphine
    """
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)

    # Get carbon neighbors
    neighbors = [n for n in phosphorus.GetNeighbors() if n.GetSymbol() == 'C']

    if len(neighbors) == 0:
        return "phosphane"  # Parent hydride PH3

    # Count carbons in each alkyl group
    alkyl_names = []
    for neighbor in neighbors:
        carbon_count = _count_alkyl_carbons(mol, neighbor.GetIdx(), {phosphorus_idx})
        if carbon_count == 0 or carbon_count > 10:
            return None  # Not a simple alkyl
        alkyl_names.append(get_alkyl_name(carbon_count))

    # Sort alphabetically for alphabetical ordering check
    sorted_names = sorted(alkyl_names)

    # Check for symmetry (all identical)
    if len(set(alkyl_names)) == 1:
        count = len(alkyl_names)
        multiplier = {1: "", 2: "di", 3: "tri"}[count]
        return f"{multiplier}{alkyl_names[0]}phosphane"
    else:
        # Different: concatenate alphabetically without spaces
        return "".join(sorted_names) + "phosphane"


def name_phosphine_oxide(mol, phosphine_oxide_atoms: Tuple[int, ...]) -> Optional[str]:
    """
    Name a phosphine oxide using substitutive nomenclature.

    IUPAC: trimethylphosphane oxide (substitutive, preferred)

    Args:
        mol: RDKit Mol object
        phosphine_oxide_atoms: Atom indices from SMARTS match

    Returns:
        Name like "trimethylphosphane oxide", or None if not simple
    """
    # Find the phosphorus atom
    phosphorus_idx = None
    for idx in phosphine_oxide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'P':
            phosphorus_idx = idx
            break

    if phosphorus_idx is None:
        return None

    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)

    # Get carbon neighbors (exclude oxygen)
    neighbors = [n for n in phosphorus.GetNeighbors() if n.GetSymbol() == 'C']
    if len(neighbors) != 3:
        return None  # Must have exactly 3 carbon substituents

    # Count carbons in each alkyl group
    alkyl_names = []
    for neighbor in neighbors:
        carbon_count = _count_alkyl_carbons(mol, neighbor.GetIdx(), {phosphorus_idx})
        if carbon_count == 0 or carbon_count > 10:
            return None
        alkyl_names.append(get_alkyl_name(carbon_count))

    # Sort alphabetically
    sorted_names = sorted(alkyl_names)

    # Check for symmetry
    if len(set(alkyl_names)) == 1:
        multiplier = {1: "", 2: "di", 3: "tri"}[len(alkyl_names)]
        return f"{multiplier}{alkyl_names[0]}phosphane oxide"
    else:
        return "".join(sorted_names) + "phosphane oxide"


def name_phosphonic_acid(mol, phosphonic_atoms: Tuple[int, ...], parent_name: str) -> str:
    """
    Name a phosphonic acid with -phosphonic acid suffix.

    Args:
        mol: RDKit Mol object
        phosphonic_atoms: Atom indices from SMARTS match
        parent_name: Parent chain/ring name (e.g., "methane", "ethane", "benzene")

    Returns:
        Name like "methanephosphonic acid", "ethanephosphonic acid"
    """
    # Phosphonic acid always attached at chain/ring end, no locant needed for simple cases
    return f"{parent_name}phosphonic acid"


def name_phosphinic_acid(mol, phosphinic_atoms: Tuple[int, ...]) -> Optional[str]:
    """
    Name a phosphinic acid with dialkyl- prefix and phosphinic acid suffix.

    R2P(O)(OH) -> dialkylphosphinic acid

    Args:
        mol: RDKit Mol object
        phosphinic_atoms: Atom indices from SMARTS match

    Returns:
        Name like "dimethylphosphinic acid", or None if not simple
    """
    # Find the phosphorus atom
    phosphorus_idx = None
    for idx in phosphinic_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'P':
            phosphorus_idx = idx
            break

    if phosphorus_idx is None:
        return None

    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)

    # Get carbon neighbors (should be exactly 2)
    neighbors = [n for n in phosphorus.GetNeighbors() if n.GetSymbol() == 'C']
    if len(neighbors) != 2:
        return None

    # Count carbons in each alkyl group
    alkyl_names = []
    for neighbor in neighbors:
        carbon_count = _count_alkyl_carbons(mol, neighbor.GetIdx(), {phosphorus_idx})
        if carbon_count == 0 or carbon_count > 10:
            return None
        alkyl_names.append(get_alkyl_name(carbon_count))

    # Sort alphabetically
    sorted_names = sorted(alkyl_names)

    # Check for symmetry
    if alkyl_names[0] == alkyl_names[1]:
        return f"di{alkyl_names[0]}phosphinic acid"
    else:
        return "".join(sorted_names) + "phosphinic acid"


def name_phosphate_ester(mol, phosphorus_idx: int) -> Optional[str]:
    """
    Name a phosphate ester using functional class nomenclature.

    Simple esters: "methyl phosphate", "dimethyl phosphate", "trimethyl phosphate"

    Args:
        mol: RDKit Mol object
        phosphorus_idx: Index of phosphorus atom

    Returns:
        Functional class name, or None if not a simple phosphate
    """
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)

    # Find O-C groups attached to P (not P=O, not P-OH)
    ester_oxygens = []
    for neighbor in phosphorus.GetNeighbors():
        if neighbor.GetSymbol() == 'O':
            # Check bond type - skip double-bonded oxygen (P=O)
            bond = mol.GetBondBetweenAtoms(phosphorus_idx, neighbor.GetIdx())
            if bond.GetBondTypeAsDouble() > 1.5:  # Double bond
                continue

            # Check if O is bonded to C (ester) vs H (acid)
            o_neighbors = [n for n in neighbor.GetNeighbors() if n.GetIdx() != phosphorus_idx]
            if o_neighbors and o_neighbors[0].GetSymbol() == 'C':
                ester_oxygens.append(neighbor.GetIdx())

    if not ester_oxygens:
        return None  # Not an ester

    # Get alkyl names for each ester group
    alkyl_names = []
    for o_idx in ester_oxygens:
        o_atom = mol.GetAtomWithIdx(o_idx)
        c_neighbors = [n for n in o_atom.GetNeighbors() if n.GetSymbol() == 'C']
        if not c_neighbors:
            continue
        c_neighbor = c_neighbors[0]
        carbon_count = _count_alkyl_carbons(mol, c_neighbor.GetIdx(), {o_idx, phosphorus_idx})
        if carbon_count > 0:
            alkyl_names.append(get_alkyl_name(carbon_count))

    if not alkyl_names:
        return None

    # Sort alphabetically
    alkyl_names.sort()

    # Format based on count and symmetry
    if len(alkyl_names) == 1:
        return f"{alkyl_names[0]} phosphate"
    elif len(alkyl_names) == 2:
        if alkyl_names[0] == alkyl_names[1]:
            return f"di{alkyl_names[0]} phosphate"
        else:
            return f"{alkyl_names[0]} {alkyl_names[1]} phosphate"
    elif len(alkyl_names) == 3:
        if len(set(alkyl_names)) == 1:
            return f"tri{alkyl_names[0]} phosphate"
        else:
            return f"{' '.join(alkyl_names)} phosphate"

    return None


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


def get_phosphorus_prefix(fg_name: str) -> Optional[str]:
    """
    Get prefix form for phosphorus functional groups.

    Returns:
        Prefix string, or None if group uses functional class naming
    """
    PHOSPHORUS_PREFIXES = {
        "phosphonic_acid": "phosphono",
        "phosphinic_acid": "phosphino",
        # These use substitutive or functional class naming, no prefix:
        "phosphine_oxide": None,
        "tertiary_phosphine": None,
        "secondary_phosphine": None,
        "primary_phosphine": None,
        "phosphate_triester": None,
        "phosphate_diester": None,
        "phosphate_monoester": None,
    }
    return PHOSPHORUS_PREFIXES.get(fg_name)
