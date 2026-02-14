"""
Phosphorus compound naming rules per IUPAC 2013.

Handles:
- Phosphines (R3P): substitutive naming with "phosphane" (PIN, not "phosphine")
- Phosphine oxides (R3P=O): functional class naming
- Phosphonic acids (RP(O)(OH)2): suffix -phosphonic acid
- Phosphinic acids (R2P(O)OH): suffix -phosphinic acid
- Phosphate esters: functional class naming (methyl phosphate)
- Phosphanyl prefix for P as substituent (e.g., diphenylphosphanyl)
"""

from typing import Optional, Tuple, List
from collections import deque, Counter
from rdkit import Chem

from ..assembly.naming_utils import get_alkyl_name


def _characterize_substituent(mol, start_idx: int, exclude: set) -> Optional[Tuple[str, str]]:
    """
    Characterize a substituent attached to phosphorus.

    Distinguishes aryl rings (phenyl, naphthyl) from alkyl chains.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the atom directly bonded to P
        exclude: Set of atom indices to exclude (typically {phosphorus_idx})

    Returns:
        Tuple of (type, name) where type is "aryl" or "alkyl" and name is the
        substituent name string (e.g., "phenyl", "naphthyl", "methyl", "ethyl").
        Returns None if substituent cannot be characterized.
    """
    start_atom = mol.GetAtomWithIdx(start_idx)

    # Check if start atom is aromatic
    if start_atom.GetIsAromatic() and start_atom.GetSymbol() == 'C':
        # Collect all aromatic carbon atoms reachable from start_idx
        aromatic_atoms = set()
        queue = deque([start_idx])
        while queue:
            idx = queue.popleft()
            if idx in aromatic_atoms or idx in exclude:
                continue
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetIsAromatic() and atom.GetSymbol() == 'C':
                aromatic_atoms.add(idx)
                for neighbor in atom.GetNeighbors():
                    nbr_idx = neighbor.GetIdx()
                    if nbr_idx not in aromatic_atoms and nbr_idx not in exclude:
                        if neighbor.GetIsAromatic():
                            queue.append(nbr_idx)

        atom_count = len(aromatic_atoms)

        if atom_count == 6:
            # Verify it's a proper 6-membered aromatic ring of all carbons
            # Find the ring containing start_idx
            ri = mol.GetRingInfo()
            for ring in ri.AtomRings():
                if start_idx in ring and len(ring) == 6:
                    all_aromatic_c = all(
                        mol.GetAtomWithIdx(a).GetIsAromatic()
                        and mol.GetAtomWithIdx(a).GetSymbol() == 'C'
                        for a in ring
                    )
                    if all_aromatic_c:
                        return ("aryl", "phenyl")
            # Fallback: if we found 6 aromatic carbons but no matching ring
            return ("aryl", "phenyl")

        elif atom_count == 10:
            # Naphthalene system (two fused 6-membered aromatic rings)
            return ("aryl", "naphthyl")

        # Unrecognized aromatic system
        return None

    # Not aromatic: count carbons via BFS (same as old _count_alkyl_carbons)
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

    if count == 0 or count > 10:
        return None

    return ("alkyl", get_alkyl_name(count))


def _count_alkyl_carbons(mol, start_idx: int, exclude: set) -> int:
    """
    Count carbon atoms in an alkyl group via BFS.

    Backward-compatible wrapper around _characterize_substituent().
    """
    result = _characterize_substituent(mol, start_idx, exclude)
    if result is None:
        return 0
    sub_type, name = result
    if sub_type == "aryl":
        # For aryl groups, return the carbon count of the ring system
        # phenyl = 6, naphthyl = 10
        aryl_counts = {"phenyl": 6, "naphthyl": 10}
        return aryl_counts.get(name, 0)
    # For alkyl, reconstruct count via BFS (keep original logic for exact count)
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


def _build_substituent_string(names: List[str]) -> str:
    """
    Build a substituent prefix string from a list of substituent names.

    Handles multipliers (di-, tri-) for identical groups, and alphabetical ordering.
    Examples:
        ["methyl", "methyl", "methyl"] -> "trimethyl"
        ["ethyl", "methyl", "methyl"] -> "ethyldimethyl"
        ["phenyl", "phenyl"] -> "diphenyl"
        ["methyl", "phenyl", "phenyl"] -> "diphenylmethyl"

    Note: IUPAC alphabetical ordering ignores multiplicative prefixes (di, tri).
    """
    counts = Counter(names)
    multiplier_map = {1: "", 2: "di", 3: "tri"}

    # Sort unique names alphabetically
    sorted_unique = sorted(counts.keys())

    parts = []
    for name in sorted_unique:
        multiplier = multiplier_map.get(counts[name], "")
        parts.append(f"{multiplier}{name}")

    return "".join(parts)


def name_phosphine(mol, phosphorus_idx: int) -> Optional[str]:
    """
    Name a phosphine using substitutive nomenclature.

    IUPAC 2013 PIN: "phosphane" (not "phosphine")
    - methylphosphane, dimethylphosphane, trimethylphosphane
    - triphenylphosphane, diphenylmethylphosphane

    Args:
        mol: RDKit Mol object
        phosphorus_idx: Index of phosphorus atom

    Returns:
        Substitutive name like "trimethylphosphane", or None if not a simple phosphine
    """
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)

    # Get carbon/aromatic-carbon neighbors
    neighbors = [n for n in phosphorus.GetNeighbors() if n.GetSymbol() == 'C']

    if len(neighbors) == 0:
        return "phosphane"  # Parent hydride PH3

    # Characterize each substituent
    sub_names = []
    for neighbor in neighbors:
        result = _characterize_substituent(mol, neighbor.GetIdx(), {phosphorus_idx})
        if result is None:
            return None  # Unrecognized substituent
        _, name = result
        sub_names.append(name)

    # Build substituent string with multipliers and alphabetical ordering
    prefix = _build_substituent_string(sub_names)
    return f"{prefix}phosphane"


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

    # Characterize each substituent
    sub_names = []
    for neighbor in neighbors:
        result = _characterize_substituent(mol, neighbor.GetIdx(), {phosphorus_idx})
        if result is None:
            return None
        _, name = result
        sub_names.append(name)

    # Build substituent string with multipliers and alphabetical ordering
    prefix = _build_substituent_string(sub_names)
    return f"{prefix}phosphane oxide"


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
    Name a phosphinic acid with dialkyl/aryl prefix and phosphinic acid suffix.

    R2P(O)(OH) -> dialkylphosphinic acid, diphenylphosphinic acid

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

    # Characterize each substituent
    sub_names = []
    for neighbor in neighbors:
        result = _characterize_substituent(mol, neighbor.GetIdx(), {phosphorus_idx})
        if result is None:
            return None
        _, name = result
        sub_names.append(name)

    # Build substituent string with multipliers and alphabetical ordering
    prefix = _build_substituent_string(sub_names)
    return f"{prefix}phosphinic acid"


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


def get_phosphanyl_prefix(mol, phosphorus_idx: int) -> Optional[str]:
    """
    Generate IUPAC P-68 phosphanyl prefix string.

    Used when phosphorus is a substituent on a parent chain/ring.
    Characterizes all C/c neighbors of P and builds the prefix.

    Examples:
        -PPh2 -> "diphenylphosphanyl"
        -PMe2 -> "dimethylphosphanyl"
        -PMePhPh -> not typical, but "methyldiphenylphosphanyl"
        -PMe -> "methylphosphanyl"

    Args:
        mol: RDKit Mol object
        phosphorus_idx: Index of phosphorus atom

    Returns:
        Prefix string like "diphenylphosphanyl", or None if no valid substituents
    """
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)

    # Get carbon neighbors
    neighbors = [n for n in phosphorus.GetNeighbors() if n.GetSymbol() == 'C']

    if len(neighbors) == 0:
        return "phosphanyl"  # Bare -PH2

    # Characterize each substituent
    sub_names = []
    for neighbor in neighbors:
        result = _characterize_substituent(mol, neighbor.GetIdx(), {phosphorus_idx})
        if result is None:
            return None  # Unrecognized substituent
        _, name = result
        sub_names.append(name)

    if not sub_names:
        return None

    # Build substituent string with multipliers and alphabetical ordering
    prefix = _build_substituent_string(sub_names)
    return f"{prefix}phosphanyl"


def get_phosphorus_prefix(fg_name: str) -> Optional[str]:
    """
    Get prefix form for phosphorus functional groups.

    Returns:
        Prefix string, or None if group uses functional class naming
    """
    PHOSPHORUS_PREFIXES = {
        "phosphonic_acid": "phosphono",
        "phosphinic_acid": "phosphino",
        # Phosphines use phosphanyl prefix when P is a substituent
        "tertiary_phosphine": "phosphanyl",
        "secondary_phosphine": "phosphanyl",
        "primary_phosphine": "phosphanyl",
        # These use functional class naming, no simple prefix:
        "phosphine_oxide": None,
        "phosphate_triester": None,
        "phosphate_diester": None,
        "phosphate_monoester": None,
    }
    return PHOSPHORUS_PREFIXES.get(fg_name)
