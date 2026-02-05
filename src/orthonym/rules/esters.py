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
from ..data.chain_names import get_chain_prefix, get_alkyl_name as _chain_alkyl_name, get_acid_stem


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

    # Systematic for others - use centralized chain naming
    return get_acid_stem(carbon_count)


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
        return _chain_alkyl_name(carbon_count)
    except ValueError:
        # Fallback for edge cases
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


# ============================================================================
# Acyloxy Prefix Naming (IUPAC P-65.6.3.2.2)
# ============================================================================
#
# When an ester is named as a substituent prefix (e.g., on a ring parent),
# the R-CO-O- portion is named as an "acyloxy" group:
#   acid name (drop "-ic") + "-yloxy"
#
# Examples:
#   formic    -> formyloxy
#   acetic    -> acetyloxy
#   benzoic   -> benzoyloxy
#   propanoic -> propanoyloxy

# Explicit mapping for trivial acid names to acyloxy prefixes.
# These are common acids whose acyloxy forms have established trivial stems.
TRIVIAL_ACID_TO_ACYLOXY = {
    "formic": "formyloxy",
    "acetic": "acetyloxy",
    "propionic": "propionyloxy",
    "butyric": "butyryloxy",
    "valeric": "valeryloxy",
    "caproic": "caproyloxy",
    "benzoic": "benzoyloxy",
    "oxalic": "oxalyloxy",
    "lactic": "lactyloxy",
}


def get_acyloxy_prefix(acid_name: str) -> str:
    """
    Convert an acid name to its acyloxy prefix form (IUPAC P-65.6.3.2.2).

    Used when an ester group is named as a substituent prefix rather than
    the principal characteristic group. The R-CO-O- portion becomes an
    "acyloxy" prefix.

    Conversion rule: drop "-ic" (or "-ic acid"), add "-yloxy".

    Args:
        acid_name: The acid name without "acid" suffix
                   (e.g., "acetic", "propanoic", "benzoic")

    Returns:
        The acyloxy prefix (e.g., "acetyloxy", "propanoyloxy", "benzoyloxy")

    Examples:
        >>> get_acyloxy_prefix("acetic")
        'acetyloxy'
        >>> get_acyloxy_prefix("propanoic")
        'propanoyloxy'
        >>> get_acyloxy_prefix("benzoic")
        'benzoyloxy'
        >>> get_acyloxy_prefix("formic")
        'formyloxy'
    """
    name = acid_name.lower().strip()

    # Strip trailing " acid" if present
    if name.endswith(" acid"):
        name = name[:-5].strip()

    # Check trivial acid lookup first
    if name in TRIVIAL_ACID_TO_ACYLOXY:
        return TRIVIAL_ACID_TO_ACYLOXY[name]

    # Systematic conversion: drop "-ic", add "-yloxy"
    # Works for both "-oic" (propanoic -> propanoyloxy) and "-ic" (generic)
    if name.endswith("ic"):
        return name[:-2] + "yloxy"

    # Fallback: just append "yloxy"
    return name + "yloxy"


def name_ester_as_prefix(mol, ester_match: tuple) -> Optional[str]:
    """
    Generate an acyloxy prefix name for an ester group.

    Used when the ester is a substituent (not the principal characteristic
    group). Extracts the acid fragment, determines its name, and converts
    to the acyloxy prefix form.

    Args:
        mol: RDKit Mol object
        ester_match: Tuple of atom indices from ester SMARTS match

    Returns:
        Acyloxy prefix string (e.g., "acetyloxy", "propanoyloxy"),
        or None if the ester is a lactone or cannot be named.

    Examples:
        For methyl acetate (COC(C)=O), returns "acetyloxy"
        For methyl propanoate (COC(=O)CC), returns "propanoyloxy"
    """
    # Lactones cannot be named as acyloxy prefixes
    if is_lactone(mol, ester_match):
        return None

    # Parse into acid and alkyl fragments
    acid_atoms, alkyl_atoms = parse_ester_fragments(mol, ester_match)

    if not acid_atoms:
        return None

    # Get the acid name from the fragment
    acid_name = get_acid_fragment_name(mol, acid_atoms)

    # Convert to acyloxy prefix
    return get_acyloxy_prefix(acid_name)


def detect_exocyclic_esters(mol) -> List[dict]:
    """
    Detect ester groups where the ester oxygen is bonded to a ring atom.

    These are "exocyclic" esters: the ring is the parent structure and
    the ester should be named as an acyloxy prefix on the ring.

    For example, cyclohexyl acetate (CC(=O)OC1CCCCC1) has an ester oxygen
    bonded to a cyclohexane ring carbon. The ester should be named as
    "acetyloxy" prefix on cyclohexane.

    Excludes lactones (cyclic esters where both the carbonyl C and alkyl C
    are in the same ring).

    Args:
        mol: RDKit Mol object

    Returns:
        List of dicts, each containing:
            - ester_match: tuple of atom indices from SMARTS match
            - ring_attach_atom_idx: index of the ring atom bonded to ester O
            - acyloxy_prefix: the acyloxy prefix string (e.g., "acetyloxy")

    Examples:
        For cyclohexyl acetate: returns [{"ester_match": (...),
            "ring_attach_atom_idx": 3, "acyloxy_prefix": "acetyloxy"}]
        For ethyl acetate: returns [] (no ring attachment)
    """
    pattern = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
    matches = mol.GetSubstructMatches(pattern)

    results = []
    for match in matches:
        # Skip lactones
        if is_lactone(mol, match):
            continue

        # match[2] is the ester oxygen, match[3] is the alkyl carbon
        # Check if the alkyl carbon (position 3) is in a ring
        alkyl_c_idx = match[3]
        alkyl_c_atom = mol.GetAtomWithIdx(alkyl_c_idx)

        if not alkyl_c_atom.IsInRing():
            continue

        # This is an exocyclic ester on a ring parent
        # Get the acyloxy prefix
        acyloxy = name_ester_as_prefix(mol, match)
        if acyloxy is None:
            continue

        results.append({
            "ester_match": match,
            "ring_attach_atom_idx": alkyl_c_idx,
            "acyloxy_prefix": acyloxy,
        })

    return results
