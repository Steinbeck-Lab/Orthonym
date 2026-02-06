"""
Acid halide naming using IUPAC functional class nomenclature.

Acid halides use two-word "functional class" naming:
    {acyl name} {halide word}

Examples:
    CC(=O)Cl     -> acetyl chloride (retained acyl name for C2)
    CCC(=O)Cl    -> propanoyl chloride
    O=C(Cl)c1ccccc1 -> benzoyl chloride (retained acyl name for benzene)
    ClC(=O)CCCC(=O)Cl -> pentanedioyl dichloride (diacid halide)

The acyl name derives from the corresponding acid:
    ethanoic acid  -> ethanoyl  (systematic)
    acetic acid    -> acetyl    (retained, preferred for C2)
    formic acid    -> formyl    (retained, preferred for C1)
    benzoic acid   -> benzoyl   (retained, preferred for benzene-attached)

Reference: IUPAC 2013 Blue Book, P-65.5.1 (Acyl halides)
"""

from typing import Optional, Dict, List
from rdkit import Chem

from ..data.chain_names import get_chain_prefix
from ..assembly.naming_utils import get_multiplier_prefix


# Retained acyl names (chain length -> retained acyl prefix)
RETAINED_ACYL_NAMES = {
    1: "formyl",     # from formic acid
    2: "acetyl",     # from acetic acid
}

# Halide word mapping
HALIDE_WORDS = {
    "acid_chloride": "chloride",
    "acid_bromide": "bromide",
    "acid_fluoride": "fluoride",
}

# Halogen prefix names (for substituent halogens that are NOT part of acid halide)
HALOGEN_PREFIX = {
    "Cl": "chloro",
    "Br": "bromo",
    "F": "fluoro",
}


def name_acid_halide(features) -> Optional[str]:
    """
    Name an acid halide compound using functional class nomenclature.

    Produces two-word names: "{acyl name} {halide word}"

    Args:
        features: MolecularFeatures with principal_group set to an acid halide type.

    Returns:
        IUPAC name string, or None if not an acid halide.
    """
    mol = features.mol
    pg = features.principal_group

    if pg not in HALIDE_WORDS:
        return None

    halide_word = HALIDE_WORDS[pg]

    # Get acid halide matches
    acid_halide_matches = features.functional_groups.get(pg, [])
    if not acid_halide_matches:
        return None

    num_halide_groups = len(acid_halide_matches)

    # Collect all atoms consumed by acid halide groups (C=O, halogen)
    consumed_atoms = set()
    for match in acid_halide_matches:
        consumed_atoms.update(match)

    # Check for ring-attached acid halide (e.g., benzoyl chloride, cyclohexanecarbonyl chloride)
    is_ring_attached = False
    ring_name = None
    for match in acid_halide_matches:
        # match = (C, O, X) from SMARTS [CX3](=O)[X]
        carbonyl_c = match[0]
        atom = mol.GetAtomWithIdx(carbonyl_c)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in consumed_atoms and neighbor.IsInRing():
                is_ring_attached = True
                # Check if it's benzene
                ring_name = _detect_ring_parent(mol, neighbor.GetIdx())
                break

    if is_ring_attached:
        return _name_ring_attached_acid_halide(mol, features, acid_halide_matches,
                                                halide_word, ring_name, consumed_atoms)

    # Acyclic acid halide: determine chain length
    # Find the principal chain (the carbon chain ending at the carbonyl C)
    chain = features.principal_chain
    chain_length = len(chain) if chain else 0

    if chain_length == 0:
        # Fallback: count carbons connected to carbonyl
        chain_length = _count_acyl_chain(mol, acid_halide_matches[0][0], consumed_atoms)

    # For diacid halides (both ends of chain have acid halide groups)
    if num_halide_groups >= 2 and _is_diacid_halide(mol, acid_halide_matches, chain):
        return _name_diacid_halide(chain_length, halide_word, num_halide_groups)

    # Single acid halide: build acyl name
    acyl_name = _build_acyl_name(chain_length)

    # Check for substituents on the chain (non-acid-halide functional groups)
    sub_prefix = _get_chain_substituent_prefix(mol, features, chain, consumed_atoms)

    if sub_prefix:
        return f"{sub_prefix}{acyl_name} {halide_word}"
    return f"{acyl_name} {halide_word}"


def _detect_ring_parent(mol, ring_atom_idx: int) -> Optional[str]:
    """Detect the ring parent name for a ring-attached acid halide."""
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if ring_atom_idx in ring:
            ring_set = set(ring)
            # Check if all ring atoms are aromatic carbons (benzene)
            all_aromatic_c = all(
                mol.GetAtomWithIdx(idx).GetIsAromatic() and
                mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                for idx in ring
            )
            if all_aromatic_c and len(ring) == 6:
                return "benzo"
            # Cycloalkane
            all_carbon = all(
                mol.GetAtomWithIdx(idx).GetSymbol() == 'C' for idx in ring
            )
            if all_carbon:
                prefix = get_chain_prefix(len(ring))
                return f"cyclo{prefix}"
    return None


def _name_ring_attached_acid_halide(mol, features, matches, halide_word,
                                     ring_name, consumed_atoms) -> str:
    """Name a ring-attached acid halide (benzoyl chloride, cyclohexanecarbonyl chloride)."""
    if ring_name == "benzo":
        return f"benzoyl {halide_word}"

    if ring_name and ring_name.startswith("cyclo"):
        # cyclohexanecarbonyl chloride format
        ring_prefix = ring_name  # e.g., "cyclohex"
        return f"{ring_prefix}anecarbonyl {halide_word}"

    # Fallback for unknown ring types
    return f"carbonyl {halide_word}"


def _build_acyl_name(chain_length: int) -> str:
    """Build the acyl name from chain length.

    Args:
        chain_length: Number of carbons including the carbonyl carbon.

    Returns:
        Acyl name string (e.g., 'acetyl', 'propanoyl', 'butanoyl').
    """
    # Check retained names first
    if chain_length in RETAINED_ACYL_NAMES:
        return RETAINED_ACYL_NAMES[chain_length]

    # Systematic: {chain_prefix}anoyl
    prefix = get_chain_prefix(chain_length)
    return f"{prefix}anoyl"


def _name_diacid_halide(chain_length: int, halide_word: str, num_groups: int) -> str:
    """Name a diacid halide (e.g., pentanedioyl dichloride).

    Args:
        chain_length: Total chain length.
        halide_word: "chloride", "bromide", or "fluoride".
        num_groups: Number of acid halide groups.

    Returns:
        IUPAC name string.
    """
    prefix = get_chain_prefix(chain_length)
    multiplier = get_multiplier_prefix(num_groups, halide_word)
    # pentane -> pentanedioyl (diacid form)
    return f"{prefix}anedioyl {multiplier}{halide_word}"


def _is_diacid_halide(mol, matches, chain) -> bool:
    """Check if acid halide groups are at both ends of the chain (diacid halide).

    Args:
        mol: RDKit Mol object.
        matches: List of acid halide SMARTS match tuples.
        chain: Principal chain atom indices.

    Returns:
        True if diacid halide pattern detected.
    """
    if len(matches) < 2:
        return False

    if not chain or len(chain) < 2:
        return False

    # Get carbonyl carbons from matches
    carbonyl_carbons = {match[0] for match in matches}
    chain_ends = {chain[0], chain[-1]}

    # Diacid if carbonyl carbons are at both chain ends
    return len(carbonyl_carbons & chain_ends) >= 2


def _count_acyl_chain(mol, carbonyl_c: int, consumed: set) -> int:
    """Count carbons in the acyl chain by BFS from carbonyl carbon.

    Args:
        mol: RDKit Mol object.
        carbonyl_c: Atom index of carbonyl carbon.
        consumed: Set of atom indices consumed by acid halide group.

    Returns:
        Chain length (number of carbons including carbonyl).
    """
    visited = {carbonyl_c}
    queue = [carbonyl_c]
    carbon_count = 1

    while queue:
        current = queue.pop(0)
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx not in visited and neighbor.GetSymbol() == 'C' and not neighbor.IsInRing():
                # Skip the carbonyl oxygen and halogen (already in consumed)
                if nidx in consumed and nidx != carbonyl_c:
                    continue
                visited.add(nidx)
                queue.append(nidx)
                carbon_count += 1

    return carbon_count


def _get_chain_substituent_prefix(mol, features, chain, consumed_atoms) -> str:
    """Build substituent prefix for non-acid-halide groups on the acyl chain.

    For example, 3-chlorobutanoyl chloride has a substituent Cl at position 3
    that is NOT part of the acid halide functional group.

    Args:
        mol: RDKit Mol object.
        features: MolecularFeatures.
        chain: Principal chain atom indices.
        consumed_atoms: Atom indices consumed by acid halide groups.

    Returns:
        Prefix string (e.g., "3-chloro") or empty string.
    """
    if not chain:
        return ""

    # Build atom-to-locant mapping for the chain
    atom_to_locant = {}
    for i, atom_idx in enumerate(chain):
        atom_to_locant[atom_idx] = i + 1

    # Find substituent halogens (and other groups) NOT consumed by acid halide
    substituent_parts = []
    chain_set = set(chain)

    for atom_idx in chain:
        atom = mol.GetAtomWithIdx(atom_idx)
        locant = atom_to_locant[atom_idx]

        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx in chain_set:
                continue
            if nidx in consumed_atoms:
                continue

            sym = neighbor.GetSymbol()
            if sym in HALOGEN_PREFIX:
                substituent_parts.append((locant, HALOGEN_PREFIX[sym]))
            # Could add more substituent types here as needed

    if not substituent_parts:
        return ""

    # Group by prefix name for multipliers
    from collections import Counter
    prefix_groups = {}
    for locant, prefix_name in sorted(substituent_parts):
        if prefix_name not in prefix_groups:
            prefix_groups[prefix_name] = []
        prefix_groups[prefix_name].append(locant)

    # Build prefix string (alphabetical order)
    parts = []
    for prefix_name in sorted(prefix_groups.keys()):
        locants = prefix_groups[prefix_name]
        locant_str = ",".join(str(l) for l in sorted(locants))
        if len(locants) > 1:
            multiplier = get_multiplier_prefix(len(locants))
            parts.append(f"{locant_str}-{multiplier}{prefix_name}")
        else:
            parts.append(f"{locant_str}-{prefix_name}")

    return "-".join(parts)


def get_acid_halide_consumed_atoms(functional_groups: Dict) -> set:
    """Get the set of atom indices consumed by acid halide groups.

    These atoms should be filtered from halogen FG detection to prevent
    double-counting (e.g., Cl appearing as both "oyl chloride" and "chloro").

    Args:
        functional_groups: Dict from detect_functional_groups().

    Returns:
        Set of atom indices consumed by acid halide groups.
    """
    consumed = set()
    for fg_name in ("acid_chloride", "acid_bromide", "acid_fluoride"):
        for match in functional_groups.get(fg_name, []):
            # match = (C, O, X) from SMARTS [CX3](=O)[X]
            # The halogen X is at index 2 (third element)
            if len(match) >= 3:
                consumed.add(match[2])  # halogen atom
    return consumed
