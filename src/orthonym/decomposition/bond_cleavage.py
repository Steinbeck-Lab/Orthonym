"""
Bond cleavage detection for decomposition engine.

Identifies ester, amide, and glycosidic bonds suitable for cleavage,
with guards to exclude cyclic variants (lactones, lactams) and
overlapping patterns (carbamates, ureas).
"""

from typing import Dict, List, Set, Tuple

from rdkit import Chem


# ---------------------------------------------------------------------------
# SMARTS patterns for cleavable bond types
# ---------------------------------------------------------------------------

# Ester: C(=O)-O-C  (atoms: 0=carbonyl C, 1==O, 2=ester O, 3=alkyl C)
_ESTER_SMARTS = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")

# Amide: C(=O)-N    (atoms: 0=carbonyl C, 1==O, 2=N)
_AMIDE_SMARTS = Chem.MolFromSmarts("[CX3](=O)[NX3]")

# Carbamate: N-C(=O)-O-C  (atoms: 0=N, 1=carbonyl C, 2==O, 3=ester O, 4=alkyl C)
_CARBAMATE_SMARTS = Chem.MolFromSmarts("[NX3][CX3](=O)[OX2][#6]")

# Glycosidic: ring-C(-O-ring)-O-C  -- anomeric C-O bond to non-ring
_GLYCOSIDIC_SMARTS = Chem.MolFromSmarts("[CX4;R]([OX2;R])[OX2;!R][#6]")


def _atoms_in_same_ring(mol, atom1: int, atom2: int) -> bool:
    """Check if two atoms share a ring membership."""
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if atom1 in ring and atom2 in ring:
            return True
    return False


def find_cleavable_bonds(mol) -> List[Dict]:
    """Find cleavable bonds in a molecule.

    Detects ester C-O, amide C-N, and glycosidic C-O-C bonds,
    excluding lactones, lactams, carbamates, and ureas.

    The detection order matters:
    1. Carbamates are detected first to mark overlapping carbonyl C atoms.
    2. Esters are detected, skipping any carbonyl C already in a carbamate.
    3. Amides are detected, skipping any carbonyl C already in a carbamate.
    4. Glycosidic bonds are detected independently.

    Args:
        mol: RDKit Mol object

    Returns:
        List of dicts with keys: bond_idx, type, match, acid_atom,
        alkyl_atom (for esters) or amine_atom (for amides).
    """
    cleavable: List[Dict] = []
    seen_bond_indices: Set[int] = set()

    # --- Step 1: Detect carbamates first to build exclusion set ---
    carbamate_carbonyl_atoms: Set[int] = set()

    if _CARBAMATE_SMARTS is not None:
        for match in mol.GetSubstructMatches(_CARBAMATE_SMARTS):
            nitrogen = match[0]
            carbonyl_c = match[1]
            # match[2] is =O, match[3] is ester O, match[4] is alkyl C
            ester_o = match[3]

            carbamate_carbonyl_atoms.add(carbonyl_c)

            # Record the C-O bond (ester-like portion) as carbamate type
            bond = mol.GetBondBetweenAtoms(carbonyl_c, ester_o)
            if bond and bond.GetIdx() not in seen_bond_indices:
                # Lactone guard: skip if both atoms in same ring
                if not _atoms_in_same_ring(mol, carbonyl_c, ester_o):
                    seen_bond_indices.add(bond.GetIdx())
                    cleavable.append({
                        "bond_idx": bond.GetIdx(),
                        "type": "carbamate",
                        "acid_atom": carbonyl_c,
                        "alkyl_atom": match[4] if len(match) > 4 else ester_o,
                        "amine_atom": nitrogen,
                        "match": match,
                    })

    # --- Step 2: Detect ester bonds (skip lactones, skip carbamate overlap) ---
    if _ESTER_SMARTS is not None:
        for match in mol.GetSubstructMatches(_ESTER_SMARTS):
            carbonyl_c = match[0]
            # match[1] is =O
            ester_o = match[2]
            alkyl_c = match[3]

            # Skip if this carbonyl C is part of a carbamate
            if carbonyl_c in carbamate_carbonyl_atoms:
                continue

            # Lactone guard: skip if carbonyl C and ester O are in the same ring
            if _atoms_in_same_ring(mol, carbonyl_c, ester_o):
                continue

            # Get the bond between carbonyl C and ester O
            bond = mol.GetBondBetweenAtoms(carbonyl_c, ester_o)
            if bond and bond.GetIdx() not in seen_bond_indices:
                seen_bond_indices.add(bond.GetIdx())
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "ester",
                    "acid_atom": carbonyl_c,
                    "alkyl_atom": alkyl_c,
                    "match": match,
                })

    # --- Step 3: Detect amide bonds (skip lactams, skip carbamate overlap) ---
    if _AMIDE_SMARTS is not None:
        for match in mol.GetSubstructMatches(_AMIDE_SMARTS):
            carbonyl_c = match[0]
            # match[1] is =O
            nitrogen = match[2]

            # Skip if this carbonyl C is part of a carbamate
            if carbonyl_c in carbamate_carbonyl_atoms:
                continue

            # Lactam guard: skip if carbonyl C and N are in the same ring
            if _atoms_in_same_ring(mol, carbonyl_c, nitrogen):
                continue

            # Get the bond between carbonyl C and N
            bond = mol.GetBondBetweenAtoms(carbonyl_c, nitrogen)
            if bond and bond.GetIdx() not in seen_bond_indices:
                seen_bond_indices.add(bond.GetIdx())
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "amide",
                    "acid_atom": carbonyl_c,
                    "amine_atom": nitrogen,
                    "match": match,
                })

    # --- Step 4: Detect glycosidic bonds ---
    if _GLYCOSIDIC_SMARTS is not None:
        for match in mol.GetSubstructMatches(_GLYCOSIDIC_SMARTS):
            anomeric_c = match[0]
            ring_o = match[1]
            glycosidic_o = match[2]
            aglycone_c = match[3]

            # Get the bond between anomeric C and glycosidic O
            bond = mol.GetBondBetweenAtoms(anomeric_c, glycosidic_o)
            if bond and bond.GetIdx() not in seen_bond_indices:
                seen_bond_indices.add(bond.GetIdx())
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "glycosidic",
                    "acid_atom": anomeric_c,
                    "alkyl_atom": aglycone_c,
                    "match": match,
                })

    return cleavable
