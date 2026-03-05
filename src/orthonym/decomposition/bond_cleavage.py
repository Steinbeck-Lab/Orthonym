"""
Bond cleavage detection for decomposition engine.

Identifies ester, amide, glycosidic, carbamate, ether, phosphodiester,
thioester, and sulfonamide bonds suitable for cleavage, with guards to
exclude cyclic variants (lactones, lactams, thiolactones, sultams,
epoxides, cyclic phosphodiesters) and overlapping patterns (carbamates,
ureas, skeletal replacement chains).
"""

from collections import deque
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

# Phosphodiester: O-P(=O)(O)-O-C
# (atoms: 0=ester_o1, 1=P, 2==O, 3=hydroxyl/anionic O, 4=ester_o2, 5=alkyl_c)
_PHOSPHODIESTER_SMARTS = Chem.MolFromSmarts("[OX2][PX4](=O)([OX2,OX1-])[OX2][#6]")

# Thioester: C(=O)-S-C  (atoms: 0=carbonyl C, 1==O, 2=sulfur, 3=alkyl C)
_THIOESTER_SMARTS = Chem.MolFromSmarts("[CX3](=O)[SX2][#6]")

# Sulfonamide: S(=O)(=O)-N  (atoms: 0=sulfur, 1==O, 2==O, 3=nitrogen)
_SULFONAMIDE_SMARTS = Chem.MolFromSmarts("[SX4](=O)(=O)[NX3]")

# Ether: C-O-C where O is divalent, NOT in a ring, and neither C is a
# carbonyl carbon or anomeric center. Excludes esters, glycosidic bonds,
# epoxides, tetrahydropyran-type ring ethers.
_ETHER_SMARTS = Chem.MolFromSmarts(
    "[#6;!$(C=O);!$(C([OX2;R])[OX2;!R])]-[OX2;!R]-[#6;!$(C=O)]"
)


def _atoms_in_same_ring(mol, atom1: int, atom2: int) -> bool:
    """Check if two atoms share a ring membership."""
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if atom1 in ring and atom2 in ring:
            return True
    return False


def _is_skeletal_ether(mol, o_idx: int) -> bool:
    """Check if an ether oxygen is part of a skeletal replacement chain.

    Returns True if either neighbor of the oxygen has another non-ring
    heteroatom (O, N, S) neighbor (excluding the oxygen itself), indicating
    a polyether or oxa-chain that should use skeletal replacement naming
    rather than ether bond cleavage.

    Args:
        mol: RDKit Mol object.
        o_idx: Atom index of the ether oxygen.

    Returns:
        True if the oxygen is part of a skeletal replacement chain.
    """
    o_atom = mol.GetAtomWithIdx(o_idx)
    for nbr in o_atom.GetNeighbors():
        for nbr2 in nbr.GetNeighbors():
            if (nbr2.GetIdx() != o_idx
                    and nbr2.GetSymbol() in ('O', 'N', 'S')
                    and not nbr2.IsInRing()):
                return True
    return False


def _bfs_heavy_atoms(mol, start: int, excluded: Set[int]) -> Set[int]:
    """BFS from start atom, skipping excluded atoms, collecting heavy atoms.

    Args:
        mol: RDKit Mol object.
        start: Starting atom index.
        excluded: Set of atom indices to not cross through.

    Returns:
        Set of heavy atom indices reachable from start.
    """
    visited: Set[int] = set()
    queue = deque([start])
    while queue:
        curr = queue.popleft()
        if curr in visited or curr in excluded:
            continue
        visited.add(curr)
        atom = mol.GetAtomWithIdx(curr)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in visited and nidx not in excluded:
                queue.append(nidx)
    # Filter to heavy atoms only (exclude H, atomic num > 1)
    return {idx for idx in visited
            if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1}


def _assign_ether_roles_by_seniority(
    mol, carbon1: int, carbon2: int,
    side1: Set[int], side2: Set[int]
) -> Tuple[int, int]:
    """Assign ether parent/substituent roles using P-44.1.1 seniority.

    The more senior side becomes acid_atom (parent). Falls back to
    atom count heuristic if seniority scoring fails or results in a tie.

    Args:
        mol: RDKit Mol object.
        carbon1: Atom index of the first ether carbon.
        carbon2: Atom index of the second ether carbon.
        side1: Heavy atom indices on carbon1's side.
        side2: Heavy atom indices on carbon2's side.

    Returns:
        (acid_atom, alkyl_atom) tuple.
    """
    try:
        from .fragment_ranker import score_fragment_seniority

        smiles1 = Chem.MolFragmentToSmiles(mol, atomsToUse=list(side1))
        smiles2 = Chem.MolFragmentToSmiles(mol, atomsToUse=list(side2))

        if smiles1 and smiles2:
            score1 = score_fragment_seniority(smiles1)
            score2 = score_fragment_seniority(smiles2)

            if score1 < score2:
                # Side 1 is more senior -> parent
                return carbon1, carbon2
            elif score2 < score1:
                # Side 2 is more senior -> parent
                return carbon2, carbon1
            # Scores equal: fall through to atom count
    except Exception:
        pass  # Any failure: fall back to atom count

    # Fallback: larger side = parent (original heuristic)
    if len(side1) > len(side2):
        return carbon1, carbon2
    elif len(side2) > len(side1):
        return carbon2, carbon1
    else:
        return min(carbon1, carbon2), max(carbon1, carbon2)


def find_cleavable_bonds(mol) -> List[Dict]:
    """Find cleavable bonds in a molecule.

    Detects 8 bond types: carbamate, phosphodiester, ester, thioester,
    amide, sulfonamide, glycosidic, and ether bonds. Excludes cyclic
    variants (lactones, lactams, thiolactones, sultams, cyclic
    phosphodiesters, epoxides) and overlapping patterns (carbamates,
    ureas, skeletal replacement chains).

    The detection order matters:
    1. Carbamates are detected first to mark overlapping carbonyl C atoms.
    2. Phosphodiesters are detected (P-O bond to alkyl C).
    3. Esters are detected, skipping any carbonyl C already in a carbamate.
    4. Thioesters are detected (C(=O)-S-C), skipping carbamate overlap
       and thiolactones.
    5. Amides are detected, skipping any carbonyl C already in a carbamate.
    6. Sulfonamides are detected (S(=O)(=O)-N), excluding sultams.
    7. Glycosidic bonds are detected independently.
    8. Ether bonds are detected with 5 guards (ring, ester-exclusion,
       glycosidic-exclusion, skeletal-replacement, minimum-fragment-size).

    Args:
        mol: RDKit Mol object

    Returns:
        List of dicts with keys: bond_idx, type, match, acid_atom,
        alkyl_atom (for esters/ethers/thioesters/phosphodiesters/sulfonamides)
        or amine_atom (for amides).
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

    # --- Step 2: Detect phosphodiester bonds ---
    if _PHOSPHODIESTER_SMARTS is not None:
        for match in mol.GetSubstructMatches(_PHOSPHODIESTER_SMARTS):
            # match[0]=ester_o1, match[1]=P, match[2]==O, match[3]=hydroxyl/anionic O,
            # match[4]=ester_o2, match[5]=alkyl_c
            phosphorus = match[1]
            ester_o2 = match[4]
            alkyl_c = match[5]

            # Ring guard: exclude cyclic phosphodiesters (sugar-phosphate rings)
            if _atoms_in_same_ring(mol, phosphorus, ester_o2):
                continue

            # Get the bond between P and ester_o2 (the P-O-C bond to cleave)
            bond = mol.GetBondBetweenAtoms(phosphorus, ester_o2)
            if bond and bond.GetIdx() not in seen_bond_indices:
                seen_bond_indices.add(bond.GetIdx())
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "phosphodiester",
                    "acid_atom": phosphorus,
                    "alkyl_atom": alkyl_c,
                    "match": match,
                })

    # --- Step 3: Detect ester bonds (skip lactones, skip carbamate overlap) ---
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

    # --- Step 4: Detect thioester bonds (C(=O)-S-C) ---
    if _THIOESTER_SMARTS is not None:
        for match in mol.GetSubstructMatches(_THIOESTER_SMARTS):
            carbonyl_c = match[0]
            # match[1] is =O
            sulfur = match[2]
            alkyl_c = match[3]

            # Skip if this carbonyl C is part of a carbamate
            if carbonyl_c in carbamate_carbonyl_atoms:
                continue

            # Thiolactone guard: skip if carbonyl C and sulfur are in the same ring
            if _atoms_in_same_ring(mol, carbonyl_c, sulfur):
                continue

            # Get the bond between carbonyl C and sulfur (C-S bond to cleave)
            bond = mol.GetBondBetweenAtoms(carbonyl_c, sulfur)
            if bond and bond.GetIdx() not in seen_bond_indices:
                seen_bond_indices.add(bond.GetIdx())
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "thioester",
                    "acid_atom": carbonyl_c,
                    "alkyl_atom": alkyl_c,
                    "match": match,
                })

    # --- Step 5: Detect amide bonds (skip lactams, skip carbamate overlap) ---
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

    # --- Step 6: Detect sulfonamide bonds (S(=O)(=O)-N) ---
    if _SULFONAMIDE_SMARTS is not None:
        for match in mol.GetSubstructMatches(_SULFONAMIDE_SMARTS):
            sulfur = match[0]
            # match[1] and match[2] are =O
            nitrogen = match[3]

            # Sultam guard: skip if sulfur and nitrogen are in the same ring
            if _atoms_in_same_ring(mol, sulfur, nitrogen):
                continue

            # Get the bond between sulfur and nitrogen (S-N bond to cleave)
            bond = mol.GetBondBetweenAtoms(sulfur, nitrogen)
            if bond and bond.GetIdx() not in seen_bond_indices:
                seen_bond_indices.add(bond.GetIdx())
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "sulfonamide",
                    "acid_atom": sulfur,
                    "alkyl_atom": nitrogen,
                    "match": match,
                })

    # --- Step 7: Detect glycosidic bonds ---
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

    # --- Step 8: Detect ether bonds (C-O-C, not ester/glycosidic/ring) ---
    if _ETHER_SMARTS is not None:
        for match in mol.GetSubstructMatches(_ETHER_SMARTS):
            carbon1 = match[0]
            oxygen = match[1]
            carbon2 = match[2]

            # Ring guard: skip if either carbon and the oxygen are in the
            # same ring (epoxides, oxetane, tetrahydropyran, etc.)
            if _atoms_in_same_ring(mol, carbon1, oxygen):
                continue
            if _atoms_in_same_ring(mol, carbon2, oxygen):
                continue

            # Skeletal replacement guard: skip if the oxygen is part of a
            # chain with multiple heteroatoms (oxa-naming applies instead)
            if _is_skeletal_ether(mol, oxygen):
                continue

            # Determine cleavage bond: bond between the larger-side carbon
            # and the ether oxygen. The oxygen stays with the smaller fragment,
            # producing an alcohol that converts to an alkoxy prefix.
            # Use BFS to count heavy atoms on each side (excluding the oxygen).
            side1 = _bfs_heavy_atoms(mol, carbon1, excluded={oxygen})
            side2 = _bfs_heavy_atoms(mol, carbon2, excluded={oxygen})

            # Minimum fragment size guard: skip if either side < 5 heavy atoms
            if len(side1) < 5 or len(side2) < 5:
                continue

            # Assign roles using P-44.1.1 seniority: the more senior side
            # becomes acid_atom (parent). Falls back to atom count if
            # seniority scoring fails or ties.
            acid_atom, alkyl_atom = _assign_ether_roles_by_seniority(
                mol, carbon1, carbon2, side1, side2
            )

            # Cleavage bond = bond between larger-side carbon and oxygen
            # This way oxygen stays with the smaller fragment (alkyl side)
            bond = mol.GetBondBetweenAtoms(acid_atom, oxygen)

            # Already-seen guard
            if bond and bond.GetIdx() not in seen_bond_indices:
                seen_bond_indices.add(bond.GetIdx())
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "ether",
                    "acid_atom": acid_atom,
                    "alkyl_atom": alkyl_atom,
                    "match": match,
                })

    return cleavable
