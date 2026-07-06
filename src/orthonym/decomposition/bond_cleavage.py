"""
Bond cleavage detection for decomposition engine.

Identifies ester, amide, glycosidic, carbamate, ether, phosphodiester,
thioester, and sulfonamide bonds suitable for cleavage, with guards to
exclude cyclic variants (lactones, lactams, thiolactones, sultams,
epoxides, cyclic phosphodiesters) and overlapping patterns (carbamates,
ureas, skeletal replacement chains).
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

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

# Thioether: C-S-C where S is divalent, NOT in a ring, and neither C
# is a carbonyl carbon. Excludes thioesters (C(=O)-S), thiolactones (ring S),
# and sulfides within ring systems.
_THIOETHER_SMARTS = Chem.MolFromSmarts(
    "[#6;!$(C=O);!$(C([SX2;R])[SX2;!R])]-[SX2;!R]-[#6;!$(C=O)]"
)

# Secondary amine: C-NH-C where N is trivalent with 1 H, NOT in a ring,
# and neither C is a carbonyl carbon. Excludes amides (C(=O)-N), lactams
# (ring N), imines (C=N), and guanidines.
_SEC_AMINE_SMARTS = Chem.MolFromSmarts(
    "[#6;!$(C=O);!$(C=N)]-[NX3H1;!R]-[#6;!$(C=O);!$(C=N)]"
)


def _atoms_in_same_ring(mol, atom1: int, atom2: int) -> bool:
    """Check if two atoms share a ring membership."""
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if atom1 in ring and atom2 in ring:
            return True
    return False


def _is_benzylic_sp3_carbon(mol, c_idx: int) -> bool:
    """True if ``c_idx`` is an acyclic sp3 carbon bonded to an aromatic ring atom.

    v22 C-T2 (V-3). Such a carbon (a benzylic ``-CH2-``/``-CHR-`` linker) marks an
    aralkyl ether (e.g. benzyl phenyl ether ``c1ccccc1COc1ccccc1``) that is named
    SUBSTITUTIVELY as an (aryloxy/alkoxy)alkyl-substituted parent —
    ``(phenoxymethyl)benzene`` — NOT by functional-class ether cleavage, which
    mis-places the O on the ring (``phenoxytoluene``, a different constitution).
    """
    atom = mol.GetAtomWithIdx(c_idx)
    if (atom.GetSymbol() != 'C' or atom.GetIsAromatic()
            or atom.IsInRing()
            or atom.GetHybridization() != Chem.HybridizationType.SP3):
        return False
    return any(n.GetIsAromatic() for n in atom.GetNeighbors())


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


def _is_skeletal_thioether(mol, s_idx: int) -> bool:
    """Check if a thioether sulfur is part of a skeletal replacement chain.

    Returns True if either neighbor of the sulfur has another non-ring
    heteroatom (O, N, S) neighbor (excluding the sulfur itself), indicating
    a polythioether or thia-chain that should use skeletal replacement naming
    rather than thioether bond cleavage.

    Args:
        mol: RDKit Mol object.
        s_idx: Atom index of the thioether sulfur.

    Returns:
        True if the sulfur is part of a skeletal replacement chain.
    """
    s_atom = mol.GetAtomWithIdx(s_idx)
    for nbr in s_atom.GetNeighbors():
        for nbr2 in nbr.GetNeighbors():
            if (nbr2.GetIdx() != s_idx
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


def _maybe_swap_parent_roles(
    mol, acid_idx: int, other_idx: int, bond_type: str,
    bridging_atoms: Optional[Set[int]] = None,
) -> Tuple[int, int, bool]:
    """Check if parent/child roles should be swapped based on P-44.1.1 seniority.

    Uses BFS to identify fragments on each side of the bond, then compares
    them using score_fragment_seniority(). Includes a size guard: when the
    non-acid fragment has >=3x heavy atoms AND equal/higher seniority, swap.

    Simple ester exemption: when bond_type == "ester" and both fragments
    have <10 heavy atoms, keep traditional acid-side parent for functional
    class naming (preserves "ethyl acetate" style).

    Args:
        mol: RDKit Mol object.
        acid_idx: Atom index currently designated as acid side.
        other_idx: Atom index currently designated as alkyl/amine side.
        bond_type: Bond type string for exemption logic.
        bridging_atoms: Set of atom indices that bridge between acid_idx
            and other_idx (e.g., ester oxygen, thioester sulfur). These
            are excluded from BFS on both sides. None for directly bonded
            atoms (amide C-N, sulfonamide S-N).

    Returns:
        (final_acid_idx, final_other_idx, roles_swapped) tuple.
    """
    try:
        from .fragment_ranker import score_fragment_seniority

        # Build BFS exclusion sets: each side excludes the other side's
        # starting atom plus any bridging atoms.
        bridge = bridging_atoms or set()
        excluded_for_acid = {other_idx} | bridge
        excluded_for_other = {acid_idx} | bridge

        acid_side = _bfs_heavy_atoms(mol, acid_idx, excluded=excluded_for_acid)
        other_side = _bfs_heavy_atoms(mol, other_idx, excluded=excluded_for_other)

        # Simple ester exemption: both sides <10 HA -> keep traditional roles
        if bond_type == "ester" and len(acid_side) < 10 and len(other_side) < 10:
            return acid_idx, other_idx, False

        # Amide exemption: never swap at detection time. The engine already
        # has a robust post-cleavage amide N-acyl seniority check
        # (engine.py:835-852) that operates on the actual capped fragment
        # SMILES (which are smaller, don't cross other cleavable bonds).
        # Detection-time swap for amides causes regressions in peptide-like
        # and macrocyclic compounds where BFS traverses through other amide
        # bonds, creating misleading size/seniority asymmetry.
        if bond_type == "amide":
            return acid_idx, other_idx, False

        # Generate fragment SMILES for seniority scoring
        smiles_acid = Chem.MolFragmentToSmiles(mol, atomsToUse=list(acid_side))
        smiles_other = Chem.MolFragmentToSmiles(mol, atomsToUse=list(other_side))

        if not smiles_acid or not smiles_other:
            return acid_idx, other_idx, False

        score_acid = score_fragment_seniority(smiles_acid)
        score_other = score_fragment_seniority(smiles_other)

        # Size guard: extreme size asymmetry overrides minor seniority diffs.
        # The acid fragment's capping (OH -> aldehyde/acid) can artificially
        # inflate its seniority rank.  Three tiers:
        #
        # 1) >=5x HA ratio: always swap (overwhelming size difference).
        # 2) >=3x HA ratio AND other has ring while acid does not: swap
        #    (ring-containing fragment is the obvious parent).
        # 3) >=3x HA ratio AND other is at least as senior: swap.
        size_ratio_met = len(other_side) >= 3 * len(acid_side)
        if size_ratio_met:
            if len(other_side) >= 5 * len(acid_side):
                return other_idx, acid_idx, True  # SWAP -- overwhelming size

            # Check ring presence: score tuple position 3 is -has_ring
            # (-1 = has ring, 0 = no ring)
            other_has_ring = score_other[3] < 0
            acid_has_ring = score_acid[3] < 0
            if other_has_ring and not acid_has_ring:
                return other_idx, acid_idx, True  # SWAP -- ring vs no ring

            if score_other <= score_acid:
                return other_idx, acid_idx, True  # SWAP -- seniority

        # Do NOT swap based on standard seniority alone. In multi-bond
        # molecules, BFS from one side crosses other cleavable bonds,
        # picking up higher-seniority FGs from distant parts of the molecule.
        # This creates misleading seniority comparisons. Only the size guard
        # above (3x+ ratio with ring/size advantage) should trigger swap.

        return acid_idx, other_idx, False  # No swap

    except Exception:
        return acid_idx, other_idx, False  # Any failure: no swap


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
                    alkyl_c = match[4] if len(match) > 4 else ester_o
                    _, _, swapped = _maybe_swap_parent_roles(
                        mol, carbonyl_c, alkyl_c, "carbamate",
                        bridging_atoms={ester_o},
                    )
                    cleavable.append({
                        "bond_idx": bond.GetIdx(),
                        "type": "carbamate",
                        "acid_atom": carbonyl_c,
                        "alkyl_atom": alkyl_c,
                        "amine_atom": nitrogen,
                        "match": match,
                        "roles_swapped": swapped,
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
                _, _, swapped = _maybe_swap_parent_roles(
                    mol, phosphorus, alkyl_c, "phosphodiester",
                    bridging_atoms={ester_o2},
                )
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "phosphodiester",
                    "acid_atom": phosphorus,
                    "alkyl_atom": alkyl_c,
                    "match": match,
                    "roles_swapped": swapped,
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

            # Anhydride guard: when the "alkyl" carbon is itself a carbonyl
            # carbon the linkage is C(=O)-O-C(=O), i.e. an acid anhydride, not
            # an ester.  Anhydrides are a distinct functional class named by the
            # dedicated handler (rules.anhydrides, P-65.7); cleaving the bridge
            # O into two acid fragments yields a constitutionally different
            # multi-component name (benzoic anhydride -> "benzoic acid
            # benzoate" = two molecules).  Skip both orientations of the match.
            _alkyl = mol.GetAtomWithIdx(alkyl_c)
            if _alkyl.GetSymbol() == 'C' and any(
                b.GetBondTypeAsDouble() == 2.0
                and b.GetOtherAtom(_alkyl).GetSymbol() == 'O'
                for b in _alkyl.GetBonds()
            ):
                continue

            # Get the bond between carbonyl C and ester O
            bond = mol.GetBondBetweenAtoms(carbonyl_c, ester_o)
            if bond and bond.GetIdx() not in seen_bond_indices:
                seen_bond_indices.add(bond.GetIdx())
                _, _, swapped = _maybe_swap_parent_roles(
                    mol, carbonyl_c, alkyl_c, "ester",
                    bridging_atoms={ester_o},
                )
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "ester",
                    "acid_atom": carbonyl_c,
                    "alkyl_atom": alkyl_c,
                    "match": match,
                    "roles_swapped": swapped,
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
                _, _, swapped = _maybe_swap_parent_roles(
                    mol, carbonyl_c, alkyl_c, "thioester",
                    bridging_atoms={sulfur},
                )
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "thioester",
                    "acid_atom": carbonyl_c,
                    "alkyl_atom": alkyl_c,
                    "match": match,
                    "roles_swapped": swapped,
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

            # Wave2 ring-hydrazide (P-66.3.1.1): a C(=O)-N bond whose nitrogen
            # belongs to an N-N unit is a HYDRAZIDE bond, not an amide bond.
            # Functional-class cleavage cannot express which nitrogen carries
            # the acyl group ('N-benzoylphenylhydrazine' names the WRONG
            # constitution on an unsymmetric hydrazine) and the Blue Book
            # rejects acyl-hydrazine names outright ['not
            # (cyclohexanecarbonyl)hydrazine', 'not 1,2-dibenzoylhydrazine'].
            # The substitutive hydrazide assemblers own this class -- skip so
            # the dispatch cascade reaches them.
            if any(nb.GetSymbol() == 'N'
                   for nb in mol.GetAtomWithIdx(nitrogen).GetNeighbors()):
                continue

            # Get the bond between carbonyl C and N
            bond = mol.GetBondBetweenAtoms(carbonyl_c, nitrogen)
            if bond and bond.GetIdx() not in seen_bond_indices:
                seen_bond_indices.add(bond.GetIdx())
                _, _, swapped = _maybe_swap_parent_roles(
                    mol, carbonyl_c, nitrogen, "amide",
                    bridging_atoms=None,
                )
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "amide",
                    "acid_atom": carbonyl_c,
                    "amine_atom": nitrogen,
                    "match": match,
                    "roles_swapped": swapped,
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
                _, _, swapped = _maybe_swap_parent_roles(
                    mol, sulfur, nitrogen, "sulfonamide",
                    bridging_atoms=None,
                )
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "sulfonamide",
                    "acid_atom": sulfur,
                    "alkyl_atom": nitrogen,
                    "match": match,
                    "roles_swapped": swapped,
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
                    "roles_swapped": False,
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

            # v22 C-T2 (V-3): aralkyl-ether guard. When an ether carbon is a
            # benzylic sp3 linker (-CH2-/-CHR- on an aromatic ring), the molecule
            # is named substitutively as (aryloxy/alkoxy)alkyl, not by ether
            # cleavage which mis-places the O on the ring (phenoxytoluene).
            if (_is_benzylic_sp3_carbon(mol, carbon1)
                    or _is_benzylic_sp3_carbon(mol, carbon2)):
                continue

            # Determine cleavage bond: bond between the larger-side carbon
            # and the ether oxygen. The oxygen stays with the smaller fragment,
            # producing an alcohol that converts to an alkoxy prefix.
            # Use BFS to count heavy atoms on each side (excluding the oxygen).
            side1 = _bfs_heavy_atoms(mol, carbon1, excluded={oxygen})
            side2 = _bfs_heavy_atoms(mol, carbon2, excluded={oxygen})

            # Minimum fragment size guard: skip if either side < 3 heavy atoms
            # (D-03: lowered from 5 to 3 to detect smaller ethers)
            if len(side1) < 3 or len(side2) < 3:
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
                    "roles_swapped": False,
                })

    # --- Step 9: Detect thioether bonds (C-S-C, not thioester/ring) ---
    if _THIOETHER_SMARTS is not None:
        for match in mol.GetSubstructMatches(_THIOETHER_SMARTS):
            carbon1 = match[0]
            sulfur = match[1]
            carbon2 = match[2]

            # Ring guard: skip if either carbon and the sulfur are in the
            # same ring (thiolane, etc.)
            if _atoms_in_same_ring(mol, carbon1, sulfur):
                continue
            if _atoms_in_same_ring(mol, carbon2, sulfur):
                continue

            # Skeletal replacement guard: skip if the sulfur is part of a
            # chain with multiple heteroatoms (thia-naming applies instead)
            if _is_skeletal_thioether(mol, sulfur):
                continue

            # Use BFS to count heavy atoms on each side (excluding sulfur)
            side1 = _bfs_heavy_atoms(mol, carbon1, excluded={sulfur})
            side2 = _bfs_heavy_atoms(mol, carbon2, excluded={sulfur})

            # Minimum fragment size guard: 3 HA per side (D-03)
            if len(side1) < 3 or len(side2) < 3:
                continue

            # Assign roles using P-44.1.1 seniority (same logic as ether)
            acid_atom, alkyl_atom = _assign_ether_roles_by_seniority(
                mol, carbon1, carbon2, side1, side2
            )

            # Cleavage bond = bond between parent-side carbon and sulfur
            bond = mol.GetBondBetweenAtoms(acid_atom, sulfur)

            if bond and bond.GetIdx() not in seen_bond_indices:
                seen_bond_indices.add(bond.GetIdx())
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "thioether",
                    "acid_atom": acid_atom,
                    "alkyl_atom": alkyl_atom,
                    "match": match,
                    "roles_swapped": False,
                })

    # --- Step 10: Detect secondary amine bonds (C-NH-C, not amide/ring) ---
    if _SEC_AMINE_SMARTS is not None:
        for match in mol.GetSubstructMatches(_SEC_AMINE_SMARTS):
            carbon1 = match[0]
            nitrogen = match[1]
            carbon2 = match[2]

            # Ring guard: skip if either carbon and the nitrogen are in the
            # same ring (pyrrolidine, piperidine, etc.)
            if _atoms_in_same_ring(mol, carbon1, nitrogen):
                continue
            if _atoms_in_same_ring(mol, carbon2, nitrogen):
                continue

            # Use BFS to count heavy atoms on each side (excluding nitrogen)
            side1 = _bfs_heavy_atoms(mol, carbon1, excluded={nitrogen})
            side2 = _bfs_heavy_atoms(mol, carbon2, excluded={nitrogen})

            # Minimum fragment size guard: 3 HA per side (D-03)
            if len(side1) < 3 or len(side2) < 3:
                continue

            # Assign roles: more senior side = acid_atom (parent),
            # less senior side = amine_atom
            acid_atom, amine_atom = _assign_ether_roles_by_seniority(
                mol, carbon1, carbon2, side1, side2
            )

            # Cleavage bond = bond between parent-side carbon and nitrogen
            bond = mol.GetBondBetweenAtoms(acid_atom, nitrogen)

            if bond and bond.GetIdx() not in seen_bond_indices:
                seen_bond_indices.add(bond.GetIdx())
                cleavable.append({
                    "bond_idx": bond.GetIdx(),
                    "type": "sec_amine",
                    "acid_atom": acid_atom,
                    "amine_atom": amine_atom,
                    "match": match,
                    "roles_swapped": False,
                })

    return cleavable
