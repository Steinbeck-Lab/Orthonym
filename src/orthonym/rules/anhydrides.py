"""
Anhydride naming using IUPAC functional class nomenclature.

Anhydrides use "functional class" naming:
    Symmetric:  "{acid name} anhydride"
    Mixed:      "{acid1} {acid2} anhydride" (alphabetical order)
    Cyclic:     "{diacid name} anhydride"

Examples:
    CC(=O)OC(=O)C       -> ethanoic anhydride (symmetric)
    CC(=O)OC(=O)CC      -> ethanoic propanoic anhydride (mixed, alphabetical)
    CCC(=O)OC(=O)CCC    -> butanoic anhydride (symmetric)
    O=C1CCC(=O)O1       -> butanedioic anhydride (cyclic, from succinic acid)
    O=C1CCCC(=O)O1      -> pentanedioic anhydride (cyclic, from glutaric acid)

The acid name derives from the acyl fragment:
    ethanoyl + ethanoyl -> ethanoic acid -> "ethanoic anhydride"

Reference: IUPAC 2013 Blue Book, P-65.3.1 (Acid anhydrides)
"""

from collections import deque
from typing import Optional, List, Tuple
from rdkit import Chem

from ..data.chain_names import get_chain_prefix


# Anhydride core SMARTS: C(=O)-O-C(=O)
ANHYDRIDE_SMARTS = "[CX3](=O)[OX2][CX3](=O)"


def name_anhydride(features) -> Optional[str]:
    """
    Name an anhydride compound using functional class nomenclature.

    Args:
        features: MolecularFeatures with principal_group == 'anhydride'.

    Returns:
        IUPAC name string, or None if not an anhydride.
    """
    mol = features.mol
    pg = features.principal_group

    if pg != "anhydride":
        return None

    # Detect anhydride core: C(=O)-O-C(=O)
    pattern = Chem.MolFromSmarts(ANHYDRIDE_SMARTS)
    if pattern is None:
        return None

    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    if not matches:
        return None

    # Take first match
    match = matches[0]
    # SMARTS [CX3](=O)[OX2][CX3](=O) matches: (C1, O1_carbonyl, O_bridge, C2, O2_carbonyl)
    # But RDKit returns atom indices in SMARTS order.
    # Let's identify the atoms from the SMARTS pattern:
    # Position 0: first carbonyl C
    # Position 1: first carbonyl O (=O)
    # Position 2: bridge O
    # Position 3: second carbonyl C
    # Position 4: second carbonyl O (=O)

    # Actually, for the SMARTS [CX3](=O)[OX2][CX3](=O), the match tuple is:
    # (C1, O1, O_bridge, C2, O2) where O1 and O2 are carbonyl oxygens
    c1_idx = match[0]
    c2_idx = match[3] if len(match) >= 5 else match[2]

    # Re-parse more carefully using atom environment
    anhydride_info = _parse_anhydride_core(mol, matches[0])
    if anhydride_info is None:
        return None

    c1_idx = anhydride_info['c1']
    c2_idx = anhydride_info['c2']
    bridge_o = anhydride_info['bridge_o']

    # Check if cyclic anhydride (both carbonyl carbons in same ring)
    ring_info = mol.GetRingInfo()
    is_cyclic = False
    for ring in ring_info.AtomRings():
        ring_set = set(ring)
        if c1_idx in ring_set and c2_idx in ring_set and bridge_o in ring_set:
            is_cyclic = True
            # Count total carbons in ring (excluding bridge oxygen)
            ring_carbons = [idx for idx in ring if mol.GetAtomWithIdx(idx).GetSymbol() == 'C']
            total_chain_length = len(ring_carbons)
            break

    if is_cyclic:
        return _name_cyclic_anhydride(total_chain_length)

    # Acyclic anhydride: count carbon chains on each side
    chain1_length = _count_acyl_fragment_carbons(mol, c1_idx, bridge_o)
    chain2_length = _count_acyl_fragment_carbons(mol, c2_idx, bridge_o)

    if chain1_length == chain2_length:
        # Symmetric anhydride
        acid_name = _build_acid_name(chain1_length)
        return f"{acid_name} anhydride"
    else:
        # Mixed/asymmetric anhydride: alphabetical order
        acid1 = _build_acid_name(chain1_length)
        acid2 = _build_acid_name(chain2_length)
        # Sort alphabetically
        acids = sorted([acid1, acid2])
        return f"{acids[0]} {acids[1]} anhydride"


def _parse_anhydride_core(mol, match: tuple) -> Optional[dict]:
    """Parse the anhydride core atoms from a SMARTS match.

    The SMARTS [CX3](=O)[OX2][CX3](=O) can return atoms in different orders
    depending on the molecule. This function identifies the key atoms reliably.

    Args:
        mol: RDKit Mol object.
        match: Tuple of atom indices from SMARTS match.

    Returns:
        Dict with keys 'c1', 'c2', 'bridge_o', 'o1', 'o2', or None.
    """
    # From the SMARTS pattern, we know:
    # - There are exactly 2 carbonyl carbons (CX3)
    # - There is exactly 1 bridge oxygen (OX2, single-bonded)
    # - There are 2 carbonyl oxygens (=O)

    match_set = set(match)
    carbons = []
    oxygens = []

    for idx in match:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'C':
            carbons.append(idx)
        elif atom.GetSymbol() == 'O':
            oxygens.append(idx)

    if len(carbons) != 2:
        return None

    # Identify bridge oxygen: bonded to both carbonyl carbons
    bridge_o = None
    carbonyl_os = []
    for oidx in oxygens:
        o_atom = mol.GetAtomWithIdx(oidx)
        bonded_carbons = set()
        for nbr in o_atom.GetNeighbors():
            if nbr.GetIdx() in set(carbons):
                bonded_carbons.add(nbr.GetIdx())
        if len(bonded_carbons) == 2:
            bridge_o = oidx
        else:
            carbonyl_os.append(oidx)

    if bridge_o is None:
        # Fallback: find the oxygen that is single-bonded to both C
        for oidx in oxygens:
            o_atom = mol.GetAtomWithIdx(oidx)
            c_neighbors = [n.GetIdx() for n in o_atom.GetNeighbors()
                           if n.GetIdx() in set(carbons)]
            if len(c_neighbors) == 2:
                bridge_o = oidx
                break

    if bridge_o is None:
        return None

    return {
        'c1': carbons[0],
        'c2': carbons[1],
        'bridge_o': bridge_o,
        'o1': carbonyl_os[0] if len(carbonyl_os) > 0 else None,
        'o2': carbonyl_os[1] if len(carbonyl_os) > 1 else None,
    }


def _count_acyl_fragment_carbons(mol, carbonyl_c: int, bridge_o: int) -> int:
    """Count carbons in an acyl fragment (one side of the anhydride).

    BFS from carbonyl carbon, following only carbon atoms, avoiding the bridge oxygen.

    Args:
        mol: RDKit Mol object.
        carbonyl_c: Atom index of the carbonyl carbon.
        bridge_o: Atom index of the bridge oxygen (to exclude).

    Returns:
        Number of carbons in the acyl fragment (including carbonyl carbon).
    """
    visited = {carbonyl_c}
    queue = deque([carbonyl_c])
    carbon_count = 1

    while queue:
        current = queue.popleft()
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx in visited:
                continue
            if nidx == bridge_o:
                continue
            # Only follow carbon atoms (skip oxygens, halogens etc.)
            if neighbor.GetSymbol() == 'C':
                visited.add(nidx)
                queue.append(nidx)
                carbon_count += 1

    return carbon_count


def _build_acid_name(chain_length: int) -> str:
    """Build the acid name from chain length.

    Args:
        chain_length: Number of carbons including the carbonyl carbon.

    Returns:
        Acid name string (e.g., 'ethanoic', 'propanoic', 'butanoic').
    """
    prefix = get_chain_prefix(chain_length)
    return f"{prefix}anoic"


def _name_cyclic_anhydride(num_carbons: int) -> str:
    """Name a cyclic anhydride from a dicarboxylic acid.

    Cyclic anhydrides are named as "{diacid name} anhydride".
    The diacid name is based on the number of carbons.

    Args:
        num_carbons: Number of carbon atoms in the ring (both carbonyl carbons + chain).

    Returns:
        IUPAC name string (e.g., 'butanedioic anhydride', 'pentanedioic anhydride').
    """
    prefix = get_chain_prefix(num_carbons)
    return f"{prefix}anedioic anhydride"


def get_anhydride_consumed_atoms(functional_groups: dict) -> set:
    """Get the set of atom indices consumed by anhydride groups.

    The bridge oxygen and ester-like bonds in anhydrides should be filtered
    from ester FG detection to prevent double-counting.

    Args:
        functional_groups: Dict from detect_functional_groups().

    Returns:
        Set of all atom indices that are part of anhydride groups.
    """
    consumed = set()
    for match in functional_groups.get("anhydride", []):
        consumed.update(match)
    return consumed
