"""
Polyfunctional compound naming coordinator.

Handles naming of compounds with multiple functional groups:
- Principal group (highest seniority) becomes the suffix
- Lower-seniority groups become prefixes with locants
- Ethers (no seniority) are always named as alkoxy prefixes

Based on IUPAC 2013 Blue Book P-41 to P-43.
"""

from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Any, Set

from ..assembly.naming_utils import (
    alpha_sort_key,
    get_multiplier_prefix,
    format_suffix_with_locants,
    get_alkyl_name,
    ALKYL_NAMES,
)
from .seniority import (
    get_principal_group,
    get_prefix,
    SENIORITY_ORDER,
    PREFIX_FORMS,
)


# Functional groups that can be detected but have no seniority
# These are always named as prefixes (never suffix)
NO_SENIORITY_GROUPS = {
    "ether",
    "vinyl_ether",
    "aromatic_ether",
    "thioether",
    "fluoro",
    "chloro",
    "bromo",
    "iodo",
    "nitro",
    "nitroso",
    "azido",
    "alkene",  # Handled separately via unsaturation
    "alkyne",  # Handled separately via unsaturation
}

# Chain prefixes for ether alkoxy naming (match ALKYL_NAMES pattern)
ALKOXY_NAMES = {
    1: "methoxy",
    2: "ethoxy",
    3: "propoxy",
    4: "butoxy",
    5: "pentyloxy",
    6: "hexyloxy",
    7: "heptyloxy",
    8: "octyloxy",
    9: "nonyloxy",
    10: "decyloxy",
}


def detect_polyfunctional(mol, functional_groups: Dict[str, List[tuple]]) -> bool:
    """
    Determine if a molecule has multiple distinct functional groups.

    Returns True if the molecule has 2+ distinct functional groups that
    participate in naming (excludes unsaturation markers and halogens).

    A compound with multiple instances of the SAME group (e.g., diol)
    is NOT considered polyfunctional by this function - that's multiplicity.

    Args:
        mol: RDKit Mol object
        functional_groups: Dict from detect_functional_groups()

    Returns:
        True if molecule is polyfunctional (multiple distinct FGs)
    """
    # Groups that count as "functional" for polyfunctional detection
    # Excludes: alkene/alkyne (unsaturation), halogens (always prefix anyway)
    exclude_groups = {
        "alkene", "alkyne",
        "fluoro", "chloro", "bromo", "iodo",
    }

    # Count distinct functional group types with seniority or special handling
    distinct_groups = set()

    for fg_name, matches in functional_groups.items():
        if not matches:
            continue
        if fg_name in exclude_groups:
            continue

        # Count this as a distinct group
        distinct_groups.add(fg_name)

    return len(distinct_groups) >= 2


def get_fg_prefix_form(
    fg_name: str,
    mol,
    atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """
    Get the prefix form for a functional group.

    For most groups, uses the standard prefix from seniority.py.
    For ethers, determines the alkoxy prefix based on substituent size.

    Args:
        fg_name: Name of the functional group
        mol: RDKit Mol object
        atoms: Atom indices matching this FG instance
        principal_chain: Atom indices of the principal chain

    Returns:
        Prefix string (e.g., "hydroxy", "oxo", "methoxy") or None if no prefix
    """
    # Handle ethers specially - determine alkoxy prefix
    if fg_name in ("ether", "vinyl_ether", "aromatic_ether"):
        return _get_alkoxy_prefix(mol, atoms, principal_chain)

    # For other groups, use standard prefix form
    return get_prefix(fg_name)


def _get_alkoxy_prefix(
    mol,
    ether_atoms: tuple,
    principal_chain: List[int]
) -> str:
    """
    Determine the alkoxy prefix for an ether.

    The ether SMARTS "[OX2]([CX4])[CX4]" matches both carbons.
    We need to determine which side is the substituent (smaller/not in chain)
    and name it as alkoxy.

    Args:
        mol: RDKit Mol object
        ether_atoms: Atom indices from ether SMARTS match (O, C, C)
        principal_chain: Atom indices of the principal chain

    Returns:
        Alkoxy prefix (e.g., "methoxy", "ethoxy")
    """
    # ether_atoms from SMARTS "[OX2]([CX4])[CX4]" = (O, C1, C2)
    if len(ether_atoms) < 3:
        return "alkoxy"  # Fallback

    oxygen_idx = ether_atoms[0]
    carbon1_idx = ether_atoms[1]
    carbon2_idx = ether_atoms[2]

    chain_set = set(principal_chain)

    # Determine which carbon is the substituent (not in principal chain)
    # or the smaller fragment if both/neither in chain
    carbon1_in_chain = carbon1_idx in chain_set
    carbon2_in_chain = carbon2_idx in chain_set

    if carbon1_in_chain and not carbon2_in_chain:
        # carbon2 is the substituent
        sub_carbon = carbon2_idx
    elif carbon2_in_chain and not carbon1_in_chain:
        # carbon1 is the substituent
        sub_carbon = carbon1_idx
    else:
        # Neither or both in chain - use smaller fragment
        # Count atoms on each side via BFS
        frag1_size = _count_fragment_atoms(mol, carbon1_idx, {oxygen_idx})
        frag2_size = _count_fragment_atoms(mol, carbon2_idx, {oxygen_idx})
        sub_carbon = carbon1_idx if frag1_size <= frag2_size else carbon2_idx

    # Check if the substituent is aromatic (phenoxy, benzyloxy)
    sub_atom = mol.GetAtomWithIdx(sub_carbon)

    # Case A: O -> aromatic C in 6-membered all-carbon ring -> "phenoxy"
    if sub_atom.GetIsAromatic():
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if sub_carbon in ring and len(ring) == 6:
                if all(mol.GetAtomWithIdx(r).GetIsAromatic()
                       and mol.GetAtomWithIdx(r).GetSymbol() == 'C'
                       for r in ring):
                    return "phenoxy"
        # Fallback for other aromatic ethers
        return "phenoxy"

    # Case B: O -> CH2 -> aromatic ring -> "benzyloxy"
    if (not sub_atom.GetIsAromatic()
            and sub_atom.GetSymbol() == 'C'
            and sub_atom.GetTotalNumHs() >= 1):
        arom_nbrs = [n for n in sub_atom.GetNeighbors()
                     if n.GetIdx() != oxygen_idx and n.GetIsAromatic()]
        non_h_non_arom = [n for n in sub_atom.GetNeighbors()
                          if n.GetIdx() != oxygen_idx
                          and not n.GetIsAromatic()
                          and n.GetSymbol() != 'H']
        if arom_nbrs and not non_h_non_arom:
            return "benzyloxy"

    # Count carbons in the substituent fragment
    carbon_count = _count_fragment_atoms(mol, sub_carbon, {oxygen_idx}, carbons_only=True)

    # Get alkoxy name
    if carbon_count in ALKOXY_NAMES:
        return ALKOXY_NAMES[carbon_count]
    else:
        # For larger groups, build name from alkyl
        try:
            alkyl = get_alkyl_name(carbon_count)
            # Remove 'yl' and add 'yloxy' for larger alkoxy groups
            if alkyl.endswith("yl"):
                return alkyl[:-2] + "yloxy"
            return alkyl + "oxy"
        except ValueError:
            return "alkoxy"  # Fallback for very large groups


def _count_fragment_atoms(
    mol,
    start_atom: int,
    exclude: Set[int],
    carbons_only: bool = False
) -> int:
    """
    Count atoms in a fragment via BFS.

    Args:
        mol: RDKit Mol object
        start_atom: Starting atom index
        exclude: Atoms to not cross (boundary)
        carbons_only: If True, only count carbon atoms

    Returns:
        Number of atoms (or carbons if carbons_only=True)
    """
    from collections import deque

    visited = set()
    queue = deque([start_atom])
    count = 0

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        if carbons_only:
            if atom.GetSymbol() == 'C':
                count += 1
        else:
            count += 1

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude:
                queue.append(nbr_idx)

    return count


def format_fg_prefix(prefix_form: str, locants: List[int], count: int) -> str:
    """
    Format functional group prefix with locants and multiplier.

    Args:
        prefix_form: Base prefix name (e.g., "hydroxy", "oxo", "methoxy")
        locants: List of locant positions
        count: Number of instances

    Returns:
        Formatted prefix string (e.g., "2-hydroxy", "3-oxo", "2,4-dihydroxy")
    """
    if not locants:
        # No locants - just return prefix with multiplier if needed
        if count > 1:
            multiplier = get_multiplier_prefix(count, prefix_form)
            return f"{multiplier}{prefix_form}"
        return prefix_form

    # Format locants
    locant_str = ",".join(str(loc) for loc in sorted(locants))

    # Get multiplier if multiple instances
    if count > 1:
        multiplier = get_multiplier_prefix(count, prefix_form)
        return f"{locant_str}-{multiplier}{prefix_form}"

    return f"{locant_str}-{prefix_form}"


def _find_fg_center_atom(mol, match: tuple, fg_name: str) -> Optional[int]:
    """
    Find the functional group CENTER atom in a SMARTS match.

    Different FGs have different definitions of "center":
    - Alcohols/thiols: The carbon attached to -OH/-SH (typically index 1 in SMARTS)
    - Ketones: The carbonyl carbon (typically index 1 in SMARTS)
    - Aldehydes: The aldehyde carbon (typically index 0 in SMARTS)

    For most FGs, the center is the FIRST carbon in the match that is
    directly bonded to a heteroatom that is ALSO in the match.

    Args:
        mol: RDKit Mol object
        match: Tuple of atom indices from SMARTS match
        fg_name: Name of the functional group

    Returns:
        Atom index of the FG center, or None if not found
    """
    match_set = set(match)

    # For each carbon in match, check if it's bonded to a heteroatom in match
    for atom_idx in match:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != 'C':
            continue

        # Check if this carbon is bonded to a heteroatom that's also in the match
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in match_set and neighbor.GetSymbol() not in ('C', 'H'):
                # This carbon is directly bonded to a heteroatom in the FG match
                return atom_idx

    # Fallback: return first carbon in match
    for atom_idx in match:
        if mol.GetAtomWithIdx(atom_idx).GetSymbol() == 'C':
            return atom_idx

    return None


def get_non_principal_fg_locants(
    mol,
    fg_atoms: List[tuple],
    principal_chain: List[int],
    atom_to_locant: Dict[int, int],
    fg_name: str = ""
) -> List[int]:
    """
    Get locants for non-principal functional groups.

    For each FG match, find the functional group CENTER atom (the carbon
    bearing the heteroatom) on the principal chain and return its locant.

    Args:
        mol: RDKit Mol object
        fg_atoms: List of atom index tuples for each FG match
        principal_chain: Atom indices of the principal chain
        atom_to_locant: Mapping from atom index to locant
        fg_name: Name of the functional group (for specialized handling)

    Returns:
        Sorted list of locants for the functional group positions
    """
    locants = []

    for match in fg_atoms:
        # Find the FG center atom
        center_idx = _find_fg_center_atom(mol, match, fg_name)

        if center_idx is not None and center_idx in atom_to_locant:
            locants.append(atom_to_locant[center_idx])
        else:
            # Fallback: find any atom in match that's on chain
            for atom_idx in match:
                if atom_idx in atom_to_locant:
                    locants.append(atom_to_locant[atom_idx])
                    break
            else:
                # Check if any atom's neighbor is on the chain
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for neighbor in atom.GetNeighbors():
                        nbr_idx = neighbor.GetIdx()
                        if nbr_idx in atom_to_locant:
                            locants.append(atom_to_locant[nbr_idx])
                            break
                    else:
                        continue
                    break

    # Do NOT deduplicate locants: two groups at the same position need
    # repeated locants (e.g., 3,3-diamino). The count and locant list
    # must agree for OPSIN compatibility.
    return sorted(locants)


def get_non_principal_groups(
    functional_groups: Dict[str, List[tuple]],
    principal_group: Optional[str]
) -> Dict[str, List[tuple]]:
    """
    Get all functional groups that are not the principal group.

    Filters out the principal group and unsaturation markers.

    Args:
        functional_groups: Dict from detect_functional_groups()
        principal_group: Name of the principal group (or None)

    Returns:
        Dict of non-principal functional group names to their atom indices
    """
    exclude_groups = {"alkene", "alkyne"}  # Handled by unsaturation

    result = {}
    for fg_name, matches in functional_groups.items():
        if fg_name == principal_group:
            continue
        if fg_name in exclude_groups:
            continue
        if matches:
            result[fg_name] = matches

    return result


def name_polyfunctional(features: Any) -> Optional[str]:
    """
    Generate IUPAC name for a polyfunctional compound.

    This is the main coordinator function for multi-FG naming.
    If the compound is not polyfunctional, returns None.

    Args:
        features: MolecularFeatures object with extracted features

    Returns:
        Complete IUPAC name string, or None if not polyfunctional
    """
    # Check if this is a polyfunctional compound
    if not getattr(features, 'is_polyfunctional', False):
        return None

    mol = features.mol
    principal_chain = features.principal_chain
    atom_to_locant = features.atom_to_locant
    principal_group = features.principal_group
    non_principal = getattr(features, 'non_principal_groups', {})

    if not principal_chain or not atom_to_locant:
        return None

    # --- EL-02: Ester demotion in polyfunctional context ---
    # When ester is the principal group in a polyfunctional compound, it should
    # NOT use "-oate" suffix. Instead, demote esters to acyloxy prefixes and
    # re-select the next-highest seniority group as the principal group.
    ester_acyloxy_prefixes = []
    _esters_demoted = False
    if principal_group == "ester":
        from ..rules.esters import name_ester_as_prefix
        ester_matches = features.functional_groups.get("ester", [])
        for match in ester_matches:
            acyloxy = name_ester_as_prefix(mol, match)
            if acyloxy:
                # Find the alkyl carbon (last in match) on the principal chain
                alkyl_c = match[-1] if len(match) >= 4 else match[3] if len(match) > 3 else None
                locant = atom_to_locant.get(alkyl_c) if alkyl_c is not None else None
                ester_acyloxy_prefixes.append((acyloxy, locant))

        if ester_acyloxy_prefixes:
            _esters_demoted = True

        # Re-select principal group excluding esters
        filtered_fgs = {k: v for k, v in features.functional_groups.items() if k != "ester"}
        new_principal, new_atoms = get_principal_group(mol, filtered_fgs)
        if new_principal:
            principal_group = new_principal
            features.principal_group_atoms = new_atoms
            # Update non_principal to exclude the new principal group
            non_principal = {k: v for k, v in filtered_fgs.items()
                            if k != new_principal and k not in ("alkene", "alkyne")}

    # Collect all prefixes (FG prefixes + alkyl substituents)
    all_prefixes = []

    # Add ester acyloxy prefixes if esters were demoted
    if ester_acyloxy_prefixes:
        # Group identical acyloxy prefixes for multipliers
        from collections import Counter as _Ctr
        acyloxy_groups: Dict[str, List] = defaultdict(list)
        for acyloxy_name, locant in ester_acyloxy_prefixes:
            acyloxy_groups[acyloxy_name].append(locant)

        for acyloxy_name, locants in sorted(acyloxy_groups.items()):
            valid_locants = sorted(loc for loc in locants if loc is not None)
            count = len(locants)
            if count > 1 and valid_locants:
                locant_str = ",".join(str(loc) for loc in valid_locants)
                multiplier = get_multiplier_prefix(count, acyloxy_name)
                all_prefixes.append(f"{locant_str}-{multiplier}({acyloxy_name})")
            elif valid_locants:
                all_prefixes.append(f"{valid_locants[0]}-({acyloxy_name})")
            else:
                if count > 1:
                    multiplier = get_multiplier_prefix(count, acyloxy_name)
                    all_prefixes.append(f"{multiplier}({acyloxy_name})")
                else:
                    all_prefixes.append(f"({acyloxy_name})")

    # --- Generate FG prefixes from non-principal groups ---
    chain_set = set(principal_chain)
    ring_fg_groups = defaultdict(list)  # FGs on ring atoms, for ring substituent naming

    # Collect substituent branch atoms for FG-on-branch filtering (BUG-B).
    # FGs located entirely on a substituent branch are already named by the
    # substituent naming path (e.g., hydroxymethyl), so skip them here.
    _branch_atoms = set()
    if features.substituents:
        for _pos, sub_list in features.substituents.items():
            for sub_atoms in sub_list:
                _branch_atoms.update(sub_atoms)

    for fg_name, matches in non_principal.items():
        if not matches:
            continue

        # When chain is parent, separate FGs on ring from FGs on chain
        if getattr(features, 'chain_is_parent', False):
            ring_atom_set = set()
            for rg in getattr(features, 'ring_substituents_as_groups', []):
                ring_atom_set.update(rg)

            chain_matches = []
            for match in matches:
                # Check if this FG's center atom is on the chain
                center = _find_fg_center_atom(mol, match, fg_name)
                if center is not None and center in chain_set and center not in ring_atom_set:
                    chain_matches.append(match)
                elif any(a in chain_set and a not in ring_atom_set for a in match):
                    chain_matches.append(match)
                else:
                    # FG is on a ring - track for ring substituent naming
                    ring_fg_groups[fg_name].append(match)
            matches = chain_matches
            if not matches:
                continue

        # BUG-B: Skip simple FG matches located entirely on a *small* substituent
        # branch (<=3 carbons) that gets named as a compound substituent by
        # _name_heteroatom_substituent() (e.g., hydroxymethyl, aminomethyl).
        # Only applies to simple FGs: alcohol, amine, halogens.
        # Long branches or complex FGs are NOT handled by substituent naming.
        _BRANCH_HANDLED_FGS = {
            'primary_alcohol', 'secondary_alcohol', 'primary_amine',
            'fluoro', 'chloro', 'bromo', 'iodo',
        }
        if _branch_atoms and fg_name in _BRANCH_HANDLED_FGS:
            filtered_matches = []
            for match in matches:
                if not all(a in _branch_atoms for a in match):
                    filtered_matches.append(match)
                    continue
                # Check if this FG is on a small branch with carbons
                # (i.e., a branch where _name_heteroatom_substituent handles the FG)
                on_small_branch = False
                for _pos, sub_list in features.substituents.items():
                    for sub_atoms in sub_list:
                        sub_set = set(sub_atoms)
                        if all(a in sub_set for a in match):
                            # This branch contains the entire FG match
                            c_count = sum(1 for a in sub_atoms
                                          if mol.GetAtomWithIdx(a).GetSymbol() == 'C')
                            if 1 <= c_count <= 3:
                                on_small_branch = True
                                break
                    if on_small_branch:
                        break
                if not on_small_branch:
                    filtered_matches.append(match)
            matches = filtered_matches
            if not matches:
                continue

        # Get prefix form for this FG
        prefix_form = get_fg_prefix_form(
            fg_name, mol, matches[0], principal_chain
        )
        if not prefix_form:
            continue

        # Get locants for this FG
        locants = get_non_principal_fg_locants(
            mol, matches, principal_chain, atom_to_locant, fg_name
        )

        # Reconcile count with locants: if we found fewer locants than
        # matches, some FG instances are off-chain (inside substituent branches)
        # and should not inflate the multiplier. Use locant count as truth.
        count = len(matches)
        if locants:
            count = len(locants)

        # Format the prefix
        formatted = format_fg_prefix(prefix_form, locants, count)
        all_prefixes.append(formatted)

    # --- Generate ring substituent prefixes (when chain is parent) ---
    if getattr(features, 'chain_is_parent', False):
        from ..assembly.composer import _generate_ring_substituent_prefixes
        ring_prefixes = _generate_ring_substituent_prefixes(features)
        for ring_prefix in ring_prefixes:
            all_prefixes.append(ring_prefix.text)

    # --- Generate alkyl substituent prefixes ---
    if features.substituents:
        alkyl_prefixes = _generate_alkyl_prefixes_for_polyfunctional(
            features, skip_acyloxy=_esters_demoted
        )
        all_prefixes.extend(alkyl_prefixes)

    # Sort all prefixes alphabetically
    all_prefixes.sort(key=alpha_sort_key)

    # --- Build the name ---
    from ..data.chain_names import get_chain_prefix

    chain_length = len(principal_chain)
    stem = get_chain_prefix(chain_length)

    # Get suffix for principal group
    from .seniority import get_suffix
    from ..rules.locants import get_functional_group_locants

    suffix = get_suffix(principal_group, is_ring=False)
    if not suffix:
        return None

    # Get locants for principal group
    suffix_locants = []
    if features.principal_group_atoms:
        suffix_locants = get_functional_group_locants(
            principal_chain,
            features.principal_group_atoms,
            atom_to_locant,
            mol=mol
        )
        # Deduplicate locants (overlapping SMARTS can produce duplicates)
        suffix_locants = sorted(set(suffix_locants))

    # For terminal groups (acid, aldehyde), locant is implicit
    from ..assembly.composer import TERMINAL_GROUPS
    if principal_group in TERMINAL_GROUPS:
        suffix_locants = []

    # Determine multiplier for multiple principal groups
    # Validate: suffix count cannot exceed parent chain/ring capacity
    count = len(features.principal_group_atoms)
    max_capacity = chain_length
    if count > max_capacity:
        count = max_capacity
    # If we have unique locants, use those as the count (more reliable)
    if suffix_locants:
        count = len(suffix_locants)
    multiplier = get_multiplier_prefix(count, suffix) if count > 1 else ""

    # Build unsaturation infix
    from ..assembly.composer import _build_unsaturation_infix
    from ..rules.locants import get_bond_locants

    double_locants = get_bond_locants(
        principal_chain, features.double_bonds, atom_to_locant
    )
    triple_locants = get_bond_locants(
        principal_chain, features.triple_bonds, atom_to_locant
    )
    unsaturation = _build_unsaturation_infix(double_locants, triple_locants)

    # Detect suffix-prefix locant collisions (safety net for ring parents)
    if suffix_locants and all_prefixes:
        import re as _re
        from .locant_validation import detect_locant_collisions

        # Extract prefix locants from formatted prefix strings
        prefix_locant_groups = []
        for ptext in all_prefixes:
            match = _re.match(r'^([\d,]+)-', ptext)
            if match:
                try:
                    locs = [int(x) for x in match.group(1).split(',')]
                    prefix_locant_groups.append(locs)
                except ValueError:
                    pass

        is_ring = getattr(features, 'is_cyclic', False)
        if prefix_locant_groups:
            collisions = detect_locant_collisions(
                suffix_locants,
                prefix_locant_groups,
                parent_type="ring" if is_ring else "chain",
            )
            if collisions:
                collision_set = set(loc for _, loc in collisions)
                # Remove colliding prefix locants from affected prefix strings
                cleaned = []
                for ptext in all_prefixes:
                    match = _re.match(r'^([\d,]+)-(.+)$', ptext)
                    if match:
                        locs = [int(x) for x in match.group(1).split(',') if int(x) not in collision_set]
                        name_part = match.group(2)
                        if locs:
                            cleaned.append(f"{','.join(str(l) for l in locs)}-{name_part}")
                        else:
                            cleaned.append(name_part)
                    else:
                        cleaned.append(ptext)
                all_prefixes = cleaned

    # Assemble the name
    name = format_suffix_with_locants(
        stem, unsaturation, suffix, suffix_locants, multiplier
    )

    # Add prefixes with proper hyphenation at boundary
    if all_prefixes:
        prefix_str = _join_prefixes(all_prefixes)
        # Ensure hyphen between prefix ending with letter and name starting with digit
        if prefix_str and name and prefix_str[-1].isalpha() and name[0].isdigit():
            name = f"{prefix_str}-{name}"
        else:
            name = f"{prefix_str}{name}"

    # Add stereodescriptors if present
    if features.stereocenters or getattr(features, 'double_bond_stereo', None):
        from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        if descriptors:
            stereo_prefix = format_stereodescriptor_string(descriptors)
            name = f"{stereo_prefix}{name}"

    return name


def _generate_alkyl_prefixes_for_polyfunctional(
    features: Any, skip_acyloxy: bool = False
) -> List[str]:
    """
    Generate alkyl substituent prefixes for polyfunctional compounds.

    Similar to _generate_alkyl_prefixes in composer.py but returns
    just the formatted strings for combination with FG prefixes.

    Args:
        features: MolecularFeatures object
        skip_acyloxy: If True, skip acyloxy detection (esters already handled
                      by ester demotion in name_polyfunctional)
    """
    from collections import defaultdict

    mol = features.mol
    substituent_groups: Dict[str, List[int]] = defaultdict(list)

    # Collect ring atoms that should be skipped (handled by ring substituent prefixes)
    ring_atoms_to_skip: set = set()
    if getattr(features, 'chain_is_parent', False):
        ring_groups = getattr(features, 'ring_substituents_as_groups', [])
        for ring_atoms in ring_groups:
            ring_atoms_to_skip.update(ring_atoms)

    for position, sub_list in features.substituents.items():
        for sub_atoms in sub_list:
            # Skip substituents whose carbon atoms are entirely within ring atoms
            # (these are handled by _generate_ring_substituent_prefixes).
            # Substituents that extend beyond the ring (large branches containing
            # a ring) should NOT be skipped -- only the ring-only substituents.
            if ring_atoms_to_skip:
                sub_carbons = {
                    idx for idx in sub_atoms
                    if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                }
                if sub_carbons and sub_carbons <= ring_atoms_to_skip:
                    continue

            # Count only carbon atoms
            carbon_count = sum(
                1 for idx in sub_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )

            # Skip non-alkyl substituents
            if carbon_count == 0:
                continue

            # Check for heteroatoms
            has_heteroatom = any(
                mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
                for idx in sub_atoms
            )

            if has_heteroatom:
                from ..assembly.composer import (
                    _check_for_acylamino,
                    _check_for_acyloxy,
                    _name_heteroatom_substituent,
                )
                # NOTE: Do NOT check alkoxy here - ethers are already handled
                # by the FG prefix system in name_polyfunctional
                het_name = _check_for_acylamino(mol, sub_atoms, features.principal_chain)
                if not het_name and not skip_acyloxy:
                    het_name = _check_for_acyloxy(mol, sub_atoms, features.principal_chain)
                if not het_name:
                    het_name = _name_heteroatom_substituent(mol, sub_atoms, features.principal_chain)
                if not het_name:
                    # Enumerator fallback for simple non-ring, non-ether branches.
                    # Ethers (O-attached) are handled by the FG prefix system.
                    ring_info = mol.GetRingInfo()
                    sub_has_ring = any(ring_info.NumAtomRings(idx) > 0 for idx in sub_atoms)
                    # Find attachment atom
                    _attach = None
                    _chain_set_tmp = set(features.principal_chain) if features.principal_chain else set()
                    for _si in sub_atoms:
                        for _nb in mol.GetAtomWithIdx(_si).GetNeighbors():
                            if _nb.GetIdx() in _chain_set_tmp:
                                _attach = _si
                                break
                        if _attach is not None:
                            break
                    _attach_sym = mol.GetAtomWithIdx(_attach).GetSymbol() if _attach is not None else ''
                    if not sub_has_ring and _attach_sym != 'O':
                        from ..assembly.substituent_enumerator import (
                            SubstituentInfo,
                            classify_and_name_fragment,
                        )
                        from ..assembly.naming_utils import needs_brackets
                        chain_set = set(features.principal_chain) if features.principal_chain else set()
                        frag_info = SubstituentInfo(
                            frag_mol=None,
                            locant=position,
                            attach_mol_idx=features.principal_chain[position - 1] if features.principal_chain and position <= len(features.principal_chain) else sub_atoms[0],
                            frag_atoms=frozenset(sub_atoms),
                        )
                        het_name = classify_and_name_fragment(mol, frag_info, chain_set, features)
                        if het_name and needs_brackets(het_name):
                            het_name = f"({het_name})"
                if het_name:
                    substituent_groups[het_name].append(position)
                continue

            try:
                alkyl_name = get_alkyl_name(carbon_count)
                substituent_groups[alkyl_name].append(position)
            except ValueError:
                continue

    # Build formatted prefixes
    prefixes = []
    for name, locants in substituent_groups.items():
        count = len(locants)
        locant_str = ",".join(str(loc) for loc in sorted(locants))
        if count > 1:
            multiplier = get_multiplier_prefix(count, name)
            formatted = f"{locant_str}-{multiplier}{name}"
        else:
            formatted = f"{locant_str}-{name}"
        prefixes.append(formatted)

    return prefixes


def _join_prefixes(prefix_texts: List[str]) -> str:
    """
    Join multiple prefix strings with proper IUPAC hyphenation.

    When concatenating prefixes like "3-ethyl" and "2-hydroxy", the result
    should be "3-ethyl-2-hydroxy" (with hyphen between letter and digit).
    """
    if not prefix_texts:
        return ""

    if len(prefix_texts) == 1:
        return prefix_texts[0]

    result = prefix_texts[0]
    for i in range(1, len(prefix_texts)):
        current = prefix_texts[i]
        if result and current:
            last_char = result[-1]
            first_char = current[0]
            # Insert hyphen between letter/paren and digit
            # e.g., "amino" + "4-methyl" → "amino-4-methyl"
            # e.g., "(ethanoyl)amino" + "4-methyl" → "(ethanoyl)amino-4-methyl"
            if first_char.isdigit() and (last_char.isalpha() or last_char == ')'):
                result += "-"
            # Also between ')' and letter for clarity
            elif last_char == ')' and first_char.isalpha():
                result += "-"
        result += current

    return result
