"""
Polyfunctional compound naming coordinator.

Handles naming of compounds with multiple functional groups:
- Principal group (highest seniority) becomes the suffix
- Lower-seniority groups become prefixes with locants
- Ethers (no seniority) are always named as alkoxy prefixes

Based on IUPAC 2013 Blue Book P-41 to P-43.
"""

import logging
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Any, Set

logger = logging.getLogger(__name__)

from ..assembly.naming_utils import (
    alpha_sort_key,
    get_multiplier_prefix,
    format_suffix_with_locants,
    get_alkyl_name,
    ALKYL_NAMES,
    BRANCH_HANDLED_FGS,
)
from .seniority import (
    get_principal_group,
    get_prefix,
    SENIORITY_ORDER,
    PREFIX_FORMS,
)
# Phase 160.1 CONTEXT D-03: lift the 5 prefix-form generators + dispatcher to
# assembly/substituent_prefix_forms.py. The functions below remain as thin
# re-exports preserving the existing polyfunctional.py public API.
from ..assembly.substituent_prefix_forms import (
    get_substituent_prefix_form as _ASSEMBLY_get_substituent_prefix_form,
    get_alkoxycarbonyl_prefix as _ASSEMBLY_get_alkoxycarbonyl_prefix,
    get_alkoxy_prefix as _ASSEMBLY_get_alkoxy_prefix,
    get_sulfinyl_prefix as _ASSEMBLY_get_sulfinyl_prefix,
    get_sulfonyl_prefix as _ASSEMBLY_get_sulfonyl_prefix,
    get_sulfanyl_prefix as _ASSEMBLY_get_sulfanyl_prefix,
)


# Functional groups that can be detected but have no seniority
# These are always named as prefixes (never suffix)
NO_SENIORITY_GROUPS = {
    "ether",
    "vinyl_ether",
    "aromatic_ether",
    "fluoro",
    "chloro",
    "bromo",
    "iodo",
    "nitro",
    "nitroso",
    "azido",
    # Wave2 T2a (P-61.8): the substitutive 'isocyanato'/'isothiocyanato'
    # prefixes are the PINs (BB VERBATIM 'isocyanatocyclohexane (PIN)
    # cyclohexyl isocyanate'), so with no senior group present the parent
    # hydride is named with the prefix (isocyanatoethane) — the functional-
    # class handlers now decline under style='pin' and keep the 'ethyl
    # isocyanate' form for general (--trivial) nomenclature.
    "isocyanate",
    "isothiocyanate",
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


# PERC-06 / D-03: Normalize FG subtypes to parent class for polyfunctional counting.
# IUPAC treats primary/secondary/tertiary alcohol as the same FG class.
PARENT_CLASS_MAP = {
    "primary_alcohol": "alcohol",
    "secondary_alcohol": "alcohol",
    "tertiary_alcohol": "alcohol",
    "phenol": "alcohol",
    "enol": "alcohol",
    "alcohol": "alcohol",
    "primary_amine": "amine",
    "secondary_amine": "amine",
    "tertiary_amine": "amine",
    "aromatic_amine": "amine",
}


def _drop_union_class_subtypes(
    non_principal: Dict[str, List[tuple]], principal_group: Optional[str]
) -> Dict[str, List[tuple]]:
    """Remove non-principal FGs already absorbed into the principal multiplied suffix.

    DD5 RC-4 / SEN-03 parity (audit fix 2026-06-22): ``get_principal_group`` unions
    every present subtype of the principal group's equal-seniority *union* class
    (alcohols, ``_RC4_UNION_CLASSES``) into ``principal_group_atoms``, so a mixed
    primary+secondary OH chain is expressed as ONE multiplied suffix
    (``…-1,3-diol``). Such same-class subtypes therefore must NOT also leak into
    the non-principal prefix loop — doing so double-listed the secondary OH as both
    a ``hydroxy`` prefix and a ``-ol`` suffix locant, producing OPSIN-invalid names
    like ``2-amino-3-hydroxyoctadecane-1,3-diol`` (sphinganine) / geminal forms.

    This mirrors the identical skip in
    ``assembly/handlers/_handler_shared._generate_prefixes`` (the general-acyclic
    path); ``polyfunctional`` is the parallel multi-FG handler and needs the same
    guard. Scoped to ``_RC4_UNION_CLASSES`` (alcohols only) via the shared
    ``_SENIORITY_PARENT`` map, so amine/other subtypes are untouched.
    """
    if not principal_group:
        return non_principal
    from .seniority import _SENIORITY_PARENT, _RC4_UNION_CLASSES
    pclass = _SENIORITY_PARENT.get(principal_group, principal_group)
    if pclass not in _RC4_UNION_CLASSES:
        return non_principal
    return {
        k: v for k, v in non_principal.items()
        if _SENIORITY_PARENT.get(k, k) != pclass
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

        # Normalize to parent class before counting (PERC-06 / D-03)
        parent = PARENT_CLASS_MAP.get(fg_name, fg_name)
        distinct_groups.add(parent)

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

    Phase 160.1 CONTEXT D-03: chemistry rules lifted to
    assembly/substituent_prefix_forms.py. This shim consults the new
    dispatcher first; on None (out-of-14-row-set), falls back to the
    static PREFIX_FORMS lookup via get_prefix() — preserving the original
    polyfunctional caller's expectation that any FG in PREFIX_FORMS
    (e.g., hydroxyl, amino, carboxy) gets its static prefix form returned.

    Args:
        fg_name: Name of the functional group
        mol: RDKit Mol object
        atoms: Atom indices matching this FG instance
        principal_chain: Atom indices of the principal chain

    Returns:
        Prefix string (e.g., "hydroxy", "oxo", "methoxy") or None if no prefix
    """
    # Phase 160.1: consult the 14-row dispatcher first (P-65 / P-66 forms).
    result = _ASSEMBLY_get_substituent_prefix_form(fg_name, mol, atoms, principal_chain)
    if result is not None:
        return result

    # Fall through to the static PREFIX_FORMS table for FGs not in the
    # 14-row Phase 160.1 closed-set (e.g., carboxylic_acid → "carboxy",
    # primary_alcohol → "hydroxy", primary_amine → "amino").
    return get_prefix(fg_name)


def _get_alkoxy_prefix(
    mol,
    ether_atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """Thin re-export — see assembly/substituent_prefix_forms.get_alkoxy_prefix.

    Phase 160.1 CONTEXT D-03: chemistry rule lifted to
    assembly/substituent_prefix_forms.py. This shim preserves the
    polyfunctional.py public API for existing callers.
    """
    return _ASSEMBLY_get_alkoxy_prefix(mol, ether_atoms, principal_chain)


def _get_alkoxycarbonyl_prefix(
    mol,
    ester_atoms: tuple,
    principal_chain: List[int],
) -> Optional[str]:
    """Thin re-export — see assembly/substituent_prefix_forms.get_alkoxycarbonyl_prefix.

    Phase 160.1 CONTEXT D-03: chemistry rule lifted to
    assembly/substituent_prefix_forms.py. This shim preserves the
    polyfunctional.py public API for existing callers.
    """
    return _ASSEMBLY_get_alkoxycarbonyl_prefix(mol, ester_atoms, principal_chain)


def _get_sulfinyl_prefix(
    mol,
    sulfoxide_atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """Thin re-export — see assembly/substituent_prefix_forms.get_sulfinyl_prefix.

    Phase 160.1 CONTEXT D-03: chemistry rule lifted to
    assembly/substituent_prefix_forms.py. This shim preserves the
    polyfunctional.py public API for existing callers.
    """
    return _ASSEMBLY_get_sulfinyl_prefix(mol, sulfoxide_atoms, principal_chain)


def _get_sulfonyl_prefix(
    mol,
    sulfone_atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """Thin re-export — see assembly/substituent_prefix_forms.get_sulfonyl_prefix.

    Phase 160.1 CONTEXT D-03: chemistry rule lifted to
    assembly/substituent_prefix_forms.py. This shim preserves the
    polyfunctional.py public API for existing callers.
    """
    return _ASSEMBLY_get_sulfonyl_prefix(mol, sulfone_atoms, principal_chain)


def _get_sulfanyl_prefix(
    mol,
    thioether_atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """Thin re-export — see assembly/substituent_prefix_forms.get_sulfanyl_prefix.

    Phase 160.1 CONTEXT D-03: chemistry rule lifted to
    assembly/substituent_prefix_forms.py. This shim preserves the
    polyfunctional.py public API for existing callers.
    """
    return _ASSEMBLY_get_sulfanyl_prefix(mol, thioether_atoms, principal_chain)


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

    Per IUPAC P-16.3.3, compound substituent prefixes (e.g., methylsulfinyl,
    methylsulfonyl) are enclosed in parentheses when used with locants.
    Simple prefixes (hydroxy, oxo, amino) are not parenthesized.

    Args:
        prefix_form: Base prefix name (e.g., "hydroxy", "oxo", "methoxy")
        locants: List of locant positions
        count: Number of instances

    Returns:
        Formatted prefix string (e.g., "2-hydroxy", "3-oxo", "2-(methylsulfinyl)")
    """
    from ..assembly.naming_utils import needs_brackets

    if not locants:
        # No locants - just return prefix with multiplier if needed
        if count > 1:
            multiplier = get_multiplier_prefix(count, prefix_form)
            return f"{multiplier}{prefix_form}"
        return prefix_form

    # Format locants
    locant_str = ",".join(str(loc) for loc in sorted(locants))

    # Check if prefix is a compound substituent needing parentheses (P-16.3.3)
    compound = needs_brackets(prefix_form)

    # Get multiplier if multiple instances
    if count > 1:
        multiplier = get_multiplier_prefix(count, prefix_form)
        if compound:
            return f"{locant_str}-{multiplier}({prefix_form})"
        return f"{locant_str}-{multiplier}{prefix_form}"

    if compound:
        return f"{locant_str}-({prefix_form})"
    return f"{locant_str}-{prefix_form}"


# WS-A task 9 / P-66.6.1.2: FGs whose PREFIX_FORMS string ('oxo',
# 'hydroxy') decorates a SKELETAL atom — valid only when the FG center
# carbon is part of the parent ring. Carbon-including prefixes
# (carboxylic_acid->carboxy, nitrile->cyano, amide->carbamoyl) are correct
# exocyclic by design and MUST NOT be listed here.
_SKELETON_DECORATION_FGS = {
    'aldehyde',
    'ketone', 'thioketone', 'selenoketone', 'telluroketone',
    'alcohol', 'primary_alcohol', 'secondary_alcohol', 'tertiary_alcohol',
}


def _exocyclic_component(mol, start: int, ring_set: set) -> set:
    """Connected exocyclic component containing ``start`` (BFS that never
    enters the parent ring)."""
    visited = {start}
    queue = [start]
    while queue:
        cur = queue.pop()
        for nbr in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = nbr.GetIdx()
            if ni not in visited and ni not in ring_set:
                visited.add(ni)
                queue.append(ni)
    return visited


def _name_exocyclic_decoration(
    mol, center: int, ring_set: set, atom_to_locant: dict, fg_name: str
):
    """Name an exocyclic decoration-FG branch with its carbon-including
    substituent prefix (P-66.6.1.2; Phase-172/MBA-02 table: formyl /
    n-oxoalkyl / hydroxyalkyl).

    Returns (formatted_prefix_or_None, component_atoms). A None prefix means
    the branch could not be faithfully named — the caller must leave its
    atoms to the universal pipeline, never emit a skeleton-decoration
    prefix for it.
    """
    comp = _exocyclic_component(mol, center, ring_set)
    ring_attach_loc = None
    for a in comp:
        for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
            ni = nbr.GetIdx()
            if ni in ring_set and ni in atom_to_locant:
                ring_attach_loc = atom_to_locant[ni]
                break
        if ring_attach_loc is not None:
            break
    if ring_attach_loc is None:
        return None, comp
    # Guard: the branch must be exactly chain-C + oxygen(s) — anything else
    # (rings, N, S decorations) is beyond the table's vocabulary.
    for a in comp:
        sym = mol.GetAtomWithIdx(a).GetSymbol()
        if sym not in ('C', 'O', 'H'):
            return None, comp
        if mol.GetAtomWithIdx(a).IsInRing():
            return None, comp
    carbon_count = sum(
        1 for a in comp if mol.GetAtomWithIdx(a).GetSymbol() == 'C'
    )
    if fg_name == 'aldehyde':
        table_key = 'aldehyde'
    elif fg_name.endswith('alcohol'):
        table_key = 'alcohol'
    else:
        return None, comp  # ketone family: no faithful table row yet
    from .benzene import _name_functionalized_chain_substituent
    sub_name = _name_functionalized_chain_substituent(carbon_count, table_key)
    if not sub_name:
        return None, comp
    return format_fg_prefix(sub_name, [ring_attach_loc], 1), comp


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


def _name_ring_as_parent_polyfunctional(features: Any) -> Optional[str]:
    """Name a polyfunctional compound where the ring is the parent structure.

    Handles the case where ``principal_chain`` is None because the molecule
    uses a ring as its parent hydride. Builds the name from:
    1. Ring parent name (from oriented ring or ring detection)
    2. Principal group suffix with locants on the ring
    3. Non-principal group prefixes with ring locants
    4. Substituent prefixes via the universal pipeline

    IUPAC 2013 P-31 through P-43.

    Args:
        features: MolecularFeatures object (must have is_cyclic=True,
            principal_group set).

    Returns:
        Complete IUPAC name, or None if ring parent cannot be determined.
    """
    mol = features.mol
    principal_group = features.principal_group
    non_principal = getattr(features, 'non_principal_groups', {})
    # audit fix 2026-06-22: drop same-union-class subtypes already in the
    # multiplied suffix (parity with the chain path; ring polyols too).
    non_principal = _drop_union_class_subtypes(non_principal, principal_group)

    if not principal_group:
        return None

    # --- Step 1: Identify ring parent ---
    ring_info = mol.GetRingInfo()
    all_rings = ring_info.AtomRings()
    if not all_rings:
        return None

    # Use oriented_ring if available, else find the largest ring
    oriented_ring = getattr(features, 'oriented_ring', None)
    if oriented_ring:
        ring_atoms = list(oriented_ring)
    else:
        # Pick the largest ring as parent
        ring_atoms = list(max(all_rings, key=len))

    ring_set = set(ring_atoms)
    ring_size = len(ring_atoms)

    # --- Step 2: Build ring parent name ---
    ring_parent_name = _get_ring_parent_name(mol, ring_atoms)
    if not ring_parent_name:
        return None

    # --- Step 3: Build locant mapping for ring atoms ---
    # Use features.atom_to_locant if available, else build from ring order
    atom_to_locant = features.atom_to_locant
    if not atom_to_locant:
        atom_to_locant = {}
        for i, idx in enumerate(ring_atoms):
            atom_to_locant[idx] = i + 1

    # --- Step 4: Generate principal group suffix with locants ---
    from .seniority import get_suffix
    suffix = get_suffix(principal_group, is_ring=True)
    if not suffix:
        # Try chain suffix as fallback
        suffix = get_suffix(principal_group, is_ring=False)
    if not suffix:
        return None

    suffix_locants = []
    if features.principal_group_atoms:
        for match in features.principal_group_atoms:
            # Wave2 T2c (P-64.7.1): the suffix locant must come from the FG
            # CENTER atom, not the first match atom that happens to be in the
            # map. The ketone SMARTS match is (C_neighbor, C=O, O, C_neighbor)
            # — taking match[0] read the NEIGHBOR's locant, producing the
            # impossible '2-aminocyclohexan-2-one' for NC1CCCCC1=O (amino and
            # oxo cannot share one carbon; PIN is 2-aminocyclohexan-1-one).
            _center = _find_fg_center_atom(mol, match, principal_group)
            if _center is not None and _center in atom_to_locant:
                suffix_locants.append(atom_to_locant[_center])
                continue
            for atom_idx in match:
                if atom_idx in atom_to_locant:
                    suffix_locants.append(atom_to_locant[atom_idx])
                    break
                # For FGs attached TO the ring (e.g., -COOH on cyclohexane),
                # the center atom may not be in the ring; check neighbors
                atom = mol.GetAtomWithIdx(atom_idx)
                for nbr in atom.GetNeighbors():
                    nidx = nbr.GetIdx()
                    if nidx in atom_to_locant and nidx in ring_set:
                        suffix_locants.append(atom_to_locant[nidx])
                        break
    suffix_locants = sorted(set(suffix_locants))

    # Determine multiplier for multiple principal groups
    count = len(suffix_locants) if suffix_locants else 1
    multiplier = get_multiplier_prefix(count, suffix) if count > 1 else ""

    # --- Step 5: Generate non-principal FG prefixes with ring locants ---
    all_prefixes = []
    # WS-A task 9 consumption plumbing: atoms named by an exocyclic
    # carbon-including prefix (formyl/n-oxoalkyl/...) must be consumed even
    # when outside the SMARTS match (the CH2 of -CH2CHO); atoms of a branch
    # we could NOT faithfully name must NOT be consumed (the universal
    # pipeline names them instead of them silently vanishing).
    _extra_consumed: set = set()
    _do_not_consume: set = set()
    for fg_name, matches in non_principal.items():
        if not matches:
            continue
        if fg_name in NO_SENIORITY_GROUPS and fg_name not in ('alkene', 'alkyne'):
            # Named as prefix if it has a prefix form
            prefix_form = get_fg_prefix_form(fg_name, mol, matches[0], None)
            if prefix_form:
                fg_locants = []
                for match in matches:
                    for atom_idx in match:
                        if atom_idx in atom_to_locant:
                            fg_locants.append(atom_to_locant[atom_idx])
                            break
                        atom = mol.GetAtomWithIdx(atom_idx)
                        for nbr in atom.GetNeighbors():
                            nidx = nbr.GetIdx()
                            if nidx in atom_to_locant and nidx in ring_set:
                                fg_locants.append(atom_to_locant[nidx])
                                break
                fg_locants = sorted(set(fg_locants))
                fg_count = len(fg_locants) if fg_locants else len(matches)
                formatted = format_fg_prefix(prefix_form, fg_locants, fg_count)
                all_prefixes.append(formatted)
            continue

        # Seniority-bearing non-principal groups
        prefix_form = get_prefix(fg_name)
        if prefix_form:
            fg_locants = []
            _diverted = 0
            for match in matches:
                # WS-A task 9 / P-66.6.1.2: skeleton-decoration prefixes
                # ('oxo', 'hydroxy') are valid ONLY when the FG center carbon
                # is IN the ring skeleton. The old context-blind path rewrote
                # an exocyclic -CHO as a ring ketone '4-oxo...' — a
                # structurally DIFFERENT molecule — by re-anchoring the
                # locant on the ring neighbor. An exocyclic center must emit
                # the carbon-including substituent prefix instead (formyl /
                # n-oxoalkyl / hydroxymethyl, the Phase-172/MBA-02 table).
                if fg_name in _SKELETON_DECORATION_FGS:
                    center = _find_fg_center_atom(mol, match, fg_name)
                    if center is not None:
                        if center in atom_to_locant:
                            # Locant from the CENTER atom (also fixes ring
                            # ketones citing the flanking match atom).
                            fg_locants.append(atom_to_locant[center])
                            continue
                        if center not in ring_set:
                            _diverted += 1
                            exo_prefix, exo_atoms = _name_exocyclic_decoration(
                                mol, center, ring_set, atom_to_locant, fg_name
                            )
                            if exo_prefix is not None:
                                all_prefixes.append(exo_prefix)
                                _extra_consumed.update(exo_atoms)
                            else:
                                # Cannot produce a faithful name — leave the
                                # branch to the universal pipeline rather
                                # than emit a structurally wrong prefix.
                                _do_not_consume.update(exo_atoms)
                            continue
                for atom_idx in match:
                    if atom_idx in atom_to_locant:
                        fg_locants.append(atom_to_locant[atom_idx])
                        break
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for nbr in atom.GetNeighbors():
                        nidx = nbr.GetIdx()
                        if nidx in atom_to_locant and nidx in ring_set:
                            fg_locants.append(atom_to_locant[nidx])
                            break
            fg_locants = sorted(set(fg_locants))
            _remaining = len(matches) - _diverted
            if fg_locants or _remaining > 0:
                fg_count = len(fg_locants) if fg_locants else _remaining
                formatted = format_fg_prefix(prefix_form, fg_locants, fg_count)
                all_prefixes.append(formatted)

    # --- Step 6: Discover substituents on the ring via universal pipeline ---
    # Collect atoms consumed by the principal group and non-principal FGs
    consumed_atoms = set()
    if features.principal_group_atoms:
        for match in features.principal_group_atoms:
            for idx in match:
                if idx not in ring_set:
                    consumed_atoms.add(idx)
    for fg_name, matches in non_principal.items():
        for match in matches:
            for idx in match:
                if idx not in ring_set:
                    consumed_atoms.add(idx)
    # WS-A task 9: see Step-5 plumbing comment.
    consumed_atoms |= _extra_consumed
    consumed_atoms -= _do_not_consume

    try:
        from ..assembly.composer import _integrate_universal_prefixes
        sub_prefix_str = _integrate_universal_prefixes(
            mol, ring_set,
            parent_type="ring",
            oriented_ring=ring_atoms,
            atom_to_locant=atom_to_locant,
            exclude_atoms=consumed_atoms,
        )
        if sub_prefix_str:
            all_prefixes.append(sub_prefix_str)
    except Exception as exc:
        logger.debug("Ring-as-parent universal prefix failed: %s", exc)

    # Sort prefixes alphabetically
    all_prefixes.sort(key=alpha_sort_key)

    # --- Step 7: Assemble the complete name ---
    # Format: [prefixes]-[ring_parent]-[suffix_locants]-[multiplier][suffix]
    # e.g., "4-hydroxy-cyclohexan-1-one"
    formatted_suffix = format_suffix_with_locants(
        ring_parent_name, "", suffix, suffix_locants, multiplier
    )

    if all_prefixes:
        prefix_str = _join_prefixes(all_prefixes)
        if prefix_str and formatted_suffix and prefix_str[-1].isalpha() and formatted_suffix[0].isdigit():
            name = f"{prefix_str}-{formatted_suffix}"
        else:
            name = f"{prefix_str}{formatted_suffix}"
    else:
        name = formatted_suffix

    # Add stereodescriptors if present
    if features.stereocenters or getattr(features, 'double_bond_stereo', None):
        from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        if descriptors:
            stereo_prefix = format_stereodescriptor_string(descriptors)
            name = f"{stereo_prefix}{name}"

    return name


def _get_ring_parent_name(mol, ring_atoms: list) -> Optional[str]:
    """Get the parent hydride name for a ring.

    Checks retained names first (benzene, pyridine, etc.), then falls
    back to systematic cyclo- naming.

    Args:
        mol: RDKit Mol object.
        ring_atoms: List of atom indices forming the ring.

    Returns:
        Ring parent stem (e.g., ``"cyclohexan"``, ``"benzene"``),
        or None if not determinable.
    """
    ring_set = set(ring_atoms)
    ring_size = len(ring_atoms)

    # Check if ring is all-carbon aromatic (benzene)
    all_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_atoms)
    all_carbon = all(mol.GetAtomWithIdx(idx).GetSymbol() == 'C' for idx in ring_atoms)

    if ring_size == 6 and all_aromatic and all_carbon:
        return "benzene"

    # Check for heterocyclic retained names
    has_heteroatom = any(
        mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
        for idx in ring_atoms
    )
    if has_heteroatom:
        try:
            from ..rules.heterocycles import classify_heterocycle
            het_info = classify_heterocycle(mol, tuple(ring_atoms))
            if het_info and het_info.get('name'):
                return het_info['name']
        except Exception:
            pass

    # Systematic cyclo- naming for carbocyclic rings
    if all_carbon:
        from ..data.chain_names import get_chain_prefix
        stem = get_chain_prefix(ring_size)
        if stem:
            # Return stem without "ane" suffix -- format_suffix_with_locants
            # will add the appropriate suffix. For cycloalkanes with
            # functional groups, we need "cyclohexan" not "cyclohexane".
            return f"cyclo{stem}an"

    return None


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
    # audit fix 2026-06-22: drop same-union-class subtypes already in the
    # multiplied suffix (prevents the 'n-hydroxy…-1,n-diol' double-listing).
    non_principal = _drop_union_class_subtypes(non_principal, principal_group)

    if not principal_chain or not atom_to_locant:
        # --- Phase 86-02: Ring-as-parent polyfunctional path ---
        # When principal_chain is None but the molecule is cyclic, attempt
        # ring-as-parent naming. Conservative guards prevent intercepting
        # compounds better handled by specialized fallthrough handlers:
        # - Only monocyclic (1 ring) -- fused/polycyclic have specialized handlers
        # - Only fully saturated rings -- unsaturated rings need -ene/-yne handling
        # - No aromatic rings -- benzene/pyridine/etc. have retained name handlers
        if getattr(features, 'is_cyclic', False) and not getattr(features, 'chain_is_parent', False) and principal_group:
            ring_info = mol.GetRingInfo()
            n_rings = ring_info.NumRings()
            if n_rings == 1:
                ring_atoms = list(ring_info.AtomRings()[0])
                ring_set = set(ring_atoms)
                ring_size = len(ring_atoms)
                total_heavy = mol.GetNumHeavyAtoms()

                # Guard: ring must be a significant portion of the molecule.
                # If the ring is < 40% of heavy atoms, it's likely a
                # substituent on a chain parent, not the parent itself.
                # E.g., sphingolipid with cyclohexane ring but 50+ chain atoms.
                if total_heavy > 0 and ring_size / total_heavy < 0.35:
                    pass  # Fall through to return None
                else:
                    # Check ring is fully saturated and non-aromatic
                    has_ring_double = False
                    is_aromatic = False
                    for idx in ring_atoms:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetIsAromatic():
                            is_aromatic = True
                            break
                        for bond in atom.GetBonds():
                            other = bond.GetOtherAtomIdx(idx)
                            if (other in ring_set
                                    and bond.GetBondTypeAsDouble() == 2.0):
                                has_ring_double = True
                                break
                        if has_ring_double:
                            break

                    if not is_aromatic and not has_ring_double:
                        ring_name = _name_ring_as_parent_polyfunctional(
                            features,
                        )
                        if ring_name:
                            return ring_name
        return None

    # --- Wave2 T5b (P-66.1.1.3): OFF-CHAIN acyl secondary/tertiary amide ---
    # When the PCG amide's acyl carbon is NOT on the principal chain (the
    # chain is the N-side), the chain-suffix machinery below is structurally
    # WRONG: it double-expresses the amide (acetamido prefix + a phantom
    # '...anamide' suffix whose carbonyl C isn't in the chain) — production
    # SELF-01 suppressed these to unknown. The PIN keeps the amide as PCG
    # with the ACYL side as parent and the whole N-side as a located
    # N-substituent prefix carrying its own junior FGs:
    # CC(CO)NC(C)=O -> N-(1-hydroxypropan-2-yl)acetamide (amide senior to
    # alcohol, P-41). Delegate to rules.amides.name_amide (whose N-substituent
    # namer now handles simple internal substituents) under fail-closed
    # guards; decline to None (jar-independent) when they don't hold.
    if principal_group in ('secondary_amide', 'tertiary_amide'):
        _amide_matches = features.functional_groups.get(principal_group, [])
        if len(_amide_matches) == 1:
            _match = list(_amide_matches[0])
            _acyl_c = next(
                (i for i in _match
                 if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                 and any(
                     nb.GetSymbol() == 'O'
                     and mol.GetBondBetweenAtoms(i, nb.GetIdx())
                            .GetBondTypeAsDouble() == 2.0
                     for nb in mol.GetAtomWithIdx(i).GetNeighbors())),
                None,
            )
            if _acyl_c is not None and _acyl_c not in set(principal_chain):
                from .amides import (
                    name_amide as _rules_name_amide, get_n_substituents,
                )
                # Fail closed on any DEFINED stereocentre/stereobond: the
                # delegated path emits no parent-level descriptors (the T3d
                # oxime-decline precedent).
                from rdkit import Chem as _Chem
                _has_stereo = bool(
                    _Chem.FindMolChiralCenters(mol, includeUnassigned=False)
                ) or any(
                    b.GetStereo() != _Chem.BondStereo.STEREONONE
                    for b in mol.GetBonds()
                )
                _n_subs = get_n_substituents(mol, tuple(_match))
                _n_atom = next(
                    (i for i in _match
                     if mol.GetAtomWithIdx(i).GetSymbol() == 'N'), None)
                # Coverage: every heavy N-neighbour besides the acyl C must
                # have produced a named substituent (get_n_substituents
                # silently omits unnameable ones), and every junior FG must
                # live wholly inside those N-substituent fragments.
                _claimed = set()
                for _s in _n_subs:
                    _claimed |= set(_s.get('atoms') or [])
                _expected_n_branches = 0
                if _n_atom is not None:
                    _expected_n_branches = sum(
                        1 for nb in mol.GetAtomWithIdx(_n_atom).GetNeighbors()
                        if nb.GetAtomicNum() > 1 and nb.GetIdx() != _acyl_c
                    )
                _fgs_contained = all(
                    set(_fm).issubset(_claimed)
                    for _fg_matches in (non_principal or {}).values()
                    for _fm in _fg_matches
                )
                if (not _has_stereo and _claimed
                        and len(_n_subs) == _expected_n_branches
                        and _fgs_contained):
                    _nm = _rules_name_amide(mol, tuple(_match))
                    if _nm:
                        return _nm
                return None

    # --- Ester is the most-senior group (P-41: esters outrank acyl halides,
    # amides, nitriles, aldehydes, ketones, alcohols) ---
    # When get_principal_group selected the ester, the ester IS the most senior
    # group present, so it STAYS the principal characteristic group (suffix
    # '-oate') and the junior groups become prefixes (oxo/halo/cyano/hydroxy).
    # Name it as "alkyl <acid-with-prefixes>oate" via the acid-analog builder.
    # (The legacy EL-02 fallback below DEMOTED the ester and promoted a LESS
    # senior group to PCG -- a different, wrong molecule; kept only as a
    # fail-closed fallback for cases the acid-analog path declines.)
    if principal_group == "ester":
        _ester_matches_all = features.functional_groups.get("ester", [])
        if len(_ester_matches_all) == 1:
            from ..rules.esters import name_polyfunctional_ester_via_acid
            _ester_pin = name_polyfunctional_ester_via_acid(
                mol, _ester_matches_all[0]
            )
            if _ester_pin:
                return _ester_pin

    # --- EL-02 (fallback): Ester demotion in polyfunctional context ---
    # Legacy path, reached only when the acid-analog naming above declined
    # (e.g. >1 ester, or a junior group on the removed alkyl side).
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

        # BUG-B guard: Skip FG matches located entirely on a small substituent
        # branch (<=3 carbons) when the substituent naming path demonstrably
        # handles them (producing e.g. "hydroxymethyl", "chloromethyl").
        # Uses shared BRANCH_HANDLED_FGS from naming_utils (unified in Phase 113).
        if _branch_atoms and fg_name in BRANCH_HANDLED_FGS:
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
            original_count = len(non_principal[fg_name])
            matches = filtered_matches
            if not matches:
                if original_count > 0:
                    logger.debug(
                        "DROP-17 polyfunc_bugb: fg_name=%s filtered=%d "
                        "(substituent naming handles these on small branches)",
                        fg_name, original_count,
                    )
                continue

        # HYG-04 site#2 double-count guard: a secondary/tertiary amine whose
        # nitrogen sits in a substituent branch is named IN FULL by the
        # substituent-naming path as a single "(...amino)" prefix
        # (methylamino / dimethylamino / ...). Emitting a bare "amino" FG-prefix
        # for the same nitrogen here double-counts it, producing geminal-diamine
        # artifacts like "2-amino-2-(dimethylamino)ethan-1-ol". Drop such matches
        # so the substituent walk owns the naming. Primary -NH2 is unaffected:
        # its branch is carbon-free, the substituent walk skips it, and this
        # FG-prefix remains its sole, correct name. Mirrors the BUG-B guard above
        # but keyed on the nitrogen (the amine match also spans the chain carbon,
        # so an "entirely on branch" test would miss it).
        _N_SUBSTITUTED_AMINE_FGS = {'secondary_amine', 'tertiary_amine'}
        if fg_name in _N_SUBSTITUTED_AMINE_FGS and features.substituents:
            _amine_kept = []
            for _match in matches:
                _n_owned_by_branch = False
                for _n in (a for a in _match
                           if mol.GetAtomWithIdx(a).GetSymbol() == 'N'):
                    for _sub_list in features.substituents.values():
                        if any(_n in _sub_atoms for _sub_atoms in _sub_list):
                            _n_owned_by_branch = True
                            break
                    if _n_owned_by_branch:
                        break
                if not _n_owned_by_branch:
                    _amine_kept.append(_match)
            if not _amine_kept:
                logger.debug(
                    "DROP-HYG04 polyfunc: skip bare 'amino' FG-prefix for %s — "
                    "N-substituted amine named in full by the substituent walk",
                    fg_name,
                )
                continue
            matches = _amine_kept

        # Non-principal acyl halide on a CHAIN parent (P-65.5.4): the
        # acyl-halide carbon is a chain member, expressed as 'oxo' (=O) +
        # 'halo' (X), NOT the 'carbonochloridoyl' prefix.  Blue Book worked
        # examples: "4-chloro-4-oxobutanoic acid" (PIN, line 5108),
        # "3-chloro-3-oxopropanoic acid" (PIN, line 31531).  The
        # 'carbonochloridoyl' prefix is the PIN only on a RING parent (the
        # carbon cannot join the ring, e.g. "2-carbonochloridoylbenzoic acid",
        # line 31533) — handled elsewhere; here we fall through to it only when
        # the acyl-halide carbon is NOT a member of the principal chain.
        _ACYL_HALIDE_HALO = {
            'acid_fluoride': 'fluoro', 'acid_chloride': 'chloro',
            'acid_bromide': 'bromo', 'acid_iodide': 'iodo',
        }
        if fg_name in _ACYL_HALIDE_HALO and chain_set:
            _ah_locs = []
            for _m in matches:
                _c = _m[0]  # acyl-halide carbonyl carbon (SMARTS index 0)
                if _c in chain_set:
                    _loc = atom_to_locant.get(_c)
                    if _loc is not None:
                        _ah_locs.append(_loc)
            if _ah_locs and len(_ah_locs) == len(matches):
                _ah_locs.sort()
                _n = len(_ah_locs)
                all_prefixes.append(format_fg_prefix('oxo', _ah_locs, _n))
                all_prefixes.append(
                    format_fg_prefix(_ACYL_HALIDE_HALO[fg_name], _ah_locs, _n)
                )
                continue

        # Get prefix form for this FG
        prefix_form = get_fg_prefix_form(
            fg_name, mol, matches[0], principal_chain
        )
        if not prefix_form:
            # Phase 169 D-01 tier-3 fallback (POLY-01): when a composite loser FG
            # has no clean strict-IUPAC prefix (the DROP-23 case), decompose it into
            # its ordered sub-group prefix components instead of dropping it. Gated
            # behind the default-OFF flag (Stage A byte-identical); the split only
            # fires for table-listed composites (ester/thioester) and is OPSIN-RT
            # gated (FAIL-CLOSED, D-05). Lazy import (Pattern-S3) avoids a cycle.
            components = None
            if getattr(features, "_enable_group_splitting", False):
                from ..assembly.group_splitting import split_composite_fg
                components = split_composite_fg(
                    fg_name, mol, matches[0], principal_chain,
                    oracle=getattr(features, "_split_oracle", None),
                )
            if not components:
                # By-design: FGs using functional class naming (ester→alkoxycarbonyl,
                # secondary_amide→acylamino, thioether, etc.) are handled by
                # specialized naming paths, not as simple prefixes. Unchanged drop
                # path for the OFF / non-splittable / RT-rejected cases.
                logger.debug(
                    "DROP-23 substituent_skip: reason=no_fg_prefix_form fg_name=%s",
                    fg_name,
                )
                continue
            # POLY-02 native: each split component re-enters the existing prefix
            # pipeline (format_fg_prefix + alpha_sort_key) sharing the central-carbon
            # locant (comp.locants is None -> the caller's locants apply to both).
            split_locants = get_non_principal_fg_locants(
                mol, matches, principal_chain, atom_to_locant, fg_name
            )
            for comp in components:
                all_prefixes.append(
                    format_fg_prefix(comp.prefix_form, comp.locants or split_locants, comp.count)
                )
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

    # --- HYG-04 site#2: N-substituent prefixes for a PRINCIPAL amine ---
    # When the amine is the principal characteristic group (suffix -amine), its
    # N-substituents are cited as N-/N,N- prefixes (e.g. N,N-dimethylethan-1-amine),
    # mirroring composer._assemble_amine_name. Without this, a tertiary amine
    # (e.g. diphenhydramine core) leaks its N-alkyls into a "(dimethylamino)"
    # substituent that double-counts the principal nitrogen. The principal-amine
    # branch itself is excluded from the substituent walk (see
    # _generate_alkyl_prefixes_for_polyfunctional). These prefixes join the same
    # alphanumerical sort below, so they interleave correctly with C-substituents.
    _AMINE_PRINCIPAL_FGS = {
        'primary_amine', 'secondary_amine', 'tertiary_amine', 'aromatic_amine',
    }
    _principal_amine_n_atoms = set()
    if principal_group in _AMINE_PRINCIPAL_FGS:
        for _pmatch in getattr(features, 'principal_group_atoms', None) or []:
            for _pa in _pmatch:
                if mol.GetAtomWithIdx(_pa).GetSymbol() == 'N':
                    _principal_amine_n_atoms.add(_pa)
    # Scope: exactly ONE principal amine nitrogen. Di-/polyamines need primed
    # N-locants (N,N,N',N'-tetramethyl...) keyed to each amine position — a
    # distinct feature; applying single-N logic there would mis-assign locants
    # (e.g. "N,N-tetramethyl"). Multi-N principal amines fall back to the prior
    # path (the substituent walk names each "(dimethylamino)"), avoiding a drop.
    if len(_principal_amine_n_atoms) == 1:
        from ..assembly.composer import _name_r_group, _wrap_n_substituent
        from collections import Counter as _NCounter

        _amine_n_atom = next(iter(_principal_amine_n_atoms))
        _chain_set_amine = set(principal_chain)
        _n_sub_names: List[str] = []
        for _pnb in mol.GetAtomWithIdx(_amine_n_atom).GetNeighbors():
            _pni = _pnb.GetIdx()
            # Skip the parent-chain attachment carbon and hydrogens;
            # the remaining heavy neighbours are the N-substituents.
            if _pni in _chain_set_amine or _pnb.GetAtomicNum() <= 1:
                continue
            _rn = _name_r_group(mol, _pni, exclude_atoms={_amine_n_atom})
            if _rn:
                _n_sub_names.append(_rn)
        if _n_sub_names:
            _n_counts = _NCounter(_n_sub_names)
            for _rn in sorted(_n_counts.keys()):
                _rc = _n_counts[_rn]
                if _rc == 1:
                    all_prefixes.append(f"N-{_wrap_n_substituent(_rn)}")
                else:
                    _rmult = get_multiplier_prefix(_rc, _rn)
                    all_prefixes.append(
                        f"N,N-{_rmult}{_wrap_n_substituent(_rn)}"
                    )

    # --- Merge duplicate bare prefix names ---
    # When two FG detection paths (e.g., primary_alcohol and secondary_alcohol)
    # both produce the same bare prefix form ("hydroxy"), merge into one entry
    # with the correct multiplier ("dihydroxy"). Only bare (no-locant) prefixes
    # are merged; locanted prefixes represent distinct chain positions.
    all_prefixes = _merge_bare_duplicate_prefixes(all_prefixes)

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
    from ..assembly.naming_utils import should_omit_locant_one
    if should_omit_locant_one(context="suffix", fg_type=principal_group):
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

        is_ring = getattr(features, 'is_cyclic', False) and not getattr(features, 'chain_is_parent', False)
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

    # ASSEMBLY_AUDIT: detect FGs present in molecule but missing from final name.
    # Guarded by logger level check so there is no performance impact in production.
    if logger.isEnabledFor(logging.DEBUG):
        detected_fgs = set()
        for fg_name_audit, fg_matches in non_principal.items():
            if fg_name_audit in ('alkene', 'alkyne'):
                continue
            if fg_matches:
                detected_fgs.add(fg_name_audit)
        missing_fgs = set()
        for fg_audit in detected_fgs:
            prefix = PREFIX_FORMS.get(fg_audit)
            if prefix is None:
                continue  # functional-class-only, no prefix form expected
            if prefix and prefix in name:
                continue
            missing_fgs.add(fg_audit)
        if missing_fgs:
            logger.debug(
                "ASSEMBLY_AUDIT: missing_fg=%s in name=%s smiles=%s",
                missing_fgs, name, getattr(features, 'canonical_smiles', '?'),
            )

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

    # HYG-04 site#2: when a PRINCIPAL amine carries N-substituents (e.g. -N(CH3)2),
    # those atoms ARE the principal group; they are expressed as N,N- prefixes on
    # the amine suffix (in name_polyfunctional), NOT walked as a "(dimethylamino)"
    # substituent. Skipping the principal-amine's own branch here prevents naming
    # the same nitrogen twice (the "1-(dimethylamino)...-1-amine" double-count).
    _AMINE_PRINCIPALS = {
        'primary_amine', 'secondary_amine', 'tertiary_amine', 'aromatic_amine',
    }
    # The principal amine's NITROGEN atom(s). A substituent branch that contains
    # the principal nitrogen IS the principal group (its alkyls extend past the
    # FG-match atoms, so an "issubset" test misses multi-carbon N-substituents).
    _principal_amine_n: set = set()
    if getattr(features, 'principal_group', None) in _AMINE_PRINCIPALS:
        for _pm in getattr(features, 'principal_group_atoms', []) or []:
            for _pa in _pm:
                if mol.GetAtomWithIdx(_pa).GetSymbol() == 'N':
                    _principal_amine_n.add(_pa)
    # Scope to a single principal amine nitrogen (matches the N,N-prefix emission
    # in name_polyfunctional). Di-/polyamines fall back to the prior path.
    _exclude_principal_amine = len(_principal_amine_n) == 1

    for position, sub_list in features.substituents.items():
        for sub_atoms in sub_list:
            # HYG-04 site#2: don't name the principal amine's own branch as a
            # substituent — its N-substituents are emitted as N,N- prefixes.
            if _exclude_principal_amine and (_principal_amine_n & set(sub_atoms)):
                continue
            # Skip substituents whose ring atoms are handled by
            # _generate_ring_substituent_prefixes (ring + its own substituents).
            # A substituent is a "ring substituent" if its attachment atom
            # (the atom bonded to the principal chain) is IN the ring.
            # Substituents that contain rings deeper in the branch (e.g.,
            # benzoylamino where N attaches to chain) are NOT skipped.
            if ring_atoms_to_skip:
                _chain_set_local = set(features.principal_chain) if features.principal_chain else set()
                _attach_in_ring = False
                for _si in sub_atoms:
                    if _si in ring_atoms_to_skip:
                        _sa = mol.GetAtomWithIdx(_si)
                        for _nb in _sa.GetNeighbors():
                            if _nb.GetIdx() in _chain_set_local:
                                _attach_in_ring = True
                                break
                    if _attach_in_ring:
                        break
                if _attach_in_ring:
                    continue

            # Skip substituent branches entirely covered by a single non-principal
            # FG match ONLY for FG types whose prefix form includes the carbon
            # (carbamoyl, carboxy, carbonochloridoyl, etc.).  These are the FG itself
            # (e.g., -C(=O)NH2 for amide) and are emitted as FG prefixes by the
            # FG prefix loop above.  Processing them here would produce incorrect
            # compound substituent names like "(aminomethyl)".
            # IUPAC P-66.1(c): non-principal amide = carbamoyl prefix.
            # NOTE: Only applies to specific terminal-C FG types. Other FGs
            # (amine, ketone, secondary_amide, etc.) must NOT trigger this guard.
            _POLY_GUARD_FG_TYPES = {
                'primary_amide', 'carboxylic_acid',
                'acid_chloride', 'acid_bromide', 'acid_fluoride',
                # v23 Phase 12 follow-on (F3 ureido): a urea substituent
                # -NH-C(=O)-NH2 is emitted as the 'carbamoylamino' FG prefix by
                # the FG loop above (its carbon is in the prefix). Without this
                # guard the SAME branch is ALSO named by _check_for_acylamino
                # below, which counts the ureido C(=O) as a 1-carbon acyl ->
                # the spurious 'methanoylamino', doubling the prefix on one
                # carbon ('5-carbamoylamino-5-(methanoylamino)' = a different,
                # OPSIN-unparseable molecule). The urea match [NX3][CX3](=O)[NX3]
                # covers the whole branch, so the subset test skips it here.
                'urea',
                # Wave2 T2b (claimed-atom mask): FG prefixes that fully name
                # their branch — isocyanato/isothiocyanato (P-35.2.1),
                # isocyano (P-66.5.3), guanidino (P-66.4.1.2.2). Without the
                # skip the SAME branch is re-walked by the generic namers,
                # which mis-read the heterocumulene/guanidine atoms and emit
                # a phantom co-substituent on the same locant
                # ('8-formamido-8-isocyanatooctanoic acid',
                #  '8-isothiocyanato-8-(methylamino)octanoic acid',
                #  '4-guanidino-4-(methylamino)butanoic acid') — a different,
                # often unparseable molecule. Subset test: each SMARTS spans
                # the whole branch (isocyanate/isothiocyanate/isocyanide
                # include the chain-anchor C; guanidine IS the branch).
                'isocyanate', 'isothiocyanate', 'isocyanide', 'guanidine',
                # Wave-2 completion: thiourea -NH-C(=S)-NH2 -> carbamothioylamino
                # FG prefix (whole branch), same double-count guard as urea.
                'thiourea',
            }
            _sub_set = set(sub_atoms)
            _skip_fg_branch = False
            _non_principal = getattr(features, 'non_principal_groups', {})
            for _fg_nm, _fg_ms in _non_principal.items():
                if _fg_nm not in _POLY_GUARD_FG_TYPES:
                    continue
                if not _fg_ms:
                    continue
                for _fg_m in _fg_ms:
                    if set(_fg_m) and _sub_set.issubset(set(_fg_m)):
                        _skip_fg_branch = True
                        break
                if _skip_fg_branch:
                    break
            if _skip_fg_branch:
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
                    # Skip S-attached branches if S is part of a named FG
                    # (sulfoxide, sulfone, thioether) -- already named by FG prefix system
                    _S_FG_NAMES = {'sulfoxide', 'sulfone', 'thioether'}
                    _skip_s_branch = False
                    if _attach is not None and _attach_sym == 'S':
                        _all_fgs = features.functional_groups if hasattr(features, 'functional_groups') else {}
                        for _fg_n in _S_FG_NAMES:
                            for _fg_match in _all_fgs.get(_fg_n, []):
                                if _attach in _fg_match:
                                    _skip_s_branch = True
                                    break
                            if _skip_s_branch:
                                break
                    if not sub_has_ring and _attach_sym != 'O' and not _skip_s_branch:
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

    # Build formatted prefixes using format_substituent_prefix for proper
    # complex substituent handling (e.g., bis(acetyloxy) not bisacetyloxy)
    from ..assembly.naming_utils import format_substituent_prefix as _fmt_sub
    prefixes = []
    for name, locants in substituent_groups.items():
        count = len(locants)
        sorted_locs = sorted(locants)
        formatted = _fmt_sub(name, sorted_locs, count)
        prefixes.append(formatted)

    return prefixes


def _merge_bare_duplicate_prefixes(prefixes: List[str]) -> List[str]:
    """Merge duplicate prefix strings (bare or locanted) into single entries.

    Merges prefixes that share the same base name, combining locants and
    adjusting multipliers. Handles both bare ("hydroxy" + "hydroxy" ->
    "dihydroxy") and locanted ("4-hydroxy" + "5-hydroxy" -> "4,5-dihydroxy").

    Prefixes with parentheses or N- are never merged (compound substituents).

    Args:
        prefixes: Formatted prefix strings.

    Returns:
        De-duplicated prefix list.
    """
    import re as _re

    if len(prefixes) <= 1:
        return prefixes

    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS

    def _parse_prefix(ptext: str):
        """Parse prefix into (locants, base_name) or None if unparseable."""
        # Skip compound/N-substituted prefixes
        if ptext.startswith('(') or ptext.startswith('N-') or ptext.startswith('N,'):
            return None

        locants = []
        rest = ptext

        # Extract leading locants: "4,5-dihydroxy" -> locants=[4,5], rest="dihydroxy"
        m = _re.match(r'^([\d,]+)-(.+)$', ptext)
        if m:
            loc_str = m.group(1)
            rest = m.group(2)
            for loc in loc_str.split(','):
                try:
                    locants.append(int(loc))
                except ValueError:
                    return None  # string locants like '3a' -- skip merge

        # Strip multiplier prefix to get base name
        base = rest
        for _cnt, mult in sorted(SIMPLE_MULTIPLIERS.items(),
                                 key=lambda x: len(x[1]), reverse=True):
            if rest.startswith(mult):
                cand = rest[len(mult):]
                if cand and cand[0].islower():
                    base = cand
                    break

        return (locants, base)

    # Group prefixes by base name
    groups: Dict[str, List[Tuple[int, List[int]]]] = {}  # base -> [(index, locants)]
    ungrouped: List[int] = []  # indices of prefixes that can't be parsed/merged

    for i, ptext in enumerate(prefixes):
        parsed = _parse_prefix(ptext)
        if parsed is None:
            ungrouped.append(i)
            continue
        locants, base = parsed
        if base not in groups:
            groups[base] = []
        groups[base].append((i, locants))

    # Check if any group needs merging
    needs_merge = any(len(entries) > 1 for entries in groups.values())
    if not needs_merge:
        return prefixes

    # Build merged result
    merged_indices: set = set()
    extra: List[str] = []

    for base, entries in groups.items():
        if len(entries) <= 1:
            continue

        # Separate bare (no locants) from locanted entries
        bare_entries = [(idx, locs) for idx, locs in entries if not locs]
        locanted_entries = [(idx, locs) for idx, locs in entries if locs]

        # Only merge within the same category (all-bare or all-locanted).
        # Mixed groups (bare + locanted) are NOT merged because bare
        # prefixes have no position info to combine with locants.
        merge_sets = []
        if len(bare_entries) > 1:
            merge_sets.append(bare_entries)
        if len(locanted_entries) > 1:
            merge_sets.append(locanted_entries)

        for merge_group in merge_sets:
            all_locants: List[int] = []
            for idx, locs in merge_group:
                all_locants.extend(locs)
                merged_indices.add(idx)
            all_locants = sorted(set(all_locants))
            total_count = len(all_locants) if all_locants else len(merge_group)

            # Rebuild using format_fg_prefix (peer function in this module)
            rebuilt = format_fg_prefix(base, all_locants, total_count)
            extra.append(rebuilt)

    result = [prefixes[i] for i in range(len(prefixes)) if i not in merged_indices]
    result.extend(extra)
    return result


# SUB-05/D-16 (4b): bare detachable oxo-acid prefixes that, when glued
# letter->letter to a preceding substituent, must be hyphen-separated. EXACT
# membership only (so 'phosphonooxy'/'phenylsulfanyl' fused tokens are never
# split). Conservative oxo-acid set; deliberately excludes the common
# nitro/nitroso to avoid false splits.
_DETACHABLE_LETTER_PREFIXES = frozenset({
    "phosphono", "sulfo", "sulfino", "arsono", "borono",
})


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
            # An italic-N locant prefix ("N-..." / "N,N-...") is a locant-bearing
            # term like a numeric locant — it must be hyphen-separated from a
            # preceding substituent (e.g. "2-methoxy" + "N,N-dimethyl" ->
            # "2-methoxy-N,N-dimethyl"). HYG-04 site#2.
            next_is_n_locant = (
                first_char == 'N' and len(current) > 1 and current[1] in (',', '-')
            )
            # SUB-05/D-16 (4b): a bare detachable oxo-acid prefix (phosphono,
            # sulfo, ...) glued letter->letter to a preceding substituent must
            # be separated ("3-oxo"+"phosphono" -> "3-oxo-phosphono", NOT
            # "3-oxophosphono"). GATED to an exact detachable-prefix token set
            # so legitimately-fused single tokens are NEVER split:
            # 'phosphonooxy'/'phenylsulfanyl' (.split('-')[-1] != a gate member)
            # and '2,3-dimethyl' (digit-initial) stay intact.
            current_tail = current.split('-')[-1]
            is_detachable = (current in _DETACHABLE_LETTER_PREFIXES
                             or current_tail in _DETACHABLE_LETTER_PREFIXES)
            # Insert hyphen between letter/paren and digit
            # e.g., "amino" + "4-methyl" → "amino-4-methyl"
            # e.g., "(ethanoyl)amino" + "4-methyl" → "(ethanoyl)amino-4-methyl"
            if first_char.isdigit() and (last_char.isalpha() or last_char == ')'):
                result += "-"
            # Also between ')' and letter for clarity
            elif last_char == ')' and first_char.isalpha():
                result += "-"
            # Letter/paren followed by an italic-N locant prefix
            elif next_is_n_locant and (last_char.isalpha() or last_char == ')'):
                result += "-"
            # Letter followed by a bare detachable oxo-acid prefix (4b)
            elif is_detachable and last_char.isalpha() and first_char.isalpha():
                result += "-"
        result += current

    return result
