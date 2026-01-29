"""
Name assembly - combining fragments into complete IUPAC names.

Assembly order:
1. Stereodescriptors (R/S, E/Z) at start in parentheses
2. Locanted prefixes (substituents, alphabetized)
3. Parent name (with unsaturation modifiers)
4. Locanted suffix (principal group)
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

from ..rules.seniority import get_suffix, get_prefix
from ..rules.locants import get_functional_group_locants, get_bond_locants
from .naming_utils import (
    format_suffix_with_locants,
    get_multiplier_prefix,
    format_substituent_prefix,
    alpha_sort_key,
    get_alkyl_name,
)


# Chain length prefixes (IUPAC Blue Book)
CHAIN_PREFIXES = {
    1: "meth", 2: "eth", 3: "prop", 4: "but", 5: "pent",
    6: "hex", 7: "hept", 8: "oct", 9: "non", 10: "dec",
    11: "undec", 12: "dodec", 13: "tridec", 14: "tetradec",
    15: "pentadec", 16: "hexadec", 17: "heptadec", 18: "octadec",
    19: "nonadec", 20: "icos", 21: "henicos", 22: "docos",
    23: "tricos", 24: "tetracos", 25: "pentacos",
    30: "triacont", 40: "tetracont", 50: "pentacont",
    60: "hexacont", 70: "heptacont", 80: "octacont",
    90: "nonacont", 100: "hect",
}

# Simple multiplicative prefixes (for simple substituent names)
SIMPLE_MULTIPLIERS = {
    2: "di", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
    11: "undeca", 12: "dodeca",
}

# Complex multiplicative prefixes (for substituents with locants/hyphens)
COMPLEX_MULTIPLIERS = {
    2: "bis", 3: "tris", 4: "tetrakis", 5: "pentakis",
    6: "hexakis", 7: "heptakis", 8: "octakis", 9: "nonakis", 10: "decakis",
}

# Prefixes to IGNORE for alphabetization
IGNORE_FOR_ALPHA = {
    "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "nona", "deca",
    "undeca", "dodeca",
    "bis", "tris", "tetrakis", "pentakis", "hexakis", "heptakis",
    "octakis", "nonakis", "decakis",
}

# Terminal functional groups that NEVER include locants in the name
# These are always at position 1 by definition (chain numbered from terminal group)
TERMINAL_GROUPS = {
    "carboxylic_acid",  # Always at chain end (locant 1)
    "aldehyde",         # Always at chain end (locant 1)
}


@dataclass
class NameFragment:
    """A fragment of an IUPAC name."""
    text: str
    locants: tuple = ()
    priority: int = 0
    fragment_type: str = "prefix"  # prefix, parent, suffix, stereo


def assemble_name(features: Any, style: str = "pin") -> str:
    """
    Assemble complete IUPAC name from molecular features.

    Args:
        features: MolecularFeatures object with extracted features
        style: Naming style ("pin", "general", "cas")

    Returns:
        Complete IUPAC name string
    """
    # Handle heterocyclic compounds FIRST
    # Heterocycles include aromatic heterocycles (pyridine) and saturated (morpholine)
    ring_type = getattr(features, 'ring_type', None)
    if ring_type == 'heterocyclic':
        return _assemble_heterocycle_name(features, style)

    # Handle benzene derivatives specially
    # Benzene naming generates the complete name directly, not fragments
    if getattr(features, 'is_benzene', False):
        return _assemble_benzene_name(features, style)

    # Handle polycyclic aromatics specially
    # Polycyclics (naphthalene, anthracene, etc.) have their own naming rules
    polycyclic_name = getattr(features, 'polycyclic_name', None)
    if polycyclic_name:
        return _assemble_polycyclic_name(features, style)

    # Handle simple cases
    if not features.principal_chain and not features.ring_systems:
        # Single atom or very simple molecule
        return _name_simple_molecule(features)

    fragments = []

    # Generate parent name (chain or ring)
    if features.principal_chain:
        parent = _generate_chain_parent(features)
    elif features.ring_systems:
        parent = _generate_ring_parent(features)
    else:
        parent = NameFragment(text="", fragment_type="parent")

    fragments.append(parent)

    # Generate suffix for principal group
    if features.principal_group:
        suffix = _generate_suffix(features)
        if suffix:
            fragments.append(suffix)

    # Generate prefixes for substituents and non-principal groups
    prefixes = _generate_prefixes(features)
    fragments.extend(prefixes)

    # Generate stereodescriptors
    if features.stereocenters:
        stereo = _generate_stereodescriptors(features)
        if stereo:
            fragments.append(stereo)

    # Assemble in correct order
    return _assemble_fragments(fragments, style)


def _name_simple_molecule(features: Any) -> str:
    """Name very simple molecules (single atom, etc.)."""
    mol = features.mol

    if mol.GetNumAtoms() == 1:
        atom = mol.GetAtomWithIdx(0)
        symbol = atom.GetSymbol()
        # Single atom molecules
        if symbol == 'C':
            return "methane"  # CH4
        # Add more as needed

    return "unknown"


def _assemble_benzene_name(features: Any, style: str) -> str:
    """
    Assemble name for benzene derivatives.

    Benzene naming is handled specially because:
    1. Substituents are named relative to the ring, not a chain
    2. Ring orientation determines locants (not chain direction)
    3. The parent is always "benzene"

    Args:
        features: MolecularFeatures with is_benzene=True
        style: Naming style (only "pin" supported for now)

    Returns:
        Complete IUPAC name for the benzene derivative
    """
    from ..rules.benzene import orient_benzene, name_substituted_benzene

    mol = features.mol
    ring_atoms = features.benzene_ring
    substituents = features.benzene_substituents

    # If no substituents, return "benzene" (should be caught by retained names,
    # but handle here as fallback)
    if not substituents:
        return "benzene"

    # Orient the ring for lowest locants
    oriented_ring = orient_benzene(mol, ring_atoms, substituents)

    # Generate systematic name
    return name_substituted_benzene(mol, ring_atoms, oriented_ring, substituents)


def _assemble_polycyclic_name(features: Any, style: str) -> str:
    """
    Assemble name for polycyclic aromatic hydrocarbons.

    Polycyclic naming is handled specially because:
    1. Substituent numbering is FIXED by IUPAC (not lowest locants)
    2. Parent is a retained name (naphthalene, anthracene, etc.)
    3. Multiple fused rings have standard numbering

    Args:
        features: MolecularFeatures with polycyclic_name set
        style: Naming style (only "pin" supported for now)

    Returns:
        Complete IUPAC name for the polycyclic aromatic
    """
    from ..rules.polycyclics import name_substituted_polycyclic

    pah_name = features.polycyclic_name
    substituents = features.polycyclic_substituents

    # If no substituents, return the PAH name
    # (This should be caught by retained names, but handle here as fallback)
    if not substituents:
        return pah_name

    # Generate systematic name with substituents
    return name_substituted_polycyclic(features.mol, pah_name, substituents)


def _assemble_heterocycle_name(features: Any, style: str) -> str:
    """
    Assemble name for heterocyclic compounds.

    For unsubstituted heterocycles, returns the parent name directly
    (either retained name or HW systematic name).

    For substituted heterocycles, adds prefixes with locants:
    - N-substitution uses N-locant format (N-methyl, N,N-dimethyl)
    - C-substitution uses numeric locants (2-methyl, 3-ethyl)

    Args:
        features: MolecularFeatures with ring_type='heterocyclic'
        style: Naming style (only "pin" supported for now)

    Returns:
        Complete IUPAC name for the heterocycle
    """
    from ..rules.heterocycles import name_heterocycle, name_substituted_heterocycle

    # Get the heterocycle parent name
    parent_name = name_heterocycle(features.mol, features.principal_ring)

    # Check for substituents
    substituents = getattr(features, 'heterocycle_substituents', None)
    atom_to_locant = getattr(features, 'heterocycle_atom_to_locant', None)

    if substituents and atom_to_locant:
        # Generate substituted name with N-locants and C-locants
        return name_substituted_heterocycle(
            features.mol,
            features.principal_ring,
            parent_name,
            substituents,
            atom_to_locant
        )

    # No substituents - return parent name directly
    return parent_name


def _generate_chain_parent(features: Any) -> NameFragment:
    """Generate parent name for acyclic chains.

    Returns a NameFragment where:
    - text: stem + unsaturation_base (e.g., "but", "prop")
    - locants: tuple of (double_bond_locants, triple_bond_locants)
    """
    chain_length = len(features.principal_chain)

    # Get chain prefix (stem)
    if chain_length in CHAIN_PREFIXES:
        stem = CHAIN_PREFIXES[chain_length]
    else:
        stem = _build_long_chain_prefix(chain_length)

    # Get bond locants if we have atom_to_locant mapping
    double_locants = []
    triple_locants = []
    if features.atom_to_locant:
        double_locants = get_bond_locants(
            features.principal_chain,
            features.double_bonds,
            features.atom_to_locant
        )
        triple_locants = get_bond_locants(
            features.principal_chain,
            features.triple_bonds,
            features.atom_to_locant
        )

    # Store stem as text, bond locants as locants tuple
    # We'll use a nested tuple: ((double_locants), (triple_locants))
    return NameFragment(
        text=stem,
        locants=(tuple(double_locants), tuple(triple_locants)),
        fragment_type="parent"
    )


def _generate_ring_parent(features: Any) -> NameFragment:
    """
    Generate parent name for cyclic compounds.

    For cycloalkanes: "cyclo" + chain prefix + "an" (e.g., "cyclohexan")
    The final 'e' is added during assembly if no suffix follows.

    For cycloalkenes: "cyclo" + chain prefix + double bond info
    - Mono-cycloalkenes: cyclo + stem + "en" (cyclohexene) - no locant
    - Cycloalkadienes: cyclo + stem + "a" + locants + "dien" (cyclohexa-1,3-diene)

    For other ring types, returns placeholder for now (to be implemented
    in subsequent plans).

    Args:
        features: MolecularFeatures object with ring_type and principal_ring

    Returns:
        NameFragment with parent text and bond locants
    """
    ring_type = getattr(features, 'ring_type', None)
    principal_ring = getattr(features, 'principal_ring', None)

    if not principal_ring:
        # Fallback: no ring identified
        return NameFragment(text="cyclo", fragment_type="parent")

    ring_size = len(principal_ring)

    if ring_size in CHAIN_PREFIXES:
        stem = CHAIN_PREFIXES[ring_size]
    else:
        stem = _build_long_chain_prefix(ring_size)

    if ring_type == 'cycloalkane':
        # Cycloalkane naming: cyclo + stem + an (e.g., cyclohexan)
        # Return stem - the "ane" will be added in assembly
        return NameFragment(
            text=f"cyclo{stem}",
            locants=((), ()),  # No bond locants for saturated rings
            fragment_type="parent"
        )

    elif ring_type == 'cycloalkene':
        # Get double bond locants from features
        ring_double_bond_locants = getattr(features, 'ring_double_bond_locants', [])

        return NameFragment(
            text=f"cyclo{stem}",
            locants=(tuple(ring_double_bond_locants), ()),  # (double_bond_locants, triple_bond_locants)
            fragment_type="parent"
        )

    elif ring_type == 'aromatic':
        # TODO: Implement aromatic naming (Plan 02-04)
        return NameFragment(text="cyclo", fragment_type="parent")

    elif ring_type == 'heterocyclic':
        # TODO: Implement heterocyclic naming (Phase 3)
        return NameFragment(text="cyclo", fragment_type="parent")

    # Unknown ring type, return placeholder
    return NameFragment(text="cyclo", fragment_type="parent")


def _get_unsaturation_suffix(features: Any) -> str:
    """
    Get the unsaturation suffix (an, en, yn, etc.).
    
    Returns suffix without the 'e' ending (that's added later based on what follows).
    """
    num_double = len(features.double_bonds)
    num_triple = len(features.triple_bonds)
    
    if num_double == 0 and num_triple == 0:
        return "an"  # Saturated
    
    parts = []
    
    # Double bonds
    if num_double == 1:
        parts.append("en")
    elif num_double > 1:
        multiplier = SIMPLE_MULTIPLIERS.get(num_double, str(num_double))
        parts.append(f"{multiplier}en")  # dien, trien, etc.
    
    # Triple bonds
    if num_triple == 1:
        parts.append("yn")
    elif num_triple > 1:
        multiplier = SIMPLE_MULTIPLIERS.get(num_triple, str(num_triple))
        parts.append(f"{multiplier}yn")
    
    if not parts:
        return "an"
    
    # For mixed unsaturation, use "a" connector
    if num_double > 0 and num_triple > 0:
        return "a" + "".join(parts)  # e.g., "adienyn"
    
    return "".join(parts)


def _generate_suffix(features: Any) -> Optional[NameFragment]:
    """Generate suffix fragment for principal group.

    Handles locant assignment for the principal functional group:
    - Terminal groups (carboxylic acid, aldehyde): locant always 1, omitted from name
    - Non-terminal groups (alcohol, ketone): locant MUST be included in PIN style
    - Multiple instances: include multiplier (di, tri) and all locants
    """
    fg_name = features.principal_group
    if not fg_name:
        return None

    is_ring = features.is_cyclic
    suffix_text = get_suffix(fg_name, is_ring=is_ring)

    if not suffix_text:
        return None

    # Get locants for functional group positions on the chain
    locants = ()
    if features.principal_chain and features.atom_to_locant and features.principal_group_atoms:
        fg_locants = get_functional_group_locants(
            features.principal_chain,
            features.principal_group_atoms,
            features.atom_to_locant,
            mol=features.mol
        )

        # Terminal groups: locant is implicitly 1, do NOT include in name
        if fg_name in TERMINAL_GROUPS:
            # For terminal groups, we don't include the locant
            locants = ()
        else:
            # For non-terminal groups (alcohol, ketone), include locants
            locants = tuple(fg_locants)

    return NameFragment(
        text=suffix_text,
        locants=locants,
        fragment_type="suffix"
    )


def _generate_prefixes(features: Any) -> List[NameFragment]:
    """
    Generate prefix fragments for alkyl substituents and non-principal groups.

    For alkane/cycloalkane naming, this extracts alkyl substituents from
    features.substituents (for chains) or features.ring_substituents (for rings),
    groups them by name (methyl, ethyl, etc.), and formats with locants and
    multiplicative prefixes.
    """
    prefixes = []

    # --- Handle alkyl substituents from features.substituents (chains) ---
    if features.substituents and features.mol:
        alkyl_prefixes = _generate_alkyl_prefixes(features)
        prefixes.extend(alkyl_prefixes)

    # --- Handle ring substituents from features.ring_substituents ---
    ring_substituents = getattr(features, 'ring_substituents', None)
    oriented_ring = getattr(features, 'oriented_ring', None)
    if ring_substituents and features.mol and oriented_ring:
        ring_prefixes = _generate_ring_alkyl_prefixes(features)
        prefixes.extend(ring_prefixes)

    # --- Handle non-principal functional groups as prefixes ---
    for fg_name, matches in features.functional_groups.items():
        if fg_name == features.principal_group:
            continue

        prefix_text = get_prefix(fg_name)
        if prefix_text and matches:
            count = len(matches)

            if count > 1:
                # Add multiplier
                multiplier = SIMPLE_MULTIPLIERS.get(count, str(count))
                prefix_text = f"{multiplier}{prefix_text}"

            prefixes.append(NameFragment(
                text=prefix_text,
                locants=(),  # TODO: Add locants
                fragment_type="prefix"
            ))

    return prefixes


def _generate_alkyl_prefixes(features: Any) -> List[NameFragment]:
    """
    Generate prefix fragments for alkyl substituents.

    This function:
    1. Iterates over features.substituents (keyed by 1-indexed position/locant)
    2. Counts carbon atoms in each substituent to determine name
    3. Groups identical substituents with their locants
    4. Formats each group with locants, multipliers
    5. Sorts alphabetically by base substituent name

    Returns:
        List of NameFragment objects for alkyl prefixes, sorted alphabetically
    """
    mol = features.mol

    # Group substituents by name: {name: [locants]}
    substituent_groups: Dict[str, List[int]] = defaultdict(list)

    for position, sub_list in features.substituents.items():
        # position is already a 1-indexed locant (from get_substituents)
        for sub_atoms in sub_list:
            # Count only carbon atoms in the substituent
            carbon_count = sum(
                1 for idx in sub_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )

            # Skip non-alkyl substituents (no carbons = functional group like -OH)
            if carbon_count == 0:
                continue

            # Check if substituent has any heteroatoms (non-C, non-H)
            has_heteroatom = any(
                mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
                for idx in sub_atoms
            )

            # For Phase 1, skip mixed substituents (carbon + heteroatom)
            # These are complex substituents handled in later phases
            if has_heteroatom:
                continue

            # Get alkyl name from carbon count
            try:
                alkyl_name = get_alkyl_name(carbon_count)
                substituent_groups[alkyl_name].append(position)
            except ValueError:
                # Carbon count > 10, skip for now (complex substituent)
                continue

    # Build prefix fragments
    prefixes = []
    for name, locants in substituent_groups.items():
        count = len(locants)
        formatted = format_substituent_prefix(name, sorted(locants), count)
        prefixes.append(NameFragment(
            text=formatted,
            locants=tuple(sorted(locants)),
            fragment_type="prefix"
        ))

    # Sort by IUPAC alphabetization rules (ignoring di-, tri-, etc.)
    prefixes.sort(key=lambda f: alpha_sort_key(f.text))

    return prefixes


def _generate_ring_alkyl_prefixes(features: Any) -> List[NameFragment]:
    """
    Generate prefix fragments for alkyl substituents on rings.

    This function:
    1. Iterates over features.ring_substituents (keyed by ring atom index)
    2. Uses features.oriented_ring to determine locants
    3. Counts carbon atoms in each substituent to determine name
    4. Groups identical substituents with their locants
    5. Formats each group with locants, multipliers
    6. Sorts alphabetically by base substituent name

    IUPAC rules for cycloalkane substituent locants:
    - Monosubstituted cycloalkanes: locant 1 is implicit and omitted
      (methylcyclohexane, not 1-methylcyclohexane)
    - Polysubstituted cycloalkanes: all locants included
      (1,2-dimethylcyclohexane)

    Returns:
        List of NameFragment objects for alkyl prefixes, sorted alphabetically
    """
    mol = features.mol
    ring_substituents = features.ring_substituents
    oriented_ring = features.oriented_ring

    # Build atom-to-locant mapping from oriented ring
    atom_to_locant = {atom_idx: i + 1 for i, atom_idx in enumerate(oriented_ring)}

    # Group substituents by name: {name: [locants]}
    substituent_groups: Dict[str, List[int]] = defaultdict(list)

    for ring_atom_idx, sub_list in ring_substituents.items():
        # Get locant for this ring position
        locant = atom_to_locant.get(ring_atom_idx)
        if locant is None:
            continue

        for sub_atoms in sub_list:
            # Count only carbon atoms in the substituent
            carbon_count = sum(
                1 for idx in sub_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )

            # Skip non-alkyl substituents (no carbons = functional group like -OH)
            if carbon_count == 0:
                continue

            # Check if substituent has any heteroatoms (non-C, non-H)
            has_heteroatom = any(
                mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
                for idx in sub_atoms
            )

            # For Phase 2, skip mixed substituents (carbon + heteroatom)
            # These are complex substituents handled in later phases
            if has_heteroatom:
                continue

            # Get alkyl name from carbon count
            try:
                alkyl_name = get_alkyl_name(carbon_count)
                substituent_groups[alkyl_name].append(locant)
            except ValueError:
                # Carbon count > 10, skip for now (complex substituent)
                continue

    # Count total number of substituents to determine if monosubstituted
    total_substituents = sum(len(locs) for locs in substituent_groups.values())
    is_monosubstituted = total_substituents == 1

    # Build prefix fragments
    prefixes = []
    for name, locants in substituent_groups.items():
        count = len(locants)

        if is_monosubstituted:
            # Monosubstituted: omit locant (it's always 1)
            # Just the substituent name: "methyl" not "1-methyl"
            formatted = name
        else:
            # Polysubstituted: include locants
            formatted = format_substituent_prefix(name, sorted(locants), count)

        prefixes.append(NameFragment(
            text=formatted,
            locants=tuple(sorted(locants)),
            fragment_type="prefix"
        ))

    # Sort by IUPAC alphabetization rules (ignoring di-, tri-, etc.)
    prefixes.sort(key=lambda f: alpha_sort_key(f.text))

    return prefixes


def _generate_stereodescriptors(features: Any) -> Optional[NameFragment]:
    """Generate stereodescriptor prefix."""
    if not features.stereocenters:
        return None
    
    descriptors = []
    for center in features.stereocenters:
        # TODO: Calculate proper locant from chain/ring numbering
        locant = center['idx'] + 1  # Placeholder: 1-indexed atom index
        cip = center['cip']
        descriptors.append(f"{locant}{cip}")
    
    if not descriptors:
        return None
    
    text = f"({','.join(descriptors)})-"
    return NameFragment(text=text, fragment_type="stereo")


def _assemble_fragments(fragments: List[NameFragment], style: str) -> str:
    """
    Assemble fragments into final name string.

    Order: stereo + prefixes (alphabetized) + parent + suffix

    For functional group compounds, uses PIN-style infix locants:
    - propan-1-ol (not propanol or 1-propanol)
    - butan-2-one (not butanone or 2-butanone)

    For unsaturated hydrocarbons, includes bond locants:
    - but-1-ene (not butene or 1-butene)
    - pent-1-en-4-yne (enyne)
    """
    stereo = ""
    prefixes = []
    parent_frag = None
    suffix_frag = None

    for frag in fragments:
        if frag.fragment_type == "stereo":
            stereo = frag.text
        elif frag.fragment_type == "prefix":
            prefixes.append(frag)
        elif frag.fragment_type == "parent":
            parent_frag = frag
        elif frag.fragment_type == "suffix":
            suffix_frag = frag

    # Alkyl prefixes are already sorted by _generate_alkyl_prefixes.
    # For non-alkyl prefixes added later, sort all together.
    prefixes.sort(key=lambda f: alpha_sort_key(f.text))

    # Concatenate prefixes with proper hyphenation
    # IUPAC rule: hyphens separate locants from names, and are needed
    # between prefixes when one ends with a letter and the next starts with a digit
    prefix_str = _join_prefixes([f.text for f in prefixes])

    # Extract parent info: stem is in text, bond locants in locants
    stem = parent_frag.text if parent_frag else ""
    double_locants = []
    triple_locants = []
    if parent_frag and parent_frag.locants:
        # locants is a tuple of (double_bond_locants, triple_bond_locants)
        double_locants = list(parent_frag.locants[0]) if parent_frag.locants[0] else []
        triple_locants = list(parent_frag.locants[1]) if len(parent_frag.locants) > 1 and parent_frag.locants[1] else []

    # Handle suffix attachment using PIN-style formatting
    if suffix_frag and suffix_frag.text:
        suffix_text = suffix_frag.text
        suffix_locants = list(suffix_frag.locants) if suffix_frag.locants else []

        # Determine multiplier for multiple functional groups
        count = len(suffix_locants)
        multiplier = get_multiplier_prefix(count, suffix_text) if count > 1 else ""

        # Build unsaturation infix with locants for compounds with functional groups
        unsaturation_infix = _build_unsaturation_infix(double_locants, triple_locants)

        name = format_suffix_with_locants(
            stem,
            unsaturation_infix,
            suffix_text,
            suffix_locants,
            multiplier
        )
    else:
        # No suffix = hydrocarbon, build name with unsaturation
        name = _build_hydrocarbon_name(stem, double_locants, triple_locants)

    # Add prefixes
    if prefix_str:
        name = f"{prefix_str}{name}"

    # Add stereodescriptors at the very start
    if stereo:
        name = f"{stereo}{name}"

    return name


def _build_unsaturation_infix(
    double_locants: List[int],
    triple_locants: List[int]
) -> str:
    """
    Build the unsaturation infix (e.g., 'an', '-1-en', '-1-en-4-yn').

    This is used when there's a functional group suffix. The unsaturation
    part is inserted between the stem and the suffix locants.

    Args:
        double_locants: Sorted list of locants for double bonds.
        triple_locants: Sorted list of locants for triple bonds.

    Returns:
        Unsaturation infix string (may be empty for saturated).

    Examples:
        >>> _build_unsaturation_infix([], [])
        'an'
        >>> _build_unsaturation_infix([1], [])
        '-1-en'
        >>> _build_unsaturation_infix([1], [4])
        '-1-en-4-yn'
    """
    num_double = len(double_locants)
    num_triple = len(triple_locants)

    if num_double == 0 and num_triple == 0:
        return "an"  # Saturated

    parts = []

    # Double bonds
    if num_double > 0:
        double_str = ",".join(str(loc) for loc in double_locants)
        if num_double == 1:
            parts.append(f"-{double_str}-en")
        else:
            multiplier = SIMPLE_MULTIPLIERS.get(num_double, str(num_double))
            # For multiple double bonds, add 'a' connector: butadiene not butdiene
            parts.append(f"a-{double_str}-{multiplier}en")

    # Triple bonds
    if num_triple > 0:
        triple_str = ",".join(str(loc) for loc in triple_locants)
        if num_triple == 1:
            parts.append(f"-{triple_str}-yn")
        else:
            multiplier = SIMPLE_MULTIPLIERS.get(num_triple, str(num_triple))
            parts.append(f"-{triple_str}-{multiplier}yn")

    # Special case: if only double bonds and parts starts with 'a-' for diene,
    # we need to handle this differently for saturated stem
    if parts:
        result = "".join(parts)
        # Clean up any double hyphens
        while "--" in result:
            result = result.replace("--", "-")
        return result

    return "an"


def _build_hydrocarbon_name(
    stem: str,
    double_locants: List[int],
    triple_locants: List[int]
) -> str:
    """
    Build a complete hydrocarbon name (no functional group suffix).

    This handles:
    - Saturated: stem + 'ane' (butane, cyclohexane)
    - Alkenes: stem + '-locant-ene' (but-1-ene) or stem + 'ene' (ethene, cyclohexene)
    - Alkynes: stem + '-locant-yne' (but-1-yne) or stem + 'yne' (ethyne)
    - Enynes: stem + '-double-en-triple-yne' (pent-1-en-4-yne)
    - Cycloalkenes: stem + 'ene' for mono (cyclohexene), stem + 'a-locants-diene' for di

    For 2-carbon compounds (ethene, ethyne), locants are omitted since
    there's only one possible position for the multiple bond.

    For mono-cycloalkenes (single double bond in ring), locants are omitted
    per IUPAC convention (cyclohexene, not cyclohex-1-ene).

    Args:
        stem: Chain prefix (e.g., 'but', 'pent', 'cyclohex').
        double_locants: Sorted list of locants for double bonds.
        triple_locants: Sorted list of locants for triple bonds.

    Returns:
        Complete hydrocarbon name.

    Examples:
        >>> _build_hydrocarbon_name("but", [], [])
        'butane'
        >>> _build_hydrocarbon_name("eth", [1], [])
        'ethene'
        >>> _build_hydrocarbon_name("but", [1], [])
        'but-1-ene'
        >>> _build_hydrocarbon_name("but", [2], [])
        'but-2-ene'
        >>> _build_hydrocarbon_name("pent", [1], [4])
        'pent-1-en-4-yne'
        >>> _build_hydrocarbon_name("cyclohex", [], [])
        'cyclohexane'
        >>> _build_hydrocarbon_name("cyclohex", [1], [])
        'cyclohexene'
        >>> _build_hydrocarbon_name("cyclohex", [1, 3], [])
        'cyclohexa-1,3-diene'
    """
    num_double = len(double_locants)
    num_triple = len(triple_locants)

    # Check if this is a cyclic compound
    is_cyclic = stem.startswith("cyclo")

    if num_double == 0 and num_triple == 0:
        # Saturated hydrocarbon
        return f"{stem}ane"

    # Check if this is a 2-carbon chain (ethene/ethyne) where locants are omitted
    # The stem 'eth' indicates 2 carbons
    is_two_carbon = stem == "eth"

    # For mono-cycloalkenes, locant is omitted (cyclohexene, not cyclohex-1-ene)
    # This is IUPAC convention for simple mono-unsaturated cyclic compounds
    omit_mono_cycloalkene_locant = is_cyclic and num_double == 1 and num_triple == 0

    parts = [stem]

    # Double bonds
    if num_double > 0:
        if num_double == 1:
            if is_two_carbon and num_triple == 0:
                # ethene - no locant needed
                parts.append("en")
            elif omit_mono_cycloalkene_locant:
                # cyclohexene - no locant needed for mono-cycloalkene
                parts.append("en")
            else:
                # Single double bond with locant: but-1-ene
                double_str = ",".join(str(loc) for loc in double_locants)
                parts.append(f"-{double_str}-en")
        else:
            # Multiple double bonds: buta-1,3-diene, cyclohexa-1,3-diene
            # Add 'a' before locants for pronunciation
            double_str = ",".join(str(loc) for loc in double_locants)
            multiplier = SIMPLE_MULTIPLIERS.get(num_double, str(num_double))
            parts.append(f"a-{double_str}-{multiplier}en")

    # Triple bonds
    if num_triple > 0:
        if num_triple == 1:
            if is_two_carbon and num_double == 0:
                # ethyne - no locant needed
                parts.append("yn")
            else:
                # Single triple bond with locant
                triple_str = ",".join(str(loc) for loc in triple_locants)
                parts.append(f"-{triple_str}-yn")
        else:
            # Multiple triple bonds
            triple_str = ",".join(str(loc) for loc in triple_locants)
            multiplier = SIMPLE_MULTIPLIERS.get(num_triple, str(num_triple))
            # If there were also double bonds, we already have 'a', otherwise add it
            if num_double == 0:
                parts.append(f"a-{triple_str}-{multiplier}yn")
            else:
                parts.append(f"-{triple_str}-{multiplier}yn")

    # Add final 'e'
    result = "".join(parts) + "e"

    # Clean up: handle edge cases
    # 1. Double hyphens shouldn't happen but clean up just in case
    while "--" in result:
        result = result.replace("--", "-")

    return result


def _split_parent_stem(parent: str) -> tuple:
    """
    Split parent string into stem and unsaturation components.

    The parent string from _generate_chain_parent is stem + unsaturation,
    e.g., "propan" = "prop" + "an", "buten" = "but" + "en".

    Returns:
        Tuple of (stem, unsaturation)
    """
    # Unsaturation suffixes in order of longest first to avoid partial matches
    unsaturation_patterns = [
        "adienyn", "adien", "adiyn", "aenyn",  # Complex unsaturation
        "dien", "diyn", "enyn",  # Double unsaturation
        "en", "yn", "an",  # Simple unsaturation
    ]

    for pattern in unsaturation_patterns:
        if parent.endswith(pattern):
            stem = parent[:-len(pattern)]
            return (stem, pattern)

    # No recognized unsaturation - return as-is with empty unsaturation
    return (parent, "")


# Note: alpha_sort_key is imported from naming_utils for IUPAC-compliant sorting.


def _join_prefixes(prefix_texts: List[str]) -> str:
    """
    Join multiple prefix strings with proper IUPAC hyphenation.

    When concatenating prefixes like "3-ethyl" and "4-methyl", the result
    should be "3-ethyl-4-methyl" (with hyphen between letter and digit).

    Args:
        prefix_texts: List of prefix strings (e.g., ["3-ethyl", "4-methyl"])

    Returns:
        Concatenated string with proper hyphenation

    Examples:
        >>> _join_prefixes(["3-ethyl", "4-methyl"])
        '3-ethyl-4-methyl'
        >>> _join_prefixes(["2,2-dimethyl"])
        '2,2-dimethyl'
        >>> _join_prefixes([])
        ''
    """
    if not prefix_texts:
        return ""

    if len(prefix_texts) == 1:
        return prefix_texts[0]

    # Join with hyphens where needed
    result = prefix_texts[0]
    for i in range(1, len(prefix_texts)):
        current = prefix_texts[i]
        # If result ends with letter and current starts with digit, add hyphen
        if result and current:
            last_char = result[-1]
            first_char = current[0]
            if last_char.isalpha() and first_char.isdigit():
                result += "-"
        result += current

    return result


def _build_long_chain_prefix(length: int) -> str:
    """
    Build prefix for chains longer than those in CHAIN_PREFIXES.
    
    Uses IUPAC multiplicative system for very long chains.
    """
    # For now, just return the numerical form
    # TODO: Implement proper long-chain naming (triacontane, etc.)
    return f"{length}C"


def format_locants(locants: tuple) -> str:
    """
    Format a tuple of locants for name insertion.
    
    Example: (2, 3) -> "2,3-"
    """
    if not locants:
        return ""
    return ",".join(str(l) for l in sorted(locants)) + "-"


def get_multiplier(count: int, is_complex: bool = False) -> str:
    """
    Get the appropriate multiplier prefix for a count.
    
    Args:
        count: Number of occurrences
        is_complex: True if the substituent name is complex (has locants/hyphens)
        
    Returns:
        Multiplier string (e.g., "di", "tris")
    """
    if count <= 1:
        return ""
    
    multipliers = COMPLEX_MULTIPLIERS if is_complex else SIMPLE_MULTIPLIERS
    return multipliers.get(count, str(count))
