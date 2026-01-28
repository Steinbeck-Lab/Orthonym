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
from ..rules.locants import get_functional_group_locants
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


def _generate_chain_parent(features: Any) -> NameFragment:
    """Generate parent name for acyclic chains."""
    chain_length = len(features.principal_chain)
    
    # Get chain prefix
    if chain_length in CHAIN_PREFIXES:
        prefix = CHAIN_PREFIXES[chain_length]
    else:
        prefix = _build_long_chain_prefix(chain_length)
    
    # Determine unsaturation
    unsaturation = _get_unsaturation_suffix(features)
    
    return NameFragment(
        text=f"{prefix}{unsaturation}",
        fragment_type="parent"
    )


def _generate_ring_parent(features: Any) -> NameFragment:
    """Generate parent name for cyclic compounds."""
    # TODO: Implement ring naming (cycloalkanes, aromatics, heterocycles)
    # For now, return placeholder
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

    For alkane naming, this extracts alkyl substituents from features.substituents,
    groups them by name (methyl, ethyl, etc.), and formats with locants and
    multiplicative prefixes.
    """
    prefixes = []

    # --- Handle alkyl substituents from features.substituents ---
    if features.substituents and features.mol:
        alkyl_prefixes = _generate_alkyl_prefixes(features)
        prefixes.extend(alkyl_prefixes)

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

    # Get parent text
    parent = parent_frag.text if parent_frag else ""

    # Handle suffix attachment using PIN-style formatting
    if suffix_frag and suffix_frag.text:
        suffix_text = suffix_frag.text
        suffix_locants = list(suffix_frag.locants) if suffix_frag.locants else []

        # Determine multiplier for multiple functional groups
        count = len(suffix_locants)
        multiplier = get_multiplier_prefix(count, suffix_text) if count > 1 else ""

        # Parse parent into stem and unsaturation
        # Parent should be like "propan" (stem + "an") or "but" + "en" etc.
        # For now, we know it's stem+unsaturation (e.g., "propan", "butan")
        # We need to split into stem and unsaturation for format_suffix_with_locants
        stem, unsaturation = _split_parent_stem(parent)

        name = format_suffix_with_locants(
            stem,
            unsaturation,
            suffix_text,
            suffix_locants,
            multiplier
        )
    else:
        # No suffix = hydrocarbon, add 'e' ending
        name = f"{parent}e"

    # Add prefixes
    if prefix_str:
        name = f"{prefix_str}{name}"

    # Add stereodescriptors at the very start
    if stereo:
        name = f"{stereo}{name}"

    return name


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
