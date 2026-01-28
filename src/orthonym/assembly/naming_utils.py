"""
Name assembly utility functions for IUPAC name generation.

Pure functions for:
- Alkyl substituent naming (methyl through decyl)
- Substituent prefix formatting with locants and multipliers
- Alphabetization sort keys (IUPAC rules for prefix ordering)
- Complex substituent detection and multiplier selection
- Vowel elision (terminal 'e' removal before vowel suffixes)
- Suffix with locants formatting (PIN infix style)

All functions are pure: they accept locants as integers (not atom indices).
The mapping from atom indices to locants is handled elsewhere.
"""

from typing import List


# ============================================================================
# Alkyl Substituent Names
# ============================================================================

ALKYL_NAMES = {
    1: "methyl",
    2: "ethyl",
    3: "propyl",
    4: "butyl",
    5: "pentyl",
    6: "hexyl",
    7: "heptyl",
    8: "octyl",
    9: "nonyl",
    10: "decyl",
}


def get_alkyl_name(carbon_count: int) -> str:
    """Get alkyl substituent name for a given carbon count.

    Returns 'methyl' for 1, 'ethyl' for 2, ..., 'decyl' for 10.

    Args:
        carbon_count: Number of carbons in the alkyl chain (1-10).

    Returns:
        Alkyl substituent name string.

    Raises:
        ValueError: If carbon_count is not in supported range (1-10).
    """
    if carbon_count not in ALKYL_NAMES:
        raise ValueError(
            f"Unsupported carbon count {carbon_count}. "
            f"Supported range: 1-10."
        )
    return ALKYL_NAMES[carbon_count]


# ============================================================================
# Multiplier Prefixes
# ============================================================================

# Simple multiplicative prefixes (for simple substituent names)
SIMPLE_MULTIPLIERS = {
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
    7: "hepta",
    8: "octa",
    9: "nona",
    10: "deca",
}

# Complex multiplicative prefixes (for substituents with locants/hyphens)
COMPLEX_MULTIPLIERS = {
    2: "bis",
    3: "tris",
    4: "tetrakis",
    5: "pentakis",
    6: "hexakis",
    7: "heptakis",
    8: "octakis",
    9: "nonakis",
    10: "decakis",
}


def is_complex_substituent(name: str) -> bool:
    """Determine if a substituent name is complex.

    A substituent is considered complex if its name contains digits,
    hyphens, or is enclosed in parentheses. Complex substituents
    require bis/tris/tetrakis multipliers instead of di/tri/tetra,
    and are enclosed in parentheses in the final name.

    Args:
        name: The substituent name (e.g., 'methyl', '1-methylethyl').

    Returns:
        True if the substituent is complex, False otherwise.

    Examples:
        >>> is_complex_substituent("methyl")
        False
        >>> is_complex_substituent("ethyl")
        False
        >>> is_complex_substituent("1-methylethyl")
        True
        >>> is_complex_substituent("2-propyl")
        True
    """
    # Check for digits (indicates locants within the substituent name)
    if any(ch.isdigit() for ch in name):
        return True
    # Check for hyphens (indicates compound substituent)
    if "-" in name:
        return True
    return False


def get_multiplier_prefix(count: int, substituent_name: str) -> str:
    """Get the appropriate multiplier prefix for a count of substituents.

    For count=1, returns empty string (no multiplier needed).
    For simple substituent names (no digits/hyphens), uses di/tri/tetra.
    For complex substituent names, uses bis/tris/tetrakis.

    Args:
        count: Number of identical substituents.
        substituent_name: The substituent name to determine simple vs complex.

    Returns:
        Multiplier prefix string, or empty string for count=1.

    Examples:
        >>> get_multiplier_prefix(1, "methyl")
        ''
        >>> get_multiplier_prefix(2, "methyl")
        'di'
        >>> get_multiplier_prefix(3, "methyl")
        'tri'
        >>> get_multiplier_prefix(2, "1-methylethyl")
        'bis'
        >>> get_multiplier_prefix(3, "1-methylethyl")
        'tris'
    """
    if count <= 1:
        return ""

    if is_complex_substituent(substituent_name):
        return COMPLEX_MULTIPLIERS.get(count, str(count))
    else:
        return SIMPLE_MULTIPLIERS.get(count, str(count))


# ============================================================================
# Substituent Prefix Formatting
# ============================================================================

def format_substituent_prefix(name: str, locants: List[int], count: int) -> str:
    """Format a substituent with locants and multiplier prefix.

    Produces a formatted substituent prefix ready for insertion into an
    IUPAC name. Handles simple and complex substituents differently:
    - Simple: locants + multiplier + name (e.g., '2,2-dimethyl')
    - Complex: locants + multiplier + (name) (e.g., '3,5-bis(1-methylethyl)')

    Args:
        name: Base substituent name (e.g., 'methyl', '1-methylethyl').
        locants: List of locant positions for this substituent.
        count: Number of identical substituents.

    Returns:
        Formatted prefix string.

    Examples:
        >>> format_substituent_prefix("methyl", [2], 1)
        '2-methyl'
        >>> format_substituent_prefix("methyl", [2, 2], 2)
        '2,2-dimethyl'
        >>> format_substituent_prefix("ethyl", [3, 5], 2)
        '3,5-diethyl'
        >>> format_substituent_prefix("1-methylethyl", [4, 7], 2)
        '4,7-bis(1-methylethyl)'
    """
    # Format locants as comma-separated string
    locant_str = ",".join(str(loc) for loc in locants)

    # Get the multiplier prefix
    multiplier = get_multiplier_prefix(count, name)

    # Complex substituents get parentheses around the name
    complex = is_complex_substituent(name)
    if complex and count > 1:
        formatted_name = f"({name})"
    else:
        formatted_name = name

    # Assemble: locants-multiplier+name
    return f"{locant_str}-{multiplier}{formatted_name}"


# ============================================================================
# Alphabetization Sort Key
# ============================================================================

# Prefixes to IGNORE for IUPAC alphabetization
# These are multiplicative prefixes and sec-/tert- detachable prefixes
IGNORE_FOR_ALPHA = {
    "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "nona", "deca",
    "undeca", "dodeca",
    "bis", "tris", "tetrakis", "pentakis", "hexakis", "heptakis",
    "octakis", "nonakis", "decakis",
    "sec", "tert",
}

# Note: iso-, neo-, cyclo- are INCLUDED in alphabetization
# (they are non-detachable prefixes that affect alphabetical order)


def alpha_sort_key(substituent_name: str) -> str:
    """Generate an alphabetization sort key for IUPAC prefix ordering.

    According to IUPAC 2013 rules:
    - Multiplicative prefixes (di-, tri-, tetra-, bis-, tris-, tetrakis-)
      are IGNORED for alphabetization.
    - sec- and tert- are IGNORED for alphabetization (treated as detachable).
    - iso-, neo-, cyclo- are INCLUDED (non-detachable, affect sort order).

    The input is the base substituent name. The function strips any
    leading multiplicative prefix or sec-/tert- to produce a sort key.

    Args:
        substituent_name: The substituent name (e.g., 'methyl', 'dimethyl',
                         'isopropyl', 'tert-butyl').

    Returns:
        Lowercase sort key string.

    Examples:
        >>> alpha_sort_key("methyl")
        'methyl'
        >>> alpha_sort_key("dimethyl")
        'methyl'
        >>> alpha_sort_key("triethyl")
        'ethyl'
        >>> alpha_sort_key("isopropyl")
        'isopropyl'
        >>> alpha_sort_key("neopentyl")
        'neopentyl'
        >>> alpha_sort_key("cyclopropyl")
        'cyclopropyl'
        >>> alpha_sort_key("tert-butyl")
        'butyl'
    """
    text = substituent_name.lower()

    # Handle hyphenated detachable prefixes: sec- and tert-
    for prefix in ("sec-", "tert-"):
        if text.startswith(prefix):
            return text[len(prefix):]

    # Handle non-hyphenated multiplicative prefixes
    # Sort by longest prefix first to avoid partial matches
    # (e.g., 'tetra' before 'tri', 'tetrakis' before 'tetra')
    sorted_prefixes = sorted(IGNORE_FOR_ALPHA - {"sec", "tert"},
                             key=len, reverse=True)
    for prefix in sorted_prefixes:
        if text.startswith(prefix):
            remainder = text[len(prefix):]
            # Only strip if there is a remainder (avoid stripping entire word)
            if remainder:
                return remainder

    return text


# ============================================================================
# Vowel Elision
# ============================================================================

# Vowels that trigger elision of terminal 'e' in the parent stem
_ELISION_VOWELS = frozenset("aiouy")
# Note: 'e' is excluded from triggering elision per IUPAC convention;
# terminal 'e' is only elided before a, i, o, u, y


def apply_vowel_elision(parent_stem: str, suffix: str) -> str:
    """Apply IUPAC vowel elision rules when joining parent stem and suffix.

    The terminal 'e' in a parent stem is elided (removed) when the suffix
    begins with 'a', 'i', 'o', 'u', or 'y'. The 'e' is NOT elided before
    consonants or before another 'e'.

    Args:
        parent_stem: Parent name stem (e.g., 'propane', 'butane', 'propan').
        suffix: Suffix to append (e.g., 'ol', 'al', 'one', 'amine', 'diol').

    Returns:
        Combined string with elision applied if appropriate.

    Examples:
        >>> apply_vowel_elision("propane", "ol")
        'propanol'
        >>> apply_vowel_elision("propane", "al")
        'propanal'
        >>> apply_vowel_elision("propane", "one")
        'propanone'
        >>> apply_vowel_elision("propane", "amine")
        'propanamine'
        >>> apply_vowel_elision("butane", "diol")
        'butanediol'
        >>> apply_vowel_elision("ethane", "oic acid")
        'ethanoic acid'
        >>> apply_vowel_elision("propan", "ol")
        'propanol'
    """
    if not parent_stem or not suffix:
        return parent_stem + suffix

    # Check if parent stem ends in 'e' and suffix starts with a vowel
    # that triggers elision
    if parent_stem[-1] == "e" and suffix[0] in _ELISION_VOWELS:
        return parent_stem[:-1] + suffix

    return parent_stem + suffix


# ============================================================================
# Suffix with Locants Formatting (PIN Infix Style)
# ============================================================================

def format_suffix_with_locants(
    parent_stem: str,
    unsaturation: str,
    suffix: str,
    suffix_locants: List[int],
    suffix_multiplier: str = "",
) -> str:
    """Assemble a parent name with infix locants in IUPAC 2013 PIN style.

    Constructs the parent name by combining stem + unsaturation infix,
    then attaching the suffix with locants using hyphens. Applies vowel
    elision rules where appropriate.

    Args:
        parent_stem: The chain prefix (e.g., 'prop', 'but', 'pent').
        unsaturation: Unsaturation infix (e.g., 'an', 'en', 'yn', '').
        suffix: The functional group suffix (e.g., 'ol', 'one', 'oic acid', 'al').
        suffix_locants: Locant positions for the suffix group(s).
        suffix_multiplier: Multiplier for multiple suffix groups (e.g., 'di', 'tri').

    Returns:
        Assembled parent name string.

    Examples:
        >>> format_suffix_with_locants("prop", "an", "ol", [1])
        'propan-1-ol'
        >>> format_suffix_with_locants("but", "an", "one", [2])
        'butan-2-one'
        >>> format_suffix_with_locants("prop", "an", "ol", [1, 2], "di")
        'propane-1,2-diol'
        >>> format_suffix_with_locants("pent", "an", "oic acid", [])
        'pentanoic acid'
        >>> format_suffix_with_locants("prop", "an", "al", [])
        'propanal'
    """
    # Build the base: stem + unsaturation (e.g., 'propan', 'buten')
    base = parent_stem + unsaturation

    if suffix_locants:
        # Format locants as comma-separated
        locant_str = ",".join(str(loc) for loc in suffix_locants)

        # Build the suffix part: multiplier + suffix (e.g., 'diol', 'ol')
        full_suffix = suffix_multiplier + suffix

        # When there's a multiplier (e.g., 'di'), the base keeps terminal 'e'
        # because 'diol' starts with 'd' (consonant), so no elision.
        # When there's no multiplier and suffix starts with vowel, apply elision.
        if suffix_multiplier:
            # With multiplier: base + 'e' + '-locants-' + multiplier + suffix
            # e.g., 'propane-1,2-diol'
            # The 'e' is added because the next character is '-' (or the
            # multiplier starts with a consonant like 'd')
            base_with_e = apply_vowel_elision(base + "e", full_suffix)
            # But we need to insert locants between base and suffix
            # So: base+'e' + '-locants-' + full_suffix
            return f"{base}e-{locant_str}-{full_suffix}"
        else:
            # Without multiplier: base + '-locants-' + suffix
            # e.g., 'propan-1-ol'
            # Apply elision to base before the hyphen-locant construct
            # In PIN style, the base does NOT get 'e' appended when
            # locants follow directly
            return f"{base}-{locant_str}-{suffix}"
    else:
        # No locants: just combine base and suffix with elision
        # e.g., 'pentanoic acid', 'propanal'
        return apply_vowel_elision(base + "e", suffix)
