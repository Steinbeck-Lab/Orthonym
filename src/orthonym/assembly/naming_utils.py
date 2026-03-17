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

import re
from typing import List


# ============================================================================
# Pre-compiled patterns and constants (hoisted from function bodies)
# ============================================================================

# Used by is_complex_substituent() — compound multiplier+substituent patterns
_MULT_SUBSTITUENT_RE = re.compile(
    r'^(?:di|tri|tetra|penta|hexa)'
    r'(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|'
    r'phenyl|naphthyl|cyclopentyl|cyclohexyl|benzyl|vinyl|allyl|'
    r'fluoro|chloro|bromo|iodo)'
)

# Halogen + alkyl compound substituent patterns (e.g., fluoromethyl,
# trifluoromethyl, chloroethyl). These are compound substituents per
# IUPAC P-31.1.2.3 and require enclosing marks.
_HALOALKYL_RE = re.compile(
    r'^(?:(?:di|tri|tetra|penta|hexa)?(?:fluoro|chloro|bromo|iodo))'
    r'(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl|'
    r'undecyl|dodecyl|tridecyl|tetradecyl|pentadecyl|'
    r'hexadecyl|heptadecyl|octadecyl|nonadecyl|icosyl)$'
)

# Alkyl + functional group compound substituent patterns (e.g., methylamino,
# ethylamino, propylamino). These are compound substituents per IUPAC P-31.1.2
# and require enclosing marks.
_ALKYLAMINO_RE = re.compile(
    r'^(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|'
    r'nonyl|decyl|undecyl|dodecyl|tridecyl|tetradecyl|pentadecyl|'
    r'hexadecyl|heptadecyl|octadecyl|nonadecyl|icosyl|'
    r'phenyl|benzyl|cyclopentyl|cyclohexyl)'
    r'(?:amino|imino)$'
)

_COMPOUND_OXY_PREFIXES = frozenset((
    'sulfooxy', 'sulfonyloxy', 'phosphonooxy', 'phosphonatoxy', 'carbonyloxy',
))

# Shared C1-C20 alkyl roots used by needs_brackets(), is_complex_substituent(),
# and regex patterns. Extends coverage beyond the original C1-C10 lists.
_ALKYL_ROOTS_FULL = (
    'methyl', 'ethyl', 'propyl', 'butyl', 'pentyl',
    'hexyl', 'heptyl', 'octyl', 'nonyl', 'decyl',
    'undecyl', 'dodecyl', 'tridecyl', 'tetradecyl', 'pentadecyl',
    'hexadecyl', 'heptadecyl', 'octadecyl', 'nonadecyl', 'icosyl',
)

_COMPOUND_S_SUFFIXES_COMPLEX = ('sulfinyl', 'sulfonyl', 'sulfanyl')

# Functional group prefixes that, when fused with alkyl roots, form compound
# substituents requiring enclosing marks per IUPAC P-14.5.2.
# Hoisted to module level for performance (was recreated inside needs_brackets).
_COMPOUND_FG_PREFIXES = (
    'hydroxy', 'carboxy', 'amino', 'oxo', 'formyl', 'cyano',
    'nitro', 'mercapto', 'sulfanyl', 'phospho',
    'fluoro', 'chloro', 'bromo', 'iodo',
    'difluoro', 'trifluoro', 'dichloro', 'trichloro',
    'dibromo', 'tribromo',
)

# Used by alpha_sort_key() — pre-compiled regex patterns
_LOCANT_PREFIX_RE = re.compile(r'^[\d,]+-')
_N_LOCANT_PREFIX_RE = re.compile(r'^[nN],?[nN]?-')


# ============================================================================
# Terminal Functional Group Types (IUPAC P-14.3.4.1)
# ============================================================================

# Terminal functional groups that always occupy position 1 by definition.
# The chain is numbered FROM the terminal group, so locant-1 is implicit.
# Canonical source: IUPAC P-14.3.4.1. Used by should_omit_locant_one().
TERMINAL_FG_TYPES = frozenset({
    "carboxylic_acid",  # Always at chain end (locant 1)
    "aldehyde",         # Always at chain end (locant 1)
    "nitrile",          # Always at chain end (locant 1)
    "primary_amide",    # Always at chain end (locant 1)
    "secondary_amide",  # Always at chain end (locant 1)
    "tertiary_amide",   # Always at chain end (locant 1)
    "acid_chloride",    # Always at chain end (locant 1)
    "acid_bromide",     # Always at chain end (locant 1)
    "acid_fluoride",    # Always at chain end (locant 1)
    "thioic_S_acid",    # Always at chain end (locant 1)
    "thioic_O_acid",    # Always at chain end (locant 1)
    "dithioic_acid",    # Always at chain end (locant 1)
    "carbamic_acid",    # Retained name, terminal (locant 1)
})


# ============================================================================
# Centralized Locant-1 Elision (IUPAC P-14.3.4)
# ============================================================================


def should_omit_locant_one(
    *,
    context: str,
    chain_length: int = 0,
    is_ring: bool = False,
    is_heterocyclic: bool = False,
    is_monosubstituted: bool = False,
    fg_type: str = "",
) -> bool:
    """Determine whether locant-1 should be omitted per IUPAC P-14.3.4.

    Centralized decision point for ALL locant-1 elision in the pipeline.
    Every call site that decides whether to omit locant-1 must use this
    function rather than reimplementing the logic inline.

    Args:
        context: One of "suffix", "prefix", or "bond".
        chain_length: Length of the parent chain (0 for rings).
        is_ring: True if the parent is a ring system.
        is_heterocyclic: True if the ring contains heteroatoms.
        is_monosubstituted: True if only one substituent/FG is present.
        fg_type: Functional group type string (e.g., "carboxylic_acid").

    Returns:
        True if locant-1 should be omitted from the name.
    """
    # Rule 1: Methane derivatives (chain_length=1): always omit
    # Only position exists, so locant is always trivially 1.
    if chain_length == 1:
        return True

    # Rule 2: 2-carbon bond locants (ethene/ethyne): omit bond locant
    # Only one possible position for the double/triple bond.
    if context == "bond" and chain_length == 2:
        return True

    # Rule 3: Terminal groups: suffix locant-1 is implicit
    # Chain is numbered from the terminal group (acid, aldehyde, nitrile, etc.)
    if context == "suffix" and fg_type in TERMINAL_FG_TYPES:
        return True

    # Rule 4: Monosubstituted rings
    if context == "prefix" and is_ring and is_monosubstituted:
        if is_heterocyclic:
            return False  # Position matters in heterocycles
        return True  # Symmetric carbocyclic: omit

    # Rule 5: Monosubstituted hydrocarbon chain at position 1
    if context == "prefix" and not is_ring and is_monosubstituted and chain_length > 1:
        return True

    # Rule 6: Mono-cycloalkene bond locant
    # Single double bond in a ring: locant omitted (cyclohexene, not cyclohex-1-ene)
    if context == "bond" and is_ring and is_monosubstituted:
        return True

    return False


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
    For carbon counts > 10, delegates to the centralized chain_names module
    which supports up to 999 carbons using IUPAC compositional naming.

    Args:
        carbon_count: Number of carbons in the alkyl chain (1-999).

    Returns:
        Alkyl substituent name string.

    Raises:
        ValueError: If carbon_count is outside supported range.
    """
    if carbon_count in ALKYL_NAMES:
        return ALKYL_NAMES[carbon_count]
    # Delegate to centralized module for chains > 10
    from ..data.chain_names import get_alkyl_name as _get_alkyl_name
    return _get_alkyl_name(carbon_count)


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
    11: "undeca",
    12: "dodeca",
    13: "trideca",
    14: "tetradeca",
    15: "pentadeca",
    16: "hexadeca",
    17: "heptadeca",
    18: "octadeca",
    19: "nonadeca",
    20: "icosa",
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
    11: "undecakis",
    12: "dodecakis",
    13: "tridecakis",
    14: "tetradecakis",
    15: "pentadecakis",
    16: "hexadecakis",
    17: "heptadecakis",
    18: "octadecakis",
    19: "nonadecakis",
    20: "icosakis",
}


def needs_brackets(name: str) -> bool:
    """Determine if a substituent name is a compound substituent needing parentheses.

    Per IUPAC P-14.5.2, compound substituents (those that contain locants,
    hyphens, or functional group prefixes fused with alkyl names) must be
    enclosed in parentheses when used as prefixes on ring parents.

    Simple substituents (single-word names like methyl, chloro, hydroxy)
    do NOT need parentheses.

    Already-bracketed names (starting with '(' or '[') are left alone.

    Args:
        name: The substituent name (e.g., 'methyl', 'hydroxymethyl',
              '2-methylpropyl', '(N,N-dimethylamino)').

    Returns:
        True if the substituent needs enclosing parentheses, False otherwise.

    Examples:
        >>> needs_brackets("methyl")
        False
        >>> needs_brackets("chloro")
        False
        >>> needs_brackets("hydroxy")
        False
        >>> needs_brackets("hydroxymethyl")
        True
        >>> needs_brackets("carboxymethyl")
        True
        >>> needs_brackets("aminoethyl")
        True
        >>> needs_brackets("2-methylpropyl")
        True
        >>> needs_brackets("(N,N-dimethylamino)")
        False
        >>> needs_brackets("methoxy")
        False
    """
    if not name:
        return False

    # Already wrapped in parentheses or square brackets -- skip
    if (name.startswith('(') and name.endswith(')')) or \
       (name.startswith('[') and name.endswith(']')):
        return False

    # Contains a digit (has locants): definitely compound
    if any(ch.isdigit() for ch in name):
        return True

    # Contains a hyphen (compound substituent): definitely compound
    if '-' in name:
        return True

    # Functional group prefixes fused with alkyl names are compound substituents.
    # Examples: hydroxymethyl, carboxymethyl, aminoethyl, oxoethyl, formylmethyl
    # But NOT: methoxy, ethoxy (these are simple ether prefixes, single concept)
    # Uses module-level _COMPOUND_FG_PREFIXES and _ALKYL_ROOTS_FULL (C1-C20).
    name_lower = name.lower()
    for fg in _COMPOUND_FG_PREFIXES:
        if name_lower.startswith(fg):
            remainder = name_lower[len(fg):]
            # Check if the remainder is an alkyl root
            for alkyl in _ALKYL_ROOTS_FULL:
                if remainder == alkyl:
                    return True

    # Compound sulfur/selenium/tellurium prefixes: alkyl + sulfinyl/sulfonyl/sulfanyl
    # Per IUPAC P-16.3.3, "methylsulfinyl" = methyl + sulfinyl = compound substituent
    # requiring parentheses: "2-(methylsulfinyl)ethanoic acid"
    # But NOT bare "sulfinyl", "sulfonyl", "sulfanyl" (simple, no alkyl prefix)
    for alkyl in _ALKYL_ROOTS_FULL:
        if name_lower.startswith(alkyl):
            remainder = name_lower[len(alkyl):]
            for s_suffix in _COMPOUND_S_SUFFIXES_COMPLEX:
                if remainder == s_suffix:
                    return True

    return False


# ============================================================================
# IUPAC Enclosing Marks (P-16.3.3)
# ============================================================================


def get_bracket_depth(name: str) -> int:
    """Determine the current bracket nesting depth of a name.

    Returns 0 if no brackets, 1 if contains (), 2 if contains [], etc.
    Used to determine what enclosing marks to use at the next level.

    Args:
        name: A substituent or compound name.

    Returns:
        Integer nesting depth (0-3).

    Examples:
        >>> get_bracket_depth("methyl")
        0
        >>> get_bracket_depth("2-methylpropyl")
        0
        >>> get_bracket_depth("(2-methylpropyl)")
        1
        >>> get_bracket_depth("2-[(1-methylethyl)]propyl")
        2
    """
    if '{' in name:
        return 3
    if '[' in name:
        return 2
    if '(' in name:
        return 1
    return 0


def apply_enclosing_marks(name: str, depth: int = 0) -> str:
    """Apply IUPAC P-16.3.3 enclosing marks at the correct nesting depth.

    Nesting order: ( ) -> [ ] -> { } -> ( ) again
    Depth 0: parentheses
    Depth 1: square brackets (name already contains parentheses)
    Depth 2: braces (name already contains brackets)

    Args:
        name: The compound substituent name (without outer brackets).
        depth: Nesting depth (0 = outermost).

    Returns:
        Name enclosed in the appropriate bracket type.

    Examples:
        >>> apply_enclosing_marks("2-methylpropyl", 0)
        '(2-methylpropyl)'
        >>> apply_enclosing_marks("2-methylpropyl", 1)
        '[2-methylpropyl]'
        >>> apply_enclosing_marks("2-methylpropyl", 2)
        '{2-methylpropyl}'
    """
    MARKS = [('(', ')'), ('[', ']'), ('{', '}')]
    open_mark, close_mark = MARKS[depth % 3]
    return f"{open_mark}{name}{close_mark}"


def is_complex_substituent(name: str) -> bool:
    """Determine if a substituent name is complex.

    A substituent is considered complex if its name contains digits,
    hyphens, is enclosed in parentheses, or contains embedded substituent
    multiplier prefixes (e.g., diphenyl, trimethyl within a compound name).
    Complex substituents require bis/tris/tetrakis multipliers instead of
    di/tri/tetra, and are enclosed in parentheses in the final name.

    Note: Modification prefixes like "tetrahydro-" or "dihydro-" are NOT
    multipliers and do NOT make a name complex.

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
        >>> is_complex_substituent("diphenylphosphanyl")
        True
        >>> is_complex_substituent("tetrahydropyranyl")
        False
    """
    # Check for digits (indicates locants within the substituent name)
    if any(ch.isdigit() for ch in name):
        return True
    # Check for hyphens (indicates compound substituent)
    if "-" in name:
        return True
    # Check for embedded multiplier + substituent name patterns (IUPAC P-14.5.2)
    if _MULT_SUBSTITUENT_RE.match(name):
        return True
    # Compound oxy-prefixes require brackets (OPSIN/IUPAC parenthesization)
    if name in _COMPOUND_OXY_PREFIXES:
        return True
    # Compound sulfur prefixes per IUPAC P-16.3.3
    name_lower = name.lower()
    for alkyl in _ALKYL_ROOTS_FULL:
        if name_lower.startswith(alkyl):
            remainder = name_lower[len(alkyl):]
            for s_suffix in _COMPOUND_S_SUFFIXES_COMPLEX:
                if remainder == s_suffix:
                    return True
    # Acyloxy compound prefixes per IUPAC P-16.3.3:
    # "acetyloxy", "benzoyloxy", "propanoyloxy" etc. are compound prefixes
    # (acyl + oxy) that require complex multipliers (bis/tris) and parenthesization.
    if name.endswith('yloxy') and len(name) > 5:
        # Matches: acetyloxy, benzoyloxy, propanoyloxy, butanoyloxy, etc.
        # Does NOT match: methoxy, ethoxy (simple alkoxy, no 'yl' before 'oxy')
        return True
    # Haloalkyl compound substituents per IUPAC P-31.1.2.3:
    # "fluoromethyl", "trifluoromethyl", "chloroethyl" etc.
    if _HALOALKYL_RE.match(name):
        return True
    # Alkyl+amino compound substituents per IUPAC P-31.1.2:
    # "methylamino", "ethylamino", "phenylamino" etc.
    if _ALKYLAMINO_RE.match(name):
        return True
    # Acylamino compound substituents: "ethanoylamino", "propanoylamino" etc.
    if name.endswith('amino') and 'oyl' in name:
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
        if count in COMPLEX_MULTIPLIERS:
            return COMPLEX_MULTIPLIERS[count]
        # For counts > 20, build compositional multiplier using chain_names
        from ..data.chain_names import get_chain_prefix
        prefix = get_chain_prefix(count)
        # Complex multipliers use "-akis" suffix (e.g., "henicosakis")
        if not prefix.endswith("a"):
            prefix += "a"
        return prefix + "kis"
    else:
        if count in SIMPLE_MULTIPLIERS:
            return SIMPLE_MULTIPLIERS[count]
        # For counts > 20, build compositional multiplier using chain_names
        from ..data.chain_names import get_chain_prefix
        prefix = get_chain_prefix(count)
        # Simple multipliers use trailing 'a' (e.g., "henicosa", "docosa")
        if not prefix.endswith("a"):
            prefix += "a"
        return prefix


# ============================================================================
# Substituent Prefix Formatting
# ============================================================================


def _has_stereo_prefix(name: str) -> bool:
    """Check if a substituent name starts with a CIP stereo descriptor prefix.

    Detects patterns like "(R)-", "(S)-", "(1R)-", "(2S,3R)-", "(11z,14z)-"
    at the start. This distinguishes stereo-prefixed names from compound
    substituent names that happen to start with parentheses (e.g.,
    "(2-methylphenyl)").

    A stereo prefix is: '(' + optional digits/comma + single letter R/S/E/Z + ')' + '-'
    Case-insensitive for E/Z since generated names may use lowercase e/z.
    """
    if not name.startswith('('):
        return False
    # Match: (R)-, (S)-, (1R)-, (2S,3R)-, (11z,14z)- etc.
    return bool(re.match(r'^\((?:\d+[RSEZrsez],)*\d*[RSEZrsez]\)-', name))


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

    # Complex substituents get parentheses around the name.
    # Per IUPAC P-14.5.2, compound substituent names containing numeric locants
    # need enclosing marks to avoid ambiguity:
    # - count > 1: always wrap for bis/tris multiplier (e.g., "3,5-bis(2-methylpropyl)")
    # - count == 1: wrap if name has bare digits at the start (e.g., "3-(2-methylpropyl)")
    #   but NOT for names that already have their own parenthesization (e.g., "(oxan-2-yl)oxy")
    #   and NOT for names that are just hyphenated (e.g., "3-sec-butyl" is fine)
    complex = is_complex_substituent(name)
    if _has_stereo_prefix(name):
        # Name has a CIP stereo descriptor prefix (e.g., "(R)-sec-butyl"):
        # use square brackets per IUPAC P-16.3.3 nesting rules
        formatted_name = f"[{name}]"
    elif complex and not name.startswith('(') and not name.startswith('['):
        # Complex name without existing enclosing marks: wrap in parentheses
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

# Pre-sorted prefix list for alpha_sort_key() — longest first to avoid partial matches
_SORTED_ALPHA_PREFIXES = sorted(IGNORE_FOR_ALPHA - {"sec", "tert"},
                                key=len, reverse=True)


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

    # Strip enclosing parentheses if present
    # e.g., "(N,N-dimethylamino)" -> "N,N-dimethylamino"
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]

    # Strip leading locants (digits and commas followed by hyphen)
    # e.g., "3-methyl" -> "methyl", "2,2-dimethyl" -> "dimethyl"
    # This handles formatted prefix strings that include locants
    text = _LOCANT_PREFIX_RE.sub('', text)

    # Strip enclosing parentheses again after locant removal
    # e.g., "4-(methylsulfinyl)" -> "(methylsulfinyl)" after locant strip -> "methylsulfinyl"
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]

    # Strip N-locant prefixes (N- or N,N-) for alphabetization
    # e.g., "N,N-dimethylamino" -> "dimethylamino" -> "amino" (after multi-prefix strip)
    text = _N_LOCANT_PREFIX_RE.sub('', text)

    # Handle hyphenated detachable prefixes: sec- and tert-
    for prefix in ("sec-", "tert-"):
        if text.startswith(prefix):
            return text[len(prefix):]

    # Handle non-hyphenated multiplicative prefixes
    # Sort by longest prefix first to avoid partial matches
    # (e.g., 'tetra' before 'tri', 'tetrakis' before 'tetra')
    # Note: This correctly strips 'tri' from 'trioxo' -> 'oxo' per IUPAC P-14.4.
    # The result is used ONLY as a sort key (in key= parameter of sorted/sort),
    # never for text reconstruction. All callers confirmed to use key= only.
    for prefix in _SORTED_ALPHA_PREFIXES:
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
        # No locants: combine base and suffix with elision
        # e.g., 'pentanoic acid', 'propanal'
        if suffix_multiplier:
            # Multiple terminal groups without locants (e.g., diacids):
            # 'butanedioic acid', 'pentanedioic acid'
            full_suffix = suffix_multiplier + suffix
            return apply_vowel_elision(base + "e", full_suffix)
        return apply_vowel_elision(base + "e", suffix)
