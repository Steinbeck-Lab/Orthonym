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

# BUG-B guard: FG types that substituent naming demonstrably handles on 1-3C branches.
# Each entry verified empirically (Phase 105-01 + Phase 113-01) to produce correct
# prefix via name_substituent() on small branches in BOTH polyfunctional.py and
# composer.py code paths. FG types NOT in this set stay in the polyfunctional
# prefix list (IUPAC P-59.1).
#
# NOTE: thiol, nitro, azido, secondary_amine, tertiary_amine were previously in
# composer.py's inline set but NOT in polyfunctional.py's set. Testing showed that
# the polyfunctional path's substituent naming does NOT reliably handle these FG
# types on small branches (drops sulfanyl, nitro, amino, azido prefixes).
# These are excluded from the unified set to prevent silent FG drops.
BRANCH_HANDLED_FGS: frozenset = frozenset({
    'primary_alcohol',    # -> "hydroxymethyl", "2-hydroxypropyl"
    'secondary_alcohol',  # -> "hydroxy" included in branch name
    'primary_amine',      # -> "aminomethyl", "2-aminoethyl"
    'fluoro', 'chloro', 'bromo', 'iodo',  # -> "fluoromethyl" etc.
})

# Shared C1-C20 alkyl roots used by needs_brackets(), is_complex_substituent(),
# and regex patterns. Extends coverage beyond the original C1-C10 lists.
_ALKYL_ROOTS_FULL = (
    'methyl', 'ethyl', 'propyl', 'butyl', 'pentyl',
    'hexyl', 'heptyl', 'octyl', 'nonyl', 'decyl',
    'undecyl', 'dodecyl', 'tridecyl', 'tetradecyl', 'pentadecyl',
    'hexadecyl', 'heptadecyl', 'octadecyl', 'nonadecyl', 'icosyl',
)

# Chalcogen compound-substituent suffixes that, fused to an alkyl root, form a
# COMPLEX (compound) substituent requiring enclosing marks per IUPAC P-16.3.3 /
# P-63.6 — '(methylsulfanyl)', '(methylselanyl)', '(methyltellanyl)'. Phase 171
# BBR-ASM (DEF-8): the Se/Te analogues (selanyl/tellanyl + the oxidized
# seleninyl/selenonyl/tellurinyl/telluronyl) were missing, so 'methylselanyl' was
# wrongly classed simple -> '1-methylselanylpropane' instead of PIN
# '1-(methylselanyl)propane'.
_COMPOUND_S_SUFFIXES_COMPLEX = (
    'sulfinyl', 'sulfonyl', 'sulfanyl',
    'seleninyl', 'selenonyl', 'selanyl',
    'tellurinyl', 'telluronyl', 'tellanyl',
)

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
# Leading indicated-hydrogen descriptor (e.g. '1H-', '2H-'). Like a locant, the
# italic indicated H is IGNORED for alphabetization (IUPAC P-14.5.2), so
# '(1H-imidazol-5-yl)' must sort at 'i', not at the leading digit '1'.
_INDICATED_H_PREFIX_RE = re.compile(r'^\d+[hH]-')


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
    # Phase 163 Tier FRN-A chalcogen acids (P-66.6.3 functional replacement; IUPAC PIN
    # 'propaneselenoic Se-acid' / 'propanetelluroic Te-acid' — never carries locant-1)
    "selenoic_Se_acid", "selenoic_O_acid", "diselenoic_acid",
    "telluroic_Te_acid", "telluroic_O_acid", "ditelluroic_acid",
    # Phase 163 Tier FRN-B chalcogen amides (P-66.1.4.1.1 functional replacement; IUPAC PIN
    # 'propanethioamide' / 'propaneselenoamide' — never carries locant-1)
    "thioamide", "selenoamide", "telluroamide",
    # Phase 163 Tier FRN-C chalcogen aldehydes (P-66.6.3 — "propanethial" / "propaneselenal" /
    # "propanetellural"; suffix is "-thial"/"-selenal"/"-tellural" per seniority.py SUFFIX_FORMS,
    # but FG identifier is the long form. Always terminal — never carries locant-1.)
    "thioaldehyde", "selenoaldehyde", "telluroaldehyde",
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

    # Rule 5: Monosubstituted hydrocarbon chain at position 1.
    # P-14.3.4 (PIN): the locant '1' is omitted ONLY where the position is
    # unambiguous — methane (Rule 1, chain_length==1) and ETHANE (chain_length==2,
    # the two carbons are equivalent so there is a single monosubstitution product
    # -> 'chloroethane'). For propane and longer the terminal substituent is
    # distinguishable from interior positions ('1-chloropropane' != '2-chloropropane',
    # '1-chloropentane' is the PIN), so the locant MUST be cited. Phase 171 DEF-4
    # (BlueBookV2 P-14.3.4 @2869; gold 'ClCCCCC' -> '1-chloropentane'). Previously this
    # used `chain_length > 1`, which wrongly elided the locant for all chains.
    if context == "prefix" and not is_ring and is_monosubstituted and chain_length == 2:
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


# Pre-compiled patterns for compute_nesting_depth (P-16.5.4.1)
# P-16.5.4.1.1: Indicated hydrogen -- (1H), (3H), (9aH) etc.
_INDICATED_H_RE = re.compile(r'\(\d+[a-z]?H\)')
# P-16.5.4.1.2: Fusion/spiro/von Baeyer brackets -- [2,3-b], [4.5], [2.2.1]
_FUSION_BRACKET_RE = re.compile(r'\[[0-9a-z,.\-]+\]')
# P-16.5.4.1.3: Stereo descriptors -- (R), (S), (E), (Z), (1R,2S), etc.
_STEREO_PAREN_RE = re.compile(r'\((?:\d+[a-z]?,)*[RSEZ](?:,\d+[a-z]?[RSEZ]?)*\)')


def compute_nesting_depth(name: str) -> int:
    """Compute effective bracket nesting depth per P-16.5.4.1 (Dec 2025).

    Analyzes a name string and returns the effective nesting depth by
    counting only nesting-relevant brackets, per the following subsections:

    - P-16.5.4.1.1: Ignore indicated hydrogen parentheses, e.g., (1H), (3H)
    - P-16.5.4.1.2: Ignore fusion/spiro/ring assembly/von Baeyer brackets
    - P-16.5.4.1.3: Count stereo descriptor and compound locant parentheses
    - P-16.5.4.1.4: Escalate if consecutive same-level marks would result
    - P-16.5.4.1.5: Isotopic labeling convention (not applicable -- not implemented)

    Args:
        name: The chemical name string to analyze.

    Returns:
        Effective nesting depth (0 = no relevant brackets, 1 = has relevant
        parentheses, 2 = has relevant square brackets, etc.).
    """
    # Start with the original name and strip out non-nesting brackets
    # by replacing them with placeholder text that contains no brackets.
    working = name

    # Remove indicated hydrogen parentheses (P-16.5.4.1.1)
    working = _INDICATED_H_RE.sub('__IH__', working)

    # Remove fusion/spiro/von Baeyer square brackets (P-16.5.4.1.2)
    working = _FUSION_BRACKET_RE.sub('__FB__', working)

    # Now count remaining bracket types to determine depth
    max_depth = 0
    if '(' in working:
        max_depth = 1
    if '[' in working:
        max_depth = 2
    if '{' in working:
        max_depth = 3

    return max_depth


def apply_enclosing_marks(name: str, depth: int = 0) -> str:
    """Apply IUPAC P-16.3.3 enclosing marks at the correct nesting depth.

    Nesting order: ( ) -> [ ] -> { } -> ( ) again
    Depth 0: parentheses
    Depth 1: square brackets (name already contains parentheses)
    Depth 2: braces (name already contains brackets)

    When depth=-1 (sentinel for auto-detect), calls compute_nesting_depth()
    on the input name to determine the effective starting depth from the
    name's existing brackets per P-16.5.4.1 (Dec 2025 errata). This makes
    all callers that use the default depth automatically benefit from
    subsection-aware nesting.

    Args:
        name: The compound substituent name (without outer brackets).
        depth: Nesting depth (0 = outermost, -1 = auto-detect from name).

    Returns:
        Name enclosed in the appropriate bracket type.

    Examples:
        >>> apply_enclosing_marks("2-methylpropyl", 0)
        '(2-methylpropyl)'
        >>> apply_enclosing_marks("2-methylpropyl", 1)
        '[2-methylpropyl]'
        >>> apply_enclosing_marks("2-methylpropyl", 2)
        '{2-methylpropyl}'
        >>> apply_enclosing_marks("(R)-butan-2-yl", -1)
        '[(R)-butan-2-yl]'
    """
    MARKS = [('(', ')'), ('[', ']'), ('{', '}')]

    if depth == -1:
        # Auto-detect: compute effective depth from name content
        depth = compute_nesting_depth(name)

    # P-16.5.4.1.4: Check for consecutive same-level marks.
    # If the name starts with the same type of bracket we'd add, escalate --
    # BUT only if the leading bracket is nesting-relevant (not indicated H,
    # not fusion/spiro brackets).
    open_mark, close_mark = MARKS[depth % 3]
    if name.startswith(open_mark):
        # Check if the leading bracket is non-nesting (indicated H or fusion)
        leading_is_nesting = True
        if open_mark == '(':
            # Check for indicated hydrogen: (1H), (3H), (9aH)
            if _INDICATED_H_RE.match(name):
                leading_is_nesting = False
            # Check for stereo: (R), (S), (E), (Z), (1R,2S) -- these ARE nesting
            elif _STEREO_PAREN_RE.match(name):
                leading_is_nesting = True
        elif open_mark == '[':
            # Check for fusion/spiro brackets: [2,3-b], [4.5]
            if _FUSION_BRACKET_RE.match(name):
                leading_is_nesting = False

        if leading_is_nesting:
            depth += 1
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
    # Check for hyphens (indicates compound substituent) — EXCEPT a leading
    # italicized 'sec-'/'tert-' detachable prefix on an otherwise-simple retained
    # name (tert-butyl, sec-butyl). IUPAC P-16.3.4 treats these as SIMPLE for
    # multiplication (di-tert-butyl, NOT bis(tert-butyl)) and does not enclose them
    # in marks (N-tert-butyl, NOT N-(tert-butyl)). Phase 171 BBR-ASM: without this,
    # coupling the paren/bis decision to is_complex_substituent over-parenthesised
    # tert-butyl. The leading-digit case above still catches genuine compounds.
    _hyphen_probe = name.lower()
    for _retained_prefix in ("sec-", "tert-"):
        if _hyphen_probe.startswith(_retained_prefix):
            _hyphen_probe = _hyphen_probe[len(_retained_prefix):]
            break
    if "-" in _hyphen_probe:
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


def _wrap_n_substituent(name: str) -> str:
    """Apply IUPAC P-16.3.3 bracket escalation to an N-substituent name.

    When an N-substituent already contains parentheses (from stereo
    descriptors or compound substituent names), the outer enclosure must
    use square brackets to maintain unambiguous nesting.

    Simple names (no parentheses) are returned unchanged -- they do not
    need brackets around them in N-prefix context.

    Names that already have balanced outer enclosing marks -- starting
    with '(' or '[' and ending with the matching close, where the opening
    mark at position 0 is balanced at the final position -- are returned
    unchanged (already properly enclosed).

    Args:
        name: The bare N-substituent name (without N- prefix).

    Returns:
        The name, optionally wrapped in square brackets.

    Examples:
        >>> _wrap_n_substituent("methyl")
        'methyl'
        >>> _wrap_n_substituent("dimethyl")
        'dimethyl'
        >>> _wrap_n_substituent("(2S)-2-(pentanoylamino)propanoyl")
        '[(2S)-2-(pentanoylamino)propanoyl]'
        >>> _wrap_n_substituent("(2R,3S)-3-hydroxy-2-(benzoylamino)butanoyl")
        '[(2R,3S)-3-hydroxy-2-(benzoylamino)butanoyl]'
        >>> _wrap_n_substituent("[already-bracketed]")
        '[already-bracketed]'
        >>> _wrap_n_substituent("(3-ethyl-1H-indolyl)")
        '(3-ethyl-1H-indolyl)'
    """
    # Already has outer square brackets -- no double-wrapping
    if name.startswith('[') and name.endswith(']'):
        return name
    # Already has balanced outer parentheses as enclosing marks --
    # this means a previous step already enclosed the name. Check that
    # the opening paren at position 0 is balanced at the final position.
    if name.startswith('(') and name.endswith(')'):
        depth = 0
        for i, ch in enumerate(name):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth == 0 and i == len(name) - 1:
                # The opening paren matches the closing paren at the end
                return name
    # Contains parentheses (stereo prefix, compound sub-substituent, etc.)
    # -> escalate to square brackets per P-16.3.3
    if '(' in name:
        return f"[{name}]"
    return name


def format_substituent_prefix(name: str, locants: List[int], count: int) -> str:
    """Format a substituent with locants and multiplier prefix.

    BBR-HYG(e) / DEF-8 inventory (Phase 169.7 — begin; consolidation lands in
    BBR-ASM / Phase 171). This is the ONE correct substituent-prefix + needs-parens
    reference (P-16.3.5 / P-16.3.3). Phase 171 consolidates the divergent
    needs-parens / enclosing-mark / prefix-assembly predicates onto THIS function;
    do NOT add a 4th. The divergent predicates as of 169.7:
      - rules/amides.py:32           _has_positional_locants(name)   (digit-only locant test)
      - assembly/naming_utils.py:525 is_complex_substituent(name)    (complex-substituent test)
      - assembly/naming_utils.py:461 apply_enclosing_marks(name, depth)
      - rules/ortho_fused.py:505     format_substituent_prefix(substituents)  (different signature)
    They DISAGREE today (e.g. on 'trifluoromethyl'/'tert-butyl': is_complex_substituent
    True but _has_positional_locants False) — see
    tests/unit/assembly/test_needs_parens_consolidation.py (xfail-strict tripwire).

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

    # Leading positional descriptors are ALWAYS ignored for alphabetization
    # (P-14.5.2): a parent locant ('3-'), a compound locant set ('2,4-'), and
    # the italic indicated-hydrogen descriptor ('1H-'). Strip them up front so
    # the enclosed-vs-simple decision below sees the bare substituent name —
    # this is what makes '3-(1H-imidazol-5-yl)' and '(1H-imidazol-5-yl)' yield
    # the same key, and '2-(2,4-dihydroxyphenyl)' route to the complete-name
    # branch just like the bare '(2,4-dihydroxyphenyl)'.
    text = _LOCANT_PREFIX_RE.sub('', text)
    text = _INDICATED_H_PREFIX_RE.sub('', text)

    # P-14.5.2 (Phase 171 BBR-ASM, DEF-8): a COMPOUND substituent cited as a
    # fully-enclosed unit is alphabetized on the first letter of its COMPLETE
    # name — its INTERNAL multiplying prefix (di/tri…) is part of the name and
    # is NOT ignored ('(2,4-dimethylpentyl)' sorts at 'd', before 'ethyl';
    # '2-(2,4-dihydroxyphenyl)' sorts at 'd', before 'hydroxy'). Contrast
    # P-14.5.1: a bare 'di'/'tri' multiplying SEPARATE simple prefixes on the
    # parent IS ignored ('2,2-dimethyl' sorts at 'm'). The enclosing marks
    # disambiguate the two rules, so only fully-enclosed input takes the
    # complete-name path.
    if (text.startswith('(') and text.endswith(')')) or (text.startswith('[') and text.endswith(']')):
        inner = text[1:-1]
        inner = _LOCANT_PREFIX_RE.sub('', inner)       # drop the inner positional locant
        inner = _INDICATED_H_PREFIX_RE.sub('', inner)  # and a leading '1H-' descriptor
        if (inner.startswith('(') and inner.endswith(')')) or (inner.startswith('[') and inner.endswith(']')):
            inner = inner[1:-1]
        return inner  # complete name; internal multiplying prefix NOT stripped

    # ---- Simple (non-enclosed) prefix: P-14.5.1 ----
    # Strip N-locant prefixes (N- or N,N-)
    # e.g., "N,N-dimethylamino" -> "dimethylamino" -> "amino" (after multi-prefix strip)
    text = _N_LOCANT_PREFIX_RE.sub('', text)

    # Handle hyphenated detachable prefixes: sec- and tert-
    for prefix in ("sec-", "tert-"):
        if text.startswith(prefix):
            return text[len(prefix):]

    # Handle non-hyphenated multiplicative prefixes (di-, tri-, ...): for a
    # SIMPLE prefix these are ignored (P-14.5.1). Sort by longest prefix first
    # to avoid partial matches (e.g., 'tetra' before 'tri'). The result is a
    # sort key only, never reconstructed.
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
            # IUPAC P-16.3.3 vowel elision: drop terminal 'e' of the parent stem
            # only when the suffix begins with a, i, o, u, or y. Before consonant-
            # leading suffixes (-thione, -selone, -tellone, etc.) preserve the 'e'.
            #
            # Examples:
            #   propan + 1 + ol     -> 'propan-1-ol'     (elide e — 'o' is vowel)
            #   butan  + 2 + one    -> 'butan-2-one'     (elide e — 'o' is vowel)
            #   propan + 2 + thione -> 'propane-2-thione' (keep e — 't' is consonant)
            #   propan + 2 + selone -> 'propane-2-selone' (keep e — 's' is consonant)
            #   ethan  + 1 + thione -> 'ethane-1-thione' (P-66.1.4.3 example PIN)
            if suffix and suffix[0] in _ELISION_VOWELS:
                return f"{base}-{locant_str}-{suffix}"
            else:
                return f"{base}e-{locant_str}-{suffix}"
    else:
        # No locants: combine base and suffix with elision
        # e.g., 'pentanoic acid', 'propanal'
        if suffix_multiplier:
            # Multiple terminal groups without locants (e.g., diacids):
            # 'butanedioic acid', 'pentanedioic acid'
            full_suffix = suffix_multiplier + suffix
            return apply_vowel_elision(base + "e", full_suffix)
        return apply_vowel_elision(base + "e", suffix)
