"""
OPSIN format validation for generated IUPAC names.

FMT-01: Heuristic validator that detects known OPSIN-incompatible patterns.
FMT-04: Bracket nesting verification (P-16.5.4.1 hierarchy).
FMT-05: Multi-word name validation against OPSIN's 21 wordRules.xml patterns.

This is a DIAGNOSTIC tool, not a postprocessor. It flags known issues
so upstream fixes can be applied at the root cause.
"""

import re
from typing import Tuple

from orthonym.data.opsin_imports.word_rules import OPSIN_WORD_RULES


# -------------------------------------------------------------------------
# Known OPSIN word rule patterns (simplified for heuristic matching)
# -------------------------------------------------------------------------

# Functional terms that OPSIN recognizes as standalone words
_MONOVALENT_GROUPS = {
    "oxide", "sulfide", "selenide", "telluride",
    "fluoride", "chloride", "bromide", "iodide",
    "cyanide", "azide", "hydroxide", "hydride",
    "peroxide", "superoxide",
}

_DIVALENT_GROUPS = {
    "ether", "sulfide", "selenide", "telluride",
    "peroxide", "disulfide",
}

_CHALCOGENIDES = {
    "oxide", "sulfide", "selenide", "telluride",
    "dioxide", "trioxide", "tetroxide",
    "disulfide", "trisulfide",
}

_ANHYDRIDE_TERMS = {"anhydride"}

_CARBONYL_REPLACEMENT = {
    "hydrazone", "oxime", "semicarbazone",
    "thiosemicarbazone", "phenylhydrazone",
    "dinitrophenylhydrazone",
}

_ACETAL_LIKE = {
    "acetal", "hemiacetal", "hemiketal", "ketal",
    "dimethyl acetal", "diethyl acetal",
}

# Acid halide/pseudo-halide terms (standalone functional terms)
_ACID_HALIDE_TERMS = _MONOVALENT_GROUPS

# Ester term
_ESTER_TERM = {"ester"}

# Functional class terms
_FUNCTIONAL_CLASS_TERMS = (
    _MONOVALENT_GROUPS | _DIVALENT_GROUPS | _CHALCOGENIDES
    | _ANHYDRIDE_TERMS | _CARBONYL_REPLACEMENT | _ACETAL_LIKE
    | _ESTER_TERM
)


def _is_ester_multiword(words: list) -> bool:
    """Check if multi-word name matches ester pattern: <sub> <name-ate>."""
    if len(words) < 2:
        return False
    last = words[-1]
    # Standard ester: "methyl propanoate" -- last word ends in -ate
    if re.search(r'at[e]\b', last) or last.endswith("ate"):
        return True
    return False


def _is_anhydride_multiword(words: list) -> bool:
    """Check if multi-word name matches anhydride pattern."""
    if len(words) < 2:
        return False
    last = words[-1].lower()
    if last == "anhydride":
        return True
    return False


def _is_oxide_multiword(words: list) -> bool:
    """Check if multi-word name matches oxide/chalcogenide pattern."""
    if len(words) < 2:
        return False
    last = words[-1].lower()
    if last in _CHALCOGENIDES:
        return True
    return False


def _is_acid_halide_multiword(words: list) -> bool:
    """Check if multi-word name matches acid halide pattern: <acid> <halide>."""
    if len(words) < 2:
        return False
    last = words[-1].lower()
    # Check if any earlier word ends with "acid" and last is a halide term
    has_acid = any(w.lower().endswith("acid") for w in words[:-1])
    if has_acid and last in _ACID_HALIDE_TERMS:
        return True
    return False


def _is_salt_multiword(words: list) -> bool:
    """Check if multi-word name matches salt pattern: <cation> <anion>."""
    if len(words) < 2:
        return False
    last = words[-1].lower()
    # Common salt patterns: "sodium chloride", "potassium acetate"
    # Cation names typically end in -ium or are element names
    first = words[0].lower()
    cation_patterns = [
        "sodium", "potassium", "lithium", "calcium", "magnesium",
        "barium", "aluminium", "aluminum", "iron", "copper", "zinc",
        "silver", "ammonium", "hydronium",
    ]
    if first in cation_patterns or first.endswith("ium"):
        return True
    return False


def _is_carbonyl_derivative_multiword(words: list) -> bool:
    """Check if multi-word name matches carbonyl derivative pattern."""
    if len(words) < 2:
        return False
    last = words[-1].lower()
    if last in _CARBONYL_REPLACEMENT:
        return True
    return False


def _is_acetal_multiword(words: list) -> bool:
    """Check if multi-word name matches acetal pattern."""
    if len(words) < 2:
        return False
    last = words[-1].lower()
    if last in _ACETAL_LIKE:
        return True
    # "dimethyl acetal" pattern
    if len(words) >= 3 and words[-1].lower() == "acetal":
        return True
    return False


def _is_amide_functional_group(words: list) -> bool:
    """Check substituent + amide functional group pattern."""
    if len(words) < 2:
        return False
    last = words[-1].lower()
    # Matches words ending in 'amide' or 'amid' (possibly with brackets)
    if re.search(r'amid[e]?[\]\)\}]*$', last):
        return True
    return False


def _is_ester_functional_class(words: list) -> bool:
    """Check 'name sub ester' pattern (functional class ester)."""
    if len(words) < 3:
        return False
    # Look for 'ester' as one of the words
    for w in words[1:]:
        if w.lower() == "ester":
            return True
    return False


def _is_glycol_multiword(words: list) -> bool:
    """Check glycol/halohydrin pattern."""
    if len(words) < 2:
        return False
    last = words[-1].lower()
    if last in {"glycol", "halohydrin", "chlorohydrin", "bromohydrin"}:
        return True
    return False


def _is_polymer_multiword(words: list) -> bool:
    """Check poly/oligo pattern."""
    if len(words) < 2:
        return False
    first = words[0].lower()
    if first in {"poly", "oligo"}:
        return True
    return False


def _is_acid_name(words: list) -> bool:
    """Check if multi-word name is a standard 'X acid' name.

    OPSIN treats names like 'propanoic acid', 'benzoic acid',
    'sulfuric acid' as single full names. The last word is 'acid'
    and the first word is the acid stem.
    """
    if len(words) < 2:
        return False
    if words[-1].lower() == "acid":
        return True
    return False


def _matches_known_multiword_pattern(words: list) -> bool:
    """Check if a multi-word name matches any known OPSIN word rule pattern."""
    checks = [
        _is_acid_name,
        _is_ester_multiword,
        _is_anhydride_multiword,
        _is_oxide_multiword,
        _is_acid_halide_multiword,
        _is_salt_multiword,
        _is_carbonyl_derivative_multiword,
        _is_acetal_multiword,
        _is_amide_functional_group,
        _is_ester_functional_class,
        _is_glycol_multiword,
        _is_polymer_multiword,
    ]
    for check in checks:
        if check(words):
            return True
    return False


def _check_bracket_nesting(name: str) -> Tuple[bool, str]:
    """Verify P-16.5.4.1 bracket hierarchy: () inside [] inside {}.

    IUPAC enclosing marks hierarchy:
    - Level 0: parentheses ()
    - Level 1: brackets []
    - Level 2: braces {}
    - Then repeat: (()) [[]] {{}}

    We check that nesting order is correct: parentheses should not
    contain brackets at a shallower depth than expected.
    """
    # Track depth per bracket type
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    total_depth = 0

    # Inside VB descriptors [x.y.z] we skip nesting checks
    in_vb = False
    i = 0
    while i < len(name):
        ch = name[i]

        # Detect VB descriptors: [digits.digits.digits...] patterns
        if ch == '[' and not in_vb:
            # Check if this looks like a VB descriptor
            j = name.find(']', i + 1)
            if j > i + 1:
                content = name[i + 1:j]
                if re.match(r'^[\d.,()]+$', content):
                    # This is a VB descriptor, skip it
                    i = j + 1
                    continue

        if ch == '(':
            paren_depth += 1
            total_depth += 1
        elif ch == ')':
            paren_depth -= 1
            total_depth -= 1
        elif ch == '[':
            bracket_depth += 1
            total_depth += 1
        elif ch == ']':
            bracket_depth -= 1
            total_depth -= 1
        elif ch == '{':
            brace_depth += 1
            total_depth += 1
        elif ch == '}':
            brace_depth -= 1
            total_depth -= 1

        i += 1

    # Check final balance (already done by unbalanced check, but verify)
    if paren_depth != 0 or bracket_depth != 0 or brace_depth != 0:
        return False, "bracket_nesting_unbalanced"

    return True, "ok"


def validate_name_format(name: str) -> Tuple[bool, str]:
    """Validate an IUPAC name for known OPSIN-incompatible patterns.

    This is a diagnostic heuristic validator (FMT-01). It checks for
    patterns known to cause OPSIN parse failures. It is NOT a full
    OPSIN parser reimplementation.

    Args:
        name: Generated IUPAC name string.

    Returns:
        Tuple of (is_valid, reason). If valid, returns (True, "ok").
        If invalid, returns (False, "reason_code: details").

    Checks performed:
        1. Empty name
        2. Unbalanced brackets (parentheses, square brackets, braces)
        3. Empty parentheses ()
        4. Bare 'oxy' prefix (not part of a larger word)
        5. Double hyphens -- (except within VB descriptors)
        6. Multi-word name validation against OPSIN word rules (FMT-05)
        7. Bracket nesting hierarchy verification (FMT-04)
    """
    # 1. Empty name check
    if not name or not name.strip():
        return False, "empty_name: name is empty or whitespace only"

    name = name.strip()

    # 2. Unbalanced brackets check
    # Count opening and closing for each bracket type
    # Depth must never go negative and must end at 0
    for open_ch, close_ch, label in [
        ('(', ')', 'parentheses'),
        ('[', ']', 'square_brackets'),
        ('{', '}', 'braces'),
    ]:
        depth = 0
        for ch in name:
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth < 0:
                    return False, f"unbalanced_{label}: closing before opening"
        if depth != 0:
            return False, f"unbalanced_{label}: {depth} unclosed"

    # 3. Empty parentheses check
    if "()" in name:
        return False, "empty_parentheses: name contains ()"

    # 4. Bare 'oxy' prefix check
    # Detect standalone 'oxy' that is not part of a larger word
    # e.g., "oxy" alone or at word boundaries, but not "methoxy", "ethoxy", etc.
    if re.search(r'(?<![a-zA-Z])oxy(?![a-zA-Z])', name, re.IGNORECASE):
        return False, "bare_oxy: unqualified 'oxy' prefix detected"

    # 5. Double hyphens check (except within VB descriptors [...])
    # First, mask VB descriptor contents
    masked = re.sub(r'\[[^\]]*\]', lambda m: '[' + 'X' * (len(m.group()) - 2) + ']', name)
    if "--" in masked:
        return False, "double_hyphen: name contains '--' outside VB descriptors"

    # 6. Multi-word validation (FMT-05)
    # Split on spaces to detect multi-word names
    words = name.split()
    if len(words) > 1:
        if not _matches_known_multiword_pattern(words):
            return False, f"unknown_multi_word_pattern: '{name}' does not match any known OPSIN word rule"

    # 7. Bracket nesting hierarchy (FMT-04)
    ok, reason = _check_bracket_nesting(name)
    if not ok:
        return False, f"bracket_nesting: {reason}"

    return True, "ok"
