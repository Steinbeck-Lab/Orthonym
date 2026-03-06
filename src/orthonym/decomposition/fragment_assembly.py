"""
Fragment assembly module for decomposition engine.

Combines individually-named fragments into correct multi-component IUPAC names.
Assembly patterns differ by bond type:
- Esters: "alkyl alkanoate" (e.g., "ethyl acetate")
- Amides: "N-[substituent][acid-amide]" (e.g., "N-methylacetamide")
- Glycosides: basic concatenation (full naming deferred to Phase 42)

Each assembler handles name transformations:
- acid name -> "-ate" form for esters
- acid name -> "-amide" form for amides
- acid name -> "-yl" (acyl) form for N-acyl naming
- alcohol name -> alkyl prefix form
"""

import re as _re
from collections import Counter as _Counter
from typing import Dict, List, Optional

from ..data.trivial_acids import get_acylate_name, TRIVIAL_ACID_TO_ACYLATE
from ..data.sugar_names import lookup_sugar, sugar_to_glycosyloxy_prefix


# ============================================================================
# Component joining helper
# ============================================================================

def _join_components(left: str, right: str) -> str:
    """Join two name components with IUPAC-compliant hyphenation.

    Inserts a hyphen when:
    - Left ends with letter/')' and right starts with digit
    - Left ends with letter/')' and right starts with uppercase letter
      (N-substitution, L/D/R/S stereodescriptors)

    This matches the logic in composer.py _join_prefix_to_name().

    Args:
        left: Left name component.
        right: Right name component.

    Returns:
        Combined string with proper hyphenation.
    """
    if not left or not right:
        return (left or "") + (right or "")

    last = left[-1]
    first = right[0]

    if (last.isalpha() or last in (')', ']')) and first.isdigit():
        return f"{left}-{right}"
    if (last.isalpha() or last in (')', ']')) and first.isupper():
        return f"{left}-{right}"

    return f"{left}{right}"


# ============================================================================
# Trivial acid conversions that don't follow simple suffix rules
# ============================================================================

# Acid full name -> amide form (for trivial names)
_TRIVIAL_ACID_TO_AMIDE = {
    "formic acid": "formamide",
    "acetic acid": "acetamide",
    "propionic acid": "propionamide",
    "butyric acid": "butyramide",
    "valeric acid": "valeramide",
    "benzoic acid": "benzamide",
}

# Acid full name -> acyl prefix (for trivial names)
_TRIVIAL_ACID_TO_ACYL = {
    "formic acid": "formyl",
    "acetic acid": "acetyl",
    "propionic acid": "propionyl",
    "butyric acid": "butyryl",
    "valeric acid": "valeryl",
    "benzoic acid": "benzoyl",
}

# Alcohol -> alkyl prefix (for trivial/common names)
_TRIVIAL_ALCOHOL_TO_ALKYL = {
    "methanol": "methyl",
    "ethanol": "ethyl",
    "propan-1-ol": "propyl",
    "propan-2-ol": "propan-2-yl",
    "butan-1-ol": "butyl",
    "butan-2-ol": "butan-2-yl",
    "pentan-1-ol": "pentyl",
    "hexan-1-ol": "hexyl",
    "heptan-1-ol": "heptyl",
    "octan-1-ol": "octyl",
    "phenol": "phenyl",
    "cyclohexanol": "cyclohexyl",
}

# Alcohol/parent -> alkoxy prefix (retained alkoxy names per IUPAC P-63.2.3.1)
_RETAINED_ALKOXY = {
    "methanol": "methoxy",
    "ethanol": "ethoxy",
    "propan-1-ol": "propoxy",
    "butan-1-ol": "butoxy",
    "phenol": "phenoxy",
}

# Amine -> prefix form for amide N-substitution
_TRIVIAL_AMINE_TO_PREFIX = {
    "methanamine": "methyl",
    "methylamine": "methyl",
    "ethanamine": "ethyl",
    "ethylamine": "ethyl",
    "propan-1-amine": "propyl",
    "propylamine": "propyl",
    "butan-1-amine": "butyl",
    "butylamine": "butyl",
    "aniline": "phenyl",
    "cyclohexanamine": "cyclohexyl",
    "cyclohexylamine": "cyclohexyl",
}


# ============================================================================
# Public API
# ============================================================================

def assemble_fragment_name(
    bond_type: str,
    fragment_names: Dict[str, str],
    style: str = "pin",
) -> Optional[str]:
    """Assemble fragment names into a multi-component IUPAC name.

    Dispatches to bond-type-specific assemblers.

    Args:
        bond_type: Type of bond that was cleaved ("ester", "amide",
                   "glycosidic", "carbamate").
        fragment_names: Dict mapping fragment roles to their names.
            For esters: {"acid": "acetic acid", "alkyl": "ethanol"}
            For amides: {"acid": "acetic acid", "amine": "methylamine"}
            For glycosides: {"sugar": "...", "aglycone": "..."}
        style: Naming style ("pin" for preferred IUPAC names).

    Returns:
        Assembled multi-component name, or None if assembly fails.

    Examples:
        >>> assemble_fragment_name("ester", {"acid": "acetic acid", "alkyl": "ethanol"})
        'ethyl acetate'
        >>> assemble_fragment_name("amide", {"acid": "acetic acid", "amine": "methylamine"})
        'N-methylacetamide'
    """
    if not bond_type or not fragment_names:
        return None

    assemblers = {
        "ester": _assemble_ester,
        "amide": _assemble_amide,
        "glycosidic": _assemble_glycoside,
        "carbamate": _assemble_carbamate,
        "ether": _assemble_ether,
        "thioester": _assemble_thioester,
        "phosphodiester": _assemble_phosphodiester,
        "sulfonamide": _assemble_sulfonamide,
    }

    assembler = assemblers.get(bond_type)
    if assembler is None:
        return None

    return assembler(fragment_names, style)


# ============================================================================
# N-substituent grouping (IUPAC P-16.3.4)
# ============================================================================

def _group_n_substituents(name: str) -> str:
    """Group repeated N-prefix patterns in a decomposition-produced name.

    Detects patterns like "N-acetyl-N-acetyltetrahydropyranamine" and
    collapses them to "N,N-diacetyltetrahydropyranamine" using IUPAC
    P-16.3.4 multiplicative prefixes (di/tri for simple substituents,
    bis/tris for complex ones).

    This operates WITHIN the decomposition assembly pipeline, not as a
    postprocessor on the final output.

    Args:
        name: A decomposition-produced name that may contain repeated
              N-prefix segments.

    Returns:
        The name with repeated N-prefixes grouped, or the original
        name if no grouping is needed.

    Examples:
        >>> _group_n_substituents("N-acetyl-N-acetylpiperidine")
        'N,N-diacetylpiperidine'
        >>> _group_n_substituents("N-formyl-N-acetyl-N-acetylpiperidine")
        'N-acetyl-N,N-diformylpiperidine'  # sorted alphabetically
        >>> _group_n_substituents("N-methylacetamide")
        'N-methylacetamide'
    """
    # Extract all N-prefix segments from the beginning of the name.
    # Pattern: "N-{substituent_name}" repeated, followed by the parent name.
    # Each N-substituent is "N-" + a lowercase name that runs until the
    # next "N-" or until the parent starts (no more "N-" prefixes).

    # Find all N-prefix segments using regex.
    # Matches: "N-acetyl", "N-formyl", "N-(2-methylpropyl)", etc.
    # The segment content is everything after "N-" until the next "-N-" boundary
    # or until the parent compound starts.
    segments: List[str] = []
    remainder = name

    while remainder.startswith("N-"):
        # Skip the "N-" prefix
        rest = remainder[2:]

        # Find where this N-substituent ends. It ends at the next "-N-"
        # that signals another N-substituent, OR where the parent starts.
        # Handle parenthesized segments: "(2-methylpropyl)" contains "N" but
        # it's inside parens so not an N-prefix boundary.
        next_n_pos = -1
        paren_depth = 0
        bracket_depth = 0
        for i, ch in enumerate(rest):
            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth -= 1
            elif ch == '[':
                bracket_depth += 1
            elif ch == ']':
                bracket_depth -= 1
            elif (ch == 'N' and paren_depth == 0 and bracket_depth == 0
                    and i > 0 and rest[i - 1] == '-'
                    and i + 1 < len(rest) and rest[i + 1] == '-'):
                # Found "-N-" boundary at position i-1 in rest
                next_n_pos = i - 1
                break

        if next_n_pos >= 0:
            # Extract this segment (without trailing hyphen)
            seg = rest[:next_n_pos]
            segments.append(seg)
            remainder = rest[next_n_pos + 1:]  # skip the hyphen, start at "N-..."
        else:
            # No more N-prefixes; rest is "substituent + parent"
            # We need to separate the substituent from the parent.
            # The substituent is the text that, when combined with "N-",
            # makes a valid N-prefix. For simple cases like "acetylpiperidine",
            # the boundary is where the substituent ends and parent begins.
            # For now, take everything as the last segment + parent.
            segments.append(rest)
            remainder = ""
            break

    if len(segments) < 2:
        # Only 0 or 1 N-prefix segments -- no grouping needed
        return name

    # The last segment contains both the final substituent AND the parent.
    # We need to split it. The substituent is the part that matches the
    # pattern of the other segments (ends with a known acyl suffix like "yl",
    # "oyl", "amido" etc., followed by the parent compound name).
    # Since we're in the decomposition pipeline, the substituent is always
    # an acyl group (ends in "yl") or a simple prefix fused with the parent.

    # Strategy: collect all segments. The last segment has the substituent
    # fused with the parent (e.g., "acetylpiperidine"). We need to detect
    # where the acyl prefix ends and the parent begins.
    # For safety, just keep the last segment as-is if we can't split it.

    # Count identical substituents
    # First, try to identify the acyl name in each segment.
    # Segments 0..n-2 are standalone substituent names (e.g., "acetyl").
    # The last segment is "substituent + parent" (e.g., "acetylpiperidine").

    # Check if we can split the last segment by matching earlier segments.
    last_seg = segments[-1]
    standalone_segs = segments[:-1]

    # Try to find a matching prefix in the last segment
    parent_name = ""
    last_sub = last_seg
    for seg in standalone_segs:
        if last_seg.startswith(seg):
            last_sub = seg
            parent_name = last_seg[len(seg):]
            break

    if not parent_name:
        # Try generic acyl suffix detection: the substituent ends at "yl"
        # followed by the parent compound start
        yl_pos = last_seg.find("yl")
        while yl_pos >= 0:
            candidate_sub = last_seg[:yl_pos + 2]
            candidate_parent = last_seg[yl_pos + 2:]
            if candidate_parent and candidate_parent[0].islower():
                last_sub = candidate_sub
                parent_name = candidate_parent
                break
            yl_pos = last_seg.find("yl", yl_pos + 1)

    if not parent_name:
        # Could not separate last segment; return original name unchanged
        return name

    # Now we have all substituent names and the parent
    all_subs = standalone_segs + [last_sub]

    # Count occurrences
    counts = _Counter(all_subs)

    # Build grouped N-prefix parts, sorted alphabetically by substituent name
    parts: List[str] = []
    for sub_name in sorted(counts.keys()):
        count = counts[sub_name]
        n_locants = ",".join(["N"] * count)
        if count == 1:
            parts.append(f"{n_locants}-{sub_name}")
        else:
            # Use lazy import for get_multiplier_prefix to avoid circular imports
            from ..assembly.naming_utils import is_complex_substituent

            if is_complex_substituent(sub_name):
                # Complex: use bis/tris/tetrakis with parentheses
                _COMPLEX_MULT = {2: "bis", 3: "tris", 4: "tetrakis",
                                 5: "pentakis"}
                mult = _COMPLEX_MULT.get(count, f"{count}kis")
                parts.append(f"{n_locants}-{mult}({sub_name})")
            else:
                # Simple: use di/tri/tetra without parentheses
                _SIMPLE_MULT = {2: "di", 3: "tri", 4: "tetra", 5: "penta"}
                mult = _SIMPLE_MULT.get(count, str(count))
                parts.append(f"{n_locants}-{mult}{sub_name}")

    # Join parts with hyphens, then append parent
    prefix = "-".join(parts)
    return _join_components(prefix, parent_name)


# ============================================================================
# Bond-type-specific assemblers
# ============================================================================

def _assemble_ester(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble ester name as 'alkyl alkanoate'.

    Args:
        fragment_names: {"acid": acid_name, "alkyl": alkyl_name}
        style: Naming style.

    Returns:
        Ester name like "ethyl acetate", or None.
    """
    acid_name = fragment_names.get("acid")
    alkyl_name = fragment_names.get("alkyl")

    if not acid_name or not alkyl_name:
        return None

    ate_name = _acid_to_ate(acid_name)
    alkyl_prefix = _alcohol_to_alkyl(alkyl_name)

    if not ate_name or not alkyl_prefix:
        return None

    return f"{alkyl_prefix} {ate_name}"


def _assemble_amide(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble amide name as 'N-[substituent][acid-amide]'.

    For simple amines, uses amide suffix: "N-methylacetamide".
    For complex amines, uses acyl prefix: "N-acetylcyclohexanamine".

    After assembly, passes the result through _group_n_substituents() to
    collapse any repeated N-prefix patterns (e.g., "N-acetyl-N-acetyl..."
    becomes "N,N-diacetyl...") per IUPAC P-16.3.4.

    Args:
        fragment_names: {"acid": acid_name, "amine": amine_name}
        style: Naming style.

    Returns:
        Amide name, or None.
    """
    acid_name = fragment_names.get("acid")
    amine_name = fragment_names.get("amine")

    if not acid_name or not amine_name:
        return None

    result = None

    # Get amine prefix (simple substituent name)
    amine_prefix = _amine_to_prefix(amine_name)

    if amine_prefix:
        # Simple amine: use amide suffix pattern
        # "N-methylacetamide", "N-ethylpropanamide"
        amide_name = _acid_to_amide(acid_name)
        if amide_name:
            if amine_prefix.startswith("N-") or amine_prefix.startswith("N,"):
                # Prefix already carries N-substitution from recursive naming
                # (e.g., "N-methylcyclohexyl" from "N-methylcyclohexanamine").
                # Do NOT prepend another "N-" -- the N is already in the prefix.
                # Result: "N-methylcyclohexylacetamide" (correct).
                result = _join_components(amine_prefix, amide_name)
            else:
                result = f"N-{_join_components(amine_prefix, amide_name)}"

    if result is None:
        # Complex amine: use acyl prefix pattern
        # "N-acetylcyclohexanamine"
        acyl_prefix = _acid_to_acyl(acid_name)
        if acyl_prefix and amine_name:
            if amine_name.startswith("N-") or amine_name.startswith("N,"):
                # Amine already has N-prefix(es) from recursive naming.
                # Insert the acyl as an additional N-substituent:
                #   amine = "N-methylcyclohexanamine"
                #   acyl  = "acetyl"
                #   -> "N-acetyl-N-methylcyclohexanamine"
                # This keeps each N-substituent as a separate "N-X" segment
                # so _group_n_substituents can properly merge identical ones.
                result = f"N-{acyl_prefix}-{amine_name}"
            else:
                result = f"N-{_join_components(acyl_prefix, amine_name)}"

    if result is None:
        return None

    # Guard: multi-level decomposition can produce "N-N-" at the start
    # (outer level adds N-, inner level already starts with N-). This is
    # never valid IUPAC notation. Deduplicate to single "N-".
    while result.startswith("N-N-"):
        result = result[2:]  # strip one "N-"

    # Group repeated N-prefix patterns (IUPAC P-16.3.4)
    # This handles cases where recursive fragment naming already produced
    # an "N-acetyl..." prefix, and our assembly adds another "N-acetyl",
    # resulting in "N-acetyl-N-acetyl..." which should be "N,N-diacetyl...".
    if result.count("N-") >= 2:
        result = _group_n_substituents(result)

    return result


def _assemble_glycoside(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble glycoside name using sugar prefix + aglycone parent.

    Accepts both key conventions from the decomposition engine:
    - Engine convention: {"acid": sugar_name, "alkyl": aglycone_name}
    - Explicit convention: {"sugar": sugar_name, "aglycone": aglycone_name}

    The sugar name should ideally already be a glycosyloxy prefix
    (e.g., "beta-D-glucopyranosyloxy"). If it already ends in "oxy",
    it's used as-is. Otherwise, basic concatenation is used as fallback.

    Output format: "(glycosyloxy-prefix)aglycone-name"

    Args:
        fragment_names: Fragment name dict with sugar/aglycone info.
        style: Naming style.

    Returns:
        Glycoside name like "(beta-D-glucopyranosyloxy)phenol", or None.
    """
    # Accept both key conventions
    sugar_name = fragment_names.get("sugar") or fragment_names.get("acid")
    aglycone_name = fragment_names.get("aglycone") or fragment_names.get("alkyl")

    if not sugar_name or not aglycone_name:
        return None

    # If sugar_name already ends in "oxy" (glycosyloxy prefix), use as-is
    if sugar_name.endswith("oxy"):
        sugar_prefix = sugar_name
    else:
        # Fallback: use as-is (may be a systematic or retained name)
        sugar_prefix = sugar_name

    # Assemble as "(prefix)aglycone" with proper hyphenation
    return _join_components(f"({sugar_prefix})", aglycone_name)


def _assemble_carbamate(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble carbamate name.

    Basic pattern: "[alkyl] [amine]carbamate"

    Args:
        fragment_names: {"acid": acid_name, "alkyl": alkyl_name, "amine": amine_name}
        style: Naming style.

    Returns:
        Carbamate name, or None.
    """
    alkyl_name = fragment_names.get("alkyl")
    amine_name = fragment_names.get("amine")

    if not alkyl_name:
        return None

    alkyl_prefix = _alcohol_to_alkyl(alkyl_name)
    if not alkyl_prefix:
        return None

    if amine_name:
        amine_prefix = _amine_to_prefix(amine_name)
        if amine_prefix:
            return f"{alkyl_prefix} {amine_prefix}carbamate"

    return f"{alkyl_prefix} carbamate"


def _assemble_ether(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble ether name as 'alkoxy + parent' (IUPAC P-63.2.3).

    The smaller fragment (alkyl side, which retains the ether oxygen as an
    alcohol) is converted to an alkoxy prefix. The larger fragment (acid side)
    is the parent compound.

    Args:
        fragment_names: {"acid": parent_name, "alkyl": alkyl_name}
            where alkyl_name is an alcohol (the ether O stayed with it).
        style: Naming style.

    Returns:
        Ether name like "methoxybenzene", or None.
    """
    parent_name = fragment_names.get("acid")
    alkyl_name = fragment_names.get("alkyl")

    if not parent_name or not alkyl_name:
        return None

    alkoxy = _alcohol_to_alkoxy(alkyl_name)
    if not alkoxy:
        return None

    return _join_components(alkoxy, parent_name)


def _assemble_thioester(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble thioester name as 'S-alkyl alkanethioate' per IUPAC P-65.6.3.3.

    Thioic S-acid esters are named as functional class names where sulfur
    replaces oxygen on the ester side. The acid fragment converts to the
    "-thioate" form and the thiol fragment becomes an "S-alkyl" prefix.

    Args:
        fragment_names: {"acid": acid_name, "alkyl": thiol_name}
        style: Naming style.

    Returns:
        Thioester name like "S-methyl ethanethioate", or None.

    Examples:
        >>> _assemble_thioester({"acid": "acetic acid", "alkyl": "methanethiol"}, "pin")
        'S-methyl ethanethioate'
        >>> _assemble_thioester({"acid": "propanoic acid", "alkyl": "ethanethiol"}, "pin")
        'S-ethyl propanethioate'
    """
    acid_name = fragment_names.get("acid")
    thiol_name = fragment_names.get("alkyl")
    if not acid_name or not thiol_name:
        return None

    thioate = _acid_to_thioate(acid_name)
    s_prefix = _thiol_to_s_prefix(thiol_name)
    if not thioate or not s_prefix:
        return None

    return f"{s_prefix} {thioate}"


def _assemble_phosphodiester(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble phosphodiester name per IUPAC P-67.1.3 compositional nomenclature.

    The acid fragment is a phosphoric acid monoester (named by phosphorus.py
    after capping). The alkyl fragment name identifies the other ester group.
    Produces: "alkyl [acid-fragment-name]".

    Args:
        fragment_names: {"acid": acid_name, "alkyl": alkyl_name}
        style: Naming style.

    Returns:
        Phosphodiester name, or None.

    Examples:
        >>> _assemble_phosphodiester({"acid": "methyl dihydrogen phosphate", "alkyl": "methanol"}, "pin")
        'methyl methyl dihydrogen phosphate'
    """
    acid_name = fragment_names.get("acid")
    alkyl_name = fragment_names.get("alkyl")
    if not acid_name or not alkyl_name:
        return None

    alkyl_prefix = _alcohol_to_alkyl(alkyl_name)
    if alkyl_prefix and acid_name:
        return f"{alkyl_prefix} {acid_name}"
    return None


def _assemble_sulfonamide(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble sulfonamide as 'N-substituent (parent)sulfonamide' per P-66.6.4.2.2.

    The acid fragment is a sulfonic acid (named after capping). The amine
    fragment provides the N-substituent prefix. For unsubstituted sulfonamides
    (amine is ammonia or absent), produces just the parent sulfonamide.

    Args:
        fragment_names: {"acid": acid_name, "amine": amine_name}
        style: Naming style.

    Returns:
        Sulfonamide name like "N-methylbenzenesulfonamide", or None.

    Examples:
        >>> _assemble_sulfonamide({"acid": "benzenesulfonic acid", "amine": "methanamine"}, "pin")
        'N-methylbenzenesulfonamide'
        >>> _assemble_sulfonamide({"acid": "benzenesulfonic acid"}, "pin")
        'benzenesulfonamide'
    """
    acid_name = fragment_names.get("acid")
    amine_name = fragment_names.get("amine")
    if not acid_name:
        return None

    sulfonamide_parent = _acid_to_sulfonamide(acid_name)
    if not sulfonamide_parent:
        return None

    # Unsubstituted sulfonamide: no amine or amine is ammonia
    if not amine_name or amine_name.lower() in ('amine', 'ammonia'):
        return sulfonamide_parent

    amine_prefix = _amine_to_prefix(amine_name)
    if amine_prefix:
        return f"N-{_join_components(amine_prefix, sulfonamide_parent)}"

    return sulfonamide_parent


def _alcohol_to_alkoxy(name: str) -> Optional[str]:
    """Convert an alcohol or fragment name to its alkoxy form.

    Uses retained alkoxy names first (IUPAC P-63.2.3.1), then falls back
    to converting via _alcohol_to_alkyl() + replacing -yl with -oxy.

    Args:
        name: Alcohol or fragment name (e.g., "methanol", "ethanol",
              "propan-1-ol").

    Returns:
        Alkoxy prefix (e.g., "methoxy", "ethoxy", "propoxy"), or None.

    Examples:
        >>> _alcohol_to_alkoxy("methanol")
        'methoxy'
        >>> _alcohol_to_alkoxy("ethanol")
        'ethoxy'
        >>> _alcohol_to_alkoxy("propan-1-ol")
        'propoxy'
        >>> _alcohol_to_alkoxy("phenol")
        'phenoxy'
    """
    stripped = name.strip().lower()

    # Check retained alkoxy lookup first
    if stripped in _RETAINED_ALKOXY:
        return _RETAINED_ALKOXY[stripped]

    # Fallback: convert alcohol -> alkyl -> alkoxy
    alkyl = _alcohol_to_alkyl(name)
    if alkyl and alkyl.endswith("yl"):
        candidate = alkyl[:-2] + "oxy"
        if candidate != "oxy":  # Guard: never return bare "oxy"
            return candidate

    # Phase 86 fallback: for complex fragments where _alcohol_to_alkyl fails,
    # attempt direct suffix conversion. If the name already ends in "yl"
    # (from decomposition engine naming), convert to "oxy".
    if stripped.endswith("yl") and len(stripped) > 2:
        candidate = stripped[:-2] + "oxy"
        if candidate != "oxy":  # Guard: never return bare "oxy"
            return candidate

    # Last resort: if the name ends in common alcohol-like patterns, try
    # to derive alkoxy. Never return bare "oxy" for organic fragments.
    # For "-anol" (e.g., cyclopentanol -> cyclopentanoxy):
    if stripped.endswith("anol") and len(stripped) > 4:
        return stripped[:-4] + "anoxy"
    if stripped.endswith("ol") and len(stripped) > 2:
        base = stripped[:-2]
        if base.endswith("an") and len(base) > 2:
            return base[:-2] + "oxy"

    return None


# ============================================================================
# Name transformation helpers
# ============================================================================

def _acid_to_ate(acid_name: str) -> str:
    """Convert acid name to '-ate' form for ester naming.

    Handles both systematic names (propanoic acid -> propanoate) and
    trivial names (acetic acid -> acetate).

    Args:
        acid_name: Full acid name (e.g., "acetic acid", "propanoic acid").

    Returns:
        The '-ate' form (e.g., "acetate", "propanoate").

    Examples:
        >>> _acid_to_ate("acetic acid")
        'acetate'
        >>> _acid_to_ate("propanoic acid")
        'propanoate'
        >>> _acid_to_ate("benzoic acid")
        'benzoate'
        >>> _acid_to_ate("cyclohexanecarboxylic acid")
        'cyclohexanecarboxylate'
    """
    name = acid_name.strip()

    # Guard: if already in -ate form, don't convert again
    if name.endswith("ate") and " acid" not in name:
        return name

    # Handle "carboxylic acid" -> "carboxylate"
    if name.endswith("carboxylic acid"):
        return name[:-len("carboxylic acid")] + "carboxylate"

    # Strip " acid" suffix to get the stem for lookup
    if name.endswith(" acid"):
        stem = name[:-5].strip()
    else:
        stem = name

    # Try trivial acid lookup via existing module
    if stem.lower() in TRIVIAL_ACID_TO_ACYLATE:
        return TRIVIAL_ACID_TO_ACYLATE[stem.lower()]

    # Systematic: "-oic acid" -> "-oate"
    if stem.endswith("oic"):
        return stem[:-2] + "ate"  # propanoic -> propanoate

    # Generic "-ic" -> "-ate"
    if stem.endswith("ic"):
        return stem[:-2] + "ate"

    # Fallback: append "ate"
    return stem + "ate"


def _acid_to_amide(acid_name: str) -> str:
    """Convert acid name to '-amide' form.

    Handles systematic (propanoic acid -> propanamide) and
    trivial (acetic acid -> acetamide) names.

    Args:
        acid_name: Full acid name.

    Returns:
        The '-amide' form.

    Examples:
        >>> _acid_to_amide("acetic acid")
        'acetamide'
        >>> _acid_to_amide("propanoic acid")
        'propanamide'
        >>> _acid_to_amide("benzoic acid")
        'benzamide'
        >>> _acid_to_amide("cyclohexanecarboxylic acid")
        'cyclohexanecarboxamide'
    """
    name = acid_name.strip()

    # Guard: if already in -amide form, don't convert again
    if name.endswith("amide") and " acid" not in name:
        return name

    # Check trivial lookup first
    if name.lower() in _TRIVIAL_ACID_TO_AMIDE:
        return _TRIVIAL_ACID_TO_AMIDE[name.lower()]

    # Handle "carboxylic acid" -> "carboxamide"
    if name.endswith("carboxylic acid"):
        return name[:-len("carboxylic acid")] + "carboxamide"

    # Strip " acid" suffix
    if name.endswith(" acid"):
        stem = name[:-5].strip()
    else:
        stem = name

    # Systematic: "-oic" -> "-amide"
    # "propanoic" -> "propanamide" (drop "oic", add "amide")
    if stem.endswith("oic"):
        return stem[:-3] + "amide"  # propanoic -> propanamide

    # Trivial: "-ic" -> "-amide"
    # "acetic" -> "acetamide" (drop "ic", add "amide")
    if stem.endswith("ic"):
        return stem[:-2] + "amide"

    # Fallback
    return stem + "amide"


def _acid_to_acyl(acid_name: str) -> str:
    """Convert acid name to acyl prefix form.

    Used for N-acyl naming of amides with complex amines.

    Args:
        acid_name: Full acid name.

    Returns:
        Acyl prefix (e.g., "acetyl", "propanoyl", "benzoyl").

    Examples:
        >>> _acid_to_acyl("acetic acid")
        'acetyl'
        >>> _acid_to_acyl("propanoic acid")
        'propanoyl'
        >>> _acid_to_acyl("benzoic acid")
        'benzoyl'
    """
    name = acid_name.strip()

    # Check trivial lookup first
    if name.lower() in _TRIVIAL_ACID_TO_ACYL:
        return _TRIVIAL_ACID_TO_ACYL[name.lower()]

    # Handle "carboxylic acid" -> "carbonyl"
    if name.endswith("carboxylic acid"):
        return name[:-len("carboxylic acid")] + "carbonyl"

    # Strip " acid" suffix
    if name.endswith(" acid"):
        stem = name[:-5].strip()
    else:
        stem = name

    # Systematic: "-oic" -> "-oyl"
    if stem.endswith("oic"):
        return stem[:-2] + "yl"  # propanoic -> propanoyl

    # Trivial: "-ic" -> "-yl"
    if stem.endswith("ic"):
        return stem[:-2] + "yl"

    # Fallback
    return stem + "yl"


def _acid_to_thioate(acid_name: str) -> Optional[str]:
    """Convert acid name to thioate form per IUPAC P-65.6.3.3.

    The "-thioate" suffix replaces "-oate" in ester naming when sulfur
    replaces the ester oxygen.

    Args:
        acid_name: Full acid name (e.g., "acetic acid", "propanoic acid").

    Returns:
        The thioate form (e.g., "ethanethioate", "propanethioate"), or None.

    Examples:
        >>> _acid_to_thioate("acetic acid")
        'ethanethioate'
        >>> _acid_to_thioate("propanoic acid")
        'propanethioate'
    """
    name = acid_name.strip()

    # For thioester naming, always use systematic form (not trivial)
    # because thio- derivatives use systematic nomenclature per IUPAC P-65.6.3.3

    # Handle "carboxylic acid" -> "carbothioate"
    if name.endswith("carboxylic acid"):
        return name[:-len("carboxylic acid")] + "carbothioate"

    # Trivial acid name -> systematic thioate conversion
    # Map common trivial acids to their systematic thioate forms
    _TRIVIAL_TO_THIOATE = {
        "formic acid": "methanethioate",
        "acetic acid": "ethanethioate",
        "propionic acid": "propanethioate",
        "butyric acid": "butanethioate",
        "valeric acid": "pentanethioate",
        "benzoic acid": "benzenecarbothioate",
    }
    if name.lower() in _TRIVIAL_TO_THIOATE:
        return _TRIVIAL_TO_THIOATE[name.lower()]

    # Strip " acid" suffix
    if name.endswith(" acid"):
        stem = name[:-5].strip()
    else:
        stem = name

    # Systematic: "propanoic" -> "propanethioate" (replace "-oic" with "ethioate")
    if stem.endswith("oic"):
        return stem[:-3] + "ethioate"  # propanoic -> propanethioate

    # Generic "-ic" -> "ethioate" (for systematic names)
    if stem.endswith("ic"):
        return stem[:-2] + "ethioate"

    # Fallback
    return stem + "thioate"


def _thiol_to_s_prefix(thiol_name: str) -> Optional[str]:
    """Convert thiol name to S-alkyl prefix for thioester naming.

    Per IUPAC P-65.6.3.3, the sulfur-bearing fragment is designated
    with an "S-" locant prefix followed by the alkyl name.

    Args:
        thiol_name: Thiol fragment name (e.g., "methanethiol", "ethanethiol").

    Returns:
        S-alkyl prefix (e.g., "S-methyl", "S-ethyl"), or None.

    Examples:
        >>> _thiol_to_s_prefix("methanethiol")
        'S-methyl'
        >>> _thiol_to_s_prefix("ethanethiol")
        'S-ethyl'
    """
    name = thiol_name.strip()
    if not name:
        return None

    # Try stripping thiol/anethiol suffixes and deriving alkyl name
    # "methanethiol" -> "meth" + "yl" -> "methyl"
    # "ethanethiol" -> "eth" + "yl" -> "ethyl"
    if name.endswith("anethiol"):
        base = name[:-len("anethiol")]  # "methanethiol" -> "meth"
        if base:
            return f"S-{base}yl"

    # Generic "thiol" suffix: "propanethiol" is already handled above
    # But "phenylthiol" or other forms:
    if name.endswith("thiol"):
        base = name[:-5]  # strip "thiol"
        if base:
            # If base ends in "ane": "propanethiol" -> "propane" -> not reached (handled above)
            # If base already looks like an alkyl form
            if base.endswith("yl"):
                return f"S-{base}"
            # Try converting what's left to alkyl
            # "propane" -> "propyl"  (if base is "propane")
            if base.endswith("ane"):
                return f"S-{base[:-3]}yl"
            if base.endswith("an"):
                return f"S-{base[:-2]}yl"
            return f"S-{base}yl"

    # If the name is already in some other form, try using _alcohol_to_alkyl
    # This handles cases where the fragment is named as an alcohol instead
    alkyl = _alcohol_to_alkyl(name)
    if alkyl and alkyl != name:
        return f"S-{alkyl}"

    return None


def _acid_to_sulfonamide(acid_name: str) -> Optional[str]:
    """Convert sulfonic acid name to sulfonamide form per IUPAC P-66.6.4.

    Replaces 'sulfonic acid' with 'sulfonamide' in the acid name.

    Args:
        acid_name: Sulfonic acid name (e.g., "benzenesulfonic acid").

    Returns:
        Sulfonamide parent name (e.g., "benzenesulfonamide"), or None.

    Examples:
        >>> _acid_to_sulfonamide("benzenesulfonic acid")
        'benzenesulfonamide'
        >>> _acid_to_sulfonamide("methanesulfonic acid")
        'methanesulfonamide'
    """
    name = acid_name.strip()
    if not name:
        return None

    # Direct conversion: "sulfonic acid" -> "sulfonamide"
    if name.endswith("sulfonic acid"):
        return name[:-len("sulfonic acid")] + "sulfonamide"

    # Already a sulfonamide name
    if name.endswith("sulfonamide"):
        return name

    return None


def _alcohol_to_alkyl(alcohol_name: str) -> str:
    """Convert alcohol name to alkyl prefix form.

    Handles both systematic (propan-1-ol -> propyl) and
    trivial (ethanol -> ethyl) names.

    Args:
        alcohol_name: Alcohol name or parent name.

    Returns:
        Alkyl prefix (e.g., "methyl", "ethyl", "propyl").

    Examples:
        >>> _alcohol_to_alkyl("methanol")
        'methyl'
        >>> _alcohol_to_alkyl("ethanol")
        'ethyl'
        >>> _alcohol_to_alkyl("propan-1-ol")
        'propyl'
        >>> _alcohol_to_alkyl("butan-1-ol")
        'butyl'
    """
    name = alcohol_name.strip()

    # Check trivial lookup first
    if name.lower() in _TRIVIAL_ALCOHOL_TO_ALKYL:
        return _TRIVIAL_ALCOHOL_TO_ALKYL[name.lower()]

    # If already in alkyl form (ends with "yl"), return as-is
    if name.endswith("yl"):
        return name

    # Systematic alcohol: "propan-1-ol" -> "propyl"
    # Strip locant + "-ol" suffix
    if "-ol" in name:
        # "propan-1-ol" -> remove "-1-ol" -> "propan" -> "propyl"
        # Find the base: everything before the first locant-ol
        import re
        # Match pattern: base-N-ol or baseol
        match = re.match(r'^(.+?)(?:-\d+)?-ol$', name)
        if match:
            base = match.group(1)
            # Convert "-an" ending to "-yl" (propan -> propyl)
            if base.endswith("an"):
                return base[:-2] + "yl"
            # Convert "-en"/"yn" endings too
            if base.endswith("en"):
                return base + "yl"
            if base.endswith("yn"):
                return base + "yl"
            return base + "yl"

    # Simple "-ol" suffix: "methanol" -> "methyl"
    if name.endswith("ol"):
        base = name[:-2]
        if base.endswith("an"):
            return base[:-2] + "yl"
        return base + "yl"

    # Alkane form: "methane" -> "methyl", "ethane" -> "ethyl"
    if name.endswith("ane"):
        return name[:-3] + "yl"

    # If it doesn't match known patterns, return as-is
    # (may already be a substituent name)
    return name


def _amine_to_prefix(amine_name: str) -> Optional[str]:
    """Convert amine name to substituent prefix for N-substitution.

    Args:
        amine_name: Amine name (e.g., "methylamine", "ethanamine").

    Returns:
        Prefix form (e.g., "methyl", "ethyl"), or None if complex.

    Examples:
        >>> _amine_to_prefix("methylamine")
        'methyl'
        >>> _amine_to_prefix("ethanamine")
        'ethyl'
        >>> _amine_to_prefix("aniline")
        'phenyl'
    """
    name = amine_name.strip()

    # Check trivial lookup first
    if name.lower() in _TRIVIAL_AMINE_TO_PREFIX:
        return _TRIVIAL_AMINE_TO_PREFIX[name.lower()]

    # Systematic: ends with "amine" -> strip and convert
    if name.endswith("amine"):
        base = name[:-5]  # "methylamine" -> "methyl"
        if not base:
            return None
        # If base already looks like a prefix (ends with "yl"), return it
        if base.endswith("yl"):
            return base

        # Systematic: "propan-1-amine" -> "propyl"
        # Strip locant + convert
        import re
        match = re.match(r'^(.+?)(?:-\d+)?$', base)
        if match:
            stem = match.group(1)
            if stem.endswith("an"):
                return stem[:-2] + "yl"
            if stem.endswith("en"):
                return stem + "yl"
            return stem + "yl"

    # If already a prefix form
    if name.endswith("yl"):
        return name

    return None
