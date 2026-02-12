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

from typing import Dict, Optional

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

    if (last.isalpha() or last == ')') and first.isdigit():
        return f"{left}-{right}"
    if (last.isalpha() or last == ')') and first.isupper():
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
    }

    assembler = assemblers.get(bond_type)
    if assembler is None:
        return None

    return assembler(fragment_names, style)


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

    # Get amine prefix (simple substituent name)
    amine_prefix = _amine_to_prefix(amine_name)

    if amine_prefix:
        # Simple amine: use amide suffix pattern
        # "N-methylacetamide", "N-ethylpropanamide"
        amide_name = _acid_to_amide(acid_name)
        if amide_name:
            return f"N-{_join_components(amine_prefix, amide_name)}"

    # Complex amine: use acyl prefix pattern
    # "N-acetylcyclohexanamine"
    acyl_prefix = _acid_to_acyl(acid_name)
    if acyl_prefix and amine_name:
        return f"N-{_join_components(acyl_prefix, amine_name)}"

    return None


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
