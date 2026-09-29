"""
Fragment assembly module for decomposition engine.

Combines individually-named fragments into correct multi-component IUPAC names.
Assembly patterns differ by bond type:
- Esters: "alkyl alkanoate" (e.g., "ethyl acetate")
- Amides: "N-[substituent][acid-amide]" (e.g., "N-methylacetamide")
- Glycosides: basic concatenation (full naming deferred to a phase)

Each assembler handles name transformations:
- acid name -> "-ate" form for esters
- acid name -> "-amide" form for amides
- acid name -> "-yl" (acyl) form for N-acyl naming
- alcohol name -> alkyl prefix form
"""

import re as _re
from collections import Counter as _Counter
from typing import Dict, List, Optional, Tuple

from rdkit import Chem as _Chem

from ..assembly.naming_utils import _wrap_n_substituent, enclose_if_compound
from ..data.sugar_names import (
    lookup_sugar,
    recognize_sugar_skeleton,
    sugar_to_glycoside_class_name,
    sugar_to_glycosyloxy_prefix,
)
from ..data.trivial_acids import TRIVIAL_ACID_TO_ACYLATE
from ..errors import is_refusal_sentinel

# a phase /: structural seniority guard primitives. The aglycone is in
# scope to flip to the functional-class form iff its principal characteristic
# group is hydroxy-class or junior (rank >= primary_alcohol). This is a
# structural decision via get_principal_group, NOT a string match on the name.
from ..perception.functional_groups import detect_functional_groups
from ..rules.seniority import SENIORITY_ORDER, get_principal_group

# The top of the hydroxy band. primary_alcohol (== 54) admits
# methanol/ethanol/2-aminoethanol/phenol; everything senior to hydroxy (ketone,
# aldehyde, acid,...) has a strictly smaller rank and is gated out (Pitfall 5:
# this is primary_alcohol, NOT phenol/57 -- 57 would wrongly block ethanol).
HYDROXY_TOP = SENIORITY_ORDER.index("primary_alcohol")


# ============================================================================
# Component joining helper
# ============================================================================

def _join_components(left: str, right: str) -> str:
    """Join two name components with IUPAC-compliant hyphenation.

    Inserts a hyphen when:
    - Left ends with letter/')' and right starts with digit
    - Left ends with letter/')' and right starts with uppercase letter
      (N-substitution, L/D/R/S stereodescriptors)

    This matches the logic in composer.py _join_prefix_to_name.

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
# Ring parent -> substituent prefix conversion (IUPAC
# ============================================================================

# When a fragment name is used as a substituent prefix (e.g., wrapped in
# parentheses), ring parent names must be converted to their substituent
# forms: "benzene" -> "phenyl", "naphthalene" -> "naphthyl", etc.
_RING_PARENT_TO_SUBSTITUENT = {
    "benzene": "phenyl",
    "naphthalene": "naphthyl",
    "anthracene": "anthryl",
    "phenanthrene": "phenanthryl",
    "toluene": "tolyl",
}


def _parent_to_substituent_prefix(name: str) -> str:
    """Convert ring parent names to substituent prefix form per IUPAC.

    When a fragment name ending in a ring parent name (e.g., "methoxybenzene")
    is used as a substituent prefix, the ring name must be converted to its
    substituent form (e.g., "methoxyphenyl").

    Only converts the LAST ring-parent-name token in the name to preserve
    any substituent prefixes already present (e.g., "methoxy" stays).

    Args:
        name: Fragment name that may contain a ring parent name.

    Returns:
        Name with ring parent converted to substituent form, or original
        name if no conversion applies.
    """
    for parent, sub in _RING_PARENT_TO_SUBSTITUENT.items():
        if name.endswith(parent):
            return name[:-len(parent)] + sub
    return name


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

# Acid-name endings whose acyl group ends in '-oyl' although the acid ends in
# '-ic' (not '-oic'), so the generic '-ic acid' -> '-yl' step would misspell
# them ('carbamyl', 'sulfamyl', 'butanimidyl'). Blue Book 2013:
# prefix list, "H2N-CO- carbamoyl (preferred prefix)"
# (the Blue Book); "H2N-C(=NH)- carbamimidoyl (preferred prefix)"
# (:17772); "H2N-CO-CO- oxamoyl (preferred prefix)" (:17784);
# "sulfamoyl (preselected prefix)" (:31330);
# "butanimidoyl (PIN)" (:40610).
# Shared by rules.lipids._acid_to_acyl and rules.radicals._acyl_from_acid_name.
ACID_ENDINGS_TO_OYL_ACYL = (
    ("carbamic acid", "carbamoyl"),
    ("sulfamic acid", "sulfamoyl"),
    ("oxamic acid", "oxamoyl"),
    ("imidic acid", "imidoyl"),
)

# ---------------------------------------------------------------------------
# OPSIN-expanded acid transformation tables
# ---------------------------------------------------------------------------

def _build_opsin_acid_lookups():
    """Build expanded acid transformation lookups from OPSIN carboxylicAcids.xml stems.

    OPSIN stems are short forms (e.g., 'form', 'acet', 'palmit') that need
    expansion to full acid name -> transformed form:
    - stem + "ic acid" -> stem + "ate" (trivial acids)
    - stem + "ic acid" -> stem + "amide" (trivial amides)
    - stem + "ic acid" -> stem + "oyl" or stem + "yl" (trivial acyl)

    Returns:
        Tuple of (ate_map, amide_map, acyl_map) dicts.
    """
    from ..data.opsin_imports.carboxylic_acids_opsin import OPSIN_ACID_STEMS

    ate_map = {}
    amide_map = {}
    acyl_map = {}

    for smiles, entry in OPSIN_ACID_STEMS.items():
        sub_type = entry.get('subType', '')
        for stem in entry['names']:
            # Build full acid name: stem + "ic acid"
            acid_name = f"{stem}ic acid"

            # ate form: stem + "ate"
            ate_map[acid_name] = f"{stem}ate"

            # amide form: stem + "amide"
            amide_map[acid_name] = f"{stem}amide"

            # acyl form depends on subType
            if sub_type == 'ylForAcyl':
                acyl_map[acid_name] = f"{stem}yl"
            else:
                # ylForYl and ylForNothing both use "oyl" suffix
                acyl_map[acid_name] = f"{stem}oyl"

    return ate_map, amide_map, acyl_map


# Build at module load time (cached)
_OPSIN_ATE, _OPSIN_AMIDE, _OPSIN_ACYL = _build_opsin_acid_lookups()


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
    # (the Blue Book): "C6H5-CH2- benzyl (preferred prefix)
    # phenylmethyl" -- the unsubstituted benzyl alcohol's organyl is 'benzyl'
    # ('phenylmethyl 2-(...)ethanoate' read worse and is not the preferred form).
    "phenylmethanol": "benzyl",
}

# Alcohol/parent -> alkoxy prefix (retained alkoxy names per IUPAC
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
    #: hydroxylamine (HO-NH2) as an N-substituent on an amide/sulfonamide
    # contributes the -OH prefix 'hydroxy', NOT 'hydroxyl'. Stripping the retained
    # name to 'hydroxyl' (its systematic-looking '...yl' tail) mis-spells the -OH
    # prefix -- 'hydroxyl' is not a prefix form. Gives N-hydroxymethanesulfonamide.
    "hydroxylamine": "hydroxy",
}


# ============================================================================
# Public API
# ============================================================================

def assemble_fragment_name(
    bond_type: str,
    fragment_names: Dict[str, str],
    style: str = "pin",
    fragment_smiles: Optional[Dict[str, str]] = None,
    parent_smiles: Optional[str] = None,
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
        fragment_smiles: Optional dict mapping fragment roles to their SMILES,
            keyed by the same side keys as fragment_names. a phase /:
            only the glycoside assembler consumes this (the sugar-skeleton
            deriver and the aglycone seniority guard both need structure); all
            other assemblers ignore it, keeping them byte-identical.

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

    # a phase /: the glycoside assembler needs the fragment SMILES to
    # derive the sugar skeleton and run the aglycone seniority guard. Special-
    # case it so every other assembler keeps its two-arg (fragment_names, style)
    # signature byte-identical.
    if bond_type == "glycosidic":
        return _assemble_glycoside(
            fragment_names, style, fragment_smiles=fragment_smiles,
            parent_smiles=parent_smiles,
        )

    # task 24: the amide assembler needs the amine SMILES for the acyl-float
    # ambiguity guard (a bare 'N-<acyl>-' onto an amine with >=2 acylatable N is
    # ambiguous). Special-cased like the glycoside above so every other assembler
    # keeps its two-arg signature byte-identical.
    if bond_type == "amide":
        return _assemble_amide(fragment_names, style, fragment_smiles=fragment_smiles)

    assemblers = {
        "ester": _assemble_ester,
        "amide": _assemble_amide,
        "carbamate": _assemble_carbamate,
        "ether": _assemble_ether,
        "thioester": _assemble_thioester,
        "phosphodiester": _assemble_phosphodiester,
        "sulfonamide": _assemble_sulfonamide,
        "thioether": _assemble_thioether,
        "sec_amine": _assemble_sec_amine,
    }

    assembler = assemblers.get(bond_type)
    if assembler is None:
        return None

    return assembler(fragment_names, style)


def _assemble_by_bond_type(
    named_fragments: List[Tuple],
    used_bond_types: set,
    style: str = "pin",
) -> Optional[str]:
    """Assemble named fragments using bond-type-specific assemblers.

    Routes each fragment pair through the appropriate assembler based on
    the bond type that connected them. This replaces the naive space-join
    in _try_iterative_mixed_decompose.

    Args:
        named_fragments: List of (frag_dict, name_str) tuples.
            Each frag_dict may have 'parent_bond_type' and 'role' keys.
        used_bond_types: Set of bond types used during cleavage.
        style: Naming style.

    Returns:
        Assembled name string, or None if assembly fails.
    """
    if not named_fragments or len(named_fragments) < 2:
        return None

    # Group fragments by their parent bond type
    typed_frags = {}  # bond_type -> list of (frag, name)
    untyped = []
    for frag, name in named_fragments:
        bt = frag.get('parent_bond_type', '')
        if bt:
            typed_frags.setdefault(bt, []).append((frag, name))
        else:
            untyped.append((frag, name))

    # If no typed fragments, return None (caller uses fallback)
    if not typed_frags:
        return None

    # For each bond type group, try to assemble using the type-specific assembler
    assembled_parts = []
    for bt, frags in typed_frags.items():
        if len(frags) < 2:
            # Single fragment for this bond type -- just use its name
            assembled_parts.append(frags[0][1])
            continue

        # Sort by role: acid/parent first, then alkyl/amine
        acid_frags = [f for f in frags if f[0].get('role') == 'acid']
        alkyl_frags = [f for f in frags if f[0].get('role') in ('alkyl', 'amine')]

        if not acid_frags:
            # Guess: longest name is probably the parent
            sorted_by_len = sorted(frags, key=lambda x: len(x[1]), reverse=True)
            acid_frags = [sorted_by_len[0]]
            alkyl_frags = sorted_by_len[1:]

        for acid_frag, acid_name in acid_frags:
            for alk_frag, alk_name in alkyl_frags:
                # Build fragment_names dict matching what assemblers expect
                fs = None
                if bt in ("amide", "sulfonamide", "sec_amine"):
                    fn = {"acid": acid_name, "amine": alk_name}
                    # a review review of 8d8c84f7: the floating-N-acyl ambiguity guard
                    # in `_assemble_amide` only fires when it can see the amine
                    # SMILES. The single-bond path threads it (engine.py:1354); this
                    # mixed-decomposition path did NOT, so the guard silently no-oped
                    # here and an ambiguous bare N-acyl could float ungated. Thread
                    # the amine/acid SMILES (available on the frag dicts) so the guard
                    # runs on this path too. Scoped to the amide family: glycosidic /
                    # ester assemblers read fragment_smiles differently and must stay
                    # on their existing no-SMILES back-compat path.
                    fs = {"acid": acid_frag.get("smiles"),
                          "amine": alk_frag.get("smiles")}
                elif bt == "glycosidic":
                    fn = {"sugar": acid_name, "aglycone": alk_name}
                else:
                    fn = {"acid": acid_name, "alkyl": alk_name}

                result = assemble_fragment_name(bt, fn, style, fragment_smiles=fs)
                if not result:
                    # The bond-type assembler cannot combine this pair. The old
                    # fallback blank-joined 'alk_name acid_name': two standalone
                    # names, not a name of the connected input. Void the whole
                    # assembly instead (Task 3, 2026-09-25).
                    return None
                assembled_parts.append(result)

    # Add any untyped fragments
    for _, name in untyped:
        assembled_parts.append(name)

    if not assembled_parts:
        return None

    if len(assembled_parts) == 1:
        return assembled_parts[0]

    # More than one part. The groups above are keyed on the bond type that was
    # CUT, not on how the parts connect, so nothing here knows how to join them
    # into ONE name. The old code blank-joined them "functional class style";
    # a functional class name has a fixed grammar, that
    # a list of independently assembled names does not follow. Measured on the
    # GPI mannoside: 4 parts glued into a string OPSIN reads as a different
    # formula (C38H75N2O31PS vs the input's C38H71N2O28PS). Void the assembly;
    # the molecule then gets its name from the next candidate.
    return None


# ============================================================================
# N-substituent grouping (IUPAC
# ============================================================================

def _group_n_substituents(name: str) -> str:
    """Group repeated N-prefix patterns in a decomposition-produced name.

    Detects patterns like "N-acetyl-N-acetyltetrahydropyranamine" and
    collapses them to "N,N-diacetyltetrahydropyranamine" using IUPAC
     multiplicative prefixes (di/tri for simple substituents,
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
        'N-acetyl-N,N-diformylpiperidine' # sorted alphabetically
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

    # seniority /: a mono-ester of a POLY-acid is
    # a PARTIAL ester — the un-esterified free -COOH is the senior principal group
    # (carboxylic acid > ester), so the ester functional-class 'alkyl...dicarboxylate'
    # is NOT the PIN. It also silently drops the 'hydrogen' the free acid needs, and
    # OPSIN then reads the name as an anion (-> RT-MISMATCH: 'ethyl benzene-1,2-
    # dicarboxylate' for ethyl hydrogen phthalate). Decline so the acid-senior path
    # names it, matching the chain analogue that is already correct ('6-ethoxy-6-oxo-
    # hexanoic acid'). A poly-acid fragment's systematic PIN name carries a multiplied
    # '{di,tri,tetra,penta}carboxylic acid' / '{di,tri,tetra,penta}…oic acid' suffix.
    _al = acid_name.lower()
    if any(p in _al for p in (
            "dicarboxylic acid", "tricarboxylic acid", "tetracarboxylic acid",
            "pentacarboxylic acid", "dioic acid", "trioic acid", "tetraoic acid",
            "pentaoic acid")):
        return None

    ate_name = _acid_to_ate(acid_name)
    alkyl_prefix = _alcohol_to_alkyl(alkyl_name)

    if not ate_name or not alkyl_prefix:
        return None

    return f"{alkyl_prefix} {ate_name}"


#: Suffix spellings of an acid with MORE than one acid group. Its acyl prefix
#: ('butanedioyl') and its amide ('butanediamide') convert EVERY acid group, so a
#: decomposition that cut only ONE of them names a different molecule.
_POLY_ACID_NAME_MARKERS = (
    "dicarboxylic acid", "tricarboxylic acid", "tetracarboxylic acid",
    "pentacarboxylic acid", "dioic acid", "trioic acid", "tetraoic acid",
    "pentaoic acid",
)


def _is_poly_acid_name(acid_name: str) -> bool:
    """True when ``acid_name`` names an acid with two or more acid groups."""
    low = (acid_name or "").lower()
    return any(m in low for m in _POLY_ACID_NAME_MARKERS)


def _expand_n_locant_for_multiplier(prefix: str) -> str:
    """Expand N-locant based on multiplier prefix on an N-substituent name.

    When an amine fragment is named with a multiplier (e.g., "dimethyl" from
    "dimethylamine"), the N-locant must repeat for each substituent on nitrogen:
      "dimethyl" -> "N,N-" (two methyls on N)
      "triethyl" -> "N,N,N-" (three ethyls on N)
      "methyl" -> "N-" (one methyl on N)

    IUPAC: When identical substituents on nitrogen, use N,N- prefix
    with multiplying prefix.

    Args:
        prefix: Substituent prefix (e.g., "dimethyl", "triethyl", "methyl").

    Returns:
        Appropriate N-locant string ("N-", "N,N-", "N,N,N-", etc.).
    """
    _MULT_MAP = {
        "di": 2, "tri": 3, "tetra": 4, "penta": 5,
    }
    # Simple alkyl/aryl suffixes that confirm the multiplier applies to
    # identical N-substituents (not e.g., "dichloro" which is one substituent
    # with two chlorines on it).
    _SIMPLE_SUFFIXES = (
        "methyl", "ethyl", "propyl", "butyl", "pentyl", "hexyl",
        "heptyl", "octyl", "nonyl", "decyl", "phenyl", "benzyl",
        "allyl", "vinyl",
    )
    lower = prefix.lower()
    for mult_prefix, count in _MULT_MAP.items():
        if lower.startswith(mult_prefix):
            rest = lower[len(mult_prefix):]
            if rest in _SIMPLE_SUFFIXES:
                return ",".join(["N"] * count) + "-"
    return "N-"


def _assemble_amide(fragment_names: Dict[str, str], style: str,
                    fragment_smiles: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Assemble amide name as 'N-[substituent][acid-amide]'.

    For simple amines, uses amide suffix: "N-methylacetamide".
    For complex amines, uses acyl prefix: "N-acetylcyclohexanamine".

    After assembly, passes the result through _group_n_substituents to
    collapse any repeated N-prefix patterns (e.g., "N-acetyl-N-acetyl..."
    becomes "N,N-diacetyl...") per IUPAC.

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

    # 2026-09-25 (pre-existing-failures plan, Task 4): both forms below convert
    # EVERY acid group of the acid fragment, but the decomposition cut ONE amide
    # bond. 'butanedioyl' is the DIVALENT group -CO-CH2-CH2-CO-,
    # the Blue Book), so the free CH2-COOH vanished:
    # 'N-({(2S)-2-[...]butanedioyl})(2S)-2-aminopropanoic acid' (TRIAGE row 17,
    # CHEBI:139249); 'butanediamide' makes the free acid an amide. The same guard
    # as the ester assembler's: the free acid is the senior class,
    #:18158, class 7 acids above amides), so neither amide form is the name.
    if _is_poly_acid_name(acid_name):
        return None

    result = None
    acyl_float = False

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
                # Check if prefix has a multiplier (di, tri, tetra) indicating
                # multiple identical N-substituents. In that case, expand the
                # N-locant: "dimethyl" -> "N,N-dimethyl", not "N-dimethyl".
                n_locant = _expand_n_locant_for_multiplier(amine_prefix)
                # A multiplied SIMPLE prefix ('dimethyl' -> 'N,N-'; the expander
                # only returns several N for di/tri/... + a simple substituent)
                # is cited bare: multiplies the simple prefix, and
                # (the Blue Book) encloses only compound/complex
                # prefixes -- enclose_if_compound reads 'dimethyl' as a
                # two-prefix compound and produced the unparseable
                # 'N,N-(dimethyl)acetamide' (TRIAGE g7 C10).
                if n_locant != "N-":
                    wrapped_prefix = amine_prefix
                else:
                    wrapped_prefix = _wrap_n_substituent(enclose_if_compound(amine_prefix))
                result = f"{n_locant}{_join_components(wrapped_prefix, amide_name)}"

    if result is None:
        # Complex amine: use acyl prefix pattern
        # "N-acetylcyclohexanamine"
        # task 24: refuse the bare 'N-<acyl>-' float when the amine fragment
        # has >=2 acylatable N -- the float would not say which N bears the acyl
        # (ambiguous name / no-jar fail-open). Same guard as engine.py's amide
        # branch. Needs the amine SMILES (threaded from assemble_fragment_name).
        _amine_smi = (fragment_smiles or {}).get("amine")
        if _amine_smi:
            from .engine import _amine_acyl_ambiguous
            if _amine_acyl_ambiguous(_amine_smi):
                return None
        acyl_prefix = _acid_to_acyl(acid_name)
        if acyl_prefix and amine_name:
            acyl_float = True
            if amine_name.startswith("N-") or amine_name.startswith("N,"):
                # Amine already has N-prefix(es) from recursive naming.
                # Insert the acyl as an additional N-substituent:
                # amine = "N-methylcyclohexanamine"
                # acyl = "acetyl"
                # -> "N-acetyl-N-methylcyclohexanamine"
                # This keeps each N-substituent as a separate "N-X" segment
                # so _group_n_substituents can properly merge identical ones.
                result = f"N-{_wrap_n_substituent(enclose_if_compound(acyl_prefix))}-{amine_name}"
            else:
                #: a parent-hydride-derived acyl prefix (furan-2-carbonyl,
                # cyclohexanecarbonyl) must be parenthesised so two parent hydrides
                # are not read as one -- 'N-(furan-2-carbonyl)furan-2-carboxamide',
                # not 'N-furan-2-carbonylfuran-2-carboxamide'. _wrap_n_substituent
                # only ESCALATES an existing enclosure, so add the base level first
                # via enclose_if_compound (idempotent for a simple acyl like acetyl),
                # mirroring the already-fixed amine_prefix sibling branch above.
                wrapped = _wrap_n_substituent(enclose_if_compound(acyl_prefix))
                result = f"N-{_join_components(wrapped, amine_name)}"

    if result is None:
        return None

    # Guard: multi-level decomposition can produce "N-N-" at the start
    # (outer level adds N-, inner level already starts with N-). This is
    # never valid IUPAC notation. Deduplicate to single "N-".
    while result.startswith("N-N-"):
        result = result[2:]  # strip one "N-"

    # Group repeated N-prefix patterns (IUPAC
    # This handles cases where recursive fragment naming already produced
    # an "N-acetyl..." prefix, and our assembly adds another "N-acetyl",
    # resulting in "N-acetyl-N-acetyl..." which should be "N,N-diacetyl...".
    if result.count("N-") >= 2:
        result = _group_n_substituents(result)

    # M2 Task 3 fail-closed splice guard: `amine_prefix`/`amine_name` can carry
    # the substituent cascade's refusal sentinel embedded in a decorated name
    # through unchanged (e.g. a recursively-named amine like
    # "N-substituentcyclohexanamine" -> amine_prefix "N-substituentcyclohexyl"
    # via the multiplier-expand + wrap-and-join above -- see
    # errors.is_refusal_sentinel's docstring for the measured leak). Check the
    # fully assembled result ONCE here rather than at each of the two branches
    # above, so neither can weave the placeholder into a shipped name.
    if is_refusal_sentinel(result):
        return None

    if acyl_float:
        # Decision A (user, 2026-09-26): the acyl-prefix float is refused (inside
        # _handle_peptide's systematic attempt) or demoted when the acylated N is not a
        # suffix nitrogen of the amine parent -- the alpha-amino N of 'glycine', a ring
        # N. Same gate as engine.py's amide branch (engine.gate_nonsuffix_nacyl_float).
        from .engine import gate_nonsuffix_nacyl_float
        result = gate_nonsuffix_nacyl_float(
            (fragment_smiles or {}).get("amine"), result, amine_name)

    return result


def _aglycone_to_substituent(
    aglycone_name: str, aglycone_smiles: Optional[str]
) -> Optional[str]:
    """Convert an aglycone fragment to a monovalent substituent prefix (/).

    The aglycone arrives from the glycoside cleavage as an *alcohol* (the
    glycosidic oxygen stays with it as -OH), so it is named e.g. ``methanol``,
    ``ethanol``, ``2-aminoethanol``, ``phenol``. For the functional-class
    glycoside form it must be cited as a preceding monovalent substituent word
    (``methyl``, ``ethyl``, ``2-aminoethyl``, ``phenyl``).

    Structural seniority guard (,: if the aglycone bears a
    characteristic group *senior to hydroxy* (ketone, aldehyde, acid,...), the
    functional-class glycoside form is NOT used -- the ``-ose`` ending is
    retained and the aglycone is cited as an O-substituent instead. We detect
    that structurally via ``get_principal_group`` (NOT by string-matching the
    name) and return ``None`` so the caller falls back to the legacy
    ``(glycosyloxy)R`` form. Returning ``None`` is also the response to any
    unconvertible aglycone (Tier-4-recursive / unparseable).

    Args:
        aglycone_name: The aglycone fragment name (an alcohol/phenol name).
        aglycone_smiles: The aglycone fragment SMILES (needed for the
            structural seniority decision). When absent, the guard cannot run
            and we fail closed (return None -> legacy form).

    Returns:
        The substituent prefix (e.g. "methyl", "phenyl", "2-aminoethyl"), or
        None when the aglycone is senior to hydroxy or cannot be converted.
    """
    if not aglycone_name:
        return None

    # Structural seniority guard . Without the SMILES we cannot make the
    # structural decision, so fail closed to the legacy form.
    if not aglycone_smiles:
        return None
    aglycone_mol = _Chem.MolFromSmiles(aglycone_smiles)
    if aglycone_mol is None:
        return None
    try:
        pg, _atoms = get_principal_group(
            aglycone_mol, detect_functional_groups(aglycone_mol)
        )
    except Exception:
        return None
    if pg is not None:
        # rank >= HYDROXY_TOP (primary_alcohol == 54) => hydroxy-class or junior
        # => in scope to flip. Anything senior to hydroxy => keep legacy form.
        rank = SENIORITY_ORDER.index(pg) if pg in SENIORITY_ORDER else 999
        if rank < HYDROXY_TOP:
            return None  # senior aglycone -> PROTECT, keep legacy

    # Convert the alcohol/phenol aglycone name to its substituent prefix.
    # Pitfall 4: use _alcohol_to_alkyl (NOT _amine_to_prefix) -- the aglycone
    # arrives as an alcohol (methanol->methyl, 2-aminoethanol->2-aminoethyl,
    # phenol->phenyl, ethanol->ethyl).
    prefix = _alcohol_to_alkyl(aglycone_name)
    if not prefix:
        return None
    # _alcohol_to_alkyl is best-effort and returns its INPUT unchanged when it
    # cannot convert (e.g. the diol "hydroquinone" -> "hydroquinone", which is
    # NOT a monovalent substituent prefix). A real substituent prefix ends in
    # "-yl"; if the converter could not produce one, fail closed to the legacy
    # (glycosyloxy)R form (zero-regression default) rather than emit a
    # malformed two-word name that the validity gate would then reject as
    # "unknown". All four in-scope aglycones convert to a -yl prefix
    # (methyl/ethyl/2-aminoethyl/phenyl/2-naphthyl).
    if not prefix.endswith("yl"):
        return None
    return prefix


def _aglycone_structural_substituent(aglycone_smiles: Optional[str]) -> Optional[str]:
    """Derive the aglycone monovalent substituent prefix from STRUCTURE (Incr-1a).

    Root-cause alternative to the string rule ``_alcohol_to_alkyl``, which FABRICATES
    an OPSIN-unparseable token for a retained-named aglycone (``borneol`` ->
    ``borneyl``, not an OPSIN substituent). The aglycone arrives from the glycoside
    cleavage as an alcohol (the glycosidic O reattached as -OH); we take the carbon
    that bears that single -OH as the free valence and name the remaining skeleton
    through the ring-substituent chokepoint ``name_ring_system_substituent``,
    yielding e.g. ``4,7,7-trimethylbicyclo[2.2.1]heptan-5-yl``.

    Determinism (a project rule): the aglycone is re-parsed through its CANONICAL SMILES
    first, so the von-Baeyer / ring numbering the chokepoint assigns is independent of
    the incoming atom order (that numbering is otherwise order-dependent).

    Structural seniority guard (,: a group senior to hydroxy keeps
    the substitutive form -> return None. Fail closed (None) on a multi-hydroxy,
    purely acyclic, or unnameable aglycone -- the caller then keeps its legacy path.
    """
    if not aglycone_smiles:
        return None
    mol = _Chem.MolFromSmiles(aglycone_smiles)
    if mol is None:
        return None
    # Determinism: canonicalize atom order before the chokepoint numbers the ring.
    mol = _Chem.MolFromSmiles(_Chem.MolToSmiles(mol))
    if mol is None:
        return None
    # Seniority guard (structural,) -- mirrors _aglycone_to_substituent so a
    # ketone/acid/aldehyde aglycone is never flipped to the functional-class form.
    try:
        pg, _atoms = get_principal_group(mol, detect_functional_groups(mol))
    except Exception:
        return None
    if pg is not None:
        rank = SENIORITY_ORDER.index(pg) if pg in SENIORITY_ORDER else 999
        if rank < HYDROXY_TOP:
            return None
    # Locate the single glycosidic -OH (degree-1 O carrying >=1 H, bonded to carbon).
    oh = [
        (a.GetIdx(), a.GetNeighbors()[0].GetIdx())
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 8 and a.GetDegree() == 1 and a.GetTotalNumHs() >= 1
        and a.GetNeighbors()[0].GetAtomicNum() == 6
    ]
    if len(oh) != 1:
        return None  # ambiguous / no single attachment -> fail closed
    o_idx, c_idx = oh[0]
    frag = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() != o_idx]
    from ..rules.ring_substituents import name_ring_system_substituent
    try:
        prefix = name_ring_system_substituent(mol, frag, c_idx, allow_mancude=True)
    except Exception:
        return None
    # Reject the unnameable sentinel and any non-substituent token; a real
    # monovalent prefix ends in "-yl".
    if not prefix or prefix == "substituent" or not prefix.endswith("yl"):
        return None
    return prefix


def _assemble_glycoside(
    fragment_names: Dict[str, str],
    style: str,
    fragment_smiles: Optional[Dict[str, str]] = None,
    parent_smiles: Optional[str] = None,
) -> Optional[str]:
    """Assemble glycoside name (a phase / -08: functional-class form).

    Emits the Blue-Book functional-class two-word form
    ``<aglycone-substituent> <sugar>oside``, e.g.
    ``methyl beta-D-glucopyranoside``, when the self-gating triad  holds:

      1. the sugar skeleton is recognized (lookup_sugar catalog fast-path FIRST
         per, then the structure-derived recognize_sugar_skeleton); AND
      2. the aglycone resolves to a monovalent substituent prefix without a
         group senior to hydroxy (/ via _aglycone_to_substituent); AND
      3. there is exactly one sugar unit (this single-glycosidic-bond assembler
         -- _assemble_multi_glycoside handles >=2, untouched).

    If ANY gate fails -- including the back-compat case where no fragment_smiles
    was threaded -- it falls through to the EXISTING legacy substitutive form
    ``(glycosyloxy)aglycone``. Zero regression is the default failure mode (,
    the project guarded-primitive standard).

    Accepts both key conventions from the decomposition engine:
    - Engine convention: {"acid": sugar_name, "alkyl": aglycone_name}
    - Explicit convention: {"sugar": sugar_name, "aglycone": aglycone_name}

    Args:
        fragment_names: Fragment name dict with sugar/aglycone info.
        style: Naming style.
        fragment_smiles: Optional dict of fragment SMILES keyed by side
            . Required for the functional-class flip; when None the
            legacy form is emitted (back-compat).

    Returns:
        Functional-class name like "methyl beta-D-glucopyranoside" when the
        triad holds, else the legacy "(beta-D-glucopyranosyloxy)phenol" form,
        or None if neither can be built.
    """
    # Accept both key conventions
    sugar_name = fragment_names.get("sugar") or fragment_names.get("acid")
    aglycone_name = fragment_names.get("aglycone") or fragment_names.get("alkyl")

    if not sugar_name or not aglycone_name:
        return None

    # --- Functional-class flip attempt (//) -------------------
    # Only attempted when the fragment SMILES were threaded . Every gate
    # failure falls through to the legacy logic below (zero regression).
    if fragment_smiles:
        sugar_smiles = fragment_smiles.get("sugar") or fragment_smiles.get("acid")
        aglycone_smiles = (
            fragment_smiles.get("aglycone") or fragment_smiles.get("alkyl")
        )

        # SUGAR gate : catalog fast-path FIRST, then the deriver on miss.
        sugar_tuple = None
        if sugar_smiles:
            sugar_mol = _Chem.MolFromSmiles(sugar_smiles)
            if sugar_mol is not None:
                try:
                    canonical = _Chem.MolToSmiles(sugar_mol)
                except Exception:
                    canonical = None
                if canonical:
                    sugar_tuple = lookup_sugar(canonical)
                if sugar_tuple is None:
                    sugar_tuple = recognize_sugar_skeleton(sugar_mol)

        # AGLYCONE gate (/): substituent prefix + structural seniority.
        aglycone_prefix = _aglycone_to_substituent(aglycone_name, aglycone_smiles)

        # SINGLE-SUGAR gate : this assembler is only reached for the
        # single-glycosidic-bond path (>=2 -> _assemble_multi_glycoside, which
        # is untouched). Defensive guard: a multi-sugar prefix would carry the
        # glycosyloxy stem more than once.
        single_sugar = (
            sugar_name.count("glycosyloxy")
            + sugar_name.count("pyranosyloxy")
            + sugar_name.count("furanosyloxy")
        ) <= 1

        if sugar_tuple is not None and single_sugar:
            anomer, config, base = sugar_tuple
            # W6-P1: a uronic sugar core must use the head form
            # "glucopyranosiduronic acid" (NOT the wrong "glucuronopyranoside"
            # sugar_to_glycoside_class_name would build) -> methyl
            # beta-D-glucopyranosiduronic acid. Fail-closed if the base is an
            # out-of-map uronic skeleton (uronic_glycoside_head returns None).
            if "urono" in base:
                from ..data.sugar_names import uronic_glycoside_head
                head = uronic_glycoside_head(anomer, config, base)
            else:
                head = sugar_to_glycoside_class_name(anomer, config, base)
            if head:
                # Functional-class form "<substituent> <sugar>oside";
                # the alpha/beta + D/L descriptors come from the sugar tuple and are
                # never dropped . Two aglycone-substituent sources:
                # - the string rule (`aglycone_prefix`), byte-identical for every
                # already-working glycoside (menthyl/cyclohexyl/phenyl/...);
                # - Incr-1a: the aglycone '-yl' derived from STRUCTURE, which
                # rescues a retained-named aglycone whose string token is
                # fabricated and OPSIN-unparseable (borneol -> 'borneyl').
                string_form = f"{aglycone_prefix} {head}" if aglycone_prefix else None
                # Incr-1a: the aglycone '-yl' derived from STRUCTURE
                # (canonicalised -> deterministic), used to (a) recognise when the
                # string rule can be trusted WITHOUT OPSIN, and (b) rescue a
                # fabricated string token. Only computed when the parent SMILES is
                # available to RT-gate against (single-bond path); the minority
                # _assemble_by_bond_type route (parent_smiles=None) is unchanged.
                structural_prefix = (
                    _aglycone_structural_substituent(aglycone_smiles)
                    if parent_smiles else None
                )
                structural_form = None
                if structural_prefix:
                    # Enclose a complex substituent (locants/brackets) per
                    #; a bare 'cyclohexyl'/'phenyl' stays unenclosed.
                    wrapped = (
                        structural_prefix
                        if _re.fullmatch(r"[a-z]+yl", structural_prefix)
                        else f"({structural_prefix})"
                    )
                    structural_form = f"{wrapped} {head}"

                # (A) FAST PATH, no OPSIN: trust the string form when the structural
                # derivation AGREES with it, or when there is no structural
                # alternative. This covers every already-working glycoside whose
                # token is a genuine substituent (cyclohexyl/phenyl/methyl/ethyl/
                #...): their OPSIN behaviour is UNCHANGED, so nothing perturbs the
                # rest of the naming run. (Back-compat: with no parent_smiles the
                # structural branch is off, so this is the sole path, as before.)
                #
                # ⚠ Case 2a (structural_prefix is None -> ship string_form unverified):
                # a simple/acyclic aglycone whose string token does NOT round-trip
                # still ships here without RT-gating -- exactly base's pre-existing
                # best-effort-tier exposure, neither introduced nor widened by this
                # change. Closing it (RT-gating the fast path) would re-introduce the
                # warm-cache OPSIN perturbation documented at branch B / in the
                # report, so it is deliberately DEFERRED (out of Incr-1a scope).
                if string_form is not None and (
                    structural_prefix is None
                    or aglycone_prefix == structural_prefix
                ):
                    return string_form

                # (B) The string prefix DISAGREES with structure -- either a valid
                # retained token whose systematic form differs (menthyl) or a
                # FABRICATED OPSIN-unparseable one (borneol -> 'borneyl'). Both
                # candidates describe the SAME aglycone but only one is OPSIN-valid.
                # 0-wrong is ABSOLUTE, so RT-select against the parent (full InChI)
                # and ship the first that round-trips; if neither does, fall through
                # (abstain). OPSIN is invoked ONLY here, off the working fast path.
                #
                # ⚠ JVM-CONTINGENT byte-identity: the COMPLEX already-working controls
                # (menthyl / decahydronaphthalenyl / tricyclo-decyl / 1-oxaspiro
                # glucoside) have a valid string token that DIFFERS from its systematic
                # form, so they reach HERE and now round-trip through OPSIN -- their
                # exact output is therefore pinned by the committed byte-identity
                # controls only when a JVM is present. Without a JVM, opsin_roundtrip_check
                # returns not-passed and these would regress to abstain. Acceptable per
                # the project's real-deployment stance (OPSIN spawns unconditionally, so
                # a no-JVM run is not a real deployment), but RECORDED here rather than
                # silently relied on. See the Incr-1a report, Finding 2.
                if parent_smiles:
                    from ..validation.opsin_roundtrip import opsin_roundtrip_check
                    for cand in (string_form, structural_form):
                        if cand and opsin_roundtrip_check(
                            parent_smiles, cand
                        ).get("passed"):
                            return cand
                elif string_form is not None:
                    return string_form

    # --- The legacy substitutive fallback is REMOVED (it could not be right) ---
    #
    # It used to emit ``_join_components(f"({sugar_prefix})", aglycone_name)``.
    # That is not a nomenclature operation: ``aglycone_name`` is the aglycone
    # named as a COMPLETE PARENT, and the glycosidic oxygen arrives from the
    # cleavage attached to it, so the aglycone is named as an alcohol/phenol.
    # Gluing the sugar prefix onto that name therefore expressed the glycosidic
    # oxygen TWICE -- once as the aglycone's own hydroxy and once inside the
    # '...osyloxy' prefix -- and, because the aglycone name is already complete,
    # the prefix could carry no attachment locant. The result denotes a molecule
    # with an extra OH and no linkage, which is why these scored
    # ``constitution_mismatch`` rather than merely reading oddly.
    #
    # This is unconditional, not a bad case among good ones: the functional-class
    # branch above is the only one that consumes the glycosidic oxygen (via
    # ``_alcohol_to_alkyl``: phenol -> phenyl), so every path that reaches HERE
    # still has it doubled. Measured on the a dev split split: the branch produced 19
    # names, 18 of them final, and NONE was correct -- it also wrapped whole
    # parent names in parentheses as pseudo-prefixes, e.g.
    # '((3S)-butan-3-ol)(8S)-...' and
    # '((3R,4S,6R)-3,4,5,6-tetrahydroxyoxane-2-carboxylic acid)apigenin'.
    # All 19 were already abstentions under the shipped configuration (both OPSIN
    # gates reject them), so no shipped name is lost by refusing here.
    #
    # Failing closed lets the dispatcher fall through to the GENERAL pipeline,
    # where the sugar is named as the compound prefix by
    # ``sugar_names.glycosyl_substituent_prefix`` (substituent cascade Tier
    # 1.75) and the ordinary parent+prefix machinery assigns the attachment
    # locant and cites only the remaining hydroxy groups -- the oxygen is then
    # expressed exactly once. That is the correct construction; this branch had
    # no way to reach it, because it never sees the aglycone's locants.
    return None


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


_POSITION_INVARIANT_PARENTS = frozenset({
    # every substitutable position equivalent — an unlocanted prefix join is
    # unambiguous (Wave-2 C2; the methoxybenzene policy gold lives here)
    'benzene', 'methane', 'ethane',
})


def _parent_is_position_invariant(parent_name: str) -> bool:
    """True when *parent_name* is a bare parent whose substitutable positions
    are all equivalent: no locant needed). Unsubstituted
    monocycloalkanes qualify; anything carrying locants, substituents or a
    positional suffix (benzonitrile, phenol, naphthalene) does not."""
    import re as _re
    if parent_name in _POSITION_INVARIANT_PARENTS:
        return True
    return bool(_re.fullmatch(r'cyclo[a-z]+ane', parent_name))


def _assemble_ether(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble ether name as 'alkoxy + parent' (IUPAC.

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

    # Wave-2 C2: this join carries NO attachment locant, so it is
    # only correct when every position of the parent is equivalent. Joining
    # onto a positional parent silently mis-placed the ether
    # ('phenoxybenzonitrile' for the 2-isomer; meta/para -suppressed).
    # Decline otherwise — the cascade falls to the benzene handler, which
    # emits the locanted form.
    if not _parent_is_position_invariant(parent_name):
        return None

    return _join_components(alkoxy, parent_name)


def _assemble_thioether(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble thioether name as 'alkylthio + parent' (IUPAC.

    The smaller fragment (alkyl side, retaining the sulfur as a thiol)
    is converted to an alkylthio prefix. The larger fragment (acid side)
    is the parent compound.

    Args:
        fragment_names: {"acid": parent_name, "alkyl": alkyl_name}
            where alkyl_name may be a thiol (the S stayed with it).
        style: Naming style.

    Returns:
        Thioether name like "methylthiobenzene", or None.
    """
    parent_name = fragment_names.get("acid")
    alkyl_name = fragment_names.get("alkyl")

    if not parent_name or not alkyl_name:
        return None

    alkylthio = _alcohol_to_alkylthio(alkyl_name)
    if not alkylthio:
        return None

    # Wave-2 C2: same locant-blind join as _assemble_ether — see there.
    if not _parent_is_position_invariant(parent_name):
        return None

    return _join_components(alkylthio, parent_name)


def _assemble_sec_amine(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble secondary amine name as 'alkylamino + parent' (IUPAC.

    The smaller fragment (amine side) is converted to an alkylamino prefix.
    The larger fragment (acid side) is the parent compound.

    Args:
        fragment_names: {"acid": parent_name, "amine": amine_name}
            where amine_name is the N-bearing fragment.
        style: Naming style.

    Returns:
        Secondary amine name like "methylaminobenzoic acid", or None.
    """
    parent_name = fragment_names.get("acid")
    amine_name = fragment_names.get("amine") or fragment_names.get("alkyl")

    if not parent_name or not amine_name:
        return None

    amino_prefix = _amine_to_prefix(amine_name)
    if not amino_prefix:
        return None

    return _join_components(amino_prefix, parent_name)


def _assemble_thioester(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble thioester name as 'S-alkyl alkanethioate' per IUPAC.

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


#: the acid-fragment name of a polyphosphoric acid (preselected
# names, the Blue Book:36901) -> the '<hydrogen> <anion>' words of its
# monoester (one ester group; the rest of the acid hydrogens stay).
_POLYPHOSPHORIC_MONOESTER_WORDS = {
    "diphosphoric acid": "trihydrogen diphosphate",
    "triphosphoric acid": "tetrahydrogen triphosphate",
}


def _assemble_phosphodiester(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble phosphodiester name per IUPAC compositional nomenclature.

    The acid fragment is a phosphoric acid monoester (named by phosphorus.py
    after capping). The alkyl fragment name identifies the other ester group.
    Produces: "alkyl [acid-fragment-name]".

    Args:
        fragment_names: {"acid": acid_name, "alkyl": alkyl_name}
        style: Naming style.

    Returns:
        Phosphodiester name, or None.

     "Esters of mononuclear noncarbon oxoacids" (BB:35918): "Partial
    acid esters of polybasic acids are named by citing alkyl groups, aryl
    groups, etc. as separate words, in alphanumeric order if more than one,
    followed by the word 'hydrogen' (with the appropriate multiplying prefix, as
    necessary) also cited as a separate word, and the name of the appropriate
    anion." BB:35940 'methyl dihydrogen phosphate (PIN)' (one ester group) and
    :35944 'sodium methyl hydrogen phosphate (PIN)' (two groups, one H).

    So the acid fragment must itself be the monoester '<R> dihydrogen
    phosphate', and the diester is '<A> <B> hydrogen phosphate'. Anything else
    is declined (None). The old body put the alkyl word in front of whatever
    the acid fragment was called: 'methyl methyl dihydrogen phosphate' (its
    own docstring example; OPSIN cannot parse it) or, for an acid fragment
    named substitutively, a glue of two standalone names --
    'ethyl 2-(phosphonooxy)ethan-1-amine', which OPSIN parses to a DIFFERENT
    molecule (C(C)C(COP(=O)(O)O)N; the GPI mannoside's single-bond
    decomposition, TRIAGE T3). Two identical groups take the multiplying
    prefix: (BB:4857) "Identical simple substituent groups are
    indicated by multiplicative prefixes, such as 'di', 'tri', etc.... For
    compound or complex substituent groups... the multiplicative prefixes
    'bis', 'tris', 'tetrakis-', etc.... are used"; BB:35946 'dimethyl
    phosphonate (PIN)'.

    Examples:
        >>> _assemble_phosphodiester({"acid": "methyl dihydrogen phosphate", "alkyl": "ethanol"}, "pin")
        'ethyl methyl hydrogen phosphate'
        >>> _assemble_phosphodiester({"acid": "methyl dihydrogen phosphate", "alkyl": "methanol"}, "pin")
        'dimethyl hydrogen phosphate'
        >>> _assemble_phosphodiester({"acid": "2-(phosphonooxy)ethan-1-amine", "alkyl": "ethanol"}, "pin") is None
        True
    """
    from ..assembly.naming_utils import alpha_sort_key

    acid_name = fragment_names.get("acid")
    alkyl_name = fragment_names.get("alkyl")
    if not acid_name or not alkyl_name:
        return None

    # "Esters of polynuclear noncarbon oxoacids" (the Blue Book
    #:36917): "Partial (acid) esters... are named by the procedures for
    # neutral esters and acid salts, except that the name 'hydrogen' denoting
    # acid hydrogen atoms is indicated by the separate word 'hydrogen' (with
    # the appropriate multiplying prefix denoting multiplicity) inserted
    # between the name of... the organic group and the name of the anion"
    # ('methyl hydrogen dithiohypodiphosphonite (PIN)',:36921). A monoester of
    # diphosphoric acid (preselected name,:36901) keeps three acid hydrogens:
    # '<R> trihydrogen diphosphate' (cf. 'xanthosine 3'-(trihydrogen
    # diphosphate)',:55031). The old body glued '<R> diphosphoric acid' (two
    # standalone names; Task 3 of 16f45443a declined it), which left the
    # polyprenyl diphosphates past the general engine's reach unnamed.
    polyphosphate = _POLYPHOSPHORIC_MONOESTER_WORDS.get(acid_name)
    if polyphosphate is not None:
        # An unsaturated '-en-1-ol'/'-yn-1-ol' keeps its free-valence locant,
        # 'prop-2-en-1-yl', which the contracting
        # _alcohol_to_alkyl drops ('prop-2-enyl').
        unsat = _re.match(r'^(.+?(?:en|yn))-1-ol$', alkyl_name.strip())
        group = (f"{unsat.group(1)}-1-yl" if unsat
                 else _alcohol_to_alkyl(alkyl_name))
        if (not group or " " in group or not group.endswith("yl")
                or is_refusal_sentinel(group)):
            return None
        return f"{group} {polyphosphate}"

    monoester_word = " dihydrogen phosphate"
    if not acid_name.endswith(monoester_word):
        return None
    first = acid_name[:-len(monoester_word)]
    # `_alcohol_to_alkyl` returns its input unchanged when it matches no
    # alcohol pattern, so both words must BE group words ('...yl'), or the
    # glue comes back by another route.
    second = _alcohol_to_alkyl(alkyl_name)
    if (not first or not second or " " in first or " " in second
            or not first.endswith("yl") or not second.endswith("yl")
            or is_refusal_sentinel(first) or is_refusal_sentinel(second)):
        return None
    if first == second:
        enclosed = enclose_if_compound(first)
        groups = f"di{first}" if enclosed == first else f"bis{enclosed}"
    else:
        groups = " ".join(sorted((first, second), key=alpha_sort_key))
    return f"{groups} hydrogen phosphate"


def _assemble_sulfonamide(fragment_names: Dict[str, str], style: str) -> Optional[str]:
    """Assemble sulfonamide as 'N-substituent (parent)sulfonamide' per.

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
        wrapped = _wrap_n_substituent(amine_prefix)
        return f"N-{_join_components(wrapped, sulfonamide_parent)}"

    return sulfonamide_parent


def _assemble_multi_ester(
    named_fragments: List[Tuple[Dict, str]],
    style: str = "pin",
) -> Optional[str]:
    """Assemble name for polyesters (triglycerides, etc.) per IUPAC.

    For polyol + multiple acids:
    - Identical acids: "glycerol triacetate" with multiplicative prefix
    - Different acids: "glycerol acetate propanoate" positionally listed

    The core fragment is identified by role label preference:
    1. Fragments with side="middle" (backbone between cleavage points)
    2. Fall back to score_fragment_seniority (existing behavior)

    Args:
        named_fragments: List of (fragment_dict, name_str) tuples. Each
            fragment_dict has "smiles" and "side" keys. Preserves duplicates.
        style: Naming style ("pin" for preferred IUPAC names).

    Returns:
        Multi-ester name like "glycerol triacetate", or None if assembly fails.
    """
    from .fragment_ranker import score_fragment_seniority

    if not named_fragments or len(named_fragments) < 2:
        return None

    # Per,: identify core by role, then by seniority
    middle_frags = [(f, n) for f, n in named_fragments if f.get("side") == "middle"]
    if middle_frags:
        # Per: multiple middles -> pick most senior
        core_frag, core_name = min(
            middle_frags,
            key=lambda pair: score_fragment_seniority(pair[0]["smiles"]),
        )
    else:
        # No middle label: fall back to seniority over unique SMILES (existing behavior)
        # Build unique representatives for seniority scoring
        seen_smiles = {}
        for f, n in named_fragments:
            if f["smiles"] not in seen_smiles:
                seen_smiles[f["smiles"]] = (f, n)
        if len(seen_smiles) < 2:
            return None  # Need at least core + one acid by unique SMILES
        core_frag, core_name = min(
            seen_smiles.values(),
            key=lambda pair: score_fragment_seniority(pair[0]["smiles"]),
        )

    if not core_name:
        return None

    # Core-size guard (a phase-05): reject when core is smaller than any non-core
    core_mol = _Chem.MolFromSmiles(core_frag["smiles"])
    if core_mol is not None:
        core_ha = core_mol.GetNumHeavyAtoms()
        for frag, name in named_fragments:
            if frag is core_frag:
                continue
            nc_mol = _Chem.MolFromSmiles(frag["smiles"])
            if nc_mol is not None and nc_mol.GetNumHeavyAtoms() > core_ha:
                return None

    # Per: collect acid names from non-core fragments, count by name (not SMILES)
    ate_names = []
    for frag, name in named_fragments:
        if frag is core_frag:
            continue
        ate = _acid_to_ate(name)
        if ate:  # Per: None from _acid_to_ate is skipped
            ate_names.append(ate)

    if not ate_names:
        return None

    # Identical acids: use multiplicative prefix (per: compare names not SMILES)
    if len(set(ate_names)) == 1:
        _MULT_PREFIX = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta"}
        mult = _MULT_PREFIX.get(len(ate_names), str(len(ate_names)))
        return f"{core_name} {mult}{ate_names[0]}"

    # Different acids: list each positionally
    return f"{core_name} {' '.join(ate_names)}"


def _assemble_multi_glycoside(
    named_fragments: List[Tuple[Dict, str]],
    style: str = "pin",
) -> Optional[str]:
    """Assemble names for molecules with 2+ glycosidic bonds.

    Pattern: (sugar1-oxy)(sugar2-oxy)aglycone per IUPAC.
    Each sugar fragment should resolve to a retained glycosyloxy prefix.
    If NO sugar fragment gets a retained name (all end up as systematic
    oxanyloxane), fall back to None to prevent garbled names.

    The core fragment is identified by role label preference:
    1. Fragments with side="middle" (backbone between cleavage points)
    2. Fall back to non-sugar seniority ranking (existing behavior)

    Args:
        named_fragments: List of (fragment_dict, name_str) tuples. Each
            fragment_dict has "smiles" and "side" keys. Preserves duplicates.
        style: Naming style.

    Returns:
        Multi-glycoside name like "bis(beta-D-glucopyranosyloxy)phenol",
        or None if assembly fails.
    """
    from .fragment_ranker import score_fragment_seniority

    if not named_fragments or len(named_fragments) < 2:
        return None

    # Identify sugar vs aglycone fragments from the named tuples.
    # Sugar fragments are identified by: name ending in "oxy" OR
    # sugar lookup succeeding on the SMILES.
    sugar_indices = set()
    for i, (frag, name) in enumerate(named_fragments):
        if name.endswith("oxy"):
            sugar_indices.add(i)
        elif lookup_sugar(frag["smiles"]):
            sugar_indices.add(i)

    # Per,: identify core by role first, then by non-sugar seniority
    middle_frags = [(f, n) for f, n in named_fragments if f.get("side") == "middle"]
    if middle_frags:
        # Per: multiple middles -> pick most senior
        core_frag, core_name = min(
            middle_frags,
            key=lambda pair: score_fragment_seniority(pair[0]["smiles"]),
        )
    else:
        # No middle label: use sugar/non-sugar classification
        non_sugar_pairs = [
            (f, n) for i, (f, n) in enumerate(named_fragments)
            if i not in sugar_indices
        ]
        if non_sugar_pairs:
            # Build unique representatives for seniority scoring
            seen_smiles = {}
            for f, n in non_sugar_pairs:
                if f["smiles"] not in seen_smiles:
                    seen_smiles[f["smiles"]] = (f, n)
            core_frag, core_name = min(
                seen_smiles.values(),
                key=lambda pair: score_fragment_seniority(pair[0]["smiles"]),
            )
        else:
            # All fragments are sugars -- pick the most senior as core
            seen_smiles = {}
            for f, n in named_fragments:
                if f["smiles"] not in seen_smiles:
                    seen_smiles[f["smiles"]] = (f, n)
            core_frag, core_name = min(
                seen_smiles.values(),
                key=lambda pair: score_fragment_seniority(pair[0]["smiles"]),
            )

    if not core_name:
        return None

    # Collect glycosyloxy prefixes from non-core fragments
    glycosyloxy_prefixes: List[str] = []
    has_retained_sugar = False

    for frag, name in named_fragments:
        if frag is core_frag:
            continue

        # Check if the name is a glycosyloxy prefix (ends in "oxy")
        if name.endswith("oxy"):
            glycosyloxy_prefixes.append(name)
            has_retained_sugar = True
        else:
            # Try to convert to glycosyloxy prefix via sugar lookup
            sugar_info = lookup_sugar(frag["smiles"])
            if sugar_info:
                anomer, config, base_name = sugar_info
                prefix = sugar_to_glycosyloxy_prefix(anomer, config, base_name)
                if prefix:
                    glycosyloxy_prefixes.append(prefix)
                    has_retained_sugar = True
                else:
                    glycosyloxy_prefixes.append(name)
            else:
                glycosyloxy_prefixes.append(name)

    if not glycosyloxy_prefixes:
        return None

    # Pitfall 4: if no sugar fragment got a retained name, fall back
    # to prevent garbled "oxanyloxy-oxanyloxy" names
    if not has_retained_sugar:
        return None

    # Handle identical sugars with multiplicative prefix
    prefix_counts = _Counter(glycosyloxy_prefixes)
    _MULT_PREFIX = {1: "", 2: "bis", 3: "tris", 4: "tetrakis", 5: "pentakis"}

    if len(prefix_counts) == 1:
        # All identical sugars
        prefix, count = list(prefix_counts.items())[0]
        mult = _MULT_PREFIX.get(count, str(count))
        if mult:
            assembled_prefix = f"{mult}({prefix})"
        else:
            assembled_prefix = f"({prefix})"
    else:
        # Different sugars: wrap each in parens
        assembled_parts = []
        for prefix in glycosyloxy_prefixes:
            assembled_parts.append(f"({prefix})")
        assembled_prefix = "".join(assembled_parts)

    return _join_components(assembled_prefix, core_name)


def _assemble_multi_amide(
    named_fragments: List[Tuple[Dict, str]],
    style: str = "pin",
) -> Optional[str]:
    """Assemble names for molecules with 3+ amide bonds.

    Pattern: N-acyl1,N-acyl2-amine per IUPAC.
    Each acyl group as N- prefix on the amine core.

    The core fragment is identified by role label preference:
    1. Fragments with side="middle" (amine backbone in polyamides)
    2. Fall back to non-acid seniority ranking (existing behavior)

    Args:
        named_fragments: List of (fragment_dict, name_str) tuples. Each
            fragment_dict has "smiles" and "side" keys. Preserves duplicates.
        style: Naming style.

    Returns:
        Multi-amide name like "N,N,N-triacetylcyclohexanamine",
        or None if assembly fails.
    """
    from .fragment_ranker import score_fragment_seniority

    if not named_fragments or len(named_fragments) < 2:
        return None

    # Per,: identify core by role first, then by non-acid seniority
    middle_frags = [(f, n) for f, n in named_fragments if f.get("side") == "middle"]
    if middle_frags:
        # Per: multiple middles -> pick most senior
        core_frag, core_name = min(
            middle_frags,
            key=lambda pair: score_fragment_seniority(pair[0]["smiles"]),
        )
    else:
        # No middle label: identify acid fragments by name and pick non-acid core
        acid_indices = set()
        for i, (frag, name) in enumerate(named_fragments):
            n = name.strip().lower()
            if n.endswith("acid") or n.endswith("ate"):
                acid_indices.add(i)

        non_acid_pairs = [
            (f, n) for i, (f, n) in enumerate(named_fragments)
            if i not in acid_indices
        ]
        if non_acid_pairs:
            # Build unique representatives for seniority scoring
            seen_smiles = {}
            for f, n in non_acid_pairs:
                if f["smiles"] not in seen_smiles:
                    seen_smiles[f["smiles"]] = (f, n)
            core_frag, core_name = min(
                seen_smiles.values(),
                key=lambda pair: score_fragment_seniority(pair[0]["smiles"]),
            )
        else:
            # All fragments look like acids -- pick the most senior as core
            seen_smiles = {}
            for f, n in named_fragments:
                if f["smiles"] not in seen_smiles:
                    seen_smiles[f["smiles"]] = (f, n)
            core_frag, core_name = min(
                seen_smiles.values(),
                key=lambda pair: score_fragment_seniority(pair[0]["smiles"]),
            )

    if not core_name:
        return None

    # Collect acyl prefixes from non-core fragments
    acyl_names: List[str] = []
    for frag, name in named_fragments:
        if frag is core_frag:
            continue
        acyl = _acid_to_acyl(name)
        if acyl:  # Per: None from _acid_to_acyl is skipped
            acyl_names.append(acyl)

    if not acyl_names:
        return None

    # Group identical acyls for multiplicative prefix
    acyl_counts = _Counter(acyl_names)
    _MULT_PREFIX = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta"}

    # Build the N-prefix assembly
    n_parts: List[str] = []
    for acyl, count in sorted(acyl_counts.items()):
        mult = _MULT_PREFIX.get(count, str(count))
        n_locant = ",".join(["N"] * count)
        n_parts.append(f"{n_locant}-{mult}{acyl}")

    if not n_parts:
        return None

    # Join N-prefix parts with hyphens and append core name
    prefix_str = "-".join(n_parts)
    result = _join_components(prefix_str, core_name)

    # Apply N-substituent grouping
    if result.count("N-") >= 2:
        result = _group_n_substituents(result)

    return result


def _alcohol_to_alkoxy(name: str) -> Optional[str]:
    """Convert an alcohol or fragment name to its alkoxy form.

    Uses retained alkoxy names first (IUPAC, then falls back
    to converting via _alcohol_to_alkyl + replacing -yl with -oxy.

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

    # Fallback: convert alcohol -> alkyl -> alkoxy via the shared BB-verbatim
    # morphology primitive: the retained set contracts (methoxy/ethoxy/propoxy/
    # butoxy, + substituted primary members), a ring / C5+ / locant-bearing free
    # valence keeps the whole '-yl' ('cyclohexyloxy', 'pentyloxy', '(oxan-2-yl)oxy'
    # -- never the mangled 'cyclohexoxy'/'pentoxy'/'oxan-2-oxy'), a decorated
    # benzene -> 'phenoxy' (F-spell-oxy).
    from ..assembly.substituent_enumerator import alkoxy_prefix_from_substituent
    alkyl = _alcohol_to_alkyl(name)
    if alkyl and alkyl.endswith("yl"):
        candidate = alkoxy_prefix_from_substituent(alkyl)
        if candidate and candidate not in ("oxy", "()oxy"):  # never bare "oxy"
            return candidate

    # a phase fallback: for complex fragments where _alcohol_to_alkyl fails,
    # attempt direct suffix conversion. If the name already ends in "yl"
    # (from decomposition engine naming), convert to "oxy".
    if stripped.endswith("yl") and len(stripped) > 2:
        candidate = alkoxy_prefix_from_substituent(stripped)
        if candidate and candidate != "oxy":  # Guard: never return bare "oxy"
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


def _alcohol_to_alkylthio(name: str) -> Optional[str]:
    """Convert a thiol or fragment name to its alkylthio prefix form.

    For thioether naming, the smaller fragment (which retains the sulfur)
    is converted to an alkylthio prefix: "methanethiol" -> "methylthio",
    "ethanol" -> "ethylthio" (if S stayed with this fragment).

    Args:
        name: Thiol or fragment name (e.g., "methanethiol", "ethanethiol",
              "methanol" for cases where fragment was named as alcohol).

    Returns:
        Alkylthio prefix (e.g., "methylthio", "ethylthio"), or None.
    """
    stripped = name.strip().lower()

    # Direct thiol conversions
    if stripped.endswith("anethiol"):
        base = stripped[:-len("anethiol")]
        if base:
            return f"{base}ylthio"

    if stripped.endswith("thiol"):
        base = stripped[:-5]
        if base and base.endswith("ane"):
            return f"{base[:-3]}ylthio"
        if base and base.endswith("an"):
            return f"{base[:-2]}ylthio"
        if base:
            return f"{base}ylthio"

    # Fallback: convert via alkyl form
    alkyl = _alcohol_to_alkyl(name)
    if alkyl and alkyl.endswith("yl"):
        return f"{alkyl}thio"

    return None


# ============================================================================
# Pre-validation helpers (through)
# ============================================================================

def _looks_like_acid_name(name: str) -> bool:
    """Check if a name looks like it could be an acid name.

    Returns True for names containing recognizable acid patterns:
    - Ends in " acid" (systematic or trivial acid names)
    - Ends in "oic" (stem of systematic acid, e.g., "propanoic")
    - Ends in "ic" and len > 4 (stem of trivial acid, e.g., "acetic")
    - Ends in "carboxylic" (ring-attached acid)
    - Is in the trivial acid lookup table

    Per: used as pre-validation before any suffix transformation.
    """
    n = name.strip().lower()
    if n.endswith(" acid"):
        return True
    if n.endswith("oic") or n.endswith("carboxylic"):
        return True
    if n.endswith("ic") and len(n) > 4:
        return True
    # Check trivial lookup (handles names like "acetic", "formic", "propionic")
    try:
        if n in TRIVIAL_ACID_TO_ACYLATE:
            return True
    except Exception:
        pass
    return False


def _looks_like_convertible_name(name: str) -> bool:
    """Check if a name can be converted to an acyl form.

    Broader than _looks_like_acid_name: also accepts amide names
    ("benzamide", "propanamide") and ester names ("propanoate", "acetate")
    since _acid_to_acyl handles all three input types.

    Per Open Question 3 in RESEARCH.md: _acid_to_acyl has a broader input
    contract than the other three transformation functions.
    """
    if _looks_like_acid_name(name):
        return True
    n = name.strip().lower()
    # Amide patterns
    if n.endswith("amide"):
        return True
    # Ester patterns
    if n.endswith("ate") and " acid" not in n:
        return True
    # N-substituted amide (e.g., "N-methylbenzamide")
    if "amide" in n:
        return True
    return False


# ============================================================================
# Name transformation helpers
# ============================================================================

def _acid_to_ate(acid_name: str) -> Optional[str]:
    """Convert acid name to '-ate' form for ester naming.

    Handles both systematic names (propanoic acid -> propanoate) and
    trivial names (acetic acid -> acetate).

    Args:
        acid_name: Full acid name (e.g., "acetic acid", "propanoic acid").

    Returns:
        The '-ate' form (e.g., "acetate", "propanoate"), or None if input
        is not a recognized acid name (per).

    Examples:
        >>> _acid_to_ate("acetic acid")
        'acetate'
        >>> _acid_to_ate("propanoic acid")
        'propanoate'
        >>> _acid_to_ate("benzoic acid")
        'benzoate'
        >>> _acid_to_ate("cyclohexanecarboxylic acid")
        'cyclohexanecarboxylate'
        >>> _acid_to_ate("ethanol") # non-acid -> None
    """
    name = acid_name.strip()

    # Guard: if already in -ate form, don't convert again
    if name.endswith("ate") and " acid" not in name:
        return name

    # Per: pre-validate that input looks like an acid name
    if not _looks_like_acid_name(name):
        return None  # Per: not an acid -- don't fabricate

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

    # OPSIN expanded lookup
    if name.lower() in _OPSIN_ATE:
        return _OPSIN_ATE[name.lower()]

    # Systematic: "-oic acid" -> "-oate"
    if stem.endswith("oic"):
        return stem[:-2] + "ate"  # propanoic -> propanoate

    # Generic "-ic" -> "-ate"
    if stem.endswith("ic"):
        return stem[:-2] + "ate"

    # Per: no recognized pattern matched -- return None instead of garbage
    return None


def _acid_to_amide(acid_name: str) -> Optional[str]:
    """Convert acid name to '-amide' form.

    Handles systematic (propanoic acid -> propanamide) and
    trivial (acetic acid -> acetamide) names.

    Args:
        acid_name: Full acid name.

    Returns:
        The '-amide' form, or None if input is not a recognized acid name
        (per).

    Examples:
        >>> _acid_to_amide("acetic acid")
        'acetamide'
        >>> _acid_to_amide("propanoic acid")
        'propanamide'
        >>> _acid_to_amide("benzoic acid")
        'benzamide'
        >>> _acid_to_amide("cyclohexanecarboxylic acid")
        'cyclohexanecarboxamide'
        >>> _acid_to_amide("ethanol") # non-acid -> None
    """
    name = acid_name.strip()

    # Guard: if already in -amide form, don't convert again
    if name.endswith("amide") and " acid" not in name:
        return name

    # Per: pre-validate that input looks like an acid name
    if not _looks_like_acid_name(name):
        return None  # Per: not an acid -- don't fabricate

    # Check trivial lookup first
    if name.lower() in _TRIVIAL_ACID_TO_AMIDE:
        return _TRIVIAL_ACID_TO_AMIDE[name.lower()]

    # OPSIN expanded lookup
    if name.lower() in _OPSIN_AMIDE:
        return _OPSIN_AMIDE[name.lower()]

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

    # Per: no recognized pattern matched -- return None instead of garbage
    return None


def _acid_to_acyl(acid_name: str) -> Optional[str]:
    """Acyl prefix form of ``acid_name`` (see ``_acid_to_acyl_impl``). An acid name
    that carries a recorded non-PIN fragment (e.g. a retained peptide name,
    'phenylalanylglutamic acid') keeps that status in its acyl form
    ('phenylalanylglutamyl'), so the name that cites the acyl group is labelled the
    same way (``metrics.provenance.record_derived_non_pin_fragment``)."""
    acyl = _acid_to_acyl_impl(acid_name)
    if acyl and acid_name:
        from ..metrics.provenance import record_derived_non_pin_fragment
        record_derived_non_pin_fragment(acid_name, acyl)
    return acyl


def _acid_to_acyl_impl(acid_name: str) -> Optional[str]:
    """Convert acid name to acyl prefix form.

    Used for N-acyl naming of amides with complex amines.

    Handles acid names ("acetic acid"), ester names ("propanedioate"),
    and amide names ("benzamide", "N-methylbenzamide") that may be passed
    when the decomposition engine mislabels fragment types.

    Args:
        acid_name: Full acid name (or amide/ester name).

    Returns:
        Acyl prefix (e.g., "acetyl", "propanoyl", "benzoyl"), or None if
        input is not a recognized convertible name (per).

    Examples:
        >>> _acid_to_acyl("acetic acid")
        'acetyl'
        >>> _acid_to_acyl("propanoic acid")
        'propanoyl'
        >>> _acid_to_acyl("benzoic acid")
        'benzoyl'
        >>> _acid_to_acyl("benzamide")
        'benzoyl'
        >>> _acid_to_acyl("N-methylbenzamide")
        'N-methylbenzoyl'
        >>> _acid_to_acyl("propanedioate")
        'propanedioyl'
        >>> _acid_to_acyl("cyclohexane") # non-convertible -> None
    """
    name = acid_name.strip()

    # Check trivial lookup first
    if name.lower() in _TRIVIAL_ACID_TO_ACYL:
        return _TRIVIAL_ACID_TO_ACYL[name.lower()]

    # OPSIN expanded lookup
    if name.lower() in _OPSIN_ACYL:
        return _OPSIN_ACYL[name.lower()]

    # Per: pre-validate that input looks like a convertible name
    # (acid, amide, or ester). Place AFTER trivial lookup to preserve shortcut.
    if not _looks_like_convertible_name(name):
        return None  # Per: not convertible -- don't fabricate

    # Handle amide names passed as acid names (e.g., "benzamide",
    # "N-methylbenzamide"). The decomposition engine sometimes labels
    # amide fragments as "acid" in the fragment_names dict.
    # Convert amide -> acid -> acyl to get the correct prefix.
    # Reverse lookup: amide -> acid for trivial names
    _trivial_amide_to_acid = {v: k for k, v in _TRIVIAL_ACID_TO_AMIDE.items()}

    # Strip N-substituent prefix if present (e.g., "N-methylbenzamide" -> "benzamide")
    n_prefix = ""
    bare_name = name
    import re as _re
    n_match = _re.match(r'^(N(?:,N)*-[a-z]+)-(.+)$', name)
    if n_match and n_match.group(2).endswith("amide"):
        n_prefix = n_match.group(1) + "-"
        bare_name = n_match.group(2)

    if bare_name.lower() in _trivial_amide_to_acid:
        acid = _trivial_amide_to_acid[bare_name.lower()]
        acyl = _acid_to_acyl(acid)  # recursive call with the acid name
        return n_prefix + acyl

    # Systematic amide -> acyl: "propanamide" -> "propanoyl"
    if bare_name.endswith("amide") and " acid" not in bare_name:
        stem = bare_name[:-5]  # strip "amide"
        if stem:
            # "propanamide" -> "propan" -> "propanoyl"
            return n_prefix + stem + "oyl"

    # Handle ester "-ate" -> acyl "-oyl": "propanedioate" -> "propanedioyl"
    # First check trivial ester->acyl via reverse lookup of acid tables
    if name.endswith("ate") and " acid" not in name:
        # Reverse lookup: build ester -> acid -> acyl chain for trivial names
        from ..data.trivial_acids import TRIVIAL_ACID_TO_ACYLATE
        _trivial_ester_to_acyl = {}
        for acid_stem, ester in TRIVIAL_ACID_TO_ACYLATE.items():
            # acid_stem = "acetic", ester = "acetate"
            acid_full = acid_stem + " acid"
            if acid_full.lower() in _TRIVIAL_ACID_TO_ACYL:
                _trivial_ester_to_acyl[ester.lower()] = _TRIVIAL_ACID_TO_ACYL[acid_full.lower()]
        if name.lower() in _trivial_ester_to_acyl:
            return _trivial_ester_to_acyl[name.lower()]

        # Systematic ester -> acyl
        stem = name[:-3]  # strip "ate"
        if stem:
            # Check systematic pattern: "-oate" -> "-oyl"
            if stem.endswith("o"):
                # "benzoate" -> "benzo" -> "benzoyl"
                # "propanoate" -> "propano" -> but we want "propanoyl"
                return stem + "yl"
            else:
                # "propanedioate" -> "propanedio" -> "propanedioyl"
                return stem + "oyl"

    # '-amic' / '-imidic' acids take '-oyl' (ACID_ENDINGS_TO_OYL_ACYL):
    # 'N-methylcarbamic acid' -> 'N-methylcarbamoyl', never '...carbamyl'.
    for end, acyl in ACID_ENDINGS_TO_OYL_ACYL:
        if name.endswith(end):
            return name[:-len(end)] + acyl

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

    # Per: no recognized pattern matched -- return None instead of garbage
    return None


def _acid_to_thioate(acid_name: str) -> Optional[str]:
    """Convert acid name to thioate form per IUPAC.

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
    # because thio- derivatives use systematic nomenclature per IUPAC

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

    # Per: pre-validate that input looks like an acid name
    # (placed AFTER carboxylic acid special case and trivial lookup)
    if not _looks_like_acid_name(name):
        return None  # Per: not an acid -- don't fabricate

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

    # Per: no recognized pattern matched -- return None instead of garbage
    return None


def _thiol_to_s_prefix(thiol_name: str) -> Optional[str]:
    """Convert thiol name to S-alkyl prefix for thioester naming.

    Per IUPAC, the sulfur-bearing fragment is designated
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
            # "propane" -> "propyl" (if base is "propane")
            if base.endswith("ane"):
                return f"S-{base[:-3]}yl"
            if base.endswith("an"):
                return f"S-{base[:-2]}yl"
            return f"S-{base}yl"

    # If the name is already in some other form, try using _alcohol_to_alkyl
    # This handles cases where the fragment is named as an alcohol instead
    alkyl = _alcohol_to_alkyl(name)
    if alkyl and alkyl != name:
        # (the Blue Book), '*S*-(2-cyanoethyl)... (PIN)' (:31765):
        # a compound or locant-bearing alkyl is enclosed after the element locant.
        from ..assembly.naming_utils import enclose_if_compound
        return f"S-{enclose_if_compound(alkyl)}"

    return None


def _acid_to_sulfonamide(acid_name: str) -> Optional[str]:
    """Convert sulfonic acid name to sulfonamide form per IUPAC.

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


def _is_saturated_chain_stem(base: str) -> bool:
    """True when ``base`` (an alcohol name minus its '-1-ol') ends in a saturated
    acyclic or monocyclic hydrocarbon stem, '<alk>an' / 'cyclo<alk>an', that is not
    the tail of a von Baeyer or spiro descriptor ('bicyclo[2.2.1]heptan')."""
    import re

    from ..data.chain_names import get_chain_prefix
    for n in range(60, 0, -1):
        stem = get_chain_prefix(n) + "an"
        if base.endswith(stem):
            head = base[: -len(stem)]
            if head.endswith("cyclo"):
                head = head[: -len("cyclo")]
            # A von Baeyer / spiro descriptor ('bicyclo[2.2.1]', 'spiro[4.5]') is
            # a ring system; a closing enclosing mark of a PREFIX ('2-[...]ethan')
            # is not.
            return not re.search(r"(?:cyclo|spiro)\[[0-9.,^]+\]$", head)
    return False


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
        # Match pattern: base-N-ol or baseol, CAPTURING the locant.
        match = re.match(r'^(.+?)(?:-(\d+))?-ol$', name)
        if match:
            base = match.group(1)
            locant = match.group(2)
            # A NON-omittable locant (>1) must be PRESERVED and the stem kept
            # intact: 'tropan-3-ol' -> 'tropan-3-yl' (NOT 'tropyl' -- dropping
            # the '-3-' AND eliding the retained ring stem 'tropan'->'trop'
            # would be a different/invalid substituent). Also fixes acyclic
            # secondary alcohols, e.g. 'heptan-3-ol' -> 'heptan-3-yl'. Locant
            # '1' (or absent) keeps the classic contracted elision below
            # (propan-1-ol -> propyl) to stay byte-identical on chains.
            if locant is not None and locant != "1":
                return f"{base}-{locant}-yl"
            # (the Blue Book): the locant 1 is omitted, and 'ane'
            # becomes 'yl', only for a SATURATED chain (and,, a saturated
            # monocycle: 'cyclohexyl'). Every other parent keeps its free-valence
            # locant: 'prop-2-en-1-yl (preferred prefix)' (:17216), a fixed-
            # numbering ring such as naphthalene ('4-chloronaphthalen-1-yl', never
            # '4-chloronaphthalenyl'), a von Baeyer or retained cage
            # ('bicyclo[2.2.1]heptan-1-yl', 'adamantan-1-yl').
            if locant == "1" and not _is_saturated_chain_stem(base):
                return f"{base}-1-yl"
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
        >>> _amine_to_prefix("glutamine")
        'glutaminyl'
    """
    name = amine_name.strip()

    # Check trivial lookup first
    if name.lower() in _TRIVIAL_AMINE_TO_PREFIX:
        return _TRIVIAL_AMINE_TO_PREFIX[name.lower()]

    # Systematic: ends with "amine" -> strip and convert
    if name.endswith("amine"):
        # Check amino acid acyl names FIRST for names ending in "amine".
        # Amino acids like glutamine, asparagine end in "amine" but are NOT
        # simple amines -- stripping "amine" gives nonsense ("glut" + "yl").
        # Their correct prefix forms are in the amino acid data module.
        from ..data.amino_acids import get_amino_acid_acyl_name
        aa_prefix = get_amino_acid_acyl_name(name)
        if aa_prefix:
            return aa_prefix

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
