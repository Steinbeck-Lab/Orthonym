"""a phase (WSA-03 / CONTEXT) — shared name-composition primitives.

Single source of truth for the IUPAC P-14.5 / P-16 / P-31.1 string-composition
grammar, called by BOTH the legacy fragment assembler
(``handlers/_handler_shared.py:_assemble_fragments``) and the Name-Tree
serializer (``name_tree_to_string._assemble_explicit_fields``). There is ONE
implementation of each rule — not two — so the legacy path and the production
flip path cannot drift (; the no-band-aid mandate in
``.claude/skills/fix-methodology.md``).

The five core helpers (``_estimate_parent_size_from_name``,
``_build_unsaturation_infix``, ``_build_hydrocarbon_name``, ``_join_prefixes``,
``_join_prefix_to_name``) were LIFTED VERBATIM from ``composer.py`` (a phase)
— their bodies are unchanged so the legacy carrier stays byte-identical.
``composer.py`` re-exports them under their original names for back-compat.

Leaf-module invariant (Pitfall 4): this module imports ONLY from
``naming_utils`` and ``rules.locant_validation`` (both confirmed leaves) plus the
``data.chain_names`` table — it MUST NOT import ``composer`` or ``NameFragment``,
so the serializer can import it at module load without an
``assembly -> composer -> assembly`` cycle.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .naming_utils import (
    SIMPLE_MULTIPLIERS,
    italicized_prefix_is_bare,
    should_omit_locant_one,
)

# ---------------------------------------------------------------------------
# Core helpers — LIFTED VERBATIM from composer.py (a phase, byte-identical).
# ---------------------------------------------------------------------------


def _estimate_parent_size_from_name(parent_name: str) -> int:
    """Estimate the number of atoms in the parent from its name.

    Used by collision detection to determine if a locant exceeds the parent
    structure capacity. Returns a conservative estimate; unknown parents
    default to 100 (effectively disabling capacity validation).

    Args:
        parent_name: The parent fragment text (e.g., 'cyclohex', 'benz', 'prop').

    Returns:
        Estimated atom count in the parent structure.
    """
    lower = parent_name.lower()

    # Common ring systems with fixed sizes
    _RING_SIZES = {
        'benzene': 6, 'phenyl': 6, 'benz': 6,
        'cycloprop': 3, 'cyclobut': 4, 'cyclopent': 5,
        'cyclohex': 6, 'cyclohept': 7, 'cycloocta': 8,
        'cyclonon': 9, 'cyclodec': 10,
        'naphthal': 10, 'naphthyl': 10,
        'indol': 9, 'indene': 9,
        'quinol': 10, 'isoquinol': 10,
        'pyrid': 6, 'pyrimid': 6, 'pyrazin': 6,
        'pyrrol': 5, 'furan': 5, 'thiophen': 5,
        'imidazol': 5, 'pyrazol': 5, 'oxazol': 5,
        'thiazol': 5, 'triazin': 6,
    }
    for key, size in _RING_SIZES.items():
        if key in lower:
            return size

    # Try chain prefix matching using FIRST_20 from chain_names
    from ..data.chain_names import FIRST_20
    # Check longest prefixes first to avoid partial matches (e.g., "eth" in "meth")
    for length in sorted(FIRST_20.keys(), reverse=True):
        prefix = FIRST_20[length]
        if lower.startswith(prefix) or lower == prefix:
            return length

    # Safe fallback: return large value to disable capacity validation
    # for unknown parent structures.
    import logging
    logging.getLogger(__name__).debug(
        "Unknown parent '%s' -- skipping locant capacity validation (fallback=100)",
        parent_name,
    )
    return 100


def _build_unsaturation_infix(
    double_locants: List[int],
    triple_locants: List[int]
) -> str:
    """
    Build the unsaturation infix (e.g., 'an', '-1-en', '-1-en-4-yn').

    This is used when there's a functional group suffix. The unsaturation
    part is inserted between the stem and the suffix locants.

    The infix is assembled structurally so that hyphens are inserted only
    where needed -- no post-hoc band-aid cleanup required.

    IUPAC P-31.1.3.4: when total unsaturation locant count >= 2, prefix
    with 'a' for euphony (e.g., 'a-1,3-dien' not '-1,3-dien').

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
        >>> _build_unsaturation_infix([1, 3], [])
        'a-1,3-dien'
        >>> _build_unsaturation_infix([], [1, 3])
        'a-1,3-diyn'
        >>> _build_unsaturation_infix([1, 3], [5])
        'a-1,3-dien-5-yn'
    """
    num_double = len(double_locants)
    num_triple = len(triple_locants)

    if num_double == 0 and num_triple == 0:
        return "an"  # Saturated

    # Determine if the 'a' euphonic connector is needed (IUPAC P-31.1.3.4):
    # used when total unsaturation locants >= 2 (diene, diyne, enyne etc.)
    total_locants = num_double + num_triple
    needs_a = (total_locants >= 2) and (num_double > 1 or num_triple > 1)

    # Build segments: each is a tuple (locant_str, multiplier, bond_type)
    segments = []

    if num_double > 0:
        loc_str = ",".join(str(loc) for loc in double_locants)
        multiplier = SIMPLE_MULTIPLIERS.get(num_double, str(num_double)) if num_double > 1 else ""
        segments.append((loc_str, multiplier, "en"))

    if num_triple > 0:
        loc_str = ",".join(str(loc) for loc in triple_locants)
        multiplier = SIMPLE_MULTIPLIERS.get(num_triple, str(num_triple)) if num_triple > 1 else ""
        segments.append((loc_str, multiplier, "yn"))

    # Assemble: "a" (if needed) then each segment as "-locants-[mult]bond"
    # joined with hyphens between them.
    result_parts = []
    if needs_a:
        result_parts.append("a")

    for loc_str, mult, bond in segments:
        result_parts.append(loc_str)
        result_parts.append(f"{mult}{bond}")

    # Join all parts with hyphens; prepend leading hyphen if no 'a' connector
    result = "-".join(result_parts)
    if not needs_a:
        result = "-" + result

    return result


def _build_hydrocarbon_name(
    stem: str,
    double_locants: List[int],
    triple_locants: List[int],
    ring_bond_locant_omittable: Optional[bool] = None,
    chain_bond_locant_omittable: Optional[bool] = None,
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

    The name is assembled structurally so that hyphens are inserted only
    where needed -- no post-hoc band-aid cleanup required.

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

    # Centralized bond locant elision: ethene/ethyne (2-carbon) and mono-cycloalkenes.
    # WSD-06 (NUM-01): the ring ene-locant is omitted only for an UNSUBSTITUTED
    # cycloalkene (P-14.3.4.2(d)); a substituent makes the double-bond position
    # distinctive, so it must be cited (`3-bromocyclohex-1-ene`, not
    # `bromocyclohexene`). The caller passes ``ring_bond_locant_omittable`` (False
    # when the ring carries a substituent); the old ``num_double==1`` count proxy is
    # the fallback for callers that don't supply it.
    # P-14.3.4.2(d): dinuclear ethene/ethyne always omit the bond locant.
    # TRInuclear propene/propyne (Wave2 T6a) omit it ONLY when the caller
    # certifies the parent is unsubstituted via chain_bond_locant_omittable —
    # a substituent makes the position distinctive (3-chloroprop-1-ene). A
    # single terminal multiple bond is implied (a 3-chain 2-ene cannot exist).
    if stem == "eth":
        _chain_len = 2
    elif (
        stem == "prop"
        and chain_bond_locant_omittable
        and (len(double_locants) + len(triple_locants)) == 1
    ):
        _chain_len = 3
    else:
        _chain_len = 0
    _ring_mono = (
        ring_bond_locant_omittable
        if (is_cyclic and ring_bond_locant_omittable is not None)
        else (num_double == 1 and num_triple == 0)
    )
    omit_bond_locant = should_omit_locant_one(
        context="bond",
        chain_length=_chain_len,
        is_ring=is_cyclic,
        is_monosubstituted=_ring_mono,
    )

    # Determine if the 'a' euphonic connector is needed (IUPAC P-31.1.3.4):
    # used when multiple bonds have multiplied locants (diene, diyne, etc.)
    needs_a = (num_double > 1) or (num_triple > 1 and num_double == 0)

    # Build segments as structured data, then join cleanly
    # Each segment: (locant_str_or_None, multiplier, bond_suffix)
    segments = []

    if num_double > 0:
        if num_double == 1:
            if omit_bond_locant:
                segments.append((None, "", "en"))
            else:
                loc_str = ",".join(str(loc) for loc in double_locants)
                segments.append((loc_str, "", "en"))
        else:
            loc_str = ",".join(str(loc) for loc in double_locants)
            mult = SIMPLE_MULTIPLIERS.get(num_double, str(num_double))
            segments.append((loc_str, mult, "en"))

    if num_triple > 0:
        if num_triple == 1:
            if omit_bond_locant and num_double == 0:
                segments.append((None, "", "yn"))
            else:
                loc_str = ",".join(str(loc) for loc in triple_locants)
                segments.append((loc_str, "", "yn"))
        else:
            loc_str = ",".join(str(loc) for loc in triple_locants)
            mult = SIMPLE_MULTIPLIERS.get(num_triple, str(num_triple))
            segments.append((loc_str, mult, "yn"))

    # Assemble: stem [+ "a" if needed] [+ segments joined with hyphens] + "e"
    result = stem
    if needs_a:
        result += "a"

    for loc_str, mult, bond in segments:
        if loc_str is not None:
            result += f"-{loc_str}-{mult}{bond}"
        else:
            # No locant: directly append bond suffix (ethene, cyclohexene)
            result += f"{mult}{bond}"

    # Final terminal 'e' for the hydrocarbon ending
    result += "e"

    return result


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
            # -CLEANUP Item 4 (MINOR 8): `}` was missing from both tuples,
            # so this copy dropped the separator after a brace-enclosed prefix
            # ('2-{[(methylcarbamoyl)amino]methyl}4-methyl') while the
            # `polyfunctional._join_prefixes` copy already hyphenated it. A closing
            # brace ends an enclosure exactly as `)` and `]` do — P-16.5.4's nesting
            # cycle `{[({[()]})]}` makes all three the same kind of boundary — and
            # 24 `}`-bearing gold rows route through THIS copy, so the two must
            # agree. (No current gold pairs a `}` with a following locant, which is
            # why the asymmetry survived; see the tests added alongside.)
            closes_enclosure = last_char in (')', ']', '}')
            if (last_char.isalpha() or closes_enclosure) and first_char.isdigit():
                result += "-"
            elif (last_char.isalpha() or closes_enclosure) and first_char == 'N':
                result += "-"
        result += current

    return result


def _join_prefix_to_name(prefix_str: str, name: str) -> str:
    """
    Join a prefix string to a parent/suffix name with proper IUPAC hyphenation.

    Ensures a hyphen is inserted when:
    - The prefix ends with a letter and the name starts with a digit

    This prevents broken names like 'pentabutyl1,4,7,10,13-pentaaza'
    by inserting a hyphen: 'pentabutyl-1,4,7,10,13-pentaaza'.

    Args:
        prefix_str: The assembled prefix string (e.g., '3-ethyl-4-methyl')
        name: The parent+suffix name (e.g., 'propan-1-ol')

    Returns:
        Properly hyphenated combined name
    """
    if not prefix_str or not name:
        return prefix_str + name

    if (prefix_str[-1].isalpha() or prefix_str[-1] in (')', ']')) and name[0].isdigit():
        return f"{prefix_str}-{name}"

    return f"{prefix_str}{name}"


# Leading top-level (alpha-carbon) locant run on a formatted prefix, e.g. the
# "2-" of "2-chloro" or the "2,2-" of "2,2-difluoro". Anchored at the string
# start so INTERNAL locants of a complex substituent — the "4-" inside
# "(4-chlorophenoxy)" — are preserved.
_ALPHA_LOCANT_RE = re.compile(r'^\d+(?:,\d+)*-')


def retained_acetic_from_prefixes(
    prefix_texts: List[str],
    stereo: str = "",
    parent: str = "acetic acid",
    enclose_subsequent: bool = False,
) -> str:
    """Assemble a SUBSTITUTED retained-functional-parent PIN from already-formatted
    substituent prefix strings (P-65.1.1.1 / P-66.6.1.2.1 retained functional parent
    + P-14.3.4.6 locant omission).

    The retained parent (``acetic acid``; or ``acetaldehyde`` via ``parent=``, per
    P-66.6.1.2.1) has a single substitutable position (the alpha carbon), so ALL
    substituent locants are omitted while multipliers and any INTERNAL locants of
    a complex substituent are preserved::

        ['2-chloro'] -> 'chloroacetic acid'
        ['2,2-difluoro'] -> 'difluoroacetic acid' (BB 3037)
        ['2-phenyl'] -> 'phenylacetic acid' (BB 6694)
        ['2-(4-chlorophenoxy)'] -> '(4-chlorophenoxy)acetic acid'
        ['2-phenoxy'], parent='acetaldehyde'
                                  -> 'phenoxyacetaldehyde' (BB 35076)

    ``enclose_subsequent`` (P-16.3.3 worked examples ``cyclopropyl(hydroxy)-
    acetaldehyde`` BB 45259, ``cyclobutyl(cyclopropyl)methanol`` BB 45247):
    when 2+ substituent prefixes are cited with their locants omitted, the FIRST
    prefix is bare and each SUBSEQUENT prefix is set off by enclosing marks::

        ['2-cyclopropyl','2-hydroxy'], parent='acetaldehyde', enclose_subsequent=True
                                  -> 'cyclopropyl(hydroxy)acetaldehyde'

    The caller must gate ``enclose_subsequent`` to a context where every prefix is a
    simple, single-alpha-locant token (the multiplied ``di(phenyl)`` shape of BB
    29852 is a DIFFERENT P-16.3.3 rule and must not be routed here) -- see the
    aldehyde arm in ``rules/polyfunctional.py``. The caller is likewise responsible
    for gating the whole assembly to the substituted-2-carbon-mono-FG context; this
    function only performs the retained-name assembly.
    """
    unlocanted = [_ALPHA_LOCANT_RE.sub('', t) for t in prefix_texts]
    # (P-16.3.3): stripping the alpha locant can leave a COMPLEX substituent
    # whose own enclosure no longer wraps the whole prefix -- e.g.
    # '[(methylsulfanyl)carbonyl]amino' (the [...] wraps only the acyl, 'amino'
    # trails outside). Joined bare it welds into '...aminoacetic acid'; P-16.3.3
    # requires the whole prefix enclosed -> '{[(methylsulfanyl)carbonyl]amino}'.
    # Wrap each unlocanted prefix that carries marks but is not already fully
    # enclosed; a fully-enclosed prefix ('(4-chlorophenoxy)') and a mark-free one
    # ('chloro') are left untouched (docstring examples preserved). Mirrors the
    # substituent_naming.py:2487 pattern.
    from .naming_utils import _is_fully_enclosed, apply_enclosing_marks

    def _mult_self_enclosed(tok: str) -> bool:
        # A 'bis(...)'/'tris(...)'/'tetrakis(...)'-shaped token is ALREADY fully
        # enclosed by its own multiplicative parentheses (P-16.5.1.1: parentheses
        # are used after 'bis','tris', etc.) -- the closing ')' is the last char
        # with nothing trailing -- so it must NOT be re-escalated to '[bis(...)]'.
        # _is_fully_enclosed only inspects the FIRST char, so it misses this (the
        # token starts with 'b'/'t', not a bracket) -> 'bis(2-hydroxyethoxy)acetic
        # acid', not '[bis(2-hydroxyethoxy)]acetic acid' (the Blue Book PIN).
        for _m in ('bis', 'tris', 'tetrakis', 'pentakis', 'hexakis'):
            if tok.startswith(_m) and _is_fully_enclosed(tok[len(_m):]):
                return True
        return False

    unlocanted = [
        apply_enclosing_marks(t, -1)
        if (('(' in t or '[' in t) and not _is_fully_enclosed(t)
            and not _mult_self_enclosed(t)) else t
        for t in unlocanted
    ]
    # P-16.5.1.3.1/.3.2 (locant-omitted multi-prefix on a single-substitutable
    # retained parent): the FIRST cited substituent is bare, each SUBSEQUENT one is
    # set off by enclosing marks -- 'anilino(oxo)acetic acid' (the Blue Book),
    # 'bromo(chloro)acetic acid' (the Blue Book). A multiplicative prefix stays OUTSIDE the
    # marks (rule's last sentence,:7272): 'diphenyl' -> 'di(phenyl)', giving
    # 'hydroxydi(phenyl)acetic acid' (the Blue Book) -- NOT the halomethane bare-all
    # carve-out (which the BB's own di(phenyl) example contradicts here).
    if enclose_subsequent and len(unlocanted) > 1:
        def _enclose_after_first(t: str) -> str:
            if _is_fully_enclosed(t):
                return t
            for _mp in _MULTIPLIER_PREFIXES:
                if t.startswith(_mp) and len(t) > len(_mp):
                    return f"{_mp}{apply_enclosing_marks(t[len(_mp):], -1)}"
            return apply_enclosing_marks(t, -1)
        unlocanted = [unlocanted[0]] + [_enclose_after_first(t) for t in unlocanted[1:]]
    name = _join_prefix_to_name(_join_prefixes(unlocanted), parent)
    return f"{stereo}{name}" if stereo else name


# ---------------------------------------------------------------------------
# Lifted P-16.5.1.3.1 mononuclear enclosing-marks block (from
# _handler_shared.py:1037-1078) as a named function; the serializer calls this
# and the legacy assembler keeps its byte-identical inline copy (to be thinned
# to a call in a later phase).
# ---------------------------------------------------------------------------

# Mononuclear parent stems (one heavy atom of any element), keyed by the bare
# parent-hydride stem. Drives the serializer's structural `is_mononuclear`
# derivation (CONTEXT — DERIVE, no NameTreeNode field add). For
# `general_acyclic` only "meth" is reachable; the rest keep parity with the
# legacy structural flag for non-carbon mononuclear parents.
_MONONUCLEAR_STEMS = frozenset(
    {"meth", "silan", "phosphan", "boran", "german", "stannan", "plumban", "alan"}
)

_MULTIPLIER_PREFIXES = (
    'di', 'tri', 'tetra', 'penta', 'hexa', 'hepta', 'octa', 'nona', 'deca',
)


def apply_mononuclear_enclosing(
    prefix_texts: List[str], is_mononuclear: bool
) -> List[str]:
    """Apply the P-16.5.1.3.1 mononuclear enclosing rule (BlueBookV2 7272).

    For a mononuclear parent hydride with >= 2 substituents: the FIRST cited
    substituent is bare; the SECOND AND FURTHER are EACH enclosed in
    parentheses (``bromo(chloro)(fluoro)methane``). Carve-out: when ANY simple
    prefix carries a multiplicative prefix (``dichloro``/``trifluoro``/...) ALL
    prefixes are left bare/unenclosed — the common-PIN form
    ``bromodichlorofluoromethane`` (the PROTECT case ``C(Br)(Cl)(Cl)F``).

    THE single implementation of this rule: ``_handler_shared._assemble_fragments``
    used to carry a second, near-identical inline copy (differing only in two
    comment words) and therefore a second copy of the compound-hyphen test; it now
    calls this function.
    """
    if not (is_mononuclear and len(prefix_texts) >= 2):
        return prefix_texts

    def _is_simple_prefix(t: str) -> bool:
        # simple = no locant prefix, no existing enclosing marks, no compound
        # hyphen, no multiplicative prefix; bromo/chloro/fluoro/iodo/nitro etc.
        # A locant-bearing first substituent already routes through the
        # `re.match(r'^\d',...)` path in the caller and is excluded here (the
        # Blue Book exempts only the locant-bearing first prefix from this rule).
        #
        # The italicized-prefix carve-out is the SHARED primitive, never a raw
        # `'-' in t`: P-16.3.3(b)/P-16.2.4.1(d) / P-29.6.1 make the hyphen of a leading italicized
        # structural prefix part of a SIMPLE retained name (BB 16286 cites
        # '*tert*-butyldi(methyl)phosphane' (PIN) with the group BARE). A raw test
        # classed 'tert-butyl' as compound, which flipped `all(...)` below to False
        # and so silently switched OFF the first-bare/rest-enclosed transform for
        # EVERY prefix in the name, not merely the tert- one.
        _ital_bare = italicized_prefix_is_bare(t)     # GUARD: P-16.3.3(b)/P-16.2.4.1(d) carve-out
        if (re.match(r'^\d', t) or '(' in t or '[' in t
                or ('-' in t and not _ital_bare)):
            return False
        if any(t.startswith(mp) for mp in _MULTIPLIER_PREFIXES):
            return False
        return True

    if all(_is_simple_prefix(t) for t in prefix_texts):
        # FIRST cited bare; SECOND AND FURTHER each enclosed (P-16.5.1.3.1).
        return [prefix_texts[0]] + [f"({t})" for t in prefix_texts[1:]]
    return prefix_texts


# ---------------------------------------------------------------------------
# Lifted P-14.7 suffix<->prefix locant-collision resolver (from
# _handler_shared.py:963-1015), refactored to operate on neutral
# (text, locants) tuples instead of NameFragment. Chains return the input
# unchanged early (no-op for general_acyclic); it exists so the serializer
# reaches byte-identity on ring parents too.
# ---------------------------------------------------------------------------


def resolve_suffix_prefix_collision(
    suffix_locants: List[int],
    prefix_pairs: List[Tuple[str, tuple]],
    is_ring: bool,
    parent_size: int,
) -> List[Tuple[str, tuple]]:
    """Remove prefix locants that collide with a suffix locant (P-14.7, suffix wins).

    Mirrors ``_assemble_fragments:963-1015``. Returns ``prefix_pairs`` unchanged
    when the parent is not a ring, there is no suffix locant, or no collision is
    detected (the general_acyclic-chain no-op path).
    """
    if not (is_ring and suffix_locants and prefix_pairs):
        return prefix_pairs

    from ..rules.locant_validation import detect_locant_collisions

    suffix_locants_list = list(suffix_locants)
    prefix_locant_groups = [list(loc) for (_, loc) in prefix_pairs if loc]
    if not (suffix_locants_list and prefix_locant_groups):
        return prefix_pairs

    collisions = detect_locant_collisions(
        suffix_locants_list,
        prefix_locant_groups,
        parent_type="ring",
        parent_size=parent_size,
    )
    if not collisions:
        return prefix_pairs

    collision_set = set(loc for _, loc in collisions)
    adjusted: List[Tuple[str, tuple]] = []
    for text, locants in prefix_pairs:
        if locants:
            new_locants = tuple(l for l in locants if l not in collision_set)
            adjusted.append((text, new_locants))
        else:
            adjusted.append((text, locants))
    return adjusted


def is_ring_parent_name(parent_text: str) -> bool:
    """True when the parent stem names a ring system (the P-14.7 collision gate).

    Same keyword scan the legacy ``_assemble_fragments`` uses inline at
    ``_handler_shared.py:966-974`` — extracted so the serializer applies the
    identical ring test.
    """
    return any(
        kw in parent_text.lower()
        for kw in ('cyclo', 'benz', 'pyrid', 'pyrrol', 'furan',
                   'thiophen', 'imidazol', 'naphthal', 'indol',
                   'quinol', 'pyrimid', 'pyrazin', 'oxazol',
                   'thiazol', 'triazol', 'morpholin', 'piperidin',
                   'pyrrolidin', 'aziridin', 'oxiran', 'thiiran',
                   'oxetan', 'azetidin', 'thietan')
    )


__all__ = [
    "_estimate_parent_size_from_name",
    "_build_unsaturation_infix",
    "_build_hydrocarbon_name",
    "_join_prefixes",
    "_join_prefix_to_name",
    "apply_mononuclear_enclosing",
    "resolve_suffix_prefix_collision",
    "is_ring_parent_name",
    "_MONONUCLEAR_STEMS",
]
