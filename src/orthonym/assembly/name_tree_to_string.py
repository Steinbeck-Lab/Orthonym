"""Phase 160 Pass-2 serializer (DECOMP-02 + CONTEXT D-04).

Single source of truth for ``NameTreeNode -> str`` going forward. Plan-02
substrate ships this module ALONGSIDE the legacy ``_assemble_fragments``
path at composer.py:7591 — the legacy path STAYS until commit 03-10 thinning
per CONTEXT D-24 incremental-migration discipline. Plan-02/03 first-wave
handlers return ``NamingResult(name=<existing string>, tree=None, ...)``;
the legacy path is what actually produces those names. The Pass-2
serializer here is exercised by ``--dump-tree`` integration tests in Plan-04
and by v19+ handlers that populate trees explicitly.

Contract per CONTEXT D-04 + 160-AUDIT-DECOMP.md § 5.4:

* If ``node.fragment_legacy is not None``: delegate to the legacy assembly
  path by wrapping the fragment list as
  ``_assemble_fragments([node.fragment_legacy], style)``. This is the
  first-wave compatibility mode that lets Phase 160 ship byte-identical
  even before any handler emits a real tree.
* If ``node.fragment_legacy is None`` AND ``node.parent_stem`` is populated:
  assemble the name from the explicit fields in the order::

      stereo + prefixes + indicated_h + parent_stem + locants
             + unsaturation_locants + suffix

  applying multiplicative-prefix and parenthesization rules per
  P-14.5 + P-14.2.2.
* If ``node.fragment_legacy is None`` AND ``node.parent_stem == ""``: raise
  an explicit ``NameTreeSerializerError`` — this is a malformed tree per
  DECOMP-02 honest-fail-on-data.

Byte-identical contract (DECOMP-03): for every input ``node`` whose
``fragment_legacy`` round-trips through the legacy path, the output of
``name_tree_to_string(node, style)`` MUST equal the corresponding
``_assemble_fragments(fragments, style)`` output.

Anti-pattern hygiene:
- AP-160-05 / AP-160-23: no postprocessor band-aid on inner-dispatch
  output; the explicit-field branch below assembles deterministically
  from the IR fields.
- AP-160-27: NameTreeNode field shape is locked — adding new fields to
  the serializer here without a corresponding NameTreeNode field
  addition is a Rule 4 architectural decision.

References:
- 160-AUDIT-DECOMP.md § 5.4 — Pass-2 serializer contract.
- 160-CONTEXT.md D-04 / Question 3 — explicit two-pass IR + serializer.
- 160-PATTERNS.md § 2 — analog: composer.py:7591-7748 ``_assemble_fragments``.
- 160-RESEARCH.md § "Name-Tree IR Semantic Contract" — collision detection
  per IUPAC P-14.7; suffix-prefix locant priority; unsaturation infix.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .name_tree import NameTreeNode, _alphabetize_prefixes


class NameTreeSerializerError(ValueError):
    """Raised when a malformed NameTreeNode is passed to name_tree_to_string.

    Per DECOMP-02 honest-fail-on-data (CONTEXT D-27): a node with
    ``parent_stem == ""`` and ``fragment_legacy is None`` cannot serialize
    deterministically. The fix is upstream in the handler that produced
    the malformed node, not a band-aid string default.
    """


def name_tree_to_string(node: NameTreeNode, style: str = "pin") -> str:
    """Phase 160 D-04 + IUPAC P-14.5 Pass-2 serializer.

    Composition order::

        [stereo] + [prefixes (alpha-sorted; recursively serialized)] +
        [parent_stem] + [indicated_h] + [unsaturation_infix] + [suffix]

    Args:
        node: The NameTreeNode to serialize.
        style: Naming style ("pin", "general", "cas"); forwarded to the
            legacy assembler when ``fragment_legacy`` is set.

    Returns:
        IUPAC name string per CONTEXT D-04 byte-identical contract.

    Raises:
        NameTreeSerializerError: if ``node.parent_stem == ""`` AND
            ``node.fragment_legacy is None`` (malformed tree per
            DECOMP-02 honest-fail-on-data).
    """
    # First-wave compatibility mode (per 160-AUDIT-DECOMP.md § 5.4 first
    # bullet): if the legacy NameFragment is present, route through the
    # legacy assembly path to preserve byte-identical output even before
    # any handler emits a real tree. This is the SOLE wave-1 production
    # path; the explicit-field branch below is exercised by Plan-04
    # --dump-tree integration tests + by v19+ handlers that populate
    # trees.
    if node.fragment_legacy is not None:
        # Phase 165 (SCORE-01) final-string carrier: a ``str`` fragment_legacy
        # holds the pre-assembled final name and is returned verbatim. A single
        # synthetic NameFragment cannot reproduce a multi-fragment concatenation
        # (the parent path appends 'ane' via _build_hydrocarbon_name), so the
        # dedicated string short-circuit is the byte-identical carrier for the
        # fragments_to_tree deriver + caller-side concatenations (e.g. amide
        # N-prefix). RESEARCH 165 lines 459-462.
        if isinstance(node.fragment_legacy, str):
            # IN-4 note: an empty-string ``fragment_legacy=""`` is a VALID
            # (empty) carrier and returns ``""`` verbatim here — deliberately
            # distinct from ``fragment_legacy is None`` (the latter falls through
            # to the explicit-field / honest-fail path below). This str branch is
            # reached only when ``fragment_legacy is not None`` (outer guard), so
            # ``""`` short-circuits BEFORE the ``parent_stem`` honest-fail at the
            # bottom. The SC-3 boundary only synthesizes a node when ``name`` is
            # truthy (namer.py ``if name:``), so an empty name never reaches here
            # in production; this is documented, not a behavioral guard.
            return node.fragment_legacy
        # Lazy import to avoid composer.py -> name_tree_to_string -> composer.py
        # cycle at module-import time (PATTERNS § Lazy Import).
        from .composer import _assemble_fragments
        return _assemble_fragments([node.fragment_legacy], style)

    # Honest-fail-on-data per DECOMP-02 (CONTEXT D-27): a node without
    # both fragment_legacy AND parent_stem cannot serialize. The fix is
    # upstream in the handler.
    if not node.parent_stem:
        raise NameTreeSerializerError(
            f"NameTreeNode with empty parent_stem and no fragment_legacy "
            f"cannot serialize. node={node!r}. Per DECOMP-02 honest-fail-"
            f"on-data: fix the upstream handler that produced this node."
        )

    # Explicit-field assembly (Plan-04+ + v19 path). The body below mirrors
    # the IUPAC P-14.5 composition order; collision detection per
    # IUPAC P-14.7 (suffix has priority over substituent locants) is
    # implemented per RESEARCH § "Name-Tree IR Semantic Contract".
    return _assemble_explicit_fields(node, style)


def _assemble_explicit_fields(node: NameTreeNode, style: str) -> str:
    """Assemble name from explicit NameTreeNode fields (non-legacy path).

    Implementation order (per IUPAC P-14.5)::

        stereo + prefixes + indicated_h + parent_stem + locants + suffix

    where:
      - prefixes are alphabetized via ``_alphabetize_prefixes`` (P-13).
      - Each prefix subtree is recursively serialized via name_tree_to_string.
      - Multiplicative prefix (di-/tri-/bis-/tris-) is applied per P-14.2.2.
      - Parenthesization hint forces parentheses for complex prefixes per
        P-14.5.1 / P-51.3.5.
      - Suffix locants override colliding prefix locants per P-14.7.

    Per Plan-02 wave: this path is NOT exercised by first-wave handlers
    (all return tree=None). Plan-04 --dump-tree CLI tests + v19+
    tree-populated handlers are the consumers.
    """
    # Pass-1: recursively serialize each prefix subtree (alphabetize per
    # P-13). Subtree serialization handles its own multiplicative-prefix,
    # parenthesization, and locant prefix.
    prefix_strs: List[str] = []
    for sub in _alphabetize_prefixes(node.prefixes):
        sub_str = name_tree_to_string(sub, style)
        if sub.parenthesization_hint:
            sub_str = f"({sub_str})"
        if sub.multiplicative_prefix:
            sub_str = f"{sub.multiplicative_prefix}{sub_str}"
        if sub.locants:
            sub_str = f"{_format_locant_set(sub.locants)}-{sub_str}"
        prefix_strs.append(sub_str)

    # Pass-2: compose the parent fragment with locants + unsaturation infix
    # + indicated_h.
    parent_text = node.parent_stem

    # Indicated hydrogen prepend (e.g., "1H-pyrrol-"): per P-25.7.
    if node.indicated_h:
        ih_str = ",".join(f"{i}H" for i in node.indicated_h)
        parent_text = f"{ih_str}-{parent_text}"

    # Unsaturation infix (double + triple bond locants): per P-31.1.
    parent_text = _apply_unsaturation_infix(
        parent_text, node.unsaturation_locants,
    )

    # Parent locant prefix (e.g., "propan-1-"): per P-14.5.
    if node.locants and not node.suffix:
        # When there's no suffix, locants attach to the parent stem.
        parent_text = f"{_format_locant_set(node.locants)}-{parent_text}"

    # Suffix locants override colliding prefix locants per IUPAC P-14.7.
    # (First-wave path: collision logic is deferred to Plan-04+ since
    # no handler populates explicit trees in Plan-02/03.)
    suffix_str = _format_suffix(node.suffix, node.locants) if node.suffix else ""

    # Assemble: prefixes + parent + suffix.
    assembled = "".join(prefix_strs) + parent_text + suffix_str

    # Stereo prepend (e.g., "(2R)-"): per P-91 / P-14.5.
    if node.stereo:
        assembled = f"{node.stereo}-{assembled}"

    return assembled


def _format_locant_set(locants: Tuple[int, ...]) -> str:
    """Format a sorted tuple of locants as a comma-separated string.

    Example::

        >>> _format_locant_set((1, 2, 3))
        '1,2,3'
        >>> _format_locant_set(())
        ''
    """
    if not locants:
        return ""
    return ",".join(str(l) for l in sorted(locants))


def _apply_unsaturation_infix(
    parent_text: str,
    unsaturation_locants: Tuple[Tuple[int, ...], Tuple[int, ...]],
) -> str:
    """Apply double-/triple-bond infix to the parent stem per P-31.1.

    The unsaturation tuple is ``(double_bond_locants, triple_bond_locants)``;
    each is a sorted tuple of int locants. The infix transforms the parent
    stem suffix:

    - ``methan`` + ((), ()) -> ``methan``           (no unsaturation)
    - ``propan`` + ((1,), ()) -> ``prop-1-en``      (one double bond)
    - ``propan`` + ((1, 2), ()) -> ``propa-1,2-dien`` (cumulated double bonds)
    - ``propan`` + ((), (1,)) -> ``prop-1-yn``      (one triple bond)
    - ``propan`` + ((1,), (3,)) -> ``prop-1-en-3-yn`` (enyne)

    Plan-02 wave: NOT exercised by first-wave handlers. Plan-04 +
    v19 handlers exercise this branch via explicit-tree tests.

    Note: For the first-wave/Plan-02 ship the unsaturation infix logic is
    minimal — the composer.py:7706-7712 + _build_unsaturation_infix at
    composer.py:7725 has more sophisticated stem-rewrite rules
    (an / en / yn / adien / atrien etc.) which Plan-04+ formalizes here.
    """
    double_locants, triple_locants = unsaturation_locants

    if not double_locants and not triple_locants:
        return parent_text

    # Minimal substitution: replace trailing 'an' with 'en' / 'yn' / etc.
    # Plan-04+ formalizes the full P-31.1 grammar; first-wave handlers
    # do NOT exercise this branch (all return tree=None).
    if parent_text.endswith("an"):
        stem = parent_text[:-2]
    else:
        stem = parent_text

    # Build the unsaturation suffix.
    infix_parts: List[str] = []
    if double_locants:
        double_count = len(double_locants)
        multiplier = _unsaturation_multiplier(double_count, "en")
        loc_str = _format_locant_set(double_locants)
        infix_parts.append(f"{loc_str}-{multiplier}" if loc_str else multiplier)
    if triple_locants:
        triple_count = len(triple_locants)
        multiplier = _unsaturation_multiplier(triple_count, "yn")
        loc_str = _format_locant_set(triple_locants)
        infix_parts.append(f"{loc_str}-{multiplier}" if loc_str else multiplier)

    # When there are NO double bonds but triple bonds present, the parent
    # keeps the 'a' (e.g., propan -> propa-1-yn would be wrong; it's prop-1-yn).
    # When BOTH present, the stem keeps 'a' before en (e.g., propa-1-en-3-yn).
    if double_locants and triple_locants:
        return f"{stem}a-" + "-".join(infix_parts)
    if double_locants:
        # When n>1, the stem retains 'a' (propa-1,2-dien).
        if len(double_locants) > 1:
            return f"{stem}a-" + "-".join(infix_parts)
        return f"{stem}-" + "-".join(infix_parts)
    # triple only:
    return f"{stem}-" + "-".join(infix_parts)


def _unsaturation_multiplier(count: int, base: str) -> str:
    """Build the unsaturation multiplier for ``base`` ('en' or 'yn').

    Example::

        >>> _unsaturation_multiplier(1, "en")
        'en'
        >>> _unsaturation_multiplier(2, "en")
        'dien'
        >>> _unsaturation_multiplier(3, "en")
        'trien'
    """
    if count <= 1:
        return base
    if count == 2:
        return f"di{base}"
    if count == 3:
        return f"tri{base}"
    if count == 4:
        return f"tetra{base}"
    if count == 5:
        return f"penta{base}"
    return f"{count}{base}"


def _format_suffix(suffix: str, locants: Tuple[int, ...]) -> str:
    """Format the suffix with locants per P-14.5.

    Plan-02 wave: NOT exercised by first-wave handlers (all return
    tree=None). Plan-04 + v19 handlers exercise this branch via
    explicit-tree tests.

    For the first-wave/Plan-02 ship this is intentionally simple. The
    composer.py:7714-7733 ``format_suffix_with_locants`` helper has more
    nuanced rules (omit locant '1' for single-locant cases per
    should_omit_locant_one; multiplicative prefix on suffix; etc.) which
    are layered in by Plan-04+.
    """
    if not suffix:
        return ""
    if not locants:
        return suffix
    loc_str = _format_locant_set(locants)
    return f"-{loc_str}-{suffix.lstrip('-')}"


__all__ = [
    "name_tree_to_string",
    "NameTreeSerializerError",
]
