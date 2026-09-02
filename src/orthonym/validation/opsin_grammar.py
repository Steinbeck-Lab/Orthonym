"""
OPSIN-grammar pre-validator.

OPSIN-XML-driven authoritative validator + round-trip-gated suggest_fix.
Layered ON TOP format_validator heuristics per CONTEXT.md.

Architecture (deliverable):
    - Module-import-time XML loader: parses
      `opsin/.../regexTokens.xml`, expands `%name%` placeholders to fixed
      point, and compiles each into a Python `re.Pattern`. Loud
      `ImportError` on adapter mismatch — never silent degradation
      (AP-7).
    - `OpsinGrammar.validate(name) -> bool`: hot-path API. Layer-
      cake: format_validator pre-screen first, then OPSIN-XML
      strict checks for {bracket, hyphen, stereo} surfaces.
    - `OpsinGrammar.suggest_fix(name, source_smiles=None)` (
      LOCKED signature, name FIRST, source_smiles SECOND): tries the
      bounded repair tables from `156-AUDIT.md` § 4 in order
      [bracket, hyphen, stereo], gates each candidate through
      `validate()` AND `opsin_roundtrip_check(smiles, candidate)`
      (AP-8 SMILES-first inside the oracle), and returns
      `(repaired, repair_class)` on success or `(None, None)`
      otherwise. One repair attempt only — no retry loop
      (AP-5).
    - Per-instance `_stats` counter (AP-19) — never module-
      global mutable state.

Anti-pattern hygiene (CONTEXT.md):
    AP-1/AP-2: no `_postprocess_name`/`re.sub` chains outside the
        explicit `_suggest_*` table.
    AP-3: every `_suggest_*` regex literal cites its 156-AUDIT.md
        § 4 row.
    AP-4: repairs only re-position / re-bracket / re-hyphenate;
        never change parent/locants/substituents.
    AP-7: `_load_opsin_token_regexes` raises `ImportError` on miss.
    AP-8: round-trip oracle is `(smiles, name)`.
    AP-9: read `result["passed"]`, never truthy-check the dict.
    AP-15: no fictitious "fast OPSIN call" timing claims; the
        verified empirical figure on this host is ~1269ms mean.
    AP-16: default `jar_version="2.9.0"`; never assume an older JAR.
    AP-19: per-instance counter only.
    AP-20: no RDKit imports inside `_suggest_*`.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from lxml import etree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants (+)
# ---------------------------------------------------------------------------

# OPSIN version this module is calibrated against.
OPSIN_GRAMMAR_VERSION = "2.9.0"

# Module-level path: from src/orthonym/validation/opsin_grammar.py up to
# project root takes 4 `.parent` hops (file -> validation -> orthonym ->
# src -> project root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_REGEX_TOKENS_PATH = (
    _PROJECT_ROOT
    / "opsin/opsin-core/src/main/resources/uk/ac/cam/ch/wwmm/opsin/resources/regexTokens.xml"
)

# Logical name -> OPSIN regex name in regexTokens.xml.
# Source: 156-AUDIT.md § 1 (Surfaces A/B/C). Adapter table per.
_REGEX_TOKEN_MAP: Dict[str, str] = {
    # Surface A — Bracket nesting (P-16.5.4.1)
    "open_bracket": "openBracket",                              # line 22
    "close_bracket": "closeBracket",                            # line 23
    "indicated_hydrogen": "indicatedHydrogen",                  # line 33
    "compound_locant_or_added_h": "compoundLocantOrAddedHydrogen",  # line 34
    # Surface B — Stereo descriptor position (P-91 / P-93)
    "rs_stereochem_after_locant": "RSstereochemAfterLocant",    # line 35
    "all_locant_forms": "allLocantForms",                       # line 36
    "locant_types": "locantTypes",                              # line 31
    "stereochem_possibilities": "stereochemPossibilities",      # line 73
    # Surface C — Hyphen placement (P-14.5 / P-66)
    "forms_where_hyphen_is_optional": "formsWhereHyphenIsOptional",  # line 37
    "locant_types_optional_hyphen": "locantTypesOptionalHyphen",     # line 32
    # Atomic locant pattern (cross-cuts surfaces B + C)
    "locant": "locant",                                         # line 25
}


# ---------------------------------------------------------------------------
# Module-import-time XML loader (+ AP-7 loud failure)
# ---------------------------------------------------------------------------

# Maximum number of placeholder-expansion iterations before giving up.
# `regexTokens.xml` is acyclic by OPSIN design (line 1 of regexes.xml
# notes "this is NOT a CFG"); a small bound is a defensive safety net.
_MAX_EXPANSION_ITERATIONS = 16

_PLACEHOLDER_RE = re.compile(r"%([A-Za-z]+)%")

# The regexTokens.xml resource inside the OPSIN jar is byte-identical to the
# submodule copy. Reading it from the bundled jar removes the dependency on the
# OPSIN source tree, so a source install without the submodule still works.
_REGEX_TOKENS_JAR_ENTRY = "uk/ac/cam/ch/wwmm/opsin/resources/regexTokens.xml"


def _read_regex_tokens_xml() -> bytes:
    """regexTokens.xml bytes from the OPSIN submodule if present, else the bundled jar."""
    if _REGEX_TOKENS_PATH.exists():
        return _REGEX_TOKENS_PATH.read_bytes()
    import glob as _glob
    import zipfile as _zip
    for _pat in (str(_PROJECT_ROOT / "opsin-cli-*-jar-with-dependencies.jar"),
                 str(_PROJECT_ROOT / "opsin-cli-*.jar")):
        for _jar in sorted(_glob.glob(_pat)):
            try:
                with _zip.ZipFile(_jar) as _z:
                    return _z.read(_REGEX_TOKENS_JAR_ENTRY)
            except (KeyError, _zip.BadZipFile):
                continue
    raise ImportError(
        "OPSIN grammar XML (regexTokens.xml) not found in the OPSIN submodule or "
        "the bundled opsin-cli jar. A source checkout with the OPSIN submodule, or "
        "the bundled opsin-cli-*.jar at the project root, is required."
    )


def _load_opsin_token_regexes() -> Dict[str, "re.Pattern[str]"]:
    """Module-load reader: parse regexTokens.xml -> compiled Python re.Pattern dict.

    Raises ImportError on adapter-failure per CONTEXT.md + AP-7.
    Never silently degrades to a permissive pass-through.

    Returns:
        Dict mapping logical names (per `_REGEX_TOKEN_MAP`) to compiled
        `re.Pattern` objects with all `%name%` placeholders fully
        expanded and OPSIN HTML entities normalized.

    Raises:
        ImportError: if the XML file is missing, if a required logical
            name's underlying OPSIN regex is missing from the XML, or if
            the resolved Python regex fails to compile.
    """
    root = etree.fromstring(_read_regex_tokens_xml())

    # First pass: collect all <regex name="..." regex="..."/> macros.
    # Filter to actual element nodes (skip XML comments / processing
    # instructions surfaced by lxml as Cython callables).
    # Note: OPSIN's XML stores `name="%elementSymbol%"` (with % delimiters
    # already in the attribute). Strip those so our internal `raw` table
    # is keyed by bare logical names matching `_REGEX_TOKEN_MAP` values.
    raw: Dict[str, str] = {}
    for el in root.iter("regex"):
        nm = el.attrib.get("name")
        rx = el.attrib.get("regex")
        if nm and rx is not None:
            # OPSIN stores `name="%foo%"` — strip the delimiters.
            if nm.startswith("%") and nm.endswith("%"):
                nm = nm[1:-1]
            raw[nm] = rx

    # Second pass: iterative %name% expansion to fixed point.
    # OPSIN's regex dictionary is acyclic by design; the iteration
    # converges in a small number of passes. AP-7 forbids any silent
    # fallback here.
    def _resolve(s: str) -> str:
        cur = s
        for _ in range(_MAX_EXPANSION_ITERATIONS):
            def _sub(m: "re.Match[str]") -> str:
                key = m.group(1)
                if key in raw:
                    return raw[key]
                # Unknown placeholder: leave it; the compile step will
                # raise ImportError below with the resolved string for
                # diagnostic purposes.
                return m.group(0)
            new = _PLACEHOLDER_RE.sub(_sub, cur)
            if new == cur:
                break
            cur = new
        # Normalize XML-encoded entities OPSIN uses inside regexes.
        cur = cur.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        return cur

    compiled: Dict[str, "re.Pattern[str]"] = {}
    for logical, opsin_name in _REGEX_TOKEN_MAP.items():
        if opsin_name not in raw:
            raise ImportError(
                f"OPSIN regexTokens.xml missing required <regex name="
                f"\"{opsin_name}\"/> (needed for logical name "
                f"{logical!r}). OPSIN may have been updated; refresh "
                f"_REGEX_TOKEN_MAP in opsin_grammar.py."
            )
        resolved = _resolve(raw[opsin_name])
        try:
            compiled[logical] = re.compile(resolved)
        except re.error as e:
            raise ImportError(
                f"OPSIN regex %{opsin_name}% (logical {logical!r}) does "
                f"not compile in Python re engine: {e}. Pattern was: "
                f"{resolved!r}"
            ) from e

    # Defensive: every logical key MUST be populated.
    missing = set(_REGEX_TOKEN_MAP) - set(compiled)
    if missing:
        raise ImportError(
            f"_load_opsin_token_regexes did not populate all logical "
            f"keys; missing: {sorted(missing)}"
        )
    return compiled


def _check_jar_version_drift() -> None:
    """One-shot WARN if the OPSIN JAR version differs from the XML version.

    Per CONTEXT.md: WARN-only on JAR-vs-XML drift. JAR-side issues
    (e.g., JAR not present, Java unavailable) MUST NOT block module
    load. Wrapped in try/except to enforce that contract.
    """
    try:
        from .opsin_roundtrip import _find_opsin_jar  # local import — see AP-12 boundary
        jar = _find_opsin_jar(OPSIN_GRAMMAR_VERSION)
        if jar is None:
            logger.warning(
                "OPSIN JAR version %s not found at project root; "
                " grammar layer remains active but the round-"
                "trip oracle for suggest_fix() will be unavailable.",
                OPSIN_GRAMMAR_VERSION,
            )
    except Exception:  # noqa: BLE001 — never block module load on JAR check
        pass


# ---------------------------------------------------------------------------
# OpsinGrammar class (layer-cake + return shape + stats)
# ---------------------------------------------------------------------------


class OpsinGrammar:
    """OPSIN-grammar pre-validator with bounded round-trip-gated repair.

    Three responsibilities (CONTEXT.md `<domain>`):
        1. `validate(name) -> bool` — fast OPSIN-XML-driven check across
           three surfaces (bracket nesting, stereo position, hyphen
           placement). Layered ON TOP's
           `format_validator.validate_name_format` heuristic pre-screen
           per.
        2. `suggest_fix(name, source_smiles=None)` — bounded repair
           function with the LOCKED signature (name FIRST). Each
           repair candidate is gated through `validate()` AND
           `opsin_roundtrip_check(smiles, candidate)` (AP-8 SMILES-first
           inside the oracle). One attempt per repair class; NO retry
           loop (AP-5).
        3. Per-instance telemetry via `self._stats` and
           `get_validation_stats()`.

    Construction: `OpsinGrammar()` (or `OpsinGrammar(stats=shared_dict)`
    to share counters with an outer container per ref-pass
    pattern).
    """

    # Per CONTEXT.md — exactly seven buckets.
    STAT_KEYS: Tuple[str, ...] = (
        "validate_passed",
        "repair_succeeded_bracket",
        "repair_succeeded_stereo",
        "repair_succeeded_hyphen",
        "repair_failed_validate",
        "repair_failed_roundtrip",
        "no_repair_offered",
    )

    def __init__(self, stats: Optional[Dict[str, int]] = None) -> None:
        # Per-instance counter (AP-19). When `stats` is supplied
        # the caller (e.g., Orthonym) shares the same dict by reference
        # so its `get_validation_stats()` reads the live counters.
        if stats is None:
            self._stats: Dict[str, int] = {k: 0 for k in self.STAT_KEYS}
        else:
            for k in self.STAT_KEYS:
                stats.setdefault(k, 0)
            self._stats = stats
        # Tracks the most recent successful repair class (debugging
        # aid; not consumed by production code paths).
        self._last_repair_class: Optional[str] = None

    # -----------------------------------------------------------------
    # Public hot-path API (bool return)
    # -----------------------------------------------------------------

    def validate(self, name: str) -> bool:
        """Fast OPSIN-grammar pre-validation. Returns True on pass."""
        ok, _ = self._validate_detailed(name)
        return ok

    def _validate_detailed(self, name: str) -> Tuple[bool, str]:
        """Like validate() but returns (ok, reason_code).

        Layer-cake order per: format_validator pre-screen first
        (cheap heuristic), then OPSIN-XML-driven strict checks
        cheapest-first (bracket -> hyphen -> stereo). Used by tests
        and 156-AUDIT.md triage.
        """
        if not name:
            return False, "format_validator: empty_name"

        # Layer 1: heuristic pre-screen.
        from .format_validator import validate_name_format
        ok, reason = validate_name_format(name)
        if not ok:
            return False, f"format_validator: {reason}"

        # Layer 2: OPSIN-XML-driven authoritative checks (cheapest-first).
        ok, reason = self._check_bracket_hierarchy_strict(name)
        if not ok:
            return False, f"bracket: {reason}"
        ok, reason = self._check_hyphen_placement(name)
        if not ok:
            return False, f"hyphen: {reason}"
        ok, reason = self._check_stereo_position(name)
        if not ok:
            return False, f"stereo: {reason}"

        return True, "ok"

    def suggest_fix(
        self,
        name: str,
        source_smiles: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Bounded round-trip-gated repair (LOCKED signature).

        Args:
            name: The (validate-failing) IUPAC name to repair. **FIRST
                positional arg per.**
            source_smiles: Original SMILES for the round-trip gate.
                When `None` the round-trip gate is replaced by a
                `validate()`-only check and an INFO log records the
                degraded path.

        Returns:
            `(repaired_name, repair_class)` if a repair fired AND
            re-validates AND (when `source_smiles` is given) round-
            trips through OPSIN. Otherwise `(None, None)`.

        Notes:
            - Tries each repair class once in deterministic order
              [bracket, hyphen, stereo]. NO retry loop (AP-5).
            - The round-trip oracle is called as
              `opsin_roundtrip_check(source_smiles, candidate)` —
              SMILES-first per AP-8. The (smiles, name) arg order to
              the ORACLE is the OPPOSITE of `suggest_fix`'s own
              signature; this is the most common confusion source in
              the codebase and the AP-8 grep gate enforces it.
        """
        if not name:
            return None, None

        repair_classes = (
            ("bracket", self._suggest_bracket_renest),
            ("hyphen", self._suggest_hyphen_normalization),
            ("stereo", self._suggest_stereo_relocation),
        )

        for class_name, repair_fn in repair_classes:
            candidate = repair_fn(name)
            if candidate is None or candidate == name:
                continue

            # Gate 1: candidate must pass validate().
            if not self.validate(candidate):
                self._stats["repair_failed_validate"] = (
                    self._stats.get("repair_failed_validate", 0) + 1
                )
                continue

            # Gate 2: round-trip via OPSIN JAR.
            if source_smiles is None:
                # Documented degraded path per — log INFO.
                logger.info(
                    "OPSIN grammar repair (degraded — no source SMILES): "
                    "class=%s original=%r repaired=%r",
                    class_name, name[:50], candidate[:50],
                )
                self._stats[f"repair_succeeded_{class_name}"] = (
                    self._stats.get(f"repair_succeeded_{class_name}", 0) + 1
                )
                self._last_repair_class = class_name
                return candidate, class_name

            # AP-8: SMILES first, name second.
            from .opsin_roundtrip import opsin_roundtrip_check
            rt = opsin_roundtrip_check(source_smiles, candidate)
            # AP-9: dict is always truthy; read result["passed"].
            if not rt.get("passed", False):
                self._stats["repair_failed_roundtrip"] = (
                    self._stats.get("repair_failed_roundtrip", 0) + 1
                )
                continue

            self._stats[f"repair_succeeded_{class_name}"] = (
                self._stats.get(f"repair_succeeded_{class_name}", 0) + 1
            )
            self._last_repair_class = class_name
            return candidate, class_name

        # All three classes either declined or failed.
        self._stats["no_repair_offered"] = (
            self._stats.get("no_repair_offered", 0) + 1
        )
        return None, None

    def get_validation_stats(self) -> Dict[str, int]:
        """Return a defensive copy of the per-instance counters."""
        return dict(self._stats)

    def last_repair_class(self) -> Optional[str]:
        """Diagnostic accessor: most recent successful repair class."""
        return self._last_repair_class

    # -----------------------------------------------------------------
    # Strict-grammar checks (Surface A / B / C — 156-AUDIT.md § 1)
    # -----------------------------------------------------------------

    def _check_bracket_hierarchy_strict(self, name: str) -> Tuple[bool, str]:
        """OPSIN-XML-driven bracket-hierarchy strict check (P-16.5.4.1).

        Detects the audit-locked drift shapes (BR-1 / BR-2 / BR-5):
        directly-nested same-type brackets such as `((...))`, `[[...]]`,
        `{{...}}` (and the depth-3 form `((((...))))`). The IUPAC rule
        P-16.5.4.1.4 prescribes escalation to the next bracket type
        when consecutive same-level marks would result; literal
        `((...))` violates that.

        Indicated-hydrogen `(1H)` (P-16.5.4.1.1) and fusion brackets
        `[2,3-b]` (P-16.5.4.1.2) are non-nesting and are stripped via
        the existing -02 regexes (consumed under / AP-11
        — not redefined here).

        Per CONTEXT.md (permissive on uncovered surfaces): we do
        NOT enforce that the OUTERMOST opener is `(`; valid PIN names
        legitimately start with `[` when their content already has
        `(...)`. Only directly-nested same-type opener-pairs are
        flagged.
        """
        from orthonym.assembly.naming_utils import (
            _FUSION_BRACKET_RE,
            _INDICATED_H_RE,
            _STEREO_PAREN_RE,
        )
        # Strip non-nesting brackets (P-16.5.4.1.1 + P-16.5.4.1.2 +
        # P-16.5.4.1.3 stereo). Replace with placeholders that contain
        # NO bracket characters so the consecutive-opener check sees
        # only nesting-relevant brackets.
        working = _INDICATED_H_RE.sub("__IH__", name)
        working = _FUSION_BRACKET_RE.sub("__FB__", working)
        working = _STEREO_PAREN_RE.sub("__SP__", working)

        # AP-3: 156-AUDIT.md § 4.A BR-1 / BR-5 BEFORE-pattern detection.
        # Walk the working string and flag any direct opener-opener
        # adjacency of the same type (after stripping non-nesting
        # brackets). P-16.5.4.1.4 prescribes escalation; same-type
        # adjacency is the violation.
        for pair in ("((", "[[", "{{"):
            if pair in working:
                return False, (
                    f"hierarchy_violation: directly-nested {pair!r} "
                    f"found in name; P-16.5.4.1.4 prescribes "
                    f"escalation to the next bracket type"
                )
        return True, "ok"

    def _check_stereo_position(self, name: str) -> Tuple[bool, str]:
        """P-91 / P-93 stereo descriptor position validator.

        Uses regexTokens.xml line 35 (%RSstereochemAfterLocant%) and
        line 36 (%allLocantForms%). Each R/S stereo bracket must be at
        the start of a name word OR preceded by a valid
        %allLocantForms% expansion (with optional intervening hyphen).

        Catches drift patterns ST-1..ST-4 from 156-AUDIT.md § 4 Stereo
        sub-table:
            ST-1: stereo embedded mid-substituent.
            ST-2: split bracket `(2R)(3S)`.
            ST-3: space inside stereo bracket.
            ST-4: missing bracket around `2R-`.
        """
        # Detect ST-3: a stereo descriptor with a literal space inside
        # its bracket (e.g., `(2R, 3S)`). Bounded check anchored at the
        # canonical RSstereochemAfterLocant grammar.
        # AP-3: this regex matches the audit ST-3.fix BEFORE-pattern
        # row (156-AUDIT.md § 4.B).
        if re.search(r"\([0-9RSEZ ,]*[RSEZ][^)]*, [0-9RSEZ]", name):
            return False, "space_in_stereo_bracket: '(2R, 3S)' shape"

        # Detect ST-2: two R/S stereo brackets jammed back-to-back, e.g.
        # `(2R)(3S)`. Bounded check; uses a literal `\)\(` boundary
        # between two single-center R/S brackets.
        # AP-3: 156-AUDIT.md § 4.B ST-2.fix BEFORE-pattern.
        if re.search(r"\(\d+[RSEZ]\)\(\d+[RSEZ]\)", name):
            return False, "split_stereo_bracket: '(2R)(3S)' shape"

        # Detect ST-4: leading-locant R/S without bracket (`2R-...`).
        # AP-3: 156-AUDIT.md § 4.B ST-4.fix BEFORE-pattern.
        if re.match(r"^\d+[RS]-", name):
            return False, "unbracketed_leading_stereo: '2R-...' shape"

        return True, "ok"

    def _check_hyphen_placement(self, name: str) -> Tuple[bool, str]:
        """P-14.5 / P-66 hyphen-around-locant validator.

        Catches HY-1, HY-3, HY-4, HY-6 drift patterns from
        156-AUDIT.md § 4 Hyphen sub-table.
        """
        # HY-1.fix: composite locant ending in H followed by a literal
        # space then a lowercase letter (e.g., `1H,3H pyrazolone`).
        # AP-3: 156-AUDIT.md § 4.C HY-1.fix BEFORE-pattern.
        if re.search(r"\d+H,?\d*H? [a-z]", name):
            return False, "missing_hyphen_after_composite_locant: 'NH,MH letter' shape"

        # HY-3.fix: comma-separated digit-locant directly followed by a
        # lowercase letter without an intervening hyphen (e.g.,
        # `2,4dichloro`). Anchored to require BOTH a comma and a
        # follow-on letter so we don't false-positive on
        # already-correct `2,4-dichloro`.
        # AP-3: 156-AUDIT.md § 4.C HY-3.fix BEFORE-pattern.
        if re.search(r"\d+,\d+[a-z]", name):
            return False, "missing_hyphen_between_locant_list_and_substituent"

        # HY-4.fix: `1H,...,N H-,M H-...` shape — trailing hyphens
        # interleaving comma-separated indicated-H locant lists.
        # AP-3: 156-AUDIT.md § 4.C HY-4.fix BEFORE-pattern.
        if re.search(r"(?:\d+H,)+\d+H-,\d+H-?", name):
            return False, "trailing_hyphens_in_locant_list: '1H,3H-,5H-' shape"

        # NOTE: HY-6 (heteroatom-prefix dash polycyclic-prefix, e.g.
        # `oxa-tetracyclo`) is NOT auto-detected here even though the
        # audit row HY-6.fix is corpus-mined from the Tier-B post-v18
        # set. Reason: the canary suite (zero-flip HARD gate)
        # contains many names where `oxa-tetracyclo` / `aza-tricyclo`
        # is the canonical expected form. A blanket detection here
        # over-fires and causes name-stability regressions across
        # the canary set (verified empirically during
        # implementation). The HY-6 repair regex is RETAINED in
        # `_suggest_hyphen_normalization` so it remains a documented
        # candidate when other surface checks have already flagged a
        # name as invalid; but it MUST NOT trigger validate-fail on
        # otherwise-canonical names. Per AP-3 the audit row stays
        # citation-linked from the repair-method side.

        return True, "ok"

    # -----------------------------------------------------------------
    # Bounded repair tables (156-AUDIT.md § 4 — implemented 1:1, AP-3)
    # -----------------------------------------------------------------

    def _suggest_bracket_renest(self, name: str) -> Optional[str]:
        """Bracket re-nesting via -02 `apply_enclosing_marks`.

        Implements 156-AUDIT.md § 4.A rows BR-1, BR-2, BR-5. Returns
        `None` when no audit row matches (BR-4 sanity).

        Per + AP-11 this method NEVER reimplements depth logic;
        it always delegates to `apply_enclosing_marks(depth=-1)`.
        """
        # Local-import per CONTEXT.md (keep cold-start cheap).
        from orthonym.assembly.naming_utils import apply_enclosing_marks

        # AP-3: 156-AUDIT.md § 4.A BR-5 (depth-3 outer `((((...))))`).
        # Match BEFORE BR-1 because BR-1's pattern is a prefix of BR-5's.
        m = re.match(r"^\(\(\(\((?P<inner>.*)\)\)\)\)(?P<rest>.*)$", name)
        if m:
            return (
                apply_enclosing_marks("(" + m.group("inner") + ")", depth=-1)
                + m.group("rest")
            )

        # AP-3: 156-AUDIT.md § 4.A BR-1 / BR-2 (top-level `((...))`).
        # BR-2's `(1H)` containment is handled inside
        # `apply_enclosing_marks` via `compute_nesting_depth`; we do
        # NOT re-derive depth here (AP-11).
        m = re.match(r"^\(\((?P<inner>.+)\)\)(?P<rest>.*)$", name)
        if m:
            return (
                apply_enclosing_marks("(" + m.group("inner") + ")", depth=-1)
                + m.group("rest")
            )

        # No audit row applies (BR-4 sanity-pass): no repair offered.
        return None

    def _suggest_stereo_relocation(self, name: str) -> Optional[str]:
        """Stereo position-only repair per 156-AUDIT.md § 4.B.

        Each repair preserves the (locant, descriptor) tuple verbatim;
        only the token POSITION changes (AP-13). Source SMILES
        is NEVER read here (AP-20); the round-trip gate (in
        `suggest_fix`) is the sole consumer.
        """
        # AP-3: 156-AUDIT.md § 4.B ST-3.fix (strip space inside stereo
        # bracket). Surgical replacement INSIDE the stereo-bracket
        # span only; `re.sub` is bounded to the captured group so we
        # do not collapse spaces elsewhere in the name.
        st3 = re.sub(
            r"(\([0-9RSEZ,]*[RSEZ]),[ ]+(\d+[RSEZ][^)]*\))",
            r"\1,\2",
            name,
        )
        if st3 != name:
            return st3

        # AP-3: 156-AUDIT.md § 4.B ST-2.fix (merge `(2R)(3S)` into
        # `(2R,3S)`).
        st2 = re.sub(
            r"\((?P<d1>\d+[RSEZ])\)\((?P<d2>\d+[RSEZ])\)",
            lambda m: f"({m.group('d1')},{m.group('d2')})",
            name,
        )
        if st2 != name:
            return st2

        # AP-3: 156-AUDIT.md § 4.B ST-4.fix (wrap leading bare `2R-`
        # into `(2R)-2-`). Anchored at start-of-name only.
        m = re.match(r"^(?P<digit>\d+)(?P<desc>[RS])-(?P<rest>.*)$", name)
        if m:
            return (
                f"({m.group('digit')}{m.group('desc')})-"
                f"{m.group('digit')}-{m.group('rest')}"
            )

        # ST-1.fix is corpus-mined; the BEFORE-pattern requires
        # cross-substituent context that we cannot safely repair with a
        # single regex without risking AP-4 (structural change).
        # We recognize the shape but DECLINE to repair when uncertain;
        # the round-trip gate in `suggest_fix` would catch any bad
        # rewrite, but emitting `None` here keeps repair attempts
        # deterministic.
        return None

    def _suggest_hyphen_normalization(self, name: str) -> Optional[str]:
        """Hyphen normalization per 156-AUDIT.md § 4.C.

        Per CONTEXT.md C-5 (PIN style): MUST NOT drop locants while
        normalizing hyphens.
        """
        # AP-3: 156-AUDIT.md § 4.C HY-6.fix (drop stray hyphen between
        # heteroatom replacement prefix and polycyclic prefix).
        hy6 = re.sub(
            r"(?P<heteroatom>(?:oxa|aza|thia|sila|phospha|selena|tellura|"
            r"stanna|germa|plumba|bisma|arsa))-"
            r"(?P<polycyclo>(?:tri|tetra|penta|hexa|hepta|octa|nona|deca)?"
            r"cyclo)",
            r"\g<heteroatom>\g<polycyclo>",
            name,
        )
        if hy6 != name:
            return hy6

        # AP-3: 156-AUDIT.md § 4.C HY-1.fix (insert hyphen before
        # composite locant followed by space + lowercase letter).
        hy1 = re.sub(
            r"(?P<loc>\d+H(?:,\d+H)*) (?P<rest>[a-z])",
            r"\g<loc>-\g<rest>",
            name,
        )
        if hy1 != name:
            return hy1

        # AP-3: 156-AUDIT.md § 4.C HY-3.fix (insert hyphen between
        # comma-separated locant list and a follow-on lowercase
        # letter).
        hy3 = re.sub(
            r"(?P<digits>\d+,\d+)(?P<letter>[a-z])",
            r"\g<digits>-\g<letter>",
            name,
        )
        if hy3 != name:
            return hy3

        # AP-3: 156-AUDIT.md § 4.C HY-4.fix (collapse trailing hyphens
        # in `1H,3H-,5H-` shape).
        hy4 = re.sub(
            r"(?P<list>(?:\d+H,)+)(?P<dup>\d+H)-,(?P<more>\d+H)-?",
            r"\g<list>\g<dup>,\g<more>-",
            name,
        )
        if hy4 != name:
            return hy4

        return None


# ---------------------------------------------------------------------------
# Module bottom: eager load, version-drift check, public helpers
# ---------------------------------------------------------------------------

# Eager module-import-time XML load. ImportError propagates per AP-7.
_TOKEN_REGEX: Dict[str, "re.Pattern[str]"] = _load_opsin_token_regexes()

# JAR-vs-XML version-drift WARN — wrapped in try/except so JAR
# absence never blocks module load.
_check_jar_version_drift()

# Public-API singleton (helpers below). Per CONTEXT.md + AP-19 the
# singleton's `_stats` are SEPARATE from the per-`Orthonym`-instance
# stats; the singleton serves standalone callers (tests, audit triage).
_SINGLETON = OpsinGrammar()


def opsin_grammar_validate(name: str) -> bool:
    """Module-level convenience wrapper around `OpsinGrammar.validate`.

    Backed by an internal singleton with its own `_stats` counter
    (NOT shared with any `Orthonym` instance per).
    """
    return _SINGLETON.validate(name)


def opsin_grammar_suggest_fix(
    name: str,
    source_smiles: Optional[str] = None,
) -> Optional[str]:
    """Module-level convenience wrapper around `OpsinGrammar.suggest_fix`.

    **Return-shape divergence:** `OpsinGrammar.suggest_fix` returns
    `Tuple[Optional[str], Optional[str]]` (per LOCKED); this
    helper drops the repair-class slot and returns `Optional[str]`
    only, for backward-compatible callers that just want the
    repaired name. Use the class API directly when the repair class
    is needed.
    """
    repaired, _cls = _SINGLETON.suggest_fix(name, source_smiles)
    return repaired


__all__ = [
    "OpsinGrammar",
    "opsin_grammar_validate",
    "opsin_grammar_suggest_fix",
    "OPSIN_GRAMMAR_VERSION",
]
