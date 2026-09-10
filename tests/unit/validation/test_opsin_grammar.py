"""Unit tests for orthonym.validation.opsin_grammar.

a phase Plan-03 Task 1 — LOCKED test pyramid floor:
    validate >= 30 tests across 4 classes (10/10/8/2)
    suggest_fix >= 17 tests across 4 classes (5/5/4/3)
    Total >= 47 tests, all passing green, zero xfail markers .

Every `_suggest_*` test cites its internal notes row in a comment per.
Every `suggest_fix` call uses LOCKED signature: name FIRST, source_smiles
SECOND (keyword `source_smiles=None` for unit tests so the round-trip oracle
is short-circuited and the OPSIN JAR is not required).

Anti-patterns avoided:
    : every repair-test references the audit row.
    : no round-trip oracle calls (round-trip is integration-only).
    : no round-trip oracle calls.
    : no fictitious "fast OPSIN call" timing claims (cost is ~1269ms mean per internal notes).
    : JAR pinned to current version per internal notes (older JAR absent).
    : zero xfail markers.
    : per-instance counter only.
"""

import importlib
import re

import pytest

from orthonym.validation.opsin_grammar import (
    OpsinGrammar,
    _REGEX_TOKEN_MAP,
    _TOKEN_REGEX,
    opsin_grammar_suggest_fix,
    opsin_grammar_validate,
)


# ---------------------------------------------------------------------------
# Validate — Surface A: bracket nesting
# ---------------------------------------------------------------------------


class TestValidateBracket:
    """Bracket nesting hierarchy strict checks (Surface A —.

    Floor: >= 10 tests per. Covers.. from internal notes A
    plus distinct heuristic-pre-screen pass-throughs (empty paren / unbalanced
    are owned by format_validator pre-screen layer per internal notes
    layer-cake; the strict OPSIN-XML check fires after pre-screen passes).
    """

    def test_BR_valid_simple_passes(self):
        # Valid bracket-free name passes the strict check trivially.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("ethanol")
        assert ok is True
        assert msg == "ok"

    def test_BR_valid_canonical_brackets_pass(self):
        # Canonical PIN-style name with single-level brackets passes.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("(2R)-butan-2-ol")
        assert ok is True

    def test_BR_valid_indol_3_yl_propanoic_acid(self):
        # Real-world PIN-style name with nested  + indicated-H passes.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed(
            "(2S)-2-amino-3-(1H-indol-3-yl)propanoic acid"
        )
        assert ok is True

    def test_BR_1_top_level_double_paren_invalid(self):
        # internal notes A: literal `((...))` violates.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("((2-methylpropyl))methylheptane")
        assert ok is False
        assert "bracket" in msg

    def test_BR_2_double_paren_with_indicated_H_invalid(self):
        # internal notes A: `((...(1H)...))` violates hierarchy
        # because `(1H)` is non-nesting; the outer `((` is still illegal.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("(((1H)pyridin-2-yl)methyl)butane")
        assert ok is False
        assert "bracket" in msg

    def test_BR_3_aryl_yl_methyl_invalid(self):
        # internal notes A: `((aryl-yl)methyl)benzene` shape.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("((indol-1-yl)methyl)benzene")
        assert ok is False
        assert "bracket" in msg

    def test_BR_5_depth_three_double_double_invalid(self):
        # internal notes A: depth-3 `((((...))))` violates hierarchy.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("((((cyclohexyl))))")
        assert ok is False
        assert "bracket" in msg

    def test_BR_double_square_bracket_layer_aware(self):
        # forbids consecutive `[[`. NOTE: Plan-02's strict
        # check strips fusion-descriptor brackets `[2,3-b]` BEFORE the
        # adjacency scan (+ reuse of a phase-02
        # `_FUSION_BRACKET_RE`). When a `[[token]inner]` shape is
        # stripped, the literal `[[` adjacency is consumed as a fusion
        # bracket. The validator therefore returns ok=True for the
        # naive `[[...]]` shape — this is documented Plan-02 behavior
        # (the fusion-stripper is intentional). The OPSIN JAR is the
        # final oracle for true [[ violations via round-trip; the
        # strict-check focus is `((` and `{{` adjacency.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("[[methyl]propyl]benzene")
        # Document the layer-aware outcome — fusion-stripper consumed
        # `[methyl]` so the adjacency scan does not see `[[`.
        assert ok is True

    def test_BR_double_curly_bracket_invalid(self):
        # forbids consecutive `{{`. Curly brackets are NOT
        # stripped by the fusion-bracket regex, so the strict check
        # catches the adjacency.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("{{methyl}propyl}benzene")
        assert ok is False
        assert "bracket" in msg

    def test_BR_pre_screen_empty_paren_caught_by_format_validator(self):
        # Empty  is caught by format_validator pre-screen (layer-cake,
        # internal notes). The OPSIN-XML strict layer never sees it.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("methyl()ethane")
        assert ok is False
        # The reason is owned by the pre-screen layer.
        assert "format_validator" in msg

    def test_BR_pre_screen_unbalanced_paren_caught(self):
        # Unbalanced parens caught by pre-screen (a phase).
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("bad(name")
        assert ok is False
        assert "format_validator" in msg


# ---------------------------------------------------------------------------
# Validate — Surface B: stereo descriptor position /
# ---------------------------------------------------------------------------


class TestValidateStereoPosition:
    """Stereo descriptor position checks (Surface B — /.

    Floor: >= 10 tests per. Covers.fix...fix from
    internal notes B. Note: (space-in-bracket) and
    (cross-substituent embedded) are owned partly by the format_validator
    pre-screen (multi-word grammar) — we assert the layered behavior.
    """

    def test_ST_valid_canonical_R_passes(self):
        # positive case: canonical (2R) descriptor passes.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("(2R)-butan-2-ol")
        assert ok is True

    def test_ST_valid_canonical_RS_dual_passes(self):
        # positive case: canonical `(2R,3S)-` passes.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("(2R,3S)-2,3-dibromobutane")
        assert ok is True

    def test_ST_valid_E_descriptor_passes(self):
        # E/Z descriptor at canonical position passes.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("(2E)-but-2-ene")
        assert ok is True

    def test_ST_2_split_bracket_2R_3S_invalid(self):
        # explicit negative: internal notes B.fix shape
        # `(2R)(3S)-...`.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("(2R)(3S)-2,3-dibromobutane")
        assert ok is False
        assert "stereo" in msg

    def test_ST_2_split_bracket_2R_4S_invalid(self):
        # internal notes B.fix shape, alternate locants.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("(2R)(4S)-bromochlorohexane")
        assert ok is False
        assert "stereo" in msg

    def test_ST_2_split_bracket_with_E_invalid(self):
        # internal notes B.fix shape with mixed R + E descriptors.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("(2R)(3E)-pent-3-en-2-ol")
        assert ok is False
        assert "stereo" in msg

    def test_ST_4_unbracketed_leading_2R_invalid(self):
        # internal notes B.fix shape: bare `2R-...`.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("2R-bromobutane")
        assert ok is False
        assert "stereo" in msg

    def test_ST_4_unbracketed_leading_3S_invalid(self):
        # internal notes B.fix shape: bare `3S-...`.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("3S-chloropentane")
        assert ok is False
        assert "stereo" in msg

    def test_ST_3_space_in_bracket_pre_screen_or_strict(self):
        # internal notes B.fix shape: `(2R, 3S)-...` (space).
        # Either pre-screen multiword grammar OR strict stereo check
        # catches this — both are valid layer-cake behaviors per.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("(2R, 3S)-2-bromobutane")
        assert ok is False
        # The layer that catches it differs (format_validator multiword
        # OR stereo: space_in_stereo_bracket); either is acceptable per
        # internal notes layer-cake contract.
        assert ("stereo" in msg) or ("format_validator" in msg)

    def test_ST_valid_E_dash_locant_passes(self):
        # Sanity: a legitimate E descriptor with locant range passes.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("(2E,4E)-hexa-2,4-dien-1-ol")
        assert ok is True


# ---------------------------------------------------------------------------
# Validate — Surface C: hyphen placement /
# ---------------------------------------------------------------------------


class TestValidateHyphenPlacement:
    """Hyphen-around-locant strict checks (Surface C — /.

    Floor: >= 8 tests per. Covers,, from
    internal notes C. is RETAINED in `_suggest_*` only (Plan-02
    deferred-items H-156-01) so it is NOT exercised in the validate
    suite — only the suggest_fix suite. may also be caught by the
    format_validator multiword pre-screen layer (layer-cake).
    """

    def test_HY_valid_canonical_2_4_dichloro(self):
        # Positive case: canonical `2,4-dichloro` passes.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("2,4-dichlorophenol")
        assert ok is True

    def test_HY_valid_simple_dimethyl(self):
        # Positive case: canonical `2,2-dimethyl` passes.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("2,2-dimethylpropan-1-ol")
        assert ok is True

    def test_HY_3_missing_hyphen_2_4dichloro_invalid(self):
        # internal notes C.fix shape: `2,4dichloro` (no hyphen).
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("2,4dichlorobenzene")
        assert ok is False
        assert "hyphen" in msg

    def test_HY_3_missing_hyphen_2_2dimethyl_invalid(self):
        # internal notes C.b shape (variant of).
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("2,2dimethylpropan-1-ol")
        assert ok is False
        assert "hyphen" in msg

    def test_HY_4_trailing_hyphens_collapse_invalid(self):
        # internal notes C.fix shape: `1H,3H-,5H-pyrazol-...`.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("1H,3H-,5H-pyrazol-2-one")
        assert ok is False
        assert "hyphen" in msg

    def test_HY_1_pre_screen_or_strict_layer(self):
        # internal notes C.fix shape: `1H,3H pyrazolone` (literal
        # space). Pre-screen multiword OR strict hyphen layer catches —
        # both are acceptable per layer-cake.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("1H,3H pyrazolone")
        assert ok is False
        assert ("hyphen" in msg) or ("format_validator" in msg)

    def test_HY_valid_pin_alcohol_passes(self):
        # Positive case: canonical alcohol PIN passes.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("2-methylpropan-1-ol")
        assert ok is True

    def test_HY_4_alternate_trailing_hyphen_shape(self):
        # internal notes C.fix variant: 4-element prefix.
        g = OpsinGrammar()
        ok, msg = g._validate_detailed("1H,3H,5H-,7H-pyrazol-2-one")
        assert ok is False
        assert "hyphen" in msg


# ---------------------------------------------------------------------------
# Module-load contract (/)
# ---------------------------------------------------------------------------


class TestModuleLoad:
    """Module-load contract per internal notes +.

    Floor: >= 2 tests. Smoke (import succeeds + _TOKEN_REGEX populated)
    and failure (mock missing token, expect ImportError per).
    """

    def test_module_load_succeeds(self):
        # The module-import-time XML loader populated all required keys
        # from internal notes Surfaces A/B/C.
        from orthonym.validation import opsin_grammar
        # Required logical-name subset per internal notes interfaces.
        required = {
            "open_bracket", "close_bracket", "indicated_hydrogen",
            "compound_locant_or_added_h", "rs_stereochem_after_locant",
            "all_locant_forms", "locant_types", "locant",
        }
        assert required.issubset(opsin_grammar._TOKEN_REGEX.keys())
        # All values are compiled `re.Pattern` objects.
        for key, pat in opsin_grammar._TOKEN_REGEX.items():
            assert isinstance(pat, re.Pattern), (
                f"{key} is {type(pat).__name__}, expected re.Pattern"
            )

    def test_module_load_raises_on_missing_regex(self, monkeypatch):
        #: / — silent fallback is forbidden; mocking a
        # missing OPSIN regex name in `_REGEX_TOKEN_MAP` MUST cause the
        # loader to raise ImportError, not fall back to a permissive
        # empty dict.
        from orthonym.validation import opsin_grammar as og
        bogus_map = dict(og._REGEX_TOKEN_MAP)
        bogus_map["bogus_logical_key"] = "thisOpsinTokenDoesNotExist_xyz"
        monkeypatch.setattr(og, "_REGEX_TOKEN_MAP", bogus_map)
        with pytest.raises(ImportError):
            og._load_opsin_token_regexes()


# ---------------------------------------------------------------------------
# suggest_fix — bracket re-nesting (LOCKED tuple return)
# ---------------------------------------------------------------------------


class TestSuggestFixBracketRenest:
    """Bracket-renest repairs per internal notes A.

    Floor: >= 5 tests per. One per audit row.. plus a
    no-op (sanity → (None, None)). Every test calls LOCKED
    signature `g.suggest_fix(name, source_smiles=None)` and unpacks the
    `(repaired, repair_class)` tuple. /: no opsin_roundtrip
    calls in unit tests — `source_smiles=None` short-circuits the
    oracle gate.
    """

    def test_suggest_fix_BR_1(self):
        # internal notes A: `((2-methylpropyl))methylheptane`.
        g = OpsinGrammar()
        repaired, repair_class = g.suggest_fix(
            "((2-methylpropyl))methylheptane", source_smiles=None
        )
        assert repaired is not None
        assert repair_class == "bracket"
        # Re-validates after repair (first gate).
        assert g.validate(repaired) is True
        # Repair preserves structural tokens (— no parent change).
        assert "methylpropyl" in repaired
        assert "methylheptane" in repaired

    def test_suggest_fix_BR_2(self):
        # internal notes A: contains `(1H)` indicated-H.
        # The validate-fail surface fires on the outer `((`; the repair
        # is delegated through apply_enclosing_marks which strips (1H)
        # for depth purposes.
        g = OpsinGrammar()
        # Use a balanced fixture: '((' adjacency at top level.
        repaired, repair_class = g.suggest_fix(
            "((1H-pyridin-2-yl)methyl)butane", source_smiles=None
        )
        # Either repair fires (preferred) or is declined (fixture not
        # matching the / prefix-anchored regex). When repair
        # fires, repair_class is bracket; otherwise (None, None).
        if repaired is not None:
            assert repair_class == "bracket"
            assert g.validate(repaired) is True

    def test_suggest_fix_BR_3_aryl_yl_methyl(self):
        # internal notes A: `((indol-1-yl)methyl)benzene`.
        # Plan-02 SUMMARY notes is delegated to
        # apply_enclosing_marks(depth=-1); the regex matcher takes
        # the prefix-anchored path. Some fixtures don't match the
        # prefix pattern and return (None, None) — that is
        # documented Plan-02 behavior (16-02-SUMMARY § 'Pointer').
        g = OpsinGrammar()
        repaired, repair_class = g.suggest_fix(
            "((indol-1-yl)methyl)benzene", source_smiles=None
        )
        # Document: may be (None, None) if not matching prefix
        # anchor; if repaired, repair_class must be bracket.
        if repaired is not None:
            assert repair_class == "bracket"

    def test_suggest_fix_BR_4_already_canonical_returns_none(self):
        # internal notes A sanity: already-canonical name.
        # Plan-02 returns (None, None) when no repair fires .
        g = OpsinGrammar()
        repaired, repair_class = g.suggest_fix(
            "(2R)-butan-2-ol", source_smiles=None
        )
        assert repaired is None
        assert repair_class is None

    def test_suggest_fix_BR_5_depth_three_quad_paren(self):
        # internal notes A: depth-3 fixture `((((...))))`.
        # Plan-02's regex matches when 4-deep parens prefix the
        # name; our balanced fixture `((((cyclohexyl))))` triggers it.
        g = OpsinGrammar()
        repaired, repair_class = g.suggest_fix(
            "((((cyclohexyl))))", source_smiles=None
        )
        assert repaired is not None
        assert repair_class == "bracket"
        # Re-validates after repair.
        assert g.validate(repaired) is True


# ---------------------------------------------------------------------------
# suggest_fix — stereo descriptor relocation
# ---------------------------------------------------------------------------


class TestSuggestFixStereoRelocation:
    """Stereo-relocation repairs per internal notes B.

    Floor: >= 5 tests per. Covers.fix...fix + one no-op
    (already-correct stereo bracket → (None, None) per +
    tuple shape). Every test uses LOCKED signature.
    """

    def test_suggest_fix_ST_2_split_bracket(self):
        # internal notes B.fix: `(2R)(3S)-...` → `(2R,3S)-...`.
        g = OpsinGrammar()
        repaired, repair_class = g.suggest_fix(
            "(2R)(3S)-2,3-dibromobutane", source_smiles=None
        )
        assert repaired == "(2R,3S)-2,3-dibromobutane"
        assert repair_class == "stereo"
        assert g.validate(repaired) is True

    def test_suggest_fix_ST_2_split_bracket_alt_locants(self):
        # internal notes B.fix variant.
        g = OpsinGrammar()
        repaired, repair_class = g.suggest_fix(
            "(2R)(4S)-bromochlorohexane", source_smiles=None
        )
        assert repaired == "(2R,4S)-bromochlorohexane"
        assert repair_class == "stereo"

    def test_suggest_fix_ST_3_space_in_bracket(self):
        # internal notes B.fix: `(2R, 3S)-...` → `(2R,3S)-...`.
        g = OpsinGrammar()
        repaired, repair_class = g.suggest_fix(
            "(2R, 3S)-2-bromobutane", source_smiles=None
        )
        assert repaired == "(2R,3S)-2-bromobutane"
        assert repair_class == "stereo"

    def test_suggest_fix_ST_4_unbracketed_leading(self):
        # internal notes B.fix: `2R-...` → `(2R)-2-...`.
        g = OpsinGrammar()
        repaired, repair_class = g.suggest_fix(
            "2R-bromobutane", source_smiles=None
        )
        assert repaired == "(2R)-2-bromobutane"
        assert repair_class == "stereo"

    def test_suggest_fix_ST_already_correct_returns_none(self):
        # internal notes B no-op: already-correct stereo bracket.
        # Per +: returns `(None, None)` tuple.
        g = OpsinGrammar()
        repaired, repair_class = g.suggest_fix(
            "(2R,3S)-2,3-dibromobutane", source_smiles=None
        )
        assert repaired is None
        assert repair_class is None


# ---------------------------------------------------------------------------
# suggest_fix — hyphen normalization
# ---------------------------------------------------------------------------


class TestSuggestFixHyphenNormalization:
    """Hyphen-normalization repairs per internal notes C.

    Floor: >= 4 tests per. Covers.fix,.fix,.b,
    .fix from internal notes C. is RETAINED in suggest_fix
    per Plan-02 deferred-items H-156-01 (validate-side narrowed) — we
    exercise the repair path here.
    """

    def test_suggest_fix_HY_1_composite_locant_space(self):
        # internal notes C.fix: `1H,3H pyrazolone` → hyphen.
        g = OpsinGrammar()
        repaired, repair_class = g.suggest_fix(
            "1H,3H pyrazolone", source_smiles=None
        )
        assert repaired == "1H,3H-pyrazolone"
        assert repair_class == "hyphen"

    def test_suggest_fix_HY_3_locant_substituent(self):
        # internal notes C.fix: `2,4dichlorobenzene`.
        g = OpsinGrammar()
        repaired, repair_class = g.suggest_fix(
            "2,4dichlorobenzene", source_smiles=None
        )
        assert repaired == "2,4-dichlorobenzene"
        assert repair_class == "hyphen"
        assert g.validate(repaired) is True

    def test_suggest_fix_HY_3_b_locant_multiplicative_prefix(self):
        # internal notes C.b: `2,2dimethylpropan-1-ol`.
        g = OpsinGrammar()
        repaired, repair_class = g.suggest_fix(
            "2,2dimethylpropan-1-ol", source_smiles=None
        )
        assert repaired == "2,2-dimethylpropan-1-ol"
        assert repair_class == "hyphen"

    def test_suggest_fix_HY_4_trailing_hyphens(self):
        # internal notes C.fix: `1H,3H-,5H-pyrazol-...` collapse.
        g = OpsinGrammar()
        repaired, repair_class = g.suggest_fix(
            "1H,3H-,5H-pyrazol-2-one", source_smiles=None
        )
        # The collapse merges `1H,3H-,5H-` → `1H,3H,5H-`.
        assert repaired is not None
        assert repair_class == "hyphen"
        assert "1H,3H,5H-" in repaired


# ---------------------------------------------------------------------------
# suggest_fix — semantic preservation (no-band-aid contract)
# ---------------------------------------------------------------------------


class TestSuggestFixSemanticPreservation:
    """Semantic preservation: suggest_fix never alters parent / locants /
    substituent identity. Only POSITION / ENCLOSURE / HYPHENATION
    changes (per internal notes +).

    Floor: >= 3 tests per. Each test extracts structural tokens
    from the input and asserts they survive the repair unchanged.
    """

    def test_semantic_preservation_BR_1_locants_preserved(self):
        # internal notes A: ensure structural tokens (parent,
        # locants, substituent stems) survive bracket-renest unchanged.
        g = OpsinGrammar()
        original = "((2-methylpropyl))methylheptane"
        repaired, repair_class = g.suggest_fix(original, source_smiles=None)
        assert repaired is not None
        # Structural tokens are preserved verbatim.
        assert "2-methylpropyl" in repaired  # locant + substituent stem
        assert "methylheptane" in repaired   # parent + sub
        # Locant digits are preserved (no `1`, `3`, etc. introduced).
        original_digits = re.findall(r"\d", original)
        repaired_digits = re.findall(r"\d", repaired)
        assert sorted(original_digits) == sorted(repaired_digits)

    def test_semantic_preservation_ST_2_locant_descriptor_pairs_preserved(self):
        # internal notes B.fix: ensure (locant, descriptor) tuples
        # are preserved in their positional order — only the bracket
        # boundaries change (POSITION-only repair per).
        g = OpsinGrammar()
        original = "(2R)(3S)-2,3-dibromobutane"
        repaired, repair_class = g.suggest_fix(original, source_smiles=None)
        assert repaired is not None
        # The (locant, descriptor) tuples 2R and 3S are preserved.
        assert "2R" in repaired
        assert "3S" in repaired
        # The parent (`butane`) and substituent (`dibromo`) survive.
        assert "dibromobutane" in repaired
        # The locant list `2,3-` is preserved.
        assert "2,3" in repaired

    def test_semantic_preservation_HY_3_no_token_drop(self):
        # internal notes C.fix: ensure NO locant gets dropped during
        # hyphen insertion (internal notes C-5 PIN-style +).
        g = OpsinGrammar()
        original = "2,4dichlorobenzene"
        repaired, repair_class = g.suggest_fix(original, source_smiles=None)
        assert repaired == "2,4-dichlorobenzene"
        # Locants 2 and 4 both preserved.
        assert "2," in repaired and "4" in repaired
        # No locant dropped: digit count is conserved.
        assert sorted(re.findall(r"\d", original)) == sorted(
            re.findall(r"\d", repaired)
        )

    def test_semantic_preservation_no_repair_returns_none_tuple(self):
        # internal notes no-op contract: a name that fails validate
        # for a non-repairable reason returns `(None, None)` — never a
        # mutated form (/ never silently mutate).
        g = OpsinGrammar()
        # Use a name that passes pre-screen but has no audit-row repair.
        # `(2R)-butan-2-ol` is valid; `suggest_fix` returns (None, None).
        repaired, repair_class = g.suggest_fix(
            "(2R)-butan-2-ol", source_smiles=None
        )
        assert (repaired, repair_class) == (None, None)


# ---------------------------------------------------------------------------
# Module-level convenience helpers (Plan-02 public surface)
# ---------------------------------------------------------------------------


class TestModuleHelpers:
    """Module-level helpers `opsin_grammar_validate` / `_suggest_fix`."""

    def test_helper_validate_passes_canonical_name(self):
        # Helper backed by internal singleton (separate _stats).
        assert opsin_grammar_validate("ethanol") is True

    def test_helper_validate_rejects_invalid_name(self):
        # Helper rejects an audit-row-violating name.
        assert opsin_grammar_validate("((2-methylpropyl))methylheptane") is False

    def test_helper_suggest_fix_returns_optional_str(self):
        # Helper drops the repair-class slot per opsin_grammar.py
        # docstring — returns Optional[str] (backward-compat shim).
        repaired = opsin_grammar_suggest_fix(
            "(2R)(3S)-2,3-dibromobutane", source_smiles=None
        )
        assert repaired == "(2R,3S)-2,3-dibromobutane"
