"""Tests for IUPAC Blue Book errata corrections to assembly/formatting logic.

ERRATA-07: Bracket nesting rules expanded per P-16.5.4.1 (5 new subsections, Dec 2025)
ERRATA-10: First substituent no enclosing marks for mononuclear parent hydrides (P-16.5.1.3)
"""

import pytest

from orthonym.assembly.naming_utils import apply_enclosing_marks, compute_nesting_depth
from orthonym.rules.phosphorus import _build_substituent_string


# ===========================================================================
# ERRATA-07: compute_nesting_depth tests (P-16.5.4.1)
# ===========================================================================

class TestComputeNestingDepth:
    """Test the new compute_nesting_depth() function per P-16.5.4.1 subsections."""

    def test_plain_name_depth_zero(self):
        """A plain name without any brackets has depth 0."""
        assert compute_nesting_depth("methylpropyl") == 0

    def test_name_with_regular_parentheses_depth_one(self):
        """A name with regular parentheses (e.g., compound sub) has depth >= 1."""
        # "(2-methylpropyl)" has nesting-relevant parentheses
        assert compute_nesting_depth("(2-methylpropyl)") >= 1

    def test_indicated_hydrogen_ignored(self):
        """P-16.5.4.1.1: Indicated hydrogen parentheses like (1H) do NOT count."""
        # A name containing (1H) should have depth 0 from (1H) alone
        assert compute_nesting_depth("(1H)-indol-3-yl") == 0

    def test_indicated_hydrogen_with_locant_ignored(self):
        """P-16.5.4.1.1: Indicated hydrogen (3H) or (9aH) also ignored."""
        assert compute_nesting_depth("(3H)-furan-2-yl") == 0
        assert compute_nesting_depth("(9aH)-carbazol-3-yl") == 0

    def test_fusion_brackets_ignored(self):
        """P-16.5.4.1.2: Fusion brackets like [2,3-b] do NOT count."""
        assert compute_nesting_depth("thieno[2,3-b]furan") == 0

    def test_spiro_brackets_ignored(self):
        """P-16.5.4.1.2: Spiro brackets like [4.3.0] do NOT count."""
        assert compute_nesting_depth("spiro[4.5]decane") == 0

    def test_von_baeyer_brackets_ignored(self):
        """P-16.5.4.1.2: Von Baeyer brackets like [2.2.1] do NOT count."""
        assert compute_nesting_depth("bicyclo[2.2.1]heptane") == 0

    def test_stereo_descriptors_counted(self):
        """P-16.5.4.1.3: Stereo descriptors (R), (S), (E), (Z) DO count."""
        assert compute_nesting_depth("(R)-butan-2-yl") >= 1
        assert compute_nesting_depth("(S)-1-phenylethyl") >= 1

    def test_compound_stereo_descriptors_counted(self):
        """P-16.5.4.1.3: Compound stereo like (1R,2S) also counts."""
        assert compute_nesting_depth("(1R,2S)-cyclohexyl") >= 1

    def test_ez_stereo_counted(self):
        """P-16.5.4.1.3: E/Z stereo descriptors count."""
        assert compute_nesting_depth("(E)-but-2-enyl") >= 1
        assert compute_nesting_depth("(Z)-but-2-enyl") >= 1

    def test_consecutive_same_level_escalation(self):
        """P-16.5.4.1.4: Consecutive same-level marks escalate.

        If a name starts with '(' and we'd add another '(', the effective
        depth should be >= 1 to escalate to '['.
        """
        # A name like "(2-methylpropyl)" already has parens; wrapping
        # in parens would give "((2-methylpropyl))" -- should escalate
        depth = compute_nesting_depth("(2-methylpropyl)")
        assert depth >= 1


# ===========================================================================
# ERRATA-07: apply_enclosing_marks tests with auto-detection
# ===========================================================================

class TestApplyEnclosingMarksAutoDetect:
    """Test that apply_enclosing_marks with default depth auto-detects nesting."""

    def test_basic_parentheses_explicit_zero(self):
        """Explicit depth=0 still works (backward compatible)."""
        assert apply_enclosing_marks("2-methylpropyl", 0) == "(2-methylpropyl)"

    def test_basic_brackets_explicit_one(self):
        """Explicit depth=1 still works (backward compatible)."""
        assert apply_enclosing_marks("2-methylpropyl", 1) == "[2-methylpropyl]"

    def test_basic_braces_explicit_two(self):
        """Explicit depth=2 still works (backward compatible)."""
        assert apply_enclosing_marks("2-methylpropyl", 2) == "{2-methylpropyl}"

    def test_auto_detect_plain_name(self):
        """Auto-detect (depth=-1) for plain name uses parentheses."""
        result = apply_enclosing_marks("2-methylpropyl", -1)
        assert result == "(2-methylpropyl)"

    def test_auto_detect_name_with_stereo(self):
        """Auto-detect escalates for name with stereo descriptor parentheses."""
        # Name contains (R) which counts as nesting -- should escalate
        result = apply_enclosing_marks("(R)-butan-2-yl", -1)
        assert result == "[(R)-butan-2-yl]"

    def test_auto_detect_name_with_indicated_h(self):
        """Auto-detect ignores indicated hydrogen -- stays at depth 0."""
        result = apply_enclosing_marks("(1H)-indol-3-yl", -1)
        assert result == "((1H)-indol-3-yl)"

    def test_auto_detect_name_with_fusion_brackets(self):
        """Auto-detect ignores fusion brackets -- stays at depth 0."""
        result = apply_enclosing_marks("thieno[2,3-b]furanyl", -1)
        assert result == "(thieno[2,3-b]furanyl)"


# ===========================================================================
# ERRATA-10: _build_substituent_string tests (P-16.5.1.3)
# ===========================================================================

class TestBuildSubstituentStringErrata10:
    """Test P-16.5.1.3 first-substituent rule for mononuclear parent hydrides."""

    def test_four_different_substituents(self):
        """4 different subs: first (butyl) no marks, rest get parentheses."""
        result = _build_substituent_string(
            ["butyl", "ethyl", "methyl", "propyl"]
        )
        assert result == "butyl(ethyl)(methyl)(propyl)"

    def test_single_unique_substituent(self):
        """Single unique substituent: standard multiplier, no marks."""
        result = _build_substituent_string(["methyl", "methyl", "methyl"])
        assert result == "trimethyl"

    def test_two_different_groups_ethyl_dimethyl(self):
        """ethyl first (no marks); the multiplicative prefix 'di' stays OUTSIDE
        the parentheses -> 'ethyldi(methyl)' (P-16.5.1.3.1, BB:7272: multiplying
        prefixes are not enclosed; verbatim PIN 'tert-butyldi(methyl)phosphane',
        BB 16286). Source corrected in 47a8624a; this expectation was stale."""
        result = _build_substituent_string(["ethyl", "methyl", "methyl"])
        assert result == "ethyldi(methyl)"

    def test_two_different_groups_ethyl_diphenyl(self):
        """As above (P-16.5.1.3.1): 'di' stays outside the marks ->
        'ethyldi(phenyl)'."""
        result = _build_substituent_string(["ethyl", "phenyl", "phenyl"])
        assert result == "ethyldi(phenyl)"

    def test_trimethylphosphane_unchanged(self):
        """Trimethylphosphane should be unchanged (single unique substituent)."""
        from orthonym.namer import name_compound
        result = name_compound("CP(C)C")
        assert result == "trimethylphosphane"

    def test_two_unique_substituents_diphenyl_methyl(self):
        """methyl first (no marks); 'di' stays OUTSIDE the parentheses ->
        'methyldi(phenyl)' (P-16.5.1.3.1, BB:7272). Stale expectation from the
        pre-errata assembler; source corrected in 47a8624a."""
        result = _build_substituent_string(["methyl", "phenyl", "phenyl"])
        assert result == "methyldi(phenyl)"
