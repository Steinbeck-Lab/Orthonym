"""v36 Milestone C — Pattern A2: primed (spiro/fused) stereo locant render + recognize.

These tests pin the SHARED stereo primitive
(`rules/stereochemistry.py::format_stereodescriptor_string` + `_STEREO_PREFIX_RE`
/ `_STEREO_EMBEDDED_RE`) for PRIMED tuple locants `(n, "'")` / `(n, "''")` that
arise from the 2nd component of a spiro/fused multi-component name.

Bug (finding V36-C1C2C6-TASK3-A2): `format_stereodescriptor_string` renders a
primed tuple locant with the Python tuple repr `(6, "'")R` instead of `6'R`, and
the recognizers don't accept a prime after the locant, so a correct
`(2S,1'R,...)-` prefix isn't seen as already-stereoed and gets double-injected.

⚠ SHARED primitive: scalar (unprimed) render + match MUST stay byte-identical —
the scalar-regression tests below guard that (they pass both RED and GREEN).
"""

from orthonym.rules.stereochemistry import (
    _STEREO_EMBEDDED_RE,
    _STEREO_PREFIX_RE,
    format_stereodescriptor_string,
)


class TestPrimedLocantRender:
    """format_stereodescriptor_string must render primed tuple locants as `n'`."""

    def test_single_primed_locant_renders_as_apostrophe(self):
        # (1, "'") + CIP R  ->  1'R   NOT   (1, "'")R
        out = format_stereodescriptor_string([((1, "'"), "R")])
        assert out == "(1'R)-", out
        assert "(1, " not in out  # no Python tuple repr leaked

    def test_double_primed_locant_renders_as_two_apostrophes(self):
        out = format_stereodescriptor_string([((2, "''"), "S")])
        assert out == "(2''S)-", out

    def test_rt_verified_target_prefix(self):
        # The finding's RT-verified spiro-VB target prefix:
        # (2S,1'R,3'R,11'R)-spiro[oxolane-2,12'-tricyclo[9.3.0.0^3,7]tetradeca-4,7-diene]
        out = format_stereodescriptor_string(
            [(2, "S"), ((1, "'"), "R"), ((3, "'"), "R"), ((11, "'"), "R")]
        )
        assert out == "(2S,1'R,3'R,11'R)-", out


class TestPrimedLocantRecognize:
    """The recognizer regexes must accept a prime after the locant number."""

    def test_prefix_re_matches_single_primed(self):
        assert _STEREO_PREFIX_RE.match("(1'R)-")

    def test_prefix_re_matches_double_primed(self):
        assert _STEREO_PREFIX_RE.match("(2''S)-")

    def test_prefix_re_matches_combined_primed_block(self):
        assert _STEREO_PREFIX_RE.match("(2S,1'R,3'R,11'R)-")

    def test_embedded_re_finds_primed_block(self):
        assert _STEREO_EMBEDDED_RE.search("spiro-(2''S)-thing")


class TestScalarByteIdenticalRegression:
    """SHARED primitive: unprimed (scalar) render + match must NOT change."""

    def test_scalar_render_unchanged(self):
        assert format_stereodescriptor_string([(2, "R")]) == "(2R)-"
        assert format_stereodescriptor_string([(2, "R"), (3, "S")]) == "(2R,3S)-"
        assert (
            format_stereodescriptor_string([(2, "E"), (3, "R"), (5, "Z")])
            == "(2E,3R,5Z)-"
        )
        assert format_stereodescriptor_string([(2, "r"), (3, "s")]) == "(2r,3s)-"
        assert format_stereodescriptor_string([]) == ""

    def test_composite_locant_render_unchanged(self):
        # composite ring-junction locant like '7a' stays byte-identical
        assert format_stereodescriptor_string([("7a", "S")]) == "(7aS)-"

    def test_scalar_prefix_and_embedded_match_unchanged(self):
        assert _STEREO_PREFIX_RE.match("(2R,3S)-")
        assert _STEREO_PREFIX_RE.match("(R)-")
        assert _STEREO_PREFIX_RE.match("(E)-")
        assert _STEREO_PREFIX_RE.match("(7aS)-")
        assert _STEREO_EMBEDDED_RE.search("foo-(3aR,7aS)-bar")
