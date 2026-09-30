""" Milestone C — Pattern A2: primed (spiro/fused) stereo locant render + recognize.

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

import pytest

from orthonym.rules.stereochemistry import (
    _STEREO_EMBEDDED_RE,
    _STEREO_PREFIX_RE,
    format_stereodescriptor_string,
)
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "C1=C[C@H]2C[C@H]3CC[C@]4(CCCO4)[C@@H]3CCC=C2C1",
    "CC1=C[C@H]2O[C@@H]3[C@H](O)C[C@](C)([C@@]2(CO)[C@H](O)C1=O)[C@]31CO1",
    "COc1cc2c3c(c1OC)C1(C=CC(=O)C=C1)C[C@H]3N(C)CC2",
    "C[C@@H]1CC[C@@]2(OC1)O[C@H]1C[C@H]3[C@@H]4CC[C@H]5C[C@@H](O)CC[C@]5(C)[C@H]4CC[C@]3(C)[C@H]1[C@@H]2C",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)



class TestPrimedLocantRender:
    """format_stereodescriptor_string must render primed tuple locants as `n'`."""

    def test_single_primed_locant_renders_as_apostrophe(self):
        # (1, "'") + CIP R -> 1'R NOT (1, "'")R
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


# ---------------------------------------------------------------------------
# Task 3 — WITNESS RT: spiro-von-Baeyer stereo-completion (needs OPSIN)
# ---------------------------------------------------------------------------
# These name real spiro-VB stereo-omission witnesses end-to-end and check the
# emitted name round-trips to the input's FULL InChI (incl. the stereo layer)
# via PLAIN OPSIN — the independent 0-wrong gate. Names are NOT string-asserted
# (a core-namer change may re-spell them); the contract is RT-pass / abstain.

# Witnesses that must round-trip WITH full stereo after the A2 fix (measured):
_RT_PASS_WITNESSES = [
    # The finding's RT-verified canonical (2S/1'R/3'R/11'R spiro-oxolane-tricyclo)
    "C1=C[C@H]2C[C@H]3CC[C@]4(CCCO4)[C@@H]3CCC=C2C1",
    # azaspiro tricyclic
    "CC1O[C@@]2(CS1)CN1CCC2CC1",
    # spiro-azatricyclo dienone (single primed R/S)
    "COc1cc2c3c(c1OC)C1(C=CC(=O)C=C1)C[C@H]3N(C)CC2",
    # steroidal spiroketal (cholestane 16,22-epoxy-22,27-epoxy)
    "C[C@@H]1CC[C@@]2(OC1)O[C@H]1C[C@H]3[C@@H]4CC[C@H]5C[C@@H](O)CC[C@]5(C)"
    "[C@H]4CC[C@]3(C)[C@H]1[C@@H]2C",
    # spiro-oxiranyl oxatricyclo, multi primed + unprimed block
    "CC1=C[C@H]2O[C@@H]3[C@H](O)C[C@](C)([C@@]2(CO)[C@H](O)C1=O)[C@]31CO1",
]

# A witness whose CONSTITUTION (bare core) is OPSIN-unparseable — the stereo
# block injects fine, but must suppress on the core → abstain (0-wrong).
_ABSTAIN_WITNESSES = [
    "c1cc2c(c3c1CNC3)O[C@@]1(CCC[C@H]3CCCC[C@@H]31)C2",
]

_ALL_WITNESSES = _RT_PASS_WITNESSES + _ABSTAIN_WITNESSES
_FAIL_NAMES = {"", "unknown organic compound", "unknown"}


def _opsin_available() -> bool:
    try:
        from orthonym.validation.opsin_roundtrip import opsin_parse

        return opsin_parse("ethane") is not None
    except Exception:
        return False


@pytest.fixture(scope="module")
def named_results():
    """Name every witness ONCE (conserve JVM launches); return {smi: (name, rt)}."""
    if not _opsin_available():
        pytest.skip("OPSIN jar unavailable — cannot RT-verify")
    from orthonym import name_compound
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

    out = {}
    for smi in _ALL_WITNESSES:
        res = _dt_name_compound(smi)
        name = res.name if hasattr(res, "name") else res
        emitted = bool(name) and name not in _FAIL_NAMES
        rt = opsin_roundtrip_check(smi, name)["passed"] if emitted else False
        out[smi] = (name, emitted, rt)
    return out


class TestSpiroVBStereoCompletionWitnessRT:
    def test_finding_witness_rt_passes_with_full_stereo(self, named_results):
        name, emitted, rt = named_results[_RT_PASS_WITNESSES[0]]
        assert emitted, name
        assert rt, f"finding witness must RT-pass with full stereo; got {name!r}"
        # the primed 2nd-component descriptors are present (no tuple-repr garbage)
        assert "(1, " not in name and "'" in name, name

    @pytest.mark.parametrize("smi", _RT_PASS_WITNESSES)
    def test_rt_pass_witnesses_roundtrip_full_stereo(self, smi, named_results):
        name, emitted, rt = named_results[smi]
        assert emitted, f"expected a stereo-completed name, got abstain: {smi}"
        assert rt, f"expected full-InChI RT pass, got {name!r} for {smi}"

    @pytest.mark.parametrize("smi", _ABSTAIN_WITNESSES)
    def test_constitutional_defect_witnesses_abstain(self, smi, named_results):
        name, emitted, _rt = named_results[smi]
        # Constitution unparseable → suppresses → abstain (never wrong).
        assert not emitted, f"expected abstain (0-wrong), got emitted name {name!r}"

    def test_zero_wrong_invariant_all_witnesses(self, named_results):
        """0-wrong ABSOLUTE: every EMITTED witness name must RT to the full
        InChI (incl. stereo). No emitted-but-RT-fail is permitted."""
        for smi, (name, emitted, rt) in named_results.items():
            if emitted:
                assert rt, (
                    f"0-WRONG VIOLATION: emitted {name!r} does NOT RT for {smi}"
                )
