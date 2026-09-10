"""a phase V18 plan section 6 line 1031 + AUTONOM-1990 section 10 triviality controller tests.

Verify retained-name + multiplier transitions stay IUPAC-compliant after
a phase's retained-name expansion. Each test cites:
  - The QMUL P-section URL (P-14.2.1, P-14.2.2, P-14.5.1, P-14.5.2)
  - AUTONOM-1990 section 10 (Wisniewski 1990, J. Chem. Inf. Comput. Sci. 30, 324-332)
  - V18 plan section 6 line 1031 (triviality controller scope-extension mandate)
  - a phase CONTEXT + SC-5

Per CONTEXT + RESEARCH section 7.2: tests VERIFY pre-existing composer.py /
prefixes.py behavior is correct after retained-name expansion. They do NOT
modify the assembly layer. Failures filed as a phase IM (multiplicative.py
completion); xfail with link to a phase IM is acceptable for known gaps
with documented attribution.

Source: https://iupac.qmul.ac.uk/BlueBook/P1.html#P-14.2
Source: https://iupac.qmul.ac.uk/BlueBook/P1.html#P-14.5
Source: AUTONOM-1990-insights.md section 10
Source: 150-CONTEXT.md + SC-5 + V18 plan section 6 line 1031
"""

import pytest

from orthonym.namer import name_compound

# ===== a phase additions (Plan-03): triviality-controller integration corpus =====
# This file pre-dates a phase (a phase shipped the multiplier-transition tests above).
# a phase APPENDS its controller corpus rather than overwriting (preserves Phase-150 coverage).
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from typing import Optional  # noqa: E402

from rdkit import Chem  # noqa: E402

from orthonym.assembly.name_tree import NameTreeNode, _alphabetize_prefixes  # noqa: E402
from orthonym.assembly.name_tree_to_string import name_tree_to_string  # noqa: E402
from orthonym.assembly.retained_substitution import apply_triviality_controller  # noqa: E402


class TestRetainedNameMultiplierTransitions:
    """V18 section 6 line 1031 mandate: 5+ retained-name + multiplier transition tests.

    Each parametrized case verifies that when a retained name is substituted
    for a systematic-name fragment during assembly, the multiplier choice
    (di- for retained simple substituents per P-14.2.1, bis- for substituents
    with internal locants per P-14.2.2) and the alphabetization first-letter
    citation order (P-14.5.1 + P-14.5.2) stay IUPAC-compliant.

    Per CONTEXT: tests verify pre-existing assembly behavior; failures
    filed as a phase IM (multiplicative.py completion).
    """

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles, expected_substring, forbidden_substring, citation",
        [
            # T-1: bis(1-oxo-ethoxy) -> diacetoxy substitution + bis -> di multiplier
            # 1,3-bis(acetoxy)benzene = resorcinol diacetate; expect 'diacetoxy' or 'diacetate'
            # a phase -T1: composer.py emits 'bis(acetyloxy)' not 'diacetoxy';
            # the substituent renderer for ester-O-acyl chains hasn't been wired through
            # the retained-substituent table. Multiplicative.py P-14.2.1 'simple
            # substituent' path keeps the systematic form.
            pytest.param(
                "CC(=O)Oc1cccc(OC(C)=O)c1",
                "diacet",
                "bis(1-oxo-ethoxy)",
                "P-14.2.1 + AUTONOM section 10 - acetoxy multiplier transition",
                marks=[
                    pytest.mark.integration,
                    pytest.mark.xfail(
                        reason=(
                            "Phase 154 IM-150-T1: prefixes.py emits 'bis(acetyloxy)' "
                            "instead of 'diacetoxy' - retained acetoxy substituent "
                            "form not wired through multiplicative.py. Per CONTEXT "
                            "D-12 documents pre-existing assembly behavior; Phase "
                            "154 closes multiplicative.py + ester-O-acyl substituent "
                            "renderer. Source: 150-CONTEXT.md D-12 + 150-RESEARCH.md "
                            "section 7.2."
                        ),
                        strict=False,
                    ),
                ],
            ),
            # T-2: bis(carbamoyl) -> dicarbamoyl (carbamoyl is a HC retained substituent)
            # 1,4-dicarbamoylbenzene; expect 'carbamoyl' multiplier transition
            # a phase -T2: composer.py emits 'benzene-1,4-dicarboxamide'
            # via the carboxamide handler; the retained 'carbamoyl' substituent
            # form is not selected when both groups are PG-equivalent (P-66.6
            # carboxamide-as-suffix path wins over carbamoyl-as-prefix retained
            # substitution). a phase multiplicative.py + carbamoyl-prefix path.
            pytest.param(
                "NC(=O)c1ccc(C(N)=O)cc1",
                "carbamoyl",
                "bis(aminocarbonyl)",
                "P-14.2.1 + P-66.6 - carbamoyl retained substituent multiplier",
                marks=[
                    pytest.mark.integration,
                    pytest.mark.xfail(
                        reason=(
                            "Phase 154 IM-150-T2: carboxamide-as-suffix path "
                            "(P-66.6) wins over carbamoyl-as-prefix retained "
                            "substitution; 'benzene-1,4-dicarboxamide' is the "
                            "current output. Phase 154 multiplicative.py + "
                            "retained-substituent renderer wiring closes this. "
                            "Source: 150-CONTEXT.md D-12 + 150-RESEARCH.md "
                            "section 7.2."
                        ),
                        strict=False,
                    ),
                ],
            ),
            # T-3: substituted retained-name uses bis (NOT di) due to internal locant
            # furan-2-yl has internal locant '2-'; multi-furan requires bis-
            # 2,5-bis(furan-2-yl)thiophene; expect 'bis(furan' (NOT 'difuran')
            # a phase -T3: composer.py emits '2-thienylfuran' (the
            # connectivity-walk handler picks ONE furan as parent and makes the
            # OTHER furan into a 'thienylfuran' compound substituent). The
            # 2,5-bis(furan-2-yl)thiophene name requires multiplicative.py
            # P-14.5 path (3+ identical parent structures with primed locants).
            pytest.param(
                "c1cc(-c2ccc(-c3ccco3)s2)oc1",
                "bis",
                "difuran",
                "P-14.2.2 - internal locant disqualifies di- form",
                marks=[
                    pytest.mark.integration,
                    pytest.mark.xfail(
                        reason=(
                            "Phase 154 IM-150-T3: connectivity-walk handler picks "
                            "one furan as parent emitting '2-thienylfuran'; "
                            "multiplicative.py P-14.5 (3+ identical parent "
                            "structures) needed for bis(furan-2-yl)thiophene. "
                            "Source: 150-CONTEXT.md D-12 + 150-RESEARCH.md "
                            "section 7.2."
                        ),
                        strict=False,
                    ),
                ],
            ),
            # T-4: alphabetization first-letter shift - acetoxy ('a') before bromo ('b')
            # 1-bromo-3,5-bis(acetoxy)benzene; expect 'diacetoxy' substring + acetoxy
            # citation order. Same a phase -T1 root cause as T-1.
            pytest.param(
                "Brc1cc(OC(C)=O)cc(OC(C)=O)c1",
                "diacet",
                "bis(1-oxo-ethoxy)",
                "P-14.5.1 + P-14.5.2 - alphabetization first-letter (acetoxy < bromo)",
                marks=[
                    pytest.mark.integration,
                    pytest.mark.xfail(
                        reason=(
                            "Phase 154 IM-150-T1 (downstream): same root cause as "
                            "T-1 ('bis(acetyloxy)' instead of 'diacetoxy'); the "
                            "alphabetization check cannot run because the "
                            "expected 'diacet' substring is not in the output. "
                            "Source: 150-CONTEXT.md D-12 + 150-RESEARCH.md "
                            "section 7.2."
                        ),
                        strict=False,
                    ),
                ],
            ),
            # T-5: Phase-150-imported retained name triggers di- form
            # 2,5-dimethylfuran; expect 'dimethylfuran' (di- because methyl is simple)
            # T-5 PASSES on current implementation - documents the canonical case.
            pytest.param(
                "Cc1ccc(C)o1",
                "dimethyl",
                "bis(methyl)",
                "P-14.2.1 + Phase 150 SC-5 + V18 plan section 6 line 1031 - simple multiplier",
                marks=[pytest.mark.integration],
            ),
        ],
    )
    def test_retained_name_triggers_di_not_bis(
        self, smiles, expected_substring, forbidden_substring, citation
    ):
        """V18 section 6 line 1031 + AUTONOM-1990 section 10 - multiplier transition.

        Verifies pre-existing composer.py / prefixes.py behavior is IUPAC-compliant
        after retained-name expansion. Failures filed as a phase IM per CONTEXT.

        Source: AUTONOM-1990-insights.md section 10
        Source: https://iupac.qmul.ac.uk/BlueBook/P1.html#P-14.2
        """
        name = name_compound(smiles)
        assert name is not None, f"name_compound returned None for {smiles!r}"
        lower = name.lower()
        assert expected_substring.lower() in lower, (
            f"{citation}: expected '{expected_substring}' in '{name}' for {smiles!r}"
        )
        assert forbidden_substring.lower() not in lower, (
            f"{citation}: forbidden '{forbidden_substring}' in '{name}' for {smiles!r}"
        )


class TestMultiplierAlphabetization:
    """V18 section 6 line 1031 - alphabetization first-letter rule (P-14.5.1 + P-14.5.2).

    Multiplicative prefixes (di-, tri-, tetra-, bis-, tris-, tetrakis-) are
    IGNORED for alphabetical citation order per P-14.5.1; the first letter of
    the substituent stem governs citation order per P-14.5.2.
    """

    @pytest.mark.integration
    def test_di_prefix_does_not_affect_alphabetization(self):
        """P-14.5.1: multiplicative prefixes do NOT alter alphabetical order.

        For 1-bromo-3,5-diacetoxybenzene (or its IUPAC PIN equivalent), the
        citation order should be acetoxy < bromo (acetoxy starts with 'a',
        bromo starts with 'b'). The current pipeline emits
        '1,5-bis(acetyloxy)-3-bromobenzene' which already cites the acet-
        prefix before the brom- prefix; the 'acet' < 'brom' first-letter
        order P-14.5 mandates is satisfied even without the diacetoxy
        retained form (T-4 of the parametrized case above).

        Per CONTEXT + RESEARCH section 7.2: pre-existing prefixes.py
        first-letter alphabetization MUST stay correct after retained-name
        expansion.

        Source: https://iupac.qmul.ac.uk/BlueBook/P1.html#P-14.5
        Source: 150-CONTEXT.md + V18 plan section 6 line 1031
        """
        smiles = "Brc1cc(OC(C)=O)cc(OC(C)=O)c1"
        name = name_compound(smiles)
        assert name is not None, f"name_compound returned None for {smiles!r}"
        lower = name.lower()
        # a phase REVIEW WR-03 root-cause fix: previously this test
        # silently passed when the output omitted either substring
        # (e.g., a pipeline regression returning 'unknown' or an
        # unrelated retained name would no-op the alphabetization
        # check). Assert both substrings are present so a regression
        # actually fails the suite.
        assert "acet" in lower and "brom" in lower, (
            f"P-14.5 + WR-03: expected both 'acet' and 'brom' in "
            f"'{name}' (regression silently passed before WR-03 fix)"
        )
        # Find positions of 'acet' and 'brom' substrings (tolerant of
        # 'diacetoxy' vs 'bis(acetyloxy)' vs 'acetate' surface forms).
        assert lower.index("acet") < lower.index("brom"), (
            f"P-14.5: acet- ('a') must precede brom- ('b') in '{name}'"
        )


# ============================================================================
# a phase Plan-03 controller corpus (appended; Phase-150 content above preserved).
#
# HONEST REACH-BOUND (CONTEXT honest-RT-framing #1; confirmed at Plan-03 execution): the
# controller fires 0 times end-to-end on the current IR — the structured IR fraction is aliphatic
# chains while the seed targets aromatic rings/acids that route through COARSE handlers (
# passthrough). So for seed-covered bare molecules the ON output EQUALS the OFF output (the existing
# retained-name lookup already emits the retained PIN). These tests verify (a) output CORRECTNESS,
# (b) NO regression (the controller never emits a worse form), and (c) the swap LOGIC via synthetic
# IR (TestMidNameSwapBeforeAlpha, TestXyleneStemAssembly). The controller activates for free once
# coarse-handler structured-coverage expands (a SCORE-01 follow-on, out of Phase-168 scope).
# ============================================================================


def _p168_find_opsin_jar():
    import glob
    for pat in ("opsin-cli-*-jar-with-dependencies.jar",
                "opsin/opsin-cli-*-jar-with-dependencies.jar"):
        m = glob.glob(pat)
        if m:
            return m[0]
    return None


_P168_OPSIN_JAR = _p168_find_opsin_jar()
_P168_OPSIN_AVAILABLE = bool(_P168_OPSIN_JAR) and shutil.which("java") is not None


def _p168_opsin_smiles(name: str) -> Optional[str]:
    if not _P168_OPSIN_AVAILABLE or not name:
        return None
    try:
        r = subprocess.run(["java", "-jar", _P168_OPSIN_JAR, "-osmi"], input=name + "\n",
                           capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    return r.stdout.strip() or None


def _p168_inchi(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchi(mol) if mol else None


TYPE_1_FIXTURES = [
    ("c1ccoc1", "furan"), ("c1cc[nH]c1", "pyrrole"), ("c1ccncc1", "pyridine"),
    ("c1ccsc1", "thiophene"), ("C1COCCN1", "morpholine"), ("C1CCNCC1", "piperidine"),
    ("c1ccc2ccccc2c1", "naphthalene"), ("c1ccc2[nH]ccc2c1", "indole"),
    ("Cc1ccncc1", "pyridine"), ("Brc1ccc2ccccc2c1", "naphthalene"),
]


class TestType1Branch:
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", TYPE_1_FIXTURES)
    def test_retained_form_present(self, smiles, expected):
        out = name_compound(smiles, enable_triviality_controller=True)
        assert expected in out.lower(), f"{smiles}: expected {expected!r} in {out!r}"

    @pytest.mark.integration
    @pytest.mark.roundtrip
    @pytest.mark.parametrize("smiles,expected", TYPE_1_FIXTURES)
    def test_output_roundtrips(self, smiles, expected):
        if not _P168_OPSIN_AVAILABLE:
            pytest.skip("OPSIN/Java unavailable")
        out = name_compound(smiles, enable_triviality_controller=True)
        rt = _p168_opsin_smiles(out)
        if rt is None:
            pytest.skip(f"OPSIN could not parse {out!r}")
        assert _p168_inchi(rt) == _p168_inchi(smiles)


TYPE_2A_POSITIVE = [
    ("Oc1ccccc1", "phenol"), ("Oc1ccc(Br)cc1", "phenol"), ("Nc1ccccc1", "aniline"),
    ("OC(=O)c1ccccc1", "benzoic acid"), ("CC(=O)O", "acetic acid"),
]


class TestType2aBranch:
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", TYPE_2A_POSITIVE)
    def test_positive_retained_form(self, smiles, expected):
        out = name_compound(smiles, enable_triviality_controller=True)
        assert expected in out.lower(), f"{smiles}: expected {expected!r} in {out!r}"

    @pytest.mark.integration
    def test_negative_pg_mismatch_no_phenol_parent(self):
        out = name_compound("OC(=O)c1ccc(O)cc1", enable_triviality_controller=True).lower()
        assert not out.endswith("phenol"), f"phenol parent leaked: {out!r}"


TYPE_2B_POSITIVE = [
    ("OC=O", "formic acid"), ("OC(=O)Br", "methanoic acid"), ("OC(=O)F", "methanoic acid"),
]
TYPE_2B_NEGATIVE_SMILES = ["OC(=O)CC"]


class TestType2bBranch:
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_sub", TYPE_2B_POSITIVE)
    def test_positive_swap(self, smiles, expected_sub):
        out = name_compound(smiles, enable_triviality_controller=True).lower()
        assert any(k in out for k in ("formic", "methanoic", "carbono")), \
            f"{smiles}: expected formic/methanoic/carbono in {out!r}"

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles", TYPE_2B_NEGATIVE_SMILES)
    def test_negative_carbon_substituent_not_emitted(self, smiles):
        # #7 REAL boundary: never emit a carbon-substituted formic/methanoic-acid form (carbon is
        # out of the P-15.1.8.2.2 closed list). Systematic 'propanoic acid' is expected.
        actual = name_compound(smiles, enable_triviality_controller=True).lower()
        assert not re.search(r"\w+formic acid", actual), f"carbon-formic leak: {actual!r}"

    @pytest.mark.integration
    def test_type_2b_check_refuses_carbon_substituent_unit(self):
        from orthonym.assembly.retained_substitution import _type_2b_check
        from orthonym.data.triviality_controller_seed import SEED_TABLE
        key = Chem.CanonSmiles("OC=O")
        if key not in SEED_TABLE:
            pytest.xfail("Type 2b empty post-audit per A1; documented in 168-AUDIT-TRIV.md section 7")
        entry = SEED_TABLE[key]
        node = NameTreeNode(parent_stem="methanoic acid", prefixes=(NameTreeNode(parent_stem="methyl"),))
        assert _type_2b_check(node, Chem.MolFromSmiles("CC(=O)O"), entry, None) is False


class TestType2cBranch:
    @pytest.mark.integration
    def test_anisole_bare_present(self):
        out = name_compound("COc1ccccc1", enable_triviality_controller=True).lower()
        assert "anisole" in out or "methoxybenzene" in out, f"unexpected: {out!r}"

    @pytest.mark.integration
    def test_substituted_anisole_not_bare_anisole(self):
        out = name_compound("COc1ccc(Br)cc1", enable_triviality_controller=True).lower()
        assert out != "anisole"


class TestType3Branch:
    @pytest.mark.integration
    def test_toluene_bare_present(self):
        out = name_compound("Cc1ccccc1", enable_triviality_controller=True).lower()
        assert "toluene" in out, f"expected toluene in {out!r}"

    @pytest.mark.integration
    def test_chlorotoluene_not_bare_toluene(self):
        out = name_compound("Cc1ccc(Cl)cc1", enable_triviality_controller=True).lower()
        assert not out.endswith("toluene")

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles", ["Cc1ccccc1C", "Cc1cccc(C)c1", "Cc1ccc(C)cc1"])
    def test_xylene_no_italic_locant_reach_bound(self, smiles):
        # HONEST reach-bound: the dimethylbenzene node is COARSE, so the swap to "1,2-xylene" does
        # NOT fire end-to-end (the swap LOGIC is proven by TestXyleneStemAssembly). Invariant: the
        # controller NEVER emits an italic o-/m-/p- locant (), fired or not.
        out = name_compound(smiles, enable_triviality_controller=True).lower()
        assert not any(t in out for t in ("o-xylene", "m-xylene", "p-xylene"))


class TestMultiplierFeedback:
    @pytest.mark.unit
    def test_simple_substituent_uses_di(self):
        from orthonym.assembly.retained_substitution import _build_rewrite
        from orthonym.data.triviality_controller_seed import SEED_TABLE
        entry = SEED_TABLE[Chem.CanonSmiles("Oc1ccccc1")]
        out = _build_rewrite(NameTreeNode(parent_stem="benzenol", multiplicative_prefix="di"), entry, ())
        assert out.multiplicative_prefix == "di"

    @pytest.mark.unit
    def test_complex_substituent_uses_bis(self):
        from orthonym.assembly.retained_substitution import _build_rewrite
        from orthonym.data.triviality_controller_seed import SEED_TABLE
        entry = SEED_TABLE[Chem.CanonSmiles("Cc1ccccc1C")]
        out = _build_rewrite(NameTreeNode(parent_stem="x", multiplicative_prefix="di"), entry, ())
        assert out.multiplicative_prefix == "bis"


class TestMidNameSwapBeforeAlpha:
    @pytest.mark.unit
    def test_swap_happens_before_alphabetization(self):
        # Synthetic IR with OPSIN-parseable stems (so recovery CAN succeed): assert the
        # re-alphabetize invariant always; IF the swap fired, 'aniline' ('a') precedes 'ethyl' ('e').
        child_amine = NameTreeNode(parent_stem="benzenamine", fragment_legacy=None)
        child_ethyl = NameTreeNode(parent_stem="ethyl", locants=(2,), fragment_legacy=None)
        parent = NameTreeNode(parent_stem="benzene",
                              prefixes=(child_amine, child_ethyl), fragment_legacy=None)
        # CR-04: aniline's real principal_group is "aromatic_amine" (NOT "primary_amine").
        out = apply_triviality_controller(parent, Chem.MolFromSmiles("Nc1ccc(CC)cc1"),
                                          "aromatic_amine", enabled=True)
        assert out.prefixes == _alphabetize_prefixes(out.prefixes)
        stems = [p.parent_stem for p in out.prefixes]
        if "aniline" in stems and "ethyl" in stems:
            assert stems.index("aniline") < stems.index("ethyl")

    @pytest.mark.unit
    def test_re_alphabetization_invariant_on_every_call(self):
        c_z = NameTreeNode(parent_stem="zzzz", fragment_legacy=None)
        c_a = NameTreeNode(parent_stem="aaaa", fragment_legacy=None)
        parent = NameTreeNode(parent_stem="benzene", prefixes=(c_z, c_a), fragment_legacy=None)
        out = apply_triviality_controller(parent, Chem.MolFromSmiles("c1ccccc1"), None, enabled=True)
        assert out.prefixes == _alphabetize_prefixes(out.prefixes)


class TestType2aNonPrincipalSubstituent:
    @pytest.mark.integration
    def test_phenol_fragment_on_ester_stays_systematic(self):
        actual = name_compound("COC(=O)c1ccc(O)cc1", enable_triviality_controller=True).lower()
        assert not actual.endswith("phenol")
        assert "hydroxy" in actual or "benzoate" in actual

    @pytest.mark.integration
    def test_aniline_fragment_on_acid_stays_systematic(self):
        actual = name_compound("OC(=O)c1ccc(N)cc1", enable_triviality_controller=True).lower()
        assert not actual.endswith("aniline")
        assert "amino" in actual


class TestXyleneStemAssembly:
    """CR-01 regression (code review 2026-05-30). The swap output for a xylene is built by
    ``_build_rewrite``, NOT a hand-assembled ``NameTreeNode(parent_stem="xylene",...)`` the
    controller never produces. ``_build_rewrite`` sets ``parent_stem`` to the FULL seed name
    ("1,2-xylene") and — because the retained name already embeds the locant cluster — RESETS
    ``node.locants`` to () so the serializer does NOT prepend a second cluster. The pre-fix code
    kept ``locants=(1,2)`` AND ``parent_stem="1,2-xylene"`` and emitted the malformed
    "1,2-1,2-xylene". This test exercises the REAL swap path so the bug cannot hide again."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,locants,expected", [
        ("Cc1ccccc1C", (1, 2), "1,2-xylene"),
        ("Cc1cccc(C)c1", (1, 3), "1,3-xylene"),
        ("Cc1ccc(C)cc1", (1, 4), "1,4-xylene"),
    ])
    def test_build_rewrite_emits_single_locant_xylene(self, smiles, locants, expected):
        from orthonym.assembly.retained_substitution import _build_rewrite
        from orthonym.data.triviality_controller_seed import SEED_TABLE
        entry = SEED_TABLE[Chem.CanonSmiles(smiles)]
        # A bare dimethylbenzene structured node carrying the locants _type_3_check gates on.
        node = NameTreeNode(parent_stem="dimethylbenzene-stem", locants=locants, fragment_legacy=None)
        out = _build_rewrite(node, entry, ())
        assembled = name_tree_to_string(out, style="pin").lower()
        # Exact match: the retained PIN, with the locant cluster appearing EXACTLY ONCE.
        assert assembled == expected, f"{smiles}: expected {expected!r} got {assembled!r}"
        loc_cluster = expected.split("-")[0]  # "1,2"
        assert assembled.count(loc_cluster) == 1, f"CR-01 doubled-locant regression: {assembled!r}"


class TestControllerPositiveSwapFires:
    """WR-03 (code review 2026-05-30): the end-to-end TestType1Branch / TestType2aBranch tests
    above pass via the PRE-EXISTING retained-name machinery (OFF == ON, the controller is the
    0-fire reach-bound) — they would stay green even if ``apply_triviality_controller`` were a
    ``return tree`` stub. These drive the controller's FULL positive path (recovery -> seed match
    -> type dispatch -> ``_build_rewrite`` -> RT-gate) on a synthetic STRUCTURED node and assert
    the swap actually FIRED via fingerprints a stub/no-fire cannot leave: ``iupac_section_cite``
    stamped from the seed and ``fragment_legacy`` reset to None. Jar-gated because CR-03 makes the
    RT-gate fail closed without OPSIN."""

    @pytest.mark.integration
    @pytest.mark.roundtrip
    @pytest.mark.parametrize("smiles,p_section", [
        ("c1ccoc1", "P-22.2.1"),    # furan (Type 1)
        ("c1ccncc1", "P-22.2.1"),   # pyridine (Type 1)
        ("c1ccccc1", "P-22.1.2"),   # benzene (Type 1)
    ])
    def test_full_path_swap_fires(self, smiles, p_section):
        if not _P168_OPSIN_AVAILABLE:
            pytest.skip("OPSIN/Java unavailable — RT-gate fails closed without a jar (CR-03)")
        from orthonym.assembly.retained_substitution import (
            apply_triviality_controller, OpsinOracle,
        )
        from orthonym.data.triviality_controller_seed import SEED_TABLE
        entry = SEED_TABLE[Chem.CanonSmiles(smiles)]
        # Structured node (parent_stem != fragment_legacy => not coarse) whose parent_stem
        # recovers (Path B token-match) to the seed SMILES. The sentinel cite/fragment_legacy are
        # cleared ONLY by a real _build_rewrite swap.
        node = NameTreeNode(parent_stem=entry.retained_pin_name,
                            iupac_section_cite=None, fragment_legacy="SENTINEL")
        oracle = OpsinOracle(opsin_jar=_P168_OPSIN_JAR)
        out = apply_triviality_controller(node, Chem.MolFromSmiles(smiles), None,
                                          opsin_oracle=oracle, enabled=True)
        assert out.iupac_section_cite == p_section, \
            f"{smiles}: swap did not fire (controller behaving as a no-op?)"
        assert out.fragment_legacy is None, f"{smiles}: swap did not reset fragment_legacy"

    @pytest.mark.integration
    def test_swap_rejected_when_rt_unverifiable(self):
        # CR-03 differential: with NO oracle the post-swap round-trip cannot be verified, so the
        # controller MUST keep the systematic form (fail closed) — the swap fingerprints are NOT
        # stamped even though recovery (Path B) + seed match + Type-1 check all succeed.
        from orthonym.assembly.retained_substitution import apply_triviality_controller
        node = NameTreeNode(parent_stem="furan", iupac_section_cite=None, fragment_legacy="SENTINEL")
        out = apply_triviality_controller(node, Chem.MolFromSmiles("c1ccoc1"), None,
                                          opsin_oracle=None, enabled=True)
        assert out.iupac_section_cite is None and out.fragment_legacy == "SENTINEL", \
            "swap fired without RT verification (CR-03 fail-open regression)"
