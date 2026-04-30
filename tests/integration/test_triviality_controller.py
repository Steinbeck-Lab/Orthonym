"""Phase 150 V18 plan section 6 line 1031 + HERITAGE-1990 section 10 triviality controller tests.

Verify retained-name + multiplier transitions stay IUPAC-compliant after
Phase 150's retained-name expansion. Each test cites:
  - The QMUL P-section URL (P-14.2.1, P-14.2.2, P-14.5.1, P-14.5.2)
  - HERITAGE-1990 section 10 (Wisniewski 1990, J. Chem. Inf. Comput. Sci. 30, 324-332)
  - V18 plan section 6 line 1031 (triviality controller scope-extension mandate)
  - Phase 150 CONTEXT D-07 + SC-5

Per CONTEXT D-12 + RESEARCH section 7.2: tests VERIFY pre-existing composer.py /
prefixes.py behavior is correct after retained-name expansion. They do NOT
modify the assembly layer. Failures filed as Phase 154 IM (multiplicative.py
completion); xfail with link to Phase 154 IM is acceptable for known gaps
with documented attribution.

Source: https://iupac.qmul.ac.uk/BlueBook/P1.html#P-14.2
Source: https://iupac.qmul.ac.uk/BlueBook/P1.html#P-14.5
Source: HERITAGE-1990-insights.md section 10
Source: 150-CONTEXT.md D-07 + SC-5 + V18 plan section 6 line 1031
"""

import pytest

from orthonym.namer import name_compound


class TestRetainedNameMultiplierTransitions:
    """V18 section 6 line 1031 mandate: 5+ retained-name + multiplier transition tests.

    Each parametrized case verifies that when a retained name is substituted
    for a systematic-name fragment during assembly, the multiplier choice
    (di- for retained simple substituents per P-14.2.1, bis- for substituents
    with internal locants per P-14.2.2) and the alphabetization first-letter
    citation order (P-14.5.1 + P-14.5.2) stay IUPAC-compliant.

    Per CONTEXT D-12: tests verify pre-existing assembly behavior; failures
    filed as Phase 154 IM (multiplicative.py completion).
    """

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles, expected_substring, forbidden_substring, citation",
        [
            # T-1: bis(1-oxo-ethoxy) -> diacetoxy substitution + bis -> di multiplier
            # 1,3-bis(acetoxy)benzene = resorcinol diacetate; expect 'diacetoxy' or 'diacetate'
            # Phase 154 IM-150-T1: composer.py emits 'bis(acetyloxy)' not 'diacetoxy';
            # the substituent renderer for ester-O-acyl chains hasn't been wired through
            # the retained-substituent table. Multiplicative.py P-14.2.1 'simple
            # substituent' path keeps the systematic form.
            pytest.param(
                "CC(=O)Oc1cccc(OC(C)=O)c1",
                "diacet",
                "bis(1-oxo-ethoxy)",
                "P-14.2.1 + HERITAGE section 10 - acetoxy multiplier transition",
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
            # Phase 154 IM-150-T2: composer.py emits 'benzene-1,4-dicarboxamide'
            # via the carboxamide handler; the retained 'carbamoyl' substituent
            # form is not selected when both groups are PG-equivalent (P-66.6
            # carboxamide-as-suffix path wins over carbamoyl-as-prefix retained
            # substitution). Phase 154 multiplicative.py + carbamoyl-prefix path.
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
            # Phase 154 IM-150-T3: composer.py emits '2-thienylfuran' (the
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
            # citation order. Same Phase 154 IM-150-T1 root cause as T-1.
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
        """V18 section 6 line 1031 + HERITAGE-1990 section 10 - multiplier transition.

        Verifies pre-existing composer.py / prefixes.py behavior is IUPAC-compliant
        after retained-name expansion. Failures filed as Phase 154 IM per CONTEXT D-12.

        Source: HERITAGE-1990-insights.md section 10
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

        Per CONTEXT D-12 + RESEARCH section 7.2: pre-existing prefixes.py
        first-letter alphabetization MUST stay correct after retained-name
        expansion.

        Source: https://iupac.qmul.ac.uk/BlueBook/P1.html#P-14.5
        Source: 150-CONTEXT.md D-07 + V18 plan section 6 line 1031
        """
        smiles = "Brc1cc(OC(C)=O)cc(OC(C)=O)c1"
        name = name_compound(smiles)
        assert name is not None, f"name_compound returned None for {smiles!r}"
        lower = name.lower()
        # Phase 150 REVIEW WR-03 root-cause fix: previously this test
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
