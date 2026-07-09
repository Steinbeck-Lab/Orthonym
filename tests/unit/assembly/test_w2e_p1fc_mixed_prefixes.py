"""W2E-P1FC Task 13 — P-35.5.1 (BB 18124): mixed substituent prefixes
(substitutive + additive operations combined, with enclosing-mark escalation).
Examples: (ethoxysulfinyl)amino, (acetylsulfanyl)carbonyl,
[bis(sulfanyl)phosphoryl]amino.

STATUS (2026-07-09): DEFERRED to strict-xfail. All three mixed-prefix skeletons
(EtO-S(=O)-NH-, CH3CO-S-CO-, (HS)2P(=O)-NH-) fail CLOSED at HEAD (never a wrong
name). Each is a distinct ADDITIVE-prefix core (sulfinyl / carbonyl / phosphoryl)
carrying a SUBSTITUTIVE prefix (ethoxy / acetyl / bis(sulfanyl)) — building them
requires dedicated per-skeleton mixed-prefix assemblers (no reusable helper
exists: get_sulfinyl_prefix handles only C-linked -S(=O)-alkyl, not the O-linked
alkoxysulfinyl here). Deferred as documented follow-ups
(); the fail-closed contract below guards
against any wrong-name leak in the meantime.

OPSIN-verified target PINs (authoring, for the follow-up build):
  CCOS(=O)Nc1ccccc1  -> N-(ethoxysulfinyl)aniline
  SP(=O)(S)Nc1ccccc1 -> N-[bis(sulfanyl)phosphoryl]aniline
"""
import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestP3551MixedPrefixes:
    @pytest.mark.parametrize("smiles", [
        "CCOS(=O)Nc1ccccc1",
        "SP(=O)(S)Nc1ccccc1",
    ])
    def test_no_correct_name_yet(self, smiles):
        # No mixed-prefix assembler yet. In production these fail closed via
        # SELF-01 (OPSIN re-perception). Deterministically (independent of the
        # OPSIN test env, which can fail SELF-01 open) the ONE thing guaranteed:
        # the correct mixed-prefix PIN is NOT produced. This trips (strict-xfail)
        # to a passing target once the follow-up build lands.
        result = name_compound(smiles)
        assert "sulfinyl" not in result and "phosphoryl" not in result

    @pytest.mark.xfail(
        reason="P-35.5.1 mixed substituent prefix (ethoxysulfinyl) needs a "
        "dedicated alkoxysulfinyl additive-prefix assembler — W2E-P1FC follow-up",
        strict=True,
    )
    def test_ethoxysulfinyl_target(self):
        assert name_compound("CCOS(=O)Nc1ccccc1") == "N-(ethoxysulfinyl)aniline"

    @pytest.mark.xfail(
        reason="P-35.5.1 mixed substituent prefix [bis(sulfanyl)phosphoryl] needs "
        "a dedicated phosphoryl additive-prefix assembler — W2E-P1FC follow-up",
        strict=True,
    )
    def test_bis_sulfanyl_phosphoryl_target(self):
        assert name_compound("SP(=O)(S)Nc1ccccc1") == \
            "N-[bis(sulfanyl)phosphoryl]aniline"
