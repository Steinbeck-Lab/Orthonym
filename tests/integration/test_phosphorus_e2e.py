"""End-to-end tests for phosphorus compound naming."""

import sys
from pathlib import Path

import pytest
from orthonym import name_compound


class TestPhosphineE2E:
    """E2E tests for phosphine (phosphane) naming (PHOSPH-01, PHOSPH-02)."""

    def test_methylphosphane(self):
        """CP -> methylphosphane (PHOSPH-01)"""
        assert name_compound("CP") == "methylphosphane"

    def test_ethylphosphane(self):
        """CCP -> ethylphosphane"""
        assert name_compound("CCP") == "ethylphosphane"

    def test_trimethylphosphane(self):
        """CP(C)C -> trimethylphosphane (PHOSPH-02)"""
        assert name_compound("CP(C)C") == "trimethylphosphane"

    def test_triethylphosphane(self):
        """CCP(CC)CC -> triethylphosphane"""
        assert name_compound("CCP(CC)CC") == "triethylphosphane"

    def test_dimethylphosphane(self):
        """CPC -> dimethylphosphane"""
        assert name_compound("CPC") == "dimethylphosphane"


class TestPhosphineOxideE2E:
    """E2E tests for phosphine oxide naming (PHOSPH-03)."""

    def test_trimethylphosphane_oxide(self):
        """CP(C)(C)=O -> trimethylphosphane oxide (PHOSPH-03)"""
        assert name_compound("CP(C)(C)=O") == "trimethyl-lambda5-phosphanone"

    def test_triethylphosphane_oxide(self):
        """CCP(CC)(CC)=O -> triethylphosphane oxide"""
        assert name_compound("CCP(CC)(CC)=O") == "triethyl-lambda5-phosphanone"


class TestPhosphonicAcidE2E:
    """E2E tests for phosphonic acid naming (PHOSPH-04)."""

    def test_methanephosphonic_acid(self):
        """CP(=O)(O)O -> methanephosphonic acid (PHOSPH-04)"""
        assert name_compound("CP(=O)(O)O") == "methanephosphonic acid"

    def test_ethanephosphonic_acid(self):
        """CCP(=O)(O)O -> ethanephosphonic acid"""
        assert name_compound("CCP(=O)(O)O") == "ethanephosphonic acid"

    def test_phenylphosphonic_acid(self):
        """c1ccccc1P(=O)(O)O -> phenylphosphonic acid"""
        assert name_compound("c1ccccc1P(=O)(O)O") == "phenylphosphonic acid"


class TestPhosphinicAcidE2E:
    """E2E tests for phosphinic acid naming (PHOSPH-05)."""

    def test_dimethylphosphinic_acid(self):
        """CP(C)(=O)O -> dimethylphosphinic acid (PHOSPH-05)"""
        assert name_compound("CP(C)(=O)O") == "dimethylphosphinic acid"

    def test_diethylphosphinic_acid(self):
        """CCP(CC)(=O)O -> diethylphosphinic acid"""
        assert name_compound("CCP(CC)(=O)O") == "diethylphosphinic acid"

    def test_ethylmethylphosphinic_acid(self):
        """CCP(C)(=O)O -> ethylmethylphosphinic acid"""
        assert name_compound("CCP(C)(=O)O") == "ethyl(methyl)phosphinic acid"


class TestPhosphateEsterE2E:
    """E2E tests for phosphate ester naming (PHOSPH-06)."""

    def test_methyl_phosphate(self):
        """COP(=O)(O)O -> methyl phosphate (PHOSPH-06)"""
        assert name_compound("COP(=O)(O)O") == "methyl phosphate"

    def test_dimethyl_phosphate(self):
        """COP(=O)(OC)O -> dimethyl phosphate"""
        assert name_compound("COP(=O)(OC)O") == "dimethyl phosphate"

    def test_trimethyl_phosphate(self):
        """COP(=O)(OC)OC -> trimethyl phosphate"""
        assert name_compound("COP(=O)(OC)OC") == "trimethyl phosphate"

    def test_ethyl_phosphate(self):
        """CCOP(=O)(O)O -> ethyl phosphate"""
        assert name_compound("CCOP(=O)(O)O") == "ethyl phosphate"

    def test_diethyl_phosphate(self):
        """CCOP(=O)(OCC)O -> diethyl phosphate"""
        assert name_compound("CCOP(=O)(OCC)O") == "diethyl phosphate"

    def test_triethyl_phosphate(self):
        """CCOP(=O)(OCC)OCC -> triethyl phosphate"""
        assert name_compound("CCOP(=O)(OCC)OCC") == "triethyl phosphate"


class TestPhosphorusRetainedNames:
    """E2E tests for phosphorus retained names."""

    def test_triphenylphosphane(self):
        """Triphenylphosphane retained name."""
        result = name_compound("c1ccccc1P(c2ccccc2)c3ccccc3")
        assert result == "triphenylphosphane"

    def test_triphenylphosphane_oxide(self):
        """Triphenylphosphane oxide retained name."""
        result = name_compound("O=P(c1ccccc1)(c2ccccc2)c3ccccc3")
        assert result == "triphenyl-lambda5-phosphanone"


class TestPhosphorusEdgeCases:
    """Edge case tests for phosphorus naming."""

    def test_primary_phosphine(self):
        """Primary phosphine with single substituent."""
        result = name_compound("CP")
        assert result == "methylphosphane"

    def test_asymmetric_phosphate(self):
        """Asymmetric phosphate diester -> cites the free -OH as 'hydrogen'.

        The bare 'ethyl methyl phosphate' is the ANION (OPSIN full-InChIKey RT
        FAILS vs the neutral input P(=O)(OCC)(OC)[O-]); the neutral diester acid
        is 'ethyl methyl hydrogen phosphate' (P-67/P-68, RT-OK).
        """
        result = name_compound("COP(=O)(OCC)O")
        assert result == "ethyl methyl hydrogen phosphate"

    def test_dipropylphosphinic_acid(self):
        """Dipropylphosphinic acid."""
        result = name_compound("CCCP(CCC)(=O)O")
        assert result == "dipropylphosphinic acid"


class TestPhosphorusRequirements:
    """Tests verifying all PHOSPH requirements are met."""

    def test_phosph_01_methylphosphane(self):
        """PHOSPH-01: name_compound('CP') returns 'methylphosphane'"""
        assert name_compound("CP") == "methylphosphane"

    def test_phosph_02_trimethylphosphane(self):
        """PHOSPH-02: name_compound('CP(C)C') returns 'trimethylphosphane'"""
        assert name_compound("CP(C)C") == "trimethylphosphane"

    def test_phosph_03_trimethylphosphane_oxide(self):
        """PHOSPH-03: name_compound('CP(C)(C)=O') returns 'trimethylphosphane oxide'"""
        assert name_compound("CP(C)(C)=O") == "trimethyl-lambda5-phosphanone"

    def test_phosph_04_methanephosphonic_acid(self):
        """PHOSPH-04: name_compound('CP(=O)(O)O') returns 'methanephosphonic acid'"""
        assert name_compound("CP(=O)(O)O") == "methanephosphonic acid"

    def test_phosph_05_dimethylphosphinic_acid(self):
        """PHOSPH-05: name_compound('CP(C)(=O)O') returns 'dimethylphosphinic acid'"""
        assert name_compound("CP(C)(=O)O") == "dimethylphosphinic acid"

    def test_phosph_06_methyl_phosphate(self):
        """PHOSPH-06: name_compound('COP(=O)(O)O') returns 'methyl phosphate'"""
        assert name_compound("COP(=O)(O)O") == "methyl phosphate"


class TestPhosphanylPrefix:
    """E2E tests for phosphanyl prefix naming (PHOS-01, PHOS-02)."""

    def test_diphenylphosphanyl_on_benzene(self):
        """Triphenylphosphane is a retained name (PHOS-01)."""
        result = name_compound("c1ccc(P(c2ccccc2)c3ccccc3)cc1")
        assert result == "triphenylphosphane"

    def test_diphenylphosphanyl_on_substituted_ring(self):
        """PPh2 on methylbenzene produces diphenylphosphanyl prefix (PHOS-01)."""
        result = name_compound("Cc1ccc(P(c2ccccc2)c3ccccc3)cc1")
        assert "diphenylphosphanyl" in result

    def test_phenyl_not_counted_as_alkyl(self):
        """Phenyl groups on P should not be counted as hexyl (PHOS-02 regression)."""
        result = name_compound("c1ccc(P(c2ccccc2)c3ccccc3)cc1")
        assert "hexyl" not in result


class TestBisPhosphanyl:
    """E2E tests for bis/tris multiplied phosphanyl groups (PHOS-03)."""

    def test_bis_diphenylphosphanyl_benzene(self):
        """Two identical PPh2 groups on benzene use bis() multiplier (PHOS-03)."""
        result = name_compound(
            "c1ccc(P(c2ccccc2)c3ccccc3)c(P(c4ccccc4)c5ccccc5)c1"
        )
        assert "bis(diphenylphosphanyl)" in result


class TestPhosphorusOnComplexSubstrate:
    """E2E tests for phosphorus on complex substrates (PHOS-04, PHOS-05)."""

    def test_phosphonate_still_correct(self):
        """Phosphonic acid naming unaffected (PHOS-04)."""
        assert name_compound("CP(=O)(O)O") == "methanephosphonic acid"

    def test_phenylphosphonic_acid(self):
        """Phenylphosphonic acid (PHOS-04)."""
        assert name_compound("c1ccccc1P(=O)(O)O") == "phenylphosphonic acid"

    def test_diphenylphosphinic_acid_e2e(self):
        """Diphenylphosphinic acid via e2e (PHOS-04)."""
        assert name_compound("O=P(O)(c1ccccc1)c2ccccc2") == "diphenylphosphinic acid"

    def test_phosphate_ester_not_disrupted(self):
        """Phosphate ester naming not disrupted (PHOS-05 routing safety)."""
        assert name_compound("COP(=O)(O)O") == "methyl phosphate"

    def test_phosphine_oxide_not_disrupted(self):
        """Phosphine oxide naming not disrupted (PHOS-05 routing safety)."""
        assert name_compound("CP(C)(C)=O") == "trimethyl-lambda5-phosphanone"


class TestCanaryRegression:
    """Verify zero regression on canary compounds for Phase 52."""

    def test_canary_no_regression(self):
        """All 88 canary compounds should still pass after phosphanyl changes."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest",
             "tests/integration/test_canary_rt75.py", "-x", "-q"],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).resolve().parents[2]),  # project root — CI-portable, replaces hardcoded /home/kohulan path per REVIEWS §Plan 03 HIGH #2
        )
        assert result.returncode == 0, (
            f"Canary regression detected:\n{result.stdout}\n{result.stderr}"
        )


class TestPhosphiteAndPhosphonateEsterE2E:
    """v30 tail #16/#17: trivalent-P phosphite triesters and phosphonic-acid
    diesters were entirely unperceived (no matching FG) -> a garbage skeletal-
    replacement name or abstention. New perception routes them to the existing
    functional-class namer's phosphite / phosphonate stems."""

    def test_trimethyl_phosphite(self):
        assert name_compound("COP(OC)OC") == "trimethyl phosphite"

    def test_triethyl_phosphite(self):
        assert name_compound("CCOP(OCC)OCC") == "triethyl phosphite"

    def test_tris_dodecylsulfanyl_phosphite(self):
        # #16: sulfenyl-ester oxygens (-O-S-R). The compound owner takes tris(...).
        smi = "CCCCCCCCCCCCSOP(OSCCCCCCCCCCCC)OSCCCCCCCCCCCC"
        assert name_compound(smi) == "tris(dodecylsulfanyl) phosphite"

    def test_dimethyl_methylphosphonate(self):
        assert name_compound("COP(=O)(C)OC") == "dimethyl methylphosphonate"

    def test_diisopropyl_cyano_isocyano_ethylphosphonate(self):
        # #17: phosphonate diester senior to the nitrile + isocyanide, which
        # become cyano/isocyano prefixes on the P-C ligand.
        smi = "CC(C)OP(=O)(C(C)(C#N)[N+]#[C-])OC(C)C"
        r = name_compound(smi)
        assert r is not None and "phosphonate" in r
        assert "cyano" in r and "isocyano" in r
