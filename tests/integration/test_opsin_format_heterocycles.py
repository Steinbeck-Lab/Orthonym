"""
Tests for OPSIN-compatible heterocycle naming.

Validates that Orthonym generates heterocycle names parseable by OPSIN CLI.
Specifically tests that 6-membered O unsaturated heterocycles produce "2H-pyran"
or "4H-pyran" instead of HW systematic "oxine" (which OPSIN cannot parse).

a phase Plan 03: OPSIN Format Fixes - Heterocycle Naming
"""

import os
import subprocess
import pytest
from orthonym import name_compound
from tests.support.jars import jar_or_none

# ---------------------------------------------------------------------------
# OPSIN CLI helper
# ---------------------------------------------------------------------------

OPSIN_JAR = jar_or_none()
OPSIN_AVAILABLE = OPSIN_JAR is not None


def opsin_parse(name: str) -> str:
    """Parse a name with OPSIN CLI and return SMILES or empty string on failure."""
    if not OPSIN_AVAILABLE:
        return ""
    try:
        result = subprocess.run(
            ["java", "-jar", OPSIN_JAR, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


# ---------------------------------------------------------------------------
# TestOxineToPyran: Verify oxine -> pyran replacement
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestOxineToPyran:
    """Test that 6-membered O unsaturated heterocycles produce pyran names, not oxine."""

    def test_2h_pyran_no_oxine(self):
        """2H-pyran should NOT contain 'oxine'."""
        name = name_compound("C1=CC=COC1")
        assert "oxine" not in name.lower(), f"Name contains 'oxine': {name}"

    def test_2h_pyran_correct_name(self):
        """2H-pyran canonical SMILES should produce '2H-pyran'."""
        name = name_compound("C1=CC=COC1")
        assert name == "2H-pyran", f"Expected '2H-pyran', got '{name}'"

    def test_4h_pyran_no_oxine(self):
        """4H-pyran should NOT contain 'oxine'."""
        name = name_compound("C1=COC=CC1")
        assert "oxine" not in name.lower(), f"Name contains 'oxine': {name}"

    def test_4h_pyran_correct_name(self):
        """4H-pyran canonical SMILES should produce '4H-pyran'."""
        name = name_compound("C1=COC=CC1")
        assert name == "4H-pyran", f"Expected '4H-pyran', got '{name}'"

    def test_34_dihydro_2h_pyran(self):
        """3,4-dihydro-2H-pyran should be named correctly."""
        name = name_compound("C1=COCCC1")
        assert "oxine" not in name.lower(), f"Name contains 'oxine': {name}"
        assert name == "3,4-dihydro-2H-pyran", f"Expected '3,4-dihydro-2H-pyran', got '{name}'"

    def test_36_dihydro_2h_pyran(self):
        """3,6-dihydro-2H-pyran should be named correctly."""
        name = name_compound("C1=CCOCC1")
        assert "oxine" not in name.lower(), f"Name contains 'oxine': {name}"
        assert name == "3,6-dihydro-2H-pyran", f"Expected '3,6-dihydro-2H-pyran', got '{name}'"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_2h_pyran_opsin_parses(self):
        """OPSIN CLI should parse '2H-pyran' successfully."""
        name = name_compound("C1=CC=COC1")
        opsin_smi = opsin_parse(name)
        assert opsin_smi, f"OPSIN failed to parse '{name}'"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_4h_pyran_opsin_parses(self):
        """OPSIN CLI should parse '4H-pyran' successfully."""
        name = name_compound("C1=COC=CC1")
        opsin_smi = opsin_parse(name)
        assert opsin_smi, f"OPSIN failed to parse '{name}'"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_dihydropyran_opsin_parses(self):
        """OPSIN CLI should parse dihydropyran names successfully."""
        for smi in ["C1=COCCC1", "C1=CCOCC1"]:
            name = name_compound(smi)
            opsin_smi = opsin_parse(name)
            assert opsin_smi, f"OPSIN failed to parse '{name}' (from {smi})"


# ---------------------------------------------------------------------------
# TestRetainedHeterocycleNames: Regression tests for existing heterocycles
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRetainedHeterocycleNames:
    """Verify that existing retained heterocycle names are not broken by pyran fix."""

    @pytest.mark.parametrize("smiles,expected", [
        # 6-membered aromatic heterocycles
        ("c1ccncc1", "pyridine"),
        ("c1cncnc1", "pyrimidine"),
        ("c1cnccn1", "pyrazine"),
        # 5-membered aromatic heterocycles
        ("c1ccoc1", "furan"),
        ("c1ccsc1", "thiophene"),
        # PIN per R1: "in preferred IUPAC names indicated hydrogen must always be
        # cited when present in the corresponding structure" the Blue Book; "1H-pyrrole
        # (PIN)":24645; OPSIN RT exact (TRIAGE.csv; re-checked in Task 7/8). The test id is
        # now 'c1cc[nH]c1-1H-pyrrole'.
        ("c1cc[nH]c1", "1H-pyrrole"),
        # Saturated heterocycles
        ("C1CCOCC1", "oxane"),
        ("C1CCNCC1", "piperidine"),
        ("C1CCOC1", "oxolane"),
        ("C1CCNC1", "pyrrolidine"),
        ("C1COCCN1", "morpholine"),
        ("C1CNCCN1", "piperazine"),
        # 3- and 4-membered heterocycles
        ("C1CO1", "oxirane"),
        ("C1COC1", "oxetane"),
        ("C1CN1", "aziridine"),
    ])
    def test_retained_name_unchanged(self, smiles, expected):
        """Retained heterocycle names should be unchanged by pyran fix."""
        name = name_compound(smiles)
        assert name == expected, f"Expected '{expected}' for {smiles}, got '{name}'"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    @pytest.mark.parametrize("smiles,expected", [
        ("c1ccncc1", "pyridine"),
        ("c1ccoc1", "furan"),
        ("c1ccsc1", "thiophene"),
        ("c1cc[nH]c1", "pyrrole"),
        ("c1cncnc1", "pyrimidine"),
        ("C1CCOCC1", "oxane"),
        ("C1CCNCC1", "piperidine"),
    ])
    def test_retained_names_opsin_compatible(self, smiles, expected):
        """Retained heterocycle names should be parseable by OPSIN."""
        name = name_compound(smiles)
        opsin_smi = opsin_parse(name)
        assert opsin_smi, f"OPSIN failed to parse '{name}' (expected '{expected}') for {smiles}"


# ---------------------------------------------------------------------------
# TestHWNamingStillWorks: Verify HW systematic naming is intact
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestHWNamingStillWorks:
    """Verify that HW systematic naming is not broken for non-pyran heterocycles."""

    def test_hw_build_oxine_still_correct(self):
        """The build_hw_name function should still produce 'oxine' for 6-member O unsaturated.

        This is the correct HW systematic name; the override happens at a higher level
        (retained name lookup in name_heterocycle).
        """
        from orthonym.rules.heterocycles import build_hw_name
        result = build_hw_name([(1, 'O')], 6, False, False)
        assert result == "oxine", f"HW build should produce 'oxine', got '{result}'"

    def test_hw_azine_still_correct(self):
        """HW naming for 6-member N unsaturated should still produce 'azine'."""
        from orthonym.rules.heterocycles import build_hw_name
        result = build_hw_name([(1, 'N')], 6, False, False)
        assert result == "azine", f"HW build should produce 'azine', got '{result}'"

    def test_hw_thiine_still_correct(self):
        """HW naming for 6-member S unsaturated should still produce 'thiine'."""
        from orthonym.rules.heterocycles import build_hw_name
        result = build_hw_name([(1, 'S')], 6, False, False)
        assert result == "thiine", f"HW build should produce 'thiine', got '{result}'"
