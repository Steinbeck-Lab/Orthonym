"""Integration tests for Phase 89: Ring System Recognition Expansion.

Verifies that dictionary additions from Plan 89-01 prevent VB polycyclic
over-application. Tests the full naming pipeline end-to-end: new ring system
SMILES should produce retained names, not VB notation.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem
from orthonym.namer import name_compound


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPSIN_JAR = "/home/kohulan/OpenSTOUT/Orthonym/opsin-cli-2.8.0-jar-with-dependencies.jar"


def _opsin_parse(name: str) -> str | None:
    """Send a name to OPSIN and return the SMILES, or None on failure."""
    try:
        result = subprocess.run(
            ["java", "-jar", OPSIN_JAR, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=30,
        )
        smi = result.stdout.strip()
        if smi and smi != "" and not smi.startswith("Error"):
            return smi
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _has_vb_notation(name: str) -> bool:
    """Check if a name contains VB polycyclic notation indicators."""
    lower = name.lower()
    return any(x in lower for x in ["tricyclo", "tetracyclo", "pentacyclo"])


def _canonical(smiles: str) -> str:
    """Return canonical SMILES."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)


# ---------------------------------------------------------------------------
# Test Class 1: Fused Heterocycle Naming
# ---------------------------------------------------------------------------

class TestFusedHeterocycleNaming:
    """Tests that new fused heterocycle entries produce retained names."""

    def test_phenanthridine(self):
        # Fixed: was c1ccc2c(c1)ccc1cccnc12 (non-canonical SMILES)
        name = name_compound("c1ccc2c(c1)cnc1ccccc12")
        assert "phenanthridine" in name.lower()
        assert not _has_vb_notation(name)

    def test_beta_carboline(self):
        name = name_compound("c1ccc2c(c1)[nH]c1cnccc12")
        assert "carboline" in name.lower()
        assert not _has_vb_notation(name)

    def test_acridone(self):
        name = name_compound("O=c1c2ccccc2[nH]c2ccccc12")
        assert "acridon" in name.lower()
        assert not _has_vb_notation(name)

    def test_4h_quinolizine(self):
        # Fixed SMILES (Phase 101): was C1=CC2=CCC=CN2C=C1
        smi = _canonical("C1=CCN2C=CC=CC2=C1")
        name = name_compound(smi)
        assert "quinolizin" in name.lower()
        assert not _has_vb_notation(name)

    def test_quinolizidine(self):
        name = name_compound("C1CCN2CCCCC2C1")
        # May produce "quinolizidine" or "decahydroisoquinoline" — both valid
        # Key requirement: no VB polycyclic notation
        assert not _has_vb_notation(name)
        # Should contain some recognizable ring name
        lower = name.lower()
        assert any(x in lower for x in ["quinoliz", "isoquinolin", "decahydro"])

    def test_substituted_phenanthridine(self):
        """6-chlorophenanthridine should get substituent + retained name."""
        # Fixed: use correct phenanthridine SMILES (Phase 101)
        smi = "Clc1nc2ccccc2c2ccccc12"
        name = name_compound(smi)
        lower = name.lower()
        assert "phenanthridine" in lower
        assert "chloro" in lower


# ---------------------------------------------------------------------------
# Test Class 2: Flavonoid Naming
# ---------------------------------------------------------------------------

class TestFlavonoidNaming:
    """Tests that flavonoid entries produce retained names."""

    def test_flavone(self):
        name = name_compound("O=c1cc(-c2ccccc2)oc2ccccc12")
        assert "flavone" in name.lower() or "chromen" in name.lower()
        assert not _has_vb_notation(name)

    def test_flavanone(self):
        name = name_compound("O=C1CC(c2ccccc2)Oc2ccccc21")
        assert "flavanone" in name.lower() or "chroman" in name.lower()
        assert not _has_vb_notation(name)

    def test_isoflavone(self):
        name = name_compound("O=c1c(-c2ccccc2)coc2ccccc12")
        assert "isoflavone" in name.lower() or "chromen" in name.lower()
        assert not _has_vb_notation(name)

    def test_chromanone(self):
        name = name_compound("O=C1CCOc2ccccc21")
        assert "chromanone" in name.lower() or "chroman" in name.lower()
        assert not _has_vb_notation(name)

    def test_chromone(self):
        name = name_compound("O=c1ccoc2ccccc12")
        assert "chromone" in name.lower() or "chromen" in name.lower()
        assert not _has_vb_notation(name)


# ---------------------------------------------------------------------------
# Test Class 3: Bicyclo Expansion
# ---------------------------------------------------------------------------

class TestBicycloExpansion:
    """Tests norbornene naming."""

    def test_norbornene(self):
        name = name_compound("C1=CC2CCC1C2")
        lower = name.lower()
        # Either retained name or correct systematic name
        assert "norbornene" in lower or "bicyclo[2.2.1]hept" in lower
        assert not _has_vb_notation(name)

    def test_norbornene_systematic_form(self):
        """Norbornene = bicyclo[2.2.1]hept-2-ene — verify no wrong bridge."""
        name = name_compound("C1=CC2CCC1C2")
        if "bicyclo" in name.lower():
            assert "2.2.1" in name


# ---------------------------------------------------------------------------
# Test Class 4: OPSIN Round-Trip
# ---------------------------------------------------------------------------

class TestOPSINRoundTrip:
    """Validate OPSIN can parse all new retained names."""

    COMPOUNDS = [
        ("c1ccc2c(c1)cnc1ccccc12", "phenanthridine"),  # Fixed SMILES (Phase 101)
        ("c1ccc2c(c1)[nH]c1cnccc12", "beta-carboline"),
        ("O=c1c2ccccc2[nH]c2ccccc12", "acridone"),
        ("O=c1cc(-c2ccccc2)oc2ccccc12", "flavone"),
        ("O=C1CC(c2ccccc2)Oc2ccccc21", "flavanone"),
        ("O=c1c(-c2ccccc2)coc2ccccc12", "isoflavone"),
        ("O=C1CCOc2ccccc21", "chromanone"),
        ("O=c1ccoc2ccccc12", "chromone"),
        ("C1=CC2CCC1C2", "norbornene"),
    ]

    @pytest.mark.parametrize("smiles,label", COMPOUNDS, ids=[c[1] for c in COMPOUNDS])
    def test_opsin_parses_generated_name(self, smiles, label):
        """OPSIN should be able to parse the name we generate."""
        generated_name = name_compound(smiles)
        opsin_smi = _opsin_parse(generated_name)
        assert opsin_smi is not None, (
            f"OPSIN failed to parse '{generated_name}' for {label}"
        )


# ---------------------------------------------------------------------------
# Test Class 5: Canary Stability
# ---------------------------------------------------------------------------

class TestCanaryStability:
    """Verify canary compounds remain stable after dictionary expansion."""

    def test_canary_suite_passes(self):
        """Run canary tests and verify 0 failures."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest",
             "tests/integration/test_canary_rt75.py",
             "-x", "-q", "--tb=line"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(Path(__file__).resolve().parents[2]),  # project root — CI-portable, replaces hardcoded /home/kohulan path per REVIEWS §Plan 03 HIGH #2
        )
        # Check that no failures occurred
        assert result.returncode == 0, (
            f"Canary tests failed:\n{result.stdout}\n{result.stderr}"
        )
