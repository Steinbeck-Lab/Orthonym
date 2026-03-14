"""Unit tests for thiocarboxylic acid detection and naming (IUPAC P-65.3).

Tests three variants:
- thioic S-acid: R-C(=O)-SH
- thioic O-acid: R-C(=S)-OH
- dithioic acid: R-C(=S)-SH
"""

import pytest
from rdkit import Chem

from src.orthonym.perception.functional_groups import detect_functional_groups
from src.orthonym.rules.seniority import get_suffix, get_prefix
from src.orthonym.namer import name_compound


# ============================================================================
# Detection tests -- SMARTS correctness and collision resolution
# ============================================================================


@pytest.mark.unit
class TestThiocarboxylicAcidDetection:
    """Verify SMARTS patterns correctly detect all three thiocarboxylic acid types."""

    def test_thioic_S_acid_detected(self):
        """CC(=O)S detects thioic_S_acid."""
        mol = Chem.MolFromSmiles("CC(=O)S")
        groups = detect_functional_groups(mol)
        assert "thioic_S_acid" in groups

    def test_thioic_O_acid_detected(self):
        """CC(=S)O detects thioic_O_acid."""
        mol = Chem.MolFromSmiles("CC(=S)O")
        groups = detect_functional_groups(mol)
        assert "thioic_O_acid" in groups

    def test_dithioic_acid_detected(self):
        """CC(=S)S detects dithioic_acid."""
        mol = Chem.MolFromSmiles("CC(=S)S")
        groups = detect_functional_groups(mol)
        assert "dithioic_acid" in groups

    def test_thioic_S_acid_no_thiol(self):
        """SH in C(=O)SH must NOT also be detected as thiol."""
        mol = Chem.MolFromSmiles("CC(=O)S")
        groups = detect_functional_groups(mol)
        assert "thiol" not in groups

    def test_dithioic_acid_no_thiol(self):
        """SH in C(=S)SH must NOT also be detected as thiol."""
        mol = Chem.MolFromSmiles("CC(=S)S")
        groups = detect_functional_groups(mol)
        assert "thiol" not in groups

    def test_thioic_S_acid_no_thioester(self):
        """C(=O)SH must NOT also be detected as thioester."""
        mol = Chem.MolFromSmiles("CC(=O)S")
        groups = detect_functional_groups(mol)
        assert "thioester" not in groups

    def test_thioic_O_acid_no_carboxylic_acid(self):
        """C(=S)OH must NOT be detected as carboxylic_acid (different SMARTS)."""
        mol = Chem.MolFromSmiles("CC(=S)O")
        groups = detect_functional_groups(mol)
        assert "carboxylic_acid" not in groups

    def test_propanethioic_S_acid_detected(self):
        """CCC(=O)S detects thioic_S_acid."""
        mol = Chem.MolFromSmiles("CCC(=O)S")
        groups = detect_functional_groups(mol)
        assert "thioic_S_acid" in groups

    def test_real_thiol_still_detected(self):
        """Ethanethiol (CCS) should still detect thiol normally."""
        mol = Chem.MolFromSmiles("CCS")
        groups = detect_functional_groups(mol)
        assert "thiol" in groups
        assert "thioic_S_acid" not in groups

    def test_real_carboxylic_acid_unaffected(self):
        """Acetic acid (CC(=O)O) should still detect carboxylic_acid normally."""
        mol = Chem.MolFromSmiles("CC(=O)O")
        groups = detect_functional_groups(mol)
        assert "carboxylic_acid" in groups
        assert "thioic_O_acid" not in groups


# ============================================================================
# Seniority/suffix tests
# ============================================================================


@pytest.mark.unit
class TestThiocarboxylicAcidSeniority:
    """Verify suffix and prefix forms are correctly defined."""

    def test_thioic_S_acid_suffix(self):
        assert get_suffix("thioic_S_acid") == "thioic S-acid"

    def test_thioic_O_acid_suffix(self):
        assert get_suffix("thioic_O_acid") == "thioic O-acid"

    def test_dithioic_acid_suffix(self):
        assert get_suffix("dithioic_acid") == "dithioic acid"

    def test_thioic_S_acid_ring_suffix(self):
        assert get_suffix("thioic_S_acid", is_ring=True) == "carbothioic S-acid"

    def test_thioic_O_acid_ring_suffix(self):
        assert get_suffix("thioic_O_acid", is_ring=True) == "carbothioic O-acid"

    def test_dithioic_acid_ring_suffix(self):
        assert get_suffix("dithioic_acid", is_ring=True) == "carbodithioic acid"

    def test_prefix_forms(self):
        """Thiocarboxylic acids have prefix forms per IUPAC P-65.1.1.4."""
        assert get_prefix("thioic_S_acid") == "sulfanylcarbonyl"
        assert get_prefix("thioic_O_acid") == "carbothioyl"
        assert get_prefix("dithioic_acid") == "dithiocarboxy"


# ============================================================================
# End-to-end naming tests
# ============================================================================


@pytest.mark.unit
class TestThiocarboxylicAcidNaming:
    """Verify full naming pipeline produces correct IUPAC names."""

    def test_ethanethioic_S_acid(self):
        """CC(=O)S -> ethanethioic S-acid"""
        assert name_compound("CC(=O)S") == "ethanethioic S-acid"

    def test_ethanethioic_O_acid(self):
        """CC(=S)O -> ethanethioic O-acid"""
        assert name_compound("CC(=S)O") == "ethanethioic O-acid"

    def test_ethanedithioic_acid(self):
        """CC(=S)S -> ethanedithioic acid"""
        assert name_compound("CC(=S)S") == "ethanedithioic acid"

    def test_propanethioic_S_acid(self):
        """CCC(=O)S -> propanethioic S-acid"""
        assert name_compound("CCC(=O)S") == "propanethioic S-acid"

    def test_benzenecarbothioic_S_acid_no_regression(self):
        """SC(=O)c1ccccc1 -> benzenecarbothioic S-acid (existing benzene path)"""
        assert name_compound("SC(=O)c1ccccc1") == "benzenecarbothioic S-acid"


# ============================================================================
# OPSIN round-trip tests
# ============================================================================


@pytest.mark.unit
class TestThiocarboxylicAcidOPSIN:
    """Verify OPSIN can parse the generated names back to structures."""

    @pytest.fixture(autouse=True)
    def _check_opsin(self):
        """Skip if OPSIN (Java) is not available."""
        import shutil
        if not shutil.which("java"):
            pytest.skip("Java not available for OPSIN round-trip")

    @staticmethod
    def _opsin_parse(name):
        """Parse a name through OPSIN and return the SMILES (or None)."""
        import subprocess
        import glob
        opsin_jars = glob.glob("/home/kohulan/OpenSTOUT/Orthonym/opsin/opsin-cli/target/opsin-cli-*-jar-with-dependencies.jar")
        if not opsin_jars:
            pytest.skip("OPSIN jar not found")
        jar = opsin_jars[0]
        proc = subprocess.run(
            ["java", "-jar", jar, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=30,
        )
        result = proc.stdout.strip()
        if not result or result == "":
            return None
        return result

    def test_opsin_ethanethioic_S_acid(self):
        smi = self._opsin_parse("ethanethioic S-acid")
        assert smi is not None, "OPSIN could not parse 'ethanethioic S-acid'"

    def test_opsin_ethanethioic_O_acid(self):
        smi = self._opsin_parse("ethanethioic O-acid")
        assert smi is not None, "OPSIN could not parse 'ethanethioic O-acid'"

    def test_opsin_ethanedithioic_acid(self):
        smi = self._opsin_parse("ethanedithioic acid")
        assert smi is not None, "OPSIN could not parse 'ethanedithioic acid'"
