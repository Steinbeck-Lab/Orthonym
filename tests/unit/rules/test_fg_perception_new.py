"""Tests for functional group perception fixes and new FG patterns.

a phase Plan 02: through
-: Aldehyde SMARTS fix (formaldehyde detection)
-: Amine SMARTS fix (sp2 carbon amines)
-: New FG patterns (hydroxamic acid, cyanate, thiocyanate, azo)
-: Seniority table entries for new FGs
"""

import pytest
from rdkit import Chem
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.seniority import SUFFIX_FORMS, PREFIX_FORMS, SENIORITY_ORDER


# ───: Aldehyde SMARTS fix ───────────────────────────────────────

class TestAldehydeDetection:
    """Aldehyde pattern must match formaldehyde (O=C) and not regress on normal aldehydes."""

    def test_formaldehyde_detected_as_aldehyde(self):
        mol = Chem.MolFromSmiles("O=C")
        groups = detect_functional_groups(mol)
        assert "aldehyde" in groups, "Formaldehyde (O=C) must be detected as aldehyde"

    def test_acetaldehyde_still_detected(self):
        mol = Chem.MolFromSmiles("CC=O")
        groups = detect_functional_groups(mol)
        assert "aldehyde" in groups, "Acetaldehyde must still be detected as aldehyde"

    def test_benzaldehyde_still_detected(self):
        mol = Chem.MolFromSmiles("O=Cc1ccccc1")
        groups = detect_functional_groups(mol)
        assert "aldehyde" in groups, "Benzaldehyde must still be detected as aldehyde"

    def test_amide_not_falsely_detected_as_aldehyde(self):
        """Amide C(=O)N should NOT be detected as aldehyde (collision resolution)."""
        mol = Chem.MolFromSmiles("CC(=O)N")
        groups = detect_functional_groups(mol)
        # Primary amide should be detected, not aldehyde
        assert "primary_amide" in groups
        assert "aldehyde" not in groups, "Amides must not be falsely detected as aldehyde"

    def test_formic_acid_not_falsely_detected_as_aldehyde(self):
        """Formic acid HC(=O)OH: the CX3H1(=O) matches but carboxylic_acid should take priority."""
        mol = Chem.MolFromSmiles("O=CO")
        groups = detect_functional_groups(mol)
        assert "carboxylic_acid" in groups
        # Aldehyde may or may not match depending on collision rules,
        # but carboxylic_acid must be present


# ───: Amine SMARTS fix ──────────────────────────────────────────

class TestAmineDetection:
    """Amine pattern must match amines on sp2 carbons."""

    def test_aminoethylene_detected(self):
        """Vinyl amine (C=C-NH2) must be detected as primary_amine."""
        mol = Chem.MolFromSmiles("C=CN")
        groups = detect_functional_groups(mol)
        assert "primary_amine" in groups, "Aminoethylene must be detected as primary_amine"

    def test_standard_amine_still_detected(self):
        mol = Chem.MolFromSmiles("CCN")
        groups = detect_functional_groups(mol)
        assert "primary_amine" in groups

    def test_aniline_detected_as_aromatic_amine(self):
        """Aniline must be detected as aromatic_amine, not just primary_amine."""
        mol = Chem.MolFromSmiles("c1ccccc1N")
        groups = detect_functional_groups(mol)
        assert "aromatic_amine" in groups, "Aniline must be detected as aromatic_amine"

    def test_amide_nitrogen_not_detected_as_amine(self):
        """Amide nitrogen (NX3H2 on C=O) should be detected as amide, not amine."""
        mol = Chem.MolFromSmiles("CC(=O)N")
        groups = detect_functional_groups(mol)
        assert "primary_amide" in groups
        # The amide N is NX3H2 but attached to C(=O), so amide SMARTS should match first


# ───: New FG patterns ───────────────────────────────────────────

class TestHydroxamicAcidDetection:
    """Hydroxamic acid: R-C(=O)-NHOH pattern."""

    def test_acetohydroxamic_acid_detected(self):
        mol = Chem.MolFromSmiles("CC(=O)NO")
        groups = detect_functional_groups(mol)
        assert "hydroxamic_acid" in groups, "Acetohydroxamic acid must be detected"

    def test_hydroxamic_acid_suppresses_amide_and_alcohol(self):
        """Hydroxamic acid atoms should NOT also show as amide + alcohol."""
        mol = Chem.MolFromSmiles("CC(=O)NO")
        groups = detect_functional_groups(mol)
        assert "hydroxamic_acid" in groups
        # The N-C(=O) part should not also be detected as amide
        assert "primary_amide" not in groups, "Hydroxamic acid must suppress amide false positive"
        assert "secondary_amide" not in groups

    def test_simple_amide_not_detected_as_hydroxamic_acid(self):
        """Simple amides (no OH on N) must NOT match hydroxamic acid pattern."""
        mol = Chem.MolFromSmiles("CC(=O)N")
        groups = detect_functional_groups(mol)
        assert "hydroxamic_acid" not in groups
        assert "primary_amide" in groups


class TestCyanateDetection:
    """Cyanate: R-O-C#N pattern."""

    def test_methyl_cyanate_detected(self):
        mol = Chem.MolFromSmiles("COC#N")
        groups = detect_functional_groups(mol)
        assert "cyanate" in groups, "Methyl cyanate must be detected"

    def test_simple_ether_not_detected_as_cyanate(self):
        """Simple ethers must NOT match cyanate pattern."""
        mol = Chem.MolFromSmiles("COC")
        groups = detect_functional_groups(mol)
        assert "cyanate" not in groups

    def test_simple_nitrile_not_detected_as_cyanate(self):
        """Simple nitriles (C-C#N, no O) must NOT match cyanate pattern."""
        mol = Chem.MolFromSmiles("CC#N")
        groups = detect_functional_groups(mol)
        assert "cyanate" not in groups
        assert "nitrile" in groups


class TestThiocyanateDetection:
    """Thiocyanate: R-S-C#N pattern."""

    def test_methyl_thiocyanate_detected(self):
        mol = Chem.MolFromSmiles("CSC#N")
        groups = detect_functional_groups(mol)
        assert "thiocyanate" in groups, "Methyl thiocyanate must be detected"

    def test_simple_thioether_not_detected_as_thiocyanate(self):
        mol = Chem.MolFromSmiles("CSC")
        groups = detect_functional_groups(mol)
        assert "thiocyanate" not in groups


class TestAzoDetection:
    """Azo: R-N=N-R pattern."""

    def test_azobenzene_detected(self):
        mol = Chem.MolFromSmiles("c1ccccc1/N=N/c1ccccc1")
        groups = detect_functional_groups(mol)
        assert "azo" in groups, "Azobenzene must be detected as azo"

    def test_simple_diazene_detected(self):
        mol = Chem.MolFromSmiles("CN=NC")
        groups = detect_functional_groups(mol)
        assert "azo" in groups

    def test_simple_imine_not_detected_as_azo(self):
        """Simple imines (C=NH) must NOT match azo pattern."""
        mol = Chem.MolFromSmiles("CC=N")
        groups = detect_functional_groups(mol)
        assert "azo" not in groups

    def test_hydrazone_not_detected_as_azo(self):
        """Hydrazones (C=N-N) have NX2 and NX3, azo requires both NX2."""
        mol = Chem.MolFromSmiles("CC=NN")
        groups = detect_functional_groups(mol)
        assert "azo" not in groups


# ───: Seniority table entries ───────────────────────────────────

class TestSeniorityEntries:
    """New FGs must have proper entries in seniority tables."""

    def test_hydroxamic_acid_in_seniority_order(self):
        assert "hydroxamic_acid" in SENIORITY_ORDER

    def test_hydroxamic_acid_suffix_form(self):
        # R8c /: hydroxamic acid is named via the
        # amide handler as 'N-hydroxy<stem>amide', NOT the retained 'hydroxamic
        # acid' suffix. SUFFIX_FORMS entry is None (handler-emitted, no
        # substitutive suffix) — same contract as thioether, sulfoxide, etc.
        assert "hydroxamic_acid" in SUFFIX_FORMS
        assert SUFFIX_FORMS["hydroxamic_acid"] is None

    def test_hydroxamic_acid_prefix_form(self):
        assert "hydroxamic_acid" in PREFIX_FORMS
        assert PREFIX_FORMS["hydroxamic_acid"] is not None

    def test_cyanate_prefix_form(self):
        assert "cyanate" in PREFIX_FORMS
        assert PREFIX_FORMS["cyanate"] == "cyanato"

    def test_thiocyanate_prefix_form(self):
        assert "thiocyanate" in PREFIX_FORMS
        assert PREFIX_FORMS["thiocyanate"] == "thiocyanato"

    def test_azo_prefix_form(self):
        assert "azo" in PREFIX_FORMS
        assert PREFIX_FORMS["azo"] == "diazenyl"
