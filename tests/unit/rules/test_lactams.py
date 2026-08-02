"""
Unit tests for monocyclic lactam detection and naming.

Tests cover:
- Detection of lactams in ring sizes 4-7 (beta through epsilon)
- Imide disambiguation (succinimide, maleimide, glutarimide rejected)
- Naming with correct HW parent + vowel elision + -2-one suffix
- End-to-end naming via name_compound()
- Negative tests (lactones, acyclic amides not detected)
- Substituent handling (N-methyl, C-alkyl)
"""

import pytest
from rdkit import Chem

from orthonym.rules.lactams import is_monocyclic_lactam, name_monocyclic_lactam
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLactamDetection:
    """Test is_monocyclic_lactam() detection logic."""

    def test_beta_lactam_detected(self):
        """4-membered ring lactam (beta-lactam) is detected."""
        mol = Chem.MolFromSmiles("C1CC(=O)N1")
        result = is_monocyclic_lactam(mol)
        assert result is not None
        assert result["ring_size"] == 4

    def test_gamma_lactam_detected(self):
        """5-membered ring lactam (gamma-lactam) is detected."""
        mol = Chem.MolFromSmiles("C1CCC(=O)N1")
        result = is_monocyclic_lactam(mol)
        assert result is not None
        assert result["ring_size"] == 5

    def test_delta_lactam_detected(self):
        """6-membered ring lactam (delta-lactam) is detected."""
        mol = Chem.MolFromSmiles("C1CCCC(=O)N1")
        result = is_monocyclic_lactam(mol)
        assert result is not None
        assert result["ring_size"] == 6

    def test_epsilon_lactam_detected(self):
        """7-membered ring lactam (epsilon-lactam) is detected."""
        mol = Chem.MolFromSmiles("C1CCCCC(=O)N1")
        result = is_monocyclic_lactam(mol)
        assert result is not None
        assert result["ring_size"] == 7

    def test_succinimide_not_detected(self):
        """Succinimide (imide, 2 C=O on N) is NOT a lactam."""
        mol = Chem.MolFromSmiles("O=C1CCC(=O)N1")
        result = is_monocyclic_lactam(mol)
        assert result is None

    def test_n_methyl_succinimide_not_detected(self):
        """N-methylsuccinimide (imide) is NOT a lactam."""
        mol = Chem.MolFromSmiles("O=C1CCC(=O)N1C")
        result = is_monocyclic_lactam(mol)
        assert result is None

    def test_maleimide_not_detected(self):
        """Maleimide (unsaturated imide) is NOT a lactam."""
        mol = Chem.MolFromSmiles("O=C1C=CC(=O)N1")
        result = is_monocyclic_lactam(mol)
        assert result is None

    def test_glutarimide_not_detected(self):
        """Glutarimide (6-membered imide) is NOT a lactam."""
        mol = Chem.MolFromSmiles("O=C1CCCC(=O)N1")
        result = is_monocyclic_lactam(mol)
        assert result is None

    def test_lactone_not_detected(self):
        """Lactone (cyclic ester) is NOT a lactam."""
        mol = Chem.MolFromSmiles("C1CC(=O)O1")
        result = is_monocyclic_lactam(mol)
        assert result is None

    def test_acyclic_amide_not_detected(self):
        """Acyclic amide is NOT a lactam."""
        mol = Chem.MolFromSmiles("CC(=O)NC")
        result = is_monocyclic_lactam(mol)
        assert result is None

    def test_none_mol_returns_none(self):
        """None input returns None."""
        result = is_monocyclic_lactam(None)
        assert result is None

    def test_n_methyl_lactam_detected(self):
        """N-substituted lactam is still detected as lactam."""
        mol = Chem.MolFromSmiles("CN1CCC1=O")
        result = is_monocyclic_lactam(mol)
        assert result is not None
        assert result["ring_size"] == 4

    def test_detection_result_keys(self):
        """Detection result has all expected keys."""
        mol = Chem.MolFromSmiles("C1CC(=O)N1")
        result = is_monocyclic_lactam(mol)
        assert "ring_atoms" in result
        assert "nitrogen_idx" in result
        assert "carbonyl_idx" in result
        assert "carbonyl_o_idx" in result
        assert "ring_size" in result


# ---------------------------------------------------------------------------
# Naming tests (via name_monocyclic_lactam)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLactamNaming:
    """Test name_monocyclic_lactam() naming logic."""

    def test_beta_lactam_name(self):
        """Beta-lactam named as azetidin-2-one."""
        mol = Chem.MolFromSmiles("C1CC(=O)N1")
        assert name_monocyclic_lactam(mol) == "azetidin-2-one"

    def test_gamma_lactam_name(self):
        """Gamma-lactam named as pyrrolidin-2-one."""
        mol = Chem.MolFromSmiles("C1CCC(=O)N1")
        assert name_monocyclic_lactam(mol) == "pyrrolidin-2-one"

    def test_delta_lactam_name(self):
        """Delta-lactam named as piperidin-2-one."""
        mol = Chem.MolFromSmiles("C1CCCC(=O)N1")
        assert name_monocyclic_lactam(mol) == "piperidin-2-one"

    def test_epsilon_lactam_name(self):
        """Epsilon-lactam named as azepan-2-one."""
        mol = Chem.MolFromSmiles("C1CCCCC(=O)N1")
        assert name_monocyclic_lactam(mol) == "azepan-2-one"

    def test_n_methyl_beta_lactam_name(self):
        """A beta-lactam's ring N is locant 1, so its methyl cites '1-'.

        CORRECTED 2026-08-02 (Task W). This asserted `N-methylazetidin-2-one`.
        A lactam's PIN is a heterocyclic pseudoketone -- P-66.1.5.1
        (`BlueBookV2.md:33224`), decisive last sentence at `:33229` "Method (1)
        generates preferred IUPAC names." -- so the ring nitrogen is numbered and
        its substituent cites that numeral. P-66.1.3 "'Hidden' amides" (`:33125`)
        demotes the italic-N reading of a heterocyclic ring nitrogen to "general
        nomenclature" only; `:33847` prints
        `1-bromopyrrolidine-2,5-dione (PIN) (not N-bromosuccinimide)`.
        """
        mol = Chem.MolFromSmiles("CN1CCC1=O")
        name = name_monocyclic_lactam(mol)
        assert name == "1-methylazetidin-2-one"

    def test_c_methyl_gamma_lactam_name(self):
        """C-methylated gamma-lactam named correctly with locant."""
        mol = Chem.MolFromSmiles("CC1CCC(=O)N1")
        name = name_monocyclic_lactam(mol)
        assert name == "5-methylpyrrolidin-2-one"

    def test_succinimide_returns_none(self):
        """Imide (succinimide) returns None from naming."""
        mol = Chem.MolFromSmiles("O=C1CCC(=O)N1")
        assert name_monocyclic_lactam(mol) is None

    def test_none_mol_returns_none(self):
        """None input returns None."""
        assert name_monocyclic_lactam(None) is None


# ---------------------------------------------------------------------------
# End-to-end tests (via name_compound)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLactamEndToEnd:
    """Test lactam naming through the full name_compound() pipeline."""

    def test_beta_lactam_e2e(self):
        """Beta-lactam through name_compound."""
        assert name_compound("C1CC(=O)N1") == "azetidin-2-one"

    def test_gamma_lactam_e2e(self):
        """Gamma-lactam through name_compound."""
        assert name_compound("C1CCC(=O)N1") == "pyrrolidin-2-one"

    def test_delta_lactam_e2e(self):
        """Delta-lactam through name_compound."""
        assert name_compound("C1CCCC(=O)N1") == "piperidin-2-one"

    def test_epsilon_lactam_e2e(self):
        """Epsilon-lactam through name_compound (returns retained name caprolactam)."""
        assert name_compound("C1CCCCC(=O)N1") == "caprolactam"

    def test_succinimide_not_lactam_e2e(self):
        """Succinimide does NOT produce a lactam name."""
        name = name_compound("O=C1CCC(=O)N1")
        # Should be named as dioxopyrrolidine, NOT as a lactam
        assert "one" not in name or "pyrrolidin" not in name

    def test_lactone_still_works(self):
        """Lactone naming is not broken by lactam integration."""
        assert name_compound("O=C1CCCO1") == "oxolan-2-one"

    def test_acyclic_amide_not_affected(self):
        """Acyclic amide naming is not affected."""
        name = name_compound("CC(=O)NC")
        assert name is not None
        # Should contain 'amide' or 'acetamide' -- not lactam
        assert "one" not in name or "azetidin" not in name
