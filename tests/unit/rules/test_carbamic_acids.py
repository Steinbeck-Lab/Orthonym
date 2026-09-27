"""Tests for carbamic acid detection and naming (IUPAC.

Carbamic acid (H2N-COOH) is a retained name for aminoformic acid.
It should be detected as its own principal group, NOT as carboxylic acid.
N-substituted forms produce names like methylcarbamic acid:
(the Blue Book) cites them without the italic-N locant,
'(CH3)2N-COOH dimethylcarbamic acid (PIN)' (:30762), 'phenylcarbamic acid (PIN)'
(:6798), '2-hydroxypropyl (2-aminoethyl)carbamate (PIN)' (:30766).
"""

import pytest
from rdkit import Chem

from orthonym.namer import name_compound
from orthonym.perception.functional_groups import detect_functional_groups


@pytest.mark.unit
class TestCarbamicAcidDetection:
    """Functional group detection tests for carbamic acid."""

    def test_carbamic_acid_detected(self):
        """NC(=O)O should detect carbamic_acid, NOT carboxylic_acid."""
        mol = Chem.MolFromSmiles("NC(=O)O")
        groups = detect_functional_groups(mol)
        assert "carbamic_acid" in groups
        assert "carboxylic_acid" not in groups, "Should not falsely detect carboxylic_acid"

    def test_n_methyl_carbamic_acid_detected(self):
        """CNC(=O)O should detect carbamic_acid."""
        mol = Chem.MolFromSmiles("CNC(=O)O")
        groups = detect_functional_groups(mol)
        assert "carbamic_acid" in groups
        assert "carboxylic_acid" not in groups

    def test_nn_dimethyl_carbamic_acid_detected(self):
        """CN(C)C(=O)O should detect carbamic_acid."""
        mol = Chem.MolFromSmiles("CN(C)C(=O)O")
        groups = detect_functional_groups(mol)
        assert "carbamic_acid" in groups
        assert "carboxylic_acid" not in groups

    def test_regular_carboxylic_acid_not_affected(self):
        """CC(=O)O (acetic acid) should still detect carboxylic_acid."""
        mol = Chem.MolFromSmiles("CC(=O)O")
        groups = detect_functional_groups(mol)
        assert "carboxylic_acid" in groups
        assert "carbamic_acid" not in groups

    def test_carbamate_not_carbamic_acid(self):
        """CCOC(=O)NC (ethyl N-methylcarbamate) should NOT detect carbamic_acid."""
        mol = Chem.MolFromSmiles("CCOC(=O)NC")
        groups = detect_functional_groups(mol)
        assert "carbamate" in groups
        assert "carbamic_acid" not in groups


@pytest.mark.unit
class TestCarbamicAcidNaming:
    """Naming tests for carbamic acid and N-substituted forms."""

    def test_unsubstituted_carbamic_acid(self):
        """NC(=O)O -> carbamic acid."""
        assert name_compound("NC(=O)O") == "carbamic acid"

    def test_n_methyl_carbamic_acid(self):
        """CNC(=O)O -> N-methylcarbamic acid."""
        assert name_compound("CNC(=O)O") == "methylcarbamic acid"

    def test_nn_dimethyl_carbamic_acid(self):
        """CN(C)C(=O)O -> N,N-dimethylcarbamic acid."""
        assert name_compound("CN(C)C(=O)O") == "dimethylcarbamic acid"

    def test_nn_diethyl_carbamic_acid(self):
        """CCN(CC)C(=O)O -> N,N-diethylcarbamic acid."""
        assert name_compound("CCN(CC)C(=O)O") == "diethylcarbamic acid"

    def test_n_phenyl_carbamic_acid(self):
        """c1ccc(NC(=O)O)cc1 -> N-phenylcarbamic acid."""
        assert name_compound("c1ccc(NC(=O)O)cc1") == "phenylcarbamic acid"


@pytest.mark.unit
class TestCarbamateNoRegression:
    """Verify existing carbamate naming still works."""

    def test_ethyl_n_methylcarbamate(self):
        """CCOC(=O)NC should still produce ethyl N-methylcarbamate."""
        assert name_compound("CCOC(=O)NC") == "ethyl methylcarbamate"
