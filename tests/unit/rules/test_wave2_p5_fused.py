import pytest
from rdkit import Chem
from orthonym.namer import name_compound


@pytest.mark.unit
class TestWave2P5FusedVerify:
    """C5 fused rows already correct at HEAD — lock them against regression."""

    @pytest.mark.parametrize("smiles,expected", [
        ("c1cc2ccc3cccc4ccc(c1)c2c34", "pyrene"),          # P-25.3.1.3
        ("c1ccc2ccccc2c1", "naphthalene"),                 # P-25.3.2.4 (g)
        ("c1ccc2ncccc2c1", "quinoline"),                   # P-25.3.2.4 (h)
        ("c1ccc2[nH]ccc2c1", "1H-indole"),                 # P-25.3.2.4 (i)
        ("c1ccc2nc[nH]c2c1", "1H-benzimidazole"),          # P-25.3.2.4 (j)
    ])
    def test_already_correct(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("s1,s2", [
        # two SMILES spellings of naphthalene -> identical output (determinism)
        ("c1ccc2ccccc2c1", "c1ccc2c(c1)cccc2"),
        # two spellings of quinoline
        ("c1ccc2ncccc2c1", "c1ccc2c(c1)nccc2"),
    ])
    def test_parent_selection_ab_order_invariant(self, s1, s2):
        # P-25.3.2.4: parent/orientation selection must not depend on the
        # RDKit atom order induced by the input SMILES spelling.
        assert name_compound(s1) == name_compound(s2)
