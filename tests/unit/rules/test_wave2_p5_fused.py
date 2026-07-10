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


@pytest.mark.unit
class TestP52BenzoGhiPerylene:
    def test_benzo_ghi_perylene(self):
        # P-52.2.4.2 / large peri-fused PAH catalog parent.
        # OPSIN-RT-verified: benzo[ghi]perylene -> c1cc2ccc3ccc4ccc5cccc6c(c1)c2c3c4c56
        assert name_compound("c1cc2ccc3ccc4ccc5cccc6c(c1)c2c3c4c56") == "benzo[ghi]perylene"


@pytest.mark.unit
class TestP25MultiparentDifuranC:
    def test_benzo_difuran_c_prime(self):
        # P-25.3.5.3: multiparent name preferred to a fused-ring name.
        # OPSIN-RT-verified: benzo[1,2-b:4,5-c']difuran -> c1cc2cc3cocc3cc2o1
        assert name_compound("c1cc2cc3cocc3cc2o1") == "benzo[1,2-b:4,5-c']difuran"

    def test_benzo_difuran_c_prime_ab_order(self):
        # determinism: alternate spelling -> identical output
        assert name_compound("o1cc2cc3ccoc3cc2c1") == name_compound("c1cc2cc3cocc3cc2o1")


@pytest.mark.unit
class TestP25MultiparentDifuranB:
    def test_benzo_difuran_b_prime(self):
        # P-25.3.7.1: multiparent, one interparent (benzene) component; primed letters, colon-separated.
        # OPSIN-RT-verified: benzo[1,2-b:4,5-b']difuran -> c1cc2cc3occc3cc2o1
        assert name_compound("c1cc2cc3occc3cc2o1") == "benzo[1,2-b:4,5-b']difuran"

    def test_benzo_difuran_b_prime_ab_order(self):
        # determinism: alternate spelling of the SAME b' structure (the plan's
        # original probe SMILES was a distinct isomer; replaced with an RDKit
        # rooted respelling that canon-matches the b' key) -> identical output.
        assert name_compound("c1coc2c1cc1occc1c2") == name_compound("c1cc2cc3occc3cc2o1")
