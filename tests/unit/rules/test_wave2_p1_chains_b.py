import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestWave2P1ChainsBVerify:
    """Rows already correct at HEAD — lock them against regression."""

    @pytest.mark.parametrize("smiles,expected", [
        # P-16.7.2(d): dialkyl-disulfide PIN is substitutive (BB 27874);
        # no-elision multiplicative parents (di/tri + acetic acid).
        ("C(C)SSCC", "(ethyldisulfanyl)ethane"),
        ("OC(=O)CN(CC(=O)O)CC(=O)O", "2,2',2''-nitrilotriacetic acid"),
        ("OC(=O)COCC(=O)O", "2,2'-oxydiacetic acid"),
        # P-22.2.2.1.5: alternate HW stems (N-form iridine/etidine/olidine,
        # no-N form irene/olane).
        ("N1CC1", "aziridine"),
        ("O1CC1", "oxirane"),
        ("C1CNC1", "azetidine"),
        ("C1CCNC1", "pyrrolidine"),
        ("C1CCOC1", "oxolane"),
        # P-22.2.5: homogeneous monocyclic parent hydrides (cyclo + chain).
        ("C1CCCCCCC1", "cyclooctane"),
        ("N1NNNN1", "pentazolidine"),
        # P-25.3.2.1.1: monocyclic hydrocarbon parent (cycloheptatriene).
        ("C1=CC=CCC=C1", "cyclohepta-1,3,5-triene"),
    ])
    def test_already_correct(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestP22MultiplierAElision:
    def test_tetrazine_elides_multiplier_a(self):
        # P-22.2.2.1.2: 'tetra' + 'aza' -> 'tetraza' -> '...tetrazine'
        assert name_compound("N1=NN=NC=C1") == "1,2,3,4-tetrazine"
