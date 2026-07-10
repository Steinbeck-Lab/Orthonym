import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestWave2P3SpiroVerify:
    """Rows already correct at HEAD — lock them against regression before
    the engine + scorer edits in this plan."""

    @pytest.mark.parametrize("smiles,expected", [
        ("C1CCC2(CC1)CC=CC2", "spiro[4.5]dec-2-ene"),   # P-31.1.5.1
        ("O1CCCC12CCCCC2", "1-oxaspiro[4.5]decane"),    # heterospiro control
    ])
    def test_already_correct(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestP24BranchedPolyspiro:
    def test_trispiro_superscript(self):
        # P-24.2.3: superscript revisit locants are essential; without them the
        # name is ambiguous (BB note at :10018).
        assert name_compound("C1CC12CCC1(CC1)CCC1(CC1)CC2") == \
            "trispiro[2.2.2^6.2.2^11.2^3]pentadecane"

    def test_dispiro_unchanged(self):
        # 2-spiro-atom path must NOT regress.
        assert name_compound("C1CC2(CC1)CC1(CC2)CCCC1") == "dispiro[4.1.4.2]tridecane"


@pytest.mark.unit
class TestP24Dispiroter:
    def test_dispiroter_bicyclohexane(self):
        # P-24.4.1: three identical bicyclo[3.1.0]hexane components, 2 spiro atoms.
        assert name_compound("C12CC3(CC2C1)CC1C2(C1C3)C3CCCC32") == \
            "3,3':6',6''-dispiroter[bicyclo[3.1.0]hexane]"


@pytest.mark.unit
class TestP24ThiaSpiroVonBaeyer:
    def test_thiaspiro_bicyclooctane_fluorene(self):
        # P-24.5.2: 'a'-replacement prefix cited before spiro.
        assert name_compound("C1=CC=CC=2C3=CC=CC=C3C3(C12)C1CCC(S3)CC1") == \
            "3-thiaspiro[bicyclo[2.2.2]octane-2,9'-fluorene]"
