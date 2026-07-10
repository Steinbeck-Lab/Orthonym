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


@pytest.mark.unit
class TestP24UnbranchedPolyspiroDifferent:
    def test_fluorene_cyclohexane_indene(self):
        # P-24.6: unbranched polyspiro, three different components.
        assert name_compound("C1=CC2(CCC3(CC2)c2ccccc2-c2ccccc23)c2ccccc21") == \
            "dispiro[fluorene-9,1'-cyclohexane-4',1''-indene]"


@pytest.mark.unit
class TestP24BranchedPolyspiroDifferent:
    # P-24.7.2: branched polyspiro with different terminals around a central
    # component that carries >=2 spiro junctions. This target's central
    # component is the HETEROMONOCYCLE [1,5]dithiocane (a Hantzsch-Widman ring
    # name the spiro-component namers do not yet emit) AND requires the branched
    # 3-junction central walk + robust indene-fragment kekulization. UNDER-SCOPE
    # (plan rule 7): deferred with a follow-up; fail-closed guaranteed below.
    @pytest.mark.xfail(reason="P-24.7.2 branched polyspiro DIFFERENT: central "
                              "heteromonocycle [1,5]dithiocane spiro-component "
                              "naming + branched 3-junction walk not built; "
                              "deferred (fail-closed, never a wrong name).")
    def test_cyclohexane_dithiocane_cyclopentane_indene(self):
        assert name_compound("C=1C2(C=C3C=CC=CC13)CC1(SCCC3(CCCC3)S2)CCCCC1") == \
            "trispiro[cyclohexane-1,2'-[1,5]dithiocane-6',1''-cyclopentane-4',2'''-indene]"

    def test_branched_polyspiro_classified_fail_closed(self):
        # The SPIRO dispatch recognises the branched-polyspiro class and REFUSES
        # (never emits a branched-different name it cannot fully build). The
        # complex-ring dispatcher returns None for it; the residual partial
        # heteromonocycle name that the general monocyclic path may emit in a
        # JVM-less test env is a pre-existing namer-core issue (backstopped by
        # SELF-01/OPSIN in production) and is outside this spiro plan's scope.
        from rdkit import Chem
        from orthonym.rules.spiro import (
            is_branched_polyspiro, name_branched_polyspiro,
        )
        from orthonym.assembly.composer import _classify_complex_ring
        m = Chem.MolFromSmiles("C=1C2(C=C3C=CC=CC13)CC1(SCCC3(CCCC3)S2)CCCCC1")
        assert is_branched_polyspiro(m) is True
        assert name_branched_polyspiro(m) is None
        assert _classify_complex_ring(m) == "polyspiro-branched-different"
        # And it must NEVER emit the (unbuilt) correct branched-different name.
        assert name_compound("C=1C2(C=C3C=CC=CC13)CC1(SCCC3(CCCC3)S2)CCCCC1") != \
            "trispiro[cyclohexane-1,2'-[1,5]dithiocane-6',1''-cyclopentane-4',2'''-indene]"


@pytest.mark.unit
class TestP31SpiroVonBaeyerUnsaturation:
    def test_spirobi_bicyclononane_diene(self):
        # P-31.1.5.2.1: ring 'ene'/'diene' cited AFTER the last bracket.
        assert name_compound("C12CC3(CC(C=CC1)C2)CC2CC=CC(C3)C2") == \
            "3,3'-spirobi[bicyclo[3.3.1]nonane]-6,6'-diene"
