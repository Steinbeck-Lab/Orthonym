import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestWave2P3SpiroVerify:
    """Rows already correct at HEAD — lock them against regression before
    the engine + scorer edits in this plan."""

    @pytest.mark.parametrize("smiles,expected", [
        ("C1CCC2(CC1)CC=CC2", "spiro[4.5]dec-2-ene"),   #
        ("O1CCCC12CCCCC2", "1-oxaspiro[4.5]decane"),    # heterospiro control
    ])
    def test_already_correct(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestP24BranchedPolyspiro:
    def test_trispiro_superscript(self):
        #: superscript revisit locants are essential; without them the
        # name is ambiguous (BB note at:10018).
        assert name_compound("C1CC12CCC1(CC1)CCC1(CC1)CC2") == \
            "trispiro[2.2.2^6.2.2^11.2^3]pentadecane"

    def test_dispiro_unchanged(self):
        # 2-spiro-atom path must NOT regress.
        assert name_compound("C1CC2(CC1)CC1(CC2)CCCC1") == "dispiro[4.1.4^7.2^5]tridecane"


@pytest.mark.unit
class TestP24Dispiroter:
    def test_dispiroter_bicyclohexane(self):
        #: three identical bicyclo[3.1.0]hexane components, 2 spiro atoms.
        assert name_compound("C12CC3(CC2C1)CC1C2(C1C3)C3CCCC32") == \
            "3,3':6',6''-dispiroter[bicyclo[3.1.0]hexane]"


@pytest.mark.unit
class TestP24ThiaSpiroVonBaeyer:
    def test_thiaspiro_bicyclooctane_fluorene(self):
        #: 'a'-replacement prefix cited before spiro.
        assert name_compound("C1=CC=CC=2C3=CC=CC=C3C3(C12)C1CCC(S3)CC1") == \
            "3-thiaspiro[bicyclo[2.2.2]octane-2,9'-fluorene]"


@pytest.mark.unit
class TestP24UnbranchedPolyspiroDifferent:
    def test_fluorene_cyclohexane_indene(self):
        #: unbranched polyspiro, three different components.
        assert name_compound("C1=CC2(CCC3(CC2)c2ccccc2-c2ccccc23)c2ccccc21") == \
            "dispiro[fluorene-9,1'-cyclohexane-4',1''-indene]"


@pytest.mark.unit
class TestP24BranchedPolyspiroDifferent:
    #: branched polyspiro with different terminals around a central
    # component that carries >=3 spiro junctions. This target's central
    # component is the HETEROMONOCYCLE [1,5]dithiocane (a Hantzsch-Widman ring
    # name now emitted by _name_hw_monocycle_component), assembled by the
    # branched 3-junction walk (_name_branched_polyspiro_different_core) with
    # the indene terminal from the carbo-PAH template. BUILT (Wave2 D5a).
    def test_cyclohexane_dithiocane_cyclopentane_indene(self):
        assert name_compound("C=1C2(C=C3C=CC=CC13)CC1(SCCC3(CCCC3)S2)CCCCC1") == \
            "trispiro[cyclohexane-1,2'-[1,5]dithiocane-6',1''-cyclopentane-4',2'''-indene]"

    def test_branched_polyspiro_classified(self):
        # The SPIRO dispatch recognises the branched-polyspiro class and now
        # BUILDS it (component-name method).
        from rdkit import Chem
        from orthonym.rules.spiro import (
            is_branched_polyspiro, name_branched_polyspiro,
        )
        from orthonym.assembly.composer import _classify_complex_ring
        m = Chem.MolFromSmiles("C=1C2(C=C3C=CC=CC13)CC1(SCCC3(CCCC3)S2)CCCCC1")
        assert is_branched_polyspiro(m) is True
        assert name_branched_polyspiro(m) is not None
        assert _classify_complex_ring(m) == "polyspiro-branched-different"

    def test_branched_polyspiro_deterministic(self):
        # P1_amide determinism lesson: spiro component ordering/numbering must be
        # spelling-independent — name the SAME molecule from randomized SMILES.
        from rdkit import Chem
        m = Chem.MolFromSmiles("C=1C2(C=C3C=CC=CC13)CC1(SCCC3(CCCC3)S2)CCCCC1")
        outs = {name_compound(Chem.MolToSmiles(m, doRandom=True))
                for _ in range(8)}
        assert len(outs) == 1
        assert next(iter(outs)) == \
            "trispiro[cyclohexane-1,2'-[1,5]dithiocane-6',1''-cyclopentane-4',2'''-indene]"


@pytest.mark.unit
class TestP31SpiroVonBaeyerUnsaturation:
    def test_spirobi_bicyclononane_diene(self):
        #: ring 'ene'/'diene' cited AFTER the last bracket.
        assert name_compound("C12CC3(CC(C=CC1)C2)CC2CC=CC(C3)C2") == \
            "3,3'-spirobi[bicyclo[3.3.1]nonane]-6,6'-diene"


@pytest.mark.unit
class TestP24LambdaMonocyclicTripleSpiro:
    #: three monocyclic rings sharing ONE nonstandard (lambda6 S)
    # spiro atom. Once the monocyclic-HW spiro-component namer landed (Wave2 D5a)
    # this resolves to the spiroter multiplicative form
    # ``1lambda6,1',1''-spiroter[thietane]`` (OPSIN-round-trippable, verified),
    # so the previously-deferred case now BUILDS via the spiroter path (the
    # dedicated name_lambda_multiring_spiro von-Baeyer form is unneeded here).
    _NEAREST = "C1CS23(C1)(CCC2)CCC3"  # lambda6 S in three 4-membered rings
    _TARGET = "1λ6,1',1''-spiroter[thietane]"

    def test_lambda_trispiro(self):
        assert name_compound(self._NEAREST) == self._TARGET

    def test_lambda_trispiro_deterministic(self):
        from rdkit import Chem
        m = Chem.MolFromSmiles(self._NEAREST)
        outs = {name_compound(Chem.MolToSmiles(m, doRandom=True))
                for _ in range(8)}
        assert outs == {self._TARGET}


@pytest.mark.unit
class TestP24LambdaSpirobiSpiroter:
    def test_lambda4_spirobi_benzodioxathiole(self):
        assert name_compound("O1S2(OC3=C1C=CC=C3)OC3=C(O2)C=CC=C3") == \
            "2λ4,2'-spirobi[[1,3,2]benzodioxathiole]"

    def test_lambda6_spiroter_benzodioxathiole(self):
        assert name_compound("O1S23(OC4=C1C=CC=C4)(OC4=C(O2)C=CC=C4)OC4=C(O3)C=CC=C4") == \
            "2λ6,2',2''-spiroter[[1,3,2]benzodioxathiole]"


@pytest.mark.unit
class TestP24LambdaSpiroDifferent:
    #: monospiro, DIFFERENT polycyclic components, >=1 with a λ spiro
    # atom. BUILT (Wave2 D5a): _name_spiro_component names both the
    # [1,3,2]benzoxazaphosphole fused catalog component and the monocyclic-HW
    # [1,3,5,2]triazaphosphinine, and the spiro-VB core assembles the indicated-H
    # (3H) + λ5 front prefix. The mancude C=N/N=P ring bonds are expressed by the
    # component names, so the von-Baeyer '-ene' suffix machinery (which is scoped
    # to cage components only) does NOT fire. OPSIN-round-trippable.
    _SMILES = "N1=P2(N=CN=C1)OC1=C(N2)C=CC=C1"
    _TARGET = ("3H-2λ5-spiro[[1,3,2]benzoxazaphosphole-2,2'-"
               "[1,3,5,2]triazaphosphinine]")

    def test_lambda5_spiro_benzoxazaphosphole_triazaphosphinine(self):
        assert name_compound(self._SMILES) == self._TARGET

    def test_lambda_spiro_different_builds(self):
        # The spiro-VB path builds the full name over every ring atom.
        from rdkit import Chem
        from orthonym.rules.spiro import name_spiro_vonbaeyer
        res = name_spiro_vonbaeyer(Chem.MolFromSmiles(self._SMILES))
        assert res is not None
        assert res[0] == self._TARGET

    def test_lambda_spiro_different_deterministic(self):
        # Determinism: name the SAME molecule from randomized SMILES spellings;
        # component citation order (alphanumerical) + numbering must be
        # spelling-independent.
        from rdkit import Chem
        m = Chem.MolFromSmiles(self._SMILES)
        outs = {name_compound(Chem.MolToSmiles(m, doRandom=True))
                for _ in range(8)}
        assert outs == {self._TARGET}


@pytest.mark.unit
class TestP24LambdaUnbranchedPolyspiro:
    #: unbranched polyspiro, DIFFERENT components, >=1 λ spiro atom.
    # Terminals are heteromonocycles (thiane, thiolane) and the central is
    # benzo[1,2-c:4,5-c']dithiophene. BUILT (Wave2 D5a): the heteromonocycle
    # spiro-component namer names thiane/thiolane, the benzo[1,2-c:4,5-c']-
    # dithiophene central is nameable, and the combined indicated-H (1'H,3'H) +
    # multi-λ (1lambda4,1''lambda4) front prefix is assembled by
    # _build_lambda_ih_front_prefix (indicated-H NOT at a spiro atom is
    # front-cited; the λ spiro atoms are cited once each). OPSIN-round-trippable.
    _SMILES = "S12(CCCC1)C=C1C(=C2)C=C2CS3(CC2=C1)CCCCC3"
    _TARGET = ("1'H,3'H-1λ4,1''λ4-dispiro[thiane-1,2'-"
               "benzo[1,2-c:4,5-c']dithiophene-6',1''-thiolane]")

    def test_lambda_dispiro_thiane_dithiophene_thiolane(self):
        assert name_compound(self._SMILES) == self._TARGET

    def test_lambda_unbranched_polyspiro_builds(self):
        # The polyspiro-different core builds the full name (component maps cover
        # every ring atom).
        from rdkit import Chem
        from orthonym.rules.spiro import name_unbranched_polyspiro_different
        res = name_unbranched_polyspiro_different(Chem.MolFromSmiles(self._SMILES))
        assert res is not None
        assert res[0] == self._TARGET

    def test_lambda_unbranched_polyspiro_deterministic(self):
        # P1_amide determinism lesson: the indicated-H/λ front prefix, component
        # citation order and numbering must be spelling-independent — name the
        # SAME molecule from randomized SMILES and require one identical output.
        from rdkit import Chem
        m = Chem.MolFromSmiles(self._SMILES)
        outs = {name_compound(Chem.MolToSmiles(m, doRandom=True))
                for _ in range(8)}
        assert outs == {self._TARGET}
