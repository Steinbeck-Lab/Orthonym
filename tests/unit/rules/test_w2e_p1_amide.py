"""Wave-2 plan P1AM (2026-07-09) — C1 amide/amidine/polyfunctional rows.

Every expected PIN in this file was OPSIN-2.9.0-parse-verified and
canonical-matched against the evidence SMILES during planning.
"""

import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestT2CarbonicFamilyParents:
    """P-66.1.1.1.1.3 (BB 32675) + P-68.3.1.2.4 (BB 38623: 'The systematic
    name is the preferred IUPAC name') + P-66.4.2.2 (BB 34480)."""

    @pytest.mark.parametrize("smiles,expected", [
        ("NNC(=O)N", "hydrazinecarboxamide"),          # was 'semicarbazide'
        ("NNC(=N)NN", "hydrazinecarboximidohydrazide"),  # was unknown
        ("NC(=N)NN", "hydrazinecarboximidamide"),      # was unknown
    ])
    def test_pins(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_protect_carbonohydrazonic_diamide_unchanged(self):
        # existing exact row in the same dict must keep working
        assert name_compound("NC(=NN)N") == "carbonohydrazonic diamide"


@pytest.mark.unit
class TestT3AromaticCarboximidamideProtect:
    """P-66.4.1.1 (BB 34173/34393). Already healed at HEAD — protect pins."""

    @pytest.mark.parametrize("smiles,expected", [
        ("NC(=N)c1ccccc1", "benzenecarboximidamide"),
        ("C(=N)(Nc1ccccc1)c1ccccc1", "N-phenylbenzenecarboximidamide"),
    ])
    def test_protect(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestT4SulfonimidamideRingAndSe:
    """P-66.4.1.1 (BB 34173): S/Se/Te imidamide suffixes; ring parent form."""

    @pytest.mark.parametrize("smiles,expected", [
        ("N=S(N)(=O)c1ccccc1", "benzenesulfonimidamide"),
        ("C[Se](=N)N", "methaneseleninimidamide"),
        ("C[Se](=N)(=O)N", "methaneselenonimidamide"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("CS(=N)(=O)N", "methanesulfonimidamide"),
        ("CS(=N)N", "methanesulfinimidamide"),
    ])
    def test_protect_chain_s_forms(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestT5Biguanide:
    """P-66.4.1.2.1.2 (BB 34298): condensed guanidines are named as diamides
    of imidodicarbonimidic acid; 'biguanide' no longer recommended."""

    def test_bare_biguanide_pin(self):
        assert name_compound("NC(=N)NC(=N)N") == "imidodicarbonimidic diamide"

    def test_substituted_keeps_rt_valid_general_name(self):
        # Substituted condensed guanidines await the N^n superscript
        # subsystem (Task 12 investigation). Until then the RT-valid
        # general name must NOT regress to unknown.
        assert name_compound("CCN=C(NC(N)=N)N(c1ccccc1)c1ccccc1") == \
            "N-carbamimidoyl-N''-ethyl-N',N'-diphenylguanidine"

    def test_protect_plain_guanidine(self):
        assert name_compound("NC(=N)N") == "guanidine"


@pytest.mark.unit
class TestT6AmidrazonePrefixes:
    """P-66.4.2.3.1 (BB 34490) + P-66.4.2.3.2 (BB 34498): chain-terminal
    amidrazone C stays IN the chain (hydrazinyl+imino / amino+hydrazinylidene);
    ring-attached keeps the full acyl prefix (hydrazinecarboximidoyl)."""

    @pytest.mark.parametrize("smiles,expected", [
        # BB 34494 verbatim PIN:
        ("N=C(NN)CC(=O)O", "3-hydrazinyl-3-iminopropanoic acid"),
        # P-66.4.2.3.2 chain-end pattern (OPSIN-verified):
        ("NC(=NN)CC(=O)O", "3-amino-3-hydrazinylidenepropanoic acid"),
        ("NN=C(N)CCC(=O)O", "4-amino-4-hydrazinylidenebutanoic acid"),
        # BB 34496 verbatim PIN (ring parent -> acyl prefix retained):
        ("NNC(=N)c1cccc(C(=O)O)c1", "3-(hydrazinecarboximidoyl)benzoic acid"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # PF-2 pass-D pins share the hydrazonamide FG machinery — protect:
        ("CC(=NN)NCCC(=O)O", "3-(ethanehydrazonamido)propanoic acid"),
        ("CC(=NN)N", "ethanehydrazonamide"),
        # AM-4 pass-D chain-amidine pins share _TERMINAL_C_FGS + the amidine
        # block — protect:
        ("CCN=C(CCC(=O)OC)N(C)C",
         "methyl 4-(dimethylamino)-4-(ethylimino)butanoate"),
    ])
    def test_protect_shared_machinery(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestT7ImidohydrazideFamily:
    """P-66.4.2.1 (BB 34430): the R-C(=NH)-NH-NH2 amidrazone tautomer takes
    the 'imidohydrazide'/'carboximidohydrazide' suffix; P-14.3.4.1 (BB 2877):
    no terminal locants. Family reps per WAVE2-BUILD-PLAN-ALL §2a."""

    @pytest.mark.parametrize("smiles,expected", [
        ("N=CNN", "methanimidohydrazide"),
        ("CC(=N)NN", "ethanimidohydrazide"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # family reps already healed elsewhere — protect:
        ("NC(=N)SSC(=N)N", "carbamimidic dithioperoxyanhydride"),
        ("NNS(=NN)c1ccccc1", "benzenesulfinohydrazonohydrazide"),
        ("CC(=NN)NN", "ethanehydrazonohydrazide"),
    ])
    def test_protect_family_reps(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestT8SulfinoSulfonoHydrazonamido:
    """P-66.4.2.3.5 (BB 34540 verbatim PIN) + P-66.4.3.2 (BB 34617):
    S(=N-NH2) N-attached branches take the e->o amide-name prefix."""

    @pytest.mark.parametrize("smiles,expected", [
        # BB 34540 verbatim:
        ("OC(=O)c1ccc(NS(=NN)c2ccccc2)cc1",
         "4-(benzenesulfinohydrazonamido)benzoic acid"),
        # sulfono analogue (OPSIN-verified during planning):
        ("O=S(=NN)(Nc1ccc(C(=O)O)cc1)c1ccccc1",
         "4-(benzenesulfonohydrazonamido)benzoic acid"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_protect_parent_direction(self):
        assert name_compound("NNS(=NN)c1ccccc1") == \
            "benzenesulfinohydrazonohydrazide"


@pytest.mark.unit
class TestT9ComplexPolyamines:
    """P-62.2.4.1.3 (BB 26375): senior parent DIAMINE retained; other amine
    N demoted into N-substituent branches; numeric N-locant tags."""

    @pytest.mark.parametrize("smiles,expected", [
        # BB verbatim PINs:
        ("NCCNCN", "N1-(aminomethyl)ethane-1,2-diamine"),
        ("CN(C)CCN(C)CCN",
         "N1-(2-aminoethyl)-N1,N2,N2-trimethylethane-1,2-diamine"),
        # same class, OPSIN-verified during planning:
        ("NCCNCCN", "N1-(2-aminoethyl)ethane-1,2-diamine"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # existing 2-N path must stay byte-identical (primed style):
        ("NCCN", "ethane-1,2-diamine"),
        ("CNCCN", "N-methylethane-1,2-diamine"),
    ])
    def test_protect_simple_diamines(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestT10Am2AcylChainSubstituents:
    """AM-2 ROOT-1 (P-66.1.7, BB 33576 verbatim): the T5b off-chain amide
    block must express acyl-chain substituents or decline."""

    def test_heals(self):
        assert name_compound("CN(CC(O)CO)C(=O)CN") == \
            "2-amino-N-(2,3-dihydroxypropyl)-N-methylacetamide"

    @pytest.mark.parametrize("smiles,expected", [
        # T5b's existing clean-coverage case must not regress — the plain
        # off-chain amide with NO acyl-chain substituents:
        ("CCCC(NC(C)=O)CC", "N-(hexan-3-yl)acetamide"),
    ])
    def test_protect_plain_off_chain(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestT11Am2PoolDiscard:
    """AM-2 ROOT-2: the correct amide candidate must win the pool; the
    polyfunctional double-express (amide N named twice) must not be
    emitted for ANY input (structure-wrong)."""

    def test_heals(self):
        assert name_compound("NCC(=O)N(C)C") == "2-amino-N,N-dimethylacetamide"

    @pytest.mark.parametrize("smiles,expected", [
        # neighbouring amide pins that exercise the same pool branch:
        ("CC(=O)N(C)C", "N,N-dimethylacetamide"),
        ("CC(=O)NC", "N-methylacetamide"),
    ])
    def test_protect_amide_pool(self, smiles, expected):
        assert name_compound(smiles) == expected
