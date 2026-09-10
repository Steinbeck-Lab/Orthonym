"""D1 gap-fix tests — N-substituted amidines with N/N' locants (Blue Book
 / /, completes C2 sub-fix (c)).

D1 EXTENDS the C2 machinery (carbamimidoyl prefix, benzenecarboximidamide
suffix, chain-count exclusion — all for UNsubstituted amidines) to
N-/N'-substituted amidines:

  N = the amino (single-bonded, -NH-) nitrogen (OPSIN label /N1)
  N' = the imino (double-bonded, =N-) nitrogen (OPSIN label /N2)

All expected PINs are OPSIN-verified (style='pin'), and the N vs N' assignment
is correctness-critical: both tautomers round-trip through OPSIN standalone, so
each produced name must denote the SPECIFIC input tautomer.

CRITICAL GUARDS (broadening the amidine SMARTS to [CX3](=[NX2])[NX3] overlaps
the guanidine central carbon):
  NC(=N)N -> guanidine (guanidine guard MUST suppress the amidine read)
  NC(=O)N -> urea (=O not =N; no risk, asserted)
  CNC(=NC)N(C)C -> a guanidine (3 N on the C): must NOT be a wrong amidine name
"""
import pytest
from orthonym.namer import name_compound


def _pin(smiles):
    return name_compound(smiles, style="pin")


# ---------------------------------------------------------------------------
# ACCEPTANCE — N-substituted amidine on a benzoic acid ring (demotion path)
# ---------------------------------------------------------------------------
class TestBenzoicAcidCarbamimidoylPrefix:
    def test_N_methyl_amino_N(self):
        # CNC(=N)-: methyl on the single-bonded (amino) N = N-methyl
        assert _pin("CNC(=N)c1ccc(C(=O)O)cc1") == "4-(N-methylcarbamimidoyl)benzoic acid"

    def test_Nprime_methyl_imino_N(self):
        # CN=C(N)-: methyl on the double-bonded (imino) N = N'-methyl
        assert _pin("CN=C(N)c1ccc(C(=O)O)cc1") == "4-(N'-methylcarbamimidoyl)benzoic acid"


# ---------------------------------------------------------------------------
# ACCEPTANCE — N-substituted amidine as the benzene ring principal group
# ---------------------------------------------------------------------------
class TestBenzeneCarboximidamideSuffix:
    def test_N_methylbenzenecarboximidamide(self):
        assert _pin("CNC(=N)c1ccccc1") == "N-methylbenzenecarboximidamide"

    def test_Nprime_methylbenzenecarboximidamide(self):
        assert _pin("CN=C(N)c1ccccc1") == "N'-methylbenzenecarboximidamide"

    def test_N_Nprime_dimethylbenzenecarboximidamide(self):
        assert _pin("CNC(=NC)c1ccccc1") == "N,N'-dimethylbenzenecarboximidamide"


# ---------------------------------------------------------------------------
# ACCEPTANCE — acyclic N-substituted amidine (imidamide suffix)
# ---------------------------------------------------------------------------
class TestAcyclicImidamide:
    def test_N_methylethanimidamide(self):
        assert _pin("CNC(=N)C") == "N-methylethanimidamide"

    def test_Nprime_methylethanimidamide(self):
        assert _pin("CN=C(N)C") == "N'-methylethanimidamide"

    def test_N_Nprime_dimethylethanimidamide(self):
        assert _pin("CNC(C)=NC") == "N,N'-dimethylethanimidamide"

    def test_N_N_dimethylethanimidamide(self):
        # both substituents on the SAME (amino) nitrogen -> N,N-dimethyl
        assert _pin("CN(C)C(C)=N") == "N,N-dimethylethanimidamide"


# ---------------------------------------------------------------------------
# GUARDS — must hold after every change (guanidine is non-negotiable)
# ---------------------------------------------------------------------------
class TestD1Guards:
    def test_guanidine_not_swallowed(self):
        assert _pin("NC(=N)N") == "guanidine"

    def test_urea_not_matched(self):
        assert _pin("NC(=O)N") == "urea"

    def test_three_nitrogen_carbon_not_wrong_amidine(self):
        # CNC(=NC)N(C)C: 3 N on the amidine-candidate C -> guanidine class.
        # MUST NOT be named as an amidine. Accept any guanidine name or unknown,
        # but never a (wrong) amidine name.
        result = _pin("CNC(=NC)N(C)C")
        assert "imidamide" not in result and "carbamimidoyl" not in result

    # C2 unsubstituted cases must stay working after the SMARTS broadening.
    def test_ethanimidamide_unchanged(self):
        assert _pin("CC(=N)N") == "ethanimidamide"

    def test_4_carbamimidoylbenzoic_acid_unchanged(self):
        assert _pin("NC(=N)c1ccc(C(=O)O)cc1") == "4-carbamimidoylbenzoic acid"

    def test_benzenecarboximidamide_unchanged(self):
        assert _pin("NC(=N)c1ccccc1") == "benzenecarboximidamide"

    def test_cyclohexanecarboximidamide_unchanged(self):
        assert _pin("N=C(N)C1CCCCC1") == "cyclohexanecarboximidamide"

    def test_propanimidamide_unchanged(self):
        assert _pin("CCC(=N)N") == "propanimidamide"

    # Adversarial: the broadened [NX3] must NOT catch imidates (C=N with O).
    def test_imidate_not_matched(self):
        # COC(=N)C is methyl ethanimidate; must not become an amidine name.
        result = _pin("COC(=N)C")
        assert "imidamide" not in result and "carbamimidoyl" not in result


# ---------------------------------------------------------------------------
# a phase task 2: N-substituent CITATION ORDER is alphanumerical by
# NAME, NOT by italic-N locant. The italic-N locant is only a tie-break
# (the Blue Book dicarboximidamide PIN 'N''1-ethyl-N1,N1-dimethyl...' cites ethyl
# before dimethyl across N''1 > N1). All rows are OPSIN-RT gold PINs.
# ---------------------------------------------------------------------------
class TestAmidineCitationOrderAlphanumerical:
    def test_Nprime_ethyl_N_methyl_benzenecarboximidamide(self):
        # ethyl(N') < methyl(N) -> ethyl cited first despite the higher locant.
        assert _pin("CCN=C(NC)c1ccccc1") == "N'-ethyl-N-methylbenzenecarboximidamide"

    def test_Nprime_methyl_NN_diphenyl_benzenecarboximidamide(self):
        # methyl(N') < phenyl(N,N) -> methyl cited first.
        assert (_pin("CN=C(c1ccccc1)N(c1ccccc1)c1ccccc1")
                == "N'-methyl-N,N-diphenylbenzenecarboximidamide")

    def test_Nprime_hydroxy_N_methyl_ethanimidamide(self):
        # amidoxime read: hydroxy(N') < methyl(N).
        assert _pin("CNC(C)=NO") == "N'-hydroxy-N-methylethanimidamide"

    def test_Nprime_ethyl_NN_dimethyl_carbamimidoyl_benzoic_acid(self):
        # carbamimidoyl prefix form: ethyl(N') < methyl(N,N).
        assert (_pin("CCN=C(c1ccc(C(=O)O)cc1)N(C)C")
                == "4-(N'-ethyl-N,N-dimethylcarbamimidoyl)benzoic acid")

    # Regression: identical substituents keep the N-before-N' locant order.
    def test_NNprime_dimethyl_regression(self):
        assert _pin("CNC(=NC)c1ccccc1") == "N,N'-dimethylbenzenecarboximidamide"
