"""C2 gap-fix tests — amidines (Blue Book, completes task 1.3).

Three sub-fixes, each OPSIN-verified:
  (a) aliphatic-chain amidine — SUPERSEDED by, BB 34338):
      a chain-terminal amidine C stays IN the chain and is cited as amino+imino
      ('4-amino-4-iminobutanoic acid'), NOT 'carbamimidoyl' (that prefix is the
      PIN only for RING / off-chain amidine carbons).
  (b) amidine-as-PCG on plain benzene emits the -carboximidamide SUFFIX
      (benzenecarboximidamide), not the carbamimidoyl PREFIX.
  (c) N-substituted amidine — N/N'-aware carbamimidoyl prefix
      (4-(N-methylcarbamimidoyl)benzoic acid). HIGHEST risk; ship only if
      provably safe (guanidine guard must hold, N/N' must be correct).

CRITICAL GUARDS (must stay correct after every step):
  NC(=N)N -> guanidine (SMARTS must NOT swallow guanidine)
  N=C(N)C1CCCCC1 -> cyclohexanecarboximidamide (non-aromatic ring path unchanged)
  CC(=N)N -> ethanimidamide (chain amidine-as-PCG unchanged)
  N=C(N)c1ccc(C(=O)O)cc1 -> 4-carbamimidoylbenzoic acid (already-working prefix case)
"""
import pytest
from orthonym.namer import name_compound


def _pin(smiles):
    return name_compound(smiles, style="pin")


# ---------------------------------------------------------------------------
# FIX (a) — aliphatic chain off-by-one (amidine C excluded from chain)
# ---------------------------------------------------------------------------
class TestAmidineChainOffByOne:
    #, BB 34338): the chain-terminal amidine carbon stays IN
    # the chain and is expressed as amino (-NH2) + imino (=NH), NOT the
    # 'carbamimidoyl' prefix. The new forms are OPSIN-RT canonical-equal to the
    # same SMILES (was '3-carbamimidoylpropanoic acid' etc. before).
    def test_4_amino_4_iminobutanoic_acid(self):
        assert _pin("N=C(N)CCC(=O)O") == "4-amino-4-iminobutanoic acid"

    def test_3_amino_3_iminopropanoic_acid(self):
        assert _pin("N=C(N)CC(=O)O") == "3-amino-3-iminopropanoic acid"

    def test_5_amino_5_iminopentanoic_acid(self):
        assert _pin("N=C(N)CCCC(=O)O") == "5-amino-5-iminopentanoic acid"

    def test_already_working_4_carbamimidoylbenzoic_acid(self):
        # benzene ring case already worked; must remain correct.
        assert _pin("N=C(N)c1ccc(C(=O)O)cc1") == "4-carbamimidoylbenzoic acid"


# ---------------------------------------------------------------------------
# FIX (b) — amidine-as-PCG on plain benzene emits -carboximidamide suffix
# ---------------------------------------------------------------------------
class TestBenzeneCarboximidamideSuffix:
    def test_benzenecarboximidamide(self):
        assert _pin("N=C(N)c1ccccc1") == "benzenecarboximidamide"

    def test_benzene_1_4_dicarboximidamide(self):
        assert _pin("N=C(N)c1ccc(C(=N)N)cc1") == "benzene-1,4-dicarboximidamide"

    def test_4_methylbenzene_1_carboximidamide(self):
        assert _pin("Cc1ccc(C(=N)N)cc1") == "4-methylbenzene-1-carboximidamide"


# ---------------------------------------------------------------------------
# FIX (c) — N-substituted amidine (N/N'-aware carbamimidoyl prefix)
# ---------------------------------------------------------------------------
# SHIPPED (D1): sub-fix (c) — the N-substituted-amidine subsystem — landed in D1
# (broadened SMARTS with the guanidine guard held + N/N' locant assignment pinned
# by C-N bond order, OPSIN round-trip verified per tautomer). Full coverage lives
# in tests/unit/rules/test_nsubst_amidine_d1.py; these two anchor cases stay here
# to document the C2->D1 continuity.
class TestNSubstitutedAmidine:
    def test_4_N_methylcarbamimidoyl_benzoic_acid(self):
        # CNC(=N)-: methyl on the single-bonded (amino) N = N-methyl
        assert _pin("CNC(=N)c1ccc(C(=O)O)cc1") == "4-(N-methylcarbamimidoyl)benzoic acid"

    def test_Nprime_methyl_tautomer_correct(self):
        # CN=C(N)-: methyl on the double-bonded (imino) N = N'-methyl.
        # Adversarial: getting N vs N' backwards flips the molecule.
        assert _pin("CN=C(N)c1ccc(C(=O)O)cc1") == "4-(N'-methylcarbamimidoyl)benzoic acid"


# ---------------------------------------------------------------------------
# CRITICAL GUARDS — must hold after every change
# ---------------------------------------------------------------------------
class TestAmidineGuards:
    def test_guanidine_not_swallowed(self):
        assert _pin("NC(=N)N") == "guanidine"

    def test_cyclohexanecarboximidamide(self):
        assert _pin("N=C(N)C1CCCCC1") == "cyclohexanecarboximidamide"

    def test_ethanimidamide_chain_amidine_pcg(self):
        assert _pin("CC(=N)N") == "ethanimidamide"

    def test_propanimidamide_chain_amidine_pcg(self):
        # amidine C stays in the chain when amidine IS the PCG (FIX(a) guard).
        assert _pin("CCC(=N)N") == "propanimidamide"
