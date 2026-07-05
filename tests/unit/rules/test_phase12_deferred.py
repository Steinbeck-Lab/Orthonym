"""v23 Phase 12 follow-on — the three deferred items resolved (F3 / F2 / F1).

  - F3 ureido : -NH-C(=O)-NH2 substituent was double-prefixed
                (carbamoylamino + phantom methanoylamino). Now a single
                (carbamoylamino). Fixes every ureido-acid, not just citrulline.
  - F2 cystine: L-cystine -> 'cystine' (was always fine); D-cystine added.
  - F1 inositol: the 7 meso inositols get their P-104.2.1 retained PIN
                (name-exact, OPSIN-unparseable). The chiral D/L-chiro pair is
                DETERMINISTICALLY refused (RDKit perceives it non-deterministically).
"""

import pytest

from rdkit import Chem

from orthonym.namer import name_compound


class TestUreido:
    """F3: ureido substituent must be a single (carbamoylamino), not split."""

    @pytest.mark.parametrize("smiles,expected", [
        ("NC(=O)NCCC[C@H](N)C(=O)O", "(2S)-2-amino-5-(carbamoylamino)pentanoic acid"),  # citrulline
        ("NC(=O)NCCC(=O)O", "3-(carbamoylamino)propanoic acid"),
        ("NC(=O)NCCCC(=O)O", "4-(carbamoylamino)butanoic acid"),
    ])
    def test_ureido_single_prefix(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("O=CNCCCC(=O)O", "4-formamidobutanoic acid"),    # real formamido — P-66.1.1.4.3(1)
        ("CC(=O)NCCC(=O)O", "3-acetamidopropanoic acid"),  # real acetamido — P-66.1.1.4.3(1)
        ("NCCC(=O)O", "3-aminopropanoic acid"),            # plain amino — unaffected
        ("CNC(N)=O", "N-methylurea"),                      # terminal urea — unaffected
    ])
    def test_acylamino_regressions(self, smiles, expected):
        assert name_compound(smiles) == expected


class TestCystine:
    """F2: cystine disulfide-dimer amino acid."""

    @pytest.mark.parametrize("smiles,expected", [
        ("N[C@@H](CSSC[C@H](N)C(=O)O)C(=O)O", "cystine"),     # L-cystine (2R,2'R)
        ("N[C@H](CSSC[C@@H](N)C(=O)O)C(=O)O", "D-cystine"),   # D-cystine (2S,2'S)
    ])
    def test_cystine(self, smiles, expected):
        assert name_compound(smiles) == expected

    # NOTE: meso-cystine (2R,2'S) is fail-closed to 'unknown' in PRODUCTION (its
    # systematic name does not OPSIN-round-trip -> the validity gate suppresses
    # it). The unit suite runs with that gate DISABLED (conftest autouse), so the
    # raw systematic name is emitted here; the fail-closed behaviour is covered by
    # the gate-on PIN-conformance run, not asserted here.


class TestInositol:
    """F1: the seven meso inositols (P-104.2.1); chiro deterministically refused."""

    # InChIKey -> (representative SMILES, retained name) for the 7 meso inositols.
    MESO = {
        "myo-inositol":    "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O",
        "scyllo-inositol": "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
        "muco-inositol":   "O[C@H]1[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1O",
        "epi-inositol":    "O[C@H]1[C@@H](O)[C@@H](O)[C@@H](O)[C@@H](O)[C@@H]1O",
        "allo-inositol":   "O[C@H]1[C@H](O)[C@H](O)[C@@H](O)[C@@H](O)[C@H]1O",
        "cis-inositol":    "O[C@H]1[C@@H](O)[C@@H](O)[C@@H](O)[C@@H](O)[C@H]1O",
        "neo-inositol":    "O[C@H]1[C@H](O)[C@H](O)[C@H](O)[C@@H](O)[C@H]1O",
    }

    @pytest.mark.parametrize("name,smiles", list(MESO.items()))
    def test_meso_inositol(self, name, smiles):
        assert name_compound(smiles) == name

    @pytest.mark.parametrize("name,smiles", list(MESO.items()))
    def test_meso_inositol_deterministic(self, name, smiles):
        m = Chem.MolFromSmiles(smiles)
        out = {name_compound(Chem.MolToSmiles(m, doRandom=True)) for _ in range(8)}
        assert out == {name}, f"{name} non-deterministic: {out}"

    @pytest.mark.parametrize("smiles", [
        "O[C@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O",   # chiro (…-LKPKBOIGSA-N)
        "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@H](O)[C@H]1O",   # chiro enantiomer (…-SHFUYGGZSA-N)
    ])
    def test_chiro_deterministic_unknown(self, smiles):
        # The chiral chiro pair is refused to a DETERMINISTIC unknown (RDKit
        # perceives its absolute config non-deterministically).
        m = Chem.MolFromSmiles(smiles)
        out = {name_compound(Chem.MolToSmiles(m, doRandom=True)) for _ in range(10)}
        assert len(out) == 1, f"chiro not deterministic: {out}"
        assert out.pop().startswith("unknown")

    def test_undefined_stereo_hexol_keeps_systematic(self):
        # Fail-closed: an undefined-stereo cyclohexanehexol is NOT an inositol match.
        assert name_compound("OC1C(O)C(O)C(O)C(O)C1O") == "cyclohexane-1,2,3,4,5,6-hexol"
