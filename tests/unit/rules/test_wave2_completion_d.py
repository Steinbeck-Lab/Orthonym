"""Wave-2 completion pass D — AM/PF amide/amidine/cyanamide investigation rows.

All expected PINs OPSIN-RT verified. Covers:
  * AM-6 (P-66.1.1.4.3): R-SO2-NH- on a chain -> '(...sulfonamido)' prefix
    (composer._check_for_acylamino sulfonyl recognizer + polyfunctional
    N-attach FG-prefix skip). S-attached 'sulfamoyl' orientation preserved.
  * AM-4 (P-66.4.1.3.2): a chain-terminal amidine carbon stays in the chain and
    is cited amino+imino, NOT 'carbamimidoyl' (ring/off-chain keeps carbamimidoyl).
  * AM-1 (P-66.1.6.2): NEW cyanamide retained-name subsystem (FG + collision
    suppression + composer._try_name_cyanamide + Tier-B handler).
  * AM-5 (P-66.4.1.6): conjoined diamidine -> N-imidoyl substituent on the other.
  * PF-2 (P-66.4.2.3.5): N-attached amidrazone -> 'hydrazonamido' prefix.

AM-2 (2-amino-N-(2,3-dihydroxypropyl)-N-methylacetamide) is DEFERRED — two
central root causes (amide T5b off-chain acyl-substituent drop + a pool discard),
documented in  It stays fail-closed.
"""

import pytest

from orthonym.namer import name_compound


@pytest.fixture()
def _validity_gate_on(monkeypatch):
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    yield


@pytest.mark.unit
class TestAM6Sulfonamido:
    @pytest.mark.parametrize("smiles,expected", [
        ("CS(=O)(=O)NCCC(=O)O", "3-(methanesulfonamido)propanoic acid"),
        ("O=S(=O)(c1ccccc1)NCCC(=O)O", "3-(benzenesulfonamido)propanoic acid"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # S-attached orientation keeps the 'sulfamoyl' FG prefix (unchanged).
        ("O=S(=O)(N)CCC(=O)O", "3-sulfamoylpropanoic acid"),
        # sulfonamide-as-parent unchanged.
        ("CCS(=O)(=O)N", "ethanesulfonamide"),
    ])
    def test_protected(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_n_alkyl_sulfonamido_fails_closed(self, _validity_gate_on):
        # N-alkylated sulfonamido is out of scope -> fail closed (unknown).
        assert "unknown" in name_compound("CS(=O)(=O)N(C)CCC(=O)O")


@pytest.mark.unit
class TestAM4AmidineOnChain:
    @pytest.mark.parametrize("smiles,expected", [
        ("CCN=C(CCC(=O)OC)N(C)C",
         "methyl 4-(dimethylamino)-4-(ethylimino)butanoate"),
        ("CCN=C(CCC(=O)O)N(C)C",
         "4-(dimethylamino)-4-(ethylimino)butanoic acid"),
        ("N=C(N)CCC(=O)O", "4-amino-4-iminobutanoic acid"),
        ("N=C(N)CC(=O)O", "3-amino-3-iminopropanoic acid"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # RING amidine keeps 'carbamimidoyl' (PIN for ring/off-chain, BB 34332).
        ("N=C(N)c1ccc(C(=O)O)cc1", "4-carbamimidoylbenzoic acid"),
        # amidine-as-principal suffix unchanged.
        ("CC(N)=N", "ethanimidamide"),
    ])
    def test_protected(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestAM1Cyanamide:
    @pytest.mark.parametrize("smiles,expected", [
        ("CC(C)NC#N", "(propan-2-yl)cyanamide"),
        ("CNC#N", "methylcyanamide"),
        ("CN(C)C#N", "dimethylcyanamide"),
        ("CCN(CC)C#N", "diethylcyanamide"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("NC#N", "cyanamide"),  # bare retained name unchanged
        # ring N-C#N stays a ring carbonitrile (SMARTS ;!R guard).
        ("N#CN1CCCCC1", "piperidine-1-carbonitrile"),
        # S-C#N (no N-C) untouched by the N-anchored SMARTS.
        ("CS(=O)(=O)C#N", "1-(methanesulfonyl)methanenitrile"),
    ])
    def test_protected(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestAM5ConjoinedAmidine:
    @pytest.mark.parametrize("smiles,expected", [
        ("CC(=N)NC(C)=N", "N-ethanimidoylethanimidamide"),
        ("CC(=N)NC(=N)c1ccccc1", "N-ethanimidoylbenzenecarboximidamide"),
    ])
    def test_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # non-conjoined diamidine (no shared N) stays on the general path.
        ("NC(=N)CCC(=N)N", "butanediimidamide"),
        # single-N amidines / amidoxime unchanged.
        ("CC(=N)NC", "N-methylethanimidamide"),
        ("CC(=NO)N", "N'-hydroxyethanimidamide"),
        ("NC(=N)c1ccccc1", "benzenecarboximidamide"),
    ])
    def test_protected(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestPF2Hydrazonamido:
    def test_heals(self):
        assert name_compound("CC(=NN)NCCC(=O)O") == \
            "3-(ethanehydrazonamido)propanoic acid"

    def test_hydrazonamide_parent_unchanged(self):
        assert name_compound("CC(=NN)N") == "ethanehydrazonamide"


@pytest.mark.unit
class TestFailClosed:
    @pytest.mark.parametrize("smiles", [
        "O=C(O)CNC#N",     # cyanamide + senior COOH coexistence (AM-1 out of scope)
        # NB: "NN=C(N)CCC(=O)O" was pass-D-fail-closed; plan P1AM Task 6
        # (P-66.4.2.3.2) now HEALS it to '4-amino-4-hydrazinylidenebutanoic
        # acid' (chain-terminal amidrazone amino/hydrazinylidene split) — it
        # is a W2E-P1AM Task-6 heal, no longer a fail-closed case.
        "CC(=N)NCCC(=O)O",  # N-attached amidine sibling (separate row)
    ])
    def test_stays_unknown(self, _validity_gate_on, smiles):
        assert "unknown" in name_compound(smiles)
