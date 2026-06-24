"""Unit tests for the λ-convention mononuclear-hydride namer (v23 Phase 6).

P-68 / P-31.1.4.2: an all-halogen mononuclear hydride with a non-standard-valence
hub (chalcogen S/Se/Te, P, or I) -> ``<halo>-lambda<n>-<stem>``. Fail-closed:
every guard narrows; standard-valence hydrides, oxoacids, oxo-halides,
interhalogens, organo-substituted hubs, di-nuclear species, ions and rings all
decline (None) and cascade onward.
"""
import pytest
from rdkit import Chem

from orthonym.rules.mononuclear_hydrides import name_mononuclear_hydride


def _name(smiles):
    return name_mononuclear_hydride(Chem.MolFromSmiles(smiles))


class TestLambdaWins:
    @pytest.mark.parametrize("smiles,expected", [
        ("FS(F)(F)(F)(F)F", "hexafluoro-lambda6-sulfane"),    # SF6
        ("FS(F)(F)F", "tetrafluoro-lambda4-sulfane"),          # SF4
        ("FP(F)(F)(F)F", "pentafluoro-lambda5-phosphane"),     # PF5
        ("ClP(Cl)(Cl)(Cl)Cl", "pentachloro-lambda5-phosphane"),  # PCl5
        ("FI(F)(F)(F)F", "pentafluoro-lambda5-iodane"),        # IF5
        ("FI(F)F", "trifluoro-lambda3-iodane"),                # IF3
        ("F[Se](F)(F)(F)(F)F", "hexafluoro-lambda6-selane"),   # SeF6
    ])
    def test_known_lambda_hydrides(self, smiles, expected):
        assert _name(smiles) == expected

    def test_mixed_halogens_alphabetical(self):
        # SF5Cl: chloro (c) before fluoro (f); multiplier ignored for ordering.
        assert _name("FS(F)(F)(F)(F)Cl") == "chloropentafluoro-lambda6-sulfane"


class TestFailClosedDecline:
    @pytest.mark.parametrize("smiles,why", [
        ("ClP(Cl)Cl", "PCl3 standard trivalent P -> Phase 7/8 element hydride"),
        ("FP(F)F", "PF3 standard trivalent P"),
        ("FSF", "SF2 standard divalent S"),
        ("F[Si](F)(F)F", "SiF4: Si is not a hub element (Group-14, Phase 8)"),
        ("O=S(=O)(O)O", "sulfuric acid: O neighbours are non-halogen"),
        ("O=P(O)(O)O", "phosphoric acid"),
        ("O=S(Cl)Cl", "thionyl chloride SOCl2: has an O"),
        ("ClC(Cl)(Cl)Cl", "CCl4: carbon hub, not in set"),
        ("FC(F)(F)F", "CF4: carbon hub"),
        ("ClCl", "Cl2 diatomic: no degree>=2 hub"),
        ("ClI", "ICl interhalogen: no degree>=2 hub"),
        ("FS(F)(F)(F)(F)S(F)(F)(F)(F)F", "S2F10: two hubs (di-nuclear)"),
        ("C[S](C)(C)CC", "organo-substituted hub: non-halogen substituents"),
    ])
    def test_declines(self, smiles, why):
        mol = Chem.MolFromSmiles(smiles)
        # Some hypervalent halides RDKit refuses outright; treat as N/A.
        if mol is None:
            pytest.skip(f"RDKit rejects SMILES ({why})")
        assert _name(smiles) is None, why

    def test_charged_declines(self):
        assert _name("F[S+](F)(F)F") is None

    def test_none_input(self):
        assert name_mononuclear_hydride(None) is None
