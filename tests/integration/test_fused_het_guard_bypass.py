"""Tests for fused heterocycle guard bypass (Phase 70, IUPAC P-44.1.1 / P-52.2.8).

The guard bypass in namer.py allows parent_selection to run for known fused
heterocycles when the chain has strictly more principal groups than the ring.
Two additional guards prevent over-broad bypass:
  1. No substantial additional ring systems beyond the matched core
  2. Non-ring heavy atoms must exceed core atom count (chain is longer)
"""
import pytest
from orthonym import name_compound
from orthonym.namer import _should_bypass_fused_guard, MolecularFeatures
from rdkit import Chem
from orthonym.data.fused_heterocycles import match_fused_heterocycle_core
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.seniority import get_principal_group


def _check_bypass(smiles):
    """Helper: returns (bypass_result, core_name) for a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    fgs = detect_functional_groups(mol)
    pg_name, pg_atoms = get_principal_group(mol, fgs)

    features = MolecularFeatures(mol=mol)
    features.principal_group = pg_name
    features.principal_group_atoms = pg_atoms
    features.ring_systems = [set(r) for r in mol.GetRingInfo().AtomRings()]

    core_match = match_fused_heterocycle_core(mol)
    if core_match is None:
        return False, None
    return _should_bypass_fused_guard(features, core_match), core_match[0]


class TestGuardBypassLogic:
    """Direct tests of _should_bypass_fused_guard() function."""

    def test_long_chain_ester_quinoline_bypasses(self):
        """Long chain (13 non-ring HA) with ester PG → bypass."""
        bypass, core = _check_bypass("CCCCCC(C)OC(=O)COc1ccc(Cl)c2cccnc12")
        assert bypass is True
        assert core == "quinoline"

    def test_long_chain_amide_indole_bypasses(self):
        """Very long chain (30 non-ring HA) with amide PG → bypass."""
        bypass, core = _check_bypass(
            "CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)NCCc1c[nH]c2ccccc12"
        )
        assert bypass is True
        assert core == "1H-indole"

    def test_bare_quinoline_no_bypass(self):
        """No PG at all → ring stays parent."""
        bypass, core = _check_bypass("c1ccc2ncccc2c1")
        assert bypass is False

    def test_bare_indole_no_bypass(self):
        """No PG at all → ring stays parent."""
        bypass, core = _check_bypass("c1ccc2[nH]ccc2c1")
        assert bypass is False

    def test_short_chain_nitrile_no_bypass(self):
        """Short -CH2CN chain on indole → chain too short, no bypass."""
        bypass, core = _check_bypass("N#CCc1c[nH]c2ccccc12")
        assert bypass is False

    def test_pg_on_ring_no_bypass(self):
        """PG directly on ring → ring_pg >= chain_pg, no bypass."""
        bypass, core = _check_bypass("OC(=O)c1cnc2ccccc2c1")
        assert bypass is False

    def test_equal_pg_ring_wins(self):
        """Equal PG count → ring wins per P-52.2.8, no bypass."""
        # Quinoline with -COOH on ring AND -COOH on chain
        bypass, core = _check_bypass("OC(=O)c1cc2ccccc2nc1CCCC(=O)O")
        # Even if chain PG = ring PG = 1, bypass should not trigger
        assert bypass is False

    def test_complex_polycyclic_no_bypass(self):
        """Large polycyclic molecule with additional ring systems → no bypass."""
        smiles = "CC1(C)C=Cc2c(cc(O)c3c(=O)c4ccc(O[C@@H]5c6c(cc(O)c7c(=O)c8cccc(O)c8oc67)O[C@H]5C(C)(C)O)c(O)c4oc23)O1"
        bypass, core = _check_bypass(smiles)
        assert bypass is False


class TestRetainedNamesPreserved:
    """Guard bypass does not break retained fused heterocycle names."""

    @pytest.mark.parametrize("smiles,expected", [
        ("c1ccc2ncccc2c1", "quinoline"),
        ("c1ccc2[nH]ccc2c1", "1H-indole"),
        ("CCc1nc2ccccc2[nH]1", "2-ethyl-1H-benzimidazole"),
        ("Nc1ccc2ncccc2c1", "quinolin-6-amine"),
        ("OC(=O)c1cnc2ccccc2c1", "quinoline-3-carboxylic acid"),
    ])
    def test_retained_name_preserved(self, smiles, expected):
        """Known fused heterocycle names must be preserved."""
        result = name_compound(smiles)
        assert result == expected, f"Expected {expected}, got {result}"


class TestTokenFilterExpansion:
    """DROP-04 token filter accepts fused heterocycle name stems."""

    def test_indole_prefix_not_dropped(self):
        """'indol' token should pass DROP-04 validation."""
        # Indole as substituent on ring-as-parent: ring naming path
        name = name_compound("N#CCc1c[nH]c2ccccc12")
        assert "indol" in name.lower(), f"Expected indol in name, got: {name}"

    def test_quinoline_prefix_not_dropped(self):
        """'quinolin' token should pass DROP-04 validation."""
        name = name_compound("Clc1ccc2ncccc2c1")
        assert "quinolin" in name.lower(), f"Expected quinolin in name, got: {name}"
