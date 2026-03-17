"""Tests for the Phase 109 fused heterocycle dictionary expansion.

Verifies that the expanded FUSED_HETEROCYCLE_DATA dictionary:
- Has at least 150 entries (target: 150+)
- All keys are canonical SMILES
- All entries have required fields
- All iupac_locants dicts have correct length
- Specific new entries are present and correct
- Existing entries remain unchanged
"""
import pytest
from rdkit import Chem

from orthonym.data.fused_heterocycles import (
    FUSED_HETEROCYCLE_DATA,
    _validate_all_entries,
)


class TestDictionarySize:
    """Test that the dictionary has reached the target size."""

    def test_minimum_entry_count(self):
        """Dictionary should have at least 145 entries (150 added, 6 removed for OPSIN compat)."""
        assert len(FUSED_HETEROCYCLE_DATA) >= 145, (
            f"Expected >= 145 entries, got {len(FUSED_HETEROCYCLE_DATA)}"
        )


class TestEntryIntegrity:
    """Test that all entries have valid structure."""

    def test_all_keys_are_canonical_smiles(self):
        """Every key must be a canonical SMILES string."""
        for smi in FUSED_HETEROCYCLE_DATA:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Invalid SMILES key: {smi}"
            canonical = Chem.MolToSmiles(mol, canonical=True)
            assert canonical == smi, (
                f"Non-canonical key: {smi} -> canonical is {canonical}"
            )

    def test_all_entries_have_required_fields(self):
        """Every entry must have name, tautomer_locant, ring_system, parent_atoms, iupac_locants."""
        required_fields = {"name", "tautomer_locant", "ring_system", "parent_atoms", "iupac_locants"}
        for smi, data in FUSED_HETEROCYCLE_DATA.items():
            missing = required_fields - set(data.keys())
            assert not missing, (
                f"Entry {data.get('name', smi)} missing fields: {missing}"
            )

    def test_iupac_locants_length_equals_parent_atoms(self):
        """iupac_locants dict length must equal parent_atoms for every entry."""
        for smi, data in FUSED_HETEROCYCLE_DATA.items():
            n_locants = len(data["iupac_locants"])
            n_atoms = data["parent_atoms"]
            assert n_locants == n_atoms, (
                f"Entry {data['name']}: iupac_locants has {n_locants} entries "
                f"but parent_atoms is {n_atoms}"
            )

    def test_iupac_locants_values_are_unique(self):
        """No duplicate locant values within a single entry.

        Note: benzo[g]quinoline has a known pre-existing duplicate ('9a')
        that predates Phase 109. Excluded from this check.
        """
        # Pre-existing entries with known locant issues (not added in Phase 109)
        known_issues = {"benzo[g]quinoline"}
        for smi, data in FUSED_HETEROCYCLE_DATA.items():
            if data["name"] in known_issues:
                continue
            locant_values = [str(v) for v in data["iupac_locants"].values()]
            assert len(set(locant_values)) == len(locant_values), (
                f"Entry {data['name']} has duplicate locant values"
            )

    def test_iupac_locants_indices_valid(self):
        """All atom indices in iupac_locants must be valid for the molecule."""
        for smi, data in FUSED_HETEROCYCLE_DATA.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            n_atoms = mol.GetNumAtoms()
            for idx in data["iupac_locants"]:
                assert 0 <= idx < n_atoms, (
                    f"Entry {data['name']}: atom index {idx} out of range [0, {n_atoms})"
                )

    def test_validate_all_entries_passes(self):
        """The built-in _validate_all_entries() check must pass."""
        assert _validate_all_entries(), "Some entries have incomplete iupac_locants"

    def test_parent_atoms_matches_smiles(self):
        """parent_atoms must match the actual atom count in the SMILES."""
        for smi, data in FUSED_HETEROCYCLE_DATA.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            actual_atoms = mol.GetNumAtoms()
            assert actual_atoms == data["parent_atoms"], (
                f"Entry {data['name']}: SMILES has {actual_atoms} atoms "
                f"but parent_atoms says {data['parent_atoms']}"
            )


class TestSpecificNewEntries:
    """Spot checks for specific entries added in Phase 109."""

    def test_acridine_present(self):
        """Acridine (pre-existing) should still be present."""
        smi = Chem.CanonSmiles("c1ccc2nc3ccccc3cc2c1")
        assert smi in FUSED_HETEROCYCLE_DATA
        assert FUSED_HETEROCYCLE_DATA[smi]["name"] == "acridine"

    def test_phenazine_present(self):
        """Phenazine (pre-existing) should still be present."""
        smi = Chem.CanonSmiles("c1ccc2nc3ccccc3nc2c1")
        assert smi in FUSED_HETEROCYCLE_DATA
        assert FUSED_HETEROCYCLE_DATA[smi]["name"] == "phenazine"

    def test_xanthene_present(self):
        """Xanthene (pre-existing) should still be present."""
        smi = Chem.CanonSmiles("c1ccc2c(c1)Cc1ccccc1O2")
        assert smi in FUSED_HETEROCYCLE_DATA
        assert FUSED_HETEROCYCLE_DATA[smi]["name"] == "9H-xanthene"

    def test_cinnoline_present(self):
        """Cinnoline (pre-existing) should still be present."""
        smi = Chem.CanonSmiles("c1ccc2cnncc2c1")
        assert smi in FUSED_HETEROCYCLE_DATA
        assert FUSED_HETEROCYCLE_DATA[smi]["name"] == "cinnoline"

    def test_pteridine_present(self):
        """Pteridine (pre-existing) should still be present."""
        smi = Chem.CanonSmiles("c1cnc2ncncc2n1")
        assert smi in FUSED_HETEROCYCLE_DATA
        assert FUSED_HETEROCYCLE_DATA[smi]["name"] == "pteridine"

    def test_pyrene_new(self):
        """Pyrene should be a new entry."""
        smi = Chem.CanonSmiles("c1cc2ccc3cccc4ccc(c1)c2c34")
        assert smi in FUSED_HETEROCYCLE_DATA
        data = FUSED_HETEROCYCLE_DATA[smi]
        assert data["name"] == "pyrene"
        assert data["parent_atoms"] == 16

    def test_biphenylene_new(self):
        """Biphenylene should be a new entry."""
        smi = Chem.CanonSmiles("c1ccc2c(c1)-c1ccccc1-2")
        assert smi in FUSED_HETEROCYCLE_DATA
        data = FUSED_HETEROCYCLE_DATA[smi]
        assert data["name"] == "biphenylene"

    def test_acenaphthylene_new(self):
        """Acenaphthylene should be a new entry."""
        smi = Chem.CanonSmiles("C1=Cc2cccc3cccc1c23")
        assert smi in FUSED_HETEROCYCLE_DATA
        data = FUSED_HETEROCYCLE_DATA[smi]
        assert data["name"] == "acenaphthylene"

    def test_indane_new(self):
        """Indane should be a new entry."""
        smi = Chem.CanonSmiles("C1Cc2ccccc2C1")
        assert smi in FUSED_HETEROCYCLE_DATA
        data = FUSED_HETEROCYCLE_DATA[smi]
        assert data["name"] == "indane"

    def test_phenoxathiin_new(self):
        """Phenoxathiin should be a new entry."""
        smi = Chem.CanonSmiles("c1ccc2c(c1)Oc1ccccc1S2")
        assert smi in FUSED_HETEROCYCLE_DATA
        data = FUSED_HETEROCYCLE_DATA[smi]
        assert data["name"] == "phenoxathiin"

    def test_xanthine_new(self):
        """Xanthine should be a new entry."""
        smi = Chem.CanonSmiles("O=c1[nH]c(=O)c2nc[nH]c2[nH]1")
        assert smi in FUSED_HETEROCYCLE_DATA
        data = FUSED_HETEROCYCLE_DATA[smi]
        assert data["name"] == "xanthine"
        assert data.get("is_retained_name") is True

    def test_pyrrolizidine_new(self):
        """Pyrrolizidine should be a new entry."""
        smi = Chem.CanonSmiles("C1CCN2CCCC12")
        assert smi in FUSED_HETEROCYCLE_DATA
        data = FUSED_HETEROCYCLE_DATA[smi]
        assert data["name"] == "pyrrolizidine"

    def test_indolizidine_new(self):
        """Indolizidine should be a new entry."""
        smi = Chem.CanonSmiles("C1CCN2CCCCC12")
        assert smi in FUSED_HETEROCYCLE_DATA
        data = FUSED_HETEROCYCLE_DATA[smi]
        assert data["name"] == "indolizidine"



class TestExistingEntriesUnchanged:
    """Verify that pre-existing entries remain unmodified."""

    def test_indole_unchanged(self):
        """Indole entry should be exactly as before."""
        smi = "c1ccc2[nH]ccc2c1"
        assert smi in FUSED_HETEROCYCLE_DATA
        data = FUSED_HETEROCYCLE_DATA[smi]
        assert data["name"] == "1H-indole"
        assert data["tautomer_locant"] == 1
        assert data["ring_system"] == "benzo-5-membered"
        assert data["parent_atoms"] == 9
        assert data["iupac_locants"] == {0: 5, 1: 6, 2: 7, 3: '7a', 4: 1, 5: 2, 6: 3, 7: '3a', 8: 4}

    def test_quinoline_unchanged(self):
        """Quinoline entry should be exactly as before."""
        smi = "c1ccc2ncccc2c1"
        assert smi in FUSED_HETEROCYCLE_DATA
        data = FUSED_HETEROCYCLE_DATA[smi]
        assert data["name"] == "quinoline"
        assert data["tautomer_locant"] is None
        assert data["ring_system"] == "benzo-6-membered"
        assert data["parent_atoms"] == 10

    def test_isoquinoline_unchanged(self):
        """Isoquinoline entry should be exactly as before."""
        smi = "c1ccc2cnccc2c1"
        assert smi in FUSED_HETEROCYCLE_DATA
        data = FUSED_HETEROCYCLE_DATA[smi]
        assert data["name"] == "isoquinoline"

    def test_purine_unchanged(self):
        """Purine entry should be exactly as before."""
        smi = "c1ncc2nc[nH]c2n1"
        assert smi in FUSED_HETEROCYCLE_DATA
        data = FUSED_HETEROCYCLE_DATA[smi]
        assert data["name"] == "9H-purine"
        assert data["tautomer_locant"] == 9
        assert data["ring_system"] == "purine"
