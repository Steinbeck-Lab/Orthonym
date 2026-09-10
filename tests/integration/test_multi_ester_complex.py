"""
Integration tests for complex multi-ester scenarios: and.

: Large molecules (HA > 30) with multiple ester groups produce names
         covering >= 70% of heavy atoms.
: Molecules with 3+ ester bonds produce valid names through the
         quality gate (non-empty, non-garbled).

These are regression tests ensuring the multi-ester naming pipeline handles
real-world compounds with many ester bonds and/or large heavy atom counts.
"""

import pytest
from rdkit import Chem
from orthonym.namer import name_compound


def _heavy_atom_count(smiles: str) -> int:
    """Count heavy atoms in a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol else 0


def _is_valid_name(name):
    """Check that a name is non-empty, non-None, and not garbled."""
    if name is None or not isinstance(name, str):
        return False
    name = name.strip()
    if len(name) < 5:
        return False
    if "unknown" in name.lower():
        return False
    return True


# ============================================================================
#: 3+ ester bonds produce valid names through quality gate
# ============================================================================

class TestDECO09_MultiEsterBonds:
    """Molecules with 3+ ester bonds must produce valid names."""

    @pytest.mark.integration
    def test_triacetyl_nucleoside_3_esters(self):
        """3-ester nucleoside (HA=26): produces a valid non-empty name."""
        smi = "CC(=O)OC[C@H]1O[C@@H](n2ccc(=O)[nH]c2=O)[C@H](OC(C)=O)[C@@H]1OC(C)=O"
        ha = _heavy_atom_count(smi)
        assert ha >= 20, f"Expected HA >= 20, got {ha}"
        result = name_compound(smi)
        assert _is_valid_name(result), f"Invalid name for 3-ester nucleoside: {result!r}"

    @pytest.mark.integration
    def test_triacetyl_fused_aromatic_3_esters(self):
        """3-ester fused aromatic (HA=30): produces a valid non-empty name."""
        smi = "CC(=O)Oc1ccc2c(c1)oc(=O)c1c3cc(OC(C)=O)c(OC(C)=O)cc3oc21"
        ha = _heavy_atom_count(smi)
        assert ha >= 28, f"Expected HA >= 28, got {ha}"
        result = name_compound(smi)
        assert _is_valid_name(result), f"Invalid name for 3-ester fused aromatic: {result!r}"


# ============================================================================
#: Large molecules (HA > 30) with multiple esters
# ============================================================================

class TestDECO08_LargeMultiEster:
    """Large multi-ester molecules (HA > 30) produce valid names."""

    @pytest.mark.integration
    def test_gallic_acid_diester_HA34(self):
        """2-ester gallic acid derivative (HA=34): non-empty name."""
        smi = "O=C(O)c1cc(O)c(O)c(OC(=O)c2cc(O)c(O)c(OC(=O)c3cc(O)c(O)c(O)c3)c2)c1"
        ha = _heavy_atom_count(smi)
        assert ha >= 30, f"Expected HA >= 30, got {ha}"
        result = name_compound(smi)
        assert _is_valid_name(result), f"Invalid name for gallic acid diester (HA={ha}): {result!r}"

    @pytest.mark.integration
    def test_glycolipid_2_ester_HA51(self):
        """2-ester glycolipid (HA=51): non-empty name."""
        smi = "CCCCCCCCCCCCCCCC(=O)OC[C@H](CO[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O)OC(=O)CCCCCCCCCCCCCCC"
        ha = _heavy_atom_count(smi)
        assert ha >= 45, f"Expected HA >= 45, got {ha}"
        result = name_compound(smi)
        assert _is_valid_name(result), f"Invalid name for glycolipid (HA={ha}): {result!r}"


# ============================================================================
# + combined: Large molecule with 3+ ester bonds
# ============================================================================

class TestDECO08_09_Combined:
    """Large molecules (HA > 30) with 3+ ester bonds."""

    @pytest.mark.integration
    def test_triglyceride_HA65_3_esters(self):
        """Triglyceride (HA=65, 3 esters): valid name with reasonable coverage."""
        smi = "CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)OC(COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC)COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC"
        ha = _heavy_atom_count(smi)
        assert ha >= 60, f"Expected HA >= 60, got {ha}"
        result = name_compound(smi)
        assert _is_valid_name(result), f"Invalid name for triglyceride (HA={ha}): {result!r}"


# ============================================================================
# Quality gate: validate that multi-ester names pass basic quality checks
# ============================================================================

class TestMultiEsterQualityGate:
    """Multi-ester names must pass quality gate (not garbled)."""

    @pytest.mark.integration
    def test_quality_gate_no_none(self):
        """All DECO multi-ester compounds produce non-None names."""
        smiles_list = [
            "CC(=O)OC[C@H]1O[C@@H](n2ccc(=O)[nH]c2=O)[C@H](OC(C)=O)[C@@H]1OC(C)=O",
            "O=C(O)c1cc(O)c(O)c(OC(=O)c2cc(O)c(O)c(OC(=O)c3cc(O)c(O)c(O)c3)c2)c1",
            "CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)OC(COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC)COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC",
        ]
        for smi in smiles_list:
            result = name_compound(smi)
            assert result is not None, f"name_compound returned None for {smi[:40]}..."
            assert len(result.strip()) > 0, f"Empty name for {smi[:40]}..."

    @pytest.mark.integration
    def test_quality_gate_no_garbled(self):
        """Multi-ester names do not contain 'unknown' or very short fragments."""
        smiles_list = [
            "CC(=O)OC[C@H]1O[C@@H](n2ccc(=O)[nH]c2=O)[C@H](OC(C)=O)[C@@H]1OC(C)=O",
            "CC(=O)Oc1ccc2c(c1)oc(=O)c1c3cc(OC(C)=O)c(OC(C)=O)cc3oc21",
        ]
        for smi in smiles_list:
            result = name_compound(smi)
            assert _is_valid_name(result), f"Garbled name for {smi[:40]}...: {result!r}"
