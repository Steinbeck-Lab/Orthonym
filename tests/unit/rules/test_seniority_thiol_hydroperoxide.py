"""
Tests verifying DATA-01: hydroperoxide outranks thiol in IUPAC P-43 Table 5.1 seniority.

Per IUPAC 2013 Blue Book P-43, hydroperoxide (Class 19) has higher seniority than
thiol (Class 20). This means for molecules containing both -SH and -OOH, the
principal characteristic group should be hydroperoxide, not thiol.
"""
import pytest
from rdkit import Chem

from orthonym.rules.seniority import SENIORITY_ORDER, get_principal_group
from orthonym.perception.functional_groups import detect_functional_groups


class TestHydroperoxideThiolSeniority:
    """Verify hydroperoxide is ranked above thiol in SENIORITY_ORDER."""

    def test_hydroperoxide_before_thiol_in_order(self):
        """Hydroperoxide must have a lower index (higher seniority) than thiol."""
        hp_idx = SENIORITY_ORDER.index("hydroperoxide")
        th_idx = SENIORITY_ORDER.index("thiol")
        assert hp_idx < th_idx, (
            f"hydroperoxide (idx={hp_idx}) should come before thiol (idx={th_idx})"
        )

    def test_hydroperoxide_before_selenol_in_order(self):
        """Hydroperoxide must also outrank selenol (which is below thiol)."""
        hp_idx = SENIORITY_ORDER.index("hydroperoxide")
        se_idx = SENIORITY_ORDER.index("selenol")
        assert hp_idx < se_idx, (
            f"hydroperoxide (idx={hp_idx}) should come before selenol (idx={se_idx})"
        )

    def test_combined_sh_ooh_selects_hydroperoxide(self):
        """For a molecule with both -SH and -OOH, principal group = hydroperoxide."""
        # 2-(hydroperoxy)ethanethiol: OOCCS
        smiles = "OOCCS"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Failed to parse SMILES: {smiles}"
        fgs = detect_functional_groups(mol)
        principal, matches = get_principal_group(mol, fgs)
        assert principal == "hydroperoxide", (
            f"Expected 'hydroperoxide' as principal group, got '{principal}'. "
            f"Detected FGs: {list(fgs.keys())}"
        )

    def test_pure_thiol_still_selects_thiol(self):
        """Ethanethiol (CCS) should still select thiol as principal group."""
        smiles = "CCS"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        fgs = detect_functional_groups(mol)
        principal, matches = get_principal_group(mol, fgs)
        assert principal == "thiol", (
            f"Expected 'thiol' as principal group, got '{principal}'"
        )

    def test_pure_hydroperoxide_selects_hydroperoxide(self):
        """Ethyl hydroperoxide (CCOO) should select hydroperoxide as principal group."""
        smiles = "CCOO"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        fgs = detect_functional_groups(mol)
        principal, matches = get_principal_group(mol, fgs)
        assert principal == "hydroperoxide", (
            f"Expected 'hydroperoxide' as principal group, got '{principal}'"
        )
