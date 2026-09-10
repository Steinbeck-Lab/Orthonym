"""
Tests verifying the thiol-vs-hydroperoxide seniority (BBR-HYG/, a phase).

CORRECTION: the prior version of this file asserted the INVERSE (hydroperoxide
outranks thiol, mis-citing " Class 19/20"). That was the audit's
 inversion bug. Per IUPAC 2013 Blue Book (the Blue Book lines
~18190-18191, verbatim):

    17 Hydroxy compounds and chalcogen analogues (alcohols, phenols, -ol, -thiol,
        -selenol, -tellurol)
    18 Hydroperoxides (peroxols), i.e. -OOH

so class 17 (hydroxy + thiol/selenol/tellurol) is SENIOR to class 18 (hydroperoxide).
The 169.7 swap moved hydroperoxide BELOW the chalcogen-ols; these tests now assert
the corrected order.
"""
import pytest
from rdkit import Chem

from orthonym.rules.seniority import SENIORITY_ORDER, get_principal_group
from orthonym.perception.functional_groups import detect_functional_groups


class TestHydroperoxideThiolSeniority:
    """Verify thiol/selenol (class 17) are ranked ABOVE hydroperoxide (class 18)."""

    def test_thiol_before_hydroperoxide_in_order(self):
        """Thiol (class 17) must have a lower index (higher seniority) than
        hydroperoxide (class 18) — P-41 Table 4.1."""
        th_idx = SENIORITY_ORDER.index("thiol")
        hp_idx = SENIORITY_ORDER.index("hydroperoxide")
        assert th_idx < hp_idx, (
            f"thiol (idx={th_idx}) should come before hydroperoxide (idx={hp_idx})"
        )

    def test_selenol_before_hydroperoxide_in_order(self):
        """Selenol (class 17 chalcogen analogue) also outranks hydroperoxide."""
        se_idx = SENIORITY_ORDER.index("selenol")
        hp_idx = SENIORITY_ORDER.index("hydroperoxide")
        assert se_idx < hp_idx, (
            f"selenol (idx={se_idx}) should come before hydroperoxide (idx={hp_idx})"
        )

    def test_combined_sh_ooh_selects_thiol(self):
        """For a molecule with both -SH and -OOH, the principal group is THIOL
        (class 17), with -OOH expressed as the hydroperoxy prefix."""
        smiles = "OOCCS"  # 2-hydroperoxyethane-1-thiol
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Failed to parse SMILES: {smiles}"
        fgs = detect_functional_groups(mol)
        principal, matches = get_principal_group(mol, fgs)
        assert principal == "thiol", (
            f"Expected 'thiol' (class 17, senior to hydroperoxide class 18) as "
            f"principal group, got '{principal}'. Detected FGs: {list(fgs.keys())}"
        )

    def test_pure_thiol_still_selects_thiol(self):
        """Ethanethiol (CCS) should still select thiol as principal group."""
        fgs = detect_functional_groups(Chem.MolFromSmiles("CCS"))
        principal, _ = get_principal_group(Chem.MolFromSmiles("CCS"), fgs)
        assert principal == "thiol"

    def test_pure_hydroperoxide_selects_hydroperoxide(self):
        """Ethyl hydroperoxide (CCOO) should still select hydroperoxide."""
        fgs = detect_functional_groups(Chem.MolFromSmiles("CCOO"))
        principal, _ = get_principal_group(Chem.MolFromSmiles("CCOO"), fgs)
        assert principal == "hydroperoxide"
