"""P-62.3.1.3 (BB 26568): "Compounds containing the group X=NH, where X is
a heteroatom and =NH the principal characteristic group, are named as
imines". BB verbatim: CH3-P=NH -> 1-methylphosphanimine (PIN).
"""
import pytest
from rdkit import Chem
from orthonym.namer import name_compound
from orthonym.rules.mononuclear_hydrides import name_heteroimine


@pytest.mark.unit
class TestHeteroimine:
    def test_methylphosphanimine(self):
        assert name_compound("CP=N", style="pin") == "1-methylphosphanimine"

    def test_bare_phosphanimine(self):
        assert name_compound("P=N", style="pin") == "phosphanimine"

    def test_direct_none_for_carbon_imine(self):
        # C=N is ethanimine territory — the heteroimine namer must decline.
        assert name_heteroimine(Chem.MolFromSmiles("CC=N")) is None

    def test_n_substituted_declines(self):
        # CP=NC would need N-methyl handling (P-62.3.1.3 silanimine
        # example) — not built in this task: fail closed here.
        assert name_heteroimine(Chem.MolFromSmiles("CP=NC")) is None
