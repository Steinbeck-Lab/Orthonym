""" (BB 34346): "-S(O)(=NH)-NH2 S-aminosulfonimidoyl
(preselected prefix)"; table BB 55487. Aryl parent per seniority.py
comment (benzenesulfonimidamide) and family.

HEAD divergence: the benzene SUFFIX_FG + SMARTS for sulfonimidamide were
already shipped (plan P1AM), so benzenesulfonimidamide works at HEAD; this
task only flips the chain PREFIX_FORMS row from the divalent connector
'sulfonimidoyl' to the preselected 'S-aminosulfonimidoyl'.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestSulfonimidamide:
    def test_benzenesulfonimidamide(self):
        assert name_compound("N=S(N)(=O)c1ccccc1", style="pin") \
            == "benzenesulfonimidamide"

    def test_s_aminosulfonimidoyl_prefix(self):
        assert name_compound("N=S(N)(=O)CCC(=O)O", style="pin") \
            == "3-(S-aminosulfonimidoyl)propanoic acid"

    def test_chain_parent_protect(self):
        assert name_compound("CS(=N)(=O)N", style="pin") \
            == "methanesulfonimidamide"

    def test_benzenesulfonamide_protect(self):
        assert name_compound("NS(=O)(=O)c1ccccc1", style="pin") \
            == "benzenesulfonamide"
