""" (BB 33557): polyfunctional amides; BB verbatim example 33580:
H2N-CH2-CO-N(CH3)-CH2-CHOH-CH2OH ->
2-amino-N-(2,3-dihydroxypropyl)-N-methylacetamide (PIN).

W2E-P1FG Task 14 : reproduce-first at the CODE level (2026-07-09)
exposed a STALE DEFERRED spec — BOTH targets already emit the correct
OPSIN-RT-verified PIN at HEAD (the acyl-substituent drop and
 pool discard the DEFERRED doc recorded were healed by an earlier
pass / the p1_amide plan). This is now a VERIFY-ONLY regression pin so the
rows can never silently regress; no src edit was needed.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestAM2PolyfunctionalAmide:
    def test_bb_verbatim_target(self):
        assert name_compound("CN(CC(O)CO)C(=O)CN", style="pin") \
            == "2-amino-N-(2,3-dihydroxypropyl)-N-methylacetamide"

    def test_minimal_tertiary_root2(self):
        assert name_compound("NCC(=O)N(C)C", style="pin") \
            == "2-amino-N,N-dimethylacetamide"

    def test_plain_tertiary_protect(self):
        assert name_compound("CC(=O)N(C)C", style="pin") \
            == "N,N-dimethylacetamide"

    def test_n_sub_amide_protect(self):
        assert name_compound("CCC(=O)NC=O", style="pin") \
            == "N-propanoylformamide"
