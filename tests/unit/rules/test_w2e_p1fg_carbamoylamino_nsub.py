"""P-66.1.6.1.1.3 (BB 33338): urea-derived prefixes are systematic;
table BB 55463: "carbamoylamino* (not ureido) | H2N-CO-NH-".
Target (DEFERRED doc, OPSIN-RT verified): NC(=O)NCCCNC=O ->
N-[3-(carbamoylamino)propyl]formamide.

Investigation: the ACID leg (NC(=O)NCC(=O)O -> 2-(carbamoylamino)ethanoic
acid) already works via the polyfunctional urea-FG prefix path; this task
fixes only the amide-N-SUBSTITUENT leg, where the propyl fragment
terminated by -NH-C(=O)-NH2 was named 'N-propylureayl' by the recursive
fallback. New recognizer in name_substituent emits
'3-(carbamoylamino)propyl' from structure.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestCarbamoylaminoNSub:
    def test_carbamoylamino_propyl_formamide(self):
        assert name_compound("NC(=O)NCCCNC=O", style="pin") \
            == "N-[3-(carbamoylamino)propyl]formamide"

    def test_acid_leg_protect(self):
        assert name_compound("NC(=O)NCC(=O)O", style="pin") \
            == "2-(carbamoylamino)ethanoic acid"

    def test_urea_parent_protect(self):
        assert name_compound("CNC(N)=O", style="pin") == "N-methylurea"
