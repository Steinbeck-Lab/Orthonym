"""P-66.1.1.4.5.1 (BB 33071): "The preferred prefix for the group
H2N-CO-CO-N= is 'oxamoylimino'." Table row BB 55479 confirms
oxamoylimino* as the preferred prefix. Target (DEFERRED doc, OPSIN-RT
verified): NC(=O)C(=O)N=CCC(=O)O -> 3-(oxamoylimino)propanoic acid.

Investigation: identify_functional_groups perceives the =N as the
'imine' FG (match (imine_C, imine_N)); the chain-terminal imine C stays
in the propanoic-acid chain, and the oxamoyl (H2N-CO-CO-) branch hangs
off the =N. The new polyfunctional block emits '(oxamoylimino)' at the
imine C's locant; a bare 'imino' (dropping the oxamoyl) would fail
coverage -> unknown, so a substituted =N-R the recognizer cannot name
fails closed.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestOxamoylimino:
    def test_oxamoylimino_propanoic(self):
        assert name_compound("NC(=O)C(=O)N=CCC(=O)O", style="pin") \
            == "3-(oxamoylimino)propanoic acid"

    def test_oxamide_protect(self):
        assert name_compound("NC(=O)C(N)=O", style="pin") == "oxamide"

    def test_plain_imine_protect(self):
        assert name_compound("CC=N", style="pin") == "ethanimine"

    def test_other_acyl_fails_closed(self):
        # acetyl-N= (would be 'acetylimino'/(1-oxoethyl)imino — not built):
        # never emit a bare 'imino' that drops the acetyl.
        n = name_compound("CC(=O)N=CCC(=O)O", style="pin")
        assert n == "unknown organic compound" or "acet" in n
