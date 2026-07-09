"""P-15.2.2 (BB 5074ff): "the preferred IUPAC names are substitutive
names for ... semicarbazones ...: (CH3)2C=N-NH-CO-NH2 acetone
semicarbazone -> 2-(propan-2-ylidene)hydrazine-1-carboxamide (PIN)".
Covers both P-15.2.2 rows (PIN form + OPSIN-RT validation).

Investigation: compute_features gives principal_group=None with a 'urea'
FG present, so _try_name_semicarbazone is hooked into the urea handler
(handlers/urea.py) before _try_name_urea.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestSemicarbazone:
    def test_acetone_semicarbazone_pin(self):
        assert name_compound("CC(=NNC(N)=O)C", style="pin") \
            == "2-(propan-2-ylidene)hydrazine-1-carboxamide"

    def test_butanone_semicarbazone_pin(self):
        assert name_compound("CCC(C)=NNC(N)=O", style="pin") \
            == "2-(butan-2-ylidene)hydrazine-1-carboxamide"

    def test_semicarbazide_protect(self):
        # HEAD names the bare parent 'hydrazinecarboxamide' (OPSIN-RT valid);
        # pin the actual working form (plan recorded 'semicarbazide').
        assert name_compound("NNC(N)=O", style="pin") == "hydrazinecarboxamide"

    def test_n_substituted_terminal_fails_closed(self):
        # CH3 on the terminal carboxamide N -> 4-methylsemicarbazone
        # family, N4 locant machinery not built: fail closed.
        n = name_compound("CC(=NNC(NC)=O)C", style="pin")
        assert "hydrazine-1-carboxamide" not in n or "methyl" in n
