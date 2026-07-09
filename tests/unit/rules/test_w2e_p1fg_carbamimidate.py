"""P-66.1.6.1.2.1 (BB 33398): "The imidic acid tautomer of urea,
H2N-C(OH)=NH, is named 'carbamimidic acid'... derivatives of carbamimidic
acid are named using the locants N and N'."
Ester: 'carbamimidic acid' -> 'carbamimidate' + alkyl word ->
'methyl carbamimidate' (OPSIN-parse verified).
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestCarbamimidateEster:
    def test_methyl_carbamimidate(self):
        assert name_compound("COC(N)=N", style="pin") == "methyl carbamimidate"

    def test_ethyl_carbamimidate(self):
        assert name_compound("CCOC(N)=N", style="pin") == "ethyl carbamimidate"

    def test_acid_protect(self):
        assert name_compound("OC(N)=N", style="pin") == "carbamimidic acid"
