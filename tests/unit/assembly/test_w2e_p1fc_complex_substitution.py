"""W2E-P1FC Task 6 — P-35.4.1 (BB 18108): complex prefixes by substitution.
"Names of complex substituent prefixes may be formed by substituting a simple
or compound substituent prefix into a compound substituent prefix."
Examples: (chloromethyl)amino, (4-chlorophenyl)methoxy."""
import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestP3541ComplexBySubstitution:
    def test_chlorophenylmethanol_head(self):
        # (4-chlorophenyl)methanol already conforms — protect pin.
        assert name_compound("OCc1ccc(Cl)cc1") == "(4-chlorophenyl)methanol"

    def test_chloromethylamino_prefix(self):
        # 4-[(chloromethyl)amino]benzoic acid (P-35.4.1). OPSIN-RT verified.
        assert name_compound("ClCNc1ccc(C(=O)O)cc1") == \
            "4-[(chloromethyl)amino]benzoic acid"
