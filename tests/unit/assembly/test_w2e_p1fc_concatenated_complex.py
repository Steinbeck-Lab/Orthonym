"""W2E-P1FC Task 8 — (BB 16250): concatenated complex substituent
groups (3+ components) with enclosing-mark escalation.
Expected PIN OPSIN-RT verified at authoring time (2026-07-09):
4-[(benzylsulfanyl)methyl]benzoic acid -> C(C1=CC=CC=C1)SCC1=CC=C(C(=O)O)C=C1
(canon-equal to OC(=O)c1ccc(CSCc2ccccc2)cc1)."""
import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestP2952ConcatenatedComplex:
    def test_concatenated_three_component(self):
        # -CH2-S-CH2-C6H5 on a benzoic-acid parent: benzyl -> sulfanyl -> methyl
        # (3-component concatenation). OPSIN-RT verified.
        assert name_compound("OC(=O)c1ccc(CSCc2ccccc2)cc1") == \
            "4-[(benzylsulfanyl)methyl]benzoic acid"

    def test_oxy_analog_unchanged(self):
        # The oxy concatenation encloses each compound prefix:
        # (the Blue Book) parentheses around compound prefixes, nested by
        # (:7446); '(benzyloxy)carbonyl (preferred prefix)' (:18116).
        assert name_compound("OC(=O)c1ccc(COCc2ccccc2)cc1") == \
            "4-[(benzyloxy)methyl]benzoic acid"
