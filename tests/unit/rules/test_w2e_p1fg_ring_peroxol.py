""" (BB 27935): "(1) substitutively using the suffix 'peroxol'...
Method (1) leads to preferred IUPAC names. Examples:
(1) 1,2,3,4-tetrahydronaphthalene-1-peroxol (PIN)" — BB verbatim example.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
class TestRingPeroxolFused:
    def test_tetrahydronaphthalene_peroxol(self):
        assert name_compound("OOC1CCCC2=CC=CC=C12", style="pin") \
            == "1,2,3,4-tetrahydronaphthalene-1-peroxol"

    def test_simple_ring_peroxol_protect(self):
        assert name_compound("OOC1CCCCC1", style="pin") == "cyclohexane-1-peroxol"

    def test_chain_peroxol_protect(self):
        assert name_compound("CC(C)(CC)OO", style="pin") == "2-methylbutane-2-peroxol"

    def test_tetralone_protect(self):
        assert name_compound("O=C1CCCC2=CC=CC=C12", style="pin") \
            == "3,4-dihydronaphthalen-1(2H)-one"

    def test_decorated_fails_closed(self):
        # extra substituent the new path does not enumerate -> never a
        # structure-dropping name
        n = name_compound("OOC1CC(C)CC2=CC=CC=C12", style="pin")
        assert n == "unknown organic compound" or "methyl" in n
