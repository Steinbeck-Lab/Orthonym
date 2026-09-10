"""(a) cyclic carbonates as pseudoketones + verify.

BB 28258: "(a) 1,3-dioxan-2-one (PIN)" — carbonyl bonded to TWO ring O.
BB 29314: "Cyclic anhydrides, esters and amides are named as pseudoketones".
OPSIN-parse verified: '1,3-dioxan-2-one' and '1,3-dioxolan-2-one' both
round-trip to the probe SMILES.
"""
import pytest
from orthonym.namer import name_compound
from orthonym.rules.lactones import name_monocyclic_lactone
from rdkit import Chem


@pytest.mark.unit
class TestCyclicCarbonate:
    def test_dioxanone_pin(self):                       # (a) target
        assert name_compound("O=C1OCCCO1", style="pin") == "1,3-dioxan-2-one"

    def test_dioxolanone_pin(self):                     # 5-ring analogue
        assert name_compound("O=C1OCCO1", style="pin") == "1,3-dioxolan-2-one"

    def test_lactone_unchanged(self):                   # protect
        assert name_compound("O=C1CCCO1", style="pin") == "oxolan-2-one"

    def test_direct_namer_dioxanone(self):
        assert name_monocyclic_lactone(Chem.MolFromSmiles("O=C1OCCCO1")) \
            == "1,3-dioxan-2-one"

    def test_fail_closed_ring_NS(self):                 # not a carbonate — no wrong name
        # thiazolidinone-type ring must NOT be claimed by the lactone path
        n = name_monocyclic_lactone(Chem.MolFromSmiles("S=C1NC(=O)CS1"))
        assert n is None
