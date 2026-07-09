"""W2E-P1FC Task 3 — P-29.6.1 (BB 16270): unsubstituted -CH2-C6H5 is the
preferred prefix 'benzyl', not 'phenylmethyl'.
"The traditional prefixes benzyl, benzylidene, benzylidyne are retained
preferred prefixes" — 2-benzylpyridine (PIN), not 2-(phenylmethyl)pyridine.
"""
import pytest

from orthonym.namer import name_compound
from rdkit import Chem
from orthonym.rules.ring_substituents import name_ring_system_substituent


@pytest.mark.unit
class TestP2961Benzyl:
    def test_producer_emits_benzyl(self):
        mol = Chem.MolFromSmiles("C(c1ccccc1)c1ccccn1")
        assert name_ring_system_substituent(mol, [0, 1, 2, 3, 4, 5, 6], 0) == "benzyl"

    def test_benzyl_on_pyridine(self):
        assert name_compound("C(c1ccccc1)c1ccccn1") == "2-benzylpyridine"

    @pytest.mark.xfail(
        reason="benzylidene routes through the alkylidene producer, not the "
        "len(path)==1 CH2 branch — W2E-P1FC follow-up: apply P-29.6.1 "
        "benzylidene at the ylidene/alkylidene site",
        strict=False,
    )
    def test_benzylidene(self):
        # -CH=C6H5 on a ring parent -> benzylidene (P-29.6.1). OPSIN-verified:
        # benzylidenecyclohexane -> C(C1=CC=CC=C1)=C1CCCCC1.
        assert name_compound("C(=C1CCCCC1)c1ccccc1") == "benzylidenecyclohexane"

    def test_chloromethylbenzene_unaffected(self):
        # benzene is the parent here; not a benzyl-substituent case.
        assert name_compound("c1ccc(CCl)cc1") == "(chloromethyl)benzene"
