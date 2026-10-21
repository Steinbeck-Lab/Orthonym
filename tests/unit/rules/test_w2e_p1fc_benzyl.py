"""W2E-P1FC Task 3 — (BB 16270): unsubstituted -CH2-C6H5 is the
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

    def test_benzylidene(self):
        # "Unsymmetrical substituent groups having terminal atoms with
        # different types of free valencies" (the Blue Book): the =CH- methine
        # double-bonds cyclohexane and single-bonds benzene, and the Blue Book names it
        # '(cyclohexylidenemethyl)benzene (PIN) [not (phenylmethylidene)cyclohexane;
        # benzene is senior to cyclohexane (see ]' (:6390-6392). Benzene is the
        # parent, so the retained 'benzylidene' is not cited here. (W2E-D2 once
        # asserted 'benzylidenecyclohexane': the senior ring was not allowed to be the
        # parent because its ring-ylidene prefix had no producer; see
        # test_ring_ylidene_parent.py.) OPSIN 2.9.0 reads both names to
        # C(C1=CC=CC=C1)=C1CCCCC1.
        assert name_compound("C(=C1CCCCC1)c1ccccc1") == "(cyclohexylidenemethyl)benzene"

    def test_chloromethylbenzene_unaffected(self):
        # benzene is the parent here; not a benzyl-substituent case.
        assert name_compound("c1ccc(CCl)cc1") == "(chloromethyl)benzene"
