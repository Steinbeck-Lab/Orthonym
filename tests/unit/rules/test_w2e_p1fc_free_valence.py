"""W2E-P1FC Task 5 — / (BB 15935) free-valence suffix
vocabulary and citation order.
"If there is a choice, low locants are assigned, in order, to the suffixes
'yl', 'ylidene', and 'ylidyne'. In names, the suffixes are cited in the
order 'yl', 'ylidene', and 'ylidyne'."
All expected PINs OPSIN-RT verified at authoring time; all conform at HEAD
2026-07-09 (verify-only pins).
"""
import pytest

from orthonym.namer import name_compound
from rdkit import Chem
from orthonym.assembly.naming_utils import unbranched_alkylidene_name


@pytest.mark.unit
class TestP292FreeValence:
    def test_alkylidene_producer(self):
        mol = Chem.MolFromSmiles("CC=C1CCCCC1")
        assert unbranched_alkylidene_name(mol, 1, 2) == "ethylidene"

    def test_ylidene_end_to_end(self):
        #...=CH-CH3 on a ring parent -> ylidene form (OPSIN-RT verified).
        assert name_compound("CC=C1CCCCC1") == "ethylidenecyclohexane"

    def test_diol_not_diyl(self):
        # -diyl is a free valence; a diol keeps the -diol suffix (contrast).
        assert name_compound("OCCCCO") == "butane-1,4-diol"

    def test_dioic_acid(self):
        assert name_compound("OC(=O)CCCCCC(=O)O") == "heptanedioic acid"

    def test_dicarbaldehyde(self):
        assert name_compound("O=Cc1ccc(C=O)cc1") == "benzene-1,4-dicarbaldehyde"
