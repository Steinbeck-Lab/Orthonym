"""W2E-P1FC Task 9 — P-58.3.2 (BB 24896): homogeneous heteroatom chain broken
on the senior characteristic group.
"an acyclic homogeneous heterocyclic chain may be broken in order to recognize
a senior function" — H2N-NH-NH-NH-COOH -> tetraazane-1-carboxylic acid (PIN)
[the carboxylic acid is senior to the carbonic acid derivative]."""
import pytest

from orthonym.namer import name_compound
from rdkit import Chem
from orthonym.rules.polyazane import name_polyazane


@pytest.mark.unit
class TestP5832HeterochainAcid:
    def test_tetraazane_carboxylic_acid(self):
        assert name_compound("NNNNC(=O)O") == "tetraazane-1-carboxylic acid"

    def test_triazane_carboxylic_acid(self):
        assert name_compound("NNNC(=O)O") == "triazane-1-carboxylic acid"

    def test_bare_azane_unchanged(self):
        # the carboxyl split must not corrupt the bare-hydride path.
        assert name_compound("NNNN") == "tetraazane"
        assert name_polyazane(Chem.MolFromSmiles("NNNN")) == "tetraazane"

    def test_2n_acid_keeps_retained_pin(self):
        # 2-N acid stays 'hydrazinecarboxylic acid' (peel gated to n>=3).
        assert name_compound("NNC(=O)O") == "hydrazinecarboxylic acid"
