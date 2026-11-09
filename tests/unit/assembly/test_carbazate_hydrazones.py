"""The ester of a hydrazinecarboxylic acid whose second nitrogen is a hydrazone carbon (program 14 lead, 2026-11-09).

 class 9 "Esters" (the Blue Book) outranks the hydrazine derivatives, so R-O-CO-NH-N=CR2 is the
ester of '2-(ylidene)hydrazine-1-carboxylic acid', numbered 1 and 2 from the carbonyl:38507), as the
amide of the same acid is '2-(propan-2-ylidene)hydrazine-1-carboxamide (PIN)' (:5096). The hydrazone handler took the
molecule first and its name failed later, so the PIN tier declined.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CC(C)=NNC(=O)OC(C)(C)C", "tert-butyl 2-(propan-2-ylidene)hydrazine-1-carboxylate"),
    ("CCOC(=O)NN=C1CCCCC1", "ethyl 2-cyclohexylidenehydrazine-1-carboxylate"),
    ("COC(=O)NN=Cc1ccccc1", "methyl 2-benzylidenehydrazine-1-carboxylate"),
    ("COC(=O)N(C)N=CC", "methyl 2-ethylidene-1-methylhydrazine-1-carboxylate"),
    # unchanged: the hydrazone stays the parent when no ester is senior to it
    ("NC(=O)NN=C(C)C", "2-(propan-2-ylidene)hydrazine-1-carboxamide"),
    ("CC(=O)NN=C(C)C", "N'-(propan-2-ylidene)acetohydrazide"),
    ("CC(C)=NN", "(propan-2-ylidene)hydrazine"),
])
def test_a_carbazate_hydrazone_is_named_as_the_ester(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")
