"""An ionic centre outside two identical units makes the parent: no multiplicative PIN to wait for.

 "SENIORITY ORDER FOR CLASSES" (the Blue Book): anions and cations come before every neutral class, so the
iodanium is the parent and the two hydroxyphenyl groups its substituents ('bis(4-hydroxyphenyl)iodanium');
``pin_vocabulary.multiplicative_candidate`` used to read the phenols as the principal group of two identical units.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.rules.pin_vocabulary import multiplicative_candidate


@pytest.mark.parametrize("smiles, expected", [
    ("Oc1ccc([I+]c2ccc(O)cc2)cc1", False),
    ("Oc1ccc([I-]c2ccc(O)cc2)cc1", False),
    ("Oc1ccc(Cc2ccc(O)cc2)cc1", True),                  # a neutral linker: '4,4'-methylenediphenol'
    ("O=[N+]([O-])c1ccc(Cc2ccc([N+](=O)[O-])cc2)cc1", True),   # nitro groups are no ionic centre
], ids=lambda v: str(v))
def test_multiplicative_candidate(smiles, expected):
    assert multiplicative_candidate(Chem.MolFromSmiles(smiles)) is expected


@pytest.mark.opsin_gate
def test_the_pin_tier_names_the_iodanium():
    row = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered("Oc1ccc([I+]c2ccc(O)cc2)cc1")
    assert row["name"] == "bis(4-hydroxyphenyl)iodanium", row
    assert row["tier"] == "pin_verified", row
