"""The retained PAH parents ovalene, rubicene and pleiadene, named with OPSIN's numbering.

 lists them with PIN status: '(4) rubicene (PIN)' (the Blue Book), '(7) pleiadene (PIN)'
(:11360); ovalene is entry (1) of the table (OCR'd as a figure before '(2) pyranthrene (PIN)',:11342). The numbering of
each entry in ``data/polycyclic_data.py`` is OPSIN's (``-o extendedsmi``, the atom labels ``$_AV``). Every SMILES below
is OPSIN 2.9.0's structure for the name in the comment, and the expected name is the lowest locant among the positions
that symmetry makes equivalent, the lowest locants), so a numbering table that points at the wrong atoms
gives a higher or a wrong locant.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.opsin_gate


ROWS = [
    ("C1=CC=C2C3=CC=CC=C3C=3C4=CC=CC=5C6=CC=CC=C6C(=C1C23)C45", "rubicene"),
    ("C1=CC2=CC=C3C=CC4=CC5=CC=C6C=CC7=CC=C8C=C1C1=C2C3=C4C4=C5C6=C7C8=C14", "ovalene"),
    ("C1=CC=C2C=CC=C3C=C4C=CC=CC4=CC1=C23", "pleiadene"),
    # 1-methylrubicene; 7-methylrubicene; 8-methylrubicene is 1-methylrubicene by symmetry
    ("CC1=CC=C2C3=CC=CC=C3C=3C4=CC=CC=5C6=CC=CC=C6C(=C1C23)C45", "1-methylrubicene"),
    ("CC=1C=CC=C2C3=CC=CC4=C5C6=CC=CC=C6C=6C=CC=C(C(C12)=C34)C65", "7-methylrubicene"),
    ("CC1=C2C=3C4=CC=CC=C4C4=CC=CC(=C5C6=CC=CC=C6C(C=C1)=C25)C43", "1-methylrubicene"),
    # 1-methylovalene; 7-methylovalene; 11-methylovalene is 3-methylovalene by symmetry
    ("CC1=CC2=CC=C3C=CC4=CC5=CC=C6C=CC7=CC=C8C=C1C1=C2C3=C4C4=C5C6=C7C8=C14", "1-methylovalene"),
    ("CC1=C2C=CC3=CC=C4C=CC5=CC6=CC=C7C=CC8=CC=C1C1=C9C2=C3C4=C5C9=C6C7=C18", "7-methylovalene"),
    ("CC1=CC2=CC=C3C=C4C=CC5=CC=C6C=CC7=CC8=CC=C1C1=C2C3=C3C4=C5C6=C7C3=C18", "3-methylovalene"),
    # pleiaden-8-ol: 8 and 11 are equivalent by symmetry, and 8 is the lower locant
    ("C1=CC=C2C=CC=C3C=C4C(=CC=CC4=CC1=C23)O", "pleiaden-8-ol"),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_the_pin_tier_names_the_parent_with_opsins_numbering(smiles, expected):
    row = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered(smiles)
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row
