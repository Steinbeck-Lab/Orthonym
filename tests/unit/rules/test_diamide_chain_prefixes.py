"""A chain diamide with substituted nitrogens and a group on a chain carbon (``rules.diamide_chain_prefixes``).

 (the Blue Book) N-locants; (:2877, '2-methylpropanediamide (PIN)':2887);, the
N-locants first in one multiplied prefix ('N,2-dimethylpropanamide',:21624).
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CNC(=O)CC(Cl)C(=O)NC", "2-chloro-N1,N4-dimethylbutanediamide"),
    ("CNC(=O)CC(C)C(=O)NC", "N1,N4,2-trimethylbutanediamide"),
    ("CCNC(=O)C(C)C(=O)NCC", "N1,N3-diethyl-2-methylpropanediamide"),
    ("CNC(=O)CCC(=O)NC", "N1,N4-dimethylbutanediamide"),
    ("NC(=O)CC(Cl)C(=O)N", "2-chlorobutanediamide"),
], ids=lambda v: v)
def test_names(smiles, name):
    row = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified", row
