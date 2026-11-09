"""Two different organyl groups on two non-equivalent acid positions take locants (2026-11-09).

 (the Blue Book): '1,1-diethyl 3-methyl butane-1,1,3-tricarboxylate (PIN)' (:31783); with equivalent
positions no locant: 'ethyl methyl benzene-1,3-dicarboxylate (PIN)' (:5068). The ring diester builder wrote
'ethyl methyl 4-methylbenzene-1,3-dicarboxylate' (pin_verified), a name of two isomers.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CCOC(=O)c1ccc(C)c(c1)C(=O)OC", "1-ethyl 3-methyl 4-methylbenzene-1,3-dicarboxylate"),
    ("COC(=O)c1ccc(C)c(c1)C(=O)OCC", "3-ethyl 1-methyl 4-methylbenzene-1,3-dicarboxylate"),
    ("CCOC(=O)c1ccc2ccccc2c1C(=O)OC", "2-ethyl 1-methyl naphthalene-1,2-dicarboxylate"),
    ("CC(C)(C)OC(=O)c1ccc2ccccc2c1C(=O)OC", "2-tert-butyl 1-methyl naphthalene-1,2-dicarboxylate"),
    # a chain '-dioate' cites no terminal locant:2877); its acid positions are 1 and n
    ("CCOC(=O)CCC(C)C(=O)OC", "5-ethyl 1-methyl 2-methylpentanedioate"),
    ("CCC(C)COC(=O)c1ccc(C)c(c1)C(=O)OC", "3-methyl 1-(2-methylbutyl) 4-methylbenzene-1,3-dicarboxylate"),
    ("CC[C@@H](C)OC(=O)c1ccc(C)c(c1)C(=O)OC", "1-[(2R)-butan-2-yl] 3-methyl 4-methylbenzene-1,3-dicarboxylate"),
    ("CCOC(=O)c1cccnc1C(=O)OC", "3-ethyl 2-methyl pyridine-2,3-dicarboxylate"),
    # equivalent positions: no locant
    ("CCOC(=O)c1cccc(c1)C(=O)OC", "ethyl methyl benzene-1,3-dicarboxylate"),
    ("CCOC(=O)c1ccccc1C(=O)OC", "ethyl methyl benzene-1,2-dicarboxylate"),
    ("COC(=O)c1ccc(C)c(c1)C(=O)OC", "dimethyl 4-methylbenzene-1,3-dicarboxylate"),
])
def test_mixed_diester_organyl_locants(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")


@pytest.mark.parametrize("smiles", ["CCOC(=O)CC(C)C(=O)OC", "CCOC(=O)CC(O)C(=O)OC"])
def test_no_located_organyls_over_an_unlocanted_prefix(smiles):
    """'methylbutanedioic acid' drops the locant of its one prefix,:2939); once the organyl locants tell
    the acid positions apart the prefix needs it again ('4-ethyl 1-methyl 2-methylbutanedioate'), which is not built:
    no '4-ethyl 1-methyl methylbutanedioate' PIN."""
    row = Orthonym().name_tiered(smiles)
    assert row["tier"] != "pin_verified" or "methyl methylbutanedioate" not in row["name"], row
