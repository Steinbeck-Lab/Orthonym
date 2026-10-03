"""The tautomer check of the bridged fused PIN builder (its design spec, section 8): the
standard InChIKey does not tell the 1H- from the 2H- tautomer of an N-H / N= ring pair, so
'4,5,6,7-tetrahydro-2H-4,7-methanoindazole' reads back to the full key of the 1H input. For
a name whose indicated hydrogen or hydro atom is a ring N or bonded to one, the builder
compares the fixed-H InChI of the input with that of OPSIN's reading and declines on a
mismatch (``validation.protonation_identity.tautomer_verdict``)."""
import pytest
from rdkit import Chem

import orthonym.rules.bridged_fused_pin as pkg
from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.validation.protonation_identity import tautomer_verdict

ONE_H = "c1n[nH]c2c1C1CCC2C1"          # 4,5,6,7-tetrahydro-1H-4,7-methanoindazole
TWO_H = "c1[nH]nc2c1C1CCC2C1"          # 4,5,6,7-tetrahydro-2H-4,7-methanoindazole
WRONG = "4,5,6,7-tetrahydro-2H-4,7-methanoindazole"


def test_the_verdict_tells_the_tautomers_apart():
    assert tautomer_verdict(ONE_H, ONE_H) == "ok"
    assert tautomer_verdict(ONE_H, TWO_H) == "mismatch"
    assert tautomer_verdict(ONE_H, "c1ccccc1") == "n/a"      # the standard InChIs differ
    assert tautomer_verdict("not a smiles", ONE_H) == "n/a"


def test_the_sensitive_positions_are_ring_nitrogens_and_their_neighbours():
    mol = Chem.MolFromSmiles(ONE_H)
    nh = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetTotalNumHs())
    ch = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C" and a.GetIsAromatic()
              and a.GetTotalNumHs() == 1)
    far = next(a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() == 2 and a.GetDegree() == 2
               and not any(nb.GetIsAromatic() for nb in a.GetNeighbors())
               and not any(nb.GetIdx() in (nh, ch) for nb in a.GetNeighbors()))
    assert pkg._tautomer_sensitive(mol, {nh}, set())
    assert pkg._tautomer_sensitive(mol, set(), {ch})
    assert not pkg._tautomer_sensitive(mol, set(), {far})


@pytest.mark.parametrize("smiles", [
    "c1nc2c([nH]1)C1CCC2C1",            # an imidazole N-H between two carbons
    "c1[nH]cc2c1C1C=CC2C1",             # the isoindole N-H between two carbons
])
def test_a_ring_nitrogen_with_the_hydrogen_is_sensitive_itself(smiles):
    # the "is a ring N" branch alone: this N-H has no ring-N neighbour
    mol = Chem.MolFromSmiles(smiles)
    nh = next(a for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetTotalNumHs())
    assert not any(nb.GetAtomicNum() == 7 for nb in nh.GetNeighbors())
    assert pkg._tautomer_sensitive(mol, {nh.GetIdx()}, set())


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    (ONE_H, "4,5,6,7-tetrahydro-1H-4,7-methanoindazole"),
    (TWO_H, "4,5,6,7-tetrahydro-2H-4,7-methanoindazole"),
    ("Cn1ncc2c1C1CCC2C1", "1-methyl-4,5,6,7-tetrahydro-1H-4,7-methanoindazole"),
    ("c1[nH]cc2c1C1C=CC2C1", "4,7-dihydro-2H-4,7-methanoisoindole"),
])
def test_both_tautomers_get_their_own_name(smiles, name):
    res = pkg.build(Chem.MolFromSmiles(smiles))
    assert res is not None and res[0] == name, res


@pytest.mark.opsin_gate
def test_a_wrong_indicated_hydrogen_is_declined(monkeypatch):
    # constructed: the builder is made to spell the 2H tautomer for the 1H input. OPSIN
    # reads it to the input's full InChIKey (the engine's exit check passes it: measured
    # with the check switched off, the engine ships it pin_verified), so the fixed-H
    # comparison is the check that stops it.
    real = pkg._choose

    def wrong(*args, **kwargs):
        got = real(*args, **kwargs)
        return (WRONG,) + tuple(got[1:]) if got else got

    monkeypatch.setattr(pkg, "_choose", wrong)
    assert pkg.build(Chem.MolFromSmiles(ONE_H)) is None
    with jvm_slots(1, purpose="bf-s2"):
        row = Orthonym().name_tiered(ONE_H)
    assert row.get("name") != WRONG, row
