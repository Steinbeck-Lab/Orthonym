"""The fusion numbering of systems with a ring other than six-membered, drawn on the hexagon
grid (``fusion_orientation.grid_orientations``): one numbering per ring system whatever the
input atom order, and OPSIN 2.9.0's numbering of the printed name.

 (the Blue Book) the permitted ring shapes; (:12081) "Polycyclic
fused ring systems are oriented in accordance with the following criteria considered in order
until a decision is reached: (a) maximum number of rings in a horizontal row; (b) maximum
number of rings in upper right quadrant..."; (:12501) "The numbering of
peripheral atoms in the preferred orientation starts from the uppermost ring. If there is more
than one uppermost ring, the ring furthest to the right is chosen."; (:12541) the
cascade (heteroatoms, fusion carbon atoms, indicated hydrogen). The rules number the ring
graph, so two drawings of one molecule must get one numbering.

 (:12493) "Anthracene, phenanthrene, acridine, carbazole, xanthene and its chalcogen
analogues, purine, and cyclopenta[a]phenanthrene are exceptions; traditional numberings are
retained": each keeps its fixed numbering (OPSIN's)."""
import json
import pathlib
import random

import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin.parents import _skeleton, _same_orbit, normalize_locant
from orthonym.rules.fusion_numbering import compute_fused_numbering

_DATA = json.loads((pathlib.Path(__file__).parent / "data" / "fusion_grid_numbering.json").read_text())
SYSTEMS = [pytest.param(s["smiles"], s["locants"], id=s["name"]) for s in _DATA["systems"]]


def _numbering_on(mol, perm):
    """compute_fused_numbering of ``mol`` with its atoms renumbered by ``perm``, mapped back."""
    m2 = Chem.RenumberAtoms(mol, perm)
    got = compute_fused_numbering(m2, set(range(m2.GetNumAtoms())))
    if not got or len(got) != m2.GetNumAtoms():
        return None
    return {perm[a]: normalize_locant(loc) for a, loc in got.items()}


@pytest.mark.parametrize("smiles,locants", SYSTEMS)
def test_one_numbering_in_every_atom_order_and_it_is_opsins(smiles, locants):
    mol = Chem.MolFromSmiles(smiles)
    n = mol.GetNumAtoms()
    key, _ = _skeleton(mol, range(n))
    opsin = {a: normalize_locant(loc) for a, loc in enumerate(locants)}
    rng = random.Random(7)
    for k in range(6):
        perm = list(range(n))
        if k:
            rng.shuffle(perm)
        got = _numbering_on(mol, perm)
        assert got is not None, k
        assert _same_orbit(key, opsin, got), (k, got)


def test_a_system_above_the_drawing_cap_gets_no_numbering():
    # dicyclopenta[f,f']pentaleno[1,2-a:6,5-a']dipentalene: eight five-membered rings, more
    # drawings than GRID_MAX_DRAWINGS; no numbering (the callers decline), never a guess
    # (the Blue Book, OPSIN 2.9.0's structure)
    mol = Chem.MolFromSmiles("C1=CC=C2C1=CC1=CC=3C(=C21)C=2C(=CC=1C2C2=C4C(C=C2C1)=CC=C4)C3")
    assert sorted(len(r) for r in mol.GetRingInfo().AtomRings()) == [5] * 8
    assert compute_fused_numbering(mol, set(range(mol.GetNumAtoms()))) is None


@pytest.mark.parametrize("name,smiles,locants", [
    ("9H-telluroxanthene", "C1=CC=CC=2[Te]C3=CC=CC=C3CC12",
     "1;2;3;4;4a;10;10a;5;6;7;8;8a;9;9a"),                         # Table 2.8:11646
    ("17H-cyclopenta[a]phenanthrene", "C1=CC=CC2=CC=C3C=4C=CCC4C=CC3=C12",
     "1;2;3;4;5;6;7;8;14;15;16;17;13;12;11;9;10"),
    ("9H-xanthene", "C1=CC=CC=2OC3=CC=CC=C3CC12", "1;2;3;4;4a;10;10a;5;6;7;8;8a;9;9a"),
    ("9H-carbazole", "C1=CC=CC=2C3=CC=CC=C3NC12", "1;2;3;4;4a;4b;5;6;7;8;8a;9;9a"),
    ("7H-purine", "N1=CN=C2NC=NC2=C1", "1;2;3;4;9;8;7;5;6"),
])
def test_the_traditional_numberings_are_kept(name, smiles, locants):
    # (:12493): the listed exceptions keep their traditional numbering (OPSIN $_AV)
    mol = Chem.MolFromSmiles(smiles)
    got = compute_fused_numbering(mol, set(range(mol.GetNumAtoms())))
    assert got is not None, name
    text = ";".join(str(normalize_locant(got[a])) for a in range(mol.GetNumAtoms()))
    assert text == locants, (name, text)
