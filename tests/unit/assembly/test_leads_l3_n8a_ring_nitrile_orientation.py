"""Leads program L3, item N8a (part 1): a saturated ring with a carbonitrile suffix is numbered from
the suffix.

 'NUMBERING' (the Blue Book) gives low locants, in order, to "(c) principal
characteristic groups and free valences (suffixes)" (:3256), so the ring atom that carries the
carbonitrile carbon is 1, and then to "(f) detachable alphabetized prefixes, all considered together
in a series of increasing numerical order" (:3301); the Blue Book's own row is
'4-(disilylamino)cyclohexane-1-carbonitrile (PIN)' (:38198). ``_assemble_ring_nitrile_name`` numbered
the ring in perception order (the atom order of the input), so the prefixes were cited from
wherever the input listed the ring: '6-(4-ethylphenoxy)cyclohexane-1-carbonitrile' for the PIN
'2-(4-ethylphenoxy)cyclohexane-1-carbonitrile' and, from one order of the atoms, the wrong-molecule
'8-[(tert-butylamino)amino]cyclooctane-1-carbonitrile' (the round trip rejected it). The ring is now
numbered by the orientation the namer computes for every other appended ring suffix.

Each molecule is named from 10 atom orders (the given one and 9 seeded permutations); every order
must give the one PIN at both tiers, and each PIN is read back by a fresh OPSIN call.
"""
import random

import pytest
from rdkit import Chem

from tests.support.pin_tiers import name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


def _orders(smiles, n=10, seed=20261009):
    mol = Chem.MolFromSmiles(smiles)
    rng = random.Random(seed)
    out = [smiles]
    for _ in range(200):
        perm = list(range(mol.GetNumAtoms()))
        rng.shuffle(perm)
        v = Chem.MolToSmiles(Chem.RenumberAtoms(mol, perm), canonical=False)
        if v not in out:
            out.append(v)
        if len(out) == n:
            break
    return out


ROWS = [
    # the three names of the leads list
    ("CCc1ccc(OC2CCCCC2C#N)cc1", "2-(4-ethylphenoxy)cyclohexane-1-carbonitrile"),
    ("C1(C#N)C(n2nc(C)c(Cl)c2C)CC(CCC)CC1",
     "2-(4-chloro-3,5-dimethyl-1H-pyrazol-1-yl)-4-propylcyclohexane-1-carbonitrile"),
    # the order that gave a wrong molecule: the substituent is on the carbon that carries the nitrile
    ("CC(C)(C)NNC1(CCCCCCC1)C#N", "1-[(tert-butylamino)amino]cyclooctane-1-carbonitrile"),
    # (f) between the two directions round the ring
    ("N#CC1(C)CCCC(C)C1", "1,3-dimethylcyclohexane-1-carbonitrile"),
    ("N#CC1CCC(Br)CC1Cl", "4-bromo-2-chlorocyclohexane-1-carbonitrile"),
    ("N#CC1CC(C)CCC1C", "2,5-dimethylcyclohexane-1-carbonitrile"),
]


@pytest.mark.parametrize("smiles, pin", ROWS)
def test_every_atom_order_gives_the_one_pin(smiles, pin):
    assert name_is_rt_exact(pin, smiles), pin
    orders = _orders(smiles)
    assert len(orders) >= 10, orders
    for order in orders:
        d, b = name_default(order), name_breadth(order)
        assert (d["name"], d["tier"]) == (pin, "pin_verified"), (order, d["name"], d["tier"])
        assert (b["name"], b["tier"]) == (pin, "pin_verified"), (order, b["name"], b["tier"])


def test_the_unoriented_ring_keeps_its_perception_order(monkeypatch):
    """A ring the namer does not orient (no ``oriented_ring``) still gets a name from the atoms in
    perception order, as before; nothing is declined for the want of an orientation."""
    import orthonym.assembly.composer as composer
    from orthonym import Orthonym

    smiles = "N#CC1CCCCC1"
    eng = Orthonym(style="pin")
    mol = Chem.MolFromSmiles(smiles)
    feats = eng._perceive(mol, smiles, Chem.MolToSmiles(mol))
    eng._classify(feats)
    feats.oriented_ring = None
    assert composer._assemble_ring_nitrile_name(feats, "pin") == "cyclohexanecarbonitrile"
