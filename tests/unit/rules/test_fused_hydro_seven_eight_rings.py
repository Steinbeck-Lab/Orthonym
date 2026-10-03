"""Two ortho-fused rings with a seven- or eight-membered ring are numbered and named as fused rings.

 (the Blue Book): the degree of hydrogenation is given by 'hydro' prefixes on the
mancude parent, '6,7-dihydro-5H-benzo[7]annulene (PIN)' (:17032); (:12501) numbers the
periphery. The fusion numbering placed rings on a lattice that holds no seven- or eight-membered ring,
so '6,7,8,9-tetrahydro-5H-cyclohepta[b]pyridine' abstained at the PIN tier (best-effort spelled the
saturated ring as indicated hydrogen, '5H,6H,7H,8H,9H-...', or fell back to a von Baeyer name). Two
rings always lie in one horizontal row, so every orientation is allowed and the cascade
picks the start (no lattice needed). The algorithmic namer also lacked the eight-membered carbocyclic
attached component 'cycloocta'; 'dibenzo[4,5:6,7]cycloocta[1,2-c]furan (PIN)':13487).

An azole component the algorithmic namer spells without its bracketed locants is never a PIN
,:11982: '[1,3]thiazole', not 'thiazole'), so '5,6,7,8-tetrahydro-4H-cyclohepta[d]thiazole'
ships at best-effort only. Every name below reads back to the input's full InChIKey with OPSIN 2.9.0.
"""
import random

import pytest
from rdkit import Chem

from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
from orthonym.rules.fusion_numbering import compute_fused_numbering
from tests.support.pin_tiers import assert_pin_at_both_tiers

PIN_ROWS = [
    ("N1=C2C(=CC=C1)CCCCC2", "6,7,8,9-tetrahydro-5H-cyclohepta[b]pyridine"),
    ("C1=NC=CC2=C1CCCCC2", "6,7,8,9-tetrahydro-5H-cyclohepta[c]pyridine"),
    ("S1C2=C(C=C1)CCCCC2", "5,6,7,8-tetrahydro-4H-cyclohepta[b]thiophene"),
    ("CC1=CC=C2C(=N1)CCCCC2", "2-methyl-6,7,8,9-tetrahydro-5H-cyclohepta[b]pyridine"),
    ("ClC1=CC=C2C(=N1)CCCCC2", "2-chloro-6,7,8,9-tetrahydro-5H-cyclohepta[b]pyridine"),
    ("NC1=C(C2=C(S1)CCCCC2)C#N",
     "2-amino-5,6,7,8-tetrahydro-4H-cyclohepta[b]thiophene-3-carbonitrile"),
    ("N1=C2C(=CC=C1)CCCCCC2", "5,6,7,8,9,10-hexahydrocycloocta[b]pyridine"),
    ("N1N=CC2=C1CCCCCC2", "4,5,6,7,8,9-hexahydro-1H-cycloocta[c]pyrazole"),
    ("N1=C2C(=CC=C1)CC=CC=C2", "5H-cyclohepta[b]pyridine"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_seven_and_eight_membered_fused_hydro_pins(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.opsin_gate
def test_an_unbracketed_azole_component_is_not_a_pin():
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    smiles = "S1C=NC2=C1CCCCC2"
    be = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert be["name"] == "5,6,7,8-tetrahydro-4H-cyclohepta[d]thiazole"
    assert be["tier"] != "pin_verified" and be["is_pin"] is False
    pin = Orthonym().name_tiered(smiles)
    assert pin["tier"] != "pin_verified"


def _fmt(loc):
    return f"{loc[0]}{loc[1]}" if isinstance(loc, tuple) else str(loc)


def _skeleton(mol):
    e = Chem.RWMol(mol)
    for b in e.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in e.GetAtoms():
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    return e.GetMol()


def _same_up_to_automorphism(mol, a, b):
    sk = _skeleton(mol)
    for auto in sk.GetSubstructMatches(sk, uniquify=False, maxMatches=5000):
        if all(a[i] == b[auto[i]] for i in a):
            return True
    return False


def _two_rings_one_large(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    rings = mol.GetRingInfo().AtomRings()
    return len(rings) == 2 and max(len(r) for r in rings) >= 7


_TWO_RING_LARGE = [k for k, v in FUSED_HETEROCYCLE_DATA.items()
                   if v.get("iupac_locants") and _two_rings_one_large(k)]


def test_the_catalogue_has_seven_and_eight_membered_two_ring_maps():
    assert len(_TWO_RING_LARGE) >= 10


@pytest.mark.parametrize("key", _TWO_RING_LARGE,
                         ids=[FUSED_HETEROCYCLE_DATA[k]["name"] for k in _TWO_RING_LARGE])
def test_two_ring_numbering_reproduces_the_catalogue(key):
    # benzo[7]annulene, benzo[8]annulene, 1H-cyclopenta[8]annulene, heptalene, the benzoxepines
    # and benzazepines: the catalogue maps come from OPSIN's numbering of each name
    mol = Chem.MolFromSmiles(key)
    ring = {a for r in mol.GetRingInfo().AtomRings() for a in r}
    num = compute_fused_numbering(mol, ring)
    assert num is not None
    have = {i: str(FUSED_HETEROCYCLE_DATA[key]["iupac_locants"][i]) for i in ring}
    assert _same_up_to_automorphism(mol, have, {i: _fmt(num[i]) for i in ring})


def test_azulene_numbering():
    mol = Chem.MolFromSmiles("c1cc2cccccc2c1")
    ring = set(range(mol.GetNumAtoms()))
    num = {i: _fmt(l) for i, l in compute_fused_numbering(mol, ring).items()}
    assert sorted(num.values()) == sorted(["1", "2", "3", "3a", "4", "5", "6", "7", "8", "8a"])
    fusion = [i for i in ring if mol.GetAtomWithIdx(i).GetDegree() == 3]
    assert sorted(num[i] for i in fusion) == ["3a", "8a"]


@pytest.mark.parametrize("smiles", ["N1=C2C(=CC=C1)CCCCC2", "N1=C2C(=CC=C1)CCCCCC2",
                                    "S1C2=C(C=C1)CCCCC2"])
def test_two_ring_numbering_does_not_depend_on_atom_order(smiles):
    base = Chem.MolFromSmiles(smiles)
    ring0 = {a for r in base.GetRingInfo().AtomRings() for a in r}
    ref = {i: _fmt(l) for i, l in compute_fused_numbering(base, ring0).items()}
    rng = random.Random(7)
    for _ in range(12):
        order = list(range(base.GetNumAtoms()))
        rng.shuffle(order)
        mol = Chem.RenumberAtoms(base, order)
        ring = {a for r in mol.GetRingInfo().AtomRings() for a in r}
        num = {i: _fmt(l) for i, l in compute_fused_numbering(mol, ring).items()}
        back = {order[new]: loc for new, loc in num.items()}   # new index -> old index
        assert _same_up_to_automorphism(base, ref, back)
