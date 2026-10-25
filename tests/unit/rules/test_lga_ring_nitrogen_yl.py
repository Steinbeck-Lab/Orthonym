"""LGA: a nitrogen that is the free valence of a ring on a benzene parent is not an amino group.

``-N1CCCCC1`` on the benzene of benzoic acid was read as an amino group with two chains (the carbons
around the ring walked as an alkyl group of five carbons: '4-(dipentylamino)benzoic acid') and
``-n1cccc1`` as 'dibutylamino'; the OPSIN read-back refused both and the molecule had no name at the
default tier. The ring is a heteromonocyclic or fused parent hydride with its free valence on the
nitrogen: "General methodology for naming substituent groups" (the Blue Book
heading ``## ****``), the names printed by the book:

* ``(1H-indol-1-yl)acetic acid (PIN)`` table,:2039);
* ``2-chloro-3-(pyrrolidin-1-yl)naphthalene-1,4-dione (PIN)`` (:28955);
* ``3-methyl-4-(morpholin-4-yl)-2,2-diphenyl-1-(pyrrolidin-1-yl)butan-1-one (PIN)`` (:29617);
* a locant-bearing simple prefix is enclosed: ``tetra(naphthalen-2-yl)`` (a),:7085,:7090).

The expected names below follow from those forms with benzoic acid as the parent.
"""
import pytest
from rdkit import Chem, RDLogger

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.tiers import emit_tier_flags
from orthonym.rules.ring_nitrogen_yl import MAX_RING_YL_SYSTEM, n_in_substituent_ring, ring_yl_in_scope
from tests.support.rt_assert import assert_full_rt

RDLogger.DisableLog("rdApp.*")

pytestmark = pytest.mark.opsin_gate

#: (id, SMILES, PIN)
RING_N_YL = [
    ("piperidin-1-yl", "OC(=O)c1ccc(cc1)N1CCCCC1", "4-(piperidin-1-yl)benzoic acid"),
    ("pyrrolidin-1-yl", "OC(=O)c1ccc(cc1)N1CCCC1", "4-(pyrrolidin-1-yl)benzoic acid"),
    ("azepan-1-yl", "OC(=O)c1ccc(cc1)N1CCCCCC1", "4-(azepan-1-yl)benzoic acid"),
    ("1H-pyrrol-1-yl", "OC(=O)c1ccc(cc1)-n1cccc1", "4-(1H-pyrrol-1-yl)benzoic acid"),
    ("1H-indol-1-yl", "OC(=O)c1ccc(cc1)-n1ccc2ccccc12", "4-(1H-indol-1-yl)benzoic acid"),
    ("9H-carbazol-9-yl", "OC(=O)c1ccc(cc1)-n1c2ccccc2c2ccccc12", "4-(9H-carbazol-9-yl)benzoic acid"),
    ("2,3-dihydro-1H-indol-1-yl", "OC(=O)c1ccc(cc1)N1CCc2ccccc12",
     "4-(2,3-dihydro-1H-indol-1-yl)benzoic acid"),
    ("3,6-dimethyl-9H-carbazol-9-yl", "OC(=O)c1ccc(cc1)-n1c2ccc(C)cc2c2cc(C)ccc12",
     "4-(3,6-dimethyl-9H-carbazol-9-yl)benzoic acid"),
]
IDS = [row[0] for row in RING_N_YL]


@pytest.fixture(scope="module")
def namer():
    with jvm_slots(1, purpose="lga-test"):
        yield Orthonym(style="pin", **emit_tier_flags("pin"))   # emit_tier="pin": the PIN-only tier


@pytest.mark.parametrize("_id,smiles,pin", RING_N_YL, ids=IDS)
def test_the_ring_nitrogen_is_a_ring_yl_prefix_and_the_name_is_the_pin(namer, _id, smiles, pin):
    row = namer.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def _n(smiles):
    mol = Chem.MolFromSmiles(smiles)
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    benzene = set(next(r for r in mol.GetRingInfo().AtomRings() if n not in r))
    return mol, n, benzene


def test_the_nitrogen_of_a_ring_substituent_is_recognised():
    mol, n, benzene = _n("OC(=O)c1ccc(cc1)N1CCCCC1")
    assert n_in_substituent_ring(mol, n, benzene)


@pytest.mark.parametrize("smiles", [
    "OC(=O)c1ccc(cc1)N(C)C",                     # an acyclic amino group
    "OC(=O)c1ccc(cc1)N(C1CCCCC1)CC",             # a ring on a chain of the nitrogen, not the nitrogen's ring
])
def test_an_acyclic_nitrogen_is_not_a_ring_member(smiles):
    mol, n, benzene = _n(smiles)
    assert not n_in_substituent_ring(mol, n, benzene)


@pytest.mark.parametrize("smiles,pin", [
    ("OC(=O)c1ccc(cc1)N(C)C", "4-(dimethylamino)benzoic acid"),
    ("OC(=O)c1ccc(cc1)N1CCOCC1", "4-(morpholin-4-yl)benzoic acid"),
])
def test_the_amino_group_and_the_morpholine_keep_their_names(namer, smiles, pin):
    row = namer.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row


def _fragment(smiles):
    """The molecule, the first atom of the substituent bonded to the benzene parent ring (the first six
    atoms of the SMILES), and the atoms of that substituent."""
    mol = Chem.MolFromSmiles(smiles)
    parent = set(range(6))
    start = next(n.GetIdx() for i in parent for n in mol.GetAtomWithIdx(i).GetNeighbors()
                 if n.GetIdx() not in parent)
    seen, stack = {start}, [start]
    while stack:
        for n in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            if n.GetIdx() not in parent and n.GetIdx() not in seen:
                seen.add(n.GetIdx())
                stack.append(n.GetIdx())
    return mol, start, seen


def test_a_pyrene_ring_yl_of_a_benzene_parent_is_in_scope():
    # "General methodology for naming substituent groups" (the Blue Book): the free
    # valence of a fused hydrocarbon, 'pyren-1-yl' (16 atoms), is named by the ring-yl writers
    mol, start, frag = _fragment("c1ccccc1-c1ccc2ccc3cccc4ccc1c2c34")
    assert len(frag) == 16 and ring_yl_in_scope(mol, start, frag)


def test_a_chain_of_ring_systems_is_not_a_ring_yl():
    # a 4-phenylene chain is one ring system after another joined by bonds: each ring is small, the
    # fragment is not. The branched polyphenylene of a dendrimer drove the cascade through the
    # fragment and the von Baeyer analysis for minutes (row 6 of the large aromatic set).
    smiles = "c1ccccc1" + "-c1ccc(cc1)" * 8 + "-c1ccccc1"
    mol, start, frag = _fragment(smiles)
    assert len(frag) > MAX_RING_YL_SYSTEM
    assert not ring_yl_in_scope(mol, start, frag)
    assert ring_yl_in_scope(mol, start, set(list(frag)[:MAX_RING_YL_SYSTEM]))
