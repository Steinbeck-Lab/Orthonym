"""Leads L2 (E), item N8a (name part): a heteromonocycle that carries a free valence and a substituent
ring is numbered ONCE.

``### **** NUMBERING`` (the Blue Book): the lowest locants go to the heteroatoms, then to
indicated hydrogen, then to "(c) principal characteristic groups and free valences (suffixes)" (:3256),
then to "(f) detachable alphabetized prefixes, all considered together in a series of increasing
numerical order" (:3301). In the compound substituent prefix of a heteromonocycle the free valence is
the suffix: so the piperazine that holds the chain on one nitrogen and a phenyl on the other is
'4-phenylpiperazin-1-yl', never '1-phenylpiperazin-4-yl' (nitrogen 1 and 4 are the same numbering in
both directions, the free valence takes the 1).

``name_mixed_ring_prefix`` read the attachment locant and the connection locant off two independent
numberings (each atom took its own lowest locant over both walking directions and over the starting
heteroatoms): the piperazine got '1-phenylpiperazin-4-yl' in ten of ten atom orders, and pyridine could
read an attachment and a connection atom as the same locant ('2-(3-phenylpyridin-3-yl)ethan-1-ol', a
different molecule that rejected, so the default tier abstained).
"""
import pytest
from rdkit import Chem, RDLogger

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules import ring_assemblies as ra
from tests.support.rt_assert import assert_full_rt

RDLogger.DisableLog("rdApp.*")

#: (id, SMILES, the PIN)
ROWS = [
    ("piperazine-tetralyloxy", "OC(COC1CCCc2ccccc21)CN1CCN(c2ccccc2)CC1",
     "1-(4-phenylpiperazin-1-yl)-3-[(1,2,3,4-tetrahydronaphthalen-1-yl)oxy]propan-2-ol"),
    ("pyridin-2-yl-6-phenyl", "OCCc1cccc(-c2ccccc2)n1", "2-(6-phenylpyridin-2-yl)ethan-1-ol"),
    ("pyridin-2-yl-5-phenyl", "OCCc1ccc(-c2ccccc2)cn1", "2-(5-phenylpyridin-2-yl)ethan-1-ol"),
    ("pyridin-3-yl-5-phenyl", "OCCc1cncc(-c2ccccc2)c1", "2-(5-phenylpyridin-3-yl)ethan-1-ol"),
    ("pyridin-2-yl-4-phenyl", "OCCc1cc(-c2ccccc2)ccn1", "2-(4-phenylpyridin-2-yl)ethan-1-ol"),
    ("piperazine-pyridyl", "OCCN1CCN(c2ccccn2)CC1", "2-[4-(pyridin-2-yl)piperazin-1-yl]ethan-1-ol"),
]
IDS = [r[0] for r in ROWS]


def _locants(mapped_smiles):
    """The numbering of the ring for the atoms mapped 1 (the free valence) and 2 (the connection)."""
    mol = Chem.MolFromSmiles(mapped_smiles)
    by_map = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
    return ra._heteromonocycle_prefix_locants(mol, set(range(mol.GetNumAtoms())), by_map[1], by_map[2])


# ------------------------------------------------------------------ the numbering alone
@pytest.mark.parametrize("mapped,expected", [
    # piperazine: the two nitrogens are 1 and 4 in either direction; the free valence takes the 1
    ("C1C[N:1]CC[N:2]1", (1, 4)),
    ("C1C[N:2]CC[N:1]1", (1, 4)),
    ("[N:2]1CC[N:1]CC1", (1, 4)),
    # 1,4-dioxane: the free valence on the carbon next to an oxygen is 2, the substituent on the carbon
    # next to the other oxygen is 3 (the old per-atom minimum read the free valence as 3 when the
    # starting oxygen was the other one)
    ("O1[C:1][C:2]OCC1", (2, 3)),
    ("C1[C:2]O[C:1]CO1", (2, 6)),
    # pyridine: free valence on a 3-position, substituent on the other meta carbon: 3 and 5, never 3 and 3
    ("c1[n]c[c:1]c[c:2]1", (3, 5)),
    ("c1[n]c[c:2]c[c:1]1", (3, 5)),
    ("[c:1]1[n]c[c:2]cc1", (2, 5)),
    # a free valence beside the nitrogen outranks a substituent beside it on the far side
    ("[c:1]1[n]cc[c:2]c1", (2, 4)),
    # locant 1 goes to the most senior heteroatom,:8284): 1,3,4-oxadiazole is O1, N3, N4,
    # never numbered from a nitrogen even though that would give the set {1,2,4}
    ("o1[c:1]nn[c:2]1", (2, 5)),
    ("o1[c:2]nn[c:1]1", (2, 5)),
    # 1,3-thiazole: S1, N3; the free valence on C2 (between them), the substituent on C5 (next to S)
    ("s1[c:1]nc[c:2]1", (2, 5)),
    ("s1[c:2]nc[c:1]1", (5, 2)),
])
def test_one_numbering_for_the_free_valence_and_the_connection(mapped, expected):
    assert _locants(mapped) == expected


@pytest.mark.parametrize("attach_atom", [1, 2, 4, 5])
def test_a_ring_with_a_free_valence_alone_gets_the_lowest_free_valence_locant(attach_atom):
    # 1,4-dioxane: every ring carbon is next to an oxygen, so the free valence is 2, never 3
    # (the single-ring substituent producer, ring_substituents.py, reads '(1,4-dioxan-3-yl)' on an
    # eval row; see the lane's proposal note)
    mol = Chem.MolFromSmiles("O1CCOCC1")
    assert ra._heteromonocycle_prefix_locants(mol, set(range(6)), attach_atom) == (2, None)


def test_the_numbering_declines_a_partly_hydrogenated_or_fused_ring():
    # hydro prefixes come before the free valence; a fused ring is numbered by its fusion name:
    # both stay with the caller's standard numbering
    tetrahydropyridine = Chem.MolFromSmiles("C1=CCNCC1")
    assert ra._heteromonocycle_prefix_locants(tetrahydropyridine, set(range(6)), 0, 3) is None
    indole = Chem.MolFromSmiles("c1ccc2[nH]ccc2c1")
    assert ra._heteromonocycle_prefix_locants(indole, set(range(9)), 0, 3) is None


# ------------------------------------------------------------------ the names read back
@pytest.mark.parametrize("_id,smiles,pin", ROWS, ids=IDS)
def test_the_pin_reads_back_to_the_full_inchikey(_id, smiles, pin):
    assert_full_rt(pin, smiles)


# ------------------------------------------------------------------ the engine
pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namers():
    with jvm_slots(1, purpose="leads-L2-test"):
        yield {"pin": Orthonym(), "best-effort": Orthonym(style="pin", **_emit_tier_flags("best-effort"))}


@pytest.mark.parametrize("_id,smiles,pin", ROWS, ids=IDS)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_engine_names_the_ring_substituent_from_one_numbering(namers, _id, smiles, pin, tier):
    row = namers[tier].name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row


@pytest.mark.parametrize("_id,smiles,pin", ROWS[:4], ids=IDS[:4])
def test_the_name_does_not_depend_on_the_atom_order(namers, _id, smiles, pin):
    mol = Chem.MolFromSmiles(smiles)
    orders = {smiles} | {Chem.MolToSmiles(mol, doRandom=True, canonical=False) for _ in range(12)}
    names = {namers["pin"].name_tiered(o)["name"] for o in orders}
    assert names == {pin}
