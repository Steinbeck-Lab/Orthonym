"""An amide N-substituent that the naming pipeline declines is left to the other producers
because the writer says so (its refusal is a typed value), not because the text of the name
reads 'substituent' or holds a space (``rules.benzene._ring_bearing_n_substituent``).

The names that are kept fall under (the Blue Book, N-substituted
amides: the substituent prefixes are cited with the italic locant N).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.substituent_enumerator import name_substituent
from orthonym.errors import CASCADE_PLACEHOLDER, is_cascade_refusal
from orthonym.rules.amides import _name_n_substituent
from orthonym.rules.benzene import _bfs_substituent_atoms

pytestmark = pytest.mark.opsin_gate

#: amides whose N-substituent has a ring and is declined by every namer of the pipeline
REFUSED = [
    "O=C(NC[Si](C)(C)c1ccccc1)c1ccccc1",
    "O=C(NCc1ccc[Se+]1)c1ccccc1",
    "O=C(NCc1cc[b-]cc1)c1ccccc1",
    "O=C(NCC1=CC=C[Ge]1)c1ccccc1",
]

#: the same class where the substituent is named: pin_verified at the pin tier
NAMED = [
    ("O=C(Nc1ccccn1)c1ccccc1", "N-(pyridin-2-yl)benzamide"),
    ("Cc1ccc(C(=O)Nc2cccnc2)cc1", "4-methyl-N-(pyridin-3-yl)benzamide"),
    ("O=C(NC[Si](C)(C)c1ccccc1)c1ccccc1", "N-[(dimethylphenylsilyl)methyl]benzamide"),
]


def _n_branch(smiles):
    mol = Chem.MolFromSmiles(smiles)
    n = next(a for a in mol.GetAtoms() if a.GetSymbol() == 'N')
    c = next(nb for nb in n.GetNeighbors()
             if any(b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(nb).GetSymbol() == 'O'
                    for b in nb.GetBonds()))
    ring_atoms = {a for r in mol.GetRingInfo().AtomRings() if c.GetIdx() in
                  [x for x in r] or True for a in r}
    # the ring of the acyl group is the one bonded to the carbonyl carbon
    acyl_ring = {a for r in mol.GetRingInfo().AtomRings() for a in r
                 if any(nb.GetIdx() == c.GetIdx() for x in r
                        for nb in mol.GetAtomWithIdx(x).GetNeighbors())}
    nbr = next(nb for nb in n.GetNeighbors() if nb.GetIdx() != c.GetIdx())
    blocked = set(acyl_ring) | {n.GetIdx(), c.GetIdx()}
    branch = _bfs_substituent_atoms(mol, nbr.GetIdx(), blocked)
    return mol, branch, ring_atoms


@pytest.mark.parametrize("smiles", REFUSED)
def test_the_writers_refusal_is_a_typed_value_and_is_none_when_asked(smiles):
    mol, branch, _ = _n_branch(smiles)
    carbons = sum(1 for a in branch if mol.GetAtomWithIdx(a).GetSymbol() == 'C')
    assert _name_n_substituent(mol, branch, carbons, refusal_as_none=True) is None
    # the default keeps the placeholder, which is the typed refusal and equals the word
    default = _name_n_substituent(mol, branch, carbons)
    assert default == CASCADE_PLACEHOLDER
    assert is_cascade_refusal(name_substituent(mol, branch, branch[0]))


def test_a_name_built_around_the_placeholder_is_not_the_refusal():
    assert not is_cascade_refusal("N-" + CASCADE_PLACEHOLDER)
    assert not is_cascade_refusal(str(CASCADE_PLACEHOLDER))


def test_the_consumer_reads_the_type_not_the_text(monkeypatch):
    # known positive: the writer's refusal declines the N-substituent; a plain
    # placeholder word is declined too
    from orthonym.rules import amides, benzene
    smiles = "O=C(NC[Si](C)(C)c1ccccc1)c1ccccc1"
    mol, branch, ring_atoms = _n_branch(smiles)
    n = next(a for a in mol.GetAtoms() if a.GetSymbol() == 'N')
    c = next(nb.GetIdx() for nb in n.GetNeighbors() if nb.GetIdx() != branch[0]
             and nb.GetIdx() not in branch)
    acyl_ring = {a for r in mol.GetRingInfo().AtomRings() for a in r
                 if any(nb.GetIdx() == c for x in r
                        for nb in mol.GetAtomWithIdx(x).GetNeighbors())}
    blocked = acyl_ring | {n.GetIdx(), c}
    # the writer's refusal declines the N-substituent (another producer names this amide)
    assert benzene._ring_bearing_n_substituent(
        mol, branch[0], blocked, acyl_ring, c) is None
    # a plain placeholder word is never a name; a name with a space is not judged by its text
    monkeypatch.setattr(amides, "_name_n_substituent",
                        lambda mol, sub, cc, refusal_as_none=False: "substituent")
    assert benzene._ring_bearing_n_substituent(
        mol, branch[0], blocked, acyl_ring, c) is None
    monkeypatch.setattr(amides, "_name_n_substituent",
                        lambda mol, sub, cc, refusal_as_none=False: "two words")
    assert benzene._ring_bearing_n_substituent(
        mol, branch[0], blocked, acyl_ring, c) == "two words"
    monkeypatch.setattr(amides, "_name_n_substituent",
                        lambda mol, sub, cc, refusal_as_none=False: None)
    assert benzene._ring_bearing_n_substituent(
        mol, branch[0], blocked, acyl_ring, c) is None


@pytest.mark.parametrize("smiles,expected", NAMED)
def test_a_named_n_substituent_keeps_its_pin(smiles, expected):
    row = Orthonym(style="pin").name_tiered(smiles)
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


@pytest.mark.parametrize("smiles", REFUSED[1:])
def test_a_refused_n_substituent_still_abstains_at_the_pin_tier(smiles):
    row = Orthonym(style="pin").name_tiered(smiles)
    assert row["tier"] == "abstain", row


@pytest.mark.parametrize("smiles,guard", [
    # the free-valence guard: a doubly bonded silicon attachment named with a '-yl' token
    ("C=[Si](C)C", "P-29.2 guard"),
    # the B2 stereo guard: a defined C=C whose token carries no E/Z descriptor
    ("N/C=C/OC", "B2 stereo-completeness"),
])
def test_name_substituents_own_refusals_are_typed(smiles, guard, caplog):
    import logging

    from orthonym.errors import CascadeRefusal
    mol = Chem.MolFromSmiles(smiles)
    attach = mol.GetAtomWithIdx(0).GetNeighbors()[0].GetIdx()
    with caplog.at_level(logging.DEBUG,
                        logger="orthonym.assembly.substituent_enumerator"):
        result = name_substituent(mol, list(range(1, mol.GetNumAtoms())), attach)
    assert any(guard in r.getMessage() for r in caplog.records), caplog.text
    assert isinstance(result, CascadeRefusal), (result, type(result))
