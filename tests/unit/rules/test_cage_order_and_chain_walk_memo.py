"""An unwritable cage branch, and the chain-walk memo of the universal floor."""
from rdkit import Chem

from orthonym.assembly import universal_substituent as us
from orthonym.rules import cage_numbering


def test_a_branch_rdkit_cannot_write_leaves_the_prefixes_uncompared(monkeypatch):
    mol = Chem.MolFromSmiles("CC12CCC(CC1)C2")
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    assert cage_numbering.cage_branches(mol, ring, set()) is not None

    def refuse(*_a, **_k):
        raise Chem.KekulizeException("Can't kekulize mol")
    monkeypatch.setattr(cage_numbering.Chem, "MolFragmentToSmiles", refuse)
    assert cage_numbering.cage_branches(mol, ring, set()) is None


def test_a_memo_hit_of_the_chain_walk_fires_the_cyano_spelling_again():
    mol = Chem.MolFromSmiles("N#CCC")
    ctx = us._Ctx(mol=mol, ring_systems=[], ring_system_of={}, budget=us._Budget(10**6),
                 canon_rank=tuple(Chem.CanonicalRankAtoms(mol)), cation_sites=[], anion_sites=[])
    component = frozenset(range(mol.GetNumAtoms()))
    first = us._tree_neighbors(ctx, 2, component)
    assert ctx.forms_fired == {"chain"}
    ctx.forms_fired.clear()
    ctx.book_forms_fired = False
    assert us._tree_neighbors(ctx, 2, component) == first == [3]
    assert ctx.forms_fired == {"chain"} and ctx.book_forms_fired
