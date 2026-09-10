"""S2 (audit 2026-09-03): ``for a in mol.GetAtoms`` goes through RDKit's
Python sequence wrapper (about 88 us per 58-atom walk); the engine walks the
same molecule hundreds of times per name, mostly in dispatch predicates. 24% of
engine CPU on a dev split was this wrapper. ``atoms_of``/``bonds_of`` return a tuple
built once per molecule per top-level naming call (the memo scope) and reuse
it; molecules that can be edited in place (RWMol) are never cached.

Byte-identity contract: the tuple must contain the same atoms, in the same
order, as ``mol.GetAtoms``; a stale tuple must be impossible to observe.
"""
import pytest
from rdkit import Chem

from orthonym.assembly import memo
from orthonym.perception import molcache


@pytest.fixture
def scope():
    tok = memo.push_scope()
    yield
    memo.pop_scope(tok)


def _sig(atoms):
    return [(a.GetIdx(), a.GetSymbol(), a.GetFormalCharge()) for a in atoms]


def test_atoms_of_matches_GetAtoms_in_order(scope):
    mol = Chem.MolFromSmiles("CC(C)CC(N)C(=O)O")
    assert _sig(molcache.atoms_of(mol)) == _sig(mol.GetAtoms())
    assert [b.GetIdx() for b in molcache.bonds_of(mol)] == [b.GetIdx() for b in mol.GetBonds()]


def test_same_tuple_is_reused_within_a_scope(scope):
    mol = Chem.MolFromSmiles("c1ccccc1O")
    assert molcache.atoms_of(mol) is molcache.atoms_of(mol)
    assert molcache.bonds_of(mol) is molcache.bonds_of(mol)


def test_not_reused_across_scopes():
    mol = Chem.MolFromSmiles("c1ccccc1O")
    tok = memo.push_scope(); t1 = molcache.atoms_of(mol); memo.pop_scope(tok)
    tok = memo.push_scope(); t2 = molcache.atoms_of(mol); memo.pop_scope(tok)
    assert t1 is not t2 and _sig(t1) == _sig(t2)


def test_no_scope_means_no_caching():
    mol = Chem.MolFromSmiles("CCO")
    assert molcache.atoms_of(mol) is not molcache.atoms_of(mol)


def test_rwmol_is_never_cached_and_tracks_edits(scope):
    rw = Chem.RWMol(Chem.MolFromSmiles("CCCC"))
    a = molcache.atoms_of(rw)
    rw.RemoveAtom(0)
    b = molcache.atoms_of(rw)
    assert a is not b and len(b) == 3 == rw.GetNumAtoms()


def test_two_different_molecules_never_share_an_entry(scope):
    m1 = Chem.MolFromSmiles("CCO"); m2 = Chem.MolFromSmiles("CCN")
    assert _sig(molcache.atoms_of(m1)) != _sig(molcache.atoms_of(m2))


def test_id_reuse_after_garbage_collection_cannot_serve_stale_atoms(scope):
    import gc
    sigs = set()
    for smi in ("CCO", "CCN", "CCS", "CCF", "CCCl"):
        mol = Chem.MolFromSmiles(smi)
        sigs.add(tuple(_sig(molcache.atoms_of(mol))))
        del mol; gc.collect()
    assert len(sigs) == 5


def test_off_mode_always_rebuilds(scope, monkeypatch):
    monkeypatch.setattr(molcache, "_MODE", "off")
    mol = Chem.MolFromSmiles("CCO")
    assert molcache.atoms_of(mol) is not molcache.atoms_of(mol)


def test_verify_mode_raises_when_the_cached_tuple_is_not_this_molecule(scope, monkeypatch):
    """Simulates the one failure class the cache could have (an entry that
    belongs to another molecule of the same size, e.g. after an id reuse):
    verify mode must catch it instead of serving it."""
    monkeypatch.setattr(molcache, "_MODE", "verify")
    mol = Chem.MolFromSmiles("CCO"); other = Chem.MolFromSmiles("CCN")
    molcache.atoms_of(mol)
    memo._cache_var.get()[(molcache._NS_ATOMS, id(mol))] = (mol, molcache._fresh_atoms(other))
    with pytest.raises(molcache.MolCacheMismatchError):
        molcache.atoms_of(mol)


def test_in_place_property_edits_are_visible_through_the_cache(scope):
    mol = Chem.MolFromSmiles("CCO")
    cached = molcache.atoms_of(mol)
    mol.GetAtomWithIdx(2).SetAtomicNum(7)
    assert cached[2].GetSymbol() == "N" == mol.GetAtomWithIdx(2).GetSymbol()


def test_verify_mode_passes_when_nothing_changed(scope, monkeypatch):
    monkeypatch.setattr(molcache, "_MODE", "verify")
    mol = Chem.MolFromSmiles("CCO")
    assert molcache.atoms_of(mol) is molcache.atoms_of(mol)
