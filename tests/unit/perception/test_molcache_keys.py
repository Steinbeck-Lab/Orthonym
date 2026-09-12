from rdkit import Chem
from orthonym.perception import molcache
from orthonym.assembly import memo


def test_inchikey_of_equals_rdkit_and_caches():
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    tok = memo.push_scope()
    try:
        k1 = molcache.inchikey_of(mol)
        assert k1 == Chem.MolToInchiKey(mol)
        cache = memo._cache_var.get()
        assert any(k[0] == "molcache.inchikey" for k in cache)
        assert molcache.inchikey_of(mol) is k1          # same object -> served from cache
    finally:
        memo.pop_scope(tok)


def test_inchikey_of_rwmol_not_cached():
    rw = Chem.RWMol(Chem.MolFromSmiles("CCO"))
    tok = memo.push_scope()
    try:
        assert molcache.inchikey_of(rw) == Chem.MolToInchiKey(rw)
        assert not any(k[0] == "molcache.inchikey" for k in memo._cache_var.get())
    finally:
        memo.pop_scope(tok)


def test_inchikey_of_without_scope():
    mol = Chem.MolFromSmiles("c1ccncc1")
    assert memo._cache_var.get() is None
    assert molcache.inchikey_of(mol) == Chem.MolToInchiKey(mol)


def test_canon_smiles_equals_rdkit():
    for smi in ("OCC", "C(C)O", "c1ccccc1O", "Oc1ccccc1"):
        assert molcache.canon_smiles(smi) == Chem.CanonSmiles(smi)


def test_inchikey_of_sees_inplace_charge_edit():
    mol = Chem.MolFromSmiles("CC(=O)O")
    tok = memo.push_scope()
    try:
        k_acid = molcache.inchikey_of(mol)
        mol.GetAtomWithIdx(3).SetFormalCharge(-1)      # in-place edit on the SAME object
        mol.GetAtomWithIdx(3).SetNumExplicitHs(0)
        Chem.SanitizeMol(mol)
        k_anion = molcache.inchikey_of(mol)
        assert k_anion == Chem.MolToInchiKey(mol) and k_anion != k_acid
    finally:
        memo.pop_scope(tok)
