"""A neutral germanium or antimony atom beside ions is cited with the oxidation number zero (``metal_scope.ZERO_VALENT``).

User decision 2026-10-30: literal 'metal(0)' names for PubChem's neutral-metal drawings, at best-effort, never a PIN
, the Blue Book); a reader of the name without the zero balances the charges itself. Germanium and
antimony joined the zero-valent metals on 2026-11-25; their organic compounds keep their parent-hydride names.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("C(C(=O)O)[S-].C(C(=O)O)[S-].[Ge]", "carboxymethanethiolate—germanium(0) (2/1)"),
    ("[Sb].[Cl-].[Cl-].[Cl-].CCCC", "butane—chloride—antimony(0) (1/3/1)"),
])
def test_a_neutral_atom_beside_anions(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("C[Ge](C)(C)C", "tetramethylgermane"),
    ("c1ccc(cc1)[Sb](c1ccccc1)c1ccccc1", "triphenylstibane"),
])
def test_the_organic_compounds_keep_their_pins(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r


@pytest.mark.parametrize("smiles,ok", [
    ("C[Ge](=[Zr])C", True),            # zirconium 2: one of its charges
    ("C[Ge](=[Ti+4])C", False),         # titanium 6: none of its charges, refused as the silicon twin is
    ("C[Si](=[Ti+4])C", False),
])
def test_a_germylidene_bond_counts_toward_the_metal_oxidation_number(smiles, ok):
    """Review of the lane: with germanium in the table, its Ge=M bond stopped counting (the check read 'in the table' as
    'a metal'), six germylidene complexes lost their names and two titanium drawings passed the check."""
    from rdkit import Chem
    from orthonym.perception.metals import bonded_valence_ok
    mol = Chem.MolFromSmiles(smiles)
    metal = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in ("Zr", "Ti"))
    assert bonded_valence_ok(mol, metal) is ok


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("C[Ge](=[Zr])C", "(dimethylgermylidene)zirconium"),
    ("C[Ge](C)=[Hf](Cl)Cl", "dichlorido(dimethylgermylidene)hafnium"),
])
def test_the_germylidene_complexes_keep_their_names(smiles, name):
    assert name_best_effort(smiles)["name"] == name
