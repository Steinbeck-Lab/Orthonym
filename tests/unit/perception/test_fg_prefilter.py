import pytest
from rdkit import Chem
from orthonym.perception import functional_groups as fg


def _unfiltered(mol):
    out = {}
    for name, pat in fg._COMPILED_FG_SMARTS.items():
        m = mol.GetSubstructMatches(pat, uniquify=True)
        if m:
            out[name] = list(m)
    return out


def _filtered(mol):
    out = {}
    have = fg._mol_elements(mol)
    for name, pat in fg._COMPILED_FG_SMARTS.items():
        if not fg._REQUIRED_ELEMENTS[name] <= have:
            continue
        m = mol.GetSubstructMatches(pat, uniquify=True)
        if m:
            out[name] = list(m)
    return out


SMILES = ["CC(=O)O", "CCN", "CS(=O)(=O)N", "c1ccccc1O", "C[N+](C)(C)C", "OP(=O)(O)O", "CC#N", "C=CC=O",
          "CN=[N+]=[N-]", "CC(=O)OC(=O)C", "C[S+](C)C", "CB(O)O", "C[Si](C)(C)C", "C[Se]C", "ClC(=O)C", "[O-][n+]1ccccc1",
          "CC(=O)NN", "CCCC(NN)C1CCC(C)C1", "NNc1ccc(C2(Cl)CC2)nn1", "C(=O)(O)C(=O)O", "OS(=O)(=O)O", "CNC(=O)N", "C1CC1", "N#N", "O=O", "[Na+].[Cl-]"]


@pytest.mark.parametrize("smi", SMILES)
def test_prefilter_never_drops_a_match(smi):
    mol = Chem.MolFromSmiles(smi)
    assert _filtered(mol) == _unfiltered(mol)


def test_required_elements_conservative():
    # every required element symbol must literally occur in the SMARTS
    pt = Chem.GetPeriodicTable()
    for name, req in fg._REQUIRED_ELEMENTS.items():
        smarts = fg.FUNCTIONAL_GROUP_SMARTS[name]
        for z in req:
            sym = pt.GetElementSymbol(z)
            assert sym.lower() in smarts.lower() or f"#{z}" in smarts, (name, sym, smarts)


def test_prefilter_skips_something():
    # a plain alkane needs no N/O/S/P/halogen patterns: the filter must remove most of the table
    mol = Chem.MolFromSmiles("CCCCCC")
    have = fg._mol_elements(mol)
    kept = sum(1 for n in fg._COMPILED_FG_SMARTS if fg._REQUIRED_ELEMENTS[n] <= have)
    assert kept < len(fg._COMPILED_FG_SMARTS) // 2
