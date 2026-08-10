"""v31 (P-31.1.4.2.4 / P-73.1.2.1): a quaternary ammonium whose parent branch is
substituted must number that branch with the N-attached carbon as C1 (the amine
principal group takes the lowest locant). Regression: `CC[N+](CC)(CC)CCF` emitted
`1-fluoro-N,N,N-triethylethanaminium` (fluoro at C1 — the chain numbered from the
wrong end when equal-length ethyl co-branches competed); correct is `2-fluoro-...`.
Mission-corpus (drug-like PubChem) bounded lever — see DRUGLIKE-BASELINE-2026-08-10.md.
"""
import pytest
from rdkit import Chem
from orthonym.rules.ions import name_quaternary_aminium
from orthonym.namer import _validity_gate_name_to_smiles


def _site(mol):
    for a in mol.GetAtoms():
        if (a.GetSymbol() == "N" and a.GetFormalCharge() == 1
                and a.GetTotalNumHs() == 0 and a.GetDegree() >= 4):
            return {"atom_idx": a.GetIdx()}
    return None


def _canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


@pytest.mark.parametrize("smiles", [
    "CC[N+](CC)(CC)CCF",   # 2-fluoro-N,N,N-triethylethan-1-aminium (the regression)
    "FCC[N+](C)(C)C",      # 2-fluoro-N,N,N-trimethylethan-1-aminium (was already OK)
    "ClCC[N+](CC)(CC)CC",  # 2-chloro-N,N,N-triethyl... (mixed-branch chloro)
    "OCC[N+](CC)(CC)CC",   # 2-hydroxy-N,N,N-triethyl... (mixed-branch hydroxy)
])
def test_substituted_quaternary_aminium_locant_rt_exact(smiles):
    mol = Chem.MolFromSmiles(smiles)
    name = name_quaternary_aminium(mol, _site(mol))
    assert name, f"no name for {smiles}"
    opsin = _validity_gate_name_to_smiles(name)
    assert opsin is not None, f"OPSIN could not parse {name!r}"
    assert _canon(opsin) == _canon(smiles), (
        f"NOT rt_exact: {name!r} -> {_canon(opsin)} != {_canon(smiles)}")
