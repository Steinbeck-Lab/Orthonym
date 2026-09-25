"""A fused-heterocycle substituent namer never names an ester as an acyl group or
as the free acid (pre-existing-failures plan, Task 4 continuation, 2026-09-25;
found on the tropisetron chain, TRIAGE row 36 / canary call 182).

`rules/fused_rings.py::_identify_functionalized_substituent` named a chain by its
carbon count once one SMARTS matched:
- its acyl branch, `[CX3](=O)[#6]`, also matches through the RING carbon, so
  -C(=O)-O-CH3 (two carbons) came out 'acetyl' and -C(=O)-O-CH2CH3 'propanoyl':
  '3-acetyl-1H-indole' for methyl 1H-indole-3-carboxylate;
- its ester branch returned the suffix 'carboxylic acid' with an 'ester_alkyl'
  word that no assembler reads: '1H-indole-3-carboxylic acid' for the ester.

'acetyl' is CH3-CO- (the Blue Book, "acyl groups are formed by subtracting
all -OH groups from oxoacids for example 'acetyl', CH3-CO-"), and an ester is
not its acid. Both are now refused, so the substituent reaches the general
substituent namer or the decomposition route. (:31663): "All
preferred IUPAC names for esters are named by functional class nomenclature."
"""
import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.errors import is_failure_name
from orthonym.rules.fused_rings import (
    _chain_is_unbranched_alkanoyl,
    _identify_functionalized_substituent,
)

pytestmark = pytest.mark.unit


def _ring_core(mol):
    """Atom indices of the ring system the substituent hangs on."""
    return {i for ring in mol.GetRingInfo().AtomRings() for i in ring}


@pytest.mark.parametrize("smiles", [
    "COC(=O)c1c[nH]c2ccccc12",    # methyl ester (was 'acetyl')
    "CCOC(=O)c1c[nH]c2ccccc12",   # ethyl ester (was 'propanoyl')
    "COC(=O)c1cnc2ccccc2c1",       # quinoline
    "COC(=O)c1coc2ccccc12",        # benzofuran
])
def test_an_ester_is_not_an_acyl_or_acid_substituent(smiles):
    mol = Chem.MolFromSmiles(smiles)
    carbonyl = next(a.GetIdx() for a in mol.GetAtoms()
                    if a.GetSymbol() == "C" and not a.IsInRing()
                    and any(n.GetIsAromatic() for n in a.GetNeighbors()))
    core = _ring_core(mol)
    assert _identify_functionalized_substituent(mol, carbonyl, core) is None


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)c1c[nH]c2ccccc12", "acetyl"),
    ("CCC(=O)c1c[nH]c2ccccc12", "propanoyl"),
])
def test_a_true_alkanoyl_keeps_its_name(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    carbonyl = next(a.GetIdx() for a in mol.GetAtoms()
                    if a.GetSymbol() == "C" and not a.IsInRing()
                    and any(n.GetIsAromatic() for n in a.GetNeighbors()))
    core = _ring_core(mol)
    info = _identify_functionalized_substituent(mol, carbonyl, core)
    assert info is not None and info["name"] == expected


@pytest.mark.parametrize("smiles,is_alkanoyl", [
    ("CC(=O)c1ccccc1", True),       # acetyl
    ("CCC(=O)c1ccccc1", True),      # propanoyl
    ("COC(=O)c1ccccc1", False),     # methoxycarbonyl (ester)
    ("ClCC(=O)c1ccccc1", False),    # chloroacetyl
    ("C=CC(=O)c1ccccc1", False),    # prop-2-enoyl
])
def test_chain_is_unbranched_alkanoyl(smiles, is_alkanoyl):
    mol = Chem.MolFromSmiles(smiles)
    ring = {i for r in mol.GetRingInfo().AtomRings() for i in r}
    chain = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in ring]
    acyl_c = next(a.GetIdx() for a in mol.GetAtoms()
                  if a.GetIdx() not in ring and a.GetSymbol() == "C"
                  and any(n.GetIdx() in ring for n in a.GetNeighbors()))
    assert _chain_is_unbranched_alkanoyl(mol, chain, acyl_c) is is_alkanoyl


@pytest.mark.parametrize("smiles", [
    "COC(=O)c1c[nH]c2ccccc12",
    "CCOC(=O)c1c[nH]c2ccccc12",
    "COC(=O)c1cnc2ccccc2c1",
    "COC(=O)c1coc2ccccc12",
    "COC(=O)c1cccc2[nH]ccc12",
    "COC(=O)c1cn(C)c2ccccc12",
])
def test_fused_heteroaryl_ester_name_describes_the_input(smiles):
    """With the gate OFF (the suite default, the raw producers): the PIN tier
    ships an RT-exact name or fails closed -- never the acyl or acid misreading."""
    from tests.support.rt_assert import name_is_rt_exact
    name = name_compound(smiles)
    assert is_failure_name(name) or name_is_rt_exact(name, smiles), name
