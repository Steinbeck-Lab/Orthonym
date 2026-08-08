"""Composer #2 — chain-rooted ring substituent with MULTI-ATOM ring decorations.

v30 sub-lever A, first increment. `_compound_ring_on_chain_substituent` folded only
DEGREE-1 ring decorations into the ring-yl, so a multi-atom decoration (methoxy = O-CH3,
isopropyl, ...) landed in the carrier and broke the carrier-path check. Under the
best-effort tier (allow_mancude) fold every non-carrier decoration subgraph into the
recursively-named ring-yl, so `(4-methoxyphenyl)methyl` and the whole decorated-aryl/
decorated-cycloalkyl-on-a-simple-carrier class names.

PIN default (allow_mancude=False) stays byte-identical (the sentinel), so no gold risk.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent


def _frag(smi):
    """Return (mol, frag_atoms, free_valence_idx) from a `[*]`-marked dummy parent."""
    m = Chem.MolFromSmiles(smi)
    assert m is not None, smi
    dummy = next(a for a in m.GetAtoms() if a.GetAtomicNum() == 0)
    fv = dummy.GetNeighbors()[0].GetIdx()
    frag = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() != 0]
    return m, frag, fv


# (dummy-parent SMILES, expected best-effort prefix)
BEST_EFFORT_CASES = [
    ("[*]Cc1ccc(OC)cc1", "(4-methoxyphenyl)methyl"),
    ("[*]Cc1ccccc1OC", "(2-methoxyphenyl)methyl"),
    ("[*]Cc1ccc(OC)cc1OC", "(2,4-dimethoxyphenyl)methyl"),
    ("[*]Cc1ccc(C(C)C)cc1", "[4-(propan-2-yl)phenyl]methyl"),
    ("[*]CC1CCC(OC)CC1", "(4-methoxycyclohexyl)methyl"),
]


@pytest.mark.parametrize("smi,expected", BEST_EFFORT_CASES)
def test_best_effort_names_decorated_ring_on_chain(smi, expected):
    m, frag, fv = _frag(smi)
    assert name_substituent(m, frag, fv, allow_mancude=True) == expected


@pytest.mark.parametrize("smi,_expected", BEST_EFFORT_CASES)
def test_pin_default_unchanged_byte_identical(smi, _expected):
    # PIN default must NOT gain these (best-effort-only scope -> 0 gold risk):
    # a multi-atom ring decoration keeps the historical sentinel refusal.
    m, frag, fv = _frag(smi)
    assert name_substituent(m, frag, fv, allow_mancude=False) == "substituent"


# regressions that must keep working (degree-1 decoration folds at BOTH tiers;
# 2-carbon carrier already emitted via the chain path — must stay identical).
CONTROL_CASES = [
    ("[*]Cc1ccc(Cl)cc1", "(4-chlorophenyl)methyl", False),
    ("[*]Cc1ccc(Cl)cc1", "(4-chlorophenyl)methyl", True),
    ("[*]CCc1ccc(OC)cc1", "2-(4-methoxyphenyl)ethyl", True),
    ("[*]CC1CCC(Br)CC1", "(4-bromocyclohexyl)methyl", True),
]


@pytest.mark.parametrize("smi,expected,am", CONTROL_CASES)
def test_controls_unchanged(smi, expected, am):
    m, frag, fv = _frag(smi)
    assert name_substituent(m, frag, fv, allow_mancude=am) == expected
