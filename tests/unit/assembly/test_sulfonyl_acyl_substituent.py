"""P-65.3.1: an S-attached sulfonic/sulfinic acyl fragment (R-SO2- / R-SO-)
names as a `{arene/alkane}sulfonyl` / `...sulfinyl` substituent prefix.

Before this branch `name_substituent_fragment` returned None for an acyl-S
attachment, which fail-closed every recursive caller (notably the disubstituted
azanide emitter P-72.2.2.2.4 for an N-tosyl secondary-amide anion). Cite:
the Blue Book P-65.3.1 (acyl groups of sulfonic/sulfinic acids).
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_naming import (
    _sulfonyl_sulfinyl_acyl_prefix,
    name_substituent_fragment,
)
from orthonym.rules.charged_router import emit_secondary_amine_azanide


def _arm_prefix(smi, s_symbol_expected="S"):
    """Name the whole molecule's single substituent (everything but the dummy
    parent carbon at atom 0) attached through its sulfur."""
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    # Convention for these fixtures: atom 0 is the parent carbon; the sulfur is
    # its sole neighbour; the substituent is every atom except atom 0.
    parent = 0
    s_idx = [nb.GetIdx() for nb in mol.GetAtomWithIdx(parent).GetNeighbors()][0]
    sub_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() != parent]
    return name_substituent_fragment(mol, sub_atoms, s_idx, [parent])


@pytest.mark.unit
@pytest.mark.parametrize(
    "smi,expected",
    [
        # Fixture convention: atom 0 is the parent carbon, bonded directly to S;
        # the substituent (everything but atom 0) attaches through that sulfur.
        ("CS(=O)(=O)C", "methanesulfonyl"),                 # parent-SO2-CH3
        ("CS(=O)(=O)c1ccc(C)cc1", "4-methylbenzene-1-sulfonyl"),  # parent-SO2-tolyl
        ("CS(=O)(=O)c1ccccc1", "benzenesulfonyl"),          # parent-SO2-Ph
        ("CS(=O)C", "methanesulfinyl"),                     # parent-SO-CH3
    ],
)
def test_sulfonyl_sulfinyl_acyl_prefix(smi, expected):
    assert _arm_prefix(smi) == expected


@pytest.mark.unit
def test_azanide_recovers_sulfonyl_arm():
    """The N-(naphthalene-2-sulfonyl) benzothiazole anion now emits the
    disubstituted azanide instead of fail-closing on the unnameable arm."""
    smi = "C1=CC=C2C=C(C=CC2=C1)S(=O)(=O)[N-]C3=NC4=CC=CC=C4S3"
    mol = Chem.MolFromSmiles(smi)
    anion = [a.GetIdx() for a in mol.GetAtoms()
             if a.GetSymbol() == "N" and a.GetFormalCharge() == -1
             and a.GetTotalNumHs() == 0 and not a.GetIsAromatic()][0]
    name = emit_secondary_amine_azanide(mol, anion, "pin")
    assert name == "(1,3-benzothiazol-2-yl)(naphthalene-2-sulfonyl)azanide"


@pytest.mark.unit
def test_fail_closed_non_acyl_sulfur():
    """A plain thioether -S-CH3 (no =O) is NOT a sulfonyl acyl -> None
    (falls through to the existing sulfanyl path, unchanged)."""
    mol = Chem.MolFromSmiles("CSC")  # CH3-S-CH3
    s_idx = 1
    sub_atoms = [1, 2]
    assert _sulfonyl_sulfinyl_acyl_prefix(mol, sub_atoms, s_idx) is None
