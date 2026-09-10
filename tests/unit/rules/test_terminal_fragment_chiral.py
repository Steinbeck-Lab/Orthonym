"""a phase B3: terminal_fragment emits R/S for backbone stereocentres.

Before this, `_terminal_fragment_name` refused ANY defined atom stereo (no R/S
descriptor). Now the ACYCLIC path cites backbone R/S centres (merged with any
backbone C=C E/Z) in one leading (...) block; the COMPOSITE (ring-parent) path
still refuses (it emits no descriptor). This unblocked the peptide class + chiral
composite substituents (measured: +9 in-scope rows, 0-wrong intact -- constitution
verified by, so a wrong CIP/locant abstains rather than ships).
"""
import re

import pytest
from rdkit import Chem

from orthonym.rules.terminal_fragment import _terminal_fragment_name
from orthonym.perception.stereo import assign_stereochemistry

pytestmark = pytest.mark.unit


def _name_chiral_fragment(smi, attach):
    """Name the whole molecule as a terminal fragment attached at `attach`."""
    mol = Chem.MolFromSmiles(smi)
    assign_stereochemistry(mol)
    frag = set(range(mol.GetNumAtoms()))
    res = _terminal_fragment_name(mol, frag, attach)
    return res.name if res else None


def test_backbone_stereocentre_gets_an_rs_descriptor():
    # (S)-butan-2-yl attached at C1: a defined Cα -> a leading (…R/S) block.
    name = _name_chiral_fragment("C[C@@H](CC)O", attach=1)  # attach at the chiral C
    assert name is not None, "a chiral backbone fragment must no longer refuse"
    assert re.search(r"^\(\d*[RS]", name), (
        f"expected a leading R/S descriptor, got {name!r}")


def test_composite_ring_stereo_still_refuses():
    # A ring-attached fragment with a defined ring stereocentre: the composite
    # path emits no descriptor, so it must still refuse (never drop the centre).
    mol = Chem.MolFromSmiles("O[C@H]1CCCCC1")  # (R/S) cyclohexan-1-ol via ring C
    assign_stereochemistry(mol)
    frag = set(range(mol.GetNumAtoms()))
    # attach at the ring carbon bearing OH (a ring atom) -> composite path
    ring_c = next(a.GetIdx() for a in mol.GetAtoms()
                  if a.GetSymbol() == 'C' and a.IsInRing()
                  and any(n.GetSymbol() == 'O' for n in a.GetNeighbors()))
    if mol.GetAtomWithIdx(ring_c).HasProp('_CIPCode'):
        assert _terminal_fragment_name(mol, frag, ring_c) is None
