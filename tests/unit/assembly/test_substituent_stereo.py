"""Wave-0 unit tests for WSB-02 substituent-stereo placement (Phase 177 Plan 02).

Covers ``substituent_naming._add_substituent_stereo``:

- Single-centre branch returns ``(R)-<name>`` with NO locant prefix (the
  correct form — this is what STEREO-06's mononuclear parent reuses at the
  handler level).
- Multi-centre branch uses the substituent's OWN threaded numbering, never the
  raw atom-index positions of the WR-06 bug (D-07). When the substituent's own
  numbering cannot be threaded, NO stereo block is emitted (D-09: missing > wrong).

These tests are RED until Task 2 replaces the WR-06 raw-index block.
"""

import pytest
from rdkit import Chem

from orthonym.assembly.substituent_naming import _add_substituent_stereo
from orthonym.perception.stereo import assign_stereochemistry


pytestmark = pytest.mark.unit


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"invalid SMILES {smiles!r}"
    assign_stereochemistry(mol)
    return mol


class TestSingleCentre:
    """Single stereocenter -> (R)-/(S)- with no locant."""

    def test_single_centre_no_locant(self):
        # sec-butyl fragment: one stereocenter -> "(R)-name" / "(S)-name".
        mol = _mol("CC[C@H](C)O")
        stereo_idx = [a.GetIdx() for a in mol.GetAtoms()
                      if a.HasProp("_CIPCode")]
        assert len(stereo_idx) == 1
        sub_atoms = [a.GetIdx() for a in mol.GetAtoms()
                     if a.GetSymbol() == "C"]
        out = _add_substituent_stereo(mol, sub_atoms, "butan-2-yl")
        # exactly one of (R)/(S), no numeric locant before it
        assert out in ("(R)-butan-2-yl", "(S)-butan-2-yl"), out


class TestMultiCentre:
    """Multi-centre: own-numbering, never raw atom-index; unthreadable -> none."""

    def test_multi_centre_unthreadable_emits_no_block(self):
        # A two-stereocentre fragment where no substituent-own numbering can be
        # threaded (we pass a bare atom-set with no numbering context) MUST emit
        # NO stereo block (D-09: missing beats wrong) — NOT the old raw-index
        # "(1R,2S)-" guess.
        mol = _mol("C[C@H](Cl)[C@@H](Br)C")  # 2 stereocentres
        stereo_idx = [a.GetIdx() for a in mol.GetAtoms()
                      if a.HasProp("_CIPCode")]
        assert len(stereo_idx) == 2
        sub_atoms = [a.GetIdx() for a in mol.GetAtoms()]
        out = _add_substituent_stereo(mol, sub_atoms, "some-substituent")
        # The legacy WR-06 path would have prepended a raw-index "(1R,2S)-"
        # block. The fix emits NO block when own-numbering is unthreadable.
        assert not out.startswith("("), out
        assert out == "some-substituent", out

    def test_multi_centre_does_not_use_raw_atom_index(self):
        # Whatever the multi-centre branch does, it must NOT key locants off the
        # raw sorted atom-index (the WR-06 bug numbers a leading heteroatom "1").
        # We assert the output is either a clean name (no block) or a block whose
        # locants are NOT the raw 1..N enumerate positions of sorted atom idxs.
        mol = _mol("Cl[C@H](C)[C@@H](C)Br")
        sub_atoms = sorted(a.GetIdx() for a in mol.GetAtoms())
        out = _add_substituent_stereo(mol, sub_atoms, "frag")
        # raw-index path would produce a fabricated "(<pos><cip>,...)-frag"
        # block keyed off enumerate(sorted(sub_atoms)). The fix must emit NO
        # fabricated block when the substituent's own numbering is unthreadable.
        assert out == "frag", out
