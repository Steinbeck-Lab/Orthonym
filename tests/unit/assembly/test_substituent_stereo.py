"""Wave-0 unit tests for -02 substituent-stereo placement (a phase Plan 02).

Covers ``substituent_naming._add_substituent_stereo``:

- Single-centre branch returns ``(R)-<name>`` with NO locant prefix (the
  correct form — this is what STEREO-06's mononuclear parent reuses at the
  handler level).
- Multi-centre branch uses the substituent's OWN threaded numbering, never the
  raw atom-index positions of the bug . When the substituent's own
  numbering cannot be threaded, NO stereo block is emitted (: missing > wrong).

These tests are RED until Task 2 replaces the raw-index block.
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
        # NO stereo block (: missing beats wrong) — NOT the old raw-index
        # "(1R,2S)-" guess.
        mol = _mol("C[C@H](Cl)[C@@H](Br)C")  # 2 stereocentres
        stereo_idx = [a.GetIdx() for a in mol.GetAtoms()
                      if a.HasProp("_CIPCode")]
        assert len(stereo_idx) == 2
        sub_atoms = [a.GetIdx() for a in mol.GetAtoms()]
        out = _add_substituent_stereo(mol, sub_atoms, "some-substituent")
        # The legacy path would have prepended a raw-index "(1R,2S)-"
        # block. The fix emits NO block when own-numbering is unthreadable.
        assert not out.startswith("("), out
        assert out == "some-substituent", out

    def test_multi_centre_does_not_use_raw_atom_index(self):
        # Whatever the multi-centre branch does, it must NOT key locants off the
        # raw sorted atom-index (the bug numbers a leading heteroatom "1").
        # We assert the output is either a clean name (no block) or a block whose
        # locants are NOT the raw 1..N enumerate positions of sorted atom idxs.
        mol = _mol("Cl[C@H](C)[C@@H](C)Br")
        sub_atoms = sorted(a.GetIdx() for a in mol.GetAtoms())
        out = _add_substituent_stereo(mol, sub_atoms, "frag")
        # raw-index path would produce a fabricated "(<pos><cip>,...)-frag"
        # block keyed off enumerate(sorted(sub_atoms)). The fix must emit NO
        # fabricated block when the substituent's own numbering is unthreadable.
        assert out == "frag", out


class TestMultiCentreThreadedViaAcyclicAlkyl:
    """ a phase L3-2a: when ``attach_idx`` IS known and the fragment is a
    plain (optionally hydroxy/halogen/amino-decorated) ACYCLIC alkyl chain,
    ``_acyclic_alkyl_located_stereo_name`` (the SAME deriver the single-centre
    branch already trusts) exposes the substituent's own /.12
    chain-position map, so the multi-centre branch can now thread it instead
    of unconditionally falling back to 's 'missing beats wrong'.

    Root cause + measured 7-row bucket: this is the class that made the T4
    floor omit real defined stereo on a decorated glycoside/lipid/steroid
    side-chain substituent ('1,2,3-trihydroxypropyl', '2,3,4-trihydroxybutyl',
    '5-(propan-2-yl)heptan-2-yl'...), which fails full-InChIKey RT and is
    voided by the L3-1 offer gate.
    """

    def test_unbranched_two_centre_chain_gets_located_block(self):
        # C7=attach(S), O8, C9(R), O10, C11(terminal CH2OH) -- exactly the
        # 'sugar side-chain' shape (row 0 / row 4 of the L3-2a bucket).
        mol = _mol("OC1CCCCC1[C@H](O)[C@H](O)CO")
        sub_atoms = [7, 8, 9, 10, 11, 12]
        attach_idx = 7
        out = _add_substituent_stereo(
            mol, sub_atoms, "1,2,3-trihydroxypropyl", attach_idx=attach_idx)
        assert out == "(1S,2R)-1,2,3-trihydroxypropyl", out

    def test_branched_chain_free_valence_not_locant_one(self):
        # Free valence at C2 of a branched heptyl chain, with BOTH
        # stereocentres off the attachment atom -- the steroid side-chain
        # shape (row 7/10/16 of the L3-2a bucket: free valence at locant 2,
        # not 1). Locants must come from the chain's OWN numbering,
        # never a raw atom-index guess (attach_idx=1 -> locant 2; the other
        # stereocentre at raw atom-index 4 -> locant 5, NOT enumerate
        # position 5 of sorted(sub_atoms), which this fragment's shape makes
        # coincide -- the assembly-site regression test below uses a
        # genuinely non-coincident shape).
        mol = _mol("C[C@H](CC[C@H](C(C)C)CC)C1CCCCC1")
        stereo_idx = [a.GetIdx() for a in mol.GetAtoms()
                      if a.HasProp("_CIPCode")]
        assert len(stereo_idx) == 2
        attach_idx = 1
        sub_atoms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        from orthonym.assembly.substituent_naming import (
            _acyclic_alkyl_located_stereo_name)
        located = _acyclic_alkyl_located_stereo_name(mol, sub_atoms, attach_idx)
        assert located is not None
        pin_form, k, pos = located
        assert k == 2, "free valence must NOT be locant 1 for this shape"
        out = _add_substituent_stereo(
            mol, sub_atoms, pin_form, attach_idx=attach_idx)
        expected_locants = sorted(
            (pos[i], mol.GetAtomWithIdx(i).GetProp("_CIPCode"))
            for i in stereo_idx)
        expected_block = "(" + ",".join(
            f"{loc}{cip}" for loc, cip in expected_locants) + ")-"
        assert out == f"{expected_block}{pin_form}", out

    def test_unthreadable_shape_still_falls_back_unchanged(self):
        # A RING stereocentre (not an acyclic alkyl chain) -- the deriver
        # declines (None) and 's fallback must still hold: no fabricated
        # block, even though attach_idx is now provided.
        mol = _mol("C[C@H]1CC[C@@H](C)CC1")
        stereo_idx = [a.GetIdx() for a in mol.GetAtoms()
                      if a.HasProp("_CIPCode")]
        assert len(stereo_idx) == 2
        sub_atoms = [a.GetIdx() for a in mol.GetAtoms()]
        out = _add_substituent_stereo(
            mol, sub_atoms, "4-methylcyclohexyl", attach_idx=stereo_idx[0])
        assert out == "4-methylcyclohexyl", out
