"""P7 (Wave-8) sub-plan 7c — sphingolipid / phospholipid PINs.

7c.1 — (4E)-sphing-4-enine.

Blue Book: the retained name 'sphinganine' is preferred to the
systematic (2S,3R)-2-aminooctadecane-1,3-diol, and it generates the preferred
names of its unsaturated / N- / O-substituted derivatives. The listed example
is ``(4E)-sphing-4-enine`` for ``(2S,3R,4E)-2-aminooctadec-4-ene-1,3-diol``
(the Blue Book). The common name 'sphingosine' is NOT a Blue Book name (0
occurrences); Orthonym currently emits it from the non-PIN OPSIN simpleGroup
import — the PIN is ``(4E)-sphing-4-enine``.

Both names are OPSIN-unparseable → name-exact carve-out. BB is explicit that
different chain lengths / other diastereomers are named systematically, so
recognition is exact-canonical-SMILES only (fail closed otherwise).
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym.namer import Orthonym

RAW = Orthonym(_disable_opsin_validity_gate=True)
GATED = Orthonym()

# (2S,3R,4E)-2-aminooctadec-4-ene-1,3-diol — verified CIP below.
SPHING_4_ENINE = "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@@H](N)CO"
# (2S,3R)-2-aminooctadecane-1,3-diol — sphinganine (retained PIN, unchanged).
SPHINGANINE = "CCCCCCCCCCCCCCC[C@@H](O)[C@@H](N)CO"


def test_sphing_4_enine_smiles_is_2s_3r_4e():
    # Guard the fixture: this SMILES really is the (2S,3R,4E) parent.
    m = Chem.MolFromSmiles(SPHING_4_ENINE)
    rdCIPLabeler.AssignCIPLabels(m)
    cips = {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms()
            if a.HasProp("_CIPCode")}
    assert sorted(cips.values()) == ["R", "S"]  # C3=R, C2=S
    assert any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in m.GetBonds())


def test_sphing_4_enine_is_pin_not_sphingosine():
    # PIN, replacing the non-BB common name 'sphingosine'.
    assert RAW.name(SPHING_4_ENINE) == "(4E)-sphing-4-enine"


def test_sphinganine_retained_name_unchanged():
    # Saturated C18 parent keeps its retained PIN (regression guard).
    assert RAW.name(SPHINGANINE) == "sphinganine"


def test_phosphatidylserine_serine_parent_pin():
    # 7c.3: phosphatidylserine is named on the L-serine parent (the
    # carboxylic acid outranks the phosphorus oxoacid,, NOT the phosphate-
    # ester-parent form. Locant/config RT-verified.
    ps = "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OC[C@H](N)C(=O)O)OC(=O)CCCCCCCCCCCCCCCCC"
    assert GATED.name(ps) == (
        "O-{[(2R)-2,3-bis(octadecanoyloxy)propoxy]hydroxyphosphoryl}-L-serine")
