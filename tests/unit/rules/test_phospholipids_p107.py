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
# (2S,3R)-2-aminooctadecane-1,3-diol — sphinganine (a Chapter name, no PIN).
SPHINGANINE = "CCCCCCCCCCCCCCC[C@@H](O)[C@@H](N)CO"


def test_sphing_4_enine_smiles_is_2s_3r_4e():
    # Guard the fixture: this SMILES really is the (2S,3R,4E) parent.
    m = Chem.MolFromSmiles(SPHING_4_ENINE)
    rdCIPLabeler.AssignCIPLabels(m)
    cips = {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms()
            if a.HasProp("_CIPCode")}
    assert sorted(cips.values()) == ["R", "S"]  # C3=R, C2=S
    assert any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in m.GetBonds())


def test_sphing_4_enine_takes_its_systematic_name():
    # NPL 2026-11-01: '(4E)-sphing-4-enine' is a Chapter name, no PIN INTRODUCTION, the Blue Book);
    # the book prints the systematic name beside it (:55235), which the engine offers first.
    assert RAW.name(SPHING_4_ENINE) == "(2S,3R,4E)-2-aminooctadec-4-ene-1,3-diol"


@pytest.mark.opsin_gate
def test_sphinganine_takes_its_systematic_name():
    # 2026-11-01 (changed asserted value; advisor ruling, internal notes): was 'sphinganine'. LIPIDS
    # (the Blue Book) sits inside Chapter, and INTRODUCTION (:50943) says "Preferred IUPAC names
    # (PINs) are not identified for the compounds in this Chapter"; (:55227) prefers 'sphinganine' inside
    # lipid nomenclature and prints the systematic name beside it, which OPSIN 2.9.0 reads back to the full InChIKey.
    assert RAW.name(SPHINGANINE) == "(2S,3R)-2-aminooctadecane-1,3-diol"
    row = GATED.name_tiered(SPHINGANINE)
    assert (row["name"], row["tier"]) == ("(2S,3R)-2-aminooctadecane-1,3-diol", "pin_verified"), row


def test_phosphatidylserine_serine_parent_pin():
    # 7c.3: phosphatidylserine is named on the L-serine parent (the
    # carboxylic acid outranks the phosphorus oxoacid,, NOT the phosphate-
    # ester-parent form. Locant/config RT-verified.
    ps = "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OC[C@H](N)C(=O)O)OC(=O)CCCCCCCCCCCCCCCCC"
    assert GATED.name(ps) == (
        "O-{[(2R)-2,3-bis(octadecanoyloxy)propoxy]hydroxyphosphoryl}-L-serine")
