"""Unit tests for the lipid backbone detector (a phase, -01).

`detect_lipid_backbone(mol)` returns a structured BackboneMatch (family in
{"glyceride","phospholipid","sphingolipid"}, acyl/phospho sites, head group,
free-OH set, backbone_atom_to_locant map) or None on any dirty/unrecognized
decoration (hard gate / fail-safe).

WAVE 0 CONTRACT: imports the not-yet-built `detect_lipid_backbone` INSIDE each
test body so `pytest --collect-only` succeeds; RED at run time until Wave 1.
"""

import pytest
from rdkit import Chem


def _mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"bad test SMILES: {smiles}"
    return m


class TestDetectorPositives:
    def test_detects_each_family(self):
        from orthonym.perception.lipids import detect_lipid_backbone

        # glyceride (triacylglycerol)
        tag = detect_lipid_backbone(_mol(
            "CCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCC"))
        assert tag is not None and tag.family == "glyceride"

        # phospholipid (phosphatidylethanolamine)
        pe = detect_lipid_backbone(_mol(
            "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(O)(=O)OCCN)OC(=O)CCCCCCCCCCCCCCC"))
        assert pe is not None and pe.family == "phospholipid"

        # sphingolipid (ceramide)
        cer = detect_lipid_backbone(_mol(
            "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](CO)NC(=O)CCCCCCCCCCCCCCC"))
        assert cer is not None and cer.family == "sphingolipid"


class TestHardGateNegatives:
    def test_negatives(self):
        from orthonym.perception.lipids import detect_lipid_backbone

        # wax ester — long acyl on a long alkyl, no polyol/sphingoid backbone
        assert detect_lipid_backbone(_mol("CCCCCCCCCCCCCCCC(=O)OCCCCCCCCCCCCCCCC")) is None

        # arbitrary tetraol/erythritol tetraester — NOT propane-1,2,3-triol-shaped
        assert detect_lipid_backbone(_mol(
            "CC(=O)OCC(OC(C)=O)C(OC(C)=O)COC(C)=O")) is None

        # ether lipid — O-alkyl on glycerol, no acyl carbonyl
        assert detect_lipid_backbone(_mol("CCCCCCCCCCCCCCCCOCC(O)CO")) is None

        # plasmalogen-like vinyl ether on glycerol
        assert detect_lipid_backbone(_mol("CCCCCCCCCCCCCC/C=C/OCC(O)CO")) is None

        # plain molecule
        assert detect_lipid_backbone(_mol("CCO")) is None


class TestDeterminism:
    def test_cip_idempotent(self):
        from orthonym.perception.lipids import detect_lipid_backbone
        m = _mol("CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(O)(=O)OCCN)OC(=O)CCCCCCCCCCCCCCC")
        a = detect_lipid_backbone(m)
        b = detect_lipid_backbone(m)
        assert (a is None) == (b is None)
        if a is not None:
            assert a.family == b.family
