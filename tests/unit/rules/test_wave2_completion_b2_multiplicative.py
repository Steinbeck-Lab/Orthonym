"""Wave-2 completion batch B2 — multiplicative bridge extensions.

Covers (all expected names OPSIN-RT verified against the evidence SMILES):
  * P-16.5.1.1    heterocyclic multiplied units: attachment locant from the
                  UNIT's own numbering (4,4'-oxybis(1,3-thiazole)).
  * P-15.3.2.4.1  PG-free SUBSTITUTED benzene units renamed with locants
                  anchored at the attachment — the root fix for the
                  structure-dropping '1,1'-oxydibromobenzene' class.
  * P-29.4.2      composite CH2-SiH2-CH2 bridge (silanediylbis(methylene)).
  * P-51.3.2.1 /  two-carbon triyl central unit over three identical ring
    P-45.1.2      parents (ethane/ethene-1,1,2-triyl).
  * P-15.3.1.1    methylenebis(disilane) + phosphanetriyltriacetic acid
                  (BB verbatim).
  * P-15.3.1.2.2.1 [azanediylbis(methylene)]bis(phosphonic acid) (BB verbatim).
"""

import pytest
from rdkit import Chem

from orthonym.rules.multiplicative import name_multiplicative


def _mult(smiles):
    return name_multiplicative(Chem.MolFromSmiles(smiles))


@pytest.mark.unit
class TestHeterocycleUnits:
    def test_oxybis_thiazole(self):
        assert _mult("O(C=1N=CSC1)C=1N=CSC1") == "4,4'-oxybis(1,3-thiazole)"


@pytest.mark.unit
class TestAttachmentAnchoredUnits:
    """P-15.3.2.4.1: units without a PG anchor are renumbered relative to the
    bridge attachment; the free-fragment name silently dropped the
    substituent position before (wrong-name class, RT-gate-masked)."""

    def test_oxybis_bromobenzene(self):
        assert (_mult("Brc1ccc(Oc2ccc(Br)cc2)cc1")
                == "1,1'-oxybis(4-bromobenzene)")

    def test_oxybis_chlorobenzene(self):
        assert (_mult("Clc1ccc(Oc2ccc(Cl)cc2)cc1")
                == "1,1'-oxybis(4-chlorobenzene)")

    def test_oxybis_methylbenzene(self):
        assert (_mult("Cc1ccc(Oc2ccc(C)cc2)cc1")
                == "1,1'-oxybis(4-methylbenzene)")

    def test_two_atom_bridge_sibling(self):
        assert (_mult("Brc1ccc(CCc2ccc(Br)cc2)cc1")
                == "1,1'-(ethane-1,2-diyl)bis(4-bromobenzene)")

    def test_ortho_attachment(self):
        assert (_mult("Brc1ccccc1Oc1ccccc1Br")
                == "1,1'-oxybis(2-bromobenzene)")

    def test_mismatched_positions_fail_closed(self):
        # para on one ring, meta on the other: H-filled fragments compare
        # canonical-equal but the anchored units differ -> decline.
        assert _mult("Brc1ccc(Oc2cccc(Br)c2)cc1") is None

    def test_unsupported_substituent_fails_closed(self):
        # CF3 unit substituent is outside the anchored table -> decline,
        # never a locant-dropping name.
        assert _mult("FC(F)(F)c1ccc(Oc2ccc(C(F)(F)F)cc2)cc1") is None


@pytest.mark.unit
class TestCompositeAndTriylBridges:
    def test_silanediylbis_methylene(self):
        assert (_mult("OC(=O)c1ccc(C[SiH2]Cc2ccc(C(=O)O)cc2)cc1")
                == "4,4'-[silanediylbis(methylene)]dibenzoic acid")

    def test_ethane_triyl_tribenzoic(self):
        smi = ("C(Cc1ccc(C(=O)O)cc1)(c1ccc(C(=O)O)cc1)"
               "c1ccc(C(=O)O)cc1")
        assert _mult(smi) == "4,4',4''-(ethane-1,1,2-triyl)tribenzoic acid"

    def test_ethene_triyl_trianiline(self):
        assert (_mult("Nc1ccc(C(=Cc2ccc(N)cc2)c2ccc(N)cc2)cc1")
                == "4,4',4''-(ethene-1,1,2-triyl)trianiline")


@pytest.mark.unit
class TestAcyclicBridges:
    def test_methylenebis_disilane(self):
        assert (_mult("[SiH3][SiH2]C[SiH2][SiH3]")
                == "1,1'-methylenebis(disilane)")

    def test_branched_silane_fails_closed(self):
        # A branched unit is not a catenated-hydride chain -> decline.
        assert _mult("[SiH3][Si]([SiH3])([SiH3])C[SiH2][SiH3]") is None

    def test_phosphanetriyltriacetic(self):
        assert (_mult("OC(=O)CP(CC(=O)O)CC(=O)O")
                == "2,2',2''-phosphanetriyltriacetic acid")

    def test_azanediylbis_methylene_phosphonic(self):
        assert (_mult("OP(=O)(O)CNCP(=O)(O)O")
                == "[azanediylbis(methylene)]bis(phosphonic acid)")


@pytest.mark.unit
class TestDeclineRowsStayClosed:
    def test_stereo_differing_units_decline(self):
        # P-45.6.2: units differing in R/S are NOT identical -> multiplicative
        # must decline (canonical SMILES carry the stereo tags).
        assert _mult(
            "C[C@@H](CC)c1ccc(Sc2ccc(cc2)[C@H](C)CC)cc1") is None

    def test_asymmetric_bridge_positions_decline(self):
        # P-51.3.3: 1,3- vs 1,2-attachment fragments differ -> decline.
        assert _mult("OC(=O)C1CCCC(CC2CCCCC2C(=O)O)C1") is None


@pytest.mark.unit
class TestEstablishedBehaviourUnchanged:
    @pytest.mark.parametrize("smiles,expected", [
        ("Nc1ccc(Cc2ccc(N)cc2)cc1", "4,4'-methylenedianiline"),
        ("OC(=O)c1ccc(Oc2ccc(C(=O)O)cc2)cc1", "4,4'-oxydibenzoic acid"),
        ("OC(=O)CNCC(=O)O", "2,2'-azanediyldiacetic acid"),
        ("N(CC(=O)O)(CC(=O)O)CC(=O)O", "2,2',2''-nitrilotriacetic acid"),
        ("OCCSCCO", "2,2'-sulfanediyldi(ethan-1-ol)"),
        ("OC(=O)COCCOCC(=O)O",
         "2,2'-[ethane-1,2-diylbis(oxy)]diacetic acid"),
        ("c1ccccc1Sc1ccccc1", "1,1'-sulfanediyldibenzene"),
        ("c1ccccc1Oc1ccccc1", "1,1'-oxydibenzene"),
    ])
    def test_names(self, smiles, expected):
        assert _mult(smiles) == expected
