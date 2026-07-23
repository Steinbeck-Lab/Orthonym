"""v28 Cluster D — PIN-backlog per-class builds (OPSIN-free, root-cause).

Each target's gold is Blue-Book-verified (research workflow wf_bebb8a05):
  D-1 subordinate thioanhydride  -> acylsulfanyl/oxo split      (P-41, P-35.5.1)
  D-2 no-PCG polyfunctional      -> prefix-only substitutive name (P-41, P-63.6)
  D-3 ring-ester decoration      -> alkoxycarbonyl ring prefix    (P-65.6.3.2.3, P-16.3.3)
  D-4 aromatic nitrile oxide     -> benzonitrile-oxide + prefix   (P-66.5.4.2, P-16.3.3)

All assertions use name_compound (PIN style, OPSIN-free) so the whole file is
safe for touched-module HEAD-A/B. Regression cases lock the blast radius.
"""
import pytest
from rdkit import RDLogger

from orthonym import name_compound

RDLogger.logger().setLevel(RDLogger.ERROR)


class TestD1SubordinateThioanhydride:
    """A -CO-S-CO- linkage SUBORDINATE to a senior acid is expressed
    substitutively (in-chain acyl -> oxo, acyl-S -> acylsulfanyl), never as an
    'anhydride' word (P-41 class seniority: acids > anhydrides)."""

    def test_thioanhydride_of_acid_is_acylsulfanyl_oxo(self):
        # HOOC-(CH2)7-C(=O)-S-C(=O)-CH3 ; BB P-35.5.1 (CH3-CO-S- = acetylsulfanyl)
        assert (name_compound("CC(=O)SC(=O)CCCCCCCC(=O)O")
                == "9-(acetylsulfanyl)-9-oxononanoic acid")

    @pytest.mark.parametrize("smiles,expected", [
        # Genuine anhydrides (anhydride IS the senior class) must be UNCHANGED.
        ("CC(=O)OC(=O)C", "acetic anhydride"),
        ("O=C(OC(=O)c1ccccc1)c1ccccc1", "benzoic anhydride"),
        # Existing thioester-of-diacid precedent (never was folded) unchanged.
        ("CCSC(=O)CCC(=O)O", "4-(ethylsulfanyl)-4-oxobutanoic acid"),
    ])
    def test_genuine_anhydrides_and_thioesters_unchanged(self, smiles, expected):
        assert name_compound(smiles) == expected
