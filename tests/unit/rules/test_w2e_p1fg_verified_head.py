"""Regression pins for six Wave-2 C2 rows already healed at HEAD (2026-07-09).

P-62.3.1.1 (BB 26510): "All imines are named substitutively using the
suffix 'imine'".
P-63.1.5 (BB 27274): "Sulfur, selenium, and tellurium analogues of hydroxy
compounds are named substitutively using the suffixes 'thiol', 'selenol',
and 'tellurol', and the prefixes 'sulfanyl', 'selanyl', and 'tellanyl'".
P-66.1.1.4.2/.4.3 (BB 33019): sulfonamide final e->o -> methanesulfonamido.
P-66.1.6.2 (BB 33531): (propan-2-yl)cyanamide, no N-locants.
P-66.4.1.3.2 (BB 34338): chain-terminal amidine C -> amino + imino prefixes.
P-66.4.1.6 (BB 34424): conjoined diamidine -> N-ethanimidoylethanimidamide.
"""
import pytest
from orthonym.namer import name_compound


def _pin(smiles):
    return name_compound(smiles, style="pin")


@pytest.mark.unit
class TestW2EP1FGVerifiedAtHead:
    def test_imine_suffix_ethanimine(self):            # P-62.3.1.1
        assert _pin("CC=N") == "ethanimine"

    def test_imine_suffix_methanimine(self):           # P-62.3.1.1
        assert _pin("C=N") == "methanimine"

    def test_imine_suffix_ring_cyclohexanimine(self):  # P-62.3.1.1
        assert _pin("N=C1CCCCC1") == "cyclohexan-1-imine"

    def test_selenol_suffix(self):                     # P-63.1.5
        assert _pin("CC[SeH]") == "ethaneselenol"

    def test_selanyl_prefix_with_senior_acid(self):    # P-63.1.5
        assert _pin("OC(=O)CC[SeH]") == "3-selanylpropanoic acid"

    def test_selanyl_prefix_with_senior_ol(self):      # P-63.1.5
        assert _pin("OCC[SeH]") == "2-selanylethan-1-ol"

    def test_sulfonamido_prefix(self):                 # P-66.1.1.4.2 (AM-6)
        assert _pin("CS(=O)(=O)NCCC(=O)O") == "3-(methanesulfonamido)propanoic acid"

    def test_sulfamoyl_orientation_protected(self):    # AM-6 guard (S-attached)
        assert _pin("O=S(=O)(N)CCC(=O)O") == "3-sulfamoylpropanoic acid"

    def test_cyanamide_parent(self):                   # P-66.1.6.2 (AM-1)
        assert _pin("CC(C)NC#N") == "(propan-2-yl)cyanamide"

    def test_chain_amidine_amino_imino(self):          # P-66.4.1.3.2 (AM-4)
        assert _pin("CCN=C(CCC(=O)OC)N(C)C") == \
            "methyl 4-(dimethylamino)-4-(ethylimino)butanoate"

    def test_diamidide_n_imidoyl(self):                # P-66.4.1.6 (AM-5)
        assert _pin("CC(=N)NC(C)=N") == "N-ethanimidoylethanimidamide"
