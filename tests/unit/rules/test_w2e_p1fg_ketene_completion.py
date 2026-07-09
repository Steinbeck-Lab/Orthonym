"""P-64.5 / P-64.2.2.4 ketene completion.

BB 29446 (P-64.5(3)): "the group =C=O is named in substitutive
nomenclature as 'oxomethylidene'". BB 29292 (P-64.2.2.4): derivatives of
ketene "are named by using the principles for naming ketones".
Scope-decision #4 OPSIN-verified PINs: diphenylethenone,
cyclohexylidenemethanone.
"""
import pytest
from rdkit import Chem
from orthonym.namer import name_compound
from orthonym.rules.ketenes import name_ketene


@pytest.mark.unit
class TestKeteneCompletion:
    def test_diphenylethenone(self):
        assert name_compound("O=C=C(c1ccccc1)c1ccccc1", style="pin") \
            == "diphenylethenone"

    def test_cyclohexylidenemethanone(self):
        assert name_compound("O=C=C1CCCCC1", style="pin") \
            == "cyclohexylidenemethanone"

    def test_ethenone_protect(self):
        assert name_compound("C=C=O", style="pin") == "ethenone"

    def test_dibromoethenone_protect(self):
        assert name_compound("BrC(=C=O)Br", style="pin") == "dibromoethenone"

    def test_alkyl_ketene_still_declines(self):
        # 2-butylhex-1-en-1-one belongs to the general ketone numbering
        # path (P-64.2.2.4), NOT this namer — must stay None here.
        assert name_ketene(Chem.MolFromSmiles("C(CCC)C(=C=O)CCCC")) is None

    def test_mixed_aryl_h_declines(self):
        # PhCH=C=O would need locant/2-phenyl reasoning — fail closed.
        assert name_ketene(Chem.MolFromSmiles("O=C=Cc1ccccc1")) is None
