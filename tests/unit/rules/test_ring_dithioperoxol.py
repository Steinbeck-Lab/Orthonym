"""-SSH (and -SeSeH, -TeTeH) is named by the suffix 'dithioperoxol', multiplied with 'bis' and enclosed.

IUPAC 2013 ('ethanedithioperoxol (PIN)', the Blue Book) and (e): 'bis' before a name
that begins with 'di' (:7190, 'bis(diazonium) (preselected suffix) (not didiazonium)').
"""

import pytest

from orthonym import name_compound
from orthonym.assembly.naming_utils import suffix_takes_derived_multiplier
from orthonym.validation.spelling.checks_thioperoxol import multiplied_dithioperoxol_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("suffix", ["dithioperoxol", "diselenoperoxol", "ditelluroperoxol"])
def test_dichalcogen_peroxol_takes_bis(suffix):
    assert suffix_takes_derived_multiplier(suffix)


@pytest.mark.parametrize("name,flagged", [
    ("ethane-1,2-didithioperoxol", True),
    ("benzene-1,4-bisdithioperoxol", True),
    ("benzene-1,4-bis(dithioperoxol)", False),
    ("benzenedithioperoxol", False),
])
def test_spelling_check(name, flagged):
    assert (multiplied_dithioperoxol_check(None, name) is not None) is flagged


@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,name", [
    ("SSc1ccccc1", "benzenedithioperoxol"),
    ("SSc1ccc(SS)cc1", "benzene-1,4-bis(dithioperoxol)"),
    ("Cc1ccc(SS)cc1", "4-methylbenzene-1-dithioperoxol"),
    ("SSCCSS", "ethane-1,2-bis(dithioperoxol)"),
    ("NC(=O)c1ccc(SS)c(SS)c1", "3,4-bis(disulfanyl)benzamide"),   #:28021, the amide is senior
    # the selenium and tellurium analogues of Table 6.1 (:27972)
    ("C[Se][SeH]", "methanediselenoperoxol"),
    ("[SeH][Se]CC[Se][SeH]", "ethane-1,2-bis(diselenoperoxol)"),
    ("C[Te][TeH]", "methaneditelluroperoxol"),
    ("[SeH][Se]c1ccccc1", "benzenediselenoperoxol"),
    ("[SeH][Se]c1ccc([Se][SeH])cc1", "benzene-1,4-bis(diselenoperoxol)"),
    ("OC(=O)C[Se][SeH]", "diselanylacetic acid"),
])
def test_names(smiles, name):
    assert name_compound(smiles) == name
