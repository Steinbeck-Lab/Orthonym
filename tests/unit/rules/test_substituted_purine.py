from rdkit import Chem
from orthonym.rules.purine import name_substituted_purine


def _mol(smi):
    return Chem.MolFromSmiles(smi)


def test_9_methyladenine_parent():
    # 9-methyladenine -- the canonical substituted-purine target
    assert name_substituted_purine(_mol("Cn1cnc2c(N)ncnc21")) == "9-methyl-9H-purin-6-amine"


def test_6_chloropurine_no_amine():
    # no amino suffix: parent hydride 9H-purine + 6-chloro prefix
    assert name_substituted_purine(_mol("Clc1ncnc2[nH]cnc12")) == "6-chloro-9H-purine"


def test_bare_adenine_declines():
    # bare adenine keeps its retained name via another path -> this producer declines
    assert name_substituted_purine(_mol("Nc1ncnc2[nH]cnc12")) is None
    assert name_substituted_purine(_mol("Nc1ncnc2nc[nH]c12")) is None


def test_non_purine_declines():
    assert name_substituted_purine(_mol("c1ccccc1")) is None
    assert name_substituted_purine(_mol("CCO")) is None
