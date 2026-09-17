import pytest
from rdkit import Chem
from orthonym.rules.purine import name_substituted_purine

pytestmark = pytest.mark.opsin_gate


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


def test_26_diaminopurine():
    # di-amino purine (amino at C2 AND C6): widens the "substituted" predicate
    # to catch len(amino) > 1, since neither amino alone would trip the old
    # c/n_substituents/other/suffix_groups check -> it used to fall through to
    # the wrong retained-core path. Indicated H is derived per structure (this
    # engine never hardcodes 9H): the two SMILES below are the 7H and 9H
    # tautomers of the SAME molecule (RDKit an InChIKey
    # for both -- confirmed by RDKit InChI's mobile-H layer, exactly like
    # adenine's own 7H/9H pair below), and OPSIN round-trips EACH engine output
    # back to that identical InChIKey (verified via in-process opsin_stdout).
    mol_7h = _mol("Nc1nc(N)c2[nH]cnc2n1")
    mol_9h = _mol("Nc1nc(N)c2nc[nH]c2n1")
    assert Chem.MolToInchiKey(mol_7h) == "MSSXOMSJDRHRMC-UHFFFAOYSA-N"
    assert Chem.MolToInchiKey(mol_9h) == "MSSXOMSJDRHRMC-UHFFFAOYSA-N"
    assert name_substituted_purine(mol_7h) == "7H-purine-2,6-diamine"
    assert name_substituted_purine(mol_9h) == "9H-purine-2,6-diamine"


def test_non_purine_declines():
    assert name_substituted_purine(_mol("c1ccccc1")) is None
    assert name_substituted_purine(_mol("CCO")) is None


def test_9_methylguanine_declines():
    # amino+oxo: the shared collector's suffix builder silently drops the oxo
    # when amino is present, which would name a DIFFERENT molecule -> decline
    # (Tier 1 is non-oxo; oxo purines defer to purine_oxo.py / a later task)
    mol = _mol("Cn1cnc2c1nc(N)[nH]c2=O")
    assert Chem.MolToInchiKey(mol) == "UUWJNBOCAPUTBK-UHFFFAOYSA-N"  # 9-methylguanine
    assert name_substituted_purine(mol) is None


def test_caffeine_declines():
    # oxo-bearing purine-2,6-dione: this producer is non-oxo-only and defers
    # to rules/purine_oxo.py, which knows the correct 3,7-dihydro-1H- spelling
    mol = _mol("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
    assert Chem.MolToInchiKey(mol) == "RYYVLZVUVIJVGH-UHFFFAOYSA-N"  # caffeine
    assert name_substituted_purine(mol) is None


def test_unclassifiable_substituent_fails_closed():
    # a boronic-acid ring substituent is NOT typed by _identify_fused_substituent
    # (returns None) -> _exocyclic_atoms_accounted must catch it and decline,
    # rather than silently omit it and name a different (des-boronic) molecule
    assert name_substituted_purine(_mol("OB(O)c1ncnc2[nH]cnc12")) is None


def test_9_methyladenine_end_to_end():
    from orthonym import Orthonym
    smi = "Cn1cnc2c(N)ncnc21"
    name = Orthonym().name(smi)
    assert name == "9-methyl-9H-purin-6-amine", name


def test_bare_adenine_unchanged_end_to_end():
    from orthonym import Orthonym
    # bare adenine still gets its retained name (standard tautomer)
    assert Orthonym().name("Nc1ncnc2nc[nH]c12") == "adenine"


def test_26_diaminopurine_end_to_end():
    from orthonym import Orthonym
    assert Orthonym().name("Nc1nc(N)c2[nH]cnc2n1") == "7H-purine-2,6-diamine"
    assert Orthonym().name("Nc1nc(N)c2nc[nH]c2n1") == "9H-purine-2,6-diamine"
