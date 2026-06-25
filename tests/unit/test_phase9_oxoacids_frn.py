"""v23 Phase 9 — P-67 oxoacids + shared functional-replacement (FRN) engine.

Covers:
  * the shared FRN name-builder (``rules.functional_replacement``)
  * organyl-stem phosphonic acids (``rules.phosphorus.name_phosphonic_acid`` +
    the ``phosphonic_acid`` inner handler) — P-67.1.1.2
  * acid-halide / amide functional-class derivatives + carbonic-family FRN acids
    tabled in ``rules.inorganic_acids`` — P-67.1.2 / P-65.2
  * Se/Te suffix acids (P-65.3) end-to-end
All end-to-end expectations are OPSIN-round-trip-confirmed PINs.
"""
import pytest
from rdkit import Chem

from orthonym.rules.functional_replacement import (
    build_acyl_halide_name,
    build_frn_acid_name,
    build_polyacid_name,
)
from orthonym.rules.inorganic_acids import name_inorganic_acid


# --- the shared FRN engine (pure string builder) -----------------------------

@pytest.mark.parametrize("base,infix,count,expected", [
    ("carbon", "peroxo", 1, "carbonoperoxoic acid"),
    ("carbon", "thio", 2, "carbonodithioic acid"),
    ("carbon", "thio", 3, "carbonotrithioic acid"),
    ("carbon", "imido", 1, "carbonimidic acid"),
    ("carbam", "imido", 1, "carbamimidic acid"),
    ("carbam", "thio", 1, "carbamothioic acid"),
    ("phosphor", "thio", 1, "phosphorothioic acid"),
    ("sulfur", "thio", 1, "sulfurothioic acid"),
])
def test_build_frn_acid_name(base, infix, count, expected):
    assert build_frn_acid_name(base, infix, count) == expected


def test_build_frn_acid_name_unknown_infix_fail_closed():
    assert build_frn_acid_name("carbon", "bogus", 1) is None


def test_build_polyacid_name():
    assert build_polyacid_name("carbonic acid", 2) == "dicarbonic acid"
    assert build_polyacid_name("carbonic acid", 3) == "tricarbonic acid"
    assert build_polyacid_name("carbonic acid", 1) is None  # single -> not multiplicative


def test_build_acyl_halide_name():
    assert build_acyl_halide_name("phosphoryl", "chloride", 3) == "phosphoryl trichloride"
    assert build_acyl_halide_name("sulfuryl", "chloride", 2) == "sulfuryl dichloride"
    assert build_acyl_halide_name("phosphorothioyl", "chloride", 3) == "phosphorothioyl trichloride"


# --- inorganic_acids table (acid halides / amides / carbonic-FRN) -------------

@pytest.mark.parametrize("smiles,expected", [
    # free acids (regression guard — pre-existing rows)
    ("OP(=O)(O)O", "phosphoric acid"),
    ("OC(=O)O", "carbonic acid"),
    # acid halides
    ("ClP(=O)(Cl)Cl", "phosphoryl trichloride"),
    ("ClS(=O)(=O)Cl", "sulfuryl dichloride"),
    ("ClP(=S)(Cl)Cl", "phosphorothioyl trichloride"),
    # amides
    ("NP(=O)(N)N", "phosphoric triamide"),
    ("NS(=O)(=O)N", "sulfuric diamide"),
    ("NS(=O)(=O)O", "sulfamic acid"),
    # carbonic-family FRN
    ("OOC(=O)O", "carbonoperoxoic acid"),
    ("SC(=O)S", "carbonodithioic acid"),
    ("SC(=S)S", "carbonotrithioic acid"),
    ("N=C(O)O", "carbonimidic acid"),
    ("N=C(N)O", "carbamimidic acid"),
    ("OC(=O)OC(=O)O", "dicarbonic acid"),
])
def test_name_inorganic_acid_table(smiles, expected):
    assert name_inorganic_acid(Chem.MolFromSmiles(smiles)) == expected


def test_name_inorganic_acid_fail_closed():
    # an organyl phosphonic acid is NOT a free inorganic acid -> None (cascades)
    assert name_inorganic_acid(Chem.MolFromSmiles("CCP(=O)(O)O")) is None
    # a phosphate ester likewise
    assert name_inorganic_acid(Chem.MolFromSmiles("COP(=O)(O)O")) is None


# --- end-to-end production names ---------------------------------------------

@pytest.fixture(scope="module")
def namer():
    from orthonym.namer import Orthonym
    return Orthonym()


@pytest.mark.parametrize("smiles,expected", [
    # 9a organyl-stem phosphonic
    ("CP(=O)(O)O", "methylphosphonic acid"),
    ("CCP(=O)(O)O", "ethylphosphonic acid"),
    ("CCCP(=O)(O)O", "propylphosphonic acid"),
    ("OP(=O)(O)c1ccccc1", "phenylphosphonic acid"),
    # phosphinic must NOT regress
    ("CCP(=O)(O)CC", "diethylphosphinic acid"),
    # 9b
    ("ClP(=O)(Cl)Cl", "phosphoryl trichloride"),
    ("NP(=O)(N)N", "phosphoric triamide"),
    ("NS(=O)(=O)N", "sulfuric diamide"),
    # 9c
    ("OOC(=O)O", "carbonoperoxoic acid"),
    ("SC(=S)S", "carbonotrithioic acid"),
    ("OC(=O)OC(=O)O", "dicarbonic acid"),
    # 9d Se/Te suffix
    ("CC[Se](=O)(=O)O", "ethaneselenonic acid"),
    ("CC[Te](=O)O", "ethanetellurinic acid"),
    ("OC(=O)CC[Se](=O)(=O)O", "3-selenonopropanoic acid"),
    # sulfonic must NOT regress
    ("CCS(=O)(=O)O", "ethanesulfonic acid"),
])
def test_production_names(namer, smiles, expected):
    assert namer.name(smiles) == expected
