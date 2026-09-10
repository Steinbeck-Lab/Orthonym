"""a phase BBR-PERC / — previously-missing neutral classes.

Hydroxylamine, selenide/telluride/tellurol, and the free
inorganic oxoacids were silently dropped or mis-cast before 169.7.
Each new SMARTS ships with positive + negative tests (the broadening-SMARTS
false-positive guard per MEMORY feedback_smarts_and_seniority).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.perception.functional_groups import detect_functional_groups as d


def _fgs(smi):
    return set(d(Chem.MolFromSmiles(smi)).keys())


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# --- hydroxylamine (R-NH-OH / R2N-OH) ---
@pytest.mark.unit
@pytest.mark.parametrize("smi", ["CCCNO", "CCCN(O)C", "CN(C)O"])
def test_hydroxylamine_perceived(smi):
    assert "hydroxylamine" in _fgs(smi)


@pytest.mark.unit
@pytest.mark.parametrize("smi", [
    "CCO",        # ethanol — alcohol, NOT hydroxylamine
    "CCN",        # ethylamine — amine, NOT hydroxylamine
    "CC=NO",      # acetaldoxime — oxime, NOT hydroxylamine
    "CC(=O)NO",   # acetohydroxamic acid — hydroxamic, NOT hydroxylamine
])
def test_hydroxylamine_negatives(smi):
    assert "hydroxylamine" not in _fgs(smi)


@pytest.mark.unit
def test_hydroxylamine_naming(namer):
    # gold target
    assert namer.name("CCCNO") == "N-propylhydroxylamine"
    assert namer.name("CN(C)O") == "N,N-dimethylhydroxylamine"


# --- selenide / telluride / tellurol (Se/Te ether analogues) ---
@pytest.mark.unit
@pytest.mark.parametrize("smi,fg", [
    ("CCC[Se]C", "selenoether"),
    ("CC[Te]C", "telluroether"),
    ("CC[TeH]", "tellurol"),
])
def test_chalcogen_ethers_perceived(smi, fg):
    assert fg in _fgs(smi)


@pytest.mark.unit
@pytest.mark.parametrize("smi", ["CSC", "CCO", "CCS"])
def test_chalcogen_ether_negatives(smi):
    # dimethyl sulfide / ethanol / ethanethiol carry no Se/Te key
    fgs = _fgs(smi)
    assert "selenoether" not in fgs and "telluroether" not in fgs and "tellurol" not in fgs


@pytest.mark.unit
def test_selenide_ships_and_round_trips(namer):
    # 169.7 delivered perception + (methylselanyl) naming + locant; a phase BBR-ASM
    # added the gold-exact enclosing parens (selanyl/tellanyl now classed complex).
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    name = namer.name("CCC[Se]C")
    assert name == "1-(methylselanyl)propane"
    assert opsin_roundtrip_check("CCC[Se]C", name)["passed"]


# --- free inorganic oxoacids (perceived; naming is, downstream) ---
@pytest.mark.unit
@pytest.mark.parametrize("smi,fg", [
    ("OP(=O)(O)O", "phosphoric_acid"),
    ("OS(=O)(=O)O", "sulfuric_acid"),
    ("O[N+](=O)[O-]", "nitric_acid"),
    ("OC(=O)O", "carbonic_acid"),
])
def test_free_oxoacids_perceived(smi, fg):
    assert fg in _fgs(smi)


@pytest.mark.unit
def test_carbonic_acid_not_carboxylic(namer):
    # HO-C(=O)-OH is a functional parent, NOT a carboxylic acid
    assert "carboxylic_acid" not in _fgs("OC(=O)O")
    # a real carboxylic acid is unaffected
    assert "carboxylic_acid" in _fgs("CC(=O)O") and "carbonic_acid" not in _fgs("CC(=O)O")
