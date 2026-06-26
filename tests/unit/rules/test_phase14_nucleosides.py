"""v23 Phase 14 — nucleoside/nucleotide decoration + NP-parent catalog + lipid PA.

All names are OPSIN-RT (verified at gold-authoring time). These unit tests lock
the production behaviour (name_compound, full dispatch + SELF-01 gate) and the
fail-closed boundaries of the strip-and-recognise nucleoside engine.
"""
import pytest
from rdkit import Chem

from orthonym.namer import name_compound
from orthonym.rules.nucleosides import name_nucleoside


# --------------------------------------------------------------------------- #
# Batch 1 — nucleoside / nucleotide decoration (P-105.2 / P-106)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("smiles,expected", [
    # 5'-phosphate chain: di / tri across bases
    ("Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
     "adenosine 5'-(tetrahydrogen triphosphate)"),
    ("Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
     "adenosine 5'-(trihydrogen diphosphate)"),
    ("Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",
     "guanosine 5'-(tetrahydrogen triphosphate)"),
    ("O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",
     "uridine 5'-(tetrahydrogen triphosphate)"),
    ("Nc1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)n1",
     "cytidine 5'-(tetrahydrogen triphosphate)"),
    # inosine via keto/enol skeleton tautomer
    ("O=c1[nH]cnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
     "inosine 5'-(tetrahydrogen triphosphate)"),
    # 2'-deoxy + phosphate
    ("Nc1ncnc2c1ncn2[C@H]1C[C@H](O)[C@@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O1",
     "2'-deoxyadenosine 5'-(tetrahydrogen triphosphate)"),
    # O-acyl esters
    ("CC(=O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](OC(C)=O)[C@@H]1OC(C)=O",
     "adenosine 2',3',5'-triacetate"),
    ("CC(=O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",
     "adenosine 5'-acetate"),
])
def test_decorated_nucleotides(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # 2'-deoxynucleoside PINs (data rename; the prime is required)
    ("Nc1ncnc2c1ncn2[C@H]1C[C@H](O)[C@@H](CO)O1", "2'-deoxyadenosine"),
    ("Nc1nc2c(ncn2[C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1", "2'-deoxyguanosine"),
    ("Nc1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)n1", "2'-deoxycytidine"),
    ("O=c1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1", "2'-deoxyuridine"),
])
def test_deoxynucleoside_pins(smiles, expected):
    assert name_compound(smiles) == expected


def test_bare_nucleoside_unaffected():
    # bare nucleosides keep the retained-name path (engine declines)
    assert name_compound("Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O") == "adenosine"


def test_retained_monophosphate_unaffected():
    # AMP MUST stay the retained '5'-adenylic acid' (RETAINED_NAME@1300 before NUCLEOSIDE@1800)
    assert name_compound("Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O") == "5'-adenylic acid"


@pytest.mark.parametrize("smiles", [
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",  # bare -> decline
    "CCO",                                              # not a nucleoside -> decline
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@H](O)[C@H]1O",        # a sugar (no base) -> decline
])
def test_engine_fail_closed(smiles):
    # the engine itself returns None for bare/irrelevant inputs (fail-closed)
    assert name_nucleoside(Chem.MolFromSmiles(smiles)) is None


def test_determinism_across_spellings():
    smi = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"
    m = Chem.MolFromSmiles(smi)
    outs = {name_compound(Chem.MolToSmiles(m, doRandom=True)) for _ in range(10)}
    assert outs == {"adenosine 5'-(tetrahydrogen triphosphate)"}


# --------------------------------------------------------------------------- #
# Batch 3 — NP parent catalog growth (P-101.2.7 Table 10.1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("smiles,expected", [
    ("C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3", "porphyrin"),
    ("C1=C2CCC(=N2)C=C2CCC(N2)C2CCC(=N2)C=C2CCC1=N2", "corrin"),
    ("C1=CC(=CC2=NC(=CC3=NC(=Cc4ccc[nH]4)C=C3)C=C2)N=C1", "21H-biline"),
    ("CC1CC[C@@]2(OC1)O[C@H]1C[C@H]3[C@@H]4CCC5CCCC[C@]5(C)[C@H]4CC[C@]3(C)[C@H]1[C@@H]2C", "spirostan"),
    ("CC(C)CCC1O[C@H]2C[C@H]3[C@@H]4CCC5CCCC[C@]5(C)[C@H]4CC[C@]3(C)[C@H]2[C@@H]1C", "furostan"),
    ("CC(C)[C@@H](C)[C@@]1(C)C[C@@H]1[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C", "gorgostane"),
    ("CC[C@@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C)C(C)C", "poriferastane"),
])
def test_np_parents(smiles, expected):
    assert name_compound(smiles) == expected


def test_poriferastane_distinct_from_stigmastane():
    # the C-24 epimer must NOT collapse to stigmastane (SELF-01 cannot catch a same-skeleton stereo error)
    poriferastane = "CC[C@@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C)C(C)C"
    stigmastane = "CC[C@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C)C(C)C"
    assert name_compound(poriferastane) == "poriferastane"
    assert name_compound(stigmastane) == "stigmastane"


# --------------------------------------------------------------------------- #
# Batch 2 — free phosphatidic acid (P-107.3.1)
# --------------------------------------------------------------------------- #
def test_free_phosphatidic_acid():
    pa = "CCCCCCCCCCCCCCCC(=O)OCC(COP(O)(O)=O)OC(=O)CCCCCCCCCCCCCCC"
    assert name_compound(pa) == "2,3-bis(hexadecanoyloxy)propyl dihydrogen phosphate"


@pytest.mark.parametrize("smiles,expected", [
    # phospholipid regression guards (unchanged by the free-phosphate branch)
    ("CCCCCCCCCCCCCCCC(=O)OCC(COP([O-])(=O)OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC",
     "[2,3-bis(hexadecanoyloxy)propyl] 2-(trimethylazaniumyl)ethyl phosphate"),
    ("CCCCCCCCCCCCCCCCCC(=O)OCC(OC(=O)CCCCCCCCCCCCCCCCC)COC(=O)CCCCCCCCCCCCCCCCC",
     "propane-1,2,3-triyl trioctadecanoate"),
])
def test_lipid_regression(smiles, expected):
    assert name_compound(smiles) == expected
