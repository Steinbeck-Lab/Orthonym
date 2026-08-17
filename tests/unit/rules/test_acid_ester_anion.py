from rdkit import Chem
from orthonym.rules.phosphorus import name_phosphate_ester_anion


def _p(smi):
    return name_phosphate_ester_anion(Chem.MolFromSmiles(smi), _p_idx(smi))


def _p_idx(smi):
    m = Chem.MolFromSmiles(smi)
    return next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'P')


def test_phosphate_monoester_dianion():
    assert _p("CCCCCCCCCCCCOP(=O)([O-])[O-]") == "dodecyl phosphate"


def test_phosphate_monoester_monoanion():
    assert _p("CCCCCCCCCCCCOP(=O)([O-])O") == "dodecyl hydrogen phosphate"


def test_phosphate_diester_monoanion():
    assert _p("CCOP(=O)([O-])OCC") == "diethyl phosphate"


def test_phosphate_monoester_dianion_methyl_regression():
    assert _p("COP(=O)([O-])[O-]") == "methyl phosphate"


def test_neutral_ester_deferred():
    # anion_count == 0 -> None (the neutral producer owns this shape)
    assert _p("CCCCCCCCCCCCOP(=O)(O)O") is None


def test_thiophosphate_ester_anion_failclosed():
    assert _p("CCCCCCCCCCCCOP(=S)([O-])[O-]") is None


def test_bare_inorganic_phosphate_no_owner_failclosed():
    # no O-C ester owner -> not our class
    assert _p("[O-]P(=O)([O-])[O-]") is None


def test_extra_cation_centre_failclosed():
    # a second (cationic) centre -> not a clean single acid-ester anion
    assert _p("C[N+](C)(C)CCOP(=O)([O-])[O-]") is None


def test_phosphonate_ester_anion():
    # methylphosphonate mono-ethyl-ester anion: one P-C ligand -> phosphonate stem
    assert _p("CP(=O)([O-])OCC") == "ethyl methylphosphonate"


def test_zwitterion_extra_charged_centre_failclosed():
    # O-phosphoserine-shaped dianion + zwitterion: a charged centre outside
    # the terminal [O-] must reject via the per-atom scan, not slip through.
    assert _p("[NH3+]C(COP(=O)([O-])[O-])C(=O)[O-]") is None
