import pytest
from rdkit import Chem
from orthonym import Orthonym
from orthonym.rules.phosphorus import name_phosphate_ester_anion
from orthonym.rules.acid_ester_anion import (
    name_sulfate_ester_anion,
    name_acid_ester_anion,
)


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


def _s_idx(smi):
    m = Chem.MolFromSmiles(smi)
    return next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'S')


def test_sulfate_ester_anion():
    smi = "CCCCCCCCCCCCOS(=O)(=O)[O-]"
    assert name_sulfate_ester_anion(Chem.MolFromSmiles(smi), _s_idx(smi)) == "dodecyl sulfate"


def test_sulfonate_not_intercepted():
    # S-C (methanesulfonate), not S-O-C -> not our class
    smi = "CS(=O)(=O)[O-]"
    assert name_sulfate_ester_anion(Chem.MolFromSmiles(smi), _s_idx(smi)) is None


def test_thiosulfate_ester_failclosed():
    smi = "CCCCCCCCCCCCOS(=S)(=O)[O-]"
    assert name_sulfate_ester_anion(Chem.MolFromSmiles(smi), _s_idx(smi)) is None


def test_sulfate_extra_charged_centre_failclosed():
    # net charge (-1) equals -anion_count, but a remote [NH3+]/carboxylate pair
    # is extra -> the per-atom charge scan must reject (a net-sum check would not)
    smi = "[NH3+]C(CC(=O)[O-])OS(=O)(=O)[O-]"
    assert name_sulfate_ester_anion(Chem.MolFromSmiles(smi), _s_idx(smi)) is None


def test_dispatcher_routes_phosphate():
    assert name_acid_ester_anion(Chem.MolFromSmiles("CCCCCCCCCCCCOP(=O)([O-])[O-]")) == "dodecyl phosphate"


def test_dispatcher_routes_sulfate():
    assert name_acid_ester_anion(Chem.MolFromSmiles("CCCCCCCCCCCCOS(=O)(=O)[O-]")) == "dodecyl sulfate"


def test_dispatcher_declines_nonester():
    assert name_acid_ester_anion(Chem.MolFromSmiles("CC(=O)[O-]")) is None  # acetate


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("CCCCCCCCCCCCOP(=O)([O-])[O-]", "dodecyl phosphate"),
    ("CCCCCCCCCCCCOP(=O)([O-])O", "dodecyl hydrogen phosphate"),
    ("CCOP(=O)([O-])OCC", "diethyl phosphate"),
    ("COP(=O)([O-])[O-]", "methyl phosphate"),
    ("CCCCCCCCCCCCOS(=O)(=O)[O-]", "dodecyl sulfate"),
])
def test_integration_ester_anion_names(namer, smi, expected):
    # opsin_gate: without the RT-validity gate ON, an earlier neutral-form
    # candidate (e.g. 'dodecyl dihydrogen phosphate') wins before the gate
    # can reject it as a different molecule and fall back to our anion name.
    assert namer.name(smi) == expected


@pytest.mark.opsin_gate
def test_integration_phosphonate_ester_anion(namer):
    # end-to-end RT for the phosphonate ester anion (a P-C ligand shifts the
    # stem to 'phosphonate'; not covered by test_integration_ester_anion_names).
    assert namer.name("CP(=O)([O-])OCC") == "ethyl methylphosphonate"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # neutral esters unchanged (name_phosphate_ester)
    ("CCCCCCCCCCCCOP(=O)(O)O", "dodecyl dihydrogen phosphate"),
    ("COP(=O)(OC)OC", "trimethyl phosphate"),
    # sulfonate unchanged (must NOT be intercepted by the sulfate-ester branch)
    ("CS(=O)(=O)[O-]", "methanesulfonate"),
])
def test_integration_regressions_unchanged(namer, smi, expected):
    assert namer.name(smi) == expected


@pytest.mark.opsin_gate
def test_integration_failclosed_never_wrong_organic(namer):
    # thiophosphate ester anion -> not a wrong name. Carries a carbon skeleton
    # (the dodecyl chain), so the B3 carbon-free honesty floor
    # (errors.py::classify_failure_limit) does not apply here -- this stays
    # the pre-existing organic-unnameable sentinel.
    out = namer.name("CCCCCCCCCCCCOP(=S)([O-])[O-]")
    assert out == "unknown organic compound"


@pytest.mark.opsin_gate
def test_integration_failclosed_never_wrong_inorganic(namer):
    # diphosphate (multi-centre) -> fail closed, never a wrong molecule.
    # B3: this input is carbon-free (O/P/H only), so it now correctly
    # abstains via errors.py's structural honesty floor to the HONEST
    # "inorganic compound (not supported)" message instead of the
    # the contributor guide-flagged-dishonest "unknown organic compound" sentinel --
    # this assertion was updated deliberately, not because the old string
    # "looked wrong": the input has zero carbon atoms (verified: no 'C' in
    # "OP(=O)([O-])OP(=O)([O-])[O-]"), and the project's own stated policy
    # (the contributor guide a project rule / V36-a trace-B3 a) is that a carbon-free
    # fragment must never surface the organic sentinel. 0-wrong is
    # unaffected either way (both strings are abstentions, never a shipped
    # name); only the honesty of the abstention message changed.
    out = namer.name("OP(=O)([O-])OP(=O)([O-])[O-]")
    assert out == "inorganic compound (not supported)"
