"""Wave-2 completion Tier 1: bare hydrides, skeletal-replacement unsaturation,
bipyridine retained-key fix, ketenes (WAVE2-COMPLETION-PLAN T1).

All expected names BB-verified (P-21.1.1.1 / P-21.1.2 / P-15.4.3.2.4 /
P-28.2.1 / P-64.2.2.4) and OPSIN-RT probed at build time.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.ketenes import name_ketene
from orthonym.rules.mononuclear_hydrides import name_mononuclear_hydride
from orthonym.rules.skeletal_replacement import try_skeletal_replacement_name

pytestmark = pytest.mark.unit


def _name(smiles):
    return name_compound(Chem.CanonSmiles(smiles))


# --- Bare / lambda mononuclear parent hydrides (P-21.1.1.1 / P-21.1.2) ---

BARE_HYDRIDES = [
    ("[SiH4]", "silane"),
    ("[GeH4]", "germane"),
    ("[SnH4]", "stannane"),
    ("[PbH4]", "plumbane"),
    ("[PH3]", "phosphane"),
    ("[SH2]", "sulfane"),
    ("[SeH2]", "selane"),
    ("[TeH2]", "tellane"),
    ("[AsH3]", "arsane"),
    ("[PH5]", "lambda5-phosphane"),
    ("[SH4]", "lambda4-sulfane"),
    ("[IH3]", "lambda3-iodane"),
    ("I", "iodane"),
]


@pytest.mark.parametrize("smiles,expected", BARE_HYDRIDES)
def test_bare_hydride_unit(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert name_mononuclear_hydride(mol) == expected


@pytest.mark.parametrize("smiles,expected", [
    ("[SiH4]", "silane"),
    ("[PH5]", "lambda5-phosphane"),
    ("[SH4]", "lambda4-sulfane"),
    ("[IH3]", "lambda3-iodane"),
    ("[SnH4]", "stannane"),
])
def test_bare_hydride_e2e(smiles, expected):
    assert _name(smiles) == expected


def test_halogen_regime_unchanged():
    # Protect: the pre-existing all-halogen regime is untouched.
    assert name_mononuclear_hydride(
        Chem.MolFromSmiles("FS(F)(F)(F)(F)F")) == "hexafluoro-lambda6-sulfane"
    assert name_mononuclear_hydride(
        Chem.MolFromSmiles("ClP(Cl)Cl")) == "trichlorophosphane"


def test_bare_hydride_fail_closed():
    # Charged, radical, ring, multi-hub each decline.
    assert name_mononuclear_hydride(Chem.MolFromSmiles("[PH4+]")) is None
    assert name_mononuclear_hydride(Chem.MolFromSmiles("PP")) is None
    assert name_mononuclear_hydride(Chem.MolFromSmiles("C1CCSC1")) is None
    # Water/ammonia/methane are not hubs (O/N/C not in _HUB_STEMS).
    assert name_mononuclear_hydride(Chem.MolFromSmiles("O")) is None
    assert name_mononuclear_hydride(Chem.MolFromSmiles("N")) is None
    assert name_mononuclear_hydride(Chem.MolFromSmiles("C")) is None


# --- Skeletal-replacement chain unsaturation (P-15.4.3.2.4) ---

SKELETAL_ENE = [
    # BB verbatim example
    ("C[SiH2]C[SiH2]C[SiH2]C[SiH2]C=C", "2,4,6,8-tetrasiladec-9-ene"),
    ("C=CCOCCOCCOCCOC", "2,5,8,11-tetraoxatetradec-13-ene"),
    ("C#CCOCCOCCOCCOC", "2,5,8,11-tetraoxatetradec-13-yne"),
    ("COCCOCCOCC=C", "2,5,8-trioxaundec-10-ene"),
]


@pytest.mark.parametrize("smiles,expected", SKELETAL_ENE)
def test_skeletal_ene_unit(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert try_skeletal_replacement_name(mol) == expected


@pytest.mark.parametrize("smiles,expected", SKELETAL_ENE[:3])
def test_skeletal_ene_e2e(smiles, expected):
    assert _name(smiles) == expected


def test_skeletal_saturated_protect():
    # Byte-identical saturated behaviour (pre-existing goldens).
    assert try_skeletal_replacement_name(
        Chem.MolFromSmiles("COCCOCCOCC")) == "2,5,8-trioxadecane"
    assert try_skeletal_replacement_name(
        Chem.MolFromSmiles("OCCOCCOCC")) == "3,6-dioxaoctan-1-ol"
    assert try_skeletal_replacement_name(
        Chem.MolFromSmiles("COCSCOC")) == "2,6-dioxa-4-thiaheptane"


def test_skeletal_ene_fail_closed():
    # ene + terminal-OH suffix integration is unbuilt -> None (never lossy).
    assert try_skeletal_replacement_name(
        Chem.MolFromSmiles("OCCOCCOCC=C")) is None
    # 3-heterounit ether-ene chain does not qualify -> substitutive territory.
    assert try_skeletal_replacement_name(
        Chem.MolFromSmiles("C=CCOCCOC")) is None


# --- Bipyridine retained-key fix (P-28.2.1) ---

@pytest.mark.parametrize("smiles,expected", [
    ("c1cc(ncc1)-c1ccncc1", "2,4'-bipyridine"),   # was wrong-keyed 2,2'
    ("c1ccc(nc1)-c1ccccn1", "2,2'-bipyridine"),   # the TRUE 2,2'
    ("c1cc(ccn1)-c1ccncc1", "4,4'-bipyridine"),
    ("c1cccnc1-c1cccnc1", "2,3'-bipyridine"),
])
def test_bipyridine_isomers(smiles, expected):
    assert _name(smiles) == expected


# --- Ketenes (P-64.2.2.4) ---

KETENES = [
    ("C=C=O", "ethenone"),               # BB verbatim
    ("BrC(Br)=C=O", "dibromoethenone"),  # BB verbatim
    ("FC(F)=C=O", "difluoroethenone"),
    ("ClC=C=O", "chloroethenone"),
    ("FC(Cl)=C=O", "chlorofluoroethenone"),
]


@pytest.mark.parametrize("smiles,expected", KETENES)
def test_ketene_unit(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert name_ketene(mol) == expected


@pytest.mark.parametrize("smiles,expected", KETENES[:2])
def test_ketene_e2e(smiles, expected):
    assert _name(smiles) == expected


def test_ketene_fail_closed():
    # Alkyl / mono-substituted ketenes needing chain-ketone numbering ->
    # general ketone principles (unbuilt) -> None.
    assert name_ketene(Chem.MolFromSmiles("CC=C=O")) is None
    # NOTE: O=C=C1CCCCC1 (cyclohexylidenemethanone) is now BUILT — see
    # W2E-P1FG Task 3 (scope-decision #4, P-64.5(3) oxomethylidene ring case).
    assert name_ketene(Chem.MolFromSmiles("O=C=C1CCCCC1")) == \
        "cyclohexylidenemethanone"
    # Non-ketene cumulenes / carbonyls decline.
    assert name_ketene(Chem.MolFromSmiles("O=C=O")) is None
    assert name_ketene(Chem.MolFromSmiles("N=C=O")) is None
    assert name_ketene(Chem.MolFromSmiles("CC(C)=O")) is None


def test_ketene_controls_e2e():
    # Protect: CO2 / isocyanic acid / acetone keep their names.
    assert _name("O=C=O") == "carbon dioxide"
    assert _name("CC(=O)C") == "propan-2-one"
