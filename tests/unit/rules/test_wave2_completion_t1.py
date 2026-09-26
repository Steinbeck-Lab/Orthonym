"""Wave-2 completion Tier 1: bare hydrides, skeletal-replacement unsaturation,
bipyridine retained-key fix, ketenes (WAVE2-COMPLETION-PLAN T1).

All expected names BB-verified / / /
 / and OPSIN-RT probed at build time.
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


# --- Bare / lambda mononuclear parent hydrides / ---

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
    ("[PH5]", "λ5-phosphane"),
    ("[SH4]", "λ4-sulfane"),
    ("[IH3]", "λ3-iodane"),
    ("I", "iodane"),
]


@pytest.mark.parametrize("smiles,expected", BARE_HYDRIDES)
def test_bare_hydride_unit(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert name_mononuclear_hydride(mol) == expected


@pytest.mark.parametrize("smiles,expected", [
    ("[SiH4]", "silane"),
    ("[PH5]", "λ5-phosphane"),
    ("[SH4]", "λ4-sulfane"),
    ("[IH3]", "λ3-iodane"),
    ("[SnH4]", "stannane"),
])
def test_bare_hydride_e2e(smiles, expected):
    assert _name(smiles) == expected


def test_halogen_regime_unchanged():
    # Protect: the pre-existing all-halogen regime is untouched.
    assert name_mononuclear_hydride(
        Chem.MolFromSmiles("FS(F)(F)(F)(F)F")) == "hexafluoro-λ6-sulfane"
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


# --- Skeletal-replacement chain unsaturation ---

SKELETAL_ENE = [
    # BB verbatim example
    ("C[SiH2]C[SiH2]C[SiH2]C[SiH2]C=C", "2,4,6,8-tetrasiladec-9-ene"),
    ("C=CCOCCOCCOCCOC", "2,5,8,11-tetraoxatetradec-13-ene"),
    ("C#CCOCCOCCOCCOC", "2,5,8,11-tetraoxatetradec-13-yne"),
    # fix a performance pass (was 2,5,8-trioxaundec-10-ene): three heterounits, no 'a' PIN
    #, the Blue Book); see test_three_unit_ene_is_substitutive.
    ("COCCOCCOCC=C", None),
]


@pytest.mark.parametrize("smiles,expected", SKELETAL_ENE)
def test_skeletal_ene_unit(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert try_skeletal_replacement_name(mol) == expected


@pytest.mark.parametrize("smiles,expected", SKELETAL_ENE[:3])
def test_skeletal_ene_e2e(smiles, expected):
    assert _name(smiles) == expected


def test_skeletal_saturated_protect():
    # fix a performance pass: these chains carry 3, 2 and 3 heterounits, so the 'a' name
    # is not the PIN, the Blue Book: "four or more heterounits";
    # "(1) 1-methoxy-2-(2-methoxyethoxy)ethane (PIN)":27756). Were
    # '2,5,8-trioxadecane', '3,6-dioxaoctan-1-ol', '2,6-dioxa-4-thiaheptane'.
    assert try_skeletal_replacement_name(Chem.MolFromSmiles("COCCOCCOCC")) is None
    assert try_skeletal_replacement_name(Chem.MolFromSmiles("OCCOCCOCC")) is None
    assert try_skeletal_replacement_name(Chem.MolFromSmiles("COCSCOC")) is None
    # The substitutive PINs (OPSIN 2.9.0 full-InChIKey RT: exact):
    assert _name("COCCOCCOCC") == "1-ethoxy-2-(2-methoxyethoxy)ethane"
    assert _name("OCCOCCOCC") == "2-(2-ethoxyethoxy)ethan-1-ol"
    assert _name("COCSCOC") == "methoxy[(methoxymethyl)sulfanyl]methane"


def test_three_unit_ene_is_substitutive():
    # fix a performance pass (was '2,5,8-trioxaundec-10-ene'): (:23348);
    # the longest chain carrying the double bond is the parent. OPSIN RT exact.
    assert _name("COCCOCCOCC=C") == "3-[2-(2-methoxyethoxy)ethoxy]prop-1-ene"


def test_skeletal_ene_fail_closed():
    # ene + terminal-OH suffix integration is unbuilt -> None (never lossy).
    assert try_skeletal_replacement_name(
        Chem.MolFromSmiles("OCCOCCOCC=C")) is None
    # 3-heterounit ether-ene chain does not qualify -> substitutive territory.
    assert try_skeletal_replacement_name(
        Chem.MolFromSmiles("C=CCOCCOC")) is None


# --- Bipyridine retained-key fix ---

@pytest.mark.parametrize("smiles,expected", [
    ("c1cc(ncc1)-c1ccncc1", "2,4'-bipyridine"),   # was wrong-keyed 2,2'
    ("c1ccc(nc1)-c1ccccn1", "2,2'-bipyridine"),   # the TRUE 2,2'
    ("c1cc(ccn1)-c1ccncc1", "4,4'-bipyridine"),
    ("c1cccnc1-c1cccnc1", "2,3'-bipyridine"),
])
def test_bipyridine_isomers(smiles, expected):
    assert _name(smiles) == expected


# --- Ketenes ---

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
    # W2E-P1FG Task 3 (scope-decision #4, (3) oxomethylidene ring case).
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
