"""v52 a phase Task 2 — focused unit tests for ``emit_onium_hydride_parent``.

Governing rule: **** (the Blue Book "Cationic compounds derived
from neutral compounds expressed by suffixes"; verbatim ``(PIN)`` examples
"ethylideneoxidanium":41463 and "acetyloxidanium":41474). A SUBSTITUTED
mononuclear chalcogen cation (O/S/Se/Te, +1, not in a ring, not catenated to
another chalcogen) is named on the mononuclear parent hydride
oxidane/sulfane/selane/tellane + '-ium' with its substituents (alkyl / acyl /
alkylidene) cited as detachable prefixes -- the systematic PIN, NOT the T4
'-a'/'-onia'-replacement spelling (``1-oxaprop-1-en-1-ium``).

The direct-call tests exercise the producer in isolation; the end-to-end tests
confirm it is reached through ``route_charged`` under the bb-harness flags (the
plain CLI abstains for these). The DECLINE tests pin the fail-closed scope: an
abstention (``''``) hands the caller's RT gate no name and preserves 0-wrong.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.rules.ions import emit_onium_hydride_parent

FLAGS = dict(general_fallback=True, general_fallback_unverified=True,
             allow_aromatic_general=True)

_CHALCOGENS = {"O", "S", "Se", "Te"}


def _cation_center(smiles):
    """The atom index of the single +1 chalcogen cation centre in ``smiles``."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    centers = [a.GetIdx() for a in mol.GetAtoms()
               if a.GetSymbol() in _CHALCOGENS and a.GetFormalCharge() == 1]
    assert len(centers) == 1, (smiles, centers)
    return mol, centers[0]


# --- Direct producer calls: the five mononuclear targets ---------
@pytest.mark.parametrize("smiles,expected", [
    ("CC=[OH+]", "ethylideneoxidanium"),                 # =CHCH3 alkylidene, 1 H
    ("CC(=O)[OH2+]", "acetyloxidanium"),                 # CH3CO- acyl, 2 H
    ("CC[O+]=C(C)C", "ethyl(propan-2-ylidene)oxidanium"),  # alkyl + alkylidene
    ("C[O+](C)C(=O)C1CCCCC1",
     "(cyclohexanecarbonyl)di(methyl)oxidanium"),        # ring acyl + di(methyl)
    ("C[S+](C)C(=O)c1ccccc1", "benzoyldi(methyl)sulfanium"),  # benzoyl + di(methyl)
])
def test_emit_onium_hydride_parent_targets(smiles, expected):
    # (the Blue Book; "ethylideneoxidanium (PIN)":41463).
    mol, center = _cation_center(smiles)
    assert emit_onium_hydride_parent(mol, center) == expected


# --- Producer reproduces the retained sulfanium controls byte-identically ---
@pytest.mark.parametrize("smiles,expected", [
    ("[SH3+]", "sulfanium"),            # bare parent-hydride cation, no prefixes
    ("C[SH+]C", "dimethylsulfanium"),   # uniform simple prefix -> no marks
    ("C[S+](C)C", "trimethylsulfanium"),
    ("[OH3+]", "oxidanium"),
])
def test_emit_onium_hydride_parent_reproduces_controls(smiles, expected):
    mol, center = _cation_center(smiles)
    assert emit_onium_hydride_parent(mol, center) == expected


# --- Fail-closed scope: the producer DECLINES ('') on out-of-scope inputs ----
@pytest.mark.parametrize("smiles", [
    "O=C(O[OH2+])c1ccccc1",  # catenated O-O (dioxidane) -> different parent (Task 3)
    "CO[O+](C)C",            # catenated O-O parent
    "C[S+](C)CCO",           # hydroxyethyl -> classify_substituent would drop the -OH
    "CO[S+](C)(C)=O",        # =O on S (sulfinate-ester cation), not a hydride parent
    "O=[S+](C)C",            # =O on S
])
def test_emit_onium_hydride_parent_declines_out_of_scope(smiles):
    mol, center = _cation_center(smiles)
    assert emit_onium_hydride_parent(mol, center) == ""


def test_ring_chalcogen_cation_declines():
    """A ring chalcogen cation (e.g. a pyrylium-type centre) is not a mononuclear
    hydride parent -> decline is the acyclic mononuclear class)."""
    mol = Chem.MolFromSmiles("C1CC[O+]1C")  # methyl-oxetanium (ring O+)
    assert mol is not None
    center = [a.GetIdx() for a in mol.GetAtoms()
              if a.GetSymbol() == "O" and a.GetFormalCharge() == 1][0]
    assert emit_onium_hydride_parent(mol, center) == ""


# --- End-to-end through the tiered namer under the bb-harness flags ----------
@pytest.mark.parametrize("smiles,expected", [
    ("CC=[OH+]", "ethylideneoxidanium"),
    ("CC(=O)[OH2+]", "acetyloxidanium"),
    ("C[S+](C)C(=O)c1ccccc1", "benzoyldi(methyl)sulfanium"),
])
def test_end_to_end_name_tiered(smiles, expected):
    namer = Orthonym(**FLAGS)
    assert namer.name_tiered(smiles)["name"] == expected
