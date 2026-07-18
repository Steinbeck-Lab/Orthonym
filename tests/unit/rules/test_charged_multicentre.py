"""Charged multi-centre subsystem (root R11; P-71..75) — W8-P5.

Task 0: guard-characterization test locking the ~44 already-built charged
golds so later work in this subsystem can never regress them (these already
pass at HEAD; this is a characterization lock, not a new-behaviour test).

Task 1 (P-73.2.3.1, BB 41623 PIN): acylium detection + neutralize-as-acid
emitter — ``classify_cation`` must return 'acylium' (not the generic
carbenium 'ylium') for a C+ double-bonded to O, and the name must be the
reconstructed-acid PIN ('acetylium'/'cyclohexanecarbonylium'), never the
generic hydride-loss aldehyde form ('acetaldehydylium').
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.perception.ions import get_ion_sites
from orthonym.rules.ions import classify_cation


GUARD = {
    "C[N-][N+](C)(C)C": "1,2,2,2-tetramethylhydrazin-2-ium-1-ide",   # P-74.1.1
    "CC(C)=[O+][O-]":   "2-(propan-2-ylidene)dioxidan-2-ium-1-ide",  # P-74.1.1
    "C[P+](C)(C)[C-](C)C": "2-(trimethylphosphaniumyl)propan-2-ide", # P-74.2.1.1
    "[C-]#[C-]": "ethynediide",                                       # P-72.2.2.1
    "[O-]CC[O-]": "ethane-1,2-bis(olate)",                            # P-72.2.2.2.2
    "[NH-]CC[NH-]": "ethane-1,2-bis(aminide)",                        # P-72.2.2.2.3
    "[NH3+]CC[NH3+]": "ethane-1,2-bis(aminium)",                      # P-73.5 poly-aminium
    "C[N+](C)(C)C": "N,N,N-trimethylmethanaminium",                   # P-73.1.2.1
    "[CH-]1CCCCC1": "cyclohexan-1-ide",                               # P-72.2.2.1 ring
    "C[B-](C)(C)C": "tetramethylboranuide",                           # P-72.3
    "C[P-](C)(C)C": "tetramethylphosphanuide",                        # P-72.3
    "NC(=[OH+])N": "uronium",                                         # P-73.1.2.2
    "CCC=[S+][O-]": "propylidene-lambda4-sulfanone",                  # P-74.2.2.1.8
    "CC(C)[O-]": "propan-2-olate",                                    # P-72.2.2.2.2
}


@pytest.mark.parametrize("smi,expected", GUARD.items())
def test_charged_guard_no_regression(smi, expected):
    assert Orthonym().name(smi) == expected


# === Task 1: acylium (P-73.2.3.1) =============================================

def test_classify_acylium():
    mol = Chem.MolFromSmiles("C[C+]=O")  # CH3-C(+)=O
    site = get_ion_sites(mol)["cations"][0]
    assert classify_cation(mol, site) == "acylium"


def test_classify_cyclohexanecarbonylium():
    mol = Chem.MolFromSmiles("[C+](=O)C1CCCCC1")
    site = get_ion_sites(mol)["cations"][0]
    assert classify_cation(mol, site) == "acylium"


def test_classify_plain_carbenium_unaffected():
    """A plain carbenium (no double-bonded O) must stay 'ylium' (no regression
    on the pre-existing carbenium path)."""
    mol = Chem.MolFromSmiles("C[C+](C)C")  # tert-butyl cation
    site = get_ion_sites(mol)["cations"][0]
    assert classify_cation(mol, site) == "ylium"


def test_acetylium_pin():
    assert Orthonym().name("C[C+]=O") == "acetylium"


def test_cyclohexanecarbonylium_pin():
    assert Orthonym().name("[C+](=O)C1CCCCC1") == "cyclohexanecarbonylium"


def test_acylium_gated_equals_raw():
    """Gated == raw (gate-off) confirms the emitter is source-level correct,
    not merely OPSIN-lucky (Global Constraint: verify with the gate-off
    namer since the RT-gate fails OPEN without Java)."""
    gated = Orthonym()
    raw = Orthonym(_disable_opsin_validity_gate=True)
    for smi in ("C[C+]=O", "[C+](=O)C1CCCCC1"):
        assert gated.name(smi) == raw.name(smi)
