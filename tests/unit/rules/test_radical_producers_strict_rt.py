""" fix a performance pass (wp1-zero-wrong, F8): every radical producer ships only a name
that the strict radical round trip confirms.

Before this fix several producers returned a name without any round trip, and
only the namer's final OPSIN gate stood in the way -- which fails OPEN when OPSIN
is 'unavailable' (a timeout on a loaded host). Under that outage these shipped as
pin_verified although each denotes a different molecule (OPSIN 2.9.0 -r parses,
independent batch call):

  NC(=O)CC[NH] '3-aminopropanamidyl' -> NCCC(=O)[NH] (radical on the amide N)
  C[Si](C)(C)[O] 'oxyl' -> [OH]
  [SH3] 'sulfanyl' -> [SH]
  O=[C]C(=O)O 'acetyl' -> C[C]=O
  [NH]C[C@H](C)[NH] '(propane-1,2-diyl)bis(aminyl)' -> stereo lost

 (the Blue Book, 'benzenaminyl (PIN)'): the radical suffix goes on
the nitrogen that carries the radical. Invariant: a name OPSIN does not parse back
to the input is never pin_verified.
"""
import pytest
from rdkit import Chem

import orthonym.namer as nm
from orthonym import Orthonym
from orthonym.perception.ions import get_radical_sites
from orthonym.rules import radicals as R
from orthonym.rules.charged_router import route_charged


def _mol_sites(smi):
    m = Chem.MolFromSmiles(smi)
    return m, get_radical_sites(m)


@pytest.mark.parametrize("smi", [
    "NC(=O)CC[NH]",          # amine-family long form: '3-aminopropanamidyl' (wrong N)
    "[SH3]",                 # mononuclear: 'sulfanyl' is [SH]
    "[PH4]",                 # mononuclear: 'phosphanyl' is [PH2]
    "[15NH2]",               # mononuclear: 'azanyl' drops the isotope
    "[15NH]CC[NH]",          # multiplicative: drops the isotope
    "[NH]C[C@H](C)[NH]",     # multiplicative: drops the stereo
])
def test_heteroatom_radical_producer_never_ships_an_unverified_name(smi):
    m, sites = _mol_sites(smi)
    assert R.name_heteroatom_radical(m, sites) == ""


@pytest.mark.parametrize("smi,expected", [
    ("[NH2]", "azanyl"),
    ("[SH]", "sulfanyl"),
    ("C[NH]", "methanaminyl"),
    ("[NH]CC[NH]", "(ethane-1,2-diyl)bis(aminyl)"),   # BB:40660 (PIN)
])
def test_heteroatom_radical_producer_keeps_verified_names(smi, expected):
    m, sites = _mol_sites(smi)
    assert R.name_heteroatom_radical(m, sites) == expected


@pytest.mark.parametrize("smi", ["C[Si](C)(C)[O]", "[O]O[Si](C)(C)C"])
def test_bare_oxyl_is_not_a_fallback(smi):
    m, sites = _mol_sites(smi)
    assert R.name_oxyl_radical(m, sites[0]) == ""


def test_oxyl_positives_unchanged():
    m, sites = _mol_sites("C[O]")
    assert R.name_oxyl_radical(m, sites[0]) == "methoxyl"


@pytest.mark.parametrize("smi,expected", [
    ("O=[C]C(=O)O", ""),          # carbon count would say 'acetyl'
    ("O=[13C]C", ""),             # isotope not expressed
    ("C[C]=O", "acetyl"),
    ("O=[C]c1ccccc1", "benzoyl"),
])
def test_acyl_radical_counts_only_verified(smi, expected):
    m, sites = _mol_sites(smi)
    assert R.name_acyl_radical(m, sites[0]) == expected


@pytest.mark.parametrize("smi,expected", [
    ("[13CH2]CC(=O)O", None),
    ("[CH2]CC(=O)[18OH]", None),
    ("[CH2]CC(=O)O", "2-carboxyethyl"),
])
def test_carboxy_alkyl_radical_is_verified(smi, expected):
    m, sites = _mol_sites(smi)
    assert R._name_carboxy_alkyl_radical(m, sites[0]["atom_idx"]) == expected


@pytest.mark.parametrize("smi,wrong", [
    ("[CH2]C[13CH3]", "(1-13C)propyl"),       # label at the wrong end
    ("[CH]C[13CH3]", "(1-13C)propylidene"),
    ("[C]C[13CH3]", "(1-13C)propylidyne"),
])
def test_simple_terminal_radical_contraction_is_verified(smi, wrong):
    assert route_charged(Chem.MolFromSmiles(smi), "pin") != wrong


@pytest.mark.parametrize("smi,expected", [
    ("[CH3]", "methyl"), ("[CH2]C", "ethyl"), ("[CH2]CC", "propyl"),
])
def test_simple_terminal_radical_positives(smi, expected):
    assert route_charged(Chem.MolFromSmiles(smi), "pin") == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", [
    "NC(=O)CC[NH]", "C[Si](C)(C)[O]", "[SH3]", "O=[C]C(=O)O", "[NH]C[C@H](C)[NH]",
])
def test_gate_outage_ships_no_wrong_radical(monkeypatch, smi):
    # The namer's final gate fails OPEN on 'unavailable'; with it simulated
    # down, no producer may ship its own unverified name as pin_verified.
    monkeypatch.setattr(nm, "_validity_gate_name_to_smiles", lambda *a, **k: None)
    monkeypatch.setattr(nm, "_validity_gate_status", lambda *a, **k: "unavailable")
    res = Orthonym(style="pin").name_tiered(smi)
    assert res["tier"] != "pin_verified", res
