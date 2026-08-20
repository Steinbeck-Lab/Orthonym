import pytest
from rdkit import Chem
from orthonym.rules.acid_ester_anion import name_acid_ester_anion

def _be():
    from orthonym.namer import Orthonym
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)

# --- current-correct poly-anion shapes that route_charged/oxoacid own: must stay byte-identical
@pytest.mark.parametrize("smi,expected", [
    ("[O-]C(=O)CC(=O)[O-]", "propanedioate"),          # dicarboxylate dianion (GUARD 2)
    ("CP(=O)([O-])[O-]", "methylphosphonate"),          # phosphonate dianion (CR-02)
])
def test_nonester_dianion_unchanged(smi, expected):
    assert _be().name(smi) == expected

# --- the producer is CORRECT when called directly (proves it is a routing bug, not scope)
@pytest.mark.parametrize("smi,expected", [
    ("O=P([O-])([O-])OC(CO)CO", "1,3-dihydroxypropan-2-yl phosphate"),
    ("O=P([O-])([O-])OCC(O)CO", "2,3-dihydroxypropyl phosphate"),
])
def test_producer_correct_when_called_directly(smi, expected):
    m = Chem.MolFromSmiles(smi)
    assert name_acid_ester_anion(m) == expected

# --- the TARGET: end-to-end these currently abstain; after the fix they must emit the word
@pytest.mark.parametrize("smi,expected", [
    ("O=P([O-])([O-])OC(CO)CO", "1,3-dihydroxypropan-2-yl phosphate"),
    ("O=P([O-])([O-])OCC(O)CO", "2,3-dihydroxypropyl phosphate"),
])
def test_glycerophosphate_class_end_to_end(smi, expected):
    assert _be().name(smi) == expected

@pytest.mark.parametrize("smi", [
    "CCOP(=O)([O-])[O-]",          # simple phosphate monoester dianion (already worked)
    "O=P([O-])([O-])OCC",          # ethyl phosphate (owner order variant)
    "OP(=O)([O-])OC(CO)CO",        # glycerol hydrogen phosphate monoanion
    "CCOS(=O)(=O)[O-]",            # sulfate ester monoanion
    "OC",                           # sanity: not an ester-anion -> must NOT be claimed here
])
def test_ester_anion_sweep_roundtrips_or_declines(smi):
    from rdkit import Chem
    from orthonym.validation.opsin_roundtrip import opsin_parse
    name = _be().name(smi)
    if name and name != "unknown organic compound":
        got = opsin_parse(name)
        assert got is not None, f"{name} did not OPSIN-parse"
        assert Chem.MolToInchiKey(Chem.MolFromSmiles(got)) == \
               Chem.MolToInchiKey(Chem.MolFromSmiles(smi)), \
               f"{smi} -> {name} did NOT round-trip (0-wrong violation)"

# --- fix round 1: cyclitol/inositol phosphate dianion -> the ester-anion producer emits a
# name whose stereo OPSIN 2.9.0 cannot CIP-parse (constitution-only match is not enough).
# Never-wrong beats never-silent: this must ABSTAIN, not ship a stereo-unverified name.
# `opsin_gate` (tests/conftest.py): the suite disables the OPSIN validity gate by
# default (most tests assert raw generator output); this test is ABOUT gate/abstain
# behaviour, so per conftest's own documented convention it must re-enable the gate
# to exercise the same path production (`.venv/bin/python -m orthonym`) runs — a
# canary confirmed the assertion is gate-invariant here (it holds with the gate on;
# without the marker pytest sees the pre-gate raw candidate instead).
@pytest.mark.opsin_gate
def test_cyclitol_phosphate_abstains_stereo_unverified():
    smi = "O=P([O-])([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O"
    assert _be().name(smi) == "unknown organic compound"

# --- fix round 2 (class closure): the SAME defect class on the single-anion sibling
# sites (ions.py:975-978, dispatch_table.py::_handle_anion_small). The strict gate
# must NEVER abstain a name that is correct + OPSIN-parseable -- verify plain ester
# monoanions still ship and full-InChIKey round-trip (RT computed, not hardcoded).
@pytest.mark.parametrize("smi", [
    "CCOP(=O)(O)[O-]",   # ethyl hydrogen phosphate (P monoester monoanion)
    "CCOS(=O)(=O)[O-]",  # sulfate ester monoanion
])
def test_plain_ester_monoanion_still_ships_and_roundtrips(smi):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    name = _be().name(smi)
    assert name and name != "unknown organic compound", \
        f"{smi} unexpectedly abstained: {name!r}"
    got = opsin_parse(name)
    assert got is not None, f"{name} did not OPSIN-parse"
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(got)) == \
           Chem.MolToInchiKey(Chem.MolFromSmiles(smi)), \
           f"{smi} -> {name} did NOT round-trip (0-wrong violation)"

# --- fix round 2: cyclitol/inositol phosphate/sulfate MONOanion -> the ester-anion
# producer's owner-namer emits a stereo OPSIN 2.9.0 cannot CIP-parse; this must
# ABSTAIN via the single-anion sibling sites, not ship stereo-unverified.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", [
    "O=P(O)([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O",   # cyclitol phosphate monoanion
    "O=S(=O)([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O",  # cyclitol sulfate monoanion
])
def test_cyclitol_monoanion_abstains_stereo_unverified(smi):
    assert _be().name(smi) == "unknown organic compound"
