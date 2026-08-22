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

# --- cyclitol/inositol phosphate/sulfate ester anions: the ester-anion PRODUCER
# still emits a name whose stereo OPSIN 2.9.0 cannot CIP-parse, and SELF-01 still
# correctly suppresses THAT stereo-bearing name (never-wrong preserved). What
# CHANGED at WS7 (v34 composed-charge): the universal coverage FLOOR now BACKSTOPS
# these with a CONSTITUTION-ONLY name (stereo omitted) instead of a silent
# abstain -- a safe stereo-OMISSION superset (feedback:
# stereo-omission is not a wrong molecule; best-effort ships the superset, never
# abstains for 0 precision gain). CONTRACT CHANGE (was
# ``== "unknown organic compound"``): these are now NAMED, not voided. The floor
# is asserted DIRECTLY (deterministic) with a 0-wrong constitution+charge check;
# the end-to-end best-effort tier ships exactly this backstop.
def _floor_constitution_charge_only(smi):
    """Assert the coverage floor names ``smi`` with a name whose OPSIN parse-back
    matches on CONSTITUTION + net CHARGE (stereo removed both sides) but NOT on
    the full InChIKey (stereo genuinely omitted -> a safe superset, not a wrong
    stereoisomer). Floor-direct: independent of the in-process OPSIN-validity
    gate's warm-up state (a documented pytest OPSIN/JVM harness hazard)."""
    from orthonym.assembly.universal_substituent import (
        name_universal_substitutive)
    from orthonym.validation.opsin_roundtrip import opsin_parse
    r = name_universal_substitutive(Chem.MolFromSmiles(smi))
    assert r is not None and r.name, f"floor voided on {smi!r}"
    got = opsin_parse(r.name)
    assert got is not None, f"floor name did not OPSIN-parse: {r.name!r}"

    def _flat(s):
        m = Chem.MolFromSmiles(s)
        Chem.RemoveStereochemistry(m)   # keep formal charge, drop stereo
        return Chem.MolToSmiles(m)
    assert _flat(got) == _flat(smi), \
        f"0-wrong: {smi} -> {r.name} differs in constitution/charge"
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(got)) != \
        Chem.MolToInchiKey(Chem.MolFromSmiles(smi)), \
        f"expected a stereo-OMISSION (block1), got a full match for {r.name!r}"


def test_cyclitol_phosphate_dianion_floor_backstops_constitution_only():
    smi = "O=P([O-])([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O"
    _floor_constitution_charge_only(smi)

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

# --- cyclitol/inositol phosphate/sulfate MONOanion: same WS7 change as the
# dianion above -- the producer's stereo name is still SELF-01-suppressed, and
# the coverage floor now backstops with a constitution-only stereo-omission
# (0-wrong), instead of a silent abstain.
@pytest.mark.parametrize("smi", [
    "O=P(O)([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O",   # cyclitol phosphate monoanion
    "O=S(=O)([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O",  # cyclitol sulfate monoanion
])
def test_cyclitol_monoanion_floor_backstops_constitution_only(smi):
    _floor_constitution_charge_only(smi)
