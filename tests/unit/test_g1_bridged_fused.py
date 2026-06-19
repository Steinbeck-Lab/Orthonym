"""v22 Phase G1 — bridged-fused P-25.4 constructor (DD7 COV-01, bridged half).

G1 turns the G0 fail-closed refusals for the *bridged-fused* class into correct
IUPAC PINs. A bridged fused ring system (P-25.4.1.1) is a fused ring system
(the recognised parent, e.g. naphthalene) plus one or more bridges across it.
The constructor:

  * identifies the bridge(s) (the minimal atom set whose excision leaves a
    recognised fused parent),
  * numbers the residual fused parent by reusing the partial-saturation engine
    (so the bridgeheads receive their true IUPAC locants, e.g. 1 and 4),
  * cites bridge prefixes (methano/ethano/epoxy/...) and hydro prefixes TOGETHER
    in alphanumerical order ignoring multiplying prefixes
    ('epoxy' < 'ethano' < 'hydro' < 'methano'), each with its own locant set
    (Blue Book P-25.4.3.4; verified against the PIN examples
    '1,4-dihydro-1,4-methanonaphthalene', '1,4-epoxy-5,8-methanonaphthalene').

Per guardrail A8 these test the rule FAMILY (any single-bridge-over-recognised-
fused-aromatic-parent) plus determinism and OPSIN constitutional round-trip,
not just literal canary rows. Systems OUTSIDE the handled class (polycomponent
fusion P-25.3.4, polyspiro) MUST stay G0 fail-closed — never a wrong name.
"""
import pytest
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from orthonym import Orthonym, name_compound
from orthonym.errors import is_failure_name


# (smiles, expected PIN) — each verified: OPSIN parse(name) InChIKey == input InChIKey.
BRIDGED_FUSED_GOLD = [
    ("C1C2C=CC1c1ccccc12", "1,4-dihydro-1,4-methanonaphthalene"),   # benzonorbornadiene
    ("C1=CC2OC1c1ccccc12", "1,4-epoxy-1,4-dihydronaphthalene"),     # epoxy bridge
    ("C1CC2CCC1c1ccccc12", "1,4-ethano-1,2,3,4-tetrahydronaphthalene"),  # ethano bridge
]


def _ik(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToInchiKey(m) if m else None


@pytest.mark.parametrize("smiles,expected", BRIDGED_FUSED_GOLD)
def test_bridged_fused_exact_pin(smiles, expected):
    """The constructor emits the exact P-25.4 PIN (gold name-exact-match)."""
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,expected", BRIDGED_FUSED_GOLD)
def test_bridged_fused_opsin_roundtrip(smiles, expected):
    """Constitutional correctness: the emitted name re-parses (OPSIN) to the input."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    out = name_compound(smiles)
    parsed = opsin_parse(out)
    psmi = parsed if isinstance(parsed, str) else getattr(parsed, "smiles", None)
    assert psmi is not None, f"OPSIN could not parse {out!r}"
    assert _ik(psmi) == _ik(smiles), f"{out!r} re-parses to a different structure"


@pytest.mark.parametrize("smiles,expected", BRIDGED_FUSED_GOLD)
def test_bridged_fused_deterministic(smiles, expected):
    """Same structure, any SMILES spelling -> the same PIN."""
    namer = Orthonym().name
    mol = Chem.MolFromSmiles(smiles)
    names = {namer(Chem.MolToSmiles(mol))}
    for seed in range(5):
        names.add(namer(Chem.MolToSmiles(mol, doRandom=True)))
    assert names == {expected}, f"{smiles} non-deterministic / wrong: {names}"


# --------------------------------------------------------------------------- #
# Protect: the constructor must NOT perturb pure-fused / pure-bridged systems  #
# --------------------------------------------------------------------------- #
PROTECT = [
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("C1CCCc2ccccc12", "1,2,3,4-tetrahydronaphthalene"),  # tetralin (residual core)
    ("C1CC2CCC1C2", "norbornane"),                        # norbornane (pure von Baeyer, retained)
    ("C1C2CC3CC1CC(C2)C3", "adamantane"),                 # pure polycyclic-bridged
]


@pytest.mark.parametrize("smiles,expected", PROTECT)
def test_protect_unchanged(smiles, expected):
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------- #
# Fail-closed boundary: out-of-class systems MUST stay refused (no wrong name) #
# --------------------------------------------------------------------------- #
STILL_REFUSED = [
    "c1cc2nc3ccoc3cc2o1",          # difuropyridine (polycomponent P-25.3.4 -> G1b)
    "c1cc2nc3ccsc3cc2o1",          # furo+thieno+pyridine (polycomponent -> G1b)
    "C1Cc2ccccc2C13Cc1ccccc1C3",   # spirobi-indane (polyspiro -> G4)
]


@pytest.mark.parametrize("smiles", STILL_REFUSED)
def test_out_of_class_still_fails_closed(smiles):
    out = name_compound(smiles)
    assert is_failure_name(out), f"{smiles} must stay refused, got {out!r}"
    assert "cyclo[" not in out.lower()  # never a de-aromatised cage
