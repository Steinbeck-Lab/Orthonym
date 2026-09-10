""" Phase G1 — bridged-fused constructor (DD7, bridged half).

G1 turns the G0 fail-closed refusals for the *bridged-fused* class into correct
IUPAC PINs. A bridged fused ring system is a fused ring system
(the recognised parent, e.g. naphthalene) plus one or more bridges across it.
The constructor:

  * identifies the bridge(s) (the minimal atom set whose excision leaves a
    recognised fused parent),
  * numbers the residual fused parent by reusing the partial-saturation engine
    (so the bridgeheads receive their true IUPAC locants, e.g. 1 and 4),
  * cites bridge prefixes (methano/ethano/epoxy/...) and hydro prefixes TOGETHER
    in alphanumerical order ignoring multiplying prefixes
    ('epoxy' < 'ethano' < 'hydro' < 'methano'), each with its own locant set
    (Blue Book; verified against the PIN examples
    '1,4-dihydro-1,4-methanonaphthalene', '1,4-epoxy-5,8-methanonaphthalene').

Per guardrail A8 these test the rule FAMILY (any single-bridge-over-recognised-
fused-aromatic-parent) plus determinism and OPSIN constitutional round-trip,
not just literal canary rows. Systems OUTSIDE the handled class (polycomponent
fusion, polyspiro) MUST stay G0 fail-closed — never a wrong name.
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
    ("C1=CC2OC1c1ccccc12", "1,4-dihydro-1,4-epoxynaphthalene"),     # epoxy bridge
    ("C1CC2CCC1c1ccccc12", "1,2,3,4-tetrahydro-1,4-ethanonaphthalene"),  # ethano bridge
]


def _ik(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToInchiKey(m) if m else None


@pytest.mark.parametrize("smiles,expected", BRIDGED_FUSED_GOLD)
def test_bridged_fused_exact_pin(smiles, expected):
    """The constructor emits the exact PIN (gold name-exact-match)."""
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
# Protect: the constructor must NOT perturb pure-fused / pure-bridged systems #
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
    # NOTE: difuropyridine + furo+thieno+pyridine were here as polycomponent
    # out-of-class examples; Phase G1b (2026-06-20) now NAMES them correctly
    # (difuro[3,2-b:2',3'-e]pyridine etc.) via the polycomponent ortho-fusion
    # constructor, so they moved out of this fail-closed list.
    # Wave-2 completion B4 (2026-07-07): spirobi-indane
    # C1Cc2ccccc2C13Cc1ccccc1C3 moved OUT of this list — the cata-fused skip
    # in is_polycyclic_system lets the spirobi path name it correctly as
    # "1,2'-spirobi[indane]" (OPSIN-RT verified); it was previously refused
    # only because the VB misroute raised before the spirobi check.
    # CRITICAL-1 (code review): ortho-fused small rings share a BOND with naphthalene
    # (fusion nomenclature, e.g. 1H-cyclopropa[b]naphthalene) — the bridgeheads are
    # adjacent + aromatic, so they are NOT a bridge. Must NOT emit
    # '2,3-methano-/ethano-/propanonaphthalene' (a non-PIN that even OPSIN re-parses
    # to the same structure, so the RT gate can't catch it).
    "c1ccc2cc3c(cc2c1)C3",         # cyclopropa[b]naphthalene
    "c1ccc2cc3c(cc2c1)CC3",        # cyclobuta[b]naphthalene
    "c1ccc2cc3c(cc2c1)CCC3",       # cyclopenta[b]naphthalene
    # CRITICAL-2 (code review): a composite / multi-heteroatom bridge would DROP atoms
    # via the length-blind heteroatom prefix (epidioxy -> 'epoxy' loses an O). Must
    # fail closed until the composite-bridge grammar is built.
    "C1=CC2OOC1c1ccccc12",         # -O-O- (epidioxy bridge)
    "C1=CC2COCC1c1ccccc12",        # -CH2-O-CH2- (composite bridge)
    "C1=CC2CSCC1c1ccccc12",        # -CH2-S-CH2- (composite bridge)
]


@pytest.mark.parametrize("smiles", STILL_REFUSED)
def test_out_of_class_still_fails_closed(smiles, monkeypatch):
    # Assert the PRODUCTION fail-closed behavior: the suite's autouse fixture
    # disables the validity gate, but these out-of-class bridged systems
    # are fail-closed IN PRODUCTION via that gate (a raw benzene/parent candidate
    # is suppressed). Re-enable it. (Wave-close note: p8's benzene parent-selection
    # chokepoint made a few of these emit a raw 'benzene' pre-gate — a handler-level
    # tightening is a documented follow-up; production stays correct via the gate.)
    import orthonym.namer as _nm
    monkeypatch.setattr(_nm, "_DISABLE_VALIDITY_GATE", False, raising=False)
    out = name_compound(smiles)
    assert is_failure_name(out), f"{smiles} must stay refused, got {out!r}"
    assert "cyclo[" not in out.lower()  # never a de-aromatised cage
