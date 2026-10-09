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
    ("C1CC2CCC1C2", "bicyclo[2.2.1]heptane"),  # R11 (2026-09-25, pre-existing-failures plan, Task 5)::9881, only adamantane/cubane are retained; "bicyclo[2.2.1]heptane (PIN)":2038
    ("C1C2CC3CC1CC(C2)C3", "adamantane"),                 # pure polycyclic-bridged
]


@pytest.mark.parametrize("smiles,expected", PROTECT)
def test_protect_unchanged(smiles, expected):
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------- #
# Ortho-fused small rings are fusion-named, never a bridge #
# --------------------------------------------------------------------------- #
# Ortho-fused small rings share a BOND with naphthalene (fusion nomenclature):
# the bridgeheads are adjacent + aromatic, so they are NOT a bridge and the
# name must NOT be '2,3-methano-/ethano-/propanonaphthalene' (a non-PIN that even
# OPSIN re-parses to the same structure, so the RT gate cannot catch it). These
# three used to be refused (no fusion producer for a carbocycle fused to a
# 6-membered ring that is not a 5-ring pair); the fusion name is the PIN now:
# "Five-membered ring requirement" (the Blue Book-23710): "Fusion
# nomenclature gives preferred IUPAC names only to compounds having at least two
# rings of at least five or more members. This requirement is not necessarily
# applied in general nomenclature, in which names such as cyclopropabenzene and
# cyclobutabenzene can be used." Here the naphthalene gives two six-membered rings, so
# the fusion name is allowed. OPSIN 2.9.0 full InChIKey = input's (assert_full_rt).
ORTHO_FUSED_SMALL_RING_NAMES = [
    ("c1ccc2cc3c(cc2c1)C3", "1H-cyclopropa[b]naphthalene"),
    ("c1ccc2cc3c(cc2c1)CC3", "1,2-dihydrocyclobuta[b]naphthalene"),
    ("c1ccc2cc3c(cc2c1)CCC3", "2,3-dihydro-1H-cyclopenta[b]naphthalene"),
]


@pytest.mark.parametrize("smiles,expected", ORTHO_FUSED_SMALL_RING_NAMES)
def test_ortho_fused_small_ring_is_fusion_named(smiles, expected, monkeypatch):
    # Production semantics: the validity gate is re-enabled (the suite's autouse
    # fixture disables it).
    import orthonym.namer as _nm
    from tests.support.rt_assert import assert_full_rt
    monkeypatch.setattr(_nm, "_DISABLE_VALIDITY_GATE", False, raising=False)
    out = name_compound(smiles)
    assert out == expected, f"{smiles}: {out!r} != {expected!r}"
    assert "methano" not in out and "ethano" not in out and "propano" not in out
    assert_full_rt(out, smiles)
