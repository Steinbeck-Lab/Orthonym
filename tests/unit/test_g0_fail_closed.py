"""v22 Phase G0 — fail-closed safety (DD7 S1 + P-69 crash).

G0 makes Orthonym REFUSE (return the 'unknown organic compound' descriptive
fallback in the default path, raise ``OrthonymLimitError`` on the opt-in path)
rather than emit a structurally-WRONG name for ring systems it cannot yet name
correctly:

  * a von-Baeyer / bicyclo cage that DROPS aromaticity (benzonorbornadiene ->
    'tricyclo[...]undec-3-ene' with the benzo ring desaturated);
  * a polycomponent fused system whose matched catalog core covers only PART of
    the fused ring system, so the leftover fused ring is mis-named as an acyclic
    substituent (difuropyridine -> '7-ethoxyfuro[3,2-b]pyridine');
  * and the P-69 two-metal species that CRASHED with
    'NoneType object has no attribute name' through the direct .name() /
    raise_on_limit API.

Per guardrail A8 these test the OUTPUT/behaviour for the rule FAMILY (any
aromatic-in-cage / partial-fused-core / unnameable-acyclic molecule), not literal
canary rows. The correct PINs (bridged-fused P-25.4, polycomponent fusion
P-25.3.4, spirobi P-24.3) are Phase-G1+ builds; until then refusal is the
correct, honest behaviour.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym, OrthonymLimitError, name_compound
from orthonym.errors import is_failure_name
from orthonym.rules.polycyclic import vonbaeyer_cage_has_aromaticity


def _is_refused(smiles: str) -> bool:
    """Default-path fail-closed signal: the descriptive 'unknown ...' fallback."""
    return is_failure_name(name_compound(smiles))


# --------------------------------------------------------------------------- #
# Family 1 — von-Baeyer / bicyclo cage that would drop aromaticity            #
# --------------------------------------------------------------------------- #
# NOTE (v22 Phase G1, DD7 COV-01): the three single-bridge naphthalene systems
# G0 refused (benzonorbornadiene / 1,4-epoxy- / 1,4-ethano-) are now named
# CORRECTLY by the P-25.4 constructor (see tests/unit/test_g1_bridged_fused.py),
# so they left the fail-closed family. The polyspiro spirobi-indane stays refused
# (its spirobi[indane] PIN is a Phase-G4 build; is_spiro_system rejects
# polycyclic-component spiro, so it still routes to von Baeyer and the aromaticity
# veto fires).
AROMATIC_IN_CAGE = [
    "C1Cc2ccccc2C13Cc1ccccc1C3",  # spirobi-indane (DD7-spiro-1; polyspiro -> G4)
]


@pytest.mark.parametrize("smiles", AROMATIC_IN_CAGE)
def test_aromatic_in_cage_fails_closed_default(smiles):
    out = name_compound(smiles)
    assert is_failure_name(out), f"{smiles} must refuse, got {out!r}"
    # must NOT emit a von-Baeyer / bicyclo descriptor (the wrong de-aromatised cage)
    assert "cyclo[" not in out.lower()


@pytest.mark.parametrize("smiles", AROMATIC_IN_CAGE)
def test_aromatic_in_cage_raises_named_limit(smiles):
    with pytest.raises(OrthonymLimitError) as ei:
        Orthonym().name(smiles, raise_on_limit=True)
    assert ei.value.code == "UNSUPPORTED_RING_SYSTEM"
    assert ei.value.heritage_ref  # cites the HERITAGE analog


@pytest.mark.parametrize("smiles", AROMATIC_IN_CAGE)
def test_aromatic_in_cage_is_deterministic(smiles):
    """Same structure, any SMILES spelling -> the same (refusal) name."""
    namer = Orthonym().name
    mol = Chem.MolFromSmiles(smiles)
    names = set()
    for seed in range(4):
        respelled = Chem.MolToSmiles(mol, doRandom=True)  # stochastic re-spelling
        names.add(namer(respelled))
    names.add(namer(Chem.MolToSmiles(mol)))  # canonical
    assert len(names) == 1, f"{smiles} non-deterministic: {names}"
    assert is_failure_name(next(iter(names)))


# --------------------------------------------------------------------------- #
# Family 2 — polycomponent fused: matched core covers only PART of the system. #
# G0 (2026-06-18) made these fail closed to kill the phantom 'ethoxy'. v22      #
# Phase G1b (2026-06-20) SUPERSEDES the outcome for the *unsubstituted* parents:#
# the polycomponent ortho-fusion constructor now names them CORRECTLY. The G0   #
# coverage veto remains the backstop for partial-core systems G1b cannot name   #
# (SUBSTITUTED variants — G1b refuses substituted -> still fail closed).        #
# --------------------------------------------------------------------------- #
PARTIAL_FUSED_CORE_NAMED = [
    ("c1cc2nc3ccoc3cc2o1", "difuro[3,2-b:2',3'-e]pyridine"),     # gold DD7-S1-safety-2
    ("c1cc2nc3ccsc3cc2o1", "furo[3,2-b]thieno[2,3-e]pyridine"),  # DD7-fusion-2
    # v26 BP-4 Phase 2 SUPERSEDES the substituted fail-closed: the polycomponent
    # constructor now decorates the star parent against the canonical P-25.3.3
    # numbering (verified OPSIN round-trip). Previously these produced the phantom
    # sub-fragment '5-methylfuran'/'5-chlorofuran' (suppressed only by SELF-01 in
    # production) — now the correct PIN, which is a strict accuracy improvement.
    ("Cc1cc2nc3ccoc3cc2o1", "2-methyldifuro[3,2-b:2',3'-e]pyridine"),
    ("Clc1cc2nc3ccoc3cc2o1", "2-chlorodifuro[3,2-b:2',3'-e]pyridine"),
]

# Partial-core systems the fusion path still CANNOT name -> the G0 veto fires.
# BP-4 Phase 2 fails closed at SOURCE when a substituent branch is unnameable
# (exotic element), so the emitted name can never silently omit a substituent.
PARTIAL_FUSED_CORE_STILL_CLOSED = [
    "[Si](C)(C)c1cc2nc3ccoc3cc2o1",  # silyl-difuropyridine: unnameable branch
]


@pytest.mark.parametrize("smiles,expected", PARTIAL_FUSED_CORE_NAMED)
def test_partial_fused_core_now_named_by_g1b(smiles, expected):
    # v22 G1b supersedes the G0 fail-closed for the unsubstituted parent: the
    # polycomponent constructor names it correctly (and still emits no phantom).
    out = name_compound(smiles)
    assert out == expected, f"{smiles} -> {out!r}, expected {expected!r}"
    assert "ethoxy" not in out.lower()  # no phantom-substituent regression


@pytest.mark.parametrize("smiles", PARTIAL_FUSED_CORE_STILL_CLOSED)
def test_partial_fused_core_fails_closed(smiles):
    out = name_compound(smiles)
    assert is_failure_name(out), f"{smiles} must refuse, got {out!r}"
    # the phantom-substituent signature was a spurious 'ethoxy'/'oxy' prefix
    assert "ethoxy" not in out.lower()


@pytest.mark.parametrize("smiles", PARTIAL_FUSED_CORE_STILL_CLOSED)
def test_partial_fused_core_raises_named_limit(smiles):
    with pytest.raises(OrthonymLimitError) as ei:
        Orthonym().name(smiles, raise_on_limit=True)
    # A classified fail-closed refusal (not a crash). The silyl-difuropyridine
    # example fails via UNNAMEABLE (its substituent branch is unnameable —
    # BP-4 Phase 2's source-level completeness check), a sibling of the
    # ring-coverage UNSUPPORTED_RING_SYSTEM veto; both are honest refusals.
    assert ei.value.code in {"UNSUPPORTED_RING_SYSTEM", "UNNAMEABLE"}


# --------------------------------------------------------------------------- #
# Family 3 — P-69 NoneType crash: unnameable two-metal / acyclic species      #
# --------------------------------------------------------------------------- #
NO_CRASH_METAL = [
    "c1ccc(cc1)[Hg]c1ccc(cc1)[Sb](c1ccccc1)c1ccccc1",  # gold P-69 (Hg + Sb)
    "[Hg]([Hg]C)C",                                      # two covalent Hg
    "c1ccccc1[Hg][Hg]c1ccccc1",
]


@pytest.mark.parametrize("smiles", NO_CRASH_METAL)
def test_metal_species_do_not_crash(smiles):
    # default path: a string (never an exception)
    out = name_compound(smiles)
    assert isinstance(out, str) and out
    # opt-in path: a NAMED limit, never a raw AttributeError/NoneType crash
    try:
        Orthonym().name(smiles, raise_on_limit=True)
    except OrthonymLimitError:
        pass  # acceptable: a named, classified refusal
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"{smiles} crashed with {type(exc).__name__}: {exc}")


def test_general_acyclic_empty_pool_returns_none_not_crash():
    """The direct .name() API must not raise a raw AttributeError when the
    general_acyclic catch-all builds no candidate (the masked P-69 crash).

    .name() returns the bare failure signal (empty string) — the descriptive
    'mercury compound (not supported)' wrapper is applied by name_compound;
    what matters here is that no exception escapes."""
    smiles = "c1ccc(cc1)[Hg]c1ccc(cc1)[Sb](c1ccccc1)c1ccccc1"
    out = Orthonym().name(smiles)  # raise_on_limit=False -> must return a string, no crash
    assert isinstance(out, str)
    assert is_failure_name(out)  # empty / 'unknown' — a failure signal, not a wrong name
    # name_compound applies the descriptive fallback wrapper
    assert "not supported" in name_compound(smiles)


# --------------------------------------------------------------------------- #
# Protect — fixes must NOT over-fire on correctly-nameable systems            #
# --------------------------------------------------------------------------- #
PROTECT = {
    "C1C2CC3CC1CC(C2)C3": "adamantane",
    "C1CCC2CCCCC2C1": "decahydronaphthalene",
    "C1CC2CCC1C2": "norbornane",
    "C1CC2CC1C=C2": "bicyclo[2.2.1]hept-2-ene",
    "c1ccc2ncccc2c1": "quinoline",
    "Cc1ccc2ccccc2n1": "2-methylquinoline",
    "c1ccc2[nH]ccc2c1": "1H-indole",
    "c1ccc2ccccc2c1": "naphthalene",
}


@pytest.mark.parametrize("smiles,expected", list(PROTECT.items()))
def test_protect_rows_still_named(smiles, expected):
    assert name_compound(smiles) == expected


def test_pendant_aromatic_substituent_not_refused():
    """A phenyl attached by a single bond to a saturated cage is a SUBSTITUENT,
    not part of the von-Baeyer ring system -> must still be named, not refused."""
    out = name_compound("c1ccccc1C12CC3CC(C1)CC(C2)C3")  # 1-phenyladamantane
    assert not is_failure_name(out)
    assert "phenyl" in out.lower()


def test_2_phenylquinoline_not_refused_by_core_coverage():
    """The pendant phenyl is a separate fused ring system, so the fused-core
    coverage guard must NOT trip on it."""
    out = name_compound("c1ccc(-c2ccc3ccccc3n2)cc1")
    assert not is_failure_name(out)
    assert "phenyl" in out.lower()


def test_pendant_aromatic_larger_than_cage_not_refused():
    """WR-01 regression guard: a saturated cage (norbornane, 7 atoms) carrying a
    pendant aromatic ring system STRICTLY LARGER than the cage (2-naphthyl, 10
    atoms) must NOT be refused — the aromaticity guard keys off the EXACT cage
    atoms the namer numbers, not the largest ring component, so the pendant
    aromatic (a separate single-bond-joined ring system) never enters the cage."""
    out = name_compound("C1CC2CCC1C2c1ccc2ccccc2c1")
    assert not is_failure_name(out), f"over-fire: pendant aromatic refused, got {out!r}"


# --------------------------------------------------------------------------- #
# Helper unit — aromaticity checked over the EXACT cage atom set (WR-01)       #
# --------------------------------------------------------------------------- #
def test_helper_checks_exact_cage_atoms():
    # An aromatic atom in the passed cage set -> True (benzo carbons are aromatic)
    mol = Chem.MolFromSmiles("C1C2C=CC1c1ccccc12")
    aromatic_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    saturated_idx = [a.GetIdx() for a in mol.GetAtoms() if not a.GetIsAromatic()]
    assert aromatic_idx and saturated_idx
    assert vonbaeyer_cage_has_aromaticity(mol, aromatic_idx) is True
    # A cage set containing ONLY the saturated (bridge) atoms -> False, even
    # though the molecule has aromatic atoms elsewhere (the WR-01 invariant:
    # only the cage being numbered counts).
    assert vonbaeyer_cage_has_aromaticity(mol, saturated_idx) is False
    # Empty cage -> False (no atoms to be aromatic)
    assert vonbaeyer_cage_has_aromaticity(mol, ()) is False
