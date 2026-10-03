""" Phase G0 — fail-closed safety (DD7 S1 + crash).

G0 makes Orthonym REFUSE (return the 'unknown organic compound' descriptive
fallback in the default path, raise ``OrthonymLimitError`` on the opt-in path)
rather than emit a structurally-WRONG name for ring systems it cannot yet name
correctly:

  * a von-Baeyer / bicyclo cage that DROPS aromaticity (benzonorbornadiene ->
    'tricyclo[...]undec-3-ene' with the benzo ring desaturated);
  * a polycomponent fused system whose matched catalog core covers only PART of
    the fused ring system, so the leftover fused ring is mis-named as an acyclic
    substituent (difuropyridine -> '7-ethoxyfuro[3,2-b]pyridine');
  * and the two-metal species that CRASHED with
    'NoneType object has no attribute name' through the direct.name /
    raise_on_limit API.

Per guardrail A8 these test the OUTPUT/behaviour for the rule FAMILY (any
aromatic-in-cage / partial-fused-core / unnameable-acyclic molecule), not literal
canary rows. The correct PINs (bridged-fused, polycomponent fusion
, spirobi are Phase-G1+ builds; until then refusal is the
correct, honest behaviour.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym, OrthonymLimitError, name_compound
from orthonym.errors import is_failure_name
from orthonym.rules.polycyclic import vonbaeyer_cage_has_aromaticity


def _is_refused(smiles: str) -> bool:
    """Default-path fail-closed signal: the descriptive 'unknown...' fallback."""
    return is_failure_name(name_compound(smiles))


# --------------------------------------------------------------------------- #
# Family 1 — von-Baeyer / bicyclo cage that would drop aromaticity #
# --------------------------------------------------------------------------- #
# NOTE (Phase G1, DD7): the three single-bridge naphthalene systems
# G0 refused (benzonorbornadiene / 1,4-epoxy- / 1,4-ethano-) are now named
# CORRECTLY by the constructor (see tests/unit/test_g1_bridged_fused.py),
# so they left the fail-closed family.
#
# Task 12 fix a performance pass (wp6-tests; t12-research rest-of-suite item 32, user decision
# D-b): the spirobi-indane (DD7-spiro-1) left the family too -- it is now named
# "1',2,3,3'-tetrahydro-1,2'-spirobi[indene]" (pin_verified, OPSIN 2.9.0
# full-InChIKey exact; the Blue Book, component indicated hydrogen not
# cited, cf. '1,1'-spirobi[indene] (PIN)':10164), asserted below. The veto keeps
# two witnesses that still reach it (trace 2026-09-26, fresh process per input:
# vonbaeyer_cage_has_aromaticity returns True on the naming path, and the default
# path refuses with UNSUPPORTED_RING_SYSTEM): triptycene (PIN
# '9,10-dihydro-9,10-[1,2]benzenoanthracene', a bridged fused name, not
# built). Best-effort names it RT-exact (von Baeyer), so breadth holds.
#
# The benzo-fused bicyclo[2.2.2]octadiene left the family with the bridged fused
# builder: it is named '1,4-dihydro-1,4-ethanonaphthalene' (pin_verified, OPSIN
# 2.9.0 full-InChIKey exact), the pattern of (j) (the Blue Book,
# '1,4-dihydro-1,4-ethanoanthracene (PIN) (not 1,2,3,4-tetrahydro-1,4-
# ethenoanthracene)',:14399), asserted below.
AROMATIC_IN_CAGE = [
    "c1ccc2c(c1)C1c3ccccc3C2c2ccccc21",  # triptycene
]

BENZO_BICYCLOOCTADIENE = "C1=CC2CCC1c1ccccc12"
BENZO_BICYCLOOCTADIENE_NAME = "1,4-dihydro-1,4-ethanonaphthalene"

SPIROBI_INDANE = "C1Cc2ccccc2C13Cc1ccccc1C3"   # DD7-spiro-1
SPIROBI_INDANE_NAME = "1',2,3,3'-tetrahydro-1,2'-spirobi[indene]"


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
    assert ei.value.design_note_ref  # cites the AUTONOM analog


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


def test_spirobi_indane_is_named_exactly_and_deterministically():
    """DD7-spiro-1 left the fail-closed family (D-b): every spelling gives one name,
    and that name round-trips to the input's full InChIKey. This replaces the three
    refusal assertions it had (test_aromatic_in_cage_*[C1Cc2ccccc2C13Cc1ccccc1C3])."""
    from tests.support.jars import jar_or_skip
    from tests.support.rt_assert import assert_full_rt
    namer = Orthonym().name
    mol = Chem.MolFromSmiles(SPIROBI_INDANE)
    names = {namer(Chem.MolToSmiles(mol, doRandom=True)) for _ in range(4)}
    names.add(namer(Chem.MolToSmiles(mol)))
    assert names == {SPIROBI_INDANE_NAME}, names
    jar_or_skip()
    assert_full_rt(SPIROBI_INDANE_NAME, SPIROBI_INDANE)


def test_benzo_bicyclooctadiene_is_named_exactly_and_deterministically():
    """The benzo-fused bicyclo[2.2.2]octadiene left the fail-closed family: every
    spelling gives one name, the (j) bridged fused PIN (more
    noncumulative double bonds in the fused parent, the Blue Book,:14399), and
    that name round-trips to the input's full InChIKey. This replaces the three
    refusal assertions it had (test_aromatic_in_cage_*[C1=CC2CCC1c1ccccc12])."""
    from tests.support.jars import jar_or_skip
    from tests.support.rt_assert import assert_full_rt
    namer = Orthonym().name
    mol = Chem.MolFromSmiles(BENZO_BICYCLOOCTADIENE)
    names = {namer(Chem.MolToSmiles(mol, doRandom=True)) for _ in range(4)}
    names.add(namer(Chem.MolToSmiles(mol)))
    assert names == {BENZO_BICYCLOOCTADIENE_NAME}, names
    jar_or_skip()
    assert_full_rt(BENZO_BICYCLOOCTADIENE_NAME, BENZO_BICYCLOOCTADIENE)


# --------------------------------------------------------------------------- #
# Family 2 — polycomponent fused: matched core covers only PART of the system. #
# G0 (2026-06-18) made these fail closed to kill the phantom 'ethoxy'. #
# Phase G1b (2026-06-20) SUPERSEDES the outcome for the *unsubstituted* parents:#
# the polycomponent ortho-fusion constructor now names them CORRECTLY. The G0 #
# coverage veto remains the backstop for partial-core systems G1b cannot name #
# (SUBSTITUTED variants — G1b refuses substituted -> still fail closed). #
# --------------------------------------------------------------------------- #
PARTIAL_FUSED_CORE_NAMED = [
    ("c1cc2nc3ccoc3cc2o1", "difuro[3,2-b:2',3'-e]pyridine"),     # gold DD7-S1-safety-2
    ("c1cc2nc3ccsc3cc2o1", "furo[3,2-b]thieno[2,3-e]pyridine"),  # DD7-fusion-2
    # a phase SUPERSEDES the substituted fail-closed: the polycomponent
    # constructor now decorates the star parent against the canonical
    # numbering (verified OPSIN round-trip). Previously these produced the phantom
    # sub-fragment '5-methylfuran'/'5-chlorofuran' (suppressed only by in
    # production) — now the correct PIN, which is a strict accuracy improvement.
    ("Cc1cc2nc3ccoc3cc2o1", "2-methyldifuro[3,2-b:2',3'-e]pyridine"),
    ("Clc1cc2nc3ccoc3cc2o1", "2-chlorodifuro[3,2-b:2',3'-e]pyridine"),
]

# Partial-core systems the fusion path still CANNOT name -> the G0 veto fires.
# a phase fails closed at SOURCE when a substituent branch is unnameable
# (exotic element), so the emitted name can never silently omit a substituent.
PARTIAL_FUSED_CORE_STILL_CLOSED = [
    "[Si](C)(C)c1cc2nc3ccoc3cc2o1",  # silyl-difuropyridine: unnameable branch
]


@pytest.mark.parametrize("smiles,expected", PARTIAL_FUSED_CORE_NAMED)
def test_partial_fused_core_now_named_by_g1b(smiles, expected):
    # b supersedes the G0 fail-closed for the unsubstituted parent: the
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
    # a phase's source-level completeness check), a sibling of the
    # ring-coverage UNSUPPORTED_RING_SYSTEM veto; both are honest refusals.
    assert ei.value.code in {"UNSUPPORTED_RING_SYSTEM", "UNNAMEABLE"}


# --------------------------------------------------------------------------- #
# Family 3 — NoneType crash: unnameable two-metal / acyclic species #
# --------------------------------------------------------------------------- #
NO_CRASH_METAL = [
    "c1ccc(cc1)[Hg]c1ccc(cc1)[Sb](c1ccccc1)c1ccccc1",  # gold (Hg + Sb)
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


def test_general_acyclic_empty_pool_returns_none_not_crash(monkeypatch):
    """The general_acyclic catch-all must return None, not raise a raw
    AttributeError, when its candidate pool stays empty (the masked crash).

    Task 12 fix a performance pass (wp6-tests; t12-research rest-of-suite item 33): the old
    witness, the Hg + Sb gold species, no longer reaches that branch -- an earlier
    producer now names it '[4-(diphenylstibanyl)phenyl](phenyl)mercury' (OPSIN 2.9.0
    full-InChIKey exact; D-b), and no other input was found that does (a line
    trace, fresh process, over 8 metal/noble-gas candidates: 0 hits on the
    empty-pool return; the tracer itself was validated on 3 inputs that reach the
    check). So the branch is exercised directly: a pool whose best is None, as
    when RATIO_REJECT_FLOOR refuses the only candidate. The molecule-level half
    keeps the old witness's no-crash intent and asserts its exact round trip."""
    from tests.support.jars import jar_or_skip
    from tests.support.rt_assert import assert_full_rt
    smiles = "c1ccc(cc1)[Hg]c1ccc(cc1)[Sb](c1ccccc1)c1ccccc1"
    out = Orthonym().name(smiles)  # raise_on_limit=False -> must return a string, no crash
    assert isinstance(out, str)
    jar_or_skip()
    assert_full_rt(out, smiles)

    import orthonym.assembly.candidate_pool as _cp
    from orthonym.assembly.handlers.general_acyclic import name_general_acyclic

    class _EmptyPool:
        def add(self, *a, **k):
            return None          # every candidate refused

        def best(self):
            return None

    namer = Orthonym(style="pin")
    mol = Chem.MolFromSmiles("CCCC")
    features = namer._perceive(mol, "CCCC", Chem.CanonSmiles("CCCC"))
    namer._classify(features)
    monkeypatch.setattr(_cp, "get_current_pool", lambda: _EmptyPool())
    assert name_general_acyclic(features, mol=features.mol, style="pin") is None


# --------------------------------------------------------------------------- #
# Protect — fixes must NOT over-fire on correctly-nameable systems #
# --------------------------------------------------------------------------- #
PROTECT = {
    "C1C2CC3CC1CC(C2)C3": "adamantane",
    "C1CCC2CCCCC2C1": "decahydronaphthalene",
    "C1CC2CCC1C2": "bicyclo[2.2.1]heptane",  # R11 (2026-09-25, pre-existing-failures plan, Task 5)::9881, only adamantane/cubane are retained; "bicyclo[2.2.1]heptane (PIN)":2038
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
    """ regression guard: a saturated cage (norbornane, 7 atoms) carrying a
    pendant aromatic ring system STRICTLY LARGER than the cage (2-naphthyl, 10
    atoms) must NOT be refused — the aromaticity guard keys off the EXACT cage
    atoms the namer numbers, not the largest ring component, so the pendant
    aromatic (a separate single-bond-joined ring system) never enters the cage."""
    out = name_compound("C1CC2CCC1C2c1ccc2ccccc2c1")
    assert not is_failure_name(out), f"over-fire: pendant aromatic refused, got {out!r}"


# --------------------------------------------------------------------------- #
# Helper unit — aromaticity checked over the EXACT cage atom set  #
# --------------------------------------------------------------------------- #
def test_helper_checks_exact_cage_atoms():
    # An aromatic atom in the passed cage set -> True (benzo carbons are aromatic)
    mol = Chem.MolFromSmiles("C1C2C=CC1c1ccccc12")
    aromatic_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    saturated_idx = [a.GetIdx() for a in mol.GetAtoms() if not a.GetIsAromatic()]
    assert aromatic_idx and saturated_idx
    assert vonbaeyer_cage_has_aromaticity(mol, aromatic_idx) is True
    # A cage set containing ONLY the saturated (bridge) atoms -> False, even
    # though the molecule has aromatic atoms elsewhere (the invariant:
    # only the cage being numbered counts).
    assert vonbaeyer_cage_has_aromaticity(mol, saturated_idx) is False
    # Empty cage -> False (no atoms to be aromatic)
    assert vonbaeyer_cage_has_aromaticity(mol, ()) is False
