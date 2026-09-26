""": complete-tier fails closed on unexpressed stereo + inline no-Java guard.

Two accuracy fixes to the general-engine ("complete" tier) path, both scoped so
the pin/default and valid/best-effort tiers are byte-identical.

FIX 2 (dropped stereo -> fail closed). (``_self_consistency_*``) is
deliberately CONSTITUTIONAL (atoms + bonds + charge, stereo-BLIND). A
general-engine emission that DROPS an E/Z or R/S descriptor the input carries
therefore passes and would ship a WRONG stereoisomer specification. The
universal stereo backstop (``_final_stereo_check``) detects it but is LOG-ONLY
for the ``handler='unknown'`` cohort that general emissions arrive as. So BOTH
general-engine acceptance points (the inline ``_name_impl`` block + the
late-recovery ``_try_general_engine_recovery``) now abstain when
``needs_stereo_injection`` is STILL True after the emission. Stereo the engine
DOES express (parent-scope ``_stereo_prefix`` / substituent-internal
``name_substituent``) leaves the predicate False, so those names still ship
WITH their stereo -- no over-abstention. The coverage-sweep witness was an
all-Z trienyl resorcinol emitted as the descriptor-less
``5-(heptadeca-5,8,11-trien-1-yl)benzene-1,3-diol``.

FIX 1 (no-Java -> inline abstain). Under ``complete`` the inline block accepted
``name = _eng.name`` after only an E1 check; the downstream
``_final_opsin_validity_gate`` FAILS OPEN with no OPSIN jar, so a no-Java
``complete`` run would ship an E1-only name from the NEW aggressive producers
(P1/P2/P5). E1 does not verify ring numbering / locants, so the inline block now
mirrors the late-recovery jar guard and keeps the abstention -- scoped to
``allow_aromatic_general`` (best-effort opts back in via
``general_fallback_unverified``).
"""
import glob
import shutil
from unittest import mock as _m

import pytest
from rdkit import Chem

import orthonym.namer as _namer_mod
from orthonym.namer import Orthonym, is_failure_name
from orthonym.rules.stereochemistry import needs_stereo_injection
from tests.support.jars import jar_or_none

pytestmark = pytest.mark.unit


# The coverage-sweep WRONG name: an all-Z trienyl resorcinol. The general engine
# emitted `5-(heptadeca-5,8,11-trien-1-yl)benzene-1,3-diol` with the 3 Z (E/Z)
# descriptors DROPPED -> a wrong stereoisomer specification. Must abstain.
TRIENYL = r"CCCCC/C=C\C/C=C\C/C=C\CCCCc1cc(O)cc(O)c1"
TRIENYL_DROPPED_NAME = "5-(heptadeca-5,8,11-trien-1-yl)benzene-1,3-diol"

# General-engine-routed molecules whose stereo the engine DOES express -> must
# STILL ship WITH stereo (no over-abstention). pin ABSTAINS; complete EMITS.
STEREO_EXPRESSED = [
    # 2026-09-25 (pre-existing-failures plan, Task 5, R17) change-asserted-value:
    # "NAMING OF STEREOISOMERS" (the Blue Book): stereodescriptors of
    # a substituent "are cited at the front of the corresponding prefix", preceded
    # by the locant -- "[(1R)-1-chloropropyl]benzene (PIN)" (:44668);
    # (:7478): the stereo parentheses count in the nesting order, hence '[(...)...]';
    # (c) (:2913): no '1-' on a monosubstituted benzene. OPSIN RT exact;
    # TRIAGE rows 123-125.
    ("C[C@H](Cl)COc1ccccc1", "[(2S)-2-chloropropoxy]benzene"),
    ("ClCC[C@@H](Cl)Oc1ccccc1", "[(1R)-1,3-dichloropropoxy]benzene"),
    # change-asserted-value: trimethylsilyl is a compound
    # substituent -> enclosed '(trimethylsilyl)'. RT-exact. R17 as above: the
    # stereo-bearing prefix takes square brackets; the locants stay (two prefixes
    # on a heterocycle).
    ("C[Si](C)(C)[n+]1ccc([C@@H](Cl)C)cc1",
     "4-[(1S)-1-chloroethyl]-1-(trimethylsilyl)pyridin-1-ium"),
]

# The former witness INLINE_P5 ("C[n+]1ccc(cc1)C", 1,4-dimethylpyridin-1-ium) was
# meant to route through the INLINE _name_impl general-engine block; the PIN path
# now names it first, so it was retired (Task 9 and Task 12, see below).

# The witness for the FIX 1 guard itself. Pre-existing-failures plan, Task 9
# (TRIAGE.csv row 122): INLINE_P5 is now named by the PIN path first, so it no
# longer reaches the inline block (trace 2026-09-26, fresh process: 0
# name_general calls, jar absent or present) and cannot witness the guard. This
# one does (same trace, jar absent, complete tier): the PIN path abstains,
# name_general returns the certified '2-silylpyridine' (the P1 monocycle
# producer), the guard's `_validity_gate_jar_present` call in the
# `_no_jar_abstain` expression runs with every other conjunct True, the name is
# not adopted, the late recovery's own jar guard also declines, and the output
# is 'unknown organic compound'. With the jar present the inline block adopts it
# (late recovery not called). OPSIN 2.9.0 RT exact.
INLINE_GUARD_WITNESS = "[SiH3]c1ccccn1"
INLINE_GUARD_WITNESS_NAME = "2-silylpyridine"


def _find_opsin_jar():
    """The pinned OPSIN jar via orthonym.jars (tests.support.jars), or None."""
    return jar_or_none()


@pytest.fixture
def production_gate(monkeypatch):
    """Re-enable the production OPSIN validity gate (the suite autouse
    fixture disables it). Skips when Java/OPSIN are unavailable -- without the
    gate the default path ships unverified names and the reproduce/emit
    semantics that surfaced the bug do not hold. Mirrors the P1 fixture."""
    if not shutil.which("java") or _find_opsin_jar() is None:
        pytest.skip("OPSIN/Java not available for production-gate semantics")
    monkeypatch.setattr(_namer_mod, "_DISABLE_VALIDITY_GATE", False)


def _complete(**kw):
    return Orthonym(style="pin", general_fallback=True,
                     allow_aromatic_general=True, **kw)


def _has_stereo_descriptor(name: str) -> bool:
    return any(tok in name for tok in
               ("(R)", "(S)", "(E)", "(Z)", "R)-", "S)-", "E)-", "Z)-"))


# --------------------------------------------------------------------------
# Predicate-level (gate-independent): the guard's trigger.
# --------------------------------------------------------------------------
def test_predicate_true_for_trienyl_dropped_stereo():
    """needs_stereo_injection detects the dropped E/Z -> the fail-closed trigger."""
    mol = Chem.MolFromSmiles(TRIENYL)
    assert needs_stereo_injection(mol, TRIENYL_DROPPED_NAME) is True


@pytest.mark.parametrize("smiles,name", STEREO_EXPRESSED)
def test_predicate_false_for_expressed_stereo(smiles, name):
    """A name that expresses its stereo -> predicate False -> NOT over-abstained."""
    mol = Chem.MolFromSmiles(smiles)
    assert needs_stereo_injection(mol, name) is False


# --------------------------------------------------------------------------
# FIX 2 integration (real OPSIN gate): dropped stereo -> abstain; expressed
# stereo -> still ships WITH the descriptor.
# --------------------------------------------------------------------------
def test_trienyl_resorcinol_no_longer_drops_its_stereo_under_complete(
        production_gate):
    """-FIX Item 1 re-baseline: this witness now SHIPS, stereo-complete.

    FIX 2's guard is unchanged and still correct — what changed is that this
    molecule no longer triggers it. The reason the descriptors were dropped was
    that ``_add_substituent_stereo`` reads only ATOM ``_CIPCode``, so an E/Z bond
    inside a substituent fragment was invisible; the alkenyl producer
    (``substituent_naming._name_unsaturated_chain``) now emits the located block
    itself via ``_located_stereo_block``, so all three Z descriptors are present
    and ``needs_stereo_injection`` is correctly False.

    Abstaining here would now be OVER-abstention — precisely what this file's own
    header warns against ("Stereo the engine DOES express... still ship WITH
    their stereo -- no over-abstention"). The emitted name round-trips EXACTLY:
    fed to the OPSIN jar it returns
    ``CCCCC/C=C\\C/C=C\\C/C=C\\CCCCc1cc(O)cc(O)c1``, the canonical input.

    ``### Stereodescriptors used in substitutive nomenclature``
    (``the Blue Book``): "*In preferred IUPAC names, stereodescriptors,
    preceded by a locant, must be cited to specify each stereogenic unit*" — the
    three located ``5Z,8Z,11Z`` descriptors are what that requires.
    """
    out = _complete().name(Chem.CanonSmiles(TRIENYL))
    # The descriptor-less spelling is the actual defect and must never return.
    assert out != TRIENYL_DROPPED_NAME
    assert not is_failure_name(out), (
        f"complete must no longer over-abstain on this witness, got {out!r}")
    # All three stereogenic double bonds expressed, in locant order.
    assert "5Z,8Z,11Z" in out, out
    #...and the guard is still armed: the OLD name would still trip it.
    assert needs_stereo_injection(
        Chem.MolFromSmiles(TRIENYL), TRIENYL_DROPPED_NAME) is True


@pytest.mark.parametrize("smiles,expected", STEREO_EXPRESSED)
def test_stereo_expressible_still_ships_with_stereo(smiles, expected,
                                                    production_gate):
    out = _complete().name(Chem.CanonSmiles(smiles))
    assert not is_failure_name(out), (
        f"OVER-ABSTENTION: engine expresses this stereo; must still ship {smiles!r}")
    assert _has_stereo_descriptor(out), (
        f"shipped name dropped its stereo descriptor: {out!r}")
    assert out == expected, f"{smiles}: {out!r} != {expected!r}"


# --------------------------------------------------------------------------
# FIX 1: inline no-Java abstain guard (validity gate fails OPEN without a jar).
# --------------------------------------------------------------------------
def test_inline_block_abstains_without_java_under_complete():
    """complete + jar absent -> the inline block must NOT ship an E1-only name
    from the new producers. INLINE_GUARD_WITNESS routes through the inline
    block (see its comment; the old witness INLINE_P5 is now named by the PIN
    path first), so this hits the FIX 1 guard. Control: with the jar present
    the same input is named by the inline block and the late recovery is not
    called, so the abstention above comes from the inline guard."""
    smi = Chem.CanonSmiles(INLINE_GUARD_WITNESS)
    with _m.patch("orthonym.namer._validity_gate_jar_present",
                  return_value=False):
        out = _complete().name(smi)
    assert is_failure_name(out), (
        f"inline no-Java guard must abstain under complete, got {out!r}")
    late = []
    real_late = Orthonym._try_general_engine_recovery

    def _spy_late(self, *a, **k):
        late.append(1)
        return real_late(self, *a, **k)

    with _m.patch("orthonym.namer._validity_gate_jar_present",
                  return_value=True), \
            _m.patch.object(Orthonym, "_try_general_engine_recovery",
                            _spy_late):
        out_jar = _complete().name(smi)
    assert out_jar == INLINE_GUARD_WITNESS_NAME and not late, (
        f"control: with the jar present the inline block must name the "
        f"witness, got {out_jar!r} (late recovery calls: {len(late)})")


def _name_with_inline_spies(namer, smi, jar_present):
    """Name ``smi`` with the jar presence forced; also return how often the inline
    block's ``general_engine.name_general`` and the late recovery ran."""
    import orthonym.assembly.general_engine as _ge
    engine, late = [], []
    real_engine = _ge.name_general
    real_late = Orthonym._try_general_engine_recovery

    def _spy_engine(*a, **k):
        engine.append(1)
        return real_engine(*a, **k)

    def _spy_late(self, *a, **k):
        late.append(1)
        return real_late(self, *a, **k)

    with _m.patch("orthonym.namer._validity_gate_jar_present",
                  return_value=jar_present), \
            _m.patch.object(_ge, "name_general", _spy_engine), \
            _m.patch.object(Orthonym, "_try_general_engine_recovery", _spy_late):
        out = namer.name(smi)
    return out, len(engine), len(late)


# Task 12 fix a performance pass (wp6-tests; TRIAGE.md ' outcome', follow-up 3): the two
# siblings below used INLINE_P5, which the PIN path names before the inline block
# (0 name_general calls, jar absent or present: trace 2026-09-26, fresh process), so
# they passed without reaching the code they describe. They now use
# INLINE_GUARD_WITNESS and assert that the inline block ran (name_general called)
# and the late recovery did not.
def test_inline_block_best_effort_still_ships_without_java():
    """best-effort OPTS IN to unverified emission via general_fallback_unverified
    -> the no-Java guard must NOT fire (behavior unchanged for that tier)."""
    smi = Chem.CanonSmiles(INLINE_GUARD_WITNESS)
    out, engine, late = _name_with_inline_spies(
        _complete(general_fallback_unverified=True), smi, jar_present=False)
    assert engine and not late, (
        f"the witness must reach the inline block (name_general calls {engine}, "
        f"late recovery calls {late})")
    assert out == INLINE_GUARD_WITNESS_NAME


def test_inline_jar_present_still_names_under_complete():
    """Positive control: with the jar present the inline block still names the
    witness (the FIX 1 guard only fires on jar ABSENCE)."""
    smi = Chem.CanonSmiles(INLINE_GUARD_WITNESS)
    out, engine, late = _name_with_inline_spies(_complete(), smi, jar_present=True)
    assert engine and not late, (
        f"the witness must reach the inline block (name_general calls {engine}, "
        f"late recovery calls {late})")
    assert out == INLINE_GUARD_WITNESS_NAME
