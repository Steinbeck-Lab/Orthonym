""" free-valence class: NO ring family may name ``=CH2`` as ``-yl``.

WHY THIS FILE EXISTS
--------------------
Commit 720d5b0c built the carbon free-valence class (``-yl`` single, ``-ylidene``
double, ``-ylidyne`` triple) but built it in only two of the ring-substituent
detectors. Two siblings -- the bicyclo one and the monocyclic-heterocycle one --
carried the textually identical defect (trace the exocyclic neighbour, count
carbons, call ``get_alkyl_name``, never look at the bond order) and each emitted
a name for a DIFFERENT molecule: ``C=C1CC2CCC1C2`` came out as
``2-methylbicyclo[2.2.1]heptane`` (C8H12 in, C8H14 out).

A source grep ("does every detector call the primitive?") would not have caught
that, because a NEW detector added tomorrow is not in any grep written today.
So this is a BEHAVIOURAL sweep instead: one representative exocyclic-ylidene
molecule per ring family, driven through the public entry point, asserting the
one property the class exists to guarantee --

    the emitted name never describes a different molecule than the input.

Abstaining is allowed. Naming correctly is allowed. Naming WRONGLY is not. A new
ring path that reintroduces the single-free-valence assumption fails here
whatever code it is written in, because the assertion is about the molecule that
comes back out, not about which function was called.

WHY BOTH JVM MODES
------------------
 (namer.py) re-perceives the emitted name through OPSIN and suppresses it
when it encodes a different molecule -- but it is deliberately fail-OPEN when no
JVM is present (namer.py ``_validity_gate_jar_present``, the guard: a
no-Java host must not have every name suppressed). The no-JVM path is therefore
the one where a wrong structure actually escapes to a user, and it is how the
original defect shipped. Both modes are exercised:

* ``jar_present`` -- production configuration, armed. Proves the release
  path is clean.
* ``no_jvm`` -- ``_find_opsin_jar`` patched to None, disarmed. Proves the
  PRODUCERS are clean, with no downstream net to hide behind. This is the mode
  with the teeth.

The round-trip verdict itself always uses the real OPSIN: the patch is applied
around the naming call only, and lifted before the name is checked.

References: IUPAC 2013 (free-valence morphology), /
(locants for the free valence).
"""
import pytest
from rdkit import Chem

import orthonym
from orthonym.validation import opsin_roundtrip
from tests.support.jars import jar_or_skip
from tests.support.default_tier import (  # noqa: E402
    assert_default_tier_declines,
    declined_pin_row,
    default_tier_rule_applies,
    strict_path_name,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "C=C1C2CC3CC1CC(C2)C3",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)



# One representative per ring family. Every one is a ring carbon carrying an
# exocyclic ``=CH2``: the smallest shape that distinguishes ``-ylidene`` from
# ``-yl``, and the shape whose mis-naming silently ADDS two hydrogens.
RING_FAMILY_YLIDENES = [
    # family SMILES expected PIN
    ("monocycle-carbocycle", "C=C1CCCCC1", "methylidenecyclohexane"),
    ("monocycle-heterocycle", "C=C1CCNCC1", "4-methylidenepiperidine"),
    ("bicyclo-bridged", "C=C1CC2CCC1C2", "2-methylidenebicyclo[2.2.1]heptane"),
    ("polycyclic-von-baeyer", "C=C1C2CC3CC1CC(C2)C3",
     "2-methylidenetricyclo[3.3.1.1^3,7]decane"),
    ("spiro", "C=C1CCC2(CC1)CCCC2", "8-methylidenespiro[4.5]decane"),
    ("ortho-fused", "C=C1Cc2ccccc2C1", None),
    ("ring-assembly", "C=C1CCC(C2CCCCC2)CC1", None),
    ("ring-as-substituent", "OCC=C1CCCCC1", "2-cyclohexylideneethan-1-ol"),
    ("chain-branch", "OC(=O)C(=C)CCCl", "4-chloro-2-methylidenebutanoic acid"),
]

# Families whose correct name this phase does not undertake to CONSTRUCT carry
# ``None`` above: for them the contract is only "never wrong". Where a PIN is
# stated it is additionally asserted, so a future regression that turns a
# correct name back into an abstention is also caught.

# Molecules that STILL name wrongly, for reasons outside this class. They are
# listed -- and asserted, as strict xfails -- rather than omitted, so the leak
# is recorded where the next person looks, and so the marker fails loudly the
# day the underlying defect is fixed.
#
# In each case the free-valence morphology is now CORRECT (the emitted name
# contains 'methylidene' / 'ethylideneamino'); what is wrong is something else,
# which is why fixing it belongs to a different phase:
#
# (decalin/tetralin used to be listed here: the producer named ONE ring of an
# ortho-fused pair as the parent and the other ring's atoms as an open
# 'butan-1-yl' chain. They are fusion-named now, FUSED_YLIDENE_NAMES below.)
# CC=NCC(=O)O -- the C=N is claimed TWICE, once by the correct
# 'ethylideneamino' prefix and again as an 'imino'. An atom double-count.
#
# Both abstain correctly in production (with a JVM); they escape
# only on a Java-less host, which is what the no_jvm parameter reproduces.
OUT_OF_CLASS_LEAKS = [
    ("imine-double-count", "CC=NCC(=O)O", "C=N claimed twice"),
    ("imine-double-count-long", "CC=NCCCC(=O)O", "C=N claimed twice"),
]


def _canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _name_under(mode, smiles, monkeypatch):
    """Name ``smiles`` with armed (``jar_present``) or not (``no_jvm``)."""
    import orthonym.namer as namer

    # The suite-wide autouse fixture disables the validity gate so tests can
    # assert raw producer output. Here the gate's behaviour IS the subject, so
    # it is re-armed and then, for no_jvm, disarmed the way a Java-less host
    # disarms it -- through the jar probe, not through the flag.
    monkeypatch.setattr(namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    if mode == "no_jvm":
        monkeypatch.setattr(
            opsin_roundtrip, "_find_opsin_jar", lambda *a, **k: None)
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        if mode == "jar_present":
            return _declined_pin_row(smiles)["name"]
        # The reduced mode without a jar is under the same emission rule (namer.py
        # ``_default_tier_policy_applies``): the default tier declines there too, and
        # the strict path's name, built with the rule off, stays under test.
        assert_default_tier_declines(smiles)
        return strict_path_name(smiles)
    try:
        return orthonym.name_compound(smiles, style="pin")
    except Exception:  # noqa: BLE001 - an exception is an abstention, not a wrong name
        return None


def _opsin_smiles(name):
    """Parse ``name`` with the REAL OPSIN, whatever the naming mode patched."""
    import importlib

    module = importlib.import_module(
        "orthonym.validation.opsin_roundtrip")
    # Re-read the attribute rather than closing over it, so a monkeypatch that
    # is still active cannot make the verdict pass vacuously.
    return module.opsin_parse.__wrapped__(name) if hasattr(
        module.opsin_parse, "__wrapped__") else module.opsin_parse(name)


def _is_abstention(name):
    from orthonym.errors import _DESCRIPTIVE_FALLBACK_NAMES

    return (not name) or name in _DESCRIPTIVE_FALLBACK_NAMES


@pytest.mark.parametrize("mode", ["jar_present", "no_jvm"])
@pytest.mark.parametrize(
    "family,smiles,expected_pin", RING_FAMILY_YLIDENES,
    ids=[case[0] for case in RING_FAMILY_YLIDENES])
def test_exocyclic_ylidene_never_names_a_different_molecule(
        family, smiles, expected_pin, mode, monkeypatch):
    """Every ring family: name it right or abstain -- never name it wrong."""
    jar_or_skip()  # OPSIN jar required to verify the emitted structure

    name = _name_under(mode, smiles, monkeypatch)
    monkeypatch.undo()

    if _is_abstention(name):
        assert expected_pin is None, (
            f"[{family}/{mode}] abstained on {smiles} but {expected_pin!r} is a "
            f"constructible PIN -- this is a coverage regression")
        return

    parsed = _opsin_smiles(name)
    if parsed is None:
        # Unparseable is a separate (emitter-discipline) concern; it cannot
        # mislead a reader about the structure, which is what this file guards.
        pytest.skip(f"[{family}/{mode}] OPSIN could not parse {name!r}")

    assert _canonical(parsed) == _canonical(smiles), (
        f"[{family}/{mode}] WRONG STRUCTURE: {smiles} named {name!r}, which "
        f"OPSIN reads back as {_canonical(parsed)}. A free valence of 2 was "
        f"named with a single-valence (-yl) morpheme (P-29.2).")


@pytest.mark.parametrize(
    "family,smiles,expected_pin", RING_FAMILY_YLIDENES,
    ids=[case[0] for case in RING_FAMILY_YLIDENES])
def test_exocyclic_ylidene_reaches_its_known_pin(
        family, smiles, expected_pin, monkeypatch):
    """Families whose PIN this phase constructs must actually reach it."""
    if expected_pin is None:
        pytest.skip(f"{family}: no PIN undertaken by this phase")
    jar_or_skip()  # OPSIN jar required

    name = _name_under("no_jvm", smiles, monkeypatch)
    monkeypatch.undo()
    assert name == expected_pin, (
        f"[{family}] {smiles} named {name!r}, expected {expected_pin!r}")


# Ortho-fused carbocycles with an exocyclic =CH2: the fusion parent (naphthalene, hydro
# prefixes) carries the 'methylidene' prefix. "Five-membered ring requirement"
# (the Blue Book-23710): fusion names are the preferred names of ortho-fused systems of
# rings of five or more members; hydro prefixes; the free valence 2 of =CH2 is
# 'methylidene'. Were: out-of-class leaks (one ring named as a parent, the other as an
# open chain). OPSIN 2.9.0 full InChIKey = the input's (assert_full_rt).
FUSED_YLIDENE_NAMES = [
    ("ortho-fused-decalin", "C=C1CCC2CCCCC2C1",
     "2-methylidenedecahydronaphthalene"),
    ("ortho-fused-tetralin", "C=C1CCc2ccccc2C1",
     "2-methylidene-1,2,3,4-tetrahydronaphthalene"),
]


@pytest.mark.parametrize("mode", ["jar_present", "no_jvm"])
@pytest.mark.parametrize(
    "label,smiles,expected", FUSED_YLIDENE_NAMES,
    ids=[case[0] for case in FUSED_YLIDENE_NAMES])
def test_fused_ylidene_is_fusion_named_and_round_trips(
        label, smiles, expected, mode, monkeypatch):
    from tests.support.rt_assert import assert_full_rt
    jar_or_skip()  # OPSIN jar required
    name = _name_under(mode, smiles, monkeypatch)
    monkeypatch.undo()
    assert name == expected, f"[{label}/{mode}] {name!r} != {expected!r}"
    assert_full_rt(name, smiles)


@pytest.mark.parametrize(
    "label,smiles,why",
    [pytest.param(*case, marks=pytest.mark.xfail(
        strict=True, reason=f"out-of-class leak: {case[2]}"))
     for case in OUT_OF_CLASS_LEAKS],
    ids=[case[0] for case in OUT_OF_CLASS_LEAKS])
def test_known_out_of_class_leaks_are_still_wrong(label, smiles, why,
                                                  monkeypatch):
    """Documented non-free-valence wrong-structure leaks (strict xfail).

    These fail for reasons this class cannot see. The marker is STRICT so that
    fixing the underlying defect turns this into a loud XPASS rather than
    quietly rotting. Each one abstains correctly with a JVM present.
    """
    jar_or_skip()  # OPSIN jar required
    name = _name_under("no_jvm", smiles, monkeypatch)
    monkeypatch.undo()
    assert not _is_abstention(name)
    parsed = _opsin_smiles(name)
    assert parsed is not None
    assert _canonical(parsed) == _canonical(smiles), (
        f"[{label}] still wrong ({why}): {name!r} -> {_canonical(parsed)}")


@pytest.mark.parametrize(
    "label,smiles,why", OUT_OF_CLASS_LEAKS,
    ids=[case[0] for case in OUT_OF_CLASS_LEAKS])
def test_known_out_of_class_leaks_abstain_in_production(label, smiles, why,
                                                        monkeypatch):
    """...and with a JVM present, catches every one of them."""
    jar_or_skip()  # OPSIN jar required
    name = _name_under("jar_present", smiles, monkeypatch)
    monkeypatch.undo()
    assert _is_abstention(name), (
        f"[{label}] SELF-01 did not suppress {name!r}")


# (smiles, fragment atoms, attachment atom, expected verdict prefix)
# The attachment atom is always a FRAGMENT atom; the linkage bond is the one
# leaving the fragment. ``None`` means the verdict must be fail-closed.
PRIMITIVE_CASES = [
    ("=CH2 on a ring", "C=C1CCCCC1", [0], 0, "methylidene"),
    ("=CH-CH3 on a ring", "CC=C1CCCCC1", [0, 1], 1, "ethylidene"),
    ("=C(CH3)2 on a ring", "CC(C)=C1CCCCC1", [0, 1, 2], 1, "propan-2-ylidene"),
    ("ring as the ylidene", "OCC=C1CCCCC1", [3, 4, 5, 6, 7, 8], 3,
     "cyclohexylidene"),
    ("=CH-CH3 on a nitrogen", "CC=NCC(=O)O", [0, 1], 1, "ethylidene"),
    # Decorated ring: the constructor does not build a substituted ring's
    # parent hydride, so the class fails CLOSED rather than saying 'cyclohexyl'.
    ("decorated ring ylidene", "OCC=C1CCCC(C)C1", [3, 4, 5, 6, 7, 8, 9], 3,
     None),
]


@pytest.mark.parametrize(
    "label,smiles,frag,attach,expected", PRIMITIVE_CASES,
    ids=[case[0] for case in PRIMITIVE_CASES])
def test_primitive_never_defers_on_a_double_attachment(
        label, smiles, frag, attach, expected):
    """The property stated at the PRIMITIVE, independent of any naming path.

    The behavioural sweep can only see families someone remembered to list.
    This pins the shared primitive itself: given a double attachment it must
    return either a prefix whose text spells two free valences, or an explicit
    fail-closed -- never a deferral, which is what tells a caller its ``-yl``
    path is safe.
    """
    from orthonym.assembly.substituent_enumerator import (
        carbon_free_valence_prefix)
    from orthonym.validation.name_morphemes import free_valence_morphology

    mol = Chem.MolFromSmiles(smiles)
    verdict = carbon_free_valence_prefix(mol, frag, attach)

    assert verdict.free_valence == 2, (
        f"[{label}] read free valence {verdict.free_valence}, expected 2")
    assert not verdict.defers, (
        f"[{label}] the primitive DEFERRED on a double attachment; the caller "
        f"would have emitted a -yl token")
    assert verdict.prefix == expected, (
        f"[{label}] prefix {verdict.prefix!r}, expected {expected!r} "
        f"({verdict.basis})")

    if expected is None:
        assert verdict.must_fail_closed
    else:
        morphology = free_valence_morphology(verdict.prefix)
        assert morphology.confident and morphology.free_valences == 2


def test_primitive_defers_on_single_bonds_and_heteroatom_attachments():
    """The two deferrals that keep existing names byte-identical.

    A single free valence must defer (or every ``-yl`` name in the project
    would have to be rebuilt here), and so must a non-carbon attachment -- an
    exocyclic ``=O`` is ``oxo``/``-one``, whose token spells no morpheme
    at all, so treating it as in-class would fail-close every ring ketone.
    """
    from orthonym.assembly.substituent_enumerator import (
        carbon_free_valence_prefix)

    single = Chem.MolFromSmiles("CC1CCCCC1")          # methylcyclohexane
    verdict = carbon_free_valence_prefix(single, [0], 0)
    assert verdict.defers and verdict.free_valence == 1
    assert not verdict.must_fail_closed

    ketone = Chem.MolFromSmiles("O=C1CCCCC1")         # cyclohexanone
    verdict = carbon_free_valence_prefix(ketone, [0], 0)
    assert verdict.free_valence == 2, "the bond order is still read honestly"
    assert not verdict.in_class, "an =O attachment is out of the carbon class"
    assert verdict.defers and not verdict.must_fail_closed
