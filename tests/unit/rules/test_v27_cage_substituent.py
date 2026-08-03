"""v27 Phase 1 — unify parent/substituent recursion: von-Baeyer cages as `-yl`.

Tests the new ``_universal_cage_substituent_name`` (routes a detached ring
system through the audited ``analyze_cage_universal`` engine + free-valence
numbering) and its opt-in wiring through ``name_substituent`` /
``get_ring_substituent_name`` behind ``allow_mancude``.

Root-cause target (Blue Book P-29.2 / P-29.3.3-5): the cage engine names any
fused/bridged polycyclic as a PARENT; before P1 it was unreachable when the
ring system is a SUBSTITUENT (``_vonbaeyer_substituent_name`` only handles a
2-bridgehead bicyclo, ``_polycyclic_substituent_name`` then fails closed), so
77.8% multi-ring-assembly molecules abstained the instant a second ring system
had to be a prefix. These tests pin the restored symmetry AND that the PIN
default (``allow_mancude=False``) stays byte-identical.
"""
import pytest
from rdkit import Chem

from orthonym.rules.ring_substituents import (
    _universal_cage_substituent_name,
    _extract_ring_submol,
    get_ring_substituent_name,
    name_ring_system_substituent,
)
from orthonym.assembly.substituent_enumerator import name_substituent


def _detach(smiles):
    """Return (sub, attach_sub) for the ring system carrying the free valence.

    Finds the first ring atom bonded to a non-ring heavy atom (the attachment
    to the acyclic parent) and extracts the ring-only submol, mirroring what
    ``_polycyclic_substituent_name`` does before calling the cage namers.
    """
    mol = Chem.MolFromSmiles(smiles)
    ring_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    attach = None
    for a in ring_atoms:
        for n in mol.GetAtomWithIdx(a).GetNeighbors():
            if n.GetIdx() not in ring_atoms and n.GetAtomicNum() > 1:
                attach = a
                break
        if attach is not None:
            break
    sub, attach_sub = _extract_ring_submol(mol, ring_atoms, attach)
    return sub, attach_sub


def _ring_frag(smiles):
    """Return (mol, frag_ring_atoms_set, attach_ring_atom) in FULL-mol indices."""
    mol = Chem.MolFromSmiles(smiles)
    ring_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    attach = None
    for a in ring_atoms:
        for n in mol.GetAtomWithIdx(a).GetNeighbors():
            if n.GetIdx() not in ring_atoms and n.GetAtomicNum() > 1:
                attach = a
                break
        if attach is not None:
            break
    return mol, set(ring_atoms), attach


# ---------------------------------------------------------------------------
# Task 1 — bare saturated cages the narrow namers decline (tricyclo/adamantane)
# ---------------------------------------------------------------------------

def test_adamantane_cage_substituent():
    """Adamantane (tricyclo, >2 bridgeheads) is declined by
    ``_vonbaeyer_substituent_name`` (not ``is_bicyclo_system``); the universal
    namer names it -- and for adamantane it must use the RETAINED PIN stem.

    v30 P3-T1c: was ``"tricyclo[3.3.1.1^3,7]decan-2-yl"``.
    **P-23.7 "RETAINED NAMES FOR VON BAEYER PARENT HYDRIDES"**
    (``BlueBookV2/BlueBookV2.md:9879``): *"The retained names adamantane and
    cubane are used in general nomenclature and as preferred IUPAC names."*
    Table 2.6 (``:9885``) prints *"adamantane (PIN) tricyclo[3.3.1.1^3,7]decane"*
    -- retained name PIN, descriptor the ALTERNATIVE. The LOCANT is unchanged (2,
    a CH2 position): OPSIN 2.9.0 resolves ``adamantan-N-ol`` and
    ``tricyclo[3.3.1.1^3,7]decan-N-ol`` to the same InChIKey for every N in 1..10,
    so the two numberings coincide. Both whole-molecule forms
    (``2-(adamantan-2-yl)ethanol`` and the descriptor form) round-trip to this
    input's FULL InChIKey -- the change is the PREFERENCE, not the structure.
    """
    sub, attach = _detach("C12C(C3CC(CC(C1)C3)C2)CCO")  # 2-(adamantan-2-yl)ethanol
    assert _universal_cage_substituent_name(sub, attach) == "adamantan-2-yl"


def test_tricyclo_bridged_cage_substituent():
    sub, attach = _detach("C12C3C(C(CC31)C2)CCO")
    assert _universal_cage_substituent_name(sub, attach) == \
        "tricyclo[2.2.1.0^2,6]heptan-5-yl"


def test_bicyclo_cage_substituent():
    """A plain 2-bridgehead bicyclo is ALSO nameable by the universal namer
    (the existing ``_vonbaeyer_substituent_name`` still wins first in the
    producer tuple — this asserts the universal path agrees)."""
    sub, attach = _detach("C1CC2CCC(C1)C2CCO")
    assert _universal_cage_substituent_name(sub, attach) == \
        "bicyclo[3.2.1]octan-8-yl"


# ---------------------------------------------------------------------------
# Task 2 — free-valence locant + ene infix + terminal-e elision
# ---------------------------------------------------------------------------

def test_bridgehead_ene_cage_substituent():
    """The P-29.2 PIN skeleton: a bridgehead ene must keep its `n(m)` locant,
    the terminal `e` of the parent stem is elided before the `-yl` suffix, and
    the free-valence locant is cited."""
    sub, attach = _detach("C12(CCCCCCC(CCCCCC1)=C2)CCO")
    name = _universal_cage_substituent_name(sub, attach)
    assert name == "bicyclo[6.6.1]pentadec-1(15)-en-8-yl"
    assert name.endswith("-yl") and "-en-" in name  # ene infix + free valence


# ---------------------------------------------------------------------------
# Task 3 — mancude (aromatic) cages gated on allow_mancude; retained precedence
# ---------------------------------------------------------------------------

def test_mancude_cage_gated_off_by_default():
    """A fused-aromatic (mancude) cage as a substituent is REFUSED unless the
    complete tier is active (``allow_mancude=True``) — PIN default unchanged."""
    sub, attach = _detach("c1ccc2ccccc2c1CCO")  # naphthalene-2-yl carrier
    assert _universal_cage_substituent_name(sub, attach, allow_mancude=False) is None


def test_mancude_cage_named_when_enabled():
    sub, attach = _detach("c1ccc2ccccc2c1CCO")
    assert _universal_cage_substituent_name(sub, attach, allow_mancude=True) == \
        "bicyclo[4.4.0]deca-1,3,5,7,9-pentaen-5-yl"


def test_retained_aromatic_precedence_over_polyene():
    """At the ``get_ring_substituent_name`` level the retained catalog wins:
    naphthalene stays ``naphthalen-2-yl``, the von-Baeyer polyene never leaks —
    even with the complete tier on."""
    mol, frag, attach = _ring_frag("c1ccc2ccccc2c1CCO")
    name = get_ring_substituent_name(mol, tuple(sorted(frag)), attach,
                                     allow_mancude=True)
    # retained catalog wins (a naphthalen-N-yl, NOT the bicyclo[4.4.0]…polyene);
    # this SMILES attaches at the peri position 1.
    assert name == "naphthalen-1-yl"


# ---------------------------------------------------------------------------
# Task 4 — fail-closed guards (never emit a wrong / coverage-incomplete cage)
# ---------------------------------------------------------------------------

def test_spiro_fragment_fails_closed():
    """analyze_cage_universal refuses spiro (< 2 bridgeheads) — Phase 3 scope —
    so the universal cage namer abstains rather than guessing."""
    sub, attach = _detach("C1CCC2(C1)CCC(CC2)CCO")  # spiro[4.5]decane carrier
    assert _universal_cage_substituent_name(sub, attach, allow_mancude=False) is None
    assert _universal_cage_substituent_name(sub, attach, allow_mancude=True) is None


# ---------------------------------------------------------------------------
# Wiring — allow_mancude threads through name_substituent; default byte-identical
# ---------------------------------------------------------------------------

def test_name_substituent_default_declines_cage():
    """PIN default (``allow_mancude=False``): the cage substituent is NOT named
    by the new path — ``name_substituent`` returns the 'substituent' sentinel
    (whole molecule then fails closed), exactly as before P1."""
    mol, frag, attach = _ring_frag("C12C(C3CC(CC(C1)C3)C2)CCO")  # adamantan-2-yl
    # attach in full-mol indices is a ring atom bonded to the chain; name the
    # ring fragment as a substituent.
    result = name_substituent(mol, frag, attach)
    assert result == "substituent"


def test_name_substituent_complete_names_cage():
    """Complete tier (``allow_mancude=True``): the cage substituent is named.

    v30 P3-T1c: was ``"tricyclo[3.3.1.1^3,7]decan-2-yl"``; adamantane's retained
    name is the PIN per **P-23.7 "RETAINED NAMES FOR VON BAEYER PARENT HYDRIDES"**
    (``BlueBookV2/BlueBookV2.md:9879``, Table 2.6 at ``:9885``). Same locant, same
    structure -- see ``test_adamantane_cage_substituent`` for the full
    justification and the OPSIN locant-equivalence evidence.
    """
    mol, frag, attach = _ring_frag("C12C(C3CC(CC(C1)C3)C2)CCO")
    result = name_substituent(mol, frag, attach, allow_mancude=True)
    assert result == "adamantan-2-yl"


def test_name_ring_system_substituent_threads_flag():
    """The FLAG THREADING is the property under test here -- ``allow_mancude=False``
    must decline and ``True`` must name -- so the route assertion stays.

    v30 P3-T1c: the expected name was ``"tricyclo[3.3.1.1^3,7]decan-2-yl"``;
    adamantane's retained name is the PIN per **P-23.7 "RETAINED NAMES FOR VON
    BAEYER PARENT HYDRIDES"** (``BlueBookV2/BlueBookV2.md:9879``, Table 2.6 at
    ``:9885``). See ``test_adamantane_cage_substituent`` for the evidence.
    """
    mol, frag, attach = _ring_frag("C12C(C3CC(CC(C1)C3)C2)CCO")
    assert name_ring_system_substituent(
        mol, sorted(frag), attach, allow_enumerator_fallback=False,
        allow_mancude=False) is None
    assert name_ring_system_substituent(
        mol, sorted(frag), attach, allow_enumerator_fallback=False,
        allow_mancude=True) == "adamantan-2-yl"
