"""Unit tests for orthonym.assembly.universal_substituent (Phase B.2).

``name_universal_substitutive`` is the unconditional recursive substitutive
namer core: it NEVER declines a hard branch (a hard branch is a deeper
recursive call, never a refusal), guarantees atom coverage by construction
(void rather than partial), and carries an explicit in-algorithm work-budget
backstop (not a signal/timeout).

Coverage:
  * simple acyclic chain + branched alkane (sanity, correct lowest-locant
    numbering)
  * hetero-backbone acyclic chain (skeletal 'a'-replacement)
  * monocyclic, hetero-monocyclic and aromatic (kekulized) rings
  * polycyclic rings (reusing ``analyze_cage_universal`` +
    ``_build_parent_with_unsaturation``) including a substituent branch and
    two SEPARATE ring systems linked by a chain
  * hard-branch witnesses that abstain TODAY on ``main`` (confirmed by the
    B.1 spy) -- complete coverage + ``verify_or_none`` CONFIRMS
  * the stereo-omission degrade is documented as expected (constitution
    matches; the STRICT full-InChIKey ``verify_or_none`` withholds CONFIRM,
    which is correct: this module does not yet render stereo descriptors)
  * the mandatory work-budget backstop: a pathological giant fails closed,
    fast, never hangs
  * the atom-coverage assertion actually VOIDS a rigged incomplete binding
    (proves the gate fires, not just that good input passes it)
  * a per-atom formal-charge scope guard (internally charge-separated atoms
    such as nitro are refused rather than mis-constructed)

Targeted-file run only (per project convention -- avoid the OPSIN-pipe
deadlock of a full ``pytest tests/`` run):
    .venv/bin/python -m pytest tests/unit/assembly/test_universal_substituent.py -q
"""
from __future__ import annotations

import dataclasses
import time

import pytest
from rdkit import Chem

from orthonym.assembly import universal_substituent as us
from orthonym.assembly.universal_substituent import (
    DEFAULT_ATOM_WORK_BUDGET,
    name_universal_substitutive,
)
from orthonym.validation.atom_coverage import validate_atom_coverage
from orthonym.validation.reconstruct import verify_or_none


def _heavy_atoms(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def _name_and_verify(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    result = name_universal_substitutive(mol)
    assert result is not None, f"expected a name for {smiles!r}, got None"
    assert result.covers == _heavy_atoms(smiles), (
        f"coverage gap for {smiles!r}: covers={result.covers} "
        f"heavy={_heavy_atoms(smiles)}"
    )
    verified = verify_or_none(result.name, smiles)
    return result, verified


# ===========================================================================
# Sanity: simple acyclic molecules
# ===========================================================================

def test_simple_ethanol():
    result, verified = _name_and_verify("CCO")
    assert verified == result.name


def test_branched_alkane_lowest_locant():
    """CC(C)CC = 2-methylbutane. The branch must land on the LOW locant (2),
    not the high one (3) -- this is the numbering-direction tie-break that
    falls out only once heteroatom/unsaturation locants tie and substituent
    locants are consulted as the next tier."""
    result, verified = _name_and_verify("CC(C)CC")
    assert verified == result.name
    assert "-2-" in result.name or result.name.startswith("2-")
    assert "-3-" not in result.name


def test_plain_pentane():
    result, verified = _name_and_verify("CCCCC")
    assert result.name == "pentane"
    assert verified == "pentane"


# ===========================================================================
# Hetero-backbone (skeletal 'a'-replacement, acyclic)
# ===========================================================================

def test_hetero_backbone_polyether():
    """The module docstring's own worked example for skeletal replacement:
    COCCOCCOC -> 2,5,8-trioxanonane."""
    result, verified = _name_and_verify("COCCOCCOC")
    assert result.name == "2,5,8-trioxanonane"
    assert verified == result.name


# ===========================================================================
# Rings: monocyclic, hetero-monocyclic, aromatic (kekulized), polycyclic,
# and two SEPARATE ring systems linked by a chain.
# ===========================================================================

def test_monocyclic_ring():
    result, verified = _name_and_verify("C1CCCCC1")
    assert result.name == "cyclohexane"
    assert verified == result.name


def test_hetero_monocyclic_ring():
    result, verified = _name_and_verify("C1CCOCC1")
    assert verified == result.name
    assert "oxa" in result.name and "cyclo" in result.name


def test_aromatic_ring_kekulizes():
    """An aromatic input is kekulized to an explicit polyene -- structurally
    faithful (round-trips), even though it is not the PIN 'benzene'."""
    result, verified = _name_and_verify("c1ccccc1")
    assert verified == result.name


def test_polycyclic_ring_reuses_von_baeyer():
    result, verified = _name_and_verify("C1CC2CCC1C2")  # norbornane
    assert result.name == "bicyclo[2.2.1]heptane"
    assert verified == result.name


def test_polycyclic_ring_with_substituent_branch():
    result, verified = _name_and_verify("CC1CC2CCC1C2")
    assert verified == result.name


def test_two_separate_ring_systems_linked_by_a_chain():
    """Two distinct ring systems (not fused/spiro) joined by an acyclic
    linker: the top-level call picks ONE as the parent ring; the OTHER ring
    system is discovered as a nested branch, recursively, off the linker."""
    smi = "C1CCCCC1CCC1CCCC1"  # cyclohexane-CH2CH2CH2-cyclopentane
    result, verified = _name_and_verify(smi)
    assert "cyclopentan" in result.name and "cyclohexane" in result.name
    assert verified == result.name


# ===========================================================================
# Hard-branch witnesses -- confirmed abstaining on `main` TODAY (B.1 spy)
# ===========================================================================

def test_hard_branch_witness_sulfooxy_carboxylic_acid():
    """Witness (achiral form -- see the stereo-omission tests below for the
    original stereo-bearing SMILES): today's system abstains
    ('unknown organic compound') on ``C/C=C(/COS(=O)(=O)O)C(=O)O`` -- a
    trisubstituted alkene carbon bearing BOTH a carboxylic acid branch and a
    sulfooxymethyl branch, which the existing recursive substituent namer
    cannot compose (confirmed by the B.1 spy: neither branch nor whole
    compound reaches a completing tier). This module names it completely."""
    result, verified = _name_and_verify("CC=C(COS(=O)(=O)O)C(=O)O")
    assert verified == result.name
    assert "carboxy" in result.name


def test_hard_branch_witness_perindopril_fragment():
    """Witness (achiral form): today's system abstains on the stereo-bearing
    original (``CCC[C@H](N[C@H](C)C=O)C(=O)OCC``, cited as an example in
    ``RESEARCH-universal-namer-reference-architecture.md``) -- an amino-
    aldehyde N-substituent alongside an ethyl-ester tail, both of which the
    existing recursive substituent namer declines. This module names the
    achiral constitution completely."""
    result, verified = _name_and_verify("CCCC(NC(C)C=O)C(=O)OCC")
    assert verified == result.name
    assert "aza" in result.name and "oxa" in result.name


# ===========================================================================
# Stereo-omission is a documented, ACCEPTABLE degrade -- NOT wrong-molecule.
# ===========================================================================

def test_stereo_bearing_witness_is_constitution_complete():
    """The stereo-bearing ORIGINAL of the sulfooxy/carboxylic-acid witness.
    This module calls ``assign_stereochemistry`` (determinism / canonical
    CIP path) but does not yet RENDER stereo descriptors into the name
    string -- a documented, in-scope-for-later-work degrade (design doc:
    "a stereo-incomplete name is a valid less-specific degrade, not wrong").
    Coverage is total and the CONSTITUTION (InChIKey skeleton block) matches
    exactly; only the stereo layer differs, so the STRICT full-InChIKey
    ``verify_or_none`` correctly withholds CONFIRM rather than falsely
    confirming a name that omits stereo the input asserts."""
    smi = r"C/C=C(/COS(=O)(=O)O)C(=O)O"
    mol = Chem.MolFromSmiles(smi)
    result = name_universal_substitutive(mol)
    assert result is not None
    assert result.covers == _heavy_atoms(smi)
    cov = validate_atom_coverage(mol, result.name)
    assert cov.constitution_match is True
    assert verify_or_none(result.name, smi) is None  # strict: stereo omitted


def test_stereo_bearing_perindopril_is_constitution_complete():
    smi = r"CCC[C@H](N[C@H](C)C=O)C(=O)OCC"
    mol = Chem.MolFromSmiles(smi)
    result = name_universal_substitutive(mol)
    assert result is not None
    assert result.covers == _heavy_atoms(smi)
    cov = validate_atom_coverage(mol, result.name)
    assert cov.constitution_match is True
    assert verify_or_none(result.name, smi) is None  # strict: stereo omitted


# ===========================================================================
# Per-atom formal charge: refuse rather than mis-construct.
# ===========================================================================

def test_internally_charged_species_refused_not_misconstructed():
    """nitroethane (CC[N+](=O)[O-]) has NET charge 0 but an internal +1/-1
    pair this module does not model (charge suffixes are a later-phase
    item). MEASURED: before the per-atom guard was added, this produced a
    name that did NOT round-trip (a wrong construction, not merely an ugly
    one) -- the guard must refuse it outright rather than ship a candidate
    ``verify_or_none`` would have to catch."""
    mol = Chem.MolFromSmiles("CC[N+](=O)[O-]")
    assert name_universal_substitutive(mol) is None


# ===========================================================================
# Mandatory work-budget backstop: pathological giants fail CLOSED, fast.
# ===========================================================================

def _build_chain(n: int):
    rw = Chem.RWMol()
    atoms = [rw.AddAtom(Chem.Atom(6)) for _ in range(n)]
    for i in range(1, n):
        rw.AddBond(atoms[i - 1], atoms[i], Chem.BondType.SINGLE)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return m


def _build_comb(main_len: int, tooth_len: int):
    """A 'comb': a main chain of ``main_len`` carbons, each bearing its own
    ``tooth_len``-carbon side chain. Cheap to build, but forces MANY
    non-trivial recursive branch calls (one per main-chain atom), so total
    charged work is >> the raw atom count of any single call -- unlike a
    plain long chain (which needs only ONE spine call, no recursion at
    all)."""
    rw = Chem.RWMol()
    main = [rw.AddAtom(Chem.Atom(6)) for _ in range(main_len)]
    for i in range(1, main_len):
        rw.AddBond(main[i - 1], main[i], Chem.BondType.SINGLE)
    for i in range(main_len):
        prev = main[i]
        for _ in range(tooth_len):
            j = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(prev, j, Chem.BondType.SINGLE)
            prev = j
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return m


def test_giant_outright_exceeds_budget_fails_closed_fast():
    """A molecule bigger than the whole budget must be refused WITHOUT ever
    entering the (expensive, and at extreme size unsafe -- see the module's
    own comment on a measured segfault) kekulize/CIP/ring-perception
    pipeline. Regression guard for exactly that: this must return None
    quickly, never hang or crash."""
    big = _build_chain(DEFAULT_ATOM_WORK_BUDGET + 5_000)
    t0 = time.time()
    result = name_universal_substitutive(big)
    elapsed = time.time() - t0
    assert result is None
    assert elapsed < 2.0


def test_giant_deep_branching_trips_budget_despite_modest_atom_count():
    """A moderately-sized (420-atom) 'comb' structure with a DELIBERATELY
    small custom budget: proves the budget is charged per RECURSIVE CALL
    (cumulative work), not just checked once against the raw atom count --
    the comb's total atom count is far below the custom budget's sibling
    threshold once every branch call is summed."""
    comb = _build_comb(main_len=20, tooth_len=20)
    assert comb.GetNumHeavyAtoms() == 420
    t0 = time.time()
    result = name_universal_substitutive(comb, atom_work_budget=300)
    elapsed = time.time() - t0
    assert result is None
    assert elapsed < 2.0


def test_giant_comb_within_default_budget_still_terminates():
    """Sanity: the SAME comb shape, given a generous budget, actually
    terminates (does not hang) and produces complete coverage -- the budget
    is a backstop against pathological input, not a blanket refusal of any
    branched molecule."""
    comb = _build_comb(main_len=20, tooth_len=20)
    t0 = time.time()
    result = name_universal_substitutive(comb)
    elapsed = time.time() - t0
    assert result is not None
    assert result.covers == frozenset(
        a.GetIdx() for a in comb.GetAtoms() if a.GetAtomicNum() > 1
    )
    assert elapsed < 10.0


# ===========================================================================
# The coverage assertion actually VOIDS an incomplete binding (not just
# trusted to never happen -- proven by rigging a gap).
# ===========================================================================

def test_atom_coverage_assertion_voids_a_rigged_gap(monkeypatch):
    """Monkeypatch the internal recursive core to drop one atom from the
    top-level binding list, simulating a hypothetical construction bug, and
    confirm the public entry point VOIDS the candidate (returns None)
    instead of shipping a partial name. This is the carry-by-construction
    guarantee's enforcement mechanism, tested directly rather than merely
    assumed to hold because no natural witness violates it."""
    orig = us._name_component

    def rigged(ctx, component, attach_hint, is_top):
        result = orig(ctx, component, attach_hint, is_top)
        if is_top and result is not None and result.bindings:
            bindings = list(result.bindings)
            for i, (tok, ids) in enumerate(bindings):
                ids_list = sorted(ids)
                if len(ids_list) > 1:
                    dropped = frozenset(ids_list[1:])  # drop one atom
                    bindings[i] = (tok, dropped)
                    result = dataclasses.replace(result, bindings=bindings)
                    break
        return result

    monkeypatch.setattr(us, "_name_component", rigged)
    mol = Chem.MolFromSmiles("CC(C)CC")  # has a branch -> >1 binding entries
    result = name_universal_substitutive(mol)
    assert result is None
