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
  * Task B2b: charge / indicated-H as suffixes (design step 4) -- a genuine
    net-charged cation/anion, a skeletal (P-74.1.1) zwitterion, a charged
    substituent on a neutral parent, and the nitroethane-class internally
    charge-separated species (now NAMED, not voided, when spellable+
    verified) reusing ``general_engine``'s own charge-suffix primitives; an
    FG-anchored anion (carboxylate) correctly VOIDS as out of scope for
    those reused primitives, never mis-named
  * a live regression test that actually exercises the public entry point's
    broad ``except Exception`` guard (monkeypatch-injected error), since the
    original fix-round-1 giant-chain witness is now short-circuited by the
    ``_MAX_ATOMS_FOR_PERCEPTION`` cap before that code path is reached

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
# Charge: named when spellable+verified via the reused general_engine
# primitives, voided (never mis-constructed) when out of their scope.
# ===========================================================================

def test_internally_charged_species_now_named_not_voided():
    """Task B2b: nitroethane (``CC[N+](=O)[O-]``, NET charge 0, an internal
    +1/-1 nitro pair) used to be refused OUTRIGHT by a blanket per-atom-
    charge guard (before B2b: any nonzero per-atom formal charge -> void,
    added because the ORIGINAL unguarded ``nitro`` leaf shortcut, keyed only
    on "N bonded to two terminal O's" with no bond-order/charge check, fired
    on the wrong shape and mis-named a different molecule). B2b replaces the
    blanket guard with a charge-and-bond-order-VALIDATED ``_nitro_shortcut``
    (nitro's charges are P-59 INTERNAL, excluded from
    ``perception.ions.get_ion_sites``'s genuine-ion-site perception, so this
    is spelled directly as a leaf, never via the charge-suffix machinery).
    This is the the contributor guide-mandated regression check: an internally
    charge-separated net-0 species must now be NAMED when it can be spelled
    correctly and verified, never silently left void."""
    result, verified = _name_and_verify("CC[N+](=O)[O-]")
    assert verified == result.name
    assert "nitro" in result.name


# ===========================================================================
# Task B2b: charge / indicated-H as suffixes (step 4 of the design).
# ===========================================================================

def test_quaternary_ammonium_cation_gets_ium_suffix():
    """A net-charged, single-sign species: tetramethylammonium
    (``C[N+](C)(C)C``) -- the nitrogen is a skeletal 'quaternary' cation
    (``general_engine.classify_cation``), reused via
    ``general_engine._charge_suffix_text`` to append a locanted ``-ium``
    suffix on whichever spine the N ends up on."""
    result, verified = _name_and_verify("C[N+](C)(C)C")
    assert verified == result.name
    assert "ium" in result.name


def test_carboxylate_anion_voids_out_of_scope_fg_anion():
    """A net-charged anion: acetate (``CC(=O)[O-]``) classifies as
    'carboxylate' (an FG-anchored anion), which is OUT OF SCOPE for the
    reused ``general_engine._charge_suffix_text`` (that function's own
    scope only covers skeletal 'carbanion'/'heteroatom_hydride_anion'/
    'uide_anion' bases -- an FG anion is explicitly declined there, "PIN
    path owns it"). This module has no functional-group-suffix layer of its
    own to build the 'ate' form, so it VOIDS rather than mis-name it --
    never a wrong or partial name for what it cannot yet spell."""
    mol = Chem.MolFromSmiles("CC(=O)[O-]")
    assert name_universal_substitutive(mol) is None


def test_zwitterion_amino_acid_voids_carboxylate_anchored_case():
    """A second, distinct zwitterion (net-0, internally charge-separated):
    the glycine zwitterion ``C(C(=O)[O-])[NH3+]``. Its cation (aminium) IS
    in ``general_engine._zwitterion_suffix_plan``'s skeletal cation-base
    table, but its anion classifies 'carboxylate' -- in neither the P-74.1.1
    skeletal anion table NOR the P-74.1.2 '-olate' table (which is disabled
    here anyway via ``allow_fg_anion=False``, since this producer has no
    FG-suffix layer to hold the anchor atom out of substituent discovery).
    Asserts the REQUIRED invariant: NEVER a wrong constitution -- void is
    the correct, honest degrade here (a specialized carboxylate-zwitterion
    mechanism, out of scope for this general recursive namer, would be
    needed to name it)."""
    mol = Chem.MolFromSmiles("C(C(=O)[O-])[NH3+]")
    assert name_universal_substitutive(mol) is None


def test_charged_substituent_on_neutral_parent_carries_on_branch_name():
    """A charged SUBSTITUENT on an otherwise-neutral parent: cyclohexane
    forces the RING to be the top-level spine (ring always beats chain when
    any ring atom is present), so the pendant ``-N+(CH3)3`` group is
    discovered as an off-spine BRANCH and named by RE-ENTERING this same
    recursive namer on the branch subgraph -- its own local spine carries
    the genuine cation, gets its own ``-ium`` suffix (SAME
    ``_resolve_spine_charge`` machinery as the top-level case), and is then
    mechanically rewritten as a ``-yl`` substituent prefix on the neutral
    cyclohexane parent. Coverage-complete and OPSIN round-trip verified."""
    result, verified = _name_and_verify("C1CCCCC1[N+](C)(C)C")
    assert verified == result.name
    assert "ium" in result.name
    assert "cyclohexane" in result.name


# ===========================================================================
# Task B2b fix round 1: the RAW-formal-charge void guard -- the whole
# internal-charge class VOIDS (never mis-names). See
# ``.superpowers/sdd/2026-08-21-no-abstain-universal-namer/
# task-B2b-fixround1-findings.md``.
# ===========================================================================

# Every one of these carries a nonzero RAW formal charge that ``get_ion_sites``
# STRIPS as a P-59 INTERNAL / P-74.2.1 semipolar bonding feature -- invisible
# to both ``_resolve_spine_charge`` and the charge-coverage assertion. Before
# the fix, the element-symbol-only spine builders absorbed each into the
# skeleton AS IF NEUTRAL and emitted a coverage-complete name of a DIFFERENT
# molecule (all 21 measured emitting wrong on dc96929f). The raw-charge guard
# voids every one: none is spellable by this producer, so the floor degrades
# to abstain, never mis-names.
_INTERNAL_CHARGE_VOID_WITNESSES = [
    ("N-oxide (aliphatic, TMAO)",        "C[N+](C)([O-])C"),
    ("N-oxide (aromatic, pyridine)",     "[O-][n+]1ccccc1"),
    ("N-oxide (NMMO)",                   "C[N+]1([O-])CCOCC1"),
    ("N-oxide (2-methylpyridine)",       "Cc1cccc[n+]1[O-]"),
    ("azide (ethyl)",                    "CCN=[N+]=[N-]"),
    ("azide (resonance twin)",           "CC[N-][N+]#N"),
    ("azide (cyclohexyl)",               "[N-]=[N+]=NC1CCCCC1"),
    ("diazo (methane)",                  "C=[N+]=[N-]"),
    ("diazo (ethane)",                   "CC=[N+]=[N-]"),
    ("nitrone",                          "CC=[N+](C)[O-]"),
    ("nitrile oxide",                    "CC#[N+][O-]"),
    ("nitronate",                        "CC=[N+]([O-])[O-]"),
    ("aci-nitro",                        "CC=[N+]([O-])O"),
    ("nitrate ester",                    "CCO[N+](=O)[O-]"),
    ("thionitro",                        "CC[N+](=S)[O-]"),
    ("S-oxide (charge-drawn, DMSO)",     "C[S+](C)[O-]"),
    ("P-oxide (charge-drawn)",           "C[P+](C)(C)[O-]"),
    ("S-oxide (ethyl-methyl)",           "C[S+]([O-])CC"),
    ("polynitro (dinitromethane)",       "C([N+](=O)[O-])[N+](=O)[O-]"),
    ("polynitro (1,2-dinitroethane)",    "[O-][N+](=O)CC[N+](=O)[O-]"),
    ("polynitro (trinitromethane)",      "[O-][N+](=O)C([N+](=O)[O-])[N+](=O)[O-]"),
]


@pytest.mark.parametrize(
    "label,smiles", _INTERNAL_CHARGE_VOID_WITNESSES,
    ids=[w[0] for w in _INTERNAL_CHARGE_VOID_WITNESSES],
)
def test_internal_charge_classes_void_never_misname(label, smiles):
    """Fix round 1: an internal-charge class the module cannot spell must VOID
    (return None), never emit a coverage-complete name of a different molecule.
    This is the fail-closed regression suite for the whole class.

    Polynitro is included deliberately: its nitro groups get shredded (a nitro
    O can seed a spine, or be a raw-``GetNeighbors`` branch root), so they are
    NEVER rendered as a ``_nitro_shortcut`` leaf -- their O(-) raw charge is
    unconsumed and trips the guard. (A future task may NAME polynitro; for now,
    voiding is correct and sufficient -- degrade to abstain, never mis-name.)"""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert name_universal_substitutive(mol) is None, (
        f"{label} ({smiles!r}) must VOID -- an internal charge this producer "
        f"cannot spell must never be absorbed into the skeleton as if neutral"
    )


def test_np_ylide_zwitterion_names_and_verifies():
    """Fix round 1 keep-green: a P-74.1.1 skeletal zwitterion (both ionic
    centres genuine and on the same spine) is IN scope and must still emit +
    verify -- the raw-charge guard allows it because both charged atoms are
    genuine ion sites (``cation_sites | anion_sites``) the suffix machinery
    consumed, NOT stripped internal charges. Covers both the N-ylide and the
    P-ylide (trimethylammonium/phosphonium methylide)."""
    for smi in ("C[N+](C)(C)[CH2-]", "C[P+](C)(C)[CH2-]"):
        result, verified = _name_and_verify(smi)
        assert verified == result.name
        assert "ium" in result.name and "ide" in result.name


@pytest.mark.parametrize(
    "smiles", ["CCON=O", "CC=NO", "CCN=O", "CCCC(CCC)N(O)O"],
    ids=["nitrite-ester", "oxime", "nitroso", "N,N-dihydroxyheptanamine"],
)
def test_internal_charge_neutral_analogues_stay_correct(smiles):
    """Fix round 1 keep-green: the NEUTRAL analogues of the voided classes
    (drawn without formal charges) carry no raw charge, so the guard never
    fires on them -- they must still name completely and verify, exactly as
    before the fix. Guards against an over-broad guard that keys on the wrong
    signal (element/shape rather than raw formal charge)."""
    result, verified = _name_and_verify(smiles)
    assert verified == result.name


def test_charged_species_determinism_across_permutations():
    """The SAME charged molecule from differently-numbered (but structurally
    identical) SMILES must produce the IDENTICAL name -- the charge-suffix
    locant derives from the SAME canonical-rank-based spine numbering the B2
    determinism fix established, so it inherits determinism for free rather
    than needing its own separate tie-break."""
    import random

    random.seed(2026_08_21)
    witnesses = [
        "C[N+](C)(C)C",
        "CC[N+](=O)[O-]",
        "C1CCCCC1[N+](C)(C)C",
    ]
    for smi in witnesses:
        mol = Chem.MolFromSmiles(smi)
        n_atoms = mol.GetNumAtoms()
        names = set()
        for _ in range(8):
            perm = list(range(n_atoms))
            random.shuffle(perm)
            renumbered = Chem.RenumberAtoms(mol, perm)
            smi2 = Chem.MolToSmiles(renumbered, canonical=False)
            mol2 = Chem.MolFromSmiles(smi2)
            assert mol2 is not None
            result = name_universal_substitutive(mol2)
            names.add(result.name if result is not None else None)
        assert len(names) == 1, f"{smi} produced {names} across permutations"


def test_broad_except_guard_actually_fires_on_an_injected_error(monkeypatch):
    """The public entry point's broad ``except Exception: return None`` (its
    own docstring reason 5) needs a regression test that actually exercises
    it -- the fix round 1 witness (a 10,000-atom chain) is now
    short-circuited by the ``_MAX_ATOMS_FOR_PERCEPTION`` size cap BEFORE the
    code path the broad except was added for is ever reached, making that
    test vacuous for THIS guard specifically (it still correctly tests the
    size cap). This test injects a ``ValueError`` directly into the inner
    unsafe function via monkeypatch and confirms the PUBLIC wrapper still
    returns ``None`` rather than letting the exception propagate -- proving
    the broad except is live, not merely present."""
    def _boom(mol, atom_work_budget):
        raise ValueError("injected failure -- proves the broad except fires")

    monkeypatch.setattr(us, "_name_universal_substitutive_unsafe", _boom)
    mol = Chem.MolFromSmiles("CCO")
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
    small custom budget is refused, fast.

    NOTE (fix round 1, finding 6): this specific case (420 atoms > the
    budget of 300) is actually caught by the top-level SIZE guard, not the
    per-call recursive charge -- see
    ``test_cumulative_recursive_charge_trips_when_size_guard_would_not``
    below for a case that isolates the recursive-charging mechanism
    specifically (total atoms UNDER the budget, cumulative charge over it).
    Kept as a regression: either mechanism refusing this input, fast, is
    correct behaviour."""
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


def test_atom_coverage_assertion_voids_a_rigged_double_count(monkeypatch):
    """Fix round 1, finding 5: the disjointness assertion
    (``sum(len(ids)) == len(heavy)``) catches a DOUBLE-COUNT (two bindings
    sharing an atom) that a union-only check cannot see, because the union
    of an overlapping set is still the full atom set. Rig one atom into TWO
    bindings and confirm the public entry point still voids the candidate."""
    orig = us._name_component

    def rigged(ctx, component, attach_hint, is_top):
        result = orig(ctx, component, attach_hint, is_top)
        if is_top and result is not None and len(result.bindings) >= 2:
            bindings = list(result.bindings)
            tok0, ids0 = bindings[0]
            tok1, ids1 = bindings[1]
            if ids0 and ids1:
                # Duplicate one atom from binding 0 into binding 1 as well --
                # the union is UNCHANGED (still the full heavy-atom set), but
                # the total bound count now exceeds it.
                dup_atom = next(iter(ids0))
                bindings[1] = (tok1, frozenset(ids1) | {dup_atom})
                result = dataclasses.replace(result, bindings=bindings)
        return result

    monkeypatch.setattr(us, "_name_component", rigged)
    mol = Chem.MolFromSmiles("CC(C)CC")
    result = name_universal_substitutive(mol)
    assert result is None


# ===========================================================================
# Fix round 1 (task-review + FABLE adversarial review of commit cad511fd)
# ===========================================================================

def test_nitro_shortcut_removed_no_wrong_constitution():
    """Fix round 1, finding 1 (CRITICAL): the deleted ``nitro`` leaf
    shortcut checked NO bond orders, so it fired on N(OH)2 (real neutral
    nitro is refused upstream by the per-atom charge guard, and a
    pentavalent-N-with-two-double-bonds shape never sanitizes) and named it
    "nitro" -- a DIFFERENT, wrong-constitution molecule. MEASURED before the
    fix: ``CCCC(CCC)N(O)O`` (N,N-dihydroxyheptan-4-amine) emitted
    "4-nitroheptane" (denotes a different molecule, C7H15NO2 vs the actual
    C7H17NO2). The shortcut is gone; the branch now falls through to the
    generic (uglier, but not wrong) skeletal-replacement construction."""
    mol = Chem.MolFromSmiles("CCCC(CCC)N(O)O")
    result = name_universal_substitutive(mol)
    if result is not None:
        assert "nitro" not in result.name
        assert result.covers == frozenset(
            a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1
        )


def test_never_raises_on_unexpected_exception():
    """Fix round 1, finding 2: a pure ``Optional``-contracted producer must
    NEVER raise. MEASURED before the fix: a 10,000-carbon chain reached
    ``data/chain_names.py``'s "chain length outside supported range
    (1-9999)" ``ValueError`` uncaught (the entry point only caught
    ``_BudgetExceeded``). This is a plain SMILES string, not RDKit
    construction, so it also exercises the real public entry point
    end-to-end exactly as a caller would use it."""
    mol = Chem.MolFromSmiles("C" * 10_000)
    assert mol is not None
    result = name_universal_substitutive(mol)  # must not raise
    assert result is None


def test_never_segfaults_on_repeated_large_molecule_calls():
    """Fix round 1 follow-up (found verifying finding 2): a raw linear chain
    crashes the PROCESS (segfault, not a Python exception -- no try/except
    can catch it) inside CIP assignment past a few thousand atoms, and
    MEASURED worse: calling this module TWICE in one process on moderately
    large chains (2,500+ atoms) crashes on the SECOND call even though the
    first succeeds -- consistent with state accumulating in the CIP bridge
    across calls, exactly the shape a batch/eval harness uses (name many
    molecules in one process). ``_MAX_ATOMS_FOR_PERCEPTION`` is a hard,
    non-overridable ceiling BELOW the region where this was ever observed
    (stress-tested at 40 repeats). This test is the regression guard: if it
    segfaults, pytest itself dies (exit 139) rather than reporting a
    failure -- that IS the signal.
    """
    for _ in range(5):
        mol = Chem.MolFromSmiles("C" * 2_500)
        result = name_universal_substitutive(mol)
        assert result is None  # refused by the hard ceiling, not attempted


def test_large_atom_work_budget_cannot_reenable_the_crash():
    """Fix round 1: a caller passing a large ``atom_work_budget`` legitimately
    raises the cumulative RECURSIVE-work ceiling (for generous branch
    allowance) but must NOT be able to re-enable the raw-size safety
    ceiling that guards against the measured CIP segfault -- the effective
    cap is ``min(atom_work_budget, _MAX_ATOMS_FOR_PERCEPTION)``, never
    ``atom_work_budget`` alone."""
    mol = Chem.MolFromSmiles("C" * 2_500)
    result = name_universal_substitutive(mol, atom_work_budget=1_000_000)
    assert result is None


def test_monocycle_scan_is_budget_charged_not_blind():
    """Fix round 1, finding 3: monocycle numbering search scores O(n)
    candidates for O(n) rotations -- O(n^2) work that used to run AFTER the
    one up-front atom-count charge, so it was budget-blind (measured: 87.5s
    at 4,001 atoms, ~550s projected at 9,999). It must now be charged BEFORE
    the scan runs, so a tight custom budget trips during the scan rather
    than after it completes. Uses a monocycle small enough to pass the
    initial size guard but whose O(len(candidates) * n) scan cost exceeds a
    deliberately tight budget."""
    smi = "C1" + "C" * 498 + "CC1"  # 501-atom monocycle
    mol = Chem.MolFromSmiles(smi)
    assert mol.GetNumHeavyAtoms() == 501
    t0 = time.time()
    result = name_universal_substitutive(mol, atom_work_budget=520)
    elapsed = time.time() - t0
    assert result is None
    assert elapsed < 5.0  # generous: covers a cold CIP-bridge warm-up


def test_monocycle_still_names_within_the_hard_size_ceiling():
    """Sanity companion to the above: an ordinary, small monocycle is
    unaffected by the budget-charging fix."""
    mol = Chem.MolFromSmiles("C1CCCCC1")
    result = name_universal_substitutive(mol)
    assert result is not None
    assert result.name == "cyclohexane"


def test_cumulative_recursive_charge_trips_when_size_guard_would_not():
    """Fix round 1, finding 6 (test gap): the EXISTING giant/comb test
    (``test_giant_deep_branching_trips_budget_despite_modest_atom_count``)
    trips the top-level SIZE guard (420 atoms > budget 300), never
    exercising the per-call RECURSIVE charging mechanism at all. This test
    isolates that mechanism: total heavy-atom count is UNDER the custom
    budget (240 <= 300, so the size guard passes), but the CUMULATIVE charge
    across the top-level spine plus its 15 branch calls (15 + 2*15*15 = 465)
    exceeds the same budget -- the trip must come from recursion, not size."""
    comb = _build_comb(main_len=15, tooth_len=15)
    assert comb.GetNumHeavyAtoms() == 240
    result = name_universal_substitutive(comb, atom_work_budget=300)
    assert result is None


def test_determinism_across_equivalent_smiles_permutations():
    """Fix round 1, finding 4: the SAME molecule from differently-numbered
    (but structurally identical) SMILES must produce the IDENTICAL name --
    every tie-break in this module must key on ``Chem.CanonicalRankAtoms``
    (numbering-invariant), never a raw RDKit atom index. MEASURED before the
    fix: ``ClC1CCCCC1`` emitted "5-chlorocyclohexane" from one atom-numbering
    and "1-chlorocyclohexane" from another -- both individually RT-valid,
    so a downstream OPSIN-RT gate could not have caught the
    nondeterminism. Also confirms LOWEST-locant numbering as a side effect
    (chlorocyclohexane must always emit "1-chlorocyclohexane", never
    "5-...")."""
    import random

    random.seed(1234567)
    witnesses = [
        "ClC1CCCCC1",
        "ClC1CCC(Br)CC1",
        "O=C(O)C1CCCC1CC(C)C",
        "C1CCC(C1)C1CCOC1",
    ]
    for smi in witnesses:
        mol = Chem.MolFromSmiles(smi)
        n_atoms = mol.GetNumAtoms()
        names = set()
        for _ in range(8):
            perm = list(range(n_atoms))
            random.shuffle(perm)
            renumbered = Chem.RenumberAtoms(mol, perm)
            smi2 = Chem.MolToSmiles(renumbered, canonical=False)
            mol2 = Chem.MolFromSmiles(smi2)
            assert mol2 is not None
            result = name_universal_substitutive(mol2)
            names.add(result.name if result is not None else None)
        assert len(names) == 1, f"{smi} produced {names} across permutations"

    # The specific lowest-locant regression named in the finding:
    mol = Chem.MolFromSmiles("ClC1CCCCC1")
    result = name_universal_substitutive(mol)
    assert result.name == "1-chlorocyclohexane"


def test_single_heavy_atom_top_level_input_no_raise():
    """Fix round 1, finding 7: a bare single-heavy-atom molecule at the top
    level (water / ammonia / hydrogen sulfide) skips the leaf-shortcut table
    (branches only) and falls into the generic chain-spine path as a
    length-1 chain. Assert it either produces a coverage-complete result or
    voids (None) -- never raises."""
    for smi in ("O", "N", "S"):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        result = name_universal_substitutive(mol)  # must not raise
        if result is not None:
            heavy = frozenset(
                a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1
            )
            assert result.covers == heavy
