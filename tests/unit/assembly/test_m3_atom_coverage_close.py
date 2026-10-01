"""M3 Task 1: populate ``NameFragment.atoms`` on the suffix / cyclic-parent /
FG-prefix producers so the existing but DEAD general-acyclic atom-coverage
close (``_w2_atom_coverage_declines``, ``assembly/handlers/_handler_shared.py``)
goes live.

Root cause (measured, `internal notes`): the close
already runs unconditionally (``handlers/general_acyclic.py``), but it
breadth-safe-skips (never verifies) whenever ANY covered ``parent``/``suffix``/
``prefix`` fragment reports ``atoms=None``. Suffix fragments reported ``None``
ALWAYS, cyclic parents reported ``None`` always, and the halogen/hydroxy/amino/
... FG-prefix loop reported ``None`` always -- so for virtually every real
functionalized molecule the close never fired, and the ONLY backstop was the
OPSIN self-consistency round-trip (jar-present only).

This file pins:
  * the ``COS(=O)(=O)O`` -> ``methane`` canonical witness (already fixed by the
    chain-parent's pre-existing atoms population -- a regression anchor);
  * a NEW witness family (``sulfooxy``-substituted acids/alcohols/amines) that
    specifically requires the suffix-atoms fix landed here: a carboxylic-acid/
    alcohol/amine suffix coexists with a carbon-free sulfate-ester branch that
    ``_generate_alkyl_prefixes`` cannot express (``substituent_is_bare_
    functional_group``/``universal_pipeline_unnameable``), so the candidate
    silently ships only the acid/alcohol/amine fragment. Before this fix the
    suffix's ``atoms=None`` made the close skip; jar-absent, nothing caught it.

All witnesses are run with the OPSIN jar forced ABSENT (the ``_force_jar_absent``
pattern from ``tests/unit/validation/test_w2_jarabsent_coverage.py``) so the
terminal self-consistency backstop cannot mask a live/dead close -- this test
exercises ONLY the producer-side, OPSIN-free coverage assertion.
"""
import json
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.namer import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse


def _force_jar_absent(monkeypatch):
    """Simulate the no-Java config (mirrors test_w2_jarabsent_coverage.py)."""
    import orthonym.namer as namer_mod
    import orthonym.validation.atom_coverage as ac_mod
    import orthonym.validation.opsin_roundtrip as rt_mod
    monkeypatch.setattr(namer_mod, "_validity_gate_jar_present", lambda: False)
    monkeypatch.setattr(ac_mod, "find_opsin_jar", lambda: None)
    monkeypatch.setattr(rt_mod, "_find_opsin_jar", lambda *a, **k: None)


# Methyl hydrogen sulfate: the documented NameFragment.atoms docstring witness.
# Already fixed by the PRE-EXISTING chain-parent atoms population (parent alone
# accounts for {C}, the sulfate {O,S,O,O,O} is unbound -> decline). Pinned here
# as a regression anchor for the whole M3 Task 1 producer family.
METHYL_SULFATE = "COS(=O)(=O)O"

# NEW witnesses: a carboxylic-acid / alcohol / amine suffix + a dropped
# carbon-free sulfooxy branch. Each currently (pre-fix) ships the fragment
# name below jar-absent; after populating suffix.atoms the close declines it.
_SUFFIX_WITNESS_WRONG = {
    "CC(C(=O)O)OS(=O)(=O)O": "propanoic acid",
    "OC(=O)C(C)OS(=O)(=O)O": "propanoic acid",
    "OC(=O)CCOS(=O)(=O)O": "propanoic acid",
    "OC(=O)COS(=O)(=O)O": "ethanoic acid",
    "CC(N)OS(=O)(=O)O": "ethan-1-amine",
    "OCC(=O)OS(=O)(=O)O": "ethan-1-ol",
}

# A halogen-FG-prefix witness (2-chloropropane): dropping the -OS(=O)(=O)O
# branch off a chain whose ONLY decoration is a "chloro" FG prefix (no suffix
# at all) -- requires the FG-prefix-loop atoms fix (not the suffix fix).
_PREFIX_WITNESS_WRONG = {
    "CC(Cl)C(=O)OS(=O)(=O)O": "2-chloropropane",
}

# Ordinary correct names that MUST keep emitting -- the fix must void ONLY
# genuine drops, never a complete/correct name (the mandatory "0 complete
# names voided" bar). Includes chain suffix+prefix and bare-ring+suffix cases
# newly brought under the close by this task.
_GUARD = {
    "CC(=O)O": "acetic acid",
    "CC(C)(C)CC(=O)O": "3,3-dimethylbutanoic acid",
    "ClCC(=O)O": "chloroacetic acid",
    "CC(Cl)C(=O)O": "2-chloropropanoic acid",
    "OCC(=O)O": "hydroxyacetic acid",
    "NCC(=O)O": "glycine",
    "O=C1CCCCC1": "cyclohexanone",
    "OC1CCCCC1": "cyclohexanol",
    "OC1CCCC1": "cyclopentanol",
}


@pytest.fixture
def eng():
    return Orthonym(style="pin")


@pytest.mark.parametrize("smiles", [METHYL_SULFATE, *list(_SUFFIX_WITNESS_WRONG),
                                     *list(_PREFIX_WITNESS_WRONG)])
def test_atom_drop_witness_abstains_jar_absent(monkeypatch, smiles):
    """Jar-absent: the witness must NOT ship its atom-dropped fragment name."""
    _force_jar_absent(monkeypatch)
    wrong = {**_SUFFIX_WITNESS_WRONG, **_PREFIX_WITNESS_WRONG}.get(
        smiles, "methane")
    name = Orthonym(style="pin").name(smiles)
    assert name != wrong, (
        f"{smiles} still ships the atom-dropped wrong name {name!r}")


@pytest.mark.parametrize("smiles", [METHYL_SULFATE, *list(_SUFFIX_WITNESS_WRONG),
                                     *list(_PREFIX_WITNESS_WRONG)])
def test_atom_drop_witness_abstains_cleanly_jar_absent(monkeypatch, smiles):
    """Strengthening: the surviving result is an honest abstention (sentinel),
    never a DIFFERENT atom-incomplete name silently swapped in."""
    _force_jar_absent(monkeypatch)
    name = Orthonym(style="pin").name(smiles)
    assert is_failure_name(name), (
        f"{smiles} jar-absent produced {name!r}, expected an abstention "
        f"sentinel (no whole-graph alternative exists for these witnesses)")


@pytest.mark.parametrize("smiles,expected", list(_GUARD.items()))
def test_by_construction_guard_still_emits_jar_present(eng, smiles, expected):
    """The fix voids ONLY atom-drops: ordinary correct names still emit
    (jar-present, the normal operating config)."""
    assert eng.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", list(_GUARD.items()))
def test_by_construction_guard_still_emits_jar_absent(monkeypatch, smiles, expected):
    """Same guard with the jar forced absent -- the newly-live coverage close
    must not void any of these complete, correct names (the mandatory
    '0 complete names voided' bar for this task)."""
    _force_jar_absent(monkeypatch)
    assert Orthonym(style="pin").name(smiles) == expected


# ===========================================================================
# M3 Task 2: benzene whole-graph atom-coverage close
# ===========================================================================
#
# Mirrors the heterocycle close (composer.py ``_assemble_heterocycle_name``,
# "task-W2 Witness-A") for the benzene assembly return
# (``composer.py::_assemble_benzene_name``, just before its
# ``name_substituted_benzene(...)`` return). Verified BEFORE implementing
# (per the task's highest-risk-direction warning) that
# ``_ring_handler_parent_atom_indices(features, 'benzene')`` reports the
# COMPLETE ring-union-substituent atom set for 20 distinct substituted-benzene
# structures spanning halogens, nitro, sulfonic acid/sulfonamide, amides
# (incl. N-methyl), esters via the acid suffix, carboxylic acids, amines,
# alkenes, haloalkyls and ethers -- 0 under-counts -- before wiring the
# unconditional decline. Those same molecules are pinned below as the
# regression guard.
#
# Real-witness search (per the task: "PREFER a real witness"): sampled 120
# real abstaining SMILES across `internal notes
# pubchem_seed5,zinc_seed17,zinc_seed42,pubchem_seed42}.jsonl`` with the
# census best-effort engine (``general_fallback=True,
# general_fallback_unverified=True, allow_aromatic_general=True``); only 4
# molecules picked ``handler='benzene'`` and every one of those already
# reported FULL coverage (no live drop). ONE genuine handler='benzene'
# atom-drop witness WAS reproduced from the family named in
# ``M3-ATOMDROP-a trace.md`` (a chloroimino-guanidine N-substituted benzamide):
# jar-absent, default (pin) tier ships ``3-(trifluoromethyl)benzamide`` for a
# 17-heavy-atom input (13 named, 4 dropped -- the ``-NH-C(=NCl)-NH2`` arm on
# the amide nitrogen). That witness does **NOT** trip this close: the
# carboxamide substituent's own ``atoms`` field (built by a full BFS walk
# BEFORE ``_detect_n_substituents`` decides whether it can render the
# N-substituent, ``rules/benzene.py:987-1003``) already claims all 4 dropped
# atoms as "covered", so ``_ring_handler_parent_atom_indices`` over-reports
# and the partition looks complete. That is a SEPARATE, narrower defect
# (``_detect_n_substituents`` silently returns an empty ``alkyl_names`` list
# for a non-pure-alkyl amide N-substituent instead of signalling failure the
# way its sulfonamide twin ``_detect_sulfonamide_n_substituents`` does,
# returning ``None``, at ``rules/benzene.py:1009-1054``) -- out of THIS
# task's scope (adding the close, not auditing every substituent-atoms
# producer for over-claiming). Recorded here, pinned by
# ``test_known_gap_amide_n_substituent_overclaims_atoms`` below, so a future
# fix to that producer is verified against this exact reproduction.
#
# The close is still shipped as breadth-safe hardening exactly as the M3 plan
# permits when no witness of the EXACT catchable shape is found naturally: it
# is correctness-neutral on every sampled molecule (0 false-voids) and
# demonstrably LIVE (the wiring test below proves it declines whenever the
# coverage data it is fed is genuinely incomplete).

BENZENE_OVERCLAIM_WITNESS = "NC(=NCl)NC(=O)c1cccc(C(F)(F)F)c1"  # 17 heavy atoms
BENZENE_OVERCLAIM_WRONG = "3-(trifluoromethyl)benzamide"  # drops 4 atoms

# Ordinary, correct substituted-benzene names that MUST keep emitting -- the
# EMPIRICALLY-VERIFIED-FULL-COVERAGE set from the pre-implementation trace
# (0/20 under-counts across this exact family).
_BENZENE_GUARD = {
    "Cc1cccc(C)c1": "1,3-dimethylbenzene",
    "NC(=O)c1ccc(Cl)cc1": "4-chlorobenzamide",
    "CNC(=O)c1ccc(Cl)cc1": "4-chloro-N-methylbenzamide",
    "OC(=O)c1cc(Cl)cc(Cl)c1": "3,5-dichlorobenzoic acid",
    "O=[N+]([O-])c1ccccc1": "nitrobenzene",
    "Cc1ccc([N+](=O)[O-])cc1": "1-methyl-4-nitrobenzene",
    "OC(=O)c1cc(Cl)ccc1N": "2-amino-5-chlorobenzoic acid",
    "CNc1ccccc1": "N-methylaniline",
    "C=Cc1ccccc1": "ethenylbenzene",
    "ClCc1ccccc1": "(chloromethyl)benzene",
    "Cc1ccc(cc1)S(=O)(=O)O": "4-methylbenzene-1-sulfonic acid",
    "Cc1ccc(cc1)S(=O)(=O)N": "4-methylbenzene-1-sulfonamide",
    "Clc1ccc(Cl)c(Cl)c1": "1,2,4-trichlorobenzene",
    "Brc1cccc([N+](=O)[O-])c1": "1-bromo-3-nitrobenzene",
    "Oc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]": "2,4-dinitrophenol",
    "CCOc1ccc(cc1)C=O": "4-ethoxybenzaldehyde",
}


@pytest.mark.parametrize("smiles,expected", list(_BENZENE_GUARD.items()))
def test_benzene_guard_still_emits_jar_present(eng, smiles, expected):
    """The M3 Task 2 close voids ONLY atom-drops: ordinary correct
    substituted-benzene names still emit (jar-present, normal operation)."""
    assert eng.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", list(_BENZENE_GUARD.items()))
def test_benzene_guard_still_emits_jar_absent(monkeypatch, smiles, expected):
    """Same guard with the jar forced absent -- the unconditional benzene
    coverage close must not void any of these complete, correct names (the
    mandatory '0 complete names voided' bar for this task)."""
    _force_jar_absent(monkeypatch)
    assert Orthonym(style="pin").name(smiles) == expected


def test_benzene_close_declines_on_incomplete_coverage(monkeypatch):
    """WIRING proof: the close is LIVE and correctly plumbed end-to-end.

    Forces ``_ring_handler_parent_atom_indices`` to under-report by exactly
    one atom for an otherwise fully-covered, correctly-named benzene molecule
    (``4-methylbenzenesulfonamide``, verified FULL 11/11 above) and confirms
    ``_assemble_benzene_name`` declines (falls through to the sentinel)
    rather than shipping the now-artificially-incomplete-covered name. This
    is independent of whether any producer manufactures such an incomplete
    set today (see the module note above on the one located real-world
    witness, which the close does NOT catch for an unrelated reason) --
    it proves the decline branch itself fires correctly when fed
    genuinely-incomplete coverage, jar-absent AND jar-present (the close is
    gated unconditionally, ahead of any OPSIN round-trip)."""
    from orthonym.assembly import composer as C

    smiles = "Cc1ccc(cc1)S(=O)(=O)N"  # 4-methylbenzenesulfonamide
    orig = C._ring_handler_parent_atom_indices

    def _truncate_one_atom(features, handler):
        result = orig(features, handler)
        if handler == "benzene" and result is not None:
            _atoms = set(result)
            _atoms.pop()
            return frozenset(_atoms)
        return result

    monkeypatch.setattr(C, "_ring_handler_parent_atom_indices", _truncate_one_atom)
    name = Orthonym(style="pin").name(smiles)
    assert name != "4-methylbenzene-1-sulfonamide", (
        "M3 Task 2 close did not decline an artificially-incomplete benzene "
        "coverage set -- the wiring is not live")


def test_benzene_close_declines_on_incomplete_coverage_jar_absent(monkeypatch):
    """Same wiring proof, jar forced absent -- isolates the OPSIN-free close
    from the terminal self-consistency backstop (proves THIS close, not
    , is what declines)."""
    from orthonym.assembly import composer as C

    _force_jar_absent(monkeypatch)
    smiles = "Cc1ccc(cc1)S(=O)(=O)N"
    orig = C._ring_handler_parent_atom_indices

    def _truncate_one_atom(features, handler):
        result = orig(features, handler)
        if handler == "benzene" and result is not None:
            _atoms = set(result)
            _atoms.pop()
            return frozenset(_atoms)
        return result

    monkeypatch.setattr(C, "_ring_handler_parent_atom_indices", _truncate_one_atom)
    name = Orthonym(style="pin").name(smiles)
    assert name != "4-methylbenzene-1-sulfonamide", (
        "M3 Task 2 close did not decline jar-absent with an artificially "
        "incomplete benzene coverage set")


def test_witness_still_ships_wrong_name_documented_gap(monkeypatch):
    """Pins the documented KNOWN GAP (module note above): a real handler=
    'benzene' atom-drop reproduction that this close does NOT catch, because
    the dropped atoms are already (wrongly) claimed by the carboxamide
    substituent's own ``atoms`` field before ``_detect_n_substituents`` even
    runs (``rules/benzene.py:987-1003`` silently empties ``alkyl_names`` for
    a non-pure-alkyl amide N-substituent instead of signalling failure).

    This is intentionally a PINNING test, not a regression target for THIS
    task: if a future fix to ``_detect_n_substituents`` (or the substituent
    ``atoms`` accounting around it) starts reporting the drop honestly, this
    close will then correctly void the name and this assertion will start
    failing -- that failure is the signal to update this test, not a
    regression in the M3 Task 2 close itself."""
    _force_jar_absent(monkeypatch)
    name = Orthonym(style="pin").name(BENZENE_OVERCLAIM_WITNESS)
    assert name == BENZENE_OVERCLAIM_WRONG, (
        f"expected the documented known-gap wrong name {BENZENE_OVERCLAIM_WRONG!r}, "
        f"got {name!r} -- if the underlying _detect_n_substituents defect was "
        f"fixed, update/remove this pin")


# ===========================================================================
# M3 Task 3: audit of the heterocycle whole-graph atom-coverage close
# (composer.py ``_assemble_heterocycle_name``, ``:5117-5140``)
# ===========================================================================
#
# MEASUREMENT (audit, a project rule/10 -- trace before you refute): traced every
# live call to the heterocycle close's own ``verify_atom_coverage``
# invocation across real heterocycle-handler atom-drop witnesses sampled from
# `internal notes`'s classified self-consistency-
# rejected set (post Task 1 + Task 2, HEAD ``). Across 6 real
# executions the close's own union vs ``features.mol`` heavy-atom count was
# EXACT every time: complete coverage -> ``ok=True`` with 0 unaccounted atoms
# (2 witnesses shipped genuinely complete, correct names this way); genuine
# drops -> ``ok=False`` with the correct unaccounted indices (the historical
# captopril exemplar pinned below, plus a second real witness with multiple
# genuine declines). NO false pass of the shape Task 3 Step 1 describes ("a
# dropped substituent's atoms wrongly included in substituents.values")
# reproduced on this population -- the ordinary chain/amide-substituent
# family is sound.
#
# A DIFFERENT false-pass WAS found and reproduced by direct construction,
# per the task brief's instruction to check the heterocycle path for the SAME
# over-claim shape as the benzene known gap above: the GENERIC carbon-
# anchored fallback in ``name_substituted_heterocycle`` (``rules/
# heterocycles.py``, the ``attach_is_carbon`` branch feeding
# ``name_substituent_fragment``) has no fail-closed guard of its own. For an
# unusual multi-heteroatom branch it does not correctly perceive (an
# N-acylchloroformamidine), it returns a WRONG but non-``None`` name
# (``'carbamoylmethyl'``, 4 heavy atoms) for the real 7-heavy-atom branch.
# ``_cov_groups`` (composer.py:5160-5166) trusts ``sub_info['atoms']`` (the
# raw, structurally-complete BFS set, computed BEFORE this rendering
# decision) rather than what the chosen name actually spells, so the close
# reports full coverage (12/12) and the wrong molecule ships jar-absent, PIN
# tier.
#
# ATTEMPTED FIX (tried, measured unsafe, REVERTED -- not shipped): reuse the
# EXISTING ``_n_anchored_substituent_covers`` OPSIN-based verifier (already
# applied to the sibling N/O/S-anchored branch, ``rules/heterocycles.py``
# #29) on this carbon-anchored branch too. This DOES correctly decline
# the witness above -- but ``OpsinOracle.name_to_smiles`` (the primitive
# ``_n_anchored_substituent_covers`` calls through
# ``_validity_gate_name_to_smiles``) returns ``None`` UNCONDITIONALLY
# whenever the OPSIN jar is absent (``retained_substitution.py:267``), so
# extending the check to this branch makes EVERY heteroatom-bearing carbon-
# anchored substituent -- not just the pathological one -- decline
# jar-absent, correct or not. MEASURED (HEAD-A/B via ``scripts/an A/B check``):
# 2 real, complete, correct heterocycle names voided jar-absent in an 8-row
# guard sample (``2-(methoxymethyl)thiophene``, ``4-(chloromethyl)pyridine``),
# violating the M3 "0 complete names voided" bar. No net benefit either:
# jar-PRESENT the terminal self-consistency  round-trip already
# catches this exact witness without any change, so the fix only ever helps
# jar-absent -- exactly the one mode it breaks. Reverted; not shipped.
#
# Disposition: same as the benzene sibling above -- pinned as a KNOWN GAP,
# not fixed here. The real remedy is in ``assembly/substituent_naming.py::
# name_substituent_fragment``'s functional-group perception for this class
# (make IT fail closed -- return ``None`` -- instead of guessing when it
# cannot fully perceive a branch), not in ``_cov_groups`` construction, and
# out of THIS task's scope.

CAPTOPRIL_DROP_WITNESS = "CC(CS)C(=O)N1CCCC1C(=O)O"  # historical exemplar

HETEROCYCLE_CARBON_ANCHOR_OVERCLAIM_WITNESS = "NC(=NCl)NC(=O)c1cccs1"
# The spelling of the wrong carbamoyl-on-chain prefix follows method
# (1) (the Blue Book) since the PIN class program Task 2; the molecule it
# names (the documented gap) is unchanged.
HETEROCYCLE_CARBON_ANCHOR_OVERCLAIM_WRONG = "2-(2-amino-2-oxoethyl)thiophene"

# Ordinary carbon-anchored heteroatom-bearing heterocycle substituents that
# MUST keep emitting jar-absent -- the exact population the reverted
# ``_n_anchored_substituent_covers``-style fix would have voided (measured
# above).
_HETEROCYCLE_CARBON_ANCHOR_GUARD = {
    "COCc1cccs1": "2-(methoxymethyl)thiophene",
    "ClCc1ccncc1": "4-(chloromethyl)pyridine",
}


def test_heterocycle_close_declines_genuine_drop_jar_absent(monkeypatch):
    """Regression anchor: the historical captopril exemplar never ships an
    atom-dropped name jar-absent. It used to be declined by the heterocycle close
    (composer.py ``:5117-5140``) because ``get_heterocycle_substituents`` skipped
    the ring nitrogen's acyl substituent outright (``rules/heterocycles.py``
    task 9). Since decision A part 2 (2026-09-27) that collector names the acyl
    as an acyl PREFIX when the ring carries the carboxylic-acid suffix
    (``_ring_n_acyl_prefix_under_ring_acid``), so the heterocycle candidate is
    complete: ``1-(2-methyl-3-sulfanylpropanoyl)pyrrolidine-2-carboxylic acid``."""
    _force_jar_absent(monkeypatch)
    name = Orthonym(style="pin").name(CAPTOPRIL_DROP_WITNESS)
    # Was ``== "N-2-methyl-3-sulfanylpropanoylproline"``, then (decision A part 1)
    # 'N-(2-methyl-3-sulfanylpropanoyl)proline'; since decision A part 2 the name is
    # '1-(2-methyl-3-sulfanylpropanoyl)pyrrolidine-2-carboxylic acid' (the acyl
    # prefix under the ring acid, the Blue Book). The spelling is not
    # pinned here (tests/unit/rules/test_decision_a_n_substituted_amino_acids.py pins
    # it, jar-present). What this anchor guards is completeness:
    # the witness's sole sulfur sits in the acyl chain the historical drop lost, so a
    # complete name carries 'sulfanyl' and the proline/pyrrolidine core
    # (test_w2_jarabsent_coverage's drop-witness check).
    assert not is_failure_name(name), name
    assert "sulfanyl" in name and ("proline" in name or "pyrrolidine" in name), (
        f"expected the complete acyl-proline name, got the atom-dropped "
        f"fragment (or something else) instead: {name!r}")
    assert name != "pyrrolidine-2-carboxylic acid"


def test_heterocycle_carbon_anchor_overclaim_documented_gap(monkeypatch):
    """Pins the KNOWN GAP found while auditing the heterocycle close (module
    note above): a rare multi-heteroatom carbon-anchored branch that
    ``name_substituent_fragment`` cannot correctly perceive still ships
    jar-absent, because the close's ``_cov_groups`` trusts the substituent's
    raw, structurally-complete atom set rather than what the chosen name
    actually spells. Intentionally a PINNING test, not a regression target
    for Task 3: the safe fix is in ``name_substituent_fragment``'s FG
    perception (or a fail-closed guard on ITS output), not in
    ``_cov_groups`` construction -- see the module note for why the obvious
    ``_cov_groups``-side fix (reusing ``_n_anchored_substituent_covers``) was
    tried and reverted."""
    _force_jar_absent(monkeypatch)
    name = Orthonym(style="pin").name(HETEROCYCLE_CARBON_ANCHOR_OVERCLAIM_WITNESS)
    assert name == HETEROCYCLE_CARBON_ANCHOR_OVERCLAIM_WRONG, (
        f"expected the documented known-gap wrong name "
        f"{HETEROCYCLE_CARBON_ANCHOR_OVERCLAIM_WRONG!r}, got {name!r} -- if "
        f"the underlying name_substituent_fragment defect was fixed, "
        f"update/remove this pin")


@pytest.mark.parametrize("smiles,expected",
                          list(_HETEROCYCLE_CARBON_ANCHOR_GUARD.items()))
def test_heterocycle_carbon_anchor_guard_jar_absent(monkeypatch, smiles, expected):
    """The reverted fix (module note above) would have voided these -- lock
    them in as correct, complete names jar-absent so a future attempt at the
    same fix shape is caught by this exact guard before it ships."""
    _force_jar_absent(monkeypatch)
    assert Orthonym(style="pin").name(smiles) == expected


# ===========================================================================
# M3 Task 4: whole-milestone atom-coverage regression TRAP
# ===========================================================================
#
# THE INVARIANT (general, across all three M3 producer closes above): a
# default-tier (``Orthonym`` == ``Orthonym(style="pin")``, jar-present)
# emitted name may never OPSIN-round-trip to FEWER heavy atoms than its
# input -- "no atom-drop name ships". Task 1 (chain/suffix), Task 2
# (benzene) and Task 3 (heterocycle, audited-not-fixed) each close one
# producer; this trap is the coarse net that would catch a REGRESSION of
# any of them without having to name which producer broke. A name may
# legitimately round-trip to MORE heavy atoms than the input (a
# wrong-direction/over-general name is a different defect class); only
# FEWER is an atom-drop.
#
# Sample: the first 35 rows of the hard-case witness corpus M3 was built
# and measured against (`internal notes`,
# 166 rows total, every row recorded there as a historical abstainer/sentinel
# ship) -- capped well under the full file per the anti-death bound (<=40
# rows, foreground, ~2 minutes). Rows that abstain (``is_failure_name``) or
# that OPSIN cannot parse back are skipped: neither is an atom-drop SHIP, so
# neither belongs in this trap's population.

# The original sample (internal notes) was never
# tracked, so collection of this whole module failed on any fresh checkout. Since
# 2026-09-03 the sample is vendored under tests/data/ (35 PubChem-10k molecules
# the default tier does not name as PIN/systematic; lines starting with "_doc"
# are comments). If the file is ever missing the trap skips loudly instead of
# taking every other test in this module down with it.
_M3_TRAP_SAMPLE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "m3_trap_sample_pubchem10k.jsonl"
)


def _load_m3_trap_sample(n=35):
    rows = []
    try:
        with open(_M3_TRAP_SAMPLE_PATH) as fh:
            for line in fh:
                rec = json.loads(line)
                if "smiles" not in rec:
                    continue  # header/comment row
                rows.append(rec["smiles"])
                if len(rows) >= n:
                    break
    except FileNotFoundError:
        return [pytest.param("MISSING", marks=pytest.mark.skip(
            reason=f"trap sample missing: {_M3_TRAP_SAMPLE_PATH}"))]
    return rows


_M3_TRAP_SAMPLE = _load_m3_trap_sample(35)

# Known blind spots (documented KNOWN GAPs above, Task 2/3 module notes):
# reproductions where a dropped branch's atoms are already (wrongly) claimed
# by an upstream substituent's own ``atoms`` field, so the coverage close's
# partition looks complete and the atom-dropped name ships anyway. xfail
# ONLY the specific witnesses proven to hit that gap -- never broaden this to
# mute a genuinely new atom-drop.
_M3_TRAP_XFAIL = {
    # smiles: reason
}


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", _M3_TRAP_SAMPLE)
def test_no_atom_drop_ships_on_sample(eng, smiles):
    """M3 whole-milestone trap: across the chain+benzene+heterocycle
    producers, no default-tier emitted name may OPSIN-parse to fewer heavy
    atoms than the input (an atom-drop ship). See module note above.

    ``opsin_gate`` (jar-present required): the OPSIN self-consistency gate
    is OFF by suite default (``tests/conftest.py``) but ON in real default-
    tier production, and it is precisely what suppresses most atom-drop
    candidates before they ship (measured: with the gate off, 10/35 of this
    sample's rows ship an atom-dropped fragment name via the general
    fallback/fused-ring path; with the gate correctly live -- as the CLI
    always runs -- every one of those 10 abstains instead). Testing the
    gate-off config here would trap a config that is not the real invariant.
    """
    if smiles in _M3_TRAP_XFAIL:
        pytest.xfail(_M3_TRAP_XFAIL[smiles])

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparseable input SMILES (bad sample row): {smiles}"
    input_heavy = mol.GetNumHeavyAtoms()

    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="test-m3-atomdrop-trap"):
        name = eng.name(smiles)
        if not name or is_failure_name(name):
            pytest.skip("abstained -- not an atom-drop ship")

        opsin_smi = opsin_parse(name)
    if not opsin_smi:
        pytest.skip("OPSIN could not parse the emitted name -- not an atom-drop ship")
    opsin_mol = Chem.MolFromSmiles(opsin_smi)
    if opsin_mol is None:
        pytest.skip("OPSIN output not RDKit-parseable -- not an atom-drop ship")
    output_heavy = opsin_mol.GetNumHeavyAtoms()

    assert output_heavy >= input_heavy, (
        f"ATOM-DROP SHIP: {smiles!r} ({input_heavy} heavy atoms) -> {name!r} "
        f"-> OPSIN {opsin_smi!r} ({output_heavy} heavy atoms)")
