"""M2 Task 1 — tier propagation for ``name_substituent`` (best-effort scope only).

Measured (internal notes,
`.superpowers/sdd/M1-PLAN/m2-task-1-report.md`): 75/166 pubchem10k rows hit the
substituent cascade's ``substituent``/``unknown`` placeholder internally under the
best-effort tier. Of the 71 unique declined root fragments, 37 (mapping to 38
molecule rows) name ONLY at best-effort (never at PIN default) when tested
standalone -- the "tier-propagation" bucket. ``name_substituent``
(``substituent_enumerator.py:873``) is called by ~25 sites with the default
``allow_mancude=False``, so even inside a best-effort whole-molecule run the
cascade was still gated at the DEFAULT tier and declined those fragments.

An empirical fresh-process a trace over all 38 candidate rows (before vs. after the
fix, via ``scripts/an A/B check``) found the TRUE conversion rate at the
best-effort tier (``general_fallback=True, general_fallback_unverified=True,
allow_aromatic_general=True``) -- i.e. rows where the FINAL emitted name flips
from the abstention sentinel to a real, OPSIN-round-trip-verified name -- is
7/38: pubchem10k rows 23, 28, 31, 121, 126, 132, 141. The other 31 either
already named correctly via a parallel candidate path (3 rows: 20, 104, 154 --
not real abstentions) or still abstain and need Task 2's routing-to-the-general-
engine lever (28 rows). All 7 witnesses below are drawn from that measured set,
not invented.

⚠ Every test below is marked ``opsin_gate``. ``tests/conftest.py`` disables the
OPSIN self-consistency/validity gate  suite-wide by DEFAULT so most
tests can assert raw generator output cheaply -- but that makes "PIN default
still abstains" and "best-effort now names it" claims meaningless without the
gate, because gate-OFF ships whatever a recovery-lane candidate happens to
build, unguarded. Measured directly: WITHOUT the marker,
``test_default_tier_unchanged`` spuriously FAILED for 3/6 witnesses (a
gate-disabled recovery-lane artifact of the test harness, not a real
default-tier widening -- confirmed absent via ``scripts/an A/B check`` against a
bare, unpatched ``orthonym`` import and reproduced/explained with
``tests/conftest.py``'s own ``_opsin_validity_gate_state`` fixture). WITH the
marker (gate ON, matching production), all 6 pass byte-identically.
"""
import pytest

from orthonym import Orthonym
from orthonym.errors import is_refusal_sentinel

# The 7 measured TRUE conversions (best-effort: abstain -> real OPSIN-valid name).
# pubchem10k.jsonl row indices kept in the comment for traceability back to the a trace.
TIER_PROPAGATION_WITNESSES = [
    "CCC[C@H](N)C(=O)N(C)[C@H]1CC[C@@H]2CN(Cc3ccc(C(F)(F)F)cc3)C[C@@H]21",  # row 23
    "CCC[C@@H](C)NC(=O)C[C@H]1Sc2ccc(C(F)(F)F)cc2NC1=O",                    # row 28
    "O=C(c1c2ccccc2cc2ccccc12)N1CCN(c2ccc(C(F)(F)F)cn2)CC1",                # row 31
    "O=C(Nc1ccc(Cl)c(C(=O)N(Cl)/N=C/CC(F)(F)F)c1)C1C(c2cc(Cl)cc(Cl)c2)C1(Cl)Cl",  # row 121
    "COc1ccc(C2Oc3cccc(OC(F)F)c3-c3ccc(NC(=O)N(C)C)cc32)cc1OC",             # row 126
    "CN=C(NCC1(O)CCSC1)N1CCN(C(C)C(F)(F)F)CC1",                            # row 132
    "CNC(=O)c1cc(COC(=O)NC2CC3(CCN(c4ccc5cc(F)ccc5n4)CC3)C2)[nH]n1",       # row 141
]

# A subset where PIN default was VERIFIED (an A/B check, HEAD vs. patched) to
# abstain byte-identically both before and after the fix -- the default-tier
# non-widening witnesses.
DEFAULT_STILL_ABSTAINS = [
    "CCC[C@H](N)C(=O)N(C)[C@H]1CC[C@@H]2CN(Cc3ccc(C(F)(F)F)cc3)C[C@@H]21",  # row 23
    "O=C(c1c2ccccc2cc2ccccc12)N1CCN(c2ccc(C(F)(F)F)cn2)CC1",                # row 31
    "O=C(Nc1ccc(Cl)c(C(=O)N(Cl)/N=C/CC(F)(F)F)c1)C1C(c2cc(Cl)cc(Cl)c2)C1(Cl)Cl",  # row 121
    "COc1ccc(C2Oc3cccc(OC(F)F)c3-c3ccc(NC(=O)N(C)C)cc32)cc1OC",             # row 126
    "CN=C(NCC1(O)CCSC1)N1CCN(C(C)C(F)(F)F)CC1",                            # row 132
    "CNC(=O)c1cc(COC(=O)NC2CC3(CCN(c4ccc5cc(F)ccc5n4)CC3)C2)[nH]n1",       # row 141
]

# M4-fold nested-branch prefixes (internal notes
# "M4-fold prefixes"): azido / sulfamoyl / methanesulfonyl on a substituent
# branch off a diol backbone. NOTE (measured via an A/B check): these three are
# UNCHANGED by the Task-1 edit -- best-effort already named them via a
# different path before the fix, so they are regression/documentation
# witnesses for the tier contrast, not conversions this fix produced.
M4_FOLD_CASES = {
    "azido": "OCC(CN=[N+]=[N-])CO",
    "sulfamoyl": "OCC(CS(N)(=O)=O)CO",
    "methanesulfonyl": "OCC(CS(C)(=O)=O)CO",
}


@pytest.fixture(scope="module")
def eng():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                      allow_aromatic_general=True)


@pytest.fixture(scope="module")
def pin_eng():
    return Orthonym()


# --- Tier propagation: the splice is gone and a real name comes back --------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", TIER_PROPAGATION_WITNESSES)
def test_tier_propagation_no_placeholder(eng, smi):
    name = eng.name(smi)
    assert name is not None
    assert not is_refusal_sentinel(name), name
    assert "substituent" not in name
    assert "unknown" not in name.lower()


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", TIER_PROPAGATION_WITNESSES)
def test_tier_propagation_roundtrips(eng, smi):
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    name = eng.name(smi)
    assert opsin_roundtrip_check(smi, name)["passed"], name


# --- Negative: PIN default is byte-identical (no default-tier widening) -----

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", DEFAULT_STILL_ABSTAINS)
def test_default_tier_unchanged(pin_eng, smi):
    # These 6 rows abstain at PIN default both BEFORE and AFTER the fix
    # (verified via scripts/an A/B check on src/orthonym/assembly/
    # substituent_enumerator.py) -- best_effort_ctx is unset at default tier,
    # so name_substituent's effective allow_mancude is unchanged there. Needs
    # the ``opsin_gate`` marker (see module docstring) -- without it the
    # suite's gate-disabled default lets an unguarded recovery-lane candidate
    # leak through, which is a test-harness artifact, not a real regression.
    name = pin_eng.name(smi)
    assert is_refusal_sentinel(name), (
        f"PIN default must still abstain (byte-identical to pre-fix HEAD); "
        f"got a NEW default-tier emission: {name!r}")


# --- M4-fold nested-branch prefixes: best-effort names, default abstains ----

@pytest.mark.opsin_gate
@pytest.mark.parametrize("tag,smi", list(M4_FOLD_CASES.items()))
def test_m4_fold_besteffort_names_nested_prefix(eng, tag, smi):
    name = eng.name(smi)
    assert name is not None
    assert not is_refusal_sentinel(name), f"{tag}: {name!r}"
    assert "substituent" not in name
    assert "unknown" not in name.lower()


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize("tag,smi", list(M4_FOLD_CASES.items()))
def test_m4_fold_besteffort_roundtrips(eng, tag, smi):
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    name = eng.name(smi)
    assert opsin_roundtrip_check(smi, name)["passed"], f"{tag}: {name}"


# =============================================================================
# M2 Task 2 -- route the declined fragment to the general engine (-yl route)
# =============================================================================
#
# Measured (internal notes,
# `.superpowers/sdd/M1-PLAN/m2-task-2-report.md`): after Task 1, 28 of the
# original 38 tier-propagation candidates still abstain because the cascade
# DECLINES a ring/cage fragment that the whole-molecule general engine
# already names standalone. `_route_fragment_to_general_engine`
# (`substituent_enumerator.py`) closes that gap: it H/OH-caps the declined
# fragment, names the capped fragment via `name_compound` at best-effort, and
# converts the result to a `-yl` prefix carrying the correct attachment
# locant (via the injected -OH's own locant, whether it wins suffix
# seniority or falls back to a `<locant>-hydroxy` prefix on the fused/spiro
# ring producers, which build suffix-free names).
#
# ⚠ MEASURED, NOT ASSUMED (a genuine negative finding, not a gap in
# searching): a fresh-process trace over the FIRST 62 rows of
# `abstentions/pubchem10k.jsonl` (outer-wrap on `name_substituent`, gate ON,
# capped per the task's efficiency bound) found 20 real molecules whose
# naming hits this exact decline (rows 8, 9, 14, 18, 25, 27, 30, 32, 33, 35,
# 37, 38, 39, 42, 44, 47, 49, 52, 57, 61) -- but EVERY one of them still
# abstains even with this fix (`scripts/an A/B check` on
# `substituent_enumerator.py`: identical abstain=True with and without the
# fix), because each carries >=1 OTHER independent blocker elsewhere in the
# same molecule (the "mean 4.08 distinct blockers per abstainer" finding).
# A further forward scan of rows 62-165 found 8 rows that now name
# successfully, but `scripts/an A/B check` proved every one of those ALSO
# already named successfully at HEAD (before this fix) via a
# DIFFERENT, pre-existing candidate -- so none of them are attributable to
# this routing either.
#
# The fragments themselves ARE genuinely routable, though (proof by
# construction, the SAME method the a trace itself used for its "row 20" claim):
# `_route_fragment_to_general_engine` returns a real, OPSIN-valid `-yl` token
# for the exact declined fragments from pubchem10k.jsonl rows 42
# (`Cc1cc2c(nc1OCCN1CCC1)OC1(CC1)CNS2=O`, attach atom 6 -- a
# spiro-cyclopropane-fused oxathiazine) and 49
# (`CCN(CC)OCC1C=CC(C2=CC3(CCCNCC3)Oc3ccc(F)cc32)=CN1`, attach atom 11 -- a
# spiro-benzopyran-azepane). The reason NO single-occurrence molecule
# (neither a real pubchem10k/pubchem_2000 row nor a constructed single-copy
# molecule bearing either fragment, `an A/B check`-verified) demonstrates the
# fix in isolation is that the pre-existing "ring/cage-as-parent" universal
# fallback (built in -, before M2) ALWAYS offers an alternative
# candidate that treats the ring system as the parent and everything else as
# a prefix -- e.g. "4-acetamido-6-fluorospiro[1-benzopyran-2,4'-azepane]"
# instead of "N-(6-fluorospiro[1-benzopyran-2,4'-azepane]-4-yl)acetamide" --
# which supersedes the need for substituent-level routing whenever only ONE
# such ring system is present. Doubling the fragment (two independent
# occurrences on the same simple acid chain) removes that escape hatch --
# only ONE ring system can be the parent, so the OTHER MUST be named as a
# substituent -- and is therefore the smallest construction that isolates
# the mechanism. Verified via `scripts/an A/B check
# src/orthonym/assembly/substituent_enumerator.py --...`: HEAD (A)
# abstains (`unknown organic compound`) on both; the working tree (B) names
# both and both round-trip (OPSIN InChI match True).
TASK2_ROUTED_WITNESSES = [
    # row-49 fragment, doubled onto a malonic-acid-shaped chain.
    "OC(=O)C(C1=CC2(CCCNCC2)Oc2ccc(F)cc21)C3=CC4(CCCNCC4)Oc4ccc(F)cc43",
    # row-42 fragment, doubled the same way.
    "OC(=O)C(c1nc2c(cc1C)S(=O)NCC1(CC1)O2)c1nc2c(cc1C)S(=O)NCC1(CC1)O2",
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", TASK2_ROUTED_WITNESSES)
def test_task2_route_no_placeholder(eng, smi):
    name = eng.name(smi)
    assert name is not None
    assert not is_refusal_sentinel(name), name
    assert "substituent" not in name
    assert "unknown" not in name.lower()


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", TASK2_ROUTED_WITNESSES)
def test_task2_route_roundtrips(eng, smi):
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    name = eng.name(smi)
    assert opsin_roundtrip_check(smi, name)["passed"], name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", TASK2_ROUTED_WITNESSES)
def test_task2_default_tier_unchanged(pin_eng, smi):
    # PIN default must still abstain byte-identically to pre-fix HEAD --
    # best_effort_ctx is unset there, so _route_fragment_to_general_engine
    # is never reached (routing is only ever tried under best_effort_ctx).
    name = pin_eng.name(smi)
    assert is_refusal_sentinel(name), (
        f"PIN default must still abstain (byte-identical to pre-fix HEAD); "
        f"got a NEW default-tier emission: {name!r}")


def test_task2_route_declines_on_ring_cutting_extraction():
    """Fail-closed guard: a frag_atoms set whose extraction would CUT a ring
    (one of the a trace's 4 fragmentation artifacts) must decline, never emit a
    fragment describing a different, ring-opened molecule. Reproduces the
    exact ring-cutting (frag_atoms, attach_idx) pairs the trace measured as the
    FIRST candidate attempt for pubchem10k rows 14/38/52 -- each a real
    intermediate the search tries and must reject."""
    from rdkit import Chem
    from orthonym.assembly.substituent_enumerator import (
        _route_fragment_to_general_engine,
    )

    cases = [
        ("CCCN1CCC2(CC1)Oc1ccccc1C1CC(c3ccc(CC)cc3)=NN12",
         [0, 1, 2, 3, 4, 5, 7, 8], 5),
        ("N#CCC[C@@H]1CC[C@]2(CCCN(C(=O)c3cccnc3)C2)O1",
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21], 10),
        ("O=C1CC[C@@]2(CN1Cc1ccccc1)NCCc1[nH]cnc12",
         [2, 3, 4, 5, 14, 15, 16, 17, 18, 19, 20, 21], 5),
    ]
    for smi, frag_atoms, attach_idx in cases:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, smi
        result = _route_fragment_to_general_engine(mol, frag_atoms, attach_idx)
        assert result is None, (smi, frag_atoms, attach_idx, result)


def test_task2_route_declines_on_noncarbon_attach():
    """Fail-closed guard: OH-capping a non-carbon attach atom would change
    the fragment's chemistry (N-OH is a hydroxylamine, O-OH a peroxide,...),
    so the route must decline rather than risk naming a different molecule.
    Reproduces pubchem10k row 8's NITROGEN-attach candidate (attach atom 14,
    a clean single-external-bond, non-ring-cutting extraction -- otherwise a
    working candidate, isolating the carbon-only guard specifically)."""
    from rdkit import Chem
    from orthonym.assembly.substituent_enumerator import (
        _route_fragment_to_general_engine,
    )

    smi = r"Cc1nn(-c2ccccc2)c(C)c1/C=N\n1c(C(F)(F)F)n[nH]c1=S"
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    frag_atoms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    attach_idx = 14
    assert mol.GetAtomWithIdx(attach_idx).GetSymbol() == "N"
    result = _route_fragment_to_general_engine(mol, frag_atoms, attach_idx)
    assert result is None


# =============================================================================
# M2 Task 3 -- central de-mask + splice-guard belt (0-wrong cleanliness)
# =============================================================================
#
# Fail-closed `errors.is_refusal_sentinel` guards added at the splice sites
# that build an N-prefix / ring-prefix / amide string from a substituent-
# cascade name: `rules/amides.py` (`format_n_substitution`),
# `rules/polyfunctional.py` (`name_polyfunctional`'s final assembled `name`),
# `decomposition/fragment_assembly.py` (`_assemble_amide`'s final `result`),
# and `assembly/composer.py` (`_enrich_complex_ring_with_subs`, widened from
# a bare `== "substituent"` equality check to the broader sentinel predicate
# so a DECORATED placeholder is caught too). Each guard VOIDS the candidate
# (returns ``None``) instead of splicing the placeholder into a name.
#
# This is defense-in-depth, not a breadth lever: Tasks 1-2 already convert or
# correctly abstain on every measured pubchem10k row, so the regression-trap
# test below is expected to PASS with zero offenders even without these
# guards (measured before adding them). The guards protect against any
# FUTURE producer that reaches one of these sites with an unrouted, unnamed
# fragment -- a genuine cage that folds to M5, for instance.


def _is_whole_name_abstention(nm: str) -> bool:
    """True when ``nm`` is itself a legitimate, honest whole-molecule refusal
    sentinel -- NOT a real name with the placeholder woven into it.

    ``Orthonym.name`` is always-emit (``namer.py:2877``): a molecule it
    cannot name at all returns one of a small, fixed set of descriptive
    fallback strings verbatim (``errors.py``'s ``_make``/``classify_scope_
    limit`` family) -- most commonly the exact string 'unknown organic
    compound'. That string legitimately CONTAINS 'unknown', so a naive
    substring test flags every honest abstention as an "offender" -- measured
    directly: 140/166 rows in this corpus abstain with exactly that string,
    and 0 of them differ from it by even one character. The defect this test
    actually guards against is the placeholder EMBEDDED in an otherwise real,
    longer, differently-shaped name (e.g. 'N,N-disubstituentacetamide',
    '11-unknownundec-9-enoate') -- distinguished here by exact match against
    the known pure-refusal forms.
    """
    if nm in ("unknown organic compound", "substituent"):
        return True
    return nm.endswith(" compound (not supported)") or nm == (
        "compound with wildcard atoms (not supported)")


@pytest.mark.opsin_gate
def test_placeholder_never_splices(eng):
    """Regression trap: no emitted name may ever contain the substituent
    cascade's placeholder or the whole-molecule 'unknown' sentinel SPLICED
    INTO a real, longer name. A fragment even routing cannot name (e.g. a
    genuine cage -> M5) must abstain outright (the bare sentinel), never
    weave 'substituent'/'unknown' into an otherwise-real name.

    Measured (this task, fresh-process probe over all 166 rows, gate ON,
    ``scratchpad/probe_task3.py``): 140 rows abstain with the exact honest
    sentinel 'unknown organic compound' (not an offender -- see
    ``_is_whole_name_abstention``); 0 timeouts; 0 exceptions; 0 rows where the
    placeholder is embedded in a longer/different constructed name reach the
    final ``eng.name`` result -- Tasks 1-2 (tier propagation + routing) plus
    the existing OPSIN self-consistency gate (visible in this run's log as
    'OPSIN validity gate suppressed unparseable name:...unknown...' for
    several internal candidates, e.g. 'unknownmethanol') already void every
    would-be splice before it reaches the caller. This test therefore passes
    with zero offenders even without the M2 Task 3 guards -- they are
    defense-in-depth against a FUTURE producer reaching one of the 4 guarded
    splice sites with an unrouted fragment, not a breadth lever.
    """
    import json
    import pathlib

    rows = [
        json.loads(line)
        for line in pathlib.Path(
            ".planning/audit-v33/abstentions/pubchem10k.jsonl"
        ).read_text().splitlines()[:400]
    ]
    offenders = [
        (r["smiles"], nm) for r in rows
        if (nm := eng.name(r["smiles"]))
        and ("substituent" in nm or "unknown" in nm.lower())
        and not _is_whole_name_abstention(nm)
    ]
    assert not offenders, offenders[:5]
