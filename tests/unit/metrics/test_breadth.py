"""v29 Phase 0 — tests for the committed breadth instrument (metrics/breadth.py).

The instrument is load-bearing: every prior breadth figure came from an ad-hoc
harness in /tmp and at least one was a ~4x mirage (see memory
project_v28_pin_backlog_clusterA MEGA-LESSON). These tests pin the metric
definitions so a future number is comparable and cannot silently change meaning.

Pure logic only — no naming, no OPSIN, no Java.
"""
import pytest
from rdkit import Chem

from orthonym.metrics.breadth import (
    aggregate,
    classify_outcome,
    molecule_components,
    parse_refusal_codes,
    refusal_structure,
    residual_refusal_code,
    ring_systems,
    row_attribution,
    terminal_basis_available,
    terminal_site,
    terminal_stage,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------- ring_systems

def test_ring_systems_groups_fused_rings_into_one_system():
    """Naphthalene is ONE ring system, not two.

    The v29 census bug (topo_census.py v1) over-counted 'fused' by treating
    separate ring systems joined by linkers as fused; the inverse error - not
    merging genuinely fused rings - would under-count per-fragment components.
    """
    mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    assert len(ring_systems(mol)) == 1
    assert len(ring_systems(mol)[0]) == 10


def test_ring_systems_keeps_linker_joined_rings_separate():
    """Biphenyl is TWO ring systems - they share no atom, only a bond."""
    mol = Chem.MolFromSmiles("c1ccccc1-c1ccccc1")
    assert len(ring_systems(mol)) == 2


def test_ring_systems_returns_empty_for_acyclic():
    mol = Chem.MolFromSmiles("CCCCO")
    assert ring_systems(mol) == []


def test_ring_systems_handles_spiro_as_one_system():
    """Spiro rings share exactly one atom, so they are one connected system."""
    mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCCC2")
    assert len(ring_systems(mol)) == 1


# ----------------------------------------------------------- classify_outcome

def test_classify_outcome_emit_when_name_present():
    res = {"name": "benzene", "tier": "T1", "source": "pin_path"}
    assert classify_outcome(res) == "EMIT"


def test_classify_outcome_abstain_on_none_name():
    """best-effort tier returns name=None on abstention (clean-abstain contract)."""
    res = {"name": None, "tier": "T5", "source": "abstain"}
    assert classify_outcome(res) == "ABSTAIN"


def test_classify_outcome_abstain_on_descriptive_failure_name():
    """The PIN tier leaks a descriptive string instead of None - still an abstention.

    Counting 'unknown organic compound' as an emission is exactly how a breadth
    number becomes a mirage.
    """
    res = {"name": "unknown organic compound", "tier": "T5", "source": "abstain"}
    assert classify_outcome(res) == "ABSTAIN"


def test_classify_outcome_abstain_on_empty_string():
    assert classify_outcome({"name": "", "tier": "T5"}) == "ABSTAIN"


# --------------------------------------------------------- parse_refusal_codes

def test_parse_refusal_codes_extracts_drop_code_and_reason():
    lines = [
        "orthonym.assembly.substituent_naming|ring_fragment_declined_by_ring_engine substituent_skip: "
        "reason=ring_fragment_declined_by_ring_engine atom_count=9",
    ]
    assert parse_refusal_codes(lines) == [
        "ring_fragment_declined_by_ring_engine:ring_fragment_declined_by_ring_engine"]


def test_parse_refusal_codes_extracts_general_engine_refusal():
    lines = ["orthonym.assembly.general_engine|general_engine_declined: "
             "branch unnameable (tier-5 fallback)"]
    assert parse_refusal_codes(lines) == ["producer_refused:branch unnameable"]


def test_parse_refusal_codes_deduplicates_within_a_molecule():
    """A code firing 5 times on one molecule is still ONE blocker for that molecule."""
    line = ("orthonym.x|ring_fragment_declined_by_ring_engine substituent_skip: "
            "reason=ring_fragment_declined_by_ring_engine atom_count=9")
    assert len(parse_refusal_codes([line, line, line])) == 1


def test_parse_refusal_codes_captures_drop_codes_that_carry_no_reason_field():
    """Real call sites omit `reason=` entirely.

    substituent_all_candidates_filtered (rules/polyfunctional.py,
    assembly/handlers/_handler_shared.py) and suppress_duplicate_bare_amino_prefix
    (rules/polyfunctional.py) log without `reason=`. A regex requiring `reason=`
    makes them vanish from the census, so a genuinely blocking site ranks as
    ZERO purely because of its log format.
    """
    lines = [
        "orthonym.rules.polyfunctional|substituent_all_candidates_filtered substituent_skip: locant=3",
        "orthonym.rules.polyfunctional|suppress_duplicate_bare_amino_prefix hygiene_skip: something",
    ]
    codes = parse_refusal_codes(lines)
    assert any(c.startswith("substituent_all_candidates_filtered") for c in codes)
    assert any(c.startswith("suppress_duplicate_bare_amino_prefix") for c in codes)


def test_parse_refusal_codes_ignores_log_lines_that_do_not_refuse():
    """Only lines that actually END the naming attempt may become codes.

    Both of these log and then continue: the stereo backstop is detection-only,
    and `namer.py:568` "OPSIN grammar validation failed" is followed by
    `return name` at `namer.py:572` — it ships the original name unchanged.
    Counting either would manufacture blockers that block nothing.

    SUPERSEDES an earlier version of this test which also listed
    "OPSIN validity gate suppressed unparseable name" as noise. Measurement
    refuted that: the OPSIN validity gate SUPPRESSES to the failure fallback
    (`namer.py:1224-1230`) and was the terminal site for a real abstainer
    (`[NH2-]x5.[Ru+2]` -> 'ruthenium(II) pentaamide'), which is why 9 of the 11
    uncoded abstainers were unattributable. It is now a code; see
    test_parse_refusal_codes_captures_the_opsin_validity_gate.
    """
    lines = ["orthonym.x|Stereo backstop: 'alanine' has 1 R/S but name lacks descriptors",
             "orthonym.y|OPSIN grammar validation failed: handler=chain name='xyz'"]
    assert parse_refusal_codes(lines) == []


def test_parse_refusal_codes_captures_the_self01_gate_suppression():
    """SELF-01 is a REFUSAL SITE, and the census was blind to it.

    Measured on the v30 P0 best-effort run, one fresh process per molecule: 8 of
    the 11 abstainers that logged no refusal code died here — a name was built and
    then rejected as a different molecule. Ranking a build order without this site
    ranks the producers only, while the project record says the 0-wrong margin is
    the GATE.
    """
    lines = ["orthonym.namer|self_consistency rejected (different molecule): "
             "'palladium(II) acetate' (opsin=CC(=O)[O-].CC(=O)[O-].[Pd+2])"]
    assert parse_refusal_codes(lines) == ["self_consistency_rejected:different_molecule"]


def test_parse_refusal_codes_captures_the_opsin_validity_gate():
    """`namer.py:1224` suppresses to the failure fallback — a terminal refusal.

    Terminal site for `[NH2-]x5.[Ru+2]` ('ruthenium(II) pentaamide'), one of the
    11 uncoded abstainers.
    """
    lines = ["orthonym.namer|OPSIN validity gate suppressed unparseable name: "
             "'ruthenium(II) pentaamide'"]
    assert parse_refusal_codes(lines) == ["opsin_unparseable:"]


def test_parse_refusal_codes_dedups_a_gate_that_fires_on_several_candidates():
    """SELF-01 rejecting four candidates is still ONE blocker for the molecule."""
    line = ("orthonym.namer|self_consistency rejected (different molecule): 'x' (opsin=C)")
    assert parse_refusal_codes([line, line, line, line]) == [
        "self_consistency_rejected:different_molecule"]


# -------------------------------------------------------- molecule_components

def test_molecule_components_counts_ring_systems_and_acyclic_pieces():
    """Per-fragment p needs the denominator: how many pieces must ALL name.

    3-[(2,3-dimethylphenyl)amino]propanenitrile: one ring system (the arene)
    plus the acyclic remainder. Whole-molecule success is the product over these,
    which is why p must approach 1 per component.
    """
    mol = Chem.MolFromSmiles("Cc1cccc(NCCC#N)c1C")
    comps = molecule_components(mol)
    assert len(comps) >= 2
    kinds = [c["kind"] for c in comps]
    assert "ring_system" in kinds
    assert "acyclic" in kinds


def test_molecule_components_covers_every_heavy_atom_exactly_once():
    """The component partition must be total and disjoint, or p is meaningless."""
    mol = Chem.MolFromSmiles("CC(=O)N(CC1CO1)C(C)C")
    comps = molecule_components(mol)
    seen = [a for c in comps for a in c["atoms"]]
    assert sorted(seen) == list(range(mol.GetNumHeavyAtoms()))


def test_molecule_components_excludes_explicit_isotopic_hydrogens():
    """RDKit keeps isotopic H as explicit graph atoms, unlike implicit H.

    For DMSO-d6-style input GetNumHeavyAtoms() and GetAtoms() disagree, so a
    partition over GetAtoms() breaks the documented "heavy atoms" contract.
    Isotopes are a real, tracked category here (P-82), not a hypothetical.
    """
    mol = Chem.MolFromSmiles("[2H]C([2H])([2H])C([2H])([2H])O")
    comps = molecule_components(mol)
    seen = sorted(a for c in comps for a in c["atoms"])
    heavy = sorted(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    assert seen == heavy


def test_molecule_components_handles_disconnected_salt():
    """Salts are an explicitly tracked class; the partition must stay total."""
    mol = Chem.MolFromSmiles("CC(=O)[O-].[Na+]")
    comps = molecule_components(mol)
    seen = sorted(a for c in comps for a in c["atoms"])
    heavy = sorted(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    assert seen == heavy
    assert len(comps) == 2


def test_molecule_components_for_acyclic_molecule_is_single_piece():
    mol = Chem.MolFromSmiles("CCCCO")
    comps = molecule_components(mol)
    assert len(comps) == 1
    assert comps[0]["kind"] == "acyclic"


# ------------------------------------------------------------------ aggregate

def _row(outcome, tier="T5", codes=(), wrong=False, ncomp=2, ncomp_named=2):
    return {"outcome": outcome, "tier": tier, "refusal_codes": list(codes),
            "structure_wrong": wrong, "n_components": ncomp,
            "n_components_named": ncomp_named}


def test_aggregate_computes_emit_rate():
    rows = [_row("EMIT", "T1"), _row("EMIT", "T4"), _row("ABSTAIN"), _row("ABSTAIN")]
    out = aggregate(rows)
    assert out["n"] == 4
    assert out["emit_rate"] == pytest.approx(0.5)


def test_aggregate_reports_tier_histogram():
    rows = [_row("EMIT", "T1"), _row("EMIT", "T1"), _row("EMIT", "T4")]
    assert aggregate(rows)["tiers"] == {"T1": 2, "T4": 1}


def test_aggregate_counts_structure_wrong_separately_from_abstention():
    """T3: constitutionally-wrong count must never be folded into the emit rate."""
    rows = [_row("EMIT", "T4", wrong=True), _row("EMIT", "T4"), _row("ABSTAIN")]
    out = aggregate(rows)
    assert out["structure_wrong"] == 1
    assert out["emit_rate"] == pytest.approx(2 / 3)


def test_aggregate_refusal_census_counts_molecules_not_occurrences():
    rows = [_row("ABSTAIN", codes=["ring_fragment_declined_by_ring_engine:x", "substituent_is_bare_functional_group:y"]),
            _row("ABSTAIN", codes=["ring_fragment_declined_by_ring_engine:x"])]
    census = aggregate(rows)["refusal_census"]
    assert census["ring_fragment_declined_by_ring_engine:x"] == 2
    assert census["substituent_is_bare_functional_group:y"] == 1


def test_aggregate_refusal_census_restricted_to_abstainers_ranks_build_order():
    """A DROP code can fire and the molecule STILL name via another path.

    Ranking the build order on the all-molecule census therefore over-counts
    sites that are already survivable. The abstainer-restricted census is the
    one that answers 'which site actually costs us breadth'.
    """
    rows = [
        # fires but the molecule recovered and emitted -> must NOT rank
        _row("EMIT", "T1", codes=["substituent_is_bare_functional_group:fg_only"]),
        _row("EMIT", "T1", codes=["substituent_is_bare_functional_group:fg_only"]),
        # genuinely blocked
        _row("ABSTAIN", codes=["ring_fragment_declined_by_ring_engine:ring_declined"]),
    ]
    out = aggregate(rows)
    assert out["refusal_census"]["substituent_is_bare_functional_group:fg_only"] == 2
    assert out["refusal_census_abstain"] == {"ring_fragment_declined_by_ring_engine:ring_declined": 1}


def test_aggregate_per_fragment_p_is_component_weighted():
    """p = named components / total components, pooled across molecules.

    Pooling (not averaging per-molecule ratios) keeps p comparable across corpora
    with different component counts.
    """
    rows = [_row("ABSTAIN", ncomp=4, ncomp_named=2), _row("EMIT", ncomp=2, ncomp_named=2)]
    assert aggregate(rows)["per_fragment_p"] == pytest.approx(4 / 6)


def test_aggregate_projects_whole_molecule_ceiling_from_p():
    rows = [_row("EMIT", ncomp=2, ncomp_named=2) for _ in range(2)]
    out = aggregate(rows)
    assert out["per_fragment_p"] == pytest.approx(1.0)
    assert out["projected_emit_independent"] == pytest.approx(1.0)


def test_aggregate_projection_uses_mean_of_powers_not_power_of_mean():
    """E[p**n], not p**E[n] — Jensen's inequality makes them differ materially.

    p**mean(n) is convex in n so it systematically UNDERSTATES the independence
    model whenever component counts vary (every real corpus). On live data
    (p=0.8097, n in 1..18) the two differ by 8pp: 45.5% vs 53.5%. The wrong one
    would make a genuinely successful phase look like it failed to show up.
    """
    rows = [_row("ABSTAIN", ncomp=1, ncomp_named=1),
            _row("ABSTAIN", ncomp=9, ncomp_named=9)]
    # p == 1.0 here, so force a sub-1 p via a third row with unnamed components
    rows.append(_row("ABSTAIN", ncomp=10, ncomp_named=5))
    out = aggregate(rows)
    p, ns = out["per_fragment_p"], [1, 9, 10]
    assert out["projected_emit_independent"] == pytest.approx(
        sum(p ** k for k in ns) / 3)
    # and it must NOT be the power-of-mean form
    assert out["projected_emit_independent"] != pytest.approx(p ** (20 / 3))


def test_aggregate_reports_context_loss_separately_from_component_loss():
    """Two distinct loss terms, mapping to different build phases.

    component loss (p < 1)  -> ring/fragment naming gaps
    context loss            -> components that name STANDALONE but fail in
                               context (the assembly gap)
    Conflating them hides which phase is responsible for a flat number.
    """
    # every component names standalone (p=1) but half the molecules abstain
    rows = [_row("EMIT", ncomp=2, ncomp_named=2),
            _row("ABSTAIN", ncomp=2, ncomp_named=2)]
    out = aggregate(rows)
    assert out["per_fragment_p"] == pytest.approx(1.0)
    assert out["projected_emit_independent"] == pytest.approx(1.0)
    assert out["context_loss"] == pytest.approx(0.5)


def test_aggregate_nulls_per_fragment_fields_when_components_not_measured():
    """A real-looking 0.0 is indistinguishable from 'measured and genuinely zero'."""
    rows = [_row("EMIT", ncomp=3, ncomp_named=0)]
    out = aggregate(rows, components_measured=False)
    assert out["per_fragment_p"] is None
    assert out["projected_emit_independent"] is None
    assert out["context_loss"] is None


def test_aggregate_counts_tautomer_differences_apart_from_wrong_structures():
    """A tautomer difference is NOT a wrong structure and must never inflate T3.

    The first v29 baseline reported structure_wrong=2; both were mobile-H
    tautomers (a benzimidazole NH hop and a guanidine), adjudicated as correct
    names. Mobile-H is ubiquitous, so folding tautomers into T3 would manufacture
    phantom 0-wrong violations on the project's #1 invariant.
    """
    rows = [
        {"outcome": "EMIT", "tier": "T1", "structure_wrong": False,
         "tautomer_differs": True, "n_components": 1, "n_components_named": 1},
        {"outcome": "EMIT", "tier": "T1", "structure_wrong": True,
         "n_components": 1, "n_components_named": 1},
    ]
    out = aggregate(rows)
    assert out["structure_wrong"] == 1
    assert out["tautomer_differs"] == 1


def test_aggregate_counts_opsin_unparseable_emissions_for_t6():
    rows = [{"outcome": "EMIT", "tier": "T4", "opsin_unparseable": True,
             "n_components": 1, "n_components_named": 1},
            {"outcome": "EMIT", "tier": "T4", "n_components": 1,
             "n_components_named": 1}]
    assert aggregate(rows)["opsin_unparseable"] == 1


def test_aggregate_handles_empty_input_without_dividing_by_zero():
    out = aggregate([])
    assert out["n"] == 0
    assert out["emit_rate"] == 0.0
    assert out["per_fragment_p"] == 0.0


def test_aggregate_carries_the_only_ranked_structure_into_the_summary():
    """v30 P0-T2 item 5: queryable from the run JSON without re-measuring.

    The measurement costs ~675 s, so a census that lives only in stdout is a
    census whose numbers cannot be re-checked.
    """
    rows = [_row("EMIT", "T1"), _row("ABSTAIN", codes=["ring_fragment_declined_by_ring_engine:x"])]
    st = aggregate(rows)["refusal_structure"]
    assert st["single_site_ceiling"] == pytest.approx(1.0)
    assert st["sites"][0]["site"] == "ring_fragment_declined_by_ring_engine:x"


# --------------------------------------------------------- residual attribution

def test_residual_refusal_code_names_an_exception_by_its_type():
    """An EXC row is a CRASH, not a refusal — and must not read as one.

    One row on the v30 P0 run: TypeError "'<' not supported between instances of
    'str' and 'int'" on CC1(CCCC2(C1CC(=O)C3=C2CCC(C3)(C)C=C)C)C. The instrument's
    own `except` turned it into an abstainer, so the census inherited a row no
    producer ever refused.
    """
    row = {"outcome": "EXC", "refusal_codes": [],
           "exc": "TypeError:'<' not supported between instances of 'str' and 'int'"}
    assert residual_refusal_code(row) == "EXC:TypeError"


def test_residual_refusal_code_reads_the_limit_catalog_code_when_nothing_logged():
    """The unsupported-element classifier returns BEFORE the namer logs anything.

    Measured: the [99Tc]-labelled sorbitol row produced ZERO log lines of any
    kind, so there is no log stream to parse and the mechanism is only knowable
    from the returned limit_code.
    """
    row = {"outcome": "ABSTAIN", "refusal_codes": [],
           "limit_code": "UNSUPPORTED_ELEMENT"}
    assert residual_refusal_code(row) == "LIMIT:UNSUPPORTED_ELEMENT"


def test_residual_refusal_code_distinguishes_its_mechanisms_rather_than_bucketing():
    """No catch-all: a single bucket would absorb the next instrument gap.

    Each mechanism gets its own name, and anything unrecognised returns None so
    it is COUNTED as a gap instead of being quietly filed.
    """
    codes = {
        residual_refusal_code({"outcome": "TIMEOUT", "refusal_codes": []}),
        residual_refusal_code({"outcome": "SKIP", "refusal_codes": [],
                               "reason": "unparseable_input"}),
        residual_refusal_code({"outcome": "ABSTAIN", "refusal_codes": [],
                               "limit_code": "UNNAMEABLE"}),
    }
    assert codes == {"TIMEOUT:per_molecule_alarm", "SKIP:unparseable_input",
                     "LIMIT:UNNAMEABLE"}
    # unrecognised -> None, never a bucket
    assert residual_refusal_code({"outcome": "ABSTAIN", "refusal_codes": []}) is None


def test_residual_refusal_code_never_displaces_a_producer_attributed_site():
    """Residual attribution is for depth-0 rows ONLY.

    If it could fire on a coded row it would inflate that row's blocker count and
    corrupt both the depth histogram and every ONLY count.
    """
    row = {"outcome": "ABSTAIN", "refusal_codes": ["ring_fragment_declined_by_ring_engine:x"],
           "limit_code": "UNNAMEABLE"}
    assert residual_refusal_code(row) is None


def test_refusal_structure_attributes_every_uncoded_abstainer():
    """T3 acceptance: unattributed abstainers -> 0, and a leftover is LOUD."""
    rows = [_row("EMIT", "T1"),
            {"outcome": "ABSTAIN", "refusal_codes": [],
             "limit_code": "UNSUPPORTED_ELEMENT", "smiles": "[Pd+2]"},
            {"outcome": "EXC", "refusal_codes": [], "exc": "TypeError:x",
             "smiles": "C"}]
    st = refusal_structure(rows)
    assert st["uncoded_abstainers"] == 2
    assert st["residual_attribution"] == {"LIMIT:UNSUPPORTED_ELEMENT": 1,
                                          "EXC:TypeError": 1}
    assert st["unattributed_abstainers"] == 0


def test_refusal_structure_reports_an_unattributable_abstainer_as_a_gap():
    """A row no branch recognises must be counted and NAMED, not absorbed.

    This is the anti-catch-all guarantee: the next instrument gap shows up as a
    nonzero unattributed count with its SMILES, instead of swelling a bucket.
    """
    rows = [{"outcome": "ABSTAIN", "refusal_codes": [], "smiles": "CCO"}]
    st = refusal_structure(rows)
    assert st["unattributed_abstainers"] == 1
    assert st["unattributed_smiles"] == ["CCO"]
    assert st["residual_attribution"] == {}


# ------------------------------------------------------ ONLY-ranked structure

def test_refusal_structure_credits_only_to_a_single_blocked_abstainer():
    """One code -> that site is the SOLE blocker, so ONLY == 1. (Acceptance 5a)"""
    rows = [_row("ABSTAIN", codes=["producer_refused:unsupported suffix for pg='ester'"])]
    st = refusal_structure(rows)
    site = st["sites"][0]
    assert site["touched"] == 1
    assert site["first"] == 1
    assert site["only"] == 1
    assert st["single_blocked_total"] == 1


def test_refusal_structure_credits_no_only_to_any_site_on_a_four_blocker_row():
    """Four codes -> all four are touched, NONE is single-blocked. (Acceptance 5b)

    This is the whole point of the metric: a molecule blocked by four sites needs
    all four cleared, so no one-site fix converts it. Under the touched ranking
    such a row credits all four sites equally, which is how ring_fragment_declined_by_ring_engine became a
    milestone target at 116 touched while being the sole blocker on 1 molecule.
    """
    codes = ["ring_fragment_declined_by_ring_engine:ring_declined", "substituent_is_bare_functional_group:fg_only",
             "producer_refused:not an analyzable spiro ring system", "polyfunctional_producer_returned_none:pf_none"]
    st = refusal_structure([_row("ABSTAIN", codes=codes)])
    assert {s["site"] for s in st["sites"]} == set(codes)
    assert all(s["touched"] == 1 for s in st["sites"])
    assert all(s["only"] == 0 for s in st["sites"])
    assert st["single_blocked_total"] == 0
    assert st["multi_blocked_total"] == 1
    assert st["depth_histogram"] == {"4": 1}
    assert st["mean_blockers_per_abstainer"] == pytest.approx(4.0)


def test_refusal_structure_never_counts_an_uncoded_abstainer_as_single_blocked():
    """Zero codes is UNCODED, which is the opposite of known-single. (Acceptance 5c)

    Folding depth-0 rows into ONLY would raise the ceiling using rows whose
    blocker is unknown — the exact way this instrument would lie in the
    optimistic direction.
    """
    rows = [_row("EMIT", "T1"),
            {"outcome": "ABSTAIN", "refusal_codes": [],
             "limit_code": "UNNAMEABLE", "smiles": "C"}]
    st = refusal_structure(rows)
    assert st["depth_histogram"]["0"] == 1
    assert st["single_blocked_total"] == 0
    assert st["sites"] == []
    # ceiling counts the emission only, NOT the uncoded abstainer
    assert st["single_site_ceiling"] == pytest.approx(0.5)


def test_refusal_structure_ceiling_is_emitted_plus_only_over_n():
    """(emitted + SUM ONLY) / n. (Acceptance 5d)

    2 emitted + 2 single-blocked + 1 double-blocked over n=5 -> 4/5.
    The double-blocked row must contribute NOTHING to the ceiling.
    """
    rows = [_row("EMIT", "T1"), _row("EMIT", "T1"),
            _row("ABSTAIN", codes=["A:1"]),
            _row("ABSTAIN", codes=["B:1"]),
            _row("ABSTAIN", codes=["A:1", "B:1"])]
    st = refusal_structure(rows)
    assert st["single_blocked_total"] == 2
    assert st["single_site_ceiling"] == pytest.approx(4 / 5)
    by = {s["site"]: s for s in st["sites"]}
    assert by["A:1"]["touched"] == 2 and by["A:1"]["only"] == 1
    assert by["B:1"]["touched"] == 2 and by["B:1"]["only"] == 1


def test_refusal_structure_ranks_by_only_not_by_touched():
    """The ordering IS the deliverable: touched cannot size a fix.

    Mirrors the live contrast — a high-touch site that is almost never the sole
    blocker must rank BELOW a low-touch site that usually is. On the v30 P0 run
    that is ring_fragment_declined_by_ring_engine (116 touched, ONLY 1) below pg='ester' (36 touched, ONLY 10).
    """
    rows = ([_row("ABSTAIN", codes=["HIGH_TOUCH", "OTHER"]) for _ in range(9)]
            + [_row("ABSTAIN", codes=["HIGH_TOUCH"])]
            + [_row("ABSTAIN", codes=["LOW_TOUCH"]) for _ in range(3)])
    st = refusal_structure(rows)
    order = [s["site"] for s in st["sites"]]
    assert order[0] == "LOW_TOUCH", order
    assert order.index("LOW_TOUCH") < order.index("HIGH_TOUCH")
    by = {s["site"]: s for s in st["sites"]}
    assert by["HIGH_TOUCH"]["touched"] == 10 and by["HIGH_TOUCH"]["only"] == 1
    assert by["LOW_TOUCH"]["touched"] == 3 and by["LOW_TOUCH"]["only"] == 3


def test_refusal_structure_only_is_computed_over_all_abstainers_not_a_subset():
    """Invariant 14's trap: a signature computed over a subset is not a defect size.

    A site that is single-blocking on some abstainers and co-occurring on others
    must report the ONLY count from the WHOLE abstainer set. Restricting to, say,
    the depth-1 rows would report ONLY == touched for every site and make every
    site look individually unlockable.
    """
    rows = [_row("ABSTAIN", codes=["S"]),
            _row("ABSTAIN", codes=["S", "T"]),
            _row("ABSTAIN", codes=["S", "T", "U"])]
    by = {s["site"]: s for s in refusal_structure(rows)["sites"]}
    assert by["S"]["touched"] == 3
    assert by["S"]["only"] == 1
    assert by["T"]["touched"] == 2 and by["T"]["only"] == 0


def test_refusal_structure_excludes_emitted_rows_from_every_count():
    """A code can fire and the molecule still name — that site blocks nothing."""
    rows = [_row("EMIT", "T1", codes=["SURVIVABLE"]),
            _row("EMIT", "T1", codes=["SURVIVABLE"]),
            _row("ABSTAIN", codes=["BLOCKING"])]
    st = refusal_structure(rows)
    assert {s["site"] for s in st["sites"]} == {"BLOCKING"}
    assert st["n_abstain"] == 1
    assert st["mean_blockers_per_abstainer"] == pytest.approx(1.0)


def test_refusal_structure_treats_a_repeated_code_as_one_blocker():
    """['X','X'] is single-blocked. Guards a legacy or hand-built row.

    Without the dedup, len(codes)==2 would exile the row to multi-blocked and
    silently lower the ceiling.
    """
    st = refusal_structure([_row("ABSTAIN", codes=["X:1", "X:1"])])
    assert st["single_blocked_total"] == 1
    assert st["sites"][0]["only"] == 1
    assert st["total_site_hits"] == 1
    # the histogram must agree, or multi_blocked_total stops reconciling
    assert st["depth_histogram"] == {"1": 1}
    assert st["multi_blocked_total"] == 0


def test_refusal_structure_site_order_is_independent_of_row_order():
    """Determinism: Counter iteration follows corpus order, so ties need a
    value-based tie-break or the same run reorders under a reshuffled corpus."""
    rows = [_row("ABSTAIN", codes=["B:1"]), _row("ABSTAIN", codes=["A:1"]),
            _row("ABSTAIN", codes=["C:1"])]
    fwd = [s["site"] for s in refusal_structure(rows)["sites"]]
    rev = [s["site"] for s in refusal_structure(list(reversed(rows)))["sites"]]
    assert fwd == rev == ["A:1", "B:1", "C:1"]


# ------------------------------------------- v30 P0-T4: TERMINAL attribution

def _term_row(outcome="ABSTAIN", codes=(), code=None, detail=None,
              measured=True, **kw):
    """A row as ``run_worker`` writes it once the abstention channel is wired."""
    row = _row(outcome, codes=codes, **kw)
    if measured:
        row["terminal_measured"] = True
        row["terminal_code"] = code
        row["terminal_detail"] = detail
    return row


def test_terminal_attribution_does_not_credit_the_exploratory_producer_codes():
    """The measured [I-](CCO)c1ccccc1 leak class.

    Log-scraped, that molecule names four sites -- substituent_is_bare_functional_group:fg_only,
    ring_fragment_declined_by_ring_engine:ring_fragment_declined_by_ring_engine, n_branch_ring_substituent_unnameable and
    producer_refused:branch unnameable -- none of which ended it: a name WAS built and
    the P10 charge-conservation veto (namer.py:3916) removed it. Those four
    codes are EXPLORATORY (they fire while the engine searches candidates), so
    on the terminal basis they must keep `touched` and lose `first`/`ONLY`.
    Crediting ONLY to any of them aims the build order at a fix that converts
    nothing, which is the whole reason this basis exists.
    """
    row = _term_row(
        codes=["substituent_is_bare_functional_group:fg_only", "ring_fragment_declined_by_ring_engine:ring_fragment_declined_by_ring_engine",
               "n_branch_ring_substituent_unnameable:<no_reason>", "producer_refused:branch unnameable"],
        code="GATE_SUPPRESSED", detail="charge_dropped")
    log = refusal_structure([row])
    term = refusal_structure([row], basis="terminal")

    innocent = {"substituent_is_bare_functional_group:fg_only",
                "ring_fragment_declined_by_ring_engine:ring_fragment_declined_by_ring_engine",
                "n_branch_ring_substituent_unnameable:<no_reason>", "producer_refused:branch unnameable"}
    by_site = {s["site"]: s for s in term["sites"]}

    # the four keep `touched` (hazard 3: the union must survive)
    assert innocent <= set(by_site)
    assert all(by_site[s]["touched"] == 1 for s in innocent)
    # ... and lose every scrap of first/ONLY credit
    assert all(by_site[s]["first"] == 0 for s in innocent)
    assert all(by_site[s]["only"] == 0 for s in innocent)

    # the veto that actually terminated it owns first and ONLY
    veto = "TERM:GATE_SUPPRESSED:charge_dropped"
    assert by_site[veto]["first"] == 1
    assert by_site[veto]["only"] == 1
    assert term["single_blocked_total"] == 1

    # and the LOG basis is left untouched -- it credited nobody with ONLY
    # (four blockers) but did credit substituent_is_bare_functional_group with `first`
    assert log["single_blocked_total"] == 0
    assert {s["site"]: s["first"] for s in log["sites"]}["substituent_is_bare_functional_group:fg_only"] == 1


def test_terminal_and_log_bases_disagree_on_the_same_row():
    """A row whose log codes and terminal code name DIFFERENT mechanisms.

    Measured on CC1=NC(C=N1)CO: the last thing logged is
    opsin_unparseable:, but the channel recorded
    BRANCH_UNNAMEABLE/enumerator_ring_fallback -- a GENERATION-stage failure.
    So the terminal site is not "the last log line" and cannot be derived from
    the log stream at all; the two bases must be able to disagree, and the
    disagreement must be visible rather than reconciled away.
    """
    row = _term_row(
        codes=["ring_fragment_declined_by_ring_engine:ring_fragment_declined_by_ring_engine",
               "producer_refused:chain too short", "opsin_unparseable:"],
        code="BRANCH_UNNAMEABLE", detail="enumerator_ring_fallback")
    union, attribution = row_attribution(row, basis="terminal")
    assert attribution == ["TERM:BRANCH_UNNAMEABLE:enumerator_ring_fallback"]
    # the terminal site is NOT any of the logged codes
    assert attribution[0] not in (row["refusal_codes"])
    # the union carries all four so depth still reports 4 blockers touched
    assert len(set(union)) == 4
    term = refusal_structure([row], basis="terminal")
    assert term["depth_histogram"] == {"4": 1}
    # log basis on the same row attributes the FIRST log code instead
    log_union, log_attr = row_attribution(row, basis="log")
    assert log_attr[0] == "ring_fragment_declined_by_ring_engine:ring_fragment_declined_by_ring_engine"
    assert log_union == row["refusal_codes"]


def test_terminal_attribution_never_leaks_between_consecutive_rows():
    """Row N must not inherit row N-1's terminal code.

    The channel is a MODULE-LEVEL ``threading.local`` slot, so a missing
    per-molecule reset would make molecule N report molecule N-1's mechanism
    and the census would be confidently wrong -- worse than the log basis it
    replaces. Two guards are needed and both are asserted here:

    * the census must read each row's OWN stamped code (never a shared
      cursor), and
    * a row the channel could not attribute must fall through to its own
      residual mechanism, not borrow the previous row's site.
    """
    first = _term_row(codes=["substituent_is_bare_functional_group:fg_only"],
                      code="GATE_SUPPRESSED", detail="charge_dropped")
    # named right after `first`, attributed by the channel to something else
    second = _term_row(codes=["producer_refused:branch unnameable"],
                       code="BRANCH_UNNAMEABLE", detail="enumerator_last_resort")
    # and a row the channel ran on but could not attribute at all
    third = _term_row(codes=[], code="OTHER", detail=None, ncomp=1)
    third["limit_code"] = "UNSUPPORTED_ELEMENT"

    assert (row_attribution(first, "terminal")[1]
            == ["TERM:GATE_SUPPRESSED:charge_dropped"])
    assert (row_attribution(second, "terminal")[1]
            == ["TERM:BRANCH_UNNAMEABLE:enumerator_last_resort"])
    assert row_attribution(third, "terminal")[1] == []

    term = refusal_structure([first, second, third], basis="terminal")
    only = {s["site"]: s["only"] for s in term["sites"]}
    assert only["TERM:GATE_SUPPRESSED:charge_dropped"] == 1
    assert only["TERM:BRANCH_UNNAMEABLE:enumerator_last_resort"] == 1
    # the third row borrowed nothing -- it is named by its OWN mechanism
    assert term["residual_attribution"] == {"LIMIT:UNSUPPORTED_ELEMENT": 1}
    assert term["unattributed_abstainers"] == 0
    assert term["no_attribution_rows"] == 1


def test_terminal_basis_is_refused_for_rows_measured_without_the_channel():
    """A pre-T4 run must NOT silently census as an empty terminal basis.

    `aggregate` returns None for the terminal structure rather than a
    zero-looking one: an all-residual census over unstamped rows is
    indistinguishable from "the channel found nothing", which is the project's
    standing 'a PERFECT harness result means it did not RUN' failure mode.
    """
    old = [_row("EMIT"), _row("ABSTAIN", codes=["substituent_is_bare_functional_group:fg_only"])]
    assert terminal_basis_available(old) is False
    out = aggregate(old, components_measured=False)
    assert out["refusal_structure_terminal"] is None
    assert out["terminal_basis_available"] is False

    new = [_row("EMIT"), _term_row(codes=["substituent_is_bare_functional_group:fg_only"],
                                   code="GATE_SUPPRESSED", detail="self01_mismatch")]
    assert terminal_basis_available(new) is True
    out2 = aggregate(new, components_measured=False)
    assert out2["refusal_structure_terminal"]["basis"] == "terminal"
    assert out2["refusal_structure"]["basis"] == "log"


def test_terminal_site_names_the_gap_when_the_channel_recorded_nothing():
    """An UNINSTRUMENTED post-generation termination stays visible and NAMED.

    The general engine's inline E1-certificate rejection (namer.py:3586) and
    its no-jar / dropped-stereo discards (namer.py:3634) throw a generated
    candidate away without recording, so the channel reports OTHER with no
    detail. Measured example: [Ni+2].[Bi+3] logs
    producer_refused:multi-fragment (G3 scope) and records nothing. A bare TERM:OTHER
    bucket would absorb exactly the instrument gaps this attribution exists to
    expose, so the site is prefixed TERMGAP: -- named, countable, and
    impossible to mistake for a channel attribution.
    """
    row = _term_row(codes=["producer_refused:multi-fragment (G3 scope)"],
                    code="OTHER", detail=None)
    assert (terminal_site(row)
            == "TERMGAP:producer_refused:multi-fragment (G3 scope)")
    # a detail-bearing OTHER is a real named site, not a gap
    named = _term_row(codes=[], code="OTHER", detail="isotope_decorator_failed")
    assert terminal_site(named) == "TERM:OTHER:isotope_decorator_failed"
    # an unmeasured row is a THIRD, separately labelled case
    unmeasured = _row("ABSTAIN", codes=["substituent_is_bare_functional_group:fg_only"])
    assert (terminal_site(unmeasured) is None)
    assert (row_attribution(unmeasured, "terminal")[1]
            == ["TERMGAP-UNMEASURED:substituent_is_bare_functional_group:fg_only"])
    # an EMIT row never has a terminal site, whatever it carries
    assert terminal_site(_term_row("EMIT", code="GATE_SUPPRESSED")) is None


def test_terminal_ceiling_is_flagged_vacuous_and_log_ceiling_is_not():
    """The ceiling must not be quoted as a build-order bound on the terminal
    basis. Terminal attribution names exactly one site per row, so every
    attributed abstainer is 'single-blocked' by construction and the ceiling
    collapses to the attribution rate. The flag is what stops a reader
    treating a 100% ceiling as a finding about the engine.
    """
    rows = [_row("EMIT")] + [
        _term_row(codes=["substituent_is_bare_functional_group:fg_only", "principal_group_branch_overlap:pg_branch_overlap"],
                  code="GATE_SUPPRESSED", detail="self01_mismatch")
        for _ in range(3)]
    log = refusal_structure(rows)
    term = refusal_structure(rows, basis="terminal")
    assert log["ceiling_is_vacuous"] is False
    assert term["ceiling_is_vacuous"] is True
    # log: two blockers each -> nobody is single-blocked -> ceiling = emit only
    assert log["single_site_ceiling"] == pytest.approx(0.25)
    assert log["multi_blocked_total"] == 3
    # terminal: all three collapse onto one terminal site
    assert term["single_site_ceiling"] == pytest.approx(1.0)
    assert term["multi_blocked_total"] == 0


def test_row_attribution_rejects_an_unknown_basis():
    """A typo'd basis must raise, not silently fall through to `log` and
    report log numbers under a terminal heading."""
    with pytest.raises(ValueError, match="basis must be one of"):
        row_attribution(_term_row(codes=["substituent_is_bare_functional_group:fg_only"]), basis="termnial")


def test_terminal_stage_rollup_separates_capability_from_correctness():
    """The bucket P1 must be sized from — and the trap it exists to prevent.

    `needs_engine` (NO_PARENT / BRANCH_UNNAMEABLE) is the only bucket new
    naming CAPABILITY converts: no candidate was produced. `suppressed`
    (GATE_SUPPRESSED / COVERAGE_DOWNGRADE) means a candidate WAS produced and a
    gate removed it -- and for a self01_mismatch that candidate denoted a
    DIFFERENT molecule, so its lever is an upstream correctness fix, never a
    looser gate. Sizing a capability milestone off `suppressed` would aim it at
    the gate that is the only thing holding the 0-wrong invariant.

    The bucket names are shared verbatim with  so
    the two censuses cannot drift into two vocabularies for one axis.
    """
    assert terminal_stage("TERM:GATE_SUPPRESSED:self01_mismatch") == "suppressed"
    assert terminal_stage("TERM:COVERAGE_DOWNGRADE:garbled") == "suppressed"
    assert terminal_stage("TERM:BRANCH_UNNAMEABLE:x") == "needs_engine"
    assert terminal_stage("TERM:NO_PARENT:UNSUPPORTED_RING_SYSTEM") == "needs_engine"
    assert terminal_stage("TERM:OTHER:isotope_decorator_failed") == "other"
    # a log-derived site is NOT silently promoted into an engine bucket
    assert terminal_stage("TERMGAP:producer_refused:multi-fragment (G3 scope)") == "uninstrumented"
    assert terminal_stage("TERMGAP-UNMEASURED:substituent_is_bare_functional_group:fg_only") == "uninstrumented"

    rows = [_row("EMIT"),
            _term_row(code="GATE_SUPPRESSED", detail="self01_mismatch"),
            _term_row(code="GATE_SUPPRESSED", detail="opsin_unparseable"),
            _term_row(code="BRANCH_UNNAMEABLE", detail="enumerator_last_resort"),
            _term_row(codes=["producer_refused:multi-fragment (G3 scope)"],
                      code="OTHER", detail=None)]
    term = refusal_structure(rows, basis="terminal")
    assert term["terminal_stage_rollup"] == {
        "suppressed": 2, "needs_engine": 1, "uninstrumented": 1}
    # the rollup exists ONLY on the terminal basis -- a log-basis rollup would
    # imply the log codes carry a stage, and they do not
    assert refusal_structure(rows)["terminal_stage_rollup"] is None
