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
    ring_systems,
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
        "orthonym.assembly.substituent_naming|DROP-24 substituent_skip: "
        "reason=ring_fragment_declined_by_ring_engine atom_count=9",
    ]
    assert parse_refusal_codes(lines) == ["DROP-24:ring_fragment_declined_by_ring_engine"]


def test_parse_refusal_codes_extracts_general_engine_refusal():
    lines = ["orthonym.assembly.general_engine|general_engine refused: "
             "branch unnameable (tier-5 fallback)"]
    assert parse_refusal_codes(lines) == ["REFUSE:branch unnameable"]


def test_parse_refusal_codes_deduplicates_within_a_molecule():
    """A code firing 5 times on one molecule is still ONE blocker for that molecule."""
    line = ("orthonym.x|DROP-24 substituent_skip: "
            "reason=ring_fragment_declined_by_ring_engine atom_count=9")
    assert len(parse_refusal_codes([line, line, line])) == 1


def test_parse_refusal_codes_captures_drop_codes_that_carry_no_reason_field():
    """Real call sites omit `reason=` entirely, and one has a non-numeric suffix.

    DROP-17 (rules/polyfunctional.py, assembly/handlers/_handler_shared.py) and
    DROP-HYG04 (rules/polyfunctional.py) log without `reason=`. A regex requiring
    `reason=` and `\\d+` makes them vanish from the census, so a genuinely
    blocking site ranks as ZERO purely because of its log format.
    """
    lines = [
        "orthonym.rules.polyfunctional|DROP-17 substituent_skip: locant=3",
        "orthonym.rules.polyfunctional|DROP-HYG04 hygiene_skip: something",
    ]
    codes = parse_refusal_codes(lines)
    assert any(c.startswith("DROP-17") for c in codes)
    assert any(c.startswith("DROP-HYG04") for c in codes)


def test_parse_refusal_codes_ignores_unrelated_log_noise():
    lines = ["orthonym.x|Stereo backstop: 'alanine' has 1 R/S but name lacks descriptors",
             "orthonym.y|OPSIN validity gate suppressed unparseable name: 'xyz'"]
    assert parse_refusal_codes(lines) == []


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
    rows = [_row("ABSTAIN", codes=["DROP-24:x", "DROP-01:y"]),
            _row("ABSTAIN", codes=["DROP-24:x"])]
    census = aggregate(rows)["refusal_census"]
    assert census["DROP-24:x"] == 2
    assert census["DROP-01:y"] == 1


def test_aggregate_refusal_census_restricted_to_abstainers_ranks_build_order():
    """A DROP code can fire and the molecule STILL name via another path.

    Ranking the build order on the all-molecule census therefore over-counts
    sites that are already survivable. The abstainer-restricted census is the
    one that answers 'which site actually costs us breadth'.
    """
    rows = [
        # fires but the molecule recovered and emitted -> must NOT rank
        _row("EMIT", "T1", codes=["DROP-01:fg_only"]),
        _row("EMIT", "T1", codes=["DROP-01:fg_only"]),
        # genuinely blocked
        _row("ABSTAIN", codes=["DROP-24:ring_declined"]),
    ]
    out = aggregate(rows)
    assert out["refusal_census"]["DROP-01:fg_only"] == 2
    assert out["refusal_census_abstain"] == {"DROP-24:ring_declined": 1}


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
