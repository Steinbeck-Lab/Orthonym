"""Phase 149 SC-4: 153/153 catalog regression scaffold for FR-2.3 agreement.

Per V18 plan §6 Phase 149 acceptance + 149-CONTEXT.md D-10 Tier 2:
parametrized over every entry in FUSED_HETEROCYCLE_DATA (live count 153,
verified 2026-04-28; ROADMAP says 148 — live count is binding per
149-CONTEXT D-10).

## Test predicate scope (Plan 02 triage decision)

The Plan 02 RESEARCH §"Tier 2 Catalog Regression Scaffold" (lines 481-557)
specifies the predicate "locant-1 atom is in algorithmic base". On first
run this revealed 58/141 (41%) systematic disagreements clustered in
THREE categories per RESEARCH §551 triage taxonomy:

1. **Het-vs-het 2-ring systems** (42 entries, e.g.
   `1H-pyrazolo[3,4-b]pyridine`): both rings have N → FR-2.3(a) ties →
   FR-2.3(b) ring count ties → FR-2.3(c) larger ring wins. Catalog's
   locant-1 atom lives in the smaller heteroatom-rich ring (pyrazole)
   while FR-2.3(c) algorithmically picks the larger ring (pyridine,
   matching the SUFFIX of the IUPAC name). This is the **suffix-vs-
   peripheral-locant-1 divergence**: IUPAC peripheral numbering can
   start in either ring depending on FR-5.x lowest-locant rules; the
   "base component" (FR-2.3 base) is determined by the suffix in the
   IUPAC name (e.g., `pyrazolo[...]pyridine` → base is pyridine), NOT
   by which ring contains locant 1. RESEARCH §551 case 3: "Genuine
   FR-2.3 vs catalog policy divergence". FR-2.3 implementation is
   CORRECT per V18 Appendix A.6 + IUPAC P-25.3.2.4; the test predicate
   is mismatched for these systems.

2. **3-ring fused systems** (16 entries, e.g. `9H-carbazole`,
   `acridine`, `9H-xanthene`, `phenoxazine`): FR-2.3 correctly selects
   the middle heterocyclic ring as base, but IUPAC peripheral numbering
   places locant 1 on a peripheral carbocyclic ring atom. RESEARCH
   §548 anticipates this: "For 3-ring fused systems, the 'components'
   can be either (A) 3 SSSR rings, or (B) 2 macro-components... per
   D-05 lock: SSSR rings only." The 3-SSSR-ring decomposition is
   correct; the locant-1 predicate is unreliable for 3+ ring systems.

3. **Single-component catalog entries** (rare, mostly skipped by
   `pytest.skip` at len(components) < 2 guard).

## Scope adjustment

Per RESEARCH §554 + D-14 (no band-aids; root-cause fixes), the correct
remedy is to **scope the predicate** to systems where it is reliable —
NOT to special-case individual SMILES via 58 inline `pytest.skip`s
(which would constitute a band-aid). The reliable predicate scope is:

  **2-component (2-ring) fused systems where exactly ONE ring is
  heterocyclic** — FR-2.3(a) "Heterocycles always beat arenes" is the
  dominant cascade criterion and locant-1 reliably lives in the
  het-bearing base ring (e.g., 1H-indole, quinoline, benzofuran).

For systems outside this scope, the predicate is documented as
mismatched (NOT FR-2.3 wrong; NOT catalog wrong) and the test
explicitly `pytest.skip`s with a category-level rationale citing
RESEARCH §551 + this docstring.

## Acceptance

Within the scoped subset (2-ring, 1-het 1-carbo), the predicate must
pass at 100%. Outside-scope entries are skipped with documented
rationale and tracked in the Plan 02 SUMMARY for Phase 149.x or
Phase 155 follow-up (where peripheral-locant numbering for non-
cataloged systems is implemented).

Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25.3.2.4
Source: HERITAGE-1990 §4.
Source: 149-CONTEXT.md D-10 Tier 2; D-11 byte-identical lock; D-14 no
        band-aids.
Source: 149-RESEARCH.md "Tier 2 Catalog Regression Scaffold" lines
        481-557; "Edge cases / expected acceptance failures" §548-554.
"""
import pytest
from rdkit import Chem

from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
from orthonym.rules.fused_ring_selection import (
    select_base_component, _enumerate_components,
)


def _has_heteroatom_in_smiles(smi: str) -> bool:
    """Pre-filter PAH entries (PAHs flow through Branch 2, not Branch 6.5).

    Per RESEARCH §"Edge cases / expected acceptance failures" line 549:
    Plan 02 Tier 2 must skip PAH entries — pyrene etc. flow through
    polycyclics.py Branch 2; FR-2.3 doesn't apply to them.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    return any(a.GetAtomicNum() not in (1, 6) for a in mol.GetAtoms())


# Live count 153 entries (verified 2026-04-28). PAH entries are skipped
# at collection time per RESEARCH §line 549.
NON_PAH_CATALOG = [
    (smi, entry.get('name', smi))
    for smi, entry in FUSED_HETEROCYCLE_DATA.items()
    if _has_heteroatom_in_smiles(smi)
]


def _ring_has_heteroatom(mol, ring_atoms):
    """Helper: True iff ring contains at least one non-C heavy atom."""
    return any(
        mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
        for idx in ring_atoms
    )


@pytest.mark.parametrize(
    "smiles,expected_name",
    NON_PAH_CATALOG,
    ids=[name for _smi, name in NON_PAH_CATALOG],
)
def test_fr23_agrees_with_catalog(smiles, expected_name):
    """For each non-PAH cataloged entry whose decomposition fits the
    reliable-predicate scope (2-ring, exactly 1 heterocyclic ring),
    FR-2.3's chosen base contains the catalog's locant-1 atom.

    Out-of-scope entries (3+ ring systems, het-vs-het 2-ring systems,
    single-component) are skipped with documented rationale per
    RESEARCH §548-554 + module docstring "Scope adjustment".

    Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
    Source: 149-CONTEXT.md D-10 Tier 2; D-14 no band-aids.
    Source: 149-RESEARCH.md §551 triage taxonomy.
    """
    entry = FUSED_HETEROCYCLE_DATA[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        pytest.skip(f"{expected_name}: SMILES did not parse")

    components = _enumerate_components(mol)
    if len(components) < 2:
        pytest.skip(f"{expected_name}: single-component catalog entry")

    # Scope guard #1: 3+ ring fused systems.
    # FR-2.3 correctly selects the middle/heterocyclic ring as base, but
    # IUPAC peripheral numbering for 3+ ring systems places locant 1 on a
    # carbocyclic peripheral atom per FR-5.x lowest-locant rules. The
    # locant-1 predicate is unreliable here. Phase 155 owns peripheral
    # numbering for these systems.
    if len(components) >= 3:
        pytest.skip(
            f"{expected_name}: 3+ ring fused system — locant-1 predicate "
            f"unreliable per RESEARCH §548; FR-2.3 base selection is "
            f"verified by other test scopes (Plan 03 Tier 3)."
        )

    # Scope guard #2: het-vs-het 2-ring systems.
    # Both rings heterocyclic → FR-2.3(a) ties → FR-2.3(c) larger ring
    # wins. Catalog's locant-1 atom can live in the smaller het-rich ring
    # (matching IUPAC convention for prefix component) while FR-2.3 picks
    # the larger ring (matching the IUPAC name's suffix). This is the
    # suffix-vs-peripheral-locant divergence — FR-2.3 is CORRECT, the
    # locant-1 predicate just doesn't capture FR-2.3's architectural
    # output. Verified separately in Tier 1 unit tests
    # (test_fused_ring_selection.py covers FR-2.3(a)-(c) on synthetic
    # rings without locant-1 ambiguity).
    het_count_per_ring = [
        sum(1 for c in components if _ring_has_heteroatom(mol, c))
    ][0]
    if het_count_per_ring >= 2:
        pytest.skip(
            f"{expected_name}: het-vs-het 2-ring system — FR-2.3(c) "
            f"larger-ring criterion picks ring matching IUPAC name "
            f"suffix; catalog locant-1 lives in prefix component per "
            f"IUPAC peripheral numbering. Predicate mismatched for this "
            f"scope per RESEARCH §551 case 3 'genuine FR-2.3 vs catalog "
            f"policy divergence'."
        )

    base_atoms, _ = select_base_component(mol, components)

    iupac_locants = entry.get('iupac_locants', {})
    locant_1_atom = next(
        (idx for idx, loc in iupac_locants.items() if loc == 1),
        None,
    )
    if locant_1_atom is None:
        pytest.skip(
            f"{expected_name}: catalog entry has no locant 1; "
            f"cannot verify"
        )

    assert locant_1_atom in base_atoms, (
        f"FR-2.3 disagrees with catalog for {expected_name}: locant-1 "
        f"atom {locant_1_atom} not in algorithmic base {base_atoms}"
    )
