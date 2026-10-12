"""a phase: 153/153 catalog regression scaffold for.3 agreement.

Per V18 plan a phase acceptance + 149-internal notes Tier 2:
parametrized over every entry in FUSED_HETEROCYCLE_DATA (live count 153,
verified 2026-04-28; ROADMAP says 148 — live count is binding per
149-internal notes).

## Test predicate scope (Plan 02 triage decision)

The Plan 02 RESEARCH §"Tier 2 Catalog Regression Scaffold" (lines 481-557)
specifies the predicate "locant-1 atom is in algorithmic base". On first
run this revealed 58/141 (41%) systematic disagreements clustered in
THREE categories per RESEARCH triage taxonomy:

1. **Het-vs-het 2-ring systems** (42 entries, e.g.
   `1H-pyrazolo[3,4-b]pyridine`): both rings have N →.3(a) ties →
   .3(b) ring count ties →.3(c) larger ring wins. Catalog's
   locant-1 atom lives in the smaller heteroatom-rich ring (pyrazole)
   while.3(c) algorithmically picks the larger ring (pyridine,
   matching the SUFFIX of the IUPAC name). This is the **suffix-vs-
   peripheral-locant-1 divergence**: IUPAC peripheral numbering can
   start in either ring depending on.x lowest-locant rules; the
   "base component" (.3 base) is determined by the suffix in the
   IUPAC name (e.g., `pyrazolo[...]pyridine` → base is pyridine), NOT
   by which ring contains locant 1. RESEARCH case 3: "Genuine
   .3 vs catalog policy divergence"..3 implementation is
   CORRECT per V18 Appendix A.6 + IUPAC; the test predicate
   is mismatched for these systems.

2. **3-ring fused systems** (16 entries, e.g. `9H-carbazole`,
   `acridine`, `9H-xanthene`, `phenoxazine`):.3 correctly selects
   the middle heterocyclic ring as base, but IUPAC peripheral numbering
   places locant 1 on a peripheral carbocyclic ring atom. RESEARCH
    anticipates this: "For 3-ring fused systems, the 'components'
   can be either (A) 3 SSSR rings, or (B) 2 macro-components... per
    lock: SSSR rings only." The 3-SSSR-ring decomposition is
   correct; the locant-1 predicate is unreliable for 3+ ring systems.

3. **Single-component catalog entries** (rare, mostly skipped by
   `pytest.skip` at len(components) < 2 guard).

## Scope adjustment

Per RESEARCH + (no band-aids; root-cause fixes), the correct
remedy is to **scope the predicate** to systems where it is reliable —
NOT to special-case individual SMILES via 58 inline `pytest.skip`s
(which would constitute a band-aid). The reliable predicate scope is:

  **2-component (2-ring) fused systems where exactly ONE ring is
  heterocyclic** —.3(a) "Heterocycles always beat arenes" is the
  dominant cascade criterion and locant-1 reliably lives in the
  het-bearing base ring (e.g., 1H-indole, quinoline, benzofuran).

For systems outside this scope, the predicate is documented as
mismatched (NOT.3 wrong; NOT catalog wrong) and the test
explicitly `pytest.skip`s with a category-level rationale citing
RESEARCH + this docstring.

## Acceptance

Within the scoped subset (2-ring, 1-het 1-carbo), the predicate must
pass at 100%. Outside-scope entries are skipped with documented
rationale and tracked in the Plan 02 SUMMARY for a phase.x or
a phase follow-up (where peripheral-locant numbering for non-
cataloged systems is implemented).

Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
Source: https://iupac.qmul.ac.uk/BlueBook/P2.html
Source: AUTONOM-1990
Source: 149-internal notes Tier 2; byte-identical lock; no
        band-aids.
Source: internal notes "Tier 2 Catalog Regression Scaffold" lines
        481-557; "Edge cases / expected acceptance failures" -554.
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
    polycyclics.py Branch 2;.3 doesn't apply to them.
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


def _locant_1_atom(entry):
    """Atom index the catalog numbers 1, or None."""
    iupac_locants = entry.get('iupac_locants', {})
    return next(
        (idx for idx, loc in iupac_locants.items() if loc == 1),
        None,
    )


def _in_scope(smiles):
    """True iff the locant-1 predicate is reliable for this catalog entry.

    The scope is the 2-ring fused systems with exactly one heterocyclic ring
    and a catalog locant 1 (module docstring, "Scope adjustment"). The entries
    outside it are not parametrized: they used to be skipped with a rationale
    per category on every run, and so never asserted anything.

    - SMILES that does not parse, or a single-component entry.
    - 3+ ring fused systems:.3 correctly selects the middle/heterocyclic
      ring as base, but IUPAC peripheral numbering places locant 1 on a
      carbocyclic peripheral atom per.x lowest-locant rules (RESEARCH
      section 548);.3 base selection is verified by other test scopes.
    - Het-vs-het 2-ring systems:.3(c) larger-ring criterion picks the ring
      matching the IUPAC name suffix while the catalog locant 1 lives in the
      prefix component (RESEARCH section 551 case 3, "genuine.3 vs
      catalog policy divergence"). Verified separately in
      test_fused_ring_selection.py, which covers.3(a)-(c) on synthetic
      rings without locant-1 ambiguity.
    - An entry without a locant 1: nothing to verify.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    components = _enumerate_components(mol)
    if len(components) != 2:
        return False
    if sum(1 for c in components if _ring_has_heteroatom(mol, c)) >= 2:
        return False
    return _locant_1_atom(FUSED_HETEROCYCLE_DATA[smiles]) is not None


IN_SCOPE_CATALOG = [(smi, name) for smi, name in NON_PAH_CATALOG if _in_scope(smi)]
assert IN_SCOPE_CATALOG, "no catalog entry is in the scope of the locant-1 predicate"


@pytest.mark.parametrize(
    "smiles,expected_name",
    IN_SCOPE_CATALOG,
    ids=[name for _smi, name in IN_SCOPE_CATALOG],
)
def test_fr23_agrees_with_catalog(smiles, expected_name):
    """For each non-PAH cataloged entry whose decomposition fits the
    reliable-predicate scope (2-ring, exactly 1 heterocyclic ring),
    .3's chosen base contains the catalog's locant-1 atom.

    Out-of-scope entries (3+ ring systems, het-vs-het 2-ring systems,
    single-component) are left out of the parameter list (``_in_scope``)
    with documented rationale per RESEARCH section 548-554 + module
    docstring "Scope adjustment".

    Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
    Source: 149-internal notes Tier 2; no band-aids.
    Source: internal notes section 551 triage taxonomy.
    """
    mol = Chem.MolFromSmiles(smiles)
    components = _enumerate_components(mol)
    base_atoms, _ = select_base_component(mol, components)
    locant_1_atom = _locant_1_atom(FUSED_HETEROCYCLE_DATA[smiles])

    assert locant_1_atom in base_atoms, (
        f"FR-2.3 disagrees with catalog for {expected_name}: locant-1 "
        f"atom {locant_1_atom} not in algorithmic base {base_atoms}"
    )
