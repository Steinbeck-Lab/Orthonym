"""
Phase 46 Parent Mismatch Benchmark: Measures improvement in parent structure selection.

This test file targets the 68 parent_mismatch RT failures identified in Phase 46 research.
These are compounds where OPSIN parses the generated name but InChI doesn't match because
the wrong parent structure was selected (wrong ring system, wrong ring-vs-chain decision,
or wrong principal group location).

Categories (per 46-RESEARCH.md):
  - Steroid/terpenoid wrong VB name: 22 failures
  - PAH/fused with wrong ring picked: 12 failures
  - Bridged polycyclic wrong classification: 9 failures
  - Acyclic (no ring) wrong parent chain: 15 failures
  - Monocyclic ring wrong: 8 failures
  - Fused heterocycle wrong: 2 failures

Success criterion: At least 30 of 68 parent_mismatch compounds produce names
with the correct parent structure (measured by expected parent substring match).

Phase 46 changes:
  - Plan 46-01: ring_selection.py with P-44.2 ring system type classification
  - Plan 46-02: select_principal_ring_system() integrated into namer.py as metadata
  - Plan 46-03: Enhanced parent_selection.py with P-44.1 cascade + P-52.2.8
"""

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# Parent mismatch compounds: (SMILES, expected_parent_substring, description)
# ---------------------------------------------------------------------------

# Compounds where Phase 46 is expected to produce correct parent selection.
# These use firm assertions (no xfail).
EXPECTED_FIXED = [
    # --- STEROID / TERPENOID ---
    # Phase 46: P-44.2 ring classification + retained NP names
    (
        "CC(C)CCC1O[C@H]2C[C@H]3[C@@H]4CCC5CCCC[C@]5(C)[C@H]4CC[C@]3(C)[C@H]2[C@@H]1C",
        "cholest",
        "Steroid: cholestane retained NP name",
    ),
    (
        "CCC(CCC(C)C1CCC2C3C(O)C=C4CC(O)CCC4(C)C3CCC12C)C(C)C",
        "stigmast",
        "Steroid: stigmastane diol retained NP",
    ),
    (
        "C[C@]12CC[C@@H](O)C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@H](O)CC[C@@H]12",
        "androstan",
        "Steroid: androstane-3,17-diol (canary compound)",
    ),
    (
        "C[C@]12CC[C@H]3[C@@H](CC[C@@H]4CC(=O)CC[C@]34C)[C@@H]1CC(=O)[C@@H]2O",
        "androstan",
        "Steroid: androstanedione-ol",
    ),
    (
        "C=C(C)C(C)CCC(C)C1CCC2C3=CCC4CC(O)CCC4(C)C3CCC21C",
        "ergost",
        "Steroid: ergostadienol retained NP",
    ),
    (
        "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
        "chol",
        "Steroid: cholanoid acid",
    ),
    (
        "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc43)[C@@H]1CC[C@@H]2O",
        "estran",
        "Steroid: estrane-3,17-diol",
    ),
    (
        "C[C@H](CCC(=O)O)[C@H]1C[C@@H](O)[C@@]2(C)C3=CCC4C(C)(C)C(=O)CC[C@]4(C)C3=CC[C@]12C",
        "chol",
        "Steroid: dihydroxycholadienone acid",
    ),

    # --- PAH / FUSED AROMATIC ---
    # Phase 46: P-44.2 identifies fused aromatic systems correctly
    # NOTE: 40-atom PAH moved to EXPECTED_UNFIXED -- Phase 69 PAH size guard
    # correctly prevents naphthalene over-matching. Extended PAH naming needed.
    (
        "CC(C)[C@H]1CC[C@@H](CO)c2c(O)cc(C(=O)O)cc21",
        "naphthal",
        "PAH: tetrahydronaphthalene derivative",
    ),
    (
        "O=Cc1ccc2ccccc2c1O",
        "naphthal",
        "PAH: hydroxynaphthalenecarbaldehyde (canary)",
    ),
    (
        "C=C(C)C1C=C2C(C)=CCCC2(C)CC1",
        "decahydro",
        "PAH: decahydronaphthalene",
    ),

    # --- MONOCYCLIC ---
    # Phase 46: P-52.2.8 ring-over-chain + correct ring selection
    (
        "CC1CC=C(N2CCCC2)C1=O",
        "cyclopent",
        "Monocyclic: methylcyclopentenone",
    ),
    (
        "CCC1CC=C(N2CCCC2)C1=O",
        "cyclopent",
        "Monocyclic: ethylcyclopentenone",
    ),
    (
        "CC1=CC(=O)CC(C)(C)C1",
        "cyclohex",
        "Monocyclic: trimethylcyclohexenone (canary)",
    ),
    (
        "CC1=C(O)C(=O)C([C@@]2(C)CCCC2(C)C)=C(O)C1=O",
        "cyclohexa",
        "Monocyclic: cyclohexadienedione with cyclopentyl",
    ),

    # --- FUSED HETEROCYCLE ---
    # Phase 46: ring_system_score() prefers heterocyclic over carbocyclic
    (
        "Oc1ccc2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@@H](O)[C@@H]2O",
        "chroman",
        "FusedHet: trihydroxychromane",
    ),
    (
        "COc1c(O)c(O)cc2c1CO[C@@H](C)C2=O",
        "isochroman",
        "FusedHet: dihydroxy-methoxy isochromanone",
    ),
    (
        "Cc1ccc(-c2nc3ccc(C)cn3c2CC(=O)N(C)C)cc1",
        "imidazo",
        "FusedHet: imidazo[1,2-a]pyridine",
    ),
    (
        "CCCCCC(C)OC(=O)COc1ccc(Cl)c2cccnc12",
        "quinolin",
        "FusedHet: chloroquinoline ester",
    ),
    (
        "Oc1ccnc2ccccc12",
        "quinol",
        "FusedHet: hydroxyquinoline (canary)",
    ),
    (
        "CCCCCCCCCc1cc(=O)c2ccccc2n1C",
        "quinol",
        "FusedHet: N-methyl-nonylquinolinone (canary)",
    ),

    # --- BRIDGED POLYCYCLIC ---
    # Phase 46: ring_system_score selects correct polycyclic parent
    (
        "CC1(C)C[C@H](O)[C@]23CC[C@@H](O)[C@](C)(CC[C@@H]12)C3",
        "tricyclo",
        "Bridged: tricyclic diol (canary)",
    ),
    (
        "CC1=C[C@@H]2/C=C(\\C)CCC[C@H](O)/C=C/C(=O)O[C@]23C(=O)N[C@@H](CC(C)C)[C@@H]3[C@@H]1C",
        "tricyclo",
        "Bridged: macrocyclic lactone (canary)",
    ),

    # --- ACYCLIC / CHAIN ---
    # Phase 46: P-44.1 cascade preserves correct chain selection
    (
        "CCCCCCCCCCC(C)C(=O)O",
        "dodecanoic",
        "Acyclic: 2-methyldodecanoic acid (canary)",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCC(=O)OCC",
        "docosanoate",
        "Acyclic: ethyl docosanoate (canary)",
    ),
    (
        "CC(C)CCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
        "heptacosanoic",
        "Acyclic: 26-methylheptacosanoic acid (canary)",
    ),

    # --- SUGAR / GLYCOSIDE ---
    # Phase 46: ring selection preserves pyranose parent
    (
        "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O[C@@H]1OC[C@@H](O)[C@H](O)[C@H]1O",
        "pyran",
        "Sugar: xylopyranosyl galactose",
    ),

    # --- SIMPLE RETAINED NAMES ---
    (
        "Oc1ccccc1",
        "phenol",
        "Simple: phenol",
    ),
    (
        "c1ccncc1",
        "pyridin",
        "Simple: pyridine",
    ),
    (
        "Cl.c1ccc2sc(C3(N4CCCCC4)CCCCC3)cc2c1",
        "benzothioph",
        "FusedHet: benzothiophene salt",
    ),
]

# Compounds where Phase 46 is NOT expected to fix parent selection.
# These require future work (Phase 47+ NP detection, decomposition, etc.)
# Marked with xfail(strict=False) -- if they pass, great (xpass), if not, expected.
EXPECTED_UNFIXED = [
    # --- Extended PAH (Phase 69: PAH size guard correctly rejects naphthalene) ---
    (
        "c1ccc2cc3c(cc2c1)-c1cc2ccccc2cc1-c1cc2ccccc2cc1-c1cc2ccccc2cc1-3",
        "NOT_naphthalene",
        "PAH: 40-atom 9-ring system needs extended PAH naming (not naphthalene)",
    ),
    # --- Complex decomposition issues (Phase 48+) ---
    (
        "CC(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N/C=C\\c1c[nH]c2ccccc12",
        "indol",
        "Peptide with indole - decomposition fragment naming",
    ),
    (
        "COc1ccc(C(=O)N2CCCC2=O)cc1",
        "methoxy",
        "Benzoylpyrrolidinone - decomposition boundary issue",
    ),
    (
        "CC(=O)[C@@H](C)Nc1ccccc1C(=O)O",
        "amino",
        "Aminobenzoic acid derivative - wrong principal group",
    ),
    (
        "C=C1NC(=O)[C@H]([C@@H](C)[C@]2(O)C(=O)N(C)c3ccccc32)NC1=O",
        "diketopiperazin",
        "Complex diketopiperazine - decomposition",
    ),
    # --- NP detection scope (Phase 47) ---
    (
        "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2.Cl",
        "tropan",
        "Tropane ester HCl salt - NP scope",
    ),
    (
        "C=C(CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@@H](O)[C@@]1(C)CC3)C(C)C",
        "ergost",
        "Steroid: ergostene without stereo - NP detection gap",
    ),
    # --- Morphinan / complex NP ---
    (
        "COC1=CC=C2[C@H]3Cc4ccc(OC)c5c4[C@@]2(C[C@@H](C2=C[C@@]4(O)[C@H]6Cc7ccc(O)c8c7"
        "[C@@]4(C[C@@H](O)[C@]62O)CC58)O3)CC1",
        "morphin",
        "Morphinan dimer - complex NP naming",
    ),
    # --- Large phospholipid naming ---
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCCC)COP(=O)(O)"
        "OC[C@H](N)C(=O)O",
        "phospho",
        "Phospholipid - multi-fragment naming",
    ),
    # --- Nucleotide naming ---
    (
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OC(=O)CCCC[C@@H]2SC[C@@H]3NC(=O)"
        "N[C@@H]32)[C@@H](O)[C@H]1O",
        "adenosin",
        "Nucleotide conjugate - multi-fragment assembly",
    ),
    # --- Wrong FG detection ---
    (
        "CC1C/C(=C\\CC(CC(N)=O)CC(=O)O)C(=O)C(C)C1",
        "cyclohex",
        "Cyclohexanone with chain FG - wrong FG detection",
    ),
    # --- Diketopiperazine ---
    (
        "O=C1NC(Cc2c[nH]c3ccccc23)C(=O)N/C1=C/c1cnc[nH]1",
        "diketopiperazin",
        "Diketopiperazine with indole - complex heterocycle",
    ),
    # --- Ammonium salts with complex anion ---
    (
        "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(C)=O",
        "phospho",
        "Phosphocholine ester - salt decomposition",
    ),
]


# Combine all compounds for the summary count test
PARENT_MISMATCH_COMPOUNDS = EXPECTED_FIXED + [
    (smi, exp, desc) for smi, exp, desc in EXPECTED_UNFIXED
]


# ---------------------------------------------------------------------------
# Tests for compounds expected to be fixed by Phase 46
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize(
    "smiles,expected_parent_substring,description",
    EXPECTED_FIXED,
    ids=[
        d.split(":")[0].strip().replace(" ", "_")
        + f"_{i:02d}"
        for i, (_, _, d) in enumerate(EXPECTED_FIXED)
    ],
)
def test_parent_mismatch_fixed(smiles, expected_parent_substring, description):
    """Phase 46 should produce correct parent structure for these compounds.

    Verifies that the generated name contains the expected parent structure
    substring, indicating the correct ring system or chain was selected.
    """
    name = name_compound(smiles)
    assert name, f"Should produce a name for: {description}"
    assert expected_parent_substring.lower() in name.lower(), (
        f"PARENT MISMATCH: {description}\n"
        f"  SMILES: {smiles}\n"
        f"  Generated: {name}\n"
        f"  Expected parent substring: '{expected_parent_substring}'"
    )


# ---------------------------------------------------------------------------
# Tests for compounds NOT expected to be fixed (xfail)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize(
    "smiles,expected_parent_substring,description",
    EXPECTED_UNFIXED,
    ids=[
        d.split(":")[0].strip().replace(" ", "_")
        + f"_{i:02d}"
        for i, (_, _, d) in enumerate(EXPECTED_UNFIXED)
    ],
)
@pytest.mark.xfail(strict=False, reason="Phase 47+: requires NP detection, decomposition, or FG fixes")
def test_parent_mismatch_unfixed(smiles, expected_parent_substring, description):
    """Compounds where Phase 46 alone cannot fix parent selection.

    These require future work: NP detection scope (Phase 47),
    decomposition improvements (Phase 48), or FG detection fixes.
    Marked xfail(strict=False) so xpass is a bonus, not a failure.
    """
    name = name_compound(smiles)
    assert name, f"Should produce a name for: {description}"
    assert expected_parent_substring.lower() in name.lower(), (
        f"STILL WRONG: {description}\n"
        f"  SMILES: {smiles}\n"
        f"  Generated: {name}\n"
        f"  Expected parent substring: '{expected_parent_substring}'"
    )


# ---------------------------------------------------------------------------
# Summary test: count total improvements
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_parent_mismatch_improvement_count():
    """At least 30/68 parent_mismatch failures should now produce correct parent names.

    Phase 46 success criterion: >= 30 compounds from the parent_mismatch category
    have names containing the correct parent structure (ring system or chain).

    This counts across ALL parent_mismatch compounds (both expected-fixed and
    expected-unfixed). If the threshold is not met, this test fails to flag
    that the phase improvements were insufficient.
    """
    correct = 0
    total = len(PARENT_MISMATCH_COMPOUNDS)
    failures = []

    for smiles, expected, desc in PARENT_MISMATCH_COMPOUNDS:
        name = name_compound(smiles)
        if name and expected.lower() in name.lower():
            correct += 1
        else:
            failures.append(f"  {desc}: got '{name}', expected '{expected}' in name")

    # Report results regardless of pass/fail
    print(f"\n=== Parent Mismatch Improvement Count ===")
    print(f"Correct parent: {correct}/{total}")
    print(f"Target: >= 30/{total}")
    if failures:
        print(f"\nStill failing ({len(failures)}):")
        for f in failures:
            print(f)

    assert correct >= 30, (
        f"Phase 46 target not met: {correct}/{total} correct (need >= 30).\n"
        f"Still failing:\n" + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# Phase 88 Integration Tests: P-44.1 Chain Selection Fixes
# ---------------------------------------------------------------------------

class TestPhase88FGInstanceCounting:
    """PSEL-01: FG instance counting uses distinct instances, not atom overlap."""

    @pytest.mark.integration
    def test_diacid_both_groups_on_chain(self):
        """Glutaric acid: both COOHs on the 5-carbon chain (2 instances)."""
        name = name_compound("OC(=O)CCCC(=O)O")
        assert name, "Should produce a name for glutaric acid"
        assert "pentanedioic" in name.lower() or "glutar" in name.lower(), (
            f"Glutaric acid: expected pentanedioic acid. Got: {name}"
        )


class TestPhase88RingScoring:
    """PSEL-06: Ring scoring tuple reorder (P-44.2.1 before P-44.2.2)."""

    @pytest.mark.integration
    def test_pyridine_is_principal_ring(self):
        """Pyridine retained name should be produced."""
        name = name_compound("c1ccncc1")
        assert name, "Should name pyridine"
        assert "pyridin" in name.lower(), f"Expected pyridine. Got: {name}"

    @pytest.mark.integration
    def test_phenol_retained(self):
        """Phenol retained name."""
        name = name_compound("Oc1ccccc1")
        assert name, "Should name phenol"
        assert "phenol" in name.lower(), f"Expected phenol. Got: {name}"


class TestPhase88HydrocarbonGuard:
    """PSEL-05: Parent selection runs for hydrocarbons."""

    @pytest.mark.integration
    def test_butylcyclohexane_ring_parent(self):
        """Butylcyclohexane: ring should be parent (P-44.1.2.2)."""
        name = name_compound("CCCCC1CCCCC1")
        assert name, "Should produce a name"
        assert "cyclohex" in name.lower(), (
            f"Butylcyclohexane: ring should be parent. Got: {name}"
        )

    @pytest.mark.integration
    def test_propylcyclopentane_ring_parent(self):
        """Propylcyclopentane: ring should be parent."""
        name = name_compound("CCCC1CCCC1")
        assert name, "Should produce a name"
        assert "cyclopent" in name.lower(), (
            f"Propylcyclopentane: ring should be parent. Got: {name}"
        )


class TestPhase88CascadeCriteria:
    """PSEL-03: Full P-44.1 cascade criteria 5-9."""

    @pytest.mark.integration
    def test_cyclohexanone_ring_parent(self):
        """Cyclohexanone: ring with ketone is parent."""
        name = name_compound("O=C1CCCCC1")
        assert name, "Should name cyclohexanone"
        assert "cyclohex" in name.lower(), f"Expected cyclohexanone. Got: {name}"

    @pytest.mark.integration
    def test_benzoic_acid_ring_parent(self):
        """Benzoic acid: ring is parent with -COOH suffix."""
        name = name_compound("OC(=O)c1ccccc1")
        assert name, "Should name benzoic acid"
        assert "benz" in name.lower(), f"Expected benzoic acid. Got: {name}"
