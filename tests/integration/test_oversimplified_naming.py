"""Integration tests for coverage-gated naming of previously oversimplified compounds.

Verifies that the coverage gates from Phase 47 (Plans 01 and 02) improve naming
quality for compounds that were previously returning bare scaffold names or
names that were too short for their molecular complexity.

Three test groups:
  A. test_coverage_gate_no_bare_scaffold -- parametrized over improved compounds
  B. test_naming_rate_maintained -- diverse compounds still produce non-unknown names
  C. test_canary_compounds_stable -- sampling of golden canary compounds still correct

Data source: benchmark_v5_results_p46.json (pre-Phase-47 baseline)
  - Pop A: heavy > 15, len(name) < heavy // 2  (29 compounds)
  - Pop B: heavy > 20, no digits/hyphens in name  (25 compounds)
  - Combined unique: 43 compounds (11 overlap)
  - Phase 47 improved: 13 of 43
"""

import pytest
from rdkit import Chem

from orthonym import name_compound


# ---------------------------------------------------------------------------
# Group A: Compounds that the coverage gates improved
# ---------------------------------------------------------------------------
# These compounds previously returned bare scaffold or oversimplified names.
# After Phase 47 coverage gates, they now produce longer, more descriptive names.
# Format: (SMILES, old_bare_name, heavy_atom_count)

IMPROVED_COMPOUNDS = [
    (
        "CC(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N/C=C\\c1c[nH]c2ccccc12",
        "1H-indole",
        35,
    ),
    (
        "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2.Cl",
        "tropane",
        22,
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCC[C@H](O)C(=O)N[C@@H](COP(=O)(O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O)[C@H](O)CCCCCCCCCCCCCCC",
        "(2S)-2-hydroxytetracosanamide",
        62,
    ),
    (
        "CCCCCC/C=C\\CCCCCCCC(=O)OC[C@@H]1COP(=O)(O)O[C@H]2[C@H](O)[C@@H](O)[C@H](O)[C@@H](CCCCCCC(=O)O1)[C@@H](O)C[C@@H](O)[C@H](/C=C/[C@@H](O)CCCCC)[C@@H](O)[C@H]2O",
        "nonacosyl hexadec-9-enoate",
        62,
    ),
    (
        "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](COC1O[C@H](CO)[C@H](O)[C@H](OS(=O)(=O)[O-])[C@H]1O)NC(=O)C(O)CCCCCCCCCCCCCCCCCCCC",
        "2-hydroxydocosanamidate",
        60,
    ),
    # Adenine nucleotide: Phase 58 allows nucleobase retained names to bypass
    # coverage gate -- "adenine" is the correct retained name for the core
    # substructure of this molecule. Removed from oversimplification test.
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCCC)COP(=O)(O)OC[C@H](N)C(=O)O",
        "(2S)-2-aminopropanoic acid",
        59,
    ),
    (
        "COC1=CC=C2[C@H]3Cc4ccc(OC)c5c4[C@@]2(C[C@@H](C2=C[C@@]4(O)[C@H]6Cc7ccc(O)c8c7[C@@]4(CCN6C)[C@@H](O8)C2=O)N3C)[C@H]1O5",
        "morphinan",
        45,
    ),
]

_IMPROVED_IDS = [
    f"{old}_{heavy}atoms"
    for _, old, heavy in IMPROVED_COMPOUNDS
]


@pytest.mark.parametrize(
    "smiles,old_name,heavy_atoms",
    IMPROVED_COMPOUNDS,
    ids=_IMPROVED_IDS,
)
def test_coverage_gate_no_bare_scaffold(smiles, old_name, heavy_atoms):
    """Verify that a previously oversimplified compound now gets a better name.

    The coverage gates should prevent returning the bare scaffold/ring name
    for large molecules. The new name should:
    1. Not be the same bare name as before
    2. Not be "unknown" (no worsening)
    3. Have adequate length for the molecule size
    """
    result = name_compound(smiles)

    # Must not worsen to unknown
    assert result is not None, f"name_compound returned None for {smiles[:40]}..."
    assert result != "unknown", (
        f"Compound previously named '{old_name}' now returns 'unknown' -- regression"
    )

    # Must not be the same bare scaffold name
    assert result != old_name, (
        f"Coverage gate did not improve naming: still returns '{old_name}' "
        f"for {heavy_atoms}-atom molecule"
    )

    # Name length should be reasonable for molecule size
    # (at minimum, longer than the bare scaffold name for large molecules)
    assert len(result) > len(old_name), (
        f"New name '{result}' ({len(result)} chars) should be longer than "
        f"old bare name '{old_name}' ({len(old_name)} chars)"
    )


# ---------------------------------------------------------------------------
# Group B: Diverse compounds that should maintain their naming
# ---------------------------------------------------------------------------
# These compounds already had adequate names. The coverage gates should NOT
# break them. Sampled from the benchmark to cover various compound classes.

DIVERSE_COMPOUNDS = [
    # Simple functional groups
    ("CCO", "ethanol"),
    ("CC(=O)O", "acetic acid"),
    ("c1ccccc1", "benzene"),
    # Steroids / large ring systems
    (
        "C=C(CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4[C@H](C)C(=O)CC[C@]4(C)C3=C[C@@H](O)[C@]12C)C(C)C",
        "(4S,5S,10S,11R,13R,14S,17R,20R)-11-hydroxyergost-7,9,24-trien-3-one",
    ),
    # Heterocycles
    ("c1ccncc1", "pyridine"),
    ("c1cc[nH]c1", "1H-pyrrole"),
    # Peptides
    (
        "CC(C)C[C@H](N)C(=O)N[C@@H](CO)C(=O)NCC(=O)O",
        "L-leucyl-L-serylglycine",
    ),
    # Acids / chains
    (
        "CCCCCCCCCCCCCCCC(=O)O",
        "hexadecanoic acid",
    ),
    # Substituted aromatics
    ("Cc1ccccc1", "toluene"),
    # Salt
    ("SS", "disulfane"),
    # Ester
    (
        "CCOC(=O)CC",
        "ethyl propanoate",
    ),
    # Amine
    (
        "CCN",
        "ethanamine",
    ),
    # Alcohol
    (
        "CCCCCCO",
        "hexan-1-ol",
    ),
    # Ketone
    (
        "CCCC(=O)CCC",
        "heptan-4-one",
    ),
    # Aldehyde
    (
        "CCCCCC=O",
        "hexanal",
    ),
    # Cycloalkane
    (
        "C1CCCCC1",
        "cyclohexane",
    ),
]

_DIVERSE_IDS = [name[:40] for _, name in DIVERSE_COMPOUNDS]


@pytest.mark.parametrize(
    "smiles,expected_name",
    DIVERSE_COMPOUNDS,
    ids=_DIVERSE_IDS,
)
def test_naming_rate_maintained(smiles, expected_name):
    """Verify that diverse well-named compounds still produce valid names.

    The coverage gates should not accidentally break naming for compounds
    that already had correct names.
    """
    result = name_compound(smiles)

    assert result is not None, f"name_compound returned None for {smiles}"
    assert result != "unknown", (
        f"Compound '{expected_name}' now returns 'unknown' -- gate regression"
    )
    # Name should be non-empty
    assert len(result) > 0, f"Empty name for {smiles}"


# ---------------------------------------------------------------------------
# Group C: Canary compound stability check
# ---------------------------------------------------------------------------
# A sampling of the 75 golden canary compounds that round-trip correctly.
# These must produce EXACTLY the expected name (not just non-unknown).

CANARY_SAMPLE = [
    (
        "COc1cc(CC(=O)C(=O)c2c(O)cc(O)c(OC)c2O)cc(OC)c1O",
        "1-(2,4,6-trihydroxy-3-methoxyphenyl)-3-(4-hydroxy-3,5-dimethoxyphenyl)propane-1,2-dione",
    ),
    (
        "Cc1ccc(C(=O)O)s1",
        "2-methylthiophene-5-carboxylic acid",
    ),
    (
        "SS",
        "disulfane",
    ),
    (
        "C[C@]12CC[C@@H](O)C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@H](O)CC[C@@H]12",
        "(3R,5R,8R,9S,10S,13S,14S,17R)-androstan-3,17-diol",
    ),
    (
        "CC(C)C[C@H](N)C(=O)N[C@@H](CO)C(=O)NCC(=O)O",
        "L-leucyl-L-serylglycine",
    ),
    (
        "C=CC(=O)CCCC",
        "hept-1-en-3-one",
    ),
    (
        "CCCCCC/C=C/C=C(\\CCCC(=O)O)[N+](=O)[O-]",
        "(5E,7E)-5-nitrotetradeca-5,7-dienoic acid",
    ),
    (
        "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CO)C(=O)O",
        "L-tryptophyl-L-tyrosyl-L-serine",
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)[C@@H](N)CCCCN)C(=O)N[C@@H](CS)C(=O)O",
        "L-lysyl-L-threonyl-L-cysteine",
    ),
    (
        "CC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)C",
        "2,6,10,14,18,22,26-heptamethylheptacosane",
    ),
    (
        "C#CCCCCCCCCCCCC(O)CC(CO)OC(C)=O",
        "2-(acetyloxy)-4-hydroxyheptadec-16-yn-1-ol",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\C[C@@H](O)CC(=O)O",
        "(3R,5Z,8Z,11Z,14Z)-3-hydroxyicosa-5,8,11,14-tetraenoic acid",
    ),
]

_CANARY_IDS = [
    name[:50].replace(" ", "_").replace(",", "")
    for _, name in CANARY_SAMPLE
]


@pytest.mark.parametrize(
    "smiles,expected_name",
    CANARY_SAMPLE,
    ids=_CANARY_IDS,
)
def test_canary_compounds_stable(smiles, expected_name):
    """Verify that golden canary compounds still produce exactly the expected name.

    These compounds round-trip correctly through OPSIN. Any change to their
    names must be investigated.
    """
    result = name_compound(smiles)
    assert result == expected_name, (
        f"Canary regression: expected '{expected_name}', got '{result}'"
    )
