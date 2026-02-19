"""
Batch regression tests for medium molecule (21-40 HA) parent selection fixes.

Phase 66 Plan 01: Tests grouped by compound class to verify parent selection
improvements for medium-sized molecules. Each test verifies that the generated
name contains expected structural features (substring matching for robustness).

Test groups:
  1. Steroid parent selection (NP scaffold + methyl/halogen decoration)
  2. Fused heterocycle with chain substituents
  3. Polycyclic VB naming
  4. Polycyclic aromatic routing
  5. Charged species / salt naming
  6. Alkaloid parent selection
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Group 1: Steroid parent selection
# Verifies that steroid NP scaffolds are detected and decorated with
# explicit methyl groups (IUPAC P-31 retained names with substitution)
# ---------------------------------------------------------------------------
STEROID_PARENT_FIXES = [
    pytest.param(
        "C=C(CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)"
        "[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1CC3)C(C)C",
        "trimethylergost",
        id="ergost-dien-ol-trimethyl",
    ),
    pytest.param(
        "CC(CC(=O)CC(C)C1C[C@H](O)[C@@]2(C)C3=C(C(=O)CC12C)"
        "C1(C)CC[C@H](O)C(C)(C)C1C[C@@H]3O)C(=O)O",
        "trimethylcholest",
        id="cholest-en-trione-trimethyl",
    ),
    pytest.param(
        "CC(=O)O[C@H]1CC[C@]2(C)C3=C(CC[C@H]2C1(C)C)[C@]1(C)"
        "C[C@@H](O)[C@H]([C@@H](C/C=C/C(C)(C)O)C(=O)O)"
        "[C@@]1(C)CC3",
        "trimethyl",
        id="cholest-dien-yl-acetate-trimethyl",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", STEROID_PARENT_FIXES)
def test_steroid_parent_selection(smiles, expected_substr):
    """Steroid NP scaffolds must include explicit methyl group decoration."""
    name = name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 2: Fused heterocycle with chain substituents
# Verifies that ring systems are preferred as parent over chains, and
# that chain substituents/stereo are enumerated correctly
# ---------------------------------------------------------------------------
FUSED_HETERO_CHAIN_FIXES = [
    pytest.param(
        "C/C1=C/C[C@@H](/C(C)=C/c2csc(C)n2)OC(=O)C[C@H](O)"
        "C(C)(C)C(=O)[C@H](C)[C@@H](O)/C(C)=C/CC1",
        "pentamethyloxacyclohexadecan",
        id="macrolide-pentamethyl-oxacyclohexadecanone",
    ),
    pytest.param(
        "C[C@@H]1CC(=O)O[C@@H](C)[C@H](O)/C=C\\C(=O)"
        "O[C@@H](C)C/C=C\\C(=O)O1",
        "trimethyloxacyclohexadecan",
        id="macrolide-trimethyl-oxacyclohexadecanone",
    ),
    pytest.param(
        "C/C(=C\\CC/C(C)=C/C/C=C(/CC(=O)c1cc(O)ccc1O)C(=O)O)CO",
        "trienoic acid",
        id="prenylated-phenol-trienoic-acid",
    ),
    pytest.param(
        "CC(C)=CCC/C(C)=C/COC[C@H]1O[C@@H]"
        "(N2CCC(=O)NC2=O)[C@H](O)[C@@H]1O",
        "tetrahydrofuran",
        id="geranyl-nucleoside-thf-parent",
    ),
    pytest.param(
        "COC(=O)/C(CC(=O)O)=C(\\CCCCCCCCCCCCCCCCC1=C(C)"
        "C(=O)OC1=O)C(=O)O",
        "acid",
        id="long-chain-dicarboxylic-acid",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", FUSED_HETERO_CHAIN_FIXES)
def test_fused_heterocycle_chain_parent(smiles, expected_substr):
    """Fused heterocycles with chains must select ring as parent."""
    name = name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 3: Polycyclic VB naming
# Verifies that polycyclic systems get appropriate VB or retained names
# instead of being reduced to small fragment names
# ---------------------------------------------------------------------------
POLYCYCLIC_VB_FIXES = [
    pytest.param(
        "Oc1ccc2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@@H](O)[C@@H]2O",
        "hydroxychromane",
        id="catechin-trihydroxychromane",
    ),
    pytest.param(
        "O=C1c2c(O)cc(O)cc2O[C@@H](c2ccc(O)c(O)c2)"
        "[C@@H]1O[C@@H]1OC[C@@H](O)[C@H](O)[C@H]1O",
        "chroman",
        id="xylopyranoside-trihydroxychromanone",
    ),
    pytest.param(
        "C=C1NC(=O)[C@H]([C@@H](C)[C@]2(O)C(=O)N(C)"
        "c3ccccc32)NC1=O",
        "indolin",
        id="indolinone-derivative",
    ),
    pytest.param(
        "CC(C)=CCC/C(C)=C/CC[C@]1(C)Cc2c(c(O)cc3c2"
        "CN([C@H]2CCCNC2=O)C3=O)C[C@@H]1O",
        "tricyclo",
        id="prenylated-tetracyclic-vb",
    ),
    pytest.param(
        "O=C1N[C@@H](C[C@@]2(O)c3ccccc3N3C(=O)[C@@H]4"
        "CCCCN4[C@@H]32)C(=O)N[C@H]1Cc1ccccc1",
        "tetracyclo",
        id="peptide-tetracyclic-vb",
    ),
    pytest.param(
        "C=C1CC[C@@H](C/C=C2/CC[C@]3(OC2)O[C@@]2(O)"
        "CC[C@]3(C)OC2(C)C)C(C)(C)[C@H]1[C@@H](O)"
        "C=C1CCOC1=O",
        "trioxa",
        id="trioxa-tricyclic-terpene",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", POLYCYCLIC_VB_FIXES)
def test_polycyclic_vb_naming(smiles, expected_substr):
    """Polycyclic compounds must get VB or retained ring names, not fragments."""
    name = name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 4: Polycyclic aromatic routing
# Verifies that polycyclic aromatic systems are correctly identified
# ---------------------------------------------------------------------------
POLYCYCLIC_AROMATIC_FIXES = [
    pytest.param(
        "O=C1NC(Cc2c[nH]c3ccccc23)C(=O)N/C1=C/c1cnc[nH]1",
        "piperazine",
        id="indole-imidazole-piperazinedione",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", POLYCYCLIC_AROMATIC_FIXES)
def test_polycyclic_aromatic_routing(smiles, expected_substr):
    """Polycyclic aromatic compounds must route to correct naming path."""
    name = name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 5: Charged species / salt naming
# Verifies that charged compounds produce meaningful structural names
# instead of generic "ammonium <acid>" fragments
# ---------------------------------------------------------------------------
CHARGED_SPECIES_FIXES = [
    pytest.param(
        "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])"
        "OCC[N+](C)(C)C)OC(C)=O",
        "palmitate",
        id="phospholipid-palmitate-ester",
    ),
    pytest.param(
        "COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1.[Cl-].[H+]",
        "hydrochloride",
        id="methoxyphenyl-amine-hydrochloride",
    ),
    pytest.param(
        "CCCCCCCCCCCCCC(O)CC(=O)OC(CC(=O)[O-])"
        "C[N+](C)(C)C",
        "acid",
        id="carnitine-palmitoyl-acid",
    ),
    pytest.param(
        "CC(C)[C@@]1(C)N=C(c2nc3ccccc3cc2C(=O)[O-])"
        "NC1=O.[NH4+]",
        "quinoline",
        id="ammonium-quinoline-carboxylate",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", CHARGED_SPECIES_FIXES)
def test_charged_species_naming(smiles, expected_substr):
    """Charged species must produce structural names, not generic salts."""
    name = name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 6: Alkaloid parent selection
# Verifies that alkaloid scaffolds are recognized with correct naming
# ---------------------------------------------------------------------------
ALKALOID_FIXES = [
    pytest.param(
        "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2.Cl",
        "indole",
        id="tropane-indole-ester",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", ALKALOID_FIXES)
def test_alkaloid_parent_selection(smiles, expected_substr):
    """Alkaloid compounds must identify correct scaffold parent."""
    name = name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"
