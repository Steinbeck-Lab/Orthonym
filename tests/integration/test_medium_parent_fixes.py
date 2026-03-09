"""
Batch regression tests for medium molecule (21-40 HA) parent selection fixes.

Phase 66 Plans 01-02: Tests grouped by compound class to verify parent selection
improvements for medium-sized molecules. Each test verifies that the generated
name contains expected structural features (substring matching for robustness).

Test groups (Plan 01):
  1. Steroid parent selection (NP scaffold + methyl/halogen decoration)
  2. Fused heterocycle with chain substituents
  3. Polycyclic VB naming
  4. Polycyclic aromatic routing
  5. Charged species / salt naming
  6. Alkaloid parent selection

Test groups (Plan 02):
  7. VB polycyclic format verification (VB names are complete)
  8. Macrocyclic compound naming
  9. STER-04 small-molecule stereo compounds (baseline documentation)
  10. Steroid decoration completeness
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
        "pentamethyl",  # P86: now also finds 6-oxo, so substring can't span methyl→oxacyclo
        id="macrolide-pentamethyl-oxacyclohexadecanone",
    ),
    pytest.param(
        "C[C@@H]1CC(=O)O[C@@H](C)[C@H](O)/C=C\\C(=O)"
        "O[C@@H](C)C/C=C\\C(=O)O1",
        "trimethyl",  # P86: now also finds 6,12-dioxo, so substring can't span methyl→oxacyclo
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
        "oxolane",
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
        "piperidine",  # depth-independent naming v11: now sees piperidine parent
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


# ===========================================================================
# Plan 02 Groups (Phase 66-02)
# ===========================================================================

# ---------------------------------------------------------------------------
# Group 7: VB polycyclic format verification
# Verifies that VB-named polycyclic compounds produce complete structural
# names with VB descriptors (these are correct names but OPSIN interprets
# differently -- classified as unfixable_vb_interpretation)
# ---------------------------------------------------------------------------
VB_FORMAT_VERIFICATION = [
    pytest.param(
        "O=C(O)c1cc2cc3c4c(c2oc1=O)CCCN4CCC3",
        "tetracyclo",
        id="vb-aza-tetracyclic-acid",
    ),
    pytest.param(
        "COc1cccc2c1C(=O)c1ccc3c(c1C2=O)C(=O)C[C@@H](C)[C@H]3O",
        "tetracyclo",
        id="vb-methoxy-tetracyclic-trione",
    ),
    pytest.param(
        "COc1cc(O)c2c(c1O)C(=O)c1c(C(C)=O)c(O)cc(O)c1C2=O",
        "tricyclo",
        id="vb-polyhydroxy-tricyclic-dione",
    ),
    pytest.param(
        "COC(=O)[C@@H]1CC23CCCN4CC[C@@]5(c6ccccc6N(C)"
        "C15CC2)[C@@H]3[C@@H]4O",
        "hexacyclo",
        id="vb-hexacyclic-diaza-alkaloid",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", VB_FORMAT_VERIFICATION)
def test_vb_format_completeness(smiles, expected_substr):
    """VB-named compounds must produce names containing VB ring descriptors."""
    name = name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 8: Macrocyclic compound naming
# Verifies that macrocyclic lactones/lactams produce correct ring-size
# prefix and substituent enumeration
# ---------------------------------------------------------------------------
MACROCYCLIC_FIXES = [
    pytest.param(
        "C/C1=C/C[C@@H](/C(C)=C/c2csc(C)n2)OC(=O)C[C@H](O)"
        "C(C)(C)C(=O)[C@H](C)[C@@H](O)/C(C)=C/CC1",
        "oxacyclohexadecan",
        id="macrolide-16-ring-oxa",
    ),
    pytest.param(
        "C[C@@H]1CC(=O)O[C@@H](C)[C@H](O)/C=C\\C(=O)"
        "O[C@@H](C)C/C=C\\C(=O)O1",
        "oxacyclohexadecan",
        id="macrolide-trilactone-16ring",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", MACROCYCLIC_FIXES)
def test_macrocyclic_naming(smiles, expected_substr):
    """Macrocyclic compounds must include correct ring-size prefix."""
    name = name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 9: STER-04 small-molecule wrong-parent stereo baseline
# Documents current naming status of 14 small stereo compounds from Phase 63.
# All are blocked by wrong parent selection, not stereo labeling errors.
# Tests verify the name is non-None (structural description produced).
# ---------------------------------------------------------------------------
SMALL_STEREO_PARENT_BASELINE = [
    pytest.param(
        "C=C(C)[C@@H]1CC[C@@H](C)[C@@]12CC=C(C)CC2",
        "spiro",
        id="ster04-spiro-isopropenyl-cyclohexene",
    ),
    pytest.param(
        "CC(=O)[C@@]1(C)C(C)=C[C@H](O)[C@H]2C[C@](C)(O)CC[C@@H]21",
        "cyclodecan",
        id="ster04-decalin-ketone",
    ),
    pytest.param(
        "CC1=C[C@]2(CC1=O)[C@H](C)CC[C@@H](C(C)(C)O)[C@H]2O",
        "spiro",
        id="ster04-spirocyclopentanone",
    ),
    pytest.param(
        "CC(C)=CCc1ccc(O)c2c1[C@H](CC(=O)O)OC2=O",
        "acid",
        id="ster04-isocoumarinone-acid",
    ),
    pytest.param(
        "CCCCCCC(=O)NC1=CC(=O)[C@@H]2CCCN12",
        "amide",
        id="ster04-pyrrolizinone-amide",
    ),
    pytest.param(
        "C/C=C/C=C/C(=O)C1=C(O)C(=C(C)C)NC1=O",
        "oxo",
        id="ster04-dienoyl-pyrrole",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", SMALL_STEREO_PARENT_BASELINE)
def test_small_stereo_parent_baseline(smiles, expected_substr):
    """STER-04 small stereo compounds must produce non-None names with structural content."""
    name = name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 10: Steroid decoration completeness
# Verifies that steroid NP names include hydroxy, ketone, and unsaturation
# decorations (these steroids already have reasonable names but may be
# missing methyls or other substituents)
# ---------------------------------------------------------------------------
STEROID_DECORATION_COMPLETENESS = [
    pytest.param(
        "CCC(CCC(C)C1CCC2C3C(O)C=C4CC(O)CCC4(C)C3CCC12C)C(C)C",
        "stigmast",
        id="stigmastane-diol-retained-name",
    ),
    pytest.param(
        "C=C(C)C(C)CCC(C)C1CCC2C3=CCC4CC(O)CCC4(C)C3CCC21C",
        "ergost",
        id="ergostane-dienol-retained-name",
    ),
    pytest.param(
        "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1"
        "[C@@H](O)[C@@H](O)[C@@H]2O",
        "estran",
        id="estrane-tetraol-decoration",
    ),
    pytest.param(
        "CC(CCCC(C)(O)COS(=O)(=O)O)[C@H]1CC[C@H]2[C@@H]3"
        "[C@H](O)C[C@@H]4C[C@H](O)CCC4(C)[C@H]3C[C@H](O)C12C",
        "cholestan",
        id="cholestane-tetraol-sulfonate",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", STEROID_DECORATION_COMPLETENESS)
def test_steroid_decoration_completeness(smiles, expected_substr):
    """Steroid NP names must include correct scaffold stem."""
    name = name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"
