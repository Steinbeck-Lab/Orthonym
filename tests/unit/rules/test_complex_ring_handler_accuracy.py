"""
Complex ring handler accuracy test suite.

Tests polycyclic aromatics, spiro systems, bicyclo systems, and fused rings
both as parent structures and as ring-as-substituent prefixes in chain-is-parent
contexts. Derived from benchmark failure analysis (diagnostics).

Key fix verified: naphthalene as ring substituent produces 'naphthalen-X-yl'
prefix, not 'cyclodecyl' (the bug was in get_ring_substituent_name not checking
retained names for multi-ring systems).
"""

import pytest
from orthonym.namer import name_compound


# ---------------------------------------------------------------------------
# 1. Polycyclic aromatics as parent structures (retained names)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected_name", [
    # Basic retained names - must be exact
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("c1ccc2cc3ccccc3cc2c1", "anthracene"),
    ("c1cc2ccc3cccc4ccc(c1)c2c34", "pyrene"),
    # F-T9/DD6: 'biphenyl' is general-only; the PIN is 1,1'-biphenyl.
    ("c1ccc(-c2ccccc2)cc1", "1,1'-biphenyl"),
])
def test_polycyclic_retained_names(smiles, expected_name):
    """Polycyclic aromatics produce correct retained / systematic PIN names."""
    result = name_compound(smiles)
    assert result == expected_name, f"Expected '{expected_name}', got '{result}'"


# ---------------------------------------------------------------------------
# 2. Substituted polycyclic aromatics (ring is parent)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected_substring, forbidden_substring", [
    # Naphthalene with OH -> naphthol or naphthalen-X-ol
    ("Oc1ccc2ccccc2c1", "naphth", "cyclodec"),
    # Naphthalene with COOH -> naphthalene-X-carboxylic acid
    ("OC(=O)c1ccc2ccccc2c1", "naphthalene", "cyclodec"),
    # Naphthalene with alkyl -> alkylnaphthalene
    ("CC(C)c1cccc2ccccc12", "naphthalene", "cyclodec"),
])
def test_substituted_polycyclic_parent(smiles, expected_substring, forbidden_substring):
    """Substituted polycyclic aromatics use retained parent names."""
    result = name_compound(smiles)
    assert expected_substring in result.lower(), \
        f"Expected '{expected_substring}' in '{result}'"
    assert forbidden_substring not in result.lower(), \
        f"Forbidden '{forbidden_substring}' found in '{result}'"


# ---------------------------------------------------------------------------
# 3. Polycyclic aromatics as ring substituents (CRITICAL: the cyclodecyl fix)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected_substring, forbidden_substring", [
    # The PRIMARY fix target: naphthyl ketone
    # Should be 1-(naphthalen-1-yl)ethan-1-one, NOT 1-cyclodecylethan-1-one
    ("CC(=O)c1cccc2ccccc12", "naphthalen", "cyclodecyl"),
    # Biphenyl as substituent on ketone chain
    ("CC(=O)c1ccc(-c2ccccc2)cc1", "biphenyl", None),
])
def test_polycyclic_ring_as_substituent(smiles, expected_substring, forbidden_substring):
    """Polycyclic aromatics as ring substituents use retained name prefixes, not cycloXyl."""
    result = name_compound(smiles)
    assert expected_substring in result.lower(), \
        f"Expected '{expected_substring}' in '{result}'"
    if forbidden_substring:
        assert forbidden_substring not in result.lower(), \
            f"Forbidden '{forbidden_substring}' found in '{result}'"


# ---------------------------------------------------------------------------
# 4. Naphthyl ketone detailed check (the primary regression target)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_naphthyl_ketone_exact():
    """1-(naphthalen-1-yl)ethan-1-one: verifies full naphthalen-yl prefix form."""
    result = name_compound("CC(=O)c1cccc2ccccc12")
    assert "naphthalen" in result, f"Expected 'naphthalen' in '{result}'"
    assert "cyclodecyl" not in result, f"Must not contain 'cyclodecyl': '{result}'"
    # Should contain a locant (1-yl or 2-yl)
    assert "-yl" in result, f"Expected '-yl' suffix in ring prefix: '{result}'"


# ---------------------------------------------------------------------------
# 5. Spiro systems
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected_name", [
    ("C1CCC2(CC1)CCCCC2", "spiro[5.5]undecane"),
    ("C1CCCC11CCCCC1", "spiro[4.5]decane"),
])
def test_spiro_systems(smiles, expected_name):
    """Spiro systems produce correct spiro[x.y]alkane names."""
    result = name_compound(smiles)
    assert result == expected_name, f"Expected '{expected_name}', got '{result}'"


# ---------------------------------------------------------------------------
# 6. Bicyclo systems
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected_name", [
    ("C1CC2CCC1CC2", "bicyclo[2.2.2]octane"),
    # PIN per R11: "RETAINED NAMES FOR VON BAEYER PARENT HYDRIDES" the Blue Book "The retained names adamantane and cubane are used in general nomenclature and as preferred IUPAC names."; "bicyclo[2.2.1]heptane (PIN)":2038; OPSIN RT exact.
    ("C1CC2CC1CC2", "bicyclo[2.2.1]heptane"),
])
def test_bicyclo_systems(smiles, expected_name):
    """Bicyclo systems produce correct bicyclo/retained names."""
    result = name_compound(smiles)
    assert result == expected_name, f"Expected '{expected_name}', got '{result}'"


# ---------------------------------------------------------------------------
# 7. Fused heterocyclic ring systems
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected_substring", [
    # Carbazole
    ("c1ccc2[nH]c3ccccc3c2c1", "carbazole"),
    # Xanthone
    ("O=c1c2ccccc2oc2ccccc12", "xanth"),
])
def test_fused_heterocyclic_retained(smiles, expected_substring):
    """Fused heterocyclic systems produce correct retained names."""
    result = name_compound(smiles)
    assert expected_substring in result.lower(), \
        f"Expected '{expected_substring}' in '{result}'"


# ---------------------------------------------------------------------------
# 8. Anthraquinone (fused ring with ketones)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.xfail(reason="Anthraquinone routed to VB handler instead of retained name")
def test_anthraquinone():
    """anthraquinone or anthracene-9,10-dione."""
    result = name_compound("O=C1c2ccccc2C(=O)c2ccccc21")
    # Should be anthraquinone or anthracene-9,10-dione
    assert "anthracene" in result.lower() or "anthraquinone" in result.lower(), \
        f"Expected anthracene/anthraquinone variant, got '{result}'"


# ---------------------------------------------------------------------------
# 9. Phenanthrene variants
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_phenanthrene_basic():
    """Phenanthrene retained name."""
    result = name_compound("c1ccc2c(c1)ccc1ccccc12")
    assert result == "phenanthrene", f"Expected 'phenanthrene', got '{result}'"


# ---------------------------------------------------------------------------
# 10. Ring-as-substituent naming function directly
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ring_substituent_name_naphthalene():
    """get_ring_substituent_name returns naphthalen-X-yl for naphthalene system."""
    from rdkit import Chem
    from orthonym.rules.ring_substituents import get_ring_substituent_name

    mol = Chem.MolFromSmiles("CC(=O)c1cccc2ccccc12")
    # Naphthalene ring atoms (indices 3-12)
    ring_atoms = tuple(range(3, 13))
    name = get_ring_substituent_name(mol, ring_atoms, attachment_point=3)
    assert "naphthalen" in name, f"Expected 'naphthalen' in '{name}'"
    assert "yl" in name, f"Expected '-yl' in '{name}'"
    assert "cyclodec" not in name, f"Must not be 'cyclodecyl': '{name}'"


@pytest.mark.unit
def test_ring_substituent_name_simple_rings():
    """get_ring_substituent_name still works for simple single rings."""
    from rdkit import Chem
    from orthonym.rules.ring_substituents import get_ring_substituent_name

    # Benzene -> phenyl
    mol = Chem.MolFromSmiles("c1ccccc1")
    ring_atoms = tuple(range(6))
    assert get_ring_substituent_name(mol, ring_atoms) == "phenyl"

    # Cyclohexane -> cyclohexyl
    mol = Chem.MolFromSmiles("C1CCCCC1")
    ring_atoms = tuple(range(6))
    assert get_ring_substituent_name(mol, ring_atoms) == "cyclohexyl"


# ---------------------------------------------------------------------------
# 11. Complex ring compounds from benchmark failures
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles, must_not_contain", [
    # Each compound that was producing cyclodecyl/cyclotetra... etc.
    # must NOT produce cyclo{large number} names
    ("CC(=O)c1cccc2ccccc12", "cyclodecyl"),         # naphthalenyl, not cyclodecyl
    ("CCc1ccc2ccccc2c1", "cyclodecyl"),              # ethylnaphthalene, not cyclodecyl
    ("Clc1ccc2ccccc2c1", "cyclodecyl"),              # chloronaphthalene, not cyclodecyl
])
def test_no_spurious_cyclo_names(smiles, must_not_contain):
    """Ring-as-substituent naming must not produce spurious cyclo-names for polycyclics."""
    result = name_compound(smiles)
    assert must_not_contain not in result.lower(), \
        f"Spurious '{must_not_contain}' found in '{result}'"


# ---------------------------------------------------------------------------
# 12. End-to-end naming accuracy for additional complex ring compounds
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected_fragment", [
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("c1ccc2cc3ccccc3cc2c1", "anthracene"),
    ("c1cc2ccc3cccc4ccc(c1)c2c34", "pyrene"),
    ("c1ccc(-c2ccccc2)cc1", "biphenyl"),
    ("C1CC2CCC1CC2", "bicyclo"),
    ("C1CC2CC1CC2", "bicyclo[2.2.1]heptane"),  # R11::9881, no retained PIN
    ("C1CCC2(CC1)CCCCC2", "spiro"),
    ("c1ccc2[nH]c3ccccc3c2c1", "carbazole"),
    ("O=c1c2ccccc2oc2ccccc12", "xanth"),
    ("Oc1ccc2ccccc2c1", "naphth"),
    ("OC(=O)c1ccc2ccccc2c1", "naphthalene"),
    ("CC(=O)c1cccc2ccccc12", "naphthalen"),
])
def test_complex_ring_expected_fragments(smiles, expected_fragment):
    """Complex ring compounds contain expected name fragments."""
    result = name_compound(smiles)
    assert expected_fragment in result.lower(), \
        f"Expected '{expected_fragment}' in '{result}' for SMILES '{smiles}'"
