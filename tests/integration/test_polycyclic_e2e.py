"""
End-to-end integration tests for a phase polycyclic naming.

Tests validate:
- Von Baeyer nomenclature (tricyclo, tetracyclo, pentacyclo+)
- Heteroatom replacement nomenclature (oxa-, aza-, thia-)
- Polycyclic lactone naming
- Bridged fused  nomenclature
- Stereodescriptors on polycyclic frameworks
- Regression testing for existing naming (bicyclo, fused, simple cycles)

Uses name_compound as the single entry point, testing the full pipeline
from SMILES to IUPAC name.
"""
import pytest
from orthonym.namer import name_compound


class TestVonBaeyerE2E:
    """Full pipeline tests for Von Baeyer nomenclature (SMILES -> name_compound)."""

    def test_norbornane_still_works(self):
        """Norbornane: existing bicyclo naming preserved."""
        # Norbornane = bicyclo[2.2.1]heptane
        result = name_compound('C1CC2CCC1C2')
        # Should be either retained name or systematic
        assert 'norbornane' in result.lower() or 'bicyclo[2.2.1]heptane' in result.lower(), \
            f"Expected norbornane or bicyclo[2.2.1]heptane, got: {result}"

    def test_adamantane_retained_name(self):
        """Adamantane: retained name per IUPAC."""
        result = name_compound('C1C2CC3CC1CC(C2)C3')
        assert result == 'adamantane', f"Expected adamantane, got: {result}"

    def test_bicyclo_222_octane(self):
        """Bicyclo[2.2.2]octane: existing naming preserved."""
        result = name_compound('C1CC2CCC1CC2')
        assert 'bicyclo' in result.lower(), f"Expected bicyclo, got: {result}"
        assert '2.2.2' in result, f"Expected [2.2.2], got: {result}"
        assert 'octane' in result.lower(), f"Expected octane, got: {result}"

    def test_norbornene_existing_naming(self):
        """Norbornene: existing unsaturation naming preserved."""
        result = name_compound('C1=CC2CCC1C2')
        assert 'ene' in result.lower() or 'norbornene' in result.lower(), \
            f"Expected -ene or norbornene, got: {result}"

    def test_cubane_retained_name(self):
        """Cubane: retained name AND PIN per IUPAC (Blue Book 9881)."""
        # Cubane = pentacyclo[4.2.0.0(2,5).0(3,8).0(4,7)]octane, retained name preferred.
        # a phase: the prior SMILES C12C3C4C1C5C3C4C25 was the WRONG (CH)8 cage
        # isomer (InChIKey BOLISNSTKUABPW); the true cubane canonical is below
        # (TXWRERCHRDBNLG), matching what OPSIN emits for 'cubane'.
        result = name_compound('C12C3C4C1C1C2C3C41')
        assert result == 'cubane', f"Expected cubane, got: {result}"


class TestSubstitutedPolycyclicE2E:
    """Tests for polycyclic systems with substituents."""

    def test_methyladamantane(self):
        """1-methyladamantane: substituent on tricyclo system."""
        # 1-methyladamantane
        result = name_compound('CC1(C2CC3CC(C2)CC1C3)C')
        # Should have methyl prefix and tricyclo descriptor
        assert 'methyl' in result.lower() or 'tricyclo' in result.lower(), \
            f"Expected methyl or tricyclo, got: {result}"

    def test_methylnorbornane(self):
        """Methylnorbornane: existing substituent naming preserved."""
        result = name_compound('CC1CC2CCC1C2')
        assert 'methyl' in result.lower(), f"Expected methyl, got: {result}"

    def test_dimethylnorbornane(self):
        """Dimethylnorbornane: multiple substituents."""
        result = name_compound('CC1CC2CCC1C2C')
        assert 'methyl' in result.lower(), f"Expected methyl, got: {result}"


class TestHeteroatomReplacementE2E:
    """Tests for "a" nomenclature (oxa-, aza-, thia-)."""

    def test_7_oxabicyclo_221_heptane(self):
        """7-oxabicyclo[2.2.1]heptane: oxygen in norbornane framework.

        Task 12 fix a performance pass (wp6-tests): the non-strict xfail marker ('Bicyclo
        heteroatom naming uses bicyclo module') was stale -- the test XPASSed. It
        now asserts the name: the von Baeyer numbering is fixed by the hydrocarbon
        and the heteroatom takes the one-atom bridge, 7, the Blue Book
        "Numbering is determined first by the fixed numbering of the hydrocarbon
        system"). OPSIN 2.9.0 full-InChIKey exact."""
        result = name_compound('C1CC2CCC1O2')
        assert result == '7-oxabicyclo[2.2.1]heptane', result

    def test_azabicyclo_system_recognized(self):
        """Azabicyclo system: at minimum recognized as bicyclo."""
        # 2-azabicyclo[2.2.1]heptane (nitrogen in ring)
        result = name_compound('C1CC2NCC1C2')
        # At minimum should produce bicyclo (full aza- support may come later)
        assert 'bicyclo' in result.lower() or 'aza' in result.lower() or 'azabicyclo' in result.lower(), \
            f"Expected bicyclo or aza, got: {result}"

    def test_oxabicyclo_with_unsaturation(self):
        """Oxa-bicyclo with unsaturation: bicyclo framework recognized."""
        # Oxygen in bicyclo with double bond
        result = name_compound('C1=CC2CCC1O2')
        # Should at minimum recognize bicyclo framework
        assert 'bicyclo' in result.lower() or 'ene' in result.lower(), \
            f"Expected bicyclo or ene, got: {result}"

    def test_tricyclo_with_oxygen_placeholder(self):
        """Tricyclo with heteroatom: tests polycyclic oxa- support."""
        # This would route through polycyclic.py which has oxa- support
        # For now, just verify tricyclo systems work
        result = name_compound('C1C2CC3CC1CC(C2)C3')  # Adamantane
        assert result == 'adamantane', f"Expected adamantane, got: {result}"


class TestPolycyclicLactoneE2E:
    """Tests for polycyclic lactone naming (ring ester -> oxa + one)."""

    def test_bicyclic_lactone_full(self):
        """Bicyclic lactone: ring O as an 'oxa' prefix, the C=O as the '-one' suffix.

        Task 12 fix a performance pass (wp6-tests): the non-strict xfail marker was stale (the
        test XPASSed). It now asserts the name: heteroatoms take the lowest
        locants the fixed numbering allows, the Blue Book "Low
        locants are assigned to the heteroatoms considered together as a set"; cf.
        '2-oxabicyclo[2.2.1]hept-5-ene (PIN)':16705), so O is 2 and the suffix 3.
        OPSIN 2.9.0 full-InChIKey exact."""
        result = name_compound('O=C1OC2CCC1CC2')
        assert result == '2-oxabicyclo[2.2.2]octan-3-one', result

    def test_bicyclic_lactone_recognized_as_bicyclo(self):
        """Bicyclic lactone: at minimum recognized as bicyclo framework."""
        result = name_compound('O=C1OC2CCC1CC2')
        # Should at minimum recognize bicyclo framework
        assert 'bicyclo' in result.lower(), f"Expected bicyclo, got: {result}"

    def test_simple_lactone_preserved(self):
        """Simple monocyclic lactone: gamma-butyrolactone."""
        # gamma-butyrolactone = oxolan-2-one
        result = name_compound('O=C1CCCO1')
        # Should be named as lactone (oxolan-2-one pattern)
        assert 'one' in result.lower() or 'lactone' in result.lower() or 'oxol' in result.lower(), \
            f"Expected lactone naming, got: {result}"


class TestBridgedFusedE2E:
    """Tests for bridged fused nomenclature."""

    def test_methanonaphthalene(self):
        """1,4-methanonaphthalene: bridged fused AROMATIC system.

         Phase G0 (DD7 S1): this molecule (10 aromatic atoms) was previously
        named as a von-Baeyer `tricyclo[…]` cage that DROPS the benzo
        aromaticity — a structurally WRONG name (it re-parses to a different,
        over-saturated molecule). The old assertion ACCEPTED that wrong 'cyclo'
        name as "fused recognition". G0 now fails closed (von Baeyer cannot
        represent aromaticity); the correct bridged-fused PIN
        (1,4-dihydro-1,4-methanonaphthalene, is a Phase-G1 build.
        Acceptable outcomes: a fail-closed refusal, OR a real fused-system
        name (when G1 lands) — but NEVER the de-aromatised von-Baeyer cage.
        """
        from orthonym.errors import is_failure_name
        result = name_compound('C12=CC=CC3=C1C=CC=C3C2')
        if is_failure_name(result):
            # G0 fail-closed: correct (refuses rather than de-aromatising).
            return
        # If a name IS produced (post-G1), it must recognise the fused aromatic
        # parent and must NOT be a de-aromatised von-Baeyer cage.
        assert any(x in result.lower() for x in ['naphthalene', 'fused', 'anthracene']), \
            f"Expected fused-aromatic recognition (or G0 refusal), got: {result}"
        assert 'cyclo[' not in result.lower(), \
            f"von-Baeyer cage drops aromaticity (G0 must refuse this), got: {result}"

    def test_pure_naphthalene_not_affected(self):
        """Naphthalene: pure fused not affected by bridged-fused routing."""
        result = name_compound('c1ccc2ccccc2c1')
        assert result.lower() == 'naphthalene', f"Expected naphthalene, got: {result}"

    def test_pure_anthracene_not_affected(self):
        """Anthracene: pure fused not affected."""
        result = name_compound('c1ccc2cc3ccccc3cc2c1')
        assert result.lower() == 'anthracene', f"Expected anthracene, got: {result}"


class TestPolycyclicStereoE2E:
    """Tests for stereodescriptors on polycyclic frameworks."""

    def test_stereo_bicyclo_standard_smiles(self):
        """Bicyclo with stereo: use standard SMILES format."""
        # Use non-stereo SMILES first to verify bicyclo detection
        result = name_compound('C1CC2CCC1C2')
        assert 'bicyclo' in result.lower() or 'norbornane' in result.lower(), \
            f"Expected bicyclo/norbornane, got: {result}"

    def test_norbornane_basic(self):
        """Norbornane: basic bicyclo detection."""
        result = name_compound('C1CC2CCC1C2')
        assert 'bicyclo' in result.lower() or 'norbornane' in result.lower(), \
            f"Expected bicyclo/norbornane, got: {result}"

    def test_stereo_on_substituted_bicyclo(self):
        """Substituted bicyclo with stereo SMILES."""
        # Methylnorbornane - substituent may trigger stereo collection
        result = name_compound('CC1CC2CCC1C2')
        assert 'methyl' in result.lower() or 'bicyclo' in result.lower(), \
            f"Expected methyl/bicyclo, got: {result}"


class TestRegressionE2E:
    """Critical regression tests - ensure existing naming unaffected."""

    def test_simple_benzene(self):
        """Benzene: NOT captured by polycyclic routing."""
        result = name_compound('c1ccccc1')
        assert result.lower() == 'benzene', f"Expected benzene, got: {result}"

    def test_simple_cyclohexane(self):
        """Cyclohexane: NOT captured by polycyclic routing."""
        result = name_compound('C1CCCCC1')
        assert result.lower() == 'cyclohexane', f"Expected cyclohexane, got: {result}"

    def test_existing_bicyclo_norbornane(self):
        """Norbornane: existing bicyclo routing preserved."""
        result = name_compound('C1CC2CCC1C2')
        assert 'norbornane' in result.lower() or 'bicyclo[2.2.1]heptane' in result.lower(), \
            f"Expected norbornane/bicyclo, got: {result}"

    def test_existing_spiro_system(self):
        """Spiro system: routing preserved."""
        result = name_compound('C1CCC2(CC1)CCCCC2')
        assert 'spiro' in result.lower(), f"Expected spiro, got: {result}"

    def test_existing_fused_indole(self):
        """Indole: fused heterocycle routing preserved."""
        result = name_compound('c1ccc2[nH]ccc2c1')
        assert 'indole' in result.lower() or 'fused' in result.lower(), \
            f"Expected indole, got: {result}"

    def test_existing_fused_quinoline(self):
        """Quinoline: fused heterocycle routing preserved."""
        result = name_compound('c1ccc2ncccc2c1')
        assert 'quinoline' in result.lower(), f"Expected quinoline, got: {result}"

    def test_simple_chain_ethanol(self):
        """Ethanol: simple chain unaffected."""
        result = name_compound('CCO')
        assert result.lower() == 'ethanol', f"Expected ethanol, got: {result}"

    def test_simple_chain_butanoic_acid(self):
        """Butanoic acid: simple chain unaffected."""
        result = name_compound('CCCC(=O)O')
        assert 'butanoic acid' in result.lower() or 'butyric acid' in result.lower(), \
            f"Expected butanoic acid, got: {result}"

    def test_natural_product_cholesterol(self):
        """Cholesterol: natural product routing unaffected."""
        # Simplified cholesterol check
        result = name_compound('CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C')
        # Should have some recognizable name or steroid pattern
        assert result is not None, "Expected non-None result for cholesterol"

    def test_tetrahydronaphthalene(self):
        """Tetrahydronaphthalene: partial saturation preserved."""
        result = name_compound('c1ccc2c(c1)CCCC2')
        assert 'naphthalene' in result.lower() or 'tetrahydro' in result.lower() or 'tetralin' in result.lower(), \
            f"Expected naphthalene derivative, got: {result}"


class TestSuccessCriteria:
    """Explicit tests for a phase success criteria from PLAN.md."""

    def test_adamantane_produces_retained_name(self):
        """SUCCESS-1: Adamantane SMILES produces retained name 'adamantane'."""
        result = name_compound('C1C2CC3CC1CC(C2)C3')
        assert result == 'adamantane', f"Criterion 1 FAIL: expected 'adamantane', got {result}"

    def test_vb_verification_formula(self):
        """SUCCESS-6: VB verification formula passes for adamantane.

        VB descriptor is still correct internally (tricyclo[3.3.1.1(3,7)]decane),
        but name_compound returns the retained name 'adamantane'. Verify via
        low-level generate_polycyclic_name instead.
        """
        from rdkit import Chem
        from orthonym.rules.polycyclic import generate_polycyclic_name
        mol = Chem.MolFromSmiles('C1C2CC3CC1CC(C2)C3')
        base_name = generate_polycyclic_name(mol)
        assert 'tricyclo[3.3.1.1' in base_name, f"Criterion 6 FAIL: {base_name}"

    def test_existing_bicyclo_routing_untouched(self):
        """SUCCESS-7: Existing bicyclo, spiro, fused routing untouched."""
        # Test each existing route
        bicyclo_result = name_compound('C1CC2CCC1C2')  # norbornane
        spiro_result = name_compound('C1CCC2(CC1)CCCCC2')  # spiro compound
        fused_result = name_compound('c1ccc2[nH]ccc2c1')  # indole

        assert 'norbornane' in bicyclo_result.lower() or 'bicyclo' in bicyclo_result.lower()
        assert 'spiro' in spiro_result.lower()
        assert 'indole' in fused_result.lower()


class TestDescriptorFormats:
    """Tests for correct VB descriptor formatting."""

    def test_bicyclo_descriptor_format(self):
        """Bicyclo[x.y.z] format preserved."""
        result = name_compound('C1CC2CCC1CC2')  # bicyclo[2.2.2]octane
        assert 'bicyclo[2.2.2]' in result.lower(), f"Expected bicyclo[2.2.2], got: {result}"

    def test_tricyclo_secondary_bridge_locants(self):
        """Tricyclo with superscript locants for the secondary bridge.

        Adamantane now returns retained name; test VB format via generate_polycyclic_name.
        Task 12 fix a performance pass (wp6-tests), change-asserted-value (was the
        parenthesized '1(3,7)'): (the Blue Book) cites the
        attachment locants "as a pair of superscript arabic numbers (lower number is
        cited first) separated by a comma", written '1^3,7' in plain text. OPSIN
        2.9.0 parses 'tricyclo[3.3.1.1^3,7]decane' to an InChIKey
        (adamantane), full InChIKey exact.
        """
        from rdkit import Chem
        from orthonym.rules.polycyclic import generate_polycyclic_name
        mol = Chem.MolFromSmiles('C1C2CC3CC1CC(C2)C3')  # adamantane
        base_name = generate_polycyclic_name(mol)
        assert 'tricyclo[3.3.1.1^3,7]' in base_name, base_name


class TestEdgeCases:
    """Edge cases for polycyclic routing."""

    def test_invalid_smiles_raises_error(self):
        """Invalid SMILES raises ValueError."""
        # name_compound raises ValueError for invalid SMILES
        with pytest.raises(ValueError, match="Invalid SMILES"):
            name_compound('invalid_smiles_xyz')

    def test_empty_string_returns_unknown(self):
        """Empty SMILES returns 'unknown'."""
        result = name_compound('')
        assert result == 'unknown', f"Expected 'unknown', got: {result}"

    def test_single_atom_not_polycyclic(self):
        """Single atom: not routed to polycyclic."""
        result = name_compound('C')
        assert result.lower() == 'methane', f"Expected methane, got: {result}"

    def test_simple_chain_not_polycyclic(self):
        """Simple chain: not captured by polycyclic."""
        result = name_compound('CCCC')
        assert result.lower() == 'butane', f"Expected butane, got: {result}"


class TestPeryleneNaming:
    """Tests for perylene retained name lookup (Plan 16-07 gap closure)."""

    def test_perylene_returns_retained_name(self):
        """Perylene SMILES returns 'perylene' via retained name lookup."""
        result = name_compound('c1cc2cccc3c4cccc5cccc(c(c1)c23)c54')
        assert result.lower() == 'perylene', f"Expected perylene, got: {result}"

    def test_pyrene_still_works(self):
        """Pyrene retained name not broken by perylene addition."""
        result = name_compound('c1cc2ccc3cccc4ccc(c1)c2c34')
        assert result.lower() == 'pyrene', f"Expected pyrene, got: {result}"

    def test_coronene_still_works(self):
        """Coronene retained name not broken by perylene addition."""
        result = name_compound('c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67')
        assert result.lower() == 'coronene', f"Expected coronene, got: {result}"

    def test_naphthalene_still_works(self):
        """Naphthalene retained name not broken by perylene addition."""
        result = name_compound('c1ccc2ccccc2c1')
        assert result.lower() == 'naphthalene', f"Expected naphthalene, got: {result}"


class TestClassificationFixes:
    """Regression tests for ring system classification fix (Plan 16-06).

    Ensures pentacyclic/tricyclo+ systems are not misclassified as bicyclo,
    while preserving correct bicyclo classification for true bicyclic systems.
    """

    def test_pentacyclic_not_classified_as_bicyclo(self):
        """Pentacyclic lactone must NOT be classified as bicyclo."""
        from rdkit import Chem
        from orthonym.rules.bicyclo import is_bicyclo_system
        mol = Chem.MolFromSmiles(
            'C=C1C[C@]23C[C@@]1(O)CC[C@H]2[C@@]12CC[C@H](O)'
            '[C@@](C)(C(=O)O1)[C@H]2[C@@H]3C(=O)O'
        )
        assert mol is not None, "Failed to parse pentacyclic SMILES"
        assert is_bicyclo_system(mol) is False, \
            "Pentacyclic lactone must NOT be classified as bicyclo"

    def test_cubane_not_classified_as_bicyclo(self):
        """Cubane must NOT be classified as bicyclo."""
        from rdkit import Chem
        from orthonym.rules.bicyclo import is_bicyclo_system
        mol = Chem.MolFromSmiles('C12C3C4C1C5C3C4C25')
        assert mol is not None, "Failed to parse cubane SMILES"
        assert is_bicyclo_system(mol) is False, \
            "Cubane (pentacyclo) must NOT be classified as bicyclo"

    def test_norbornane_still_bicyclo(self):
        """Norbornane must still be classified as bicyclo (regression guard)."""
        from rdkit import Chem
        from orthonym.rules.bicyclo import is_bicyclo_system
        mol = Chem.MolFromSmiles('C1CC2CC1CC2')
        assert mol is not None, "Failed to parse norbornane SMILES"
        assert is_bicyclo_system(mol) is True, \
            "Norbornane must still be classified as bicyclo"

    def test_camphor_still_bicyclo(self):
        """Camphor is named on its bicyclo skeleton (regression guard).

        Task 12 fix a performance pass (wp6-tests), change-asserted-value (was the trivial
        'camphor'): (the Blue Book) "Preferred IUPAC names (PINs) are
        not identified for the compounds in this Chapter"; the Blue Book gives
        '(1R,4R)-bornan-2-one (+)-camphor' with the systematic
        '(1R,4R)-1,7,7-trimethylbicyclo[2.2.1]heptan-2-one',:52646-52648).
        OPSIN 2.9.0 full-InChIKey exact for this stereo-free input."""
        result = name_compound('CC1(C)C2CCC1(C)C(=O)C2')
        assert result == '1,7,7-trimethylbicyclo[2.2.1]heptan-2-one', result

    def test_pentacyclic_reaches_polycyclic_path(self):
        """Pentacyclic lactone must be recognized by is_polycyclic_system."""
        from rdkit import Chem
        from orthonym.rules.polycyclic import is_polycyclic_system
        mol = Chem.MolFromSmiles(
            'C=C1C[C@]23C[C@@]1(O)CC[C@H]2[C@@]12CC[C@H](O)'
            '[C@@](C)(C(=O)O1)[C@H]2[C@@H]3C(=O)O'
        )
        assert mol is not None, "Failed to parse pentacyclic SMILES"
        assert is_polycyclic_system(mol) is True, \
            "Pentacyclic lactone must be recognized as polycyclic system"


class TestPolycyclicFunctionalGroups:
    """Tests for functional group detection and naming on polycyclic ring systems (Plan 16-09).

    Validates that:
    - Exocyclic C=O on ring carbons produces -one suffix
    - -OH on ring carbons produces -ol suffix or hydroxy- prefix
    - Seniority determines which FG is suffix vs prefix
    - Retained names (camphor, norbornane) are not broken
    - Non-FG polycyclic naming remains unchanged
    """

    def test_oxatricyclo_lactone_has_one_suffix(self):
        """Oxatricyclo lactone: oxa prefix + one suffix for C=O on ring carbon."""
        result = name_compound('O=C1OC2CC3CC1CC(O)(C3)C2')
        assert 'oxa' in result.lower(), f"Expected oxa prefix, got: {result}"
        assert 'on' in result.lower(), f"Expected -one suffix (as 'on' in name), got: {result}"

    def test_oxatricyclo_lactone_has_hydroxy_prefix(self):
        """Oxatricyclo lactone with OH: hydroxy prefix present for -OH group."""
        result = name_compound('O=C1OC2CC3CC1CC(O)(C3)C2')
        assert 'hydroxy' in result.lower(), f"Expected hydroxy prefix, got: {result}"

    def test_simple_oxatricyclo_lactone_has_one_suffix(self):
        """Simple oxatricyclo lactone: oxa prefix and -one suffix."""
        result = name_compound('O=C1OC2CC3CC(C2)CC1C3')
        assert 'oxa' in result.lower(), f"Expected oxa prefix, got: {result}"
        assert 'on' in result.lower(), f"Expected -one suffix (as 'on' in name), got: {result}"

    def test_tricyclo_ketone(self):
        """Tricyclo ketone (adamantanone): -one suffix on ring C=O."""
        result = name_compound('O=C1C2CC3CC1CC(C2)C3')
        assert 'on' in result.lower(), f"Expected -one suffix, got: {result}"
        assert 'tricyclo' in result.lower(), f"Expected tricyclo descriptor, got: {result}"

    def test_tricyclo_alcohol(self):
        """Tricyclo alcohol (1-adamantanol): -ol suffix on ring -OH."""
        result = name_compound('OC1C2CC3CC1CC(C2)C3')
        assert 'ol' in result.lower(), f"Expected -ol suffix, got: {result}"
        assert 'tricyclo' in result.lower(), f"Expected tricyclo descriptor, got: {result}"

    def test_polycyclic_no_fg_unchanged(self):
        """Adamantane without FGs: retained name (regression guard)."""
        result = name_compound('C1C2CC3CC1CC(C2)C3')
        assert result == 'adamantane', f"Expected adamantane, got: {result}"

    def test_norbornane_systematic_name_preserved(self):
        """Bicyclo[2.2.1]heptane keeps its von Baeyer PIN (no retained name)."""
        # PIN per R11: "RETAINED NAMES FOR VON BAEYER PARENT HYDRIDES" the Blue Book "The retained names adamantane and cubane are used in general nomenclature and as preferred IUPAC names."; "bicyclo[2.2.1]heptane (PIN)":2038; OPSIN RT exact.
        result = name_compound('C1CC2CC1CC2')
        assert result == 'bicyclo[2.2.1]heptane', f"Expected bicyclo[2.2.1]heptane, got: {result}"

    def test_camphor_retained_name_preserved(self):
        """Camphor's name is not broken by FG changes.

        Task 12 fix a performance pass (wp6-tests), change-asserted-value (was 'camphor'):
        the systematic name, as in test_camphor_still_bicyclo:50943;
        :52646-52648). OPSIN full-InChIKey exact."""
        result = name_compound('CC1(C)C2CCC1(C)C(=O)C2')
        assert result == '1,7,7-trimethylbicyclo[2.2.1]heptan-2-one', result


class TestOPSINCompatibleVBFormat:
    """Regression tests ensuring VB descriptors use an OPSIN-parseable format.

    The secondary-bridge locants are superscripts, the Blue Book:
    "cited as a pair of superscript arabic numbers... separated by a comma"),
    written in plain text with a caret: '1^3,7'. OPSIN parses it (and the older
    parenthesized '1(3,7)' as well; both give an InChIKey for
    adamantane's descriptor). This class guards against the LaTeX '^{...}' form.
    """

    def test_adamantane_opsin_format(self):
        """Adamantane: retained name is OPSIN-compatible.

        Adamantane now returns retained name; VB descriptor format tested via
        generate_polycyclic_name in other tests.
        """
        result = name_compound('C1C2CC3CC1CC(C2)C3')
        assert result == 'adamantane', f"Expected retained name 'adamantane', got: {result}"
        # Must NOT contain old LaTeX-style notation
        assert '^{' not in result, \
            f"Found old ^{{}} notation in: {result}"

    def test_cubane_opsin_format(self):
        """Cubane: retained name is OPSIN-compatible.

        Cubane now returns retained name; VB descriptor format tested via
        generate_polycyclic_name. a phase: SMILES corrected from the wrong
        (CH)8 cage isomer to the true cubane canonical (TXWRERCHRDBNLG).
        """
        result = name_compound('C12C3C4C1C1C2C3C41')
        assert result == 'cubane', f"Expected retained name 'cubane', got: {result}"
        # Must NOT contain old LaTeX-style notation
        assert '^{' not in result, \
            f"Found old ^{{}} notation in cubane: {result}"

    def test_norbornane_no_secondary_bridges(self):
        """Norbornane (bicyclo[2.2.1]heptane) has no secondary bridges - format unchanged."""
        result = name_compound('C1CC2CCC1C2')
        # Norbornane has no secondary bridges, so no superscript locants at all
        assert 'bicyclo[2.2.1]' in result.lower() or 'norbornane' in result.lower(), \
            f"Expected bicyclo[2.2.1]heptane or norbornane, got: {result}"
        # No locant superscripts should appear (no secondary bridges)
        assert '^{' not in result, f"Found unexpected ^{{}} in norbornane: {result}"

    def test_no_caret_brace_in_any_polycyclic(self):
        """No generated polycyclic name should contain '^{' (old format guard)."""
        test_smiles = [
            'C1C2CC3CC1CC(C2)C3',     # Adamantane (tricyclo)
            'C12C3C4C1C5C3C4C25',     # Cubane (pentacyclo)
            'C1CC2CCC1CC2',           # Bicyclo[2.2.2]octane (bicyclo, no secondary)
            'C1CC2CCC1C2',            # Norbornane (bicyclo, no secondary)
        ]
        for smi in test_smiles:
            result = name_compound(smi)
            assert '^{' not in result, \
                f"Old VB format '^{{}}' found for SMILES {smi}: {result}"

    def test_simple_tricyclo_one_secondary_bridge(self):
        """Tricyclo system with exactly one secondary bridge uses OPSIN format."""
        # Adamantane has exactly one secondary bridge of length 1
        from rdkit import Chem
        from orthonym.rules.polycyclic import VonBaeyerAnalyzer
        mol = Chem.MolFromSmiles('C1C2CC3CC1CC(C2)C3')
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)
        # Verify one secondary bridge
        secondary = [b for b in desc.bridge_info_list if b.is_secondary]
        assert len(secondary) == 1, f"Expected 1 secondary bridge, got {len(secondary)}"
        # Task 12 fix a performance pass (wp6-tests), change-asserted-value (was the
        # parenthesized '(3,7)' and 'no caret', the phase-28 form of e418b259c):
        # the superscript pair is written '^3,7', the Blue Book),
        # the project's plain-text superscript that other passing tests assert
        # ('tricyclo[3.3.1.1^3,7]decane-1-carboxylate'); OPSIN 2.9.0 parses it.
        assert desc.descriptor_string == 'tricyclo[3.3.1.1^3,7]', desc.descriptor_string
        assert '^{' not in desc.descriptor_string
