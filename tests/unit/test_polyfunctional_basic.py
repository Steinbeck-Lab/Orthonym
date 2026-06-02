"""Tests for polyfunctional compound naming - basic prefix handling."""
import pytest
from orthonym import name_compound


class TestHydroxyAcids:
    """Test hydroxy acid naming (POLY-06)."""

    def test_glycolic_acid_systematic(self):
        # 2-hydroxyacetic acid (glycolic acid trivial name)
        # Note: ethanoic = acetic, both are valid
        result = name_compound("OCC(=O)O")
        assert result == "2-hydroxyethanoic acid"

    def test_3_hydroxypropanoic_acid(self):
        assert name_compound("OCCC(=O)O") == "3-hydroxypropanoic acid"

    def test_2_hydroxypropanoic_acid(self):
        # Lactic acid systematic
        assert name_compound("CC(O)C(=O)O") == "2-hydroxypropanoic acid"

    def test_dihydroxy_acid(self):
        # 2,3-dihydroxypropanoic acid (glyceric acid)
        # Can be named as "2-hydroxy-3-hydroxy..." or "2,3-dihydroxy..."
        # Both are acceptable IUPAC; we currently generate the longer form
        result = name_compound("OCC(O)C(=O)O")
        assert "hydroxy" in result and "propanoic acid" in result


class TestKetoAcids:
    """Test keto acid naming (POLY-08)."""

    def test_pyruvic_acid_systematic(self):
        # 2-oxopropanoic acid (pyruvic acid trivial name)
        assert name_compound("CC(=O)C(=O)O") == "2-oxopropanoic acid"

    def test_3_oxopropanoic_acid(self):
        # Aldehyde as oxo prefix when part of chain
        assert name_compound("O=CCC(=O)O") == "3-oxopropanoic acid"

    def test_3_oxobutanoic_acid(self):
        # Acetoacetic acid systematic
        assert name_compound("CC(=O)CC(=O)O") == "3-oxobutanoic acid"


class TestEthers:
    """Test ether naming as alkoxy prefixes (POLY-05)."""

    def test_methoxymethane(self):
        # Dimethyl ether substitutive name
        assert name_compound("COC") == "methoxymethane"

    def test_methoxyethane(self):
        assert name_compound("COCC") == "methoxyethane"

    def test_ethoxyethane(self):
        # Diethyl ether substitutive name
        assert name_compound("CCOCC") == "ethoxyethane"

    def test_methoxy_with_acid(self):
        # 3-methoxypropanoic acid
        assert name_compound("COCCC(=O)O") == "3-methoxypropanoic acid"


class TestPolyfunctionalDetection:
    """Test detection of polyfunctional compounds."""

    def test_single_fg_not_polyfunctional(self):
        # Simple alcohol - single FG type
        from rdkit import Chem
        from orthonym.rules.polyfunctional import detect_polyfunctional
        from orthonym.perception.functional_groups import detect_functional_groups

        mol = Chem.MolFromSmiles("CCO")
        fgs = detect_functional_groups(mol)
        assert detect_polyfunctional(mol, fgs) is False

    def test_multiple_same_fg_not_polyfunctional(self):
        # Diol - multiple instances of same FG type
        from rdkit import Chem
        from orthonym.rules.polyfunctional import detect_polyfunctional
        from orthonym.perception.functional_groups import detect_functional_groups

        mol = Chem.MolFromSmiles("OCCO")
        fgs = detect_functional_groups(mol)
        # A diol has only alcohol FGs, so it's NOT polyfunctional
        assert detect_polyfunctional(mol, fgs) is False

    def test_two_different_fgs_is_polyfunctional(self):
        # Hydroxy acid - two different FGs
        from rdkit import Chem
        from orthonym.rules.polyfunctional import detect_polyfunctional
        from orthonym.perception.functional_groups import detect_functional_groups

        mol = Chem.MolFromSmiles("OCC(=O)O")
        fgs = detect_functional_groups(mol)
        assert detect_polyfunctional(mol, fgs) is True


class TestPolyfunctionalIntegration:
    """End-to-end polyfunctional naming tests."""

    def test_hydroxy_acid_full_pipeline(self):
        result = name_compound("OCC(=O)O")
        assert "hydroxy" in result
        assert "oic acid" in result

    def test_keto_acid_full_pipeline(self):
        result = name_compound("CC(=O)C(=O)O")
        assert "oxo" in result
        assert "oic acid" in result

    def test_ether_hydrocarbon_parent(self):
        assert name_compound("COCC") == "methoxyethane"

    def test_ether_acid_parent(self):
        result = name_compound("COCCC(=O)O")
        assert "methoxy" in result
        assert "oic acid" in result

    def test_existing_alkane_naming_unchanged(self):
        # Ensure alkane naming still works
        assert name_compound("CCCC") == "butane"
        assert name_compound("CC(C)C") == "2-methylpropane"

    def test_existing_alcohol_naming_unchanged(self):
        # Ensure simple alcohol naming still works
        assert name_compound("CCO") == "ethanol"
        assert name_compound("CCCO") == "propan-1-ol"

    def test_existing_acid_naming_unchanged(self):
        # Ensure simple acid naming still works
        result = name_compound("CC(=O)O")
        assert result in ["acetic acid", "ethanoic acid"]  # Both are valid

    def test_existing_ketone_naming_unchanged(self):
        # Ensure simple ketone naming still works
        # propan-2-one is the IUPAC 2013 PIN (acetone removed from retained names)
        result = name_compound("CC(=O)C")
        assert result == "propan-2-one"


class TestPrefixFormatting:
    """Test FG prefix formatting functions."""

    def test_format_single_hydroxy(self):
        from orthonym.rules.polyfunctional import format_fg_prefix
        assert format_fg_prefix("hydroxy", [2], 1) == "2-hydroxy"

    def test_format_single_oxo(self):
        from orthonym.rules.polyfunctional import format_fg_prefix
        assert format_fg_prefix("oxo", [3], 1) == "3-oxo"

    def test_format_dihydroxy(self):
        from orthonym.rules.polyfunctional import format_fg_prefix
        result = format_fg_prefix("hydroxy", [2, 4], 2)
        assert "dihydroxy" in result
        assert "2,4" in result

    def test_format_no_locants(self):
        from orthonym.rules.polyfunctional import format_fg_prefix
        assert format_fg_prefix("hydroxy", [], 1) == "hydroxy"


class TestAlkoxyPrefixForm:
    """Test alkoxy prefix determination for ethers."""

    def test_methoxy_prefix(self):
        from rdkit import Chem
        from orthonym.rules.polyfunctional import get_fg_prefix_form

        mol = Chem.MolFromSmiles("COCC")
        # Match for methoxy: ether oxygen + methyl
        prefix = get_fg_prefix_form("ether", mol, (1, 0, 2), [2, 3])
        assert prefix == "methoxy"

    def test_ethoxy_prefix(self):
        from rdkit import Chem
        from orthonym.rules.polyfunctional import get_fg_prefix_form

        mol = Chem.MolFromSmiles("CCOCC")
        # Ether oxygen is at index 2
        prefix = get_fg_prefix_form("ether", mol, (2, 0, 3), [3, 4])
        assert prefix == "ethoxy"


class TestLocantDetermination:
    """Test FG locant finding."""

    def test_alcohol_locant_in_hydroxy_acid(self):
        from rdkit import Chem
        from orthonym.rules.polyfunctional import get_non_principal_fg_locants

        # 2-hydroxypropanoic acid: CC(O)C(=O)O
        mol = Chem.MolFromSmiles("CC(O)C(=O)O")
        # secondary_alcohol match: (O, C-bearing-OH, neighbor1, neighbor2)
        fg_atoms = [(2, 1, 0, 3)]
        principal_chain = [3, 1, 0]  # acid at 1, then C, then methyl
        atom_to_locant = {3: 1, 1: 2, 0: 3}

        locants = get_non_principal_fg_locants(mol, fg_atoms, principal_chain, atom_to_locant)
        assert locants == [2]  # Alcohol at position 2

    def test_ketone_locant_in_keto_acid(self):
        from rdkit import Chem
        from orthonym.rules.polyfunctional import get_non_principal_fg_locants

        # 2-oxopropanoic acid: CC(=O)C(=O)O
        mol = Chem.MolFromSmiles("CC(=O)C(=O)O")
        # ketone match: (neighbor, carbonyl_C, O, neighbor2)
        fg_atoms = [(0, 1, 2, 3)]
        principal_chain = [3, 1, 0]  # acid at 1, ketone C at 2, methyl at 3
        atom_to_locant = {3: 1, 1: 2, 0: 3}

        locants = get_non_principal_fg_locants(mol, fg_atoms, principal_chain, atom_to_locant)
        assert locants == [2]  # Ketone at position 2


# ============================================================================
# Phase 169.5 SUB-05 — detachable-prefix separators + enclosing marks (Wave 0)
#
# Must-not-split negatives assert NOW (D-16: legitimately-fused tokens +
# single-formatter 2,3-dimethyl stay intact). Glued/dropped-locant/nested-stereo
# targets xfail until Plan 04 (4a off-chain locant recovery, 4b _join_prefixes
# separator, D-17 enclosing-mark unify).
# ============================================================================

import pytest  # noqa: E402
from orthonym.rules.polyfunctional import _join_prefixes  # noqa: E402


@pytest.mark.unit
class TestSUB05JoinMustNotSplit:
    """_join_prefixes MUST keep legitimately-fused / single-token prefixes intact."""

    def test_join_keeps_phosphonooxy(self):
        assert _join_prefixes(["phosphonooxy"]) == "phosphonooxy"

    def test_join_keeps_phenylsulfanyl(self):
        assert _join_prefixes(["phenylsulfanyl"]) == "phenylsulfanyl"

    def test_join_keeps_dimethyl_locant_token(self):
        # '2,3-dimethyl' is a SINGLE formatter token — structurally immune.
        assert _join_prefixes(["2,3-dimethyl"]) == "2,3-dimethyl"


@pytest.mark.unit
class TestSUB05GluedAndDroppedPrefix:
    """Glued detachable-prefix pair + dropped off-chain locant — xfail (Plan 04)."""

    def test_join_separates_oxo_phosphono(self):
        # SUB-05 Plan 04 (4b): gated detachable-prefix separator.
        assert _join_prefixes(["3-oxo", "phosphono"]) != "3-oxophosphono"

    def test_glued_oxophosphono_in_name(self):
        # SUB-05 Plan 04 (4b): the 'oxophosphono' glue is gone (now 'oxo-phosphono').
        name = name_compound("CC(=O)C(CP(O)(O)=O)C(=O)O")
        assert "oxophosphono" not in name

    @pytest.mark.xfail(reason="DEFERRED (169.5, honest-fail): the off-chain CH2-PO3H2 must unify into the compound substituent '(phosphonomethyl)' (the plan's literal '3-phosphono' is structurally WRONG — agent OPSIN-proven). Option A (unify off-chain FG + suppress the dup alkyl) is a complex substituent-enumerator change; narrow reach (0/300 subset). 4b separated the glue but the unification is deferred.", strict=False)
    def test_dropped_locant_phosphonomethyl(self):
        # HOOC-CH2-CH(CH2PO3H2)-COOH -> 2-(phosphonomethyl)butanedioic acid.
        name = name_compound("OC(=O)CC(CP(=O)(O)O)C(=O)O")
        assert "(phosphonomethyl)" in name


@pytest.mark.unit
class TestSUB05NestedStereoEnclosing:
    """D-17: monosubstituted ring-substituent stereo must be enclosed — xfail (Plan 04)."""

    @pytest.mark.xfail(reason="DEFERRED (169.5, honest-fail): for a benzene parent the stereocentre lives INSIDE the substituent, but _generate_stereodescriptors emits it as a MOLECULAR-level '(R)-' prefix, not enclosed with the substituent. The generic-ring D-17 enclosing path IS fixed; the benzene/molecular-stereo case needs substituent-local stereodescriptor generation (a deep architectural change, related to the SUB-04 fragment-context issue). Out of SUB-05 scope.", strict=False)
    def test_nested_stereo_enclosed(self):
        # (R)-3-methylpentylbenzene -> [(R)-3-methylpentyl]benzene (P-16.3.3).
        name = name_compound("CC[C@@H](C)CCc1ccccc1")
        assert name == "[(R)-3-methylpentyl]benzene"
