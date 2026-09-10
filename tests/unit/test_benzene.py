"""
Tests for benzene derivative naming.

Tests IUPAC 2013 PIN style benzene naming:
- Retained names (benzene, toluene, phenol, aniline)
- Monosubstituted benzenes (chlorobenzene, bromobenzene)
- Disubstituted benzenes (1,2-dimethylbenzene, 1,3-dimethylbenzene, 1,4-dimethylbenzene)
- Trisubstituted benzenes
- Numeric locants (not ortho/meta/para)
- Alphabetization of substituents
- Toluene NOT used as substitutable parent
"""

import pytest
from orthonym import name_compound
from orthonym.rules.benzene import (
    is_benzene_ring,
    get_benzene_ring,
    get_benzene_substituents,
    orient_benzene,
    name_substituted_benzene,
)
from rdkit import Chem


# =============================================================================
# Test Retained Names
# =============================================================================

class TestBenzeneRetainedNames:
    """Test that common benzene derivatives use retained names."""

    @pytest.mark.unit
    def test_benzene(self):
        """Unsubstituted benzene should return 'benzene'."""
        assert name_compound("c1ccccc1") == "benzene"

    @pytest.mark.unit
    def test_toluene(self):
        """Methylbenzene should return retained name 'toluene'."""
        assert name_compound("Cc1ccccc1") == "toluene"

    @pytest.mark.unit
    def test_phenol(self):
        """Hydroxybenzene should return retained name 'phenol'."""
        assert name_compound("Oc1ccccc1") == "phenol"

    @pytest.mark.unit
    def test_aniline(self):
        """Aminobenzene should return retained name 'aniline'."""
        assert name_compound("Nc1ccccc1") == "aniline"

    @pytest.mark.unit
    def test_ethylbenzene(self):
        """Ethylbenzene should return retained name 'ethylbenzene'."""
        assert name_compound("CCc1ccccc1") == "ethylbenzene"

    @pytest.mark.unit
    def test_styrene(self):
        """In PIN style the systematic 'ethenylbenzene' is emitted, NOT the retained
        'styrene' (BBR-HYG, a phase;, the Blue Book: styrene is
        retained for general nomenclature only, ethenylbenzene is the PIN)."""
        assert name_compound("C=Cc1ccccc1") == "ethenylbenzene"

    @pytest.mark.unit
    def test_cumene(self):
        """F-T9/DD6 /: 'cumene' is general-only; the PIN is the
        substitutive '(propan-2-yl)benzene' (enclosing marks per."""
        assert name_compound("CC(C)c1ccccc1") == "(propan-2-yl)benzene"


# =============================================================================
# Test Monosubstituted Benzene
# =============================================================================

class TestMonosubstitutedBenzene:
    """Test monosubstituted benzenes - no locant needed."""

    @pytest.mark.unit
    def test_chlorobenzene(self):
        """Chlorobenzene - no locant needed."""
        assert name_compound("Clc1ccccc1") == "chlorobenzene"

    @pytest.mark.unit
    def test_bromobenzene(self):
        """Bromobenzene - no locant needed."""
        assert name_compound("Brc1ccccc1") == "bromobenzene"

    @pytest.mark.unit
    def test_fluorobenzene(self):
        """Fluorobenzene - no locant needed."""
        assert name_compound("Fc1ccccc1") == "fluorobenzene"

    @pytest.mark.unit
    def test_iodobenzene(self):
        """Iodobenzene - no locant needed."""
        assert name_compound("Ic1ccccc1") == "iodobenzene"

    @pytest.mark.unit
    def test_nitrobenzene(self):
        """Nitrobenzene - no locant needed."""
        assert name_compound("[N+](=O)([O-])c1ccccc1") == "nitrobenzene"

    @pytest.mark.unit
    def test_propylbenzene(self):
        """Propylbenzene - no locant needed."""
        assert name_compound("CCCc1ccccc1") == "propylbenzene"

    @pytest.mark.unit
    def test_butylbenzene(self):
        """Butylbenzene - no locant needed."""
        assert name_compound("CCCCc1ccccc1") == "butylbenzene"


# =============================================================================
# Test Disubstituted Benzene
# =============================================================================

class TestDisubstitutedBenzene:
    """Test disubstituted benzenes - all three isomers."""

    @pytest.mark.unit
    def test_dimethylbenzene_1_2(self):
        """ortho-xylene should be named 1,2-dimethylbenzene."""
        assert name_compound("Cc1ccccc1C") == "1,2-dimethylbenzene"

    @pytest.mark.unit
    def test_dimethylbenzene_1_3(self):
        """meta-xylene should be named 1,3-dimethylbenzene."""
        assert name_compound("Cc1cccc(C)c1") == "1,3-dimethylbenzene"

    @pytest.mark.unit
    def test_dimethylbenzene_1_4(self):
        """para-xylene should be named 1,4-dimethylbenzene."""
        assert name_compound("Cc1ccc(C)cc1") == "1,4-dimethylbenzene"

    @pytest.mark.unit
    def test_dichlorobenzene_1_2(self):
        """ortho-dichlorobenzene."""
        assert name_compound("Clc1ccccc1Cl") == "1,2-dichlorobenzene"

    @pytest.mark.unit
    def test_dichlorobenzene_1_3(self):
        """meta-dichlorobenzene."""
        assert name_compound("Clc1cccc(Cl)c1") == "1,3-dichlorobenzene"

    @pytest.mark.unit
    def test_dichlorobenzene_1_4(self):
        """para-dichlorobenzene."""
        assert name_compound("Clc1ccc(Cl)cc1") == "1,4-dichlorobenzene"

    @pytest.mark.unit
    def test_chloromethylbenzene_alphabetized(self):
        """4-chlorotoluene should be 1-chloro-4-methylbenzene (alphabetized)."""
        # chloro comes before methyl alphabetically
        result = name_compound("Cc1ccc(Cl)cc1")
        assert result == "1-chloro-4-methylbenzene"

    @pytest.mark.unit
    def test_bromomethylbenzene_1_2(self):
        """1-bromo-2-methylbenzene (alphabetized)."""
        result = name_compound("Cc1ccccc1Br")
        assert result == "1-bromo-2-methylbenzene"

    @pytest.mark.unit
    def test_diethylbenzene_1_4(self):
        """1,4-diethylbenzene."""
        assert name_compound("CCc1ccc(CC)cc1") == "1,4-diethylbenzene"


# =============================================================================
# Test Trisubstituted Benzene
# =============================================================================

class TestTrisubstitutedBenzene:
    """Test trisubstituted benzenes."""

    @pytest.mark.unit
    def test_trimethylbenzene_1_2_4(self):
        """1,2,4-trimethylbenzene."""
        assert name_compound("Cc1ccc(C)c(C)c1") == "1,2,4-trimethylbenzene"

    @pytest.mark.unit
    def test_trimethylbenzene_1_3_5(self):
        """1,3,5-trimethylbenzene (mesitylene)."""
        assert name_compound("Cc1cc(C)cc(C)c1") == "1,3,5-trimethylbenzene"

    @pytest.mark.unit
    def test_trimethylbenzene_1_2_3(self):
        """1,2,3-trimethylbenzene."""
        assert name_compound("Cc1cccc(C)c1C") == "1,2,3-trimethylbenzene"

    @pytest.mark.unit
    def test_trichlorobenzene_1_2_4(self):
        """1,2,4-trichlorobenzene."""
        assert name_compound("Clc1ccc(Cl)c(Cl)c1") == "1,2,4-trichlorobenzene"

    @pytest.mark.unit
    def test_trichlorobenzene_1_3_5(self):
        """1,3,5-trichlorobenzene."""
        assert name_compound("Clc1cc(Cl)cc(Cl)c1") == "1,3,5-trichlorobenzene"


# =============================================================================
# Test Benzene Numbering Rules
# =============================================================================

class TestBenzeneNumberingRules:
    """Test IUPAC 2013 numbering rules for benzene."""

    @pytest.mark.unit
    def test_lowest_locants_1_4_not_1_4_via_different_path(self):
        """Locants should be lowest via first-point-of-difference."""
        # Both orientations give [1,4], so alphabetically first at 1
        result = name_compound("Cc1ccc(Cl)cc1")
        # chloro < methyl alphabetically, so chloro should be at position 1
        assert result.startswith("1-chloro")

    @pytest.mark.unit
    def test_alphabetization_bromo_before_chloro(self):
        """Bromo comes before chloro alphabetically."""
        result = name_compound("Brc1ccc(Cl)cc1")
        # bromo < chloro, so bromo at position 1
        assert result.startswith("1-bromo")

    @pytest.mark.unit
    def test_no_ortho_meta_para_in_pin(self):
        """PIN style uses numeric locants, not o/m/p."""
        result = name_compound("Cc1ccc(C)cc1")
        assert "ortho" not in result
        assert "meta" not in result
        assert "para" not in result
        assert "1,4-" in result

    @pytest.mark.unit
    def test_locants_sorted_ascending(self):
        """Locants should be in ascending order."""
        result = name_compound("Cc1ccc(C)c(C)c1")  # 1,2,4-trimethyl
        assert "1,2,4" in result

    @pytest.mark.unit
    def test_lowest_locant_set_preferred(self):
        """The orientation giving lowest locant set is preferred."""
        # 1,2,4 is preferred over 1,3,4 (first-point-of-difference: 2 < 3)
        result = name_compound("Cc1ccc(C)c(C)c1")
        assert "1,2,4" in result


# =============================================================================
# Test Toluene Not As Parent
# =============================================================================

class TestTolueneNotAsParent:
    """Test that toluene is NOT used as a substitutable parent."""

    @pytest.mark.unit
    def test_chlorotoluene_becomes_chloromethylbenzene(self):
        """'4-chlorotoluene' should be '1-chloro-4-methylbenzene'."""
        result = name_compound("ClC1=CC=C(C)C=C1")
        assert "toluene" not in result.lower()
        assert "methylbenzene" in result

    @pytest.mark.unit
    def test_bromotoluene_becomes_bromomethylbenzene(self):
        """Bromotoluene should use methylbenzene parent."""
        result = name_compound("Brc1ccc(C)cc1")
        assert "toluene" not in result.lower()
        assert "methylbenzene" in result

    @pytest.mark.unit
    def test_nitrotoluene_becomes_nitromethylbenzene(self):
        """Nitrotoluene should use methylbenzene parent."""
        result = name_compound("[N+](=O)([O-])c1ccc(C)cc1")
        assert "toluene" not in result.lower()
        # Note: could be nitro...methylbenzene


# =============================================================================
# Test Benzene Ring Detection
# =============================================================================

class TestBenzeneRingDetection:
    """Test the benzene ring detection functions."""

    @pytest.mark.unit
    def test_is_benzene_ring_true(self):
        """is_benzene_ring returns True for benzene."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert is_benzene_ring(mol, ring) is True

    @pytest.mark.unit
    def test_is_benzene_ring_false_for_pyridine(self):
        """is_benzene_ring returns False for pyridine (contains N)."""
        mol = Chem.MolFromSmiles("c1ccncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert is_benzene_ring(mol, ring) is False

    @pytest.mark.unit
    def test_is_benzene_ring_false_for_cyclohexane(self):
        """is_benzene_ring returns False for cyclohexane (not aromatic)."""
        mol = Chem.MolFromSmiles("C1CCCCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert is_benzene_ring(mol, ring) is False

    @pytest.mark.unit
    def test_get_benzene_ring_finds_ring(self):
        """get_benzene_ring returns the benzene ring atoms."""
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        ring = get_benzene_ring(mol)
        assert ring is not None
        assert len(ring) == 6

    @pytest.mark.unit
    def test_get_benzene_ring_none_for_pyridine(self):
        """get_benzene_ring returns None for pyridine."""
        mol = Chem.MolFromSmiles("c1ccncc1")
        ring = get_benzene_ring(mol)
        assert ring is None


# =============================================================================
# Test Substituent Detection
# =============================================================================

class TestBenzeneSubstituentDetection:
    """Test the substituent detection functions."""

    @pytest.mark.unit
    def test_detect_methyl_substituent(self):
        """Detect methyl substituent on toluene."""
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        ring = get_benzene_ring(mol)
        subs = get_benzene_substituents(mol, ring)
        # Should have one substituent
        assert len(subs) == 1
        # The substituent should be methyl
        sub_names = [s['name'] for s_list in subs.values() for s in s_list]
        assert 'methyl' in sub_names

    @pytest.mark.unit
    def test_detect_chloro_substituent(self):
        """Detect chloro substituent on chlorobenzene."""
        mol = Chem.MolFromSmiles("Clc1ccccc1")
        ring = get_benzene_ring(mol)
        subs = get_benzene_substituents(mol, ring)
        sub_names = [s['name'] for s_list in subs.values() for s in s_list]
        assert 'chloro' in sub_names

    @pytest.mark.unit
    def test_detect_two_substituents(self):
        """Detect two methyl substituents on xylene."""
        mol = Chem.MolFromSmiles("Cc1ccc(C)cc1")
        ring = get_benzene_ring(mol)
        subs = get_benzene_substituents(mol, ring)
        # Should have two substituted positions
        assert len(subs) == 2
        # Both should be methyl
        sub_names = [s['name'] for s_list in subs.values() for s in s_list]
        assert sub_names.count('methyl') == 2


# =============================================================================
# Test Orient Benzene
# =============================================================================

class TestOrientBenzene:
    """Test the benzene orientation function."""

    @pytest.mark.unit
    def test_orient_monosubstituted(self):
        """Monosubstituted benzene: substituent at position 1."""
        mol = Chem.MolFromSmiles("Clc1ccccc1")
        ring = get_benzene_ring(mol)
        subs = get_benzene_substituents(mol, ring)
        oriented = orient_benzene(mol, ring, subs)

        # First atom should have the chloro substituent
        assert oriented[0] in subs

    @pytest.mark.unit
    def test_orient_para_disubstituted(self):
        """Para-disubstituted: positions 1 and 4."""
        mol = Chem.MolFromSmiles("Cc1ccc(C)cc1")
        ring = get_benzene_ring(mol)
        subs = get_benzene_substituents(mol, ring)
        oriented = orient_benzene(mol, ring, subs)

        # Calculate locants
        locants = []
        for i, atom_idx in enumerate(oriented):
            if atom_idx in subs:
                locants.append(i + 1)

        assert sorted(locants) == [1, 4]

    @pytest.mark.unit
    def test_orient_mixed_para_alphabetic(self):
        """Para with different substituents: alphabetically first at 1."""
        mol = Chem.MolFromSmiles("Cc1ccc(Cl)cc1")  # chloro and methyl
        ring = get_benzene_ring(mol)
        subs = get_benzene_substituents(mol, ring)
        oriented = orient_benzene(mol, ring, subs)

        # Chloro should be at position 1 (c < m)
        pos1_atom = oriented[0]
        assert pos1_atom in subs
        pos1_sub_name = subs[pos1_atom][0]['name']
        assert pos1_sub_name == 'chloro'


# =============================================================================
# Additional Edge Cases
# =============================================================================

class TestBenzeneEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.unit
    def test_pentasubstituted_benzene(self):
        """Pentamethylbenzene."""
        result = name_compound("Cc1c(C)c(C)c(C)c(C)c1")
        assert "pentamethylbenzene" in result

    @pytest.mark.unit
    def test_hexasubstituted_benzene(self):
        """Hexamethylbenzene - all positions substituted."""
        result = name_compound("Cc1c(C)c(C)c(C)c(C)c1C")
        assert "hexamethylbenzene" in result

    @pytest.mark.unit
    def test_tetrasubstituted_benzene(self):
        """1,2,3,4-tetramethylbenzene."""
        result = name_compound("Cc1c(C)c(C)c(C)cc1")
        assert "tetramethylbenzene" in result


# =============================================================================
# Test Benzonitrile Naming (Fix - a phase)
# =============================================================================

class TestBenzonitrileNaming:
    """Tests for benzonitrile naming (fix).

    IUPAC 2013 PIN: benzonitrile (not cyanobenzene) per
    Substituents are numbered relative to the nitrile carbon (position 1).
    """

    @pytest.mark.unit
    def test_simple_benzonitrile(self):
        """c1ccccc1C#N -> benzonitrile"""
        result = name_compound('c1ccccc1C#N')
        assert result == 'benzonitrile', f"Got {result}"

    @pytest.mark.unit
    def test_benzonitrile_canonical_smiles(self):
        """N#Cc1ccccc1 (canonical) -> benzonitrile"""
        result = name_compound('N#Cc1ccccc1')
        assert result == 'benzonitrile', f"Got {result}"

    @pytest.mark.unit
    def test_4_chlorobenzonitrile(self):
        """Clc1ccc(C#N)cc1 (para) -> 4-chlorobenzonitrile"""
        result = name_compound('Clc1ccc(C#N)cc1')
        assert result == '4-chlorobenzonitrile', f"Got {result}"

    @pytest.mark.unit
    def test_4_methylbenzonitrile(self):
        """Cc1ccc(C#N)cc1 (para) -> 4-methylbenzonitrile"""
        result = name_compound('Cc1ccc(C#N)cc1')
        assert result == '4-methylbenzonitrile', f"Got {result}"

    @pytest.mark.unit
    def test_2_chlorobenzonitrile(self):
        """Clc1ccccc1C#N (ortho) -> 2-chlorobenzonitrile"""
        result = name_compound('Clc1ccccc1C#N')
        assert result == '2-chlorobenzonitrile', f"Got {result}"

    @pytest.mark.unit
    def test_3_chlorobenzonitrile(self):
        """Clc1cccc(C#N)c1 (meta) -> 3-chlorobenzonitrile"""
        result = name_compound('Clc1cccc(C#N)c1')
        assert result == '3-chlorobenzonitrile', f"Got {result}"

    @pytest.mark.unit
    def test_2_methylbenzonitrile(self):
        """Cc1ccccc1C#N (ortho methyl) -> 2-methylbenzonitrile"""
        result = name_compound('Cc1ccccc1C#N')
        assert result == '2-methylbenzonitrile', f"Got {result}"

    @pytest.mark.unit
    def test_3_methylbenzonitrile(self):
        """Cc1cccc(C#N)c1 (meta methyl) -> 3-methylbenzonitrile"""
        result = name_compound('Cc1cccc(C#N)c1')
        assert result == '3-methylbenzonitrile', f"Got {result}"

    @pytest.mark.unit
    def test_4_bromobenzonitrile(self):
        """Brc1ccc(C#N)cc1 (para bromo) -> 4-bromobenzonitrile"""
        result = name_compound('Brc1ccc(C#N)cc1')
        assert result == '4-bromobenzonitrile', f"Got {result}"

    @pytest.mark.unit
    def test_4_fluorobenzonitrile(self):
        """Fc1ccc(C#N)cc1 (para fluoro) -> 4-fluorobenzonitrile"""
        result = name_compound('Fc1ccc(C#N)cc1')
        assert result == '4-fluorobenzonitrile', f"Got {result}"


# =============================================================================
# Test Nitrile Detection Functions
# =============================================================================

class TestNitrileDetection:
    """Test nitrile detection helper functions."""

    @pytest.mark.unit
    def test_detect_benzene_nitrile_true(self):
        """_detect_benzene_nitrile returns True for benzonitrile."""
        from orthonym.rules.benzene import _detect_benzene_nitrile, get_benzene_ring
        mol = Chem.MolFromSmiles('c1ccccc1C#N')
        ring = get_benzene_ring(mol)
        result = _detect_benzene_nitrile(mol, ring)
        assert result['is_nitrile'] is True
        assert len(result['nitrile_positions']) == 1

    @pytest.mark.unit
    def test_detect_benzene_nitrile_false_for_plain_benzene(self):
        """_detect_benzene_nitrile returns False for plain benzene."""
        from orthonym.rules.benzene import _detect_benzene_nitrile, get_benzene_ring
        mol = Chem.MolFromSmiles('c1ccccc1')
        ring = get_benzene_ring(mol)
        result = _detect_benzene_nitrile(mol, ring)
        assert result['is_nitrile'] is False
        assert len(result['nitrile_positions']) == 0

    @pytest.mark.unit
    def test_detect_benzene_nitrile_false_for_toluene(self):
        """_detect_benzene_nitrile returns False for toluene."""
        from orthonym.rules.benzene import _detect_benzene_nitrile, get_benzene_ring
        mol = Chem.MolFromSmiles('Cc1ccccc1')
        ring = get_benzene_ring(mol)
        result = _detect_benzene_nitrile(mol, ring)
        assert result['is_nitrile'] is False

    @pytest.mark.unit
    def test_identify_nitrile_substituent(self):
        """_identify_substituent detects nitrile as a substituent."""
        from orthonym.rules.benzene import get_benzene_ring, get_benzene_substituents
        mol = Chem.MolFromSmiles('c1ccccc1C#N')
        ring = get_benzene_ring(mol)
        subs = get_benzene_substituents(mol, ring)

        # Check that nitrile was detected (suffix FG detection returns 'carbonitrile')
        sub_names = [s['name'] for s_list in subs.values() for s in s_list]
        assert 'carbonitrile' in sub_names or 'nitrile' in sub_names
