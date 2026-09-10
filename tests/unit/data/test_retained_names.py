"""
Tests for retained names data (a phase additions).

Tests the benzonitrile and thiazolidine entries added to fix and.
"""

import pytest
from rdkit import Chem
from orthonym.data.retained_names import RETAINED_NAMES, get_retained_name
from orthonym import name_compound


# =============================================================================
# Test Benzonitrile Retained Name
# =============================================================================

class TestBenzonitrileRetainedName:
    """Tests for benzonitrile in retained names (fix)."""

    @pytest.mark.unit
    def test_benzonitrile_in_retained_names(self):
        """Benzonitrile should be in RETAINED_NAMES values."""
        assert 'benzonitrile' in RETAINED_NAMES.values()

    @pytest.mark.unit
    def test_benzonitrile_canonical_smiles_key(self):
        """Benzonitrile canonical SMILES key should be correct."""
        # Verify the canonical SMILES
        mol = Chem.MolFromSmiles('c1ccccc1C#N')
        canonical = Chem.MolToSmiles(mol)
        assert canonical == 'N#Cc1ccccc1', f"Expected 'N#Cc1ccccc1', got '{canonical}'"
        assert canonical in RETAINED_NAMES
        assert RETAINED_NAMES[canonical] == 'benzonitrile'

    @pytest.mark.unit
    def test_get_retained_name_benzonitrile(self):
        """get_retained_name should return 'benzonitrile' for canonical SMILES."""
        result = get_retained_name('N#Cc1ccccc1')
        assert result == 'benzonitrile', f"Got {result}"

    @pytest.mark.unit
    def test_benzonitrile_naming_e2e(self):
        """name_compound should return 'benzonitrile' via retained name lookup."""
        # This tests the full pipeline through retained names
        result = name_compound('N#Cc1ccccc1')
        assert result == 'benzonitrile', f"Got {result}"


# =============================================================================
# Test Thiazolidine Retained Names
# =============================================================================

class TestThiazolidineRetainedNames:
    """Tests for thiazolidine and isothiazolidine in retained names (fix)."""

    @pytest.mark.unit
    def test_thiazolidine_in_retained_names(self):
        """Thiazolidine should be in RETAINED_NAMES values."""
        assert 'thiazolidine' in RETAINED_NAMES.values()

    @pytest.mark.unit
    def test_thiazolidine_canonical_smiles_key(self):
        """Thiazolidine canonical SMILES key should be correct.

        1,3-thiazolidine: S at position 1, N at position 3 (not adjacent)
        The canonical SMILES is C1CSCN1.
        """
        # Verify the canonical SMILES for thiazolidine
        mol = Chem.MolFromSmiles('C1SCNC1')  # Input form
        canonical = Chem.MolToSmiles(mol)
        assert canonical == 'C1CSCN1', f"Expected 'C1CSCN1', got '{canonical}'"
        assert canonical in RETAINED_NAMES
        assert RETAINED_NAMES[canonical] == 'thiazolidine'

    @pytest.mark.unit
    def test_isothiazolidine_in_retained_names(self):
        """Isothiazolidine should be in RETAINED_NAMES values."""
        assert 'isothiazolidine' in RETAINED_NAMES.values()

    @pytest.mark.unit
    def test_isothiazolidine_canonical_smiles_key(self):
        """Isothiazolidine canonical SMILES key should be correct.

        1,2-isothiazolidine: S at position 1, N at position 2 (adjacent)
        The canonical SMILES is C1CNSC1.
        """
        # Verify the canonical SMILES for isothiazolidine
        mol = Chem.MolFromSmiles('C1CCSN1')  # Input form (S-N adjacent)
        canonical = Chem.MolToSmiles(mol)
        assert canonical == 'C1CNSC1', f"Expected 'C1CNSC1', got '{canonical}'"
        assert canonical in RETAINED_NAMES
        assert RETAINED_NAMES[canonical] == 'isothiazolidine'

    # ----------------------------------------------------------------------
    # Task AA5: the six expectations below asserted the BARE stem, which the
    # Blue Book prints as NOT the PIN. Section ** "Retained names of
    # heteromonocycles"** (heading ``the Blue Book``), whose lead-in at
    # ``:8117`` reads "Retained names for saturated heteromonocycles are given
    # in Table 2.3". Both governing lines are single, non-interleaved lines --
    # no OCR column reconstruction is involved for either name:
    #
    #:8182 "oxazolidine 1,3-oxazolidine (PIN) thiazolidine (S instead of O)
    # 1,3-thiazolidine (PIN) selenazolidine (Se instead of O)"
    #:8184 "1,2-oxazolidine (PIN) isothiazolidine (S instead of O)
    # 1,2-thiazolidine (PIN) isoselenazolidine (Se instead of O)..."
    #
    # Independent corroboration (does NOT use the code under test): OPSIN 2.9.0
    # parses '1,3-thiazolidine' -> C1CSCN1 and '1,2-thiazolidine' -> C1CNSC1, so
    # the new names denote exactly these molecules. ⚠ That check confirms
    # VALIDITY only -- OPSIN resolves the old bare names to the same structures,
    # so it cannot adjudicate PIN-preference. Preference rests on the Blue Book
    # lines above, which state it explicitly. Second corroboration, internal:
    # every sibling in the same two table rows already emitted the
    # locant-bearing form (1,3-oxazolidine, 1,3-selenazolidine,
    # 1,3-tellurazolidine, 1,2-oxazole, 1,2-thiazole) -- members of one series
    # that must agree, and did not.
    #
    # What would make this wrong: if the "(PIN)" marker in Table 2.3 attached to
    # the trivial column rather than the locant-bearing one. It does not -- both
    # pairs are printed left-to-right on one line, and Table 2.2's:8135/:8137
    # print the identical mapping for the mancude analogues.
    # ----------------------------------------------------------------------

    @pytest.mark.unit
    def test_get_retained_name_no_longer_returns_bare_thiazolidine(self):
        """The PIN-path lookup must not serve the non-PIN bare stem.

        ``get_retained_name`` reads the merged, deny-filtered
        ``ALL_RETAINED_NAMES``; the adjudicated deny row withdraws
        'thiazolidine' from it so the generic path can supply the PIN.
        """
        assert get_retained_name('C1CSCN1') != 'thiazolidine'

    @pytest.mark.unit
    def test_get_retained_name_no_longer_returns_bare_isothiazolidine(self):
        assert get_retained_name('C1CNSC1') != 'isothiazolidine'

    @pytest.mark.unit
    def test_bare_stems_are_demoted_not_deleted(self):
        """The deny is a demotion to the --trivial surface, not a deletion.

        Keeps teeth on the mechanism: if a future change deletes the rows
        instead of denying them, the legitimate general-nomenclature names are
        silently lost and this fails.
        """
        from orthonym.data import GENERAL_RETAINED_NAMES
        demoted = set(GENERAL_RETAINED_NAMES.values())
        assert 'thiazolidine' in demoted
        assert 'isothiazolidine' in demoted

    @pytest.mark.unit
    def test_thiazolidine_naming_e2e(self):
        """name_compound returns the PIN '1,3-thiazolidine' (the Blue Book)."""
        result = name_compound('C1CSCN1')
        assert result == '1,3-thiazolidine', f"Got {result}"

    @pytest.mark.unit
    def test_thiazolidine_naming_alt_input(self):
        """Alternate input form gives the same PIN."""
        result = name_compound('C1SCNC1')
        assert result == '1,3-thiazolidine', f"Got {result}"

    @pytest.mark.unit
    def test_isothiazolidine_naming_e2e(self):
        """name_compound returns the PIN '1,2-thiazolidine' (the Blue Book)."""
        result = name_compound('C1CNSC1')
        assert result == '1,2-thiazolidine', f"Got {result}"

    @pytest.mark.unit
    def test_isothiazolidine_naming_alt_input(self):
        """Alternate input form gives the same PIN."""
        result = name_compound('C1CCSN1')
        assert result == '1,2-thiazolidine', f"Got {result}"


# =============================================================================
# Test Thiazolidine vs Isothiazolidine Distinction
# =============================================================================

class TestThiazolidineDistinction:
    """Test that thiazolidine and isothiazolidine are correctly distinguished."""

    @pytest.mark.unit
    def test_different_canonical_smiles(self):
        """Thiazolidine and isothiazolidine have different canonical SMILES."""
        mol_tz = Chem.MolFromSmiles('C1SCNC1')  # 1,3-thiazolidine
        mol_itz = Chem.MolFromSmiles('C1CCSN1')  # 1,2-isothiazolidine

        can_tz = Chem.MolToSmiles(mol_tz)
        can_itz = Chem.MolToSmiles(mol_itz)

        assert can_tz != can_itz, f"Expected different SMILES, got {can_tz} and {can_itz}"
        assert can_tz == 'C1CSCN1'
        assert can_itz == 'C1CNSC1'

    @pytest.mark.unit
    def test_thiazolidine_s_n_not_adjacent(self):
        """In 1,3-thiazolidine, S and N are NOT adjacent (separated by C)."""
        mol = Chem.MolFromSmiles('C1CSCN1')
        # Find S and N atoms
        s_atom = None
        n_atom = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'S':
                s_atom = atom
            elif atom.GetSymbol() == 'N':
                n_atom = atom

        # Check if S and N are neighbors
        s_neighbors = [n.GetSymbol() for n in s_atom.GetNeighbors()]
        assert 'N' not in s_neighbors, "In thiazolidine, S and N should NOT be adjacent"

    @pytest.mark.unit
    def test_isothiazolidine_s_n_adjacent(self):
        """In 1,2-isothiazolidine, S and N ARE adjacent (S-N bond)."""
        mol = Chem.MolFromSmiles('C1CNSC1')
        # Find S and N atoms
        s_atom = None
        n_atom = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'S':
                s_atom = atom
            elif atom.GetSymbol() == 'N':
                n_atom = atom

        # Check if S and N are neighbors
        s_neighbors = [n.GetSymbol() for n in s_atom.GetNeighbors()]
        assert 'N' in s_neighbors, "In isothiazolidine, S and N SHOULD be adjacent"
