"""
Integration tests for a phase: Early Return Handler Retrofit.

Tests the shared _integrate_universal_prefixes helper and verifies
that retrofitted handlers (acid halide, lactone, lactam, ether) correctly
discover and name substituents via the universal pipeline.

Reference: IUPAC 2013 Blue Book, (detachable prefixes)
"""

import pytest
from rdkit import Chem


# ============================================================================
# Helper function tests: _integrate_universal_prefixes
# ============================================================================


class TestIntegrateUniversalPrefixes:
    """Tests for the _integrate_universal_prefixes helper in composer.py."""

    def _get_helper(self):
        """Import the helper function."""
        from orthonym.assembly.composer import _integrate_universal_prefixes
        return _integrate_universal_prefixes

    def test_chain_parent_with_methyl(self):
        """Chain parent (propane) with a methyl substituent at C-2 produces '2-methyl' prefix."""
        helper = self._get_helper()
        # 2-methylpropane: CC(C)C
        mol = Chem.MolFromSmiles("CC(C)C")
        assert mol is not None
        # Principal chain: atoms 0, 1, 3 (the longest chain C-C-C)
        # Atom 2 is the methyl substituent at C-2
        # Find the chain: index 0, 1, 3
        principal_chain = [0, 1, 3]
        parent_atoms = set(principal_chain)

        result = helper(
            mol, parent_atoms,
            parent_type="chain",
            principal_chain=principal_chain,
        )
        assert "methyl" in result
        assert "2" in result

    def test_ring_parent_with_chloro(self):
        """Ring parent (cyclohexane) with Cl substituent produces locanted prefix."""
        helper = self._get_helper()
        # Chlorocyclohexane: ClC1CCCCC1
        mol = Chem.MolFromSmiles("ClC1CCCCC1")
        assert mol is not None
        # Ring atoms: 1, 2, 3, 4, 5, 6
        ring_atoms = set([1, 2, 3, 4, 5, 6])
        oriented_ring = [1, 2, 3, 4, 5, 6]

        result = helper(
            mol, ring_atoms,
            parent_type="ring",
            oriented_ring=oriented_ring,
        )
        assert "chloro" in result

    def test_no_substituents_returns_empty(self):
        """Parent with no substituents returns empty string."""
        helper = self._get_helper()
        # Propane: CCC
        mol = Chem.MolFromSmiles("CCC")
        assert mol is not None
        principal_chain = [0, 1, 2]
        parent_atoms = set(principal_chain)

        result = helper(
            mol, parent_atoms,
            parent_type="chain",
            principal_chain=principal_chain,
        )
        assert result == ""

    def test_exclude_atoms_skips_excluded(self):
        """Excluded atoms are not discovered as substituents."""
        helper = self._get_helper()
        # Acetyl chloride: CC(=O)Cl
        mol = Chem.MolFromSmiles("CC(=O)Cl")
        assert mol is not None
        # Chain: atoms 0, 1 (C-C chain)
        # Atom 2 = O (=O), Atom 3 = Cl
        # Both are consumed by acid halide -- exclude them
        principal_chain = [0, 1]
        parent_atoms = set(principal_chain)
        exclude_atoms = {2, 3}  # O and Cl

        result = helper(
            mol, parent_atoms,
            parent_type="chain",
            principal_chain=principal_chain,
            exclude_atoms=exclude_atoms,
        )
        assert result == ""

    def test_multiple_substituents_alphabetical(self):
        """Multiple different substituents are alphabetically sorted."""
        helper = self._get_helper()
        # 2-chloro-3-methylbutane: CC(Cl)C(C)C -- but let's use a cleaner case
        # 1-bromo-2-chloroethane: BrCCCl
        mol = Chem.MolFromSmiles("BrCCCl")
        assert mol is not None
        # Chain: atoms 1, 2
        principal_chain = [1, 2]
        parent_atoms = set(principal_chain)

        result = helper(
            mol, parent_atoms,
            parent_type="chain",
            principal_chain=principal_chain,
        )
        # Should contain both bromo and chloro, with bromo before chloro alphabetically
        assert "bromo" in result
        assert "chloro" in result
        bromo_pos = result.index("bromo")
        chloro_pos = result.index("chloro")
        assert bromo_pos < chloro_pos, f"bromo should come before chloro alphabetically, got: {result}"


# ============================================================================
# Acid halide handler tests (Task 2)
# ============================================================================


class TestAcidHalideRetrofit:
    """Tests for acid halide handler with universal pipeline."""

    def test_3_methylbutanoyl_chloride(self):
        """3-methylbutanoyl chloride: CC(C)CC(=O)Cl"""
        from orthonym.namer import name_compound
        result = name_compound("CC(C)CC(=O)Cl")
        assert result is not None
        assert "methyl" in result.lower(), f"Expected 'methyl' in '{result}'"
        assert "butanoyl" in result.lower(), f"Expected 'butanoyl' in '{result}'"
        assert "chloride" in result.lower(), f"Expected 'chloride' in '{result}'"

    def test_2_chloroacetyl_bromide(self):
        """2-chloroacetyl bromide: ClCC(=O)Br -- halogen substituent on chain."""
        from orthonym.namer import name_compound
        result = name_compound("ClCC(=O)Br")
        assert result is not None
        assert "chloro" in result.lower(), f"Expected 'chloro' in '{result}'"
        assert "bromide" in result.lower(), f"Expected 'bromide' in '{result}'"


# ============================================================================
# Lactone handler tests (Task 2)
# ============================================================================


class TestLactoneRetrofit:
    """Tests for lactone handler with universal pipeline."""

    def test_3_methyloxolan_2_one(self):
        """3-methyloxolan-2-one (gamma-butyrolactone with methyl): CC1CCOC1=O"""
        from orthonym.namer import name_compound
        result = name_compound("CC1CCOC1=O")
        assert result is not None
        assert "methyl" in result.lower(), f"Expected 'methyl' in '{result}'"
        assert "oxolan" in result.lower(), f"Expected 'oxolan' in '{result}'"

    def test_5_ethyloxan_2_one(self):
        """5-ethyloxan-2-one (delta-valerolactone with ethyl): CCC1CCOC(=O)C1"""
        from orthonym.namer import name_compound
        result = name_compound("CCC1CCOC(=O)C1")
        assert result is not None
        assert "ethyl" in result.lower(), f"Expected 'ethyl' in '{result}'"
        assert "oxan" in result.lower(), f"Expected 'oxan' in '{result}'"


# ============================================================================
# Lactam handler tests (Task 2)
# ============================================================================


class TestLactamRetrofit:
    """Tests for lactam handler with universal pipeline."""

    def test_1_methylpyrrolidin_2_one(self):
        """1-methylpyrrolidin-2-one (N-methylpyrrolidone): CN1CCCC1=O"""
        from orthonym.namer import name_compound
        result = name_compound("CN1CCCC1=O")
        assert result is not None
        assert "methyl" in result.lower(), f"Expected 'methyl' in '{result}'"
        assert "pyrrolidin" in result.lower(), f"Expected 'pyrrolidin' in '{result}'"

    def test_3_ethylpiperidin_2_one(self):
        """3-ethylpiperidin-2-one: CCC1CCCNC1=O"""
        from orthonym.namer import name_compound
        result = name_compound("CCC1CCCNC1=O")
        assert result is not None
        assert "ethyl" in result.lower(), f"Expected 'ethyl' in '{result}'"
        assert "piperidin" in result.lower(), f"Expected 'piperidin' in '{result}'"


# ============================================================================
# Ether bare oxy fix tests (Task 2)
# ============================================================================


class TestEtherOxyFix:
    """Tests for ether naming -- no bare 'oxy' prefix."""

    def test_simple_methoxy(self):
        """Methoxy: COc1ccccc1 -> methoxybenzene (anisole retained)."""
        from orthonym.decomposition.fragment_assembly import _alcohol_to_alkoxy
        result = _alcohol_to_alkoxy("methanol")
        assert result == "methoxy"

    def test_simple_ethoxy(self):
        """Ethoxy: ethanol -> ethoxy."""
        from orthonym.decomposition.fragment_assembly import _alcohol_to_alkoxy
        result = _alcohol_to_alkoxy("ethanol")
        assert result == "ethoxy"

    def test_no_bare_oxy(self):
        """_alcohol_to_alkoxy never returns bare 'oxy' for organic fragments."""
        from orthonym.decomposition.fragment_assembly import _alcohol_to_alkoxy
        # Test complex alcohol names that might fail
        test_names = ["cyclopentanol", "cyclohexanol", "phenol"]
        for name in test_names:
            result = _alcohol_to_alkoxy(name)
            if result is not None:
                assert result != "oxy", f"Bare 'oxy' returned for '{name}'"
                assert result.endswith("oxy"), f"Expected alkoxy form for '{name}', got '{result}'"


# ============================================================================
# Ester handler tests (Plan 02, Task 1)
# ============================================================================


class TestEsterRetrofit:
    """Tests for ester handler with universal pipeline (acid-side substituents)."""

    def test_methyl_3_methylbutanoate(self):
        """Branched acid chain: methyl 3-methylbutanoate -- CC(C)CC(=O)OC."""
        from orthonym.namer import name_compound
        result = name_compound("CC(C)CC(=O)OC")
        assert result is not None
        assert "methyl" in result.lower(), f"Expected 'methyl' prefix in '{result}'"
        assert "butanoate" in result.lower(), f"Expected 'butanoate' in '{result}'"
        # The 3-methyl substituent on the acid chain must be present
        assert "3-methyl" in result.lower(), f"Expected '3-methyl' locanted prefix in '{result}'"

    def test_methyl_2_chloropropanoate(self):
        """Halogen substituent on acid chain: methyl 2-chloropropanoate."""
        from orthonym.namer import name_compound
        result = name_compound("CC(Cl)C(=O)OC")
        assert result is not None
        assert "chloro" in result.lower(), f"Expected 'chloro' in '{result}'"
        assert "propanoate" in result.lower(), f"Expected 'propanoate' in '{result}'"

    def test_simple_ethyl_acetate_no_regression(self):
        """Simple ester without substituents: ethyl acetate still works."""
        from orthonym.namer import name_compound
        result = name_compound("CC(=O)OCC")
        assert result is not None
        assert "ethyl" in result.lower(), f"Expected 'ethyl' in '{result}'"
        assert "acetate" in result.lower(), f"Expected 'acetate' in '{result}'"

    def test_methyl_propanoate_no_regression(self):
        """Simple 3-carbon acid ester: methyl propanoate still works."""
        from orthonym.namer import name_compound
        result = name_compound("CCC(=O)OC")
        assert result is not None
        assert "propanoate" in result.lower(), f"Expected 'propanoate' in '{result}'"

    def test_methyl_2_ethylbutanoate(self):
        """Multiple substituents on acid chain: methyl 2-ethylbutanoate."""
        from orthonym.namer import name_compound
        result = name_compound("CCC(CC)C(=O)OC")
        assert result is not None
        assert "ethyl" in result.lower(), f"Expected 'ethyl' in '{result}'"
        assert "butanoate" in result.lower(), f"Expected 'butanoate' in '{result}'"


# ============================================================================
# Amide handler tests (Plan 02, Task 1)
# ============================================================================


class TestAmideRetrofit:
    """Tests for amide handler with correct prefix locant joining."""

    def test_n_methylacetamide_no_regression(self):
        """Simple N-substituted amide: N-methylacetamide."""
        from orthonym.namer import name_compound
        result = name_compound("CC(=O)NC")
        assert result is not None
        assert result.lower() == "n-methylacetamide", f"Expected 'N-methylacetamide', got '{result}'"

    def test_2_methylbutanamide(self):
        """Chain substituent on saturated amide: 2-methylbutanamide."""
        from orthonym.namer import name_compound
        result = name_compound("CCC(C)C(=O)N")
        assert result is not None
        assert "methyl" in result.lower(), f"Expected 'methyl' in '{result}'"
        assert "butanamide" in result.lower(), f"Expected 'butanamide' in '{result}'"
        # Must NOT have double locant (e.g., "2-2-methyl")
        assert "2-2-" not in result, f"Double locant found in '{result}'"

    def test_2_ethylpentanamide(self):
        """Chain substituent on longer saturated amide: 2-ethylpentanamide."""
        from orthonym.namer import name_compound
        result = name_compound("CCCC(CC)C(=O)N")
        assert result is not None
        assert "ethyl" in result.lower(), f"Expected 'ethyl' in '{result}'"
        assert "pentanamide" in result.lower(), f"Expected 'pentanamide' in '{result}'"
        assert "2-2-" not in result, f"Double locant found in '{result}'"

    def test_nndimethylacetamide_no_regression(self):
        """N,N-disubstituted amide: N,N-dimethylacetamide."""
        from orthonym.namer import name_compound
        result = name_compound("CC(=O)N(C)C")
        assert result is not None
        assert result.lower() == "n,n-dimethylacetamide", f"Expected 'N,N-dimethylacetamide', got '{result}'"

    def test_2_hydroxypropanamide(self):
        """Hydroxy substituent on saturated amide: 2-hydroxypropanamide."""
        from orthonym.namer import name_compound
        result = name_compound("CC(O)C(=O)N")
        assert result is not None
        assert "hydroxy" in result.lower(), f"Expected 'hydroxy' in '{result}'"
        assert "propanamide" in result.lower(), f"Expected 'propanamide' in '{result}'"


# ============================================================================
# Polyfunctional ring-as-parent handler tests (Plan 02, Task 2)
# ============================================================================


class TestPolyfunctionalRingAsParent:
    """Tests for polyfunctional ring-as-parent path ."""

    def test_ring_as_parent_function_exists(self):
        """_name_ring_as_parent_polyfunctional is importable."""
        from orthonym.rules.polyfunctional import _name_ring_as_parent_polyfunctional
        assert callable(_name_ring_as_parent_polyfunctional)

    def test_4_aminocyclohexanol(self):
        """Ring-as-parent: 4-aminocyclohexan-1-ol (amino + alcohol on cyclohexane)."""
        from orthonym.namer import name_compound
        result = name_compound("OC1CCC(N)CC1")
        assert result is not None
        assert "amino" in result.lower(), f"Expected 'amino' in '{result}'"
        assert "cyclohexan" in result.lower(), f"Expected 'cyclohexan' in '{result}'"
        assert "ol" in result.lower(), f"Expected '-ol' suffix in '{result}'"

    def test_4_hydroxycyclohexanone(self):
        """Ring-as-parent: 4-hydroxycyclohexan-1-one (hydroxy + ketone on cyclohexane)."""
        from orthonym.namer import name_compound
        result = name_compound("OC1CCC(=O)CC1")
        assert result is not None
        assert "hydroxy" in result.lower(), f"Expected 'hydroxy' in '{result}'"
        assert "cyclohexan" in result.lower(), f"Expected 'cyclohexan' in '{result}'"
        assert "one" in result.lower(), f"Expected '-one' suffix in '{result}'"

    def test_chain_polyfunctional_unchanged(self):
        """Chain-as-parent polyfunctional still works: 4-aminobutanoic acid (GABA)."""
        from orthonym.namer import name_compound
        result = name_compound("NCCCC(=O)O")
        assert result is not None
        assert "amino" in result.lower(), f"Expected 'amino' in '{result}'"
        assert "butanoic" in result.lower(), f"Expected 'butanoic' in '{result}'"
        assert "acid" in result.lower(), f"Expected 'acid' in '{result}'"

    def test_complex_sphingolipid_no_ring_intercept(self):
        """Large molecule with small ring: ring-as-parent should NOT intercept.

        Sphingolipids have a cyclohexane ring but the chain dominates.
        The ring-as-parent guard must reject compounds where the ring
        is < 35% of total heavy atoms.
        """
        from orthonym.namer import name_compound
        smiles = (
            "CCCCCCCCCCCCCCCCCCCCCC[C@H](O)C(=O)N[C@@H]"
            "(COP(=O)(O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)"
            "[C@H](O)[C@H]1O)C(CCCCCCCCCCCCCCC)"
            "/C=C/CCCCCCCCCCCCC"
        )
        result = name_compound(smiles)
        assert result is not None
        # Must NOT contain "cyclohexane" as parent (ring is too small relative to total)
        assert "aminocyclohexane" not in result.lower(), (
            f"Ring-as-parent should not intercept sphingolipid, got: '{result}'"
        )


# ============================================================================
# Benzene handler retrofit tests (Plan 03, Task 1)
# ============================================================================


class TestBenzeneRetrofit:
    """Tests for benzene handler with universal C-substituent fallback.

    a phase Plan 03: _identify_substituent no longer returns None
    for complex C-substituents (<=10 atoms, non-carbonyl). Uses
    name_substituent from universal pipeline as fallback.
    """

    def test_complex_c_sub_not_none(self):
        """_identify_substituent returns non-None for complex C-substituents."""
        from rdkit import Chem
        from orthonym.rules.benzene import _identify_substituent

        # Cyanomethyl on benzene
        mol = Chem.MolFromSmiles("c1ccc(CC#N)cc1")
        ring_atoms = set()
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if len(ring) == 6 and all(
                mol.GetAtomWithIdx(i).GetIsAromatic() and
                mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                for i in ring
            ):
                ring_atoms = set(ring)
                break

        for ra in ring_atoms:
            for nbr in mol.GetAtomWithIdx(ra).GetNeighbors():
                if nbr.GetIdx() not in ring_atoms and nbr.GetSymbol() == 'C':
                    result = _identify_substituent(mol, nbr.GetIdx(), ring_atoms)
                    assert result is not None, (
                        "Complex C-substituent (cyanomethyl) returned None"
                    )
                    assert 'name' in result
                    assert 'atoms' in result
                    return
        pytest.fail("Did not find C-substituent on benzene ring")

    def test_carbamoylmethyl_on_benzene(self):
        """Carbamoylmethyl on benzene is identified (not None)."""
        from rdkit import Chem
        from orthonym.rules.benzene import _identify_substituent

        mol = Chem.MolFromSmiles("c1ccc(CC(=O)N)cc1")
        ring_atoms = set()
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if len(ring) == 6 and all(
                mol.GetAtomWithIdx(i).GetIsAromatic() and
                mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                for i in ring
            ):
                ring_atoms = set(ring)
                break

        for ra in ring_atoms:
            for nbr in mol.GetAtomWithIdx(ra).GetNeighbors():
                if nbr.GetIdx() not in ring_atoms and nbr.GetSymbol() == 'C':
                    result = _identify_substituent(mol, nbr.GetIdx(), ring_atoms)
                    assert result is not None, "Carbamoylmethyl returned None"
                    return
        pytest.fail("Did not find C-substituent on benzene ring")

    def test_benzoic_acid_no_double_counting(self):
        """Benzoic acid remains 'benzoic acid' -- no 'carboxy' prefix."""
        from orthonym.namer import name_compound
        result = name_compound("OC(=O)c1ccccc1")
        assert result is not None
        assert result.lower() == "benzoic acid", f"Expected 'benzoic acid', got '{result}'"

    def test_benzamide_no_double_counting(self):
        """Benzamide remains 'benzamide' -- no 'carbamoyl' prefix."""
        from orthonym.namer import name_compound
        result = name_compound("NC(=O)c1ccccc1")
        assert result is not None
        assert result.lower() == "benzamide", f"Expected 'benzamide', got '{result}'"

    def test_benzonitrile_no_double_counting(self):
        """Benzonitrile remains 'benzonitrile' -- no 'cyano' prefix."""
        from orthonym.namer import name_compound
        result = name_compound("N#Cc1ccccc1")
        assert result is not None
        assert result.lower() == "benzonitrile", f"Expected 'benzonitrile', got '{result}'"

    def test_benzaldehyde_no_double_counting(self):
        """Benzaldehyde remains 'benzaldehyde' -- no 'formyl' prefix."""
        from orthonym.namer import name_compound
        result = name_compound("O=Cc1ccccc1")
        assert result is not None
        assert result.lower() == "benzaldehyde", f"Expected 'benzaldehyde', got '{result}'"

    def test_toluene_unchanged(self):
        """Toluene still works (simple alkyl)."""
        from orthonym.namer import name_compound
        result = name_compound("Cc1ccccc1")
        assert result is not None
        assert result.lower() == "toluene", f"Expected 'toluene', got '{result}'"

    def test_chlorobenzene_unchanged(self):
        """Chlorobenzene still works (halogen)."""
        from orthonym.namer import name_compound
        result = name_compound("Clc1ccccc1")
        assert result is not None
        assert result.lower() == "chlorobenzene", f"Expected 'chlorobenzene', got '{result}'"

    def test_4_methylbenzoic_acid_unchanged(self):
        """4-methylbenzoic acid retains methyl substituent."""
        from orthonym.namer import name_compound
        result = name_compound("Cc1ccc(C(=O)O)cc1")
        assert result is not None
        assert "methyl" in result.lower(), f"Expected 'methyl' in '{result}'"
        assert "benzoic acid" in result.lower(), f"Expected 'benzoic acid' in '{result}'"

    def test_4_nitrobenzamide_unchanged(self):
        """4-nitrobenzamide retains nitro substituent."""
        from orthonym.namer import name_compound
        result = name_compound("O=C(N)c1ccc([N+](=O)[O-])cc1")
        assert result is not None
        assert "nitro" in result.lower(), f"Expected 'nitro' in '{result}'"
        assert "benzamide" in result.lower(), f"Expected 'benzamide' in '{result}'"

    def test_ester_on_benzene_not_named_as_sub(self):
        """Ester carbonyl directly on benzene returns None (handled by ester handler)."""
        from rdkit import Chem
        from orthonym.rules.benzene import _identify_substituent

        # Methyl benzoate: COC(=O)c1ccccc1
        mol = Chem.MolFromSmiles("COC(=O)c1ccccc1")
        ring_atoms = set()
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if len(ring) == 6 and all(
                mol.GetAtomWithIdx(i).GetIsAromatic() and
                mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                for i in ring
            ):
                ring_atoms = set(ring)
                break

        # Find the ester C (has =O)
        for ra in ring_atoms:
            for nbr in mol.GetAtomWithIdx(ra).GetNeighbors():
                if nbr.GetIdx() not in ring_atoms and nbr.GetSymbol() == 'C':
                    result = _identify_substituent(mol, nbr.GetIdx(), ring_atoms)
                    # Ester carbonyl should return None (carbonyl guard)
                    assert result is None, (
                        f"Ester carbonyl on benzene should return None, got {result}"
                    )
                    return
        pytest.fail("Did not find ester C on benzene ring")

    def test_succinimide_not_collapsed_to_carboxamide(self):
        """N-phenylsuccinimide: the imide ring is the parent, never collapsed to 'carboxamide'.

        a phase Plan 03 requirement: when a succinimide ring is attached to
        benzene via N, it must not be collapsed to a 'carboxamide' suffix. The
        benzene handler's helper now declines the N-linked imide ring (returns None
        and falls through) because the ring that carries the principal characteristic
        group is the parent, not benzene: IMIDES (the Blue Book)
        '1-bromopyrrolidine-2,5-dione (PIN) (not N-bromosuccinimide; substitution is
        not allowed on succinimide)' and:33851 '2-phenyl-1H-isoindole-1,3(2H)-dione
        (PIN)... N-phenylphthalimide'. So no 'succinimidyl' prefix exists to be
        identified, and the whole-molecule name pins the outcome.
        """
        from rdkit import Chem
        from orthonym.rules.benzene import _identify_substituent
        from orthonym.namer import name_compound

        mol = Chem.MolFromSmiles("O=C1CCC(=O)N1c1ccccc1")
        ring_atoms = set()
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if len(ring) == 6 and all(
                mol.GetAtomWithIdx(i).GetIsAromatic() and
                mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                for i in ring
            ):
                ring_atoms = set(ring)
                break

        for ra in ring_atoms:
            for nbr in mol.GetAtomWithIdx(ra).GetNeighbors():
                if nbr.GetIdx() not in ring_atoms and nbr.GetSymbol() == 'N':
                    result = _identify_substituent(mol, nbr.GetIdx(), ring_atoms)
                    # Must NOT be 'carboxamide' (suffix collapse); None = declined.
                    assert result is None or result['name'] != 'carboxamide', (
                        f"Succinimide collapsed to 'carboxamide': {result}"
                    )
                    assert name_compound("O=C1CCC(=O)N1c1ccccc1") == (
                        "1-phenylpyrrolidine-2,5-dione")
                    return
        pytest.fail("Did not find N-substituent on benzene ring")

    def test_isobutylbenzene_simple_alkyl(self):
        """Isobutylbenzene: branched alkyl correctly named (existing path)."""
        from orthonym.namer import name_compound
        result = name_compound("CC(C)Cc1ccccc1")
        assert result is not None
        # 'Retained prefixes no longer recommended as approved prefixes'
        # (the Blue Book): '2-methylpropyl (preferred prefix) (not isobutyl)' (:16412).
        assert result == "(2-methylpropyl)benzene", f"Expected '(2-methylpropyl)benzene', got '{result}'"

    def test_benzoyl_chloride_no_double_counting(self):
        """Benzoyl chloride: suffix FG correctly handled, no carbonyl prefix."""
        from orthonym.namer import name_compound
        result = name_compound("ClC(=O)c1ccccc1")
        assert result is not None
        assert "benzene" in result.lower() or "benzoyl" in result.lower(), (
            f"Expected benzene-based name, got '{result}'"
        )


# ============================================================================
# a phase Final Accounting: All Early-Return Handlers in assemble_name
# ============================================================================
#
# Category A — Safe as-is (handler returns None for complex cases → fallthrough):
# Line 547: Oxime (_name_oxime_or_hydrazone) — recursive name_compound
# Line 553: Hydrazone (_name_oxime_or_hydrazone) — recursive name_compound
# Line 561: N-oxide (_try_name_n_oxide) — recursive name_compound
# Line 569: Isocyanate (_name_isocyanate) — returns None for complex R
# Line 577: Isothiocyanate (_name_isothiocyanate) — returns None for complex R
# Line 585: Carbamic acid (_name_carbamic_acid) — retained name, N-subs only
# Line 593: Carbamate (_name_carbamate) — returns None for complex R
# Line 602: Urea (_try_name_urea) — retained name, N-subs only
# Line 610: Guanidine (_try_name_guanidine) — retained name, N-subs only
# Line 629: Anhydride (name_anhydride) — dedicated module, full naming
# Line 675: Ring-attached ester (_assemble_ring_with_ester_prefixes) — own prefix gen
# Line 697: Multi-ester (dicarboxylic/polyol) — dedicated module, full naming
# Line 725: Sulfoxide (name_sulfoxide) — returns None for complex R
# Line 733: Sulfone (name_sulfone) — returns None for complex R
# Line 742: Thioether/Sulfide (name_sulfide) — returns None for complex R
# Line 762: Phosphine oxide (name_phosphine_oxide) — returns None for complex R
# Line 771: Phosphate ester (name_phosphate_ester) — dedicated module
# Line 786: Phosphine (name_phosphine) — returns None for complex R
# Line 825: Phosphinic acid (name_phosphinic_acid) — dedicated module
# Line 834: Boronic acid (_name_boronic_acid) — returns None for complex R
# Line 841: Ring assembly (name_ring_assembly) — own substituent handling
# Line 872: Complex ring (_assemble_complex_ring_name) — own substituent pipeline
# Line 890: Polycyclic (_assemble_polycyclic_name) — own prefix generation
# Line 898: Partial sat carbocycle — own naming pipeline
# Line 910: Heterocycle (_assemble_heterocycle_name) — own prefix generation
# Line 982: Simple molecule (_name_simple_molecule) — single-atom/trivial
#
# Category B — Retrofitted (Plans 86-01, 86-02, 86-03):
# Line 619: Acid halide (name_acid_halide) — 86-01: universal pipeline for chain subs
# Line 642: Lactone (name_monocyclic_lactone) — 86-01: universal pipeline for ring subs
# Line 660: Lactam (name_monocyclic_lactam) — 86-01: universal pipeline for ring subs
# Ether (_alcohol_to_alkoxy fallback) — 86-01: name_substituent fallback for bare oxy
# Line 713: Ester (name_ester) — 86-02: universal pipeline for acid-side subs
# Line 970: Amide (_assemble_amide_name) — 86-02: fixed double-locant bug
# Line 684: Polyfunctional (name_polyfunctional) — 86-02: ring-as-parent path
# Line 929: Benzene (_assemble_benzene_name) — 86-03: universal fallback for complex C-subs
# Line 963: Ring nitrile (_assemble_ring_nitrile_name) — 86-03: universal pipeline for ring subs
#
# Category C — Verified safe (audit in 86-03 Task 2):
# Line 530: Ion (assemble_ion_name) — own naming pipeline
# Line 976: Amine (_assemble_amine_name) — extensive R-group handling (phenyl, fused het, ring, alkyl)
#
# Functional class handlers (isocyanate, isothiocyanate, carbamic acid, carbamate,
# urea, guanidine, boronic acid) use _name_r_group which has universal pipeline
# fallback (lines 1489-1500) for complex R-groups that can't be named by simple
# alkyl/phenyl/benzyl classification.
#
# All handlers either: (1) successfully name ALL atoms, or (2) return None → fallthrough.
# No handler silently drops substituents.
# ============================================================================


# ============================================================================
# Ring nitrile handler retrofit tests (Plan 03, Task 2)
# ============================================================================


class TestRingNitrileRetrofit:
    """Tests for ring nitrile handler with universal substituent discovery.

    a phase-03: _assemble_ring_nitrile_name now uses
    _integrate_universal_prefixes to discover ring substituents.
    Previously, ALL substituents on substituted ring nitriles were silently dropped.
    """

    def test_methylcyclohexanecarbonitrile(self):
        """Methyl substituent on ring nitrile is now included."""
        from orthonym.namer import name_compound
        result = name_compound("CC1(C#N)CCCCC1")
        assert result is not None
        assert "methyl" in result.lower(), f"Expected 'methyl' in '{result}'"
        assert "cyclohexane" in result.lower() or "carbonitrile" in result.lower(), (
            f"Expected ring nitrile name, got '{result}'"
        )

    def test_chlorocyclohexanecarbonitrile(self):
        """Chloro substituent on ring nitrile is included."""
        from orthonym.namer import name_compound
        result = name_compound("ClC1CCC(C#N)CC1")
        assert result is not None
        assert "chloro" in result.lower(), f"Expected 'chloro' in '{result}'"
        assert "carbonitrile" in result.lower(), (
            f"Expected 'carbonitrile' in '{result}'"
        )

    def test_unsubstituted_cyclohexanecarbonitrile_no_regression(self):
        """Unsubstituted cyclohexanecarbonitrile still works."""
        from orthonym.namer import name_compound
        result = name_compound("C1CCCCC1C#N")
        assert result is not None
        assert result.lower() == "cyclohexanecarbonitrile", (
            f"Expected 'cyclohexanecarbonitrile', got '{result}'"
        )

    def test_unsubstituted_cyclopentanecarbonitrile_no_regression(self):
        """Unsubstituted cyclopentanecarbonitrile still works."""
        from orthonym.namer import name_compound
        result = name_compound("C1CCCC1C#N")
        assert result is not None
        assert result.lower() == "cyclopentanecarbonitrile", (
            f"Expected 'cyclopentanecarbonitrile', got '{result}'"
        )


# ============================================================================
# Category C handler safety tests (Plan 03, Task 2)
# ============================================================================


class TestCategoryCHandlerSafety:
    """Verify Category C handlers use safe fallthrough pattern.

    These handlers return None when they can't fully name a molecule,
    allowing the general pipeline to handle it. This test class verifies
    the pattern works for representative compounds.

    a phase-03 audit findings:
    - Functional class handlers (isocyanate, isothiocyanate, carbamic acid,
      carbamate, urea, guanidine, boronic acid) all use _name_r_group which
      has a universal pipeline fallback for complex R-groups.
    - Sulfoxide, sulfone, thioether handlers return None for complex R -> fallthrough.
    - Phosphorus handlers have dedicated modules with full naming.
    - Amine handler has extensive R-group handling (phenyl, fused het, ring, alkyl).
    - Ring nitrile was retrofitted (previously dropped all substituents).
    """

    def test_sulfide_complex_r_falls_through(self):
        """Sulfide handler returns None for complex R-groups, molecule falls through."""
        from orthonym import name_compound
        # Methyl(cyclopentyl) sulfide -- cyclopentyl is complex for simple alkyl naming
        result = name_compound("C1CCCC1SC")
        assert result is not None  # Falls through to general naming

    def test_isocyanate_simple_works(self):
        """Isocyanate handler correctly names simple R-groups."""
        from orthonym import name_compound
        result = name_compound("CN=C=O")
        # 'ISOCYANATES' (the Blue Book): "Preferred IUPAC names are
        # generated substitutively using the prefix isocyanato"; the functional class
        # name 'methyl isocyanate' is the previous recommendation (cf.:26007
        # 'isocyanatocyclohexane (PIN) cyclohexyl isocyanate').
        assert result == "isocyanatomethane"

    def test_isothiocyanate_simple_works(self):
        """Isothiocyanate handler correctly names simple R-groups."""
        from orthonym import name_compound
        result = name_compound("CN=C=S")
        # 'ISOCYANATES' covers the chalcogen analogues: ':26009 isothiocyanatobenzene
        # (PIN) phenyl isothiocyanate'.
        assert result == "isothiocyanatomethane"

    def test_boronic_acid_simple_works(self):
        """Boronic acid handler names simple R-groups correctly."""
        from orthonym import name_compound
        result = name_compound("CB(O)O")
        assert "boronic acid" in result.lower()

    def test_urea_unsubstituted(self):
        """Urea handler produces retained name."""
        from orthonym import name_compound
        result = name_compound("NC(=O)N")
        assert result is not None

    def test_urea_n_substituted(self):
        """Urea handler correctly names N-substituted ureas."""
        from orthonym import name_compound
        result = name_compound("CN(C)C(=O)N")
        assert result is not None
        assert "dimethyl" in result.lower(), f"Expected 'dimethyl' in '{result}'"
        assert "urea" in result.lower(), f"Expected 'urea' in '{result}'"

    def test_amine_n_methyl(self):
        """Amine handler produces N-methyl prefix."""
        from orthonym import name_compound
        result = name_compound("CNCC")
        assert result is not None
        assert "methyl" in result.lower()

    def test_carbamic_acid_retained(self):
        """Carbamic acid handler produces retained name."""
        from orthonym import name_compound
        result = name_compound("NC(=O)O")
        assert result is not None
        assert "carbamic" in result.lower() or "amino" in result.lower(), (
            f"Expected carbamic-based name, got '{result}'"
        )

    def test_n_oxide_pyridine(self):
        """N-oxide handler correctly names pyridine 1-oxide."""
        from orthonym import name_compound
        result = name_compound("[O-][n+]1ccccc1")
        assert result is not None
        assert "oxide" in result.lower(), f"Expected 'oxide' in '{result}'"

    def test_sulfoxide_simple(self):
        """Sulfoxide handler names simple dimethyl sulfoxide."""
        from orthonym import name_compound
        result = name_compound("CS(=O)C")
        assert result is not None
        # 'SULFOXIDES AND SULFONES' (the Blue Book): "Methods (1) and (3)
        # generate preferred names." -- (1) substitutively, by prefixing the acyl group
        # R'-SO-; (2) functional class 'sulfoxide' is not a PIN method.
        # '(methanesulfinyl)methane (PIN)' is printed at:46154.
        assert result == "(methanesulfinyl)methane", f"Expected '(methanesulfinyl)methane', got '{result}'"

    def test_sulfone_simple(self):
        """Sulfone handler names simple dimethyl sulfone."""
        from orthonym import name_compound
        result = name_compound("CS(=O)(=O)C")
        assert result is not None
        # 'SULFOXIDES AND SULFONES' (the Blue Book): "Methods (1) and (3)
        # generate preferred names."; the functional class name 'sulfone' is method (2)
        # ('(ethanesulfonyl)ethane (PIN)... diethyl sulfone',:28118).
        assert result == "(methanesulfonyl)methane", f"Expected '(methanesulfonyl)methane', got '{result}'"

    def test_guanidine_retained(self):
        """Guanidine handler produces retained name."""
        from orthonym import name_compound
        result = name_compound("NC(=N)N")
        assert result is not None

    def test_oxime_functional_class(self):
        """Oxime handler uses functional class naming."""
        from orthonym import name_compound
        result = name_compound("CC(=NO)C")
        assert result is not None
        # 'Oximes' (under, the Blue Book): "Preferred IUPAC
        # names are formed substitutively as N-hydroxy derivatives of imines" (example
        #:38468 'N-hydroxypentan-2-imine (PIN)... pentan-2-one oxime'); the oxime name
        # is the functional class form.
        assert result == "N-hydroxypropan-2-imine", f"Expected 'N-hydroxypropan-2-imine', got '{result}'"

    def test_carbamate_functional_class(self):
        """Carbamate handler uses functional class naming."""
        from orthonym import name_compound
        result = name_compound("COC(=O)NC")
        assert result is not None
        assert "carbamate" in result.lower(), f"Expected 'carbamate' in '{result}'"
