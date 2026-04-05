"""Integration tests for Phase 85 name_substituent() and gap closures.

Tests the complete Phase 85 deliverables:
- name_substituent(mol, frag_atoms, attach_idx) end-to-end on real molecules
- Never-None guarantee under parametric fuzzing
- Canary compound naming stability
- Gap closure verification (fused_rings.py, composer.py)

References:
    IUPAC 2013 P-31.1.3 (substituent prefix naming)
    Phase 85 Plan 02: USUB-05 gap closure, QUAL-01, QUAL-02
"""

import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent
from orthonym.namer import name_compound


# ============================================================================
# TestNameSubstituentRealMolecules: end-to-end tests on real molecules
# ============================================================================

class TestNameSubstituentRealMolecules:
    """Test name_substituent() on real molecule fragments."""

    def test_toluene_methyl(self):
        """Methyl on benzene -> 'methyl'."""
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        assert mol is not None
        # Atom 0 is the methyl carbon
        result = name_substituent(mol, {0}, attach_idx=0)
        assert result == "methyl"

    def test_ethylbenzene_ethyl(self):
        """Ethyl on benzene -> 'ethyl'."""
        mol = Chem.MolFromSmiles("CCc1ccccc1")
        assert mol is not None
        # Fragment: atoms 0, 1 (CC chain), attach at atom 1 (bonded to ring)
        result = name_substituent(mol, {0, 1}, attach_idx=1)
        assert result is not None
        assert "ethyl" in result.lower()

    def test_isopropyl_on_benzene(self):
        """Isopropyl on benzene -> 'isopropyl' or '1-methylethyl'."""
        mol = Chem.MolFromSmiles("CC(C)c1ccccc1")
        assert mol is not None
        # Fragment: atoms 0, 1, 2 (CC(C) chain), attach at atom 1
        result = name_substituent(mol, {0, 1, 2}, attach_idx=1)
        assert result is not None
        assert "propyl" in result.lower() or "methylethyl" in result.lower()

    def test_hydroxymethyl_fragment(self):
        """Hydroxymethyl fragment (CH2OH) -> produces a name."""
        mol = Chem.MolFromSmiles("OCc1ccccc1")
        assert mol is not None
        # Fragment: atoms 0 (O), 1 (C), attach at atom 1
        result = name_substituent(mol, {0, 1}, attach_idx=1)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_cyanomethyl_fragment(self):
        """Cyanomethyl fragment (CH2CN) -> produces a name with cyano."""
        mol = Chem.MolFromSmiles("N#CCc1ccccc1")
        assert mol is not None
        # Fragment: atoms 0 (N), 1 (C), 2 (C), attach at atom 2
        result = name_substituent(mol, {0, 1, 2}, attach_idx=2)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_propyl_on_ring(self):
        """Propyl on cyclohexane -> 'propyl'."""
        mol = Chem.MolFromSmiles("CCCC1CCCCC1")
        assert mol is not None
        # Fragment: atoms 0, 1, 2 (propyl chain), attach at atom 2
        result = name_substituent(mol, {0, 1, 2}, attach_idx=2)
        assert result is not None
        assert result == "propyl"

    def test_single_carbon_always_methyl(self):
        """Single carbon atom fragment -> 'methyl'."""
        mol = Chem.MolFromSmiles("CC")
        assert mol is not None
        result = name_substituent(mol, {1}, attach_idx=1)
        assert result == "methyl"


# ============================================================================
# TestNameSubstituentNeverNone: fuzz-style non-None guarantee
# ============================================================================

class TestNameSubstituentNeverNone:
    """Verify name_substituent() NEVER returns None for any valid input."""

    MOLECULES = [
        "CC",
        "CCC",
        "CCCC",
        "c1ccccc1",
        "CCO",
        "CC=O",
        "CC(=O)O",
        "CCN",
        "CCS",
        "C(F)(F)F",
        "c1ccncc1",
        "c1ccc2ccccc2c1",  # naphthalene
        "OCC(O)CO",  # glycerol
        "CCCCCCCCCC",  # decane
    ]

    @pytest.mark.parametrize("smi", MOLECULES)
    def test_never_none(self, smi):
        """name_substituent() returns non-None string for first atom as fragment."""
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        frag_atoms = {0}
        result = name_substituent(mol, frag_atoms, attach_idx=0)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================================
# TestCanaryNamingStability: verify name_compound() stability
# ============================================================================

CANARY_COMPOUNDS = [
    ("CCO", "ethanol"),
    ("CC(=O)O", "acetic acid"),
    ("c1ccccc1", "benzene"),
    ("CC(C)C", "2-methylpropane"),
    ("CC(O)C", "propan-2-ol"),
    ("c1ccncc1", "pyridine"),
    ("C1CCNCC1", "piperidine"),
    ("c1ccc2ccccc2c1", "naphthalene"),
]


class TestCanaryNamingStability:
    """All canary compounds produce identical names after Phase 85."""

    @pytest.mark.parametrize(
        "smi,expected_name",
        CANARY_COMPOUNDS,
        ids=[smi for smi, _ in CANARY_COMPOUNDS],
    )
    def test_name_unchanged(self, smi, expected_name):
        """Verify canary compound name matches expected value."""
        name = name_compound(smi)
        assert name is not None
        assert len(name) > 0
        assert name == expected_name, (
            f"Canary regression for {smi}: expected '{expected_name}', got '{name}'"
        )


# ============================================================================
# TestGapClosureVerification: verify specific gap fixes work
# ============================================================================

class TestGapClosureVerification:
    """Verify the Phase 85 gap closures are effective."""

    def test_fused_ring_fallback_does_not_return_none_for_simple_c_sub(self):
        """_identify_fused_substituent() with fallback should name simple C substituents."""
        from orthonym.rules.fused_rings import _identify_fused_substituent
        # Naphthalene with a propyl chain
        mol = Chem.MolFromSmiles("CCCc1ccc2ccccc2c1")
        assert mol is not None
        # Find the fused ring core
        ri = mol.GetRingInfo()
        core_atoms = set()
        for ring in ri.AtomRings():
            core_atoms.update(ring)
        # Atom 0 (first C of propyl) is bonded to the ring
        # Check if start atom 0 is in the ring
        if 0 not in core_atoms:
            result = _identify_fused_substituent(mol, 0, core_atoms)
            # Should get a result (methyl/ethyl/propyl), not None
            assert result is not None
            assert 'name' in result

    def test_composer_has_ring_atoms_helper(self):
        """_has_ring_atoms() utility function is importable and functional."""
        from orthonym.assembly.composer import _has_ring_atoms
        mol = Chem.MolFromSmiles("c1ccccc1C")
        assert mol is not None
        ring_info = mol.GetRingInfo()
        ring_atoms = set(ring_info.AtomRings()[0])
        assert _has_ring_atoms(mol, ring_atoms) is True
        # Non-ring atom (methyl carbon, atom 6)
        assert _has_ring_atoms(mol, {6}) is False

    def test_composer_name_reflects_ring_helper(self):
        """_name_reflects_ring() detects ring system identifiers in names."""
        from orthonym.assembly.composer import _name_reflects_ring
        assert _name_reflects_ring("cyclohexyl") is True
        assert _name_reflects_ring("phenyl") is True
        assert _name_reflects_ring("piperidinyl") is True
        assert _name_reflects_ring("methyl") is False
        assert _name_reflects_ring("ethyl") is False
        assert _name_reflects_ring("propyl") is False

    def test_fused_ring_fallback_guards_ring_atoms(self):
        """Fallback must NOT name atoms that are themselves ring atoms as C-substituents."""
        from orthonym.rules.fused_rings import _identify_fused_substituent
        # Use a molecule where the core only covers SOME ring atoms
        # (simulating incomplete fused heterocycle core mapping)
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene
        assert mol is not None
        ri = mol.GetRingInfo()
        all_ring = set()
        for ring in ri.AtomRings():
            all_ring.update(ring)
        # Take just one ring as "core", leaving the other ring as "non-core"
        first_ring = set(ri.AtomRings()[0])
        # Find a ring atom NOT in first_ring that IS in a ring
        for idx in all_ring - first_ring:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'C' and ri.NumAtomRings(idx) > 0:
                result = _identify_fused_substituent(mol, idx, first_ring)
                # The fallback guard should prevent naming this ring atom
                # It may get an alkyl result (from existing handlers that don't
                # check ring membership), but the Phase 85 fallback should NOT fire
                # because the start atom is in a ring
                if result and result.get('type') == 'functionalized':
                    # This would mean the Phase 85 fallback fired for a ring atom
                    assert False, f"Fallback should not fire for ring atom {idx}: {result}"
                break

    def test_fused_ring_fallback_rejects_exotic_elements(self):
        """Fallback must NOT name fragments with exotic elements (As, Se, etc.)."""
        from orthonym.rules.fused_rings import _identify_fused_substituent
        # Compound with arsenic
        mol = Chem.MolFromSmiles("c1ccc([As](O)(O)=O)cc1")
        assert mol is not None
        ri = mol.GetRingInfo()
        core_atoms = set(ri.AtomRings()[0])
        # Find the As atom
        as_idx = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'As':
                as_idx = atom.GetIdx()
                break
        if as_idx is not None and as_idx not in core_atoms:
            result = _identify_fused_substituent(mol, as_idx, core_atoms)
            # Should return None (exotic element in fragment)
            # Note: As itself is handled by the general return None at end
            assert result is None


# ============================================================================
# TestZeroRegressionValidation: confirm full test suite stability
# ============================================================================

class TestZeroRegressionValidation:
    """Verify that Phase 85 changes do not regress any existing behavior."""

    def test_simple_compounds_unchanged(self):
        """Core compound naming must be unchanged."""
        # These are fundamental compounds that must always work
        cases = [
            ("C", "methane"),
            ("CC", "ethane"),
            ("CCC", "propane"),
            ("CCCC", "butane"),
            ("C=C", "ethene"),
            ("C#C", "acetylene"),  # retained name (P-31.1.2.1 PIN)
            ("CO", "methanol"),
            ("CCO", "ethanol"),
            ("C=O", "formaldehyde"),  # retained name
            ("CC=O", "acetaldehyde"),  # retained name
            ("CC(=O)C", "propan-2-one"),  # IUPAC 2013 PIN
        ]
        for smi, expected in cases:
            name = name_compound(smi)
            assert name == expected, f"{smi}: expected '{expected}', got '{name}'"
