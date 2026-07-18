"""
Unit tests for ring assembly detection and naming (IUPAC P-28).

Tests cover:
  - Detection of identical ring systems connected by single bonds
  - Naming with primed locant notation (1,1'-biphenyl format)
  - Substituted ring assemblies (4-chloro-1,1'-biphenyl)
  - Negative cases: fused rings, different ring types
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.perception.rings import get_ring_systems
from orthonym.rules.ring_assemblies import detect_ring_assembly, name_ring_assembly


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestDetection:
    """Tests for detect_ring_assembly()."""

    @pytest.mark.unit
    def test_biphenyl_detected(self):
        """Biphenyl (two identical benzene rings) is detected as assembly."""
        mol = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None
        assert info["count"] == 2
        assert info["ring_type"] == "carbocyclic"

    @pytest.mark.unit
    def test_bipyridine_detected(self):
        """Bipyridine (two identical pyridine rings) is detected as assembly."""
        mol = Chem.MolFromSmiles("c1ccncc1-c2ccncc2")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None
        assert info["count"] == 2
        assert info["ring_type"] == "heterocyclic"

    @pytest.mark.unit
    def test_bithiophene_detected(self):
        """Bithiophene (two identical thiophene rings) is detected as assembly."""
        mol = Chem.MolFromSmiles("c1ccsc1-c1ccsc1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None
        assert info["count"] == 2
        assert info["ring_type"] == "heterocyclic"

    @pytest.mark.unit
    def test_naphthalene_not_assembly(self):
        """Naphthalene (fused rings, 1 ring system) is NOT a ring assembly."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_phenylpyridine_not_assembly(self):
        """Phenylpyridine (different ring types) is NOT a ring assembly."""
        mol = Chem.MolFromSmiles("c1ccc(-c2ccccn2)cc1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_biphenylene_not_assembly(self):
        """Biphenylene (fused, shared atoms) is NOT a ring assembly."""
        mol = Chem.MolFromSmiles("c1ccc2c(c1)-c1ccccc12")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_phenylcyclohexane_not_assembly(self):
        """Phenylcyclohexane (different aromaticity) is NOT a ring assembly."""
        mol = Chem.MolFromSmiles("C1CCCCC1c1ccccc1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_phenanthrene_not_assembly(self):
        """Phenanthrene (fused tricyclic) is NOT a ring assembly."""
        mol = Chem.MolFromSmiles("c1ccc2c(c1)ccc1ccccc12")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_single_ring_not_assembly(self):
        """A single ring (benzene) is NOT a ring assembly."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_cyclooctenyl_cyclooctane_not_assembly(self):
        """W8-P11 leak fix: cyclooctene + cyclooctane (same ring size/
        aromaticity but DIFFERENT saturation) is NOT an identical-ring
        assembly. Pre-fix, ``_system_signature`` ignored ring double-bond
        count, so this pair falsely matched and the namer emitted
        '1,1'-bi(cyclooctene)' for a molecule where only ONE ring actually
        has the double bond -- a wrong-structure name that OPSIN's SELF-01
        gate caught (fails-OPEN without Java) but no source-level check did.
        """
        mol = Chem.MolFromSmiles(r"C1CCCCC/C=C\1C1CCCCCCC1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is None

    @pytest.mark.unit
    def test_bifuran_detected(self):
        """Bifuran (two identical furan rings) is detected as assembly."""
        mol = Chem.MolFromSmiles("c1ccoc1-c1ccoc1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None
        assert info["count"] == 2
        assert info["ring_type"] == "heterocyclic"


# ---------------------------------------------------------------------------
# Naming tests (E2E through name_compound)
# ---------------------------------------------------------------------------

class TestNaming:
    """Tests for ring assembly naming via name_compound()."""

    @pytest.mark.unit
    def test_biphenyl_name(self):
        """Biphenyl SMILES produces the PIN '1,1'-biphenyl' (F-T9/DD6 RET-01: bare
        'biphenyl' is general-only; the ring-assembly PIN is 1,1'-biphenyl, P-28.2.1)."""
        name = name_compound("c1ccc(-c2ccccc2)cc1")
        assert name == "1,1'-biphenyl"

    @pytest.mark.unit
    def test_bipyridine_contains_bipyridine(self):
        """Bipyridine SMILES produces a name containing 'bipyridine'."""
        name = name_compound("c1ccncc1-c2ccncc2")
        assert "bipyridine" in name

    @pytest.mark.unit
    def test_22_bipyridine(self):
        """2,2'-bipyridine SMILES produces '2,2'-bipyridine'."""
        name = name_compound("c1ccc(-c2ccccn2)nc1")
        assert name == "2,2'-bipyridine"

    @pytest.mark.unit
    def test_bithiophene_contains_bithiophene(self):
        """Bithiophene SMILES produces a name containing 'bithiophene'."""
        name = name_compound("c1ccsc1-c1ccsc1")
        assert "bithiophene" in name

    @pytest.mark.unit
    def test_4_chlorobiphenyl(self):
        """4-chlorobiphenyl SMILES produces '4-chloro-1,1'-biphenyl'."""
        name = name_compound("Clc1ccc(-c2ccccc2)cc1")
        assert name == "4-chloro-1,1'-biphenyl"

    @pytest.mark.unit
    def test_biphenyl_uses_phenyl_not_benzene(self):
        """Biphenyl assembly name uses 'phenyl', not 'benzene'."""
        name = name_compound("c1ccc(-c2ccccc2)cc1")
        assert "phenyl" in name
        assert "benzene" not in name

    @pytest.mark.unit
    def test_primed_locant_format(self):
        """Primed locants use ASCII apostrophe (U+0027).

        Uses substituted biphenyl since unsubstituted biphenyl is a retained name.
        """
        name = name_compound("Clc1ccc(-c2ccccc2)cc1")
        assert "'" in name  # ASCII apostrophe

    @pytest.mark.unit
    def test_bifuran_contains_bifuran(self):
        """Bifuran SMILES produces a name containing 'bifuran'."""
        name = name_compound("c1ccoc1-c1ccoc1")
        assert "bifuran" in name


# ---------------------------------------------------------------------------
# Boundary / negative naming tests
# ---------------------------------------------------------------------------

class TestNamingNegatives:
    """Tests that non-assemblies are NOT routed to assembly naming."""

    @pytest.mark.unit
    def test_phenanthrene_not_assembly_name(self):
        """Phenanthrene retains its PAH name, not misrouted to assembly."""
        name = name_compound("c1ccc2c(c1)ccc1ccccc12")
        assert name == "phenanthrene"

    @pytest.mark.unit
    def test_naphthalene_not_assembly_name(self):
        """Naphthalene retains its PAH name."""
        name = name_compound("c1ccc2ccccc2c1")
        assert name == "naphthalene"

    @pytest.mark.unit
    def test_phenylcyclohexane_unchanged(self):
        """Phenylcyclohexane is NOT named as assembly (different ring types)."""
        name = name_compound("C1CCCCC1c1ccccc1")
        # Should not contain 'bi' prefix for assembly naming
        assert "biphenyl" not in name
        assert "bicyclo" not in name.lower() or "cyclohex" in name


# ---------------------------------------------------------------------------
# RING-01: Higher multiplier tests (quinque through deci)
# ---------------------------------------------------------------------------

class TestHigherMultipliers:
    """Tests for ASSEMBLY_MULTIPLIERS extension to 5-10 (IUPAC P-28.2)."""

    @pytest.mark.unit
    def test_quinque_in_dict(self):
        """ASSEMBLY_MULTIPLIERS[5] == 'quinque'."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert ASSEMBLY_MULTIPLIERS[5] == "quinque"

    @pytest.mark.unit
    def test_sexi_in_dict(self):
        """ASSEMBLY_MULTIPLIERS[6] == 'sexi'."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert ASSEMBLY_MULTIPLIERS[6] == "sexi"

    @pytest.mark.unit
    def test_septi_in_dict(self):
        """ASSEMBLY_MULTIPLIERS[7] == 'septi'."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert ASSEMBLY_MULTIPLIERS[7] == "septi"

    @pytest.mark.unit
    def test_octi_in_dict(self):
        """ASSEMBLY_MULTIPLIERS[8] == 'octi'."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert ASSEMBLY_MULTIPLIERS[8] == "octi"

    @pytest.mark.unit
    def test_novi_in_dict(self):
        """ASSEMBLY_MULTIPLIERS[9] == 'novi'."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert ASSEMBLY_MULTIPLIERS[9] == "novi"

    @pytest.mark.unit
    def test_deci_in_dict(self):
        """ASSEMBLY_MULTIPLIERS[10] == 'deci'."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert ASSEMBLY_MULTIPLIERS[10] == "deci"

    @pytest.mark.unit
    def test_all_eleven_entries(self):
        """ASSEMBLY_MULTIPLIERS has exactly 11 entries (2 through 12).

        P-28.5 (Wave2 P1CB Task 8): extended past the old deci(10) cap with
        undeci(11)/dodeci(12) so ring assemblies of >6 (and >10) identical
        systems name correctly. Counts above 12 keep the .get->None decline.
        """
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert len(ASSEMBLY_MULTIPLIERS) == 11
        for k in range(2, 13):
            assert k in ASSEMBLY_MULTIPLIERS

    @pytest.mark.unit
    def test_quinquepyridine_name(self):
        """name_ring_assembly() with 5 identical pyridine rings returns name
        containing 'quinquepyridine'."""
        # 5 pyridines linked: c1ccncc1-c1ccncc1-c1ccncc1-c1ccncc1-c1ccncc1
        smiles = "c1ccncc1-c1ccncc1-c1ccncc1-c1ccncc1-c1ccncc1"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None, "Should detect 5-ring pyridine assembly"
        assert info["count"] == 5
        result = name_ring_assembly(mol, info, None)
        assert result is not None
        assert "quinquepyridine" in result

    @pytest.mark.unit
    def test_quinquephenyl_prefix(self):
        """name_ring_assembly() with 5 identical benzene rings returns name
        containing 'quinquephenyl'."""
        from orthonym.rules.ring_assemblies import name_ring_assembly_prefix
        smiles = "c1ccc(-c2ccc(-c3ccc(-c4ccc(-c5ccccc5)cc4)cc3)cc2)cc1"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None, "Should detect 5-ring benzene assembly"
        assert info["count"] == 5
        result = name_ring_assembly(mol, info, None)
        assert result is not None
        assert "quinquephenyl" in result

    @pytest.mark.unit
    def test_count_11_12_supported_13_unsupported(self):
        """P-28.5 (Wave2 P1CB Task 8): undeci(11)/dodeci(12) supported (OPSIN-RT
        verified); count 13 and above keep the .get->None decline (no
        OPSIN-verifiable affix -> fail closed)."""
        from orthonym.rules.ring_assemblies import ASSEMBLY_MULTIPLIERS
        assert ASSEMBLY_MULTIPLIERS[11] == "undeci"
        assert ASSEMBLY_MULTIPLIERS[12] == "dodeci"
        assert 13 not in ASSEMBLY_MULTIPLIERS

    @pytest.mark.unit
    def test_regression_biphenyl(self):
        """Regression: biphenyl detection + naming produces the PIN 1,1'-biphenyl
        (F-T9/DD6 RET-01)."""
        name = name_compound("c1ccc(-c2ccccc2)cc1")
        assert name == "1,1'-biphenyl"

    @pytest.mark.unit
    def test_regression_bipyridine(self):
        """Regression: bipyridine detection and naming still works."""
        name = name_compound("c1ccncc1-c2ccncc2")
        assert "bipyridine" in name


# ---------------------------------------------------------------------------
# RING-02: Fused ring assembly unit tests
# ---------------------------------------------------------------------------

class TestFusedRingAssemblyUnits:
    """Tests for fused ring systems as ring assembly units (IUPAC P-28.1)."""

    @pytest.mark.unit
    def test_get_ring_parent_name_naphthalene(self):
        """_get_ring_parent_name() with naphthalene system atoms returns 'naphthalene'."""
        from orthonym.rules.ring_assemblies import _get_ring_parent_name
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        # naphthalene is a single ring system; all atoms form the system
        system_atoms = set(range(mol.GetNumAtoms()))
        name = _get_ring_parent_name(mol, system_atoms)
        assert name is not None, "_get_ring_parent_name returned None for naphthalene"
        assert name == "naphthalene"

    @pytest.mark.unit
    def test_get_ring_parent_name_quinoline(self):
        """_get_ring_parent_name() with quinoline system atoms returns 'quinoline'."""
        from orthonym.rules.ring_assemblies import _get_ring_parent_name
        mol = Chem.MolFromSmiles("c1ccc2ncccc2c1")
        system_atoms = set(range(mol.GetNumAtoms()))
        name = _get_ring_parent_name(mol, system_atoms)
        assert name is not None, "_get_ring_parent_name returned None for quinoline"
        assert name == "quinoline"

    @pytest.mark.unit
    def test_get_ring_parent_name_indole(self):
        """_get_ring_parent_name() with indole system atoms returns a name containing 'indole'."""
        from orthonym.rules.ring_assemblies import _get_ring_parent_name
        mol = Chem.MolFromSmiles("c1ccc2[nH]ccc2c1")
        system_atoms = set(range(mol.GetNumAtoms()))
        name = _get_ring_parent_name(mol, system_atoms)
        assert name is not None, "_get_ring_parent_name returned None for indole"
        assert "indole" in name

    @pytest.mark.unit
    def test_binaphthalene_detection(self):
        """detect_ring_assembly() on two identical naphthalene systems returns count==2."""
        # 1,1'-binaphthalene
        smiles = "c1ccc2ccccc2c1-c1ccc2ccccc2c1"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None, "Should detect binaphthalene as 2-ring assembly"
        assert info["count"] == 2

    @pytest.mark.unit
    def test_binaphthalene_name(self):
        """name_compound() on binaphthalene SMILES produces name containing 'binaphthalene'."""
        smiles = "c1ccc2ccccc2c1-c1ccc2ccccc2c1"
        name = name_compound(smiles)
        assert name is not None
        assert "binaphthalene" in name or "binaphthalen" in name, \
            f"Expected 'binaphthalene' in name, got: {name}"

    @pytest.mark.unit
    def test_biquinoline_detection(self):
        """detect_ring_assembly() on two identical quinoline systems returns count==2."""
        smiles = "c1ccc2ncccc2c1-c1ccc2ncccc2c1"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None, "Should detect biquinoline as 2-ring assembly"
        assert info["count"] == 2


# ---------------------------------------------------------------------------
# G3 / COV-03: P-28.2.1 enclosing marks (von Baeyer confusion)
# ---------------------------------------------------------------------------

class TestP28EnclosingMarks:
    """IUPAC P-28.2.1: 'bi' + parent hydride name *enclosed in parentheses, if
    necessary*. Parentheses are used to avoid confusion with von Baeyer names,
    so cycloalkane / von-Baeyer components are enclosed (``1,1'-bi(cyclopropane)``)
    while mancude rings (phenyl / pyridine / furan / naphthalene) are NOT
    (``1,1'-biphenyl``, ``2,2'-bipyridine``). All PINs OPSIN-RT-verified.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("C1CC1C1CC1", "1,1'-bi(cyclopropane)"),
        ("C1CCC1C1CCC1", "1,1'-bi(cyclobutane)"),
        ("C1CCCC1C1CCCC1", "1,1'-bi(cyclopentane)"),
        ("C1CCCCC1C1CCCCC1", "1,1'-bi(cyclohexane)"),
    ])
    def test_cycloalkane_assembly_enclosing_marks(self, smiles, expected):
        """Cycloalkane ring assemblies get enclosing marks (P-28.2.1)."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,present,absent", [
        ("c1ccc(-c2ccccc2)cc1", "1,1'-biphenyl", "("),    # benzene -> no parens
        ("c1ccc(-c2ccccn2)nc1", "2,2'-bipyridine", "("),  # pyridine -> no parens
        ("c1ccoc1-c1ccoc1", "bifuran", "bi("),            # furan -> no parens
        ("c1ccc2ccccc2c1-c1ccc2ccccc2c1", "binaphthalene", "bi("),  # naphthalene
    ])
    def test_mancude_assembly_no_enclosing_marks(self, smiles, present, absent):
        """Mancude ring assemblies must NOT be parenthesized (no von Baeyer clash)."""
        name = name_compound(smiles)
        assert present in name, f"expected {present!r} in {name!r}"
        assert absent not in name, f"unexpected {absent!r} in {name!r}"


# ---------------------------------------------------------------------------
# G3 / COV-03: P-28.2.2 double-bond junction (ylidene)
# ---------------------------------------------------------------------------

class TestP28DoubleBondJunction:
    """IUPAC P-28.2.2: two identical cyclic systems linked by a *double bond*
    are named with the ylidene substituent form, enclosed in parentheses:
    ``1,1'-bi(cyclopentylidene)``. All PINs OPSIN-RT-verified.
    """

    @pytest.mark.unit
    def test_double_bond_detector_accepts_identical_carbocycles(self):
        """detect_ring_assembly now recognises a double-bond junction between
        identical saturated carbocycles (P-28.2.2)."""
        mol = Chem.MolFromSmiles("C1CCCC1=C1CCCC1")
        rs = get_ring_systems(mol)
        info = detect_ring_assembly(mol, rs)
        assert info is not None
        assert info["count"] == 2
        assert info["ring_type"] == "carbocyclic"

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("C1CCCC1=C1CCCC1", "1,1'-bi(cyclopentylidene)"),
        ("C1CCCCC1=C1CCCCC1", "1,1'-bi(cyclohexylidene)"),
        ("C1CCC1=C1CCC1", "1,1'-bi(cyclobutylidene)"),
    ])
    def test_cycloalkane_ylidene_assembly(self, smiles, expected):
        """Double-bond cycloalkane assemblies name as bi(...ylidene) (P-28.2.2)."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_multiring_saturated_dimer_not_claimed(self):
        """CR-01: a double-bond dimer of a MULTI-ring saturated carbocycle
        (norbornane) is NOT claimed by detect_ring_assembly — the monocyclic
        ylidene namer cannot name it, so it must fall through to the prior path
        rather than trigger the namer.py early-return into a fail-closed dead end.
        """
        mol = Chem.MolFromSmiles("C1CC2CCC1C2=C1CC2CCC1C2")
        assert detect_ring_assembly(mol, get_ring_systems(mol)) is None

    @pytest.mark.unit
    def test_double_bond_junction_deterministic(self):
        """Same structure from a different SMILES writing gives the same PIN."""
        a = name_compound("C1CCCC1=C1CCCC1")
        b = name_compound("C1(=C2CCCC2)CCCC1")
        assert a == b == "1,1'-bi(cyclopentylidene)"


# ---------------------------------------------------------------------------
# G3 / COV-03: P-28.2.1 + P-28.3.1 citation-order locant determinism
# ---------------------------------------------------------------------------

class TestP28CitationOrderDeterminism:
    """IUPAC P-28.2.1 ("lowest possible locants ... for the positions of
    attachment") + P-28.3.1 erratum (8 Oct 2025, "lowest locant set, then order
    of citation"): for a 2-component assembly of identical rings, the unprimed
    (first-cited) ring must carry the lower attachment locant. The name must be
    invariant to SMILES atom order (2,3'-bifuran, never 3,2'-bifuran).
    BB P-28.2.1 worked example: '2,3'-bifuran (PIN)'.
    """

    @pytest.mark.unit
    def test_bifuran_is_2_3prime(self):
        assert name_compound("c1ccoc1-c1ccoc1") == "2,3'-bifuran"

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles", [
        "c1ccoc1-c1ccoc1",
        "c1oc(cc1)-c1ccoc1",
        "c1(cocc1)-c1occc1",
        "c1cc(-c2ccoc2)oc1",
    ])
    def test_bifuran_deterministic_across_respellings(self, smiles):
        """Every SMILES spelling of bifuran yields the lowest-locant PIN
        (the unprimed ring takes locant 2, not 3)."""
        assert name_compound(smiles) == "2,3'-bifuran"
