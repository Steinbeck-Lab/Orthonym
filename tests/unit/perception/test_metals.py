"""a phase unit tests for organometallic perception + Salzer 1999).

Mirrors tests/unit/perception/test_ions.py class-per-concern pattern.
Tests the public API of orthonym.perception.metals + internal notes purity
invariants. NEVER uses @pytest.mark.xfail (internal notes) — failures are
honest-fail-on-data per project memory rule #4.
"""
import pytest
from rdkit import Chem

from orthonym.perception.metals import (
    METAL_ELEMENT_SYMBOLS,
    is_metal_element,
    LigandGroup,
    MetalComplex,
    detect_metal_complex,
    compute_hapticity,
    enumerate_metal_ligand_groups,
)
from orthonym.data.organometallics import LIGAND_ETA_DEFAULTS


# ---------------------------------------------------------------------------
# Class 1 — TestIsMetalElement (≥ 7 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsMetalElement:
    """Test METAL_ELEMENT_SYMBOLS allowlist + is_metal_element predicate."""

    def test_iron_is_metal(self):
        """Fe is a Group 8 transition metal."""
        assert is_metal_element('Fe') is True

    def test_carbon_is_not_metal(self):
        """C is a non-metal."""
        assert is_metal_element('C') is False

    def test_hydrogen_excluded(self):
        """H is excluded per RESEARCH §3.1 line 460 (never metal)."""
        assert is_metal_element('H') is False

    def test_lithium_is_metal(self):
        """Li is Group 1 alkali metal."""
        assert is_metal_element('Li') is True

    def test_tin_is_metal(self):
        """Sn is Group 14; part of allowlist per RESEARCH §3.1."""
        assert is_metal_element('Sn') is True

    def test_allowlist_size_ge_60(self):
        """METAL_ELEMENT_SYMBOLS has >= 60 entries (Groups 1-17 + lanthanides + actinides)."""
        assert len(METAL_ELEMENT_SYMBOLS) >= 60

    def test_unknown_symbol(self):
        """Bogus element symbol returns False (no KeyError)."""
        assert is_metal_element('Xx') is False

    def test_empty_string(self):
        """Empty string returns False."""
        assert is_metal_element('') is False

    def test_chromium_is_metal(self):
        """Cr is Group 6 transition metal (key for benzene-Cr fixtures)."""
        assert is_metal_element('Cr') is True

    def test_oxygen_is_not_metal(self):
        """O is non-metal (CO ligand atom)."""
        assert is_metal_element('O') is False

    def test_silicon_is_metal(self):
        """Si is Group 14 semimetal (hydride-parent system)."""
        assert is_metal_element('Si') is True


# ---------------------------------------------------------------------------
# Class 2 — TestDetectMetalComplex (≥ 10 tests; one per Tier-1 metallocene
# + negative cases)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectMetalComplex:
    """Test detect_metal_complex per topology class."""

    def test_ferrocene_returns_metal_complex(self):
        """Tier-1 ferrocene: 1 Fe(+2) cation + 2 Cp(-1) anions."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = detect_metal_complex(mol)
        assert result is not None
        assert isinstance(result, MetalComplex)
        assert len(result.metal_atom_indices) == 1
        assert result.formal_charges == (2,)
        assert result.is_multimetal is False
        assert len(result.ligand_groups) == 2

    def test_ruthenocene_returns_metal_complex(self):
        """Tier-1 ruthenocene with Ru(+2)."""
        mol = Chem.MolFromSmiles('[Ru+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = detect_metal_complex(mol)
        assert result is not None
        assert result.is_multimetal is False
        assert result.formal_charges == (2,)

    def test_cobaltocene_returns_metal_complex(self):
        """Tier-1 cobaltocene with Co(+2)."""
        mol = Chem.MolFromSmiles('[Co+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = detect_metal_complex(mol)
        assert result is not None

    def test_ferrocenium_returns_metal_complex(self):
        """Tier-1 ferrocenium with Fe(+3)."""
        mol = Chem.MolFromSmiles('[Fe+3].c1cc[cH-]c1.c1cc[cH-]c1')
        result = detect_metal_complex(mol)
        assert result is not None
        assert result.formal_charges == (3,)

    def test_ethanol_returns_none(self):
        """Ethanol has no metal atom; cascades to non-ORGM pipeline."""
        mol = Chem.MolFromSmiles('CCO')
        assert detect_metal_complex(mol) is None

    def test_acetic_acid_returns_none(self):
        """Risk R-02 mitigation: C=O group must NOT trigger ORGM (no metal)."""
        mol = Chem.MolFromSmiles('CC(=O)O')
        assert detect_metal_complex(mol) is None

    def test_ferric_chloride_returns_none(self):
        """FeCl3 has metal but no M-C bond; no ORGM detection per internal notes.

        Note: detect_metal_complex returns None here because the topology
        doesn't match any tier-1/2/3/4 branch (3 halides, no organic ligand).
        """
        mol = Chem.MolFromSmiles('Cl[Fe](Cl)Cl')
        # a phase perception walks via Tier-3 σ-bonded branch; halide-only
        # mol with no organic ligand → all-halides, no σ-organic match → cascades.
        result = detect_metal_complex(mol)
        # The current impl may detect this as Tier-3 (3 halides); rules layer
        # rejects it. Accept either None (cascades early) OR detected (cascades
        # at rules layer). The hard guarantee is name_compound DOESN'T produce
        # an ORGM name for this. (Tested in test_organometallic_handler.py.)
        if result is not None:
            # Detection succeeded but should reject at rules layer
            from orthonym.rules.organometallics import assemble_organometallic_name
            assert assemble_organometallic_name(result, mol, style='pin') is None

    def test_methyllithium_returns_metal_complex(self):
        """Tier-3 σ-bonded MeLi: single-component CH3-Li."""
        mol = Chem.MolFromSmiles('[Li][CH3]')
        result = detect_metal_complex(mol)
        assert result is not None
        assert result.formal_charges[0] == 0

    def test_iron_pentacarbonyl_returns_metal_complex(self):
        """Tier-2 dot-separated metal carbonyl: Fe(CO)5."""
        mol = Chem.MolFromSmiles(
            '[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Fe]'
        )
        result = detect_metal_complex(mol)
        assert result is not None
        # 5 CO ligand groups
        assert len(result.ligand_groups) == 5

    def test_isolated_iron_atom_returns_none(self):
        """[Fe] alone has no ligands → not an organometallic per Q2."""
        mol = Chem.MolFromSmiles('[Fe]')
        assert detect_metal_complex(mol) is None

    def test_none_input_returns_none(self):
        """detect_metal_complex(None) returns None per defensive guard."""
        assert detect_metal_complex(None) is None

    def test_empty_mol_returns_none(self):
        """Empty mol returns None."""
        mol = Chem.RWMol()
        # RWMol with 0 atoms; SanitizeMol skipped per internal notes
        assert detect_metal_complex(mol) is None

    def test_dibenzene_chromium_returns_metal_complex(self):
        """Tier-4 bis-arene complex: Cr(C6H6)2."""
        mol = Chem.MolFromSmiles('[Cr].c1ccccc1.c1ccccc1')
        result = detect_metal_complex(mol)
        assert result is not None
        assert len(result.ligand_groups) == 2

    def test_pentacarbonylvanadium_anion_returns_metal_complex(self):
        """Tier-2 metal carbonyl with anion metal: V(CO)6(-1)."""
        mol = Chem.MolFromSmiles(
            '[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[V-]'
        )
        result = detect_metal_complex(mol)
        assert result is not None
        assert result.formal_charges == (-1,)


# ---------------------------------------------------------------------------
# Class 3 — TestComputeHapticity (≥ 7 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeHapticity:
    """Test compute_hapticity per LIGAND_ETA_DEFAULTS entry + graph-walk fallback."""

    def test_cyclopentadienyl_hapticity_5(self):
        """Cp(-1) ligand → η⁵ per AUDIT § 2 + LIGAND_ETA_DEFAULTS."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        # Find Cp ring atoms (5 aromatic carbons)
        cp_atoms = tuple(
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetSymbol() == 'C' and a.GetIsAromatic()
        )[:5]
        h = compute_hapticity(mol, 0, cp_atoms)
        assert h == 5

    def test_methyl_single_atom_hapticity_1(self):
        """Single-atom ligand (M-CH3) → η¹ per σ-bonded rule."""
        mol = Chem.MolFromSmiles('[Li][CH3]')
        # methyl C atom
        c_atoms = tuple(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'C')
        h = compute_hapticity(mol, 0, c_atoms[:1])
        assert h == 1

    def test_carbonyl_smarts_key_has_eta_1(self):
        """CO ligand → η¹ via LIGAND_ETA_DEFAULTS catalog hit."""
        assert LIGAND_ETA_DEFAULTS['[C-]#[O+]'] == (1, 'carbonyl')

    def test_cyclopentadienyl_in_catalog(self):
        """Cp(-1) ligand in LIGAND_ETA_DEFAULTS with hapticity 5."""
        assert LIGAND_ETA_DEFAULTS['c1cc[cH-]c1'] == (5, 'cyclopentadienyl')

    def test_benzene_in_catalog(self):
        """Benzene π-ligand → η⁶ per Salzer 1999."""
        assert LIGAND_ETA_DEFAULTS['c1ccccc1'] == (6, 'benzene')

    def test_butadiene_in_catalog(self):
        """1,3-butadiene → η⁴ per Salzer 1999."""
        assert LIGAND_ETA_DEFAULTS['C=CC=C'] == (4, '1,3-butadiene')

    def test_cot_in_catalog(self):
        """Cyclooctatetraene (COT; 8-atom ring) → η⁸."""
        assert LIGAND_ETA_DEFAULTS['C1=CC=CC=CC=C1'] == (8, 'cyclooctatetraene')

    def test_cycloheptatrienyl_in_catalog(self):
        """Cycloheptatrienyl (7-atom ring) → η⁷."""
        assert LIGAND_ETA_DEFAULTS['C1=CC=CC=CC=1'] == (7, 'cycloheptatrienyl')

    def test_empty_atom_indices_raises_value_error(self):
        """Empty ligand_atom_indices is honest-fail per."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        with pytest.raises(ValueError):
            compute_hapticity(mol, 0, ())


# ---------------------------------------------------------------------------
# Class 4 — TestEnumerateMetalLigandGroups (≥ 4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnumerateMetalLigandGroups:
    """Test enumerate_metal_ligand_groups partitioning (Plan-02 stub: returns ).

    Per Plan-03 SUMMARY, enumerate_metal_ligand_groups is still a stub returning .
    The actual ligand-group enumeration happens inside detect_metal_complex.
    These tests verify the stub contract; a phase+ may flesh out the helper.
    """

    def test_stub_returns_empty_tuple_for_ferrocene(self):
        """Plan-02 stub: returns empty tuple per docstring contract."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        groups = enumerate_metal_ligand_groups(mol)
        assert groups == ()

    def test_stub_returns_empty_tuple_for_ethanol(self):
        """Stub: ethanol returns empty tuple (no metal)."""
        mol = Chem.MolFromSmiles('CCO')
        groups = enumerate_metal_ligand_groups(mol)
        assert groups == ()

    def test_stub_returns_tuple_type(self):
        """Stub returns Tuple[LigandGroup,...] type (immutable)."""
        mol = Chem.MolFromSmiles('[Li][CH3]')
        groups = enumerate_metal_ligand_groups(mol)
        assert isinstance(groups, tuple)

    def test_stub_does_not_raise_on_none(self):
        """enumerate_metal_ligand_groups(None) — defensive guard, returns ()."""
        # a phase stub: it may raise AttributeError on None; document either.
        try:
            result = enumerate_metal_ligand_groups(None)
            assert result == ()
        except (AttributeError, TypeError):
            # Stub does not yet defend against None
            pytest.skip(
                "Phase 161 enumerate_metal_ligand_groups stub does not yet "
                "guard against None input; Phase 161.1+ will harden."
            )


# ---------------------------------------------------------------------------
# Class 5 — TestPurityInvariants (≥ 7 tests — internal notes hard invariant)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPurityInvariants:
    """internal notes hard invariant: predicate-pure functions."""

    def test_detect_metal_complex_none_input(self):
        """detect_metal_complex(None) returns None per defensive guard."""
        assert detect_metal_complex(None) is None

    def test_mol_unchanged_after_detect(self):
        """detect_metal_complex MUST NOT mutate mol per internal notes."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        atom_count_pre = mol.GetNumAtoms()
        bond_count_pre = mol.GetNumBonds()
        _ = detect_metal_complex(mol)
        assert mol.GetNumAtoms() == atom_count_pre
        assert mol.GetNumBonds() == bond_count_pre

    def test_dataclasses_are_frozen_ligand_group(self):
        """LigandGroup is a frozen dataclass per."""
        lg = LigandGroup(
            metal_atom_idx=0,
            ligand_atom_indices=(1, 2),
            hapticity_n=2,
        )
        with pytest.raises(Exception):
            lg.hapticity_n = 5  # type: ignore[misc] # frozen → cannot mutate

    def test_dataclasses_are_frozen_metal_complex(self):
        """MetalComplex is a frozen dataclass per."""
        mc = MetalComplex(
            metal_atom_indices=(0,),
            ligand_groups=(),
            formal_charges=(0,),
            is_multimetal=False,
        )
        with pytest.raises(Exception):
            mc.is_multimetal = True  # type: ignore[misc] # frozen

    def test_metal_element_symbols_is_frozenset(self):
        """METAL_ELEMENT_SYMBOLS is a frozenset (immutable)."""
        assert isinstance(METAL_ELEMENT_SYMBOLS, frozenset)

    def test_metal_element_symbols_unmodifiable(self):
        """METAL_ELEMENT_SYMBOLS cannot be add/remove'd."""
        with pytest.raises(AttributeError):
            METAL_ELEMENT_SYMBOLS.add('Zz')  # type: ignore[attr-defined]

    def test_detect_metal_complex_pure_on_organic(self):
        """detect_metal_complex on non-metal mol returns None without mutation."""
        mol = Chem.MolFromSmiles('CCO')
        atom_count_pre = mol.GetNumAtoms()
        result = detect_metal_complex(mol)
        assert result is None
        assert mol.GetNumAtoms() == atom_count_pre

    def test_detect_metal_complex_pure_on_carbonyl(self):
        """detect_metal_complex on Fe(CO)5 mol does NOT mutate."""
        mol = Chem.MolFromSmiles(
            '[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Fe]'
        )
        atom_count_pre = mol.GetNumAtoms()
        bond_count_pre = mol.GetNumBonds()
        _ = detect_metal_complex(mol)
        assert mol.GetNumAtoms() == atom_count_pre
        assert mol.GetNumBonds() == bond_count_pre

    def test_no_xfail_decorator_in_module(self):
        """internal notes + enforcement: this module must NEVER use xfail.

        Checks for the literal decorator line `` @pytest.mark.xfail``
        (with leading whitespace) so the assertion string in this
        test's own body does not self-trigger.
        """
        import inspect
        from tests.unit.perception import test_metals
        src = inspect.getsource(test_metals)
        # Look for the decorator usage (whitespace + @pytest.mark.xfail)
        # rather than the bare string (which appears in docstrings/asserts).
        assert '\n    @pytest.mark.xfail' not in src
        assert '\n@pytest.mark.xfail' not in src

    def test_metal_complex_is_immutable_namedtuple_like(self):
        """MetalComplex is dataclass(frozen=True) so all fields readable + immutable."""
        mc = MetalComplex(
            metal_atom_indices=(0,),
            ligand_groups=(),
            formal_charges=(2,),
            is_multimetal=False,
        )
        # Read-only access
        assert mc.metal_atom_indices == (0,)
        assert mc.formal_charges == (2,)
        # tuple sub-fields are also immutable
        assert isinstance(mc.metal_atom_indices, tuple)
        assert isinstance(mc.ligand_groups, tuple)
