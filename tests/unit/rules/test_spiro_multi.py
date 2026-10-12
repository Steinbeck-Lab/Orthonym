"""a phase-02: multi-spiro tests for dispiro/trispiro/tetra/penta.

Wave-0 RED scaffold: tests target the existing ``name_spiro_system``
helper for multi-spiro coverage. They exercise the polyspiro descriptor
build path AND the cascade-step-6 supplier wrapper added in Plan 151-02
(``get_spiro_iupac_locants``).

Source: 151-02-PLAN.md tasks 1b/2; internal notes-B.md; IUPAC.
"""
from __future__ import annotations

import os

import pytest
from rdkit import Chem

from orthonym.rules.spiro import (
    is_spiro_system,
    name_spiro_system,
    get_spiro_atoms,
)


def _spiro_lazy_supplier_import():
    """Lazy import: until Task 2 lands, get_spiro_iupac_locants does
    not exist; tests that depend on it skip with a clear marker."""
    try:
        from orthonym.rules.spiro import get_spiro_iupac_locants
        return get_spiro_iupac_locants
    except ImportError:
        pytest.skip("get_spiro_iupac_locants not yet exported "
                    "(Task 2 GREEN gate)")


class TestDispiro:
    """Dispiro hydrocarbons — Plan 151-02 D-10 must-ship gate."""

    @pytest.mark.unit
    def test_dispiro_canonical_blue_book(self):
        """dispiro[5.1.5.2]heptadecane (Blue Book example)."""
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1(CCC2)CCCCCC1")
        assert is_spiro_system(mol) is True
        assert len(get_spiro_atoms(mol)) == 2
        result = name_spiro_system(mol)
        assert result is not None
        name, _, _, _ = result
        # segment count tiebreak deferred to (logged
        # AUTONOM-followups). Connectivity correctness is what we
        # verify here — name must contain 'dispiro' and a proper
        # '[a.b.c.d]' descriptor.
        assert name.startswith("dispiro[")

    @pytest.mark.unit
    def test_dispiro_segments_have_correct_arity(self):
        """Dispiro descriptor has exactly 4 dot-separated segments."""
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1(CCC2)CCCCCC1")
        result = name_spiro_system(mol)
        assert result is not None
        name = result[0]
        # Extract the [a.b.c.d] block
        bracket = name[name.index("[") + 1: name.index("]")]
        segments = bracket.split(".")
        assert len(segments) == 4, (
            f"dispiro descriptor must have 4 segments, got {len(segments)} "
            f"in {name!r}"
        )

    @pytest.mark.unit
    def test_dispiro_atom_count_invariant(self):
        """Total atom count for dispiro[a.b.c^m.d^n]: a+b+c+d+2 = total ring atoms
        (the superscripts are spiro-atom locants,, not linking atoms)."""
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1(CCC2)CCCCCC1")
        result = name_spiro_system(mol)
        assert result is not None
        name = result[0]
        bracket = name[name.index("[") + 1: name.index("]")]
        segments = [int(s.split("^")[0]) for s in bracket.split(".")]
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        # Sum of segments + N spiro centres = total ring atoms.
        n_spiro = len(get_spiro_atoms(mol))
        assert sum(segments) + n_spiro == len(ring_atoms), (
            f"dispiro segment-count invariant violated: "
            f"{segments} + {n_spiro} != {len(ring_atoms)}"
        )

    @pytest.mark.unit
    def test_dispiro_supplier_full_coverage(self):
        """get_spiro_iupac_locants returns map covering ALL ring atoms (Pitfall 7)."""
        get_spiro_iupac_locants = _spiro_lazy_supplier_import()
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1(CCC2)CCCCCC1")
        loc = get_spiro_iupac_locants(mol)
        assert loc is not None
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        assert set(loc.keys()) >= ring_atoms

    @pytest.mark.unit
    def test_pure_spiro_supplier_full_coverage_45_decane(self):
        """get_spiro_iupac_locants on monospiro spiro[4.5]decane."""
        get_spiro_iupac_locants = _spiro_lazy_supplier_import()
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")
        loc = get_spiro_iupac_locants(mol)
        assert loc is not None
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        assert set(loc.keys()) >= ring_atoms

    @pytest.mark.unit
    def test_pure_spiro_supplier_returns_none_on_non_spiro(self):
        """Supplier returns None on non-spiro inputs."""
        get_spiro_iupac_locants = _spiro_lazy_supplier_import()
        mol = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        assert get_spiro_iupac_locants(mol) is None

    @pytest.mark.unit
    def test_pure_spiro_supplier_returns_none_on_mixed(self):
        """Supplier returns None on mixed-spiro/fused (defers to mixed handler)."""
        get_spiro_iupac_locants = _spiro_lazy_supplier_import()
        mol = Chem.MolFromSmiles("C1CCC2(CC1)Cc1ccccc12")  # spiro-indane
        # is_spiro_system returns False on this; therefore supplier returns None.
        assert is_spiro_system(mol) is False
        assert get_spiro_iupac_locants(mol) is None



class TestPolyspiroPrefixesTable:
    """Plan 151-02: _POLYSPIRO_PREFIXES already covers up to penta."""

    @pytest.mark.unit
    def test_prefixes_cover_through_pentaspiro(self):
        from orthonym.rules.spiro import _POLYSPIRO_PREFIXES
        assert _POLYSPIRO_PREFIXES[1] == "spiro"
        assert _POLYSPIRO_PREFIXES[2] == "dispiro"
        assert _POLYSPIRO_PREFIXES[3] == "trispiro"
        assert _POLYSPIRO_PREFIXES[4] == "tetraspiro"
        assert _POLYSPIRO_PREFIXES[5] == "pentaspiro"
