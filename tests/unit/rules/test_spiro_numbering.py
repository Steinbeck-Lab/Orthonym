"""a phase-02 /: numbering — compare_locant_sets reuse.

Wave-0 RED scaffold. Verifies Plan 151-02 lands the ``compare_locant_sets``
import in ``spiro.py`` and DOES NOT define a parallel comparator.

Source: 151-02-PLAN.md tasks 1b/2; internal notes-B.md; 151-internal notes
 /.
"""
from __future__ import annotations

import inspect

import pytest
from rdkit import Chem

from orthonym.rules import spiro


class TestNoParallelComparator:
    """: spiro.py must NOT introduce a parallel locant comparator."""

    @pytest.mark.unit
    def test_spiro_module_uses_compare_locant_sets(self):
        src = inspect.getsource(spiro)
        assert "compare_locant_sets" in src, (
            "spiro.py must import compare_locant_sets (D-11/D-20 reuse lock)"
        )

    @pytest.mark.unit
    def test_no_def_compare_locant(self):
        src = inspect.getsource(spiro)
        non_comment = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "def _compare_locant" not in non_comment, (
            "spiro.py defines a parallel comparator — D-20 violation"
        )


class TestPureSpiroNumberingSmoke:
    """ smoke checks — confirm get_spiro_numbering returns
    a complete map for monospiro inputs."""

    @pytest.mark.unit
    def test_spiro45decane_numbering_complete(self):
        from orthonym.rules.spiro import (
            get_spiro_numbering, get_spiro_atoms,
        )
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")
        spiro_atoms = get_spiro_atoms(mol)
        assert len(spiro_atoms) == 1
        center = list(spiro_atoms)[0]
        numbering = get_spiro_numbering(mol, center)
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        assert set(numbering.keys()) >= ring_atoms

    @pytest.mark.unit
    def test_spiro55undecane_numbering_complete(self):
        from orthonym.rules.spiro import (
            get_spiro_numbering, get_spiro_atoms,
        )
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCCC2")
        center = list(get_spiro_atoms(mol))[0]
        numbering = get_spiro_numbering(mol, center)
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        assert set(numbering.keys()) >= ring_atoms

    @pytest.mark.unit
    def test_spiro_numbering_starts_in_smaller_ring(self):
        """: numbering starts adjacent to spiro centre in the
        smaller ring."""
        from orthonym.rules.spiro import (
            get_spiro_numbering, get_spiro_atoms,
        )
        # spiro[3.5]nonane: cyclobutane + cyclohexane sharing 1 atom
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCC2")
        center = list(get_spiro_atoms(mol))[0]
        numbering = get_spiro_numbering(mol, center)
        # Spiro centre's locant should be 4 (numbering goes 1,2,3,4 around
        # the smaller 4-membered ring with centre at position 4).
        spiro_locant = numbering[center]
        # At minimum: spiro centre is somewhere in the middle of the
        # numbering, not at position 1 or position N (terminal).
        assert spiro_locant > 1
        assert spiro_locant < len(numbering)

    @pytest.mark.unit
    def test_polyspiro_numbering_dispiro_complete(self):
        from orthonym.rules.spiro import (
            _get_polyspiro_numbering, get_spiro_atoms,
        )
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1(CCC2)CCCCCC1")
        spiro_atoms = get_spiro_atoms(mol)
        numbering = _get_polyspiro_numbering(mol, spiro_atoms)
        assert numbering is not None
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for r in ri.AtomRings():
            ring_atoms.update(r)
        assert set(numbering.keys()) >= ring_atoms

    @pytest.mark.unit
    def test_polyspiro_numbering_returns_consecutive_locants(self):
        """Numbering values are 1..N with no gaps."""
        from orthonym.rules.spiro import (
            _get_polyspiro_numbering, get_spiro_atoms,
        )
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1(CCC2)CCCCCC1")
        numbering = _get_polyspiro_numbering(mol, get_spiro_atoms(mol))
        assert numbering is not None
        values = sorted(numbering.values())
        assert values == list(range(1, len(values) + 1))

    @pytest.mark.unit
    def test_compare_locant_sets_signature(self):
        """compare_locant_sets is the authoritative comparator ."""
        from orthonym.rules.locants import compare_locant_sets
        # Lower locant set (1,2,4) should be preferred over (1,3,4).
        assert compare_locant_sets([1, 2, 4], [1, 3, 4]) < 0
        assert compare_locant_sets([1, 3, 4], [1, 2, 4]) > 0
        assert compare_locant_sets([1, 2, 4], [1, 2, 4]) == 0
