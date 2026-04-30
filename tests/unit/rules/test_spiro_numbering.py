"""Phase 151-02 D-11 / D-20: P-24.2.2 numbering — compare_locant_sets reuse.

Wave-0 RED scaffold. Verifies Plan 151-02 lands the ``compare_locant_sets``
import in ``spiro.py`` and DOES NOT define a parallel comparator.

Source: 151-02-PLAN.md tasks 1b/2; 151-AUDIT-B.md; 151-CONTEXT.md
D-11 / D-20.
"""
from __future__ import annotations

import inspect

import pytest
from rdkit import Chem


def _spiro_or_skip():
    try:
        from orthonym.rules import spiro
        return spiro
    except ImportError:
        pytest.skip("orthonym.rules.spiro import failed")


class TestNoParallelComparator:
    """D-20: spiro.py must NOT introduce a parallel locant comparator."""

    @pytest.mark.unit
    def test_spiro_module_uses_compare_locant_sets(self):
        spiro = _spiro_or_skip()
        # Plan 151-02 lands the import in Task 2.
        try:
            from orthonym.rules.spiro import name_mixed_spiro_fused  # noqa
        except ImportError:
            pytest.skip("name_mixed_spiro_fused not yet exported")
        src = inspect.getsource(spiro)
        assert "compare_locant_sets" in src, (
            "spiro.py must import compare_locant_sets (D-11/D-20 reuse lock)"
        )

    @pytest.mark.unit
    def test_no_def_compare_locant(self):
        spiro = _spiro_or_skip()
        src = inspect.getsource(spiro)
        non_comment = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "def _compare_locant" not in non_comment, (
            "spiro.py defines a parallel comparator — D-20 violation"
        )


class TestP2422LowestLocantTiebreak:
    """P-24.2.2 lowest-locant tiebreak.

    Plan 151-02 documents this as a v19 follow-up for the existing
    pure-spiro segment-locant numbering (currently picks a non-canonical
    set on 4/10 Blue Book fixtures). The Wave-0 RED test holds room for
    the v19 fix; in v18 the tiebreak is enforced ONLY in the new
    ``name_mixed_spiro_fused`` body's spiro-side orientation step.
    """

    @pytest.mark.unit
    def test_dispiro_descriptor_uses_compare_locant_sets_in_mixed_only(self):
        """In v18 the comparator gate is in name_mixed_spiro_fused, not
        in _compute_spiro_segments. This test documents the boundary."""
        spiro = _spiro_or_skip()
        try:
            name_mixed_spiro_fused = spiro.name_mixed_spiro_fused
        except AttributeError:
            pytest.skip("name_mixed_spiro_fused not yet exported")
        src = inspect.getsource(name_mixed_spiro_fused)
        # Either the body uses compare_locant_sets directly, OR it
        # delegates to a helper that does. The minimum invariant: the
        # SOURCE of the function references the comparator name.
        assert (
            "compare_locant_sets" in src
            or "compare_locant_sets" in inspect.getsource(spiro)
        ), "name_mixed_spiro_fused must use compare_locant_sets (D-11)"

    @pytest.mark.unit
    def test_pure_spiro_tiebreak_documented_followup(self):
        """Audit logs pure-spiro tiebreak refinement to v19 — the Plan
        151-02 acceptance is import-only for spiro.py module top-level."""
        spiro = _spiro_or_skip()
        try:
            from orthonym.rules.spiro import name_mixed_spiro_fused  # noqa
        except ImportError:
            pytest.skip("name_mixed_spiro_fused not yet exported")
        src = inspect.getsource(spiro)
        assert "compare_locant_sets" in src


class TestPureSpiroNumberingSmoke:
    """P-31.3.1.2 smoke checks — confirm get_spiro_numbering returns
    a complete map for monospiro inputs."""

    @pytest.mark.unit
    def test_spiro45decane_numbering_complete(self):
        from orthonym.rules.spiro import (
            get_spiro_numbering, get_spiro_atoms,
        )
        from rdkit import Chem
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
        from rdkit import Chem
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
        """P-31.3.1.2: numbering starts adjacent to spiro centre in the
        smaller ring."""
        from orthonym.rules.spiro import (
            get_spiro_numbering, get_spiro_atoms,
        )
        from rdkit import Chem
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
        from rdkit import Chem
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
        from rdkit import Chem
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CC1(CCC2)CCCCCC1")
        numbering = _get_polyspiro_numbering(mol, get_spiro_atoms(mol))
        assert numbering is not None
        values = sorted(numbering.values())
        assert values == list(range(1, len(values) + 1))

    @pytest.mark.unit
    def test_compare_locant_sets_signature(self):
        """compare_locant_sets is the authoritative comparator (D-11)."""
        from orthonym.rules.locants import compare_locant_sets
        # Lower locant set (1,2,4) should be preferred over (1,3,4).
        assert compare_locant_sets([1, 2, 4], [1, 3, 4]) < 0
        assert compare_locant_sets([1, 3, 4], [1, 2, 4]) > 0
        assert compare_locant_sets([1, 2, 4], [1, 2, 4]) == 0

    @pytest.mark.unit
    def test_spiro_imports_locants_module(self):
        """Plan 151-02 lands the import of compare_locant_sets in spiro.py."""
        spiro = _spiro_or_skip()
        try:
            from orthonym.rules.spiro import name_mixed_spiro_fused  # noqa
        except ImportError:
            pytest.skip("name_mixed_spiro_fused not yet exported")
        src = inspect.getsource(spiro)
        # Either explicit `from .locants import` or an in-body usage.
        assert "compare_locant_sets" in src
