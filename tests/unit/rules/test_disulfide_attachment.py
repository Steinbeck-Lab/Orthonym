""" regression suite: disulfide attachment-point bug.

The disulfide SMARTS is ``[#6][SX2][SX2][#6]``; SMARTS atom 0 is a flanking
carbon, not one of the disulfide sulfurs. The parent-selection helpers
historically used ``pg_atoms[0]`` as the canonical attachment for ring/chain
locant comparison, which is wrong for any FG whose SMARTS leads with a
non-locant-bearing atom. For 5-membered S-S-in-ring heterocycles fused with an
exocyclic alkyl chain of equal size (e.g. CHEBI:174033 trithiolane), this
caused the (f) cascade to flip ring->chain incorrectly and silently
drop the ring sulfurs from the generated name.

a phase surfaced this latent bug by switching ring numbering to true IUPAC
locants (sulfurs at locants 1,2,4 instead of sorted-fallback). The fix
introduces ``PG_ATTACHMENT_INDICES`` in ``rules/seniority.py`` and a
``_pg_attachment_atoms`` helper used by every PG-locant comparison.

Evidence: `internal notes`
"""

import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestDisulfideAttachmentRegression:
    """Reproducer cases that were broken before."""

    def test_chebi_174033_trithiolane_pentyl_methyl(self):
        """CHEBI:174033 ``CCCCCC1SSC(C)S1`` must keep the trithiolane ring.

        Pre- (post-147): ``1-cyclopentylpentane`` -- ring sulfurs vanish.
        Pre-Phase-147 (v17 baseline): ``3-methyl-5-pentyl-1,2,4-trithiolane``.
        """
        result = name_compound("CCCCCC1SSC(C)S1")
        # Hard requirement: must not silently drop the ring heteroatoms.
        assert "trithiolane" in result, (
            f"Disulfide attachment bug: ring sulfurs dropped. Got: {result!r}"
        )
        assert "cyclopent" not in result, (
            f"Disulfide attachment bug: ring rendered as cyclopentyl. Got: {result!r}"
        )


@pytest.mark.unit
class TestDisulfideRingCanaries:
    """Currently-correct disulfide-containing rings that must not regress.

    These are simple cases that bypass the buggy (f) cascade today
    (chain length 0 or 1 -> ring is parent automatically). They serve as
    regression guards: the fix must keep them naming correctly.
    """

    def test_dithiane_unsubstituted(self):
        """``C1CCSSC1`` -> ``1,2-dithiane``."""
        assert name_compound("C1CCSSC1") == "1,2-dithiane"

    def test_dithiolane_unsubstituted(self):
        """``C1CSSC1`` -> ``1,2-dithiolane``."""
        assert name_compound("C1CSSC1") == "1,2-dithiolane"

    def test_trithiane_unsubstituted(self):
        """``C1CSSSC1`` -> ``1,2,3-trithiane``."""
        assert name_compound("C1CSSSC1") == "1,2,3-trithiane"

    def test_methyl_dithiolane(self):
        """``CC1CCSS1`` -> ``3-methyl-1,2-dithiolane``."""
        assert name_compound("CC1CCSS1") == "3-methyl-1,2-dithiolane"

    def test_ethyl_dithiolane(self):
        """``CCC1CCSS1`` -> ``3-ethyl-1,2-dithiolane``."""
        assert name_compound("CCC1CCSS1") == "3-ethyl-1,2-dithiolane"


@pytest.mark.unit
class TestAcyclicDisulfideCanaries:
    """Acyclic disulfides use ``a-thia`` replacement nomenclature.

    These do NOT exercise the disulfide-as-PG path through, but they
    confirm the fix does not perturb the disulfide perception layer.
    """

    def test_dimethyl_disulfide(self):
        """``CSSC`` -> ``(methyldisulfanyl)methane``.

        DD2 (Phase D, (1)): a dialkyl disulfide is a SUBSTITUTIVE PIN —
        the senior R as parent + ``(R'disulfanyl)`` prefix — not the ``dithia``
        skeletal-replacement chain (the old ``2,3-dithiabutane`` consumed the S-S
        bond as two skeletal thia atoms, the catalog C3 defect)."""
        assert name_compound("CSSC") == "(methyldisulfanyl)methane"

    def test_diethyl_disulfide(self):
        """``CCSSCC`` -> ``(ethyldisulfanyl)ethane`` (substitutive PIN, (1))."""
        assert name_compound("CCSSCC") == "(ethyldisulfanyl)ethane"


@pytest.mark.unit
class TestPgAttachmentIndicesTable:
    """Sanity checks on the ``PG_ATTACHMENT_INDICES`` data structure."""

    def test_disulfide_entry_targets_sulfurs(self):
        """Disulfide SMARTS ``[#6][SX2][SX2][#6]`` -- attachment indices are
        the two ``SX2`` atoms (SMARTS positions 1, 2)."""
        from orthonym.rules.seniority import PG_ATTACHMENT_INDICES
        assert PG_ATTACHMENT_INDICES["disulfide"] == [1, 2]

    def test_default_is_zero_for_back_compat(self):
        """A FG name not present in the table must default to ``[0]``
        (preserves pre- behaviour for every other FG)."""
        from orthonym.rules.parent_selection import _pg_attachment_atoms
        # Synthetic match tuple; FG name not in dict.
        atoms = _pg_attachment_atoms("not_a_real_fg", (10, 20, 30))
        assert atoms == [10]

    def test_disulfide_returns_both_sulfurs(self):
        """For disulfide, the helper returns BOTH heteroatom indices so
        downstream comparators can use ``min(locants)`` per IUPAC."""
        from orthonym.rules.parent_selection import _pg_attachment_atoms
        # Synthetic: (flanking_C, S, S, flanking_C).
        atoms = _pg_attachment_atoms("disulfide", (5, 6, 7, 8))
        assert atoms == [6, 7]

    def test_none_fg_name_defaults_to_zero(self):
        """If FG name is None (caller didn't thread it), behaves as default."""
        from orthonym.rules.parent_selection import _pg_attachment_atoms
        atoms = _pg_attachment_atoms(None, (10, 20, 30))
        assert atoms == [10]
