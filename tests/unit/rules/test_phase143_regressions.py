"""Tests for Phase 143 regression investigation.

Validates that the heterocyclic ring locant fix in _get_polycyclic_attachment_locant
correctly assigns IUPAC positions using Hantzsch-Widman numbering for retained-name
heterocyclic rings used as substituents.
"""
import pytest
from orthonym.namer import name_compound


class TestHeterocyclicSubstituentLocants:
    """Verify correct IUPAC locant numbering for heterocyclic ring-as-substituent.

    Wave2 T3a rewrite: these originally asserted substrings of WHOLE-MOLECULE
    raw names for antibiotic-scale structures. Those raw names silently
    dropped the heterocyclyl ring's own substituents (production, jar-armed,
    was ALREADY 'unknown organic compound' at HEAD for all three — verified
    by worktree A/B), and the T3a constitution-conservation guard now makes
    the raw path honestly fail closed too. The locant logic under test is
    the ring-substituent namer's — so assert it at the FRAGMENT level, which
    is exactly the machinery the original Phase-143 fix patched.
    Naming the decorated heterocyclyl substituent itself (e.g.
    2,2-dimethyl-1,3-dioxolan-5-yl) is the documented 3c follow-on.
    """

    @staticmethod
    def _fragment_name(smiles, ring_filter, attach_filter):
        from rdkit import Chem
        from orthonym.rules.ring_substituents import get_ring_substituent_name
        mol = Chem.MolFromSmiles(smiles)
        ring = next(
            r for r in mol.GetRingInfo().AtomRings() if ring_filter(mol, r)
        )
        attach = next(a for a in ring if attach_filter(mol, a, set(ring)))
        return get_ring_substituent_name(mol, tuple(ring), attach)

    def test_dioxolane_locant_not_at_oxygen(self):
        """1,3-dioxolane: attachment at C should give position 5, not 1 (which is O)."""
        def ring_filter(mol, r):
            syms = sorted(mol.GetAtomWithIdx(a).GetSymbol() for a in r)
            return len(r) == 5 and syms.count('O') == 2

        def attach_filter(mol, a, ring_set):
            # the attachment ring C: its exocyclic neighbour is the chain
            # CH2 (>= 2 heavy neighbours), not a methyl
            return any(
                nb.GetIdx() not in ring_set and nb.GetDegree() >= 2
                for nb in mol.GetAtomWithIdx(a).GetNeighbors()
            )

        name = self._fragment_name(
            "C=C[C@](C)(O)CCC=C(C)CCC1OC(C)(C)OC1(C)C",
            ring_filter, attach_filter,
        )
        assert name is not None and "dioxolan-5-yl" in name
        # Must NOT be dioxolan-1-yl (position 1 is oxygen)
        assert "dioxolan-1-yl" not in name

    def test_thiazole_locant_at_carbon_4(self):
        """Thiazole: S=1, C=2, N=3, C=4, C=5. Attachment at C-4 gives thiazol-4-yl."""
        def ring_filter(mol, r):
            syms = sorted(mol.GetAtomWithIdx(a).GetSymbol() for a in r)
            return (len(r) == 5 and 'S' in syms and 'N' in syms
                    and all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r))

        def attach_filter(mol, a, ring_set):
            # the attachment c: exocyclic neighbour is the oxime carbon
            # (>= 2 heavy neighbours), not the -NH2
            return any(
                nb.GetIdx() not in ring_set and nb.GetSymbol() == 'C'
                and nb.GetDegree() >= 2
                for nb in mol.GetAtomWithIdx(a).GetNeighbors()
            )

        name = self._fragment_name(
            r"C=CCO/N=C(\C(=O)N[C@H]1CN2CC(S(C)(=O)=O)=C(C(=O)O)N2C1=O)c1csc(N)n1",
            ring_filter, attach_filter,
        )
        assert name is not None and "thiazol-4-yl" in name
        # Must NOT be thiazol-1-yl (position 1 is sulfur)
        assert "thiazol-1-yl" not in name

    def test_thiazolidine_locant_at_carbon_2(self):
        """Thiazolidine: S=1, C=2, N=3, C=4, C=5. Attachment at C-2 gives thiazolidin-2-yl."""
        def ring_filter(mol, r):
            syms = sorted(mol.GetAtomWithIdx(a).GetSymbol() for a in r)
            return (len(r) == 5 and 'S' in syms and 'N' in syms
                    and not any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r))

        def attach_filter(mol, a, ring_set):
            # C-2 is the unique ring carbon bonded to BOTH the ring S and N
            nb_syms = {
                nb.GetSymbol() for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                if nb.GetIdx() in ring_set
            }
            return {'S', 'N'} <= nb_syms

        name = self._fragment_name(
            "CC1(C)SC(C(NC(=O)COc2ccccc2)C(=O)O)NC1C(=O)O",
            ring_filter, attach_filter,
        )
        assert name is not None and "thiazolidin-2-yl" in name
        # Must NOT be thiazolidin-3-yl (position 3 is nitrogen)
        assert "thiazolidin-3-yl" not in name


class TestOPSINParseRegressions:
    """Document the 8 compounds that lost OPSIN parseability in Phase 143.

    All 8 are trade-offs or OPSIN limitations: the new names are more correct
    IUPAC but OPSIN 2.9.0 cannot parse them. None of these had RT=1 in baseline.
    """

    def test_ajmaline_retained_name(self):
        """Ajmaline is correctly resolved as retained NP name (was VB hexacyclo...).

        Phase 157 cleanup: removed stale @pytest.mark.xfail. Phase 150
        OPSIN XML retained-name expansion (914 entries; commit `1d1301f8`)
        brought ajmaline into the registry; the test passes cleanly.
        """
        name = name_compound(
            "CC[C@H]1[C@@H]2CC3[C@@H]4N(C)c5ccccc5[C@]45C[C@@H](C2C5O)N3[C@@H]1O"
        )
        assert name == "ajmaline"

    def test_berberine_retained_name(self):
        """Berberine is correctly resolved as retained NP name (was VB tetracyclo...).

        Phase 157 cleanup: removed stale @pytest.mark.xfail. Phase 150
        OPSIN XML retained-name expansion brought berberine into the
        registry; the test passes cleanly.
        """
        name = name_compound("COc1cc2c(cc1OC)[C@H]1Cc3ccc(OC)c(OC)c3CN1CC2")
        assert name == "berberine"
