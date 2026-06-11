"""WS-A task 9 — fused-ring routing + ring-system substituent emission.

Two coupled defect families (among-rings gold reds, cluster A):

ROUTING (A-i): the PAH early-return (namer) and the whole-molecule
complex-ring path (tier_a_ring sub-path 1) fire BEFORE the P-44 among-rings
decision, so a non-senior fused system seizes the parent
(naphthalene over furan/pyridine/pyrrolidine; benzofuran over pyridine) and
even a PG on another ring is ignored. `select_principal_ring_system` returns
the correct P-44.2 winner for every one of these rows — the fix is to gate
the early paths on that decision, not to re-derive it.

EMISSION (A-ii): the fused-parent paths name ring-system substituents by
carbon count as alkyl (phenyl -> 'hexyl' on naphthalene!) or silently drop
them (fused_rings._identify_fused_substituent returns None for ring-rooted
fragments; polycyclics._identify_pah_substituent same disease). The
universal producer (substituent_enumerator.name_substituent /
ring_substituents.get_ring_substituent_name) already names these fragments
on the chain-parent path and must be the single delegate.

LOCANT: get_ring_substituent_name cites the attachment by one arbitrary
substructure match ('naphthalen-6-yl'); P-29.2/P-14.4 require the LOWEST
locant over the ring system's symmetry ('naphthalen-2-yl').

All PINs below are the OPSIN-verified among-rings gold oracle values.
"""

import pytest
from rdkit import Chem

from orthonym.namer import name_compound


@pytest.mark.unit
class TestFusedRoutingGold:
    """A-i: parent must follow the P-44 among-rings decision."""

    def test_naphthalenyl_furan(self):
        assert (
            name_compound("c1ccc2cc(-c3ccco3)ccc2c1").strip()
            == "2-(naphthalen-2-yl)furan"
        )

    def test_naphthalenylmethyl_pyridine(self):
        assert (
            name_compound("C(c1ccc2ccccc2c1)c1ccccn1").strip()
            == "2-[(naphthalen-2-yl)methyl]pyridine"
        )

    def test_naphthalenyl_pyrrolidine(self):
        assert (
            name_compound("c1ccc2cc(N3CCCC3)ccc2c1").strip()
            == "1-(naphthalen-2-yl)pyrrolidine"
        )

    def test_benzofuranyl_pyridine(self):
        assert (
            name_compound("c1ccc2c(c1)cc(-c3ccccn3)o2").strip()
            == "2-(1-benzofuran-2-yl)pyridine"
        )

    def test_pg_ring_beats_pah(self):
        # P-44.1: the -ol bearing cyclohexane is the parent; naphthalene is a
        # substituent even though it is the P-44.2-senior ring system.
        # (Currently names bare 'naphthalene' — PG silently discarded.)
        assert (
            name_compound("OC1CCCCC1c1ccc2ccccc2c1").strip()
            == "2-(naphthalen-2-yl)cyclohexan-1-ol"
        )


@pytest.mark.unit
class TestFusedParentSubstituentEmission:
    """A-ii: a fused PARENT must emit its ring-system substituents."""

    def test_pyridinyl_quinoline(self):
        assert (
            name_compound("c1ccc2nc(-c3ccccn3)ccc2c1").strip()
            == "2-(pyridin-2-yl)quinoline"
        )

    def test_indolyl_quinoline(self):
        assert (
            name_compound("c1ccc2nc(-c3cc4ccccc4[nH]3)ccc2c1").strip()
            == "2-(1H-indol-2-yl)quinoline"
        )

    def test_phenylnaphthalene_not_hexyl(self):
        # The silent-corruption case: phenyl was being named 'hexyl' by the
        # carbon-count alkyl identifier (2-hexylnaphthalene — a DIFFERENT
        # molecule). Naphthalene stays parent (P-44.2 senior over benzene).
        assert (
            name_compound("c1ccc(-c2ccc3ccccc3c2)cc1").strip()
            == "2-phenylnaphthalene"
        )


@pytest.mark.unit
class TestFusedRoutingProtect:
    """Currently-correct neighbors that must not move."""

    def test_naphthalene_bare(self):
        assert name_compound("c1ccc2ccccc2c1").strip() == "naphthalene"

    def test_methylnaphthalene(self):
        assert name_compound("Cc1ccc2ccccc2c1").strip() == "2-methylnaphthalene"

    def test_methylquinoline(self):
        assert name_compound("Cc1ccc2ccccc2n1").strip() == "2-methylquinoline"

    def test_quinoline_bare(self):
        assert name_compound("c1ccc2ncccc2c1").strip() == "quinoline"

    def test_phenylfuran(self):
        # The S2 keystone row — among-rings winner is furan.
        assert name_compound("c1ccc(-c2ccco2)cc1").strip() == "2-phenylfuran"

    def test_phenylthiazole(self):
        # Gold oracle accepts both forms (accept_also: 4-phenylthiazole).
        assert name_compound("c1ccc(-c2cscn2)cc1").strip() in (
            "4-phenyl-1,3-thiazole",
            "4-phenylthiazole",
        )


@pytest.mark.unit
class TestRingSubstituentAttachmentLocant:
    """P-29.2: the free valence takes the LOWEST locant the ring system's
    numbering (incl. symmetry) allows — naphthalen-2-yl, never -6-yl."""

    def test_naphthalenyl_lowest_locant(self):
        from orthonym.rules.ring_substituents import get_ring_substituent_name
        mol = Chem.MolFromSmiles("c1ccc2cc(-c3ccco3)ccc2c1")
        furan = {6, 7, 8, 9, 10}
        naph = tuple(a for a in range(mol.GetNumAtoms()) if a not in furan)
        attach = next(
            a for a in naph
            for nbr in mol.GetAtomWithIdx(a).GetNeighbors()
            if nbr.GetIdx() in furan
        )
        assert get_ring_substituent_name(mol, naph, attach) == "naphthalen-2-yl"
