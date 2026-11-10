"""A von Baeyer cage named as a substituent does not name its own parent as one of its prefixes.

 NUMBERING (the Blue Book): "(c) principal characteristic groups and free valences (suffixes)"
(:3256) take their low locants before "(f) detachable alphabetized prefixes, all considered together in a
series of increasing numerical order" (:3300). The bond from a cage to the rest of the molecule is the
free valence of the group that is being named, so it counts at (c) and is no prefix at (f) or (g).

``terminal_ring_name`` used to analyse such a cage without the atom set of its group, so the bond to the
parent was a prefix: the (g) tie-break named it ("name_substituent_for_ordering"), that name held the
parent's own cages, and each of them did the same for the cage that called it. One 30-atom molecule
(PubChem CID 98563356) ran ``choose_numbering`` 1,922 times to a recursion depth of 23 and took 71 s; with
the group passed it runs it about 100 times and takes about 7 s. The test counts the calls, not the seconds.
"""
import pytest
from rdkit import Chem

from orthonym.rules import cage_numbering
from orthonym.rules.terminal_fragment import terminal_fragment_name
from orthonym.rules.terminal_ring import terminal_ring_name
from tests.support.large_polycycles import be

pytestmark = pytest.mark.unit

# PubChem CID 98563356: a norbornene spiro-cyclopropane and a benzodioxole on one pyrrolidinone
CID_98563356 = "C1CC(=O)N([C@H]1CCNCC2=CC3=C(C=C2Cl)OCO3)C[C@@H]4C[C@H]5C=C[C@H]4C56CC6"


class _ReEntered(BaseException):
    """Not an ``Exception``: the namer's own ``except Exception`` guards must not swallow it, so that the
    loop is cut at its first turn instead of running for a minute before the assertion reads the count."""


def _spy_choose_numbering(monkeypatch):
    """Record every call of ``choose_numbering`` and cut the run when a cage is numbered again while its
    own numbering is still being chosen (a cage whose prefix holds that same cage)."""
    stats = {"calls": 0, "depth": 0, "max_depth": 0}
    active = []
    real = cage_numbering.choose_numbering

    def spy(mol, ring_atoms, candidates, suffix_pairs, substituent_atoms=None):
        key = (mol.GetNumAtoms(), frozenset(ring_atoms))
        if key in active:
            raise _ReEntered(sorted(ring_atoms))
        stats["calls"] += 1
        active.append(key)
        stats["depth"] = len(active)
        stats["max_depth"] = max(stats["max_depth"], len(active))
        try:
            return real(mol, ring_atoms, candidates, suffix_pairs, substituent_atoms)
        finally:
            active.pop()

    monkeypatch.setattr(cage_numbering, "choose_numbering", spy)
    return stats


@pytest.mark.opsin_gate
def test_a_cage_substituent_does_not_rank_the_bond_to_its_parent_as_a_prefix(monkeypatch):
    stats = _spy_choose_numbering(monkeypatch)
    try:
        name = be(CID_98563356)["name"]
    except _ReEntered as loop:
        pytest.fail(f"the numbering of the cage {loop.args[0]} was asked for again inside its own choice")
    assert name
    assert stats["calls"] > 0           # the trace sees the numbering at all (see the norbornyl positive control below)
    # measured: 100 calls at depth 2 (1,922 at depth 23 before). The bound is loose, the re-entry above is exact.
    assert stats["calls"] < 400, stats
    assert stats["max_depth"] <= 3, stats


def _chloro_norbornyl_fragment():
    """2-[(3-chlorobicyclo[2.2.1]heptan-2-yl)methyl]pyridine: the cage, its chlorine, and the free valence."""
    mol = Chem.MolFromSmiles("ClC1C(Cc3ccccn3)C2CCC1C2")
    cage = sorted(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing() and not a.GetIsAromatic())
    attach = next(i for i in cage for n in mol.GetAtomWithIdx(i).GetNeighbors()
                  if n.GetSymbol() == "C" and not n.IsInRing())
    return mol, cage, attach, {0} | set(cage)       # atom 0 is the chlorine


def test_the_free_valence_of_a_cage_substituent_has_the_lower_locant_than_its_prefix():
    # (c) before (f): the free valence takes locant 2 and the chlorine 3, never chloro-2 and -3-yl
    mol, cage, attach, group = _chloro_norbornyl_fragment()
    assert terminal_ring_name(mol, cage, attach, substituent_atoms=group).name == "bicyclo[2.2.1]heptan-2-yl"
    assert terminal_fragment_name(mol, group, attach).name == "3-chlorobicyclo[2.2.1]heptan-2-yl"


def test_a_cage_without_a_group_keeps_the_numbering_of_the_parent_of_its_molecule():
    # No group: ``substituent_atoms`` is unchanged by the default, the ring system is the parent of ``mol`` and its
    # neighbours (the chlorine and the CH2) are prefixes, numbered as before. This is NOT the correct -yl name: the
    # free valence takes locant 2 at (c) "principal characteristic groups and free valences (suffixes)"
    # (the Blue Book), before the prefixes at (f); 'heptan-3-yl' is what the call returned when every caller was
    # in this position, and it is pinned only to show that a caller that holds no group is not changed.
    mol, cage, attach, _group = _chloro_norbornyl_fragment()
    assert terminal_ring_name(mol, cage, attach).name == "bicyclo[2.2.1]heptan-3-yl"


@pytest.mark.parametrize("smiles, calls_expected, name", [
    ("OCC1CCC2(CC1)CCCC2", 0, "spiro[4.5]decan-8-yl"),         # the spiro analyzer numbers a ring-only submolecule
    ("OCC1CC2CCC1C2", 1, "bicyclo[2.2.1]heptan-2-yl"),          # positive control: a von Baeyer cage chooses a numbering
])
def test_only_the_von_baeyer_analyzer_reads_the_group_the_spiro_analyzer_never_sees_the_parent(
        monkeypatch, smiles, calls_expected, name):
    stats = _spy_choose_numbering(monkeypatch)
    mol = Chem.MolFromSmiles(smiles)
    ring = sorted(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
    attach = next(i for i in ring for n in mol.GetAtomWithIdx(i).GetNeighbors()
                  if n.GetIdx() not in ring and n.GetSymbol() == "C")
    # the group is the ring system: the CH2OH is the parent
    assert terminal_ring_name(mol, ring, attach, substituent_atoms=set(ring)).name == name
    assert stats["calls"] == calls_expected


def test_the_branch_of_a_ring_prefix_of_a_chain_parent_is_the_ring_and_its_groups():
    from orthonym.assembly.ring_branch import ring_branch_atoms
    mol = Chem.MolFromSmiles("OCCC1C(Cl)C2CCC1C2")          # the chain O-C-C-, the norbornane, its chlorine
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    chain = {0, 1, 2}
    cl = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl")
    assert ring_branch_atoms(mol, ring, chain) == ring | {cl}
