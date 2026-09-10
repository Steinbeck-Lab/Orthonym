""".1 S1 — unsaturation tier in ring_system_score.

Among ring systems that tie on every general criterion AND on the
 type hierarchy, the more-unsaturated ring is senior.
This breaks the benzene-vs-cyclohexane tie that previously resolved to the
arbitrary list-order winner.

Assertions check the RELATION (more-unsaturated scores lower under min),
never a literal bond count — RDKit reports benzene bonds as AROMATIC, so the
count is implementation-defined; only the ordering is contractual.
"""
from rdkit import Chem

from orthonym.rules.ring_selection import (
    ring_system_score,
    select_principal_ring_system,
)


def _systems(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol, [set(r) for r in mol.GetRingInfo().AtomRings()]


class TestUnsaturationTierRelation:
    def test_benzene_more_senior_than_cyclohexane(self):
        mol, systems = _systems("C1CCCCC1c1ccccc1")
        benzene = next(s for s in systems
                       if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in s))
        cyclohexane = next(s for s in systems
                           if not any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in s))
        # min selects the senior system -> benzene must score strictly lower
        assert ring_system_score(mol, benzene) < ring_system_score(mol, cyclohexane)

    def test_cyclohexene_more_senior_than_cyclohexane(self):
        mol, systems = _systems("C1CCCCC1C1CCCCC=1")  # cyclohexane + cyclohexene
        ene = next(s for s in systems
                   if any(mol.GetBondBetweenAtoms(a, b).GetBondType() == Chem.BondType.DOUBLE
                          for a in s for b in s
                          if mol.GetBondBetweenAtoms(a, b)))
        ane = next(s for s in systems if s != ene)
        assert ring_system_score(mol, ene) < ring_system_score(mol, ane)

    def test_unsaturation_appended_after_type_rank(self):
        # The tier must not perturb the senior elements: a heterocycle
        # still beats a more-unsaturated carbocycle (pyridine vs benzene-less
        # case). Use furan (O-het, aromatic) vs benzene (carbocycle, aromatic):
        # furan wins on (a) heteroatom, regardless of unsaturation.
        mol, systems = _systems("c1ccc(-c2ccco2)cc1")
        furan = next(s for s in systems
                     if any(mol.GetAtomWithIdx(a).GetSymbol() == 'O' for a in s))
        benzene = next(s for s in systems
                       if all(mol.GetAtomWithIdx(a).GetSymbol() == 'C' for a in s))
        assert ring_system_score(mol, furan) < ring_system_score(mol, benzene)

    def test_score_tuple_length_is_29(self):
        mol, systems = _systems("c1ccccc1")
        assert len(ring_system_score(mol, systems[0])) == 29


class TestSelectPrincipalRingSystem:
    def test_benzene_wins_among_rings(self):
        mol, systems = _systems("C1CCCCC1c1ccccc1")
        winner = set(select_principal_ring_system(mol, systems))
        assert all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in winner)
