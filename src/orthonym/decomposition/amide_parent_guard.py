"""An amide named from its acid part must not leave a second amide of the same ring system as a prefix (2026-11-20).

 (the principal chain or ring has the maximum number of principal characteristic groups) and: when two
carboxamide groups sit on one ring system, both are the suffix ('naphthalene-1,8-dicarboxamide', N groups
N and N'). The decomposition cuts one amide bond and names the acid part with the other amide as a prefix
('8-(methylcarbamoyl)naphthalene-1-carboxylic acid'), so the assembled name cited one amide as the suffix and the other
as 'methylcarbamoyl': 'N-methyl-8-(methylcarbamoyl)naphthalene-1-carboxamide' shipped pin_verified (lb2+fa review)."""
from typing import Optional

from rdkit import Chem

_COOH = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
_AMIDE = Chem.MolFromSmarts('[CX3](=O)[NX3]')


def second_amide_on_acid_ring(acid_smiles: Optional[str]) -> bool:
    """True when the acid part (``acid_smiles``, the cut amide written as its acid) carries an amide whose carbonyl
    carbon is bonded to the ring system that bears the carboxyl carbon."""
    mol = Chem.MolFromSmiles(acid_smiles) if acid_smiles else None
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    systems = []
    for ring in ring_info.AtomRings():
        ring = set(ring)
        for system in systems:
            if len(system & ring) >= 2:
                system |= ring
                break
        else:
            systems.append(ring)

    def ring_system_of(carbonyl_idx):
        hosts = [nb.GetIdx() for nb in mol.GetAtomWithIdx(carbonyl_idx).GetNeighbors() if nb.IsInRing()]
        return next((frozenset(s) for s in systems if hosts and hosts[0] in s), None)

    carboxyl_carbons = {m[0] for m in mol.GetSubstructMatches(_COOH)}
    acid_systems = {ring_system_of(c) for c in carboxyl_carbons} - {None}

    def ring_carboxamide(match):
        # a carboxamide of the ring system: its carbonyl carbon outside every ring (a lactam C=O belongs to its ring,
        # end check CT7), not the carboxyl itself (a carboxyl on a ring N, the acid part of a urea, also matches the
        # amide pattern), and its amide N outside that system (an acyl group on a ring N, '3-butanoyl', makes the
        # ring the amine part, not a second carboxamide)
        carbon, nitrogen = match[0], match[2]
        system = ring_system_of(carbon)
        return (carbon not in carboxyl_carbons and not mol.GetAtomWithIdx(carbon).IsInRing()
                and system in acid_systems and nitrogen not in system)

    return any(ring_carboxamide(m) for m in mol.GetSubstructMatches(_AMIDE))
