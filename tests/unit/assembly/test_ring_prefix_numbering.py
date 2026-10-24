"""The numbering of a decorated ring prefix when the free valence leaves the ring two ways round.

 NUMBERING (the Blue Book) decides it in this order: (f) "detachable alphabetized prefixes,
all considered together in a series of increasing numerical order" (:3300), then (g) "lowest locants
for the substituent cited first as a prefix in the name" (:3307; ``1-methyl-4-nitronaphthalene (PIN)
(not 4-methyl-1-nitronaphthalene)``), then (j) the CIP descriptors Z, R before E, S (:3346).

``substituent_enumerator._recursive_fragment_substituent_name`` numbered the ring by (f) and left the
rest to the first of the tied numberings it met, which is the order of the atoms of the input: the
mesylate of a cyclohexane-pentol came out as '2,3-dihydroxy-4,5,6-tris[(methanesulfonyl)oxy]' for
one spelling and '5,6-dihydroxy-2,3,4-tris[...]' for another ('hydroxy' is cited before
'(methanesulfonyl)oxy', so it takes the lower locants). ``assembly.ring_prefix_numbering`` applies
(g) then (j) to the numberings that tie on everything before them.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

#: (what, SMILES, the name the criteria give)
CASES = [
    ("hydroxy cited before (methanesulfonyl)oxy: the pentol mesylate",
     "CS(=O)(=O)OC1C(C(C(C(C1OS(=O)(=O)C)OS(=O)(=O)C)OS(=O)(=O)C)O)O",
     "2,3-dihydroxy-4,5,6-tris[(methanesulfonyl)oxy]cyclohexyl methanesulfonate"),
    ("three hydroxy and two (methanesulfonyl)oxy, 'hydroxy' lower",
     "CS(=O)(=O)OC1C(O)C(O)C(O)C(OS(C)(=O)=O)C1OS(C)(=O)=O",
     "2,3,4-trihydroxy-5,6-bis[(methanesulfonyl)oxy]cyclohexyl methanesulfonate"),
    ("chloro cited before methyl, both next to the free valence",
     "CS(=O)(=O)OC1C(Cl)CCCC1C",
     "2-chloro-6-methylcyclohexyl methanesulfonate"),
    ("oxo cited after chloro and methyl",
     "CS(=O)(=O)OC1C(Cl)C(=O)CCC1C",
     "2-chloro-6-methyl-3-oxocyclohexyl methanesulfonate"),
]


def _spellings(smiles, n=6):
    mol = Chem.MolFromSmiles(smiles)
    out = list(dict.fromkeys([smiles, Chem.MolToSmiles(mol),
                              *Chem.MolToRandomSmilesVect(mol, n, randomSeed=5)]))
    assert len(out) >= 3
    return out


@pytest.mark.parametrize("what, smiles, name", CASES, ids=[c[0] for c in CASES])
def test_the_prefix_cited_first_takes_the_lowest_locants_in_every_spelling(what, smiles, name):
    """ (g) (:3307): one name for every spelling, and it reads back to the input."""
    got = {s: name_best_effort(s).get("name") for s in _spellings(smiles)}
    assert set(got.values()) == {name}, got
    assert name_is_rt_exact(name, smiles)


def test_j_key_prefers_the_descriptor_r_at_the_lower_locant():
    """ (j) (:3346): "the lower locant is assigned to CIP stereodescriptors Z, R, M, and r
    (pseudoasymmetry) that are preferred to E, S, P, and s"."""
    from orthonym.assembly.ring_prefix_numbering import j_key
    from rdkit.Chem import rdCIPLabeler
    mol = Chem.MolFromSmiles("C[C@H]1CCC[C@@H](C)C1")
    rdCIPLabeler.AssignCIPLabels(mol)
    centres = {a.GetIdx(): a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")}
    assert sorted(centres.values()) == ["R", "S"], centres
    r_atom = next(i for i, c in centres.items() if c == "R")
    s_atom = next(i for i, c in centres.items() if c == "S")
    core = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    low_r = {**{a: 9 for a in core}, r_atom: 1, s_atom: 3}
    low_s = {**{a: 9 for a in core}, s_atom: 1, r_atom: 3}
    assert j_key(mol, core, low_r) < j_key(mol, core, low_s)
    assert j_key(mol, core, low_r) == ((1,), (3,))


def test_a_numbering_that_the_automorphisms_do_not_tie_is_not_offered():
    """The numberings offered to (g) are ``pos`` composed with automorphisms of the core that keep
    each atom's element, hydrogen count and exocyclic double bonds: a ring that carries one chlorine
    and one methyl has two (the two ways round from C-1), and the winner is the one that numbers the
    chlorine ('chloro' is cited before 'methyl') lower."""
    from orthonym.assembly.ring_prefix_numbering import g_winners, tied_numberings
    mol = Chem.MolFromSmiles("C1C(Cl)CCCC1C")      # a0 free valence; Cl on a2, CH3 (a7) on a6
    core = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    order = [0]
    while len(order) < len(core):
        order.append(next(n.GetIdx() for n in mol.GetAtomWithIdx(order[-1]).GetNeighbors()
                          if n.IsInRing() and n.GetIdx() not in order))
    pos = {a: i + 1 for i, a in enumerate(order)}
    cl = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl")
    cl_carrier = mol.GetAtomWithIdx(cl).GetNeighbors()[0].GetIdx()
    me_carrier = next(n.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C" and not a.IsInRing()
                      for n in a.GetNeighbors())
    named = [(cl_carrier, "chloro"), (me_carrier, "methyl")]
    assert len(tied_numberings(mol, core, 0, pos, [cl_carrier, me_carrier])) == 2
    (winner,) = g_winners(mol, core, 0, pos, named)
    assert winner[cl_carrier] < winner[me_carrier]
    # and whichever way round the numbering was found, the same one wins
    flipped = {a: (len(core) + 2 - p if p > 1 else 1) for a, p in pos.items()}
    (winner2,) = g_winners(mol, core, 0, flipped, named)
    assert winner2 == winner
