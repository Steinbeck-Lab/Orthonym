"""A symmetric natural-product scaffold gets its numbering by rule, not by input order.

perception.natural_products used the FIRST substructure match; for tropane (C1<->C5)
that depended on how the SMILES was written, so one molecule got '(1R,3r,5S)-...' or
'(1S,3r,5R)-...' (full gate, determinism). The match is now chosen by the
ladder (the Blue Book;:52478 applies.. to NP derivatives):
(c) the suffix / ester '-yl' (:3256), (e) multiple bonds then double bonds
(:3288), (f) the prefixes together (:3301), (g) the prefix cited first (:3307),
(j) the CIP descriptors with their locants (:3346), then canonical ranks.
 fix a performance pass (wp2): the first tier used to lump the suffix with the prefixes,
so '(1R,2S,5S,7S)-2-methyltropan-7-ol' shipped where (c) gives '...-4-methyl-
tropan-6-ol' (the tropane mirror swaps C2/C4 and C6/C7; '6-hydroxy-8-methyltropan-
3-one',:29593, has the prefix at 6). Every name below was checked by an
independent OPSIN full-InChIKey round trip.
"""
import random

import pytest
from rdkit import Chem

from orthonym import Orthonym


def _respell(smiles: str, seed: int) -> str:
    mol = Chem.MolFromSmiles(smiles)
    order = list(range(mol.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    return Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)


@pytest.mark.opsin_gate
def test_tropane_ester_name_does_not_depend_on_spelling():
    smiles = "CN1[C@@H]2CC[C@H]1C[C@H](C2)OC(=O)C(CO)c1ccccc1"
    namer = Orthonym(style="pin")
    names = {namer.name(_respell(smiles, s)) for s in range(8)} | {namer.name(smiles)}
    assert names == {"(1R,3r,5S)-tropan-3-yl 3-hydroxy-2-phenylpropanoate"}


P14_4_LADDER = [
    # (1S,4S,5R,6S)-4-methyltropan-6-ol
    ('CN1[C@@H]2C[C@H](O)[C@H]1[C@@H](C)CC2', '(1S,4S,5R,6S)-4-methyltropan-6-ol'),
    ('C1[C@@H]([C@H]2N(C)[C@@H](C1)C[C@@H]2O)C', '(1S,4S,5R,6S)-4-methyltropan-6-ol'),
    ('C1[C@H]2N([C@H]([C@@H](C)CC2)[C@H]1O)C', '(1S,4S,5R,6S)-4-methyltropan-6-ol'),
    ('[C@H]12[C@H](C[C@H](CC[C@@H]1C)N2C)O', '(1S,4S,5R,6S)-4-methyltropan-6-ol'),
    ('C1[C@@H]([C@H]2N([C@H](C[C@@H]2O)C1)C)C', '(1S,4S,5R,6S)-4-methyltropan-6-ol'),
    ('N1(C)[C@H]2CC[C@@H]([C@@H]1[C@@H](O)C2)C', '(1S,4S,5R,6S)-4-methyltropan-6-ol'),
    # (1S,4S,5R)-4-hydroxytropan-6-one
    ('CN1[C@@H]2CC(=O)[C@H]1[C@@H](O)CC2', '(1S,4S,5R)-4-hydroxytropan-6-one'),
    ('N1(C)[C@@H]2CC([C@H]1[C@@H](O)CC2)=O', '(1S,4S,5R)-4-hydroxytropan-6-one'),
    ('C1[C@H]2N([C@H]([C@@H](O)CC2)C1=O)C', '(1S,4S,5R)-4-hydroxytropan-6-one'),
    ('[C@H]12C(C[C@H](CC[C@@H]1O)N2C)=O', '(1S,4S,5R)-4-hydroxytropan-6-one'),
    ('C1[C@@H]([C@H]2N([C@H](CC2=O)C1)C)O', '(1S,4S,5R)-4-hydroxytropan-6-one'),
    ('N1(C)[C@H]2CC[C@@H]([C@@H]1C(=O)C2)O', '(1S,4S,5R)-4-hydroxytropan-6-one'),
    # (1S,4S,5R,6S)-4-methyltropan-6-yl acetate
    ('CN1[C@@H]2C[C@H](OC(C)=O)[C@H]1[C@@H](C)CC2', '(1S,4S,5R,6S)-4-methyltropan-6-yl acetate'),
    ('[C@H]12[C@H](CC[C@@H](C[C@@H]2OC(C)=O)N1C)C', '(1S,4S,5R,6S)-4-methyltropan-6-yl acetate'),
    ('C1C[C@@H]([C@H]2N(C)[C@@H]1C[C@@H]2OC(C)=O)C', '(1S,4S,5R,6S)-4-methyltropan-6-yl acetate'),
    ('C1[C@H](C)[C@H]2N([C@H](C[C@@H]2OC(C)=O)C1)C', '(1S,4S,5R,6S)-4-methyltropan-6-yl acetate'),
    ('C[C@H]1CC[C@H]2C[C@H](OC(C)=O)[C@@H]1N2C', '(1S,4S,5R,6S)-4-methyltropan-6-yl acetate'),
    ('[C@@H]1(C)[C@H]2N([C@@H](CC1)C[C@@H]2OC(C)=O)C', '(1S,4S,5R,6S)-4-methyltropan-6-yl acetate'),
    # (1S,5R)-trop-2-ene
    ('CN1[C@@H]2CC[C@H]1C=CC2', '(1S,5R)-trop-2-ene'),
    ('C1C[C@@H]2N(C)[C@@H](CC2)C=1', '(1S,5R)-trop-2-ene'),
    ('C1C[C@@H]2CC=C[C@H]1N2C', '(1S,5R)-trop-2-ene'),
    ('C1C=C[C@H]2N([C@@H]1CC2)C', '(1S,5R)-trop-2-ene'),
    ('C1[C@H]2N(C)[C@H](C=CC2)C1', '(1S,5R)-trop-2-ene'),
    ('C1=C[C@H]2N(C)[C@@H](C1)CC2', '(1S,5R)-trop-2-ene'),
    # (1S,5S,7S)-7-methyltrop-2-ene
    ('CN1[C@@H]2C[C@H](C)[C@H]1C=CC2', '(1S,5S,7S)-7-methyltrop-2-ene'),
    ('C1[C@H]2N(C)[C@@H]([C@H]1C)C=CC2', '(1S,5S,7S)-7-methyltrop-2-ene'),
    ('N1([C@H]2CC=C[C@@H]1[C@H](C2)C)C', '(1S,5S,7S)-7-methyltrop-2-ene'),
    ('[C@@H]12N(C)[C@H](C=CC2)[C@@H](C)C1', '(1S,5S,7S)-7-methyltrop-2-ene'),
    ('[C@H]1(C)C[C@@H]2CC=C[C@H]1N2C', '(1S,5S,7S)-7-methyltrop-2-ene'),
    ('C[C@H]1C[C@@H]2CC=C[C@H]1N2C', '(1S,5S,7S)-7-methyltrop-2-ene'),
    # (1S,2R,4S,5R)-2-chloro-4-methyltropane
    ('C[C@H]1C[C@@H](Cl)[C@@H]2CC[C@H]1N2C', '(1S,2R,4S,5R)-2-chloro-4-methyltropane'),
    ('Cl[C@@H]1C[C@H](C)[C@@H]2N(C)[C@H]1CC2', '(1S,2R,4S,5R)-2-chloro-4-methyltropane'),
    ('[C@H]1(C[C@H](C)[C@@H]2N(C)[C@H]1CC2)Cl', '(1S,2R,4S,5R)-2-chloro-4-methyltropane'),
    ('C1[C@H]2[C@H](Cl)C[C@H](C)[C@H](N2C)C1', '(1S,2R,4S,5R)-2-chloro-4-methyltropane'),
    ('N1([C@H]2[C@@H](C)C[C@@H](Cl)[C@@H]1CC2)C', '(1S,2R,4S,5R)-2-chloro-4-methyltropane'),
    ('[C@H]1(C)C[C@H]([C@H]2N(C)[C@@H]1CC2)Cl', '(1S,2R,4S,5R)-2-chloro-4-methyltropane'),
    # (1S,2S,4S,5R)-2-chloro-4-methyltropane
    ('C[C@H]1C[C@H](Cl)[C@@H]2CC[C@H]1N2C', '(1S,2S,4S,5R)-2-chloro-4-methyltropane'),
    ('[C@H]1(C)C[C@H](Cl)[C@H]2N([C@@H]1CC2)C', '(1S,2S,4S,5R)-2-chloro-4-methyltropane'),
    ('[C@@H]1(C[C@H](C)[C@@H]2N(C)[C@H]1CC2)Cl', '(1S,2S,4S,5R)-2-chloro-4-methyltropane'),
    ('C1[C@H]2[C@@H](Cl)C[C@H](C)[C@H](N2C)C1', '(1S,2S,4S,5R)-2-chloro-4-methyltropane'),
    ('N1([C@H]2[C@@H](C)C[C@H](Cl)[C@@H]1CC2)C', '(1S,2S,4S,5R)-2-chloro-4-methyltropane'),
    ('[C@H]1(C)C[C@@H]([C@H]2N(C)[C@@H]1CC2)Cl', '(1S,2S,4S,5R)-2-chloro-4-methyltropane'),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", P14_4_LADDER)
def test_scaffold_numbering_follows_the_p14_4_ladder(smiles, expected):
    assert Orthonym(style="pin").name(smiles) == expected
