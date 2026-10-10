"""Leads program L3, item N8f (producer): ONE principal-chain selector for acyclic substituents.

The names of acyclic substituent prefixes (radicals and the prefixes of ordinary molecules alike)
depended on the order of the atoms of the input, and the names of some orders were not the PIN.
 'THE PRINCIPAL SUBSTITUENT CHAIN' (the Blue Book) chooses the principal chain by
criteria "applied successively in the order given": (b) the longest chain,:22660);
(d) the greater number of multiple bonds regardless of type, then of double bonds,
:22682); (h) the lowest locant for the free valence,:22718); (i) the lowest locants
for multiple bonds, then for double bonds,:22726; 'hept-1-en-6-yn-4-yl (preferred
prefix)',:17256); (k) the greatest number of substituents,:22740); (l) their lowest
locants,:22768; '4-hydroxy-3-(2-hydroxyethyl)pentan-2-yl (preferred prefix)',
:22780); (m) the lowest locants for the substituent cited first in alphanumerical order
,:22802).

``substituent_naming._located_acyclic_alkyl_name``, ``_located_fg_assemble`` and
``_name_branched_alkenyl_substituent`` took the first of tied arms (or the first of tied chains) in
atom order, and ``_name_unsaturated_chain`` had no double-bond tie-break. They now share
``_principal_chain_candidates``. Each molecule below is named from 12 atom orders (the given one
and 11 seeded permutations): every order must give the one preferred name, and each preferred name
is read back to the molecule by a fresh OPSIN call.
"""
import random

import pytest
from rdkit import Chem

from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


def _orders(smiles, n=12, seed=20261009):
    """``smiles`` and ``n - 1`` SMILES of the same molecule with the atoms in seeded random order."""
    mol = Chem.MolFromSmiles(smiles)
    rng = random.Random(seed)
    out = [smiles]
    for _ in range(200):
        perm = list(range(mol.GetNumAtoms()))
        rng.shuffle(perm)
        v = Chem.MolToSmiles(Chem.RenumberAtoms(mol, perm), canonical=False)
        if v not in out:
            out.append(v)
        if len(out) == n:
            break
    return out


#: (molecule, the preferred name, the criterion that chooses it)
PREFERRED = [
    # (l) lowest locants for substituents, {3,4} not {3,5}: the book's own pair,:22780
    ("C[CH]C(CCO)C(C)O", "4-hydroxy-3-(2-hydroxyethyl)pentan-2-yl", "(l)"),
    ("OC(=O)c1ccc(cc1)C(C)C(CCO)C(C)O", "4-[4-hydroxy-3-(2-hydroxyethyl)pentan-2-yl]benzoic acid", "(l)"),
    # (d) the greater number of double bonds, as many multiple bonds in each chain
    ("C#CCC([CH]C)CC=C", "3-(prop-2-yn-1-yl)hex-5-en-2-yl", "(d)"),
    ("OC(=O)c1ccc(cc1)C(C)C(CC#C)CC=C", "4-[3-(prop-2-yn-1-yl)hex-5-en-2-yl]benzoic acid", "(d)"),
    # (i) the multiple bonds at {1,6} either way, the double bond at 1,:16489)
    ("C#CC[CH]CC=C", "hept-1-en-6-yn-4-yl", "(i)"),
    # (k) the maximum number of substituents: the book's pair at:22746
    ("C[CH]CCC(CC(C)Cl)C(Cl)C(C)Cl", "6,7-dichloro-5-(2-chloropropyl)octan-2-yl", "(k)"),
]


@pytest.mark.parametrize("smiles, pin, criterion", PREFERRED)
def test_every_atom_order_gives_the_one_preferred_name(smiles, pin, criterion):
    assert name_is_rt_exact(pin, smiles), pin
    orders = _orders(smiles)
    assert len(orders) >= 10, orders
    for order in orders:
        d, b = name_default(order), name_breadth(order)
        assert (d["name"], d["tier"]) == (pin, "pin_verified"), (order, d["name"], d["tier"])
        assert (b["name"], b["tier"]) == (pin, "pin_verified"), (order, b["name"], b["tier"])


@pytest.mark.parametrize("smiles, pin, criterion", PREFERRED[:1] + PREFERRED[2:3] + PREFERRED[4:5])
def test_the_pin_holds_at_both_tiers(smiles, pin, criterion):
    assert_pin_at_both_tiers(smiles, pin)


#: preferred prefixes the Blue Book prints, each of which was named from the first of tied arms
#: or not at all: (:22744,:22746), (:22774,:22778,:22780), (:22806)
BOOK_PREFIXES = [
    ("OC[CH]C", "1-hydroxypropan-2-yl"),                                    #:22744
    ("OCC[CH]C(C)O", "1,4-dihydroxypentan-3-yl"),                           #:22774
    ("BrCC[CH]C(C)C", "1-bromo-4-methylpentan-3-yl"),                       #:22778
    ("CC(Br)[CH]C(C)Cl", "2-bromo-4-chloropentan-3-yl"),                    #:22806, (m)
    ("CC[CH]CC(CC)CC(C)CC", "5-ethyl-7-methylnonan-3-yl"),                  #:22806
    ("C=CC[CH]CC=C", "hepta-1,6-dien-4-yl"),
    ("OC[C](C)C", "1-hydroxy-2-methylpropan-2-yl"),
    ("OCC(C)(C)[CH]C", "4-hydroxy-3,3-dimethylbutan-2-yl"),
]


@pytest.mark.parametrize("smiles, name", BOOK_PREFIXES)
def test_the_books_preferred_prefixes_come_from_every_atom_order(smiles, name):
    assert name_is_rt_exact(name, smiles), name
    for order in _orders(smiles, n=8):
        d = name_default(order)
        assert (d["name"], d["tier"]) == (name, "pin_verified"), (order, d["name"], d["tier"])


def test_the_pantoyl_chain_goes_through_the_arm_with_more_substituents():
    """The acyl-CoA pantoyl chain has three one-carbon arms at its quaternary C-3. The chain through
    the CH2-O has five substituents (1-oxo, 2-hydroxy, 3,3-dimethyl, 4-oxy) against four through a
    methyl; (:22740) 'greatest number of substituents of any kind' takes the first."""
    coa = ("CCCCCCCCCCCCCC[C@@H](O)C(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)"
           "OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O")
    res = name_best_effort(coa)
    name = res["name"]
    assert "(2R)-4-[" in name and "-2-hydroxy-3,3-dimethyl-1-oxobutyl" in name, name
    assert "3-methyl-1-oxobutyl" not in name, name
    assert name_is_rt_exact(name, coa), name


def test_a_tie_that_citation_order_does_not_separate_declines():
    """Three equal arms with different halogens: the chains through (fluoro, chloro) and
    (fluoro, bromo) tie on (k) and (l) with different substituents; (m) compares the same
    substituents only, so the producer is not built rather than choosing by atom order."""
    from orthonym.assembly.substituent_naming import _located_acyclic_alkyl_name
    mol = Chem.MolFromSmiles("FCCC(CCCl)(CCBr)C")           # the last carbon is the parent atom
    sub = [i for i in range(mol.GetNumAtoms()) if i != 10]
    assert _located_acyclic_alkyl_name(mol, sub, 3) is None


def test_the_selector_declines_beyond_its_enumeration_cap(monkeypatch):
    import orthonym.assembly.substituent_naming as sn
    mol = Chem.MolFromSmiles("FCCC(CCCl)(CCBr)C")
    sub = [i for i in range(mol.GetNumAtoms()) if i != 10]
    carbon = {i for i in sub if mol.GetAtomWithIdx(i).GetSymbol() == "C"}
    assert sn._principal_chain_candidates(mol, set(sub), carbon, 3) is not None
    monkeypatch.setattr(sn, "_CHAIN_ENUM_CAP", 2)
    assert sn._principal_chain_candidates(mol, set(sub), carbon, 3) is None
    assert sn._located_acyclic_alkyl_name(mol, sub, 3) is None


def test_candidates_do_not_depend_on_the_order_of_the_atoms():
    """The chains that tie, as the atoms the chain runs through (in the numbering of the
    molecule's canonical ranks), are the same set for every atom order."""
    import orthonym.assembly.substituent_naming as sn
    seen = set()
    for order in _orders("C[CH]C(CCO)C(C)O", n=10):
        mol = Chem.MolFromSmiles(order)
        attach = next(a.GetIdx() for a in mol.GetAtoms() if a.GetNumRadicalElectrons())
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        sub = set(range(mol.GetNumAtoms()))
        carbon = {i for i in sub if mol.GetAtomWithIdx(i).GetSymbol() == "C"}
        cands = sn._principal_chain_candidates(mol, sub, carbon, attach)
        seen.add(tuple(sorted(tuple(ranks[a] for a in c) for c in cands)))
    assert len(seen) == 1, seen
