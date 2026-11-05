"""PB lane S, task S4: the count-named alkyl of ``rules/esters.get_alkyl_fragment_name``.

The tail of the function named an ester's alcohol component from its carbon COUNT when the
substituent primitive and the unsaturated namer had declined it, so ``C=C(C)C(C#N)-O-`` came
out ``pentyl`` (the alkene, the methyl branch and the nitrile all dropped; d519 on dev2000)
and the name described a different molecule that then withheld.

Rule: "GENERAL METHODOLOGY FOR NAMING SUBSTITUENT GROUPS" (the Blue Book), method (1)
"is recommended primarily for saturated acyclic and monocyclic hydrocarbon substituent groups":
a count word spells a saturated unbranched carbon chain. "Substituents derived from
unsaturated acyclic compounds" (:17205) gives the form for the rest ('prop-2-en-1-yl').
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.rules.esters import _skeleton_is_saturated, get_alkyl_fragment_name

D519 = "C=C(C)C(C#N)OC(=O)CCCCCCC/C=C\\C/C=C\\C/C=C\\CC"


def _alkyl_atoms(smiles, first):
    """Atoms of the alcohol component that starts at atom ``first`` (the O is excluded)."""
    mol = Chem.MolFromSmiles(smiles)
    o = next(n for n in mol.GetAtomWithIdx(first).GetNeighbors() if n.GetSymbol() == "O")
    seen, stack = {first}, [first]
    while stack:
        for n in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            if n.GetIdx() not in seen and n.GetIdx() != o.GetIdx():
                seen.add(n.GetIdx())
                stack.append(n.GetIdx())
    return mol, sorted(seen)


@pytest.mark.unit
def test_saturated_skeleton_predicate():
    mol = Chem.MolFromSmiles("CCC(C)CC=CC#N")
    assert _skeleton_is_saturated(mol, [0, 1, 2, 3])                  # CCC(C)
    assert not _skeleton_is_saturated(mol, [4, 5, 6])                 # the alkene
    assert not _skeleton_is_saturated(mol, [7])                       # the nitrile carbon
    assert not _skeleton_is_saturated(Chem.MolFromSmiles("CC(=O)O"), [0, 1])  # a carbonyl carbon


@pytest.mark.unit
def test_unsaturated_cyano_fragment_is_not_counted():
    """The carbon COUNT never names the fragment ('pentyl' dropped the alkene, the methyl and the nitrile).

    change-asserted-value (lane/l17 fix a performance pass): the expectation was '' (the substituent primitive declined a nitrile
    on a chain, and the count tail refused). The primitive now spells the cyano group as a prefix:
    (the Blue Book) "the -CN group is designated by the preferred prefix 'cyano'", on the unsaturated group of
     (:17205) 'prop-2-en-1-yl', numbered from the free valence: '1-cyano-2-methylprop-2-en-1-yl'. It is
    never the count word. Independent check: a fresh OPSIN call reads '1-cyano-2-methylprop-2-en-1-yl acetate' to the
    input's full InChIKey. Mutation: without Pass 1a of
    ``substituent_naming._name_polyfunctional_acyclic_substituent_impl`` the function returns '' again."""
    mol, atoms = _alkyl_atoms("C=C(C)C(C#N)OC(C)=O", 3)  # C=C(C)C(C#N)-O...
    name = get_alkyl_fragment_name(mol, atoms)
    assert name == "1-cyano-2-methylprop-2-en-1-yl" and name != "pentyl"
    from tests.support.rt_assert import assert_full_rt
    assert_full_rt(name + " acetate", "C=C(C)C(C#N)OC(C)=O")


@pytest.mark.unit
@pytest.mark.parametrize("smiles,first,word", [
    ("CCCCCOC(C)=O", 4, "pentyl"),
    ("CC(C)OC(C)=O", 1, "propan-2-yl"),
])
def test_saturated_chains_keep_their_name(smiles, first, word):
    mol, atoms = _alkyl_atoms(smiles, first)
    assert get_alkyl_fragment_name(mol, atoms) == word


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
def test_d519_is_named_with_its_cyano_alkenyl_alcohol():
    row = Orthonym().name_tiered(D519)
    assert row["name"] == ("1-cyano-2-methylprop-2-en-1-yl "
                           "(9Z,12Z,15Z)-octadeca-9,12,15-trienoate")
    assert row["tier"] == "pin_verified"
    assert not row["name"].startswith("pentyl ")
