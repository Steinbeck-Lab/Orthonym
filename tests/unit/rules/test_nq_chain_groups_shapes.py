"""Roadmap item 12a: the N/O group prefixes (``assembly.hetero_group_prefixes``).

The writers of the best-effort tier spelled a branch that no leaf names as a skeletal
replacement ('a') chain ('2-oxa-1-azaethan-1-ylidene' for =N-OH). An 'a' chain "must be
terminated by a C atom or one of the following heteroatoms: P, As, Sb, Bi, Si, Ge, Sn, Pb,
B, Al, Ga, In, or Tl", the Blue Book, section General rules), so
none of these is a book name. The module reads the structure of one group and composes the
prefix the book names it by:

* =N-OH, =N-OR, =N-R 'hydroxyimino', '(methoxyimino)',:38485
  "4-[(ethoxyimino)methyl]benzene-1-sulfonic acid (PIN)";:1695 'imino' only for =NH);
* =N-NR2 'dimethylhydrazinylidene' (Changes from the 1979 edition 7(j),:1703;
  ,:24611 "(dimethylcarbamoyl)hydrazinylidene (preferred prefix)");
* -N=N-R 'phenyldiazenyl',:38732 "The prefix 'diazenyl' is a
  preselected prefix",:38740 "(methyldiazenyl)acetic acid (PIN)");
* -NR-NR2 '2-phenylhydrazinyl',:16037; locants,:2869);
* -OO-R, -OOH '(methylperoxy)', 'hydroperoxy',:27858,:27870;,
  :27944);
* =[NR2]+, =[OR]+ 'dimethylazaniumylidene', 'oxidaniumylidene',:42298,:42350);
* -NR2(+)-O(-) 'dimethyl(oxido)azaniumyl',:26648);
* -N=O 'nitroso',:25935).

``group_shape`` is a pure reading of one group and ``compose_group`` a pure join of the names
the host gives its parts, so these tests need no OPSIN and no engine."""
import pytest
from rdkit import Chem

from orthonym.assembly.hetero_group_prefixes import (
    LATE_KINDS, compose_group, group_shape)


def _shape(smiles, root, parent, **kw):
    mol = Chem.MolFromSmiles(smiles)
    seen, stack = set(), [root]
    while stack:
        cur = stack.pop()
        if cur in seen or cur == parent:
            continue
        seen.add(cur)
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            stack.append(nb.GetIdx())
    return mol, group_shape(mol, root, frozenset(seen), parent, **kw)


#: (SMILES, root atom, atom it hangs on, kind, names of the parts in order, the prefix)
CASES = [
    ("CC=NO", 2, 1, "imino", ["hydroxy"], "hydroxyimino"),
    ("CC=NOC", 2, 1, "imino", ["methoxy"], "methoxyimino"),
    ("CC=NCC", 2, 1, "imino", ["ethyl"], "ethylimino"),
    ("CC=NN", 2, 1, "hydrazinylidene", [], "hydrazinylidene"),
    ("CC=NN(C)C", 2, 1, "hydrazinylidene", ["methyl", "methyl"], "dimethylhydrazinylidene"),
    ("CC=NNC(=O)N(C)C", 2, 1, "hydrazinylidene", ["dimethylcarbamoyl"],
     "(dimethylcarbamoyl)hydrazinylidene"),
    ("CN=Nc1ccccc1", 1, 0, "diazenyl", ["phenyl"], "phenyldiazenyl"),
    ("CN=N", 1, 0, "diazenyl", [], "diazenyl"),
    ("CNN", 1, 0, "hydrazinyl", [], "hydrazinyl"),
    ("CN(C)NC", 1, 0, "hydrazinyl", ["methyl", "methyl"], "1,2-dimethylhydrazinyl"),
    ("CNNc1ccccc1", 1, 0, "hydrazinyl", ["phenyl"], "2-phenylhydrazinyl"),
    ("COOC", 1, 0, "peroxy", ["methyl"], "methylperoxy"),
    ("COO", 1, 0, "hydroperoxy", [], "hydroperoxy"),
    ("CC=[NH2+]", 2, 1, "azaniumylidene", [], "azaniumylidene"),
    ("CC=[N+](C)C", 2, 1, "azaniumylidene", ["methyl", "methyl"], "dimethylazaniumylidene"),
    ("CC[N+](C)(C)[O-]", 2, 1, "oxidoazaniumyl", ["methyl", "methyl"],
     "dimethyl(oxido)azaniumyl"),
    ("CC=[N+](C)[O-]", 2, 1, "oxidoazaniumylidene", ["methyl"], "methyl(oxido)azaniumylidene"),
    ("CC=[OH+]", 2, 1, "oxidaniumylidene", [], "oxidaniumylidene"),
    ("CCN=O", 2, 1, "nitroso", [], "nitroso"),
    ("c1ccccc1ON=CC", 6, 5, "aminooxy", ["(ethylidene)amino"], "[(ethylidene)amino]oxy"),
]


@pytest.mark.parametrize("smiles,root,parent,kind,names,prefix", CASES)
def test_group_prefix(smiles, root, parent, kind, names, prefix):
    mol, shape = _shape(smiles, root, parent)
    assert shape is not None and shape.kind == kind
    assert len(shape.parts) == len(names)
    assert compose_group(shape, names) == prefix


def test_parts_are_the_atoms_not_in_the_unit():
    """Every atom of the group is in the unit or in exactly one part."""
    for smiles, root, parent, kind, names, _ in CASES:
        mol, shape = _shape(smiles, root, parent)
        covered = set(shape.unit)
        for part in shape.parts:
            assert not (covered & set(part.atoms))
            covered |= set(part.atoms)
        group = set()
        stack = [root]
        while stack:
            cur = stack.pop()
            if cur in group or cur == parent:
                continue
            group.add(cur)
            stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors())
        assert covered == group, (smiles, kind)


def test_iminyl_needs_a_defined_geometry_unless_asked():
    """-N=CR2 without a defined geometry is the amino prefix the floor already builds
    (``universal_substituent._hetero_root_leaf``); with a defined geometry, or on request
    (the terminal-fragment writer), it is '[(E)-R-ylidene]amino'."""
    mol, plain = _shape("CC=Nc1ccccc1", 2, 3)
    assert plain is None
    _, asked = _shape("CC=Nc1ccccc1", 2, 3, unstereo_iminyl=True)
    assert asked is not None and asked.kind == "iminyl"
    assert asked.parts[0].order == 2
    _, defined = _shape("C/C=N/c1ccccc1", 2, 3)
    assert defined is not None and defined.kind == "iminyl"
    assert compose_group(defined, ["(E)-ethylidene"]) == "[(E)-ethylidene]amino"


def test_hydrazone_far_nitrogen_takes_one_ylidene_part():
    """-NH-N=CR2 is '2-[(E)-R-ylidene]hydrazinyl': the far nitrogen's one double bond is a part
    of order 2 with the hydrazine locant 2."""
    mol, shape = _shape("CNN=CC", 1, 0)
    assert shape.kind == "hydrazinyl"
    assert [(p.locant, p.order) for p in shape.parts] == [(2, 2)]
    assert compose_group(shape, ["ethylidene"]) == "2-ethylidenehydrazinyl"


def test_azine_is_a_hydrazinylidene_with_one_ylidene_part():
    mol, shape = _shape("CC=NN=CC", 2, 1)
    assert shape.kind == "hydrazinylidene"
    assert [p.order for p in shape.parts] == [2]


def test_charges_the_prefix_spells():
    _, onium = _shape("CC=[N+](C)C", 2, 1)
    assert onium.charged == frozenset({2})
    _, oxide = _shape("CC[N+](C)(C)[O-]", 2, 1)
    assert oxide.internal == frozenset({2, 3}) or len(oxide.internal) == 2
    assert not oxide.charged


@pytest.mark.parametrize("smiles,root,parent", [
    ("CN1CC1", 1, 0),                # a nitrogen in a ring is no group root
    ("C[N-]N", 1, 0),                # a charged hydrazine nitrogen
    ("C[N+](C)(C)C", 1, 0),          # an ammonium: 'azaniumyl' is the host's
    ("CN=C=O", 1, 0),                # an isocyanate: its own prefix
    ("CN(C)C", 1, 0),                # an amine: 'amino' is the host's
    ("COC", 1, 0),                   # an ether: 'oxy' is the host's
])
def test_not_a_group_here(smiles, root, parent):
    mol, shape = _shape(smiles, root, parent)
    assert shape is None


def test_an_isotope_on_the_root_blocks_the_shape():
    m = Chem.MolFromSmiles("CC=[15N]O")
    assert group_shape(m, 2, frozenset({2, 3}), 1) is None
    assert group_shape(Chem.MolFromSmiles("CC=NO"), 2, frozenset({2, 3}), 1) is not None


def test_late_kinds_are_only_the_ones_a_writer_spells_itself():
    assert LATE_KINDS == frozenset({"aminooxy"})


def test_compose_group_refuses_a_missing_part_name():
    _, shape = _shape("CN=Nc1ccccc1", 1, 0)
    assert compose_group(shape, []) is None
    assert compose_group(shape, [""]) is None


def test_a_prefix_with_parts_is_recorded_as_enclosed():
    """'tert-butyl(oxido)azaniumylidene' reads as a simple prefix by its first letters; the
    record says it is compound (``prefix_derivation``)."""
    from orthonym.assembly.prefix_derivation import derivation_of
    _, shape = _shape("CC=[N+](C(C)(C)C)[O-]", 2, 1)
    token = compose_group(shape, ["tert-butyl"])
    assert token == "tert-butyl(oxido)azaniumylidene"
    rec = derivation_of(token)
    assert rec is not None and rec.substituted and rec.enclosed
    assert derivation_of(compose_group(_shape("CC=[NH2+]", 2, 1)[1], [])) is None
