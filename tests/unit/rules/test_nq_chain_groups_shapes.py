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
* -NR-NR2 '2-phenylhydrazin-1-yl',:16037; locants,:2869;
                        '4-(1-acetyl-2-ethylhydrazin-1-yl)benzoic acid (PIN)',:34042);
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
    ("CN(C)NC", 1, 0, "hydrazinyl", ["methyl", "methyl"], "1,2-dimethylhydrazin-1-yl"),
    ("CNNc1ccccc1", 1, 0, "hydrazinyl", ["phenyl"], "2-phenylhydrazin-1-yl"),
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
    """-NH-N=CR2: the far nitrogen's one double bond is a part of order 2 with the hydrazine
    locant 2, and the prefix cites no locant at all: "Omission of locants"
    (the Blue Book) 'ethylidenehydrazinyl (preferred prefix) (not 2-ethylidenehydrazin-1-yl)'
    (:2959), '2-(benzylidenehydrazinyl)-2-oxoacetamide (PIN)' (:34143)."""
    mol, shape = _shape("CNN=CC", 1, 0)
    assert shape.kind == "hydrazinyl"
    assert [(p.locant, p.order) for p in shape.parts] == [(2, 2)]
    assert compose_group(shape, ["ethylidene"]) == "ethylidenehydrazinyl"


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


#: (SMILES, root atom, atom it hangs on, names of the parts, the prefix). A unit that cites one
#: locant cites all of them, the free valence's too "Citation of locants",
#: the Blue Book); the locants are left out only where no isomer can be generated by moving a
#: substituent or interchanging two,:2953) and where all three positions hold the same
#: substituent,:3007, 'tetrafluorourea (PIN)')
HYDRAZINYL_LOCANTS = [
    ("CNN", 1, 0, [], "hydrazinyl"),                                       #:2911 (not hydrazin-1-yl)
    ("CNNC", 1, 0, ["methyl"], "2-methylhydrazin-1-yl"),                   #:34042,:36437
    ("CN(C)N", 1, 0, ["methyl"], "1-methylhydrazin-1-yl"),
    ("CNN(C)C", 1, 0, ["methyl", "methyl"], "2,2-dimethylhydrazin-1-yl"),
    ("CN(CC)N(C)C", 1, 0, ["ethyl", "methyl", "methyl"], "1-ethyl-2,2-dimethylhydrazin-1-yl"),
    ("CN(C)N(C)C", 1, 0, ["methyl", "methyl", "methyl"], "trimethylhydrazinyl"),
    ("CN(C(C)=O)NCC", 1, 0, ["acetyl", "ethyl"], "1-acetyl-2-ethylhydrazin-1-yl"),  #:34042 (PIN)
    ("CNNC(N)=O", 1, 0, ["carbamoyl"], "2-carbamoylhydrazin-1-yl"),         #:38644 (PIN)
    ("CNN=CC", 1, 0, ["ethylidene"], "ethylidenehydrazinyl"),              #:2959
]


@pytest.mark.parametrize("smiles,root,parent,names,prefix", HYDRAZINYL_LOCANTS)
def test_hydrazinyl_locants(smiles, root, parent, names, prefix):
    mol, shape = _shape(smiles, root, parent)
    assert shape is not None and shape.kind == "hydrazinyl"
    assert compose_group(shape, names) == prefix


def test_an_ylidene_with_a_substituent_on_the_first_nitrogen_is_not_built():
    """'ethylidene(methyl)hydrazinyl' is the form gives (no isomer), and OPSIN 2.9.0
    reads it as ethylidene on the methyl carbon (it parses to C(C)=CNN-), so no prefix is built
    and the host keeps its other spelling."""
    _, shape = _shape("CN(C)N=CC", 1, 0)
    assert shape.kind == "hydrazinyl"
    assert compose_group(shape, ["methyl", "ethylidene"]) is None


#: (SMILES, root atom, atom it hangs on, is the group R-CO-NH-NH- or R-SO2-NH-NH-?)
ACYLHYDRAZIDE = [
    ("CNNC(C)=O", 1, 0, True),        # 'acetohydrazido' is the PIN prefix,:34020)
    ("CNNC(=O)c1ccccc1", 1, 0, True),
    ("CNNS(=O)(=O)C", 1, 0, True),
    ("CNNC(N)=O", 1, 0, False),       # carbamoyl: '2-carbamoylhydrazin-1-yl' is the PIN (:38644)
    ("CNNC(=O)N(C)C", 1, 0, False),   # a substituted carbamoyl, an amino nitrogen without N-N
    ("CN(C(C)=O)N", 1, 0, False),     # on the first nitrogen: '1-acetylhydrazin-1-yl' (:34042)
    ("CNNC", 1, 0, False),
    # a double-bonded nitrogen on the carbon is an acyl marker as the chalcogens are:
    # (:34548) method (1) generates preferred prefixes, 'methanimidohydrazido' (:34559)
    ("CNNC=N", 1, 0, True),
    ("CNNC(C)=N", 1, 0, True),
    ("CNNC(C)=NC", 1, 0, True),
    # the second atom on the carbonyl a hetero atom other than an amino nitrogen without N-N
    ("CNNC(=O)NN", 1, 0, True),       # 'hydrazinecarbohydrazido' (PIN prefix,:38712)
    ("CNNC(=O)N=N", 1, 0, True),      # 'diazenecarbohydrazido (preferred prefix)' (:38912)
    ("CNNC(=O)OC", 1, 0, True),       # an ester oxygen::34129 'formohydrazido' (ASSUMED analogy)
    ("CNNC(=O)O", 1, 0, True),
    ("CNNC(=S)SC", 1, 0, True),
]


@pytest.mark.parametrize("smiles,root,parent,expected", ACYLHYDRAZIDE)
def test_acylhydrazide_flag(smiles, root, parent, expected):
    """ "Substituent groups derived from hydrazides" (the Blue Book): a hydrazide
    group R-CO-NH-NH- or R-SO2-NH-NH- is named (1) 'acetohydrazido', (2) as an acylhydrazinyl
    group; "Method (1) generates preferred IUPAC names", so the acylhydrazinyl form is valid and
    never the PIN. The groups R-CO-N(NH2)- are hydrazinyl prefixes, "This method leads to
    preferred prefixes". Also the hydrazides of imidic acids,:34548) and the
    carbonyl groups with a hetero atom other than an amino nitrogen without N-N (:38712,:38912,
    :34129, ASSUMED for the ester oxygen); carbamoyl alone stays the hydrazinyl prefix (:38640)."""
    _, shape = _shape(smiles, root, parent)
    assert shape.kind == "hydrazinyl" and shape.acylhydrazide is expected
