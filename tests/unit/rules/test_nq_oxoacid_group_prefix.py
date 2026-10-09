"""P and S oxoacid groups are substituent prefixes at the wider tiers (lane nq-forms-ps).

 "Compound and complex substituent groups" (the Blue Book): a group built on
P, attached through O, S or N to a compound with a senior group, "is named by a compound or
complex prefix built from prefixes described above": '(phosphonooxy)acetic acid (PIN)' (:36333),
'3-[(dimethoxyphosphoryl)sulfanyl]propanoic acid (PIN)' (:36335),
'3-{[hydroxy(sulfanyl)phosphorothioyl]amino}propanoic acid (PIN)' (:36337).
(:36484): '3-(sulfooxy)propanoic acid (PIN)' (:36488), '3-(sulfamoyloxy)propanoic acid (PIN)'
(:36494). (:36937) method (1): '3-{[hydroxy(phosphonooxy)phosphoryl]oxy}propanoic
acid'. (:41211,:41213): 'sulfonato', 'phosphonato' are the preselected prefixes of
the anions. A skeletal 'a' chain ending on O ('1,1-dioxo-2-oxa-1lambda6-thiaethyl') is not
one of these: (:6465) "The chain must be terminated by a C atom or one of the
following heteroatoms: P, As, Sb,...".
"""
import pytest
from rdkit import Chem

from orthonym.rules.oxoacid_group_prefix import (
    has_replacement_chain, is_oxoacid_centre, is_oxoacid_linker, oxoacid_group_prefix)


def _group(smiles, attach_map=1, parent_map=2):
    """(mol, group atoms, attach atom): the group is what lies beyond the bond between the
    atom mapped ``attach_map`` and the atom mapped ``parent_map``."""
    mol = Chem.MolFromSmiles(smiles)
    a = next(x.GetIdx() for x in mol.GetAtoms() if x.GetAtomMapNum() == attach_map)
    p = next(x.GetIdx() for x in mol.GetAtoms() if x.GetAtomMapNum() == parent_map)
    seen, stack = {a}, [a]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            k = nb.GetIdx()
            if k != p and k not in seen:
                seen.add(k)
                stack.append(k)
    for x in mol.GetAtoms():
        x.SetAtomMapNum(0)
    return mol, seen, a


# (group SMILES with the attachment atom mapped 1 and its parent-side neighbour mapped 2, prefix)
# A compound prefix that carries no enclosing mark of its own is returned enclosed;
# one with an internal mark ('[ethoxy(hydroxy)phosphoryl]oxy') is enclosed by the caller.
GROUPS = [
    ("Nc1cccc[c:2]1[S:1](=O)(=O)O", "sulfo"),
    ("c1cccc[c:2]1[S:1](=O)(=O)[O-]", "sulfonato"),
    ("C[C:2][S:1](=O)O", "sulfino"),
    ("C[C:2][S:1](=O)(=O)N", "sulfamoyl"),
    ("C[C:2][S:1](=O)(=O)Cl", "(chlorosulfonyl)"),
    ("C[C:2][S:1](=O)(=O)OC", "(methoxysulfonyl)"),
    ("C[C:2][O:1]S(=O)(=O)O", "(sulfooxy)"),
    ("C[C:2][O:1]S(=O)(=O)[O-]", "(sulfonatooxy)"),
    ("c1cccc[c:2]1[O:1]P(=O)([O-])[O-]", "(phosphonatooxy)"),
    ("C[C:2][NH:1]S(=O)(=O)O", "(sulfoamino)"),
    ("C[C:2][O:1]P(=O)(O)OCC", "[ethoxy(hydroxy)phosphoryl]oxy"),
    ("C[C:2][O:1]P(=S)(OC)OC", "(dimethoxyphosphorothioyl)oxy"),
    ("C[C:2][O:1]P(=O)(O)OP(=O)(O)O", "(1,3,3-trihydroxy-1,3-dioxo-1λ5,3λ5-diphosphoxan-1-yl)oxy"),
    ("C[C:2][O:1]P(=O)([O-])OCC(O)CO", "[(2,3-dihydroxypropoxy)(oxido)phosphoryl]oxy"),
    ("C[C:2][S:1]P(=S)(OC)OC", "(dimethoxyphosphorothioyl)sulfanyl"),
    ("C[C:2][P:1](=O)(O)OC", "hydroxy(methoxy)phosphoryl"),
    ("C[C:2][S:1](=O)(=O)N(C)CC", "ethyl(methyl)sulfamoyl"),
    # compound ligands: enclosed, the first included, and multiplied with bis,
    # the Blue Book 'bis(dimethylamino) (preferred prefix)';:26318;:18116)
    ("C[C:2][O:1]P(=O)(NC)NC", "[bis(methylamino)phosphoryl]oxy"),
    ("C[C:2][O:1]P(=O)(N(C)C)N(C)C", "[bis(dimethylamino)phosphoryl]oxy"),
    ("C[C:2][O:1]P(=O)(SCC)SCC", "[bis(ethylsulfanyl)phosphoryl]oxy"),
    ("C[C:2][O:1]P(=S)(SC)SC", "[bis(methylsulfanyl)phosphorothioyl]oxy"),
    ("C[C:2][O:1]P(=O)(Nc1ccccc1)Nc1ccccc1", "[bis(phenylamino)phosphoryl]oxy"),
    ("C[C:2][O:1]P(=O)(OCc1ccccc1)OCc1ccccc1", "[bis(benzyloxy)phosphoryl]oxy"),
    ("C[C:2][O:1]P(=O)(OCc1ccccc1)OC", "[(benzyloxy)(methoxy)phosphoryl]oxy"),
    ("C[C:2][S:1](=O)(=O)OCc1ccccc1", "(benzyloxy)sulfonyl"),
    ("C[C:2][S:1](=O)(=O)NC(=O)NC1CCCCC1", "(cyclohexylcarbamoyl)sulfamoyl"),
    ("C[C:2][O:1]S(=O)OS(=O)OC", "{[(methoxysulfinyl)oxy]sulfinyl}oxy"),
]


@pytest.mark.parametrize("smiles,expected", GROUPS)
def test_group_prefix_from_structure(smiles, expected):
    mol, grp, att = _group(smiles)
    assert oxoacid_group_prefix(mol, grp, att) == expected


@pytest.mark.parametrize("smiles", [
    "C[C:2][S:1](=O)(=O)C",            # a sulfone is not an acid group
    "C[C:2][S:1](=O)C",                # a sulfoxide
    "C[C:2][P:1](=O)(C)C",             # a phosphine oxide
    "C[C:2][O:1]C",                    # no P or S
    "C[C:2][S:1]C",                    # a thioether
])
def test_not_an_oxoacid_group_declines(smiles):
    mol, grp, att = _group(smiles)
    assert oxoacid_group_prefix(mol, grp, att) is None


def test_a_group_with_an_unaccounted_atom_declines():
    mol, grp, att = _group("C[C:2][O:1]P(=O)(O)OC1CCCC1")
    assert oxoacid_group_prefix(mol, grp, att) is not None
    grp.discard(max(grp))                                          # one ring atom missing
    assert oxoacid_group_prefix(mol, grp, att) is None


def test_centre_and_linker_perception():
    mol = Chem.MolFromSmiles("CCOS(=O)(=O)O")
    assert [is_oxoacid_centre(mol, i) for i in range(mol.GetNumAtoms())][3] is True
    assert is_oxoacid_linker(mol, 2) is True and is_oxoacid_linker(mol, 1) is False
    sulfone = Chem.MolFromSmiles("CS(=O)(=O)C")
    assert not any(is_oxoacid_centre(sulfone, i) for i in range(sulfone.GetNumAtoms()))


def test_replacement_chain_detector():
    assert has_replacement_chain("2-oxa-1-phosphaethyl")
    assert has_replacement_chain("1,1-dioxo-2-oxa-1λ6-thiaethyl")
    assert has_replacement_chain("3,5-dioxa-4-phospha")
    assert not has_replacement_chain("phosphonooxy")
    assert not has_replacement_chain("2-(sulfooxy)ethyl")


@pytest.mark.parametrize("smiles,atom,ok", [
    ("OC(=O)CCOP(=O)(OC)OC", 6, True),
    ("OC(=O)CCOS(=O)(=O)O", 6, True),
    ("P(OC)(OC)=O", 0, False),                         # P-H
    ("C[P@@](=O)(OC)OCC(=O)O", 1, False),              # a stereo tag on the centre
    ("CN(N)P(=O)(N(N)C)N(N)C", 4, False),              # an N-N ligand
    ("S(=O)(=O)(N=C=O)N=C=O", 0, False),               # isocyanato ligands
    ("OC(=O)CCOP(=O)(OCCl)OC", 6, True),
])
def test_one_shape_predicate_for_the_spine_exclusion_and_the_namer(smiles, atom, ok):  # noqa: ARG001
    """``is_oxoacid_centre`` (the chain-spine exclusion) is True only where the namer builds
    a prefix, so an atom is never kept off the spine without a leaf to name it."""
    mol = Chem.MolFromSmiles(smiles)
    centres = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in ("P", "S")]
    assert is_oxoacid_centre(mol, centres[0]) is ok, (smiles, centres)


@pytest.mark.parametrize("smiles,preferred", [
    ("C[C:2][O:1]S(=O)(=O)O", True),                               # sulfooxy,:31342
    ("C[C:2][O:1]P(=O)(O)O", True),                                # phosphonooxy,:36333
    ("c1cccc[c:2]1[S:1](=O)(=O)[O-]", True),                       # sulfonato,:41211
    ("C[C:2][S:1](=O)(=O)Cl", True),                               # (chlorosulfonyl),:36492
    ("C[C:2][S:1](=O)(=O)OC", True),                               # (methoxysulfonyl),:36540
    ("C[C:2][S:1]P(=S)(OC)OC", True),                              # (dimethoxyphosphoryl)sulfanyl,:36335
    ("C[C:2][O:1]P(=O)(OC)OC", True),
    ("C[C:2][O:1]P(=O)(O)OP(=O)(O)O", True),                       # method (2),:36949
    ("C[C:2][P:1](=O)(O)c1ccccc1", False),                         # hydroxy(phenylphosphonoyl),:36244
    ("C[C:2][P:1](=O)(Cl)Cl", False),                              # phosphorodichloridoyl,:36166
    ("C[C:2][P:1](=O)(N(C)C)N(C)C", False),                        # phosphoramidoyl,:36178
    ("C[C:2][P:1](=S)(O)O", False),                                # thiophosphono
    ("C[C:2][O:1]P(=O)(O)OCC", False),                             # acid function left on the centre
    ("C[C:2][O:1]P(=O)([O-])O", False),                            # monoanion
])
def test_preferred_flag_of_the_spelling(smiles, preferred):
    from orthonym.rules.oxoacid_group_prefix import oxoacid_group_prefix_ex
    mol, grp, att = _group(smiles)
    tok, pref = oxoacid_group_prefix_ex(mol, grp, att)
    assert tok is not None and pref is preferred, (tok, pref)
