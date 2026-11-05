"""A sulfonyl or sulfinyl group is one acyl prefix, never an 'a' chain through its sulfur.

 "Substituent groups formed by concatenation" (the Blue Book): "The name
sulfamoyl for H2N-SO2- is a retained name used as a preferred IUPAC name" (:31316); the acyl
groups -SO2- 'sulfonyl' and -SO- 'sulfinyl' take the prefixes of their ligands ('methoxysulfonyl',
:31322). "Substituents of the types -SO2-NH2, -SO-NH2" (:32971): "For -SO2-NH2,
method (2) generates preferred IUPAC names" (:32978, '2-(dimethylsulfamoyl)benzene-1-sulfonic acid
(PIN)',:32982); the carbamoyl group of "may be substituted in the normal way"
(:32928, '5-methyl-2-[methyl(phenyl)carbamoyl]benzoic acid (PIN)',:32957).
"Substituent groups derived from hydrazides" (:33980): 'hydrazinesulfonyl' is the preferred prefix
of -SO2-NH-NH2 (:33996). "Names for acyl groups derived from sulfonic, sulfinic, etc.
acids" (:31260): the preferred acyl prefix is formed "directly from the name of the sulfonic acid,
sulfinic acid, etc., as in 'benzenesulfonyl', instead of the traditional method of concatenation
as in 'phenylsulfonyl'" (:31262). "Naming of stereoisomers" (:44639): "When they relate to
substituent groups, they are cited at the front of the corresponding prefix" (:44643).
(:6465): the chain of a skeletal replacement name "must be terminated by a C atom or one of the
following heteroatoms", and (:23348) allows the 'a' chain "when none of the heteroatoms
constitute all or part of the principal characteristic group", so a chain that runs through the S of
a sulfonamide ('2-methyl-1,1-dioxo-3-oxa-1λ6-thia-2-azabutyl') is not the name of the group.
"""
import pytest
from rdkit import Chem

from orthonym.rules.oxoacid_group_prefix import has_replacement_chain, oxoacid_group_prefix

from ._helpers import assert_clean

pytestmark = pytest.mark.opsin_gate


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


# (group SMILES, attachment atom mapped 1 and its parent-side neighbour mapped 2; the prefix)
GROUPS = [
    #: the N-substituents of sulfamoyl in the normal way, cited alphanumerically
    # without N locants, as 'methyl(phenyl)carbamoyl' (:32957)
    ("C[C:2][S:1](=O)(=O)NOC", "(methoxysulfamoyl)"),
    ("C[C:2][S:1](=O)(=O)NO", "(hydroxysulfamoyl)"),
    ("C[C:2][S:1](=O)(=O)N(C)OC", "methoxy(methyl)sulfamoyl"),
    ("C[C:2][S:1](=O)(=O)N(C)O", "hydroxy(methyl)sulfamoyl"),
    ("C[C:2][S:1](=O)(=O)N(CC)OC", "ethyl(methoxy)sulfamoyl"),
    # an N-ylidene group replaces both hydrogens of the nitrogen; its C=N geometry is cited at
    # the front of the ylidene prefix,:44643), without a locant for a one-carbon group
    ("C[C:2][S:1](=O)(=O)N=CN(C)C", "[(dimethylamino)methylidene]sulfamoyl"),
    ("C[C:2][S:1](=O)(=O)/N=C/N(C)C", "[(E)-(dimethylamino)methylidene]sulfamoyl"),
    ("C[C:2][S:1](=O)(=O)/N=C\\N(C)C", "[(Z)-(dimethylamino)methylidene]sulfamoyl"),
    # method (1): 'hydrazinesulfonyl', 'hydrazinesulfinyl' (:33996); a substituent
    # on the hydrazine cites its locant,:2869) with the free valence on N-1
    ("C[C:2][S:1](=O)(=O)NN", "(hydrazinesulfonyl)"),
    ("C[C:2][S:1](=O)NN", "(hydrazinesulfinyl)"),
    ("C[C:2][S:1](=O)(=O)NN(C)C", "(2,2-dimethylhydrazine-1-sulfonyl)"),
    ("C[C:2][S:1](=O)(=O)NNC", "(2-methylhydrazine-1-sulfonyl)"),
    ("C[C:2][S:1](=O)(=O)N(C)NC", "(1,2-dimethylhydrazine-1-sulfonyl)"),
    ("C[C:2][S:1](=O)(=O)NNC(N)=S", "(2-carbamothioylhydrazine-1-sulfonyl)"),
    # the substituent of a nitrogen may be the acyl group of a second S centre
    ("C[C:2][S:1](=O)(=O)N(C)S(=O)(=O)C", "(methanesulfonyl)(methyl)sulfamoyl"),
    ("C[C:2][S:1](=O)(=O)NNS(=O)(=O)C", "2-(methanesulfonyl)hydrazine-1-sulfonyl"),
    # 'aminooxy' is the prefix of H2N-O-; the sulfamate N-O ester is '(aminooxy)sulfonyl'
    ("C[C:2][S:1](=O)(=O)ON", "(aminooxy)sulfonyl"),
    # (the Blue Book): the preferred prefix of the acyl group of a sulfonic or
    # sulfinic acid is formed from the name of the acid, 'benzenesulfonyl', 'methanesulfonyl';
    # the concatenation 'phenylsulfonyl' "may be used in general nomenclature"
    ("C[N:2][S:1](=O)(=O)c1ccccc1", "(benzenesulfonyl)"),
    ("C[N:2][S:1](=O)(=O)C", "(methanesulfonyl)"),
    ("C[N:2][S:1](=O)c1cc(C)ccc1Cl", "(2-chloro-5-methylbenzene-1-sulfinyl)"),
]


@pytest.mark.parametrize("smiles,expected", GROUPS)
def test_sulfonyl_group_prefix_from_structure(smiles, expected):
    mol, grp, att = _group(smiles)
    assert oxoacid_group_prefix(mol, grp, att) == expected


def test_enclosed_amino_branch_is_not_enclosed_again():
    # (the Blue Book): a compound prefix takes one pair of enclosing marks
    from orthonym.assembly.composer import _assemble_decorated_amino_prefix
    assert _assemble_decorated_amino_prefix(
        [("acetyl", False), ("(methylsulfamoyl)", False)], enclose=False
    ) == "acetyl(methylsulfamoyl)amino"
    assert _assemble_decorated_amino_prefix(
        [("(methylsulfamoyl)", False), ("(methylsulfamoyl)", False)], enclose=False
    ) == "bis(methylsulfamoyl)amino"


# (b) (the Blue Book): the basic numerical prefixes 'di', 'tri',... multiply simple
# substituent prefixes ('dimethyl'); (a) (:7104): 'bis' multiplies compound prefixes
# ('bis(dimethylamino)'). Two ligands that are the same group on one centre are one prefix with a
# multiplier, whatever atom each was read from.
@pytest.mark.parametrize("smiles,expected", [
    ("C[N:2][P:1](=O)(C)C", "(dimethylphosphoryl)"),
    ("C[N:2][P:1](=O)(CC)CC", "(diethylphosphoryl)"),
    ("C[N:2][P:1](=O)(c1ccccc1)c1ccccc1", "(diphenylphosphoryl)"),
    ("C[N:2][P:1](=O)(OC)OC", "(dimethoxyphosphoryl)"),
    ("C[N:2][P:1](=O)(N(C)C)N(C)C", "bis(dimethylamino)phosphoryl"),
    ("C[N:2][P:1](=O)(C)CC", "ethyl(methyl)phosphoryl"),
])
def test_identical_ligands_of_one_centre_are_multiplied(smiles, expected):
    mol, grp, att = _group(smiles)
    assert oxoacid_group_prefix(mol, grp, att) == expected


@pytest.mark.parametrize("smiles,prefix", [
    ("CNP(=O)(C)C", "[(dimethylphosphoryl)amino]"),
    ("CCP(=O)(CC)Nc1ccccc1", "[(diethylphosphoryl)amino]"),
])
def test_phosphoryl_group_with_identical_carbon_ligands_is_one_prefix(smiles, prefix):
    assert prefix in assert_clean(smiles)["name"]


def test_group_with_a_nitrogen_nitrogen_double_bond_ligand_declines():
    # the diazenyl ligand has no prefix in this producer; it declines, never half-names
    mol, grp, att = _group("C[C:2][S:1](=O)(=O)N(C)N=CC")
    assert oxoacid_group_prefix(mol, grp, att) is None


# (SMILES of a whole molecule, a prefix that the name must contain)
ROWS = [
    # N-methoxy-N-methylsulfonamide
    ("CON(C)S(=O)(=O)c1cccc(C(=O)NCc2ccc(F)cc2)c1", "[methoxy(methyl)sulfamoyl]"),
    # N-methoxysulfonamide
    ("CONS(=O)(=O)c1ccc(C(=O)N(C)C(C)(C)C(C)(C)C)cc1", "(methoxysulfamoyl)"),
    # N-sulfonylformamidine, geometry undefined and defined
    ("COc1ccc(CC(C)=O)cc1S(=O)(=O)N=CN(C)C", "{[(dimethylamino)methylidene]sulfamoyl}"),
    ("CN(C)/C=N/S(=O)(=O)c1cc(F)c(CO)c(F)c1", "{[(E)-(dimethylamino)methylidene]sulfamoyl}"),
    # sulfonohydrazides
    ("CN(C)NS(=O)(=O)c1c(F)cc(N)cc1F", "(2,2-dimethylhydrazine-1-sulfonyl)"),
    ("NNS(=O)(=O)c1ccc(Oc2ccccc2S(=O)(=O)NN)cc1", "(hydrazinesulfonyl)"),
    ("Cc1cc(F)ccc1S(=O)(=O)NNS(C)(=O)=O", "2-(methanesulfonyl)hydrazine-1-sulfonyl"),
    # the sulfamoyl group on the nitrogen of an acetamide keeps its marks in second place
    ("CNS(=O)(=O)N(C(C)=O)C(c1ccccc1)c1ccccc1", "acetyl(methylsulfamoyl)amino"),
    # an acid-stem prefix with a substituent is a substituted prefix: 'bis' (c),
    # the Blue Book); the unsubstituted one takes 'di'
    ("Cc1ccc(S(=O)(=O)N2CCN(S(=O)(=O)c3ccc(C)cc3)CC2)cc1", "1,4-bis(4-methylbenzene-1-sulfonyl)"),
    # a sulfonate ester on a methyl group is a branch of that group. The row was 'COS(=O)(=O)CC1CCC1', whose
    # PIN is the ester itself, 'methyl cyclobutylmethanesulfonate' class 9, the ring-bearing CH2 is
    # the parent: rules.parent_selection.single_carbon_hosts_suffix); the branch needs a senior group beside it.
    ("COS(=O)(=O)CC1CCC1C(=O)O", "[(methoxysulfonyl)methyl]"),
    # the centre of a sulfamide is never a skeletal atom of a chain, however many heteroatoms
    # the chain holds, the Blue Book;,:23348). The row was the 3-cyanophenyl one, whose
    # PIN is the carbamate itself ('ethyl {[(3-cyanophenyl)methyl](methyl)sulfamoyl}carbamate': class 9
    # esters,:18182, rank above class 14 nitriles,:18187); the branch needs a senior group beside it, here the acid
    ("CCOC(=O)NS(=O)(=O)N(C)Cc1cccc(C(=O)O)c1", "{[(ethoxycarbonyl)sulfamoyl](methyl)amino}methyl"),
    # the descriptor of a stereogenic sulfinyl centre stands at the front of its prefix
    ("COc1ccc([S@@](N)=O)cc1C", "[(S)-aminosulfinyl]"),
    ("CCCCCCC#CCO[S@](=O)c1ccc(C)cc1", "{(S)-[(non-2-yn-1-yl)oxy]sulfinyl}"),
]


@pytest.mark.parametrize("smiles,prefix", ROWS)
def test_sulfonyl_group_is_named_as_one_prefix(smiles, prefix):
    row = assert_clean(smiles, must_contain=prefix)
    assert not has_replacement_chain(row["name"]), row["name"]
