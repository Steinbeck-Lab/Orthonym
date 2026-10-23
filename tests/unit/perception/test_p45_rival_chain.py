"""The chain comparison is made only where another chain could be taken.

``chains.p45_principal_chain`` takes the chain that makes senior (``principal_chain_with_ring_prefixes``)
instead of the default chain only when it has the length of the default chain and holds the
principal-group atoms the default chain holds: a chain that does not hold them is not
taken), and in the direction the default chain has. The comparison names the whole branch of every chain it ties, to put them in order of
citation, the Blue Book "the lower locant or set of locants for substituents cited
as prefixes... in their order of citation in the name"); for a fragment of a peptide with two
glutamic acid residues the two chains through the two carboxyl groups tie and the comparison named
an 87-atom branch (28 s), to discard the chain of the other carboxyl group: p45_principal_chain made
731 comparisons and ``name`` 1,172 calls for one 12-residue peptide, 7 and 406 before the
comparison was made for every fragment, and the name was the same
(``chains._comparison_cannot_change_the_chain``).

The shortcut changes no answer: for every molecule below the chain ``p45_principal_chain`` returns is
the one it returns when the shortcut is switched off (the same list of atoms, in the same order). It
is made only for a principal group whose atoms are its match atoms, with no rival chain and with the
two directions of the default chain giving the principal-group atoms different locants: where they
give the same locants (a dicarboxylic acid chain, a secondary alcohol in the middle of a chain)
``find_principal_chain`` turns the chain it found round, and the two selectors do not find it alike.
"""
import pytest
from rdkit import Chem

from orthonym.perception import chains
from orthonym.perception.chains import find_principal_chain

pytestmark = pytest.mark.unit


def _features(smiles):
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.seniority import get_principal_group
    mol = Chem.MolFromSmiles(smiles)
    fgs = detect_functional_groups(mol)
    pg, atoms = get_principal_group(mol, fgs)

    class F:  # the features the comparison reads
        pass

    f = F()
    f.mol, f.functional_groups, f.principal_group, f.principal_group_atoms = mol, fgs, pg, atoms
    f.ring_systems = [set(r) for r in mol.GetRingInfo().AtomRings()]
    return f


def _default(f):
    ring_atoms = {a for r in f.ring_systems for a in r}
    return find_principal_chain(f.mol, f.functional_groups, f.principal_group, exclude_atoms=ring_atoms)


#: two carboxyl groups on two chains of one length, a ring on each: the chains tie, hold different
#: carboxyl groups, and the chain of the other group is never taken
TWO_GROUPS = [
    "OC(=O)CCC(NC(=O)CCC(=O)O)c1ccccc1",
    "OC(=O)CCC(N)C(=O)NC(CCC(=O)O)Cc1ccccc1",
    "NC(CCC(=O)O)C(=O)NC(CCC(=O)O)C(=O)NCc1ccccc1",
    "OC(=O)CCCC(=O)NC(Cc1ccccc1)CCC(=O)O",
]

#: one group, two chains of one length that hold it: the comparison decides
ONE_GROUP = [
    "OC(=O)C(C)Cc1ccccc1",
    "CC(Cc1ccccc1)CO",
    "N#CC(CO)Cc1ccc([N+](=O)[O-])cc1",
    "COC(=O)C(C)Cc1ccccc1",
    "NC(=O)C(C)Cc1ccccc1",
]


#: no rival chain, but the shortcut is not made: an alcohol, an amine (their bearing carbons count),
#: a dicarboxylic acid chain (the two directions give the same locants), a ketone (skeletal suffix)
NOT_SHORT_CUT = [
    "OCC(O)Cc1ccccc1",
    "CN(C)Cc1ccccc1",
    "OC(=O)CC(Cc1ccccc1)CC(=O)O",
    "CC(=O)CCc1ccccc1",
]


@pytest.mark.parametrize("smiles", TWO_GROUPS)
def test_no_other_chain_holds_the_group_so_the_comparison_is_not_made(smiles, monkeypatch):
    f = _features(smiles)
    default = _default(f)
    assert default and f.principal_group == "carboxylic_acid"
    held = chains._group_atoms_held(f)
    assert chains._rival_chain_holds_the_group(f, default, held) is False
    assert chains._comparison_cannot_change_the_chain(f, default, held) is True

    def made(_features):
        raise AssertionError("the comparison was made")

    monkeypatch.setattr(chains, "principal_chain_with_ring_prefixes", made)
    assert chains.p45_principal_chain(f, default) == default


@pytest.mark.parametrize("smiles", ONE_GROUP)
def test_a_rival_chain_holds_the_group_so_the_comparison_is_made(smiles):
    f = _features(smiles)
    default = _default(f)
    held = chains._group_atoms_held(f)
    assert chains._rival_chain_holds_the_group(f, default, held) is True
    assert chains._comparison_cannot_change_the_chain(f, default, held) is False
    senior = chains.principal_chain_with_ring_prefixes(f)
    assert senior and set(senior) != set(default)            # the comparison does change the chain
    assert chains.p45_principal_chain(f, default) == senior


@pytest.mark.parametrize("smiles", TWO_GROUPS + ONE_GROUP + NOT_SHORT_CUT + [
    "CC(=O)N[C@@H](Cc1ccccc1)c1cc(OC)cc(=O)o1",
    "OCC(C)Cc1ccc(O)cc1",
    "OC(=O)CCc1ccc(CCC(=O)O)cc1",
    "CN(C)Cc1ccccc1",
    "CCN(CC)CCc1ccccc1",
])
def test_the_answer_is_the_one_of_the_comparison_made_in_full(smiles, monkeypatch):
    f = _features(smiles)
    default = _default(f)
    if not default:
        pytest.skip("no chain")
    answer = chains.p45_principal_chain(f, default)
    for attr in ("_p45_ring_prefix_chain", "_p45_ring_prefix_tie", "_p45_acyclic_tie"):
        if hasattr(f, attr):
            delattr(f, attr)
    monkeypatch.setattr(chains, "_comparison_cannot_change_the_chain", lambda *args: False)
    assert chains.p45_principal_chain(f, default) == answer


@pytest.mark.parametrize("smiles", NOT_SHORT_CUT)
def test_the_shortcut_is_not_made_where_the_direction_or_the_group_atoms_are_not_the_same(smiles):
    f = _features(smiles)
    default = _default(f)
    if default:
        assert chains._comparison_cannot_change_the_chain(
            f, default, chains._group_atoms_held(f)) is False


def test_a_molecule_without_a_ring_is_never_short_cut():
    # nothing to compare: the comparison answers None, and so does the full one
    f = _features("OC(=O)C(C)CC")
    default = _default(f)
    assert chains._rival_chain_holds_the_group(f, default, chains._group_atoms_held(f)) is True
