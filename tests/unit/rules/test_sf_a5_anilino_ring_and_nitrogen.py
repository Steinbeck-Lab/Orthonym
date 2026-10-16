"""SF lane A item 5: an anilino prefix whose ring AND whose nitrogen are substituted.

 "Retained names" (the Blue Book): "The prefix name 'anilino' is retained as the
preferred prefix for C6H5-NH- with full substitution allowed. The name 'phenylamino' may be used in
general nomenclature." Its examples (:26153) '4-chloroanilino (preferred prefix) | (4-chlorophenyl)amino';
 (:26308) '3-(N-methylanilino)phenol (PIN) | 3-[methyl(phenyl)amino]phenol'.

The substituents of the ring and the substituent of the nitrogen are cited in ONE alphanumerical
sequence "ALPHANUMERICAL ORDER",:3436), as in the name of the aniline parent
('2-bromo-4-chloro-N-(2,4-dibromophenyl)aniline (PIN)',:22250); a substituent that is also on the ring
joins that group with the locant N first ('N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)',:32879). A
substituted benzene ring on the nitrogen stays the aniline of the other drawing,:21604), so
that pair is not changed. Every expected name here was read back by OPSIN to the input's InChIKey (the
run of the lane); the round trip proves the structure only, the rules above the spelling.

The tests assert the PRODUCER: ``tests/conftest.py`` disables the OPSIN validity gate.
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.rules.anilino_nitrogen import anilino_for_branches
from orthonym.rules.ring_prefix_sequence import ring_prefix_sequence

pytestmark = pytest.mark.unit

ESTER = "CC(OC(=O)c1ccccc1)C(=O)N{n}"


def _name(smiles: str) -> str:
    return Orthonym().name(smiles)


@pytest.mark.parametrize("n_group, expected", [
    # the lane's target row: ring 4-chloro, nitrogen methyl
    ("(C)c1ccc(Cl)cc1", "1-(4-chloro-N-methylanilino)-1-oxopropan-2-yl benzoate"),
    # 'chloro' before 'methyl': the N locant leads the group of the two methyls, ring locant after it
    ("(C)c1cc(Cl)ccc1C", "1-(5-chloro-N,2-dimethylanilino)-1-oxopropan-2-yl benzoate"),
    # the nitrogen substituent is the same as the ring one: one multiplied prefix, N first
    ("(C)c1ccc(C)cc1", "1-(N,4-dimethylanilino)-1-oxopropan-2-yl benzoate"),
    # an unsubstituted phenyl on the nitrogen: the substituted ring is the anilino ring
    ("(c1ccccc1)c1ccc(Cl)cc1", "1-(4-chloro-N-phenylanilino)-1-oxopropan-2-yl benzoate"),
    # a compound N-substituent takes its own marks, the outer marks escalate to brackets
    ("(CCO)c1ccc(Cl)cc1", "1-[4-chloro-N-(2-hydroxyethyl)anilino]-1-oxopropan-2-yl benzoate"),
])
def test_ring_and_nitrogen_substituted_anilino_on_an_acyl_end(n_group, expected):
    assert _name(ESTER.format(n=n_group)) == expected


@pytest.mark.parametrize("n_group, expected", [
    # two substituted benzene rings: the more substituted ring is the aniline, no anilino
    ("(c1ccc(C)cc1)c1ccc(Cl)cc1",
     "1-[(4-chlorophenyl)(4-methylphenyl)amino]-1-oxopropan-2-yl benzoate"),
    # every position of the ring substituted alike takes no locants: no N merge defined
    ("(C)c1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl",
     "1-[methyl(pentachlorophenyl)amino]-1-oxopropan-2-yl benzoate"),
])
def test_shapes_the_merge_does_not_define_keep_their_construction(n_group, expected):
    assert _name(ESTER.format(n=n_group)) == expected


@pytest.mark.parametrize("n_group, expected", [
    # the unsubstituted ring and the one-group forms are unchanged
    ("(C)c1ccccc1", "1-(N-methylanilino)-1-oxopropan-2-yl benzoate"),
    ("c1ccc(Cl)cc1", "1-(4-chloroanilino)-1-oxopropan-2-yl benzoate"),
])
def test_unchanged_anilino_forms(n_group, expected):
    assert _name(ESTER.format(n=n_group)) == expected


def _side_atoms(mol, n_idx, start):
    """The atoms of the group that starts at ``start`` and does not pass through the nitrogen."""
    seen, stack = set(), [start]
    while stack:
        cur = stack.pop()
        if cur in seen or cur == n_idx:
            continue
        seen.add(cur)
        stack.extend(x.GetIdx() for x in mol.GetAtomWithIdx(cur).GetNeighbors())
    return sorted(seen)


def test_anilino_for_branches_merges_n_substituent_into_the_ring_sequence():
    mol = Chem.MolFromSmiles("CC(=O)N(C)c1ccc(Cl)cc1")     # N is atom 3, methyl atom 4, ring 5-10
    ring, methyl = _side_atoms(mol, 3, 5), _side_atoms(mol, 3, 4)
    assert anilino_for_branches(mol, 3, [(ring, "4-chlorophenyl"), (methyl, "methyl")]) == \
        "4-chloro-N-methylanilino"
    assert anilino_for_branches(mol, 3, [(ring, "4-chlorophenyl"), (methyl, "methyl")], enclose=True) == \
        "(4-chloro-N-methylanilino)"
    # a substituted benzene ring on the other group declines
    assert anilino_for_branches(mol, 3, [(ring, "4-chlorophenyl"), (methyl, "4-methylphenyl")]) is None


def test_ring_prefix_sequence_without_n_prefix_is_the_ring_alone():
    """The unchanged rendering of the sequence: 3-chloro-2-methyl, alphanumerical, locants cited."""
    mol = Chem.MolFromSmiles("Cc1ccccc1")
    groups = {"chloro": [3], "methyl": [2]}
    order = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    atom_prefixes = {i: [] for i in range(7)}
    atom_prefixes[1] = ["methyl"]
    atom_prefixes[2] = ["chloro"]
    assert ring_prefix_sequence(mol, groups, atom_prefixes, order, 0) == "3-chloro-2-methyl"
    assert ring_prefix_sequence(mol, groups, atom_prefixes, order, 0, n_prefix="ethyl") == \
        "3-chloro-N-ethyl-2-methyl"


ESTER_SMILES = "CC(OC(=O)c1ccccc1)C(=O)N(C)c1ccc(Cl)cc1"


@pytest.mark.parametrize("smiles, name", [
    (ESTER_SMILES, "1-[(4-chlorophenyl)(methyl)amino]-1-oxopropan-2-yl benzoate"),
    ("CN(c1ccc(Cl)cc1)c1ccc(O)cc1", "4-[(4-chlorophenyl)(methyl)amino]phenol"),
    ("OC(=O)CN(CCO)c1ccc(Cl)cc1", "{(4-chlorophenyl)[2-(hydroxy)ethyl]amino}acetic acid"),
])
def test_spelling_check_lowers_the_general_form_with_the_phenyl_cited_first(smiles, name):
    """The check of read '(phenyl)amino' and '(X-phenyl)amino' but not the phenyl group
    cited FIRST of two ('(4-chlorophenyl)(methyl)amino'), which is how the target row shipped at
    pin_verified. (:26308) '3-(N-methylanilino)phenol (PIN) | 3-[methyl(phenyl)amino]phenol'."""
    from orthonym.validation.pin_spelling import check_pin_spelling
    rules = [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)]
    assert "P-62.2.1.1.1" in rules


@pytest.mark.parametrize("smiles, name", [
    (ESTER_SMILES, "1-(4-chloro-N-methylanilino)-1-oxopropan-2-yl benzoate"),
    ("Oc1cccc(N(C)c2ccccc2)c1", "3-(N-methylanilino)phenol"),
    ("CNc1ccc(Cl)cc1", "4-chloro-N-methylaniline"),
])
def test_spelling_check_keeps_the_preferred_forms(smiles, name):
    from orthonym.validation.pin_spelling import check_pin_spelling
    assert check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True) == []
