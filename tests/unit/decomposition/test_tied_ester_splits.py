"""Which ester the decomposition cleaves when several rank alike (``decomposition.tied_splits``).

``engine._select_best_bond`` ranks the cleavable bonds by type and by the evenness of the split and
cleaved the first of the ester bonds that tie, in the order of the atoms of the input. The names of
the splits differ and the Blue Book ranks them (module docstring of ``tied_splits``):

* (1), "Within the same class, a ring or ring system has seniority over a chain"
  (the Blue Book), then, "The principal chain has the greater number of skeletal
  atoms [criterion (b) in ]" (:20938): the acid component;
*, "The senior ring or ring system has the lowest locants for points of attachment (if a
  substituent group)" (:21354, 'pyridin-2-yl' before 'pyridin-3-yl'): the alkyl component;
*, "The preferred IUPAC name is the name that is earlier in alphanumerical order... Then, if
  still there is a choice, numerical locants are considered in the order of their appearance in the
  name" (:22234).

The rows are eval rows (milestone1500, best-effort tier) whose name depended on the atom order.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.unit

ROWS = [
    ("P-45.5 locants in order of appearance: 2-methyl before 3-methyl",
     "CC1CN(CCN1C(=O)C=CC(=O)OC2CCCCC2)C(=O)C=CC(=O)OC3CCCCC3",
     "cyclohexyl 4-{4-[4-(cyclohexyloxy)-4-oxobut-2-enoyl]-2-methylpiperazin-1-yl}-4-oxobut-2-enoate"),
    ("P-44.3.2 the longer acid chain (but-2-enoate, four atoms) is the parent",
     "CC(=O)N/C(=C\\C[C@@H]1C[C@@H](C1(C)C)/C=C(/C(=O)OC)\\NC(=O)C)/C(=O)OC",
     "methyl (2Z)-2-acetamido-4-{(1S,3R)-3-[(1Z)-2-acetamido-3-methoxy-3-oxoprop-1-en-1-yl]-2,2-"
     "dimethylcyclobutyl}but-2-enoate"),
    ("P-44.4.1.9 the free valence at locant 1 and not 11",
     "CCCC(C)O[C@H]1[C@H](C(=C)[C@@H]([C@H]2[C@H]([C@](C[C@@]2(C(=O)[C@H](/C=C/C(C1=O)(C)C)C)OC(=O)C)"
     "(C)OC(=O)C)OC(=O)C)OC(=O)C)OC(=O)C",
     "(1R,3S,4E,8S,9S,11R,12S,13R,14R)-9,11,13,14-tetrakis(acetyloxy)-3,6,6,14-tetramethyl-10-"
     "methylidene-2,7-dioxo-8-[(pentan-2-yl)oxy]bicyclo[10.3.0]pentadec-4-en-1-yl acetate"),
]


def _spellings(smiles, n=5):
    mol = Chem.MolFromSmiles(smiles)
    out = list(dict.fromkeys([smiles, Chem.MolToSmiles(mol),
                              *Chem.MolToRandomSmilesVect(mol, n, randomSeed=3)]))
    assert len(out) >= 3
    return out


@pytest.mark.opsin_gate
@pytest.mark.parametrize("rule, smiles, name", ROWS, ids=[r[0] for r in ROWS])
def test_the_split_that_the_criteria_rank_first_names_every_spelling(rule, smiles, name):
    got = {s: name_best_effort(s).get("name") for s in _spellings(smiles)}
    assert set(got.values()) == {name}, got
    assert name_is_rt_exact(name, smiles)


def test_the_acyl_chain_counts_the_carboxyl_carbon_and_stops_at_a_ring():
    from orthonym.decomposition.tied_splits import acyl_chain_atoms
    ester = Chem.MolFromSmiles("CCCC(=O)OC")             # butanoate: four atoms
    assert acyl_chain_atoms(ester, 3) == 4
    branched = Chem.MolFromSmiles("CC(C)CC(=O)OC")       # the longest chain from the acyl carbon: C, C, C(C)... 4
    assert acyl_chain_atoms(branched, 4) == 4
    ring = Chem.MolFromSmiles("O=C(OC)c1ccccc1")         # an acyl carbon on a ring has no chain
    assert acyl_chain_atoms(ring, 1) == 0
    ring_chain = Chem.MolFromSmiles("O=C(OC)CCc1ccccc1")  # the chain stops at the ring: three atoms
    assert acyl_chain_atoms(ring_chain, 1) == 3


def test_the_free_valence_locant_is_read_from_the_alkyl_word_only():
    from orthonym.decomposition.tied_splits import attachment_locant
    assert attachment_locant("(1R)-bicyclo[10.3.0]pentadec-4-en-11-yl acetate") == 11
    assert attachment_locant("pentan-2-yl acetate") == 2
    assert attachment_locant("2H-pyran-3-yl acetate") == 3
    assert attachment_locant("cyclohexyl 4-oxobut-2-enoate") == 0
    assert attachment_locant("methyl 4-(pyridin-2-yl)butanoate") == 0


def test_the_name_order_compares_letters_then_the_free_valence_then_locants_in_order():
    from orthonym.decomposition.tied_splits import name_order_key
    two = "cyclohexyl 4-(2-methylpiperazin-1-yl)-4-oxobutanoate"
    three = "cyclohexyl 4-(3-methylpiperazin-1-yl)-4-oxobutanoate"
    assert name_order_key(two) < name_order_key(three)
    one_yl = "(1R)-bicyclo[10.3.0]pentadec-4-en-1-yl acetate"
    eleven_yl = "(1R)-bicyclo[10.3.0]pentadec-4-en-11-yl acetate"
    assert name_order_key(one_yl) < name_order_key(eleven_yl)
    # different skeletons are decided by the letters, whatever the locant of the free valence
    assert name_order_key("hexan-3-yl acetate") < name_order_key("pentan-2-yl acetate")


# ---- the choice itself, with the names of the splits fixed so that the atom order cannot decide ----

DIESTER = "COC(=O)C(NC(C)=O)=CCC1CC(C=C(NC(C)=O)C(=O)OC)C1(C)C"


def _ester_bonds(mol):
    from orthonym.decomposition.bond_cleavage import find_cleavable_bonds
    bonds = [b for b in find_cleavable_bonds(mol) if b["type"] == "ester"]
    assert len(bonds) == 2
    return bonds


def _chain_atoms(mol, bond):
    from orthonym.decomposition.tied_splits import acyl_chain_atoms
    return acyl_chain_atoms(mol, bond["acid_atom"])


def _choose(mol, group, names, adequate=lambda name, m, b: True):
    from orthonym.decomposition.tied_splits import choose_split
    return choose_split(mol, group, group[0], "pin", lambda m, b, style: names[b["bond_idx"]],
                        adequate, lambda: False)


@pytest.mark.parametrize("reverse", [False, True], ids=["as found", "reversed"])
def test_the_longer_acid_chain_wins_over_the_earlier_name(reverse):
    """ (:20938) ranks the parents before (:22234) ranks the names: the split whose
    acid chain has four atoms wins although its name is later in alphanumerical order."""
    mol = Chem.MolFromSmiles(DIESTER)
    bonds = _ester_bonds(mol)
    assert sorted(_chain_atoms(mol, b) for b in bonds) == [3, 4]
    longer = next(b for b in bonds if _chain_atoms(mol, b) == 4)
    shorter = next(b for b in bonds if _chain_atoms(mol, b) == 3)
    names = {longer["bond_idx"]: "zzz name", shorter["bond_idx"]: "aaa name"}
    group = bonds[::-1] if reverse else bonds
    bond, name = _choose(mol, group, names)
    assert bond["bond_idx"] == longer["bond_idx"] and name == "zzz name"


@pytest.mark.parametrize("reverse", [False, True], ids=["as found", "reversed"])
def test_a_tie_of_the_parents_goes_to_the_earlier_name(reverse):
    mol = Chem.MolFromSmiles("CCCC(=O)OC1CCCCC1OC(=O)CCC")        # two butanoates
    bonds = _ester_bonds(mol)
    assert _chain_atoms(mol, bonds[0]) == _chain_atoms(mol, bonds[1])
    names = {bonds[0]["bond_idx"]: "2-methyl name", bonds[1]["bond_idx"]: "3-methyl name"}
    group = bonds[::-1] if reverse else bonds
    bond, name = _choose(mol, group, names)
    assert (bond["bond_idx"], name) == (bonds[0]["bond_idx"], "2-methyl name")


def test_a_split_without_an_adequate_name_is_not_chosen():
    mol = Chem.MolFromSmiles(DIESTER)
    bonds = _ester_bonds(mol)
    longer = next(b for b in bonds if _chain_atoms(mol, b) == 4)
    shorter = next(b for b in bonds if _chain_atoms(mol, b) == 3)
    names = {longer["bond_idx"]: "long", shorter["bond_idx"]: "short"}
    bond, name = _choose(mol, bonds, names, adequate=lambda n, m, b: b is shorter)
    assert (bond["bond_idx"], name) == (shorter["bond_idx"], "short")
    bond, name = _choose(mol, [longer, shorter], names, adequate=lambda n, m, b: False)
    assert (bond["bond_idx"], name) == (longer["bond_idx"], "long")      # the engine's own choice


def test_symmetric_esters_are_one_split():
    """The two acetates of ethane-1,2-diyl diacetate give one name; only one is named."""
    from orthonym.decomposition.engine import _split_balance
    from orthonym.decomposition.tied_splits import tied_ester_splits
    mol = Chem.MolFromSmiles("CC(=O)OCCOC(C)=O")
    bonds = _ester_bonds(mol)
    assert len(tied_ester_splits(mol, bonds, bonds[0], _split_balance)) == 1
    mol = Chem.MolFromSmiles(DIESTER)
    bonds = _ester_bonds(mol)
    assert len(tied_ester_splits(mol, bonds, bonds[0], _split_balance)) == 2
