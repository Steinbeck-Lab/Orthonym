"""a performance pass, lane R2B: the chain comparison runs after parent selection, for a chain parent only.

 'Maximum number of substituents cited as prefixes' (the Blue Book): "The preferred
IUPAC name is based on the senior parent structure that has the maximum number of substituents
cited as prefixes (other than 'hydro/dehydro') to the parent structure." The comparison picks one
of two chains of one length, and it only matters when the chain is the parent.

``Orthonym._classify`` made the comparison (``chains.p45_principal_chain``) before ``select_parent``
for every cyclic molecule with a chain of two carbons or more. Counting the ring branches of a
large ring system names whole macrocyclic branches (about 2 s each for a 105-atom fragment, six
branches a call), so a molecule whose parent is the ring either way paid for it: +35 s on an allyl
oligosaccharide and +20 s on a decacyclic macrocycle (both over the 60 s limit of the evals).

The comparison now runs after the first selection and only when the chain was chosen; selection is
made again, with the same arguments, when the chain it returns differs from the one passed. That
changes no name. ``p44_scorer.compare_with_reason`` ranks a chain against a ring on the count of
principal groups it holds, its atom class and then 'ring senior to chain'
 'Systems composed of rings and chains (exclusive of linear phanes)', the Blue Book:
"Within the same class, a ring or ring system has seniority over a chain"). The senior chain holds
the same principal-group atoms and has the same length as the default one (``p45_principal_chain``),
so none of the three tells them apart.

Every ring-parent name below is read back by OPSIN's own StdInChIKey against RDKit's key of the
input; the names are the ones the integration branch gave before the change.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym, namer
from orthonym.perception import chains
from orthonym.perception.chains import find_principal_chain
from tests.unit.namer.leads_l1_support import opsin_key, rdkit_key

pytestmark = pytest.mark.opsin_gate

# ring parent (the ring and the chain both hold a principal group: (1) makes the ring the
# parent), a chain of two or more carbons with a ring on one of two chains of one length for the
# first four, so that the comparison used to change the chain and to leave the parent alone
RING_PARENTS = [
    ("OCC(C)Cc1ccc(O)cc1", "4-(3-hydroxy-2-methylpropyl)phenol"),
    ("OCC(C)Cc1ccccc1O", "2-(3-hydroxy-2-methylpropyl)phenol"),
    ("OC(=O)C(C)Cc1ccc(C(=O)O)cc1", "4-(2-carboxypropyl)benzoic acid"),
    ("OC(=O)C(C)CC1CCC(C(=O)O)CC1", "4-(2-carboxypropyl)cyclohexane-1-carboxylic acid"),
    ("OCCC1CCC(O)CC1", "4-(2-hydroxyethyl)cyclohexan-1-ol"),
    ("OCCc1ccc(O)cc1", "4-(2-hydroxyethyl)phenol"),
]

# chain parent, the chain of the comparison differs from the default one
CHAIN_PARENTS = [
    ("OC(=O)C(C)Cc1ccccc1", "2-methyl-3-phenylpropanoic acid"),
    ("CC(Cc1ccccc1)CO", "2-methyl-3-phenylpropan-1-ol"),
    ("COC(=O)C(C)Cc1ccccc1", "methyl 2-methyl-3-phenylpropanoate"),
]


@pytest.fixture
def strict():
    token = namer._DEFAULT_TIER_POLICY_OFF.set(True)
    try:
        yield Orthonym(style="pin")
    finally:
        namer._DEFAULT_TIER_POLICY_OFF.reset(token)


@pytest.fixture
def events(monkeypatch):
    """The order in which ``_classify`` calls the chain comparison and the parent selection."""
    import orthonym.rules.parent_selection as ps
    log = []
    real_p45, real_select = chains.p45_principal_chain, ps.select_parent

    def p45(features, default_chain):
        result = real_p45(features, default_chain)
        log.append(("p45", result != default_chain))
        return result

    def select(*args, **kwargs):
        result = real_select(*args, **kwargs)
        log.append(("select_parent", result.parent_type))
        return result

    monkeypatch.setattr(chains, "p45_principal_chain", p45)
    monkeypatch.setattr(ps, "select_parent", select)
    return log


@pytest.mark.parametrize("smiles, name", RING_PARENTS)
def test_the_ring_parent_names_are_the_molecule_and_unchanged(smiles, name, strict):
    assert opsin_key(name) == rdkit_key(smiles)
    row = strict.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")


@pytest.mark.parametrize("smiles, name", RING_PARENTS)
def test_a_ring_parent_never_runs_the_chain_comparison(smiles, name, strict, events):
    strict.name_tiered(smiles)
    assert [e for e in events if e[0] == "p45"] == []
    assert events[0] == ("select_parent", "ring")


@pytest.mark.parametrize("smiles, name", CHAIN_PARENTS)
def test_a_chain_parent_compares_after_the_selection_and_selects_again(smiles, name, strict, events):
    row = strict.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")
    assert opsin_key(name) == rdkit_key(smiles)
    kinds = [e[0] for e in events]
    # the selection comes first, then the comparison, and the selection is made again with the
    # senior chain: the first call that names the whole molecule is the one that decides
    assert kinds[:3] == ["select_parent", "p45", "select_parent"]
    assert events[0] == ("select_parent", "chain") and events[1] == ("p45", True)


def test_an_equal_chain_is_not_selected_again(strict, events):
    # 2-phenylethan-1-ol: the chain of the comparison is the default chain
    row = strict.name_tiered("OCCc1ccccc1")
    assert (row["name"], row["tier"]) == ("2-phenylethan-1-ol", "pin_verified")
    assert events == [("select_parent", "chain"), ("p45", False)]


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


@pytest.mark.parametrize("smiles", [s for s, _ in RING_PARENTS[:4] + CHAIN_PARENTS])
def test_the_parent_does_not_depend_on_which_of_the_two_chains_is_passed(smiles):
    # the premise of the deferral: two chains of one length that hold the same principal-group atoms
    # are ranked the same way against every ring,,, so the choice
    # between ring and chain is the same for either, and the ring parent's atoms too
    from orthonym.rules.parent_selection import select_parent
    f = _features(smiles)
    ring_atoms = {a for r in f.ring_systems for a in r}
    default = find_principal_chain(f.mol, f.functional_groups, f.principal_group,
                                   exclude_atoms=ring_atoms)
    senior = chains.p45_principal_chain(f, default)
    assert senior != default and len(senior) == len(default)      # the two chains do differ

    def selected(chain):
        return select_parent(mol=f.mol, ring_systems=f.ring_systems, principal_chain=chain,
                             principal_group=f.principal_group,
                             principal_group_atoms=f.principal_group_atoms, ring_info=None)

    a, b = selected(default), selected(senior)
    assert a.parent_type == b.parent_type
    assert a.parent_pool_size == b.parent_pool_size
    if a.parent_type == "ring":
        assert a.parent_atoms == b.parent_atoms


def test_a_comparison_does_not_start_inside_the_naming_of_the_branches_of_another():
    # The comparison names the whole branches of the chains that tie on every count and locant
    #, the Blue Book: order of citation), to put them in order. Those names are
    # sort keys, and a comparison that starts inside one is not made (it answers None, 'undecided',
    # for the chain and the acid chain of an ester): without that the comparison of a macrocycle
    # re-enters itself through every branch and runs the rescue ladders of a whole naming at each
    # level (994 name calls and 323 s under a profiler for the cobyrinate of test_m25_workbudget,
    # 2 calls and 3.6 s for the release, which made no such comparison).
    import orthonym.assembly.substituent_enumerator as se
    ester = Chem.MolFromSmiles("COC(=O)C(C)Cc1ccccc1")
    acid_atoms = [a for a in range(2, ester.GetNumAtoms())]
    # positive controls: outside a branch naming both comparisons are made and give a chain
    assert chains.principal_chain_with_ring_prefixes(_features("OC(=O)C(C)Cc1ccccc1"))
    assert chains.p45_acid_chain(ester, acid_atoms, 2)

    tie = _features("OC(=O)C(CCc1ccccc1)CCc1ccccn1")     # two butanoic chains, one aryl branch each
    inside = []
    real = se.name_substituent_for_ordering

    def spy(mol, frag, attach):
        inside.append((chains.principal_chain_with_ring_prefixes(_features("OC(=O)C(C)Cc1ccccc1")),
                       chains.p45_acid_chain(ester, acid_atoms, 2)))
        return real(mol, frag, attach)

    se.name_substituent_for_ordering = spy
    try:
        chains.principal_chain_with_ring_prefixes(tie)
    finally:
        se.name_substituent_for_ordering = real
    assert inside, "the tie did not name a branch"
    assert all(pair == (None, None) for pair in inside)
    # and the flag is down again afterwards
    assert chains.principal_chain_with_ring_prefixes(_features("OC(=O)C(C)Cc1ccccc1"))
    assert chains.p45_acid_chain(ester, acid_atoms, 2)
