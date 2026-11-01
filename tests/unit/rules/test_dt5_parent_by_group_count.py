"""Lane DT5, cause D3: the fused system is not the parent because another ring system holds more principal groups.

Blue Book (VERIFIED, ``the Blue Book Blue Book``):

* ``### **** SENIORITY ORDER FOR PARENT STRUCTURES`` (:18871), **** (:18875): "The senior parent
  structure has the maximum number of substituents corresponding to the principal characteristic group (suffix) or
  senior parent hydride in accord with the seniority of classes and the seniority of suffixes." It
  comes before the seniority of ring systems. Table 4.1 (:18189) class 17 "Hydroxy compounds and chalcogen
  analogues (includes alcohols and phenols, which no longer have a separate ranking order)": the phenolic and the
  alcoholic hydroxy groups count together.
* "General methodology for naming substituent groups" (:15800): in a substituent prefix the free valence is the
  'suffix', so a ring system that is a substituent cites its -OH as the prefix 'hydroxy' ('2-hydroxydecahydro-
  naphthalen-1-yl').

Each expected name is read back by OPSIN to the full InChIKey of its molecule (an independent parse)."""
import pytest
from rdkit import Chem

from orthonym.rules.parent_selection import ring_system_with_most_groups
from orthonym.rules.polycyclics import bare_terminal_prefix
from tests.support.pin_tiers import name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# The census row of cause D3 (rsu91 row 15): the benzene-1,3-diol holds more hydroxy groups than the fused system,
# so it is the parent and the fused system a '-yl' prefix. (Rows 25 and 44, a sugar ring with more hydroxy groups
# than the aglycone, need the fusion prefix of the fragment namer: not built, see the lane report.)
CENSUS_ROWS = [
    ('Cc1cc(O)c(C[C@@H]2[C@@]3(C)CCCC(C)(C)[C@@H]3CC[C@@]2(C)O)c(O)c1',
     '2-{[(1R,2R,4aS,8aS)-2-hydroxy-2,5,5,8a-tetramethyldecahydronaphthalen-1-yl]methyl}-5-methylbenzene-1,3-diol'),
]


@pytest.mark.parametrize("smiles,name", CENSUS_ROWS)
def test_census_rows_name_the_ring_with_more_groups_as_parent(smiles, name):
    assert name_is_rt_exact(name, smiles)
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (name, "pin_verified", True), d
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (name, "pin_verified"), b


# (SMILES, name): the two hydroxy groups of the benzene ring outweigh the one of the decalin
GROUP_COUNT_DECIDES = [
    ('Oc1cccc(O)c1C1C(O)CCC2CCCCC12', '2-(2-hydroxydecahydronaphthalen-1-yl)benzene-1,3-diol'),
    ('Oc1cccc(O)c1CC1C(O)CCC2CCCCC12', '2-[(2-hydroxydecahydronaphthalen-1-yl)methyl]benzene-1,3-diol'),
    ('Oc1cccc(O)c1CC1C(C)CCC2CCCCC12', '2-[(2-methyldecahydronaphthalen-1-yl)methyl]benzene-1,3-diol'),
    ('Oc1cccc(O)c1CC1CCCC2CCCCC12', '2-[(decahydronaphthalen-1-yl)methyl]benzene-1,3-diol'),
]


@pytest.mark.parametrize("smiles,name", GROUP_COUNT_DECIDES)
def test_the_ring_with_more_principal_groups_is_the_parent(smiles, name):
    assert name_is_rt_exact(name, smiles)
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (name, "pin_verified", True), d


# Rows the change must leave as they were (named before it, recorded before it): equal numbers of groups leave the
# parent to the seniority of rings: the decalin has two rings), a ring without a group is a substituent
# whatever the size of its chain, and a benzene ring is never the parent of a larger ring system that has no group.
CONTROLS = [
    ('Oc1ccccc1C1C(O)CCC2CCCCC12', '1-(2-hydroxyphenyl)decahydronaphthalen-2-ol'),
    ('c1ccccc1CC1CCCC2CCCCC12', '1-benzyldecahydronaphthalene'),
    ('Oc1ccccc1CC1CCCC2CCCCC12', '2-[(decahydronaphthalen-1-yl)methyl]phenol'),
    ('Oc1ccccc1CC1CCC2CCCC2C1', '2-[(octahydro-1H-inden-5-yl)methyl]phenol'),
    ('Oc1cccc(O)c1C1CCC2CCCCC2C1', '2-(decahydronaphthalen-2-yl)benzene-1,3-diol'),
]


@pytest.mark.parametrize("smiles,name", CONTROLS)
def test_controls_keep_their_names(smiles, name):
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier")) == (name, "pin_verified"), d


def test_ring_system_with_most_groups_needs_a_group_on_the_senior_system_and_a_unique_richer_one():
    from orthonym.perception.rings import get_ring_systems

    def pick(smiles):
        mol = Chem.MolFromSmiles(smiles)
        systems = get_ring_systems(mol)
        senior = max(systems, key=len)                      # the decalin
        from orthonym.perception.functional_groups import detect_functional_groups
        from orthonym.rules.seniority import get_principal_group
        pg, atoms = get_principal_group(mol, detect_functional_groups(mol))
        got = ring_system_with_most_groups(mol, systems, senior, atoms, pg)
        return got is not None and not (set(got) & senior)
    assert pick('Oc1cccc(O)c1C1C(O)CCC2CCCCC12')            # 2 on the benzene ring, 1 on the decalin
    assert not pick('Oc1ccccc1C1C(O)CCC2CCCCC12')           # 1 and 1: a tie, the seniority of rings decides
    assert not pick('Cc1ccccc1C1C(O)CCC2CCCCC12')           # no group on the benzene ring
    assert not pick('Oc1cccc(O)c1C1CCC2CCCCC12')            # none on the senior system: other callers decide


def test_bare_terminal_prefix_is_the_prefix_of_a_ring_that_is_a_substituent():
    mol = Chem.MolFromSmiles('OC1CCCCC1.NC1CCCCC1.SC1CCCCC1.CC1CCCCC1.[O-]C1CCCCC1.O=C1CCCCC1')

    def of(symbol_pos):
        atom = mol.GetAtomWithIdx(symbol_pos)
        ring = next(n.GetIdx() for n in atom.GetNeighbors())
        return bare_terminal_prefix(mol, symbol_pos, ring)
    first = {a.GetIdx(): a.GetSymbol() for a in mol.GetAtoms() if not a.IsInRing() and a.GetDegree() == 1}
    by_symbol = {}
    for idx, sym in first.items():
        by_symbol.setdefault(sym, []).append(idx)
    assert of(by_symbol['O'][0]) == 'hydroxy'               # -OH
    assert of(by_symbol['N'][0]) == 'amino'                 # -NH2
    assert of(by_symbol['S'][0]) == 'sulfanyl'              # -SH
    assert of(by_symbol['C'][0]) is None                    # a methyl is no bare group
    assert of(by_symbol['O'][1]) is None                    # -O-: charged
    assert of(by_symbol['O'][2]) is None                    # =O: a double bond
