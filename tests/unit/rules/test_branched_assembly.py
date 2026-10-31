""": a branched ring assembly of identical benzene rings is named by substituting into its longest unbranched
assembly (``rules.branched_assembly``).

``## **** BRANCHED RING ASSEMBLIES OF IDENTICAL CYCLIC SYSTEMS`` (the Blue Book): "Preferred IUPAC names are
formed by substituting into the longest unbranched assembly. The names of substituent groups are formed in accord with the
methods described in. If necessary, the criteria for selecting the principal chain are applied, namely, the
longest chain, the maximum number of substituents, the lowest locants for substituents considered together, then
alphanumerical order." The examples (:15721-:15731) print the primed names of general nomenclature next to the PINs, whose
composite locants ('2^5-phenyl-1^1,2^1:2^3,3^1-terphenyl') OPSIN 2.9.0 does not read and this producer does not write
(``****``,:15651: "serially primed locants... may be used in general nomenclature"). Every name is read back by
OPSIN 2.9.0 to the full InChIKey of the input before it ships.
"""
import random

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.perception.rings import get_ring_systems
from orthonym.rules import branched_assembly
from orthonym.rules.ring_cluster import is_one_ring_cluster
from tests.support.opsin_marks import composite_or_primed

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# (SMILES, name): the general names of the examples of
BOOK_EXAMPLES = [
    #:15721 '5'-phenyl-1,1':3',1''-terphenyl' (PIN '2^5-phenyl-1^1,2^1:2^3,3^1-terphenyl')
    ("c1ccc(cc1)-c1cc(cc(c1)-c1ccccc1)-c1ccccc1", "5'-phenyl-1,1':3',1''-terphenyl"),
    #:15725 '5''-([1,1'-biphenyl]-3-yl)-1,1':3',1'':3'',1''':3''',1''''-quinquephenyl'
    # (PIN '3^5-([1,1'-biphenyl]-3-yl)-1^1,2^1:2^3,3^1:3^3,4^1:4^3,5^1-quinquephenyl')
    ("C1(=CC(=CC=C1)C=1C=C(C=C(C1)C=1C=C(C=CC1)C1=CC=CC=C1)C1=CC(=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1",
     "5''-([1,1'-biphenyl]-3-yl)-1,1':3',1'':3'',1''':3''',1''''-quinquephenyl"),
    #:15729 '3',4',5',6'-tetraphenyl-1,1':2',1''-terphenyl' (not 2',3',5',6'-...-1,1':4',1''-terphenyl nor
    # 2',4',5',6'-...-1,1':3',1''-terphenyl: "the locant set 1,1':2',1'' is lower than either 1,1':4',1'' or 1,1':3',1''")
    ("c1ccc(cc1)-c1c(-c2ccccc2)c(-c2ccccc2)c(-c2ccccc2)c(-c2ccccc2)c1-c1ccccc1",
     "3',4',5',6'-tetraphenyl-1,1':2',1''-terphenyl"),
]

# ten rings: a benzene ring with three 3,5-disubstituted phenylene arms (a first-generation polyphenylene dendrimer):
# the chain is a quinquephenyl, and the third arm is a substituent that is itself an assembly
DENDRIMER = "c1ccc(-c2cc(-c3ccccc3)cc(-c3cc(-c4cc(-c5ccccc5)cc(-c5ccccc5)c4)cc(-c4cc(-c5ccccc5)cc(-c5ccccc5)c4)c3)c2)cc1"
DENDRIMER_NAME = "5',5'''-diphenyl-5''-([1,1':3',1''-terphenyl]-5'-yl)-1,1':3',1'':3'',1''':3''',1''''-quinquephenyl"


def _read_back_key(name):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    parsed = opsin_parse(name)
    mol = Chem.MolFromSmiles(parsed) if parsed else None
    return Chem.MolToInchiKey(mol) if mol is not None else None


def _tiered(smiles):
    from orthonym.cli import _tiered_namer
    from orthonym.tiers import default_tier
    return _tiered_namer("pin", default_tier("pin")).name_tiered(smiles)


def _star(levels):
    """SMILES of a tree of benzene rings: a ring with six children, each with five,... (``levels`` generations)."""
    mol = Chem.RWMol()

    def ring():
        idx = [mol.AddAtom(Chem.Atom(6)) for _ in range(6)]
        for i in range(6):
            mol.AddBond(idx[i], idx[(i + 1) % 6], Chem.BondType.AROMATIC)
        for a in idx:
            mol.GetAtomWithIdx(a).SetIsAromatic(True)
        return idx

    def grow(parent, positions, depth):
        if depth == 0:
            return
        for p in positions:
            child = ring()
            mol.AddBond(parent[p], child[0], Chem.BondType.SINGLE)
            grow(child, range(1, 6), depth - 1)

    grow(ring(), range(6), levels)
    out = mol.GetMol()
    Chem.SanitizeMol(out)
    return Chem.MolToSmiles(out)


@pytest.mark.parametrize("smiles,name", BOOK_EXAMPLES)
def test_the_general_names_of_the_examples_of_p28_6(smiles, name):
    row = _tiered(smiles)
    assert row["name"] == name
    assert row["producer"] == "inner:ring_assembly"
    assert row["tier"] == "systematic_verified"     # primed locants: general nomenclature, not the PIN
    assert _read_back_key(name) == Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("smiles,name", [*BOOK_EXAMPLES, (DENDRIMER, DENDRIMER_NAME)])
def test_the_name_does_not_depend_on_how_the_structure_is_written(smiles, name):
    """Ring order, atom order and the Kekule form of the input change nothing, determinism)."""
    mol = Chem.MolFromSmiles(smiles)
    spellings = set(Chem.MolToRandomSmilesVect(mol, 6, randomSeed=7))
    assert len(spellings) > 1
    assert {name_compound(s) for s in spellings} == {name}


def test_a_dendrimer_is_named_with_its_longest_chain_and_an_assembly_as_substituent():
    """Ten rings: the longest path of the tree has five (leaf, branch ring, core, branch ring, leaf); the third arm is the
    substituent '[1,1':3',1''-terphenyl]-5'-yl': the free valence of the middle ring at 5', lower than any
    locant the walk from one of its junctions alone would give it)."""
    row = _tiered(DENDRIMER)
    assert (row["name"], row["producer"], row["opsin"]) == (DENDRIMER_NAME, "inner:ring_assembly", "verified")
    assert _read_back_key(DENDRIMER_NAME) == Chem.MolToInchiKey(Chem.MolFromSmiles(DENDRIMER))


def test_the_junction_locants_come_before_the_substituent_locants():
    """:15729: hexaphenylbenzene is '3',4',5',6'-tetraphenyl-1,1':2',1''-terphenyl, although the chain 1,1':4',1'' would
    give its substituents the lower set 2',3',5',6'."""
    name = name_compound("c1ccc(cc1)-c1c(-c2ccccc2)c(-c2ccccc2)c(-c2ccccc2)c(-c2ccccc2)c1-c1ccccc1")
    assert name == "3',4',5',6'-tetraphenyl-1,1':2',1''-terphenyl"


def test_halogens_and_alkyl_groups_are_substituents_of_the_chain():
    smiles = "Clc1ccc(cc1)-c1cc(cc(c1)-c1ccc(C)cc1)-c1ccccc1"     # 1,3,5-triphenylbenzene with a Cl and a CH3 on two rings
    name = name_compound(smiles)
    assert name == name_compound(Chem.MolToRandomSmilesVect(Chem.MolFromSmiles(smiles), 1, randomSeed=3)[0])
    assert _read_back_key(name) == Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    assert "chloro" in name and "methyl" in name and name.endswith("phenyl")


@pytest.mark.parametrize("smiles", [
    "c1ccc(cc1)-c1cc(cc(c1)-c1ccccc1)-c1ccccn1",             # a pyridine ring: the rings are not identical
    "c1ccc(cc1)-c1cc(cc(c1)-c1ccccc1)-c1ccc(O)cc1",          # a hydroxy group: a principal characteristic group
    "c1ccc(cc1)-c1cc(cc(c1)-c1ccccc1)-c1ccc(cc1)[N+](=O)[O-]",
    "C[C@H](c1ccccc1)c1cc(cc(c1)-c1ccccc1)-c1ccccc1",        # a stereocentre
])
def test_what_the_producer_does_not_take_is_declined(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert branched_assembly.detect_branched_ring_assembly(mol, get_ring_systems(mol)) is None


def test_an_unbranched_assembly_is_not_branched():
    mol = Chem.MolFromSmiles("c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccccc1")
    assert branched_assembly.detect_branched_ring_assembly(mol, get_ring_systems(mol)) is None


def test_the_chain_candidates_are_bounded():
    """A tree of 187 rings (a ring with six children, each with five, each with five) has more longest paths than
    ``MAX_CHAIN_CANDIDATES``: the producer declines it at once and the molecule keeps the name it had."""
    mol = Chem.MolFromSmiles(_star(3))
    systems = get_ring_systems(mol)
    assert len(systems) == 1 + 6 + 30 + 150
    assert branched_assembly.detect_branched_ring_assembly(mol, systems) is None


def test_an_unbranched_assembly_with_substituents_on_a_middle_ring_is_numbered_by_its_junctions():
    """The substituent locants of a middle ring are read from the numbering that gives the junctions their locants:
    '5'-chloro-6'-fluoro-1,1':3',1'':4'',1'''-quaterphenyl' (ring 1': junction to the first ring 1', to the third ring 3',
    chlorine at 5', fluorine at 6')."""
    smiles = "Fc1c(Cl)cc(-c2ccc(-c3ccccc3)cc2)cc1-c1ccccc1"
    name = name_compound(smiles)
    # (the Blue Book): where OPSIN reads composite locants the PIN has them (junction set
    # 1^1,2^1:2^3,3^1:3^4,4^1 before the substituents,:16109), else the primed name of general nomenclature
    assert name == composite_or_primed("5'-chloro-6'-fluoro-1,1':3',1'':4'',1'''-quaterphenyl",
                                       "2^5-chloro-2^6-fluoro-1^1,2^1:2^3,3^1:3^4,4^1-quaterphenyl")
    assert _read_back_key(name) == Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def test_an_unbranched_assembly_that_is_named_today_keeps_its_name():
    """``detect_assembly_or_branched`` returns the result of ``detect_ring_assembly`` unchanged when its name is built
    and read back."""
    assert name_compound("Cc1ccc(cc1)-c1ccc(cc1)-c1ccc(C)cc1") == composite_or_primed(
        "4,4''-dimethyl-1,1':4',1''-terphenyl", "1^4,3^4-dimethyl-1^1,2^1:2^4,3^1-terphenyl")   #,:24078
    assert name_compound("Cc1ccc(cc1)-c1ccccc1") == "4-methyl-1,1'-biphenyl"


@pytest.mark.parametrize("smiles,locants", [
    ("c1ccc(cc1)-c1cc(cc(c1)-c1ccccc1)C", {8: "5'", 11: "2'"}),     # meta: the methyl-bearing atom is 5', not 3'
    ("c1ccc(cc1)-c1ccc(cc1)-c1ccccc1", {7: "2'", 8: "2'"}),         # para: the junctions tie, the lower locant is kept
])
def test_the_free_valence_of_a_middle_ring_is_numbered_from_the_junctions(smiles, locants):
    """ (the Blue Book): "Low locants are assigned to ring junctions, then to free valences"."""
    from orthonym.rules import ring_assemblies
    mol = Chem.MolFromSmiles(smiles)
    info = ring_assemblies.detect_ring_assembly(mol, get_ring_systems(mol))
    for atom, locant in locants.items():
        prefix = ring_assemblies.name_ring_assembly_prefix(mol, info, atom)
        assert prefix.endswith(f"-{locant}-yl"), prefix


# ---------------------------------------------------------------------------------------------------------------
# the ring cluster: rings that share atoms are one ring system, rings joined by bonds are not
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("c1ccccc1", True),
    ("c1ccc2ccccc2c1", True),                    # fused
    ("C1CC2CCC1C2", True),                       # bridged
    ("C1CCC2(CC1)CCCC2", True),                  # spiro
    ("c1ccc(cc1)-c1ccccc1", False),              # joined by a bond: a ring assembly
    ("c1ccc2ccccc2c1-c1ccccc1", False),
    ("c1ccc(cc1)-c1ccc(cc1)-c1ccccc1", False),
])
def test_is_one_ring_cluster(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    atoms = {a for ring in mol.GetRingInfo().AtomRings() for a in ring}
    assert is_one_ring_cluster(mol, atoms) is expected


def test_a_ring_system_that_is_part_of_the_molecule_is_judged_alone():
    mol = Chem.MolFromSmiles("c1ccc2ccccc2c1-c1ccccc1")
    naphthalene = max(get_ring_systems(mol), key=len)
    assert is_one_ring_cluster(mol, naphthalene)


def test_a_random_tree_of_rings_is_named_the_same_in_every_spelling():
    """Trees of 5 to 8 benzene rings, a ring joined to a ring of the tree at atoms that carry nothing else: five
    spellings of each give one name, and that name is read back to the structure."""
    rnd = random.Random(3)
    for seed in range(5):
        mol = Chem.RWMol()
        rings, used = [], set()
        for _ in range(rnd.randint(5, 8)):
            idx = [mol.AddAtom(Chem.Atom(6)) for _ in range(6)]
            for i in range(6):
                mol.AddBond(idx[i], idx[(i + 1) % 6], Chem.BondType.AROMATIC)
            for a in idx:
                mol.GetAtomWithIdx(a).SetIsAromatic(True)
            if rings:
                parent = rnd.choice([r for r in rings if any(a not in used for a in r)])
                a = rnd.choice([a for a in parent if a not in used])
                b = rnd.choice(idx)
                mol.AddBond(a, b, Chem.BondType.SINGLE)
                used.update((a, b))
            rings.append(idx)
        built = mol.GetMol()
        Chem.SanitizeMol(built)
        smiles = Chem.MolToSmiles(built)
        names = {name_compound(s) for s in set(Chem.MolToRandomSmilesVect(Chem.MolFromSmiles(smiles), 5, randomSeed=seed))}
        assert len(names) == 1, names
        assert _read_back_key(names.pop()) == Chem.MolToInchiKey(built)
