"""LO4a class A4 (the part the builder can spell): a ring hydrazide is a pseudoketone, and a substituted
amino prefix is spelled as one.

``bridged_fused_pin.principal_suffix`` declined 19 of the fused-guanidine rows. Two of the causes were
bounded and are fixed here; the others need a decorated fused ring as a substituent (see the lane
record, LO4a Left).

1. ``CCCC1C(=O)N2C(N(C)C)=Nc3ccc(C)cc3N2C1=O``: the principal group the perception reports is 'hydrazide'
   (the ring -CO-N-N-CO-), a class the builder did not read as a ring group, so it declined.
   Pseudoketones (the Blue Book),:28262 "(a) cyclic compounds in which a carbonyl group in a ring is
   bonded to one or two skeletal heteroatoms"; (:29314) "Cyclic anhydrides, esters and amides are
   named as pseudoketones; the resulting names are preferred IUPAC names"; the class is 16 of (:18189). The carbonyls of a ring N-N diacyl are bonded to ring nitrogens, so they are pseudoketones.
   ASSUMED (the book prints no cyclic diacylhydrazine, and hydrazides are class 12): the whole -CO-N-N-
   unit must lie in the ring system (an N-amino lactam keeps declining); the engine's default tier
   already names '5-amino-2,3-dihydrophthalazine-1,4-dione' the same way.
2. The same row's exocyclic ring-guanidine nitrogen carries two methyls, a substituted amino group. The
   fragment namer (``name_substituent_fragment``) reads an N-attached fragment as a carbon chain and wrote
   'amino-N-methylmethyl' for 'dimethylamino' and 'aminomethyl' for 'methylamino', a WRONG molecule that
   OPSIN rejected, so the builder never shipped an N-alkylamino prefix. /
   (:7272) "The second and further substituents are each enclosed with parentheses even for simple
   substituents": '(dimethylamino)', 'ethyl(methyl)amino'. The enumerator's namer
   (``substituent_enumerator.name_substituent``) spells them; ``polycyclics.exocyclic_substituent_name``
   routes a neutral substituted nitrogen to it.
"""
import pytest
from rdkit import Chem

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

HYDRAZIDE_ROWS = [
    # the item row: a ring diacylhydrazine plus a dimethylamino on the ring guanidine carbon
    ("CCCC1C(=O)N2C(N(C)C)=Nc3ccc(C)cc3N2C1=O",
     "5-(dimethylamino)-9-methyl-2-propyl-1H-pyrazolo[1,2-a][1,2,4]benzotriazine-1,3(2H)-dione"),
    ("O=C1CC(=O)N2N1CCC2", "6,7-dihydro-1H,5H-pyrazolo[1,2-a]pyrazole-1,3(2H)-dione"),
    ("O=C1CC(=O)N2N1c1ccccc1C2", "1H,9H-pyrazolo[1,2-a]indazole-1,3(2H)-dione"),
    ("CN(C)C1=NN2C(=O)CCC2=N1", "2-(dimethylamino)-6,7-dihydro-5H-pyrrolo[1,2-b][1,2,4]triazol-5-one"),
]


@pytest.mark.parametrize("smiles,pin", HYDRAZIDE_ROWS)
def test_ring_hydrazide_is_a_pseudoketone(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_an_n_amino_lactam_is_not_a_ring_hydrazide():
    """The N-NH2 of a lactam is outside the ring system: not the -CO-N-N- unit of a ring, and the
    builder keeps declining it (the carbonyl is a pseudoketone, the N-amino an 'amino' prefix, but the
    class order puts the hydrazide above it and the book decides nothing)."""
    from orthonym.rules.bridged_fused_pin import build_fused
    assert build_fused(Chem.MolFromSmiles("NN1C(=O)CCc2ccccc21")) is None


# the builder's own name, with the prefix spelled by the enumerator's namer
AMINO_ROWS = [
    ("CNC1CCc2ccccc2C1=O", "2-(methylamino)-3,4-dihydronaphthalen-1(2H)-one"),
    ("CN(C)C1CCc2ccccc2C1=O", "2-(dimethylamino)-3,4-dihydronaphthalen-1(2H)-one"),
    ("CCN(C)C1CCc2ccccc2C1=O", "2-[ethyl(methyl)amino]-3,4-dihydronaphthalen-1(2H)-one"),
    ("c1ccccc1NC1CCc2ccccc2C1=O", "2-anilino-3,4-dihydronaphthalen-1(2H)-one"),
    ("NC1CCc2ccccc2C1=O", "2-amino-3,4-dihydronaphthalen-1(2H)-one"),
]


@pytest.mark.parametrize("smiles,name", AMINO_ROWS)
def test_the_builder_spells_a_substituted_amino_prefix(smiles, name):
    """Before: 'aminomethyl' for 'methylamino' (a different molecule), 'amino-N-methylmethyl' for
    'dimethylamino'."""
    from orthonym.rules.bridged_fused_pin import build_fused
    from tests.support.rt_assert import name_is_rt_exact
    got = build_fused(Chem.MolFromSmiles(smiles))
    assert got is not None and got[0] == name, got
    assert name_is_rt_exact(name, smiles), name


def test_an_n_aryl_prefix_keeps_the_fragment_namer():
    """A first version routed every N-attached fragment to the enumerator's namer, whose best-effort
    spelling of '4-(trifluoromethoxy)anilino' is '{[4-(trifluoromethoxy)phenyl]amino}' (pin_unverified): the
    N-aryl fragment keeps the namer that spells it right."""
    assert_pin_at_both_tiers("O=c1nc(Nc2ccc(OC(F)(F)F)cc2)c2ccccc2o1",
                             "4-[4-(trifluoromethoxy)anilino]-2H-1,3-benzoxazin-2-one")


@pytest.mark.parametrize("smiles,expected", [
    ("CNC1CCc2ccccc2C1=O", "methylamino"),
    ("CN(C)C1CCc2ccccc2C1=O", "dimethylamino"),
    ("O=C=NC1CCc2ccccc2C1=O", "isocyanato"),
    ("CN=C1CCc2ccccc2C1=O", "methylimino"),
    ("NC(=N)NC1CCc2ccccc2C1=O", "carbamimidoylamino"),        #, the Blue Book
    ("CC(=O)NC1CCc2ccccc2C1=O", "acetamido"),
    ("C[N+](C)(C)C1CCc2ccccc2C1=O", "N,N-dimethylmethanaminiumyl"),  # a charged nitrogen keeps the fragment namer (:42473)
    ("NNC1CCc2ccccc2C1=O", "hydrazinyl"),                     # a nitrogen with a heteroatom neighbour too
])
def test_exocyclic_substituent_name(smiles, expected):
    from orthonym.rules.polycyclics import exocyclic_substituent_name
    mol = Chem.MolFromSmiles(smiles)
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    nitrogen = next(a.GetIdx() for a in mol.GetAtoms()
                    if a.GetSymbol() == "N" and not a.IsInRing()
                    and any(n.GetIdx() in ring for n in a.GetNeighbors()))
    frag, stack, seen = [], [nitrogen], set(ring)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        frag.append(cur)
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors())
    assert exocyclic_substituent_name(mol, sorted(frag), nitrogen, sorted(ring)) == expected


def test_a_dihydroxyamino_group_is_not_read_as_nitro():
    """The enumerator's namer reads N(OH)2 as 'nitro' (another molecule): a nitrogen with a heteroatom neighbour
    keeps the fragment namer, which declines it, so the builder gives no name (the guard of
    test_bf_s2_groups.test_a_group_nitro_would_misname_gets_no_nitro_name)."""
    from orthonym.rules.bridged_fused_pin import build_fused
    assert build_fused(Chem.MolFromSmiles("OC(=O)c1cc2C3CCC(C3)c2cc1N(O)O")) is None
