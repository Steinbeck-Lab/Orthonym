"""Ring-yl prefixes of the drug lane (``rules.ring_substituents``).

- A fragment of two or more ring systems joined by bonds: the ring system that holds the free
  valence is the stem, the others are its ring-yl prefixes, the Blue Book;
  '4-(pyridin-4-yl)benzamide (PIN)',:32893); identical rings joined by a bond are a ring
  assembly ('[1,1'-biphenyl]-4-yl',,:15560).
- A ring system joined to the parent through one -NH-, -O- or -S-: 'anilino',
  :26139), '(R)amino' ('N-phenylpyridin-3-amine (PIN)',:26245), 'phenoxy', '(R)oxy'
  ('2-[(pyridin-3-yl)oxy]pyrazine (PIN)',:27772), '(R)sulfanyl' ('3-chloro-7-[(4-chloro-3-
  nitroquinolin-7-yl)sulfanyl]-4-nitroquinoline (PIN)',:22088).
- "When they relate to substituent groups, they are cited at the front of the corresponding
  prefix",:44643): the multi-system producer and the connective cite no
  stereodescriptor, so a fragment with a stereocentre is left to the producers that were there
  before them, at every tier.
- (2) (:23829): where a linear phane name may be the PIN, the PIN tier builds no
  substitutive name through these prefixes (``rules.linear_phane_screen``).
- (:15542): a molecule that joins two identical ring systems by a bond is a ring assembly,
  the parent of its PIN ('(1P)-2',5'-dimethoxy-6-nitro[1,1'-biphenyl]-2-carboxylic acid (PIN)',
  :49805), mancude and saturated forms of one ring alike,:24153); these producers
  decline it at every tier (``rules.ring_assembly_screen``).
"""
import pytest
from rdkit import Chem

from orthonym.rules import linear_phane_screen, ring_substituents
from tests.support.pin_tiers import assert_not_pin_labelled, assert_pin_at_both_tiers, name_default
from tests.support.rt_assert import assert_full_rt, name_best_effort
from tests.support.spelling_checks import disable_spelling_rule

pytestmark = pytest.mark.opsin_gate


def _side(mol, a, b):
    """The atoms on ``b``'s side of the bond a-b."""
    out, stack = set(), [b]
    while stack:
        x = stack.pop()
        if x in out:
            continue
        out.add(x)
        stack += [n.GetIdx() for n in mol.GetAtomWithIdx(x).GetNeighbors()
                  if n.GetIdx() != a and n.GetIdx() not in out]
    return sorted(out)


#: (SMILES, bond (parent atom, first substituent atom), the prefix at both tiers); main: None
PREFIXES = [
    ("NC(=O)c1cccc(c1)-c1nccc(n1)-c1cccnc1", 7, 9, "4-(pyridin-3-yl)pyrimidin-2-yl"),
    ("NC(=O)c1cccc2cn(-c3ccc(C4CCCNC4)cc3)nc12", 9, 10, "4-(piperidin-3-yl)phenyl"),
    ("NC(=O)c1cccc(c1)Nc1ncccn1", 7, 9, "(pyrimidin-2-yl)amino"),
    ("NC(=O)c1cccc(c1)Nc1ccccc1", 7, 9, "anilino"),
    ("NC(=O)c1cccc(c1)Nc1ccc(Cl)cc1", 7, 9, "4-chloroanilino"),
    ("NC(=O)c1cccc(c1)Oc1ccccc1", 7, 9, "phenoxy"),
    ("NC(=O)c1cccc(c1)Oc1ccccn1", 7, 9, "(pyridin-2-yl)oxy"),
    ("NC(=O)c1cccc(c1)Sc1ccccn1", 7, 9, "(pyridin-2-yl)sulfanyl"),
    # identical rings joined by a bond stay the ring assembly (unchanged)
    ("NC(=O)c1ccc(cc1)-c1ccc(cc1)-c1ccccc1", 6, 9, "[1,1'-biphenyl]-4-yl"),
]


@pytest.mark.parametrize("smiles,a,b,prefix", PREFIXES)
@pytest.mark.parametrize("mancude", [False, True])
def test_ring_yl_prefixes(smiles, a, b, prefix, mancude):
    mol = Chem.MolFromSmiles(smiles)
    assert ring_substituents.name_ring_system_substituent(
        mol, _side(mol, a, b), b, allow_enumerator_fallback=False,
        allow_mancude=mancude) == prefix


def test_identical_rings_joined_by_a_bond_are_not_a_multi_system_ring_yl():
    mol = Chem.MolFromSmiles("NC(=O)c1ccc(cc1)-c1ccc(cc1)-c1ccccc1")
    frag = _side(mol, 6, 9)
    assert ring_substituents._multi_system_ring_yl(mol, frag, set(frag), 9) is None


def test_a_stereocentre_in_a_multi_system_fragment_declines():
    # niraparib: the producer would write '4-(piperidin-3-yl)phenyl' and the outer emitter
    # '(3S)-2-[4-(piperidin-3-yl)phenyl]-2H-indazole-7-carboxamide'
    mol = Chem.MolFromSmiles("NC(=O)c1cccc2cn(-c3ccc([C@@H]4CCCNC4)cc3)nc12")
    frag = _side(mol, 9, 10)
    assert ring_substituents._multi_system_ring_yl(mol, frag, set(frag), 10) is None
    for mancude in (False, True):
        assert ring_substituents.name_ring_system_substituent(
            mol, frag, 10, allow_enumerator_fallback=False, allow_mancude=mancude) is None


#: a stereocentre in the ring-yl of a bare connective: (SMILES, parent atom, heteroatom)
STEREO_CONNECTIVES = [
    ("OC(=O)c1ccc(N[C@@H]2CCCNC2)cc1", 6, 7),
    ("OC(=O)c1ccc(O[C@@H]2CCCNC2)cc1", 6, 7),
    ("OC(=O)c1ccc(S[C@@H]2CCCNC2)cc1", 6, 7),
    ("NC(=O)c1cccc(N[C@@H]2CCCNC2)c1", 7, 8),
]


@pytest.mark.parametrize("smiles,a,b", STEREO_CONNECTIVES)
@pytest.mark.parametrize("mancude", [False, True])
def test_a_stereocentre_in_a_connective_ring_yl_declines(smiles, a, b, mancude):
    # the connective would hand the outer emitter '(piperidin-3-yl)amino', which then writes
    # '(3R)-' in front of the whole name with the prefix's locant
    mol = Chem.MolFromSmiles(smiles)
    assert ring_substituents.name_ring_system_substituent(
        mol, _side(mol, a, b), b, allow_enumerator_fallback=False,
        allow_mancude=mancude) is None


@pytest.mark.parametrize("smiles,misplaced,best_effort", [
    # main's best-effort names (the descriptor at the front of the prefix, without its locant)
    ("OC(=O)c1ccc(N[C@@H]2CCCNC2)cc1", "(3R)-4-[(piperidin-3-yl)amino]benzoic acid",
     "4-[(R)-(piperidin-3-yl)amino]benzoic acid"),
    ("NC(=O)c1cccc(N[C@@H]2CCCNC2)c1", "(3R)-3-[(piperidin-3-yl)amino]benzamide",
     "3-[(R)-(piperidin-3-yl)amino]benzamide"),
    ("OC(=O)c1ccc(S[C@@H]2CCCNC2)cc1", "(3R)-4-[(piperidin-3-yl)sulfanyl]benzoic acid",
     "4-[(R)-(piperidin-3-yl)sulfanyl]benzoic acid"),
])
def test_a_connective_stereocentre_is_not_cited_in_front_of_the_whole_name(
        smiles, misplaced, best_effort):
    assert_not_pin_labelled(smiles, misplaced)
    row = name_best_effort(smiles)
    assert row["name"] == best_effort, row


#: (SMILES, parent atom, first substituent atom): the parent ring and a ring of the molecule
#: are identical ring systems joined by a bond
RING_ASSEMBLY_FRAGMENTS = [
    # '5-[(pyrimidin-2-yl)amino][1,1'-biphenyl]-2-carboxamide', not
    # '2-phenyl-4-[(pyrimidin-2-yl)amino]benzamide'
    ("NC(=O)c1ccc(cc1-c1ccccc1)Nc1ncccn1", 6, 15),
    # "4'-(1H-imidazol-1-yl)[1,1'-biphenyl]-4-carboxamide", not
    # '4-[4-(1H-imidazol-1-yl)phenyl]benzamide'
    ("NC(=O)c1ccc(cc1)-c1ccc(cc1)-n1ccnc1", 6, 9),
    # a pyridine bonded to a piperidine is a ring assembly named with hydro prefixes,
    #:24153; '1,2,3,4,5,6-hexahydro-2,2'-bipyridine (PIN)',:24159), so not
    # '4-{[5-(piperidin-4-yl)pyridin-2-yl]amino}benzoic acid' / '...oxy}benzoic acid' (with the
    # screen off the connective names '[5-(piperidin-4-yl)pyridin-2-yl]amino' / '...oxy')
    ("OC(=O)c1ccc(Nc2ccc(cn2)C2CCNCC2)cc1", 6, 7),
    ("OC(=O)c1ccc(Oc2ccc(cn2)C2CCNCC2)cc1", 6, 7),
]


@pytest.mark.parametrize("smiles,a,b", RING_ASSEMBLY_FRAGMENTS)
@pytest.mark.parametrize("mancude", [False, True])
def test_a_ring_assembly_molecule_gets_no_prefix_from_these_producers(smiles, a, b, mancude):
    # known positives: the same fragments without the ring assembly, '(pyrimidin-2-yl)amino'
    # (PREFIXES) and '3-(1H-imidazol-1-yl)phenyl' (Task L3.6, the benzamide N-substituent)
    mol = Chem.MolFromSmiles(smiles)
    assert ring_substituents.name_ring_system_substituent(
        mol, _side(mol, a, b), b, allow_enumerator_fallback=False,
        allow_mancude=mancude) is None


def test_the_bare_connective_takes_no_enumerator_ring_yl():
    # the enumerator numbers this 1,3,4-oxadiazol-2-yl group '...-1,3,4-oxadiazol-3-yl'; the
    # connective must not carry that into '(...-3-yl)sulfanyl' (best-effort tier)
    mol = Chem.MolFromSmiles("CC1=CC=C(C=C1)C(=O)CSC2=NN=C(O2)COC3=CC=CC4=C3N=CC=C4")
    frag = _side(mol, 9, 10)
    for mancude in (False, True):
        assert ring_substituents.name_ring_system_substituent(
            mol, frag, 10, allow_enumerator_fallback=False, allow_mancude=mancude) is None


@pytest.mark.parametrize("smiles,pin", [
    ("CC(C)(O)CNc1nc(Nc2ccnc(C(F)(F)F)c2)nc(-c2cccc(C(F)(F)F)n2)n1",        # enasidenib
     "2-methyl-1-[(4-[6-(trifluoromethyl)pyridin-2-yl]-6-{[2-(trifluoromethyl)pyridin-4-yl]"
     "amino}-1,3,5-triazin-2-yl)amino]propan-2-ol"),
    ("NC(=O)c1cccc(c1)Nc1ncccn1", "3-[(pyrimidin-2-yl)amino]benzamide"),
    ("NC(=O)c1cccc(c1)Nc1nccs1", "3-[(1,3-thiazol-2-yl)amino]benzamide"),
    ("NC(=O)c1cccc(c1)Nc1ccnc2ccccc12", "3-[(quinolin-4-yl)amino]benzamide"),
    ("NC(=O)c1cccnc1Nc1ccccn1", "2-[(pyridin-2-yl)amino]pyridine-3-carboxamide"),
    ("OC(=O)c1cccc(c1)Nc1ncccn1", "3-[(pyrimidin-2-yl)amino]benzoic acid"),
    ("NC(=O)c1cccc2cn(-c3ccc(C4CCCNC4)cc3)nc12",
     "2-[4-(piperidin-3-yl)phenyl]-2H-indazole-7-carboxamide"),
])
def test_names_built_on_these_prefixes_are_pins(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_niraparib_is_not_certified_with_a_misplaced_stereodescriptor():
    smiles = "NC(=O)c1cccc2cn(-c3ccc([C@@H]4CCCNC4)cc3)nc12"
    assert_not_pin_labelled(smiles, "(3S)-2-[4-(piperidin-3-yl)phenyl]-2H-indazole-7-carboxamide")


def test_the_ketone_keeps_its_suffix_at_best_effort():
    # a connective that took the enumerator's '...-1,3,4-oxadiazol-3-yl' cost the ketone parent
    # its candidates, and best-effort fell to a von Baeyer name without the suffix; whichever
    # producer builds the sulfanyl prefix, the ketone stays the suffix of the
    # 1-(4-methylphenyl)ethan-1-one parent and no prefix cites the free valence at 3
    smiles = "CC1=CC=C(C=C1)C(=O)CSC2=NN=C(O2)COC3=CC=CC4=C3N=CC=C4"
    row = name_best_effort(smiles)
    name = row["name"]
    assert name.endswith("ethan-1-one") and "1-(4-methylphenyl)" in name, row
    assert "oxadiazol-3-yl" not in name, row
    assert_full_rt(name, smiles)


def test_a_phane_class_molecule_gets_no_pin_through_these_prefixes(monkeypatch):
    # four benzene rings and three NH on one chain, the carboxylic acid off it: a phane name
    # may be the PIN (the acid can be its suffix), so the PIN tier declines
    smiles = "OC(=O)c1ccc(Nc2ccc(Nc3ccc(Nc4ccccc4)cc3)cc2)cc1"
    assert linear_phane_screen.phane_may_be_pin(Chem.MolFromSmiles(smiles)) is True
    row = name_default(smiles)
    assert row["tier"] != "pin_verified", row
    # known positive: without the screen the same producers certify the substitutive name; a
    # label check registered with the spelling checks (lane L5) is disabled too,
    # so the known positive still shows what the screen alone stops
    monkeypatch.setattr(linear_phane_screen, "phane_may_be_pin", lambda mol: False)
    disable_spelling_rule(monkeypatch, "P-52.2.5.1")
    row = name_default(smiles)
    assert (row["tier"], row["name"]) == (
        "pin_verified", "4-[4-(4-anilinoanilino)anilino]benzoic acid"), row
