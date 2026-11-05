"""Which ester of a polyester is the parent of its functional-class name: the senior ring by (lane/l17 item 8).

 "Polyester names formed by using substitutive nomenclature, and/or multiplicative nomenclature, and
functional class nomenclature" (the Blue Book): "When required, seniority of rings, ring systems, and chains; number
and location of substituents; and alphanumerical order is applied in forming the alcoholic component of a functional
class ester name (see through ". The book's own rows decide the first criteria ('6-[4-(acetyloxy)phenyl]pyridin-3-yl
acetate (PIN) (not 4-[5-(acetyloxy)pyridin-2-yl]phenyl acetate; the nitrogenous ring is senior to the carbocyclic ring)',
:31895; 'phenyl 3-(benzoyloxy)benzoate (PIN) [not 3-(phenoxycarbonyl)phenyl benzoate; the substituted benzoic acid is senior
to the unsubstituted one; see ]',:31880). The book gives no row of two different ring acids, so the rest is read off
the criteria of " SENIORITY ORDER ONLY FOR RINGS AND RING SYSTEMS" (:19402), (:19408) "The senior ring or ring
system: (a) is a heterocycle; (b) has at least one nitrogen atom; (c) has at least one heteroatom (in the absence of nitrogen)
that occurs earlier in the following sequence: F > Cl > Br > I > O > S > Se > Te > P >...; (d) has the greater number of rings;
(e) has the greater number of skeletal atoms; (f) has the greater number of heteroatoms of any kind; (g) has the greater number
of heteroatoms occurring earlier in the following sequence: F > Cl > Br > I > O > S > Se > Te > N > P >...", which come
before the criteria of ((a) "the greater number of multiple bonds",:21016) and before "the senior parent
structure that has the maximum number of substituents cited as prefixes" (:21604). The acid component is a ring ranked as the
ring system (``rules.ring_selection.ring_system_score``); the key used to hold only the ring kind (N, other hetero, carbon), so
that two rings of one kind were left to the number of substituents (the benzoate over a naphthalene-2-carboxylate or a
spiro[4.5]decane-2-carboxylate) and to the canonical order of the atoms (furan against thiophene).
"""
import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit]

_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


def _senior(smiles):
    """The match of the senior ester of ``smiles`` (``rules.ester_seniority.senior_ester``)."""
    from orthonym.rules.ester_seniority import senior_ester
    mol = Chem.MolFromSmiles(smiles)
    return mol, senior_ester(mol, [tuple(m) for m in mol.GetSubstructMatches(_ESTER)])


def _ring_hetero(mol, atom):
    """The sorted heteroatom symbols of the ring system that holds ``atom``."""
    from orthonym.rules.ester_seniority import _ring_system
    return sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in _ring_system(mol, atom) if mol.GetAtomWithIdx(i).GetSymbol() != "C")


def _acid_ring_atom(mol, match):
    return next(nb.GetIdx() for nb in mol.GetAtomWithIdx(match[0]).GetNeighbors() if nb.GetAtomicNum() == 6 and nb.IsInRing())


# ---------------------------------------------------------------------------------------------------------------------
# the Blue Book's rows of and: the parent they name, at both tiers
# ---------------------------------------------------------------------------------------------------------------------
_BOOK = [
    #:31880 the substituted benzoic acid is senior to the unsubstituted one
    ("O=C(c1ccccc1)Oc1cccc(c1)C(=O)Oc1ccccc1", "phenyl 3-(benzoyloxy)benzoate"),
    #:31895 the nitrogenous ring is senior to the carbocyclic ring (a), (b))
    ("CC(=O)Oc1ccc(cc1)-c1ccc(OC(C)=O)cn1", "6-[4-(acetyloxy)phenyl]pyridin-3-yl acetate"),
    #:31813 the parent structure of the PIN has more substituents
    ("COC(=O)c1cc(Oc2cccc(c2)C(=O)OCC)ccc1Cl", "methyl 2-chloro-5-[3-(ethoxycarbonyl)phenoxy]benzoate"),
    #:31897 the ring is senior to the chain
    ("CC(=O)OCCc1ccccc1OC(C)=O", "2-[2-(acetyloxy)ethyl]phenyl acetate"),
    #:31899 the ethyl chain is senior to the methyl chain
    ("CC(=O)OCc1ccccc1CCOC(C)=O", "2-{2-[(acetyloxy)methyl]phenyl}ethyl acetate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", _BOOK)
def test_the_book_rows_keep_their_parent(smiles, name, tier):
    row = _row(smiles, tier)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


# ---------------------------------------------------------------------------------------------------------------------
# (a) to (g), and between the two rings: (SMILES, the name with the senior ring as the parent,
# the criterion). The acid rings are compared in the first block, the alcohol rings in the second.
# ---------------------------------------------------------------------------------------------------------------------
_ACID_RINGS = [
    ("O=C(c1ccco1)Oc1ccc(C(=O)OC)cc1", "4-(methoxycarbonyl)phenyl furan-2-carboxylate", "(a) a heterocycle over a carbocycle"),
    ("O=C(c1cccnc1)Oc1ccc(C(=O)OC)o1", "5-(methoxycarbonyl)furan-2-yl pyridine-3-carboxylate", "(b) a nitrogen atom"),
    ("O=C(c1ccco1)Oc1ccc(C(=O)OC)s1", "5-(methoxycarbonyl)thiophen-2-yl furan-2-carboxylate", "(c) O before S, no nitrogen"),
    ("O=C(c1ccc2ccccc2n1)Oc1ccc(C(=O)OC)nc1", "6-(methoxycarbonyl)pyridin-3-yl quinoline-2-carboxylate", "(d) two rings over one"),
    ("O=C(C1CCC2(C1)CCCCC2)Oc1ccc(C(=O)OC)cc1", "4-(methoxycarbonyl)phenyl spiro[4.5]decane-2-carboxylate", "(d) two rings over one"),
    ("O=C(c1cccnc1)Oc1ccc(C(=O)OC)[nH]1", "5-(methoxycarbonyl)-1H-pyrrol-2-yl pyridine-3-carboxylate", "(e) six skeletal atoms over five"),
    ("O=C(c1cncnc1)Oc1ccc(C(=O)OC)nc1", "6-(methoxycarbonyl)pyridin-3-yl pyrimidine-5-carboxylate", "(f) two nitrogen atoms over one"),
    ("O=C(c1ncco1)Oc1cc(C(=O)OC)sn1", "5-(methoxycarbonyl)-1,2-thiazol-3-yl 1,3-oxazole-2-carboxylate", "(g) O before S, both with N"),
    ("O=C(C1=CCCCC1)OC1CCC(CC1)C(=O)OC", "4-(methoxycarbonyl)cyclohexyl cyclohex-1-ene-1-carboxylate",
     "P-44.4.1.1 more multiple bonds"),
    # the ring that is junior by carries three substituents, the benzene none: comes after
    ("O=C(c1ccccc1)OCC1CC(C)CC(C)C1C(=O)OC", "[2-(methoxycarbonyl)-3,5-dimethylcyclohexyl]methyl benzoate",
     "P-44.4.1.1 before P-45.2.1"),
]
_ALCOHOL_RINGS = [
    ("CC(=O)Oc1ccc(o1)-c1ccc(OC(C)=O)s1", "5-[5-(acetyloxy)thiophen-2-yl]furan-2-yl acetate", "(c) O before S, no nitrogen"),
    ("CC(=O)Oc1ccc2ccccc2c1-c1ccc(OC(C)=O)cc1", "1-[4-(acetyloxy)phenyl]naphthalen-2-yl acetate", "(d) two rings over one"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name,criterion", _ACID_RINGS + _ALCOHOL_RINGS)
def test_the_parent_is_the_ester_of_the_senior_ring_at_best_effort(smiles, name, criterion):
    row = _row(smiles, "best-effort")
    assert row["name"] == name, (criterion, row)
    assert_full_rt(row["name"], smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name,criterion", _ACID_RINGS + _ALCOHOL_RINGS)
def test_the_pin_tier_names_the_senior_ring_or_nothing(smiles, name, criterion):
    """The PIN tier never gives the junior ring as the parent: the name of the senior ring, or no name."""
    row = _row(smiles, "pin")
    assert row["tier"] == "abstain" or row["name"] == name, (criterion, row)
    if row["tier"] != "abstain":
        assert_full_rt(row["name"], smiles)


# a ring that the key leaves to the order of the atoms is given the same parent by every spelling of the molecule
_SPELLINGS = [
    ("c1nc(C(Oc2cc(C(OC)=O)sn2)=O)oc1", "5-(methoxycarbonyl)-1,2-thiazol-3-yl 1,3-oxazole-2-carboxylate"),
    ("n1sc(cc1OC(c1occn1)=O)C(OC)=O", "5-(methoxycarbonyl)-1,2-thiazol-3-yl 1,3-oxazole-2-carboxylate"),
    ("c1cc(OC(C)=O)sc1-c1ccc(OC(C)=O)o1", "5-[5-(acetyloxy)thiophen-2-yl]furan-2-yl acetate"),
    ("C(Oc1oc(cc1)-c1ccc(OC(C)=O)s1)(C)=O", "5-[5-(acetyloxy)thiophen-2-yl]furan-2-yl acetate"),
    ("COC(c1ccc(OC(=O)C2CCC3(C2)CCCCC3)cc1)=O", "4-(methoxycarbonyl)phenyl spiro[4.5]decane-2-carboxylate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", _SPELLINGS)
def test_every_spelling_gives_the_same_parent(smiles, name):
    row = _row(smiles, "best-effort")
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)


# ---------------------------------------------------------------------------------------------------------------------
# the key itself (no OPSIN): the senior ester of each molecule, by the heteroatoms of the ring of its acid or its alcohol
# ---------------------------------------------------------------------------------------------------------------------
_SENIOR_ACID_RING = [(s, hetero) for s, hetero in [
    ("O=C(c1ccco1)Oc1ccc(C(=O)OC)s1", ["O"]),                            # (c)
    ("O=C(c1cccs1)Oc1ccc(C(=O)OC)o1", ["O"]),
    ("O=C(c1ncco1)Oc1cc(C(=O)OC)sn1", ["N", "O"]),                       # (g)
    ("O=C(c1ccsn1)Oc1nc(C(=O)OC)co1", ["N", "O"]),
    ("O=C(c1cncnc1)Oc1ccc(C(=O)OC)nc1", ["N", "N"]),                     # (f)
    ("O=C(c1cccnc1)Oc1cnc(nc1)C(=O)OC", ["N", "N"]),
    ("O=C(c1ccc2ccccc2n1)Oc1ccc(C(=O)OC)nc1", ["N"]),                    # (d): quinoline has two rings
    ("O=C(c1cccnc1)Oc1ccc(C(=O)OC)[nH]1", ["N"]),                        # (e): pyridine has six atoms
    ("O=C(c1ccccc1)Oc1ccc(C(=O)OC)o1", ["O"]),                           # (a)
    ("O=C(c1ccccc1)OCC1CC(C)CC(C)C1C(=O)OC", []),                        #: the benzene, not the cyclohexane
    ("O=C(C1CCC2(C1)CCCCC2)Oc1ccc(C(=O)OC)cc1", []),                     # (d): the spiro system, not the benzene
]]


@pytest.mark.parametrize("smiles,hetero", _SENIOR_ACID_RING)
def test_the_senior_ester_has_the_senior_acid_ring(smiles, hetero):
    mol, match = _senior(smiles)
    ring = _acid_ring_atom(mol, match)
    assert _ring_hetero(mol, ring) == hetero, (smiles, match)
    if not hetero:
        from orthonym.rules.ester_seniority import _ring_system
        # the carbocycle of the senior ester is the benzene or the spiro system (d))
        system = _ring_system(mol, ring)
        assert (len(system) == 6 and mol.GetAtomWithIdx(ring).GetIsAromatic()) or len(system) == 10, (smiles, match)


def test_ring_kind_alone_no_longer_decides_between_two_hetero_rings():
    """The key held the kind (N, other hetero, carbon): two rings of one kind were ranked by their substituents."""
    from orthonym.rules.ester_seniority import ester_seniority_key
    mol = Chem.MolFromSmiles("O=C(c1ccco1)Oc1ccc(C(=O)OC)s1")
    furoate, thiophenecarboxylate = [tuple(m) for m in mol.GetSubstructMatches(_ESTER)]
    assert ester_seniority_key(mol, furoate) < ester_seniority_key(mol, thiophenecarboxylate)


def test_a_chain_acid_stays_below_every_ring_acid():
    """ (1), the Blue Book: a ring has seniority over a chain, in the key as before."""
    mol, match = _senior("O=C(C1CCCCC1)OCC(=O)OC")
    assert mol.GetAtomWithIdx(_acid_ring_atom(mol, match)).IsInRing()
