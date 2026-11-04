"""A polyester is named from its senior ester, and the alcohol component by its longest chain (``rules.ester_seniority``).

 "Polyester names formed by using substitutive nomenclature, and/or multiplicative nomenclature, and
functional class nomenclature": "When required, seniority of rings, ring systems, and chains; number and location of
substituents; and alphanumerical order is applied in forming the alcoholic component of a functional class ester name
(see through ", with the examples '2-[2-(acetyloxy)ethyl]phenyl acetate (PIN) (not 2-[2-(acetyloxy)phenyl]ethyl
acetate; the ring is senior to the chain)' and '2-{2-[(acetyloxy)methyl]phenyl}ethyl acetate (PIN) (not 2-{2-[(acetyloxy)
ethyl]phenyl}methyl acetate; the ethyl chain is senior to the methyl chain)'. The acid component: 'phenyl 3-(benzoyloxy)
benzoate (PIN) [not 3-(phenoxycarbonyl)phenyl benzoate; the substituted benzoic acid is senior to the unsubstituted one;
see ]'.

The alcohol group is cited by its principal chain: "THE PRINCIPAL SUBSTITUENT CHAIN", (b) "the greater number
of skeletal atoms, i.e., the longest chain" ('1-hydroxypropan-2-yl (preferred prefix)... [not 1-(hydroxymethyl)ethyl]',
under, not by the two carbons of '1-(R-methyl)-2-X-ethyl'.
"""
import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from orthonym.decomposition import bond_cleavage, engine
from orthonym.rules.ester_seniority import ester_seniority_key
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_BUTANOYL = "CCCCOCC(CN(CCOC(=O)CCC)C(=O)CCC)OC(=O)CCC"
_HEXADECANOYL = "CCCCCCCCCCCCCCCCOCC(CN(CCOC(=O)CCCCCCCCCCCCCCC)C(=O)CCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCC"
_PHENYLENE = "CC(=O)OCCc1ccccc1COC(C)=O"
_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")

_CASES = [
    (_BUTANOYL, "1-{butanoyl[2-(butanoyloxy)ethyl]amino}-3-butoxypropan-2-yl butanoate"),
    (_HEXADECANOYL, "1-{hexadecanoyl[2-(hexadecanoyloxy)ethyl]amino}-3-(hexadecyloxy)propan-2-yl hexadecanoate"),
]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", _CASES)
def test_polyester_alcohol_component_by_seniority(smiles, name, tier):
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", _CASES)
def test_the_pin_tier_labels_the_senior_ester_name_a_pin(smiles, name):
    assert _row(smiles, "pin")["tier"] == "pin_verified"


def _split_name(smiles):
    """The name of the split the decomposition takes among the ester bonds that rank alike."""
    mol = Chem.MolFromSmiles(smiles)
    return engine._tied_single_split(mol, bond_cleavage.find_cleavable_bonds(mol), "pin")[1]


@pytest.mark.parametrize("smiles,name", [
    # the ethyl chain is senior to the methyl chain; alphanumerical order alone takes '{2-[2-(acetyloxy)ethyl]...}methyl'
    (_PHENYLENE, "2-{2-[(acetyloxy)methyl]phenyl}ethyl acetate"),
    # the ring is senior to the chain
    ("CC(=O)OCCc1ccccc1OC(C)=O", "2-[2-(acetyloxy)ethyl]phenyl acetate"),
])
def test_the_split_follows_the_seniority_of_the_alcohol_component(smiles, name):
    assert _split_name(smiles) == name
    assert_full_rt(name, smiles)


def _matches(mol):
    return {m[0]: m for m in mol.GetSubstructMatches(_ESTER)}


def test_the_alcohol_on_the_longer_chain_is_the_senior_ester():
    """propan-2-yl (three carbons through the attachment carbon) before ethyl (two): the ester of
    '1-{...amino}-3-butoxypropan-2-yl butanoate' is senior to that of the 2-(butanoyloxy)ethyl arm."""
    mol = Chem.MolFromSmiles(_BUTANOYL)
    esters = _matches(mol)
    assert ester_seniority_key(mol, esters[23]) < ester_seniority_key(mol, esters[12])


def test_the_acid_component_ring_before_chain_then_the_longer_chain():
    """: a ring is senior to a chain;: butanedioic acid is preferred to propanedioic acid
    (the longer acid chain)."""
    ethyl_esters = [Chem.MolFromSmiles(s) for s in ("CCOC(=O)c1ccccc1", "CCCCC(=O)OCC", "CCCC(=O)OCC", "CC(=O)OCC")]
    keys = [ester_seniority_key(m, m.GetSubstructMatch(_ESTER)) for m in ethyl_esters]
    assert keys == sorted(keys) and len(set(keys)) == 4, keys


def test_the_substituted_benzoic_acid_is_senior_to_the_unsubstituted_one():
    """'phenyl 3-(benzoyloxy)benzoate (PIN) [not 3-(phenoxycarbonyl)phenyl benzoate; the substituted benzoic acid is
    senior to the unsubstituted one]'. Atom 1 is the carbonyl carbon of the substituted benzoate."""
    smiles = "O=C(Oc1ccccc1)c1cccc(OC(=O)c2ccccc2)c1"
    mol = Chem.MolFromSmiles(smiles)
    esters = _matches(mol)
    substituted, = (m for c, m in esters.items() if c == 1)
    plain, = (m for c, m in esters.items() if c != 1)
    assert ester_seniority_key(mol, substituted) < ester_seniority_key(mol, plain)
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == ("phenyl 3-(benzoyloxy)benzoate", "pin_verified"), row


def _on_nitrogenous_ring(mol, atom_idx):
    return any(atom_idx in ring and any(mol.GetAtomWithIdx(i).GetSymbol() == "N" for i in ring)
               for ring in mol.GetRingInfo().AtomRings())


def test_the_nitrogenous_ring_is_the_senior_alcohol_ring():
    """ (the Blue Book): "The senior ring or ring system is a heterocycle [criterion (a) in ]";
    the book's example (:31895) '6-[4-(acetyloxy)phenyl]pyridin-3-yl acetate (PIN) (not 4-[5-(acetyloxy)pyridin-2-yl]
    phenyl acetate; the nitrogenous ring is senior to the carbocyclic ring)' builds the name on the pyridine ester. A methyl
    on the benzene gives its ester the greater substituent count, which decided before the ring kind did."""
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccc(cc1C)-c1ccc(OC(C)=O)cn1")
    esters = list(_matches(mol).values())
    pyridine, = (m for m in esters if _on_nitrogenous_ring(mol, m[3]))
    benzene, = (m for m in esters if not _on_nitrogenous_ring(mol, m[3]))
    assert ester_seniority_key(mol, pyridine) < ester_seniority_key(mol, benzene)


def test_the_more_substituted_ring_is_senior_where_the_rings_are_of_one_kind():
    """ (the substituent count): between two benzene rings the ester on the ring that carries a methyl is senior
    (atom 4 is the carbon of the methylated ring that bears the ester oxygen)."""
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccc(cc1C)-c1ccc(OC(C)=O)cc1")
    senior = min(_matches(mol).values(), key=lambda m: ester_seniority_key(mol, m))
    assert senior[3] == 4, senior


def test_the_nitrogenous_ring_is_the_senior_acid_ring():
    """ (:19445): the isonicotinate is the senior acid component against the benzoate, a methyl on the benzene
    notwithstanding."""
    mol = Chem.MolFromSmiles("COC(=O)c1ccc(OC(=O)c2ccncc2)c(C)c1")
    esters = list(_matches(mol).values())
    isonicotinate, = (m for m in esters if any(_on_nitrogenous_ring(mol, nb.GetIdx())
                                               for nb in mol.GetAtomWithIdx(m[0]).GetNeighbors()))
    benzoate, = (m for m in esters if m is not isonicotinate)
    assert ester_seniority_key(mol, isonicotinate) < ester_seniority_key(mol, benzoate)


_LACTONE_ROW = ("CCCC[C@H](O)/C=C/C(=O)O[C@@H]1[C@@H](C)[C@@H]2OC(=O)[C@H](C)[C@H](OC(C)=O)[C@@]2(C)O[C@H]1C")
_LACTONE_ROW_PIN = ("(2S,3R,4R,4aS,7R,8S,8aS)-8-(acetyloxy)-2,4,7,8a-tetramethyl-6-oxooctahydropyrano[3,2-b]pyran-3-yl "
                    "(2E,4S)-4-hydroxyoct-2-enoate")


def test_a_lactone_is_never_the_senior_ester_beside_an_acyclic_ester():
    """ "Cyclic esters" (the Blue Book): "For these compounds [lactones] heterocyclic names are preferred
    IUPAC names": a lactone is a heterocyclic pseudoketone, with no alcohol component to be the parent of a functional
    class ester name. The key ranked its ring acid before the chain acids and the namer built the ester name on it; the
    pyranopyranone below (octenoate, lactone, acetate) lost its PIN."""
    from orthonym.rules.ester_seniority import senior_ester
    mol = Chem.MolFromSmiles(_LACTONE_ROW)
    matches = list(_matches(mol).values())
    lactone, = (m for m in matches if mol.GetBondBetweenAtoms(m[0], m[2]).IsInRing())
    senior = senior_ester(mol, matches)
    assert senior is not None and tuple(senior) != tuple(lactone)
    assert ester_seniority_key(mol, lactone) < ester_seniority_key(mol, senior)   # the key alone would have chosen it
    # a molecule whose only ester is a lactone keeps its (first and only) match
    gamma = Chem.MolFromSmiles("CC1CCC(=O)O1")
    only = list(_matches(gamma).values())
    assert senior_ester(gamma, only) == only[0]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_pyranopyranone_octenoate_keeps_its_pin_beside_a_lactone(tier):
    """A a holdout split row: the PIN tier named it at the lane start with the octenoate as the parent ester (the acid
    component of the longer chain,, the Blue Book) and the lactone cited as 'oxo' on the
    pyrano[3,2-b]pyran,:32080); the senior-ester choice gave the lactone and the PIN tier abstained."""
    row = _row(_LACTONE_ROW, tier)
    assert (row["name"], row["tier"]) == (_LACTONE_ROW_PIN, "pin_verified"), row
    assert_full_rt(row["name"], _LACTONE_ROW)


def test_a_ring_acid_word_never_drops_the_chain_ester_beside_it():
    """The senior ester of 'methyl 4-(3-methoxy-3-oxopropyl)benzoate' is the benzoate: a ring is senior to a
    chain). Where the prefix pipeline names no ester prefix (the PIN tier), its ring acid word was 'methyl benzoate',
    another molecule: the match declines, as the chain acid does, and the decomposition names the molecule."""
    from orthonym.rules.esters import name_ester
    smiles = "COC(=O)CCc1ccc(cc1)C(=O)OC"
    mol = Chem.MolFromSmiles(smiles)
    ring_ester, = (m for m in mol.GetSubstructMatches(_ESTER)
                   if any(n.GetIsAromatic() for n in mol.GetAtomWithIdx(m[0]).GetNeighbors()))
    assert name_ester(mol, ring_ester) is None
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert (row["name"], row["tier"]) == ("methyl 4-(3-methoxy-3-oxopropyl)benzoate", "pin_verified"), row


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)OC(C)COC", "1-methoxypropan-2-yl"),                                   # an ether oxygen links the chain
    ("CC(=O)OC(C)COC(C)=O", "1-(acetyloxy)propan-2-yl"),                          # an ester oxygen
    ("CC(=O)OC(C)CN(CCOC(C)=O)C(C)=O", "1-{acetyl[2-(acetyloxy)ethyl]amino}propan-2-yl"),   # an amide nitrogen
    ("CC(=O)OC(C)CNC", None),       # an amine nitrogen: the deriver reads '-CH2-NH-CH3' as '(aminomethyl)'
    ("CC(=O)OC(CN(C)C)COC", None),
])
def test_the_principal_chain_group_links_by_ether_ester_or_amide_only(smiles, name):
    """ (b), the longest chain: 'propan-2-yl', not '1-methylethyl'; the groups whose chain a plain amine links
    stay with the backbone walk."""
    from orthonym.rules.esters import parse_ester_fragments
    from orthonym.rules.terminal_fragment import _book_principal_chain_fragment
    mol = Chem.MolFromSmiles(smiles)
    match = mol.GetSubstructMatch(_ESTER)
    _acid, alkyl = parse_ester_fragments(mol, match)
    got = _book_principal_chain_fragment(mol, set(alkyl), match[3])
    assert (got.name if got else None) == name
