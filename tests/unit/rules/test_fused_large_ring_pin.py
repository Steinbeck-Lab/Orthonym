"""Fused ring systems with a ring of eight or more members (lane DT4-B).

The Blue Book sets no upper ring size for a fusion component: (the Blue Book,
"Prefixes for attached components") "There is no upper ring-size limit to this criterion";
 (:11984, "Heteromonocycles") "Heteromonocycles having more than ten members... are
used as parent components in preferred IUPAC fusion names"; (:11968, "Monocyclic
hydrocarbons (annulenes)") the [n]annulene parent; (:23831, "Mancude systems attached
to adjacent atoms of an alicyclic large ring") "Fusion names... are preferred IUPAC names", with
'5,6,7,8,9,10,11,12,13,14,15,16-dodecahydrobenzo[14]annulene (PIN)'. An ortho-fused benzo on a
large ring is not a cyclophane: the mancude ring must be attached at nonadjacent
positions), so the fusion name stands.

Three groups: BOOK holds the book's own fused names (PIN, or the (PIN) of a name that is also
a component of a bridged name) rebuilt from SMILES; CENSUS holds natural-product macrolides named
since the cap on the ring size was lifted, every name read back with OPSIN 2.9.0; CONTROLS holds
fused names of rings of up to seven members that were named before the change and must stay
byte-identical (they are the OFFER-not-replace check)."""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

# (name printed in the book, SMILES of that structure read by OPSIN 2.9.0)
BOOK = [
    ('5,12-benzodioxacyclooctadecine', 'C1=CC=COC=CC=CC=COC=CC=CC2=C1C=CC=C2'),
    ('1H-3-benzazacycloundecine', 'C1C=NC=CC=CC=CC2=C1C=CC=C2'),
    ('9,2,5-benzoxathiaazacyclododecine', 'C=1SC=CN=CC=COC=C2C1C=CC=C2'),
    ('benzo[8]annulene', 'C1=CC=CC2=C1C=CC=CC=C2'),
    ('1H-cyclopenta[8]annulene', 'C1C=CC2=C1C=CC=CC=C2'),
    ('benzo[1,2:4,5]di[7]annulene', 'C1=CC=CC=C2C1=CC1=CC=CC=CC1=C2'),
    ('1H-benzo[1,2:3,4:5,6]tri[7]annulene', 'C1C=CC=CC=2C1=C1C=CC=CC=C1C1=CC=CC=CC12'),
    ('dibenzo[4,5:6,7]cycloocta[1,2-c]furan', 'C=1OC=C2C1C=C1C(=C3C(=C2)C=CC=C3)C=CC=C1'),
    ("[1,4,7,10]tetraoxacyclohexadecino[13,12-b:14,15-b']dipyridine", 'C=1C=2C(N=CC1)=C1N=CC=CC1=COC=COC=COC=COC2'),
    ('5,6,7,8,9,10,11,12,13,14,15,16-dodecahydrobenzo[14]annulene', 'C1=CC=CC2=C1CCCCCCCCCCCC2'),
    ('3,4-dihydro-2,5-benzodioxocine-1,6-dione', 'C1(OCCOC(C2=C1C=CC=C2)=O)=O'),
    ('dibenzo[b,g][1,6]diazecine', 'C1=CC=CC2=NC=CC3=C(N=CC=C21)C=CC=C3'),
    ('dibenzo[b,f][1,5]diazocine', 'C1=CC=CC2=NC=C3C(=NC=C21)C=CC=C3'),
    ('dibenzo[a,e][8]annulene', 'C1=CC=CC2=C1C=CC1=C(C=C2)C=CC=C1'),
    ('1-benzazocine', 'N1=CC=CC=CC2=C1C=CC=C2'),
    ('naphtho[2,3-c][1]benzazocine', 'C1=CC=CC2=C1C=CC1=C(C=N2)C=C2C=CC=CC2=C1'),
    ('3-benzoxepine', 'C1=COC=CC2=C1C=CC=C2'),
    ('6,7-dihydro-5H-benzo[7]annulene', 'C1=CC=CC2=C1C=CCCC2'),
    ('benzo[14]annulene', 'C1=CC=CC2=C1C=CC=CC=CC=CC=CC=C2'),
    ('2,3,4,5-tetrahydro-1-benzazocine', 'N=1CCCCC=C2C1C=CC=C2'),
]

# (SMILES, name emitted at the PIN tier; each read back to the input InChIKey)
CENSUS = [
    ('COc1cc(O)cc2c1C(=O)O[C@@H](C)CC[C@@H](O)CC/C=C/C(=O)C2',
     '(3S,6S,9E)-6,14-dihydroxy-16-methoxy-3-methyl-3,4,5,6,7,8-hexahydro-1H-2-benzoxacyclotetradecine-1,11(12H)-dione'),
    ('CC1CCC/C=C/C2CC(O)CC2C(O)/C=C\\C(=O)O1',
     '(2Z,10E)-1,13-dihydroxy-6-methyl-1,6,7,8,9,11a,12,13,14,14a-decahydro-4H-cyclopenta[f][1]oxacyclotridecin-4-one'),
    ('CC1=CC(=O)c2c(O)cc(O)cc2CC(=O)O1',
     '7,9-dihydroxy-4-methyl-2H-3-benzoxocine-2,6(1H)-dione'),
    ('CC1CCCC2OC2C(=O)c2c(O)cc(O)cc2CC(=O)O1',
     '10,12-dihydroxy-5-methyl-1a,4,5,13a-tetrahydro-3H-oxireno[2,3-h][3]benzoxacyclododecine-7,13(2H,8H)-dione'),
    ('COc1cc(O)c2c(=O)c3c(oc2c1)-c1ccc(O)cc1OC(C)(C)CC3',
     '3,10-dihydroxy-12-methoxy-6,6-dimethyl-7,8-dihydro-6H,9H-[1]benzopyrano[3,2-e][1]benzoxocin-9-one'),
    ('CC1=CC[C@@H]2[C@H]3/C(O)=C/C(=O)[C@H](C)C(=O)O[C@H](C)/C=C/C=C\\[C@H](C)[C@H]3C=C[C@H]2C1',
     '(1Z,4S,7R,8E,10Z,12S,12aR,14aR,18aS,18bR)-1-hydroxy-4,7,12,16-tetramethyl-7,12,12a,14a,15,18,18a,18b-octahydro-3H-naphtho[1,2-g][1]oxacyclotetradecine-3,5(4H)-dione'),
    ('C\\1COC(=O)C[C@H]2CCC(=O)[C@@H]2C/C=C1',
     '(6E,8aR,11aR)-1,4,5,8,8a,10,11,11a-octahydrocyclopenta[d]oxecine-2,9-dione'),
    # two macrocyclic rows that left test_vonbaeyer_parent_fallback.ASSEMBLY_POSITIVES (von Baeyer names
    # before; the PIN tier abstained)
    ("CC1CCCC(O)/C=C/C2C(O)CC(O)CC2/C=C/C=C\\C/C=C/C=C\\C(=O)O1",
     "(1E,10Z,12E,15Z,17E)-3,20,22-trihydroxy-7-methyl-3,4,5,6,7,14,18a,19,20,21,22,22a-dodecahydro-9H-8-benzoxacycloicosin-9-one"),
    ("CCN1CC(=O)Nc2ccccc2C(=O)Nc2ccccc2C(=O)O[C@H](Cc2ccccc2)C1=O",
     "(10R)-10-benzyl-8-ethyl-7,8-dihydro-6H-dibenzo[h,l][1,4,7,11]oxatriazacyclotetradecine-6,9,12,18(5H,10H,17H)-tetrone"),
]

# (SMILES, name emitted before the change)
CONTROLS = [
    ('C1=CC=CC2=CC=CC=C12', 'naphthalene'),
    ('N1=CC=CC2=CC=CC=C12', 'quinoline'),
    ('C1C=CC2=CC=CC=C12', '1H-indene'),
    ('C1=CC=C2C=CC=CC=C12', 'azulene'),
    ('C1=COC=CC2=C1C=CC=C2', '3-benzoxepine'),
    ('C1=CC=CC2=C1C=CCCC2', '6,7-dihydro-5H-benzo[7]annulene'),
    ('C1C=CC=2C1=C1C=CC=CC1=CC2', '1H-cyclopenta[a]naphthalene'),
    ('C1=CC=CC=2OC3=C(C21)C=CC=C3', 'dibenzo[b,d]furan'),
    ('C=1OC=C2C1C=C1C=CC=CC1=C2', 'naphtho[2,3-c]furan'),
    ('O1C=CCC2=CC=CC=C12', '4H-1-benzopyran'),
    ('C1=CC=CC=2C3=CC=CC=C3C=CC12', 'phenanthrene'),
    ('C1=CC=CC2=NC3=CC=CC=C3C=C12', 'acridine'),
    ('C=1C=2N(C=CN1)C=CC2', 'pyrrolo[1,2-a]pyrazine'),
    ('C1C=CC=C2C1=CC=CC=C2', '1H-benzo[7]annulene'),
    ('C1=CC=CC=2OC3=CC=CC=C3CC12', '9H-xanthene'),
    ('N1=CN=CC2=C1N=CC=C2', 'pyrido[2,3-d]pyrimidine'),
    ('C1=CC=CC=C2C=CC=CC=C12', 'heptalene'),
    ('C1=CC=CC=2NC3=C(C=CC21)C=CC=C3', '5H-dibenzo[b,f]azepine'),
    ('C1CCC2=NC=3C=CC=CC3C=C21', '2,3-dihydro-1H-cyclopenta[b]quinoline'),
    ('N=1C=CN2C1C=CC=C2', 'imidazo[1,2-a]pyridine'),
    ('S1C2=C(C=C1)SC=C2', 'thieno[3,2-b]thiophene'),
]



def _key(smiles):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


@pytest.fixture(scope="module")
def pin():
    return Orthonym(style="pin")


def _name(pin, smiles):
    got = pin.name_tiered(smiles)
    assert got["tier"] == "pin_verified", (smiles, got["tier"], got["name"])
    return got["name"]


def _reads_back(name, smiles):
    parsed = opsin_parse(name)
    assert parsed, name
    assert _key(parsed) == _key(smiles), (name, parsed, smiles)


@pytest.mark.parametrize("book_name,smiles", BOOK, ids=[b[0][:40] for b in BOOK])
def test_book_fused_names_with_a_large_ring(pin, book_name, smiles):
    name = _name(pin, smiles)
    assert name == book_name
    _reads_back(name, smiles)


@pytest.mark.parametrize("smiles,expected", CENSUS, ids=[str(i) for i in range(len(CENSUS))])
def test_census_macrolides_get_their_fusion_name(pin, smiles, expected):
    name = _name(pin, smiles)
    assert name == expected
    _reads_back(name, smiles)


@pytest.mark.parametrize("smiles,expected", CONTROLS, ids=[c[1][:30] for c in CONTROLS])
def test_smaller_ring_fused_names_do_not_move(pin, smiles, expected):
    assert _name(pin, smiles) == expected


def test_a_ring_above_the_lane_limit_declines_cleanly(pin):
    # benzo-fused 40-membered ring: above MAX_FUSED_RING_SIZE (30, the lane's limit, not the book's)
    got = pin.name_tiered("c1ccc2c(c1)" + "C" * 38 + "2")
    assert got["tier"] == "abstain"


@pytest.mark.parametrize("smiles,name", [
    # (the Blue Book) hydro prefixes stand immediately before the parent; the book writes 'hydro[' (6 times)
    # and never 'hydro-[': 'octahydro[1,4]dioxocino[2,3-c][1,6]dioxecine-2,5,9,12-tetrone (PIN)' (:32191)
    ("O1C(CCC(OC2C1COC(CCC(OC2)=O)=O)=O)=O", "octahydro[1,4]dioxocino[2,3-c][1,6]dioxecine-2,5,9,12-tetrone"),
    ("C1COC2=C(O1)OCCO2", "2,3,6,7-tetrahydro[1,4]dioxino[2,3-b][1,4]dioxine"),
])
def test_a_hydro_prefix_before_an_enclosing_mark_takes_no_hyphen(pin, smiles, name):
    got = pin.name_tiered(smiles)
    assert (got["name"], got["tier"]) == (name, "pin_verified"), got
    assert name_is_rt_exact(name, smiles)
