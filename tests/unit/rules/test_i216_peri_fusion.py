"""Fusion names of bare ortho- and peri-fused ring systems of two first-order components (issue 216).

 "LIMITATIONS OF FUSION NOMENCLATURE: THREE COMPONENTS ORTHO- AND PERI-FUSED TOGETHER" (the Blue Book):
"It is important to recall that benzo heterocycles are considered as one component, thus permitting the construction
of names that would not otherwise be possible." (:11938): "All letters referring to the parent component
are cited, but only locants of the nonfused atoms of the attached component are indicated." (:14472):
"When the fusion principles discussed in through do apply, no skeletal replacement ('a') name is
recommended", so `4,8-dioxacyclopenta[def]fluorene` is not the PIN of the issue molecule.

The expected names other than the issue molecule's are printed in the Blue Book: naphtho[2,1,8-def]quinoline and
quinolizino[4,5,6-bc]quinazoline (:14003-14005,, naphtho[2,1,8-mna]acridine (:11948),
naphtho[1,8-cd][1,2]oxathiole, indeno[7,1-bc]furan (:14347, the fusion parent of a bridged PIN).
"""
import pytest
from rdkit import Chem

from orthonym.rules.fused_ring_selection import _horizontal_row_count
from orthonym.rules.fusion_descriptors import get_fusion_prefix
from orthonym.rules.peri_fusion import peri_fusion_name

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_BOOK = [
    ("C=12C=CC=C3OC=4C=CC=C(C4C31)O2", "[1]benzofuro[4,3,2-bcd][1]benzofuran"),
    ("c1cc2ccc3ccnc4ccc(c1)c2c34", "naphtho[2,1,8-def]quinoline"),
    ("C1=CC2=CC=CC3=c4ccccc4=NC(=C1)N23", "quinolizino[4,5,6-bc]quinazoline"),
    ("c1cc2ccc3nc4ccccc4c4ccc(c1)c2c34", "naphtho[2,1,8-mna]acridine"),
    ("c1cc2c3c(cccc3c1)SO2", "naphtho[1,8-cd][1,2]oxathiole"),
    ("C1=Cc2coc3cccc1c23", "indeno[7,1-bc]furan"),
]


@pytest.mark.parametrize("smiles, expected", _BOOK)
def test_bare_peri_fused_system_gets_the_book_name(smiles, expected):
    assert peri_fusion_name(Chem.MolFromSmiles(smiles)) == expected


def test_issue_216_molecule_is_a_pin():
    from orthonym.cli import _tiered_namer
    row = _tiered_namer("pin", "pin").name_tiered("C=12C=CC=C3OC=4C=CC=C(C4C31)O2")
    assert (row["name"], row["tier"]) == ("[1]benzofuro[4,3,2-bcd][1]benzofuran", "pin_verified")


# Substituted members, on the numbering of interior atom '3a^1', never cited) with the added
# hydrogen of: naphthalimide and ellagic acid.
_SUBSTITUTED = [
    ("O=C1NC(=O)c2cccc3cccc1c23", "1H-benzo[de]isoquinoline-1,3(2H)-dione"),
    ("CN1C(=O)c2cccc3cccc(c23)C1=O", "2-methyl-1H-benzo[de]isoquinoline-1,3(2H)-dione"),
    ("Cc1ccc2cccc3c2c1C(=O)NC3=O", "4-methyl-1H-benzo[de]isoquinoline-1,3(2H)-dione"),
    ("Oc1cc2c(=O)oc3c(O)c(O)cc4c(=O)oc(c1O)c2c34",
     "2,3,7,8-tetrahydroxy[1]benzopyrano[5,4,3-cde][1]benzopyran-5,10-dione"),
    # a five- and six-membered system on the grid drawing (canthin-6-one); 'beta-carboline' is not a PIN component
    # ('9H-pyrido[3,4-b]indole'), so the components are indole and 1,5-naphthyridine
    ("O=c1ccc2nccc3c4ccccc4n1c23", "6H-indolo[3,2,1-de][1,5]naphthyridin-6-one"),
    ("O=c1cc(O)c2nccc3c4ccccc4n1c23", "4-hydroxy-6H-indolo[3,2,1-de][1,5]naphthyridin-6-one"),
]


@pytest.mark.parametrize("smiles, expected", _SUBSTITUTED)
def test_substituted_peri_fused_system(smiles, expected):
    assert peri_fusion_name(Chem.MolFromSmiles(smiles)) == expected


@pytest.mark.parametrize("smiles", [
    "c1ccc2c(c1)oc1ccccc12",                      # dibenzofuran: no interior atom, not this producer's
    "c1ccc2c(c1)ccc1ccccc12",                     # phenanthrene: carbocyclic
    "C1CC2CCCC3CCCC1C23",                         # saturated
])
def test_out_of_scope_is_not_named_here(smiles):
    assert peri_fusion_name(Chem.MolFromSmiles(smiles)) is None


def test_ene_prefix_replaces_the_final_e():
    # (:11905): "replacing the last letter 'e' by 'o'... i.e., indeno from indene"
    assert get_fusion_prefix("indene") == "indeno"
    assert get_fusion_prefix("azulene") == "azuleno"
    assert get_fusion_prefix("naphthalene") == "naphtho"


@pytest.mark.parametrize("smiles, rows", [
    ("c1ccc2cc3ccccc3cc2c1", 3),                  # anthracene
    ("c1ccc2c(c1)ccc1ccccc12", 2),                # phenanthrene
    ("c1ccc2nc3ccccc3cc2c1", 3),                  # acridine
    ("c1ccc2c(c1)cnc1ccccc12", 2),                # phenanthridine
])
def test_rings_in_a_horizontal_row(smiles, rows):
    # (g): "the greatest number of rings in a horizontal row"; acridine is senior to phenanthridine
    m = Chem.MolFromSmiles(smiles)
    assert _horizontal_row_count(m, [tuple(r) for r in m.GetRingInfo().AtomRings()]) == rows


@pytest.mark.parametrize("smiles, expected", [
    #: indicated hydrogen is cited when the mancude parent takes it; OPSIN reads the bare name with its own
    # default hydrogen and the standard InChIKey cannot see where it sits, so the bare name is never shipped for it
    ("c1ccc2c(c1)[nH]c1cccc3ccnc2c13", "7H-pyrido[4,3,2-kl]acridine"),
    ("c1cnc2cccc3[nH]ccc1c23", "6H-benzo[ij][2,7]naphthyridine"),
])
def test_parent_that_takes_indicated_hydrogen_cites_it(smiles, expected):
    assert peri_fusion_name(Chem.MolFromSmiles(smiles)) == expected


def test_ester_outside_the_ring_system_is_not_cited_as_a_prefix():
    # "SENIORITY ORDER FOR CLASSES": the ester outranks the ring ketone, so the ring system is not the parent
    assert peri_fusion_name(Chem.MolFromSmiles("COC(=O)c1ccc2cccc3c2c1C(=O)OC3=O")) is None


@pytest.mark.parametrize("smiles, expected", [
    # (b) (:12163): acridine (3 rings) is the parent, not quinoline; the naphtho name of the 4-ring
    # generate-and-test producer is valid but not the PIN, so the PIN path offers 'benzo[c]acridine'
    ("c1ccc2c(c1)ccc1cc3ccccc3nc12", "benzo[c]acridine"),
    ("c1ccc2c(c1)ccc1nc3ccccc3cc12", "benzo[a]acridine"),
])
def test_three_ring_retained_parent_wins_over_a_naphtho_name(smiles, expected):
    from orthonym.cli import _tiered_namer
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row
