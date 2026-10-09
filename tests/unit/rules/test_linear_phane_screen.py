""" (2): may a linear phane name be the PIN? (``rules.linear_phane_screen``).

 (2) (the Blue Book) "linear phanes consist of four or more rings or ring
systems, two of which must be terminal, and together with acyclic atoms or chains must consist
of at least seven nodes (components)."; (:23901) such compounds take phane PINs "even
though the compounds could also be named by substitutive or multiplicative nomenclature".
 (:18873) is applied first: a phane that cites fewer principal-group suffixes than another
parent is not the PIN,:18875). The screen fails closed: True means a substitutive
name of the molecule must not be certified by the drug lane's producers.
"""
import pytest
from rdkit import Chem

from orthonym.rules import linear_phane_screen
from orthonym.rules.linear_phane_screen import phane_may_be_pin

#: the book's linear phane PINs and their boundary rows
BOOK_ROWS = [
    #:23917 '4(3,5)-pyridina-1,7(1),2,3,5,6(1,3)-hexabenzenaheptaphane (PIN, a phane name)'
    ("c1ccc(cc1)-c1cccc(c1)-c1cccc(c1)-c1cncc(c1)-c1cccc(c1)-c1cccc(c1)-c1ccccc1", True),
    #:23959 '2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphane (PIN, a phane name)'
    ("c1ccc(cc1)Oc1ccc(cc1)Oc1ccc(cc1)Oc1ccccc1", True),
    #:26409 '2-(4-aminophenyl)-2,4,6-triaza-...-tetrabenzenaheptaphane-1^4,7^4-diamine (PIN)'
    ("Nc1ccc(cc1)N(c1ccc(N)cc1)c1ccc(cc1)Nc1ccc(cc1)Nc1ccc(N)cc1", True),
    #:18915 '4-{4-[(pyridin-4-yl)methyl]phenyl}-...-tetrabenzenaheptaphane-1^4,7^4-dicarboxylic
    # acid (PIN)'
    ("OC(=O)c1ccc(cc1)Cc1cccc(c1)C(c1ccc(Cc2ccncc2)cc1)c1ccc(cc1)Cc1ccc(cc1)C(O)=O", True),
    #:23913 '3,5-di([1,1'-biphenyl]-3-yl)pyridine (PIN, substitutive name)': five rings, five nodes
    ("c1ccc(cc1)-c1cccc(c1)-c1cncc(c1)-c1cccc(c1)-c1ccccc1", False),
    #:23925 "1,1'-[1,4-phenylenebis(oxy)]dibenzene (PIN, a multiplicative name)": three rings
    ("c1ccc(cc1)Oc1ccc(cc1)Oc1ccccc1", False),
]


@pytest.mark.parametrize("smiles,expected", BOOK_ROWS)
def test_the_books_phane_rows_and_their_boundaries(smiles, expected):
    assert phane_may_be_pin(Chem.MolFromSmiles(smiles)) is expected


#: approved drugs that meet the structure test of (2)
DRUG_ROWS = [
    # the amide links two chain rings: its carbon and nitrogen are skeletal nodes, a phane cites
    # it as 'oxo' + 'aza' (pseudoketone definition,:1878 "except for nitrogen"), picks
    # the benzamide (nilotinib, imatinib, ponatinib) or the thiazolecarboxamide (dasatinib)
    ("CC1=C(C=C(C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N2C=NC(=C2)C)C=C1)NC1=NC=CC(=N1)C=1C=NC=CC1", False),
    ("Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", False),
    ("Cc1ccc(cc1C#Cc1cnc2cccnn12)C(=O)Nc1ccc(CN2CCN(C)CC2)c(c1)C(F)(F)F", False),
    ("Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1", False),
    # an N-acyl ring nitrogen is a pseudoketone,:33127): its carbon on the skeleton
    # is cited as '-one', the phane wins (venadaparib, olaparib)
    ("C1(CC1)NCC1CN(C1)C(=O)C=1C=C(C=CC1F)CC1=NNC(C2=CC=CC=C12)=O", True),
    ("O=C(c1cc(Cc2n[nH]c(=O)c3ccccc23)ccc1F)N1CCN(C(=O)C2CC2)CC1", True),
    # the principal group (hydroxy) sits off the chain: the phane may cite it (bazedoxifene)
    ("Cc1c(-c2ccc(O)cc2)n(Cc2ccc(OCCN3CCCCCC3)cc2)c2ccc(O)cc12", True),
    # fewer than four ring systems, or fewer than seven nodes, on any path
    ("CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1", False),   # sorafenib
    ("CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1", False),          # losartan
]


@pytest.mark.parametrize("smiles,expected", DRUG_ROWS)
def test_approved_drugs(smiles, expected):
    assert phane_may_be_pin(Chem.MolFromSmiles(smiles)) is expected


def test_a_ketone_chain_keeps_the_phane():
    # Ph-CO-C6H4-CO-C6H4-CO-Ph: four rings and seven nodes; the phane cites the three ketones
    # as a trione, a substitutive parent ('methanone') one
    assert phane_may_be_pin(Chem.MolFromSmiles(
        "O=C(c1ccccc1)c1ccc(cc1)C(=O)c1ccc(cc1)C(=O)c1ccccc1")) is True


def test_an_acyl_ring_nitrogen_on_the_chain_keeps_the_phane():
    assert phane_may_be_pin(Chem.MolFromSmiles(
        "O=C(c1ccccc1)N1CCN(CC1)c1ccc(cc1)Oc1ccc(cc1)Oc1ccccc1")) is True


def test_an_ester_on_the_chain_is_absorbed():
    # the two esters link chain rings: 'oxo' + 'oxa' in a phane name (as the inner esters of
    # 'phenyl 3,6,9-trioxo-2,5,8-trioxa-...-decaphane-1^3-carboxylate (PIN)',:31927)
    assert phane_may_be_pin(Chem.MolFromSmiles(
        "O=C(Oc1ccccc1)c1ccc(cc1)C(=O)Oc1ccc(cc1)Oc1ccccc1")) is False


def test_an_error_fails_closed(monkeypatch):
    from orthonym.rules import seniority

    def boom(*a, **k):
        raise RuntimeError("perception failed")
    monkeypatch.setattr(seniority, "get_principal_group", boom)
    mol = Chem.MolFromSmiles("CC1=C(C=C(C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N2C=NC(=C2)C)C=C1)"
                             "NC1=NC=CC(=N1)C=1C=NC=CC1")
    assert phane_may_be_pin(mol) is True


def test_the_answer_is_memoised_per_molecule_object():
    mol = Chem.MolFromSmiles("c1ccc(cc1)Oc1ccc(cc1)Oc1ccc(cc1)Oc1ccccc1")
    assert phane_may_be_pin(mol) is True
    assert linear_phane_screen._MEMO_MOL is mol and linear_phane_screen._MEMO == {"v": True}
