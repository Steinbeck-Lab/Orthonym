"""Unnamed-breadth lane S1a (plan 2026-10-15-unnamed-breadth): PubChem rows that main left unnamed
because the floor's name has no E/Z descriptor for the double bond between a ring and a ring
'-ylidene' substituent; best-effort must now name each with a full read-back (full InChIKey +
radical identity + protonation site, a fresh OPSIN call).

Governing rule, 'Specification by stereodescriptors E and Z' of
'Specification of exo-cyclic double bonds' (the Blue Book): "The stereodescriptors 'E' and
'Z' are used to specify the configuration of an exo-cyclic double bond." (:48273); method (a) "the
double bond is considered as an integral part of the parent structure; the stereodescriptor is
placed at the front of the substitutive name, preceded by the locant indicating its point of
attachment to the parent structure" (:48276), method (b) "the stereodescriptor is cited at the
front of the name of the substituent group" (:48277); "Method (a) generates preferred IUPAC
names." (:48279).

PIN track: none (best-effort only). Whether these rows have a Blue Book PIN is not decided by the
read-back, and the lane does not widen the default tier: no row may claim a PIN here.
"""
import pytest
from rdkit import Chem

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

# (cid, smiles): every row is a real PubChem CID of the unnamed panel; main gives it no name at
# best-effort. The first eleven are the strategy study's rows, the others the smallest of the
# panel's rows of the same kind (a ring=ring double bond with a defined E/Z; some also carry a
# C=N 'imino' bond, ring stereocentres or a second ring=ring bond).
ROWS = [
    ("18842095", "CC(=O)N1C2=CC=CC=C2/C(=C/3\\C(=O)N(/C(=N\\C4=CC=CC=N4)/S3)C5=CC=CC=C5)/C1=O"),
    ("58454846", "CCN1C(=O)/C(=C\\2/N(C(=O)/C(=C\\C3=CC4=C(C=C3)N(C5C4CCC5)C6=CC=C(C=C6)CO)/S2)CC(=O)O)/SC1=S"),
    ("140148282", "COC\\1C(=CC(=C/C1=C/2\\N(C=CS2)C[C@H]3CCCO3)Cl)C(=O)N"),
    ("172247065", "CC/C=C(/NC\\1=C(/C(=C\\2/C=CC(=C3C2=C(C(=N)S3)C#N)F)/C=C/C1=C(/C)\\N(C)C(CCC(C)CC)[C@H](C)CC)F)\\OC"),
    ("177991378", "C[C@@H]1CN2C(=CC(=N2)C3C(/C(=C/4\\C(=C(\\C5=C(C=C(C(=C5)C=N)NC)OC/C=C/COCC(C)O)/N)\\C=CS4)/C6=C3C=C(C=C6F)F)C)CN1C(=O)C=C"),
    ("164946607", "CCC1=C([C@H](NC1=O)CC2=C(C(=C(N2)CC3=C(C(=C)C(N3)C[C@H]4/C(=C\\5/CS5)/C(C(=O)N4)C)CCC(=O)O)CCC(=O)O)C)C"),
    ("123390446", "CCN1C(=O)/C(=C\\2/N(C(=O)/C(=C\\3/N(C(=O)/C(=C/C4=CC5=C(C=C4)N(C6C5CCC6)C7=CC=C(C=C7)SC)/S3)CC(=O)O)/S2)CC8=CC=CC=C8)/SC1=S"),
    ("164946525", "C[C@@H]1[C@H](/C(=C/C2=C(C(=C(N2)/C=C\\3/C(=C(C(N3)C[C@@H]4C(/C(=C\\5/CS5)/C(=O)N4)C)C)CCC(=O)O)CCC(=O)O)C)/NC1=O)[C@@H]6CS6"),
    ("89659580", "CCCCCCCCN1C(=O)/C(=C\\2/N(C(=O)/C(=C\\C3=CN=C(S3)C4=CC=C(C5=NSN=C45)C6=NC=C(S6)C7=CC=C(C=C7)N(C8=CC=CC=C8)C9=CC=CC=C9)/S2)NOC=O)/SC1=O"),
    ("102088994", "CCN1C(=O)/C(=C\\2/N(C(=O)/C(=C\\C3=CC4=C(C=C3)N(C5C4CCC5)C6=CC=C(C=C6)C=C(C7=CC=CC=C7)C8=CC=CC=C8)/S2)CCCC(=O)O)/SC1=S"),
    ("11698035", "COC1=CC=C(C=C1)CC2=C(C=CC(=C2)[C@H]3[C@@H]([C@H]([C@@H](/C(=C/4\\CCOC4=O)/O3)O)O)O)Cl"),
    ("13040099", "CC(C)(C)C1=C/C(=C/2\\C=CC(=C2)C(C)(C)C)/C=C1"),
    ("98569879", "[C@@H]1(/C(=C/2\\[C@@H](C(=O)C(=C2Cl)Cl)Cl)/C(=C(C1=O)Cl)Cl)Cl"),
    ("142802876", "C\\1C=C(C=C/C1=C\\2/[C@H](C(=O)N2)CCC(=O)C3=CC=C(C=C3)F)O"),
    ("6206837", "CN1C2=CC=CC=C2/C(=C/3\\C(=O)N(/C(=N\\C4=NC=CS4)/S3)CC=C)/C1=O"),
    ("16438916", "CCN\\1C(=O)/C(=C\\2/N(C3=CC=CC=C3S2)C)/S/C1=N/S(=O)(=O)C4=CC=CS4"),
    ("135585542", "CN1C2=CC=CC=C2/C(=C/3\\C(=O)N/C(=N/C4=NN=C(O4)C5=CC=CC=C5)/S3)/C1=O"),
    ("3098106", "CCN\\1C2=CC=CC=C2S/C1=C\\3/C(=O)N(/C(=C\\4/C(=NNC4=O)C)/S3)CC"),
    ("143616614", "CC1=CC=C(C=C1)C2=CC(=C)/C(=C/3\\C(=C)C=C(C(=O)O3)C)/C(=C2F)F"),
    ("20615556", "CCN(CC)C1=CC2=C(C=C1)C(=C/C(=C/3\\C(=O)C(C(=S)N(C3=O)C)C)/O2)C"),
    ("137428244", "C/C/1=C/C/C(=C\\2/C=CC(=O)CC(=C2)C)/C=C\\C(=O)C1"),
    ("168306046", "C1C/C(=C\\C2=CC=CC=C2)/C(=C\\3/CCC4=CC=CC=C4S3)/C1"),
    ("162794477", "CC[C@@H]([C@@H]1CCC[C@@]23N1C[C@@H](C=C2)C=C4[C@@H]3[C@@H](/C(=C/5\\C(=C(C(=O)O5)CO)OC)/O4)CO)O"),
    ("163003571", "CCOC(=O)C1=C(/C(=C/2\\C[C@@H](C(=O)[C@H](C2)C3=CC4=C(C=C3)NC=C4)CO)/C5=C(C6=C(C(=C5O1)CO)O[C@@H](C6)C(C)(C)O)OC7CCCCC7)C=O"),
]

PIN_ROWS = set()  # none: best-effort only (a row is added only with its cited Blue Book PIN)


@pytest.mark.opsin_gate  # what SHIPS: the engine's own validity gate on, as in production
@pytest.mark.parametrize("cid,smiles", ROWS, ids=[c for c, _ in ROWS])
def test_best_effort_names_row_with_full_read_back(cid, smiles):
    res = name_best_effort(smiles)
    assert_full_rt(res.get("name"), smiles, what=f"CID {cid}: ")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("cid,smiles", ROWS, ids=[c for c, _ in ROWS])
def test_default_tier_never_claims_a_pin_without_a_citation(cid, smiles):
    res = default_tier_row(smiles)
    assert not res.get("is_pin") or cid in PIN_ROWS, f"CID {cid}: PIN claimed without a cited rule"


# ---------------------------------------------------------------------------------------------
# Rows the lane must leave as they are (named rows keep their name: the floor's rung 1d is offered
# only where nothing named the row before, and only for a bond the InChIKey checks).
# ---------------------------------------------------------------------------------------------

# 'General criteria (other than phanes)' (the Blue Book-19413): the senior ring or
# ring system "(a) is a heterocycle; (b) has at least one nitrogen atom". These names (verified by a
# fresh read-back) keep the senior parent; the floor's name of the same molecule has a benzene
# (CID 21937604) or a pyran (the XCHEBI rows) parent. The XCHEBI rows hold an amide, class 11 of
# 'SENIORITY ORDER FOR CLASSES' (the Blue Book), senior to the ketones (class 16) and the
# nitrogen heterocycles (class 21), so their parent is the acetamide,:18875: "the maximum
# number of substituents corresponding to the principal characteristic group (suffix)"), with the
# pyrrolidine ring as its substituent; the earlier '2-(2-amino-2-oxoethyl)-...-3,5-dioxopyrrolidine'
# cited the amide as a prefix (lane TI2: the 3,5-dioxopyrrolidin-2-yl prefix is now the strict one).
SENIOR_PARENT_ROWS = [
    ("CID21937604", "CCNC1=C(C=C(C=C1)C#N)N=C2N(C(=O)/C(=C/3\\N(C(=C(S3)C)CC4=CC=CC=C4)C)/S2)CC5=CC=CC=C5",
     "(2Z)-4-benzyl-2-(3-benzyl-2-{[5-cyano-2-(ethylamino)phenyl]imino}-4-oxo-1,3-thiazolidin-5-ylidene)"
     "-3,5-dimethyl-2,3-dihydro-1,3-thiazole"),
    ("XCHEBIb3", "CC(C)C1O/C(=C2\\C(=O)NC(CC(N)=O)C2=O)C=CC1C",
     "(2Z)-2-{4-[5-methyl-6-(propan-2-yl)-5,6-dihydro-2H-pyran-2-ylidene]-3,5-dioxopyrrolidin-2-yl}"
     "acetamide"),
    ("XCHEBIb9", "CC(C)C1O/C(=C2/C(=O)NC(CC(N)=O)C2=O)C=CC1C",
     "(2E)-2-{4-[5-methyl-6-(propan-2-yl)-5,6-dihydro-2H-pyran-2-ylidene]-3,5-dioxopyrrolidin-2-yl}"
     "acetamide"),
]

# A squaraine dye: its InChIKey ends -UHFFFAOYSA-O, no stereo layer, so the key cannot tell the E and
# the Z bond between its rings apart and a descriptor for it would be unchecked. It keeps the name
# without one.
KEY_BLIND_ROW = (
    "CID91868672",
    "CCCCCCCCN(CCCCCCCC)C1=CC(=C(C=C1)C2=C(/C(=C\\3/C=CC(=[N+](CCCCCCCC)CCCCCCCC)C=C3O)/C2=O)O)O",
    "4-(dioctylamino)-2-hydroxy-1-{2-hydroxy-3-[2-hydroxy-4-(dioctylazaniumylidene)cyclohexa-2,5-dien-"
    "1-ylidene]-4-oxocyclobut-1-en-1-yl}benzene")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("cid,smiles,name", SENIOR_PARENT_ROWS, ids=[r[0] for r in SENIOR_PARENT_ROWS])
def test_named_row_keeps_its_senior_parent_name(cid, smiles, name):
    res = name_best_effort(smiles)
    assert res.get("name") == name, f"CID {cid}: {res.get('name')!r}"
    assert_full_rt(name, smiles, what=f"CID {cid}: ")


@pytest.mark.opsin_gate
def test_key_blind_ring_ylidene_bond_gets_no_descriptor():
    cid, smiles, name = KEY_BLIND_ROW
    res = name_best_effort(smiles)
    assert res.get("name") == name, f"CID {cid}: {res.get('name')!r}"
    assert_full_rt(name, smiles, what=f"CID {cid}: ")


# ---------------------------------------------------------------------------------------------
# The helper's scopes, without a read-back: which scope cites the one bond (never both), and which
# bonds the key can check.
# ---------------------------------------------------------------------------------------------

def _indolinone_thiazolidinone():
    """3-(2,4-dioxo-1,3-thiazolidin-5-ylidene)-indolin-2-one with a defined Z bond; returns
    (mol, the ring=ring bond, the indolinone atom of the bond, the thiazolidinone atom)."""
    from orthonym.rules.stereochemistry import ring_ylidene_bonds
    mol = Chem.MolFromSmiles("O=C1Nc2ccccc2/C1=C1/SC(=O)NC1=O")
    (bond,) = ring_ylidene_bonds(mol)
    a, b = bond.GetBeginAtom(), bond.GetEndAtom()
    in_indolinone = any(n.GetSymbol() == "N" for n in
                        (x for r in mol.GetRingInfo().AtomRings() if a.GetIdx() in r
                         for x in (mol.GetAtomWithIdx(i) for i in r)))
    return (mol, bond, a, b) if in_indolinone else (mol, bond, b, a)


def test_ring_ylidene_bond_is_cited_by_exactly_one_scope():
    from orthonym.assembly.ring_ylidene_stereo import (
        BRANCH, HOST, RingYlidene, ring_ylidene_descriptors)
    mol, bond, host_end, branch_end = _indolinone_thiazolidinone()
    host_map, branch_map = {host_end.GetIdx(): 3}, {branch_end.GetIdx(): 5}
    code = bond.GetProp("_CIPCode")
    host, branch = (RingYlidene(m, frozenset({bond.GetIdx()})) for m in (HOST, BRANCH))
    # method (a): the host cites it with its own locant, the branch (its attachment bond) does not
    assert ring_ylidene_descriptors(mol, host_map, host) == [(3, code)]
    assert ring_ylidene_descriptors(mol, branch_map, host, attach_bond=bond) == []
    # method (b): the branch cites it with its ylidene locant, the host does not
    assert ring_ylidene_descriptors(mol, host_map, branch) == []
    assert ring_ylidene_descriptors(mol, branch_map, branch, attach_bond=bond) == [(5, code)]
    # a scope that holds both ends (a bond of its own spine) or neither leaves the bond alone
    both = {host_end.GetIdx(): 3, branch_end.GetIdx(): 5}
    assert ring_ylidene_descriptors(mol, both, host) == []
    assert ring_ylidene_descriptors(mol, {}, branch) == []
    # a bond the spec does not name is never cited
    assert ring_ylidene_descriptors(mol, host_map, RingYlidene(HOST, frozenset())) == []


def test_molecule_without_a_ring_ring_bond_is_left_alone():
    from orthonym.rules.stereochemistry import ring_ylidene_bonds
    # an ethylidene on a ring (the acyclic end: cited by the ordinary exocyclic licence)
    assert ring_ylidene_bonds(Chem.MolFromSmiles("C/C=C1/CCCCC1C")) == []
    # a ring double bond and an acyclic one
    assert ring_ylidene_bonds(Chem.MolFromSmiles("C/C=C/C1=CC=CC=C1")) == []


def test_only_bonds_the_inchikey_encodes_count_as_key_visible():
    from orthonym.assembly.ring_ylidene_stereo import key_visible_ring_ylidene_bonds
    from orthonym.rules.stereochemistry import ring_ylidene_bonds
    mol, bond, _, _ = _indolinone_thiazolidinone()
    assert [b.GetIdx() for b in key_visible_ring_ylidene_bonds(mol)] == [bond.GetIdx()]
    blind = Chem.MolFromSmiles(KEY_BLIND_ROW[1])
    assert len(ring_ylidene_bonds(blind)) == 1  # a ring=ring bond with a defined E/Z...
    assert key_visible_ring_ylidene_bonds(blind) == []  #... that the key does not encode
