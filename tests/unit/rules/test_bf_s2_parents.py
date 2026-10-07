"""The fused parent of a bridged fused system: its PIN name and every IUPAC numbering
 :14241,:14245,:14177,:12493,
 :12501), from the engine's parent tables (``bridged_fused_pin.parents``)."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin.parents import (
    PIN_PARENT_NAMES, fused_parent, is_pin_parent_name, normalize_locant)


def _parent(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol, fused_parent(mol, set(range(mol.GetNumAtoms())))


@pytest.mark.parametrize("smiles,name,n_numberings", [
    ("c1ccc2ccccc2c1", "naphthalene", 4),
    ("c1ccc2cc3ccccc3cc2c1", "anthracene", 4),
    ("c1ccc2c(c1)ccc1ccccc12", "phenanthrene", 2),
    ("c1ccc2cccc-2cc1", "azulene", 2),
    ("C1=Cc2ccccc2C1", "indene", 2),
    ("c1ccc2c(c1)Cc1ccccc1-2", "fluorene", 2),
    ("c1ccc2[nH]ccc2c1", "indole", 1),
    ("c1ccc2c[nH]cc2c1", "isoindole", 2),
    ("c1ccc2[nH]ncc2c1", "indazole", 1),
    ("c1ccc2ncccc2c1", "quinoline", 1),
    ("C1=Cc2ccccc2OC1", "1-benzopyran", 1),
    ("c1ccc2cocc2c1", "2-benzofuran", 2),
    ("c1ccc2c(c1)Cc1ccccc1S2", "thioxanthene", 2),
    ("c1ccc2Oc3ccccc3Oc2c1", "oxanthrene", 4),
    # the catalogue map agrees with the fusion numbering now (S 5 and 10, OPSIN 2.9.0)
    ("c1ccc2Sc3ccccc3Sc2c1", "thianthrene", 4),
    ("C1=CCc2ccccc2C=C1", "benzo[7]annulene", 2),
    ("c1ccc2ccccccc2c1", "benzo[8]annulene", 2),
    ("c1ccc2nc3ccccc3cc2c1", "acridine", 2),
    ("c1ccc2ocnc2c1", "1,3-benzoxazole", 1),
    ("C1=CC=Cc2ccccc2O1", "1-benzoxepine", 1),
    ("c1ccc2cc3cc4ccccc4cc3cc2c1", "tetracene", 4),
])
def test_a_pin_parent_is_named_with_all_its_numberings(smiles, name, n_numberings):
    _, fp = _parent(smiles)
    assert fp is not None and fp.name == name, fp
    assert len(fp.numberings) == n_numberings


def test_locants_follow_p25_3_3_1_1():
    #:12501 "including fusion heteroatoms but not fusion carbon atoms. Each fusion carbon
    # atom is given the same number as the immediately preceding nonfusion skeletal atom,
    # modified by a Roman letter": integers for nonfusion atoms and heteroatoms, '3a' for
    # fusion carbons
    mol, fp = _parent("c1ccc2[nH]ccc2c1")
    for numbering in fp.numberings:
        for atom, loc in numbering.items():
            fusion = mol.GetAtomWithIdx(atom).GetDegree() == 3
            assert isinstance(loc, str) == fusion, (atom, loc)
    nitrogen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    assert fp.numberings[0][nitrogen] == 1


def test_thioxanthene_keeps_its_traditional_numbering():
    # (:12493): "xanthene and its chalcogen analogues... traditional numberings
    # are retained" (S is 10, CH2 is 9)
    mol, fp = _parent("c1ccc2c(c1)Cc1ccccc1S2")
    sulfur = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S")
    assert {n[sulfur] for n in fp.numberings} == {10}


def test_a_lettered_fusion_heteroatom_map_is_discarded():
    # numbers fusion heteroatoms (4H-quinolizine N is 5); the catalogue
    # map used to label the N '4a' (corrected), and a lettered fusion-heteroatom map
    # is still discarded in favour of the fusion numbering
    mol, fp = _parent("C1C=CC=C2C=CC=CN12")
    nitrogen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    assert fp.name == "quinolizine"
    assert {n[nitrogen] for n in fp.numberings} == {5}


@pytest.mark.parametrize("smiles,name", [
    # the catalogue's '9H-beta-carboline' is not a PIN (no Blue Book line) and its
    # 'imidazo[2,1-b]thiazole' spelled the thiazole without its locants:11982);
    # since S2c-1 the two-component producer names both skeletons, numbered by OPSIN
    ("c1ccc2c(c1)[nH]c1cnccc12", "pyrido[3,4-b]indole"),
    ("c1cn2ccsc2n1", "imidazo[2,1-b][1,3]thiazole"),
])
def test_parents_the_tables_cannot_certify_get_the_producers_name(smiles, name):
    _, fp = _parent(smiles)
    assert fp is not None and fp.name == name and fp.source == "hetero_fusion_name+opsin", fp


def test_benzo_names_need_heteroatom_locants():
    # (:11815): "for preferred IUPAC names locants must be cited"
    assert is_pin_parent_name("1,3-benzoxazole") and is_pin_parent_name("3-benzoxepine")
    assert not is_pin_parent_name("benzoxazole") and not is_pin_parent_name("1,2-benzisoxazole")
    assert "naphthacene" not in PIN_PARENT_NAMES and "tetracene" in PIN_PARENT_NAMES


def test_the_table_match_is_an_isomorphism_not_a_substructure():
    # one layer of the exact-skeleton rule: _iso maps a table molecule onto the residual's
    # skeleton only when the two have the same atoms and bonds; naphthalene is a
    # substructure of the anthracene skeleton, not that skeleton
    from orthonym.rules.bridged_fused_pin.parents import _iso, _key_of
    anthracene = Chem.MolFromSmiles("c1ccc2cc3ccccc3cc2c1")
    skeleton = Chem.MolFromSmiles(_key_of(anthracene))
    assert _iso(Chem.MolFromSmiles("c1ccc2ccccc2c1"), skeleton) is None
    assert _iso(anthracene, skeleton) is not None


def test_normalize_locant():
    assert [normalize_locant(x) for x in (4, (4, "a"), "4a", "5")] == [4, "4a", "4a", 5]


def test_the_parent_inside_a_bridged_system_is_numbered_on_the_residual():
    # 4,7-methanoazulene: the azulene residual has two numberings
    mol = Chem.MolFromSmiles("C1=CC2=CC3=CC=C(C3)C2=C1")
    bridge = next(a.GetIdx() for a in mol.GetAtoms()
                  if a.GetTotalNumHs() == 2 and a.GetDegree() == 2)
    fp = fused_parent(mol, set(range(mol.GetNumAtoms())) - {bridge})
    assert fp.name == "azulene" and len(fp.numberings) == 2
