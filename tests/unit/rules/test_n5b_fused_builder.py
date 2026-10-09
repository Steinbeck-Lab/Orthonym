"""The fusion name of an all-carbon ortho- or ortho- and peri-fused ring system
(``bridged_fused_pin.build_fused``), in any hydrogenation state (lane N5b step 1).

 "Five-membered ring requirement" (the Blue Book-23710): "Fusion nomenclature
gives preferred IUPAC names only to compounds having at least two rings of at least five or more
members. [...] When fusion names are not allowed, unsaturated von Baeyer ring system names are
preferred IUPAC names". The numbering is "NUMBERING" (:3219): (b) indicated hydrogen
(:3246), (c) principal characteristic groups (:3256), (e) hydro prefixes (:3288), (f) detachable
prefixes (:3301), (g) the prefix cited first (:3307). Hydro prefixes stand directly before the
parent hydride,:16872); total hydrogenation omits their locants,
:17024-:17026); indicated hydrogen accommodates a suffix,:24766; added hydrogen
,:24689). Every name is read back by a fresh OPSIN call to the input's full InChIKey."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import build_fused, build_fused_substituent, selection
from tests.support.rt_assert import assert_full_rt

NAMED = [
    # hydrindane and decalin family (the von Baeyer names these had: bicyclo[4.3.0], bicyclo[4.4.0])
    ("OC1CCC2CCCC2C1", "octahydro-1H-inden-5-ol"),
    ("OC1CCCC2CCCC12", "octahydro-1H-inden-4-ol"),
    ("O=C1CCC2CCCC2C1", "octahydro-5H-inden-5-one"),
    ("O=C1CCC2CCCCC2C1", "octahydronaphthalen-2(1H)-one"),
    ("O=C1CCCC2CCCCC12", "octahydronaphthalen-1(2H)-one"),
    ("CC1CCC2CCCC2C1", "5-methyloctahydro-1H-indene"),
    ("OC1CCC2CCCC2(C1)C", "3a-methyloctahydro-1H-inden-5-ol"),
    ("CC12CCCCC1CCCC2", "4a-methyldecahydronaphthalene"),
    ("OC12CCCCC1CCCC2", "octahydronaphthalen-4a(2H)-ol"),
    ("OC1CC2CCCC2C1", "octahydropentalen-2-ol"),
    ("OC1CCC2CC=CC2C1", "3a,4,5,6,7,7a-hexahydro-1H-inden-5-ol"),
    ("BrC12C=CC=CC1C=CC=C2", "4a-bromo-4a,8a-dihydronaphthalene"),   # the Blue Book
    ("OC(=O)C1CCC2CCCC2C1", "octahydro-1H-indene-5-carboxylic acid"),
    ("NC1CCC2CCCC2C1", "octahydro-1H-inden-5-amine"),
    ("N#CC1CCC2CCCC2C1", "octahydro-1H-indene-5-carbonitrile"),
    ("OC1CCC2CCCC2C1O", "octahydro-1H-indene-4,5-diol"),
    ("O=C1CCC2CCCC2C1=O", "hexahydro-1H-indene-4,5-dione"),
    ("O[C@H]1CC[C@H]2CCC[C@@H]2C1", "(3aR,5S,7aR)-octahydro-1H-inden-5-ol"),
    # three and four rings, fused parents the tables or the derived table name
    ("c1ccc2c(c1)CC1CCCC21", "1,2,3,3a,8,8a-hexahydrocyclopenta[a]indene"),
    ("C1CCC2C(C1)CC1CCCCC12", "dodecahydro-1H-fluorene"),
    ("OC1CCC2C(C1)CC1CCCCC12", "dodecahydro-1H-fluoren-2-ol"),
    ("C1CCC2C(C1)CCC1CCCC12", "dodecahydro-1H-cyclopenta[a]naphthalene"),
    ("C1CCC2C(C1)CCC1C2CCC2CCCC12", "hexadecahydro-1H-cyclopenta[a]phenanthrene"),
    ("CC12CCCCC1CCC1C2CCC2(C)C(O)CCC12",
     "10,13-dimethylhexadecahydro-1H-cyclopenta[a]phenanthren-17-ol"),
    ("OC1CCC2C(C1)CCC1C2CCC2CCCC12", "hexadecahydro-1H-cyclopenta[a]phenanthren-3-ol"),
    # peri-fused, and a seven-membered ring
    ("C1CCC2CCCC3CCCC1C23", "dodecahydro-1H-phenalene"),
    ("OC1CCC2CCCC3CCCC1C23", "dodecahydro-1H-phenalen-1-ol"),
    ("C1CCC2CCCCCC2C1", "decahydro-1H-benzo[7]annulene"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", NAMED)
def test_the_builder_names_the_system_and_opsin_reads_it_back(smiles, expected):
    got = build_fused(Chem.MolFromSmiles(smiles))
    assert got is not None and got[0] == expected, got
    assert got[3] is True and set(got[2]) == set(got[1])      # every ring atom is numbered
    assert_full_rt(expected, smiles)


# names that the engine builds today; the builder spells the same bytes
UNCHANGED = [
    ("OC1CCC2CCCCC2C1", "decahydronaphthalen-2-ol"),
    ("CC1CCC2CCCCC2C1", "2-methyldecahydronaphthalene"),
    ("C1CCC2CCCC2C1", "octahydro-1H-indene"),
    ("Oc1ccc2CCCCc2c1", "5,6,7,8-tetrahydronaphthalen-2-ol"),
    ("O=C1CCCc2ccccc12", "3,4-dihydronaphthalen-1(2H)-one"),
    ("C1=CC2CCCCC2C1", "3a,4,5,6,7,7a-hexahydro-1H-indene"),
]


@pytest.mark.parametrize("smiles,expected", UNCHANGED)
def test_the_builder_spells_the_names_the_engine_already_builds(smiles, expected):
    assert build_fused(Chem.MolFromSmiles(smiles))[0] == expected


DECLINED = [
    "C1CCC2CCC2C1",                    # bicyclo[4.2.0]octane: one ring of five or more,
    "C1CC2CCC1C2",                     # bridged, not fused
    "OC1CCC2(CC1)CCCC2",               # spiro
    "C[n+]1ccc2ccccc2c1",              # a ring cation: declined with every ring atom of nonstandard
                                       # bonding or charge, (the Blue Book)
    "C1CCCC2CCCCC2CC1",                # a ring of ten members: left to the von Baeyer name
    "C1CC[C@H]2CCCC[C@@H]2C1",         # trans-decalin: the junction descriptors are r, s
]


@pytest.mark.parametrize("smiles", DECLINED)
def test_the_builder_declines_what_it_does_not_name(smiles):
    assert build_fused(Chem.MolFromSmiles(smiles)) is None


def test_a_skeleton_no_parent_table_names_declines(monkeypatch):
    monkeypatch.setattr(selection, "_parent_name", lambda mol, residual: None)
    assert build_fused(Chem.MolFromSmiles("OC1CCC2CCCC2C1")) is None


def test_a_fused_ring_substituent_is_a_yl_prefix():
    mol = Chem.MolFromSmiles("C1CCC2CCCC2C1")
    assert build_fused_substituent(mol, 2) == "octahydro-1H-inden-4-yl"
    name = build_fused_substituent(Chem.MolFromSmiles("C1CCC2CCCC2N1"), 2)
    assert name == "octahydro-1H-cyclopenta[b]pyridin-4-yl"
    assert build_fused_substituent(Chem.MolFromSmiles("C[n+]1ccc2ccccc2c1"), 2) is None
