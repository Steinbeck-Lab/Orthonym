"""DT4 F2b: the '-yl' prefix of a fused ring system that carries its own prefixes and stereodescriptors
(``bridged_fused_pin.build_fused_decorated_substituent``, `internal notes`).

Governing text (the Blue Book): ``## **** GENERAL METHODOLOGY FOR NAMING SUBSTITUENT GROUPS`` (:15800);
``### **** Five-membered ring requirement`` (:23708), "Fusion nomenclature gives preferred IUPAC names only
to compounds having at least two rings of at least five or more members" (:23710), so such a system keeps its fusion
name as a substituent too; ``****`` NAMING OF STEREOISOMERS (:44639), "When they relate to substituent groups,
they are cited at the front of the corresponding prefix" (:44643; '[(1R)-1-chloropropyl]benzene (PIN)':44668);
``### **** NUMBERING`` (f) (:3301) "detachable alphabetized prefixes, all considered together in a series of
increasing numerical order". cyclopenta[a]phenanthrene keeps its traditional numbering,:12493).

Every name is read back by OPSIN 2.9.0 to the input's full standard InChIKey (``name_is_rt_exact``), outside the
engine.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.bridged_fused_pin import build_fused_decorated_substituent
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _pin(smiles):
    with jvm_slots(1, purpose="test-f2b"):
        return Orthonym(style="pin").name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", [
    # a prefix on the ring system (3-hydroxy)
    ("CC(CCC(=O)O)C1CCC2C1CCC1C2CCC2CC(O)CCC12",
     "4-(3-hydroxyhexadecahydro-1H-cyclopenta[a]phenanthren-17-yl)pentanoic acid"),
    # methyl groups on fusion atoms (10,13) and an oxo prefix
    ("CC(CCC(=O)O)C1CCC2C1(C)CCC1C2CCC2CC(=O)CCC12C",
     "4-(10,13-dimethyl-3-oxohexadecahydro-1H-cyclopenta[a]phenanthren-17-yl)pentanoic acid"),
    # not a steroid: a methyl on a fusion atom of phenanthrene
    ("OC(=O)CCC1CCC2C(C1)CCC1CCCCC12C", "3-(4b-methyltetradecahydrophenanthren-2-yl)propanoic acid"),
    # dev2000 rows: the ring centres' descriptors at the front of the prefix, the chain's at the front of the name
    ("C[C@H](CCC[C@@H](C)[C@H]1CC[C@H]2C3C(=O)C=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)C(=O)O",
     "(2R,6R)-6-[(3S,9S,10R,13R,14S,17R)-3-hydroxy-10,13-dimethyl-7-oxo-2,3,4,7,8,9,10,11,12,13,14,15,16,17-"
     "tetradecahydro-1H-cyclopenta[a]phenanthren-17-yl]-2-methylheptanoic acid"),
    ("C[C@H](CC(=O)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)[C@@H](O)[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
     "(3R)-3-[(3R,5R,6S,7S,8S,9S,10R,13R,14S,17R)-3,6,7-trihydroxy-10,13-dimethylhexadecahydro-1H-"
     "cyclopenta[a]phenanthren-17-yl]butanoic acid"),
])
def test_a_decorated_fused_ring_system_is_a_substituent_prefix(smiles, name):
    row = _pin(smiles)
    assert row["name"] == name and row["tier"] in ("pin_verified", "pin_unverified"), row
    assert name_is_rt_exact(name, smiles)


def _attach_and_outside(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    attach = next(a for a in ring if any(not n.IsInRing() and n.GetDegree() > 1
                                         for n in mol.GetAtomWithIdx(a).GetNeighbors()))
    outside = {a.GetIdx() for a in mol.GetAtoms() if not a.IsInRing()} - {
        n.GetIdx() for r in ring for n in mol.GetAtomWithIdx(r).GetNeighbors() if n.GetDegree() == 1}
    return mol, attach, outside


def test_a_stereocentre_on_the_substituent_side_outside_the_ring_is_cited_by_its_prefix():
    # NPL 2026-11-01 (was: the builder declines): the (1R)-1-hydroxyethyl on C-3 is a stereogenic unit of a prefix of
    # the ring system; its namer cites the descriptor at the front of that prefix, the Blue Book,
    # "When they relate to substituent groups, they are cited at the front of the corresponding prefix") and the
    # builder's name is read back with the whole molecule. The same branch without the stereocentre keeps its name
    # (atoms 0-6: the pentanoic acid side). The whole name reads back by OPSIN 2.9.0 to the input's full InChIKey.
    with_stereo_smiles = "CC(CCC(=O)O)C1CCC2C1CCC1C2CCC2CC([C@@H](C)O)CCC12"
    with_stereo = Chem.MolFromSmiles(with_stereo_smiles)
    without = Chem.MolFromSmiles("CC(CCC(=O)O)C1CCC2C1CCC1C2CCC2CC(C(C)O)CCC12")
    outside = set(range(7))
    assert build_fused_decorated_substituent(with_stereo, 7, outside) == (
        "3-[(1R)-1-hydroxyethyl]hexadecahydro-1H-cyclopenta[a]phenanthren-17-yl")
    assert build_fused_decorated_substituent(without, 7, outside) == (
        "3-(1-hydroxyethyl)hexadecahydro-1H-cyclopenta[a]phenanthren-17-yl")
    assert name_is_rt_exact("4-{3-[(1R)-1-hydroxyethyl]hexadecahydro-1H-cyclopenta[a]phenanthren-17-yl}pentanoic acid",
                            with_stereo_smiles)


def test_every_prefix_counts_for_p14_4_f():
    # {4,4,5} is lower than {4,5,5}: the gem pair cites its locant twice (a set of atoms tied the two numberings
    # and (g) then gave the first-cited hydroxy the 4)
    smiles = "OC(=O)CCC12C(C)(C)CCCC1CCCC2O"
    row = _pin(smiles)
    assert row["name"] == "3-(5-hydroxy-4,4-dimethyloctahydronaphthalen-4a(2H)-yl)propanoic acid", row
    assert name_is_rt_exact(row["name"], smiles)


def test_a_double_bond_to_the_parent_side_is_no_yl_prefix():
    # '-ylidene', not '-yl': the builder declines (it used to give '6-hydroxydecahydronaphthalen-2-yl')
    mol = Chem.MolFromSmiles("OC(=O)CC=C1CCC2CC(O)CCC2C1")
    assert build_fused_decorated_substituent(mol, 5, {0, 1, 2, 3, 4}) is None


def test_a_ring_system_bonded_twice_to_the_parent_side_declines():
    # the methoxy oxygen on C-6 is passed as parent side too: two bonds into the system, not one '-yl'
    mol = Chem.MolFromSmiles("OC(=O)CC1CCC2CC(OC)CCC2C1")
    attach, ether_o = 4, 10
    assert mol.GetAtomWithIdx(ether_o).GetSymbol() == "O"
    assert build_fused_decorated_substituent(mol, attach, {0, 1, 2, 3}) is not None
    assert build_fused_decorated_substituent(mol, attach, {0, 1, 2, 3, ether_o, 11}) is None


def test_a_single_ring_declines():
    mono = Chem.MolFromSmiles("OC(=O)CCC1CCC(O)CC1")
    assert build_fused_decorated_substituent(mono, 5, {0, 1, 2, 3, 4}) is None


@pytest.mark.parametrize("smiles,name", [
    ("OCC(C)[C@H]1CC[C@H]2[C@@H](CC[C@H]3C(C)(C)CCC[C@]23C)C1",
     "2-[(2S,4aS,4bR,8aS,10aS)-4b,8,8-trimethyltetradecahydrophenanthren-2-yl]propan-1-ol"),
    ("CC(CCC(=O)O)C1CCC2C1(C)CCC1C2CCC2CC(=O)CCC12C",
     "4-(10,13-dimethyl-3-oxohexadecahydro-1H-cyclopenta[a]phenanthren-17-yl)pentanoic acid"),
    # a two-ring system of 8 atoms: no longer numbered as one ring at the PIN tier ('4-methyl...')
    ("OC(=O)CC1CCC2CC(C)CC12", "(5-methyloctahydropentalen-1-yl)acetic acid"),
    ("OC(=O)CC1CC2CC(C)CC2C1", "(5-methyloctahydropentalen-2-yl)acetic acid"),
])
def test_the_default_tier_gives_the_fusion_prefix_not_the_von_baeyer_form(smiles, name):
    # the best-effort switch also offers the von Baeyer composer for the ring system ('1,11,11-trimethyltricyclo
    # [8.4.0.0^2,7]tetradecan-5-yl'); the fusion prefix of the PIN tier comes first there too
    with jvm_slots(1, purpose="test-f2b"):
        row = Orthonym(style="pin", general_fallback=True, general_fallback_unverified=True).name_tiered(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified", row
    assert _pin(smiles)["name"] == name
    assert name_is_rt_exact(name, smiles)
