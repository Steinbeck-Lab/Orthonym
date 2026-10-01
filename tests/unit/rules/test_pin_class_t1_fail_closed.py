"""PIN class program, Task 1: two producers fail closed on the wrong-molecule candidates they built.

0-wrong (the program's Global Constraints): the skeleton a name describes must be the input's.
Each producer below returned a name that OPSIN reads to a DIFFERENT molecule; only the round
trip held it back. The fix is at the producer, so the candidate is never built.

(1) ``rules/ions.py``: the ligand walk ``_collect_substituent_atoms`` of the mononuclear
    centre emitters (``_emit_group13_uide``, ``emit_halogen_onium`` and their siblings)
    ran round a ring through the centre and came back to it, so one ring was named as two
    chain ligands: 'dimethyldipentylboranuide' for the borinan-1-uide ``C[B-]1(C)CCCCC1``,
    'dipentyliodanium' for ``C1CC[I+]CC1``. A ligand is a branch of the centre; a walk that
    reaches the centre again through another atom is a ring through the centre, and the
    walk now declines it (None), as it already declines a walk that re-enters a ring.

(2) ``rules/skeletal_replacement.py:_try_cyclic_replacement_name``: the cyclic 'a' builder
    spells every ring atom at its standard bonding number and every ring bond as single or
    double. A neutral ring atom with another bonding number needs the λ-convention,
     "Nonstandard bonding numbers" (the Blue Book): "A nonstandard bonding
    number of a **neutral** skeletal atom of a parent hydride is indicated by the symbol
    'λn', cited in conjunction with an appropriate locant." (:2758); the PIN of the first
    ring row is '1-oxa-4λ4-thiacyclotetradecane (PIN)',:9486). A triple bond
    needs the 'yne' ending, (:16489): "The presence of one or more double or
    triple bonds in an otherwise saturated parent hydride... is denoted by changing the
    ending 'ane' of the name of a saturated parent hydride to 'ene' or 'yne'." The builder
    has no 'yne' form, so it declines such a ring. It now writes the λ form (PIN class
    program Task 11,:9158 "The symbol λn, where n is the bonding number, is
    cited immediately after the locant denoting the heteroatom with the nonstandard bonding
    number"), so the λ rings are named: '1-oxa-4λ4-thiacyclotetradecane (PIN)' (:9486). A
    CHARGED ring atom stays in scope: the ring-ion emitters pass the ion and append the
    charge suffix to the parent name the builder returns ('1-oxa-4-azacyclotetradecan-4-ium').

At both tiers the rows never ship the wrong names; the best-effort tier keeps its RT-exact
general-engine name.
"""
import pytest
from rdkit import Chem

from orthonym.rules import ions
from orthonym.rules.skeletal_replacement import _try_cyclic_replacement_name
from tests.support.pin_tiers import assert_not_pin_labelled, assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

# (smiles, the wrong candidate the producer returned at 39803cb6e; OPSIN reads each to a
# DIFFERENT molecule). The best-effort tier names each RT-exact.
NOT_PIN_ROWS = [
    ("C[B-]1(C)CCC2(CCCCC2)C1", "dicyclononyldimethylboranuide"),
    ("C1C[PH-]2CCC1CC2", "triheptylphosphanuide"),
    ("C[B-]1(C)CCCCC1", "dimethyldipentylboranuide"),
    ("C1CCCCC[SH2]CCOCCCC1", "1-oxa-4-thiacyclotetradecane"),
    ("C1CC[SH2]CCCSCCOC1", "1-oxa-4,8-dithiacyclododecane"),
    ("C1#CC[SiH2]CCCCCCCCC[SiH2]CCC=CC=C1", "1,11-disilacycloicosa-4,6-diene"),
    ("C1#CC[SiH2]CCCCCCCC[SiH2]CC=CC=CC=C1", "1,10-disilacycloicosa-12,14,16-triene"),
    # class members beyond the plan's rows (same producers, same defect)
    ("C1CC[P-]CC1", "dipentylphosphanuide"),
    ("C1#CCCCCOCCCCC1", "1-oxacyclododecane"),
    ("C1CCCCC[SH4]CCOCCCC1", "1-oxa-4-thiacyclotetradecane"),
]
# (smiles, PIN the base already ships at pin_verified at both tiers) -- protection
CONTROL_ROWS = [
    ("[B-](C)(C)(C)C", "tetramethylboranuide"),
    ("C[P-](C)(C)C", "tetramethylphosphanuide"),
    ("C1CCCCCSCCOCCCC1", "1-oxa-4-thiacyclotetradecane"),
    ("C1CC[SiH2]CCCCCCCCC[SiH2]CCC=CC=C1", "1,11-disilacycloicosa-4,6-diene"),
    ("C1CCCCC[NH2+]CCOCCCC1", "1-oxa-4-azacyclotetradecan-4-ium"),
    ("C[Si-]1CCCC1", "1-methylsilolan-1-ide"),
]

# Direct calls. The ring centres: every emitter declines; the open-chain centres keep their names.
UIDE_DECLINED = ["C[B-]1(C)CCC2(CCCCC2)C1", "C1C[PH-]2CCC1CC2", "C[B-]1(C)CCCCC1",
                 "[B-]12(CCCC1)CCCC2", "C1CC[P-]CC1", "C[Si-]1CCCC1"]
UIDE_NAMED = [("[B-](C)(C)(C)C", "tetramethylboranuide"), ("C[P-](C)(C)C", "tetramethylphosphanuide"),
              ("F[B-](F)(F)c1ccccc1", "trifluorophenylboranuide")]
HALONIUM_DECLINED = ["C1CC[I+]CC1", "c1ccc2c(c1)[I+]c1ccccc1-2"]
HALONIUM_NAMED = [("c1ccc(cc1)[I+]c1ccccc1", "diphenyliodanium")]
RING_DECLINED = ["C1#CC[SiH2]CCCCCCCCC[SiH2]CCC=CC=C1",
                 "C1#CC[SiH2]CCCCCCCC[SiH2]CC=CC=CC=C1", "C1#CCCCCOCCCCC1"]
RING_NAMED = [("C1CCCCCSCCOCCCC1", "1-oxa-4-thiacyclotetradecane"),
              # the λ form (Task 11; OPSIN 2.9.0 read-back FULL)
              ("C1CCCCC[SH2]CCOCCCC1", "1-oxa-4λ4-thiacyclotetradecane"),          #:9486
              ("C1CC[SH2]CCCSCCOC1", "1-oxa-4,8λ4-dithiacyclododecane"),
              ("C1CCCCC[SH4]CCOCCCC1", "1-oxa-4λ6-thiacyclotetradecane"),
              ("C1CC[SiH2]CCCCCCCCC[SiH2]CCC=CC=C1", "1,11-disilacycloicosa-4,6-diene"),
              # a charged ring atom: the ring-ion emitters append the suffix to this parent name
              ("C1CCCCC[NH2+]CCOCCCC1", "1-oxa-4-azacyclotetradecane"),
              ("C1CCCCC[BH2-]CCOCCCC1", "1-oxa-4-boracyclotetradecane")]


def _centre(mol):
    return next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() != 0)


@pytest.mark.parametrize("smiles", UIDE_DECLINED)
def test_uide_declines_a_ring_centre(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert ions._emit_group13_uide(mol, _centre(mol)) == ""


@pytest.mark.parametrize("smiles,name", UIDE_NAMED)
def test_uide_names_an_open_centre(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    assert ions._emit_group13_uide(mol, _centre(mol)) == name


@pytest.mark.parametrize("smiles", HALONIUM_DECLINED)
def test_halonium_declines_a_ring_centre(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert ions.emit_halogen_onium(mol, _centre(mol)) == ""


@pytest.mark.parametrize("smiles,name", HALONIUM_NAMED)
def test_halonium_names_an_open_centre(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    assert ions.emit_halogen_onium(mol, _centre(mol)) == name


def test_ligand_walk_declines_a_ring_through_the_centre():
    mol = Chem.MolFromSmiles("C[B-]1(C)CCCCC1")
    centre = _centre(mol)
    ring_nbs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(centre).GetNeighbors() if nb.IsInRing()]
    methyl = [nb.GetIdx() for nb in mol.GetAtomWithIdx(centre).GetNeighbors() if not nb.IsInRing()]
    assert all(ions._collect_substituent_atoms(mol, centre, i, set()) is None for i in ring_nbs)
    assert [ions._collect_substituent_atoms(mol, centre, i, set()) for i in methyl] == [{i} for i in methyl]


@pytest.mark.parametrize("smiles", RING_DECLINED)
def test_cyclic_replacement_declines_triple_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert _try_cyclic_replacement_name(mol, mol.GetRingInfo()) is None


@pytest.mark.parametrize("smiles,name", RING_NAMED)
def test_cyclic_replacement_names_the_standard_rings(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    assert _try_cyclic_replacement_name(mol, mol.GetRingInfo()) == name


@pytest.mark.parametrize("smiles,non_pin", NOT_PIN_ROWS)
def test_not_pin_labelled(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


@pytest.mark.parametrize("smiles,pin", CONTROL_ROWS)
def test_control(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
