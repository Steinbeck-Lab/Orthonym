"""v27 Phase S Task 1 — per-element stereo COMPLETENESS predicate (accuracy
keystone).

``general_engine_stereo_complete(mol, name)`` returns True iff the name
expresses EVERY defined CIP stereo element the mol carries (all-or-nothing).
This closes the verified latent hole in the old name-side boolean
(``needs_stereo_injection`` Pattern A): a PARTIAL-stereo name — e.g. cage R/S
expressed but one substituent E/Z dropped — used to look "already stereoed" and
ship past the stereo-blind SELF-01. Under PS-1 a partial name is INCOMPLETE →
complete abstains / best-effort flags (never a partial ship in the PIN tiers).
"""
from rdkit import Chem

from orthonym.rules.stereochemistry import general_engine_stereo_complete


_CAMPHOR = "CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2"  # 2 ring stereocentres


def test_fully_expressed_is_complete():
    mol = Chem.MolFromSmiles(_CAMPHOR)
    # both centres expressed
    assert general_engine_stereo_complete(
        mol, "(1R,4R)-1,7,7-trimethylbicyclo[2.2.1]heptan-2-one") is True


def test_partial_expression_is_incomplete():
    """THE keystone: 2 defined centres, only 1 expressed -> incomplete."""
    mol = Chem.MolFromSmiles(_CAMPHOR)
    assert general_engine_stereo_complete(
        mol, "(1R)-1,7,7-trimethylbicyclo[2.2.1]heptan-2-one") is False


def test_no_stereo_expressed_is_incomplete():
    mol = Chem.MolFromSmiles(_CAMPHOR)
    assert general_engine_stereo_complete(
        mol, "1,7,7-trimethylbicyclo[2.2.1]heptan-2-one") is False


def test_stereo_free_molecule_is_complete():
    mol = Chem.MolFromSmiles("CCO")
    assert general_engine_stereo_complete(mol, "ethanol") is True


def test_chain_full_expression():
    mol = Chem.MolFromSmiles("C[C@H](O)[C@@H](O)CC")  # 2 stereocentres
    assert general_engine_stereo_complete(mol, "(2R,3S)-pentane-2,3-diol") is True
    assert general_engine_stereo_complete(mol, "(2R)-pentane-2,3-diol") is False


def test_small_ring_ez_not_counted():
    """A double bond in a <8 ring is ring-strain-fixed (not a free stereogenic
    unit) — RDKit assigns no free descriptor, so the constitutional name is
    complete without an E/Z token (mirrors collect_stereodescriptors)."""
    mol = Chem.MolFromSmiles("C1CCC=CC1")  # cyclohexene, no free E/Z
    assert general_engine_stereo_complete(mol, "cyclohexene") is True


def test_macrocycle_ez_is_counted():
    """A double bond in an 8+ ring CAN carry E/Z (P-31.1.3) -> must be expressed."""
    mol = Chem.MolFromSmiles("C1CCC/C=C/CC1")  # (E)-cyclooctene
    # only counts if RDKit assigned a bond CIP code
    from orthonym.perception.stereo import assign_stereochemistry
    assign_stereochemistry(mol)
    has_bond_cip = any(b.HasProp("_CIPCode") for b in mol.GetBonds())
    if has_bond_cip:
        assert general_engine_stereo_complete(mol, "cyclooctene") is False
        assert general_engine_stereo_complete(mol, "(E)-cyclooctene") is True


def test_empty_name_and_none_mol_fail_closed():
    assert general_engine_stereo_complete(None, "anything") is False
    assert general_engine_stereo_complete(Chem.MolFromSmiles("CCO"), "") is False
