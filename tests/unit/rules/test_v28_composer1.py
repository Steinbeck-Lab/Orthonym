"""v28 Composer1 Task 1: detach-and-name ring-substituent primitive.

Fragment-level unit test for
``orthonym.assembly.substituent_enumerator._detach_and_name_ring_substituent``
-- the reusable FIRST primitive for the always-emit recursive substituent
composer. Given a substituent fragment's ring atoms and an attachment ring
atom, it detaches the ring core and names it via the existing general
(von-Baeyer/spiro/cage) ring engine, returning a ``-yl``/``-ylidene`` token.

No ``source=='general_engine'`` provenance assertion here -- that applies to
the end-to-end tests in later v28 composer tasks. This test calls the
fragment-level primitive directly, offline.
"""
from rdkit import Chem


def test_detach_and_name_isolated_cage_fragment():
    from orthonym.assembly.substituent_enumerator import _detach_and_name_ring_substituent
    # adamantane attached via a ring carbon (whole molecule = adamantan-1-yl-acetic acid)
    smi = "OC(=O)CC12CC3CC(CC(C3)C1)C2"
    mol = Chem.MolFromSmiles(smi)
    # frag = the adamantane ring atoms; attach = the ring C bonded to the CH2
    ring_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    attach = next(i for i in ring_atoms
                  if any((not mol.GetAtomWithIdx(n.GetIdx()).IsInRing())
                         for n in mol.GetAtomWithIdx(i).GetNeighbors()))
    name = _detach_and_name_ring_substituent(mol, ring_atoms, attach, allow_mancude=True)
    assert name and name != "substituent" and " " not in name
    assert name.endswith("yl") or name.endswith("ylidene")
