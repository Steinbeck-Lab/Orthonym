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
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym


def _be():
    # best-effort / general-engine namer; OPSIN gates off for offline
    # structural assertions.
    return Orthonym(style="pin", general_fallback=True,
                     general_fallback_unverified=True,
                     allow_aromatic_general=True,
                     _disable_opsin_validity_gate=True,
                     _disable_grammar_validation=True)


def _be_rt():
    # same but with the OPSIN RT gate ON -- for RT-valid assertions (needs
    # Java; skip if absent).
    return Orthonym(style="pin", general_fallback=True,
                     general_fallback_unverified=True,
                     allow_aromatic_general=True)


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


# ============================================================================
# v28 Composer1 Task 2: recursive decoration composition (end-to-end via engine)
# ============================================================================


@pytest.mark.parametrize("smi", [
    "C[C@@H](N)c1ccc(OCc2ccc(Cl)cc2)nc1",       # pyridine core + -O-CH2-(4-Cl-phenyl) decoration
    "CC(C)(C)Sc1ccc(-c2nc3ccccc3c(=O)[nH]2)cc1", # benzene core + -S-C(CH3)3 decoration
])
def test_decorated_ring_substituent_emits_via_engine(smi):
    row = _be().name_tiered(smi)
    assert row["source"] == "general_engine", row
    assert row["name"] and "unknown" not in row["name"] and " substituent" not in row["name"]
