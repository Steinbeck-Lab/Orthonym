""" Phase S Task 5 — ring / endocyclic & von-Baeyer ring-bond E/Z.

Empirical REFRAME (probed 2026-07-22): the general engine ALREADY emits ring /
endocyclic E/Z for rings >= 8 and for von-Baeyer ring bonds, and preserves the
<= 7-membered ring-strain-fixed SUPPRESSION — all via
``collect_stereodescriptors``'s ring-bond branch (min-ring < 8 skip,
 /. These tests LOCK that behaviour and the small-ring
suppression that keeps 0-wrong / no over-abstention.
"""
from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

from orthonym.namer import Orthonym
from orthonym.assembly.general_engine import name_general
from orthonym.perception.stereo import assign_stereochemistry
from orthonym.rules.stereochemistry import (
    count_defined_stereo_elements, count_expressed_stereo_descriptors,
)


def _engine(smi):
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, smi
    nm = Orthonym(general_fallback=True, allow_aromatic_general=True,
                   _disable_opsin_validity_gate=True,
                   _disable_grammar_validation=True)
    canon = Chem.MolToSmiles(mol)
    feats = nm._perceive(mol, smi, canon)
    nm._classify(feats)
    res = name_general(mol, feats, allow_aromatic_general=True)
    return mol, (res.name if res else None)


def test_cyclooctene_ez_expressed():
    mol, name = _engine("C1CCC/C=C/CC1")  # (E)-cyclooctene, 8-ring
    assert name and "E" in name
    assert count_expressed_stereo_descriptors(name) == count_defined_stereo_elements(mol)


def test_macrocycle_ez_expressed():
    mol, name = _engine("C1CCCCC/C=C/CCCCC1")  # 13-ring
    assert name and "E" in name
    assert count_expressed_stereo_descriptors(name) == count_defined_stereo_elements(mol)


def test_small_ring_ez_suppressed():
    """<= 7-membered ring double bond is strain-fixed -> no E/Z token, and the
    completeness gate treats it as complete (no over-abstention)."""
    mol, name = _engine("C1CCC=CC1")  # cyclohexene
    assert name
    assert count_defined_stereo_elements(mol) == 0
    assert count_expressed_stereo_descriptors(name) == 0


def test_vonbaeyer_ring_bond_ez_expressed():
    mol, name = _engine("C1CCC2CCCC/C=C/CCC2CC1")  # bridged macrocycle w/ ring E/Z
    assert name and "E" in name
    assert count_expressed_stereo_descriptors(name) == count_defined_stereo_elements(mol)
