""" Phase S Task 2 — native ring-parent R/S on the general engine.

Empirical REFRAME (probed 2026-07-22): the von-Baeyer polyene path ALREADY
expresses ring-parent R/S completely — tetrahedral R/S, pseudoasymmetric r/s,
and partial-saturation fused-aromatic (mancude) parents (every ring atom carries
an ``atom_to_locant`` entry, so ``_stereo_prefix`` covers them). These tests
LOCK that behaviour and verify the one hardening adds: the fusion-PIN
early-return no longer ships a stereo-dropping bare fusion word — a stereo-
bearing mancude parent falls through to the stereo-expressing polyene form.
"""
from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

from orthonym.assembly.general_engine import name_general
from orthonym.namer import Orthonym
from orthonym.perception.stereo import assign_stereochemistry
from orthonym.rules.stereochemistry import (
    count_defined_stereo_elements, count_expressed_stereo_descriptors,
    general_engine_stereo_complete,
)


def _name_via_engine(smi, allow_aromatic_general=True):
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, smi
    nm = Orthonym(general_fallback=True, allow_aromatic_general=True,
                   _disable_opsin_validity_gate=True,
                   _disable_grammar_validation=True)
    canon = Chem.MolToSmiles(mol)
    feats = nm._perceive(mol, smi, canon)
    nm._classify(feats)
    res = name_general(mol, feats, allow_aromatic_general=allow_aromatic_general)
    return mol, (res.name if res else None)


def _assert_complete(smi):
    mol, name = _name_via_engine(smi)
    assert name, smi
    assert general_engine_stereo_complete(mol, name), (smi, name)
    return name


def test_vonbaeyer_cage_rs_complete():
    name = _assert_complete("C[C@H]1CC[C@@H]2CC[C@H](C)CC2C1")
    assert "S" in name or "R" in name


def test_vonbaeyer_cage_diol_rs_complete():
    _assert_complete("O[C@H]1CC[C@@H]2CC[C@H](O)CC2C1")


def test_pseudoasymmetric_lowercase_rs():
    """Pseudoasymmetric centres emit lowercase r/s /.4.4)."""
    _, name = _name_via_engine("C[C@H]1CC2CCC(C1)[C@@H]2C")
    assert name
    assert ("r" in name or "s" in name)  # (3s,8s)-... form


def test_fused_aromatic_partial_saturation_rs_complete():
    """A stereocentre on the saturated ring of a partial-sat fused-aromatic
    parent (tetralin) is expressed via the VB polyene form."""
    name = _assert_complete("C[C@H]1CCc2ccccc2C1")
    assert "S" in name or "R" in name


def test_aza_partial_saturation_rs_complete():
    _assert_complete("C[C@H]1CCc2ncccc2C1")


def test_carboxylic_acid_cage_three_centres_complete():
    name = _assert_complete("OC(=O)[C@H]1CC[C@@H]2CCCC[C@@H]12")
    # three centres all expressed
    mol = Chem.MolFromSmiles("OC(=O)[C@H]1CC[C@@H]2CCCC[C@@H]12")
    assert count_expressed_stereo_descriptors(name) == count_defined_stereo_elements(mol)


def test_achiral_mancude_parent_still_bare_fusion_or_polyene():
    """The guard is byte-identical for an achiral mancude parent
    (nd == 0 -> guard passes): naphthalene names without any stereo token."""
    mol, name = _name_via_engine("c1ccc2ccccc2c1")
    assert name
    assert count_expressed_stereo_descriptors(name) == 0
