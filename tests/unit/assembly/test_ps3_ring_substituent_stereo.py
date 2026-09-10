""" Phase S Task 3 — ring-substituent internal R/S (+ nested-block
double-apply fix).

Empirical REFRAME (probed 2026-07-22): ring-MEMBER stereocentres inside a
ring-system substituent are ALREADY expressed with substituent-local locants
(``ring_substituents.py`` threads the ring's own numbering through
``collect_stereodescriptors``). These tests LOCK that, and verify the
correctness fix: ``_add_substituent_stereo`` no longer double-applies a spurious
bare ``(R)-`` when a stereocentre is already expressed inside a NESTED
sub-substituent block. The exocyclic-only residual fails CLOSED (0-wrong).
"""
from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

from orthonym.namer import Orthonym
from orthonym.perception.stereo import assign_stereochemistry
from orthonym.rules.stereochemistry import (
    count_defined_stereo_elements, count_expressed_stereo_descriptors,
)
from orthonym.errors import is_failure_name


def _mk():
    return Orthonym(general_fallback=True, allow_aromatic_general=True,
                     _disable_opsin_validity_gate=True,
                     _disable_grammar_validation=True)


def test_ring_member_stereo_in_substituent_expressed():
    """Achiral senior ring, both stereocentres in the ring substituent."""
    smi = "c1ccc(cc1)[C@H]1CCCC[C@@H]1C"
    name = _mk().name(smi)
    assert not is_failure_name(name)
    assert "(1S,2S)" in name or "(1R,2R)" in name or "(1S,2R)" in name or "(1R,2S)" in name
    mol = Chem.MolFromSmiles(smi)
    assert count_expressed_stereo_descriptors(name) == count_defined_stereo_elements(mol)


def test_ring_member_stereo_with_principal_group():
    smi = "OC(=O)c1ccc(cc1)[C@H]1CCCC[C@@H]1C"
    name = _mk().name(smi)
    mol = Chem.MolFromSmiles(smi)
    assert count_expressed_stereo_descriptors(name) == count_defined_stereo_elements(mol)


def test_nested_block_no_spurious_double_descriptor():
    """The double-apply fix: a stereocentre expressed in a NESTED
    sub-substituent block must NOT also get a spurious leading (R)-.

    Molecule has exactly ONE defined stereocentre (the exocyclic hydroxyethyl
    carbon); the emitted name must carry exactly ONE descriptor, not two."""
    smi = "OC(=O)c1ccc(cc1)C1CCCCC1[C@H](O)C"
    mol = Chem.MolFromSmiles(smi)
    assert count_defined_stereo_elements(mol) == 1
    name = _mk().name(smi)
    assert not is_failure_name(name)
    # exactly one descriptor token — no spurious bare (R)- on the cyclohexyl
    assert count_expressed_stereo_descriptors(name) == 1, name


def test_add_substituent_stereo_skips_when_already_expressed():
    """Unit-level: _add_substituent_stereo is idempotent w.r.t. a name that
    already expresses the fragment's stereo."""
    from orthonym.assembly.substituent_naming import _add_substituent_stereo
    mol = Chem.MolFromSmiles("OC(=O)c1ccc(cc1)C1CCCCC1[C@H](O)C")
    assign_stereochemistry(mol)
    # fragment = the whole cyclohexyl+hydroxyethyl substituent (atoms off the ring)
    frag = [a.GetIdx() for a in mol.GetAtoms()
            if not a.GetIsAromatic() and a.GetSymbol() != "O"
            or (a.GetSymbol() == "O" and not any(
                n.GetIsAromatic() for n in a.GetNeighbors()))]
    # a name that already carries the (1R) descriptor must be returned unchanged
    already = "2-[(1R)-1-hydroxyethyl]cyclohexyl"
    out = _add_substituent_stereo(mol, frag, already, attach_idx=None)
    assert out == already  # no spurious (R)- prepended
