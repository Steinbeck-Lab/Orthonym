"""terminal_fragment: an UNSATURATED replacement-chain substituent with DEFINED
backbone C=C geometry now emits a leading (nE)/(nZ) descriptor instead of
refusing (v30 internal-C=C acyl breadth lever).

The producer already names a stereo-FREE unsaturated backbone
(``2-oxo-1-azapent-3-en-1-yl`` for ``-N-C(=O)-CH=CH-CH3``); the only blocker was
``_has_defined_stereo``, which refused any fragment with a defined double-bond
configuration because the module emitted no stereodescriptor. It now emits one
for a BACKBONE C=C (numbered from the free valence, P-29.2), so the amido /
acyloxy enamide + unsaturated-fatty-acyl class stops aborting the whole molecule.

Every expected token below is OPSIN-round-trip-verified IN A PARENT (a bare
token does not parse):
  ``2-[(3E)-2-oxo-1-azapent-3-en-1-yl]acetic acid``      -> C/C=C/C(=O)NCC(=O)O
  ``2-[(3Z)-2-oxo-1-azapent-3-en-1-yl]acetic acid``      -> C/C=C\\C(=O)NCC(=O)O
  ``2-[(3E)-2-oxo-1-oxapent-3-en-1-yl]ethanol``          -> OCCOC(=O)/C=C/C
  ``2-[(3E)-2-oxo-1-oxaoct-3-en-1-yl]ethanol``           -> OCCOC(=O)/C=C/CCCC

Reached ONLY under ``allow_mancude`` (best-effort tier): the single caller,
``substituent_enumerator.py:2218``, sits inside ``if allow_mancude`` -> the PIN
path is byte-identical by construction.
"""
import pytest
from rdkit import Chem

from orthonym.rules.terminal_fragment import terminal_fragment_name


def _tf(smiles, attach):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol, set(range(mol.GetNumAtoms())), attach


# ---- NEW FEATURE: defined backbone C=C -> leading (nE)/(nZ) descriptor -------

@pytest.mark.parametrize("smiles,attach,expected", [
    # N-rooted amido enamide: -N-C(=O)-CH=CH-CH3 (attach = amide N)
    ("C/C=C/C(=O)N", 5, "(3E)-2-oxo-1-azapent-3-en-1-yl"),
    ("C/C=C\\C(=O)N", 5, "(3Z)-2-oxo-1-azapent-3-en-1-yl"),
    # O-rooted acyloxy: -O-C(=O)-CH=CH-R (attach = ester O)
    ("C/C=C/C(=O)O", 5, "(3E)-2-oxo-1-oxapent-3-en-1-yl"),
    ("CCCC/C=C/C(=O)O", 8, "(3E)-2-oxo-1-oxaoct-3-en-1-yl"),
])
def test_defined_backbone_ez_emits_descriptor(smiles, attach, expected):
    mol, frag, at = _tf(smiles, attach)
    got = terminal_fragment_name(mol, frag, at)
    assert got is not None, f"{smiles} refused (E/Z descriptor not emitted)"
    assert got.name == expected
    assert got.atoms == frozenset(frag), "completeness: every atom accounted"


# ---- REGRESSION: stereo-free unsaturated backbone unchanged (no descriptor) --

def test_undefined_double_bond_still_names_without_descriptor():
    mol, frag, at = _tf("CC=CC(=O)N", 5)
    got = terminal_fragment_name(mol, frag, at)
    assert got is not None
    assert got.name == "2-oxo-1-azapent-3-en-1-yl"


def test_saturated_backbone_byte_identical():
    mol, frag, at = _tf("CCCC(=O)N", 5)
    got = terminal_fragment_name(mol, frag, at)
    assert got is not None
    assert got.name == "2-oxo-1-azapentyl"


# ---- REGRESSION: fail closed on stereo we cannot express (0-wrong) -----------

def test_chiral_atom_still_declines():
    """A defined R/S centre is not expressible here (no R/S emission); the guard
    must still refuse rather than ship a stereo-dropped wrong molecule."""
    mol, frag, at = _tf("C[C@H](O)C(=O)N", 6)
    assert terminal_fragment_name(mol, frag, at) is None


def test_backbone_ez_plus_chiral_centre_declines():
    """Backbone E/Z AND an R/S centre cannot both be expressed -> fail closed
    (mirrors _unsaturated_substituent_name's merged-descriptor guard)."""
    mol, frag, at = _tf("C/C=C/[C@H](O)C(=O)N", 8)
    assert terminal_fragment_name(mol, frag, at) is None
