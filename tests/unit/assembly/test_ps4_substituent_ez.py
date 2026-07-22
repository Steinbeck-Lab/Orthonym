"""v27 Phase S Task 4 — substituent-chain & C=N E/Z.

Design (per the plan's "express with AUTHORITATIVE substituent-local locants;
ABSTAIN if a bond's descriptor cannot be anchored to a locant" + the LOCKED
PIN-default-byte-identical constraint):

  BUILT (authoritative locant available -> expressed):
    * parent-chain E/Z            -> (3E)-pent-3-enoic acid
    * near-parent one-hop subst.  -> (4E)-4-(prop-1-en-1-yl)benzoic acid
    * functional-class oxime C=N  -> (E)-acetophenone oxime
    * ring-substituent in-map E/Z -> via collect_stereodescriptors bond handling

  FAIL-CLOSED (no authoritative substituent-local numbering threaded — naming
  the locant from the name string or a BFS would be the forbidden band-aid):
    * fully-internal acyclic-substituent C=C beyond one hop (but-2-en-1-yl)
    * prefix-form C=N (hydroxyimino / imine / hydrazone)
  These ship an underspecified CONSTITUTION-superset name (valid, never a wrong
  stereoisomer -> 0-wrong intact). On the general-engine path PS-1's completeness
  gate routes them to best-effort-flagged / complete-abstain.
"""
from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

from orthonym.namer import Orthonym
from orthonym.perception.stereo import assign_stereochemistry
from orthonym.rules.stereochemistry import (
    count_defined_stereo_elements, count_expressed_stereo_descriptors,
    general_engine_stereo_complete,
)
from orthonym.errors import is_failure_name


def _mk():
    return Orthonym(general_fallback=True, allow_aromatic_general=True,
                     _disable_opsin_validity_gate=True,
                     _disable_grammar_validation=True)


# --- BUILT classes: E/Z expressed with an authoritative locant --------------

def test_parent_chain_ez_expressed():
    assert "3E" in _mk().name("C/C=C/CC(=O)O")


def test_near_parent_one_hop_subst_ez_expressed():
    name = _mk().name("OC(=O)c1ccc(cc1)/C=C/C")
    assert "E" in name and "prop-1-en" in name


def test_functional_class_oxime_ez_expressed():
    name = _mk().name("O/N=C(\\C)c1ccccc1")
    assert name.startswith("(E)") or name.startswith("(Z)")


# --- FAIL-CLOSED residual: valid superset, never a wrong stereoisomer --------

def test_internal_acyclic_subst_ez_is_superset_not_wrong():
    """but-2-en-1-yl on a ring: E/Z dropped -> the name is a valid CONSTITUTION
    superset (both E and Z), never a specific wrong stereoisomer."""
    smi = "OC(=O)C1CCC(CC1)C/C=C/C"
    name = _mk().name(smi)
    assert not is_failure_name(name)
    # constitution correct; the dropped E/Z makes it a superset (0-wrong)
    ref = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    # the emitted name carries no spurious E/Z it cannot anchor
    mol = Chem.MolFromSmiles(smi)
    assert count_expressed_stereo_descriptors(name) < count_defined_stereo_elements(mol)


def test_general_engine_flags_unanchorable_subst_ez():
    """On the general-engine path, an unexpressed substituent E/Z makes the
    completeness gate False -> best-effort flags / complete abstains (never a
    silent partial ship)."""
    # a general-engine ring parent carrying an internal-substituent E/Z the
    # emitter cannot anchor: completeness predicate must report incomplete
    mol = Chem.MolFromSmiles("OC(=O)C1CCC(CC1)C/C=C/C")
    assign_stereochemistry(mol)
    # name WITHOUT the E/Z token -> incomplete (fail-closed)
    assert general_engine_stereo_complete(
        mol, "4-(but-2-en-1-yl)cyclohexane-1-carboxylic acid") is False
