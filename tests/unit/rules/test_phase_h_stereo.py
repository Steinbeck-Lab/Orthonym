"""Phase H  — stereo configuration sub-cases  + centres CIP .

A8 discipline: test the NAME OUTPUT and the structural invariant, not a flag.

: single cyclic double bond on an UNSUBSTITUTED monocyclic cycloalkene
elides the stereodescriptor locant per (BB '(E)-cyclooctene').
: the centres CIP engine is the default, and centres_label_mol maps
centres' canonical-SMILES-keyed labels back onto the mol's OWN atom indices
(the output-order remap) so descriptors never land on the wrong atom.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym.namer import name_compound


# ---------------------------------------------------------------------------
# — single cyclic double bond locant elision (whole-class)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("C/1=C/CCCCCC1", "(Z)-cyclooctene"),
    ("C1CC/C=C/CCC1", "(E)-cyclooctene"),
    ("C/1=C/CCCCCCC1", "(Z)-cyclononene"),
    ("C1CCC/C=C/CCC1", "(E)-cyclononene"),
    ("C/1=C/CCCCCCCC1", "(Z)-cyclodecene"),
])
def test_ster01_monocyclic_cycloalkene_elides_locant(smiles, expected):
    """An unsubstituted monocyclic single-double-bond cycloalkene: '(E)-' not '(1E)-'."""
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles", [
    "C/1=C/CCCCCC1", "C1CC/C=C/CCC1", "C/1=C/CCCCCCC1", "C/1=C/CCCCCCCC1",
])
def test_ster01_elided_block_carries_no_numeric_locant(smiles):
    """The stereo block must be bare '(E)-'/'(Z)-' — no digit before the descriptor."""
    import re
    name = name_compound(smiles)
    m = re.match(r"^\(([^)]*)\)-", name)
    assert m, f"expected a leading stereo block in {name!r}"
    assert not re.search(r"\d", m.group(1)), f"stereo block should have no locant: {name!r}"


@pytest.mark.parametrize("smiles,expected", [
    # ACYCLIC enes keep their locant (ring-bond guard excludes them)
    ("C/C=C/C", "(2E)-but-2-ene"),
    ("CC/C=C/C(=O)O", "(2E)-pent-2-enoic acid"),
    # SUFFIXED ring: OH pushes C=C to locant 2, which the name cites (all-carbon guard)
    ("OC1CCCCCC=C1", "cyclooct-2-en-1-ol"),
    # MULTI-ene macrocycle keeps locants (len(descriptors) > 1)
    ("C/1=C/C=C\\CCCCCC1", "(1Z,3Z)-cyclodeca-1,3-diene"),
])
def test_ster01_boundary_keeps_locant(smiles, expected):
    """The elision must NOT over-fire: acyclic / suffixed / multi-ene keep locants."""
    assert name_compound(smiles) == expected


def test_ster01_mononuclear_precedent_unchanged():
    """The pre-existing mononuclear elision ('(S)-' on methane) is undisturbed."""
    assert name_compound("[C@H](F)(Cl)Br") == "(S)-bromo(chloro)(fluoro)methane"


def test_ster01_rs_ring_block_unaffected():
    """A multi-centre R/S front block on a ring is never elided."""
    assert (name_compound("C[C@@H]1CC[C@@H](C(C)C)[C@H](O)C1")
            == "(1R,2S,5R)-5-methyl-2-(propan-2-yl)cyclohexan-1-ol")


@pytest.mark.parametrize("smiles,expect", [
    ("C1=CCCCCCC1", True),     # cyclooctene (bare, single ring C=C)
    ("C1=CCCCCC1", True),      # cycloheptene (bare; descriptor permanently omitted anyway)
    ("CC1=CCCCCCC1", False),   # methyl substituent present
    ("OC1=CCCCCCC1", False),   # heteroatom present
    ("C1=CC=CCCCC1", False),   # two ring double bonds
    ("C1CCCCCCC1", False),     # no double bond
    ("C/C=C/C", False),        # acyclic
    ("C1=CC2CCCCC2CC1", False),  # bicyclic (>1 ring)
])
def test_is_unsubstituted_monocyclic_cycloalkene(smiles, expect):
    from orthonym.assembly.handlers._handler_shared import (
        _is_unsubstituted_monocyclic_cycloalkene,
    )
    mol = Chem.MolFromSmiles(smiles)
    assert _is_unsubstituted_monocyclic_cycloalkene(mol) is expect


# ---------------------------------------------------------------------------
# — centres default ON + correct atom-index remap
# ---------------------------------------------------------------------------

def test_ster02_centres_default_on():
    """The centres CIP engine is the default (jar+Java present in this env)."""
    from orthonym.perception import stereo
    assert stereo._USE_CENTRES_CIP is True


@pytest.mark.parametrize("smiles", [
    "N[C@@H](CS)C(=O)O",                 # L-cysteine (canonical == input)
    "O[C@H]1C[C@@H](C(=O)O)NC1",         # 4-hydroxyproline (canonical reorders)
    "C[C@@H](C(=O)O)N",                  # L-alanine (canonical reorders)
    "[C@H](F)(Cl)Br",                    # CHFClBr (canonical reorders)
    "O[C@@H]([C@H](O)C(=O)O)C(=O)O",     # tartaric (canonical reorders)
    "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O",   # menthol
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  # glucose
])
def test_ster02_centres_labels_land_on_correct_atoms(smiles):
    """centres_label_mol must set _CIPCode on the SAME atom/bond loci as RDKit.

    The descriptor VALUE may legitimately differ (the +46 centres gain), but a
    centres label landing on an atom RDKit does not flag is a wrong/spurious
    descriptor — the pre-Phase-H bug the output-order remap fixes.
    """
    from orthonym.perception.centres_bridge import centres_label_mol, _find_centres_jar, _java_available
    if _find_centres_jar() is None or not _java_available():
        pytest.skip("centres jar / Java unavailable")

    m_rd = Chem.MolFromSmiles(smiles)
    rdCIPLabeler.AssignCIPLabels(m_rd)
    rd_atoms = {a.GetIdx() for a in m_rd.GetAtoms() if a.HasProp("_CIPCode")}
    rd_bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                for b in m_rd.GetBonds() if b.HasProp("_CIPCode")}

    m = Chem.MolFromSmiles(smiles)
    assert centres_label_mol(m) is True
    c_atoms = {a.GetIdx() for a in m.GetAtoms() if a.HasProp("_CIPCode")}
    c_bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
               for b in m.GetBonds() if b.HasProp("_CIPCode")}

    assert c_atoms == rd_atoms, f"centres atom loci {c_atoms} != RDKit {rd_atoms}"
    assert c_bonds == rd_bonds, f"centres bond loci {c_bonds} != RDKit {rd_bonds}"


def test_ster02_missing_output_order_declines():
    """If _smilesAtomOutputOrder is absent and labels exist, decline (fall back)."""
    from orthonym.perception import centres_bridge as cb
    # A mol that has never been through MolToSmiles has no output-order prop.
    assert cb._smiles_output_order(Chem.MolFromSmiles("CCO")) is None


# ---------------------------------------------------------------------------
# Resolved audit cases V-18 / V-19 / V-20 (no malformed / correct config)
# ---------------------------------------------------------------------------

def test_v18_cysteine_no_malformed_locant():
    """V-18: L-cysteine must not carry a malformed '(1R)-' block on the retained name."""
    name = name_compound("N[C@@H](CS)C(=O)O")
    # STALE EXPECTATION CORRECTED 2026-07-30: was `== "cysteine"`. The REG
    # work made the L-descriptor explicit, and `data/amino_acids.py` now emits
    # `L-cysteine` -- which is right: the SMILES is specifically the (R)/L-enantiomer,
    # and a bare `cysteine` would under-specify it. There is NO code site to fix here;
    # only this assertion was stale.
    assert name == "L-cysteine"
    # ★ THIS is the actual V-18 tripwire and it has passed throughout: the retained
    # name must not carry a malformed '(1R)-'/'(1S)-' block. Do not weaken or merge
    # it into the line above -- the defect V-18 guards against is the malformed
    # locant block, not the stereo-prefix spelling.
    assert "(1R)" not in name and "(1S)" not in name


def test_v19_hydroxyproline_2S():
    """V-19: the 2-position must be S (audit reported a wrong 2R)."""
    assert (name_compound("O[C@@H]1C[C@H](NC1)C(=O)O")
            == "(2S,4R)-4-hydroxypyrrolidine-2-carboxylic acid")


def test_v20_sucrose_alpha_anomer():
    """V-20: the glucopyranoside anomer must be alpha (audit reported a wrong beta)."""
    name = name_compound(
        "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O"
    )
    assert name == "β-D-fructofuranosyl α-D-glucopyranoside"
