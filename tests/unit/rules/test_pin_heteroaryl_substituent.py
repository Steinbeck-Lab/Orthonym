"""Unit tests for PIN heteroaryl substituent numbering (Phase 173.5 L1).

Covers ``pin_heteroaryl_substituent_name`` — the free-valence-aware PIN
substituent numbering for monocyclic heteroaromatic rings acting as
substituents (IUPAC 2013 P-31.1.4.3.4 / P-29.3.5).

Numbering criterion order (first point of difference wins):
  1. heteroatoms as a set get lowest locants
  2. heteroatoms in element-seniority order (O > S > Se > N ...)
  3. indicated hydrogen gets lowest locant
  4. the free valence (attachment) gets lowest locant

The helper is *guarded*: it returns ``None`` (caller keeps its current
locant-less form, zero-regression) whenever the locant is not provably
PIN-correct — unsupported ring, pyrazole misclassified as imidazole,
extra ring substituents, non-monocyclic, or non-aromatic.
"""

import pytest
from rdkit import Chem

from orthonym.rules.ring_substituents import pin_heteroaryl_substituent_name


def _ring_and_attach(smiles, attach_smiles_idx=0):
    """Build (mol, ring_atoms, attachment_atom) from a SMILES with carbon handle(s).

    ``attach_smiles_idx`` selects which handle-bearing ring atom is the
    attachment point, by order of discovery (used for the multi-substituent
    guard test). The attachment atom returned is always a RING atom (the ring
    atom bonded to an exocyclic heavy neighbor), matching how both production
    callers pass it.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    rings = mol.GetRingInfo().AtomRings()
    assert len(rings) == 1, f"test expects a single ring: {smiles}"
    ring = rings[0]
    ring_set = set(ring)
    handle_ring_atoms = []
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in ring_set and nbr.GetAtomicNum() > 1:
                handle_ring_atoms.append(idx)
                break
    assert handle_ring_atoms, f"no exocyclic handle found: {smiles}"
    return mol, ring, handle_ring_atoms[attach_smiles_idx]


# ---------------------------------------------------------------------------
# Positive cases — provably PIN-correct positional names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # Pyridine: N=1, free valence lowest in the 2/3/4 family
        ("Cc1ccccn1", "pyridin-2-yl"),
        ("Cc1cccnc1", "pyridin-3-yl"),
        ("Cc1ccncc1", "pyridin-4-yl"),
        # Furan / thiophene: O/S = 1, attachment 2 or 3
        ("Cc1ccco1", "furan-2-yl"),
        ("Cc1ccoc1", "furan-3-yl"),
        ("Cc1cccs1", "thiophen-2-yl"),
        ("Cc1ccsc1", "thiophen-3-yl"),
        # Pyrrole: indicated H on the NH -> 1H, attachment 2 or 3
        ("Cc1ccc[nH]1", "1H-pyrrol-2-yl"),
        ("Cc1cc[nH]c1", "1H-pyrrol-3-yl"),
        # Imidazole: indicated H beats free valence -> 1H-imidazol-5-yl (the
        # ground-truth case from OC(=O)C(O)Cc1cnc[nH]1). NOT imidazol-4-yl.
        ("Cc1cnc[nH]1", "1H-imidazol-5-yl"),
        ("Cc1ncc[nH]1", "1H-imidazol-2-yl"),
        # Diazines
        ("Cc1cncnc1", "pyrimidin-5-yl"),
        ("Cc1cnccn1", "pyrazin-2-yl"),
        ("Cc1cccnn1", "pyridazin-3-yl"),
    ],
)
def test_pin_positive(smiles, expected):
    mol, ring, attach = _ring_and_attach(smiles)
    assert pin_heteroaryl_substituent_name(mol, ring, attach) == expected


# ---------------------------------------------------------------------------
# Guard cases — must return None (no wrong locant; caller keeps current form)
# ---------------------------------------------------------------------------
def test_pyrazole_named_not_misclassified_as_imidazole():
    # v26 BP-3 R-bug fix: identify_ring_system() returns 'imidazole' for BOTH
    # N,N 5-rings; the adjacent-N case is pyrazole and is now NAMED with the
    # retained pyrazol- stem (P-25.2.1) instead of refused (which previously fell
    # through to a WRONG imidazol-*-yl name downstream). Attachment is the ring
    # carbon adjacent to the =N- (locant 3). OPSIN round-trip verified.
    mol, ring, attach = _ring_and_attach("Cc1cc[nH]n1")  # pyrazol-3-yl (C is the stub)
    assert pin_heteroaryl_substituent_name(mol, ring, attach) == "1H-pyrazol-3-yl"


def test_guard_benzene_not_heteroaryl():
    mol, ring, attach = _ring_and_attach("Cc1ccccc1")  # toluene
    assert pin_heteroaryl_substituent_name(mol, ring, attach) is None


def test_guard_extra_ring_substituent():
    # Pyridine bearing a second ring substituent besides the attachment: the
    # numbering would also have to place the methyl, which this primitive does
    # not claim -> guard to None.
    mol = Chem.MolFromSmiles("Cc1cc(C)cnc1")  # a dimethylpyridine
    ring = mol.GetRingInfo().AtomRings()[0]
    ring_set = set(ring)
    handles = [
        idx for idx in ring
        if any(n.GetIdx() not in ring_set and n.GetAtomicNum() > 1
               for n in mol.GetAtomWithIdx(idx).GetNeighbors())
    ]
    assert len(handles) >= 2
    assert pin_heteroaryl_substituent_name(mol, ring, handles[0]) is None


def test_guard_saturated_ring():
    # pin_heteroaryl_substituent_name claims heteroarenes and saturated
    # HETEROcyclic monocycles (piperidin-4-yl, oxan-2-yl — WS-A task 9 / Phase 4
    # SUBST-01d), but NOT a saturated carbocycle: those are named by the
    # cycloalkyl path, so the heteroaryl primitive must decline them.
    mol, ring, attach = _ring_and_attach("CC1CCCCC1")  # methyl-cyclohexane
    assert pin_heteroaryl_substituent_name(mol, ring, attach) is None
    # A saturated heteromonocycle IS now claimed (free-valence numbering).
    mol2, ring2, attach2 = _ring_and_attach("CC1CCNCC1")  # methyl-piperidine
    assert pin_heteroaryl_substituent_name(mol2, ring2, attach2) == "piperidin-4-yl"


def test_guard_attachment_not_in_ring():
    mol = Chem.MolFromSmiles("Cc1cccnc1")
    ring = mol.GetRingInfo().AtomRings()[0]
    methyl_idx = 0  # the exocyclic carbon, not a ring atom
    assert pin_heteroaryl_substituent_name(mol, ring, methyl_idx) is None
