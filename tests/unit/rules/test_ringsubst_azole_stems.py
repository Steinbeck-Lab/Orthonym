"""v28 ring-substituent tranche T2 — 5-membered heteroarene substituent stems
with three heteroatoms (oxadiazole / thiadiazole).

Root cause (reproduced): these rings returned None from the bare narrow producer
(`get_ring_substituent_name`) because they were absent from the retained-name
table — the SAME authoritative path that already names isoxazole/oxazole/
thiazole/tetrazole (NOT `identify_ring_system`, which has no 3-heteroatom 5-ring
branch). Each key is canonical(OPSIN(name)) so the bare name AND the free-valence
substituent locant round-trip by construction (all OPSIN-RT verified).

Triazoles (1,2,3-/1,2,4-triazole) are intentionally NOT added here: the shared
monocyclic numberer cannot anchor an all-nitrogen ring to its indicated-H
nitrogen, so the substituent locant is not provably PIN — fail closed until a
dedicated indicated-H-anchored numbering fix (documented follow-up).
"""
import pytest
from rdkit import Chem

from orthonym.data import get_retained_name
from orthonym.rules.ring_substituents import get_ring_substituent_name


# (canonical SMILES key, expected bare PIN)
BARE_CASES = [
    ("c1ncon1", "1,2,4-oxadiazole"),
    ("c1nnco1", "1,3,4-oxadiazole"),
    ("c1cnon1", "1,2,5-oxadiazole"),   # PIN (furazan is general-only)
    ("c1csnn1", "1,2,3-thiadiazole"),
    ("c1ncsn1", "1,2,4-thiadiazole"),
    ("c1cnsn1", "1,2,5-thiadiazole"),
    ("c1nncs1", "1,3,4-thiadiazole"),
]


@pytest.mark.parametrize("smi,pin", BARE_CASES)
def test_bare_azole_retained_name(smi, pin):
    assert get_retained_name(smi) == pin


# (SMILES, ring-atom index of a C-H, expected PIN substituent). Locants are the
# PIN LOWEST free-valence locants (symmetric rings rely on the numberer's
# lowest-locant tie-break, e.g. 1,3,4-oxadiazol-2-yl not -5-yl).
SUBST_CASES = [
    ("c1ncon1", 0, "1,2,4-oxadiazol-3-yl"),
    ("c1ncon1", 2, "1,2,4-oxadiazol-5-yl"),
    ("c1nnco1", 0, "1,3,4-oxadiazol-2-yl"),   # symmetric -> lowest locant 2
    ("c1cnon1", 0, "1,2,5-oxadiazol-3-yl"),   # symmetric -> lowest locant 3
    ("c1csnn1", 0, "1,2,3-thiadiazol-4-yl"),
    ("c1csnn1", 1, "1,2,3-thiadiazol-5-yl"),
    ("c1ncsn1", 0, "1,2,4-thiadiazol-3-yl"),
    ("c1ncsn1", 2, "1,2,4-thiadiazol-5-yl"),
    ("c1cnsn1", 0, "1,2,5-thiadiazol-3-yl"),  # symmetric -> lowest locant 3
    ("c1nncs1", 0, "1,3,4-thiadiazol-2-yl"),  # symmetric -> lowest locant 2
]


@pytest.mark.parametrize("smi,attach,pin", SUBST_CASES)
def test_azole_substituent_name(smi, attach, pin):
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, smi
    assert mol.GetAtomWithIdx(attach).GetSymbol() == "C", "attach must be a ring C"
    got = get_ring_substituent_name(
        mol, tuple(range(mol.GetNumAtoms())), attach)
    assert got == pin


def test_symmetric_ring_takes_lowest_free_valence_locant():
    """Regression: the numberer applies the free-valence lowest-locant rule to
    symmetric rings (both equivalent carbons -> the lower locant), rather than
    the arbitrary first-direction result."""
    mol = Chem.MolFromSmiles("c1nncs1")  # 1,3,4-thiadiazole, carbons at 2 & 5
    for c in [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"]:
        got = get_ring_substituent_name(mol, tuple(range(mol.GetNumAtoms())), c)
        assert got == "1,3,4-thiadiazol-2-yl", got
