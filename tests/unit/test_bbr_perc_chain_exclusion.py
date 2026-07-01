"""Phase 169.7 BBR-PERC / DEF-3 — structural chain-exclusion of prefix-only FG atoms.

Azide / diazo / nitroso / nitrite / nitro / N-oxide heteroatoms are characteristic
groups (P-59 / P-65.5 / P-61), NOT chain skeletal atoms. They must never be walked
into an aza/oxa parent chain (the audit's `CN=[N+]=[N-] -> 2,3-diazabutane` bug).

The fix is STRUCTURAL (CONTEXT D-04): `get_chain_excluded_atoms` derives the atom
set from perception's own FG matches (not a per-FG SMARTS blocklist), the skeletal-
replacement path suppresses on it, and `get_principal_group` skips prefix-only
groups so the FG is emitted as a substitutive prefix.
"""
import inspect

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.perception.functional_groups import get_chain_excluded_atoms


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# --- get_chain_excluded_atoms: graph indices of prefix-only FG heteroatoms ---
@pytest.mark.unit
def test_excluded_atoms_azide():
    m = Chem.MolFromSmiles("CN=[N+]=[N-]")
    ex = get_chain_excluded_atoms(m)
    assert len(ex) == 3  # the 3 azide N's
    assert all(m.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in ex)


@pytest.mark.unit
def test_excluded_atoms_empty_for_alkane():
    assert get_chain_excluded_atoms(Chem.MolFromSmiles("CCCC")) == set()


@pytest.mark.unit
def test_excluded_atoms_nitroso_are_heteroatoms():
    m = Chem.MolFromSmiles("CN=O")
    ex = get_chain_excluded_atoms(m)
    assert len(ex) >= 2 and all(m.GetAtomWithIdx(i).GetAtomicNum() in (7, 8) for i in ex)


@pytest.mark.unit
def test_structural_not_blocklist_edit():
    # CONTEXT D-04: derived from detect_functional_groups, NOT a _PRIORITY_FG_SMARTS edit.
    src = inspect.getsource(get_chain_excluded_atoms)
    assert "_PRIORITY_FG_SMARTS" not in src
    assert "detect_functional_groups" in src


# --- end-to-end: prefix-only FG atoms are NOT walked into an aza chain ---
@pytest.mark.unit
def test_azidomethane_not_diazabutane(namer):
    assert namer.name("CN=[N+]=[N-]") == "azidomethane"


@pytest.mark.unit
def test_diazomethane_prefix_not_dropped(namer):
    # diazo was previously consumed-as-principal-then-dropped -> 'methane'
    assert namer.name("C=[N+]=[N-]") == "diazomethane"


@pytest.mark.unit
def test_genuine_skeletal_replacement_unaffected(namer):
    # a real oxa chain (no prefix-only FG) must STILL skeletal-replace.
    # R4 / P-63.2.4: CCOCCOCC has exactly 2 embedded O-ethers with no terminal
    # -ol suffix; the substitutive PIN is '1,2-diethoxyethane' (not skeletal).
    # Updated from pre-R4 '3,6-dioxaoctane'.  A 3-O chain still keeps skeletal:
    assert namer.name("COCCOCCOC") == "2,5,8-trioxanonane"
