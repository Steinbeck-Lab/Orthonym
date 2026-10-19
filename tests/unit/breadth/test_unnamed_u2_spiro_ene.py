"""Unnamed-breadth lane U2 (plan 2026-10-15-unnamed-breadth): PubChem rows that main left unnamed
because the carbocyclic side ring of a spiro system with a double bond was named by the bare word
``cyclohexene`` (``spiro[2,3-dihydro-1H-indole-2,1'-cyclohexene]``, no locant, and 'cyclohexene' even
for a ring with two double bonds), which OPSIN 2.9.0 cannot place. A component name cites its double
bonds with the ring's own locants (``cyclohex-2-ene``, ``cyclohexa-2,4-diene``): best-effort must now
name each row with a full read-back (full InChIKey + radical identity + protonation site, a fresh
OPSIN call).

Governing rules (the Blue Book Blue Book):
* 'Monospiro ring systems with different ring components at least one being a polycyclic
  ring and at least one ring component requiring the use of skeletal replacement ('a') nomenclature'
  (:10272), Explanation (:10291): "The spiro atom of the monocyclic hydrocarbon component is given
  preference for low locant." The spiro atom of the ring is locant 1.
* 'Spiro compounds composed of unsaturated rings', (:16715): "Low locants are
  assigned to double bonds in accordance with the fixed numbering of the spiro compound."
* 'Unbranched polyspiro ring systems with different ring components one being a polycyclic
  ring system' (:10705), (:10719): "If there is a choice of locants, the lowest set of locants
  for all spiro atoms is selected when compared in increasing numerical order and, if there is still a
  choice, in order of their citation in the name". The ring between two spiro atoms numbers the junction
  toward the earlier-cited component 1, whatever the atom order of the input.
* NUMBERING (:3219), (c) 'principal characteristic groups and free valences (suffixes)' (:3256),
  (e) 'saturation/unsaturation' (:3288) and (f) "detachable alphabetized prefixes, all considered
  together in a series of increasing numerical order" (:3301): taken in that order after the spiro
  atoms; (f) also decides which of two terminal components with one name is cited first.
* (:16497): "when the endings 'ene' and 'yne' are preceded by a multiplying prefix and a
  locant the letter 'a' is inserted": ``cyclohexa-2,4-diene``.

PIN track: none (best-effort only). The Blue Book spells a monocyclic spiro component with a double
bond with the 'ene' ending after the closing bracket, 'Spiro compounds' (:17042), the
example (:17052) '4,5-dihydro-3H-spiro[[1]benzofuran-2,1'-cyclohexan]-2'-ene (PIN)'. OPSIN 2.9.0
has no grammar for that spelling (it reads ``spiro[indene-1,1'-cyclohex-2-ene]``), the PIN builder
(``_monospiro_component_state``) does not write it, and no twin spelling covers it, so the names of
this lane are not PINs: the default tier must not claim one for any row below.
"""
import re

import pytest
from rdkit import Chem, RDLogger

from orthonym.rules.spiro import (
    _name_carbocyclic_monocycle_component, name_mixed_spiro_fused)
from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

RDLogger.DisableLog("rdApp.*")

# A test of what SHIPS runs with the OPSIN validity gate on (tests/conftest.py, the `opsin_gate` marker): with
# it off, a producer's unreadable spelling of another part of the molecule ('quinolizidine') wins and
# the read-back fails for a reason that has nothing to do with the spiro component.
pytestmark = pytest.mark.opsin_gate

# ---------------------------------------------------------------------------------------------
# 1. The rows: real PubChem rows that main did not name (the lead's trace: the name is built by
# ``_name_side_ring``, 'cyclohexene' with no locant), and rows of the unnamed panel of the same class.
# ---------------------------------------------------------------------------------------------
ROWS = [
    # (cid, smiles)
    ("144725228", "CC1=CC(=CC=C1)C2=CC(=CCC2)N3C4=CC=CC=C4CC35CC=CC=C5"),
    ("127253346", "C[C@@H]1CC(=CC(=O)[C@]12C(=O)C3=C(O2)C(=C(C=C3OC)OC)Cl)NC[C@@H]4CCCN5[C@@H]4CCCC5"),
    ("167493953", "CC(C)(C)NS(=O)C1=CC=CC(=C1)C(=O)N2CC3(CC=CC3)C4=C2C=CC(=C4)NSC"),
    ("170080924", "C[C@@H]1C=C(C=C([C@]12C(=O)C3=C(O2)C(=CC(=C3Cl)C)C)OC)C4CCN(CC4)C(=O)OC(C)(C)C"),
    ("155486670", "C[C@@H]1CC(=O)C(=C([C@]12C(=O)C3=C(O2)C(=C(C=C3OC)OC)Cl)O)C(=NCCCN4CCCCCC4=O)C"),
    ("163012507", "CC1CC(=O)C(=C(C12C(=O)C3=C(O2)C(=C(C=C3OC)OC)Cl)O)C(CC(=O)NCC(=O)NC4=NC=CS4)C5=CC(=C(C(=C5)OC)OC)OC"),
]

PANEL_ROWS = [
    # unnamed-panel rows (benchmarks/unnamed_panel/unnamed_panel.tsv) with a carbocyclic spiro ring that has a
    # double bond; main names none of them. cyclopent-2-ene, -3-ene, cyclopenta-2,4-diene, cyclohex-2-ene,
    # -3-ene, cyclohexa-2,4-diene, -2,5-diene, cyclohept-2-ene, and two dispiro systems whose middle ring
    # carries both spiro atoms and a double bond.
    ("92278023", "CC1=CC[C@@]2([C@H](C1)C(=O)C)C3=CC=CC=C3NC2=O"),
    ("638833", "COC(=O)[C@@H]1CC=CC[C@]12C3=CC=CC=C3NC2=O"),
    ("155929676", "C1C=CCC12CC3=C(C2=O)C=CC(=C3)C4=CC=CC=C4"),
    ("71528299", "CCOC(=O)C1=CC[C@@H]([C@]12C3=CC=CC=C3OC2=O)C4=CC=CC=C4"),
    ("164674669", "C1C[C@@]2(C=CC(=O)C[C@@H]2C3=CC=CC=C3)C4=CC=CC=C41"),
    ("177627369", "C1C(=O)C=C(C12C=NC3=CC=CC=C23)C4=CC=C(C=C4)C5=CC=CC=C5"),
    ("101753536", "CC1=CC=C(C=C1)S(=O)(=O)N2C[C@@]3(CCCCC=C3)C(=C)C4=CC=CC=C42"),
    ("2342144", "CC1(C2=CC=CC=C2N([C@@]13C=C(C=C([C@@H]3C(=O)C4=CC=CS4)C5=CC=CC=C5)C6=CC=CS6)C)C"),
    ("4388243", "CC(C)(C)C1=CC2(C=C(C1=O)C(C)(C)C)C(N(C3=CC=CC=C3N2)S(=O)(=O)CC4=CC=CC=C4)C(=O)C5=CC=C(C=C5)Br"),
    ("101964333", "CC1=CC2=C(C=C1)N(C(=O)[C@]23C(=C(C=C3C(=O)OC)S(=O)(=O)C4=CC=CC=C4)C)C(=O)OC(C)(C)C"),
    ("11709529", "C1=CC2=C3C(=C1)OC4(C=CC(C45C(=O)C=CC5=O)O)OC3=CC=C2"),
    ("101023241", "CCOP(=O)(C1=C([C@H](C2(C13C4=CC=CC=C4NC3=O)C5=CC=CC=C5NC2=O)C#N)N)OCC"),
    ("177654730", "CN1C2=CC=CC=C2C3(C1=O)CC=CCC34C5=CC=CC=C5NC4=O"),
]

ROWS = ROWS + PANEL_ROWS
IDS = [c for c, _ in ROWS]


@pytest.mark.parametrize("cid,smiles", ROWS, ids=IDS)
def test_best_effort_names_row_with_full_read_back(cid, smiles):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles, what=f"CID {cid}: ")
    # the component cites its double bonds with the ring's own locants, never the bare 'cyclohexene'
    assert not re.search(r"cyclo[a-z]{3,6}ene\b", name), name


@pytest.mark.parametrize("cid,smiles", ROWS, ids=IDS)
def test_default_tier_never_claims_a_pin_without_a_citation(cid, smiles):
    res = default_tier_row(smiles)
    assert not res.get("is_pin") or cid in PIN_ROWS, f"CID {cid}: PIN claimed without a cited rule"


PIN_ROWS = set()  # none: see the module docstring, "PIN track"


# ---------------------------------------------------------------------------------------------
# 2. The component name and its numbering, on the molecule alone (no OPSIN).
# ---------------------------------------------------------------------------------------------

def _ring_of(m, spiro_idx, size, with_ene=True):
    """The ring through ``spiro_idx`` of ``size`` atoms that has (or has not) a C=C."""
    for ring in m.GetRingInfo().AtomRings():
        if spiro_idx in ring and len(ring) == size:
            has = any(m.GetBondBetweenAtoms(a, b).GetBondType() == Chem.BondType.DOUBLE
                      for a in ring for b in ring if a < b and m.GetBondBetweenAtoms(a, b))
            if has == with_ene:
                return set(ring)
    raise AssertionError("no such ring")


def _spiro_atom(m):
    return next(a.GetIdx() for a in m.GetAtoms() if a.GetDegree() == 4 and a.GetSymbol() == "C"
                and m.GetRingInfo().NumAtomRings(a.GetIdx()) == 2)


# (smiles, ring size, name, locants of the sp2 ring atoms, where the book prints the rule)
COMPONENTS = [
    ("C1=CCCCC12CCCC2", 6, "cyclohex-2-ene", {2, 3}, "P-31.1.5.1.1 :16715 lowest locant to the double bond"),
    ("C1CC=CCC12CCCC2", 6, "cyclohex-3-ene", {3, 4}, "the direction that gives 3,4 and not 4,5"),
    ("C1=CC=CCC12CCCC2", 6, "cyclohexa-2,4-diene", {2, 3, 4, 5}, "P-31.1.1.1/2 'a' before 'diene'; {2,4} not {3,5}"),
    ("C1=CCC=CC12CCCC2", 6, "cyclohexa-2,5-diene", {2, 3, 5, 6}, "symmetric about the spiro atom"),
    ("C1C=CCC12CCCC2", 5, "cyclopent-3-ene", {3, 4}, "symmetric about the spiro atom"),
]


@pytest.mark.parametrize("smiles,size,name,sp2,why", COMPONENTS, ids=[c[2] for c in COMPONENTS])
def test_component_cites_its_double_bonds_with_the_rings_own_locants(smiles, size, name, sp2, why):
    m = Chem.MolFromSmiles(smiles)
    s = _spiro_atom(m)
    ring = _ring_of(m, s, size)
    got = _name_carbocyclic_monocycle_component(m, ring, s)
    assert got is not None, why
    assert got[0] == name, why
    amap = got[1]
    assert amap[s] == 1, "P-24.5.2 :10291: the spiro atom is locant 1"
    assert {amap[a.GetIdx()] for a in m.GetAtoms() if a.GetIdx() in ring and a.GetHybridization()
            == Chem.HybridizationType.SP2} == sp2, why
    assert sorted(amap.values()) == list(range(1, size + 1))


def test_a_saturated_ring_keeps_its_name():
    m = Chem.MolFromSmiles("C1CCCCC12CCCC2")
    s = _spiro_atom(m)
    got = _name_carbocyclic_monocycle_component(m, _ring_of(m, s, 6, with_ene=False), s)
    assert got is not None and got[0] == "cyclohexane" and got[1][s] == 1


def test_tie_between_directions_goes_to_the_lowest_substituent_locant():
    """cyclopent-3-ene reads the same both ways round; the methyl atom gets locant 2 whatever the
    atom order of the input (f),:3301): the two spellings below are one molecule."""
    for smiles in ("CC1C=CCC12CCCC2", "C1C=CC(C)C12CCCC2"):
        m = Chem.MolFromSmiles(smiles)
        s = _spiro_atom(m)
        ring = _ring_of(m, s, 5)
        got = _name_carbocyclic_monocycle_component(m, ring, s, substituent_tiebreak=True)
        me_bearing = next(a.GetIdx() for a in m.GetAtoms() if a.GetIdx() in ring and a.GetIdx() != s
                          and any(n.GetIdx() not in ring for n in a.GetNeighbors()))
        assert got[0] == "cyclopent-3-ene" and got[1][me_bearing] == 2, smiles


def test_a_ring_with_two_spiro_atoms_numbers_the_second_spiro_atom_before_the_double_bond():
    """ (:10719): the lowest locants for the spiro atoms first, then the rest. The ring
    s1, CH2, s2, CH=CH, CH2 closes at s1: round via CH2 the junction s2 is 3 and the double bond
    4 (cyclohex-4-ene); the other way the double bond would be 3 but s2 would be 5."""
    m = Chem.MolFromSmiles("C1CCC2(C1)CC1(CCCC1)C=CC2")
    spiro = sorted(a.GetIdx() for a in m.GetAtoms() if a.GetDegree() == 4
                   and m.GetRingInfo().NumAtomRings(a.GetIdx()) == 2)
    assert len(spiro) == 2
    ring = next(set(r) for r in m.GetRingInfo().AtomRings() if set(spiro) <= set(r))
    got = _name_carbocyclic_monocycle_component(m, ring, spiro[0], other_spiro_atoms=(spiro[1],))
    assert got[0] == "cyclohex-4-ene" and got[1][spiro[0]] == 1 and got[1][spiro[1]] == 3


def test_a_suffix_atom_is_numbered_before_the_double_bond():
    """ (c) (:3256) comes before (e) (:3288): the direction that gives the ring atom with the
    principal characteristic group its lowest locant wins over the one that gives the double bond
    its lowest locant. The ring is C(=O), CH2, CH=CH, CH2, spiro: double bond first gives
    cyclohex-3-ene with the C=O at 6, suffix first gives cyclohex-4-ene with the C=O at 2."""
    m = Chem.MolFromSmiles("C1(=O)CC=CCC12CCCC2")
    s = _spiro_atom(m)
    ring = _ring_of(m, s, 6)
    keto = next(a for a in ring if m.GetAtomWithIdx(a).GetDegree() == 3 and a != s
                and any(n.GetSymbol() == "O" for n in m.GetAtomWithIdx(a).GetNeighbors()))
    by_ene = _name_carbocyclic_monocycle_component(m, ring, s)
    assert by_ene[0] == "cyclohex-3-ene" and by_ene[1][keto] == 6
    by_suffix = _name_carbocyclic_monocycle_component(m, ring, s, suffix_atoms=(keto,))
    assert by_suffix[0] == "cyclohex-4-ene" and by_suffix[1][keto] == 2


# ---------------------------------------------------------------------------------------------
# 2b. The chain of a polyspiro system: numbering never depends on the atom order of the input.
# ---------------------------------------------------------------------------------------------

def _permuted(smiles, seed):
    """The molecule of ``smiles`` with its atoms renumbered by a seeded permutation, written in that
    atom order (seed 0 is the input itself)."""
    import random
    m = Chem.MolFromSmiles(smiles)
    if seed:
        perm = list(range(m.GetNumAtoms()))
        random.Random(seed).shuffle(perm)
        m = Chem.RenumberAtoms(m, perm)
    return Chem.MolToSmiles(m, canonical=False)


# (cid, smiles, the one name, why)
CHAINS = [
    #: the ring between the two spiro atoms numbers the junction toward the earlier-cited
    # component 1 (3,1'), the other junction 2' (the lowest set), the hydroxyl 3' and the ene 4'. Another
    # atom order used to give 3,2'-cyclopent-3-ene-1',1'' with 5'-hydroxy.
    ("11709529", "C1=CC2=C3C(=C1)OC4(C=CC(C45C(=O)C=CC5=O)O)OC3=CC=C2",
     "3'-hydroxy-2'',5''-dioxodispiro[2,4-dioxatricyclo[7.3.1.0^5,13]trideca-1(12),5(13),6,8,10-pentaene-"
     "3,1'-cyclopent-4-ene-2',1''-cyclopent-3-ene]"),
    # the two spiro atoms are adjacent: 1',2' and not 2',1'; the two terminals are the same component, so
    # the one with the substituent at the lowest locant is cited first (f)): 1-methyl, not 1''-methyl
    ("177654730", "CN1C2=CC=CC=C2C3(C1=O)CC=CCC34C5=CC=CC=C5NC4=O",
     "1-methyl-2,2''-dioxodispiro[2,3-dihydro-1H-indole-3,1'-cyclohex-4-ene-2',3''-2,3-dihydro-1H-indole]"),
    # spiro locants 1',3' (the junction of the second component at 3', not 5')
    ("p09", "C1CCC2(C1)CC1(Cc3ccccc3O1)C=CC2",
     "dispiro[2,3-dihydro-1-benzofuran-2,1'-cyclohex-5-ene-3',1''-cyclopentane]"),
]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("cid,smiles,name", CHAINS, ids=[c[0] for c in CHAINS])
def test_polyspiro_chain_is_named_the_same_in_every_atom_order(cid, smiles, name, seed):
    atom_order = _permuted(smiles, seed)
    res = name_best_effort(atom_order)
    assert res.get("name") == name, f"CID {cid}, atom order {seed}: {atom_order}"
    assert_full_rt(res.get("name"), atom_order, what=f"CID {cid}: ")
    assert default_tier_row(atom_order).get("is_pin") is False


# ---------------------------------------------------------------------------------------------
# 3. The spiro parents the engine builds from a component with a double bond (no OPSIN), and the
# whole small molecule, read back.
# ---------------------------------------------------------------------------------------------

# (smiles, the parent name_mixed_spiro_fused builds, the whole name best-effort gives)
PARENTS = [
    ("C1=CCCCC12Cc3ccccc3O2", "spiro[2,3-dihydro-1-benzofuran-2,1'-cyclohex-2-ene]",
     "spiro[2,3-dihydro-1-benzofuran-2,1'-cyclohex-2-ene]"),
    ("C1CC=CCC12Cc3ccccc3O2", "spiro[2,3-dihydro-1-benzofuran-2,1'-cyclohex-3-ene]",
     "spiro[2,3-dihydro-1-benzofuran-2,1'-cyclohex-3-ene]"),
    ("C1=CC=CCC12Cc3ccccc3N2", "spiro[2,3-dihydro-1H-indole-2,1'-cyclohexa-2,4-diene]",
     "spiro[2,3-dihydro-1H-indole-2,1'-cyclohexa-2,4-diene]"),
    ("C1=CCCCC12C=Cc3ccccc23", "spiro[cyclohex-2-ene-1,1'-indene]", "spiro[cyclohex-2-ene-1,1'-indene]"),
    ("C1=CCC=CC12C=Cc3ccccc23", "spiro[cyclohexa-2,5-diene-1,1'-indene]",
     "spiro[cyclohexa-2,5-diene-1,1'-indene]"),
    ("CC1C=CCC12C=Cc3ccccc23", "spiro[cyclopent-3-ene-1,1'-indene]",
     "2-methylspiro[cyclopent-3-ene-1,1'-indene]"),
    ("C1C=CC(C)C12C=Cc3ccccc23", "spiro[cyclopent-3-ene-1,1'-indene]",
     "2-methylspiro[cyclopent-3-ene-1,1'-indene]"),
    ("O=C1C=CCCC12Cc3ccccc3O2", "spiro[2,3-dihydro-1-benzofuran-2,1'-cyclohex-3-ene]",
     "2'-oxospiro[2,3-dihydro-1-benzofuran-2,1'-cyclohex-3-ene]"),
]


@pytest.mark.parametrize("smiles,parent,whole", PARENTS, ids=[p[2] for p in PARENTS])
def test_the_spiro_parent_carries_the_component_locants(smiles, parent, whole):
    built = name_mixed_spiro_fused(Chem.MolFromSmiles(smiles))
    assert built is not None and built[0] == parent


@pytest.mark.parametrize("smiles,parent,whole", PARENTS, ids=[p[2] for p in PARENTS])
def test_the_whole_small_molecule_is_named_and_read_back(smiles, parent, whole):
    res = name_best_effort(smiles)
    assert res.get("name") == whole
    assert_full_rt(res.get("name"), smiles)
    # not the PIN (module docstring): the default tier declines it
    assert default_tier_row(smiles).get("is_pin") is False
