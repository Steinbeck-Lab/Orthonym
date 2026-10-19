"""Unnamed-breadth lane C (plan 2026-10-15-unnamed-breadth): PubChem rows that main left unnamed
because a SUBSTITUTED unbranched polyspiro system with a polycyclic (bridged) component was refused
by the polyspiro chain builder, which accepted the bare ring system only (the adamantane-spiro-
trioxolane / tetroxane family). Best-effort must now name each row with a full read-back (full
InChIKey + radical identity + protonation site, a fresh OPSIN call).

Governing rules (the Blue Book Blue Book):
* 'UNBRANCHED POLYSPIRO RING SYSTEMS WITH DIFFERENT RING COMPONENTS ONE BEING A POLYCYCLIC
  RING SYSTEM' (:10705): "Unbranched polyspiro ring systems with at least two different ring
  components and at least one of which is a polycyclic ring are named by placing the ring component
  names in order of their occurrence in the structure beginning with the terminal ring component
  lower in alphabetical order and enclosing within square brackets." The example (:10717) is
  '3,6-dioxadispiro[bicyclo[2.2.1]heptane-2,2'-[1,4]dioxane-5',2''-pyran] (PIN)'.
* (:10719): "If there is a choice of locants, the lowest set of locants for all spiro atoms
  is selected when compared in increasing numerical order and, if there is still a choice, in order
  of their citation in the name". The ring between two spiro atoms numbers the junction toward the
  earlier-cited component 1 and gives the other junction its lowest locant.
* NUMBERING (:3219), (f) "detachable alphabetized prefixes, all considered together in a
  series of increasing numerical order" (:3301): after the spiro atoms, a substituent takes the
  lowest locant its ring allows, whatever the atom order of the input.

PIN track: none (best-effort only). 'RETAINED NAMES FOR VON BAEYER PARENT HYDRIDES' (:9879):
"The retained names adamantane and cubane are used in general nomenclature and as preferred IUPAC
names", so the PIN of the adamantane rows cites 'adamantane', and as the first-cited component
(alphanumerical order,. The strict path spells the cage 'tricyclo[3.3.1.1^3,7]decane' (it
builds no retained cage name inside a spiro name) and labels the bare ring system
``systematic_verified``; so none of the names of this lane is a PIN, and the default tier must
decline every row below. The PIN path is also untouched for a substituted fluorene chain: it keeps
declining it (a test below).
"""
import random

import pytest
from rdkit import Chem, RDLogger

from orthonym.rules.spiro import (
    is_unbranched_polyspiro_different, name_unbranched_polyspiro_different)
from orthonym.rules.spiro_polyspiro import ring_blocks
from orthonym.errors import is_failure_name
from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

RDLogger.DisableLog("rdApp.*")

# A test of what SHIPS runs with the OPSIN validity gate on (tests/conftest.py, the `opsin_gate` marker).
pytestmark = pytest.mark.opsin_gate

# ---------------------------------------------------------------------------------------------
# 1. The rows: real PubChem rows that main did not name (the lead's trace: the chain builder returns
# None at ``len(spiro_atoms) != 2`` or at its unsubstituted-only gate), and rows of the unnamed
# panel with the same signature.
# ---------------------------------------------------------------------------------------------
RED_ROWS = [
    # (cid, smiles)
    ("46238243", "C1CC2(CCC1CC(=O)O)OC3(C4CC5CC3CC(C4)C5(F)F)OO2"),
    ("171402277", "C1CC2(CCC1CC(=O)NCCCN)OOC3(C4CC5CC(C4)CC3C5)OO2"),
    ("161931703", "CCCNC(=O)OC1=CC=C(C=C1)C2CCC[C@@]3(C2)OC4(C5CC6CC(C5)CC4C6)OO3"),
]

PANEL_ROWS = [
    # unnamed-panel rows (benchmarks/unnamed_panel/unnamed_panel.tsv) of the family: a cyclohexane, a
    # dioxolane, a dithiane or a bicyclo[2.2.2]octane beside the tricyclo[3.3.1.1^3,7]decane cage
    ("10088383", "C1CC2(CCC1CCC(=O)O)OC3(C4CC5CC(C4)CC3C5)OO2"),
    ("164618150", "C1CC(CC2(C1)OOC3(C4CC5CC(C4)CC3C5)OO2)C6=CC=C(C=C6)CN7CCOCC7"),
    ("71601513", "CC(C)(CNC(=O)CC1CCC2(CC1)OOC3(C4CC5CC(C4)CC3C5)OO2)N"),
    ("167581739", "CC(C)(C)CNCCCOC1=CC=C(C=C1)C2CCC[C@@]3(C2)OC4(C5CC6CC(C5)CC4C6)OO3"),
    ("146581979", "C1CC(C[C@]2(C1)OC3(C4CC5CC(C4)CC3C5)OO2)C6=CC(=CC=C6)O"),
    ("46238245", "C1CC2(CCC1CC(=O)O)OCC3(O2)C4CC5CC(C4)CC3C5"),
    ("10024188", "C1CC2(CCC1CC(=O)N3CCNC(=O)C3)OC4(C5CC6CC(C5)CC4C6)OO2"),
    ("153490008", "CC(C)(CNC(=O)OC1=CC=C(C=C1)C2CCCC3(C2)OC4(C5CC6CC(C5)CC4C6)OO3)N"),
    ("155511618", "C1CC2(CCC1C3=CC=C(C=C3)OCC(=O)O)OC4(C5CC6CC4CC(C5)C6=O)OO2"),
    ("89034935", "C1CCN(CC1)C2CCN(CC2)CC3=C(C=CC(=C3)C4CCC5(CC4)OOC6(C7CC8CC(C7)CC6C8)OO5)O"),
    ("91355325", "C1CC2(C3(CC1C4=CC(=C(C=C4)O)CN5CCOCC5)C6CC7CC(C6)CC3C7)OOCOO2"),
    ("142199592", "C1CN(CCC12OC3(C4CC5CC(C4)C3C5)OO2)C(=O)C6=CC=CC=C6C(=O)O"),
    ("132504550", "C=CC(=C)COC(=O)C1CCC2(CC1)OOC3(C4CC5CC(C4)CC3C5)OO2"),
    ("119093797", "C1CC(CC2(C1)OC3(C4CC5CC(C4)CC3C5)OO2)OC(=O)OC6=CC=C(C=C6)[N+](=O)[O-]"),
    ("76447603", "C=CC(=O)OCCOC12CC3CC(C1)C4(C(C3)C2)C5(C6CC7CC5CC(C7)(C6)OCCOC(=O)C=C)OO4"),
    ("92164573", "CCOC(=O)[C@@]12CC[C@@](CC13OCCO3)(C4(C2)OCCO4)C(=O)OCC"),
]

ROWS = RED_ROWS + PANEL_ROWS
IDS = [c for c, _ in ROWS]


@pytest.mark.parametrize("cid,smiles", ROWS, ids=IDS)
def test_best_effort_names_row_with_full_read_back(cid, smiles):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles, what=f"CID {cid}: ")
    # the ring system is cited in the component-name form
    assert "dispiro[" in name, name


@pytest.mark.parametrize("cid,smiles", ROWS, ids=IDS)
def test_default_tier_never_claims_a_pin_without_a_citation(cid, smiles):
    res = default_tier_row(smiles)
    assert not res.get("is_pin") or cid in PIN_ROWS, f"CID {cid}: PIN claimed without a cited rule"


PIN_ROWS = set()  # none: see the module docstring, "PIN track"


@pytest.mark.parametrize("cid,smiles", RED_ROWS, ids=IDS[:len(RED_ROWS)])
def test_default_tier_declines_every_red_row(cid, smiles):
    """The strict path builds none of these (UNNAMEABLE), so the default tier declines, as before."""
    row = default_tier_row(smiles)
    assert row["tier"] == "abstain" and row["is_pin"] is False and is_failure_name(row["name"]), row


# ---------------------------------------------------------------------------------------------
# 2. One molecule, one name: every atom order gives the same name (the invariant of determinism).
# ---------------------------------------------------------------------------------------------

def _atom_orders(smiles, n, seed):
    """``n`` SMILES of one molecule written in different atom orders (a fixed seed)."""
    m0 = Chem.MolFromSmiles(smiles)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        perm = list(range(m0.GetNumAtoms()))
        rng.shuffle(perm)
        out.append(Chem.MolToSmiles(Chem.RenumberAtoms(m0, perm), canonical=False))
    return out


@pytest.mark.parametrize("cid,smiles", RED_ROWS[:2], ids=IDS[:2])
def test_one_name_in_every_atom_order(cid, smiles):
    names = set()
    for s in [smiles] + _atom_orders(smiles, 3, 20261019):
        names.add(assert_full_rt(name_best_effort(s).get("name"), s, what=f"CID {cid} as {s}: "))
    assert len(names) == 1, names


@pytest.mark.parametrize("smiles,locant", [
    ("C1CC(C)CC2(C1)OC3(C4CC5CC(C4)CC3C5)OO2", "3"),
    ("CC1CCCC2(C1)OC3(C4CC5CC(C4)CC3C5)OO2", "3"),
    ("C1CC(C)CCC12OC3(C4CC5CC(C4)CC3C5)OO2", "4"),
])
def test_a_ring_substituent_takes_the_lowest_locant_whatever_the_atom_order(smiles, locant):
    """ (f) (:3301): the methyl of the cyclohexane component is 3 (or 4), never 5, in every
    atom order of the input; the cyclohexane spiro atom is locant 1,:10719)."""
    stem = "-methyldispiro[cyclohexane-1,3'-[1,2,4]trioxolane-5',2''-tricyclo[3.3.1.1^3,7]decane]"
    for s in [smiles] + _atom_orders(smiles, 3, 7):
        name = assert_full_rt(name_best_effort(s).get("name"), s, what=f"{s}: ")
        assert f"{locant}{stem}" in name, (s, name)


# ---------------------------------------------------------------------------------------------
# 3. The chain builder on the molecule alone (no OPSIN).
# ---------------------------------------------------------------------------------------------
FAMILY = [r[1] for r in RED_ROWS[:2]] + [PANEL_ROWS[0][1], PANEL_ROWS[5][1], PANEL_ROWS[12][1]]


@pytest.mark.parametrize("smiles", FAMILY)
def test_components_are_read_from_the_ring_bonds_in_every_atom_order(smiles):
    """The smallest set of smallest rings of a cage depends on the atom order (RDKit keeps 3 of the 4
    six-membered rings of adamantane), so the spiro atom of the cage can sit in 2 or in 3 of the rings
    it is listed in. The ring components do not: always the three components and two spiro atoms."""
    for s in [smiles] + _atom_orders(smiles, 6, 11):
        blocks = ring_blocks(Chem.MolFromSmiles(s))
        assert len(blocks) == 3, s
        assert sum(len(b[1]) for b in blocks) == 4, s  # two spiro atoms, each in two components
        assert sorted(len(b[1]) for b in blocks) == [1, 1, 2], s


@pytest.mark.parametrize("smiles", FAMILY)
def test_the_numbering_is_the_same_in_every_atom_order(smiles):
    """The component-name descriptor and the locant of every substituent-bearing ring atom (read off
    the canonical class of the substituent, an atom-order invariant) are one set for all orders."""
    seen = set()
    for s in [smiles] + _atom_orders(smiles, 6, 3):
        m = Chem.MolFromSmiles(s)
        res = name_unbranched_polyspiro_different(m, allow_substituents=True)
        assert res is not None, s
        name, ring_atoms, locants, _ = res
        cls = list(Chem.CanonicalRankAtoms(m, breakTies=False))
        subs = frozenset((str(locants[a]), cls[n.GetIdx()]) for a in ring_atoms
                         for n in m.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() not in ring_atoms)
        seen.add((name, subs))
    assert len(seen) == 1, seen


def test_the_pin_path_keeps_its_unsubstituted_only_reading():
    """The composer's ``is_unbranched_polyspiro_different`` (the PIN path) still refuses a
    substituted chain; only the best-effort request names one."""
    sub = Chem.MolFromSmiles("C1CC2(CCC1C)OC3(C4CC5CC3CC(C4)C5)OO2")
    assert not is_unbranched_polyspiro_different(sub)
    assert name_unbranched_polyspiro_different(sub) is None
    assert name_unbranched_polyspiro_different(sub, allow_substituents=True) is not None


def test_a_ring_that_touches_no_spiro_atom_is_a_substituent_not_a_component():
    """The phenyl of 4-phenylphenol-type rows hangs off the chain by an exocyclic bond; it is no
    fourth component."""
    m = Chem.MolFromSmiles("C1CC(CC2(C1)OC3(C4CC5CC(C4)CC3C5)OO2)C6=CC(=CC=C6)O")
    res = name_unbranched_polyspiro_different(m, allow_substituents=True)
    assert res is not None
    assert res[0] == "dispiro[cyclohexane-1,3'-[1,2,4]trioxolane-5',2''-tricyclo[3.3.1.1^3,7]decane]"
    assert not any(a.GetIdx() in res[1] for a in m.GetAtoms() if a.GetIsAromatic())


def test_a_substituted_fluorene_chain_stays_off_the_default_tier():
    """The PIN path is not widened: the substituted fluorene chain is declined at the default tier
    as before (the example (:10711) is the unsubstituted system)."""
    row = default_tier_row("Cc1ccc2c(c1)-c1ccccc1C21CCC2(CC1)C=Cc1ccccc12")
    assert row["tier"] == "abstain" and row["is_pin"] is False and is_failure_name(row["name"]), row
