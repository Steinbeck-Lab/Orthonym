""" NUMBERING, criterion (i): the direction of numbering of an isotopically modified compound.

Cited rules (the Blue Book, with their section headings):

* NUMBERING (:3219), criterion (i) (:3336): "When there is a choice between equivalent numberings in an
  isotopically unmodified compound, the starting point and the direction of numbering of the analogous
  isotopically substituted compound are chosen so as to give lowest locants to the modified atoms or groups
  considered together as a set in increasing numerical order. If a choice still remains, the lower locant is
  given to the nuclide of higher atomic number. In the case of different nuclides of the same element, the
  lower locant is assigned to the nuclide of higher mass number". Its example (:3344): '(2S,4R)-(4-2H1,2-3H1)
  pentane (PIN) [not (2R,4S)-(2-2H1,4-3H1)pentane; isotopic modifications have seniority for low locants over
  stereodescriptors described in (j) below]'.
* Priority between isotopically substituted and unmodified atoms or groups (:44170): the same rule.
* (:21516),.6: the same order for the senior parent structure.

The names are checked against the rule independently of the decorator: every assignment of the cited locants to
the same descriptor that OPSIN reads as the same molecule is an equivalent numbering, and the name's own
must be the first by the criterion (``_best_numbering``).
"""
import itertools
import re

import pytest
from rdkit import Chem

from orthonym.rules import isotopes as I
from orthonym.validation.opsin_roundtrip import opsin_parse
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.unit

_GROUP = re.compile(r"(?:(?P<locs>\d+(?:,\d+)*)-)?(?P<mass>\d+)(?P<el>[A-Z][a-z]?)(?P<cnt>\d*)")


def _blind(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def _best_numbering(smiles, name):
    """``(own, best)``: the keys (locants cited, then nuclides from the lowest locant up as ``(-Z, -mass)``)
    of the name's assignment and of the best of the assignments that read back as the molecule."""
    table = Chem.GetPeriodicTable()
    m = re.search(r"\(((?:\d+(?:,\d+)*-)?\d+[A-Z][a-z]?\d*(?:,(?:\d+(?:,\d+)*-)?\d+[A-Z][a-z]?\d*)*)\)",
                  name.split(")-", 1)[-1] if name.startswith("(") and re.match(r"\(\d+[a-z]?[RS]", name) else name)
    start = name.index(m.group(0))
    groups, pos, inner = [], 0, m.group(1)
    while pos < len(inner):
        g = _GROUP.match(inner, pos)
        groups.append((g.group("locs").split(",") if g.group("locs") else None, int(g.group("mass")),
                       g.group("el"), g.group("cnt")))
        pos = g.end() + 1
    n = Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()
    target = _blind(smiles)

    def key(assignment):
        atoms = sorted(assignment)
        return [loc for loc, _z, _m in atoms], tuple((-z, -mass) for _loc, z, mass in atoms)

    movable = [(i, g) for i, g in enumerate(groups) if g[0]]
    own = key([(int(loc), table.GetAtomicNumber(g[2]), g[1]) for _i, g in movable for loc in g[0]])
    valid = []
    for pick in itertools.product(*[itertools.combinations(range(1, n + 1), len(g[0])) for _i, g in movable]):
        flat = [x for p in pick for x in p]
        if len(set(flat)) != len(flat):
            continue
        chosen = dict(zip([i for i, _g in movable], pick))
        parts, assignment = [], []
        for i, (locs, mass, el, cnt) in enumerate(groups):
            if i in chosen:
                parts.append(f"{','.join(map(str, sorted(chosen[i])))}-{mass}{el}{cnt}")
                assignment += [(x, table.GetAtomicNumber(el), mass) for x in chosen[i]]
            else:
                parts.append(f"{mass}{el}{cnt}")
        candidate = name[:start] + "(" + ",".join(parts) + ")" + name[start + len(m.group(0)):]
        candidate = re.sub(r"^\([^)]*[RS]\)-", "", candidate)
        read = opsin_parse(candidate)
        if read and _blind(read) == target:
            valid.append(key(assignment))
    return own, min(valid)


# -- the key (no OPSIN) ---------------------------------------------------------------------------------


def test_the_key_gives_the_lower_locant_to_the_higher_mass_number():
    d2_t4 = [(2, 2, "H", 1), (4, 3, "H", 1)]
    t2_d4 = [(2, 3, "H", 1), (4, 2, "H", 1)]
    assert I._p4542_p4543_key(t2_d4) < I._p4542_p4543_key(d2_t4)      # (4-2H1,2-3H1) over (2-2H1,4-3H1)
    c13_14 = [((1,), 13, "C", 1), ((6,), 14, "C", 1)]
    c14_13 = [((1,), 14, "C", 1), ((6,), 13, "C", 1)]
    assert I._p4542_p4543_key(c14_13) < I._p4542_p4543_key(c13_14)    # (6-13C,1-14C) over (1-13C,6-14C)


def test_the_key_gives_the_lower_locant_to_the_higher_atomic_number_first():
    carbon_low = [(1, 14, "C", 1), (2, 15, "N", 1)]
    nitrogen_low = [(1, 15, "N", 1), (2, 14, "C", 1)]
    assert I._p4542_p4543_key(nitrogen_low) < I._p4542_p4543_key(carbon_low)
    # an element the module's table does not hold: silicon (14) against carbon (6)
    assert I._p4542_p4543_key([(1, 29, "Si", 1), (2, 13, "C", 1)]) < I._p4542_p4543_key(
        [(1, 13, "C", 1), (2, 29, "Si", 1)])


def test_the_key_reads_every_locant_of_a_group_in_locant_order():
    # (2,4-2H2,3-3H1) cites 2H at 2 and 4 and 3H at 3: from the lowest locant, 2H, 3H, 2H
    assert I._p4542_p4543_key([((2, 4), 2, "H", 2), ((3,), 3, "H", 1)]) == ((-1, -2), (-1, -3), (-1, -2))
    assert I._p4542_p4543_key([(None, 15, "N", 1), (2, 13, "C", 1)]) == ((-6, -13), (-7, -15))


# -- the names (OPSIN) ----------------------------------------------------------------------------------


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    # the Blue Book's own example, with and without its stereo
    ("[2H][C@H](C)C[C@@H]([3H])C", "(2S,4R)-(4-2H1,2-3H1)pentane"),
    ("[2H]C(C)CC([3H])C", "(4-2H1,2-3H1)pentane"),
    ("[2H]CCC[3H]", "(3-2H1,1-3H1)propane"),
    ("[2H]CCCC[3H]", "(4-2H1,1-3H1)butane"),
    ("[13CH3]CCCC[14CH3]", "(6-13C,1-14C)hexane"),
    ("OC[13CH2][14CH2]CO", "(3-13C,2-14C)butane-1,4-diol"),
    ("C1CC[14CH2][13CH2]C1", "(2-13C,1-14C)cyclohexane"),
    ("C1C[15NH]CC[14NH]1", "(4-14N,1-15N)piperazine"),
])
def test_the_numbering_gives_the_lower_locant_to_the_higher_mass_number(smiles, expected):
    from orthonym import Orthonym
    for flags in ({}, {"general_fallback": True, "general_fallback_unverified": True, "allow_aromatic_general": True}):
        row = Orthonym(style="pin", **flags).name_tiered(smiles)
        assert row["name"] == expected
        assert row["tier"] == "pin_verified"
        assert name_is_rt_exact(row["name"], smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "[2H]C(C)CC([3H])C", "[2H]CCCC[3H]", "[2H]c1ccc([3H])cc1", "[2H]C1CCC([3H])CC1", "[13CH3]CCCCC[14CH3]",
    "OC(=O)[13CH2]C[14CH2]C(=O)O", "OC[13CH2]CC[14CH2]CO", "C1[13CH2]C[14CH2]CC1",
    "[2H]C(C)CCC([3H])C([3H])C", "CC[13CH2]CCC[14CH2][14CH3]",
])
def test_no_name_the_engine_certifies_breaks_the_criterion(smiles):
    """The independent check: among the assignments of the cited locants that OPSIN reads as the same molecule
    the name's own is the first by criterion (i)."""
    row = name_best_effort(smiles)
    if row["tier"] != "pin_verified":
        pytest.skip("a name below the PIN is not certified")
    own, best = _best_numbering(smiles, row["name"])
    assert own == best, row["name"]
