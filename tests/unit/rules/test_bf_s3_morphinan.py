"""Slice S3 on the slice S4 parent: the morphinan-6-one ketones and a suffix on the fusion atom 4a
of '4,12-methano[1]benzofuro[3,2-e]isoquinoline' (``bridged_fused_pin.derived_parents``).

The mancude bridged parent has one indicated hydrogen, at 1, 3, 4a, 6 or 7a; no arrangement of its
double bonds puts it at 7. So the 7-one is accommodated by (the Blue Book) "When
the indicated hydrogen atoms of a parent structure cannot be used to accommodate all of the
principal characteristic groups of the structure, the rules described in are applied":
"(1) at least one of the indicated hydrogen atoms is assigned to a nonfusion peripheral atom having
the lowest locants" (1H) and "(3) principal characteristic groups... that cannot be accommodated
... are accommodated by using 'added indicated hydrogen atoms'" (:24806), cited after the suffix
locant:24693): '...-1H-4,12-methano[1]benzofuro[3,2-e]isoquinolin-7(7aH)-one'. The
4a-hydroxy group of oxycodone is a prefix (the ketone is senior, and needs no hydrogen of the
parent. A 4a-ol suffix (nalbuphine) sits on a fusion atom with no hydrogen in the mancude parent;
the one indicated hydrogen accommodates it, (:24768) "the indicated hydrogen atoms
are placed at peripheral atoms that will accommodate these principal characteristic groups"
('1,3b,4,5,6,6a,7,7a-octahydro-3aH-cyclopenta[a]pentalene-3a,4-diol (PIN)':24790,
'1,4-dihydro-3aH-indene-3a-carboxylic acid (PIN)':24778). (:50943): the natural-product
names ('oxycodone', 'hydromorphone',...) are not PINs. Every name was read back by OPSIN 2.9.0
to the input's full InChIKey; the test reads it back again (``name_is_rt_exact``)."""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules import bridged_fused_pin as bfp
from tests.support.rt_assert import name_is_rt_exact

_P = "4,12-methano[1]benzofuro[3,2-e]isoquinolin"

#: (SMILES, the PIN): the morphinan-6-ones the S4 builder declined (ring ketone) and nalbuphine,
#: whose S4 name put the indicated hydrogen at 1 although the 4a-ol needs it
TARGETS = [
    ("COc1ccc2c3c1O[C@H]1C(=O)CC[C@H]4[C@@H](C2)N(C)CC[C@]314",                       # hydrocodone
     f"(4R,4aR,7aR,12bS)-9-methoxy-3-methyl-2,3,4,4a,5,6-hexahydro-1H-{_P}-7(7aH)-one"),
    ("COc1ccc2c3c1O[C@H]1C(=O)CC[C@@]4(O)[C@@H](C2)N(C)CC[C@]314",                    # oxycodone
     f"(4R,4aS,7aR,12bS)-4a-hydroxy-9-methoxy-3-methyl-2,3,4,4a,5,6-hexahydro-1H-{_P}-7(7aH)-one"),
    ("CN1CC[C@]23c4c5ccc(O)c4O[C@H]2C(=O)CC[C@H]3[C@H]1C5",                           # hydromorphone
     f"(4R,4aR,7aR,12bS)-9-hydroxy-3-methyl-2,3,4,4a,5,6-hexahydro-1H-{_P}-7(7aH)-one"),
    ("CN1CC[C@]23c4c5ccc(O)c4O[C@H]2C(=O)C=C[C@H]3[C@H]1C5",                          # morphinone
     f"(4R,4aR,7aR,12bS)-9-hydroxy-3-methyl-2,3,4,4a-tetrahydro-1H-{_P}-7(7aH)-one"),
    ("COc1ccc2c3c1O[C@H]1C(=O)C=C[C@H]4[C@@H](C2)N(C)CC[C@]314",                      # codeinone
     f"(4R,4aR,7aR,12bS)-9-methoxy-3-methyl-2,3,4,4a-tetrahydro-1H-{_P}-7(7aH)-one"),
    ("C1CC(C1)CN2CC[C@]34[C@@H]5[C@H](CC[C@]3([C@H]2CC6=C4C(=C(C=C6)O)O5)O)O",       # nalbuphine
     f"(4R,4aS,7S,7aR,12bS)-3-(cyclobutylmethyl)-1,2,3,4,5,6,7,7a-octahydro-4aH-{_P}e-4a,7,9-triol"),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s3"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", TARGETS)
def test_the_builder_names_the_member(smiles, name):
    got = bfp.build(Chem.MolFromSmiles(smiles))
    assert got is not None and got[0] == name, got


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", TARGETS)
def test_the_member_is_the_verified_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)


def test_the_ketone_takes_added_hydrogen_because_no_indicated_hydrogen_reaches_c7(monkeypatch):
    # hydromorphone: under the chosen numbering the mancude bridged parent allows indicated
    # hydrogen at 1, 3, 4a, 6 or 7a, not at 7; the 7-one takes 1H and the added hydrogen 7aH
    from orthonym.rules.bridged_fused_pin import hydro
    calls = []
    real = hydro.accommodate

    def spy(state, a2l, need):
        got = real(state, a2l, need)
        if got is not None:
            calls.append(({str(a2l[a]) for s in state.ih_sets for a in s},
                          [str(a2l[a]) for a in need], [str(a2l[a]) for a in got[0]],
                          [str(a2l[a]) for a in got[1]]))
        return got
    monkeypatch.setattr(hydro, "accommodate", spy)
    got = bfp.build(Chem.MolFromSmiles(TARGETS[2][0]))
    assert got is not None and got[0] == TARGETS[2][1], got
    chosen = [c for c in calls if c[1:] == (["7"], ["1"], ["7a"])]
    assert chosen and all(c[0] == {"1", "3", "4a", "6", "7a"} for c in chosen), calls
