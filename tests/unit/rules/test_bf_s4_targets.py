"""Slice S4 targets: the 4,5-epoxymorphinans named by their bridged fused PIN,
'...-4,12-methano[1]benzofuro[3,2-e]isoquinoline...', at both tiers.

 (the Blue Book): "a 'systematic name' may be generated in accordance with Rules
described in Chapters through... Preferred IUPAC names (PINs) are not identified for
the compounds in this Chapter"; (:23816) "A bridged fused system... is used to
generate names for structures that cannot be named by normal fusion nomenclature";
 (:23843) "fused ring systems > bridged fused systems > non-fused bridged
systems". The parent is the rule-derived '[1]benzofuro[3,2-e]isoquinoline'
(``bridged_fused_pin.derived_parents``). Every name was read back by OPSIN 2.9.0 to the
input's full InChIKey; the test reads it back again (``name_is_rt_exact``)."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

_P = "4,12-methano[1]benzofuro[3,2-e]isoquinolin"

#: molecules no natural-product route names: the derived parent alone makes them live
PARENT_TARGETS = [
    ("C1C=NC2=C3C=CC=C4C13C1=C(O4)C=CC=C1C2", f"1H-{_P}e"),
    ("CN1CCC23c4c5cccc4OC2CCCC3C1C5", f"3-methyl-2,3,4,4a,5,6,7,7a-octahydro-1H-{_P}e"),
    ("CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5",
     f"3-methyl-2,3,4,4a,7,7a-hexahydro-1H-{_P}e-7,9-diol"),
    ("C1CC(C1)CN2CC[C@]34[C@@H]5[C@H](CC[C@]3([C@H]2CC6=C4C(=C(C=C6)O)O5)O)O",       # nalbuphine
     # slice S3: the one indicated hydrogen accommodates the 4a-ol on the fusion atom
     #, the Blue Book; '...-octahydro-3aH-cyclopenta[a]pentalene-3a,4-diol
     # (PIN)':24790)
     f"(4R,4aS,7S,7aR,12bS)-3-(cyclobutylmethyl)-1,2,3,4,5,6,7,7a-octahydro-4aH-{_P}e-4a,7,9-triol"),
]

#: molecules the natural-product route named ('morphine', 'codeine', names): the
#: systematic PIN is offered first
ROUTED_TARGETS = [
    ("CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",                      # morphine
     f"(4R,4aR,7S,7aR,12bS)-3-methyl-2,3,4,4a,7,7a-hexahydro-1H-{_P}e-7,9-diol"),
    ("CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3[C@H](C=C4)O",                # codeine
     f"(4R,4aR,7S,7aR,12bS)-9-methoxy-3-methyl-2,3,4,4a,7,7a-hexahydro-1H-{_P}-7-ol"),
    ("CN1CC[C@]23[C@@H]4C(=CC=C2[C@H]1CC5=C3C(=C(C=C5)OC)O4)OC",                      # thebaine
     f"(4R,7aR,12bS)-7,9-dimethoxy-3-methyl-2,3,4,7a-tetrahydro-1H-{_P}e"),
    ("CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)CC[C@H]3[C@H]1C5",                       # dihydromorphine
     f"(4R,4aR,7S,7aR,12bS)-3-methyl-2,3,4,4a,5,6,7,7a-octahydro-1H-{_P}e-7,9-diol"),
    ("CN1CC[C@]23c4c5ccc(OC)c4O[C@H]2[C@@H](O)CC[C@H]3[C@H]1C5",                      # dihydrocodeine
     f"(4R,4aR,7S,7aR,12bS)-9-methoxy-3-methyl-2,3,4,4a,5,6,7,7a-octahydro-1H-{_P}-7-ol"),
    ("CCN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",                     # N-ethylnormorphine
     f"(4R,4aR,7S,7aR,12bS)-3-ethyl-2,3,4,4a,7,7a-hexahydro-1H-{_P}e-7,9-diol"),
    ("CN1CC[C@@]23c4c5ccc(O)c4O[C@@H]2[C@H](O)C=C[C@@H]3[C@@H]1C5",                   # ent-morphine
     f"(4S,4aS,7R,7aS,12bR)-3-methyl-2,3,4,4a,7,7a-hexahydro-1H-{_P}e-7,9-diol"),
]


ALL_TARGETS = PARENT_TARGETS + ROUTED_TARGETS


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s4"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", ALL_TARGETS)
def test_s4_target_is_the_verified_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)
