"""Fused parents with a =X group on a ring atom take the hydrogen of their mancude parent
(``fused_rings``: the two-component and the polycomponent fusion producers, through the
shared rule ``ring_hydrogen``), at both tiers.

The rules: (the Blue Book) "the indicated hydrogen atoms are placed at
peripheral atoms that will accommodate these principal characteristic groups";
(:24794) the others "to the lowest nonfusion peripheral atom"; (:24689) 'added
indicated hydrogen' for a group no indicated hydrogen accommodates; (:16880) hydro
prefixes "precede" the indicated hydrogen; (:25036-25044) indicated
hydrogen (c) before the suffix (d) for low locants. Printed: '2H,7H-pyrano[2,3-b]pyran-2,7-
dione (PIN)' (:24782). The bare spellings ('pyrido[1,2-b]pyridazin-6-one') are the general
names of (:14607) and (:28414, '4H-pyran-4-one (PIN) pyran-4-one').
Every expected name reads back to the input's full InChIKey with OPSIN 2.9.0; so do the
old spellings, which is why only the rule decides between them."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots

FUSED = [
    ("O=c1ccn2ncccc2c1", "6H-pyrido[1,2-b]pyridazin-6-one"),
    ("CC1=CC(=O)C=C2C=CC=NN12", "8-methyl-6H-pyrido[1,2-b]pyridazin-6-one"),
    ("O=c1cccc2cccnn12", "8H-pyrido[1,2-b]pyridazin-8-one"),
    ("O=c1ccnc2ccccn12", "4H-pyrido[1,2-a]pyrimidin-4-one"),
    ("O=C1C=Cc2ccncc21", "7H-cyclopenta[c]pyridin-7-one"),
    ("C1=CC2=C(C=CC(=O)C=C2)N=C1", "7H-cyclohepta[b]pyridin-7-one"),
    ("COc1cc(=O)oc2c1CO[C@@](C)(OC)[C@@]2(C)Br",
     "(7R,8S)-8-bromo-4,7-dimethoxy-7,8-dimethyl-7,8-dihydro-2H,5H-pyrano[4,3-b]pyran-2-one"),
    ("O=c1ccc2ccc(=O)oc2o1", "2H,7H-pyrano[2,3-b]pyran-2,7-dione"),            #:24782
    ("O=c1ccoc2c1CC=CO2", "4H,5H-pyrano[2,3-b]pyran-4-one"),
    ("O=C1OCc2cocc21", "1H,3H-furo[3,4-c]furan-1-one"),
    ("O=c1[nH]cnc2sccc12", "thieno[2,3-d]pyrimidin-4(3H)-one"),
    ("O=c1[nH]nc2ccccn12", "[1,2,4]triazolo[4,3-a]pyridin-3(2H)-one"),
    ("C=c1ccc2c(n1)C=CN=2", "5-methylidene-5H-pyrrolo[3,2-b]pyridine"),
    ("O=c1ccn2ccn3ccoc3c1-2", "10H-oxazolo[3,2-a]pyrrolo[2,1-c]pyrazin-10-one"),
]

#: systems without a =X ring atom keep the hydrogen the producers gave them
UNCHANGED = [
    ("c1cc2c[nH]cc2s1", "5H-thieno[2,3-c]pyrrole"),
    ("Cc1cc2c[nH]cc2s1", "2-methyl-5H-thieno[2,3-c]pyrrole"),
    ("O=c1[nH]ccc2sccc12", "thieno[3,2-c]pyridin-4(5H)-one"),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="ring-hydrogen"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", FUSED + UNCHANGED)
def test_a_fused_parent_with_a_group_cites_its_hydrogen(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
