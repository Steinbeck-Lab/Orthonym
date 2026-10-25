"""LGA: a two-component ring assembly of a fused mancude heterocycle joined at its heteroatom.

``c1ccc2c(c1)c1ccccc1n2-c1ccc2[nH]c3ccccc3c2c1`` (a carbazole joined by its nitrogen to the 3 position
of another) had no name at the default tier. The junction locant of a ring atom of a fused system was
read off the walk of the single ring that holds it, nitrogen as position 1 (``_get_connection_locant``):
locant 1 for the nitrogen of carbazole, which is locant 9 in the fixed numbering of the component
, and the engine wrote ``9H,9'H-1,3'-bicarbazole``; the round trip refused it. The junction
now takes the locant of the one numbering of the whole component, as the carbocyclic components have
had since v52.

The indicated hydrogen of a fused component was replicated for every component. "Indicated
hydrogen of components in two-component ring assemblies" (the Blue Book): "named and numbered
according to, ignoring the indicated hydrogen atoms of the component rings or ring systems.
The maximum number of noncumulative double bonds is then added taking into account the junction
positions. Any remaining saturated ring positions are designated as indicated hydrogen, placed together
with the appropriate locant(s) at the front of the name of the assembly." The book's
``1,1'-bipyrrole (PIN) (no indicated hydrogen needed)`` (:15603) is the junction at the nitrogen: that
nitrogen holds no saturated position. So the carbazole whose nitrogen is a junction cites no ``9'H``:
``9H-3,9'-bicarbazole``, the ``9H`` of the other unit only.
"""
import pytest
from rdkit import RDLogger

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.tiers import emit_tier_flags
from tests.support.rt_assert import assert_full_rt

RDLogger.DisableLog("rdApp.*")

pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namer():
    with jvm_slots(1, purpose="lga-test"):
        yield Orthonym(style="pin", **emit_tier_flags("pin"))   # emit_tier="pin": the PIN-only tier


#: (id, SMILES, PIN)
FUSED_HETERO_ASSEMBLY = [
    ("3,9'-bicarbazole", "c1ccc2c(c1)c1ccccc1n2-c1ccc2[nH]c3ccccc3c2c1", "9H-3,9'-bicarbazole"),
    ("3'-methyl-3,9'-bicarbazole", "Cc1ccc2c(c1)c1ccccc1n2-c1ccc2[nH]c3ccccc3c2c1",
     "3'-methyl-9H-3,9'-bicarbazole"),
]


@pytest.mark.parametrize("_id,smiles,pin", FUSED_HETERO_ASSEMBLY, ids=[r[0] for r in FUSED_HETERO_ASSEMBLY])
def test_the_junction_nitrogen_of_a_fused_component_has_its_own_locant_and_no_indicated_hydrogen(
        namer, _id, smiles, pin):
    row = namer.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,pin", [
    ("c1ccc2c(c1)[nH]c1ccc(cc21)-c1ccc2[nH]c3ccccc3c2c1", "9H,9'H-3,3'-bicarbazole"),
    ("c1ccc2c(c1)c(c[nH]2)-c1c[nH]c2ccccc12", "1H,1'H-3,3'-biindole"),
    ("c1ccc2c(c1)cc([nH]2)-c1cc2ccccc2[nH]1", "1H,1'H-2,2'-biindole"),
    ("c1ccc2ccccc2c1-c1ccc2ccccc2c1", "1,2'-binaphthalene"),
])
def test_assemblies_joined_at_carbon_keep_their_names(namer, smiles, pin):
    row = namer.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row


#: (id, SMILES, PIN): the attachment locants are the lowest the numberings of the components allow
LOWEST_ATTACHMENT_LOCANTS = [
    ("1,2'-bi-1,3-benzimidazole", "c1ccc2c(c1)ncn2-c1nc2ccccc2[nH]1", "1'H-1,2'-bi-1,3-benzimidazole"),
    ("1,2'-bi-1,3-benzimidazole, the atoms the other way round", "c1ccc2[nH]c(nc2c1)-n1cnc2ccccc12",
     "1'H-1,2'-bi-1,3-benzimidazole"),
    ("1,2'-biindole", "c1ccc2c(c1)ccn2-c1cc2ccccc2[nH]1", "1'H-1,2'-biindole"),
]


@pytest.mark.parametrize("_id,smiles,pin", LOWEST_ATTACHMENT_LOCANTS, ids=[r[0] for r in LOWEST_ATTACHMENT_LOCANTS])
def test_the_junction_nitrogen_takes_the_lowest_locant_its_numberings_allow(namer, _id, smiles, pin):
    # "Ring assemblies with a single bond junction" (the Blue Book): "Lowest possible
    # locants must be used to denote the positions of attachment": N1 and N3 of 1,3-benzimidazole are
    # equivalent in the parent hydride, so the junction nitrogen is 1 (and the indicated hydrogen of
    # the other unit 1'H:,:15593; the book's own N-to-C junction '2'H-1,2'-biindole (PIN)',:15611).
    # The bare skeleton of the ring system with its nitrogen bonded onward does not sanitise, which left
    # the identity as the only numbering and the junction at 3: '1H-2,3'-bi-1,3-benzimidazole'.
    row = namer.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
