"""The organometallic route (ORGANOMETALLIC@50) leaves a chain that has an 'a' PIN to the skeletal route.

``## **** Skeletal replacement ('a') nomenclature in acyclic chains`` (the Blue Book):
an unbranched chain of four or more heterounits, none of them part of the principal characteristic group, has an
'a' PIN. ``## ****`` (:39737) names organometallic parent hydrides substitutively, and the 'a' rule overrides
that substitutive name once the chain has four or more heterounits: the silicon is one of the heterounits.

Before the decline the hub silicon was claimed by ORGANOMETALLIC@50 before SKELETAL_REPLACEMENT and the 'a'
name was never built, so the row had no name from ``Orthonym``, whose class default is the PIN-only tier (the
substitutive silane is refused by the spelling check of; the default tier of ``name_compound`` ships it,
labelled ``systematic_verified``). A molecule with three heterounits at most has no 'a' name, and the route keeps it.
"""

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.metrics.provenance import general_fallback_ctx
from orthonym.routing.dispatch_table import _is_organometallic
from tests.support.pin_tiers import name_breadth
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

SILANE = "COCCOCCOCCOCCO[Si](C)(C)C"
SILANE_NAME = "2,2-dimethyl-3,6,9,12,15-pentaoxa-2-silahexadecane"
# One O and the Si: two heterounits, no 'a' name; the name and tier the engine gave at the start of this task.
KEPT = [("CO[Si](C)(C)C", "methoxytri(methyl)silane", "pin_verified")]


def _organometallic(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return _is_organometallic(mol, smiles, Chem.MolToSmiles(mol))


def test_the_silane_has_its_a_pin():
    row = Orthonym().name_tiered(SILANE)
    assert row["name"] == SILANE_NAME, row
    assert row["tier"] == "pin_verified", row
    assert row["spelling_failures"] == [], row
    assert name_is_rt_exact(SILANE_NAME, SILANE)
    wider = name_breadth(SILANE)
    assert (wider["name"], wider["tier"]) == (SILANE_NAME, "pin_verified"), wider


def test_the_route_declines_the_a_chain():
    assert _organometallic(SILANE) is False


@pytest.mark.parametrize("smiles, name, tier", KEPT)
def test_a_molecule_without_an_a_name_keeps_its_route(smiles, name, tier):
    assert _organometallic(smiles) is True
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == name, row
    assert row["tier"] == tier, row


def test_the_general_tiers_non_pin_a_name_does_not_take_the_route_away():
    """On the general tiers the skeletal route also names a lambda chain with fewer than four heterounits
    ('C[Si]C' -> '2λ2-silapropane', a name that is never the PIN). The route is declined for an 'a' PIN only: the
    probe is asked the default tier's way, so the molecule stays here and the probe marks nothing."""
    token = general_fallback_ctx.set(True)
    try:
        assert _organometallic("C[Si]C") is True
        assert _organometallic(SILANE) is False
    finally:
        general_fallback_ctx.reset(token)
