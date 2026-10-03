"""Slice S4 routing: the natural-product route (NATURAL_PRODUCT, priority 1100) offers the
bridged fused PIN first (``dispatch_table._systematic_pin_before_natural_product``).

 (the Blue Book): "When the full structure is known, a 'systematic name' may be
generated in accordance with Rules described in Chapters through... Preferred IUPAC
names (PINs) are not identified for the compounds in this Chapter." Where the strict path
builds that name, it replaces the trivial or name at every tier; where it does not, or
where OPSIN does not read it back to the input's full InChIKey, the natural-product name is
returned as before."""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.routing import dispatch_table as dt

MORPHINE = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
MORPHINE_PIN = ("(4R,4aR,7S,7aR,12bS)-3-methyl-2,3,4,4a,7,7a-hexahydro-1H-4,12-methano"
                "[1]benzofuro[3,2-e]isoquinoline-7,9-diol")
CHOLESTEROL = "C[C@H](CCCC(C)C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
DIAMORPHINE = "CN1CC[C@]23c4c5ccc(OC(C)=O)c4O[C@H]2[C@@H](OC(C)=O)C=C[C@H]3[C@H]1C5"


def _handle(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return dt._handle_natural_product(mol, smiles, Chem.MolToSmiles(mol))


@pytest.mark.opsin_gate
def test_the_route_returns_the_bridged_fused_pin_where_the_builder_names_the_molecule():
    with jvm_slots(1, purpose="bf-s4"):
        assert _handle(MORPHINE) == MORPHINE_PIN


def test_a_name_opsin_does_not_read_back_leaves_the_natural_product_name(monkeypatch):
    # fail closed: the builder's name replaces the natural-product name only after the full
    # InChIKey read-back; without it the molecule keeps 'morphine' instead of abstaining
    monkeypatch.setattr(dt, "_rt_full_match", lambda name, mol: False)
    assert _handle(MORPHINE) == "morphine"


def test_the_builder_declining_leaves_the_natural_product_name():
    # an ester principal group (the builder's principal-group set): the builder declines,
    # the route keeps its name (the ring ketones are named since slice S3)
    assert _handle(DIAMORPHINE) == "diamorphine"


def test_a_natural_product_without_a_bridged_fused_reading_keeps_its_name():
    assert _handle(CHOLESTEROL) == "cholesterol"


def _raise(*args, **kwargs):
    raise RuntimeError("probe: an internal error of the bridged fused builder")


def test_an_internal_error_of_the_builder_leaves_the_natural_product_name(monkeypatch):
    # name_bridged_fused_pin runs the builder under builder_declining_on_error: an internal
    # error is a decline, so the route keeps 'morphine' and name does not raise
    import orthonym.rules.bridged_fused_pin as bfp
    monkeypatch.setattr(bfp, "build", _raise)
    assert _handle(MORPHINE) == "morphine"


def test_an_internal_error_of_the_class_check_keeps_the_name_and_its_label(monkeypatch):
    # has_bridged_fused_reading runs the selection under the same guard: an internal error
    # answers False, so the route returns 'diamorphine' (a member the builder declines, an
    # ester principal group) as before the check existed (no relabel) instead of raising
    # out of name
    import orthonym.metrics.provenance as prov
    import orthonym.rules.bridged_fused_pin as bfp
    from orthonym.rules.bridged_fused_pin import selection
    monkeypatch.setattr(selection, "best_splits", _raise)
    recorded = []
    monkeypatch.setattr(prov, "record_non_pin_fragment", recorded.append)
    assert bfp.has_bridged_fused_reading(Chem.MolFromSmiles(DIAMORPHINE)) is False
    assert _handle(DIAMORPHINE) == "diamorphine"
    assert recorded == []


@pytest.mark.opsin_gate
def test_the_trivial_option_does_not_bring_back_the_trivial_name():
    # '--trivial': "A preferred name that can be built is never replaced"
    with jvm_slots(1, purpose="bf-s4"):
        row = Orthonym(trivial_fallback=True).name_tiered(MORPHINE)
    assert (row.get("name"), row["tier"]) == (MORPHINE_PIN, "pin_verified"), row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier_flags", [{}, {"general_fallback": True, "general_fallback_unverified": True,
                                              "allow_aromatic_general": True}])
def test_cholesterol_keeps_its_natural_product_name(tier_flags):
    with jvm_slots(1, purpose="bf-s4"):
        row = Orthonym(style="pin", **tier_flags).name_tiered(CHOLESTEROL)
    assert (row.get("name"), row["tier"]) == ("cholesterol", "pin_verified"), row
