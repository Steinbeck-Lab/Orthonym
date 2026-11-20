"""Ring cations whose stereo sits on a bridged parent ring (ChEBI C4, lane/c4 2026-11-19).

Two causes left the tropane cations of ChEBI without a name.

1. ``ions._p74_bare_ring_stem`` names the bare parent ring, and that ring keeps the chiral tags of the ion's ring
   atoms, so the stem came back with its own block, joined after the ion's prefixes:
   '3-hydroxy-8,8-dimethyl(1R,5S)-8-azabicyclo[3.2.1]octan-8-ium'. "NAMING OF STEREOISOMERS"
   (the Blue Book): stereodescriptors "are placed at the front of the complete name when related to the parent
   structure". ``rules.ring_ion_stereo.ring_stereo_front`` drops the stem's block and cites the ion's own block, in
   the ion's numbering, at the front.
2. ``stereochemistry.inject_stereo_reanchored_rt_gated`` flattened every name OPSIN does not read back in full, so a
   name whose lowercase r/s OPSIN cannot read lost its stereo and was then withdrawn. At the best-effort request it
   now keeps a block OPSIN cannot read when the descriptor check places every descriptor; the best-effort gate holds
   it and ``assembly.stereo_unread_offer`` offers it when the second CIP labeller agrees (user decision 2026-11-19, D1:
   below the PIN). This part leaves the PIN tiers as they were: it runs at the best-effort request only.

The default tier gives the quaternary cation of cause 1 with the label pin_unverified (never the PIN label): a name of
the strict path whose stereo OPSIN cannot read ships there through the stereo-layer carve-out of
``namer._final_opsin_validity_gate`` (constitution read back by OPSIN, every descriptor checked against the input's CIP
labels), as before this lane for any such name; the user decision of 2026-11-19 (D1) is about the names the gate
withdraws, offered at best-effort when two labellers agree (``assembly.stereo_unread_offer``)."""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def test_a_quaternary_tropane_cation_cites_its_block_at_the_front():
    smiles = "C[N+]1(C)[C@@H]2CC[C@H]1C[C@H](O)C2"
    name = "(1R,3s,5S)-3-hydroxy-8,8-dimethyl-8-azabicyclo[3.2.1]octan-8-ium"
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (name, "pin_unverified"), (tier, row)


@pytest.mark.parametrize("smiles,name", [
    ("C[NH+]1[C@@H]2CC[C@H]1C[C@H](O)C2", "(1R,3s,5S)-3-hydroxy-8-methyl-8-azabicyclo[3.2.1]octan-8-ium"),
    ("C[NH+]1[C@@H]2CC[C@H]1C[C@@H](OC(=O)[C@H](CO)c1ccccc1)C2",
     "(1R,3r,5S)-3-{[(2S)-3-hydroxy-2-phenylpropanoyl]oxy}-8-methyl-8-azabicyclo[3.2.1]octan-8-ium"),
])
def test_a_protonated_tropane_cation_ships_below_the_pin_at_best_effort(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"], row["is_pin"]) == (name, "best_effort", False), row
    assert tier_row(smiles, "pin")["tier"] == "abstain"


def test_the_bare_ring_block_is_replaced_by_the_ions_block():
    from rdkit import Chem

    from orthonym.rules.ring_ion_stereo import ring_stereo_front
    mol = Chem.MolFromSmiles("C[N+]1(C)[C@@H]2CC[C@H]1C[C@H](O)C2")
    locants = {1: 8, 3: 1, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 10: 2}
    assert ring_stereo_front(mol, "(1R,5S)-8-azabicyclo[3.2.1]octane", locants) == (
        "8-azabicyclo[3.2.1]octane", "(1R,3s,5S)-")
    assert ring_stereo_front(mol, "pyridine", locants) == ("pyridine", "")


def test_the_injector_keeps_an_unread_block_only_at_the_best_effort_request_and_only_when_checked():
    """``inject_stereo_reanchored_rt_gated`` keeps a block OPSIN cannot read only at the best-effort request, and only
    when the descriptor check places it: the PIN tiers and a wrong descriptor get the flat name, as before."""
    from rdkit import Chem

    from orthonym.jvm_budget import jvm_slots
    from orthonym.metrics.provenance import best_effort_request_ctx
    from orthonym.perception.stereo import assign_stereochemistry
    from orthonym.rules.stereochemistry import inject_stereo_reanchored_rt_gated
    smiles = "C[NH+]1[C@@H]2CC[C@H]1C[C@H](O)C2"
    flat = "3-hydroxy-8-methyl-8-azabicyclo[3.2.1]octan-8-ium"
    locants = {1: 8, 2: 1, 3: 7, 4: 6, 5: 5, 6: 4, 7: 3, 9: 2}
    mol = Chem.MolFromSmiles(smiles)
    assign_stereochemistry(mol)
    wrong = Chem.MolFromSmiles("C[NH+]1[C@@H]2CC[C@H]1C[C@@H](O)C2")     # C3 the other way: '3r' for the '3s' input
    assign_stereochemistry(wrong)
    with jvm_slots(1, purpose="tropane-injector"):
        assert inject_stereo_reanchored_rt_gated(flat, mol, locants, input_smiles=smiles) == flat
        token = best_effort_request_ctx.set(True)
        try:
            assert inject_stereo_reanchored_rt_gated(flat, mol, locants, input_smiles=smiles) == (
                "(1R,3s,5S)-" + flat)
            assert inject_stereo_reanchored_rt_gated(flat, wrong, locants, input_smiles=smiles) == flat
        finally:
            best_effort_request_ctx.reset(token)
