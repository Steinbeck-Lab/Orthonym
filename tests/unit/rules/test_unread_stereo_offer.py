"""Names whose stereodescriptors OPSIN cannot read, at the best-effort tier (user decision 2026-11-19), and the
arsenic and antimony oxoacid anions.

OPSIN 2.9.0 and OPSIN main read no lowercase r/s "Achiral cyclic compounds", the Blue Book;
dan2097/opsin#351) and assign no CIP labels on some polyol rings. The best-effort tier ships such a name when OPSIN reads
the stereo-stripped name to the input's constitution, every descriptor matches the input's CIP labels at its unit (the
stereo-layer carve-out), and RDKit's rdCIPLabeler gives every cited unit the code the centres engine gives it
(``validation.second_cip_labeller``); the label is ``best_effort``, never the PIN.

 "Names of mononuclear noncarbon oxoacids" (the Blue Book) names arsonic, arsinic and stibonic acids as
the phosphorus acids are named, and their anions take the same 'ic acid' -> 'ate', 'ous acid' -> 'ite' swap."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.metrics import provenance as pv
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

#: CHEBI:1090 (a 2-deoxystreptamine phosphate); CHEBI:11369 (an inositol pentakisphosphate)
AMINOCYCLITOL = "N[C@@H]1C[C@H](N)[C@@H](O)[C@H](OP(=O)(O)O)[C@H]1O"
INOSITOL_PHOSPHATE = "O=P(O)(O)O[C@H]1[C@@H](OP(=O)(O)O)[C@H](OP(=O)(O)O)[C@@H](O)[C@H](OP(=O)(O)O)[C@H]1OP(=O)(O)O"


def _row(smiles, tier):
    with jvm_slots(1, purpose="unread-stereo-offer"):
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", [
    (AMINOCYCLITOL, "(1R,2r,3S,4R,6S)-4,6-diamino-2-(phosphonooxy)cyclohexane-1,3-diol"),
    (INOSITOL_PHOSPHATE, "(1s,2R,3S,4s,5R,6S)-2,3,4,5,6-pentakis(phosphonooxy)cyclohexan-1-ol"),
])
def test_unread_stereo_name_ships_at_best_effort(smiles, name):
    row = _row(smiles, "best-effort")
    assert (row["name"], row["tier"], row["is_pin"]) == (name, "best_effort", False)
    assert row["gate_outcome"] == pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY


def test_default_tier_still_declines():
    assert _row(AMINOCYCLITOL, "pin").get("tier") == "abstain"


def test_labellers_that_disagree_decline(monkeypatch):
    import orthonym.validation.second_cip_labeller as second
    monkeypatch.setattr(second, "cited_codes_agree", lambda *a: False)
    assert _row(AMINOCYCLITOL, "best-effort").get("tier") == "abstain"


@pytest.mark.parametrize("smiles,name", [
    ("C[As]([O-])[O-]", "methylarsonite"),
    ("C[As](=O)([O-])[O-]", "methylarsonate"),
    ("C[As](C)(=O)[O-]", "dimethylarsinate"),
    ("c1ccccc1[As](=O)([O-])[O-]", "phenylarsonate"),
    ("C[Sb](=O)([O-])[O-]", "methylstibonate"),
])
def test_arsenic_and_antimony_oxoacid_anions(smiles, name):
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified")
    assert name_is_rt_exact(name, smiles)


def test_blocks_sharing_a_locant_are_not_offered():
    """Two blocks with the same locants could be read swapped onto each other's groups (review 2026-11-19): the
    swapped name of this urea must not pass, so neither name is offered."""
    from orthonym.rules.stereochemistry import strip_stereo_blocks
    from orthonym.validation.second_cip_labeller import cited_codes_agree
    smiles = "C[C@H]1CC[C@H](NC(=O)N[C@H]2CC[C@@H](CC)CC2)CC1"
    swapped = "1-[(1r,4r)-4-ethylcyclohexyl]-3-[(1s,4s)-4-methylcyclohexyl]urea"
    with jvm_slots(1, purpose="unread-stereo-offer"):
        assert not cited_codes_agree(swapped, smiles, strip_stereo_blocks(swapped)[0])
