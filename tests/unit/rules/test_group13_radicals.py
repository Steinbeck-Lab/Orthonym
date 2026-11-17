"""A lone Group 13 radical drawing named on its parent hydride (2026-11-17, PubChem rerun 5 triage item 6).

PubChem draws many aluminium, gallium and indium compounds with the metal short of its hydrogens (`CC/C=C/[Al]`).
The heteroatom radical namer names such a centre on its parent hydride: 'Monovalent radicals'
(the Blue Book), "named by adding the suffix 'yl' to the name of the parent hydride, eliding the final letter
'e'", and 'General method' (:40498) for '-ylidene'; 'alumane', 'gallane', 'indigane' and 'thallane' are
preselected names "PRESELECTED NAMES",:2062) with standard bonding number 3 "Standard bonding
numbers", Table 1.3,:2746). The charged route let only a tin or lead centre reach that namer; a Group 13 centre now
reaches it at the best-effort tier, for the caller's own input. No such name is a PIN,:2056).
"""
import pytest

from orthonym import Orthonym
from orthonym.rules import charged_router
from orthonym.tiers import emit_tier_flags

pytestmark = pytest.mark.opsin_gate


def _tier(name):
    return Orthonym(style="pin", **emit_tier_flags(name))


_PINS = [
    ("CC/C=C/[Al]", "[(1E)-but-1-en-1-yl]alumanylidene"),
    ("CCC[Al]C(C)Cl", "(1-chloroethyl)(propyl)alumanyl"),
    ("C1=CC=C(C=C1)C[Ga]CC2=CC=CC=C2", "dibenzylgallanyl"),
    ("CCO[In]", "ethoxyindiganylidene"),
]


@pytest.mark.parametrize("smiles, name", _PINS)
def test_group13_radical_is_named_on_its_parent_hydride(smiles, name):
    row = _tier("best-effort").name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified")


@pytest.mark.parametrize("tier", ["pin", "valid", "complete"])
def test_the_narrower_tiers_still_decline(tier):
    namer = Orthonym() if tier == "pin" else _tier(tier)
    for smiles, _ in _PINS:
        assert namer.name_tiered(smiles)["tier"] == "abstain", (tier, smiles)


@pytest.mark.parametrize("smiles, name", [
    ("C[Al]", "methylaluminium"),             # the organometallic producer runs first
    ("C[Al](C)Cl", "chlorodi(methyl)alumane"),  # no radical electron
    ("CC[Sn](CC)CC", "triethylstannyl"),      # the tin route as before
])
def test_names_shipped_before_keep_their_spelling(smiles, name):
    assert _tier("best-effort").name_tiered(smiles)["name"] == name


@pytest.mark.parametrize("smiles", ["CC/C=C/[Al]", "CCO[In]"])
def test_the_gate_is_on_the_path(monkeypatch, smiles):
    """Mutation: with the gate back at tin and lead alone, the Group 13 rows decline."""
    real = charged_router._group14_metal_radical_element

    def tin_and_lead_only(mol):
        symbol = real(mol)
        return symbol if symbol in ("Sn", "Pb") else None

    monkeypatch.setattr(charged_router, "_group14_metal_radical_element", tin_and_lead_only)
    assert _tier("best-effort").name_tiered(smiles)["tier"] == "abstain"


def test_a_group13_radical_piece_of_a_mixture_is_not_admitted():
    """The radical word is for the caller's whole input: a piece of a mixture keeps today's naming (unlike tin,
    a Group 13 piece is not admitted to the last mixture producer either, whatever OPSIN reads)."""
    row = _tier("best-effort").name_tiered("CC/C=C/[Al].CCO")
    assert "alumanylidene" not in (row["name"] or "")
