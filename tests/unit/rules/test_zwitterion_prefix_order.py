"""A zwitterion named on its anion cites the cation prefix among the anion parent's own prefixes.

 "Anionic and cationic centers on different parent structures" (the Blue Book) names the
cationic part as a prefix of the anionic parent (``rules/charged_router.py::_route_zwitterion``):
'(N,N-dimethylmethanaminiumyl)acetate (PIN)' (:42473). When the parent cites prefixes of its own, the
cation prefix is one of them: "Simple prefixes... are arranged alphabetically" (:3448),
"The name of a prefix for a substituent is considered to begin with the first letter of its complete name"
(:3477), and "Hyphens are used in substitutive names:... (b) after parentheses, if the closing
parenthesis is followed by a locant" (:6936). The route used to write the prefix in front of the parent's
name, so 10,110 PubChem names read '4-(trimethylazaniumyl)3-(acetyloxy)butanoate'.

The PIN tier spells the cation 'N,N-dimethylmethanaminiumyl' and verifies it through its
'trimethylazaniumyl' twin (validation/twin_spellings.py); the best-effort tier names the same atoms with the
twin. OPSIN 2.9.0 reads every best-effort name below back to the full InChIKey of its input.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,pin,best_effort", [
    ("CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C",
     "3-(acetyloxy)-4-(N,N-dimethylmethanaminiumyl)butanoate",
     "3-(acetyloxy)-4-(trimethylazaniumyl)butanoate"),
    # the parent is named without the cation, so its own descriptor ('(3S)' for this (3R) centre) is left out
    # and the descriptor of the whole molecule is cited
    ("CC(=O)O[C@H](CC(=O)[O-])C[N+](C)(C)C",
     None,
     "(3R)-3-(acetyloxy)-4-(trimethylazaniumyl)butanoate"),
    ("C[N+](C)(C)CC(Cl)CC(=O)[O-]",
     "3-chloro-4-(N,N-dimethylmethanaminiumyl)butanoate",
     "3-chloro-4-(trimethylazaniumyl)butanoate"),
    ("C[N+](C)(C)[C@@H](Cc1c[nH]c2ccccc12)C(=O)[O-]",
     None,   # the PIN tier declines this R/S zwitterion
     "(2S)-3-(1H-indol-3-yl)-2-(trimethylazaniumyl)propanoate"),
    # an unsaturated parent: its numbering starts at the carboxylate too
    ("C[N+](C)(C)CC(O)/C=C/C(=O)[O-]",
     "(2E)-5-(N,N-dimethylmethanaminiumyl)-4-hydroxypent-2-enoate",
     "(2E)-4-hydroxy-5-(trimethylazaniumyl)pent-2-enoate"),
    # the parent's E descriptor stays in front: an exocyclic '-ylidene' bond takes the locant of its point of
    # attachment, (the Blue Book), not the substituent's own '(1Z)'
    ("C[N+](C)(C)CCC/C(=C/C)C(=O)[O-]",
     "(2Z)-5-(N,N-dimethylmethanaminiumyl)-2-ethylidenepentanoate",
     "(2Z)-2-ethylidene-5-(trimethylazaniumyl)pentanoate"),
    ("CCCC(=O)OCC(COCCC(C(=O)[O-])[N+](C)(C)C)OC(=O)CCC",
     "4-[2,3-bis(butanoyloxy)propoxy]-2-(N,N-dimethylmethanaminiumyl)butanoate",
     "4-[2,3-bis(butanoyloxy)propoxy]-2-(trimethylazaniumyl)butanoate"),
])
def test_the_cation_prefix_is_cited_in_alphanumerical_order(smiles, pin, best_effort):
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (best_effort, "systematic_verified"), row
    assert name_is_rt_exact(best_effort, smiles), row
    row = tier_row(smiles, "pin")
    if pin is None:
        assert row["tier"] == "abstain", row
    else:
        assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
