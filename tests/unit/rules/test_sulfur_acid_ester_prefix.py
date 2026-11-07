"""The ester of a sulfonic or sulfinic acid cited as a prefix, and the ring-nitrogen ester senior to it.

 "Esters cited as prefixes" (the Blue Book): "When, in an ester with the general structure R-CO-O-R′
or R-S(O)x-O-R′, another group is present that has priority for citation as the principal group... an ester group is
indicated by prefixes as 'acyloxy' for the group R-CO-O–, and 'alkoxy...oxo', '(alkyloxy)...oxo'...". The acyl group of
a sulfonic acid is 'methanesulfonyl', 'benzenesulfonyl (preferred prefix)',:31266), enclosed in parentheses
,:1754), so the ester R-SO2-O- on a parent is '(methanesulfonyl)oxy' and R-O-SO2- is '(methoxysulfonyl)'
('[2-(methoxysulfonyl)phenyl]phosphonic acid (PIN)',:36540).

The acid on a ring nitrogen is a carboxylic acid (:24910 'tetraazane-1-carboxylic acid (PIN) [not
(triazan-1-yl)carbamic acid; the carboxylic acid is senior to the carbonic acid derivative]'), first of class 7a of
 (:18172: "carboxylic..., sulfonic, sulfinic... Chalcogen analogues follow each of the corresponding
oxygen acids"), so its ester is senior to a sulfonic, sulfinic or thio ester: the PIN of a Boc ring that carries one is
the carbamate name, not the sulfonate or thioate name. A chain carbamate (carbamic acid, class 7b) stays junior.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    # the Boc-ring esters of junior acids: the ring-nitrogen carboxylic ester is the parent
    ("CC(C)(C)OC(=O)N1CCC(CC1)OS(C)(=O)=O", "tert-butyl 4-[(methanesulfonyl)oxy]piperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(CC1)OS(=O)(=O)c1ccc(C)cc1",
     "tert-butyl 4-[(4-methylbenzene-1-sulfonyl)oxy]piperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(CC1)OS(=O)(=O)C(F)(F)F",
     "tert-butyl 4-[(trifluoromethanesulfonyl)oxy]piperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(CC1)OS(=O)C", "tert-butyl 4-[(methanesulfinyl)oxy]piperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(CC1)SC(C)=O", "tert-butyl 4-(acetylsulfanyl)piperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CC(C1)OS(C)(=O)=O", "tert-butyl 3-[(methanesulfonyl)oxy]azetidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(CC1)COS(C)(=O)=O",
     "tert-butyl 4-{[(methanesulfonyl)oxy]methyl}piperidine-1-carboxylate"),
    # the prefix on chain, ring and benzene parents, both orientations
    ("CC(OS(C)(=O)=O)C(=O)O", "2-[(methanesulfonyl)oxy]propanoic acid"),
    ("OC(=O)C1CCC(CC1)OS(C)(=O)=O", "4-[(methanesulfonyl)oxy]cyclohexane-1-carboxylic acid"),
    ("OC(=O)c1ccc(OS(=O)(=O)c2ccccc2)cc1", "4-[(benzenesulfonyl)oxy]benzoic acid"),
    ("OC(=O)CCS(=O)(=O)OC", "3-(methoxysulfonyl)propanoic acid"),
    ("OP(=O)(O)c1ccccc1S(=O)(=O)OC", "[2-(methoxysulfonyl)phenyl]phosphonic acid"),   #:36540 verbatim
    ("OC(=O)C(C)S(=O)(=O)Oc1ccccc1", "2-(phenoxysulfonyl)propanoic acid"),
    ("CCO/C=C(\\C(=O)OC1=CC=CC=C1)/S(=O)(=O)OC2=CC=CC=C2", "phenyl (2E)-3-ethoxy-2-(phenoxysulfonyl)prop-2-enoate"),
])
def test_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


def test_sulfooxy_unchanged():
    """-O-SO2-OH keeps its preselected prefix 'sulfooxy',:31342)."""
    row = tier_row("OC(=O)c1ccc(OS(O)(=O)=O)cc1", "pin")
    assert (row["name"], row["tier"]) == ("4-(sulfooxy)benzoic acid", "pin_verified"), row


@pytest.mark.parametrize("smiles,expected", [
    # a ring in the acid side of a sulfonate on a branch: the group is one prefix, consumed before the ring branches of
    # the chain are read (substituent_naming Pass 0)
    ("CC(C)(C)OC(=O)N1CCC[C@H]1COS(=O)(=O)c1ccc(C)cc1",
     "tert-butyl (2S)-2-{[(4-methylbenzene-1-sulfonyl)oxy]methyl}pyrrolidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(CC1)CCOS(=O)(=O)c1ccccc1",
     "tert-butyl 4-{2-[(benzenesulfonyl)oxy]ethyl}piperidine-1-carboxylate"),
    ("OC(=O)C1CCC(CCOS(=O)(=O)c2ccccc2)CC1", "4-{2-[(benzenesulfonyl)oxy]ethyl}cyclohexane-1-carboxylic acid"),
    ("OC(=O)C1CCC(CC(C(=O)O)COS(=O)(=O)c2ccccc2)CC1",
     "4-{3-[(benzenesulfonyl)oxy]-2-carboxypropyl}cyclohexane-1-carboxylic acid"),
])
def test_sulfonate_with_a_ring_in_a_branch(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


def test_chain_carbamate_stays_junior_to_the_sulfonate():
    """Carbamic acid is a carbonic acid (class 7b), junior to the sulfonic acid (7a): the sulfonate is the parent."""
    row = tier_row("CCOC(=O)NCCOS(C)(=O)=O", "best-effort")
    assert row["name"] == "2-[(ethoxycarbonyl)amino]ethyl methanesulfonate", row


@pytest.mark.parametrize("smiles", [
    # "Polyesters formed from a single 'alcoholic' component" (:31815): the PIN names the polyol as a
    # multivalent group ('butane-1,4-diyl dimethanesulfonate'), not one ester with the others as acyloxy prefixes
    "CS(=O)(=O)OCCCCOS(C)(=O)=O",
    "CC(CCC(C)OS(C)(=O)=O)OS(C)(=O)=O",
    "C1=CC=C(C=C1)CS(=O)(=O)OC2=C(C=C(C=C2)C=O)OS(=O)(=O)CC3=CC=CC=C3",
    # different anions on one polyol: method (1) of (:31831) is the PIN, the ring-nitrogen ester a
    # carboxylic ester of the polyol like any other ('1,4-phenylene acetate dichloroacetate (PIN)')
    "O=C(OCCOS(C)(=O)=O)N1CCCCC1",
    "O=C(Oc1ccc(OS(C)(=O)=O)cc1)N1CCCCC1",
    "CC(=O)OCCOS(C)(=O)=O",
])
def test_polyol_sulfonates_by_acyloxy_prefixes_are_no_pin(smiles):
    row = tier_row(smiles, "pin")
    assert row["tier"] != "pin_verified", row


def test_sulfonyl_prefix_with_a_compound_acyl_takes_brackets():
    """'[4-(1H-imidazol-1-yl)benzene-1-sulfonyl]oxy': the acyl group in the next enclosing mark."""
    row = tier_row("CC(=O)Oc1ccc(OS(=O)(=O)c2ccc(-n3ccnc3)cc2)cc1C(=O)O", "best-effort")
    assert "{(" not in row["name"] and "(4-(" not in row["name"], row
