"""N-acyl ring nitrogens ('hidden' amides) named as pseudoketones (``rules.pseudoketones``).

 "'Hidden' amides" (the Blue Book): "An N-acyl group attached to a nitrogen atom of a heterocyclic system
is now preferably named as a pseudoketone (see, and not as an acyl substituent on the nitrogen as in
previous recommendations": '1-(piperidin-1-yl)ethan-1-one (PIN)' (:33129). A ring acyl group gives the methanone with
both groups as a mononuclear parent ('phenyl(tetrasulfanyl)methanone (PIN)',:24934;:7272). A group
senior to a ketone, or a second ketone, keeps its own parent.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)N1CCCCC1", "1-(piperidin-1-yl)ethan-1-one"),                         #:33129, unchanged
    ("O=C(c1ccccc1)N1CCCC1", "phenyl(pyrrolidin-1-yl)methanone"),
    ("O=C(c1ccccc1)N1CCOCC1", "(morpholin-4-yl)(phenyl)methanone"),
    ("O=C(C1CCCCC1)N1CCCCC1", "cyclohexyl(piperidin-1-yl)methanone"),
    ("O=C(c1ccncc1)N1CCCCC1", "(piperidin-1-yl)(pyridin-4-yl)methanone"),       # was '4-(piperidine-1-carbonyl)pyridine'
    ("O=C(c1ccc(Cl)cc1)N1CCCCC1", "(4-chlorophenyl)(piperidin-1-yl)methanone"),
    ("CC(=O)N1CCN(CC1)c1ccccc1", "1-(4-phenylpiperazin-1-yl)ethan-1-one"),
    ("CC(=O)N1CCC(O)CC1", "1-(4-hydroxypiperidin-1-yl)ethan-1-one"),
    ("S=C(c1ccccc1)N1CCCC1", "phenyl(pyrrolidin-1-yl)methanethione"),
])
def test_pin(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


@pytest.mark.parametrize("smiles", [
    "N#Cc1ccc(cc1)C(=O)N1CCCCC1",     # the nitrile is the principal group
    "CC(=O)N1CCC(CC1)C=O",            # the aldehyde too
    "CC(=O)N1CCC(=O)CC1",             # a second ketone: the ring parent of
])
def test_no_pseudoketone_beside_a_senior_group(smiles):
    from orthonym.rules.pseudoketones import name_pseudoketone
    from tests.support.pin_tiers import tier_row as _row
    row = _row(smiles, "pin")
    assert "methanone" not in (row["name"] or "") and "ethan-1-one" not in (row["name"] or ""), row


@pytest.mark.parametrize("smiles,expected", [
    # a carbamate is an ester (class 9), senior to the pseudoketone (a ketone, class 16;:18180-18190)
    ("CC(=O)N1CCC(CC1)NC(=O)OC(C)(C)C", "tert-butyl (1-acetylpiperidin-4-yl)carbamate"),
    ("O=C(c1ccccc1)N1CCN(CC1)C(=O)OCc1ccccc1", "benzyl 4-benzoylpiperazine-1-carboxylate"),
    ("CC(=O)N1CCN(CC1)C(=O)OCC", "ethyl 4-acetylpiperazine-1-carboxylate"),
])
def test_a_carbamate_keeps_the_parent(smiles, expected):
    for tier in ("pin", "best-effort"):
        assert tier_row(smiles, tier)["name"] == expected


@pytest.mark.parametrize("smiles,expected", [
    # (:7442) nesting {}: a ring prefix that holds marks of its own takes brackets
    ("CC(=O)N1CCC[C@H]1C", "1-[(2R)-2-methylpyrrolidin-1-yl]ethan-1-one"),
    ("CC(=O)N1CCN(CC1)c1ccc(Cl)cc1", "1-[4-(4-chlorophenyl)piperazin-1-yl]ethan-1-one"),
])
def test_nesting(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row
