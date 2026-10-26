"""Precision leads of lane PREC2: names that were labelled pin_verified and are not the PIN, spellings of the
best-effort floor, the added hydrogen of a ring ketone, and a silane that took a principal group.

Every expected name is read back by OPSIN (``opsin_roundtrip_check``: the full InChIKey).

A. "Skeletal replacement ('a') nomenclature in acyclic chains" (heading ``## ****``:23346, text
   :23348): the 'a' name "must be used to generate preferred IUPAC names for acyclic structures when four or more
   heterounits are present in a unbranched chain containing at least one carbon atom and when none of the
   heteroatoms constitute all or part of the principal characteristic group". "Acyl halides from suffix
   acids" (:31412, text:31414): the acyl group, then "the name(s) of the specific class(es) as a separate word(s), in
   alphabetical order, each preceded by a multiplicative prefix, as needed"; 'butanedioyl bromide chloride (PIN)'
   (:31454); (:31460): 'butanedioyl isocyanide isothiocyanate (PIN)' (:31470). A halide beside a pseudohalide:
    (:5102, text:5104) and, "Halogen atoms are senior to pseudohalogen groups".
B. "Substituent prefixes derived from acyclic parent hydrides" (:15935) with "GENERAL METHODOLOGY
   FOR NAMING SUBSTITUENT GROUPS" (:15800): 'propan-2-yl (preferred prefix) 1-methylethyl' (:15830); (g)
   (:3307, heading:3219 " NUMBERING"): "lowest locants for the substituent cited first as a prefix in the
   name".
C. "'Added indicated hydrogen'" (:3725); (:24695) 'pyridin-1(2H)-yl (preferred prefix)'.
D. "ORGANOMETALLIC COMPOUNDS INVOLVING THE ELEMENTS OF GROUPS 13 THROUGH 16" (:39737): "named
   substitutively by prefixing the appropriate substituent names to the name of a parent hydride"; "SENIORITY
   ORDER FOR CLASSES" (:18158), Table 4.1: the suffix classes (1-20) are senior to the compound classes of the
   elements (21-43), a silicon compound being class 26.
"""
import pytest

from orthonym.cli import _tiered_namer
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


# --- A. the 'a' name of an acyl halide chain, -------------------------------------------------


@pytest.mark.parametrize("smiles,expected", [
    # the lead row: four ether oxygens in an unbranched chain between two acyl chlorides
    ("ClC(=O)COCCOCCOCCOCC(=O)Cl", "3,6,9,12-tetraoxatetradecanedioyl dichloride"),
    ("COCCOCCOCCOCC(=O)Cl", "2,5,8,11-tetraoxatridecan-13-oyl chloride"),
    ("CCOCCOCCOCCOCCC(=O)Cl", "3,6,9,12-tetraoxapentadecan-15-oyl chloride"),
    #: "in alphabetical order" ('butanedioyl bromide chloride (PIN)',:31454)
    ("BrC(=O)COCCOCCOCCOCC(=O)Cl", "3,6,9,12-tetraoxatetradecanedioyl bromide chloride"),
    ("FC(=O)COCCOCCOCCOCC(=O)F", "3,6,9,12-tetraoxatetradecanedioyl difluoride"),
    #: the pseudohalides, 'butanedioyl isocyanide isothiocyanate (PIN)' (:31470) in the same order
    ("O=C=NC(=O)COCCOCCOCCOCC(=O)N=C=O", "3,6,9,12-tetraoxatetradecanedioyl diisocyanate"),
    ("S=C=NC(=O)COCCOCCOCCOCC(=O)N=C=O", "3,6,9,12-tetraoxatetradecanedioyl isocyanate isothiocyanate"),
    ("COCCOCCOCCOCC(=O)N=[N+]=[N-]", "2,5,8,11-tetraoxatridecan-13-oyl azide"),
])
def test_four_heterounits_in_an_acyl_halide_chain_are_named_by_replacement(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == expected and row["tier"] == "pin_verified", (tier, row["name"])
    assert opsin_roundtrip_check(smiles, expected)["passed"], expected


def test_three_heterounits_keep_the_substitutive_name():
    """'s own boundary: below four heterounits the 'a' name is not the PIN (:27756)."""
    row = _row("ClC(=O)COCCOCCOCC(=O)Cl", "pin")
    assert "oxa" not in row["name"]


def test_the_acid_and_the_acyl_halide_of_one_chain_are_named_alike():
    acid = _row("OC(=O)COCCOCCOCCOCC(=O)O", "pin")["name"]
    halide = _row("ClC(=O)COCCOCCOCCOCC(=O)Cl", "pin")["name"]
    assert acid == "3,6,9,12-tetraoxatetradecanedioic acid"
    assert halide == acid.replace("ic acid", "yl dichloride")


def test_an_acid_beside_an_acyl_halide_is_not_named_by_the_halide_suffix():
    """: the acid (class 7) is senior to the acyl halide (class 10), so the chain is not 'dioyl'."""
    name = _row("ClC(=O)COCCOCCOCCOCC(=O)O", "best-effort")["name"]
    assert "dioyl" not in name and "oyl chloride" not in name


@pytest.mark.parametrize("smiles", ["CCOCCOCCOCCOC(=O)Cl", "CCOCCOCCOCCOC(=O)O"])
def test_a_carbonic_acid_derivative_is_not_a_carboxylic_acid_chain(smiles):
    """The acyl carbon of R-O-CO-X is bonded to oxygen, not carbon: it is an ester of carbonic acid, never the
    '-oic acid' / '-oyl' end of a heterochain."""
    name = _row(smiles, "best-effort")["name"]
    assert "tridecan-1-oic" not in name and "tridecan-1-oyl" not in name and "tetraoxatridecan" not in name


# --- A. a halide beside a pseudohalide, ---------------------------------------------------------


def test_a_halide_beside_a_pseudohalide_is_named_with_the_halide_as_the_class():
    """ "Seniority of functional classes" (heading:5102, text:5104): "When two classes are present and both
    named by functional class nomenclature, the senior class must be chosen... The seniority order of halides and
    pseudohalides is discussed in. The senior class is expressed by functional class nomenclature and the lower
    ranking classes are expressed by prefixes"; (:31460): "Halogen atoms are senior to pseudohalogen
    groups"; the book's parallel example is 'methyl 4-chloro-4-oxobutanoate (PIN)' (:5108). So the chloride is the
    class and the isocyanate a prefix (3),:31517): '4-isocyanato-4-oxobutanoyl chloride' is the PIN, at
    both tiers, not 'butanedioyl chloride isocyanate'."""
    smiles = "O=C=NC(=O)CCC(=O)Cl"
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == "4-isocyanato-4-oxobutanoyl chloride", (tier, row["name"])
        assert row["tier"] == "pin_verified" and row["is_pin"], tier
    assert opsin_roundtrip_check(smiles, "4-isocyanato-4-oxobutanoyl chloride")["passed"]


def test_one_class_on_both_ends_keeps_its_pin():
    for smiles, expected in (("O=C=NC(=O)CCC(=O)N=C=O", "butanedioyl diisocyanate"),
                             ("ClC(=O)CCC(=O)Br", "butanedioyl bromide chloride")):
        row = _row(smiles, "pin")
        assert row["name"] == expected and row["tier"] == "pin_verified"


# --- B. the prefix of a group whose free valence is not on the end of its chain ----------------------------------------


@pytest.mark.parametrize("smiles,expected", [
    ("Cc1nnc(C(C)C)n1C", "3,4-dimethyl-5-(propan-2-yl)-4H-1,2,4-triazole"),
    ("Cc1nnc(C(C)CC)n1C", "3-(butan-2-yl)-4,5-dimethyl-4H-1,2,4-triazole"),
    ("Cc1nnc(C(CC)CC)n1C", "3,4-dimethyl-5-(pentan-3-yl)-4H-1,2,4-triazole"),
    ("Cc1nnc(C(C)CCC)n1C", "3,4-dimethyl-5-(pentan-2-yl)-4H-1,2,4-triazole"),
    ("Cc1nnc(C(C)C(C)C)n1C", "3,4-dimethyl-5-(3-methylbutan-2-yl)-4H-1,2,4-triazole"),
    ("Cc1nnc(C(C)C)n1CC", "4-ethyl-3-methyl-5-(propan-2-yl)-4H-1,2,4-triazole"),
    # an end-of-chain free valence keeps the alkyl prefix:15922), and tert-butyl is retained
    ("Cc1nnc(CC(C)C)n1C", "3,4-dimethyl-5-(2-methylpropyl)-4H-1,2,4-triazole"),
    ("Cc1nnc(C(C)(C)C)n1C", "3-tert-butyl-4,5-dimethyl-4H-1,2,4-triazole"),
])
def test_the_chain_of_a_group_runs_through_its_free_valence(smiles, expected):
    row = _row(smiles, "best-effort")
    assert row["name"] == expected and row["producer"] == "engine:t4", row["name"]
    assert opsin_roundtrip_check(smiles, expected)["passed"], expected


def test_equal_numberings_are_decided_by_the_prefix_cited_first():
    """ (g): the two numberings of the ring give {3,4,5}; 'methyl' is cited before 'propan-2-yl', so it takes
    the lower locants whichever way the input is written."""
    names = {_row(s, "best-effort")["name"] for s in
             ("Cc1nnc(C(C)C)n1C", "CC(C)c1nnc(C)n1C", "Cn1c(C)nnc1C(C)C", "CC(C)C1=NN=C(C)N1C")}
    assert names == {"3,4-dimethyl-5-(propan-2-yl)-4H-1,2,4-triazole"}


# --- C. the added hydrogen of a ring ketone ----------------------------------------------------------------------------


@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)CN1C=Nc2ccccc2C1=O", "(4-oxoquinazolin-3(4H)-yl)acetic acid"),
    ("OC(=O)Cn1c(=O)ccc2ccccc21", "(2-oxoquinolin-1(2H)-yl)acetic acid"),
    ("OC(=O)Cn1ccc2ccccc2c1=O", "(1-oxoisoquinolin-2(1H)-yl)acetic acid"),
    ("OC(=O)Cc1ccc2c(=O)[nH]cnc2c1", "(4-oxo-3,4-dihydroquinazolin-7-yl)acetic acid"),
    ("OC(=O)Cn1c(=O)[nH]c2ccccc2c1=O", "(2,4-dioxo-1,4-dihydroquinazolin-3(2H)-yl)acetic acid"),
])
def test_a_ring_ketone_of_a_fused_substituent_cites_its_added_hydrogen(smiles, expected):
    row = _row(smiles, "best-effort")
    assert row["name"] == expected, row["name"]
    assert opsin_roundtrip_check(smiles, expected)["passed"], expected


def test_the_oxo_quinazolinyl_of_a_urea_cites_its_hydro_prefixes():
    smiles = "CCCCN(C(CC)c1nc2ccccc2c(=O)n1-c1ccc(Cl)cc1)C(=O)Nc1cccc(Br)c1"
    expected = ("N'-(3-bromophenyl)-N-butyl-N-{1-[3-(4-chlorophenyl)-4-oxo-3,4-dihydroquinazolin-2-yl]"
                "propyl}urea")
    row = _row(smiles, "best-effort")
    assert row["name"] == expected, row["name"]
    assert opsin_roundtrip_check(smiles, expected)["passed"], expected


# --- D. a Group 13-16 hub does not outrank a principal characteristic group --------------------------------------------


@pytest.mark.parametrize("smiles,expected", [
    ("NCCCO[Si](C)(C)C", "3-[(trimethylsilyl)oxy]propan-1-amine"),
    ("NCCO[Si](C)(C)C", "2-[(trimethylsilyl)oxy]ethan-1-amine"),
    ("OCCO[Si](C)(C)C", "2-[(trimethylsilyl)oxy]ethan-1-ol"),
    ("CC(C)(C)[Si](C)(C)OCCN", "2-{[tert-butyldi(methyl)silyl]oxy}ethan-1-amine"),
])
def test_the_amine_or_alcohol_of_a_silyl_ether_is_the_suffix(smiles, expected):
    row = _row(smiles, "best-effort")
    assert row["name"] == expected and row["producer"] != "outer:organometallic", row["name"]
    assert opsin_roundtrip_check(smiles, expected)["passed"], expected


def test_a_hub_with_no_principal_group_or_a_true_metal_keeps_the_organometallic_name():
    assert _row("CO[Si](C)(C)C", "pin")["name"] == "methoxytri(methyl)silane"
    assert _row("[Zn](CCN)CCN", "best-effort")["name"] == "bis(2-aminoethyl)zinc"


# --- fix a performance pass ---------------------------------------------------------------------------------------------------------


def test_the_acid_of_a_chain_that_ends_on_silicon_keeps_its_pin():
    """ (:23348, section ``## ****``:23346) and its exclusion of the acids "when representing the
    parent compound or the principal group" (:23350) concern carbonic acid and the P, As, Sb acids; a chain that ends
    on Si is licensed,:6465: the chain ends on C or on P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, Tl).
    Only the neighbours that make a carbonic acid derivative (O, N, S, Se, Te) are no chain."""
    smiles = "OC(=O)[SiH2]CCOCCOCCOCCOC"
    expected = "2,5,8,11-tetraoxa-14-silapentadecan-15-oic acid"
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == expected and row["tier"] == "pin_verified", (tier, row["name"])
    assert opsin_roundtrip_check(smiles, expected)["passed"]


def test_a_substituent_chain_does_not_end_on_nitrogen():
    """ (:6465): a heterochain ends on C or on one of P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, Tl. The
    chain through the free valence of the thiohydroximate keeps its carbon chain and takes the imine as a prefix,
    as before ('6-oxo-1-[(sulfonatooxy)imino]-6-thiaheptan-1-yl'), not '...-1-azaoct-1-en-2-yl'."""
    smiles = "CS(=O)CCCC/C(=N/OS(=O)(=O)[O-])S[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"
    row = _row(smiles, "best-effort")
    assert "6-oxo-1-[(sulfonatooxy)imino]-6-thiaheptan-1-yl" in row["name"] and "azaoct" not in row["name"]
    assert opsin_roundtrip_check(smiles, row["name"])["passed"]


GLYCOPEPTIDE = (
    "CC(C)C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CCCCN)NC(=O)[C@@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CCCCN)NC(=O)CNC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)CCSCCC(=O)OCCOCCOCCOCCC(=O)NCCSCCCO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]1O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]1O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]1O)C(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](C)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CCC(=O)O)C(=O)NCCOCCOCCOCCC(=O)N[C@@H](CCCCN=[N+]=[N-])C(=O)O"
)


def test_a_licensed_heterochain_is_not_lost_to_a_chain_with_a_carbonyl_oxygen_end():
    """The longest chain through the free valence used to take a carbonyl oxygen as its short arm: a chain that ends on
    O is not licensed,:6465), so the engine fell back to the carbon chain and lost its 'a' chain
    ,:23348; (a),:20916, the most heteroatoms). An arm is cut back to a permitted end atom."""
    row = _row(GLYCOPEPTIDE, "best-effort")
    assert "20,23,26-trioxa-5,8,11,14,17,30-hexaazapentatriacontan-3-yl" in row["name"]
    assert row["tier"] == "systematic_verified"


def test_a_prefix_with_a_multiplied_group_on_a_cation_is_cited_at_its_first_letter():
    """ (:3477): a compound prefix "is considered to begin with the first letter of its complete name"
    ('dimethyl[6-(trimethylazaniumyl)hexyl]sulfanium',:42360): 'trimethylazaniumyl' is cited at 't', not at 'm'."""
    from orthonym.assembly.naming_utils import alpha_sort_key
    for name in ("trimethylazaniumyl", "triphenylphosphaniumyl", "dimethylsulfaniumyl", "dimethylboranyl"):
        assert str(alpha_sort_key(name)) == name, name
    assert str(alpha_sort_key("dimethyl")) == "methyl"                  #: a multiplier of simple prefixes
    smiles = "C[N+](C)(C)CC(CC(=O)[O-])OC(=O)CCCCCCCCCCCCCCCCC(=O)O"
    expected = "1-hydroxy-18-{[4-oxido-4-oxo-1-(trimethylazaniumyl)butan-2-yl]oxy}-1,18-dioxooctadecane"
    assert _row(smiles, "best-effort")["name"] == expected
    assert opsin_roundtrip_check(smiles, expected)["passed"]


def test_pseudohalide_words_whose_alphabetical_and_seniority_orders_differ_are_not_certified():
    """ (:31460) orders the groups N3 > CN > NC > NCO > NCS > NCSe > NCTe for the choice and prints no order
    of citation. Where alphabetical order is the order of seniority the pair keeps its PIN ('cyanide isocyanate',
    'isocyanate isothiocyanate'); 'isocyanate isocyanide' (alphabetical) and 'isocyanide isocyanate' (seniority)
    differ, so the name is labelled pin_unverified."""
    from orthonym.metrics import provenance as prov
    from orthonym.rules.skeletal_replacement import _acyl_class_words
    prov.clear_provenance()
    assert _acyl_class_words(["isocyanate", "isothiocyanate"]) == "isocyanate isothiocyanate"
    assert _acyl_class_words(["isocyanate", "cyanide"]) == "cyanide isocyanate"
    assert not prov._UNCERTIFIED_PIN_NAMES.get()
    assert _acyl_class_words(["isocyanide", "isocyanate"]) == "isocyanate isocyanide"
    assert "isocyanate isocyanide" in prov._UNCERTIFIED_PIN_NAMES.get()
    prov.clear_provenance()
    assert opsin_roundtrip_check("N#CC(=O)COCCOCCOCCOCC(=O)N=C=O",
                                 "3,6,9,12-tetraoxatetradecanedioyl cyanide isocyanate")["passed"]
