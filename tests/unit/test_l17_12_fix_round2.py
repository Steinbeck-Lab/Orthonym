"""Second fix round of the merged lane/l17 (item 12): the oxoacid ester beside a lactone, a lactam or a sultone, the
sultone prefix, the acyl name of a ketone with an amido prefix, the cyano prefix of a chain, and the class of an ester beside
the ester of sulfuric acid.

 SENIORITY ORDER FOR CLASSES, Table 4.1 (the Blue Book): "9 Esters (functional class names are given to noncyclic
esters; lactones and other cyclic esters are named as heterocycles; see 16 below)" and "16 Ketones... pseudoketones", with
the note "See Classes 8, 9, 11 and 13, on lactones, lactams, anhydrides, imides" (:18189): a lactone, a lactam and a sultone
are pseudoketones, below every acyclic ester, so the ester of sulfuric or phosphoric acid is the principal group beside them
 "Esters of mononuclear noncarbon oxoacids",:35916) and the ring is cited as a prefix of its organyl group.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import assert_full_rt
from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _principal(smiles):
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.seniority import get_principal_group
    mol = Chem.MolFromSmiles(smiles)
    return get_principal_group(mol, detect_functional_groups(mol))[0]


# ---- I1: the ester of an oxoacid is the parent beside a pseudoketone ring, at every tier -------------------------------

_ESTER_OVER_RING = [
    ("O=C1CCCN1CCOS(=O)(=O)O", "2-(2-oxopyrrolidin-1-yl)ethyl hydrogen sulfate"),            # a lactam
    ("O=C1CCCN1CCOP(=O)(O)O", "2-(2-oxopyrrolidin-1-yl)ethyl dihydrogen phosphate"),
    ("O=C1OCCC1CCOS(=O)(=O)O", "2-(2-oxooxolan-3-yl)ethyl hydrogen sulfate"),                # a lactone
    ("O=C1OCCC1CCOP(=O)(O)O", "2-(2-oxooxolan-3-yl)ethyl dihydrogen phosphate"),
]


@pytest.mark.parametrize("smiles,name", _ESTER_OVER_RING)
def test_an_oxoacid_ester_is_the_parent_beside_a_lactam_or_a_lactone(smiles, name):
    """ (:18182,:18189): the acyclic ester is class 9, the lactam and the lactone class 16; the book's
    '2-oxopyrrolidin-1-yl' / '2-oxooxolan-3-yl' prefixes are those of a pseudoketone ring "In cyclic
    pseudoketones, the prefix 'oxo'... [is] used to denote a carbonyl group"). The lactam handler named these
    '1-[2-(sulfooxy)ethyl]pyrrolidin-2-one' at the best-effort tier (and declined at the PIN tier for the phosphate), the
    lactone one '3-[2-(sulfooxy)ethyl]oxolan-2-one', while the PIN tier gave the ester name for the first.

    change-asserted-value: the old best-effort names were systematic_verified, the new are the PIN of the book's class
    order, pin_verified at both tiers (the PIN tier gave the first of them already; the phosphates and the lactone were
    abstentions or the sulfooxy name). Independent check: ``assert_full_rt`` reads each name back with a fresh OPSIN
    call to the input's full InChIKey. Mutation: with ``has_senior_group_outside`` (the lactone and lactam handlers'
    guard) as it was and ``_promoted_oxoacid_ester`` asking the head group alone, the best-effort tier names the old
    spelling and these rows fail."""
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert row["name"] == name, row
        assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(name, smiles)


def test_the_principal_group_of_an_oxoacid_ester_beside_a_lactone_is_the_ester():
    """A lactone head is a pseudoketone "Cyclic anhydrides, esters and amides are named as pseudoketones",
    the Blue Book) and ranks as a ketone, below the esters of sulfuric and phosphoric acid; the head group that is only a
    cyclic ester no longer blocks the promotion (it was ranked as the carboxylic ester it is, class 9, senior to these)."""
    assert _principal("O=C1OCCC1CCOS(=O)(=O)O") == "sulfate_monoester"
    assert _principal("O=C1OCCC1CCOP(=O)(O)O") == "phosphate_monoester"
    assert _principal("O=C1CCCN1CCOS(=O)(=O)O") == "sulfate_monoester"
    # an acyclic carboxylic ester beside the lactone is the head, which the esters of the oxoacids do not outrank
    assert _principal("O=C1OCCC1CCOC(C)=O") == "ester"
    # a lactone alone, or with the ester of the oxoacid declined (a stereo layer is not offered), keeps its place
    assert _principal("O=C1OCCC1C") == "ester"
    assert _principal("O=C1OCC[C@@H]1CCOS(=O)(=O)O") == "ester"


# ---- I3: the Hantzsch-Widman name of a sultone ring is the prefix of its group ----------------------------------------

_SULTONE_PREFIXES = [
    ("CC(=O)NCC1CCOS1(=O)=O", "N-[(2,2-dioxo-1,2λ6-oxathiolan-3-yl)methyl]acetamide"),
    ("CC(=O)N(C)CC1CCOS1(=O)=O", "N-[(2,2-dioxo-1,2λ6-oxathiolan-3-yl)methyl]-N-methylacetamide"),
    ("CC(=O)NCC1CCCOS1(=O)=O", "N-[(2,2-dioxo-1,2λ6-oxathian-3-yl)methyl]acetamide"),
    ("CC(=O)NCC1COS(=O)(=O)O1", "N-[(2,2-dioxo-1,3,2λ6-dioxathiolan-4-yl)methyl]acetamide"),
]


@pytest.mark.parametrize("smiles,name", _SULTONE_PREFIXES)
def test_the_sultone_ring_prefix_is_the_hantzsch_widman_name(smiles, name):
    """ "Heteroacyclic and heteromonocycles", (the Blue Book): "Preferred IUPAC names for
    heteromonocyclic rings with no more than ten ring members are Hantzsch-Widman names"; "Heteromonocyclic
    hydrides having heteroatoms with nonstandard bonding numbers" (:9158): "The λ-convention is used to denote heteroatoms
    with nonstandard bonding numbers in heteromonocycles... The symbol λ^n... is cited immediately after the locant denoting
    the heteroatom with the nonstandard bonding number" ('1,2λ6-oxathiolane', the ring namer's name of the sultone:
    'OCC1CCOS1(=O)=O' -> '3-(hydroxymethyl)-1,2λ6-oxathiolane-2,2-dione'). The prefix of the ring was the
    skeletal replacement name '2,2-dioxo-1-oxa-2λ6-thiacyclopentan-3-yl' (a von Baeyer-style spelling for a ring the
    Hantzsch-Widman rule names), because the monocycle form of the substituent namer declined every ring atom of another
    bonding number.

    change-asserted-value: '(2,2-dioxo-1-oxa-2λ6-thiacyclopentan-3-yl)' -> '(2,2-dioxo-1,2λ6-oxathiolan-3-yl)'; the label
    moves from systematic_verified (the replacement spelling is recorded as not a PIN) to pin_unverified (a name in the PIN
    form that the PIN tier does not build). Independent check: a fresh OPSIN call reads each new name to the input's full
    key. Mutation: with ``allow_lambda`` off in ``_book_monocycle_core_tail`` the old spelling returns and these rows fail."""
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    assert row["tier"] == "pin_unverified" and not row["is_pin"], row
    assert_full_rt(name, smiles)


def test_the_ring_name_of_the_sultone_is_the_namers_own():
    row = tier_row("OCC1CCOS1(=O)=O", "pin")
    assert row["name"] == "3-(hydroxymethyl)-1,2λ6-oxathiolane-2,2-dione", row


# ---- the acyl name of a ketone whose chain carries an amido prefix -----------------------------------------------------

_KETONE_ACYL = [
    ("CC(=O)N(C)CC(=O)c1ccc(OS(=O)(=O)O)cc1", "4-[(N-methylacetamido)acetyl]phenyl hydrogen sulfate"),
    ("CC(=O)NCC(=O)c1ccc(OS(=O)(=O)O)cc1", "4-(acetamidoacetyl)phenyl hydrogen sulfate"),
    ("CC(=O)N(C)CCC(=O)c1ccc(OS(=O)(=O)O)cc1", "4-[3-(N-methylacetamido)propanoyl]phenyl hydrogen sulfate"),
    ("O=C(CNC(=O)c1ccccc1)c1ccc(OS(=O)(=O)O)cc1", "4-(benzamidoacetyl)phenyl hydrogen sulfate"),
]


@pytest.mark.parametrize("smiles,name", _KETONE_ACYL)
def test_a_ketone_acyl_with_an_amido_prefix_is_named_by_the_acyl_name(smiles, name):
    """ (the Blue Book) "substitution on acyl groups is identical to that of acids... Preferred IUPAC names
    are systematic substitutive names" ('chloroacetyl', (chloroacetyl)oxyl (PIN),:40694); the nitrogen of the chain is the
    amido prefix of (:32991, 'Method (1) generates preferred IUPAC names'). The same acyl group on a benzoic
    acid is '4-[(N-methylacetamido)acetyl]benzoic acid' (a PIN of the ring parent); on the owner of an ester of sulfuric
    acid the consumer of the amido prefix left every C=O to an acyl tier that had no such prefix, and the group was
    '2-[acetyl(methyl)amino]-1-oxoethyl' (best-effort) or no name (PIN tier).

    change-asserted-value: '4-{2-[acetyl(methyl)amino]-1-oxoethyl}phenyl hydrogen sulfate' (systematic_verified) and the
    abstentions of the PIN tier -> the names above, pin_verified at both tiers. Independent check: a fresh OPSIN call.
    Mutation: with ``chain_amide_prefixes._is_ketone_acyl`` returning False the old spelling returns."""
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert row["name"] == name, row
        assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(name, smiles)


# ---- the cyano prefix of a chain: one label for the ester of sulfuric acid with a cyano group on its organyl group -------

_CYANO_CHAINS = [
    ("N#CCCOS(=O)(=O)O", "2-cyanoethyl hydrogen sulfate"),
    ("N#CCCCOS(=O)(=O)O", "3-cyanopropyl hydrogen sulfate"),
    ("N#CCCOP(=O)(O)O", "2-cyanoethyl dihydrogen phosphate"),
    ("N#CCCOC(C)=O", "2-cyanoethyl acetate"),
    ("N#CC(O)COC(C)=O", "2-cyano-2-hydroxyethyl acetate"),
    ("N#CCC(C#N)COC(C)=O", "2,3-dicyanopropyl acetate"),
    ("N#CCCOc1ccccc1C(=O)O", "2-(2-cyanoethoxy)benzoic acid"),
    ("N#Cc1ccc(OS(=O)(=O)O)cc1", "4-cyanophenyl hydrogen sulfate"),
]


@pytest.mark.parametrize("smiles,name", _CYANO_CHAINS)
def test_the_cyano_prefix_of_a_chain_substituent_is_a_pin_prefix(smiles, name):
    """ (the Blue Book): "the -CN group is designated by the preferred prefix 'cyano'"; the substituent
    group of a chain that carries it is '2-cyanoethyl' ('S-(2-cyanoethyl) cyclohexanesulfinothioate (PIN)',:31765),
    numbered from the free valence. The polyfunctional chain producer consumed the carboxy group as a prefix and
    declined every C#N ('C#C / C#N / C=N... -> decline'), so the PIN tier had no name for a 2-cyanoethyl group
    ('2-cyanoethyl acetate', 'N#CCCOS(=O)(=O)O') and the best-effort tier named it 'pin_unverified' (the strict path did not
    build it), while the aryl ('4-cyanophenyl hydrogen sulfate') and the one-carbon ('cyanomethyl') group were
    pin_verified.

    change-asserted-value: pin abstain / best-effort pin_unverified -> pin_verified at both tiers (the rows of the left
    column of the list; the last row is the unchanged control). Independent check: a fresh OPSIN call reads each name to the
    input's full key. Mutation: without Pass 1a of ``_name_polyfunctional_acyclic_substituent_impl`` the rows fail."""
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert row["name"] == name, row
        assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(name, smiles)


# ---- the class of an ester beside the ester of sulfuric acid --------------------------------------------------------

def test_a_carboxylic_ester_ranks_above_the_ester_of_sulfuric_acid():
    """ "When anions are different, two methods are used" (the Blue Book): "(2) one anion is chosen as
    principal anion and all other ester groups are expressed as prefixes in the name of the organyl group. The seniority
    order of anions is corresponding to that of acids (see seniority order of acids in ", so the anion of acetic acid
    (class 7a) is the principal anion beside that of sulfuric acid (class 7d): '2-(sulfooxy)ethyl acetate'; the book's own
    name of a phosphatidylethanolamine is a diester of the carboxylic acid with the phosphate in the prefix
    Phosphatidylethanolamine,:55180 '(2R)-3-{[(2-aminoethoxy)hydroxyphosphoryl]oxy}propane-1,2-diyl dihexadecanoate'). A hydrogen
    sulfate keeps an acid hydroxy group, but (:35916) names the partial ester of an inorganic acid as an ester
    (by the method of 'hydrogen salts',,:40993), not as an acid, so it is no class 7 acid that ranks above the
    class 9 ester. The name is method (2); method (1), 'ethane-1,2-diyl acetate hydrogen sulfate' (PIN), is not read by OPSIN
    2.9.0 and is not built (``esters.record_polyol_mixed_anion_non_pin`` labels method (2) below the PIN)."""
    assert _principal("CC(=O)OCCOS(=O)(=O)O") == "ester"
    row = tier_row("CC(=O)OCCOS(=O)(=O)O", "best-effort")
    assert row["name"] == "2-(sulfooxy)ethyl acetate", row
    assert row["tier"] == "systematic_verified" and not row["is_pin"], row
    assert_full_rt(row["name"], "CC(=O)OCCOS(=O)(=O)O")


# ---- I2: the suffix of a ring ester stays where the amide of a chain is the principal group ---------------------------

_DEPSIPEPTIDES = [
    ("CC(=O)N[C@@H](CC(C)C)C(=O)N[C@@H]1C(=O)N[C@H](Cc2ccc(O)cc2)C(=O)N[C@H](C)C(=O)N[C@@H](C)C(=O)N[C@H](C)C(=O)"
     "N[C@@H](C(C)C)C(=O)O[C@@H]1C",
     "(2S)-2-acetamido-N-{(3S,6R,9S,12R,15R,18S,19R)-15-[(4-hydroxyphenyl)methyl]-6,9,12,19-tetramethyl-2,5,8,11,14,17-"
     "hexaoxo-3-(propan-2-yl)-1-oxa-4,7,10,13,16-pentaazacyclononadecan-18-yl}-4-methylpentanamide", "systematic_verified"),
    ("CCCCCCCCCCCC(O)CC(=O)NC(CC(C)C)C(=O)N[C@@H]1C(=O)N[C@H](CO)C(=O)N[C@@H]([C@H](C)O)C(=O)NC(CC(C)C)C(=O)"
     "NC(CC(C)C)C(=O)O[C@H]1C",
     "3-hydroxy-N-[1-({(9S,12R,15S,16S)-9-[(1S)-1-hydroxyethyl]-12-(hydroxymethyl)-16-methyl-3,6-bis(2-methylpropyl)-2,5,8,11,"
     "14-pentaoxo-1-oxa-4,7,10,13-tetraazacyclohexadecan-15-yl}carbamoyl)-3-methylbutyl]tetradecanamide", "pin_unverified"),
]


@pytest.mark.parametrize("smiles,name,tier", _DEPSIPEPTIDES)
def test_the_amide_of_a_chain_is_the_parent_of_a_cyclic_depsipeptide(smiles, name, tier):
    """ (:18184,:18189): a chain amide (class 11) is senior to the lactone and the lactams of the ring
    (class 16), so the ring is a prefix of the amide parent with its oxo groups: at the commit that let the ring esters
    step aside (``) these molecules were named on the ring with every group cited as a prefix and no suffix
    ('...-hexaoxo-...-1-oxa-...cyclononadecane', the amide an amido prefix), because the amide parent was not built; the
    amide-parent selection of item 7 (``, builds it."""
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    assert row["tier"] == tier, row
    assert name.endswith("amide"), name      # the principal group is the suffix of the parent, not a prefix of a ring
    assert_full_rt(name, smiles)
