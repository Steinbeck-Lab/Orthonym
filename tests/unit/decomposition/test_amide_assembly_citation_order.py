"""The amide and sulfonamide assemblers of the decomposition engine write 'N-<prefix>' in
front of the other fragment's name; such a name keeps the Blue Book's citation order only
when that name has no stereodescriptor block at its front and no alphanumerically ordered
prefix that sorts before the N-prefix (``fragment_assembly.n_prefix_breaks_citation_order``).
Otherwise the name still ships at the wider tiers but is labelled below the PIN, and the
default tier declines it.

 (the Blue Book) "Stereodescriptors placed at the front of the complete name
or name fragment to which they apply"; (:3448) "Simple prefixes... are arranged
alphabetically; multiplicative prefixes, if necessary, are then inserted"; (:3477)
"The name of a prefix for a substituent is considered to begin with the first letter of its
complete name", so of two prefixes one of which begins with the other the shorter is cited
first (:21663 '4-methyl-3-methylidenehexanoic acid (PIN)'); (:3517) "priority for
order of citation is given to the group that contains the lowest locant(s) at the first point
of difference" ('4-(2-methylbutyl)-N-(3-methylbutyl)aniline (PIN)',:3523); (:3495)
italic letters only where the Roman letters tie ('1-(butan-2-yl)-3-tert-butylbenzene (PIN)',
:3507); (:3440) hydro and dehydro are not part of the alphanumerical order.

The measured positives: four morphinan rows that the bridged fused slice S4 makes nameable
(its rule-derived parent names the acid fragment), and one milestone1500 row the base
already shipped pin_verified."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.decomposition.fragment_assembly import n_prefix_breaks_citation_order
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

_P = "4,12-methano[1]benzofuro[3,2-e]isoquinoline"

#: (N-prefixes, the name they are put in front of, breaks the order)
ORDER_CASES = [
    #: the N-prefix would stand in front of the stereodescriptors
    (["prop-2-en-1-yl"], "(4R,4aS,6S,7S,7aR,12bS)-9-(benzyloxy)-3-(cyclopropylmethyl)-4a,7-"
     f"dihydroxy-2,3,4,4a,5,6,7,7a-octahydro-1H-{_P}-6-carboxamide", True),
    (["methyl"], "(2R)-2-hydroxypropanamide", True),
    # /: a prefix of the name sorts before the N-prefix
    (["phenyl"], "3-(cyclopropylmethyl)-7-hydroxy-9-methoxy-1,2,3,4,5,6,7,7a-octahydro-"
     f"4a,7-(epoxymethano)-{_P}-14-carboxamide", True),
    (["cyclohexyl"], "3-acetamido-9-azabicyclo[3.3.1]nonane-9-carboxamide", True),
    ([f"[(4R,4aS,7aR,12bS)-3-(cyclopropylmethyl)-4a,7,9-trihydroxy-2,3,4,4a,5,7a-hexahydro-"
      f"1H-{_P}-6-carbonyl]"], "2-chlorobenzene-1-sulfonamide", True),
    (["methyl"], "4-chlorobenzamide", True),
    (["phenyl"], "3-oxobutanamide", True),
    (["propyl"], "N-methylcyclohexanamine", True),
    (["phenyl"], "2,2-bis(2-chloroethyl)acetamide", True),
    # identical letters: multiplies them ('N,2-dimethylpentanamide')
    (["methyl"], "2-methylpentanamide", True),
    (["methyl"], f"3-methyl-2,3,4,4a,5,6,7,7a-octahydro-1H-{_P}-7-carboxamide", True),
    (["cyclopropyl"], "1-cyclopropylmethanamine", True),
    #: an N-prefix that begins with the name's first prefix sorts after it
    (["ethylidene"], "2-ethylpentanamide", True),
    (["2-(hydroxymethyl)propanoyl"], "3-hydroxypiperidin-2-one", True),
    #: identical letters, the lower locant first ('4-(2-methylbutyl)-N-(3-methylbutyl)')
    (["3-methylbutyl"], "4-(2-methylbutyl)benzamide", True),
    #: 'tert-butyl' is ordered at 'b'
    (["methyl"], "4-tert-butylbenzamide", True),
    # in order
    (["ethyl"], f"3-methyl-2,3,4,4a,5,6,7,7a-octahydro-1H-{_P}-7-carboxamide", False),
    (["benzyl"], "2-methylpentanamide", False),
    (["tert-butyl"], "2-[2-(3-methylphenoxy)propanamido]benzamide", False),
    (["ethyl"], "2-hydroxybenzamide", False),
    (["methyl"], "3-oxobutanamide", False),
    (["acetyl"], "N-methylcyclohexanamine", False),
    (["ethyl"], "furan-2-carboxamide", False),
    #: a prefix of the name that begins with the N-prefix's letters sorts after it
    (["cyclopropyl"], f"3-(cyclopropylmethyl)-2,3,4,4a,5,6,7,7a-octahydro-1H-{_P}-7-carboxamide",
     False),
    (["phenyl"], "4-(phenylsulfanyl)benzamide", False),
    (["methyl"], "4-(methylsulfanyl)benzamide", False),
    (["methyl"], "2-methylidenebutanamide", False),
    (["propyl"], "2-propylidenepentanamide", False),
    (["nitro"], "4-nitrosobenzamide", False),
    #: identical letters, the N-prefix has the lower locant
    (["2-methylbutyl"], "4-(3-methylbutyl)benzamide", False),
    #: the multiplying prefix of a compound prefix does not alter its order
    (["butyl"], "2,2-bis(2-chloroethyl)acetamide", False),
    #: 'tert-butyl' at 'b', after 'acetyl'
    (["acetyl"], "4-tert-butylcyclohexan-1-amine", False),
    (["methyl"], "1H-indole-3-carboxamide", False),
    # no alphanumerically ordered prefix: hydro,:3440), bridges, skeletal
    # replacement, and parent names that begin with heteroatom locants
    (["phenyl"], "1,2,3,4-tetrahydronaphthalene-1-carboxamide", False),
    (["phenyl"], "2,3-dihydro-1,4-benzodioxine-2-carboxamide", False),
    (["methyl"], f"4a,7-ethano-{_P}-6-carboxamide", False),
    (["phenyl"], "2-oxa-6-azaspiro[3.3]heptane-6-carboxamide", False),
    (["phenyl"], "1-azabicyclo[2.2.2]octane-3-carboxamide", False),
    (["methyl"], "1,3-thiazole-4-carboxamide", False),
    (["phenyl"], "1,3-dioxolane-2-carboxamide", False),
    (["phenyl"], "1,4-dioxane-2-carboxamide", False),
    (["phenyl"], "1,2,4-triazole-3-carboxamide", False),
    (["phenyl"], "1,2,4-oxadiazole-3-carboxamide", False),
    (["phenyl"], "1-benzofuran-2-carboxamide", False),
    (["phenyl"], "1,3-benzodioxole-5-carboxamide", False),
    (["phenyl"], "2,2'-bipyridine-4-carboxamide", False),
    (["methyl"], "1,1':4',1''-terphenyl-4-carboxamide", False),
]


@pytest.mark.parametrize("n_prefixes,name,breaks", ORDER_CASES)
def test_the_citation_order_of_an_n_prefix_put_in_front_of_a_name(n_prefixes, name, breaks):
    assert n_prefix_breaks_citation_order(n_prefixes, name) is breaks


#: (SMILES, the assembled name the wider tiers keep, labelled below the PIN)
BELOW_PIN = [
    ("C=CCNC(=O)[C@H]1C[C@]2([C@H]3CC4=C5[C@@]2(CCN3CC6CC6)[C@H]([C@H]1O)OC5=C(C=C4)OCC7=CC=CC=C7)O",
     "N-(prop-2-en-1-yl)(4R,4aS,6S,7S,7aR,12bS)-9-(benzyloxy)-3-(cyclopropylmethyl)-4a,7-"
     f"dihydroxy-2,3,4,4a,5,6,7,7a-octahydro-1H-{_P}-6-carboxamide"),
    ("CC(C)NC(=O)C1=C([C@H]2[C@@]34CCN([C@@H]([C@@]3(C1)O)CC5=C4C(=C(C=C5)O)O2)CC6CC6)C#CC7=CC=CC=C7",
     "N-(propan-2-yl)(4R,4aS,7aS,12bS)-3-(cyclopropylmethyl)-4a,9-dihydroxy-7-(2-phenylethynyl)-"
     f"2,3,4,4a,5,7a-hexahydro-1H-{_P}-6-carboxamide"),
    ("COC1=C2C3=C(CC4C56C3(CCN4CC7CC7)C(O2)C(CC5)(C(O6)C(=O)NC8=CC=CC=C8)O)C=C1",
     "N-phenyl-3-(cyclopropylmethyl)-7-hydroxy-9-methoxy-1,2,3,4,5,6,7,7a-octahydro-4a,7-"
     f"(epoxymethano)-{_P}-14-carboxamide"),
    # roadmap N5e (name-quality lane L2): the acyl group is cited by its acyl name and
    # the best-effort tier names the row in citation order ('chloro' before the complete
    # name of the N-prefix, the Blue Book). The name has no part the label
    # guard records as not the PIN, so it is labelled pin_unverified (``BEST_EFFORT_TIER``),
    # is_pin False. It is not in PIN form: amides rank "in the order of the corresponding
    # acids" class 11,:18184;:33559), carboxylic before sulfonic
    #:18304,:18307), so the PIN is named on the ring carboxamide, a parent choice
    # this producer does not make. The default tier still declines it. Was
    # 'N-[(4R,4aS,7aR,12bS)-...-6-carbonyl]-2-chlorobenzene-1-sulfonamide—methane (1/1)'.
    ("C.C1CC1CN2CC[C@]34[C@@H]5C(=C(C[C@]3([C@H]2CC6=C4C(=C(C=C6)O)O5)O)C(=O)NS(=O)(=O)C7=CC=CC=C7Cl)O",
     "2-chloro-N-[(4R,4aS,7aR,12bS)-3-(cyclopropylmethyl)-4a,7,9-trihydroxy-2,3,4,4a,5,7a-"
     f"hexahydro-1H-{_P}-6-carbonyl]benzene-1-sulfonamide—methane (1/1)"),
    ("CN1CCC23c4c5cccc4OC2C(C(=O)NCCC)CCC3C1C5",
     f"N-propyl-3-methyl-2,3,4,4a,5,6,7,7a-octahydro-1H-{_P}-7-carboxamide"),
]
#: the best-effort label of a row whose name is in citation order (systematic_verified for
#: the others)
BEST_EFFORT_TIER = {BELOW_PIN[3][0]: "pin_unverified"}
#: shipped pin_verified at the base (milestone1500); best-effort names it otherwise
BASE_ROW = "CC(=O)NC1CC2CCCC(C1)N2C(=O)NC3CCCCC3"
#: assembled names in citation order keep the PIN label (the second: 'cyclopropyl' before
#: 'cyclopropylmethyl',
IN_ORDER = [
    ("CN1CCC23c4c5cccc4OC2C(C(=O)NCC)CCC3C1C5",
     f"N-ethyl-3-methyl-2,3,4,4a,5,6,7,7a-octahydro-1H-{_P}-7-carboxamide"),
    ("C1CC1CN1CCC23c4c5cccc4OC2C(C(=O)NC4CC4)CCC3C1C5",
     f"N-cyclopropyl-3-(cyclopropylmethyl)-2,3,4,4a,5,6,7,7a-octahydro-1H-{_P}-7-carboxamide"),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="amide-order"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", BELOW_PIN)
def test_the_default_tier_declines_an_assembled_name_out_of_citation_order(smiles, name):
    row = _row(smiles, "pin")
    assert (row["tier"], row.get("limit_code")) == ("abstain", "NO_VERIFIED_PIN"), row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", BELOW_PIN)
def test_best_effort_keeps_the_assembled_name_below_the_pin(smiles, name):
    row = _row(smiles, "best-effort")
    assert (row.get("name"), row["tier"]) == (
        name, BEST_EFFORT_TIER.get(smiles, "systematic_verified")), row
    assert row["is_pin"] is False, row
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
def test_the_base_row_out_of_citation_order_is_not_a_pin():
    row = _row(BASE_ROW, "pin")
    assert (row["tier"], row.get("limit_code")) == ("abstain", "NO_VERIFIED_PIN"), row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", IN_ORDER)
def test_an_assembled_name_in_citation_order_keeps_the_pin_label(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), row
    assert name_is_rt_exact(name, smiles)
