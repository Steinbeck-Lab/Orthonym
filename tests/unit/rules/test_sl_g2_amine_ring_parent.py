"""SL lane G, G2: the ring is the parent of an amine whose nitrogen is bonded to a ring and a chain.

 "Selection of the principal chain or senior ring system in secondary and tertiary amines"
(the Blue Book): "When a choice for parent hydride is possible between a ring and a chain, the
ring is preferred." Its examples: 'N-cyclohexylaniline (PIN)' (:26288), 'N-butylcyclopropanamine (PIN)'
(:26292, not 'N-cyclopropylbutan-1-amine'). (:19340): within one class a ring is senior to a
chain. The nitrogen is bonded to a carbon of each parent it can have, so each carries the amine as its
characteristic group and the ring wins.

Before the change the parent depended on how the input was written. The group is normalised to ONE
bearing carbon (the one with most heavy neighbours, a tie by atom index), which the parent selection
counted (lane M fixes the count, SL-M1; this lane moves the bearing carbon into the parent chosen and builds
the names of the ring parents): 'CC(C)Nc1ccccc1' (a branched chain carbon ties the ring carbon) had no name at the PIN tier,
'c1ccccc1NC(C)C' was 'N-(propan-2-yl)aniline', and 'CCC(CSc1ccccc1)Nc1ccccc1' shipped as the chain name
'N-phenyl-1-(phenylsulfanyl)butan-2-amine' at pin_verified. The producers of the ring parents read a
chain that holds more than alkyl as the prefix of the nitrogen. Every PIN below reads back to the input's
full InChIKey (OPSIN 2.9.0); the old spelling is the mutation each NOT_PIN row rejects.
"""
import pytest

import orthonym.rules.parent_selection as _parent_selection
from tests.support.pin_tiers import (
    assert_not_pin_labelled,
    assert_pin_at_both_tiers,
    name_breadth,
    name_default,
)
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

# The count of the groups a parent holds reads the nitrogen, not the one carbon the perceived match kept
# (lane M, SL-M1: ``parent_selection.bridging_group_reaches``). Without it the parent of a row whose
# chain carbon ties the ring carbon still follows the order of the input; the rows below that need the ring
# as parent run when that change is in the tree. The chain name never ships as a PIN either way
# (NOT_PIN_ROWS: the spelling check of.
needs_nitrogen_count = pytest.mark.skipif(
    not hasattr(_parent_selection, "bridging_group_reaches"),
    reason="needs the nitrogen count of the parent selection (SL-M1)")

PIN_ROWS = [
    # the headline row of the brief: a ring on the nitrogen, a substituted chain
    ("CCC(CSc1ccccc1)Nc1ccccc1", "N-[1-(phenylsulfanyl)butan-2-yl]aniline"),
    ("c1ccccc1NC(CC)CSc1ccccc1", "N-[1-(phenylsulfanyl)butan-2-yl]aniline"),
    ("c1ccccc1SCC(CC)Nc1ccccc1", "N-[1-(phenylsulfanyl)butan-2-yl]aniline"),
    ("CCC(CSC)Nc1ccccc1", "N-[1-(methylsulfanyl)butan-2-yl]aniline"),
    ("CCC(COc1ccccc1)Nc1ccccc1", "N-(1-phenoxybutan-2-yl)aniline"),
    ("CCC(CCl)Nc1ccccc1", "N-(1-chlorobutan-2-yl)aniline"),
    ("CC(Cl)Nc1ccccc1", "N-(1-chloroethyl)aniline"),
    ("CCC(CBr)N(C)c1ccccc1", "N-(1-bromobutan-2-yl)-N-methylaniline"),
    ("CCC(CSc1ccccc1)N(C)c1ccccc1", "N-methyl-N-[1-(phenylsulfanyl)butan-2-yl]aniline"),
    # a cycloalkyl or a heteroaryl on the nitrogen
    ("CCC(CSc1ccccc1)NC1CCCCC1", "N-[1-(phenylsulfanyl)butan-2-yl]cyclohexanamine"),
    ("C(NC1CCCCC1)(CC)CSc1ccccc1", "N-[1-(phenylsulfanyl)butan-2-yl]cyclohexanamine"),
    ("CCC(CSc1ccccc1)Nc1cccnc1", "N-[1-(phenylsulfanyl)butan-2-yl]pyridin-3-amine"),
    ("CCC(CSc1ccccc1)N(CC)c1cccnc1", "N-ethyl-N-[1-(phenylsulfanyl)butan-2-yl]pyridin-3-amine"),
    # a plain alkyl, in the two orders of the input: one name
    ("CC(C)Nc1ccccc1", "N-(propan-2-yl)aniline"),
    ("c1ccccc1NC(C)C", "N-(propan-2-yl)aniline"),
    ("CCC(C)Nc1ccccc1", "N-(butan-2-yl)aniline"),
    ("c1ccccc1NC(C)CC", "N-(butan-2-yl)aniline"),
    ("CC(C)(C)Nc1ccccc1", "N-tert-butylaniline"),
    ("CC(C)N(C)c1ccccc1", "N-methyl-N-(propan-2-yl)aniline"),
    ("CC(C)NC1CCCCC1", "N-(propan-2-yl)cyclohexanamine"),
    ("C1CCCCC1NC(C)C", "N-(propan-2-yl)cyclohexanamine"),
    ("CC(C)Nc1cccnc1", "N-(propan-2-yl)pyridin-3-amine"),
    # unchanged: the book's own PINs and their neighbours
    ("CCCCNC1CC1", "N-butylcyclopropanamine"),                       #:26292
    ("c1ccc(NC2CCCCC2)cc1", "N-cyclohexylaniline"),                  #:26288
    ("c1ccccc1Nc1cccnc1", "N-phenylpyridin-3-amine"),                # heteroring over benzene
    ("CCCCNc1ccccc1", "N-butylaniline"),
    ("CCC(CO)Nc1ccccc1", "2-anilinobutan-1-ol"),                     # the alcohol is the senior group
]

# What shipped at pin_verified before the change: a correct name that is not the PIN.
NOT_PIN_ROWS = [
    ("CCC(CSc1ccccc1)Nc1ccccc1", "N-phenyl-1-(phenylsulfanyl)butan-2-amine"),
    ("CCC(CSc1ccccc1)NC1CCCCC1", "N-cyclohexyl-1-(phenylsulfanyl)butan-2-amine"),
    ("CCC(CSc1ccccc1)N(C)c1ccccc1", "N-methyl-N-phenyl-1-(phenylsulfanyl)butan-2-amine"),
    ("CCC(COc1ccccc1)Nc1ccccc1", "1-phenoxy-N-phenylbutan-2-amine"),
    ("CCC(Cc1ccccc1)Nc1ccccc1", "N,1-diphenylbutan-2-amine"),
    ("CC(c1ccccc1)Nc1ccccc1", "N,1-diphenylethan-1-amine"),
]


@needs_nitrogen_count
@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", NOT_PIN_ROWS)
def test_the_chain_parent_is_not_pin_labelled(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


@needs_nitrogen_count
def test_a_chain_that_holds_a_ring_of_its_own_is_named_at_the_default_tier_only_when_the_namer_can():
    """The default tier names the substituent '1-phenylethyl' ('N-(1-phenylethyl)aniline'); its
    namer does not build '1-phenylbutan-2-yl', so that row declines there and the breadth tier names it
    on the ring (a name that is the PIN but is not certified at the default tier)."""
    res = name_default("CC(c1ccccc1)Nc1ccccc1")
    assert (res["name"], res["tier"]) == ("N-(1-phenylethyl)aniline", "pin_verified"), res
    smiles = "CCC(Cc1ccccc1)Nc1ccccc1"
    assert name_is_rt_exact("N-(1-phenylbutan-2-yl)aniline", smiles)
    assert name_default(smiles)["tier"] == "abstain"
    assert name_breadth(smiles)["name"] == "N-(1-phenylbutan-2-yl)aniline"


# a performance pass of the review: a chain on the nitrogen that holds another amine or a group senior to the amine
# makes the chain the parent, never the aniline. "The senior parent structure has the maximum
# number of substituents corresponding to the principal characteristic group (suffix)..." (:18875);
# "Complex polyamines... The senior parent amine structure is chosen in accordance with the
# choice of a principal chain or a senior ring or ring system, as described in " (:26375), whose example
# 'N1-(aminomethyl)ethane-1,2-diamine (PIN)' puts the diamine first; the class of the principal group comes
# before the ring-over-chain rule of,:18162: the alcohols and the hydroperoxides rank
# above the amines). Each row below read back exactly and shipped as pin_verified with the wrong parent.
CHAIN_IS_THE_PARENT_ROWS = [
    ("NCCNc1ccccc1", "N-(2-aminoethyl)aniline"),                         # PIN: N1-phenylethane-1,2-diamine
    ("NCCNc1ccc(Cl)cc1", "N-(2-aminoethyl)-4-chloroaniline"),
    ("CC(N)CNc1ccccc1", "N-(2-aminopropyl)aniline"),
    ("NCCCNc1ccccc1", "N-(3-aminopropyl)aniline"),
    ("Cc1ccc(NCCN)cc1", "N-(2-aminoethyl)-4-methylaniline"),
    ("NCCCCNc1ccc(C)cc1", "N-(4-aminobutyl)-4-methylaniline"),
    ("NCc1ccc(NCCN)cc1", "N-(2-aminoethyl)-4-(aminomethyl)aniline"),
    ("CCN(CC)CCNc1ccccc1", "N-[2-(diethylamino)ethyl]aniline"),
    # the same on a heteroring and a fused ring (pin_verified on main as well)
    ("NCCNc1ccncc1", "N-(2-aminoethyl)pyridin-4-amine"),
    ("NCCNc1cccc2ccccc12", "N-(2-aminoethyl)naphthalen-1-amine"),
]


@pytest.mark.parametrize("smiles,wrong", CHAIN_IS_THE_PARENT_ROWS)
def test_a_chain_that_holds_another_amine_or_a_senior_group_is_never_named_as_a_ring_amine_pin(smiles, wrong):
    for res in (name_default(smiles), name_breadth(smiles)):
        assert res.get("tier") != "pin_verified", res
        assert not (res.get("name") == wrong and res.get("tier") == "pin_verified"), res
    # the breadth tier still names the molecule, and it reads back exactly
    b = name_breadth(smiles)
    assert b.get("name") and name_is_rt_exact(b["name"], smiles), b


def test_the_hydroperoxy_chain_keeps_the_peroxol_for_the_parent():
    """The hydroperoxide ranks above the amine,:18162, class 18 before 19), so 2-anilinoethane-1-peroxol is
    the PIN. The row was in the list above with 'N-(2-hydroperoxyethyl)aniline' as the wrong name, the PIN being
    unbuilt: the peroxol suffix was no host of a one-carbon-chain hydroperoxide until the hydroperoxide was a
    skeletal suffix (rules.parent_selection.SKELETAL_SUFFIX_PGS), and the name is now the PIN (OPSIN reads it back)."""
    res = name_default("OOCCNc1ccccc1")
    assert (res["name"], res["tier"]) == ("2-anilinoethane-1-peroxol", "pin_verified"), res
    assert name_is_rt_exact("2-anilinoethane-1-peroxol", "OOCCNc1ccccc1")


def test_the_hydroxy_chain_keeps_the_alcohol_for_the_parent():
    """The alcohol is the senior class, so 2-anilinoethan-1-ol is the PIN, found with and without the change."""
    res = name_default("OCCNc1ccccc1")
    assert (res["name"], res["tier"]) == ("2-anilinoethan-1-ol", "pin_verified"), res
