"""Esters of one 'alcoholic' component with two different anions, and the locant of a substituted
chain prefix.

 (the Blue Book): "When anions are different, two methods are used"; "Method
(1) generates preferred IUPAC names but names formed by using method (2) are acceptable in general
nomenclature" (:31836); '(1) methylene acetate formate (PIN) (2) (formyloxy)methyl acetate'
(:31840). Esters of noncarbon acids "are named in the same way",:35918). The method
(1) names are not read by OPSIN 2.9.0, so they are not built; the method (2) names ('(acetyloxy)
methyl formate', '2-(nitrosooxy)ethyl acetate') shipped pin_verified and now ship at best-effort
labelled systematic_verified, the default tier declining.

The string converter parent_to_prefix turned the capped name '(nitrooxy)ethane' into
'(nitrooxy)ethyl' (no locant for the 2-position) and '1-(nitrooxy)propane' into '1-(nitrooxy)propyl'
(the other end, a different molecule): /, the free valence takes locant 1 and the
prefixes are numbered from it. It declines a prefixed stem of two or more carbons now; the located
chain namers number it ('2-(nitrooxy)ethyl', '3-(nitrooxy)propyl'). Names read back by OPSIN 2.9.0
to the input's full InChIKey.
"""
import pytest

from orthonym.assembly.substituent_naming import ATTACH_LOCANT_UNKNOWN, parent_to_prefix
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)OCCO[N+](=O)[O-]", "2-(nitrooxy)ethyl acetate"),
    ("CC(=O)OCCCO[N+](=O)[O-]", "3-(nitrooxy)propyl acetate"),
    ("CC(=O)OCCON=O", "2-(nitrosooxy)ethyl acetate"),
    ("O=COCOC(C)=O", "(acetyloxy)methyl formate"),
    ("O=NOCCO[N+](=O)[O-]", "2-(nitrooxy)ethyl nitrite"),
    ("O=C(OCCO[N+](=O)[O-])c1ccccc1", "2-(nitrooxy)ethyl benzoate"),
])
def test_method_2_polyol_ester_is_below_the_pin(smiles, name):
    d = name_default(smiles)
    assert d.get("tier") == "abstain", d
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (name, "systematic_verified"), b
    assert name_is_rt_exact(name, smiles)


def test_a_chain_prefix_keeps_its_locant_at_best_effort():
    # an acid outranks the nitrate ester, so the 2-(nitrooxy)ethyl group stays a prefix and
    # keeps its locant (was '[(nitrooxy)ethyl]amino'). The old witness, the acetamide
    # CC(=O)NCCO[N+](=O)[O-], is named on the senior ester since quick-wins Q4e.
    b = name_breadth("OC(=O)CC(=O)NCCO[N+](=O)[O-]")
    assert b.get("name") == "3-{[2-(nitrooxy)ethyl]amino}-3-oxopropanoic acid", b


@pytest.mark.parametrize("smiles,pin", [
    ("COCOC(C)=O", "methoxymethyl acetate"),
    ("CC(=O)OCCOC", "2-methoxyethyl acetate"),
    ("CC(=O)OCCCN=[N+]=[N-]", "3-azidopropyl acetate"),
    ("CC(=O)OCCCl", "2-chloroethyl acetate"),
    ("CCC(CC(C)=O)C([SiH3])[SiH3]", "4-(disilylmethyl)hexan-2-one"),
])
def test_controls_keep_their_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("capped", ["(nitrooxy)ethane", "1-(nitrooxy)propane", "1-azidopropane",
                                    "2-methylpropane", "1-chlorobutane", "1-chloroundecane",
                                    "2-methylicosane"])
def test_the_string_converter_declines_a_prefixed_chain_stem(capped):
    assert parent_to_prefix(capped, 2, attach_locant=ATTACH_LOCANT_UNKNOWN) is None


@pytest.mark.parametrize("capped,prefix", [("propane", "propyl"), ("methoxymethane", "methoxymethyl"),
                                           ("disilylmethane", "disilylmethyl")])
def test_the_string_converter_keeps_plain_and_one_carbon_stems(capped, prefix):
    assert parent_to_prefix(capped, 3, attach_locant=ATTACH_LOCANT_UNKNOWN) == prefix


# (the Blue Book): "Systematic names for the higher members of this series consist
# of a numerical term (see Table 1.4) followed by the ending 'ane'"; (:2807): "The
# numerical term for the number '11' is 'undeca'", and '2' in association is 'do'. The 'un'/'do'/
# 'tri'... of an unbranched chain stem is its numerical term, not a substituent prefix.
@pytest.mark.parametrize("capped,prefix", [
    ("decane", "decyl"), ("undecane", "undecyl"), ("dodecane", "dodecyl"),
    ("tridecane", "tridecyl"), ("tetradecane", "tetradecyl"), ("hexadecane", "hexadecyl"),
    ("octadecane", "octadecyl"), ("nonadecane", "nonadecyl"), ("icosane", "icosyl"),
    ("henicosane", "henicosyl"), ("triacontane", "triacontyl"),
])
def test_the_string_converter_keeps_unbranched_long_chain_stems(capped, prefix):
    assert parent_to_prefix(capped, 3, attach_locant=ATTACH_LOCANT_UNKNOWN) == prefix
