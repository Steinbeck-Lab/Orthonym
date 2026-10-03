"""One rule decides whether a retained contracted alkoxy prefix is substituted.

 (the Blue Book): 'methoxy', 'ethoxy', 'propoxy', 'butoxy' and 'phenoxy' are
"considered as simple prefixes requiring the numerical prefixes 'di', 'tri', etc.". One of
them carrying a substituent prefix is substituted: (c) (:7035) "any component which
is substituted automatically requires use of the multiplicative forms 'bis', 'tris', etc.",
and it is a compound prefix that (:7232) encloses ('1-(chloromethoxy)-4-
nitrobenzene (PIN)',:27711). An isotope descriptor is not a substituent: '1,2-di[(13C)methyl]
benzene (PIN. ' (:7492).

The bridged fused S3 lane read the class in needs_brackets / is_complex_substituent
(is_substituted_contracted_alkoxy); the quick wins enclosed through is_substituted_substituent
in enclose_if_compound / format_substituent_prefix, and the multiplier asks
is_substituted_substituent too. After the merge the class predicate IS is_substituted_substituent
restricted to the contracted stems, so the enclosure (both routes) and the multiplier agree on
every token below.
"""
import pytest

from orthonym.assembly.naming_utils import (
    enclose_if_compound,
    get_multiplier_prefix,
    is_complex_substituent,
    is_substituted_contracted_alkoxy,
    is_substituted_substituent,
    needs_brackets,
)

# The lane's COMPOUND rows, plus roots the quick-wins vocabulary split did not know.
SUBSTITUTED = [
    "methoxymethoxy", "ethoxyethoxy", "chloromethoxy", "hydroxyethoxy", "carboxymethoxy",
    "phenylmethoxy", "chlorophenoxy", "methylsulfanylmethoxy", "(methylsulfanyl)methoxy",
    "(trifluoromethyl)methoxy", "trifluoromethoxy",
    "acetylmethoxy", "formylmethoxy", "ethenylmethoxy", "oxiranylmethoxy",
    "bis(trifluoromethyl)methoxy", "carboxy(phenyl)methoxy",
]
# The lane's SIMPLE rows, plus isotope descriptors in front of the stem.
SIMPLE = [
    "methoxy", "ethoxy", "propoxy", "butoxy", "phenoxy", "tert-butoxy", "isopropoxy",
    "dimethoxy", "(2H3)methoxy", "(13C)methoxy", "(18O)methoxy",
]


@pytest.mark.parametrize("name", SUBSTITUTED)
def test_a_substituted_contracted_alkoxy_is_substituted_on_every_route(name):
    assert is_substituted_substituent(name), name
    assert is_substituted_contracted_alkoxy(name.lower()), name
    assert (needs_brackets(name), is_complex_substituent(name)) == (True, True), name
    assert enclose_if_compound(name)[0] in "([{", name
    assert get_multiplier_prefix(2, name) == "bis", name


@pytest.mark.parametrize("name", SIMPLE)
def test_a_contracted_alkoxy_without_a_substituent_prefix_takes_di(name):
    assert not is_substituted_substituent(name), name
    assert not is_substituted_contracted_alkoxy(name.lower()), name
    # 'di-' before an italic prefix: (d), 'di-tert-butyl' (:6964)
    assert get_multiplier_prefix(2, name) in ("di", "di-"), name


@pytest.mark.parametrize("name", SUBSTITUTED + SIMPLE)
def test_the_class_predicate_is_the_substitution_rule_on_the_contracted_stems(name):
    assert is_substituted_contracted_alkoxy(name.lower()) == is_substituted_substituent(name.lower())
