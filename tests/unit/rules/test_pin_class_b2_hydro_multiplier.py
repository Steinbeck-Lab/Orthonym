"""PIN class program, batch 2 fix a performance pass: the multiplier of a prefix that carries hydro prefixes.

The rule
--------
 (the Blue Book): "Parentheses (round brackets) (see are used to enclose
multiplied components that are: (a) simple substituent prefixes having locants"
('di(propan-2-yl)'); (b) "simple substituent prefixes modified by 'ene' and 'yne' endings and
that have locants" ('di(prop-1-en-2-yl)',:7104). (:7104): "The numerical prefixes
'bis', 'tris', 'tetrakis', etc. are used to indicate a multiplicity of: (a) compound or complex
(i.e. substituted) prefixes". Hydro prefixes "are now classified as detachable prefixes but are
not included in the category of alphabetized detachable prefixes which describe substitution"
(the summary of changes, item 5,:1682), so the rule text does not settle the multiplier.

The Blue Book's one multiplied prefix that carries hydro prefixes takes 'bis':
'bis(4,5-dihydrothiophen-2-yl)di(methyl)germane (PIN) (Ge is senior to S)'
Substituted parent hydrides,:38232). The book has no 'di(' before such a prefix (a search of
the whole text for a basic multiplier followed by a hydro-prefixed component finds none).
Indicated hydrogen alone keeps 'di': 'di(1H-imidazol-1-yl)methanethione (PIN)',
:29544); 'di(naphthalen-2-yl)ethanedione (PIN)',:28380).

At the batch-2 fix-a performance pass head the shared multiplier (``naming_utils.get_multiplier_prefix``)
and the duplicate-prefix merger of the composer (``composer._merge_duplicate_prefixes``, a
private 'di'/'tri' table) gave 'di' ('di(2,3-dihydro-1H-indol-1-yl)methanone',
'di(4,5-dihydrothiophen-2-yl)methanone', '1,3-di(2,3-dihydro-1H-inden-5-yl)propan-2-ol'), all
labelled pin_verified.

Fix a performance pass: ``carries_hydro_prefix`` counted the chalcogen prefixes 'hydroseleno' and
'hydrotelluro' ('2-hydroselenoethyl') as hydro prefixes. They are not; a hydro prefix on a
selenophene or tellurophene ring ('4,5-dihydroselenophen-2-yl') still is.
"""
import pytest

from orthonym.assembly.naming_utils import carries_hydro_prefix, get_multiplier_prefix
from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    # the ketone parent of a C=O bridge between ring nitrogens (rules/multiplicative.py)
    ("O=C(N1CCc2ccccc21)N1CCc2ccccc21", "bis(2,3-dihydro-1H-indol-1-yl)methanone"),
    ("O=C(N1CCc2ccccc2C1)N1CCc2ccccc2C1", "bis(1,2,3,4-tetrahydroisoquinolin-2-yl)methanone"),
    ("O=C(N1CCCc2ccccc21)N1CCCc2ccccc21", "bis(1,2,3,4-tetrahydroquinolin-1-yl)methanone"),
    # the prefix of:38232 on a ketone (the shared multiplier)
    ("O=C(C1=CCCS1)C1=CCCS1", "bis(4,5-dihydrothiophen-2-yl)methanone"),
    ("OC(C1CCc2ccccc2C1)C1CCc2ccccc2C1", "bis(1,2,3,4-tetrahydronaphthalen-2-yl)methanol"),
    ("C[Si](C)(c1ccc2CCCc2c1)c1ccc2CCCc2c1",
     "bis(2,3-dihydro-1H-inden-5-yl)di(methyl)silane"),                                 # as:38232
    ("CC(=O)N(c1ccc2CCCc2c1)c1ccc2CCCc2c1", "N,N-bis(2,3-dihydro-1H-inden-5-yl)acetamide"),
    ("c1cc(C2CCc3ccccc32)nc(C2CCc3ccccc32)c1", "2,6-bis(2,3-dihydro-1H-inden-1-yl)pyridine"),
    # the duplicate-prefix merger of the composer
    ("O=C(c1ccc2c(c1)CCC2)c1ccc2c(c1)CCC2", "bis(2,3-dihydro-1H-inden-5-yl)methanone"),
    ("OC(Cc1ccc2CCCc2c1)Cc1ccc2CCCc2c1", "1,3-bis(2,3-dihydro-1H-inden-5-yl)propan-2-ol"),
    ("O=C(CC1CCc2ccccc21)CC1CCc2ccccc21", "1,3-bis(2,3-dihydro-1H-inden-1-yl)propan-2-one"),
]

# Unchanged: indicated hydrogen alone, an unsubstituted ring system, a saturated ring named
# without hydro prefixes, a substituted prefix.
CONTROL_ROWS = [
    ("S=C(n1ccnc1)n1ccnc1", "di(1H-imidazol-1-yl)methanethione"),                        #:29544
    ("O=C(c1ccc2ccccc2c1)C(=O)c1ccc2ccccc2c1", "di(naphthalen-2-yl)ethanedione"),        #:28380
    ("O=C(N1CCCC1)N1CCCC1", "di(pyrrolidin-1-yl)methanone"),
    ("O=C(c1ccc(Cl)cc1)c1ccc(Cl)cc1", "bis(4-chlorophenyl)methanone"),
]

HELPER_ROWS = [
    ("2,3-dihydro-1H-indol-1-yl", True, "bis"),
    ("4,5-dihydrothiophen-2-yl", True, "bis"),                                          #:38232
    ("(2,3-dihydro-1H-inden-5-yl)", True, "bis"),
    ("1,2,3,4-tetrahydronaphthalen-1-yl", True, "bis"),
    ("4a,8a-dihydronaphthalen-2-yl", True, "bis"),
    ("1H-imidazol-1-yl", False, "di"),                                                  #:29544
    ("naphthalen-2-yl", False, "di"),                                                   #:28380
    ("2H-pyran-3(4H)-yl", False, "di"),
    ("propan-2-yl", False, "di"),
    ("2,3-dihydroxypropyl", False, "bis"),
    ("hydroxy", False, "di"),
    # fix a performance pass: the 'hydroseleno' / 'hydrotelluro' prefixes are not hydro prefixes;
    # a hydro prefix on a selenophene / tellurophene ring is
    ("hydroseleno", False, "di"),
    ("hydrotelluro", False, "di"),
    ("4,5-dihydroselenophen-2-yl", True, "bis"),
    ("4,5-dihydrotellurophen-2-yl", True, "bis"),
]

# (prefix, carries a hydro prefix): the chalcogen prefixes inside a larger prefix
CHALCOGEN_PREFIX_ROWS = [
    ("2-hydroselenoethyl", False),
    ("2-hydrotelluroethyl", False),
    ("4-hydroselenophenyl", False),
    ("2,3-dihydroselenopheno[2,3-b]pyridin-5-yl", True),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", CONTROL_ROWS)
def test_control(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("prefix,hydro,mult", HELPER_ROWS)
def test_multiplier_of_a_prefix(prefix, hydro, mult):
    assert carries_hydro_prefix(prefix) is hydro
    assert get_multiplier_prefix(2, prefix) == mult


@pytest.mark.parametrize("prefix,hydro", CHALCOGEN_PREFIX_ROWS)
def test_chalcogen_prefix_is_not_a_hydro_prefix(prefix, hydro):
    assert carries_hydro_prefix(prefix) is hydro


def test_merger_multiplies_a_hydro_prefixed_prefix_with_bis():
    from orthonym.assembly.composer import NameFragment, _merge_duplicate_prefixes

    def frag(text, locants=()):
        return NameFragment(text=text, locants=tuple(locants), fragment_type="prefix")

    merged = _merge_duplicate_prefixes([frag("1-(2,3-dihydro-1H-inden-5-yl)", (1,)),
                                        frag("3-(2,3-dihydro-1H-inden-5-yl)", (3,))])
    assert [m.text for m in merged] == ["1,3-bis(2,3-dihydro-1H-inden-5-yl)"]
    merged = _merge_duplicate_prefixes([frag("2-methyl", (2,)), frag("3-methyl", (3,))])
    assert [m.text for m in merged] == ["2,3-dimethyl"]
    merged = _merge_duplicate_prefixes([frag("2-hydroxy", (2,)), frag("3,4-dihydroxy", (3, 4))])
    assert [m.text for m in merged] == ["2,3,4-trihydroxy"]
