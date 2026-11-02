"""The word 'hydrate' on a salt is general nomenclature, so such a name is no PIN.

 Organic adducts (the Blue Book): "Solvates, including hydrates (see, are treated as
adducts and preferred IUPAC names must use the notation for denoting the proportion of components described
above. In general nomenclature, hydrates may be named by adding the words 'hydrate' to the name preceded by an
appropriate numerical prefix, such as 'mono','di', 'tri', etc." Mixed organic - inorganic adducts
(:4667): "preferred IUPAC names cannot be assigned to mixed adducts because preferred IUPAC names have not yet
been determined for inorganic components". ``rules.adducts._label_mixed_adduct`` records the adduct form
('acetic acid—water (1/1)') below the PIN; the salt route cites the water as the word and now records the name
too. A carbon-free hydrate is labelled the same way (user decision 2026-11-01; since 2026-11-02 every carbon-free
name is below the PIN,:2056,:2058, the salt without water included); (:37563): "these 'adduct
names' are not PINs". """
import pytest

from tests.support.default_tier import assert_default_tier_declines
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

SALT_HYDRATES = [
    ("CC(=O)[O-].[Na+].O", "sodium acetate monohydrate"),
    ("CC(=O)[O-].[Na+].O.O.O", "sodium acetate trihydrate"),
    ("[O-]C(=O)c1ccccc1.[K+].O", "potassium benzoate monohydrate"),
    ("CCCCCCCCCCCCCCCC[n+]1ccccc1.[Cl-].O", "1-hexadecylpyridin-1-ium chloride monohydrate"),
    ("O.[Na+].[Cl-]", "sodium chloride monohydrate"),
    ("O.O.[Cu+2].[O-]S(=O)(=O)[O-]", "copper(II) sulfate dihydrate"),
]


@pytest.mark.parametrize("smiles,expected", SALT_HYDRATES)
def test_the_hydrate_word_of_a_salt_is_below_the_pin(smiles, expected):
    assert_default_tier_declines(smiles)
    be = name_best_effort(smiles)
    assert be["name"] == expected and be["tier"] == "systematic_verified" and be["is_pin"] is False, be
    assert_full_rt(be["name"], smiles, what="best-effort: ")


@pytest.mark.parametrize("smiles,expected,tier", [
    ("[Na+].CC(=O)[O-]", "sodium acetate", "pin_verified"),
    # changed asserted label (2026-11-02, user decision "decide what the Blue Book says"; was pin_verified): a
    # carbon-free salt has a preselected name at most, the Blue Book "... and that also contain at
    # least one carbon atom in their structure";:2058), with or without its water
    ("[Na+].[Cl-]", "sodium chloride", "systematic_verified"),
])
def test_the_salt_without_water_keeps_its_label(smiles, expected, tier):
    be = name_best_effort(smiles)
    assert be["name"] == expected and be["tier"] == tier, be
