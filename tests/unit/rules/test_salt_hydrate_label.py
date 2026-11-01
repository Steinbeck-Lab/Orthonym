"""The word 'hydrate' on a salt is general nomenclature, so such a name is no PIN.

 Organic adducts (the Blue Book): "Solvates, including hydrates (see, are treated as
adducts and preferred IUPAC names must use the notation for denoting the proportion of components described
above. In general nomenclature, hydrates may be named by adding the words 'hydrate' to the name preceded by an
appropriate numerical prefix, such as 'mono','di', 'tri', etc." Mixed organic - inorganic adducts
(:4667): "preferred IUPAC names cannot be assigned to mixed adducts because preferred IUPAC names have not yet
been determined for inorganic components". ``rules.adducts._label_mixed_adduct`` records the adduct form
('acetic acid—water (1/1)') below the PIN; the salt route cites the water as the word and now records the name
too. A carbon-free hydrate is labelled the same way (user decision 2026-11-01, which narrows the 2026-09-30
one: a carbon-free salt keeps the label of its naming path, its hydrate does not); (:37563): "these 'adduct
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


@pytest.mark.parametrize("smiles,expected", [
    ("[Na+].CC(=O)[O-]", "sodium acetate"),
    ("[Na+].[Cl-]", "sodium chloride"),
])
def test_the_salt_without_water_keeps_its_label(smiles, expected):
    be = name_best_effort(smiles)
    assert be["name"] == expected and be["tier"] == "pin_verified", be
