"""A compound prefix of a retained acyl group is cited at the first letter of its complete name.

 (the Blue Book, under ``### **** ALPHANUMERICAL ORDER``:3436): "The name of a prefix for a
substituent is considered to begin with the first letter of its complete name." 'trifluoroacetyl' is one compound prefix,
'trifluoro' on 'acetyl', and is cited at 't': the key was 'fluoroacetyl' because 'acetyl' was no recognised unit, so the
multiplier read as a separate one that (:3448) leaves out of the order. 'trimethylazaniumyl' had the same defect for the
cationic hydride groups (``_CATIONIC_HYDRIDE_YLS``); the retained acyl groups of "Definitions and name
formation" (:30426) are the same fix.
"""
import pytest

from orthonym.assembly.naming_utils import alpha_sort_key


@pytest.mark.parametrize("prefix,key", [
    ("trifluoroacetyl", "trifluoroacetyl"),
    ("2,2,2-trifluoroacetyl", "trifluoroacetyl"),
    ("trichloroacetyl", "trichloroacetyl"),
    ("fluoroformyl", "fluoroformyl"),
    ("trimethylbenzoyl", "trimethylbenzoyl"),
])
def test_a_multiplier_inside_a_compound_acyl_prefix_is_part_of_its_first_letter(prefix, key):
    assert str(alpha_sort_key(prefix)) == key


@pytest.mark.parametrize("prefix,key", [
    ("diacetyl", "acetyl"),          # the multiplier of the whole prefix is left out of the order
    ("(chloroacetyl)", "chloroacetyl"),
    ("methyl", "methyl"),
])
def test_the_multiplier_of_a_whole_prefix_stays_out_of_the_order(prefix, key):
    assert str(alpha_sort_key(prefix)) == key


def test_the_prefixes_of_a_name_are_cited_in_alphanumerical_order():
    prefixes = ["trifluoroacetyl", "methyl", "chloro"]
    assert sorted(prefixes, key=alpha_sort_key) == ["chloro", "methyl", "trifluoroacetyl"]
