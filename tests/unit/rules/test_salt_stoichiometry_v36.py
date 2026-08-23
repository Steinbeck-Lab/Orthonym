import pytest
from orthonym.rules.salts import (
    _anion_needs_enclosing_multiplier,
    _apply_stoichiometric_prefix,
)


@pytest.mark.parametrize("name,expected", [
    # simple one-word anions -> di/tri (NOT wrapped)
    ("acetate", False),
    ("benzenesulfonate", False),
    ("chloride", False),
    ("methoxide", False),
    # composite -> bis(...) (wrapped). Each reason is OPSIN-verified.
    ("D-gluconate", True),                      # leading stereo prefix (hyphen)
    ("2-hydroxypropanoate", True),              # digit-locant (di2- fuses badly)
    ("(2R)-2-hydroxybutanedioate", True),       # paren + digits
    ("naphthalene-2-sulfonate", True),          # internal locant (digit)
    ("hydrogen phosphate", True),               # embedded space (multi-word)
    ("bis(oxidanyl)phosphinate", True),         # already-enclosed paren
])
def test_needs_enclosing_multiplier(name, expected):
    assert _anion_needs_enclosing_multiplier(name) is expected


# Review-finding coverage: the startswith/endswith branches of
# _anion_needs_enclosing_multiplier are never exercised by the parametrize
# table above without being short-circuited by an earlier check (the hyphen
# or digit checks fire first for every existing composite case). Isolate the
# two remaining branches directly. Predicate-only, no OPSIN.
@pytest.mark.parametrize("name,expected", [
    ("methylphosphonate", True),    # endswith 'phosphonate' branch
    ("tetrafluoroborate", True),    # startswith 'tetra' branch
])
def test_needs_enclosing_multiplier_startswith_endswith_branches(name, expected):
    assert _anion_needs_enclosing_multiplier(name) is expected


@pytest.mark.parametrize("name,count,expected", [
    ("sodium", 1, "sodium"),                       # count 1 -> unchanged
    ("acetate", 2, "diacetate"),                   # simple -> di, glued
    ("acetate", 3, "triacetate"),
    ("benzenesulfonate", 2, "dibenzenesulfonate"), # simple -> di
    ("D-gluconate", 2, "bis(D-gluconate)"),        # composite -> bis, wrapped
    ("2-hydroxypropanoate", 2, "bis(2-hydroxypropanoate)"),
    ("(2R)-2-hydroxybutanedioate", 3, "tris((2R)-2-hydroxybutanedioate)"),
])
def test_apply_stoichiometric_prefix(name, count, expected):
    assert _apply_stoichiometric_prefix(name, count) == expected
