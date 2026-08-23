import pytest
from orthonym.rules.salts import _anion_needs_enclosing_multiplier


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
