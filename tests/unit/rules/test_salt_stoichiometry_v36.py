import pytest

from orthonym.namer import name_compound
from orthonym.rules.salts import (
    _ion_needs_enclosing_multiplier,
    _apply_stoichiometric_prefix,
)
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


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
    assert _ion_needs_enclosing_multiplier(name) is expected


# Review-finding coverage: the startswith/endswith branches of
# _ion_needs_enclosing_multiplier are never exercised by the parametrize
# table above without being short-circuited by an earlier check (the hyphen
# or digit checks fire first for every existing composite case). Isolate the
# two remaining branches directly. Predicate-only, no OPSIN.
@pytest.mark.parametrize("name,expected", [
    ("methylphosphonate", True),    # endswith 'phosphonate' branch
    ("tetrafluoroborate", True),    # startswith 'tetra' branch
])
def test_needs_enclosing_multiplier_startswith_endswith_branches(name, expected):
    assert _ion_needs_enclosing_multiplier(name) is expected


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


# salt witnesses 11 & 12 (calcium bis-aldonate) + 1 verified synthetic case.
A1_SALT_WITNESSES = [
    # w11: calcium D-gluconate
    "O=C([O-])[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO."
    "O=C([O-])[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO.[Ca+2]",
    # w12: calcium aldonate (C2 stereo unspecified variant)
    "O=C([O-])C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO."
    "O=C([O-])C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO.[Ca+2]",
    # verified synthetic: calcium bis(2-hydroxypropanoate) (lactate)
    "CC(O)C(=O)[O-].CC(O)C(=O)[O-].[Ca+2]",
]


@pytest.mark.parametrize("smiles", A1_SALT_WITNESSES)
def test_a1_salt_witness_round_trips(smiles):
    name = name_compound(smiles, style="pin")
    assert name, f"abstained on {smiles}"
    result = opsin_roundtrip_check(smiles, name)
    assert result["passed"], f"{name!r} did not round-trip: {result}"


# Task 5 (defect C): metal salt of a FULLY-DEPROTONATED phosphate ester anion
# (disodium dexamethasone-21-phosphate). Both acid oxygens are [O-]; the
# composed name must NOT carry a 'hydrogen'/'dihydrogen' word.
W7_ESTER_SALT = ("C[C@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)"
                 "[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)COP(=O)([O-])[O-].[Na+].[Na+]")


def test_w7_ester_salt_round_trips():
    name = name_compound(W7_ESTER_SALT, style="pin")
    assert name, "abstained on the disodium ester-phosphate salt"
    assert opsin_roundtrip_check(W7_ESTER_SALT, name)["passed"], name
