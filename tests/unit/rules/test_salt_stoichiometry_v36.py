import pytest

from orthonym.namer import name_compound
from orthonym.rules.salts import (
    _ion_needs_enclosing_multiplier,
    _apply_stoichiometric_prefix,
    _DI_COLLISION_ANIONS,
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


# === -A1 FABLE finding: the di-collision oxoanion class ==================
# FABLE cross-model review BLOCKER: for a bare mononuclear oxoanion X, ``di``+X
# is a REAL OPSIN word for a DIFFERENT (pyro/condensed) species. Ca3(PO4)2 emitted
# 'tricalcium diphosphate' -> OPSIN reads calcium PYROPHOSPHATE (P2O7) ->
# suppresses (0-wrong safe) -> the salt needlessly ABSTAINS, even though
# 'tricalcium bis(phosphate)' round-trips. _DI_COLLISION_ANIONS (VERIFIED
# 2026-08-23 by probing every INORGANIC_ANIONS value + every emittable single-word
# oxoanion word against opsin_parse("di"+W): 19 collisions) forces the enclosing
# multiplier bis(W)/tris(W) for these bare words.


# Predicate unit (no OPSIN): EVERY verified di-collision word forces enclosing.
@pytest.mark.parametrize("name", sorted(_DI_COLLISION_ANIONS))
def test_di_collision_word_forces_enclosing(name):
    assert _ion_needs_enclosing_multiplier(name) is True


# Predicate unit (no OPSIN): the FABLE-cited seed words, spelled out explicitly.
@pytest.mark.parametrize("name", [
    "phosphate", "sulfate", "carbonate", "chromate", "sulfite",
])
def test_di_collision_seed_words_true(name):
    assert _ion_needs_enclosing_multiplier(name) is True


# Predicate unit (no OPSIN): non-colliding controls keep the simple ``di`` and
# glue it directly (di<name>) — the class fix must NOT over-broaden.
@pytest.mark.parametrize("name", ["acetate", "chloride", "benzoate"])
def test_non_colliding_controls_keep_di(name):
    assert _ion_needs_enclosing_multiplier(name) is False
    assert _apply_stoichiometric_prefix(name, 2) == "di" + name


def test_collision_word_wraps_with_enclosing_multiplier():
    # a collision word is wrapped, never glued
    assert _apply_stoichiometric_prefix("phosphate", 2) == "bis(phosphate)"
    assert _apply_stoichiometric_prefix("sulfate", 3) == "tris(sulfate)"
    assert _apply_stoichiometric_prefix("carbonate", 2) == "bis(carbonate)"


# --- Integration RED->GREEN (OPSIN): the FABLE witness Ca3(PO4)2 -----------
# RED (pre-fix, captured in fix1-report.md): name_compound emitted the
# not-supported placeholder 'calcium compound (not supported)' because
# 'tricalcium diphosphate' (= pyrophosphate) was -suppressed and did NOT
# round-trip. GREEN: emits 'tricalcium bis(phosphate)', which round-trips.
CA3_PO4_2 = "[O-]P(=O)([O-])[O-].[O-]P(=O)([O-])[O-].[Ca+2].[Ca+2].[Ca+2]"


def test_fable_witness_ca3_po4_2_round_trips():
    name = name_compound(CA3_PO4_2, style="pin")
    assert name, "abstained on the FABLE witness Ca3(PO4)2"
    # must not emit the colliding 'diphosphate' word
    assert "diphosphate" not in name, f"still emits collision word: {name!r}"
    result = opsin_roundtrip_check(CA3_PO4_2, name)
    assert result["passed"], f"{name!r} did not round-trip: {result}"


# --- Second colliding-oxoanion salt (RED->GREEN): sodium carbonate, count 2 -
# RED: 'sodium compound (not supported)' (dicarbonate = pyrocarbonate C2O5,
# -suppressed). GREEN: 'tetrasodium bis(carbonate)', RT-verified.
NA4_CO3_2 = "[O-]C(=O)[O-].[O-]C(=O)[O-].[Na+].[Na+].[Na+].[Na+]"


def test_sodium_carbonate_count2_round_trips():
    name = name_compound(NA4_CO3_2, style="pin")
    assert name, "abstained on the disodium-carbonate (count 2) salt"
    assert "dicarbonate" not in name, f"still emits collision word: {name!r}"
    assert opsin_roundtrip_check(NA4_CO3_2, name)["passed"], name


# --- No-regression guard: aluminium sulfate Al2(SO4)3 (count 3) -------------
# Pre-fix this PASSED as 'dialuminium trisulfate' (OPSIN read tri+sulfate as
# 3xSO4). Post-fix it becomes 'dialuminium tris(sulfate)' (the unambiguous
# enclosing form) which ALSO round-trips -> 0-wrong preserved, no abstain.
AL2_SO4_3 = ("[O-]S(=O)(=O)[O-].[O-]S(=O)(=O)[O-].[O-]S(=O)(=O)[O-]."
             "[Al+3].[Al+3]")


def test_aluminium_sulfate_count3_no_regression():
    name = name_compound(AL2_SO4_3, style="pin")
    assert name, "regressed to abstain on Al2(SO4)3"
    assert opsin_roundtrip_check(AL2_SO4_3, name)["passed"], name
