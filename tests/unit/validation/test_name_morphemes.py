"""v29 Phase 1: the independent morpheme-arity oracle.

Soundness first: a CONFIDENT answer must be right; anything unrecognised must
come back unconfident rather than guessed.
"""
import pytest

from orthonym.validation.binding_spine import BindingKind
from orthonym.validation.name_morphemes import token_arity

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("token,kind,expected", [
    ("meth", BindingKind.PARENT, 1),
    ("eth", BindingKind.PARENT, 2),
    ("prop", BindingKind.PARENT, 3),
    ("dec", BindingKind.PARENT, 10),
    ("methyl", BindingKind.PREFIX, 1),
    ("ethyl", BindingKind.PREFIX, 2),
    ("phenyl", BindingKind.PREFIX, 6),
    ("benzene", BindingKind.PARENT, 6),
    ("cyclohexane", BindingKind.PARENT, 6),
    ("cyclohexyl", BindingKind.PREFIX, 6),
    ("naphthalene", BindingKind.PARENT, 10),
    ("methoxy", BindingKind.PREFIX, 2),      # 1 C + 1 O
    ("hydroxy", BindingKind.PREFIX, 1),      # 1 O
    ("amino", BindingKind.PREFIX, 1),        # 1 N
])
def test_confident_arities(token, kind, expected):
    est = token_arity(token, kind)
    assert est.confident, f"{token}: {est.basis}"
    assert est.heavy_atoms == expected


def test_multiplier_prefix_is_ignored_arity_is_one_instance():
    assert token_arity("dimethyl", BindingKind.PREFIX).heavy_atoms == \
        token_arity("methyl", BindingKind.PREFIX).heavy_atoms


def test_carboxyl_carbon_belongs_to_the_parent_not_the_suffix():
    # propanoic acid: parent 'propan' spells 3 C, suffix 'oic acid' spells 2 O
    assert token_arity("propan", BindingKind.PARENT).heavy_atoms == 3
    est = token_arity("oic acid", BindingKind.SUFFIX)
    assert est.confident and est.heavy_atoms == 2


def test_skeleton_only_suffixes_add_no_atoms():
    for suffix in ("ol", "one", "amine", "ane", "ene", "yne"):
        est = token_arity(suffix, BindingKind.SUFFIX)
        assert est.confident, f"{suffix}: {est.basis}"
    # -ol/-one/-amine each name ONE heteroatom; -ane/-ene/-yne name none
    assert token_arity("ol", BindingKind.SUFFIX).heavy_atoms == 1
    assert token_arity("ane", BindingKind.SUFFIX).heavy_atoms == 0


@pytest.mark.parametrize("token", [
    "quux",                       # not a morpheme at all
    "lambda5-phosphanyl",         # lambda convention: out of scope
    "bicyclo",                    # descriptor missing
    "spiro",                      # descriptor missing
])
def test_unrecognised_tokens_are_unconfident_not_guessed(token):
    est = token_arity(token, BindingKind.PREFIX)
    assert not est.confident
    assert est.heavy_atoms is None
    assert est.basis  # always explains why


@pytest.mark.parametrize("token,expected", [
    ("methylbenzene", 7),              # methyl 1 + benzene 6
    ("2-chloroethyl", 3),              # chloro 1 + eth 2
    ("2-hydroxyethyl", 3),             # hydroxy 1 + eth 2
    ("4-(methoxymethyl)phenyl", 9),    # methoxy 2 + methyl 1 + phenyl 6
    ("dimethylphenyl", 8),             # 2x methyl + phenyl 6
])
def test_composite_tokens_sum_their_morphemes(token, expected):
    """The real producers bind a WHOLE branch under one composed token
    (general_engine.py:405-406 etc.), so composite parsing is the common case,
    not an edge case."""
    est = token_arity(token, BindingKind.PREFIX)
    assert est.confident, f"{token}: {est.basis}"
    assert est.heavy_atoms == expected


def test_never_confidently_wrong_on_a_composite():
    est = token_arity("methylbenzene", BindingKind.PARENT)
    assert (not est.confident) or est.heavy_atoms == 7


@pytest.mark.parametrize("token", ["oxan", "thian", "oxane", "thiane"])
def test_elided_hantzsch_widman_stem_is_not_read_as_bare_saturation(token):
    """Regression, found by auditing real production tokens.

    A HW stem drops its terminal 'e' before a suffix ("oxane" -> "oxan-4-yl").
    With only the full stem registered, 'oxan' parsed uniquely as ox+an
    (replacement prefix + saturation ending) and answered a CONFIDENT 0 for a
    six-atom ring -- the fatal failure mode. Both readings must now be visible
    so the token is refused rather than answered wrongly.
    """
    est = token_arity(token, BindingKind.PARENT)
    assert (not est.confident) or est.heavy_atoms == 6, (
        f"{token}: confidently answered {est.heavy_atoms} ({est.basis})")


@pytest.mark.parametrize("token,truth", [
    # A multiplier is token-level (ignored) ONLY when its group runs to the end
    # of the token: "dimethyl" spells one methyl. As soon as anything follows
    # the multiplied group, the multiplier is internal and must be APPLIED --
    # whether what follows is a skeletal head ("dimethylphenyl") or an
    # attachment affix ("dimethylamino" = 2 methyl + 1 N = 3).
    ("dimethylamino", 3),
    ("bis(2-chloroethyl)amino", 7),      # 2 x (Cl + C2) + N
    ("dimethylphenyl", 8),
    ("dimethoxyphenyl", 10),             # 2 x (C + O) + 6
    ("trichloromethyl", 4),
    ("dichloroethyl", 4),
    ("tetrahydronaphthalene", 10),       # 'hydro' adds no heavy atom
    ("dimethyl", 1),                     # nothing follows -> one instance
])
def test_multiplier_scope_is_never_confidently_wrong(token, truth):
    """Regression: mis-scoping a multiplier is the one way this oracle can be
    confidently WRONG, which would make P6 reject a correct name."""
    est = token_arity(token, BindingKind.PREFIX)
    assert (not est.confident) or est.heavy_atoms == truth, (
        f"{token}: confidently answered {est.heavy_atoms}, truth {truth} "
        f"({est.basis})")
