""" a phase: the independent morpheme-arity oracle.

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
    "λ5-phosphanyl",              # lambda convention: out of scope
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


@pytest.mark.parametrize("token,truth", [
    # a phase Task 2b: a multiplier ("di") directly in front of a zero-atom
    # skeletal REPLACEMENT prefix ("oxa", "aza", "phospha") must contribute NO
    # extra atoms, exactly like the pre-existing "dihydro"/"tetrahydro" AFFIX
    # case -- REPL is 0 net atoms BY DEFINITION regardless of how many
    # positions are replaced, because the stem's own count ("hept" = 7, "but"
    # = 4) already includes those positions. Isolated (nothing precedes in
    # the same token), this was already right BY ACCIDENT of the
    # "multiplier governs the whole token" branch:
    ("1,3-dioxa-6-aza-2-phosphaheptyl", 7),
    ("2,4-dioxabutyl", 4),
    ("2,4-dioxabut-3-en-1-yl", 4),
    #... but a SUBST prefix (hydroxy/oxo/methyl) EARLIER in the SAME
    # composite token used to flip the multiplier into the "internal" branch
    # and multiply the replacement chain's atom count as if `di` were
    # multiplying a real substituent group -- the exact false ARITY_MISMATCH
    # that voided 3 correct, OPSIN-round-tripping a dev split rows (a phase
    # Task 2 measurement) via P6. The real correct counts, verified
    # atom-by-atom against the molecule graph in that measurement:
    # O-P(=O)(OH)-O-C-C-N-C (7-atom "1,3-dioxa-6-aza-2-phosphaheptyl"
    # skeleton + hydroxy-O + oxo-O = 9), and a 4-atom "2,4-dioxabutyl"
    # skeleton + one methyl branch = 5.
    ("2-hydroxy-2-oxo-1,3-dioxa-6-aza-2-phosphaheptyl", 9),
    ("3-hydroxy-3-oxo-2,4-dioxa-3-phosphabutyl", 6),
    ("3-methyl-2,4-dioxabut-3-en-1-yl", 5),
])
def test_multiplier_before_a_replacement_prefix_adds_no_atoms(token, truth):
    """The false-positive this fix repairs (a phase Task 2b): see
    ``.the workflow tooling/sdd/2026-08-12-phase0c-coverage-certificate-and-locant/task-2-report.md``.
    """
    est = token_arity(token, BindingKind.PREFIX)
    assert (not est.confident) or est.heavy_atoms == truth, (
        f"{token}: confidently answered {est.heavy_atoms}, truth {truth} "
        f"({est.basis})")


def test_stereo_descriptor_regex_does_not_swallow_a_parenthesised_substituent():
    """Regression guard for the NARROWNESS of ``_STEREO_GROUP``.

    Stripping stereo descriptors is safe only while the pattern matches the
    descriptor grammar exactly. A permissive character class would strip
    ``(chloro)`` as if it were ``(R)``, deleting a real substituent's atom and
    answering a CONFIDENT UNDERCOUNT -- the one failure mode this oracle may
    never have, because P6 would then reject a CORRECT name. The hardening
    that prevents it had no test; this is it.
    """
    est = token_arity("(chloro)methyl", BindingKind.PREFIX)
    assert est.confident, est.basis
    assert est.heavy_atoms == 2      # Cl + C, the '(chloro)' NOT stripped
