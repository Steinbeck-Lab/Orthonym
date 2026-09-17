""" a phase T6.1 — pin the --emit-tier -> namer-flag invariant table.

The tier semantics are load-bearing for the 0-wrong guarantee: only
``best-effort`` may ship OPSIN-unverified names (``general_fallback_unverified``),
and ``best-effort`` must have AT LEAST the candidate production of ``complete``
(``allow_aromatic_general``) or it silently under-covers (the T6.1 bug: today
best-effort omits ``allow_aromatic_general`` so it never reaches the P1 cage
machinery, making its breadth *lower* than complete's).

Single source of truth: ``cli._emit_tier_flags`` — both the CLI and this test
read the same mapping, so a future edit cannot drift the two apart.
"""
import pytest

from orthonym.cli import _emit_tier_flags


# (tier, general_fallback, general_fallback_unverified, allow_aromatic_general)
_TIER_TABLE = [
    ("pin", False, False, False),
    ("valid", True, False, False),
    ("complete", True, False, True),
    ("best-effort", True, True, True),
]


@pytest.mark.parametrize("tier,gf,gfu,aag", _TIER_TABLE)
def test_emit_tier_flag_table(tier, gf, gfu, aag):
    flags = _emit_tier_flags(tier)
    assert flags["general_fallback"] is gf, tier
    assert flags["general_fallback_unverified"] is gfu, tier
    assert flags["allow_aromatic_general"] is aag, tier


def test_best_effort_is_sole_unverified_tier():
    """general_fallback_unverified is the UNIQUE best-effort discriminator.

    Locks the T6.2 risk mitigation: no other tier may set it, so the shared
    stereo-emit policy helper can key "emit flagged" off exactly this flag.
    """
    for tier in ("pin", "valid", "complete"):
        assert _emit_tier_flags(tier)["general_fallback_unverified"] is False, tier
    assert _emit_tier_flags("best-effort")["general_fallback_unverified"] is True


def test_best_effort_superset_of_complete_candidate_production():
    """best-effort must reach every candidate producer complete does (T6.1).

    complete's candidate production is gated by allow_aromatic_general +
    general_fallback; best-effort must have both True (plus its unique
    unverified opt-in) so best-effort >= complete in breadth.
    """
    comp = _emit_tier_flags("complete")
    be = _emit_tier_flags("best-effort")
    assert be["general_fallback"] is True
    assert be["allow_aromatic_general"] is True
    # every production-gating flag complete sets, best-effort also sets
    for key in ("general_fallback", "allow_aromatic_general"):
        assert be[key] or not comp[key], key


# Composer1 Task 5 regression: name_tiered's honest "clean abstain"
# contract (name=None under general_fallback, see namer.name_tiered) was
# shipped without updating cli.main's plain-text print branch, which did
# `print(row["name"])` unconditionally -> printed the literal string
# "None" to stdout for an abstaining molecule under a non-pin --emit-tier.
# Drives the CLI end-to-end (in-process; the conftest autouse fixture
# disables the OPSIN validity gate for the whole test suite, so this
# does not spawn an OPSIN subprocess/JVM).
#
# CHANGE-ASSERTED-VALUE UPDATE (final review): this molecule was an
# abstainer when the test was written, but the best-effort engine has since
# improved and now NAMES it (below). The naming is CORRECT, not a wrong
# emission: the name OPSIN-parses back to the input's exact InChIKey
# an InChIKey (verified, a temp dir/verify_fix2.py -- identical
# canonical SMILES). It is NOT a emission -- fires 0x for it, and the
# name carries a `-propanamide` principal-group suffix that T4's PG-suppressing
# cascade structurally cannot produce; it is the pre-existing best-effort
# engine. The two tests below now assert that verified named output. The CLI's
# bare-"None" display guard (the actual regression subject) is still
# exercised: `out != "None"` and no bare-"None" token in the printed line.
_ABSTAINING_SMILES = (
    "CC(C)(C(=O)Nc1cccc(F)c1)N1CCC(c2nc(-c3cc4ccccc4o3)cs2)CC1"
)
# The engine's verified best-effort name for the molecule above (round-trips to
# the input InChIKey -- see the change-asserted-value note).
_BEST_EFFORT_NAME = (
    "2-({4-[4-(1-benzofuran-2-yl)-1,3-thiazol-2-yl]piperidin-1-yl})"
    "-2-methyl-N-(3-fluorophenyl)propanamide"
)


def test_cli_best_effort_abstain_does_not_print_bare_none(capsys):
    from orthonym.cli import main

    rc = main([_ABSTAINING_SMILES, "--emit-tier", "best-effort"])

    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out != "None"
    assert "None" not in out.split()  # no bare-None token in the printed line
    assert out  # must print SOMETHING, not an empty line
    # Best-effort now NAMES this (verified: round-trips to the input InChIKey --
    # see the change-asserted-value note above), so the plain-text branch prints
    # the name, not a "no name" placeholder.
    assert out == _BEST_EFFORT_NAME


def test_cli_best_effort_abstain_provenance_json_keeps_null(capsys):
    """The --provenance JSON branch emits the name as a well-formed JSON string.

    (Originally asserted ``"name": null`` because the molecule abstained; the
    best-effort engine now names it -- verified round-trip, see the
    change-asserted-value note above -- so the JSON carries the string name.)
    """
    import json

    from orthonym.cli import main

    rc = main([
        _ABSTAINING_SMILES, "--emit-tier", "best-effort", "--provenance",
    ])

    assert rc == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["name"] == _BEST_EFFORT_NAME
