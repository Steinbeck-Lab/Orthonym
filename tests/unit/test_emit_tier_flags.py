"""v27 Phase 6 T6.1 — pin the --emit-tier -> namer-flag invariant table.

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


# v28 Composer1 Task 5 regression: name_tiered's honest T5 "clean abstain"
# contract (name=None under general_fallback, see namer.name_tiered) was
# shipped without updating cli.main's plain-text print branch, which did
# `print(row["name"])` unconditionally -> printed the literal string
# "None" to stdout for an abstaining molecule under a non-pin --emit-tier.
# Drives the CLI end-to-end (in-process; the conftest autouse fixture
# disables the OPSIN validity gate for the whole test suite, so this
# does not spawn an OPSIN subprocess/JVM).
_ABSTAINING_SMILES = (
    "CC(C)(C(=O)Nc1cccc(F)c1)N1CCC(c2nc(-c3cc4ccccc4o3)cs2)CC1"
)


def test_cli_best_effort_abstain_does_not_print_bare_none(capsys):
    from orthonym.cli import main

    rc = main([_ABSTAINING_SMILES, "--emit-tier", "best-effort"])

    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out != "None"
    assert "None" not in out.split()  # no bare-None token in the printed line
    assert out  # must print SOMETHING, not an empty line
    assert "no name" in out.lower()


def test_cli_best_effort_abstain_provenance_json_keeps_null(capsys):
    """The --provenance JSON branch is untouched: name:null stays well-defined
    JSON, only the plain-text branch gets the display guard."""
    from orthonym.cli import main

    rc = main([
        _ABSTAINING_SMILES, "--emit-tier", "best-effort", "--provenance",
    ])

    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert '"name": null' in out
