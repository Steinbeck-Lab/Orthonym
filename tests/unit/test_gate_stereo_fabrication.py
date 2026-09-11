""" a phase (C3): gate backstop rejects OPSIN-parseable stereo
FABRICATION (nb > na) while still tolerating the achiral case (na == nb == 0).

See.the workflow tooling/sdd/2026-08-16--phase1-stereo-honesty/task-5-brief.md.
"""
from orthonym.namer import _self_consistency_verdict


def test_fabrication_over_specifies_is_rejected():
    # input flat (na=0), name's OPSIN parse adds stereo (nb>0) -> mismatch
    flat = "CC(N)C(=O)O"                 # flat alanine
    stereo = "N[C@@H](C)C(=O)O"          # L-alanine (what a retained name reparses to)
    assert _self_consistency_verdict(flat, stereo) == "mismatch"


def test_achiral_both_zero_is_ok():
    assert _self_consistency_verdict("NCC(=O)O", "NCC(=O)O") == "ok"


def test_exact_stereo_match_is_ok():
    s = "N[C@@H](C)C(=O)O"
    assert _self_consistency_verdict(s, s) == "ok"
