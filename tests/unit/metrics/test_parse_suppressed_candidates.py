"""The SELF-01 log line carries a PAYLOAD, and `_GATE_RES` was throwing it away.

`metrics/breadth.py:109` already matches

    SELF-01 suppressed (different molecule): '<name>' (opsin=<smiles>)

to attribute the abstention, but its regex has no capture groups, so the built
candidate and the molecule OPSIN read it as were both discarded. Without them the
162 `GATE_SUPPRESSED:self01_mismatch` rows can only be counted, never split by
mechanism -- and v28 built a single-lever fix for a class of that shape and
measured 0/231 afterwards.

The apostrophe case is load-bearing: IUPAC names routinely contain primes
(2,2'-bi-3,1,5-benzoxadiarsepine, 1,1'-bistibinane), so a `'([^']+)'` pattern
truncates the name at the first prime and silently reports a different candidate
than the one that was suppressed.
"""
from orthonym.metrics.breadth import parse_suppressed_candidates

SELF01 = "SELF-01 suppressed (different molecule): {!r} (opsin={})"


def test_extracts_name_and_opsin_smiles():
    lines = [SELF01.format("cholest-5-ene", "CC1CCC2C1")]
    got = parse_suppressed_candidates(lines)
    assert got["suppressed_name"] == "cholest-5-ene"
    assert got["suppressed_opsin_smiles"] == "CC1CCC2C1"
    assert got["suppressed_count"] == 1


def test_a_name_containing_primes_is_not_truncated():
    """Greedy match to the LAST "' (opsin=" -- not to the first apostrophe."""
    name = "2,2'-bi-3,1,5-benzoxadiarsepine"
    lines = ["SELF-01 suppressed (different molecule): "
             f"'{name}' (opsin=C1CC[As]OC1)"]
    got = parse_suppressed_candidates(lines)
    assert got["suppressed_name"] == name
    assert got["suppressed_opsin_smiles"] == "C1CC[As]OC1"


def test_last_suppression_wins_and_all_are_kept():
    """The retry cascade suppresses several candidates; the LAST is terminal.

    All are kept so a follow-up can tell "one bad candidate" from "the producer
    kept offering variants of the same wrong structure".
    """
    lines = [
        SELF01.format("first-wrong", "CCO"),
        "some unrelated line",
        SELF01.format("second-wrong", "CCC"),
    ]
    got = parse_suppressed_candidates(lines)
    assert got["suppressed_name"] == "second-wrong"
    assert got["suppressed_opsin_smiles"] == "CCC"
    assert got["suppressed_count"] == 2
    assert got["suppressed_all"] == [("first-wrong", "CCO"),
                                     ("second-wrong", "CCC")]


def test_no_suppression_yields_no_keys():
    """An empty dict, so callers can `row.update(...)` without adding null keys."""
    assert parse_suppressed_candidates(["DROP-09 substituent_skip: reason=x"]) == {}
    assert parse_suppressed_candidates([]) == {}


def test_an_unrelated_gate_line_is_not_mistaken_for_self01():
    """The OPSIN validity gate suppresses too, but is a different mechanism."""
    lines = ["OPSIN validity gate suppressed unparseable name: 'oxarsane'"]
    assert parse_suppressed_candidates(lines) == {}


def test_a_logger_prefix_does_not_break_the_match():
    """Handlers may hand over formatted records, not bare messages."""
    lines = ["WARNING:orthonym.namer:" + SELF01.format("wrong", "CCO")]
    got = parse_suppressed_candidates(lines)
    assert got["suppressed_name"] == "wrong"
