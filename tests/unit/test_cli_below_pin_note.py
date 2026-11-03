"""The CLI note for a name below the PIN carries the first reason a check gave (S2I-4)."""
from orthonym.cli import _BELOW_PIN_NOTE, _below_pin_note


def test_no_reason_keeps_the_note():
    assert _below_pin_note({"spelling_failures": []}) == _BELOW_PIN_NOTE


def test_the_first_reason_is_appended():
    row = {"spelling_failures": [{"rule": "P-16.3.1", "detail": "'methyl' is cited 2 times"}, {"rule": "x", "detail": "y"}]}
    note = _below_pin_note(row)
    assert note.startswith(_BELOW_PIN_NOTE)
    assert note.endswith("first reason: P-16.3.1 'methyl' is cited 2 times")
