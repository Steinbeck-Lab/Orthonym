"""Test helper: switch off the spelling checks of one Blue Book rule id for one test.

``validation.pin_spelling`` keeps its checks in a registry keyed by rule id ("",
"",...). A known-positive test that shows what a producer screen alone stops must
switch off a label check that would stop the same name (for example the check of the
linear phane lane). This helper is the one place that touches the registry, so a test does not
depend on its layout.
"""
from orthonym.validation import pin_spelling


def disable_spelling_rule(monkeypatch, rule_id: str) -> None:
    """No check of ``rule_id`` runs until the test ends (``monkeypatch`` restores the registry).
    Works whether or not a check is registered under ``rule_id``."""
    pin_spelling.registered_rules()  # loads the built-in check modules first
    monkeypatch.setitem(pin_spelling._REGISTRY, rule_id, [])
