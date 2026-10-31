"""The default is the ``best-effort`` tier; ``emit_tier="pin"`` is the default of releases up to 1.0.x (lane DF).

User decision 2026-10-25: the default of ``name_compound`` and of the command line is the ``best-effort`` tier as
it runs when it is asked for by name (one pass, the strict PIN path first among its producers), and
``--emit-tier pin`` is the old default, byte for byte. Without OPSIN to read the names back, and for the styles
other than ``pin``, the default stays the PIN-only tier. The fixtures are the engine's own names, not claims
about the Blue Book.
"""
import logging

import pytest

from orthonym import Orthonym, classify_limit, name_compound, name_with_tree
from orthonym.errors import is_failure_name
from orthonym.cli import _emit_tier_flags
from orthonym.tiers import default_flags, default_tier, emit_tier_flags, resolve_switches

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

#: a molecule the PIN-only tier declines (NO_VERIFIED_PIN) and the best-effort tier names
BELOW_PIN = "OCCNCc1ccccc1"
BELOW_PIN_NAME = "2-(benzylamino)ethan-1-ol"
PINS = [("CCO", "ethanol"), ("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "(2R)-2-[4-(2-methylpropyl)phenyl]propanoic acid")]

#: structures no tier names: the code and the label of the decline, as the default gave them before it was the
#: best-effort tier (the label names the element)
UNNAMED = [
    ("O=[Th](=O)=O", "UNSUPPORTED_ELEMENT", "inorganic compound (not supported)"),   # (uranium: named since 2026-10-30)
    ("[NH3][Bi]([NH3])(Cl)Cl", "UNSUPPORTED_ELEMENT", "bismuth compound (not supported)"),   # (nickel, platinum: named)
    ("*CC", "WILDCARD_ATOMS", "compound with wildcard atoms (not supported)"),
]


def _best_effort():
    return Orthonym(style="pin", **_emit_tier_flags("best-effort"))


@pytest.mark.parametrize("smiles, expected", PINS)
def test_a_molecule_with_a_pin_gets_it_at_the_default_and_at_pin(smiles, expected):
    assert name_compound(smiles) == expected == name_compound(smiles, emit_tier="pin")


def test_a_molecule_without_a_pin_gets_the_best_effort_name_and_label():
    assert name_compound(BELOW_PIN) == BELOW_PIN_NAME == name_compound(BELOW_PIN, emit_tier="best-effort")
    assert name_compound(BELOW_PIN, emit_tier="pin") == ""          # the old default declines it
    row = _best_effort().name_tiered(BELOW_PIN)
    assert (row["name"], row["tier"]) == (BELOW_PIN_NAME, "pin_unverified")


def test_the_default_names_what_the_best_effort_tier_names():
    smiles_set = [s for s, _ in PINS] + [BELOW_PIN, "C1CC[C@H]2CCCC[C@H]2C1", "CC(=O)OCC(OC(=O)C)COC(=O)CC",
                                         "C[N+](C)(C)CC(=O)[O-]"]
    namer = _best_effort()
    for smiles in smiles_set:
        assert name_compound(smiles) == namer.name(smiles), smiles


def test_the_default_of_name_compound_logs_nothing():
    records = []

    class _Catch(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("orthonym")
    handler = _Catch(level=logging.DEBUG)
    logger.addHandler(handler)
    try:
        assert name_compound(BELOW_PIN) == BELOW_PIN_NAME
    finally:
        logger.removeHandler(handler)
    assert not [r for r in records if r.levelno >= logging.WARNING], records


def test_the_confidence_record_carries_the_tier_for_a_tier_above_pin_only():
    assert name_compound("CCO", include_confidence=True)["tier"] == "pin_verified"
    record = name_compound(BELOW_PIN, include_confidence=True)
    assert (record["name"], record["tier"]) == (BELOW_PIN_NAME, "pin_unverified")
    # the PIN-only tier is byte-identical to the release before: no key
    assert "tier" not in name_compound("CCO", include_confidence=True, emit_tier="pin")
    assert "tier" not in Orthonym().name_with_confidence("CCO")
    assert Orthonym(**_emit_tier_flags("best-effort")).name_with_confidence("CCO")["tier"] == "pin_verified"


def test_orthonym_without_options_is_still_the_pin_only_engine():
    namer = Orthonym()
    assert (namer._general_fallback, namer._general_fallback_unverified, namer._allow_aromatic_general,
            namer._full_coverage) == (False, False, False, False)


def test_a_tier_and_a_switch_given_together_is_an_error():
    with pytest.raises(ValueError):
        name_compound("CCO", emit_tier="pin", general_fallback=True)
    with pytest.raises(ValueError):
        name_compound("CCO", emit_tier="wide")


def test_the_default_switches():
    assert default_tier() == "best-effort" and default_flags() == emit_tier_flags("best-effort")
    assert default_tier("general") == "pin" and default_tier("cas") == "pin"
    assert resolve_switches("pin", None, own_call=True) == (True, True, True, False)
    # a nested call inherits the tier of the call that makes it; a switch given selects its tier
    assert resolve_switches("pin", None, own_call=False) == (None, None, None, None)
    assert resolve_switches("pin", None, True, own_call=True) == (True, None, None, None)
    assert resolve_switches("pin", "pin") == (False, False, False, False)
    assert resolve_switches("pin", "full-coverage") == (True, True, True, True)


def test_without_opsin_the_default_is_the_pin_only_tier(monkeypatch):
    """A wider tier ships only names that read back: with no OPSIN (the reduced mode, the raw-output switches)
    the default is the tier it was."""
    import orthonym.tiers as tiers
    monkeypatch.setattr(tiers, "verification_active", lambda: False)
    assert default_tier() == "pin" and resolve_switches("pin", None, own_call=True) == (False, False, False, False)


def test_a_nested_call_inherits_the_tier_of_the_call_that_makes_it(monkeypatch):
    from orthonym.assembly import fragment_naming
    monkeypatch.setattr(fragment_naming, "name_scope_depth", lambda: 1)
    seen = []
    real = resolve_switches

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("own_call"))
        return real(*args, **kwargs)

    monkeypatch.setattr("orthonym.tiers.resolve_switches", _spy)
    name_compound("CCO")
    assert seen == [False]


@pytest.mark.parametrize("smiles, code, label", UNNAMED)
def test_a_structure_no_tier_names_gets_the_empty_string_and_its_label(smiles, code, label):
    """The empty name, the label of ``on_failure="label"`` (which names the element) and the limit code of
    ``on_failure="raise"`` are the default's, as they were before it was the best-effort tier."""
    from orthonym import OrthonymLimitError
    assert name_compound(smiles) == ""
    assert name_compound(smiles, on_failure="label") == label
    assert name_compound(smiles, emit_tier="pin", on_failure="label") == label
    with pytest.raises(OrthonymLimitError) as err:
        name_compound(smiles, on_failure="raise")
    assert (err.value.code, err.value.message) == (code, label)
    assert classify_limit(smiles).code == code
    assert is_failure_name(name_compound(smiles, on_failure="label"))


def test_name_with_tree_and_classify_limit_follow_the_default():
    assert name_with_tree(BELOW_PIN).name == BELOW_PIN_NAME
    assert classify_limit(BELOW_PIN) is None
    assert classify_limit("CCO") is None
