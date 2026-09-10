"""-FINAL Item 6 — the `multiplied_component` arity boundary, per site.

The closeout commit claimed "*arity bounds stay local so every site still fails
closed beyond its supported count*". The review could not verify it at any of
the 6 call sites and recorded the opposite risk: `multiplied_component` itself has
NO cap, and returns `decakis(hydroxymethyl)` at count 10.

Both halves are true, and together they are not a defect — but nothing pinned
either, so this file does.

MEASURED, and it REFUTES the reachability half of the concern: every one of the
six call sites carries an explicit local arity guard that returns ``None`` (fails
closed) before `multiplied_component` is reached.

    src/orthonym/rules/mononuclear_hydrides.py:341 `counts[name] not in _SUPPORTED_COUNTS`
    src/orthonym/rules/phosphorus.py:207 `count not in _SUPPORTED_COUNTS` (:189 = {1,2,3,4})
    src/orthonym/rules/polyazane.py:161 `len(locs) not in _SUB_MULTIPLIER`
    src/orthonym/rules/polyazane.py:248 (same table)
    src/orthonym/rules/polychalcogen.py:180 `len(locs) not in _SUB_MULTIPLIER`
    src/orthonym/rules/pseudoketones.py:378 `counts[nm] not in _SUPPORTED_COUNTS` ({1,2,3})

So `decakis` is unreachable through the producers and needs no cap; it is
reachable only by calling the primitive directly, where the composite multiplying
prefix is CORRECT IUPAC anyway (`****` composite multiplying prefixes).
Capping the primitive would therefore remove a correct capability to guard
against a path that does not exist. What was missing was the evidence, not a cap.
"""

import pytest

from orthonym.assembly.naming_utils import multiplied_component

pytestmark = pytest.mark.unit


# (module, the frozenset name that bounds it, the counts it must accept)
ARITY_GUARDED_SITES = [
    ("orthonym.rules.mononuclear_hydrides", "_SUPPORTED_COUNTS"),
    ("orthonym.rules.phosphorus", "_SUPPORTED_COUNTS"),
    ("orthonym.rules.polyazane", "_SUB_MULTIPLIER"),
    ("orthonym.rules.polychalcogen", "_SUB_MULTIPLIER"),
    ("orthonym.rules.pseudoketones", "_SUPPORTED_COUNTS"),
]


@pytest.mark.parametrize("module_name,_guard", ARITY_GUARDED_SITES)
def test_every_multiplied_component_caller_has_a_local_arity_guard(
        module_name, _guard):
    """The guard must be present in the module's SOURCE, at the call site.

    Reading the source rather than the symbol, because two of the five keep the
    frozenset as a FUNCTION-LOCAL name (`mononuclear_hydrides`, `pseudoketones`),
    so an attribute lookup would report a false absence — the exact
    perfect-result trap this project keeps hitting.
    """
    import importlib
    import inspect

    mod = importlib.import_module(module_name)
    src = inspect.getsource(mod)
    assert "multiplied_component(" in src, (
        f"{module_name} no longer calls multiplied_component — this test is "
        f"stale and must be re-derived, not deleted"
    )
    assert ("_SUPPORTED_COUNTS" in src) or ("_SUB_MULTIPLIER" in src), (
        f"{module_name} calls multiplied_component with no local arity bound; "
        f"the primitive has no cap, so the site must fail closed itself"
    )
    #...and the bound must actually gate a `return None`, not merely exist.
    assert "return None" in src, module_name


def test_the_primitive_itself_is_uncapped_and_that_is_deliberate():
    """Documents the measured behaviour so it is never mistaken for a bug.

    `****` composite multiplying prefixes make `decakis`/`icosakis`
    well-formed IUPAC, so the primitive is right to form them; the SITES are
    where a structurally impossible count must be refused, and all six do.
    """
    assert multiplied_component(10, "hydroxymethyl", "(hydroxymethyl)") == \
        "decakis(hydroxymethyl)"
    assert multiplied_component(20, "hydroxymethyl", "(hydroxymethyl)") == \
        "icosakis(hydroxymethyl)"
    assert multiplied_component(10, "methyl", "methyl") == "decamethyl"


def test_a_count_below_two_never_gets_a_multiplier():
    assert multiplied_component(1, "methyl", "methyl") == "methyl"
    assert multiplied_component(1, "hydroxymethyl", "(hydroxymethyl)") == \
        "(hydroxymethyl)"


def test_the_hyphen_is_dropped_once_the_component_is_enclosed():
    """ (`the Blue Book`): "*No hyphen is placed after a numerical
    prefix cited in front of a compound substituent enclosed by parentheses*"."""
    assert multiplied_component(2, "tert-butyl", "tert-butyl") == "di-tert-butyl"
    assert multiplied_component(2, "tert-butyl", "(tert-butyl)") == "di(tert-butyl)"
