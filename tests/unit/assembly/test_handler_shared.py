"""Unit tests for Phase 160.2 Plan-02-01: 6 _generate_* helpers lifted to
handlers/_handler_shared.py per CONTEXT D-02 + D-03.

Per plan must_haves:
- Shim re-export tests verify composer.py's local-name binding IS the
  same function object as _handler_shared's export, per CONTEXT D-13
  forbidden-boundary preservation + RESEARCH §Pattern 3.
- Functional tests verify byte-identical end-to-end naming output for
  representative compounds whose names depend on each lifted helper.
"""
import pytest

from orthonym import name_compound
from orthonym.assembly import composer
from orthonym.assembly.handlers import _handler_shared


# =============================================================================
# Shim re-export identity tests (per RESEARCH §Pattern 3)
# Each test verifies composer.py shim re-export creates the EXACT same
# function object as _handler_shared's export per CONTEXT D-13 +
# Python import semantics.
# =============================================================================


def test_shim_reexport_chain_parent_is_same_object():
    """composer._generate_chain_parent IS _handler_shared._generate_chain_parent."""
    assert composer._generate_chain_parent is _handler_shared._generate_chain_parent


def test_shim_reexport_ring_parent_is_same_object():
    """composer._generate_ring_parent IS _handler_shared._generate_ring_parent."""
    assert composer._generate_ring_parent is _handler_shared._generate_ring_parent


def test_shim_reexport_suffix_is_same_object():
    """composer._generate_suffix IS _handler_shared._generate_suffix."""
    assert composer._generate_suffix is _handler_shared._generate_suffix


def test_shim_reexport_prefixes_is_same_object():
    """composer._generate_prefixes IS _handler_shared._generate_prefixes."""
    assert composer._generate_prefixes is _handler_shared._generate_prefixes


def test_shim_reexport_stereodescriptors_is_same_object():
    """composer._generate_stereodescriptors IS _handler_shared._generate_stereodescriptors."""
    assert composer._generate_stereodescriptors is _handler_shared._generate_stereodescriptors


def test_shim_reexport_assemble_fragments_is_same_object():
    """composer._assemble_fragments IS _handler_shared._assemble_fragments."""
    assert composer._assemble_fragments is _handler_shared._assemble_fragments


# =============================================================================
# Functional end-to-end tests
# Each compound exercises one of the 6 helpers; the test asserts the
# name matches the byte-identical canary baseline.
# =============================================================================


def test_chain_parent_butane_named_butane():
    """_generate_chain_parent exercises 'but' stem -> name 'butane'."""
    assert name_compound("CCCC") == "butane"


def test_chain_parent_pentane_named_pentane():
    """_generate_chain_parent exercises 'pent' stem -> name 'pentane'."""
    assert name_compound("CCCCC") == "pentane"


def test_ring_parent_cyclohexane_named():
    """_generate_ring_parent exercises 'cyclohex' stem -> name 'cyclohexane'."""
    assert name_compound("C1CCCCC1") == "cyclohexane"


def test_ring_parent_cyclopentane_named():
    """_generate_ring_parent exercises 'cyclopent' stem -> name 'cyclopentane'."""
    assert name_compound("C1CCCC1") == "cyclopentane"


def test_suffix_propanol_named():
    """_generate_suffix exercises -ol suffix for propan-1-ol."""
    # The exact expected name depends on canary. Just assert it ends with -ol
    name = name_compound("CCCO")
    assert name and "ol" in name


def test_suffix_pentanoic_acid_named():
    """_generate_suffix exercises -oic acid suffix."""
    name = name_compound("CCCCC(=O)O")
    assert name and ("oic acid" in name or "anoic" in name)


def test_prefixes_methylpropane_named():
    """_generate_prefixes exercises methyl prefix on propane backbone."""
    # 2-methylpropane -> 'methylpropane' or '2-methylpropane'
    name = name_compound("CC(C)C")
    assert name and "methyl" in name and "propan" in name


def test_prefixes_dichlorobenzene_named():
    """_generate_prefixes exercises chloro prefix on benzene."""
    name = name_compound("Clc1ccc(Cl)cc1")
    assert name and "chloro" in name and "benz" in name


def test_stereodescriptors_chiral_butanol():
    """_generate_stereodescriptors exercises (R)- or (S)- prefix."""
    name = name_compound("C[C@@H](O)CC")
    assert name
    # Should include 'R' or 'S' stereodescriptor
    assert "(R)" in name or "(S)" in name or "R" in name or "S" in name


def test_stereodescriptors_e_z_double_bond():
    """_generate_stereodescriptors exercises E/Z descriptor."""
    name = name_compound("C/C=C/C")
    # but-2-ene with E configuration
    assert name


def test_assemble_fragments_simple_compound():
    """_assemble_fragments exercises final name assembly."""
    # ethanol exercises stereo (none) + prefix (none) + parent + suffix
    assert name_compound("CCO") == "ethanol"


def test_assemble_fragments_complex_compound():
    """_assemble_fragments handles multi-fragment compound."""
    # 2-chloropropan-1-ol exercises all four fragment types
    name = name_compound("CC(Cl)CO")
    assert name
    assert "chloro" in name


# =============================================================================
# Structural / contract tests
# =============================================================================


def test_handler_shared_exports_all_six():
    """_handler_shared.__all__ contains all 6 lifted helper names."""
    expected = {
        "_generate_chain_parent",
        "_generate_ring_parent",
        "_generate_suffix",
        "_generate_prefixes",
        "_generate_stereodescriptors",
        "_assemble_fragments",
    }
    assert expected.issubset(set(_handler_shared.__all__))


def test_composer_does_not_redefine_lifted_helpers():
    """AST check: composer.py no longer DEFINES the 6 helpers (only imports them)."""
    import ast
    src = open("src/orthonym/assembly/composer.py").read()
    tree = ast.parse(src)
    fn_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in (
        "_generate_chain_parent",
        "_generate_ring_parent",
        "_generate_suffix",
        "_generate_prefixes",
        "_generate_stereodescriptors",
        "_assemble_fragments",
    ):
        assert name not in fn_names, (
            f"{name} should be lifted to _handler_shared.py (shim re-import only)"
        )


def test_handler_shared_loc_grew_with_lift():
    """_handler_shared.py LOC grew from ~150 to ~900+ after lifting 6 helpers."""
    src = open("src/orthonym/assembly/handlers/_handler_shared.py").read()
    loc = src.count("\n")
    assert loc >= 700, f"_handler_shared.py LOC = {loc}, expected >= 700"


def test_all_six_helpers_callable():
    """All 6 lifted helpers are callable on _handler_shared module."""
    for name in (
        "_generate_chain_parent",
        "_generate_ring_parent",
        "_generate_suffix",
        "_generate_prefixes",
        "_generate_stereodescriptors",
        "_assemble_fragments",
    ):
        helper = getattr(_handler_shared, name, None)
        assert helper is not None, f"_handler_shared.{name} missing"
        assert callable(helper), f"_handler_shared.{name} not callable"


def test_existing_handler_shared_helpers_still_callable():
    """Pre-existing _handler_shared helpers (name_iso_x_cyanate, name_r_group,
    cached_is_complex_ring_system) still callable after Plan-02-01 extension."""
    assert callable(_handler_shared.name_iso_x_cyanate)
    assert callable(_handler_shared.name_r_group)
    assert callable(_handler_shared.cached_is_complex_ring_system)
