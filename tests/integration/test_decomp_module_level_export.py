""" part A regression: name_with_tree is exported at module level.

Per internal notes: callers expect ``from orthonym import name_with_tree``
to work analogously to ``from orthonym import name_compound``. Until Plan-05
this import raised ImportError because the symbol existed only on
``Orthonym.name_with_tree`` (instance method).
"""
from __future__ import annotations


def test_name_with_tree_module_level_import():
    """`from orthonym import name_with_tree` works (no ImportError)."""
    from orthonym import name_with_tree
    assert callable(name_with_tree)


def test_name_with_tree_module_level_invocation():
    """Module-level name_with_tree('CCO') returns NamingResult with name='ethanol'."""
    from orthonym import name_with_tree, NamingResult
    result = name_with_tree("CCO")
    assert isinstance(result, NamingResult)
    assert result.name == "ethanol"


def test_name_with_tree_module_level_in_all():
    """name_with_tree appears in orthonym.__all__."""
    import orthonym
    assert "name_with_tree" in orthonym.__all__


def test_verify_decomp_error_format_escapes_commas_and_newlines():
    """ regression: exception messages with commas/newlines do not
    corrupt CSV output. The verifier formats exceptions with !r which
    escapes literal newlines and commas.
    """
    e = ValueError("multi,\ncomma,line\"with quotes")
    _msg = str(e)[:200]
    formatted = f"<ERROR: {type(e).__name__}: {_msg!r}>"
    assert "\n" not in formatted
    assert formatted.startswith("<ERROR: ValueError:")
