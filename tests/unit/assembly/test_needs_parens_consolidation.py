"""Phase 169.7 BBR-HYG(e) — needs-parens predicate consolidation tripwire.

The audit (DEF-8) found N divergent "needs-parens / enclosing-mark" predicates:
  - rules/amides.py:_has_positional_locants        (digit-only locant test)
  - assembly/naming_utils.py:is_complex_substituent (complex-substituent test)
  - assembly/naming_utils.py:apply_enclosing_marks
  - rules/ortho_fused.py:format_substituent_prefix  (a second, different-signature copy)
with assembly/naming_utils.py:format_substituent_prefix as the ONE correct reference.

169.7 BEGINS the consolidation (BBR-HYG(e)): it inventories the predicates (a comment
block at the canonical reference) and arms THIS xfail-strict tripwire. The bulk
consolidation lands in BBR-ASM / Phase 171 (CONTEXT D-14 scope guard). When Phase 171
unifies the predicates, this assertion holds and xfail(strict) turns it into a FAILURE,
forcing removal of the marker. NO production behaviour is changed by 169.7 here.
"""
import pytest

# Audit-cited divergence inputs: substituted substituents with NO digit locant.
# is_complex_substituent() correctly flags them complex (need enclosing marks per
# P-16.3.3), but the digit-only _has_positional_locants() does not.
_DIVERGENCE_INPUTS = ["trifluoromethyl", "tert-butyl", "chloromethyl", "methylsulfanyl"]


@pytest.mark.unit
@pytest.mark.parametrize("name", _DIVERGENCE_INPUTS)
def test_needs_parens_predicates_agree(name):
    # Phase 171 BBR-ASM CONSOLIDATED the enclosing-mark decision onto ONE predicate:
    # amides._has_positional_locants now delegates to is_complex_substituent (the same
    # predicate get_multiplier_prefix uses for bis/tris), so the two views agree.
    from orthonym.assembly.naming_utils import is_complex_substituent
    from orthonym.rules.amides import _has_positional_locants
    assert is_complex_substituent(name) == _has_positional_locants(name)


@pytest.mark.unit
def test_canonical_reference_documents_consolidation():
    """The canonical reference names the Phase-171 consolidation + the divergent
    predicates (the inventory the 'begin' requires)."""
    import inspect
    from orthonym.assembly import naming_utils
    src = inspect.getsource(naming_utils.format_substituent_prefix)
    assert "Phase 171" in src and "_has_positional_locants" in src and "is_complex_substituent" in src
