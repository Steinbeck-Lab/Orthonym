"""Orthonym.name restores the caller's provenance context on EVERY exit.

name publishes four ContextVars for a top-level call (general_fallback_ctx,
best_effort_ctx, allow_aromatic_general_ctx, full_coverage_ctx) so that
recursively built namers inherit the tier. Two exits leaked them (TRIAGE C2,
found through test_v52_phase02_noncarbon's N-hydroxy rows, which failed only
after some other test on the same xdist worker):

* the isotope decorator names the stripped skeleton with the SAME instance
  (``namer.name(...)``); that re-entry also runs as a top-level naming, and the
  tokens lived on the instance, so it overwrote the outer frame's tokens, reset
  its own, and left the outer ones unreset;
* the isotope exits sat before the main try/finally and ended only the session.

After one best-effort naming of an isotope-labelled molecule, every later
default (PIN-tier) call in the process ran with best-effort flags, e.g.
ON[C@@H](C)CC -> '(2S)-2-(hydroxyamino)butane' instead of the PIN tier's
decline. A name must not depend on what the process named before.
"""
import pytest

import orthonym.namer as _namer
from orthonym import Orthonym, name_compound
from orthonym.assembly import fragment_naming
from orthonym.metrics.provenance import (allow_aromatic_general_ctx, best_effort_ctx,
                                         full_coverage_ctx, general_fallback_ctx)

_CTX = (general_fallback_ctx, best_effort_ctx, allow_aromatic_general_ctx, full_coverage_ctx)
N_HYDROXY = "ON[C@@H](C)CC"


CLEAN = (False, False, False, False)


def _ctx_values():
    return tuple(v.get() for v in _CTX)


@pytest.fixture(autouse=True)
def _clean_context():
    """Each test starts from the default (PIN-tier) context and gives the caller's
    context back whatever the code under test leaks (a ContextVar reset restores
    the value from before that token's set, later sets included)."""
    toks = [(v, v.set(False)) for v in _CTX]
    yield
    for var, tok in reversed(toks):
        var.reset(tok)


def _best_effort_engine():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                    allow_aromatic_general=True)


@pytest.mark.parametrize("isotope_smiles", [
    "[2H]C([2H])([2H])O",          # (2H3)methanol
    "[2H]C([2H])([2H])[13CH2]O",   # two nuclides
])
def test_best_effort_isotope_naming_restores_the_context(isotope_smiles):
    before_name = name_compound(N_HYDROXY)
    name = _best_effort_engine().name(isotope_smiles)
    assert name and "unknown" not in name, name        # the witness path really ran
    assert _ctx_values() == CLEAN
    assert name_compound(N_HYDROXY) == before_name


def test_same_instance_reentry_keeps_the_outer_tokens():
    """The isotope path re-enters the SAME engine; after the outer call the ctx is
    back to what the caller had, and a second call on the engine does not
    inherit anything either."""
    eng = _best_effort_engine()
    eng.name("[2H]C([2H])([2H])O")
    assert _ctx_values() == CLEAN
    eng.name("[2H]C([2H])([2H])O")
    assert _ctx_values() == CLEAN


def test_an_exception_in_the_prelude_restores_context_and_session(monkeypatch):
    """An exception before the main try (here: the acid-salt normalizer, which
    runs only for a multi-fragment input with a cation) still resets the four
    ctx vars and ends the naming session."""
    import orthonym.rules.salts as salts

    def _boom(smiles):
        raise RuntimeError("injected")
    monkeypatch.setattr(salts, "normalize_imbalanced_acid_salt", _boom)
    with pytest.raises(RuntimeError, match="injected"):
        _best_effort_engine().name("[Na+].OC(=O)O")
    assert _ctx_values() == CLEAN
    assert fragment_naming.is_top_level_naming()       # session depth back to 0
    assert name_compound("CCO") == "ethanol"


def test_reset_helper_is_lifo_and_idempotent():
    toks = [(general_fallback_ctx, general_fallback_ctx.set(True)),
            (general_fallback_ctx, general_fallback_ctx.set(False))]
    _namer.Orthonym._reset_published_ctx(toks)
    assert toks == [] and general_fallback_ctx.get() is False
    _namer.Orthonym._reset_published_ctx(toks)          # no-op
    assert general_fallback_ctx.get() is False
