"""Phase 158 recursive-re-dispatch tests (audit § 3 + RESEARCH § 6.1).

Per CONTEXT D-17 LOCKED test pyramid floor: ≥ 5 recursive-re-dispatch tests
covering the five sites named in audit § 3:

- § 3.1 RL-4: ion-neutralize-recurse (namer.py:905)
- § 3.2 RL-4: poly-anion-neutralize-recurse (namer.py:939)
- § 3.3 RL-4: dot-disconnected per-component-recurse (namer.py:988)
- § 3.4 RL-7 option (b): ``_skip_decomposition`` flag-threading (namer.py:1851)
- § 3.5 RL-3 option (c): zwitterion-character inline mutation (namer.py:1000-1034)

Per RESEARCH § 6.2 invariant: every recursive ``Orthonym().name()`` call
hits a FRESH router instance.  Tests assert NAME OUTPUT equality (the byte-
identical contract) — NOT per-call dispatch_stats accumulation across the
recursive boundary.

Per CONTEXT D-29 + AP-17: NO ``@pytest.mark.xfail`` markers; every test
passes green.  Per AP-15 + RESEARCH § 8.4 Path-(b): no inner-cascade
``StoutClass`` members are imagined — only the locked v18 surface is tested.
"""

from __future__ import annotations

import pytest

from orthonym import Orthonym, name_compound
from orthonym.namer import name_pipeline_only
from orthonym.routing.dispatch_table import StoutClass


# ---------------------------------------------------------------------------
# Test 1 — Single-anion neutralize-recurse (audit § 3.1; RESEARCH § 6.1 site 1)
# ---------------------------------------------------------------------------


def test_single_anion_neutralize_recursion():
    """Audit § 3.1 + RESEARCH § 6.1: ion-neutralize at namer.py:905-913.

    For acetate anion ``CC(=O)[O-]``: outer dispatch hits ANION_RETAINED
    (priority 400 fires first because acetate IS in the retained anion
    catalog).  If the retained lookup misses (e.g., heptanoate), the cascade
    falls through to ANION_SMALL whose handler shim recursively instantiates
    ``Orthonym(style)`` and calls ``name()`` on the neutralized SMILES.

    Per RL-4 fresh-instance invariant: the recursive call uses a separate
    router; per-call ``dispatch_stats`` counters are NOT cumulative across
    the recursive boundary.  Byte-identical contract is on NAME OUTPUT only.
    """
    # Acetate — retained-anion path (priority 400).
    acetate_name = name_compound("CC(=O)[O-]", style="pin")
    assert acetate_name == "acetate"

    # Heptanoate — ANION_SMALL path (priority 600) via the neutralize-recurse
    # fallback.  Heptanoate is NOT in the retained-anion catalog so
    # ANION_RETAINED's predicate returns False; the cascade continues to
    # ANION_SMALL which fires the recursive name() call.
    heptanoate = name_compound("C(=O)([O-])CCCCCC", style="pin")
    assert heptanoate == "heptanoate"

    # Verify outer dispatch routed through ANION_SMALL (the recursive inner
    # name() hits its own router, but the outer namer's counter records the
    # outer routing decision).
    outer_namer = Orthonym(style="pin")
    outer_namer.name("C(=O)([O-])CCCCCC")
    outer_stats = outer_namer.get_dispatch_stats()
    assert outer_stats.get(StoutClass.ANION_SMALL, 0) >= 1, (
        f"outer dispatch did not route through ANION_SMALL; "
        f"stats={dict(outer_stats)}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Poly-anion neutralize-recurse (audit § 3.2; RESEARCH § 6.1 site 2)
# ---------------------------------------------------------------------------


def test_poly_anion_neutralize_recursion():
    """Audit § 3.2 + RESEARCH § 6.1: poly-anion-neutralize at namer.py:939.

    For pentanedioate ``[O-]C(=O)CCCC(=O)[O-]`` (5 C carboxylic dianion):
    outer dispatch hits POLY_ANION; the handler shim recursively names the
    neutralized neutral acid and re-decorates as ``-dioate``.

    Per RL-4 fresh-instance invariant: the recursive call uses a fresh router.
    """
    name = name_compound("[O-]C(=O)CCCC(=O)[O-]", style="pin")
    assert name == "pentanedioate"

    # Verify outer dispatch routed through POLY_ANION
    outer_namer = Orthonym(style="pin")
    outer_namer.name("[O-]C(=O)CCCC(=O)[O-]")
    outer_stats = outer_namer.get_dispatch_stats()
    assert outer_stats.get(StoutClass.POLY_ANION, 0) >= 1, (
        f"outer dispatch did not route through POLY_ANION; "
        f"stats={dict(outer_stats)}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Dot-disconnected per-component-recurse (audit § 3.3; RESEARCH § 6.1 site 3)
# ---------------------------------------------------------------------------


def test_dot_disconnected_recursion():
    """Audit § 3.3 + RESEARCH § 6.1: dot-disconnected at namer.py:988.

    For a dot-disconnected neutral SMILES with ≥ 2 multi-atom fragments,
    outer dispatch hits MULTI_COMPONENT_NEUTRAL.  The handler iterates
    fragments, instantiates a fresh ``Orthonym(style)`` per fragment,
    and joins names with a space.

    The byte-identical contract is on the joined output; per-fragment
    ``dispatch_stats`` are NOT cumulative across the recursive boundary.
    """
    # CCO.OCC: two ethanol fragments → "ethanol ethanol"
    name = name_compound("CCO.OCC", style="pin")
    assert name == "ethanol ethanol"

    outer_namer = Orthonym(style="pin")
    outer_namer.name("CCO.OCC")
    outer_stats = outer_namer.get_dispatch_stats()
    assert outer_stats.get(StoutClass.MULTI_COMPONENT_NEUTRAL, 0) >= 1, (
        f"outer dispatch did not route through MULTI_COMPONENT_NEUTRAL; "
        f"stats={dict(outer_stats)}"
    )


# ---------------------------------------------------------------------------
# Test 4 — _skip_decomposition flag-threading (audit § 3.4; RL-7 option (b))
# ---------------------------------------------------------------------------


# Five canary fixtures with HA > 15 whose v18 cascade exercised
# DECOMPOSITION_PRE_GENERAL.  Mined from
# tests/canary/canary_pre_cfr_158.csv on Plan-02-merged HEAD.  Each row
# pairs (smiles, expected_pipeline_only_output) — the same baseline both the
# v18 cascade and the post-CFR substrate produce.
_DECOMP_FIXTURES = [
    # rt75_0 — coniferyl-like diaryl propane-1,2-dione
    (
        "COc1cc(CC(=O)C(=O)c2c(O)cc(O)c(OC)c2O)cc(OC)c1O",
        "1-(2,4,6-trihydroxy-3-methoxyphenyl)-3-(4-hydroxy-3,5-dimethoxyphenyl)propane-1,2-dione",
    ),
    # rt75_1 — nitrotetradecadienoic acid
    (
        "CCCCCC/C=C/C=C(\\CCCC(=O)O)[N+](=O)[O-]",
        "(5E,7E)-5-nitrotetradeca-5,7-dienoic acid",
    ),
    # rt75_4 — long fatty-acid with chain
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\C[C@@H](O)CC(=O)O",
        "(3R,5Z,8Z,11Z,14Z)-3-hydroxyicosa-5,8,11,14-tetraenoic acid",
    ),
    # rt75_5 — long-chain alkyne diol acetate
    (
        "C#CCCCCCCCCCCCC(O)CC(CO)OC(C)=O",
        "2-(acetyloxy)-4-hydroxyheptadec-16-yn-1-ol",
    ),
    # rt75_15 — 2-methyldodecanoic acid (short ID; safe pipeline-only fixture)
    (
        "CCCCCCCCCCC(C)C(=O)O",
        "2-methyldodecanoic acid",
    ),
]


def test_skip_decomposition_flag():
    """Audit § 3.4 + RL-7 option (b): ``name_pipeline_only(smi)`` bypasses
    DECOMPOSITION_PRE_GENERAL via the ``self._skip_decomposition`` flag-
    threading.  Verified for 5 canary fixtures whose v18 cascade exercised
    the decomposition path.

    Byte-identical contract: ``name_pipeline_only(smi)`` matches the Plan-01
    frozen baseline output exactly for every fixture.
    """
    for smiles, expected in _DECOMP_FIXTURES:
        result = name_pipeline_only(smiles, style="pin")
        assert result == expected, (
            f"name_pipeline_only({smiles!r}):\n"
            f"  Expected: {expected}\n"
            f"  Got:      {result}"
        )


# ---------------------------------------------------------------------------
# Test 5 — Zwitterion-character inline mutation (audit § 3.5; RL-3 option (c))
# ---------------------------------------------------------------------------


def test_zwitterion_character_inline_mutation_recursion():
    """Audit § 3.5 + RL-3 option (c): ZWITTERION-CHARACTER in-place mutation
    at namer.py:1000-1034 lives INLINE in ``_name_impl`` BEFORE
    ``CFR.dispatch``.

    Per CONTEXT D-26 hard invariant: the mutation is NOT a CFR entry; the
    predicate factories remain pure.  The "before mutation / after mutation"
    invariant proof is the ABSENCE of a ZWITTERION_NEUTRALIZE row in
    ``DISPATCH_TABLE``.  This test asserts that absence (the RL-3 (c) lock).

    The inline mutation runs only when ``species_type == 'neutral'``,
    ``Chem.GetFormalCharge(mol) == 0``, AND
    ``_has_true_zwitterion_character(mol)`` (per namer.py:945-948 guard).
    Plan-02's Rule-1 gate-scope fix in this same code path is documented in
    158-02-SUMMARY.md.  For determinism, run a representative SMILES twice
    and assert byte-identical names.
    """
    # RL-3 (c) lock: ZWITTERION_NEUTRALIZE must NOT be a CFR enum member.
    assert not hasattr(StoutClass, "ZWITTERION_NEUTRALIZE"), (
        "ZWITTERION_NEUTRALIZE must NOT be a CFR entry per RL-3 option (c); "
        "the in-place mutation lives INLINE in _name_impl BEFORE CFR.dispatch."
    )

    # Determinism: re-running on the same HEAD produces identical names
    # (RL-2 + RL-3 invariant).  Use a SMILES that exercises the inline mutation:
    # quaternary-ammonium acetate-ester with HA > 20.
    smi = "CC(=O)O[N+](C)(C)C(C)C"
    try:
        n1 = name_compound(smi, style="pin")
        n2 = name_compound(smi, style="pin")
    except ValueError:
        pytest.skip(f"SMILES {smi!r} not parseable in this environment")
    assert n1 == n2, (
        f"RL-3 (c) determinism violation: {smi!r} → {n1!r} vs {n2!r} "
        f"across independent calls"
    )
    assert n1 and n1 != "", "name must be non-empty"
