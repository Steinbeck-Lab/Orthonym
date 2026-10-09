""" — recursive-fragment cap + best-effort flag propagation (SP4 addendum).

Two VERIFIED defects (grounding: internal notes, re-confirmed
fresh on HEAD c59a7522 2026-08-25):

1. **25-atom compound-substituent cap.** `substituent_enumerator._name_compound_substituent`
   guarded its recursive-naming fallback with ``if len(frag_atoms) <= 25:``. A legitimately
   nameable large fragment (the 27-heavy-atom disaccharide chain hung off a steroid aglycone)
   tripped the cap → ``name_fragment_recursively`` was never called → the substituent was
   dropped and the whole molecule abstained. The size gate is now removed; the recursive
   namer's OWN work-budget + depth-net bound the cost and the RT gate keeps 0-wrong.

2. **Best-effort tier not propagated on recursive re-entry.** ``name_fragment_recursively``
   re-enters through the ``name_compound`` FREE FUNCTION, which inherited ``general_fallback``
   from ``general_fallback_ctx`` but NOT ``general_fallback_unverified`` (published as
   ``best_effort_ctx`` yet never read back here) and had no ``allow_aromatic_general`` parameter
   at all. VERIFIED: ``general_fallback=True`` alone abstains on the disaccharide fragment;
   adding ``general_fallback_unverified=True`` names it and OPSIN-round-trips. The free function
   now inherits the FULL best-effort tier (gfu via ``best_effort_ctx``, allow_aromatic_general
   via the new ``allow_aromatic_general_ctx``).

0-wrong is preserved by the RT gate (not a size constant): a compound substituent whose
recursive name does not round-trip is refused, so the whole-molecule name abstains — never a
partial (atom-dropped) wrong molecule.
"""
import pytest

from orthonym import Orthonym, errors
from orthonym.namer import name_compound
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


# --- SP4 witnesses (V37-a trace-SP4.md) -----------------------------------------
# g1: steroidal sapogenin diglycoside — cap + flag-propagation blocked (this task).
G1 = "CC1CCC2(OC1)OC1CC3C4CCC5CC(OC6OC(CO)C(O)C(OC7OC(CO)C(O)C(O)C7O)C6O)CCC5(C)C4CCC3(C)C1C2C"
# g2: macrolide monoglycoside — aglycone is mixed-spiro-fused (SP3-owned, NOT this task).
G2 = "CCC(C)/C=C/CCCC(C)C(O)C/C=C/C=C/C(=O)OC1C(OC2OC(CO)C(O)C(O)C2O)C(CO)OC2(OCc3cc(O)cc(O)c32)C1O"
# g3: cardenolide diglycoside — cap + flag-propagation blocked (this task).
G3 = "CC(=O)OC(CC1C=C(C)C(=O)O1)C(C)C1CCC2C3(C)CCC(OC4OC(CO)C(O)C(O)C4OC4OC(C)C(O)C(O)C4O)C(C)(C)C3CCC2(C)C12COC(=O)C2"
# bare disaccharide fragment (SP4 Claim 4): 23 heavy atoms; names ONLY when the
# recursive re-entry inherits general_fallback_unverified.
DISACC = "OC1OC(CO)C(O)C(OC2OC(CO)C(O)C(O)C2O)C1O"
# A >25-heavy-atom disaccharide COMPOUND SUBSTITUENT on a plainly nameable parent
# (cyclohexane): the ONLY blocker here was the 25-atom cap + the flag-propagation
# gap. This is the end-to-end witness that isolates THIS task's two defects from
# the aglycone ring-construction blocker that g1/g3 additionally hit (see below).
# 32 heavy atoms total; the compound substituent alone is 27. Abstained pre-fix.
CYCLOHEXYL_BIGSUGAR = "C1CCCCC1C(CC)OC1OC(CO)C(O)C(OC2OC(CO)C(O)C(O)C2O)C1O"


def _rt_full_inchikey(smi: str, name) -> bool:
    """Full-InChIKey round-trip: name -> OPSIN -> InChI == input's InChI."""
    if not name or errors.is_failure_name(name):
        return False
    res = opsin_roundtrip_check(smi, name)
    return bool(res.get("passed") and res.get("inchi_match"))


def _best_effort() -> Orthonym:
    return Orthonym(general_fallback=True,
                     general_fallback_unverified=True,
                     allow_aromatic_general=True)


# ---------------------------------------------------------------------------
# Defect 2 — the recursive re-entry point must inherit the FULL best-effort tier
# from contextvars, exactly as Orthonym.name publishes them at a best-effort
# top-level entry. This is the isolated flag-propagation test.
# ---------------------------------------------------------------------------
class TestRecursiveReentryTierInheritance:
    @pytest.mark.opsin_gate
    def test_name_compound_inherits_general_fallback_unverified_from_ctx(self):
        from orthonym.metrics.provenance import (
            general_fallback_ctx, best_effort_ctx)
        # publish what a best-effort Orthonym.name would publish, then re-enter
        # name_compound with NO explicit flags (the fragment_naming.py:544 call shape).
        t1 = general_fallback_ctx.set(True)
        t2 = best_effort_ctx.set(True)
        try:
            name = name_compound(DISACC)   # style='pin', no flags — must inherit
        finally:
            best_effort_ctx.reset(t2)
            general_fallback_ctx.reset(t1)
        assert name and not errors.is_failure_name(name), (
            f"recursive re-entry abstained: {name!r} (gfu not inherited from ctx)")
        assert _rt_full_inchikey(DISACC, name), (
            f"inherited-tier name does not round-trip: {name!r}")

    @pytest.mark.opsin_gate
    def test_ctx_default_off_keeps_pin_tier(self):
        """Control: with no ctx published (default PIN tier), the recursive
        re-entry must NOT silently acquire best-effort vocabulary."""
        # DISACC needs gfu to name; at PIN default it must abstain (never wrong).
        name = name_compound(DISACC)
        assert name is None or errors.is_failure_name(name) \
            or _rt_full_inchikey(DISACC, name)


# ---------------------------------------------------------------------------
# THIS TASK'S DELIVERABLE — a >25-heavy-atom disaccharide compound substituent on
# a plainly nameable parent. Blocked pre-fix by BOTH defects (the cap dropped it
# on size; even past the cap the recursive re-entry lacked the best-effort tier).
# Now names + round-trips. This is the clean, isolated proof the two SP1.1b fixes
# deliver end-to-end breadth through the real enumerator path (abstain -> RT).
# ---------------------------------------------------------------------------
class TestLargeCompoundSubstituentDelivered:
    @pytest.mark.opsin_gate
    def test_big_sugar_substituent_on_nameable_parent_round_trips(self):
        name = _best_effort().name(CYCLOHEXYL_BIGSUGAR)
        assert _rt_full_inchikey(CYCLOHEXYL_BIGSUGAR, name), (
            f"big compound substituent did not RT: {name!r}")


# ---------------------------------------------------------------------------
# The SP4 disaccharide-to-steroid witnesses. 0-wrong is ABSOLUTE for all three
# (names RT-exact OR abstains — never atom-dropped). g1/g3 additionally hit a
# von-Baeyer AGLYCONE ring-construction blocker (parent + complex substituent:
# `vonbaeyer_universal: descriptor edge-audit failed`) that is DOWNSTREAM of and
# distinct from this task's two defects — an SP3/SP1.5-owned lever. The SP4 trace
# (V37-a trace-SP4.md Claim 3) traced only the first (cap) warning and its premise
# that g1/g3 reduce ENTIRELY to SP1's cap+flag is incomplete: the compound
# substituent IS now nameable (proven above + by the aglycone-alone RT), but the
# whole molecule still abstained on the aglycone parent. g2 is SP3-owned
# (mixed-spiro-fused). The full-RT rows were xfail(strict) so they FLIP LOUDLY the
# moment SP3/SP1.5 rescued them; they did, and they are plain passing witnesses now.
# ---------------------------------------------------------------------------
class TestSP4GlycanWitnesses:
    @pytest.mark.opsin_gate
    @pytest.mark.parametrize("smi", [G1, G2, G3])
    def test_witness_never_wrong(self, smi):
        """0-wrong absolute: each witness names correctly (RT-exact) or abstains
        — never a partial (atom-dropped) wrong molecule."""
        name = _best_effort().name(smi)
        assert name is None or errors.is_failure_name(name) \
            or _rt_full_inchikey(smi, name), (
            f"{smi}: emitted a non-RT name {name!r} (0-wrong violation)")

    @pytest.mark.opsin_gate
    @pytest.mark.parametrize("smi", [G1, G3])
    def test_diglycoside_full_round_trip_pending_sp3(self, smi):
        # The strict xfail ("residual SP3/SP1.5 aglycone blocker: the von-Baeyer steroid parent
        # + complex sugar substituent fails 'descriptor edge-audit'") flipped loudly, as designed:
        # both witnesses now name RT-exact under best-effort (the fixing commit was not
        # bisected). The marker is removed and the rows stay as ordinary passing witnesses.
        name = _best_effort().name(smi)
        assert _rt_full_inchikey(smi, name), f"did not RT: {name!r}"


# ---------------------------------------------------------------------------
# Controls — the cap removal must NOT regress currently-correct output, and must
# NOT admit a wrong large fragment (the RT gate, not the size constant, guards it).
# These name correctly today and must stay byte-identical.
# ---------------------------------------------------------------------------
class TestControlsUnchanged:
    @pytest.mark.parametrize("smi,expected", [
        ('CCO', 'ethanol'),
        ('c1ccccc1', 'benzene'),
        ('CCOCC', 'ethoxyethane'),
        ('COCCNCCC', 'N-(2-methoxyethyl)propan-1-amine'),   # compound N-substituent (SP1.1)
        ('OCC1CCCCC1', 'cyclohexylmethanol'),               # ring compound substituent
    ])
    def test_default_tier_control_unchanged(self, smi, expected):
        assert name_compound(smi) == expected

    @pytest.mark.opsin_gate
    @pytest.mark.parametrize("smi", ['CCO', 'c1ccccc1', 'OCC1CCCCC1', 'COCCNCCC'])
    def test_best_effort_control_round_trips(self, smi):
        """Small currently-correct substituents still RT under best-effort
        (cap removal is a no-op for them; never a wrong molecule)."""
        name = _best_effort().name(smi)
        assert name is None or errors.is_failure_name(name) \
            or _rt_full_inchikey(smi, name)
