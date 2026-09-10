"""a phase Fix 1 — ring stereo emission for SATURATED bare steroid scaffolds.

Root cause (verified in the brief, `.superpowers/sdd/2026-08-12-phase0b-descriptor-kind-
degradation/fix-1-brief.md`): `name_natural_product`'s Step 4 bare-scaffold branch only
calls the alpha/beta stereo collector `_collect_np_stereo` inside the UNSATURATED
(`ene`/`yne`) arm. A fully-saturated bare scaffold with a non-natural ring stereocentre
(e.g. 5-beta rather than the reference 5-alpha) falls straight to
`return scaffold_info["scaffold_name"]` -- a stereo-free bare name -- which a phase's
SELF-01 gate then abstains as a stereo omission (the emitted name OPSIN-round-trips to a
DIFFERENT, epimeric molecule).

This test file locks:
  1. RED->GREEN: the witness sterane (5-beta ring stereocentre) now emits
     '5β-pregnane' from the direct producer, and the full pipeline (OPSIN gate ON)
     ships a stereo-carrying, non-failure name instead of abstaining.
  2. Control (no regression): a bare steroid whose ring stereocentres ALL match the
     natural reference configuration (`_collect_np_stereo` -> ('', {}), i.e. nothing to
     emit) must keep shipping its existing bare name, unchanged, through the new `else`
     branch's fall-through.
"""

import pytest

from orthonym import name_compound
from orthonym.errors import is_failure_name
from orthonym.rules.natural_products import name_natural_product

# The a phase witness: a fully-saturated pregnane skeleton with the C-5 ring-fusion
# stereocentre inverted relative to the natural (5-alpha) reference configuration in
# data/natural_products.py, so it is legitimately named '5β-pregnane' rather than
# the bare, stereo-free 'pregnane'.
WITNESS_5BETA_PREGNANE = (
    "CC[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC[C@H]4[C@@]3(CCCC4)C)C"
)

# Control: the exact canonical scaffold-table reference SMILES for androstane
# (data/natural_products.py) -- every ring stereocentre matches the natural
# configuration, so `_collect_np_stereo` must return ('', {}) and the bare name must
# be emitted completely unchanged by the new saturated-branch code.
CONTROL_ANDROSTANE = "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2"

# a review BLOCKER witness (2026-08-12): a pregnane skeleton that leaves a ring centre
# UNDEFINED. Any 'X-pregnane' name over-specifies (the '-pregnane' stem itself asserts
# 6 of 7 ring centres), and SELF-01 TOLERATES over-specification (namer.py:992), so it
# cannot catch a wrong over-specified stereoisomer. Pre-fix this shipped the wrong
# '(5S,8S,9S,10S,13R,14S)-pregnane'; the RT-gate on the whole-graph fallback makes it
# abstain instead. NEVER ship the wrong name.
BLOCKER_PARTIAL_STEREO = "CCC1CC[C@H]2[C@@H]3CC[C@@H]4CCCC[C@]4(C)[C@H]3CC[C@]12C"
WRONG_OVERSPEC_NAME = "(5S,8S,9S,10S,13R,14S)-pregnane"


class TestSaturatedBareScaffoldStereoEmission:
    def test_witness_sterane_direct_producer_emits_5β(self):
        """Direct `name_natural_product` unit assertion (brief's cheaper form)."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles(WITNESS_5BETA_PREGNANE)
        assert mol is not None
        name = name_natural_product(mol)
        assert name == "5β-pregnane", name

    @pytest.mark.opsin_gate
    def test_witness_sterane_full_pipeline_names_with_stereo(self):
        """End-to-end (OPSIN validity gate ON): must ship a stereo-carrying name,
        not abstain to a failure signal ('unknown organic compound' etc.)."""
        name = name_compound(WITNESS_5BETA_PREGNANE)
        assert not is_failure_name(name), name
        assert ("α" in name or "β" in name
                or "R)" in name or "S)" in name
                or "R," in name or "S," in name), name

    def test_control_androstane_bare_name_unchanged(self):
        """No-regression control: a bare steroid whose ring centres all match the
        natural reference configuration must keep emitting its plain bare name --
        `_collect_np_stereo` returns ('', {}) for it, so the new `else` branch's
        `if ring_ab:` guard is False and control falls through to the pre-existing
        `return scaffold_info["scaffold_name"]` path, unchanged."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles(CONTROL_ANDROSTANE)
        assert mol is not None
        name = name_natural_product(mol)
        assert name == "androstane", name

    @pytest.mark.opsin_gate
    def test_control_androstane_full_pipeline_unchanged(self):
        """Same control through the full pipeline with the OPSIN gate ON: verified
        pre-fix to emit the clean bare name with no SELF-01 suppression; must stay
        byte-identical after the fix."""
        name = name_compound(CONTROL_ANDROSTANE)
        assert name == "androstane", name

    @pytest.mark.opsin_gate
    def test_partial_stereo_sterane_never_ships_wrong_overspecified_name(self):
        """a review BLOCKER (2026-08-12): the whole-graph-R/S fallback must NOT ship an
        over-specified name for an input that leaves a ring centre undefined. SELF-01
        tolerates over-specification (namer.py:992), so it cannot catch it -- the
        RT-gate on the fallback (ship only if it OPSIN-round-trips) makes this abstain
        rather than ship the wrong '(5S,8S,9S,10S,13R,14S)-pregnane'. If a future fix
        legitimately recovers this input RT-exact, update this assertion accordingly
        (it must still never equal the WRONG_OVERSPEC_NAME)."""
        name = name_compound(BLOCKER_PARTIAL_STEREO)
        assert name != WRONG_OVERSPEC_NAME, name
        assert is_failure_name(name), name
