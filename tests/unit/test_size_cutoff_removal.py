"""Size-cutoff band-aid removal gate — Phase 169.6 Plan 04 (Task 3).

The three size-cutoff band-aids DROP the charge on a large ion / zwitterion and
reclassify it ``neutral`` (perception/ions.py:130-135 >10 HA single charge;
:157-165 >20 HA quat-N zwitterion; dispatch_table.py _is_anion_small <=25 HA).
This silently loses the ionic character (``CCCCCCCCCCCC[NH3+]`` -> ``dodecane``,
the amine is discarded). Now that route_charged names large ions structurally
(169.6-03/04), the cutoffs are removable.

RESEARCH open question 2 / Assumption A2: the removal is a MEASURED step. This
file is the gate:

  1. LARGE NEUTRAL negative-test set — molecules with NO charge whose names MUST
     stay byte-identical after removal (the cutoffs only fire on charged species,
     so a true neutral must be untouched). Asserted as literal expected strings.
  2. CUTOFF CASUALTIES — large ions/zwitterions that were forced ``neutral`` and
     are now named with their charge preserved (route_charged), RT-verified.

If removing a cutoff regressed a neutral, that cutoff would be KEPT with a
documented honest-fail reason (a real out-of-scope boundary, NOT a band-aid).
The measured outcome: NO large-neutral regression (the cutoffs are charged-only),
so all three are removed.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.perception.ions import detect_species_type


# ---------------------------------------------------------------------------
# 1. LARGE NEUTRAL negative-test set (MUST be byte-identical after removal).
#    Each has > 10 HA (and one > 20 HA), no formal charge. The cutoffs never
#    fired on these (they require has_any_charge), so removal cannot change them.
# ---------------------------------------------------------------------------
LARGE_NEUTRALS = {
    # (smiles, expected_name)
    "octadecanoic acid":     ("CCCCCCCCCCCCCCCCCC(=O)O", "octadecanoic acid"),
    "icosan-1-ol":           ("CCCCCCCCCCCCCCCCCCCCO", "icosan-1-ol"),
    "phenanthrene":          ("c1ccc2c(c1)ccc1ccccc12", "phenanthrene"),
    "pristane":              ("CC(C)CCCC(C)CCCC(C)CCCC(C)C", "pristane"),
    "hexadecan-1-amine":     ("CCCCCCCCCCCCCCCCN", "hexadecan-1-amine"),
}


@pytest.mark.unit
class TestLargeNeutralByteIdentical:
    """The negative-test set: large NEUTRAL molecules stay byte-identical (the
    cutoffs are charged-only, so removal must not touch them)."""

    @pytest.mark.parametrize("label", list(LARGE_NEUTRALS.keys()))
    def test_large_neutral_unchanged(self, label):
        smiles, expected = LARGE_NEUTRALS[label]
        # It is a neutral species ...
        assert detect_species_type(Chem.MolFromSmiles(smiles)) == "neutral"
        # ... and its name is the recorded fixture (byte-identical pre/post).
        assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# 2. CUTOFF CASUALTIES — large ions/zwitterions whose charge was DROPPED by the
#    cutoff and is now preserved (charge survives, routed through route_charged).
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCutoffCasualtiesNowCharged:
    """After removal, a >10 HA cation and a >25 HA anion are no longer forced
    neutral — their charge is preserved and they name structurally."""

    def test_large_cation_charge_preserved(self):
        """>10 HA protonated amine: cutoff-1 forced it ``neutral`` -> ``dodecane``
        (charge dropped!). Now it is an ``ion`` -> dodecan-1-aminium (RT=1)."""
        smi = "CCCCCCCCCCCC[NH3+]"
        assert detect_species_type(Chem.MolFromSmiles(smi)) == "ion"
        assert name_compound(smi) == "dodecan-1-aminium"

    def test_large_aryl_cation_charge_preserved(self):
        """>10 HA aryl-alkyl ammonium: was 'unknown organic compound'; now
        6-phenylhexan-1-aminium (RT=1)."""
        smi = "c1ccccc1CCCCCC[NH3+]"
        assert detect_species_type(Chem.MolFromSmiles(smi)) == "ion"
        assert name_compound(smi) == "6-phenylhexan-1-aminium"

    def test_large_anion_above_25ha_routes(self):
        """>25 HA single anion: cutoff-3 (_is_anion_small <=25 HA) excluded it
        from the anion handler. After removal it routes through route_charged
        and keeps its charge (a long-chain carboxylate -> ...oate)."""
        # Triacontanoate: 30-carbon carboxylate (30 HA O2 = 32 HA) > 25.
        smi = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCC C(=O)[O-]".replace(" ", "")
        st = detect_species_type(Chem.MolFromSmiles(smi))
        assert st == "ion"
        name = name_compound(smi)
        # The charge is preserved (an -oate anion, not the neutral acid).
        assert name.endswith("oate"), name
