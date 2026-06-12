"""Wave-0 unit coverage for Pattern D (D/L + peptide-acyl) stereo suppression.

Phase 177 Plan 01 / WS-B.0 (D-01/D-02/D-03).

`needs_stereo_injection` (src/orthonym/rules/stereochemistry.py) is the SHARED
suppression predicate consumed by the top-level injection seams (the namer
backstop and ``inject_stereo_from_locant_map``).  Phase 177 extends it with
Pattern D: a leading/embedded ``D-``/``L-`` configurational token, OR a peptide
acyl chain (``L-…yl-`` / ``D-…yl-``), is treated as "stereo-already-present"
(return ``False``) so the Plan-02 backstop flip from detect-only to inject does
NOT double-encode the conformant amino-acid / peptide subsystem.

These tests are written BEFORE Pattern D lands (Task 2) — they are RED on
HEAD until the predicate extension ships, then GREEN.  Per the fix-methodology
mandate the suppression is at the predicate source, never a postprocessor.

The two configurational tests assert Pattern D does NOT over-suppress:
  * a phenol with no CIP stereo and no D/L token falls through to the CIP
    probe and is descriptor-free *because the molecule has no stereo* (not
    because Pattern D forced it);
  * a genuinely-injectable stereo chain name WITHOUT a D/L token still returns
    True (Pattern D must not swallow legitimate injection targets).
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym.rules.stereochemistry import needs_stereo_injection


@pytest.mark.unit
class TestPatternDSuppression:
    """D/L configurational token + peptide-acyl recognition (D-01/D-02/D-03)."""

    # --- D/L configurational token suppression ---------------------------

    def test_leading_capital_dl_token_suppressed(self):
        # 'D-alanine' carries a leading D- configurational token -> already
        # stereo-configured -> no injection.
        mol = Chem.MolFromSmiles("C[C@@H](N)C(=O)O")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert needs_stereo_injection(mol, "D-alanine") is False

    def test_leading_lowercase_dl_token_suppressed(self):
        # 'd-glyceraldehyde' (lowercase d-) is the traditional configurational
        # notation -> suppressed.
        mol = Chem.MolFromSmiles("OC[C@@H](O)C=O")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert needs_stereo_injection(mol, "d-glyceraldehyde") is False

    def test_simple_l_amino_acid_suppressed(self):
        # 'L-valine' -> leading L- token -> suppressed.
        mol = Chem.MolFromSmiles("CC(C)[C@@H](N)C(=O)O")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert needs_stereo_injection(mol, "L-valine") is False

    # --- peptide acyl chain suppression ----------------------------------

    def test_peptide_acyl_chain_suppressed(self):
        # 'L-valyl-L-tyrosyl-L-isoleucine' is a peptide acyl chain
        # (L-…yl-) -> every residue already D/L-configured -> suppressed.
        mol = Chem.MolFromSmiles(
            "N[C@@H](C(C)C)C(=O)N[C@@H](CC1=CC=C(C=C1)O)"
            "C(=O)N[C@@H]([C@@H](C)CC)C(=O)O"
        )
        rdCIPLabeler.AssignCIPLabels(mol)
        assert (
            needs_stereo_injection(mol, "L-valyl-L-tyrosyl-L-isoleucine") is False
        )

    # --- no over-suppression (negative controls) -------------------------

    def test_phenol_without_dl_token_not_forced_false_by_pattern_d(self):
        # 2,3-dimethylphenol has NO D/L token and NO CIP stereo: Pattern D must
        # not fire; it falls through to the CIP probe and is descriptor-free
        # *because the molecule has no stereo*.
        mol = Chem.MolFromSmiles("Cc1cccc(O)c1C")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert needs_stereo_injection(mol, "2,3-dimethylphenol") is False

    def test_injectable_stereo_chain_without_dl_token_still_true(self):
        # A genuinely-injectable stereo chain name WITHOUT a D/L token must
        # still return True -- Pattern D must NOT over-suppress real injection
        # targets.  (2R)-butan-2-ol's bare descriptor-free form 'butan-2-ol'
        # carries no Pattern A/B/C/D match but the mol has CIP stereo.
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        rdCIPLabeler.AssignCIPLabels(mol)
        assert needs_stereo_injection(mol, "butan-2-ol") is True
