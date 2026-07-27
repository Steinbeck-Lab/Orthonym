"""
Tests for C7 gap-fix cluster: enclosing marks + 1,2,4-triazine naming.

BB rules:
  P-16.5.1.1  — enclosing marks cycle; a substituent already containing
              parentheses must be enclosed in square brackets when it is not
              fully enclosed (e.g. (pyrimidin-5-yl)methyl needs []).
  P-16.5.1.1  — compound substituent prefixes must be parenthesized.
  P-31.1.4.2 — 1,2,4-triazine PIN for 6-membered ring N@1,2,4.
  P-31.1.4.3.3 — numbering direction picks lowest heteroatom locant set.
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Part (a) — enclosing marks for (aryl)methyl N-substituents
# ---------------------------------------------------------------------------

class TestEnclosingMarks:
    """P-16.5.1.1: (arylmethyl) substituents must be wrapped in square brackets."""

    @pytest.mark.unit
    def test_pyrimidinyl_methyl_morpholine(self):
        """4-[(pyrimidin-5-yl)methyl]morpholine — the core 1.1 residual case."""
        assert name_compound("C1COCCN1Cc1cncnc1") == "4-[(pyrimidin-5-yl)methyl]morpholine"

    @pytest.mark.unit
    def test_pyridinyl_methyl_morpholine(self):
        """4-[(pyridin-4-yl)methyl]morpholine."""
        assert name_compound("C1COCCN1Cc1ccncc1") == "4-[(pyridin-4-yl)methyl]morpholine"

    @pytest.mark.unit
    def test_pyrazinyl_methyl_morpholine(self):
        """4-[(pyrazin-2-yl)methyl]morpholine."""
        assert name_compound("C1COCCN1Cc1cnccn1") == "4-[(pyrazin-2-yl)methyl]morpholine"

    @pytest.mark.unit
    def test_naphthyl_methyl_morpholine(self):
        """4-[(naphthalen-2-yl)methyl]morpholine (adversarial — naphthyl)."""
        assert name_compound("C1COCCN1Cc1ccc2ccccc2c1") == "4-[(naphthalen-2-yl)methyl]morpholine"

    @pytest.mark.unit
    def test_pyridinyl_methyl_piperidine(self):
        """4-[(piperidin-1-yl)methyl]pyridine — pyridine is the senior ring (P-44.4.1 aromatic > sat)."""
        assert name_compound("C1CCCCN1Cc1ccncc1") == "4-[(piperidin-1-yl)methyl]pyridine"

    # --- must-NOT-change cases (regression guards) ---

    @pytest.mark.unit
    def test_phenylmethyl_piperidine_no_extra_brackets(self):
        """1-(phenylmethyl)piperidine — single-token, no inner parens -> stays ()."""
        assert name_compound("C1CCCCN1Cc1ccccc1") == "1-(phenylmethyl)piperidine"

    @pytest.mark.unit
    def test_ethyl_pyrimidinyl_morpholine_stays_bracketed(self):
        """4-[2-(pyrimidin-5-yl)ethyl]morpholine — ethyl analogue; already worked."""
        assert name_compound("C1COCCN1CCc1cncnc1") == "4-[2-(pyrimidin-5-yl)ethyl]morpholine"

    @pytest.mark.unit
    def test_direct_ring_attach_stays_parens(self):
        """4-(pyrimidin-5-yl)morpholine — direct ring attachment (fully enclosed)."""
        assert name_compound("C1COCCN1c1cncnc1") == "4-(pyrimidin-5-yl)morpholine"


# ---------------------------------------------------------------------------
# Part (b) — 1,2,4-triazine naming
# ---------------------------------------------------------------------------

class TestTriazine:
    """P-31.1.4.2 / P-31.1.4.3.3: triazine PIN names and locants."""

    @pytest.mark.unit
    def test_124_triazine_bare(self):
        """c1cnncn1 -> 1,2,4-triazine (PIN)."""
        assert name_compound("c1cnncn1") == "1,2,4-triazine"

    @pytest.mark.unit
    def test_135_triazine_bare(self):
        """c1ncncn1 -> 1,3,5-triazine (PIN); must not regress."""
        assert name_compound("c1ncncn1") == "1,3,5-triazine"

    @pytest.mark.unit
    def test_124_triazine_methyl(self):
        """Cc1cnncn1 -> 5-methyl-1,2,4-triazine."""
        assert name_compound("Cc1cnncn1") == "5-methyl-1,2,4-triazine"

    @pytest.mark.unit
    def test_124_triazine_morpholinomethyl(self):
        """c1nncc(CN2CCOCC2)n1 -> 5-[(morpholin-4-yl)methyl]-1,2,4-triazine.

        Note: 1,2,4-triazine (3 N) is senior to morpholine (N+O) per P-44.2.1(f)
        heteroatom count, so triazine is the parent ring. Both names are OPSIN-verified;
        parent-ring seniority rules pick triazine.
        """
        assert name_compound("c1nncc(CN2CCOCC2)n1") == "5-[(morpholin-4-yl)methyl]-1,2,4-triazine"
