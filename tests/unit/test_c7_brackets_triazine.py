"""
Tests for C7 gap-fix cluster: enclosing marks + 1,2,4-triazine naming.

BB rules:
    — enclosing marks cycle; a substituent already containing
              parentheses must be enclosed in square brackets when it is not
              fully enclosed (e.g. (pyrimidin-5-yl)methyl needs ).
    — compound substituent prefixes must be parenthesized.
   — 1,2,4-triazine PIN for 6-membered ring N@1,2,4.
   — numbering direction picks lowest heteroatom locant set.
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Part (a) — enclosing marks for (aryl)methyl N-substituents
# ---------------------------------------------------------------------------

class TestEnclosingMarks:
    """: (arylmethyl) substituents must be wrapped in square brackets."""

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
        """1-benzylpiperidine — the unsubstituted C6H5-CH2- is the retained prefix
        'benzyl'. (The method keeps its historical name so the node id is stable: it
        used to pin '1-(phenylmethyl)piperidine', the single-token compound prefix in
        parentheses.) "The following retained names are used as preferred
        prefixes for which no substitution is recommended" (under PREFIXES DERIVED
        FROM PARENT HYDRIDES, the Blue Book): 'C6H5-CH2- benzyl (preferred prefix)
        phenylmethyl' (:24414). Both names read back to the input's full InChIKey with
        OPSIN 2.9.0 (an InChIKey). The enclosing-mark guard this row was
        written for (a single-token compound prefix with no inner parentheses stays in
        parentheses) is kept by the next test, on a prefix that has no retained name."""
        assert name_compound("C1CCCCN1Cc1ccccc1") == "1-benzylpiperidine"

    @pytest.mark.unit
    def test_cyclohexylmethyl_piperidine_single_token_stays_parenthesised(self):
        """1-(cyclohexylmethyl)piperidine — single-token compound prefix, no inner
        parens -> stays . The Blue Book writes this prefix in parentheses too:
        '1-[(cyclohexylmethoxy)methyl]-4-{[4-(cyclohexylmethyl)cyclohexyl]methyl}
        cyclohexane (PIN)', NONALPHANUMERICAL ORDER, the Blue Book. OPSIN 2.9.0 reads the name back to the input's
        full InChIKey (an InChIKey)."""
        assert name_compound("C1CCCCN1CC1CCCCC1") == "1-(cyclohexylmethyl)piperidine"

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
    """ /: triazine PIN names and locants."""

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

        Note: 1,2,4-triazine (3 N) is senior to morpholine (N+O) per (f)
        heteroatom count, so triazine is the parent ring. Both names are OPSIN-verified;
        parent-ring seniority rules pick triazine.
        """
        assert name_compound("c1nncc(CN2CCOCC2)n1") == "5-[(morpholin-4-yl)methyl]-1,2,4-triazine"
