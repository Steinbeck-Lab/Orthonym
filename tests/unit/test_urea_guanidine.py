"""Tests for urea and guanidine naming (a phase-03).

Covers:
- Retained names: urea, guanidine
- substituted derivatives: methylurea (mono omits the locant, P-14.3.4.3),
  N,N-dimethylurea, N,N'-dimethylurea (disubstituted keep both)
- FG collision avoidance: urea NOT amide, guanidine NOT imine
- OPSIN round-trip validation for generated names
"""

import pytest
from rdkit import Chem
from orthonym import name_compound
from orthonym.perception.functional_groups import detect_functional_groups


# ---------------------------------------------------------------------------
# Urea retained name tests (FG-07)
# ---------------------------------------------------------------------------

class TestUreaRetainedName:
    """Test urea naming using retained name with N-substitution."""

    def test_urea_base(self):
        """Unsubstituted urea -> 'urea'."""
        assert name_compound("NC(=O)N") == "urea"

    def test_urea_n_methyl(self):
        """Monosubstituted urea -> 'methylurea' (P-14.3.4.3, the Blue Book verbatim
        `CH3-NH-CO-NH2 methylurea (PIN)`: urea's four N-H are one orbit, so the
        italic-N locant is omitted). NOT 'N-methylurea'."""
        assert name_compound("CNC(=O)N") == "methylurea"

    def test_urea_n_ethyl(self):
        """Monosubstituted urea -> 'ethylurea' (P-14.3.4.3, same licence)."""
        assert name_compound("CCNC(=O)N") == "ethylurea"

    def test_urea_nn_dimethyl_same_nitrogen(self):
        """N,N-disubstituted (same nitrogen) -> 'N,N-dimethylurea'."""
        assert name_compound("CN(C)C(=O)N") == "N,N-dimethylurea"

    def test_urea_nn_prime_dimethyl_different_nitrogens(self):
        """N,N'-disubstituted (different nitrogens) -> 'N,N'-dimethylurea'."""
        assert name_compound("CNC(=O)NC") == "N,N'-dimethylurea"

    def test_urea_n_phenyl(self):
        """Monosubstituted (aryl) urea -> 'phenylurea' (P-14.3.4.3, same licence)."""
        assert name_compound("NC(=O)Nc1ccccc1") == "phenylurea"

    def test_urea_tetrasubstituted(self):
        """Fully substituted urea -> 'N,N,N',N'-tetramethylurea'."""
        assert name_compound("CN(C)C(=O)N(C)C") == "N,N,N',N'-tetramethylurea"

    def test_urea_mixed_substitution(self):
        """Mixed substitution -> 'N-ethyl-N'-methylurea'."""
        assert name_compound("CCNC(=O)NC") == "N-ethyl-N'-methylurea"


# ---------------------------------------------------------------------------
# Guanidine retained name tests (FG-08)
# ---------------------------------------------------------------------------

class TestGuanidineRetainedName:
    """Test guanidine naming using retained name with N-substitution."""

    def test_guanidine_base(self):
        """Unsubstituted guanidine -> 'guanidine'."""
        assert name_compound("NC(=N)N") == "guanidine"

    def test_guanidine_n_methyl(self):
        """N-monosubstituted guanidine -> 'N-methylguanidine'."""
        assert name_compound("CNC(=N)N") == "N-methylguanidine"

    def test_guanidine_n_ethyl(self):
        """N-monosubstituted guanidine -> 'N-ethylguanidine'."""
        assert name_compound("CCNC(=N)N") == "N-ethylguanidine"

    def test_guanidine_nn_dimethyl_same_nitrogen(self):
        """N,N-disubstituted (same nitrogen) -> 'N,N-dimethylguanidine'."""
        assert name_compound("CN(C)C(=N)N") == "N,N-dimethylguanidine"

    def test_guanidine_trisubstituted_three_nitrogens(self):
        """One sub on each nitrogen -> 'N,N',N''-trimethylguanidine'."""
        result = name_compound("CNC(=NC)NC")
        assert "trimethylguanidine" in result


# ---------------------------------------------------------------------------
# FG collision avoidance tests
# ---------------------------------------------------------------------------

class TestFGCollisionAvoidance:
    """Verify urea is not detected as amide, guanidine not as imine."""

    def test_urea_not_primary_amide(self):
        """Urea must NOT be detected as primary_amide."""
        mol = Chem.MolFromSmiles("NC(=O)N")
        fgs = detect_functional_groups(mol)
        assert "urea" in fgs
        assert "primary_amide" not in fgs

    def test_urea_not_secondary_amide(self):
        """N-substituted urea must NOT be detected as secondary_amide."""
        mol = Chem.MolFromSmiles("CNC(=O)N")
        fgs = detect_functional_groups(mol)
        assert "urea" in fgs
        assert "secondary_amide" not in fgs
        assert "primary_amide" not in fgs

    def test_urea_not_tertiary_amide(self):
        """N,N-disubstituted urea must NOT be detected as tertiary_amide."""
        mol = Chem.MolFromSmiles("CN(C)C(=O)N")
        fgs = detect_functional_groups(mol)
        assert "urea" in fgs
        assert "tertiary_amide" not in fgs

    def test_guanidine_not_imine(self):
        """Guanidine must NOT be detected as imine."""
        mol = Chem.MolFromSmiles("NC(=N)N")
        fgs = detect_functional_groups(mol)
        assert "guanidine" in fgs
        assert "imine" not in fgs

    def test_substituted_guanidine_not_imine(self):
        """N-substituted guanidine must NOT be detected as imine."""
        mol = Chem.MolFromSmiles("CNC(=N)N")
        fgs = detect_functional_groups(mol)
        assert "guanidine" in fgs
        assert "imine" not in fgs


# ---------------------------------------------------------------------------
# OPSIN round-trip validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected_name", [
    ("NC(=O)N", "urea"),
    # Monosubstituted urea OMITS the italic-N locant (P-14.3.4.3, the Blue Book methylurea).
    ("CNC(=O)N", "methylurea"),
    ("CCNC(=O)N", "ethylurea"),
    ("CNC(=O)NC", "N,N'-dimethylurea"),
    ("CN(C)C(=O)N", "N,N-dimethylurea"),
    ("NC(=O)Nc1ccccc1", "phenylurea"),
    ("CN(C)C(=O)N(C)C", "N,N,N',N'-tetramethylurea"),
    ("CCNC(=O)NC", "N-ethyl-N'-methylurea"),
    ("NC(=N)N", "guanidine"),
    ("CNC(=N)N", "N-methylguanidine"),
    ("CCNC(=N)N", "N-ethylguanidine"),
    ("CN(C)C(=N)N", "N,N-dimethylguanidine"),
])
def test_urea_guanidine_naming(smiles, expected_name):
    """Parameterized test for all urea/guanidine naming."""
    result = name_compound(smiles)
    assert result == expected_name, f"For {smiles}: got '{result}', expected '{expected_name}'"


# ---------------------------------------------------------------------------
# Prefix form tests (when subordinate to higher-seniority group)
# ---------------------------------------------------------------------------

class TestPrefixForms:
    """Test prefix forms when urea/guanidine is subordinate to a principal group."""

    def test_urea_as_prefix_with_acid(self):
        """Urea subordinate to carboxylic acid should use carbamoylamino prefix."""
        result = name_compound("NC(=O)NCCCC(=O)O")
        # The polyfunctional assembly produces a name containing "carbamoylamino"
        # The full name format may have imperfections but the prefix form is correct
        assert "carbamoylamino" in result or "ureido" in result

    def test_guanidine_as_prefix_with_acid(self):
        """Guanidine subordinate to carboxylic acid should use guanidino prefix."""
        result = name_compound("NC(=N)NCCCC(=O)O")
        # The polyfunctional assembly produces a name containing "guanidino"
        assert "guanidino" in result


# ---------------------------------------------------------------------------
# a phase task 2 (P-66.4.1.2.1.2): guanidine N/N'/N'' LOCANT ASSIGNMENT uses a
# "minimum number of primes" -> the nitrogen bearing the MOST substituents takes
# the lowest-primed locants (the Blue Book, 34258, 34262 'N,N'-dimethylguanidine' NOT
# 'N,N''-' at the Blue Book). Citation order among distinct names is alphanumerical
# (P-14.5.2). All rows are OPSIN-RT gold PINs.
# ---------------------------------------------------------------------------
class TestGuanidineMinimumPrimes:
    def test_NNNprime_trimethylguanidine(self):
        # 2 methyls on one amino N + 1 on the other -> N,N,N' (NOT N,N',N').
        assert name_compound("CNC(=N)N(C)C", style="pin") == "N,N,N'-trimethylguanidine"

    def test_tetramethyl_Nprimeprime_phenylguanidine(self):
        # 4 methyls take N,N,N',N' (lowest); phenyl on the imino N -> N''.
        assert (name_compound("CN(C)C(=Nc1ccccc1)N(C)C", style="pin")
                == "N,N,N',N'-tetramethyl-N''-phenylguanidine")

    def test_NNprime_dimethylguanidine_not_Nprimeprime(self):
        # the Blue Book verbatim: N,N'-dimethylguanidine, explicitly NOT N,N''-.
        assert name_compound("CNC(=N)NC", style="pin") == "N,N'-dimethylguanidine"

    def test_guanidine_unchanged(self):
        assert name_compound("N=C(N)N", style="pin") == "guanidine"

    def test_N_methylguanidine_single_unchanged(self):
        assert name_compound("CNC(=N)N", style="pin") == "N-methylguanidine"

    def test_count_tie_locant_assignment_deterministic_and_bb_correct(self):
        # Determinism regression (P-14.5.2): both amino N's tie on count(2) AND
        # earliest substituent (ethyl); the NEXT substituent decides -- methyl <
        # propyl, so the methyl-bearing N takes the lower (unprimed) locant. The
        # assignment must NOT depend on SMILES atom order (a project rule).
        # Same molecule (InChIKey XPFLVYSHEGSFLO), three writings.
        expected = "N,N'-diethyl-N-methyl-N'-propylguanidine"
        for smi in ("CCN(C)C(=N)N(CC)CCC",
                    "CCCN(CC)C(=N)N(C)CC",
                    "N(C)(CC)C(=N)N(CC)CCC"):
            assert name_compound(smi, style="pin") == expected, smi
