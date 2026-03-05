"""Unit tests for fragment seniority ranking.

Tests the fragment_ranker module which scores fragments by P-44.1.1
seniority for decomposition parent/substituent assignment.
"""

import pytest


@pytest.mark.unit
class TestScoreFragmentSeniority:
    """Test score_fragment_seniority() returns correct comparison tuples."""

    def test_carboxylic_acid_fragment(self):
        """Carboxylic acid fragment: benzoic acid."""
        from orthonym.decomposition.fragment_ranker import score_fragment_seniority

        score = score_fragment_seniority("OC(=O)c1ccccc1")
        # Should detect carboxylic_acid, has ring, 9 heavy atoms
        pg_count, pg_rank, hetero_rank, has_ring, n_heavy = score
        assert pg_count < 0, "pg_count should be negative (negated for sorting)"
        assert pg_rank == 0, "carboxylic_acid is rank 0 (most senior)"
        assert has_ring < 0, "should detect ring (negated for sorting)"

    def test_amide_fragment(self):
        """Amide fragment: 2-pyrrolidinone (cyclic amide / lactam)."""
        from orthonym.decomposition.fragment_ranker import score_fragment_seniority

        score = score_fragment_seniority("O=C1CCCN1")
        pg_count, pg_rank, hetero_rank, has_ring, n_heavy = score
        assert pg_count < 0, "should detect at least 1 FG"
        assert has_ring < 0, "should detect ring"

    def test_plain_hydrocarbon(self):
        """Plain hexane: no FG, no ring."""
        from orthonym.decomposition.fragment_ranker import score_fragment_seniority

        score = score_fragment_seniority("CCCCCC")
        pg_count, pg_rank, hetero_rank, has_ring, n_heavy = score
        assert pg_count == 0, "no functional groups"
        assert pg_rank == 999, "no FG -> sentinel rank"
        assert has_ring == 0, "no ring"
        assert n_heavy == -6, "6 heavy atoms (negated)"

    def test_alcohol_fragment(self):
        """Ethanol fragment."""
        from orthonym.decomposition.fragment_ranker import score_fragment_seniority

        score = score_fragment_seniority("CCO")
        pg_count, pg_rank, hetero_rank, has_ring, n_heavy = score
        assert pg_count < 0, "should detect alcohol"
        # Alcohol ranks below carboxylic acid
        assert pg_rank > 0, "alcohol is not rank 0"

    def test_ketone_fragment(self):
        """Acetone fragment."""
        from orthonym.decomposition.fragment_ranker import score_fragment_seniority

        score = score_fragment_seniority("CC(=O)C")
        pg_count, pg_rank, hetero_rank, has_ring, n_heavy = score
        assert pg_count < 0, "should detect ketone"

    def test_invalid_smiles(self):
        """Invalid SMILES returns lowest seniority."""
        from orthonym.decomposition.fragment_ranker import score_fragment_seniority

        score = score_fragment_seniority("INVALID_SMILES")
        assert score == (0, 999, 3, 0, 0)

    def test_none_input(self):
        """None input returns lowest seniority."""
        from orthonym.decomposition.fragment_ranker import score_fragment_seniority

        score = score_fragment_seniority(None)
        assert score == (0, 999, 3, 0, 0)

    def test_carboxylic_acid_more_senior_than_amide(self):
        """Carboxylic acid fragment should score more senior (lower tuple) than amide."""
        from orthonym.decomposition.fragment_ranker import score_fragment_seniority

        acid_score = score_fragment_seniority("OC(=O)c1ccccc1")  # benzoic acid
        amide_score = score_fragment_seniority("O=C1CCCN1")  # pyrrolidinone
        assert acid_score < amide_score, "acid should be more senior (lower score)"

    def test_ring_beats_chain_equal_fg(self):
        """Ring-containing fragment beats chain-only at equal FG seniority."""
        from orthonym.decomposition.fragment_ranker import score_fragment_seniority

        ring_alcohol = score_fragment_seniority("OC1CCCCC1")  # cyclohexanol
        chain_alcohol = score_fragment_seniority("CCCCCCO")  # hexanol
        # Both have alcohol; ring should win (lower score)
        assert ring_alcohol < chain_alcohol, "ring fragment should be more senior"

    def test_larger_fragment_wins_tiebreaker(self):
        """More heavy atoms wins when FG and ring status are equal."""
        from orthonym.decomposition.fragment_ranker import score_fragment_seniority

        larger = score_fragment_seniority("CCCCCCCC")  # octane (8)
        smaller = score_fragment_seniority("CCCCC")  # pentane (5)
        assert larger < smaller, "larger fragment should be more senior"


@pytest.mark.unit
class TestRankFragments:
    """Test rank_fragments() ordering."""

    def test_rank_two_fragments(self):
        """Acid fragment ranked before plain alkyl."""
        from orthonym.decomposition.fragment_ranker import rank_fragments

        frags = [
            {"smiles": "CCCCCC", "side": "alkyl"},
            {"smiles": "OC(=O)c1ccccc1", "side": "acid"},
        ]
        ranked = rank_fragments(frags)
        assert ranked[0]["side"] == "acid", "acid should be first (most senior)"
        assert ranked[1]["side"] == "alkyl"

    def test_rank_preserves_dict_keys(self):
        """Ranking preserves all keys in fragment dicts."""
        from orthonym.decomposition.fragment_ranker import rank_fragments

        frags = [
            {"smiles": "CCO", "side": "alkyl", "extra": "data"},
        ]
        ranked = rank_fragments(frags)
        assert ranked[0]["extra"] == "data"


@pytest.mark.unit
class TestAcidIsMoreSenior:
    """Test acid_is_more_senior() convenience function."""

    def test_acid_is_more_senior_true(self):
        """Carboxylic acid vs plain alkane -> True."""
        from orthonym.decomposition.fragment_ranker import acid_is_more_senior

        assert acid_is_more_senior("CC(=O)O", "CCCCCC") is True

    def test_acid_is_more_senior_false(self):
        """Plain alkane vs carboxylic acid (acid arg is less senior) -> False."""
        from orthonym.decomposition.fragment_ranker import acid_is_more_senior

        assert acid_is_more_senior("CCCCCC", "CC(=O)O") is False

    def test_acid_equal_seniority(self):
        """Equal seniority -> True (acid wins ties)."""
        from orthonym.decomposition.fragment_ranker import acid_is_more_senior

        assert acid_is_more_senior("CCO", "CCO") is True
