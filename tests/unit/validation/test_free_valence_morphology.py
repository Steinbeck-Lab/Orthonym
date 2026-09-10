"""The free-valence morphology oracle.

a phase. A pure text oracle, exactly the ``token_arity`` discipline: it reads
a prefix token's own morphemes and reports how many free valences that text
ASSERTS, or refuses. It is shared by the two places that must agree about it --
the producer guard in ``substituent_enumerator`` and the spine's P7 -- so that
"what the name says" is decided once rather than twice.

A confident answer must be correct. Everything the ending does not settle is
refused, because a wrong confident verdict here either suppresses a good name
(producer side) or certifies a bad one (proof side).
"""
import pytest

from orthonym.validation.name_morphemes import free_valence_morphology

pytestmark = pytest.mark.unit


class TestConfidentReadings:

    @pytest.mark.parametrize("token,expected", [
        # one free valence
        ("methyl", 1),
        ("propan-2-yl", 1),
        ("phenyl", 1),
        ("2-hydroxyethyl", 1),
        ("acetyl", 1),
        ("methoxycarbonyl", 1),
        # two on the same atom
        ("methylidene", 2),
        ("ethylidene", 2),
        ("propan-2-ylidene", 2),
        ("benzylidene", 2),
        ("sulfanylidene", 2),
        ("hydrazinylidene", 2),
        # three on the same atom
        ("methylidyne", 3),
        ("ethylidyne", 3),
    ])
    def test_reads_the_ending(self, token, expected):
        est = free_valence_morphology(token)
        assert est.confident, est.basis
        assert est.free_valences == expected

    def test_case_and_whitespace_insensitive(self):
        assert free_valence_morphology("  Methylidene ").free_valences == 2


class TestRefusals:
    """Refused, never guessed. ``free_valences`` is None when unconfident."""

    @pytest.mark.parametrize("token", [
        # spells no free-valence morpheme at all -- these are correct prefixes
        # for multivalent attachments but say nothing in the -yl lexicon
        "oxo", "hydroxy", "chloro", "amino", "imino", "nitro", "cyano",
        "azido", "phenoxy",
        # multiplied endings spread their free valences over several atoms,
        # so a single-attachment rule cannot judge them
        "ethane-1,2-diyl", "benzene-1,3,5-triyl", "propane-1,2,3-triyl",
        "butane-1,4-diylidene",
        # degenerate input
        "", "   ", "1,2-",
    ])
    def test_refused(self, token):
        est = free_valence_morphology(token)
        assert not est.confident, f"{token!r} was answered {est.free_valences}"
        assert est.free_valences is None
        assert est.basis

    def test_non_string_refused(self):
        assert not free_valence_morphology(None).confident


class TestTheOracleAgreesWithItsTwoConsumers:
    """The point of sharing it: producer and proof cannot drift apart."""

    VOCABULARY = [
        "methyl", "propan-2-yl", "phenyl", "acetyl", "2-hydroxyethyl",
        "methylidene", "ethylidene", "propan-2-ylidene", "sulfanylidene",
        "methylidyne", "ethylidyne",
        "oxo", "hydroxy", "chloro", "imino", "cyano", "nitro",
        "ethane-1,2-diyl", "benzene-1,3,5-triyl", "", "1,2-",
    ]

    def test_producer_guard_is_the_oracle_over_the_whole_vocabulary(self):
        """Not a spot check: for EVERY token the producer's guard must be
        exactly 'the oracle is confident and says one'. Any independent second
        implementation would show up here as a disagreement."""
        from orthonym.assembly.substituent_enumerator import (
            _token_asserts_single_free_valence,
        )
        for token in self.VOCABULARY:
            est = free_valence_morphology(token)
            expected = bool(est.confident and est.free_valences == 1)
            assert _token_asserts_single_free_valence(token) is expected, (
                f"guard and oracle disagree on {token!r}"
            )

    def test_the_guard_is_not_vacuously_true_or_false(self):
        """Mutation guard for the loop above: the vocabulary really does
        contain both answers, so a guard hardwired to either constant fails."""
        from orthonym.assembly.substituent_enumerator import (
            _token_asserts_single_free_valence,
        )
        answers = {_token_asserts_single_free_valence(t)
                   for t in self.VOCABULARY}
        assert answers == {True, False}
