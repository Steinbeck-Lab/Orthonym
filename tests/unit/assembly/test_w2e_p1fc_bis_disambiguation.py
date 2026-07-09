"""W2E-P1FC Task 2 — P-16.3.6 (a) (BB 7134): bis/tris before ambiguous
mononuclear prefixes. "bis(sulfanyl) ... defines two -SH groups ... whereas
disulfanyl defines the -SSH group".

STALE-SPEC NOTE (2026-07-09): the plan recorded HEAD as emitting the WRONG
`disulfanyl` form for two -SH on one carbon and proposed adding a new
`AMBIGUOUS_MONONUCLEAR_PREFIXES` frozenset. At execution HEAD ALREADY emits
`bis(sulfanyl)` correctly, via the pre-existing
`naming_utils.CATENATION_AMBIGUOUS_PREFIXES` frozenset (consumed by
`get_multiplier_prefix`). No new symbol was added (that would duplicate the
existing constant). This file therefore pins the existing correct behaviour
so it can never regress, and protects the genuine catenated -SSH disulfanyl.
"""
import pytest

from orthonym.namer import name_compound
from orthonym.assembly.naming_utils import (
    CATENATION_AMBIGUOUS_PREFIXES,
    get_multiplier_prefix,
)
from orthonym.rules.polyfunctional import format_fg_prefix


@pytest.mark.unit
class TestP1636BisDisambiguation:
    def test_frozenset_covers_the_p1636a_class(self):
        # P-16.3.6 (a) note: the S, Se, Te, N, P, As, Sb analogues whose
        # di-form collides with a catenated-hydride prefix, plus -OOH (dioxidanyl
        # vs peroxy). CATENATION_AMBIGUOUS_PREFIXES is the existing owner.
        required = {
            "sulfanyl", "selanyl", "tellanyl", "azanyl",
            "phosphanyl", "arsanyl", "stibanyl",
        }
        assert required <= CATENATION_AMBIGUOUS_PREFIXES

    def test_multiplier_switches_to_bis(self):
        assert get_multiplier_prefix(2, "sulfanyl") == "bis"
        assert get_multiplier_prefix(3, "sulfanyl") == "tris"
        # non-ambiguous simple prefixes keep di/tri
        assert get_multiplier_prefix(2, "methyl") == "di"
        assert get_multiplier_prefix(2, "hydroxy") == "di"

    def test_format_fg_prefix_parenthesizes(self):
        assert format_fg_prefix("sulfanyl", [3, 3], 2) == "3,3-bis(sulfanyl)"
        # count==1 stays bare
        assert format_fg_prefix("sulfanyl", [3], 1) == "3-sulfanyl"

    @pytest.mark.parametrize("smiles,expected", [
        ("SC(S)CC(=O)O", "3,3-bis(sulfanyl)propanoic acid"),
        ("SC(S)CCC(=O)O", "4,4-bis(sulfanyl)butanoic acid"),
    ])
    def test_end_to_end(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # genuine catenated -SSH keeps 'disulfanyl' (P-63.4.2.2)
        ("SSCCC(=O)O", "3-disulfanylpropanoic acid"),
        ("SCCC(=O)O", "3-sulfanylpropanoic acid"),
    ])
    def test_protect_unchanged(self, smiles, expected):
        assert name_compound(smiles) == expected
