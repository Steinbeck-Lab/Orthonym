"""W2F-P1 Tasks 6, 8, 9 — P-29.6.2.1 α-substituted benzyl prefixes + the
P-16.5.2.4/P-16.5.4.1 citation-layer escalation both this class and the
asymmetric benzylic ether (Task 7) require.

BB P-29.6.2.1 (BlueBookV2.md:16322): 'bromo(4-methylphenyl)methyl (preferred
prefix)'. The enclosing marks are STRUCTURE-BEARING: OPSIN parses the
marks-dropped 'bromo(phenyl)methylbenzene' to a DIFFERENT molecule
(Brc1ccccc1Cc1ccccc1 — research §3.A).

All expected names OPSIN-2.9-verified in
 §3.D (and §2.D for Task-7 shapes).
"""
import pytest
from rdkit import Chem

from orthonym.namer import name_compound

UNKNOWN = "unknown organic compound"


@pytest.mark.unit
class TestCitationLayerEscalation:
    """Task 6: format_substituent_prefix must never cite a mark-bearing
    compound prefix bare, and must escalate (not double) the outer mark."""

    def test_interior_parens_escalate_to_brackets(self):
        from orthonym.assembly.naming_utils import format_substituent_prefix
        assert format_substituent_prefix("bromo(phenyl)methyl", [4], 1) == \
            "4-[bromo(phenyl)methyl]"

    def test_trailing_stem_after_brackets_escalates_to_braces(self):
        from orthonym.assembly.naming_utils import format_substituent_prefix
        assert format_substituent_prefix(
            "[(4-methoxyphenyl)methoxy]methyl", [4], 1
        ) == "4-{[(4-methoxyphenyl)methoxy]methyl}"

    def test_leading_paren_shape_unchanged(self):
        # pre-existing behavior (W2E-P1FC Task 8 branch) must be preserved
        from orthonym.assembly.naming_utils import format_substituent_prefix
        assert format_substituent_prefix("(benzylsulfanyl)methyl", [4], 1) == \
            "4-[(benzylsulfanyl)methyl]"

    def test_simple_and_complex_markless_shapes_unchanged(self):
        from orthonym.assembly.naming_utils import format_substituent_prefix
        assert format_substituent_prefix("methyl", [2, 2], 2) == "2,2-dimethyl"
        assert format_substituent_prefix("1-methylethyl", [4, 7], 2) == \
            "4,7-bis(1-methylethyl)"
        assert format_substituent_prefix("chloromethyl", [1], 1) == \
            "1-(chloromethyl)"

    def test_fusion_brackets_not_escalated(self):
        # P-16.5.4.1.2: fusion brackets are nesting-IGNORED -> plain parens
        from orthonym.assembly.naming_utils import format_substituent_prefix
        assert format_substituent_prefix("furo[3,2-b]pyridin-2-yl", [4], 1) == \
            "4-(furo[3,2-b]pyridin-2-yl)"
