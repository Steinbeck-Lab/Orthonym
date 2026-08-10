"""v30 breadth — sulfinyl / sulfonyl substituent on a heteroaromatic parent.

Benzene names a -S(=O)R / -S(=O)(=O)R substituent as (R)sulfinyl / (R)sulfonyl
(`(methanesulfinyl)benzene`), but every HETEROaromatic parent dropped it: the
heterocycle substituent collector counted the R carbons and dropped the S + its
=O, so the whole molecule abstained (SELF-01 caught the atom-drop). The recursive
`name_substituent(..., allow_mancude=True)` already builds these prefixes; the
heterocycle path now routes a ring-borne S(=O)-substituent through it, behind a
gate-independent atom-coverage guard. best-effort-gated -> PIN default byte-identical.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.unit


def _be():
    return Orthonym(style="pin", **_emit_tier_flags("best-effort"))


@pytest.mark.parametrize("smi,expected", [
    ("CS(=O)c1ccncc1", "4-(methanesulfinyl)pyridine"),
    ("CS(=O)(=O)c1ccncc1", "4-(methanesulfonyl)pyridine"),
])
def test_heteroaryl_sulfinyl_names_at_best_effort(smi, expected):
    assert _be().name_tiered(smi)["name"] == expected


def test_pin_default_byte_identical():
    """The block is best-effort-gated (best_effort_ctx default False), so the PIN
    default path is unchanged — pyridine-sulfinyl still abstains at PIN."""
    from orthonym.errors import is_failure_name
    assert is_failure_name(Orthonym(style="pin").name("CS(=O)c1ccncc1"))


def test_benzene_control_unchanged():
    assert Orthonym(style="pin").name("CS(=O)c1ccccc1") == "(methanesulfinyl)benzene"


def test_complex_arm_fails_closed():
    """A benzyl/aryl-methyl arm (`benzylsulfinyl`) is not yet buildable by the
    recursive namer -> the coverage guard fails and the molecule abstains (safe,
    no atom-dropping name). Documented follow-up (blocks the full omeprazole)."""
    r = _be().name_tiered("O=S(Cc1ccccc1)c1ccncc1")
    assert r["name"] is None or r["tier"] == "T5"
