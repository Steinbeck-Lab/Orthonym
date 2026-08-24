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


def test_complex_arm_names_substitutively():
    """v36 Wave F (core-namer item 3): a benzyl/aryl-methyl arm now names
    SUBSTITUTIVELY via the sulfinyl/sulfonyl-rooted guard in
    ``name_substituent`` -> ``4-(benzylsulfinyl)pyridine`` (was the stale
    abstain pin, and before the guard the 'a'-replacement
    ``4-(2-(cyclohexa-1,3,5-trien-1-yl)-1-oxo-1-thiaethyl)pyridine``). OPSIN
    round-trips to the input InChIKey (change-asserted-value verified)."""
    from rdkit import Chem
    from rdkit.Chem import inchi
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smi = "O=S(Cc1ccccc1)c1ccncc1"
    r = _be().name_tiered(smi)
    assert r["name"] == "4-(benzylsulfinyl)pyridine"
    assert "oxo" not in r["name"] and "thiaethyl" not in r["name"]
    got = opsin_parse(r["name"])
    assert got and (inchi.MolToInchiKey(Chem.MolFromSmiles(got))
                    == inchi.MolToInchiKey(Chem.MolFromSmiles(smi)))
