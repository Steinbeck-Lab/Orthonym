""" RISK 5 Class 3 (backstop) — a substituent prefix can never contain a SPACE.

A space marks a FUNCTIONAL-CLASS multi-word name (e.g. 'urea oxime', 'taxifoline acetate'),
which is a whole-molecule name, never a valid single substituent token. `parent_to_prefix`
used to run its `-e`->`-yl` fallback on such a name and fabricate the OPSIN-UNPARSEABLE token
`urea oximyl` (spliced into `2-amino-5-urea oximylpentanoic acid`). Fail closed instead, so the
caller degrades to a clean abstention / a valid uglier name rather than shipping garbage.

A space in a prefix forces parentheses / voids the token, and an unnameable ligand
voids the candidate. The standalone
functional-class name `urea oxime` itself is OPSIN-valid and is NOT affected (this guards only
the substituent-prefix conversion).
"""
from orthonym.assembly.substituent_naming import parent_to_prefix


def test_spaced_functional_class_name_declines():
    assert parent_to_prefix("urea oxime", 3, attach_locant=1) is None


def test_ordinary_prefix_unchanged():
    assert parent_to_prefix("ethanol", 2, attach_locant=1) == "hydroxyethyl"
    assert parent_to_prefix("pyridine", 5, attach_locant=1) == "pyridinyl"


def test_no_spaced_token_in_full_name():
    # end-to-end: the arginine-oxime side chain must not splice a spaced token.
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    r = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(
        "NC(=NO)NCCCC(N)C(=O)O")
    nm = r["name"] if isinstance(r, dict) else r
    if nm:
        assert " oxim" not in nm and "urea oxim" not in nm, nm
