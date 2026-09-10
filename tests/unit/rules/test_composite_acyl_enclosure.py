"""P-16.5.1.4 — a substituent prefix that includes a parent-hydride name is enclosed.

"Parentheses are placed around substituent groups including the name of a parent
hydride in order to avoid any confusion from having two parent hydrides in a
substitutive name" (P-16.5.1.4, the Blue Book; the rule's own example is
cyclohexanecarbonyl). Two independent producers missed it:

  * needs_brackets() had no parent-hydride + acyl-suffix branch (Site A):
      hydrazinecarbonyl -> (hydrazinecarbonyl), 2-(hydrazinecarbonyl)benzene-1-
      sulfonic acid (the Blue Book); hydrazinecarbohydrazonoyl likewise (the Blue Book).
  * decomposition/fragment_assembly.py wrapped the acyl N-substituent with
      _wrap_n_substituent (escalate-only) instead of enclose_if_compound first
      (Site B): N-(furan-2-carbonyl)furan-2-carboxamide (the Blue Book).

The Site-A branch is gated on a CURATED parent-hydride stem list: 'hydroxy' in
hydroxycarbonimidoyl is a simple substituent on the acyl carbon, not a second parent
hydride, so it is NOT matched by the new branch (its enclosure, when present, comes
from the pre-existing C-locant rule).

Asserted at the producer level (needs_brackets) for determinism; the corrected PINs
are round-trip-checked. End-to-end flips are covered by the bb_conformance gate.
"""
import pytest
from rdkit import Chem


@pytest.mark.parametrize("prefix", [
    "hydrazinecarbonyl", "hydrazinecarbohydrazonoyl", "cyclohexanecarbonyl",
])
def test_parent_hydride_acyl_prefix_needs_brackets(prefix):
    from orthonym.assembly.naming_utils import needs_brackets
    assert needs_brackets(prefix) is True, prefix


def test_hydroxy_acyl_root_not_matched_by_new_branch():
    """The curated-stem gate: 'hydroxy' is not a parent hydride, so the NEW acyl
    branch does not fire for it (guards the hydroxycarbonimidoyl MATCH rows)."""
    from orthonym.assembly.naming_utils import (
        _ACYL_PARENT_HYDRIDE_STEMS, _ACYL_SUFFIX_FAMILY,
    )
    name = "hydroxycarbonimidoyl"
    fired = any(
        name.startswith(stem)
        and name[len(stem):].lstrip("-0123456789") in _ACYL_SUFFIX_FAMILY
        for stem in _ACYL_PARENT_HYDRIDE_STEMS
    )
    assert fired is False


@pytest.mark.parametrize("smiles,expected_pin", [
    ("NNC(=O)c1ccccc1S(=O)(=O)O", "2-(hydrazinecarbonyl)benzene-1-sulfonic acid"),
    ("NN=C(NN)c1cccc(C(=O)O)c1", "3-(hydrazinecarbohydrazonoyl)benzoic acid"),
    ("O=C(NC(=O)c1ccco1)c1ccco1", "N-(furan-2-carbonyl)furan-2-carboxamide"),
])
def test_expected_pins_round_trip(smiles, expected_pin):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    rt = opsin_parse(expected_pin)
    ref = Chem.MolFromSmiles(smiles)
    got = Chem.MolFromSmiles(rt) if rt else None
    assert got is not None and Chem.MolToInchiKey(got) == Chem.MolToInchiKey(ref), \
        f"RT failed for {expected_pin!r}"
