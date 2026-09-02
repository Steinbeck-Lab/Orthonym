"""v41 M4 subpart #1 (P-23.2.4 main-bridge selection) -- the decisive
no-regression guard.

The main-bridge fix (``_find_main_ring`` Case 3 + ``_find_main_bridge``
largest-bridge selection, both in ``rules/polycyclic.py``) changes how the main
bicycle of a von Baeyer cage is chosen. This module pins that it does NOT move
any existing von-Baeyer gold row: every ``bicyclo``/``tricyclo``/... /``spiro``
row in the PIN oracle must name byte-identically to its pre-fix output. The
subpart-#1 SPY measured 0 changes across all such rows; this test locks that in.

It also pins the seven Blue Book main-bridge cases at the shippable name level:
- the three subpart-#1 PIN wins now emit the exact Blue Book PIN, and
- the four subpart-#2 cases DEGRADE cleanly -- a valid, round-tripping but
  non-preferred name, or an abstain -- never a wrong name (0-wrong holds).

Requires the OPSIN jar (as the phase gate does); skipped otherwise, because the
namer's output is only defined against a present validator.
"""
import glob
import json
import re

import pytest

from orthonym import name_compound
from orthonym.validation.atom_coverage import find_opsin_jar

pytestmark = [
    pytest.mark.roundtrip,
    pytest.mark.skipif(find_opsin_jar() is None,
                       reason="von-Baeyer gold stability is defined against OPSIN"),
]

# Any von Baeyer / spiro descriptor in the emitted PIN.
_VB = re.compile(r"(?:bi|tri|tetra|penta|hexa|hepta|octa|nona|deca)cyclo\[|spiro")

# Excluded from the stability guard: a substituted-ester row whose emitted name
# is context-dependent inside a shared pytest process (its ester substituent
# recursion abstains to the tropane core under warm budget/state, giving
# ``(1R,3S)-tropane``) -- MEASURED identical on HEAD and after the fix, and its
# von-Baeyer component (bicyclo[3.2.1]) is not a main-bridge case. Pinning a
# single value would be flaky; the main-bridge fix leaves this row byte-identical
# either way (verified in a clean process).
_EXCLUDE = {
    "CN1[C@@H]2CC[C@H]1C[C@H](C2)OC(=O)C(CO)c1ccccc1",
}


def _vb_gold_rows():
    files = ["benchmarks/pin_oracle/gold_pins.json"] + sorted(
        glob.glob("benchmarks/pin_oracle/packs/*.json"))
    rows = []
    seen = set()
    for f in files:
        d = json.load(open(f))
        rr = d["rows"] if isinstance(d, dict) and "rows" in d else (
            d if isinstance(d, list) else [])
        for r in rr:
            pin = r.get("expected_pin") or r.get("pin") or r.get("name")
            smi = r.get("smiles")
            if not (pin and smi):
                continue
            if _VB.search(pin) and smi not in seen and smi not in _EXCLUDE:
                seen.add(smi)
                rows.append((f.split("/")[-1], smi, pin))
    return rows


_VB_GOLD = _vb_gold_rows()


def test_there_are_von_baeyer_gold_rows_to_guard():
    """Guard the guard: if the filter finds nothing, the parametrized test would
    vacuously pass and hide a regression."""
    assert len(_VB_GOLD) >= 60, (
        f"expected the von-Baeyer gold set to be sizable, found {len(_VB_GOLD)}")


@pytest.mark.parametrize(
    "fname,smiles,pin", _VB_GOLD,
    ids=[f"{f}:{s}" for f, s, _ in _VB_GOLD])
def test_von_baeyer_gold_row_is_stable(fname, smiles, pin):
    """Every von-Baeyer gold row must name byte-identically to its gold PIN,
    unchanged by the P-23.2.4 main-bridge fix."""
    got = name_compound(smiles)
    assert got == pin, (
        f"{fname}: {smiles}\n  expected {pin!r}\n  got      {got!r}")


# The three subpart-#1 PIN wins: the P-23.2.4 fix makes these emit the exact
# Blue Book PIN (BlueBookV2.md:9631 / 9721 / 9761). Also added to the gold oracle
# (packs/rings_numbering.json, V41-M41-VBMB-01..03).
_PIN_WINS = [
    ("C123CCCCCCCCCC(CCC1)(CCC2)C3", "tricyclo[9.3.3.1^1,11]octadecane"),
    ("C12C3CCC(C(C4CCC1CC4)CC2)CC3", "tetracyclo[4.4.2.2^2,5.2^7,10]hexadecane"),
    ("C12CC34CCCC(CC(CCCC1)CCC2)(CC3)C4", "tetracyclo[7.4.3.2^3,7.1^3,7]nonadecane"),
]


@pytest.mark.parametrize("smiles,pin", _PIN_WINS)
def test_subpart1_pin_win(smiles, pin):
    assert name_compound(smiles) == pin


# The four subpart-#2 cases degrade cleanly with NO regression: 9749/9753/9731
# fall back to their exact legacy (valid, round-tripping, non-preferred) name,
# and 9739 abstains -- each identical to the HEAD output before the fix.
_DEGRADES = [
    ("C12C3C4CCC(CC(CCC1)CCC2)(CC4)C3", "tetracyclo[6.3.1.3^3,7.2^1,9]heptadecane"),
    ("C12C3C4CCC(CC(CCC1)CCC2)(C4)CC3", "tetracyclo[7.2.1.3^3,7.2^1,8]heptadecane"),
    ("C12C3C4CCCC5CCCC(C(C6CCCC(CCC1)CC26)C3)C4C5",
     "hexacyclo[9.9.1.2^2,6.2^16,20.0^10,25.0^12,22]pentacosane"),
    ("C12CC3CCCCC4CCCCC(CC(CCC5CC5CC1)CCCC2)CC(C4)C3",
     "unknown organic compound"),
]


@pytest.mark.parametrize("smiles,expected", _DEGRADES)
def test_subpart2_degrades_without_regression(smiles, expected):
    assert name_compound(smiles) == expected
