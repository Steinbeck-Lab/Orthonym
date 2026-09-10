"""-REGRESSION — end-to-end oracle for the four gold targets that regressed.

These four are `category: target` rows in `benchmarks/the gold set/packs/
characteristic_groups.json`. They passed the FULL gate at and at
, then regressed at (`target_passes` 1641 -> 1637), because
`polyfunctional.format_fg_prefix` enclosed an already-enclosed prefix a second time.

WHY THIS FILE LIVES IN `tests/integration/` AND RE-ENABLES THE GATE
-------------------------------------------------------------------
Two hazards make the obvious version of this test worthless:

1. `tests/conftest.py:267` is an **autouse** fixture that sets
   `namer._DISABLE_VALIDITY_GATE = True` for every test, so a plain unit test
   asserts RAW output, not the shipped string. Each test here restores the
   production setting first, the way `test_opsin_validity_gate.py` does.
2. `pytest tests/unit` is known to DEADLOCK on an OPSIN pipe. These cases each
   spawn OPSIN, so they belong with the integration tier, not in `tests/unit`.

All four names round-trip through OPSIN cleanly in BOTH the correct and the broken
spelling — OPSIN re-parses the redundant brackets and the missing hyphen and
returns the right structure. So no round-trip check can defend this class; only an
exact-match oracle can. That is the whole reason the PIN gate exists.
"""

import pytest

pytestmark = [pytest.mark.integration]


# (SMILES, expected PIN) — verbatim from the gold rows, def_ids in the comments.
REGRESSED_GOLD_TARGETS = [
    # W2C-D-AM-04
    ("CCN=C(CCC(=O)OC)N(C)C",
     "methyl 4-(dimethylamino)-4-(ethylimino)butanoate"),
    # W2F-P2-04
    ("O=C(O)CCCC(=O)OCCCO",
     "5-(3-hydroxypropoxy)-5-oxopentanoic acid"),
    # W2F-P2-08
    ("CCSC(=O)CCC(=O)O",
     "4-(ethylsulfanyl)-4-oxobutanoic acid"),
    # W3-P02-6
    ("ON=C(O)CCCC(=O)O",
     "5-hydroxy-5-(hydroxyimino)pentanoic acid"),
]

# The four CRITICALs fixed by. The regression fix must not undo them:
# it narrows only the ENCLOSURE question, never the multiplier question.
MULTIPLIER_CRITICAL_CONTROLS = [
    ("O=Nc1ccc(N=O)cc1", "1,4-dinitrosobenzene"),
    ("COCc1ccc(COC)cc1", "1,4-bis(methoxymethyl)benzene"),
    ("CC(C)(C)c1ccccc1C(C)(C)C", "1,2-di-tert-butylbenzene"),
    ("CC(C)c1ccc(C(C)C)cc1", "1,4-di(propan-2-yl)benzene"),
]


@pytest.fixture
def shipped_namer(monkeypatch):
    """A namer with the production OPSIN validity gate ON.

    Without this the autouse fixture in `tests/conftest.py` leaves the gate OFF and
    the assertions below would be about raw output rather than shipped output.
    """
    import orthonym.namer as _namer

    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    return _namer.Orthonym()


def test_the_gate_is_really_on(shipped_namer):
    """Guards the fixture itself — if the flag stopped taking effect, every
    assertion in this file would silently become a gate-OFF measurement."""
    import orthonym.namer as _namer

    assert _namer._DISABLE_VALIDITY_GATE is False


@pytest.mark.parametrize("smiles,expected", REGRESSED_GOLD_TARGETS)
def test_regressed_gold_target_emits_its_pin(shipped_namer, smiles, expected):
    assert shipped_namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", MULTIPLIER_CRITICAL_CONTROLS)
def test_multiplier_criticals_still_hold(shipped_namer, smiles, expected):
    assert shipped_namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", REGRESSED_GOLD_TARGETS)
def test_no_doubled_enclosure_in_the_emitted_name(shipped_namer, smiles, expected):
    """The defect's fingerprint, asserted independently of the exact PIN.

    `[(` / `)]` around a single prefix, and a locant glued straight onto a closing
    mark, are both malformed regardless of which name is preferred — so this catches
    the class even if one of the golds above is ever legitimately re-based.
    """
    import re

    name = shipped_namer.name(smiles)
    assert "[(" not in name, name
    assert ")]" not in name, name
    assert not re.search(r"[)\]}]\d", name), name
