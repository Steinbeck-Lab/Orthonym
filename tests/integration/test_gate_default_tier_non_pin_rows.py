"""The PIN-conformance gate and the default-tier rule (user decision 2026-09-30).

The paper, Methods, "Tiers" (L73): "The default configuration emits a name only when the
pipeline can build the preferred IUPAC name (PIN); otherwise, it declines." 23 gold
rows expect a name the code records as not the PIN, so the default tier declines them
with NO_VERIFIED_PIN. The gold rows are not edited: ``scripts/pin_conformance_eval.py``
evaluates the rows of ``benchmarks/the gold set/default_tier_non_pin_rows.json`` at the
best-effort tier and counts one as passed only when the default tier declines it AND the
best-effort tier gives the expected name exactly.
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pin_conformance_eval as pce  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]

_ROWS_FILE = PROJECT_ROOT / "benchmarks" / "pin_oracle" / "default_tier_non_pin_rows.json"


def _rows():
    return json.loads(_ROWS_FILE.read_text(encoding="utf-8"))["rows"]


def _gold_rows():
    return {(pack, r.get("def_id", ""), r["smiles"]): r
            for pack, rows in pce.load_packs(pce._DEFAULT_PACK_DIR, pce._DEFAULT_LEGACY)
            for r in rows}


def test_every_listed_row_is_an_unchanged_gold_row_with_its_reason():
    rows = _rows()
    assert len(rows) == 23
    gold = _gold_rows()
    for r in rows:
        g = gold.get((r["pack"], r["def_id"], r["smiles"]))
        assert g is not None, r
        assert g["expected_pin"] == r["expected_pin"], r      # gold data unchanged
        assert g.get("category", "target") == r["category"], r
        assert r["reason"] and "BlueBookV2.md:" in r["bluebook"], r
    assert {r["category"] for r in rows} == {"target", "protect", "tripwire"}


@pytest.fixture(scope="module")
def namers():
    namer_on, _off, _rt = pce._build_namers(False)
    return namer_on, pce._build_best_effort_namer()


@pytest.mark.parametrize("row", _rows(), ids=[r["expected_pin"][:40] for r in _rows()])
def test_a_listed_row_passes_only_through_the_default_decline_and_best_effort(row, namers):
    namer_on, namer_be = namers
    keys = pce.load_default_tier_non_pin_rows()
    rec = pce._name_row(row["pack"], row, 60, False, namer_on, None, None, namer_be, keys)
    assert rec["evaluated_at"] == "best-effort", rec
    assert rec["default_tier_limit_code"] == "NO_VERIFIED_PIN", rec
    assert rec["verdict"] == "MATCH" and rec["shipped_name"] == row["expected_pin"], rec


class _Fake:
    def __init__(self, name, limit_code=None):
        self._name, self._limit = name, limit_code

    def name(self, smiles):
        return self._name

    def name_tiered(self, smiles):
        return {"name": self._name, "limit_code": self._limit}


def test_the_verdicts_of_a_listed_row():
    row = _rows()[0]
    keys = pce.load_default_tier_non_pin_rows()
    shipped = _Fake(row["expected_pin"])
    declined = _Fake("unknown organic compound", "NO_VERIFIED_PIN")
    # the default tier emits the name again: that is not a pass
    rec = pce._name_row(row["pack"], row, 60, False, shipped, None, None, shipped, keys)
    assert rec["verdict"] == "NON_PIN_SHIPPED_AT_DEFAULT", rec
    # the best-effort tier gives another name
    rec = pce._name_row(row["pack"], row, 60, False, declined, None, None,
                        _Fake("another name"), keys)
    assert rec["verdict"] == "MISMATCH", rec
    rec = pce._name_row(row["pack"], row, 60, False, declined, None, None, shipped, keys)
    assert rec["verdict"] == "MATCH", rec


def test_a_row_outside_the_list_is_named_at_the_default_tier():
    row = {"smiles": "CCO", "expected_pin": "ethanol", "def_id": "x", "category": "target"}
    rec = pce._name_row("p", row, 60, False, _Fake("ethanol"), None, None,
                        _Fake("not used"), pce.load_default_tier_non_pin_rows())
    assert rec["verdict"] == "MATCH" and "evaluated_at" not in rec, rec
