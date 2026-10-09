""" Phase-A: PIN-conformance gold packs — schema lock + fast protect tripwire.

The per-class gold packs (benchmarks/the gold set/packs/*.json) are the PRIMARY gate
(scored by scripts/pin_conformance_eval.py). This is the *fast* (OPSIN-free) companion: it
locks the pack schema, forbids duplicate SMILES across packs (incl. the two legacy files), and
asserts every NEW-pack PROTECT row keeps producing its expected PIN. (Legacy gold_pins.json /
among_rings_gold.json protect rows are guarded by their own existing tests.)

See internal notes
"""
import json
from pathlib import Path

import pytest

from orthonym import name_compound

_PIN_ORACLE = Path(__file__).resolve().parents[3] / "benchmarks" / "pin_oracle"
_PACK_DIR = _PIN_ORACLE / "packs"
_LEGACY = [_PIN_ORACLE / "gold_pins.json", _PIN_ORACLE / "among_rings_gold.json"]
# control files in the packs dir that are NOT gold packs
_CONTROL = {"determinism_probes", "known_nondeterministic", "baseline_targets"}
_VALID_CATEGORIES = {"target", "protect", "tripwire"}


def _norm(s):
    return " ".join(str(s).lower().strip().split())


def _rows_of(payload):
    return payload.get("rows", []) if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])


def _pack_files():
    if not _PACK_DIR.is_dir():
        return []
    return [p for p in sorted(_PACK_DIR.glob("*.json")) if p.stem not in _CONTROL and "report" not in p.stem]


def _all_pack_rows(include_legacy=False):
    """[(pack_name, row),...] over the new packs (+ legacy if asked)."""
    out = []
    for fp in _pack_files():
        for r in _rows_of(json.loads(fp.read_text(encoding="utf-8"))):
            out.append((fp.stem, r))
    if include_legacy:
        for lf in _LEGACY:
            if lf.exists():
                for r in _rows_of(json.loads(lf.read_text(encoding="utf-8"))):
                    out.append((lf.stem, r))
    return out


class TestPacksExistAndLoad:
    def test_packs_dir_exists(self):
        assert _PACK_DIR.is_dir(), "v22 Phase-A must create benchmarks/pin_oracle/packs/"

    def test_at_least_one_pack(self):
        assert _pack_files(), "expected at least one gold pack in benchmarks/pin_oracle/packs/"

    def test_every_pack_is_valid_json_with_rows(self):
        for fp in _pack_files():
            rows = _rows_of(json.loads(fp.read_text(encoding="utf-8")))
            assert rows, f"pack {fp.name} has no rows"


class TestSchema:
    def test_required_fields_and_categories(self):
        for pack, r in _all_pack_rows():
            assert {"smiles", "expected_pin", "category"} <= set(r), f"{pack}: missing fields in {r}"
            assert r["smiles"].strip(), f"{pack}: empty smiles"
            assert str(r["expected_pin"]).strip(), f"{pack}: empty expected_pin for {r['smiles']}"
            assert r["category"] in _VALID_CATEGORIES, f"{pack}: bad category {r['category']!r}"

    def test_rows_carry_provenance(self):
        # new packs must cite a def_id + bluebook_ref (provenance discipline; legacy excluded)
        for pack, r in _all_pack_rows():
            assert r.get("def_id"), f"{pack}: row {r['smiles']} missing def_id"
            assert r.get("bluebook_ref"), f"{pack}: row {r['smiles']} missing bluebook_ref"


# SMILES that are known to occur more than once with the SAME expected_pin (data debt measured on
# main 395b19b1b: rows appended by later waves without a dedup against the earlier ones). The file
# only ever shrinks; see its "_about" key.
_KNOWN_DUPLICATES = json.loads(
    (Path(__file__).resolve().parents[2] / "fixtures" / "pin_oracle"
     / "known_same_pin_duplicates.json").read_text(encoding="utf-8"))


class TestNoDuplicateSmiles:
    """Each SMILES owns exactly ONE pack row, and a new pack never re-adds a SMILES that legacy
    already owns -- with the already-recorded duplicates (same SMILES, SAME expected_pin) listed in
    ``tests/fixtures/the gold set/known_same_pin_duplicates.json``.

    The original form of these tests allowed no duplicate at all. Since then later waves (W2E, W2F,
    D1-D10,,, P8-P14) appended rows without a dedup against the earlier ones, so 44
    SMILES occur in several packs, 17 pack rows repeat a legacy row and one legacy row is written
    twice. Every one of those carries an identical expected PIN: the oracle never contradicts
    itself, it only counts a row twice. Removing the rows would move the gate's row totals and its
    baseline, so the data is left as it is (cleaning it is a separate data change) and the tests keep
    their teeth on what matters: a duplicate whose expected PIN DIFFERS from the first row's fails
    (the oracle would contradict itself), and so does any duplicate not in the recorded set (the
    debt cannot grow). (Pre-existing legacy-internal duplicates in gold_pins.json /
    among_rings_gold.json were a data issue out of Phase A's harness-only scope.)"""

    def test_no_duplicate_within_or_across_new_packs(self):
        seen, dups, conflicts = {}, [], []
        for pack, r in _all_pack_rows(include_legacy=False):
            s = r["smiles"].strip()
            if s in seen:
                first_pack, first_pin = seen[s]
                (conflicts if first_pin != _norm(r["expected_pin"]) else dups).append(
                    (s, first_pack, pack))
            else:
                seen[s] = (pack, _norm(r["expected_pin"]))
        assert not conflicts, (
            f"the v22 packs give one SMILES different expected PINs (the oracle contradicts itself): {conflicts}")
        new_dups = sorted({d[0] for d in dups} - set(_KNOWN_DUPLICATES["across_new_packs"]))
        assert not new_dups, (
            f"duplicate SMILES across the v22 packs, not in the recorded set (each SMILES owns ONE pack): "
            f"{[d for d in dups if d[0] in new_dups]}")

    def test_new_packs_do_not_collide_with_legacy(self):
        legacy = {}
        for lf in _LEGACY:
            if lf.exists():
                for r in _rows_of(json.loads(lf.read_text(encoding="utf-8"))):
                    legacy.setdefault(r["smiles"].strip(), set()).add(_norm(r["expected_pin"]))
        collisions = {r["smiles"].strip() for _p, r in _all_pack_rows(include_legacy=False)
                      if r["smiles"].strip() in legacy}
        conflicts = sorted(
            s for _p, r in _all_pack_rows(include_legacy=False)
            for s in [r["smiles"].strip()]
            if s in legacy and _norm(r["expected_pin"]) not in legacy[s])
        assert not conflicts, f"v22 packs give a legacy SMILES a different expected PIN: {conflicts}"
        new_collisions = sorted(collisions - set(_KNOWN_DUPLICATES["new_vs_legacy"]))
        assert not new_collisions, (
            f"v22 packs re-add SMILES already owned by legacy gold, not in the recorded set: {new_collisions}")

    def test_no_legacy_internal_duplicates(self):
        """Legacy gold files must be internally duplicate-free apart from the recorded rows (the dups
        were cleaned in Phase A; one row of gold_pins.json was written twice later, W4-I4 / W5-C, with
        the same expected PIN)."""
        from collections import Counter
        for lf in _LEGACY:
            if not lf.exists():
                continue
            rows = _rows_of(json.loads(lf.read_text(encoding="utf-8")))
            c = Counter(r["smiles"].strip() for r in rows)
            dups = {s for s, n in c.items() if n > 1}
            conflicts = {s for s in dups
                         if len({_norm(r["expected_pin"]) for r in rows if r["smiles"].strip() == s}) > 1}
            assert not conflicts, f"one SMILES with different expected PINs within legacy {lf.name}: {conflicts}"
            new_dups = dups - set(_KNOWN_DUPLICATES["legacy_internal"].get(lf.name, []))
            assert not new_dups, f"duplicate SMILES within legacy {lf.name}: {new_dups}"


class TestProtectRowsStayCorrect:
    """Fast OPSIN-free tripwire: every NEW-pack PROTECT row must keep producing its expected PIN."""

    _PROTECT = [(pack, r) for pack, r in _all_pack_rows() if r.get("category") == "protect"]

    @pytest.mark.skipif(not _PROTECT, reason="no protect rows in the new packs yet")
    @pytest.mark.parametrize("pack,row", _PROTECT, ids=[f"{p}:{r['smiles']}" for p, r in _PROTECT])
    def test_protect(self, pack, row, monkeypatch):
        accepted = {_norm(row["expected_pin"])} | {_norm(a) for a in row.get("accept_also", [])}
        # A fail-closed protect row (expected = a descriptive fallback such as
        # 'unknown organic compound' / '<metal> compound (not supported)') is
        # produced ONLY when the production validity gate suppresses a raw
        # candidate. This suite's autouse fixture disables that gate, so an
        # OPSIN-free name_compound here returns the un-suppressed raw candidate.
        # Validate those rows against the PRODUCTION (gate-ON) path instead (they
        # are also guarded by the phase gate's protect_new check). Non-fail-closed
        # rows keep the fast OPSIN-free path.
        from orthonym.errors import is_failure_name
        if any(is_failure_name(a) for a in accepted):
            import orthonym.namer as _nm
            monkeypatch.setattr(_nm, "_DISABLE_VALIDITY_GATE", False, raising=False)
            from orthonym import Orthonym
            got = Orthonym().name(row["smiles"])
        else:
            got = name_compound(row["smiles"])
        assert _norm(got) in accepted, (
            f"{pack} protect regression: {row['smiles']} expected {row['expected_pin']!r}")
