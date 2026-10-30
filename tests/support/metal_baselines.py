"""The baseline runs of the metal and organometallic naming work, and what an identity test reads.

A baseline is a run file of ``scripts/metal_panel.py`` made at the start commit of the work:
one JSON row per (input, tier). Three of them are the identity fixtures that every task editing
shared code compares with, so that a metal not yet admitted keeps its names:

* ``zinc`` ``tests/fixtures/zinc/baseline_zinc_<START>.jsonl`` the 87-pair zinc panel
* ``other`` ``tests/fixtures/zinc/baseline_other_<START>.jsonl`` the 73-drawing other-metals panel
* ``other98`` ``tests_local/fixtures/zinc/baseline_other98_<START>.jsonl`` the 98 non-zinc census rows
  (``tests_local`` is not published: a published test never asks for this kind)

A row also carries the time the process took, the read-back, the structure class and the input's
own columns. None of those is part of the identity of a name, and two runs of one tree differ in
them (the time always does). An identity test therefore compares only ``IDENTITY_FIELDS``:
the name, the label (tier), whether the label claims a PIN, and the limit code of an unnamed row.
"""
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]

#: the fields of a run row that make up the identity of its name
IDENTITY_FIELDS = ("name", "tier", "is_pin", "limit_code")

_KINDS = {
    "zinc": (ROOT / "tests" / "fixtures" / "zinc", "baseline_zinc_*.jsonl"),
    "other": (ROOT / "tests" / "fixtures" / "zinc", "baseline_other_*.jsonl"),
    "other98": (ROOT / "tests_local" / "fixtures" / "zinc", "baseline_other98_*.jsonl"),
}

Key = Tuple[str, str]


def baseline_path(kind: str, directory: Optional[Path] = None) -> Path:
    """The one baseline file of ``kind``. Exactly one file must match: a second start commit's
    baseline beside the first would let a test read the wrong one."""
    folder, pattern = _KINDS[kind]
    found = sorted((directory or folder).glob(pattern))
    if len(found) != 1:
        raise AssertionError(f"{kind}: expected exactly one file {pattern} in {directory or folder}, "
                             f"found {[p.name for p in found]}")
    return found[0]


def read_rows(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def identity_view(row: dict) -> Tuple:
    """The identity of one run row: ``(name, tier, is_pin, limit_code)``."""
    return tuple(row.get(field) for field in IDENTITY_FIELDS)


def identity_rows(rows: Iterable[dict]) -> Dict[Key, Tuple]:
    """``{(smiles, tier_req): identity_view}`` of a run file; a repeated key is an error."""
    out: Dict[Key, Tuple] = {}
    for row in rows:
        key = (row["smiles"], row["tier_req"])
        if key in out:
            raise AssertionError(f"two rows for {key}")
        out[key] = identity_view(row)
    return out


def identity_diff(base_rows: Iterable[dict], new_rows: Iterable[dict]) -> List[Tuple[Key, Optional[Tuple], Optional[Tuple]]]:
    """Every (smiles, tier_req) whose identity differs between two run files, and every one that is
    in one file only (the missing side is ``None``). Empty when the two are identical."""
    base, new = identity_rows(base_rows), identity_rows(new_rows)
    return [(key, base.get(key), new.get(key)) for key in sorted(set(base) | set(new))
            if base.get(key) != new.get(key)]


def declared_bonded_metal_moves() -> Dict[Key, Tuple]:
    """The rows of the other-metals baseline that the admission of the bonded metals moves (Al, Hg, Pd, Pt, Ru, Ti,
    V, W to every mechanism of zinc, 2026-10-30, ``data.metal_scope.BONDED_METAL_VALENCES``):
    ``{(smiles, tier_req): identity_view}`` of each row as it is named now. A guard of the metals the table does not
    admit compares such a row with this file, and reads a name of it back to the full key."""
    path = _KINDS["zinc"][0] / "declared_moves_bonded_b1.tsv"
    out: Dict[Key, Tuple] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            if not line.strip():
                continue
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            out[(row["smiles"], row["tier_req"])] = (row["name"] or None, row["tier"], row["is_pin"] == "true",
                                                     row["limit_code"] or None)
    return out
