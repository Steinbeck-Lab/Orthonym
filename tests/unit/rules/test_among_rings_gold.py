"""WS-A among-rings gate: corpus integrity + the Phase-171 regression guard.

The among-rings PIN gold corpus (benchmarks/the gold set/among_rings_gold.json)
is the authoritative gate for the WS-A parent-selection chokepoint — it is
scored by scripts/pin_strict_eval.py. This module is the *fast* (OPSIN-free)
companion: it locks the corpus schema, asserts every current PROTECT row
stays correct, and encodes the one regression that the corpus rows alone
cannot express — the Phase-171 CHEBI:59269 stereo-drop.

Phase-171 finding (reproduced this session in an isolated worktree): the P-1
trial that made the P-44.2 among-rings winner authoritative flipped
4-cyclohexylpyridine and 2-phenylfuran correctly but regressed the
OPSIN-self-test-500 by exactly one row — CHEBI:59269. The root cause is NOT
the parent decision itself; it is a DOWNSTREAM coupling: the molecule's true
parent is the propan-1-amine CHAIN (amine is the PCG, both rings are mere
substituents), but flipping features.principal_ring from the benzene to the
pyridine ring perturbed handler dispatch on a chain-parented molecule and the
(3S) stereodescriptor was dropped. The WS-A S2 step (deleting size_diff>=3)
must therefore NOT reassign principal_ring when the chain is the parent, and
must leave the chain handler's stereo emission intact.
"""
import json
from pathlib import Path

import pytest

from orthonym import name_compound

GOLD = (Path(__file__).resolve().parents[3]
        / "benchmarks" / "pin_oracle" / "among_rings_gold.json")


def _load():
    return json.loads(GOLD.read_text())


def _norm(s):
    return " ".join(s.lower().strip().split())


class TestCorpusIntegrity:
    def test_corpus_loads_and_has_rows(self):
        rows = _load()["rows"]
        assert len(rows) >= 40

    def test_schema_and_categories(self):
        for r in _load()["rows"]:
            assert {"smiles", "expected_pin", "category"} <= set(r)
            assert r["category"] in ("target", "protect")

    def test_has_both_targets_and_protects(self):
        cats = [r["category"] for r in _load()["rows"]]
        assert "target" in cats and "protect" in cats

    def test_no_duplicate_smiles(self):
        smis = [r["smiles"] for r in _load()["rows"]]
        assert len(smis) == len(set(smis))


class TestProtectRowsStayCorrect:
    """Fast (OPSIN-free) regression tripwire: every PROTECT row must keep
    producing its expected PIN (or an accept_also spelling). The WS-A
    chokepoint must not regress any of these."""

    @pytest.mark.parametrize("row", [
        r for r in _load()["rows"] if r["category"] == "protect"
    ], ids=lambda r: r["smiles"])
    def test_protect(self, row):
        accepted = {_norm(row["expected_pin"])}
        accepted |= {_norm(a) for a in row.get("accept_also", [])}
        assert _norm(name_compound(row["smiles"])) in accepted


class TestPhase171StereoGuard:
    """The load-bearing downstream-coupling guard (not a corpus row)."""

    SMILES = "CN(C)CC[C@@H](c1ccc(Br)cc1)c1ccccn1"  # CHEBI:59269

    def test_stereodescriptor_survives(self):
        # The (3S) configuration MUST appear. The Phase-171 trial dropped it;
        # any WS-A parent change that perturbs chain-parented stereo emission
        # will fail here. Spelling-robust: only the descriptor is asserted.
        out = name_compound(self.SMILES).lower()
        assert "3s" in out.replace("(", "").replace(")", ""), (
            f"(3S) stereodescriptor dropped — Phase-171 regression: {out!r}"
        )

    def test_both_rings_named_as_substituents(self):
        # Sanity: the chain is the parent (propan-1-amine), so both the
        # bromophenyl and the pyridinyl appear as substituents.
        out = name_compound(self.SMILES).lower()
        assert "bromophenyl" in out
        assert "pyridin" in out


class TestNeverDropCarboxy:
    """/: a demoted ring's carboxylic acid must be carried as a
    (carboxy...) prefix, never silently dropped. This test is RED until the
    carboxy branch lands in ring_atom_fg_prefixes (Plan 03)."""

    def test_carboxy_carried_not_dropped(self):
        # OC(=O)CCCCC(C(=O)O)C1CCCCC1C(=O)O: the heptanedioic-acid chain is
        # the parent; the cyclohexane ring carries its own -COOH which must
        # survive as (2-carboxycyclohexyl). HEAD currently emits
        # 2-cyclohexylheptanedioic acid (ring COOH DROPPED, RT-False).
        out = name_compound("OC(=O)CCCCC(C(=O)O)C1CCCCC1C(=O)O")
        assert _norm(out) == "2-(2-carboxycyclohexyl)heptanedioic acid", (
            f"ring carboxy dropped (D-07 violation): {out!r}"
        )

    def test_ester_not_called_carboxy(self):
        # COC(=O)C1CCCCC1CCCC(=O)O: the ring bears a -C(=O)OMe ester, NOT a
        # free acid. The carboxy branch must NOT fire (ester-exclusion,).
        # Acceptable: the methoxycarbonyl PIN, OR the legacy drop form (RT-False
        # but never WRONG-carboxy). The single hard assertion is: no 'carboxy'.
        out = name_compound("COC(=O)C1CCCCC1CCCC(=O)O").lower()
        assert "carboxy" not in out, (
            f"ester wrongly emitted as carboxy (D-08 ester-exclusion violation): {out!r}"
        )
