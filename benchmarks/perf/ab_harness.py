#!/usr/bin/env python3
"""M1 perf — byte-identity A/B + timing harness (Step 0, FABLE finding 6).

The load-bearing safety gate for the memoization levers: name a fixed corpus with
ORTHONYM_MEMO=off and =on and diff the emitted names byte-for-byte. Zero diffs is
the contract; any diff is a wrong cache key. The harness is corpus-agnostic and runs
one molecule at a time under a hard per-molecule timeout so a giant cannot stall it.

Usage:
  # produce a name map (jsonl: {"smiles":..., "name":...}) for a corpus + tier
  ab_harness.py name  --corpus dev500|chebi200  --tier pin|besteffort  --out FILE
  # diff two name maps produced by `name` (byte-for-byte)
  ab_harness.py diff  --a FILE --b FILE
  # timing: median marginal s/mol over a corpus (excludes molecule #1 = one-time cost)
  ab_harness.py time  --corpus chebi200 --tier besteffort --runs 3

Deterministic corpus selection is baked in (see _load_corpus) so before/after numbers
are reproducible. ORTHONYM_MEMO is read by the engine (once it exists); this harness
only sets/omits it via the environment of the caller.
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import logging
logging.getLogger().setLevel(logging.ERROR)
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from orthonym import name_compound
from orthonym.jvm_budget import jvm_slots

TIMEOUT_SEC = 90


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _load_corpus(corpus: str):
    """Deterministic corpus → list[smiles]. dev500 = the byte-identity split;
    chebi200 = first 200 ChEBI rows with 25-60 heavy atoms (perf slice, matches the
    PERF-OPTIMIZATION-PLAN Step 0 selection)."""
    if corpus == "dev500":
        d = json.load(open(PROJECT_ROOT / "eval/splits/dev500.json"))
        return list(d["smiles"])
    if corpus == "chebi200":
        import csv
        out = []
        with open(PROJECT_ROOT / "benchmarks/chebi_5000.csv", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    ha = int(r.get("heavy_atoms") or 0)
                except ValueError:
                    ha = 0
                if 25 <= ha <= 60:
                    out.append(r["smiles"])
                if len(out) >= 200:
                    break
        return out
    raise SystemExit(f"unknown corpus {corpus!r}")


def _name_one(smiles: str, tier: str):
    kwargs = {}
    if tier == "besteffort":
        kwargs = dict(general_fallback=True, general_fallback_unverified=True,
                      allow_aromatic_general=True)
    signal.setitimer(signal.ITIMER_REAL, TIMEOUT_SEC)
    try:
        return name_compound(smiles, **kwargs)
    except _Timeout:
        return "<TIMEOUT>"
    except Exception as e:  # noqa: BLE001 -- harness must not die mid-corpus
        return f"<ERROR:{type(e).__name__}>"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def cmd_name(args):
    signal.signal(signal.SIGALRM, _alarm)
    smis = _load_corpus(args.corpus)
    out = open(args.out, "w")
    with jvm_slots(1, purpose="perf-ab"):
        for i, smi in enumerate(smis):
            name = _name_one(smi, args.tier)
            out.write(json.dumps({"smiles": smi, "name": name}) + "\n")
            out.flush()
            if (i + 1) % 50 == 0:
                print(f"[name] {i+1}/{len(smis)}", file=sys.stderr, flush=True)
    out.close()
    print(f"[name] DONE {len(smis)} -> {args.out}", file=sys.stderr)


def cmd_diff(args):
    a = {json.loads(l)["smiles"]: json.loads(l)["name"] for l in open(args.a)}
    b = {json.loads(l)["smiles"]: json.loads(l)["name"] for l in open(args.b)}
    keys = sorted(set(a) | set(b))
    diffs = [(k, a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)]
    print(f"[diff] {len(keys)} molecules, {len(diffs)} DIFFERENCES")
    for k, av, bv in diffs[:40]:
        print(f"  DIFF {k[:50]}\n    A={av!r}\n    B={bv!r}")
    sys.exit(1 if diffs else 0)


def cmd_time(args):
    signal.signal(signal.SIGALRM, _alarm)
    smis = _load_corpus(args.corpus)
    marginals = []
    with jvm_slots(1, purpose="perf-ab"):
        for run in range(args.runs):
            t0 = None
            first = None
            start = time.time()
            for i, smi in enumerate(smis):
                ts = time.time()
                _name_one(smi, args.tier)
                dt = time.time() - ts
                if i == 0:
                    first = dt  # one-time cost lives here
            total = time.time() - start
            marginal = (total - first) / max(1, len(smis) - 1)
            marginals.append(marginal)
            print(f"[time] run {run+1}: total={total:.1f}s first={first:.2f}s "
                  f"marginal={marginal*1000:.1f}ms/mol", file=sys.stderr)
    import statistics as st
    print(f"[time] median marginal = {st.median(marginals)*1000:.1f} ms/mol "
          f"over {len(smis)} mol, {args.runs} runs")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("name"); p.add_argument("--corpus", required=True)
    p.add_argument("--tier", default="pin"); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_name)
    p = sub.add_parser("diff"); p.add_argument("--a", required=True)
    p.add_argument("--b", required=True); p.set_defaults(fn=cmd_diff)
    p = sub.add_parser("time"); p.add_argument("--corpus", required=True)
    p.add_argument("--tier", default="besteffort"); p.add_argument("--runs", type=int, default=3)
    p.set_defaults(fn=cmd_time)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
