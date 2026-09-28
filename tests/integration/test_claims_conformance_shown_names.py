"""Claims conformance (2026-09-27): at the best-effort tier every shown name passed
the engine's round trip, apart from the exact-match list names.

The paper's claims (a), (b), (d) and (i): OPSIN reads the name and the FULL InChIKey
of what it read equals the input's; a name that fails is withdrawn and the row
abstains with its limit code; a name OPSIN rejects is never emitted at best-effort,
the only exceptions being the exact-match list names (NAME_EXACT natural-product
parents, coordination retained names).

At 5b30f0248 twenty names on a dev split and a holdout split shipped with the provenance
``opsin='unverified'``. Nineteen of them round-trip: they were offers that
``_select_rt_passing_offer_name`` picked, by its full-key check, after the gate had
suppressed another string, so the gate outcome recorded for the shipped string was
'bypassed' / 'not_run', or a general-engine name whose own check was recorded as
'unverified'. ``name_tiered`` now checks such a name itself and labels it verified.
The twentieth, the myo-inositol pentakisphosphate below, is a name with lowercase
r/s descriptors, which OPSIN 2.9.0 cannot read: the last resort that had adopted it
 is gone, so the row abstains again, as in the code the paper measured
.

Every row is also checked against the OPSIN jar directly (a Java subprocess, not
the engine's own oracle), so the test does not rest on the code it tests.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_flags import java_cmd
from orthonym.namer import _is_exact_match_list_name
from tests.support.jars import jar_or_skip

pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]

_SHOWN = ("pin_verified", "pin_unverified", "systematic_verified", "best_effort")

# One or more rows per path that shipped opsin='unverified' at 5b30f0248, each
# round-tripping (full InChIKey) before and after the fix.
ROUND_TRIPPING_ROWS = [
    # a t4_floor offer that won after the gate suppressed the primary (milestone1500)
    "C[N@@+]1([O-])CCC[C@H]1c1cccnc1",
    "COC1=NC2C(=NC=NC2=N1)N",
    # a pin_path offer adopted after the gate suppressed another string (a dev split)
    "CCCCCCC[C@H]1OC(=O)CC(=O)[C@H](Cc2ccccc2)N(C)C(=O)[C@H](C(C)C)OC(=O)[C@H]1C",
    "C[C@@H]1CCC2C(C)(C)[C@H](O)CC[C@]2(C)[C@@]12Cc1c(O)cc3c(c1O2)CNC3=O",
    # a pin_path offer for which no gate outcome was recorded at all (a dev split)
    "CC1=C2O[C@@]3(C[C@]2(C)C(C)=C(C(=O)[O-])C1=O)[C@@H](C)CC[C@H]1C(C)(C)OC(=O)CC[C@@]13C",
]

# The inositol pentakisphosphate: '(1s,2R,3S,4s,5R,6S)-2,3,4,5,6-pentakis(phosphono
# oxy)cyclohexan-1-ol' shipped pin_verified at best-effort, but OPSIN 2.9.0 parses
# no lowercase r/s descriptor (nor an uppercase one at a pseudoasymmetric centre,
# nor 'myo-inositol'), so no spelling of it round-trips.
PENTAKIS = ("O=P(O)(O)O[C@H]1[C@@H](OP(=O)(O)O)[C@H](OP(=O)(O)O)[C@@H](O)"
            "[C@H](OP(=O)(O)O)[C@H]1OP(=O)(O)O")
PENTAKIS_NAME = "(1s,2R,3S,4s,5R,6S)-2,3,4,5,6-pentakis(phosphonooxy)cyclohexan-1-ol"


def _be_row(smiles):
    return Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)


def _jar_full_key(name):
    """InChIKey of the structure the OPSIN jar reads for ``name``; '' if none."""
    p = subprocess.run(java_cmd() + ["-jar", jar_or_skip(), "-o", "smi"],
                       input=name + "\n", capture_output=True, text=True)
    out = p.stdout.strip().splitlines()
    mol = Chem.MolFromSmiles(out[0]) if out and out[0] else None
    return inchi.MolToInchiKey(mol) if mol is not None else ""


def _input_key(smiles):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("smiles", ROUND_TRIPPING_ROWS)
def test_round_tripping_name_is_kept_and_labelled_verified(smiles):
    row = _be_row(smiles)
    assert row["tier"] in _SHOWN, row
    assert row["opsin"] == "verified", row
    assert row["verified"] == "opsin", row
    assert row["gates_passed"], row
    assert _jar_full_key(row["name"]) == _input_key(smiles)


def test_pentakisphosphate_r_s_name_is_not_shown_at_best_effort():
    assert _jar_full_key(PENTAKIS_NAME) == ""  # the reason: OPSIN cannot read it
    row = _be_row(PENTAKIS)
    assert row["name"] is None, row
    assert row["tier"] == "abstain" and row["limit_code"], row
    assert row["gate_outcome"] == "suppressed", row
    # the plain name at best-effort does not ship it either
    name = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name(PENTAKIS)
    assert name != PENTAKIS_NAME


_SCAN = r'''
import json, sys, multiprocessing as mp
def work(s):
    import logging
    logging.disable(logging.WARNING)
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    from orthonym.namer import _is_exact_match_list_name
    r = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(s)
    if r["tier"] == "abstain":
        return None if r["name"] is None else [s, r["name"], r["tier"], "abstain row carries a name"]
    if r["opsin"] != "verified" and not _is_exact_match_list_name(s, r["name"]):
        return [s, r["name"], r["tier"], r["opsin"]]
    return None
if __name__ == "__main__":
    split = json.load(open(sys.argv[1]))
    smiles = split["smiles"] if isinstance(split, dict) else split
    smiles = [x["smiles"] if isinstance(x, dict) else x for x in smiles]
    with mp.Pool(int(sys.argv[2])) as pool:
        bad = [b for b in pool.imap_unordered(work, smiles, chunksize=1) if b]
    print(json.dumps({"n": len(smiles), "bad": bad}))
'''


@pytest.mark.slow
def test_dev500_scan_ships_no_unverified_name(tmp_path):
    """The whole a dev split split at best-effort: no shown name without a verified
    round trip unless it is an exact-match list name."""
    repo = Path(__file__).resolve().parents[2]
    split = repo / "eval" / "splits" / "dev500.json"
    if not split.exists():
        pytest.skip("dev500 split not present")
    jar_or_skip()
    script = tmp_path / "scan.py"
    script.write_text(_SCAN)
    src = str(Path(sys.modules["orthonym"].__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE", None)
    workers = str(max(2, min(16, (os.cpu_count() or 4) // 2)))
    p = subprocess.run([sys.executable, str(script), str(split), workers],
                       capture_output=True, text=True, env=env, timeout=3600)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["n"] == 500
    assert out["bad"] == [], out["bad"]
