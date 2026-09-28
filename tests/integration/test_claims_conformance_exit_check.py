"""Claims conformance part 2 (2026-09-27): every public naming entry point returns
only a name whose own round trip passed.

The paper's claim (i): "Apart from the list names above, every shown name passed the
engine's round-trip check", and (d): a name OPSIN rejects is not emitted at
best-effort (exact-match list names excepted; the ten grammar carve-outs are
default-tier only). ba4c1a58c put that check in ``name_tiered`` only. Plain
``Orthonym(...).name``, ``name_compound`` and ``name_with_confidence`` could still
return a name that no round trip had passed: with OPSIN unavailable (a timeout on a
loaded host, the jar present) the trivial last resort (``_apply_trivial_fallback``)
returned 'benzol' for benzene at the default AND best-effort tiers, ungated, and an
offer could win on one of ``_offer_rt_ok``'s advisory fail-open branches.
``Orthonym._exit_round_trip_check`` now runs at every exit.

Gap 2: ``rules.adducts._name_component`` accepted a component with a
lowercase r/s descriptor ('(1R,3r,5S)-tropan-3-yl 1H-indole-3-carboxylate'), which
OPSIN 2.9.0 cannot read, so tropisetron hydrochloride shipped an unreadable salt name
at the default tier. It now refuses a component OPSIN rejects unless it is an
exact-match list name, and the salt abstains again, as in the code the paper
measured.
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
from orthonym.errors import is_failure_name
from orthonym.jvm_flags import java_cmd
from orthonym.metrics import provenance as pv
from tests.support.jars import jar_or_skip

pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]

_ROOT = Path(__file__).resolve().parents[2]

TROPISETRON = "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2"
TROPISETRON_HCL = TROPISETRON + ".Cl"
TROPISETRON_RS_NAME = "(1R,3r,5S)-tropan-3-yl 1H-indole-3-carboxylate"
GERMACRANE = "CC(C)[C@@H]1CC[C@H](C)CCC[C@H](C)CC1"


def _jar_full_key(name):
    """InChIKey of the structure the OPSIN jar reads for ``name``; '' if none."""
    p = subprocess.run(java_cmd() + ["-jar", jar_or_skip(), "-o", "smi"],
                       input=name + "\n", capture_output=True, text=True)
    out = p.stdout.strip().splitlines()
    mol = Chem.MolFromSmiles(out[0]) if out and out[0] else None
    return inchi.MolToInchiKey(mol) if mol is not None else ""


def _input_key(smiles):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


# --------------------------------------------------------------- gap 1, end to end

# A fresh process with every OPSIN route stubbed 'unavailable' while the jar is
# present (the state an OPSIN timeout leaves). Fresh, so no parse cached by an
# earlier test can answer for the stub.
_STUBBED = r'''
import json, subprocess, sys, logging
logging.disable(logging.WARNING)
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
stub = sys.argv[1] == "stub"
if stub:
    _real_run = subprocess.run
    def _run(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and any("opsin" in str(c).lower() for c in cmd):
            raise subprocess.TimeoutExpired(cmd, 1)
        return _real_run(cmd, *a, **k)
    subprocess.run = _run
import orthonym
from orthonym import jvm_bridge, namer as N
from orthonym.validation import opsin_roundtrip, opsin_server
from orthonym.assembly.retained_substitution import OpsinOracle
if stub:
    jvm_bridge.opsin_stdout = lambda *a, **k: (None, False)
    jvm_bridge.opsin_extended_smiles = lambda *a, **k: (None, False)
    opsin_roundtrip.opsin_extended_smiles = lambda *a, **k: (None, False)
    opsin_server.get_persistent_opsin = lambda *a, **k: None
    OpsinOracle._invoke_opsin = lambda self, n: (None, False)
    OpsinOracle._one_shot = lambda self, t: (None, False)
from orthonym.cli import _emit_tier_flags
out = {}
for smi in sys.argv[2:]:
    for tier in ("pin", "best-effort"):
        flags = {} if tier == "pin" else _emit_tier_flags(tier)
        out[f"{smi}|{tier}|name"] = N.Orthonym(style="pin", **flags).name(smi)
        out[f"{smi}|{tier}|name_compound"] = N.name_compound(
            smi, **{k: v for k, v in flags.items() if k != "full_coverage"})
        out[f"{smi}|{tier}|name_with_confidence"] = N.Orthonym(
            style="pin", **flags).name_with_confidence(smi)["name"]
        out[f"{smi}|{tier}|name_tiered"] = N.Orthonym(
            style="pin", **flags).name_tiered(smi)["name"]
print("OUT" + json.dumps(out))
'''


def _run_fresh(mode, *smiles):
    jar_or_skip()
    env = {k: v for k, v in os.environ.items()
           if k not in ("ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE", "ORTHONYM_ALLOW_REDUCED")}
    env["PYTHONPATH"] = str(_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run([sys.executable, "-c", _STUBBED, mode, *smiles],
                       capture_output=True, text=True, timeout=900, env=env, cwd=_ROOT)
    for line in p.stdout.splitlines():
        if line.startswith("OUT"):
            return json.loads(line[3:])
    raise AssertionError(p.stderr[-2000:])


@pytest.mark.slow
def test_unavailable_opsin_returns_no_unverified_name_from_any_entry_point():
    """With OPSIN unavailable no entry point returns a name for benzene (the trivial
    last resort 'benzol' used to ship at both tiers), while the documented default-
    tier grammar carve-out 'methane-SO-thioperoxol' still ships at the default tier
    only. Control: with OPSIN reachable the same process code names benzene."""
    stubbed = _run_fresh("stub", "c1ccccc1", "CSO")
    for key, name in stubbed.items():
        smi, tier, entry = key.split("|")
        if smi == "CSO" and tier == "pin":
            assert name == "methane-SO-thioperoxol", (key, name)
            continue
        assert name is None or is_failure_name(name), (key, name)
    control = _run_fresh("real", "c1ccccc1")
    assert set(control.values()) == {"benzene"}, control


# ------------------------------------------------------------ gap 1, the check itself

def _fresh_check(namer, name, smiles):
    pv.clear_provenance()
    return namer._exit_round_trip_check(name, smiles)


def _best_effort():
    return Orthonym(style="pin", **_emit_tier_flags("best-effort"))


def test_exit_check_general_tier_full_key():
    """A general-tier name no round trip covered is checked at the exit: the right
    name passes (labelled full-key verified), a wrong one is withdrawn."""
    be = _best_effort()
    assert _fresh_check(be, "ethanol", "CCO") == "ethanol"
    assert pv.get_provenance()["gate_outcome"] == pv.GATE_OUTCOME_FULL_KEY_VERIFIED
    assert is_failure_name(_fresh_check(be, "propan-1-ol", "CCO"))
    assert pv.get_provenance()["gate_outcome"] == pv.GATE_OUTCOME_SUPPRESSED


def test_exit_check_default_tier_gates_an_ungated_name():
    """At the default tier a string the gate never judged goes through the gate."""
    pin = Orthonym(style="pin")
    assert _fresh_check(pin, "ethanol", "CCO") == "ethanol"
    assert is_failure_name(_fresh_check(pin, "propan-1-ol", "CCO"))


def test_exit_check_fails_closed_on_unavailable_and_inconclusive(monkeypatch):
    """The fail-open branches reaching a caller are closed: an 'unavailable' round
    trip at a general tier, and an 'inconclusive' recorded for the string at
    the default tier, both withdraw the name."""
    import orthonym.namer as namer_mod
    monkeypatch.setattr(namer_mod, "_shipped_name_round_trip",
                        lambda name, smiles: "unavailable")
    assert is_failure_name(_fresh_check(_best_effort(), "ethanol", "CCO"))
    pv.clear_provenance()
    pv.record_gate_outcome(pv.GATE_OUTCOME_SELF01_INCONCLUSIVE, "ethanol")
    assert is_failure_name(Orthonym(style="pin")._exit_round_trip_check("ethanol", "CCO"))


def test_exit_check_keeps_list_names_and_default_tier_carveouts(monkeypatch):
    """Exact-match list names pass at every tier without a round trip; a recorded
    grammar carve-out passes at the default tier (its documented behaviour)."""
    import orthonym.namer as namer_mod
    monkeypatch.setattr(namer_mod, "_shipped_name_round_trip",
                        lambda name, smiles: "failed")
    assert _fresh_check(_best_effort(), "germacrane", GERMACRANE) == "germacrane"
    pv.clear_provenance()
    pv.record_gate_outcome(pv.carveout_outcome("thioperoxol"), "methane-SO-thioperoxol")
    assert Orthonym(style="pin")._exit_round_trip_check(
        "methane-SO-thioperoxol", "CSO") == "methane-SO-thioperoxol"


def test_exit_check_reduced_mode(monkeypatch):
    """No jar (the opt-in reduced mode): best-effort fails closed, as its gate does;
    the default tier keeps its documented unverified behaviour."""
    import orthonym.namer as namer_mod
    monkeypatch.setattr(namer_mod, "_validity_gate_jar_present", lambda: False)
    assert is_failure_name(_fresh_check(_best_effort(), "ethanol", "CCO"))
    assert _fresh_check(Orthonym(style="pin"), "ethanol", "CCO") == "ethanol"


# ------------------------------------------------------------------------ gap 2

def test_rs_component_is_refused_and_the_salt_name_is_readable_or_absent():
    """The r/s component is refused (OPSIN reads no r/s: the jar returns no
    structure for it), so no salt name built on it can ship at the default tier;
    whatever ships for the hydrochloride must round-trip in the jar."""
    from orthonym.namer import name_compound
    from orthonym.rules.adducts import _name_component
    assert _jar_full_key(TROPISETRON_RS_NAME) == ""
    assert _name_component(TROPISETRON, "pin") is None
    name = name_compound(TROPISETRON_HCL)
    assert "3r" not in name, name
    assert is_failure_name(name) or _jar_full_key(name) == _input_key(TROPISETRON_HCL), name


def test_component_list_name_exception():
    """An exact-match list name OPSIN rejects is still accepted as a component."""
    from orthonym.rules.adducts import _name_component
    assert _jar_full_key("germacrane") == ""
    assert _name_component(GERMACRANE, "pin") == "germacrane"
