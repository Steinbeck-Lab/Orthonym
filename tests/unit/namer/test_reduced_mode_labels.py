"""The opt-in reduced mode (no OPSIN jar) never labels a name pin_verified (fix
a performance pass, wp7; whole-branch verification panel RISK 2).

With ``ORTHONYM_ALLOW_REDUCED=1`` and no jar, the validity gate cannot consult OPSIN
and records ``unavailable``; the name ships unverified (the historical fail-OPEN for
the PIN tier). name_tiered labelled such names ``pin_verified`` / ``is_pin=True``,
and the radical cation C[NH2+] shipped as the closed-shell 'methanaminium': the
radical producers ship only a round-tripped name (so they declined without a jar)
and the composer's suffix swap, which ignores the radical electron, named the
closed-shell parent's ion instead.

Now: an unverified name is at most ``pin_unverified``; the suffix swap declines for a
radical ion, so C[NH2+] fails closed there. Without the opt-in the entry point still
raises ``JarUnavailable`` (fail closed). Each case runs in a fresh process, since the
jar lookup is fixed at start-up.
"""
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

_PROBE = r"""
import json, sys
from orthonym import Orthonym
out = {}
for smi in sys.argv[1:]:
    r = Orthonym(style='pin').name_tiered(smi)
    out[smi] = {k: r.get(k) for k in ('name', 'tier', 'is_pin', 'opsin', 'gate_outcome')}
print(json.dumps(out))
"""

_SMILES = ["CCO", "O=C1OC(=O)C=C1Br", "CN(C)C(=O)Cl", "C[NH2+]", "C[NH2+]CCC(=O)NC"]


def _reduced_run(smiles):
    # Start from an environment with NO engine settings: another test in the same
    # worker may have exported ORTHONYM_OPSIN_JAR (an explicit jar path) or disabled
    # the gate, and either would make this probe run against a jar.
    env = {k: v for k, v in os.environ.items() if not k.startswith("ORTHONYM_")}
    env.update(ORTHONYM_JAR_DIR="/nonexistent", ORTHONYM_NO_DOWNLOAD="1",
               ORTHONYM_ALLOW_REDUCED="1")
    proc = subprocess.run([sys.executable, "-c", _PROBE, *smiles], env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def reduced():
    return _reduced_run(_SMILES)


def test_the_probe_really_ran_without_a_jar(reduced):
    # a perfect result means nothing if OPSIN was secretly consulted
    assert reduced["CCO"]["gate_outcome"] == "unavailable", reduced["CCO"]
    assert reduced["CCO"]["opsin"] == "unverified"


@pytest.mark.parametrize("smiles", _SMILES)
def test_no_unverified_name_is_labelled_pin_verified(reduced, smiles):
    r = reduced[smiles]
    assert r["tier"] != "pin_verified" and not r["is_pin"], r


def test_radical_cation_is_not_named_as_its_closed_shell_parent(reduced):
    assert reduced["C[NH2+]"]["name"] != "methanaminium", reduced["C[NH2+]"]


def test_names_are_unchanged_in_reduced_mode(reduced):
    """Only the label changes; the unverified names themselves still ship."""
    assert reduced["CCO"]["name"] == "ethanol"
    assert reduced["O=C1OC(=O)C=C1Br"]["name"] == "3-bromofuran-2,5-dione"
