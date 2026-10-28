"""GitHub issue 121: two polyphenylene dendrimers that ran for more than ten minutes, and a ``--time-limit`` that did not
stop them.

The rows are C222H150 (37 benzene rings, 36 bonds between rings) and C258H174 (43 rings, 42 bonds). The naming sent
ring-only fragments of the molecule to the cage analysis of ``rules.vonbaeyer_universal``, which starts by matching the
fragment onto its own canonical re-parse (``Mol.GetSubstructMatch``): a VF2 search that grows with the symmetry of the
branches, in one RDKit call with the GIL released, in which the alarm of ``--time-limit`` (``wallclock``) cannot be handled
(both rows ran to the 901 s of an outer ``timeout 900`` under ``--time-limit 850``). Rings joined by bonds are an assembly
, the Blue Book), not a ring system, and are no longer sent there (``ring_cluster.is_one_ring_cluster``); the
assembly is named by ``rules.branched_assembly``,:15715).
"""
import json
import subprocess
import sys
import time

import pytest
from rdkit import Chem

from orthonym import cli_time_limit

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# the two rows of the issue: C222H150 and C258H174
ROW1 = ('C1(=CC=CC=C1)C1=C(C=C(C(=C1C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=C(C=C1)C1=C(C(=C(C(=C1C1=CC=C(C=C1)C1=C(C(=C(C(=C1)'
        'C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=C(C=C1)C1=C(C(=C(C(=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=C'
        'C=C1)C1=CC=C(C=C1)C1=C(C(=C(C(=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=C(C=C1)C1=C(C(=C(C(=C1)C1=CC=CC='
        'C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=C(C=C1)C1=C(C(=C(C(=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1')
ROW2 = ('C1(=CC=CC=C1)C1=C(C=C(C(=C1C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=C(C=C1)C1=CC=C(C=C1)C1=C(C(=C(C(=C1C1=CC=C(C=C1)C1='
        'CC=C(C=C1)C1=C(C(=C(C(=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=C(C=C1)C1=CC=C(C=C1)C1=C(C(=C(C(=C1)C1=C'
        'C=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=C(C=C1)C1=CC=C(C=C1)C1=C(C(=C(C(=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C'
        '1)C1=CC=CC=C1)C1=CC=C(C=C1)C1=CC=C(C=C1)C1=C(C(=C(C(=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=C(C=C1)C1='
        'CC=C(C=C1)C1=C(C(=C(C(=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1')
ROW1_NAME = ("3''''',4',4''''',5',5''''',6'-hexaphenyl-3''',4''',5''',6'''-tetrakis[4-(2,3,4,5-tetraphenylphenyl)phenyl]-"
             "1,1':2',1'':4'',1''':2''',1'''':4'''',1''''':2''''',1''''''-septiphenyl")


ROW2_NAME = ("3''''''',4',4''''''',5',5''''''',6'-hexaphenyl-3'''',4'''',5'''',6''''-tetrakis{4-[4-(2,3,4,5-tetraphenylphenyl)phenyl]phenyl}-"
             "1,1':2',1'':4'',1''':4''',1'''':2'''',1''''':4''''',1'''''':4'''''',1''''''':2''''''',1''''''''-noviphenyl")


# ``cli.main`` of one structure arms the net of ``cli_time_limit.on_hard_stop``, which ends the PROCESS when it fires:
# the runs below are runs of a process of their own, never of the test process (a worker of xdist would be gone).
RUN = """
import contextlib, io, json, sys, time
from orthonym import cli, cli_time_limit
from orthonym.tiers import default_tier

cli_time_limit.prepare(1.0, "pin", default_tier("pin"))     # the start-up of the process is not the run
out = io.StringIO()
start = time.monotonic()
with contextlib.redirect_stdout(out):
    status = cli.main(json.loads(sys.argv[1]))
print(json.dumps({"seconds": time.monotonic() - start, "status": status, "out": out.getvalue()}))
"""


def _row(smiles, *flags):
    """``orthonym SMILES --provenance *flags`` in a process of its own: (wall seconds of the run without the start-up of the
    process, the provenance row, the status)."""
    args = json.dumps([smiles, "--provenance", *flags])
    done = subprocess.run([sys.executable, "-c", RUN, args], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    run = json.loads(done.stdout.strip().splitlines()[-1])
    return run["seconds"], json.loads(run["out"]), run["status"]


def _key(smiles):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def test_the_cage_analysis_is_never_asked_about_rings_joined_by_bonds(monkeypatch):
    """The call that did not return: ``analyze_cage_universal`` on a fragment of rings that share no atom."""
    from orthonym.rules import vonbaeyer_universal
    from orthonym.rules.ring_cluster import is_one_ring_cluster
    real, asked = vonbaeyer_universal.analyze_cage_universal, []

    def spy(mol, cage_atoms=None, *args, **kwargs):
        asked.append(mol is not None and cage_atoms is not None and is_one_ring_cluster(mol, cage_atoms))
        return real(mol, cage_atoms, *args, **kwargs)

    monkeypatch.setattr(vonbaeyer_universal, "analyze_cage_universal", spy)
    from orthonym.cli import _tiered_namer
    from orthonym.tiers import default_tier
    name = _tiered_namer("pin", default_tier("pin")).name_tiered(ROW1)["name"]
    assert name
    assert all(asked), "the cage analysis was asked about rings that are joined by bonds"


def test_row_1_is_named_in_seconds_with_a_full_read_back():
    """C222H150: 37 rings. The chain of the tree is a septiphenyl (its longest path, ``****`` criterion 1), the
    four 4-(2,3,4,5-tetraphenylphenyl)phenyl arms and six phenyl groups are its substituents. Above six rings the PIN is a
    phane name (``****``,:15705), which OPSIN 2.9.0 does not read: the name is of general nomenclature."""
    seconds, row, status = _row(ROW1)
    assert status == 0
    assert (row["name"], row["tier"], row["producer"], row["opsin"]) == (
        ROW1_NAME, "systematic_verified", "inner:ring_assembly", "verified")
    assert seconds < 60, f"{seconds:.0f} s"          # the limit of the harness; measured 4 s
    from orthonym.validation.opsin_roundtrip import opsin_parse
    assert _key(opsin_parse(row["name"])) == _key(ROW1)


def test_row_2_is_named_in_seconds_with_a_full_read_back():
    """C258H174: 43 rings, a chain of nine (``noviphenyl``, ``ASSEMBLY_MULTIPLIERS``): the same four arms, one ring longer."""
    seconds, row, status = _row(ROW2)
    assert status == 0
    assert (row["name"], row["tier"], row["producer"], row["opsin"]) == (
        ROW2_NAME, "systematic_verified", "inner:ring_assembly", "verified")
    assert seconds < 60, f"{seconds:.0f} s"          # measured 6 s
    from orthonym.validation.opsin_roundtrip import opsin_parse
    assert _key(opsin_parse(row["name"])) == _key(ROW2)


def test_row_1_returns_within_the_limit_and_its_slack():
    """``--time-limit 20``: the row is named (about 4 s) or, on a machine that is slower than that by a factor five,
    declined with the outcome of a timeout; either way the run returns within the limit and its slack."""
    seconds, row, status = _row(ROW1, "--time-limit", "20")
    assert status == 0
    assert row["name"] or row["limit_code"] == "TIME_LIMIT"
    assert seconds < 20 + cli_time_limit.HARD_STOP_GRACE_S + 5


def test_row_1_has_the_time_limit_outcome_at_a_limit_it_cannot_meet():
    """``--time-limit 0.5``: the naming of the row takes seconds; the limit ends it with the documented outcome (a row
    with the tier ``abstain`` and the limit code ``TIME_LIMIT``) within its slack. No call of the naming of this row
    outlasts the alarm any more."""
    seconds, row, status = _row(ROW1, "--time-limit", "0.5")
    assert status == 0
    assert (row["name"], row["tier"], row["limit_code"], row["formula"]) == (None, "abstain", "TIME_LIMIT", "C222H150")
    assert seconds < 0.5 + 5


DRIVER = """
import signal, sys, time
from orthonym import cli, cli_time_limit
from orthonym.namer import Orthonym

STUCK = {stuck!r}
cli_time_limit.HARD_STOP_GRACE_S = 1.0
real = Orthonym.name_tiered
{patch}

def name_tiered(self, smiles):
    if smiles == STUCK:
        # a call that holds no Python and cannot be interrupted: the alarm of the limit is pending and cannot be handled
        signal.pthread_sigmask(signal.SIG_BLOCK, {{signal.SIGALRM}})
        time.sleep(120)
    return real(self, smiles)

Orthonym.name_tiered = name_tiered
sys.exit(cli.main([STUCK, "--provenance", "--time-limit", "1"]))
"""

STUCK = "CCCCCCCCCCCCCCCCCC"


def _stuck_run(patch=""):
    start = time.monotonic()
    done = subprocess.run([sys.executable, "-c", DRIVER.format(stuck=STUCK, patch=patch)], capture_output=True, text=True,
                          timeout=110)
    return time.monotonic() - start, done


def test_a_call_that_the_alarm_cannot_interrupt_ends_a_run_of_one_structure():
    """The net of ``cli_time_limit.on_hard_stop``: the call below ignores the alarm for 120 s; the run ends
    ``HARD_STOP_GRACE_S`` after the limit with the row of a timeout and the status of any other."""
    seconds, done = _stuck_run()
    assert seconds < 60
    assert done.returncode == 0, done.stderr[-2000:]
    row = json.loads(done.stdout.strip().splitlines()[-1])
    assert (row["name"], row["tier"], row["limit_code"], row["formula"]) == (None, "abstain", "TIME_LIMIT", "C18H38")


@pytest.mark.parametrize("flags,status", [([], 0), (["--on-failure", "raise"], 1)])
def test_the_end_of_the_run_is_reached_when_its_output_fails(flags, status):
    """A closed standard output (``BrokenPipeError``) raises in the callback of the net: the process must still end, with the
    status that a timeout of the run has (1 for ``--on-failure raise``, else 0), not wait for the call it was stopping."""
    patch = (
        "def broken(parsed):\n    raise BrokenPipeError('closed')\ncli._single_timed_out = broken\n"
        + "".join(f"sys.argv.append({f!r})\n" for f in flags))
    source = DRIVER.format(stuck=STUCK, patch=patch).replace(
        '[STUCK, "--provenance", "--time-limit", "1"]', '[STUCK, "--time-limit", "1", *sys.argv[1:]]')
    start = time.monotonic()
    done = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=110)
    assert time.monotonic() - start < 60
    assert done.returncode == status, done.stderr[-2000:]
