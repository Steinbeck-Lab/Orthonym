"""The in-process OPSIN answer equals the OPSIN command line, line by line.

The PR run answers the test code's ``java -jar opsin -osmi`` calls through
``orthonym.jvm_bridge._opsin_stdout_uncached`` (tests/support/opsin_cli_inproc.py).
That is only sound if the two give the same bytes, so this test runs ONE real
``java -jar`` call per mode over 40 names (valid names, names OPSIN rejects, names that
only parse with radicals allowed) and compares every output line with the in-process
answer, and with what ``subprocess.run`` returns when the shim is active.
"""
import subprocess

from orthonym.jvm_bridge import _opsin_stdout_uncached
from orthonym.jvm_flags import java_cmd
from tests.support import opsin_cli_inproc
from tests.support.jars import jar_or_skip

NAMES = [
    # parses
    "benzene", "toluene", "pyridine", "quinoline", "2-methylnaphthalene",
    "4H-furo[3,2-b]pyrrole", "1H-1-benzazepine", "pentaphene", "hexahelicene",
    "N'-methylbenzohydrazide", "sodium methoxide", "ethanol", "acetic acid",
    "(2S)-2-aminopropanoic acid", "(1R,2S)-2-methylcyclohexan-1-ol",
    "bicyclo[2.2.1]heptane", "spiro[4.5]decane", "2,3-dihydro-1H-indole",
    "methyl benzoate", "N,N-dimethylformamide", "propan-2-yl acetate",
    "cyclohexane-1,2-diol", "4-nitrophenol", "adamantane", "cyclohexene",
    # rejected by OPSIN
    "notarealname999zzz", "xyz!!!bad", "benzene-7-ol", "1,1,1,1-tetramethylmethane",
    "ethyl(", "cyclohexylidene-propane-9-amine",
    # only with radicals allowed
    "methyl", "ethyl", "propan-2-yl", "phenyl", "butan-2-yl", "cyclohexyl",
    "ethan-1-yl-2-ylidene", "benzyl",
    # a tab: the command line reads up to the first one
    "benzene\tignored",
]


def _compare_one_mode(jar, radicals):
    argv = java_cmd("-jar", jar, *(["-r"] if radicals else []), "-osmi")
    text = "\n".join(NAMES) + "\n"

    real = opsin_cli_inproc.REAL_RUN(argv, input=text, capture_output=True, text=True, timeout=180)
    assert real.returncode == 0, real.stderr[:500]
    real_lines = real.stdout.split("\n")[:-1]
    assert len(real_lines) == len(NAMES), (len(real_lines), real.stdout[:300])

    in_process = []
    for name in NAMES:
        stdout, served = _opsin_stdout_uncached(name, radicals, jar)
        assert served, f"the in-process path did not answer {name!r} (radicals={radicals})"
        assert stdout.endswith("\n") and stdout.count("\n") == 1, repr(stdout)
        in_process.append(stdout[:-1])
    assert in_process == real_lines, f"radicals={radicals}"

    # the shim's own path: what subprocess.run returns for this call from test code
    before = opsin_cli_inproc.SERVED
    shimmed = subprocess.run(argv, input=text, capture_output=True, text=True, timeout=180)
    assert shimmed.returncode == 0
    assert shimmed.stdout == real.stdout, f"radicals={radicals}"
    served_by_shim = 0 if opsin_cli_inproc.forced_real() else 1
    assert opsin_cli_inproc.SERVED == before + served_by_shim, "unexpected shim use"


def test_in_process_opsin_equals_cli():
    assert len(NAMES) == 40
    jar = jar_or_skip()
    for radicals in (False, True):
        _compare_one_mode(jar, radicals)

    # another output mode and another program are never answered in process
    before = opsin_cli_inproc.SERVED
    inchikey = subprocess.run(java_cmd("-jar", jar, "-ostdinchikey"), input="benzene\n",
                              capture_output=True, text=True, timeout=180)
    assert inchikey.returncode == 0
    assert inchikey.stdout.strip() == "UHOVQNZJYSORNB-UHFFFAOYSA-N", inchikey.stdout
    assert subprocess.run(["echo", "x"], capture_output=True, text=True).stdout == "x\n"
    assert opsin_cli_inproc.SERVED == before
