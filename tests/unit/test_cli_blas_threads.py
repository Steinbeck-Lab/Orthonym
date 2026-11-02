"""The CLI starts one BLAS thread, not one per CPU.

numpy's bundled OpenBLAS starts one thread per CPU when it loads, and those threads spin: ``orthonym CCO`` used 10.8 s of
user CPU for 2.2 s of wall time (3.2 s with one thread). ``import orthonym`` therefore does not load numpy, and
``cli.main`` sets the thread variables (unless the user did) before anything can load it. A library user's environment
is left alone. Each check runs in a fresh child process, so ``sys.modules`` and the environment are its own.
"""
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _child(code: str, env: dict = None) -> str:
    base = {k: v for k, v in os.environ.items() if k not in _VARS}
    base.update(env or {})
    out = subprocess.run([sys.executable, "-c", code], env=base, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


_AFTER_CLI_VERSION_EXIT = ("import os, sys\nfrom orthonym import cli\ntry:\n    cli.main(['--version'])\n"
                           "except SystemExit:\n    pass\n"
                           "print('numpy' in sys.modules, os.environ.get('OPENBLAS_NUM_THREADS'))")


def test_import_orthonym_does_not_import_numpy():
    assert _child("import sys, orthonym; print('numpy' in sys.modules)") == "False"


def test_cli_sets_blas_var_before_numpy_loads():
    assert _child(_AFTER_CLI_VERSION_EXIT) == "False 1"


def test_cli_keeps_user_value():
    assert _child(_AFTER_CLI_VERSION_EXIT, env={"OPENBLAS_NUM_THREADS": "4"}) == "False 4"


def test_import_orthonym_leaves_env_alone():
    assert _child("import os, orthonym; print(os.environ.get('OPENBLAS_NUM_THREADS'))") == "None"
