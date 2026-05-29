"""Phase 168 Plan-02: triviality-controller feature-flag unit tests.

Mirrors the Phase 162 feature-flag env-var-override pattern. Verifies the Stage A SACRED
default-OFF invariant at the ctor + pool + CLI surfaces, and the env-var override.

The env-var test runs in a SUBPROCESS (not importlib.reload) so it cannot pollute the parent
test process's already-imported ``orthonym.namer`` module.

Source: 168-CONTEXT.md D-07/D-08; 168-PATTERNS.md feature-flag analog.
"""

import os
import subprocess
import sys

import pytest

from orthonym.namer import Orthonym
from orthonym.assembly.candidate_pool import CandidatePool


class TestDefaultOff:
    """The controller is OFF by default (Stage A SACRED byte-identical canary invariant)."""

    @pytest.mark.unit
    def test_ctor_default_off(self):
        assert Orthonym()._enable_triviality_controller is False

    @pytest.mark.unit
    def test_ctor_default_no_oracle(self):
        # Flag OFF => no OpsinOracle instantiated (zero cost, byte-identical Stage A).
        assert Orthonym()._triv_oracle is None

    @pytest.mark.unit
    def test_ctor_on_instantiates_oracle(self):
        # WARNING #9: flag ON MUST instantiate the OpsinOracle for the TRIV-03 T2 RT-safety gate.
        assert Orthonym(enable_triviality_controller=True)._triv_oracle is not None


class TestEnvVarOverride:
    """ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER=1/true/yes/on flips the module-level default."""

    @pytest.mark.unit
    def test_env_var_flips_default_triv(self):
        code = (
            "from orthonym.namer import _DEFAULT_TRIV; "
            "assert _DEFAULT_TRIV is True, 'env var should flip _DEFAULT_TRIV'; "
            "print('OK')"
        )
        env = {**os.environ, "ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER": "1"}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert r.returncode == 0, f"stderr={r.stderr}"
        assert "OK" in r.stdout

    @pytest.mark.unit
    def test_env_var_absent_default_off(self):
        code = (
            "from orthonym.namer import _DEFAULT_TRIV; "
            "assert _DEFAULT_TRIV is False; print('OK')"
        )
        env = {k: v for k, v in os.environ.items()
               if k != "ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER"}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert r.returncode == 0, f"stderr={r.stderr}"


class TestStageABytewiseInvariant:
    """The pool defaults the flag OFF (BLOCKER #9) and the CLI exposes the flag."""

    @pytest.mark.unit
    def test_flag_off_pool_defaults(self):
        p = CandidatePool()
        assert p._enable_triviality_controller is False
        assert p._triv_oracle is None

    @pytest.mark.unit
    def test_cli_flag_exposed(self):
        r = subprocess.run(
            [sys.executable, "-m", "orthonym.cli", "--help"],
            capture_output=True, text=True,
        )
        assert "--enable-triviality-controller" in r.stdout
