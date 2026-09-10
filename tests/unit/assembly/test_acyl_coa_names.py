"""End-to-end regression locks for the substituted-purine build.

adenosine (the largest purine-bearing substructure) now names RT-exact — the
load-bearing proof the purine ring-substituent path is closed. acetyl-CoA still
abstains on two NON-purine residual blockers (phosphate-ester protonation +
pantetheine polyamide/giant-parent assembly), a separate future build; it is
pinned as a strict xfail so 0-wrong holds (it abstains, never emits wrong) and
the target is never lost.
"""
import pytest
from orthonym import Orthonym
from orthonym.validation import opsin_roundtrip_check

# This module asserts on real emitted behavior (0-wrong / RT-exact), so it must
# run with Orthonym's OPSIN validity gate ON. tests/conftest.py's autouse
# `_opsin_validity_gate_state` fixture disables the gate by default for every
# test unless opted in via this marker — without it, a candidate the gate
# would normally reject slips through and this suite observes fabricated
# behavior that never reaches production/eval (gate-on by default there).
pytestmark = pytest.mark.opsin_gate

ADENOSINE = "OCC1OC(n2cnc3c(N)ncnc32)C(O)C1O"
ACETYL_COA = "CC(=O)SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O"


def test_adenosine_names_and_round_trips():
    name = Orthonym().name(ADENOSINE)
    assert name == "5-(6-amino-9H-purin-9-yl)-2-(hydroxymethyl)oxolane-3,4-diol", name
    rt = opsin_roundtrip_check(ADENOSINE, name)
    assert rt["passed"], rt


def test_9_methyladenine_names():
    # the canonical substituted-purine target, end-to-end
    assert Orthonym().name("Cn1cnc2c(N)ncnc21") == "9-methyl-9H-purin-6-amine"


def test_acetyl_coa_0wrong_holds():
    # 0-wrong is ABSOLUTE: whatever acetyl-CoA emits, it must never be a WRONG
    # molecule. Today it abstains (residual non-purine blockers); if it ever
    # emits, that name must round-trip. This assertion holds in BOTH states.
    name = Orthonym().name(ACETYL_COA)
    assert name == "unknown organic compound" or opsin_roundtrip_check(ACETYL_COA, name)["passed"], name


@pytest.mark.xfail(strict=True, reason="acetyl-CoA residual NON-purine blockers: "
                   "(1) phosphate-ester protonation naming, (2) pantetheine "
                   "polyamide/giant-parent assembly. Purine path is closed "
                   "(adenosine names RT-exact). Separate future build.")
def test_acetyl_coa_fully_names_TARGET():
    name = Orthonym().name(ACETYL_COA)
    assert name != "unknown organic compound"
    assert opsin_roundtrip_check(ACETYL_COA, name)["passed"], name
