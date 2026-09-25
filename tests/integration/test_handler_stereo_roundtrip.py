"""a phase: Handler-level stereo injection - OPSIN round-trip integration tests.

Per /, every test compound is round-tripped through OPSIN
(name -> SMILES -> canonicalize) and compared with the original canonical
SMILES INCLUDING the stereo layer. The four handler classes
(TestHeterocycle, TestBenzene, TestCycloalkane, TestCycloalkene) are
populated from JSON fixtures under tests/data/stereo_handlers/. Every
fixture entry is sourced per (no hand-curated SMILES).

Skipped if Java or OPSIN JAR is unavailable.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import name_compound
from tests.support.jars import jar_or_none
from tests.support.rt_assert import assert_tier_contract

JAVA_AVAILABLE = shutil.which("java") is not None
OPSIN_JAR = jar_or_none()
OPSIN_AVAILABLE = JAVA_AVAILABLE and OPSIN_JAR is not None

pytestmark = [
    pytest.mark.skipif(
        not OPSIN_AVAILABLE,
        reason="Java or OPSIN JAR (2.9.0) not available for round-trip tests",
    ),
    pytest.mark.roundtrip,
]


def opsin_parse(name: str) -> str:
    """Parse IUPAC name to SMILES using OPSIN 2.9.0."""
    try:
        result = subprocess.run(
            ["java", "-jar", OPSIN_JAR, "-osmi"],
            input=name, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


_FIXTURES_DIR = Path(__file__).parent.parent / "data" / "stereo_handlers"


def _load_cases(handler: str):
    """Load fixture cases for *handler* from the JSON file under
    tests/data/stereo_handlers/<handler>/cases.json. Each entry MUST have
    smiles, expected_stereo_prefix_pattern, source_corpus, source_id .
    """
    path = _FIXTURES_DIR / handler / "cases.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    return data.get("compounds", [])


def _assert_handler_roundtrip(case: dict, handler: str):
    """Assert handler stereo correctness in two stages.

     FIX (2026-05-03 — see internal notes + internal notes):
    Previously this function called pytest.xfail before ANY assertion,
    making xfail-marked fixtures execute zero asserts. Now the assertion
    pipeline is split:
      (1) ALWAYS-RUN GATES: name_compound returns non-empty AND
           prefix matches `expected_stereo_prefix_pattern`. These run
          for every fixture, regardless of xfail_reason. They detect
          regressions in the a phase wiring contract independently of
          OPSIN's parsing capability.
      (2) CONDITIONAL OPSIN ROUND-TRIP: name -> OPSIN -> canonical SMILES
          -> compare. Only THIS step is xfail-able when the fixture's
          `xfail_reason` indicates a known OPSIN-side or
          handler-output-format issue tracked outside a phase.
    """
    import re as _re
    smiles = case["smiles"]
    if case.get("source_id") in _TIER_CONTRACT:
        return _assert_tier_contract_case(case, handler)
    name = name_compound(smiles)

    # ---- ALWAYS-RUN GATES (fix) ----
    assert name and name != "unknown", (
        f"name_compound returned empty/unknown for {smiles} "
        f"(handler={handler}, source={case.get('source_corpus')}/{case.get('source_id')})"
    )
    pat = case["expected_stereo_prefix_pattern"]
    assert _re.match(pat, name), (
        f"P-91 prefix mismatch: pattern={pat!r} name={name!r} "
        f"(handler={handler}, source={case.get('source_corpus')}/{case.get('source_id')})"
    )

    # ---- CONDITIONAL OPSIN ROUND-TRIP (still xfail-able) ----
    if case.get("xfail_reason"):
        pytest.xfail(case["xfail_reason"])
    opsin_smiles = opsin_parse(name)
    assert opsin_smiles, (
        f"OPSIN could not parse name {name!r} (handler={handler}, "
        f"source={case.get('source_corpus')}/{case.get('source_id')})"
    )
    canonical_input = Chem.CanonSmiles(smiles)
    canonical_output = Chem.CanonSmiles(opsin_smiles)
    assert canonical_input == canonical_output, (
        f"SMILES round-trip mismatch: input={canonical_input} "
        f"output={canonical_output} name={name!r} (handler={handler})"
    )
    return name


# Cases whose PIN the PIN tier cannot build (the molecule is outside the
# handler's class). For these the tier contract is asserted instead
# (tests/support/rt_assert.py::assert_tier_contract): best-effort is RT-exact
# (full InChIKey, so every stereodescriptor is checked) and the PIN tier ships
# an RT-exact name or fails closed.
# Pre-existing-failures plan, Task 4 (TRIAGE.csv rows 17, 19 and 20; CHEBI:137531
# and p152-carryover-2 are xfails of row 19's class that hid a wrong name).
_TIER_CONTRACT = {
    # A fused TRICYCLIC sesquiterpene, not a cycloalkene. Its PIN is a hydro
    # fusion name: (the Blue Book) "Preferred IUPAC names for
    # the partially saturated and fully saturated compounds are formed by using
    # 'hydro' prefixes" ("decahydronaphthalene (PIN) bicyclo[4.4.0]decane",
    #:24233). The PIN tier has no producer for it and fails closed; before the
    # whole-ring-system guard it shipped
    # '(1E,5Z,7S)-1,5-dimethyl-6-nonylcyclonona-1,5-diene' (OPSIN-unparseable,
    # and a monocycle for a tricycle).
    "CHEBI:177888": "fused tricycle: PIN is a hydro-fusion name (P-54.4.3.1)",
    # A fused BICYCLE (cyclooctadiene ortho-fused to a cyclopentene), same
    # rule. Before the guard the PIN tier shipped the monocycle
    # '(1Z,6S)-4-(2,2-dimethylpropyl)-2,6-dimethylcycloocta-1,4-diene'
    # (a different molecule: the 5-ring opened into a neopentyl), which the
    # fixture's round-trip xfail hid; best-effort now gives the RT-exact
    # '(3Z,7S)-3,7,10,10-tetramethylbicyclo[6.3.0]undeca-1(8),3-diene'.
    "CHEBI:137531": "fused bicycle: PIN is a hydro-fusion name (P-54.4.3.1)",
    # A pentacyclic triterpenoid (ortho-fused) with a cinnamate ester: the PIN
    # is a hydro-fusion name, out of the PIN tier's reach. Before
    # the guard the PIN tier shipped the WRONG '1-[(2E)-3-(4-hydroxy-
    # phenyl)(nonanoyloxy)prop-2-enediolyl]-3-hydroxy-3,4-dimethylcyclohexane-1-
    # carboxylic acid' (a monocycle), which the fixture's round-trip xfail hid.
    "p152-carryover-2": "fused pentacycle: PIN is a hydro-fusion name (P-54.4.3.1)",
    # TRIAGE row 17: an N-(2,1,3-benzoxadiazol-4-yl)aspartyl-alanine. The PIN
    # tier has no producer for its substitutive PIN (an acid chain carrying a
    # [methyl(7-nitro-2,1,3-benzoxadiazol-4-yl)amino] prefix and the
    # (1-carboxyethyl)amino-oxo arm). It used to ship the decomposition glue
    # 'N-({(2S)-2-[methyl(7-nitro2,1,3-benzoxadiazol-4-yl)amino]butanedioyl})
    # (2S)-2-aminopropanoic acid': a '-dioyl' acyl with ONE attachment (the free
    # CH2-COOH lost) floated onto an amino ACID name: the N- float
    # names the nitrogen of an amine suffix). Both joins now decline.
    "CHEBI:139249": "no PIN producer; the poly-acid N-acyl glue declines (P-62.2.2.1)",
    # TRIAGE row 20 (D-abstain): a 10-membered lactone.
    # (the Blue Book) "Preferred IUPAC names for heteromonocyclic rings
    # with no more than ten ring members are Hantzsch-Widman names" (an oxecine),
    # which the PIN tier cannot build, so it fails closed; best-effort gives the
    # RT-exact skeletal-replacement name '(3Z,5R,6E,8S,10R)-5,8-dihydroxy-10-
    # methyl-2-oxo-1-oxacyclodeca-3,6-diene' (not the PIN), E/Z block included.
    "CHEBI:190592": "10-ring lactone: PIN is a Hantzsch-Widman oxecinone (P-52.2.2.2)",
}


def _assert_tier_contract_case(case: dict, handler: str) -> str:
    """The tier contract for a case the PIN tier cannot name. Returns the
    best-effort name, which the caller's own extra gates (the E/Z block of the
    8+ ring class) then check.

    The prefix regex is NOT applied to the best-effort name: it encodes
    the handler's PIN shape (parent stereodescriptors in front), and a
    best-effort name may rightly carry every descriptor inside a substituent.
    The full-InChIKey round trip in `assert_tier_contract` is the stronger
    stereo check: it fails on any missing or wrong descriptor.

    These cases run with the OPSIN validity gate ON (`opsin_gate`, set by
    `_params`): the contract is about what SHIPS. With the gate off (the suite
    default) the raw producers still emit candidates the gate voids.
    """
    _pin, be = assert_tier_contract(case["smiles"])
    return be


def _params(cases):
    """pytest params for a fixture: the named xfails, and `opsin_gate` for the
    tier-contract cases."""
    out = []
    for c in cases:
        sid = c.get("source_id")
        if sid in _COMPLEX_RING_XFAIL:
            out.append(pytest.param(c, marks=pytest.mark.xfail(
                strict=True, reason=_COMPLEX_RING_XFAIL[sid])))
        elif sid in _TIER_CONTRACT:
            out.append(pytest.param(c, marks=pytest.mark.opsin_gate))
        else:
            out.append(c)
    return out


# ----------------------------------------------------------------------
# Heterocycle (commit 3)
# ----------------------------------------------------------------------

_HETEROCYCLE_CASES = _load_cases("heterocycle")


@pytest.mark.parametrize(
    "case",
    _HETEROCYCLE_CASES,
    ids=lambda c: f"{c.get('source_corpus','?')}/{c.get('source_id','?')}",
)
class TestHeterocycle:
    def test_heterocycle_stereo_roundtrip(self, case):
        _assert_handler_roundtrip(case, "heterocycle")


# ----------------------------------------------------------------------
# Benzene (commit 4)
# ----------------------------------------------------------------------

_BENZENE_CASES = _load_cases("benzene")


@pytest.mark.parametrize(
    "case",
    _BENZENE_CASES,
    ids=lambda c: f"{c.get('source_corpus','?')}/{c.get('source_id','?')}",
)
class TestBenzene:
    def test_benzene_stereo_roundtrip(self, case):
        _assert_handler_roundtrip(case, "benzene")


# ----------------------------------------------------------------------
# Cycloalkane (commit 5)
# ----------------------------------------------------------------------

_CYCLOALKANE_CASES = _load_cases("cycloalkane")


@pytest.mark.parametrize(
    "case",
    _CYCLOALKANE_CASES,
    ids=lambda c: f"{c.get('source_corpus','?')}/{c.get('source_id','?')}",
)
class TestCycloalkane:
    def test_cycloalkane_stereo_roundtrip(self, case):
        _assert_handler_roundtrip(case, "cycloalkane")


# ----------------------------------------------------------------------
# Cycloalkene (commit 5)
# ----------------------------------------------------------------------

_CYCLOALKENE_CASES = _load_cases("cycloalkene")


@pytest.mark.parametrize(
    "case",
    _CYCLOALKENE_CASES,
    ids=lambda c: f"{c.get('source_corpus','?')}/{c.get('source_id','?')}",
)
class TestCycloalkene:
    def test_cycloalkene_stereo_roundtrip(self, case):
        _assert_handler_roundtrip(case, "cycloalkene")


# ----------------------------------------------------------------------
# a phase: ComplexRing (commit 3 -- / / / wiring + fixtures)
# ----------------------------------------------------------------------

_COMPLEX_RING_CASES = _load_cases("complex_ring")

# Named blockers, one case each (pre-existing-failures plan, Task 3 ruling
# 2026-09-25). A fixture's own `xfail_reason` cannot express these: it xfails
# only the OPSIN round-trip stage, and these cases fail at the always-run name
# gate above it. CHEBI:131506: the sentinel splice is fixed and the best-effort
# tier names it RT-exact; the PIN tier abstains because it has no producer for
# a decorated saturated von Baeyer ring-yl prefix (TRIAGE.md, " producer
# sites"). strict=True, so the build that adds that producer must remove this.
_COMPLEX_RING_XFAIL = {
    "CHEBI:131506": (
        "needs a PIN producer for substituted saturated von Baeyer ring-yl "
        "prefixes (class incl. BB 8-methyl-8-azabicyclo[3.2.1]octan-3-yl "
        "acetate); named blocker, TRIAGE.md 'T4 outcome'"
    ),
}
_COMPLEX_RING_PARAMS = _params(_COMPLEX_RING_CASES)


@pytest.mark.parametrize(
    "case",
    _COMPLEX_RING_PARAMS,
    ids=lambda c: f"{c.get('source_corpus','?')}/{c.get('source_id','?')}",
)
class TestComplexRing:
    def test_complex_ring_stereo_roundtrip(self, case):
        _assert_handler_roundtrip(case, "complex_ring")


# ----------------------------------------------------------------------
# a phase: Cycloalkene >= 8 mandatory E/Z (commit 4 -- errata)
# ----------------------------------------------------------------------

_CYCLOALKENE_8PLUS_CASES = _load_cases("cycloalkene_8plus")


@pytest.mark.parametrize(
    "case",
    _params(_CYCLOALKENE_8PLUS_CASES),
    ids=lambda c: f"{c.get('source_corpus','?')}/{c.get('source_id','?')}-r{c.get('ring_size','?')}",
)
class TestCycloalkene8PlusMandatory:
    def test_cycloalkene_8plus_emits_ez_block(self, case):
        # First the standard 2-stage gates (name + prefix + OPSIN
        # round-trip per split-stage assertion).
        name = _assert_handler_roundtrip(case, "cycloalkene_8plus")
        # a phase /: in addition, the prefix MUST contain
        # an [EZ] descriptor, located anywhere in the leading block
        # (the regex tolerates names like '(4S,7Z,...)' where R/S
        # descriptors precede the E/Z one). `name` is the name the gates
        # above checked (the best-effort one for a _TIER_CONTRACT case).
        import re as _re
        assert _re.search(r"^\([^)]*[0-9][EZ]", name), (
            f"P-31.1.3 mandatory E/Z block missing for ring_size="
            f"{case.get('ring_size')} name={name!r} (handler=cycloalkene_8plus, "
            f"source={case.get('source_corpus')}/{case.get('source_id')})"
        )
