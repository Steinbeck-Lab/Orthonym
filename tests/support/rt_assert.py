"""Shared best-effort rt_exact assertions for the composition-completeness lever.

The acceptance bar for the best-effort tier is FULL isomeric round-trip: the emitted
name, parsed by OPSIN, must canonicalise to the same molecule as the input (constitution
+ all stereo + charge + isotope). These helpers verify that INDEPENDENTLY of the naming
pipeline's own OPSIN validity gate (which ``tests/conftest.py`` disables suite-wide), so a
test is correct whether the gate is on or off. Every parse here is a FRESH OPSIN call
(``_independent_parse``): it bypasses the engine's validity oracle and its name caches, so
a stale or wrong cached parse inside the engine cannot make a test pass. ``rt_exact_name`` returns the name only if
it genuinely round-trips, else ``None``. That makes the RED baseline (``assert_not_rt_exact``)
gate-independent too — "we do not yet produce a correct name for this molecule".
"""
from rdkit import Chem
from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

_NAMER = Orthonym(style="pin", **_emit_tier_flags("best-effort"))


def name_best_effort(smiles: str) -> dict:
    """Name at the best-effort tier; returns the full name_tiered dict."""
    return _NAMER.name_tiered(smiles)


def _canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def _independent_parse(name: str):
    """OPSIN 2.9.0's SMILES for ``name`` (``-r -osmi``), from a FRESH call that does not
    go through the engine: not its validity oracle, not its per-scope or process-wide
    name memo. The in-process JVM serves it when it can (``_opsin_stdout_uncached``, the
    un-memoised entry point, byte-identical to the CLI); otherwise a ``java -jar``
    subprocess does. None when OPSIN rejects the name or cannot be run."""
    import subprocess
    from orthonym.jvm_bridge import _opsin_stdout_uncached
    from orthonym.jvm_flags import JVM_HYGIENE_FLAGS
    from orthonym.validation.opsin_roundtrip import _find_opsin_jar
    if not name:
        return None
    jar = _find_opsin_jar("2.9.0")
    if not jar:
        return None
    txt, served = _opsin_stdout_uncached(name, True, jar)
    if not served:
        try:
            proc = subprocess.run(
                ["java", *JVM_HYGIENE_FLAGS, "-jar", jar, "-r", "-osmi"],
                input=name + "\n", capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            return None
        txt = proc.stdout
    smi = (txt or "").strip()
    return smi or None


def rt_exact_name(smiles: str):
    """Return the best-effort name IFF it OPSIN-parses back to the FULL isomeric input,
    else None. Independent of the pipeline's own validity gate."""
    res = name_best_effort(smiles)
    name = res.get("name")
    if not name or res.get("source") == "abstain":
        return None
    opsin_smi = _independent_parse(name)
    if opsin_smi is None:
        return None
    if _canon(opsin_smi) != _canon(smiles):
        return None
    return name


def assert_rt_exact(smiles: str) -> str:
    """Assert the molecule names at best-effort AND the name round-trips exactly."""
    name = rt_exact_name(smiles)
    res = name_best_effort(smiles)
    assert name is not None, (
        f"NOT rt_exact for {smiles}\n  name:  {res.get('name')!r} (source={res.get('source')})")
    return name


def _full_inchikey(smi):
    from rdkit.Chem import inchi
    m = Chem.MolFromSmiles(smi) if smi else None
    return inchi.MolToInchiKey(m) if m is not None else ""


def name_is_rt_exact(name: str, smiles: str) -> bool:
    """True iff OPSIN parses ``name`` back to the input's FULL standard InChIKey
    (constitution + stereo + charge + isotope layers) AND to the same radical graph.
    Independent of the pipeline's own validity gate and of its OPSIN caches.

    The full key encodes neither radical electrons nor bond order: 'ethane-1,2-diyl'
    parses to [CH2][CH2], whose key equals that of C=C, and 'but-3-en-2-yl' shares
    the key of [CH2]C=CC. So a radical on either side must also pass
    ``radical_identity_verdict`` (validation/radical_identity.py, the one
    definition every gate uses). Nor does it place a hydron: the /p layer is
    mobile, so 'N-methyl-3-(methylamino)propanamidium' shares the key of the
    protonated amine C[NH2+]CCC(=O)NC. When both sides carry a protonated atom,
    ``protonation_site_verdict`` (validation/protonation_identity.py) compares
    the fixed-H InChIs."""
    from orthonym.validation.protonation_identity import protonation_site_verdict
    from orthonym.validation.radical_identity import radical_identity_verdict
    if not name:
        return False
    opsin_smi = _independent_parse(name)
    key = _full_inchikey(smiles)
    if not (opsin_smi and key and _full_inchikey(opsin_smi) == key):
        return False
    return (radical_identity_verdict(smiles, opsin_smi) != "mismatch"
            and protonation_site_verdict(smiles, opsin_smi) != "mismatch")


def assert_full_rt(name: str, smiles: str, what: str = "") -> str:
    """Assert ``name`` is a real name (not a failure signal) that OPSIN parses
    back to the input's FULL standard InChIKey (constitution + stereo + charge +
    isotope layers). Returns the name.

    This is the assertion of the pre-existing-failures plan's decision D-b
    (2026-09-24): where a wider tier now names a molecule instead of declining,
    the test asserts the exact round trip, not the spelling. OPSIN runs as a
    fresh call outside the engine (with ``-r``), so radical names parse; the
    full-key comparison alone would NOT reject a radical reading of a
    closed-shell input (the key encodes neither radical electrons nor bond
    order), so ``name_is_rt_exact`` adds the radical-identity check.
    """
    from orthonym.errors import is_failure_name
    assert name and not is_failure_name(name) and name_is_rt_exact(name, smiles), (
        f"{what}{smiles}: name {name!r} does not round-trip to the input's "
        f"full InChIKey")
    return name


def assert_tier_contract(smiles: str) -> tuple:
    """The tier contract for a molecule whose PIN the PIN tier cannot build.

    Best-effort must name it, and the name must round-trip to the input's full
    InChIKey (breadth never drops). The PIN tier (``name_compound``, the
    default) may fail closed, but whatever it ships must be RT-exact too: it
    may never describe a different molecule. Returns ``(pin_name, be_name)``.

    It checks in whatever OPSIN-validity-gate state the caller runs. A test of
    what SHIPS runs it with the gate on (``@pytest.mark.opsin_gate``); a test
    of a raw producer may run it gate-off (the suite default), which is
    stricter.
    """
    from orthonym import name_compound
    from orthonym.errors import is_failure_name
    res = name_best_effort(smiles)
    be = res.get("name")
    assert be and res.get("source") != "abstain" and name_is_rt_exact(be, smiles), (
        f"best-effort name is not RT-exact for {smiles}\n"
        f"  name: {be!r} (tier={res.get('tier')}, source={res.get('source')})")
    pin = name_compound(smiles)
    assert is_failure_name(pin) or name_is_rt_exact(pin, smiles), (
        f"PIN tier shipped a name that is not RT-exact for {smiles}: {pin!r}")
    return pin, be


def assert_not_rt_exact(smiles: str) -> None:
    """RED baseline: assert we do NOT yet produce a correctly round-tripping name."""
    name = rt_exact_name(smiles)
    assert name is None, f"expected no rt_exact name yet for {smiles}, got {name!r}"
