"""Shared best-effort rt_exact assertions for the composition-completeness lever.

The acceptance bar for the best-effort tier is FULL isomeric round-trip: the emitted
name, parsed by OPSIN, must canonicalise to the same molecule as the input (constitution
+ all stereo + charge + isotope). These helpers verify that INDEPENDENTLY of the naming
pipeline's own OPSIN validity gate (which ``tests/conftest.py`` disables suite-wide), so a
test is correct whether the gate is on or off: ``rt_exact_name`` returns the name only if
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


def rt_exact_name(smiles: str):
    """Return the best-effort name IFF it OPSIN-parses back to the FULL isomeric input,
    else None. Independent of the pipeline's own validity gate."""
    from orthonym.namer import _validity_gate_name_to_smiles
    res = name_best_effort(smiles)
    name = res.get("name")
    if not name or res.get("source") == "abstain":
        return None
    opsin_smi = _validity_gate_name_to_smiles(name)
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
    (constitution + stereo + charge + isotope layers). Independent of the
    pipeline's own validity gate."""
    from orthonym.namer import _validity_gate_name_to_smiles
    if not name:
        return False
    opsin_smi = _validity_gate_name_to_smiles(name)
    key = _full_inchikey(smiles)
    return bool(opsin_smi) and bool(key) and _full_inchikey(opsin_smi) == key


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
