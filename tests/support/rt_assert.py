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


def _independent_parse(name: str, smiles=None):
    """OPSIN 2.9.0's SMILES for ``name`` (``-osmi``), from a FRESH call that does not
    go through the engine: not its validity oracle, not its per-scope or process-wide
    name memo. The in-process JVM serves it when it can (``_opsin_stdout_uncached``, the
    un-memoised entry point, byte-identical to the CLI); otherwise a ``java -jar``
    subprocess does. None when OPSIN rejects the name or cannot be run.

    ``smiles`` is the structure the name is read back for. OPSIN's ``-r`` (allowRadicals) is
    passed only when that structure has radical electrons (``validation.read_back_policy``,
    the engine's own rule): a name with a free valence ('(phosphoryl)oxy') reads back to a
    closed-shell structure under ``-r``, and a test that checks a closed-shell input
    with it passes for a name that describes a substituent group. With no ``smiles`` (a
    name read for its own sake: a retained name, a gold substituent string) the legacy
    reading, ``-r``, stands."""
    import subprocess
    from orthonym.jvm_bridge import _opsin_stdout_uncached
    from orthonym.jvm_flags import JVM_HYGIENE_FLAGS
    from orthonym.validation.opsin_roundtrip import _find_opsin_jar
    from orthonym.validation.read_back_policy import smiles_has_radical_electrons
    if not name:
        return None
    jar = _find_opsin_jar("2.9.0")
    if not jar:
        return None
    allow_radicals = smiles is None or smiles_has_radical_electrons(smiles)
    txt, served = _opsin_stdout_uncached(name, allow_radicals, jar)
    if not served:
        try:
            proc = subprocess.run(
                ["java", *JVM_HYGIENE_FLAGS, "-jar", jar, *(["-r"] if allow_radicals else []), "-osmi"],
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
    opsin_smi = _independent_parse(name, smiles)
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


_DATIVE_TYPES = (Chem.BondType.DATIVE, Chem.BondType.DATIVEONE,
                 Chem.BondType.DATIVEL, Chem.BondType.DATIVER)


def smiles_without_dative_bonds(smiles: str):
    """RDKit's SMILES of ``smiles`` with every dative bond removed (the donor keeps its lone pair
    and its hydrogens, the acceptor its charge; no charge moves), or None when ``smiles`` has no
    dative bond or does not parse. RDKit gives a drawing with dative bonds no InChIKey, so the
    full key of this drawing is the reference key of the read-back of such an input. Written here
    with RDKit alone: it shares no code with ``assembly.metal_offer``, whose split it checks."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    dative = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
              if b.GetBondType() in _DATIVE_TYPES]
    if not dative:
        return None
    rw = Chem.RWMol(mol)
    for idx in {i for pair in dative for i in pair}:
        atom = rw.GetAtomWithIdx(idx)
        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)
    for a, b in dative:
        rw.RemoveBond(a, b)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return Chem.MolToSmiles(out)


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
    the fixed-H InChIs. A drawing with dative bonds has no key: it is read back
    against the key of the same drawing without those bonds."""
    from orthonym.validation.protonation_identity import protonation_site_verdict
    from orthonym.validation.radical_identity import radical_identity_verdict
    if not name:
        return False
    key = _full_inchikey(smiles)
    if not key:
        # a drawing with dative bonds has no key: the drawing without them has (decision 1 of
        # the zinc plan, 2026-10-13); every other drawing without a key is never read back
        smiles = smiles_without_dative_bonds(smiles) or smiles
        key = _full_inchikey(smiles)
    opsin_smi = _independent_parse(name, smiles)
    if not (opsin_smi and key and _full_inchikey(opsin_smi) == key):
        return False
    return (radical_identity_verdict(smiles, opsin_smi) != "mismatch"
            and protonation_site_verdict(smiles, opsin_smi, name) != "mismatch")


def assert_full_rt(name: str, smiles: str, what: str = "") -> str:
    """Assert ``name`` is a real name (not a failure signal) that OPSIN parses
    back to the input's FULL standard InChIKey (constitution + stereo + charge +
    isotope layers). Returns the name.

    This is the assertion of the pre-existing-failures plan's decision D-b
    (2026-09-24): where a wider tier now names a molecule instead of declining,
    the test asserts the exact round trip, not the spelling. OPSIN runs as a
    fresh call outside the engine, with ``-r`` only for an input that has radical
    electrons (``_independent_parse``): a name with a free valence is no name of a
    closed-shell input, and OPSIN reads it back to that input's structure under
    ``-r``. The full-key comparison alone would NOT reject a radical reading of a
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
    assert is_failure_name(pin) or name_reads_back(pin, smiles), (
        f"PIN tier shipped a name that is not RT-exact for {smiles}: {pin!r}")
    return pin, be


def name_reads_back(name: str, smiles: str) -> bool:
    """``name`` is RT-exact, or OPSIN has no grammar for it and reads the twin spelling of the same structure
    (``validation.twin_spellings``, a default-tier exception: a PIN spelling OPSIN reads back through its twin)
    to the input's full InChIKey. Both reads are fresh OPSIN calls outside the engine."""
    if name_is_rt_exact(name, smiles):
        return True
    from orthonym.validation.twin_spellings import twin_of
    found = twin_of(name)
    return found is not None and name_is_rt_exact(found[0], smiles)


def assert_not_rt_exact(smiles: str) -> None:
    """RED baseline: assert we do NOT yet produce a correctly round-tripping name."""
    name = rt_exact_name(smiles)
    assert name is None, f"expected no rt_exact name yet for {smiles}, got {name!r}"
