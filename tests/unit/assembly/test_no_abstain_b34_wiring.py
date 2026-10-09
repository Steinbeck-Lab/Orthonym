"""Phase B3/B4 -- wiring the universal recursive namer into the decline sites
and the cascade final rung.

These tests lock the four guarantees the task brief requires:
  1. >=3 real a dev split best-effort abstainers now EMIT a complete universal name
     that round-trips (constitution-correct), NOT a wrong molecule.
  2. A wrong-constitution universal candidate does NOT ship (abstains).
  3. A PIN-tier molecule is byte-identical (the PIN path never reaches the
     universal namer).
  4. Determinism: a wired witness from two permuted SMILES yields one name.

The three best-effort-emitting tests carry ``@pytest.mark.opsin_gate``: the
conftest autouse default DISABLES the OPSIN validity gate, and that gate IS the
 net the phase's 0-wrong contract depends on. With the gate off, an
earlier ``_name_impl`` producer's atom-DROPPED name (``methane`` for
``COS(=O)(=O)O``, ``methylcyclohexane`` for the carbamate) ships unverified and
the universal floor is never consulted; with the gate live (production
top-level reality) suppresses the atom-drop and the coverage-complete
universal name ships. This mirrors the sibling ``test_t4_coverage.py`` tests,
which mark ``opsin_gate`` for exactly this reason.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.namer import Orthonym, _validity_gate_name_to_smiles, _validity_gate_jar_present
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
import orthonym.assembly.universal_substituent as US

_BE = _emit_tier_flags("best-effort")

# Real a dev split best-effort abstainers (BEFORE this phase) that the universal
# floor now names, each verified constitution-correct (RT-exact for these
# achiral / fully-covered cases).
_WITNESSES = [
    "COS(=O)(=O)O",                          # monomethyl sulfate
    "CCCC1OC(=O)C(O)CCC=CC(O)C1O",           # a 10-membered lactone polyol
    "C1CCC(CC1)OC(=O)NP(=O)(Cl)Cl",          # the former "unroutable" carbamate
]


def _ikey(smi):
    m = Chem.MolFromSmiles(smi)
    return inchi.InchiToInchiKey(inchi.MolToInchi(m)) if m is not None else None


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.skipif(not _validity_gate_jar_present(),
                    reason="T4 universal floor requires a live OPSIN jar (SELF-01)")
@pytest.mark.parametrize("smi", _WITNESSES)
def test_witness_now_emits_and_is_not_wrong(smi):
    """A molecule that ABSTAINED at best-effort before B3/B4 now emits a complete
    name, and it is the RIGHT molecule (constitution match) -- never wrong."""
    name = Orthonym(style="pin", **_BE).name(smi)
    assert name and not is_failure_name(name), f"still abstains: {name!r}"
    osm = _validity_gate_name_to_smiles(name)
    assert osm is not None, f"emitted an OPSIN-unparseable name: {name!r}"
    ia, ib = _ikey(smi), _ikey(osm)
    assert ia and ib
    # constitution MUST match (0-wrong); these achiral/covered cases are exact.
    assert ia.split("-")[0] == ib.split("-")[0], (
        f"WRONG MOLECULE shipped: {name!r} -> {osm}")


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.skipif(not _validity_gate_jar_present(),
                    reason="the SELF-01 suppression under test needs the jar")
def test_wrong_constitution_universal_candidate_does_not_ship(monkeypatch):
    """If the universal namer ever produced a name for a DIFFERENT molecule, the
    downstream round-trip gate must suppress it to an abstention -- the
    0-wrong net. Stub the producer to emit a wrong-constitution name and assert
    the namer abstains rather than shipping it."""
    smi = "CCCCCCCCO"  # octan-1-ol; PIN path abstains? no -> force via by stub
    # A molecule the PIN path abstains on, so the floor is actually consulted:
    smi = "CCCC1OC(=O)C(O)CCC=CC(O)C1O"
    wrong = US.UniversalResult(
        name="ethane", bindings=(("ethane", frozenset({0, 1})),),
        covers=frozenset({0, 1}))
    monkeypatch.setattr(US, "name_universal_substitutive", lambda *a, **k: wrong)
    name = Orthonym(style="pin", **_BE).name(smi)
    # 'ethane' denotes CC, not the witness -> must reject -> abstain
    # (or fall to another producer, but NEVER ship 'ethane').
    assert name != "ethane", "shipped a wrong-constitution universal name!"
    if name and not is_failure_name(name):
        osm = _validity_gate_name_to_smiles(name)
        if osm:
            assert _ikey(smi).split("-")[0] == _ikey(osm).split("-")[0]


@pytest.mark.unit
def test_pin_tier_is_byte_identical_and_never_reaches_universal(monkeypatch):
    """A PIN-tier molecule names byte-identically and never invokes the universal
    namer (best-effort-only reachability -- PIN default byte-identical)."""
    calls = {"whole": 0, "prefix": 0}
    _w, _p = US.name_universal_substitutive, US.name_universal_substituent_prefix
    monkeypatch.setattr(US, "name_universal_substitutive",
                        lambda *a, **k: (calls.__setitem__("whole", calls["whole"] + 1) or _w(*a, **k)))
    monkeypatch.setattr(US, "name_universal_substituent_prefix",
                        lambda *a, **k: (calls.__setitem__("prefix", calls["prefix"] + 1) or _p(*a, **k)))
    expected = {
        "CCO": "ethanol",
        "CC(C)(C)CC(=O)O": "3,3-dimethylbutanoic acid",
        "c1ccccc1": "benzene",
        "CCN(CC)CC": "N,N-diethylethanamine",
    }
    for smi, want in expected.items():
        got = Orthonym(style="pin").name(smi)  # default = PIN-or-abstain
        assert got == want, f"{smi}: {got!r} != {want!r}"
    assert calls == {"whole": 0, "prefix": 0}, (
        f"PIN tier reached the universal namer: {calls}")


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.skipif(not _validity_gate_jar_present(),
                    reason="best-effort emission requires the OPSIN jar")
def test_universal_floor_is_deterministic_across_permuted_smiles():
    """The universal floor keys every tie-break on the canonical rank, so two
    differently-numbered SMILES of one molecule yield ONE name."""
    smi_a = "COS(=O)(=O)O"
    smi_b = Chem.MolToSmiles(Chem.MolFromSmiles(smi_a), canonical=True,
                             doRandom=False)
    # a genuinely different atom order via a random-root renumbering
    m = Chem.MolFromSmiles(smi_a)
    smi_c = Chem.MolToSmiles(m, rootedAtAtom=m.GetNumAtoms() - 1)
    names = {Orthonym(style="pin", **_BE).name(s) for s in (smi_a, smi_b, smi_c)}
    assert len(names) == 1, f"non-deterministic across permutations: {names}"


@pytest.mark.unit
def test_b3_fallback_guards_undecidable_bond_order():
    """Fix-round Finding 1: the B3 branch fallback must NOT invoke
    ``name_universal_substituent_prefix`` when the attachment free valence is
    ``None`` (UNDECIDABLE -- a bridge/spiro,, or an aromatic/dative
    linkage). ``_free_valence_at_attachment`` documents "None means UNDECIDABLE
    and must never be read as 1", yet ``_render_as_substituent`` silently
    defaults ``None`` -> ``"yl"``, which would name a DIFFERENT free-valence
    morphology. The guard ``free_valence in (1, 2, 3)`` keeps the prior decline
    on ``None``.

    Pure unit test -- gate-independent (the guard blocks before any naming). The
    NONE case is non-vacuous: ``name_substituent`` returns the decline value
    (``None`` -- since 3028d3ad3 ``name_substituent`` honours the ambient
    ``best_effort_ctx`` as ``allow_mancude``, and under ``allow_mancude`` a
    declined fragment is ``None`` rather than the ``'substituent'`` sentinel; see
    its docstring), proving the B3 block WAS reached (``best_effort_ctx`` True,
    ``token`` declined) and that the ONLY thing suppressing the call is the
    bond-order guard. The FV=1 positive control proves the trace actually fires
    when the guard permits, so the empty NONE-case list is not a broken-trace
    artifact (``feedback_harness_that_reports_success``).
    """
    import orthonym.assembly.universal_substituent as US
    from orthonym.assembly import substituent_enumerator as SE
    from orthonym.metrics.provenance import best_effort_ctx

    def _run(smi, frag, attach):
        mol = Chem.MolFromSmiles(smi)
        fv = SE._free_valence_at_attachment(mol, frag, attach)
        calls = []
        real = US.name_universal_substituent_prefix
        US.name_universal_substituent_prefix = (
            lambda *a, **k: (calls.append(k.get("bond_order")), real(*a, **k))[1])
        tok = best_effort_ctx.set(True)
        try:
            ret = SE.name_substituent(mol, frag, attach)
        finally:
            best_effort_ctx.reset(tok)
            US.name_universal_substituent_prefix = real
        return fv, calls, ret

    # NONE case: spiro[4.5]decane fragment {5,6,7,8,9} attaches by two bonds
    # (a bridge), so _free_valence_at_attachment is None -> guard must block.
    fv, calls, ret = _run("C1CCC2(CC1)CCCC2", {5, 6, 7, 8, 9}, 5)
    assert fv is None, f"fixture no longer has an undecidable free valence: {fv}"
    assert ret is None, (
        f"B3 block not reached (no best-effort decline): {ret!r}")
    assert calls == [], (
        f"universal prefix invoked with UNDECIDABLE bond_order: {calls}")

    # POSITIVE CONTROL: the trimethylazaniumyl branch of a choline-type cation has
    # a single single-bonded attachment (free_valence == 1), so the SAME B3 block
    # DOES call the universal prefix -- proving the empty list above is the guard,
    # not a dead trace. (The control used to be the carbamate
    # O-C(=O)-N-P(=O)Cl2 branch; the cascade now names that branch before the B3
    # block is reached, so it no longer reaches the universal prefix and cannot
    # show the trace firing. The cationic nitrogen branch still does.)
    fv, calls, ret = _run("C[N+](C)(C)CCO", {0, 1, 2, 3}, 1)
    assert fv == 1, f"positive-control fixture free valence changed: {fv}"
    assert calls == [1], (
        f"universal prefix NOT called for a decidable free valence: {calls}")
