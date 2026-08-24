"""v29 Phase 2 T2a (H1): total skeletal ('a') replacement-prefix construction
for RING (von-Baeyer / spiro) systems.

The hole this closes, reproduced before the fix
-----------------------------------------------
``polycyclic.py::get_heteroatom_replacement_prefix`` gated every ring atom on
``symbol in HETEROATOM_PREFIXES`` (14 elements). Any OTHER skeletal ring element
was skipped, contributing **no** replacement morpheme -- while still counting
toward the von-Baeyer stem (``total_atoms = len(ring_atoms)``). So
``C1CC2CC[Hg]C2C1`` came back as ``bicyclo[3.3.0]octane``: a hydrocarbon name for
a mercury-containing ring, i.e. a WRONG STRUCTURE, on both the default and the
complete path. Nothing downstream caught it -- SELF-01 (the name->structure round
trip) fails OPEN when the OPSIN jar is absent, which is a supported mode and is
the mode this whole module runs in.

The spiro sibling was NOT "already failing closed" (a premise this module
disproves): ``spiro.py::_build_hetero_prefix`` reaches
``polycyclic_bridged.get_heteroatom_prefix``, whose fallback is
``symbol.lower() + 'a'`` -- so an off-table element got an INVENTED morpheme
(``3-znaspiro[5.5]undecane``, ``3-feaspiro[…]``, ``3-alaspiro[…]``) instead of
being dropped. A different failure mode, equally unsound.

The fix: ``rules/ring_replacement.py::build_replacement_prefix`` returns the
prefix *together with* ``unexpressed`` -- every skeletal non-carbon ring atom the
element table could not spell -- and both universal analyzers fail closed when it
is non-empty. The element table is NOT extended here (a wrong table entry ships a
wrong name; extension is citation-gated and lands separately). This task delivers
the sound floor: the 14 in-table elements keep working BYTE-IDENTICALLY, and
everything else refuses instead of lying.

(The name ``rules/skeletal_replacement.py`` was already taken by the ACYCLIC
P-15.4 chain namer, hence ``ring_replacement``.)
"""
import pytest
from rdkit import Chem

import orthonym.namer as _namer_mod
from orthonym.namer import Orthonym, is_failure_name
from orthonym.rules.polycyclic import (
    HETEROATOM_PREFIXES, VonBaeyerAnalyzer, get_heteroatom_replacement_prefix,
)
from orthonym.rules.ring_replacement import (
    ReplacementPrefix, build_replacement_prefix,
)
from orthonym.rules.vonbaeyer_universal import (
    analyze_cage_universal, analyze_spiro_universal,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Byte-identity corpus.
#
# Every string below was CAPTURED from ``get_heteroatom_replacement_prefix`` at
# 02c1271f (before this task touched it) and is written here as an explicit
# literal, so the assertion still means something after the refactor -- the new
# primitive is compared against recorded behaviour, never against itself.
#
# Coverage: all 14 in-table elements as single occurrences; x2/x3 multipliers;
# all-heteroatom cages (heptasila / heptagerma / heptatellura -- the cases the
# builder's own docstring cites); 10 multi-element systems exercising the
# Table-2.8 citation order (note ``5-oxa-3-sila`` and ``5-aza-3-bora``: cited by
# ELEMENT seniority, not by locant); three lambda cases (P-31.1.4.2).
# ---------------------------------------------------------------------------
# v36 REBASELINE (2026-08-24): the 25 rows below were re-captured after v34
# (58209798, analyze_cage_universal) corrected the von-Baeyer heteroatom
# numbering to give the LOWEST locant set (P-31.1.4.3.4). The pre-v34 captures
# (3-oxa, 3-oxa-5-aza, 2,4,7-trioxa, 3lambda3-tellura ...) put heteroatoms at
# NON-minimal locants; a valid lower-locant numbering exists for each (OPSIN
# round-trips the new full name to the input InChIKey), so the old values
# VIOLATED the lowest-locant rule and were non-preferred. New values are
# PIN-correct: locant set lowest, and within a tied set the senior element
# (O before N/S, per the replacement seniority order) takes the lower locant
# (e.g. 2-oxa-6-aza, not 6-oxa-2-aza). change-asserted-value: old provably
# non-minimal (decisive); cross-checked; mutation-verified.
BASELINE = [
    # --- single occurrence, every element in the table ---
    ("O1CC2CCC1C2", "2-oxa"),
    ("S1CC2CCC1C2", "2-thia"),
    ("[Se]1CC2CCC1C2", "2-selena"),
    ("[Te]1CC2CCC1C2", "2-tellura"),
    ("N1CC2CCC1C2", "2-aza"),
    ("P1CC2CCC1C2", "2-phospha"),
    ("[AsH]1CC2CCC1C2", "2-arsa"),
    ("[SbH]1CC2CCC1C2", "2-stiba"),
    ("[BiH]1CC2CCC1C2", "2-bisma"),
    ("[SiH2]1CC2CCC1C2", "2-sila"),
    ("[GeH2]1CC2CCC1C2", "2-germa"),
    ("[SnH2]1CC2CCC1C2", "2-stanna"),
    ("[PbH2]1CC2CCC1C2", "2-plumba"),
    ("B1CC2CCC1C2", "2-bora"),
    # --- repeats of one element (SIMPLE_MULTIPLIERS) ---
    ("O1COC2CC1C2", "2,4-dioxa"),
    ("N1CNC2CC1C2", "2,4-diaza"),
    ("O1COC2OC1C2", "2,4,6-trioxa"),
    ("S1CSC2CC1C2", "2,4-dithia"),
    # --- all-heteroatom cages ---
    ("[SiH2]1[SiH2][SiH]2[SiH2][SiH2][SiH]1[SiH2]2",
     "1,2,3,4,5,6,7-heptasila"),
    ("[GeH2]1[GeH2][GeH]2[GeH2][GeH2][GeH]1[GeH2]2",
     "1,2,3,4,5,6,7-heptagerma"),
    ("[Te]1[Te][Te]2[Te][Te][Te]1[Te]2",
     "1λ3,2,3,4λ3,5,6,7-heptatellura"),
    # --- multi-element citation order (element seniority, NOT locant order) ---
    ("O1CC2CNC1C2", "2-oxa-6-aza"),
    ("O1CC2CSC1C2", "2-oxa-6-thia"),
    ("S1CC2CNC1C2", "2-thia-6-aza"),
    ("O1CSC2CNC1C2", "4-oxa-2-thia-6-aza"),
    ("[SiH2]1CC2COC1C2", "2-oxa-6-sila"),
    ("B1CC2CNC1C2", "2-aza-6-bora"),
    ("[Se]1CC2C[Te]C1C2", "2-selena-6-tellura"),
    ("P1CC2C[AsH]C1C2", "2-phospha-6-arsa"),
    ("[SnH2]1CC2C[PbH2]C1C2", "2-stanna-6-plumba"),
    ("[SbH]1CC2C[BiH]C1C2", "2-stiba-6-bisma"),
    # --- lambda IS cited (valence exceeds the connectivity-forced value) ---
    ("C1C[SH2]C2CCC1C2", "2λ4-thia"),
    ("C1C[SH4]C2CCC1C2", "2λ6-thia"),
    ("C1C[PH3]C2CCC1C2", "2λ5-phospha"),
    ("[TeH]1CC2CCC1C2", "2λ3-tellura"),
    ("[PH]12CCC(CC1)C2", "1λ4-phospha"),
    ("[SbH]12CCC(CC1)C2", "1λ4-stiba"),
    # --- lambda is SUPPRESSED (bridgehead valence forced by skeletal degree, so
    #     an explicit lambda there is redundant and rejected) ---
    ("[TeH]12CCC(CC1)C2", "1-tellura"),
    ("[SeH]12CCC(CC1)C2", "1-selena"),
    ("[SH]12CCC(CC1)C2", "1-thia"),
    ("[SnH]12CCC(CC1)C2", "1-stanna"),
    # --- suppressed on one atom, cited on others, in one prefix ---
    ("[TeH]1[Te][TeH]2[Te][Te][Te]1[Te]2",
     "1,2,3λ3,4λ3,5,6,7-heptatellura"),
    # --- no heteroatoms ---
    ("C1CC2CCC1C2", ""),
]

#: Bridgehead heteroatoms whose valence EQUALS the value their skeletal degree
#: already forces. ``vb_lambda_for_atom`` must suppress the lambda there (an
#: explicit one is redundant AND rejected on re-parse), even though the shared
#: ``nonstandard_bonding_number`` does report a bonding number.
LAMBDA_SUPPRESSED = [
    ("[TeH]12CCC(CC1)C2", "1-tellura"),
    ("[SeH]12CCC(CC1)C2", "1-selena"),
    ("[SH]12CCC(CC1)C2", "1-thia"),
]

# Skeletal elements the table cannot spell. Each is a real von-Baeyer cage
# (bicyclo[3.3.0]) whose stem counts 8 atoms.
#
# Two DIFFERENT reasons to refuse, both covered here (v29 P2-T2b):
#   * Hg/Zn/Fe -- in no replacement table at all; no morpheme exists.
#   * I/At/Po  -- Table 1.5 rows with a real morpheme (``ioda``/``astata``/
#     ``polona``) that P-23.3.1 and/or P-23.3.2.2 do not rank, so there is no
#     sanctioned von Baeyer citation position or numbering rank. See
#     ``ring_replacement.VB_INADMISSIBLE``.
# ``[AlH]`` was in this list until T2b, which made Al emittable (``alumina``);
# the Al cases now assert the SPELLING, in test_v29_table_1_5_completion.
OFF_TABLE_CAGES = [
    "C1CC2CC[Hg]C2C1",
    "C1CC2CC[Zn]C2C1",
    "C1CC2CC[Fe]C2C1",
    "C1CC2CC[IH]C2C1",
    "C1CC2CC[AtH]C2C1",
    "C1CC2CC[PoH]C2C1",
]

# The same unspellable elements in a spiro system (the sibling analyzer).
OFF_TABLE_SPIRO = [
    "C1CCC2(CC1)CC[Hg]CC2",
    "C1CCC2(CC1)CC[Zn]CC2",
    "C1CCC2(CC1)CC[Fe]CC2",
    "C1CCC2(CC1)CC[IH]CC2",
    "C1CCC2(CC1)CC[AtH]CC2",
    "C1CCC2(CC1)CC[PoH]CC2",
]


def _vb(smiles):
    """(mol, numbering, ring_atoms) for a von-Baeyer cage. No Java."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparseable test SMILES {smiles!r}"
    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)
    desc = VonBaeyerAnalyzer().analyze(mol, ring_atoms)
    assert desc and desc.numbering, f"no von-Baeyer descriptor for {smiles!r}"
    return mol, desc.numbering, ring_atoms


def _off_table_ring_atoms(mol, ring_atoms):
    return tuple(sorted(
        i for i in ring_atoms
        if mol.GetAtomWithIdx(i).GetSymbol() != 'C'
        and mol.GetAtomWithIdx(i).GetSymbol() not in HETEROATOM_PREFIXES))


# ---------------------------------------------------------------------------
# 1. Byte-identity: the primitive AND the retained wrapper reproduce the
#    captured strings exactly. A single character of drift here breaks the
#    1630/1630 PIN gold, so this is the hard requirement of the task.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", BASELINE)
def test_primitive_prefix_is_byte_identical(smiles, expected):
    mol, numbering, ring_atoms = _vb(smiles)
    assert build_replacement_prefix(
        mol, numbering, ring_atoms).prefix == expected


@pytest.mark.parametrize("smiles,expected", BASELINE)
def test_wrapper_prefix_is_byte_identical(smiles, expected):
    """``get_heteroatom_replacement_prefix`` keeps its exact contract (a bare
    ``str``) for its three PIN callers (polycyclic x2, bicyclo)."""
    mol, numbering, ring_atoms = _vb(smiles)
    got = get_heteroatom_replacement_prefix(mol, numbering, ring_atoms)
    assert got == expected
    assert isinstance(got, str)


@pytest.mark.parametrize("smiles,expected", LAMBDA_SUPPRESSED)
def test_lambda_suppressed_when_connectivity_forces_the_valence(smiles, expected):
    """Direct test of the rule, not just of the string: the shared
    ``nonstandard_bonding_number`` DOES report a bonding number for this atom, and
    the ring-prefix refinement must still omit it."""
    from orthonym.rules.lambda_convention import nonstandard_bonding_number
    from orthonym.rules.ring_replacement import vb_lambda_for_atom

    mol, numbering, ring_atoms = _vb(smiles)
    bridgehead = next(
        i for i in ring_atoms
        if mol.GetAtomWithIdx(i).GetSymbol() != 'C'
        and mol.GetAtomWithIdx(i).GetDegree() >= 3)
    assert nonstandard_bonding_number(mol, bridgehead) is not None, (
        "precondition: the shared check reports a bonding number here")
    assert vb_lambda_for_atom(mol, bridgehead) is None, (
        "the ring refinement must suppress a connectivity-forced valence")

    res = build_replacement_prefix(mol, numbering, ring_atoms)
    assert res.prefix == expected
    assert 'lambda' not in res.prefix


def test_prefix_never_carries_a_trailing_hyphen():
    """P-23.3.1: the 'a'-prefix attaches DIRECTLY to the descriptor
    (``2-oxabicyclo[2.2.2]octane``). Callers concatenate, so a trailing '-'
    would spell ``3-oxa-bicyclo…``."""
    for smiles, expected in BASELINE:
        if not expected:
            continue
        mol, numbering, ring_atoms = _vb(smiles)
        assert not build_replacement_prefix(
            mol, numbering, ring_atoms).prefix.endswith('-'), smiles


# ---------------------------------------------------------------------------
# 2. ``unexpressed``: the signal that closes the hole.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,_expected", BASELINE)
def test_unexpressed_empty_for_in_table_and_all_carbon(smiles, _expected):
    mol, numbering, ring_atoms = _vb(smiles)
    assert build_replacement_prefix(
        mol, numbering, ring_atoms).unexpressed == ()


@pytest.mark.parametrize("smiles", OFF_TABLE_CAGES)
def test_unexpressed_lists_the_off_table_skeletal_atom(smiles):
    mol, numbering, ring_atoms = _vb(smiles)
    res = build_replacement_prefix(mol, numbering, ring_atoms)
    off = _off_table_ring_atoms(mol, ring_atoms)
    assert off, "test corpus bug: no off-table ring atom in this SMILES"
    assert res.unexpressed == off
    # the atom contributes NO morpheme -- which is exactly why the caller must
    # refuse rather than consume ``prefix``.
    assert res.prefix == ""


def test_unexpressed_reports_a_heteroatom_with_no_locant():
    """An in-table heteroatom the numbering does not reach cannot be cited
    either -- the same silent-drop shape, so it is reported the same way."""
    mol, numbering, ring_atoms = _vb("O1CC2CCC1C2")
    oxygen = next(i for i in ring_atoms
                  if mol.GetAtomWithIdx(i).GetSymbol() == 'O')
    partial = {k: v for k, v in numbering.items() if k != oxygen}
    res = build_replacement_prefix(mol, partial, ring_atoms)
    assert res.unexpressed == (oxygen,)
    assert res.prefix == ""


# ---------------------------------------------------------------------------
# 3. ``per_atom``: one entry per EXPRESSED heteroatom, carrying the morpheme
#    that spells it (consumed by the per-token proof bindings).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,_expected", BASELINE)
def test_per_atom_covers_exactly_the_expressed_heteroatoms(smiles, _expected):
    mol, numbering, ring_atoms = _vb(smiles)
    res = build_replacement_prefix(mol, numbering, ring_atoms)
    hetero = {i for i in ring_atoms
              if mol.GetAtomWithIdx(i).GetSymbol() != 'C'}
    idxs = [idx for idx, _m in res.per_atom]
    assert idxs == sorted(idxs), "per_atom must be deterministic"
    assert set(idxs) == hetero
    assert len(idxs) == len(hetero), "one entry per atom, no duplicates"
    for idx, morpheme in res.per_atom:
        symbol = mol.GetAtomWithIdx(idx).GetSymbol()
        assert morpheme == HETEROATOM_PREFIXES[symbol][0]
        assert morpheme in res.prefix


#: 22 skeletal Si in a bicyclo[10.9.1] cage -- INSIDE ``MAX_CAGE_ATOMS = 40``, so
#: the >20 multiplying-prefix fallback is reachable today and spelled the bare
#: integer: ``1,…,22-22sila``. Not a word.
SILA22 = ('[SiH2]1' + '[SiH2]' * 9 + '[SiH]2' + '[SiH2]' * 9
          + '[SiH]1[SiH2]2')
SILA22_LEGACY_PREFIX = (
    "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22-22sila")


def test_multiplicity_past_the_table_is_unexpressed_but_prefix_unchanged():
    """``prefix`` must keep the legacy (malformed) string, and the WRAPPER must
    now refuse outright.

    The primitive's ``prefix`` contract is unchanged: it still reports the legacy
    ``22sila`` string, because that field is defined as byte-identical to what the
    retired inline builder produced for every input.

    ``get_heteroatom_replacement_prefix`` no longer passes that string on. This
    test used to assert that it did, on the reasoning that "withholding the part
    there would turn a malformed name into one that silently drops 22 skeletal
    atoms" -- true only while the wrapper's ``-> str`` signature gave its three
    PIN callers no way to refuse. Now that it returns ``Optional[str]`` and the
    callers refuse on ``None``, refusing is strictly better than emitting a name
    containing ``22sila``: both alternatives were wrong, and only this one emits
    nothing.
    """
    mol, numbering, ring_atoms = _vb(SILA22)
    res = build_replacement_prefix(mol, numbering, ring_atoms)
    assert res.prefix == SILA22_LEGACY_PREFIX
    assert get_heteroatom_replacement_prefix(mol, numbering, ring_atoms) is None
    assert res.unexpressed == tuple(sorted(ring_atoms))
    assert res.per_atom == ()


def test_analyze_cage_universal_refuses_multiplicity_past_the_table():
    mol = Chem.MolFromSmiles(SILA22)
    assert mol is not None
    assert analyze_cage_universal(mol) is None
    assert analyze_cage_universal(mol, allow_mancude=True) is None


@pytest.mark.parametrize(
    "smiles", [s for s, _e in BASELINE] + OFF_TABLE_CAGES + [SILA22])
def test_per_atom_and_unexpressed_partition_the_heteroatoms(smiles):
    """The two channels are disjoint and together cover every non-carbon ring
    atom -- so 'the caller refuses iff some skeletal atom is unaccounted for' is
    a complete rule, not a partial one."""
    mol, numbering, ring_atoms = _vb(smiles)
    res = build_replacement_prefix(mol, numbering, ring_atoms)
    expressed = {idx for idx, _m in res.per_atom}
    unexpressed = set(res.unexpressed)
    hetero = {i for i in ring_atoms
              if mol.GetAtomWithIdx(i).GetSymbol() != 'C'}
    assert expressed & unexpressed == set()
    assert expressed | unexpressed == hetero


def test_per_atom_excludes_unexpressed_atoms():
    mol, numbering, ring_atoms = _vb("C1CC2CC[Hg]C2C1")
    res = build_replacement_prefix(mol, numbering, ring_atoms)
    assert res.per_atom == ()
    assert res.unexpressed != ()


def test_result_is_frozen():
    mol, numbering, ring_atoms = _vb("O1CC2CCC1C2")
    res = build_replacement_prefix(mol, numbering, ring_atoms)
    assert isinstance(res, ReplacementPrefix)
    with pytest.raises(Exception):
        res.prefix = "nope"


# ---------------------------------------------------------------------------
# 4. The contract at the two callers: refuse, never a stem that counts an atom
#    no morpheme spells.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles", OFF_TABLE_CAGES)
@pytest.mark.parametrize("allow_mancude", [False, True])
def test_analyze_cage_universal_refuses_off_table(smiles, allow_mancude):
    mol = Chem.MolFromSmiles(smiles)
    assert analyze_cage_universal(mol, allow_mancude=allow_mancude) is None


@pytest.mark.parametrize("smiles,expected_prefix", [
    ("C1CC2CC[Se]C2C1", "2-selena"),
    ("C1CC2CCOC2C1", "2-oxa"),
])
@pytest.mark.parametrize("allow_mancude", [False, True])
def test_analyze_cage_universal_still_names_in_table(
        smiles, expected_prefix, allow_mancude):
    mol = Chem.MolFromSmiles(smiles)
    cage = analyze_cage_universal(mol, allow_mancude=allow_mancude)
    assert cage is not None, "in-table hetero cage must still be named"
    assert cage.hetero_prefix == expected_prefix
    assert cage.descriptor == "bicyclo[3.3.0]"
    assert cage.total_atoms == 8


@pytest.mark.parametrize("smiles", OFF_TABLE_SPIRO)
def test_analyze_spiro_universal_refuses_off_table(smiles):
    """The sibling analyzer adopts the SAME rule. Before this task it emitted an
    invented morpheme (``3-zna``, ``3-fea``, ``3-ala``) from
    ``get_heteroatom_prefix``'s ``symbol.lower() + 'a'`` fallback."""
    mol = Chem.MolFromSmiles(smiles)
    assert analyze_spiro_universal(mol) is None


def test_analyze_spiro_universal_still_names_in_table():
    mol = Chem.MolFromSmiles("C1CCC2(CC1)CCOCC2")
    spiro = analyze_spiro_universal(mol)
    assert spiro is not None
    assert spiro.hetero_prefix == "3-oxa"
    assert spiro.descriptor == "spiro[5.5]"


# ---------------------------------------------------------------------------
# 5. End-to-end, with the OPSIN jar made UNAVAILABLE.
#
# This is the whole point of the task. SELF-01 fails OPEN with no jar, so a fix
# verified only with the jar present is not verified (Phase 1b established
# this). The suite-wide autouse fixture already forces
# ``_DISABLE_VALIDITY_GATE = True``; the fixture below additionally removes the
# jar itself, so no downstream round trip can mask a wrong name.
# ---------------------------------------------------------------------------
@pytest.fixture
def no_opsin_jar(monkeypatch):
    import orthonym.validation.opsin_roundtrip as _rt
    monkeypatch.setattr(_rt, "_find_opsin_jar", lambda *a, **k: None)
    monkeypatch.setattr(_namer_mod, "_DISABLE_VALIDITY_GATE", True)
    yield


@pytest.mark.parametrize("smiles", OFF_TABLE_CAGES)
def test_namer_never_gives_a_hydrocarbon_name_to_an_off_table_cage(
        smiles, no_opsin_jar):
    """``bicyclo[3.3.0]octane`` for ``C1CC2CC[Hg]C2C1`` was the reproduced wrong
    structure. With the jar gone, only the fix can prevent it."""
    nm = Orthonym(general_fallback=True, _disable_opsin_validity_gate=True)
    name = nm.name(smiles)
    assert name != "bicyclo[3.3.0]octane"
    assert is_failure_name(name), (
        f"{smiles!r} produced a positive name {name!r}; the ring stem counts a "
        f"skeletal atom that no morpheme in it spells")


@pytest.mark.parametrize("smiles,expected", [
    ("C1CC2CC[Se]C2C1", "2-selenabicyclo[3.3.0]octane"),
    ("C1CC2CCOC2C1", "2-oxabicyclo[3.3.0]octane"),
])
def test_namer_still_names_in_table_cages(smiles, expected, no_opsin_jar):
    """The fail-closed rule must cost NOTHING on the 14 in-table elements."""
    nm = Orthonym(general_fallback=True, _disable_opsin_validity_gate=True)
    assert nm.name(smiles) == expected
