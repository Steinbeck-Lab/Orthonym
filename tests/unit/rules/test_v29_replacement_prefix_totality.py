"""Skeletal-replacement ('a') prefixes: NEVER a fabricated morpheme.

The defect this suite locks down
-------------------------------
``rules/polycyclic_bridged.get_heteroatom_prefix`` ended in::

    return prefixes.get(symbol, symbol.lower() + 'a')

a *generator* for a table the Blue Book makes CLOSED. Off-table skeletal
elements got an invented morpheme that looks like nomenclature but is not:
``ala`` (Al), ``zna`` (Zn), ``hga`` (Hg), ``asa`` (As), ``sba`` (Sb), ``bia``
(Bi), ``sna`` (Sn), ``pba`` (Pb), ``gea`` (Ge), ``tea`` (Te), ``fea`` (Fe).
Several of those elements have real prefixes that are nothing like the invented
one (``arsa``, ``stiba``, ``bisma``, ``stanna``, ``plumba``, ``germa``,
``tellura``), so this shipped a plausible-looking WRONG name rather than merely
failing to name.

Reachability, measured (not assumed)
------------------------------------
Two families actually shipped a wrong name end-to-end before this change:

* ``C1C[AlH]CC2(C1)CCCCC2`` -> ``2-alaspiro[5.5]undecane`` -- the FABRICATION,
  via ``spiro.name_spiro_system`` -> ``_build_hetero_prefix``.
* ``C1C[AlH]CCC1`` -> ``inane``, ``C1C[AlH]CCCC1`` -> ``epane``,
  ``C1C[AlH]C2CCCC12`` -> ``olane`` -- the SILENT DROP, via the
  Hantzsch-Widman consumers in ``heterocycles.py``, which did
  ``if not hw_prefix: continue`` while the HW stem kept counting the atom.
  The heteroatom vanishes and a bare ring stem ships.

The other off-table metals (Zn, Cd, Hg, Sn, Pb) are intercepted upstream by an
organometallic "<element> compound (not supported)" refusal, and As/Sb/Bi/Ge/Te
were already spelled correctly because ``get_hw_prefix`` is consulted first.
That is why the fabrication is asserted at the FUNCTION level too: an upstream
interception is not a fix, and it is not stable under refactoring.

The two tables are deliberately DIFFERENT -- do not unify them
-------------------------------------------------------------
* **Table 1.5** (P-15.4.1.1, 25 elements) governs GENERAL skeletal replacement:
  heteroacyclic chains, von Baeyer (P-23.3), spiro (P-24.2.4), rings > 10.
  Aluminium is ``alumina``; indium is ``inda``.
* **Table 2.4** (P-22.2.2.1.1, 22 elements) governs Hantzsch-Widman monocycles
  (3-10 members) ONLY. Aluminium is ``aluma``; indium is ``indiga``.
  [BBv2:8245] prints ``aluminium | 3 | aluma`` with the parenthetical
  ``(not alumina)``, and [BBv2:8250] footnote 1 says "Compare with Table 1.5".

So the prefix is a function of *(element, nomenclature context)*, never of the
element alone. A prefix source is therefore correct only relative to its
context, and a von Baeyer/spiro path that reaches into the HW table is a latent
context bug even when today's spellings happen to coincide (they coincide for
all 14 elements both tables share).

Mercury is not merely absent, it is DELETED
-------------------------------------------
[BBv2:8218] **P-22.2.2**: "The elements aluminium, gallium, indium, and thallium
are now included in the recommended Hantzsch-Widman system and mercury has been
deleted." Zn/Cd/Hg appear in NEITHER Table 1.5 nor Table 2.4 (only in the
Appendix 1 *seniority* list), so ``mercura`` must not be offered by a
replacement-prefix accessor at all; organomercury is named by P-69
organometallic nomenclature, which keeps its own ``METALLACYCLE_A_PREFIX``.

Every assertion here runs with the OPSIN jar effectively absent, because SELF-01
fails OPEN in that mode and that is exactly where these wrong names shipped.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym.rules.ring_replacement import (
    HETEROATOM_PREFIXES, build_replacement_prefix,
)


# ---------------------------------------------------------------------------
# The CLOSED legal vocabulary: Table 1.5 (25) union Table 2.4's two divergent
# spellings. Any other morpheme in an 'a'-prefix position is a fabrication.
# ---------------------------------------------------------------------------
TABLE_1_5 = {
    'F': 'fluora', 'Cl': 'chlora', 'Br': 'broma', 'I': 'ioda', 'At': 'astata',
    'O': 'oxa', 'S': 'thia', 'Se': 'selena', 'Te': 'tellura', 'Po': 'polona',
    'N': 'aza', 'P': 'phospha', 'As': 'arsa', 'Sb': 'stiba', 'Bi': 'bisma',
    'C': 'carba', 'Si': 'sila', 'Ge': 'germa', 'Sn': 'stanna', 'Pb': 'plumba',
    'B': 'bora', 'Al': 'alumina', 'Ga': 'galla', 'In': 'inda', 'Tl': 'thalla',
}
# Table 2.4 (HW) diverges for exactly these two; everything else it shares with
# Table 1.5 is spelled identically.
TABLE_2_4_ONLY = {'Al': 'aluma', 'In': 'indiga'}

LEGAL_A_MORPHEMES = set(TABLE_1_5.values()) | set(TABLE_2_4_ONLY.values())

# Elements confirmed to have been given an invented morpheme by the retired
# ``symbol.lower() + 'a'`` fallback, with the Blue-Book-correct Table 1.5 form.
CONFIRMED_FABRICATIONS = {
    'Al': ('ala', 'alumina'),
    'Zn': ('zna', None),          # in NO replacement table
    'Cd': ('cda', None),          # in NO replacement table
    'Hg': ('hga', None),          # DELETED from HW by P-22.2.2
    'As': ('asa', 'arsa'),
    'Sb': ('sba', 'stiba'),
    'Bi': ('bia', 'bisma'),
    'Sn': ('sna', 'stanna'),
    'Pb': ('pba', 'plumba'),
    'Ge': ('gea', 'germa'),
    'Te': ('tea', 'tellura'),
    'Fe': ('fea', None),          # transition metal: no replacement prefix
}


@pytest.fixture
def namer(monkeypatch):
    """A namer with the OPSIN jar made UNAVAILABLE.

    SELF-01 (name -> structure round trip) fails OPEN with no jar -- a supported
    mode -- and that is exactly the mode in which these wrong names shipped. A
    fix verified only with the jar present is not verified.
    """
    import orthonym.namer as _namer_mod
    import orthonym.validation.opsin_roundtrip as _rt
    monkeypatch.setattr(_rt, '_find_opsin_jar', lambda *a, **k: None)
    monkeypatch.setattr(_namer_mod, '_DISABLE_VALIDITY_GATE', True)
    from orthonym import Orthonym
    return Orthonym(general_fallback=True, _disable_opsin_validity_gate=True)


def fabricated_form(symbol: str) -> str:
    """What the retired fallback would have invented for ``symbol``."""
    return symbol.lower() + 'a'


SPIRO_TEMPLATE = 'C1C[{atom}]CC2(C1)CCCCC2'


def spiro_smiles(symbol: str):
    """A spiro[5.5]undecane skeleton with ``symbol`` at ring position 2.

    Several bracket-atom spellings are tried because no single one is valid for
    every element: ``[OH]`` in a ring is a valence error (three bonds on a
    divalent O) while ``[AlH]`` needs the H to be accepted, and a bare ``[N]`` is
    a nitrene rather than the ring NH we want. Forms leaving no radical electron
    are preferred so the sweep tests ordinary closed-shell rings; an accepted
    radical form is used only if nothing else parses. ``None`` means RDKit will
    not put this element in a ring at all, so the sweep skips it.
    """
    fallback = None
    for atom in (f'{symbol}H', symbol, f'{symbol}H2'):
        smi = SPIRO_TEMPLATE.format(atom=atom)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        if radicals == 0:
            return smi
        if fallback is None:
            fallback = smi
    return fallback


def cites_morpheme(name: str, morpheme: str) -> bool:
    """Does ``name`` cite ``morpheme`` in an 'a'-prefix position?

    An 'a' prefix is preceded by a locant, a comma, a hyphen, or starts the
    name, and is followed by the ring descriptor or another prefix. Anchoring on
    the left boundary is what keeps this from matching the same letters inside
    an unrelated stem.
    """
    if not name:
        return False
    return bool(re.search(rf'(?:^|[-,\d]){re.escape(morpheme)}', name))


REFUSAL_MARKERS = ('not supported', 'not currently supported')


def is_refusal(name) -> bool:
    """A refusal: ``None``, empty, or an explicit unsupported-sentinel string."""
    if not name:
        return True
    return any(marker in name for marker in REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# 1. THE REPRODUCER. This is the name that shipped.
# ---------------------------------------------------------------------------
REPRODUCER = 'C1C[AlH]CC2(C1)CCCCC2'


def test_reproducer_does_not_ship_the_fabricated_ala_prefix(namer):
    """``2-alaspiro[5.5]undecane`` -- ``ala`` is not a Blue Book term.

    Aluminium IS in Table 1.5 (``alumina``) and in Table 2.4 (``aluma``), but
    ``ring_replacement.HETEROATOM_PREFIXES`` -- the Table-1.5 source this spiro
    path spells from -- deliberately carries only its 14 verified rows, so the
    correct behaviour here is a REFUSAL, not a renamed prefix. Refusing is the
    sound end state for a closed table: a wrong table row ships a wrong name.
    """
    name = namer.name(REPRODUCER)
    assert not cites_morpheme(name, 'ala'), (
        f"fabricated morpheme 'ala' still shipped: {name!r}"
    )
    # And it must not have silently dropped the aluminium either: a bare
    # hydrocarbon spiro name for an aluminium ring is the other wrong answer.
    assert is_refusal(name) or not cites_morpheme(name, 'spiro'), (
        f"aluminium neither spelled nor refused: {name!r}"
    )


# ---------------------------------------------------------------------------
# 2. The root fabricator itself.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('symbol', sorted(CONFIRMED_FABRICATIONS))
def test_root_fabricator_returns_none_off_table(symbol):
    """``polycyclic_bridged.get_heteroatom_prefix`` must never invent.

    Asserted at the function level on purpose: for most of these elements an
    upstream organometallic refusal currently hides the fabrication, and an
    upstream accident is not a fix.
    """
    from orthonym.rules.polycyclic_bridged import get_heteroatom_prefix
    got = get_heteroatom_prefix(symbol)
    fab = fabricated_form(symbol)
    assert got != fab, f"{symbol}: fabricated {got!r}"
    assert got is None or got in LEGAL_A_MORPHEMES, (
        f"{symbol}: {got!r} is not a Blue Book 'a' prefix"
    )


@pytest.mark.parametrize('symbol', ['Fe', 'U', 'Xx', 'Zn', 'Cd', 'Hg', 'Al'])
def test_root_fabricator_fails_closed_for_everything_off_table(symbol):
    from orthonym.rules.polycyclic_bridged import get_heteroatom_prefix
    assert get_heteroatom_prefix(symbol) is None


@pytest.mark.parametrize('symbol,expected', sorted(
    (s, p) for s, (p, _k) in HETEROATOM_PREFIXES.items()
))
def test_root_fabricator_still_spells_every_in_table_element(symbol, expected):
    """Byte-identity floor: the 14 Table-1.5 rows keep their exact spelling.

    The retired 7-entry local dict (O N S Se P Si B) is a strict subset of the
    canonical table with identical spellings, so routing through the canonical
    table is byte-identical for those AND adds the 7 rows the local copy lacked
    (Te As Sb Bi Ge Sn Pb) instead of inventing them.
    """
    from orthonym.rules.polycyclic_bridged import get_heteroatom_prefix
    assert get_heteroatom_prefix(symbol) == expected


# ---------------------------------------------------------------------------
# 3. Whole-periodic-table sweep at the spiro site (the shipping path).
# ---------------------------------------------------------------------------
def _ring_capable_elements():
    """Every element RDKit will actually accept as a spiro-ring skeletal atom."""
    ok = []
    for z in range(1, 104):
        try:
            sym = Chem.GetPeriodicTable().GetElementSymbol(z)
        except Exception:
            continue
        if sym in ('C', '*'):
            continue
        if spiro_smiles(sym) is not None:
            ok.append(sym)
    return ok


RING_ELEMENTS = _ring_capable_elements()


def test_the_sweep_actually_covers_the_periodic_table():
    """Guard the guard: an empty/whittled sweep would pass vacuously."""
    assert len(RING_ELEMENTS) > 50, len(RING_ELEMENTS)
    assert set(CONFIRMED_FABRICATIONS) <= set(RING_ELEMENTS)


@pytest.mark.parametrize('symbol', RING_ELEMENTS)
def test_spiro_site_never_fabricates_for_any_element(symbol, namer):
    """For EVERY ring-capable element: a legal prefix, or fail closed.

    The invariant is one-directional and total -- the emitted name may not
    contain a morpheme outside the closed Blue Book vocabulary, and an element
    the Table-1.5 source cannot spell may not yield a replacement-style ring
    name at all (which would mean the stem counted an atom no morpheme spells).
    """
    name = namer.name(spiro_smiles(symbol))
    fab = fabricated_form(symbol)
    if fab not in LEGAL_A_MORPHEMES:
        assert not cites_morpheme(name, fab), (
            f"{symbol}: fabricated {fab!r} in {name!r}"
        )
    if symbol not in HETEROATOM_PREFIXES:
        assert is_refusal(name) or not cites_morpheme(name, 'spiro'), (
            f"{symbol}: unspellable by Table 1.5 source yet named {name!r}"
        )


@pytest.mark.parametrize('symbol', sorted(HETEROATOM_PREFIXES))
def test_spiro_site_still_spells_in_table_elements(symbol, namer):
    """No coverage lost: every in-table element still gets its real prefix."""
    name = namer.name(spiro_smiles(symbol))
    if is_refusal(name):
        pytest.skip(f"{symbol}: refused upstream for an unrelated reason")
    expected = HETEROATOM_PREFIXES[symbol][0]
    assert cites_morpheme(name, expected), (
        f"{symbol}: expected {expected!r} in {name!r}"
    )


def test_spiro_prefix_builder_fails_closed_at_function_level():
    """``_build_hetero_prefix`` itself must refuse, not just its callers.

    It has two callers -- ``name_spiro_system`` (the PIN dispatch, tried BEFORE
    the general engine) and ``vonbaeyer_universal.analyze_spiro_universal``. Only
    the latter had a totality guard, which is why the PIN path shipped ``ala``.
    Putting the guard inside the builder makes both callers safe and any future
    third caller safe by construction.
    """
    from orthonym.rules.spiro import _build_hetero_prefix
    from orthonym.perception.rings import get_spiro_atoms
    for symbol in ('Al', 'Zn', 'Hg', 'Fe'):
        mol = Chem.MolFromSmiles(f'C1C[{symbol}H]CC2(C1)CCCCC2')
        assert mol is not None
        spiro_atoms = set(get_spiro_atoms(mol))
        ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
        assert _build_hetero_prefix(mol, spiro_atoms, ring) is None, (
            f"{symbol}: builder returned a prefix for an unspellable element"
        )


def test_spiro_prefix_builder_still_builds_in_table():
    from orthonym.rules.spiro import _build_hetero_prefix
    from orthonym.perception.rings import get_spiro_atoms
    mol = Chem.MolFromSmiles('C1CCC2(CC1)CCOCC2')
    spiro_atoms = set(get_spiro_atoms(mol))
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    assert _build_hetero_prefix(mol, spiro_atoms, ring) == '3-oxa'


def test_spiro_prefix_builder_refuses_unlocated_heteroatom():
    """An in-table element the numbering does not reach is also unexpressed.

    The builder used to ``if locant is not None``-skip such an atom, which is the
    silent-drop shape again: no locant to cite, so no morpheme, while the stem
    still counts it.
    """
    from orthonym.rules.spiro import _build_hetero_prefix
    mol = Chem.MolFromSmiles('C1CCC2(CC1)CCOCC2')
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    # No spiro atom supplied -> no numbering -> nothing can be located.
    assert _build_hetero_prefix(mol, set(), ring) is None


# ---------------------------------------------------------------------------
# 4. The second copy in the same file (site 6, found while auditing).
# ---------------------------------------------------------------------------
def test_spiro_vonbaeyer_a_prefix_fails_closed_off_table():
    """``spiro.py::_spiro_vb_a_prefix`` is a THIRD spiro-context speller.

    P-24.5.2, so a Table 1.5 context. It carried the same
    ``get_hw_prefix(...) or get_heteroatom_prefix(...)`` line and no totality
    guard. Its fabrication was latent (the surrounding path refuses these cages
    for other reasons today), which is precisely why it needs a test: latent is
    not safe, it is undetected.
    """
    from orthonym.rules.spiro import _spiro_vb_a_prefix
    mol = Chem.MolFromSmiles('C1CC2([AlH]C1)CCCCC2')
    assert mol is not None
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    hetero = [i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == 'Al']
    unprimed = {i: n + 1 for n, i in enumerate(sorted(ring))}
    got = _spiro_vb_a_prefix(mol, set(hetero), -1, unprimed, {})
    assert got is None, f"off-table element spelled as {got!r}"


# ---------------------------------------------------------------------------
# 5. Hantzsch-Widman: mercury is DELETED (Table 2.4 context).
# ---------------------------------------------------------------------------
def test_hw_does_not_offer_mercura():
    """P-22.2.2: "mercury has been deleted" [BBv2:8218].

    Hg is in neither Table 1.5 nor Table 2.4. Organomercury routes via P-69,
    whose own ``METALLACYCLE_A_PREFIX`` still carries ``mercura`` -- that path is
    untouched, so nothing legitimate is lost by removing it here.
    """
    from orthonym.data.hw_heteroatoms import get_hw_prefix, HW_PREFIXES
    assert get_hw_prefix('Hg') is None
    assert 'Hg' not in HW_PREFIXES


@pytest.mark.parametrize('symbol', ['Zn', 'Cd', 'Hg', 'Fe', 'U'])
def test_hw_offers_no_prefix_for_elements_in_no_replacement_table(symbol):
    from orthonym.data.hw_heteroatoms import get_hw_prefix
    assert get_hw_prefix(symbol) is None


def test_hw_table_keeps_the_table_2_4_spellings_distinct_from_table_1_5():
    """The two tables must NOT be unified; the divergence is a BB requirement.

    Guard against a future "cleanup" that points the HW consumers at the
    Table-1.5 source (or vice versa). Whatever each table offers for Al/In, it
    must never be the other table's spelling.
    """
    from orthonym.data.hw_heteroatoms import HW_PREFIXES
    for symbol, hw_spelling in TABLE_2_4_ONLY.items():
        if symbol in HW_PREFIXES:
            assert HW_PREFIXES[symbol] == hw_spelling, (
                f"{symbol}: HW table must spell {hw_spelling!r} (Table 2.4), "
                f"not {HW_PREFIXES[symbol]!r}"
            )
        if symbol in HETEROATOM_PREFIXES:
            assert HETEROATOM_PREFIXES[symbol][0] == TABLE_1_5[symbol], (
                f"{symbol}: Table-1.5 source must spell {TABLE_1_5[symbol]!r}"
            )


# ---------------------------------------------------------------------------
# 6. The HW consumers: refuse, do not drop.
# ---------------------------------------------------------------------------
# Before this change these shipped a bare ring stem with the heteroatom gone.
HW_SILENT_DROP_CASES = [
    ('C1C[AlH]CCC1', 'inane'),
    ('C1C[AlH]CCCC1', 'epane'),
    ('C1C[AlH]C2CCCC12', 'olane'),
]


@pytest.mark.parametrize('smiles,shipped_stem', HW_SILENT_DROP_CASES)
def test_hw_consumers_refuse_instead_of_dropping_the_heteroatom(
        smiles, shipped_stem, namer):
    """``if not hw_prefix: continue`` dropped the atom while the stem counted it.

    ``inane`` / ``epane`` / ``olane`` are bare Hantzsch-Widman stems: every
    heteroatom morpheme was skipped and the ring size still came from the full
    atom count. A name that describes a carbocycle for a metallacycle is a
    wrong structure, not a coverage gap.
    """
    name = namer.name(smiles)
    assert name != shipped_stem, f"still ships the stripped stem {name!r}"
    assert is_refusal(name) or 'alum' in name, (
        f"aluminium neither spelled nor refused: {name!r}"
    )


# ---------------------------------------------------------------------------
# 7. The thin wrapper that used to discard the signal (site 5).
# ---------------------------------------------------------------------------
def test_polycyclic_wrapper_reports_unexpressed_instead_of_discarding_it():
    """``get_heteroatom_replacement_prefix`` fed three PIN von Baeyer callers.

    Its old ``-> str`` signature could not report an atom it failed to express,
    so the three callers relied on an unverified assumption that an off-table
    element never reaches them. It now returns ``None`` when the primitive
    reports ``unexpressed``, which gives all three the fail-closed signal
    rather than a proof obligation.
    """
    from orthonym.rules.polycyclic import get_heteroatom_replacement_prefix
    mol = Chem.MolFromSmiles('C1CC2CC[Hg]C2C1')
    assert mol is not None
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    numbering = {i: n + 1 for n, i in enumerate(sorted(ring))}
    # sanity: the primitive really does flag it
    assert build_replacement_prefix(mol, numbering, ring).unexpressed
    assert get_heteroatom_replacement_prefix(mol, numbering, ring) is None


def test_polycyclic_wrapper_still_returns_the_prefix_when_total():
    from orthonym.rules.polycyclic import get_heteroatom_replacement_prefix
    mol = Chem.MolFromSmiles('C1CC2CCOC2C1')
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    numbering = {i: n + 1 for n, i in enumerate(sorted(ring))}
    got = get_heteroatom_replacement_prefix(mol, numbering, ring)
    assert got is not None and 'oxa' in got


def test_polycyclic_wrapper_returns_empty_string_for_a_carbocycle():
    """'' (nothing to express) must stay distinguishable from None (refuse)."""
    from orthonym.rules.polycyclic import get_heteroatom_replacement_prefix
    mol = Chem.MolFromSmiles('C1CC2CCCC2C1')
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    numbering = {i: n + 1 for n, i in enumerate(sorted(ring))}
    assert get_heteroatom_replacement_prefix(mol, numbering, ring) == ''


# ---------------------------------------------------------------------------
# 8. The higher-polycyclo (von Baeyer) speller, site 3.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('symbol', ['Al', 'Zn', 'Hg', 'Fe'])
def test_tricyclo_prefix_generator_fails_closed_off_table(symbol):
    """``tricyclo._generate_heteroatom_prefix`` was a third implementation.

    It had its own hard-coded 7-element ``priority_order``, so Te/As/Sb/Bi/Ge/
    Sn/Pb fell into a trailing "remaining elements" loop that both fabricated
    their morpheme and cited them out of seniority order.
    """
    from orthonym.rules.tricyclo import _generate_heteroatom_prefix
    mol = Chem.MolFromSmiles(f'C1C2C[{symbol}H]C3C1C23')
    if mol is None:
        pytest.skip(f'{symbol} not accepted in this cage by RDKit')
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    hetero = [(i, mol.GetAtomWithIdx(i).GetSymbol()) for i in ring
              if mol.GetAtomWithIdx(i).GetSymbol() != 'C']
    assert _generate_heteroatom_prefix(mol, hetero, ring) is None


def test_tricyclo_prefix_generator_cites_in_seniority_order():
    """Te is in-table and must be spelled ``tellura``, never ``tea``.

    This is the case the review verified broken at function level: a Te cage
    produced ``7-teatetracyclo[...]undecane``.
    """
    from orthonym.rules.tricyclo import _generate_heteroatom_prefix
    mol = Chem.MolFromSmiles('C1C2C[TeH]C3C1C23')
    assert mol is not None
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    hetero = [(i, mol.GetAtomWithIdx(i).GetSymbol()) for i in ring
              if mol.GetAtomWithIdx(i).GetSymbol() != 'C']
    got = _generate_heteroatom_prefix(mol, hetero, ring)
    assert got is None or 'tellura' in got, got
    assert got is None or 'tea' not in got, got


# ---------------------------------------------------------------------------
# 9. The universal analyzers already honour the contract -- lock it.
# ---------------------------------------------------------------------------
def test_both_universal_analyzers_honour_the_unexpressed_contract():
    """Verified by reading, then locked here: cage AND spiro both refuse."""
    from orthonym.rules.vonbaeyer_universal import (
        analyze_cage_universal, analyze_spiro_universal,
    )
    cage = Chem.MolFromSmiles('C1CC2CC[Hg]C2C1')
    assert analyze_cage_universal(cage) is None
    spiro = Chem.MolFromSmiles('C1C[AlH]CC2(C1)CCCCC2')
    assert analyze_spiro_universal(spiro) is None


# ---------------------------------------------------------------------------
# 9b. The ACYCLIC / macrocyclic P-15.4 sibling (rules/skeletal_replacement.py).
#
# Same class, different parent: its collect loops filtered on
# ``symbol in REPLACEMENT_TERMS`` and SKIPPED anything else, while
# ``chain_length``/``ring_size`` kept counting the atom -- so the off-table atom
# was renamed as a CARBON of the stem. Asserted at the PRODUCER level, because
# once this module abstains the molecule falls through to other producers that
# have their own (separate, pre-existing) drop bugs; conflating the two would
# make this test pass or fail for the wrong reason.
# ---------------------------------------------------------------------------
SKELETAL_DROP_CASES = [
    ('CC[Tl]CCSCC', '3-thiaoctane'),
    ('CCS[Zn]SCC', '3,5-dithiaheptane'),
    ('CCOCCOCCOCCOCC[Tl]C', '3,6,9,12-tetraoxahexadecane'),
    ('OCCOCCOCC[Tl]CCOCCOCC', '3,6,12,15-tetraoxaheptadecan-1-ol'),
    ('CC[Tl]CC[Tl]CCSCC', '3-thiaundecane'),
]


@pytest.mark.parametrize('smiles,shipped', SKELETAL_DROP_CASES)
def test_skeletal_replacement_refuses_off_table_backbone(smiles, shipped):
    """The named element must not be absorbed into the carbon stem."""
    from orthonym.rules.skeletal_replacement import try_skeletal_replacement_name
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    got = try_skeletal_replacement_name(mol)
    assert got != shipped, f"still renames the off-table atom as carbon: {got!r}"
    assert got is None, f"expected refusal, got {got!r}"


def test_skeletal_replacement_refuses_off_table_ring():
    """The cyclic (> 10 ring) path takes the same rule."""
    from orthonym.rules.skeletal_replacement import try_skeletal_replacement_name
    mol = Chem.MolFromSmiles('C1CCOCCOCC[Tl]CCOCC1')
    assert mol is not None
    got = try_skeletal_replacement_name(mol)
    assert got != '1,4,10-trioxacyclopentadecane'
    assert got is None, f"expected refusal, got {got!r}"


@pytest.mark.parametrize('smiles,expected', [
    ('CCOCCOCCOCCOC', '2,5,8,11-tetraoxatridecane'),
    ('C1CCOCCOCCCCOCC1', '1,4,9-trioxacyclotetradecane'),
])
def test_skeletal_replacement_still_names_in_table_backbones(smiles, expected):
    """The gate must cost NOTHING on in-table elements -- byte-identical."""
    from orthonym.rules.skeletal_replacement import try_skeletal_replacement_name
    mol = Chem.MolFromSmiles(smiles)
    assert try_skeletal_replacement_name(mol) == expected


def test_skeletal_totality_predicate_is_element_keyed_not_terminal_keyed():
    """The old terminator gate (P-51.4.1.4) inspected only the two chain ENDS.

    An off-table atom in the MIDDLE therefore passed. This asserts the predicate
    covers interior atoms, which is the whole difference.
    """
    from orthonym.rules.skeletal_replacement import (
        _skeletal_atoms_all_expressible,
    )
    mol = Chem.MolFromSmiles('CC[Tl]CCSCC')
    assert mol is not None
    interior = [a.GetIdx() for a in mol.GetAtoms()]
    assert _skeletal_atoms_all_expressible(mol, interior) is False
    ok = Chem.MolFromSmiles('CCOCCOCC')
    assert _skeletal_atoms_all_expressible(
        ok, [a.GetIdx() for a in ok.GetAtoms()]) is True


# ---------------------------------------------------------------------------
# 10. TRIPWIRE: a sixth copy must not appear silently.
# ---------------------------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[3] / 'src' / 'orthonym'

# ``symbol.lower() + 'a'`` and its spelling variants, in code (not comments).
FABRICATION_PATTERNS = [
    # .lower() + 'a'   /   .lower()+"a"
    re.compile(r"\.lower\(\)\s*\+\s*['\"]a['\"]"),
    # f"{sym.lower()}a"
    re.compile(r"\{[^{}]*\.lower\(\)\}a['\"]"),
    # '%sa' % symbol.lower()
    re.compile(r"['\"]%sa['\"]\s*%"),
]


def _code_lines(path: Path):
    """Yield (lineno, line) for EXECUTABLE lines only.

    Comments and DOCSTRINGS are skipped, because the retired fallback is
    deliberately quoted in the docstrings that explain why it was removed --
    ``ring_replacement``, ``polycyclic_bridged``, ``spiro``, ``tricyclo`` and this
    module all do that, and those references must not trip the wire. A
    ``startswith('#')`` test is not enough: the quotes sit mid-docstring.

    Only COMMENT tokens and TRIPLE-QUOTED strings are masked -- deliberately NOT
    every ``STRING`` token. Masking all strings by line is what the first version
    of this helper did, and it made the tripwire incapable of ever firing: the
    fabrication being hunted is ``symbol.lower() + 'a'``, whose own ``'a'``
    literal is a STRING token on that very line, so every offending line masked
    itself. The mutation test (plant a fabrication, expect a failure) is what
    exposed that; without it this tripwire would have been decorative.
    """
    import io
    import tokenize as _tok

    text = path.read_text()
    lines = text.splitlines()
    masked = {}
    try:
        for tok in _tok.generate_tokens(io.StringIO(text).readline):
            is_docstring = (tok.type == _tok.STRING
                            and tok.string.lstrip('rbuRBUf')[:3] in ('"""', "'''"))
            if tok.type == _tok.COMMENT or is_docstring:
                for n in range(tok.start[0], tok.end[0] + 1):
                    masked[n] = True
    except (_tok.TokenError, IndentationError, SyntaxError):
        # Unparseable file: report everything rather than silently exempting it.
        masked = {}
    for lineno, line in enumerate(lines, 1):
        if masked.get(lineno):
            continue
        yield lineno, line


def test_the_tripwire_scanner_can_actually_see_code():
    """Meta-test: the scanner must NOT mask a line just because it has a string.

    Guards the exact defect described in ``_code_lines``. A tripwire that cannot
    see ``x.lower() + 'a'`` is worse than no tripwire, because it reports safety.
    """
    import tempfile
    src = (
        '"""Docstring mentioning symbol.lower() + \'a\' which must be IGNORED."""\n'
        '\n'
        '\n'
        'def offender(symbol):\n'
        "    return symbol.lower() + 'a'\n"
        '\n'
        '\n'
        'def fine(symbol):\n'
        "    # a comment mentioning symbol.lower() + 'a', also IGNORED\n"
        '    return TABLE.get(symbol)\n'
    )
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as fh:
        fh.write(src)
        tmp = Path(fh.name)
    try:
        hits = [(n, l) for n, l in _code_lines(tmp)
                if any(p.search(l) for p in FABRICATION_PATTERNS)]
        assert len(hits) == 1, f"expected exactly the code line, got {hits}"
        assert hits[0][0] == 5, hits
    finally:
        tmp.unlink()


def test_no_module_fabricates_a_replacement_prefix_from_an_element_symbol():
    """Tripwire. The Blue Book's 'a'-prefix tables are CLOSED lists.

    There is no rule that derives a prefix from a symbol -- ``S`` is ``thia``,
    ``Sb`` is ``stiba``, ``Pb`` is ``plumba``. Any code that synthesises one is
    manufacturing nomenclature, so the pattern is banned outright rather than
    reviewed case by case.
    """
    offenders = []
    for py in sorted(SRC_ROOT.rglob('*.py')):
        for lineno, line in _code_lines(py):
            for pattern in FABRICATION_PATTERNS:
                if pattern.search(line):
                    offenders.append(
                        f"{py.relative_to(SRC_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "element-symbol -> 'a'-prefix FABRICATION found. The Blue Book tables "
        "are closed; return None and let the caller fail closed instead:\n"
        + "\n".join(offenders)
    )


def test_the_table_1_5_replacement_source_lives_in_exactly_one_place():
    """No new module may declare its own von Baeyer/spiro 'a'-prefix table.

    Divergent copies are what produced this defect: a 7-entry dict in
    ``polycyclic_bridged``, a 19-entry HW dict reached from a Table-1.5 context,
    and a 7-element ``priority_order`` in ``tricyclo`` all disagreed with the
    canonical 14-row table and with each other.
    """
    canonical = SRC_ROOT / 'rules' / 'ring_replacement.py'
    declaration = re.compile(r"^\s*_?[A-Z][A-Z0-9_]*\s*[:=].*\{\s*$")
    offenders = []
    for py in sorted(SRC_ROOT.rglob('*.py')):
        if py == canonical:
            continue
        lines = list(_code_lines(py))
        for idx, (lineno, line) in enumerate(lines):
            if not declaration.match(line):
                continue
            # Look at the dict body: a replacement table is recognisable by
            # mapping several element symbols to 'a'-term morphemes.
            body = ' '.join(l for _n, l in lines[idx + 1: idx + 40])
            hits = {m for m in LEGAL_A_MORPHEMES
                    if re.search(rf"['\"]{m}['\"]", body)}
            if len(hits) >= 5:
                offenders.append(
                    f"{py.relative_to(SRC_ROOT)}:{lineno}: {line.strip()} "
                    f"(declares {len(hits)} 'a' morphemes)")
    allowed = {
        # Table 2.4 -- the Hantzsch-Widman context, deliberately a SEPARATE
        # table with different spellings (aluma/indiga). See P-22.2.2.1.1.
        'data/hw_heteroatoms.py',
        # P-15.4 ACYCLIC chain replacement (2,5,8-trioxanonane): same
        # nomenclature family, different parent class and numbering source.
        'rules/skeletal_replacement.py',
        # P-69.4 organometallic metallacycles -- a different nomenclature
        # system entirely, and the legitimate home of Hg/Zn/Cd.
        'data/organometallics.py',
    }
    unexpected = [o for o in offenders
                  if not any(o.startswith(a + ':') for a in allowed)]
    assert not unexpected, (
        "new replacement-prefix table declared outside "
        "rules/ring_replacement.py; import HETEROATOM_PREFIXES instead:\n"
        + "\n".join(unexpected)
    )
