"""Blue Book Table 1.5, whole: which rows we may emit, and why not the rest.

v29 Phase 2, T2b. Before this change the ring 'a'-replacement table carried 14 of
Table 1.5's 25 rows and the other 11 were simply absent -- sound (they failed
closed) but undecided, so nobody could tell "not implemented" from "the Blue Book
gives us no rule". This suite makes the disposition of every one of the 25 rows
explicit and cited.

The trap this suite locks: THREE orders, THREE different element sets
---------------------------------------------------------------------
=============  ============================================  ========  ==========
rule           governs                                       elements  citation
=============  ============================================  ========  ==========
P-15.4.1.2     general / chains, "naming and numbering"        **25**   [BBv2:6446]
P-23.3.1       citation order INSIDE a von Baeyer name         **22**   [BBv2:9765]
P-23.3.2.2     numbering seniority when there IS a choice      **18**   [BBv2:9789]
=============  ============================================  ========  ==========

Emitting a replacement prefix needs BOTH a citation position and a numbering rank,
so the emittable set is the intersection: P-23.3.2.2's eighteen. Reading a
P-15.4.1.2 position for an element P-23.3.1 never ranks would be inventing a rule,
which is why ``At``/``Po``/``C`` fail closed despite having Table 1.5 morphemes.

What T2b actually changed, and what it deliberately did not
----------------------------------------------------------
* **+4 emittable**: Al ``alumina``, Ga ``galla``, In ``inda``, Tl ``thalla`` --
  the four elements P-22.2.2 [BBv2:8218] ADDED in the same sentence that deleted
  mercury. Both P-23.3.1 and P-23.3.2.2 rank all four, and their standard bonding
  number is 3, the same as boron's, which already worked. No lambda is involved.
* **7 refused, with a reason each**: At/Po/C (ranked by neither von Baeyer rule)
  and F/Cl/Br/I (ranked for citation but NOT for numbering -- and a skeletal RING
  halogen always has a nonstandard bonding number, so P-15.4.1.3 makes its lambda
  mandatory while the ring lambda helper suppresses exactly that
  connectivity-forced case). The halogens are marked NOT ESTABLISHED in-code.
* **Zn/Cd/Hg unchanged**: in NO replacement table, so still off Table 1.5 itself.
* **The Table 1.5 / Table 2.4 divergence is now LIVE**, not latent: both tables
  carry Al and In, and they disagree on purpose (``aluma`` "(not alumina)"
  [BBv2:8245], ``indiga`` "(not inda)", footnote "Compare with Table 1.5"
  [BBv2:8250]). A prefix is a function of *(element, nomenclature context)*.
* The ACYCLIC chain table (``rules/skeletal_replacement.REPLACEMENT_TERMS``) was
  deliberately NOT extended: whether replacement or the substitutive parent
  hydride (``alumane``/``gallane``/``indigane``/``thallane``, Table 2.1
  [BBv2:7924-7930]) is the PIN for a Group-13 atom in a CHAIN is a P-51.4
  selection question T2b did not answer, so that path still fails closed.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym.data.hw_heteroatoms import (
    HETEROATOM_PRIORITY, HW_PREFIXES, get_hw_prefix,
    sort_heteroatoms_by_priority,
)
from orthonym.rules.lambda_convention import STANDARD_BONDING_NUMBER
from orthonym.rules.polycyclic_bridged import get_heteroatom_prefix
from orthonym.rules.ring_replacement import (
    HETEROATOM_PREFIXES, TABLE_1_5, VB_CITATION_ORDER, VB_INADMISSIBLE,
    VB_NUMBERING_SENIORITY, build_replacement_prefix,
)

# ---------------------------------------------------------------------------
# Independently transcribed expectations. These are NOT imported from the code
# under test: they are re-read from [BBv2:6436-6443] (the 5x5 grid, whose column
# headers are the standard bonding numbers), [BBv2:9765] and [BBv2:9789]. A test
# that reads its expectation from the implementation cannot catch a wrong row.
# ---------------------------------------------------------------------------
BB_TABLE_1_5 = {
    'B': ('bora', 3), 'C': ('carba', 4), 'N': ('aza', 3),
    'O': ('oxa', 2), 'F': ('fluora', 1),
    'Al': ('alumina', 3), 'Si': ('sila', 4), 'P': ('phospha', 3),
    'S': ('thia', 2), 'Cl': ('chlora', 1),
    'Ga': ('galla', 3), 'Ge': ('germa', 4), 'As': ('arsa', 3),
    'Se': ('selena', 2), 'Br': ('broma', 1),
    'In': ('inda', 3), 'Sn': ('stanna', 4), 'Sb': ('stiba', 3),
    'Te': ('tellura', 2), 'I': ('ioda', 1),
    'Tl': ('thalla', 3), 'Pb': ('plumba', 4), 'Bi': ('bisma', 3),
    'Po': ('polona', 2), 'At': ('astata', 1),
}

# P-23.3.1 [BBv2:9765] verbatim.
BB_P_23_3_1 = (
    'F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'N', 'P', 'As', 'Sb', 'Bi',
    'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga', 'In', 'Tl',
)
# P-23.3.2.2 [BBv2:9789] verbatim.
BB_P_23_3_2_2 = (
    'O', 'S', 'Se', 'Te', 'N', 'P', 'As', 'Sb', 'Bi',
    'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga', 'In', 'Tl',
)

T2B_ADDED = ('Al', 'Ga', 'In', 'Tl')
NO_VON_BAEYER_RULE = ('At', 'Po', 'C')
HALOGENS = ('F', 'Cl', 'Br', 'I')
NO_REPLACEMENT_TABLE_AT_ALL = ('Zn', 'Cd', 'Hg', 'Fe', 'U')


@pytest.fixture
def namer(monkeypatch):
    """A namer with the OPSIN jar made UNAVAILABLE.

    SELF-01 (name -> structure round trip) fails OPEN with no jar -- a supported
    mode -- so a source-level refusal must hold on its own. Mirrors the fixture in
    ``test_v29_replacement_prefix_totality`` on purpose: an OPSIN-verified name is
    evidence about the WITH-jar path only.
    """
    import orthonym.namer as _namer_mod
    import orthonym.validation.opsin_roundtrip as _rt
    monkeypatch.setattr(_rt, '_find_opsin_jar', lambda *a, **k: None)
    monkeypatch.setattr(_namer_mod, '_DISABLE_VALIDITY_GATE', True)
    from orthonym import Orthonym
    return Orthonym(general_fallback=True, _disable_opsin_validity_gate=True)


def _refused(name) -> bool:
    return (not name) or 'not supported' in name or name == 'unknown organic compound'


# ---------------------------------------------------------------------------
# 1. The book's table, whole.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_table_1_5_has_all_twenty_five_rows():
    assert len(TABLE_1_5) == 25, sorted(TABLE_1_5)
    assert set(TABLE_1_5) == set(BB_TABLE_1_5)


@pytest.mark.unit
@pytest.mark.parametrize('symbol', sorted(BB_TABLE_1_5))
def test_table_1_5_row_matches_the_book(symbol):
    """Prefix AND standard bonding number, per row, against [BBv2:6436-6443]."""
    assert TABLE_1_5[symbol] == BB_TABLE_1_5[symbol]


@pytest.mark.unit
def test_every_table_1_5_row_has_an_explicit_disposition():
    """No row may be silently absent: emitted, or refused with a stated reason.

    This is the property T2b exists to establish. Adding a row to ``TABLE_1_5``
    without deciding its von Baeyer status fails here.
    """
    decided = set(HETEROATOM_PREFIXES) | set(VB_INADMISSIBLE)
    assert decided == set(TABLE_1_5), {
        'undecided': sorted(set(TABLE_1_5) - decided),
        'not in table': sorted(decided - set(TABLE_1_5)),
    }
    assert not (set(HETEROATOM_PREFIXES) & set(VB_INADMISSIBLE)), (
        "an element cannot be both emittable and inadmissible"
    )
    for symbol, reason in VB_INADMISSIBLE.items():
        assert reason.strip(), f"{symbol}: empty reason"


# ---------------------------------------------------------------------------
# 2. The three orders and their three different SETS.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_citation_order_is_p_23_3_1_verbatim():
    assert VB_CITATION_ORDER == BB_P_23_3_1
    assert len(VB_CITATION_ORDER) == 22


@pytest.mark.unit
def test_numbering_seniority_is_p_23_3_2_2_verbatim():
    assert VB_NUMBERING_SENIORITY == BB_P_23_3_2_2
    assert len(VB_NUMBERING_SENIORITY) == 18


@pytest.mark.unit
def test_the_three_sets_differ_exactly_as_the_book_does():
    """At/Po/C absent from P-23.3.1; those three AND the halogens from P-23.3.2.2."""
    assert set(TABLE_1_5) - set(VB_CITATION_ORDER) == set(NO_VON_BAEYER_RULE)
    assert (set(VB_CITATION_ORDER) - set(VB_NUMBERING_SENIORITY)
            == set(HALOGENS))


@pytest.mark.unit
def test_numbering_order_is_a_subsequence_of_the_citation_order():
    """One integer key can serve both orders only if this holds.

    ``HETEROATOM_PREFIXES``'s rank is used as the citation order (P-23.3.1) by
    ``build_replacement_prefix`` and as the numbering rank (P-23.3.2.2) by the
    numbering consumers. That is sound only because dropping the halogens from
    P-23.3.1 leaves P-23.3.2.2 in the same relative order.
    """
    position = {el: i for i, el in enumerate(VB_CITATION_ORDER)}
    assert list(VB_NUMBERING_SENIORITY) == sorted(
        VB_NUMBERING_SENIORITY, key=position.__getitem__)


# ---------------------------------------------------------------------------
# 3. The emittable set = the intersection, and the pre-T2b keys did not move.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_emittable_set_is_citation_intersect_numbering():
    assert (set(HETEROATOM_PREFIXES)
            == set(VB_CITATION_ORDER) & set(VB_NUMBERING_SENIORITY))
    assert len(HETEROATOM_PREFIXES) == 18


@pytest.mark.unit
def test_ranks_follow_the_numbering_seniority_order():
    ordered = sorted(HETEROATOM_PREFIXES, key=lambda e: HETEROATOM_PREFIXES[e][1])
    assert tuple(ordered) == VB_NUMBERING_SENIORITY


@pytest.mark.unit
def test_the_pre_t2b_fourteen_keep_ranks_1_to_14():
    """Extending the table must not renumber -- or reorder -- what already shipped."""
    legacy = ('O', 'S', 'Se', 'Te', 'N', 'P', 'As', 'Sb', 'Bi',
              'Si', 'Ge', 'Sn', 'Pb', 'B')
    assert [HETEROATOM_PREFIXES[e][1] for e in legacy] == list(range(1, 15))


@pytest.mark.unit
@pytest.mark.parametrize('symbol,expected', [
    ('Al', 'alumina'), ('Ga', 'galla'), ('In', 'inda'), ('Tl', 'thalla'),
])
def test_t2b_added_rows_carry_the_table_1_5_spelling(symbol, expected):
    assert HETEROATOM_PREFIXES[symbol][0] == expected
    assert get_heteroatom_prefix(symbol) == expected


# ---------------------------------------------------------------------------
# 4. The refusals, by reason.
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize('symbol', NO_VON_BAEYER_RULE + HALOGENS)
def test_inadmissible_rows_are_not_emittable(symbol):
    """A Table 1.5 morpheme exists, but no ring context may spell it."""
    assert symbol in TABLE_1_5
    assert symbol in VB_INADMISSIBLE
    assert symbol not in HETEROATOM_PREFIXES
    assert get_heteroatom_prefix(symbol) is None


@pytest.mark.unit
def test_carba_is_ruled_out_by_the_book_not_just_by_the_orders():
    """P-21.1.1.1 [BBv2:7915]: "The name 'carbane' ... is not recommended.\"

    Recorded separately because ``C`` is the one inadmissible row with a SECOND,
    independent Blue Book reason -- so a future session that finds a carbon
    citation position still must not enable it here.
    """
    assert 'carbane' in VB_INADMISSIBLE['C']
    assert 'not' in VB_INADMISSIBLE['C']


@pytest.mark.unit
@pytest.mark.parametrize('symbol', HALOGENS)
def test_halogen_refusal_reason_records_both_blockers(symbol):
    """NOT ESTABLISHED, not merely unimplemented -- and the record must say why.

    Two independent blockers: no P-23.3.2.2 numbering rank, and a mandatory
    lambda (P-15.4.1.3) we cannot currently spell. A future reader must be able
    to tell this from an ordinary "we did not get to it".
    """
    reason = VB_INADMISSIBLE[symbol]
    if reason.strip() == "see 'F'":
        reason = VB_INADMISSIBLE['F']
    assert 'P-23.3.2.2' in reason
    assert 'lambda' in reason.lower()


@pytest.mark.unit
@pytest.mark.parametrize('symbol', NO_REPLACEMENT_TABLE_AT_ALL)
def test_elements_in_no_replacement_table_are_not_even_in_table_1_5(symbol):
    """The fail-closed FLOOR: Zn/Cd/Hg are a different refusal from At/Po/C.

    They have no morpheme anywhere in replacement nomenclature -- P-22.2.2
    DELETED mercury -- so they must not appear in the book's table either, let
    alone in the emittable set. ``mercura`` lives only in the P-69 organometallic
    table.
    """
    assert symbol not in TABLE_1_5
    assert symbol not in HETEROATOM_PREFIXES
    assert symbol not in VB_INADMISSIBLE
    assert get_heteroatom_prefix(symbol) is None
    assert get_hw_prefix(symbol) is None


# ---------------------------------------------------------------------------
# 5. Table 1.5 vs Table 2.4: the divergence is LIVE and must stay locked.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_hw_table_covers_exactly_the_p_23_3_1_element_set():
    """Table 2.4's 22 rows are the same 22 elements P-23.3.1 ranks [BBv2:8236].

    A cross-check on both transcriptions at once: two independently printed Blue
    Book tables agreeing element-for-element is strong evidence neither was
    mis-copied.
    """
    assert set(HW_PREFIXES) == set(VB_CITATION_ORDER)


@pytest.mark.unit
@pytest.mark.parametrize('symbol,hw,table_1_5', [
    ('Al', 'aluma', 'alumina'),
    ('In', 'indiga', 'inda'),
])
def test_the_two_tables_diverge_where_the_book_says_they_do(symbol, hw, table_1_5):
    """[BBv2:8245] prints ``aluma`` with an explicit "(not alumina)".

    This is the test that must NOT be "cleaned up" by pointing one table at the
    other. The prefix is keyed on (element, context).
    """
    assert HW_PREFIXES[symbol] == hw
    assert get_hw_prefix(symbol) == hw
    assert TABLE_1_5[symbol][0] == table_1_5
    assert HETEROATOM_PREFIXES[symbol][0] == table_1_5
    assert hw != table_1_5


@pytest.mark.unit
@pytest.mark.parametrize('symbol', ['Ga', 'Tl'])
def test_the_two_tables_agree_where_the_book_agrees(symbol):
    """Only Al and In diverge; asserting the divergence is exactly two rows wide."""
    assert HW_PREFIXES[symbol] == TABLE_1_5[symbol][0]


@pytest.mark.unit
def test_exactly_two_rows_diverge_between_the_tables():
    diverging = {s for s in set(HW_PREFIXES) & set(TABLE_1_5)
                 if HW_PREFIXES[s] != TABLE_1_5[s][0]}
    assert diverging == {'Al', 'In'}


# ---------------------------------------------------------------------------
# 6. The determinism hazard T2b had to fix alongside the table.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_priority_table_ranks_every_hw_element():
    """In and Tl were MISSING here while B..Ga were present.

    ``get_heteroatom_priority`` returns 999 for an unknown element, so In and Tl
    TIED with each other -- and ``sort_heteroatoms_by_priority`` is a stable sort
    over a list built by iterating a ``set`` of ring atoms, which made the
    citation order depend on set iteration order. Harmless while neither element
    could be spelled; a nondeterministic NAME the moment one could.
    """
    missing = [e for e in VB_CITATION_ORDER if e not in HETEROATOM_PRIORITY]
    assert not missing, missing
    ranks = [HETEROATOM_PRIORITY[e] for e in VB_CITATION_ORDER]
    assert len(set(ranks)) == len(ranks), "duplicate ranks would re-tie"
    assert ranks == sorted(ranks), "priority order must match P-23.3.1 / Table 2.4"


@pytest.mark.unit
def test_priority_sort_separates_indium_from_thallium():
    assert sort_heteroatoms_by_priority(['Tl', 'In']) == ['In', 'Tl']
    assert sort_heteroatoms_by_priority(['In', 'Tl']) == ['In', 'Tl']


@pytest.mark.unit
def test_mercury_rank_is_not_a_replacement_position():
    """Hg keeps a value for the P-69 consumers but must rank BELOW every real row.

    If it ever sorted above In/Tl again it would re-create the tie this fixed.
    """
    assert HETEROATOM_PRIORITY['Hg'] > max(
        HETEROATOM_PRIORITY[e] for e in VB_CITATION_ORDER)
    assert get_hw_prefix('Hg') is None
    assert 'Hg' not in TABLE_1_5


# ---------------------------------------------------------------------------
# 7. Standard bonding numbers come from the book, not a periodic-table default.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_lambda_table_agrees_with_table_1_5_on_every_shared_element():
    disagreements = {
        e: (n, TABLE_1_5[e][1]) for e, n in STANDARD_BONDING_NUMBER.items()
        if e in TABLE_1_5 and n != TABLE_1_5[e][1]
    }
    assert not disagreements, disagreements


@pytest.mark.unit
def test_lambda_table_omits_carbon_and_astatine_on_purpose():
    """Adding a row here changes emitted NAMES, so the omission is deliberate.

    ``'C': 4`` would let any neutral carbon this helper is handed acquire a
    spurious lambda -- a radical carbon has total valence 3. Neither C nor At is
    emittable as a ring 'a' prefix, so an entry would buy nothing.
    """
    assert set(TABLE_1_5) - set(STANDARD_BONDING_NUMBER) == {'C', 'At'}


@pytest.mark.unit
@pytest.mark.parametrize('symbol', T2B_ADDED)
def test_added_elements_have_boron_s_standard_bonding_number(symbol):
    """All four are Group 13 with standard bonding number 3, like B.

    B already worked, so no new lambda behaviour is reachable through them: a
    ring atom of degree 2 or 3 with valence 3 is standard and cites no lambda.
    """
    assert TABLE_1_5[symbol][1] == 3 == TABLE_1_5['B'][1]
    assert STANDARD_BONDING_NUMBER[symbol] == 3


# ---------------------------------------------------------------------------
# 8. The primitive: spells the four, reports the rest unexpressed.
# ---------------------------------------------------------------------------
def _ring_prefix(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    Chem.Kekulize(mol, clearAromaticFlags=True)
    ring = sorted(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
    numbering = {idx: i + 1 for i, idx in enumerate(ring)}
    return build_replacement_prefix(mol, numbering, set(ring))


@pytest.mark.unit
@pytest.mark.parametrize('symbol,morpheme', [
    ('Al', 'alumina'), ('Ga', 'galla'), ('In', 'inda'), ('Tl', 'thalla'),
])
def test_primitive_spells_the_added_elements_and_leaves_nothing_unexpressed(
        symbol, morpheme):
    result = _ring_prefix(f'C1CC2CC[{symbol}H]C2C1')
    assert result.unexpressed == (), result
    assert morpheme in result.prefix
    assert [m for _idx, m in result.per_atom] == [morpheme]


@pytest.mark.unit
@pytest.mark.parametrize('symbol,morpheme', [
    ('O', 'oxa'), ('Al', 'alumina'), ('Ga', 'galla'),
    ('In', 'inda'), ('Tl', 'thalla'),
])
def test_the_tricyclo_speller_also_reaches_the_added_rows(symbol, morpheme):
    """The four Table-1.5 consumer sites must not diverge again.

    ``tricyclo._generate_heteroatom_prefix`` is the site that historically kept
    its OWN 7-element ``priority_order`` and fabricated morphemes for everything
    outside it. It now reads the shared table, so extending that table has to
    reach here too -- otherwise Al/Ga/In/Tl would name in a bicyclo and refuse in
    a tricyclo, which is the sibling-drift shape this milestone keeps hitting.
    """
    from orthonym.rules.tricyclo import _generate_heteroatom_prefix
    mol = Chem.MolFromSmiles(f'C1C2C[{symbol}H]C3C1C23')
    if mol is None:
        pytest.skip(f'{symbol} not accepted in this cage by RDKit')
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    hetero = [(i, mol.GetAtomWithIdx(i).GetSymbol()) for i in ring
              if mol.GetAtomWithIdx(i).GetSymbol() != 'C']
    got = _generate_heteroatom_prefix(mol, hetero, ring)
    assert got is not None and morpheme in got, got


@pytest.mark.unit
def test_primitive_reports_a_ring_halogen_as_unexpressed():
    """Never a name: the ring stem would count an atom no morpheme spells."""
    result = _ring_prefix('C1CC2CC[IH]C2C1')
    assert len(result.unexpressed) == 1, result
    assert 'ioda' not in result.prefix


@pytest.mark.unit
@pytest.mark.parametrize('smiles', ['C1CC2CC[AtH]C2C1', 'C1CC2CC[PoH]C2C1',
                                    'C1CC2CC[ZnH]C2C1', 'C1CC2CC[Hg]C2C1'])
def test_primitive_reports_at_po_and_the_off_table_metals_as_unexpressed(smiles):
    result = _ring_prefix(smiles)
    assert len(result.unexpressed) == 1, result
    for morpheme in ('astata', 'polona', 'zna', 'mercura', 'hga'):
        assert morpheme not in result.prefix


# ---------------------------------------------------------------------------
# 9. End to end, with the OPSIN jar absent.
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize('smiles,expected', [
    ('[AlH]1CC2CCC1CC2', '2-aluminabicyclo[2.2.2]octane'),
    ('[GaH]1CC2CCC1CC2', '2-gallabicyclo[2.2.2]octane'),
    ('[InH]1CC2CCC1CC2', '2-indabicyclo[2.2.2]octane'),
    ('[TlH]1CC2CCC1CC2', '2-thallabicyclo[2.2.2]octane'),
    ('C12C[AlH]CC(CC1)C2', '3-aluminabicyclo[3.2.1]octane'),
    ('C1C[AlH]CC2(C1)CCCCC2', '2-aluminaspiro[5.5]undecane'),
    ('[AlH]1C[AlH]C2CCC1CC2', '2,4-dialuminabicyclo[3.2.2]nonane'),
    ('C1CCCCCC[AlH]CCCCC1', 'aluminacyclotridecane'),
    ('C1CCCCCC[InH]CCCCC1', 'indacyclotridecane'),
])
def test_added_elements_name_end_to_end(smiles, expected, namer):
    """Every one of these is also OPSIN round-trip verified (see the T2b report).

    Byte-exact, so a spelling, locant or stray-hyphen regression fails here even
    with no jar present. The bicyclo[2.2.2] and bicyclo[3.2.1] skeletons are the
    two shipped gold von Baeyer targets with the heteroatom swapped, and the
    13-membered ring exercises the ring > 10 Table-1.5 path in ``heterocycles``.
    """
    assert namer.name(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize('element,expected', [
    ('O', 'oxacyclotridecane'), ('S', 'thiacyclotridecane'),
    ('Al', 'aluminacyclotridecane'), ('In', 'indacyclotridecane'),
])
def test_sole_heteroatom_locant_is_omitted_in_a_saturated_macrocycle(
        element, expected):
    """P-22.2.3.2.1 [BBv2:8494]: the sole heteroatom's '1' is OMITTED.

    Asserted on the ring>10 builder directly, because the defect this locks was
    invisible end to end: ``_build_replacement_name`` emitted ``1-<prefix>`` for
    EVERY element, but O/S/N/Si never reach it -- an earlier producer that already
    omits the locant handles them, which is why the shipped gold targets
    ``thiacyclododecane`` and ``azacyclotridecane`` (same citation) were green.
    Al and In started falling through to this builder when T2b made them
    spellable, so without the fix T2b would have shipped
    ``1-aluminacyclotridecane`` beside ``oxacyclotridecane``.
    """
    from orthonym.rules.heterocycles import _build_replacement_name
    assert _build_replacement_name(
        [(1, element)], 13, True, [], bare_ring=True) == expected


@pytest.mark.unit
@pytest.mark.parametrize('element,expected', [
    ('O', '1-oxacyclotridecane'), ('Se', '1-selenacyclotridecane'),
    ('Al', '1-aluminacyclotridecane'),
])
def test_sole_heteroatom_locant_is_KEPT_when_anything_else_is_numbered(
        element, expected):
    """``bare_ring=False`` -> keep the locant. This is the regression guard.

    The shipped gold target ``1-selenacyclotridecan-3-one``
    (W2-RINGKET-SELENA-PROTECT) is a SATURATED, single-heteroatom macrocycle that
    KEEPS its '1', because a suffix locant is cited. This builder returns only the
    ``...ane`` parent -- its caller appends ``-3-one`` -- so an omission decided
    from these arguments alone cannot see the suffix, and a first version of this
    fix regressed exactly that target (caught by an A/B pre-gate against the base
    commit: 1641/1641 -> 1640/1641). Hence ``bare_ring`` must be asserted by the
    caller, which alone knows the molecule is nothing but the ring.
    """
    from orthonym.rules.heterocycles import _build_replacement_name
    assert _build_replacement_name(
        [(1, element)], 13, True, [], bare_ring=False) == expected


@pytest.mark.unit
def test_the_selena_ring_ketone_gold_target_still_holds():
    """End-to-end guard on the target the tightened condition exists to protect."""
    from orthonym import Orthonym
    got = Orthonym(general_fallback=True).name('O=C1C[Se]CCCCCCCCCC1')
    assert got == '1-selenacyclotridecan-3-one', got


@pytest.mark.unit
def test_unsaturated_macrocycle_keeps_the_sole_heteroatom_locant():
    """The omission is scoped to the SATURATED case, per the book's own PINs.

    ``1-oxacycloundeca-2,4,6,8,10-pentaene`` (PIN) and
    ``1-azacyclotetradeca-1,3,5,7,9,11,13-heptaene`` (PIN) [BBv2:8488-8490] both
    RETAIN the '1'. A broader fix that dropped it there would contradict them.
    """
    from orthonym.rules.heterocycles import _build_replacement_name
    got = _build_replacement_name([(1, 'O')], 11, False, [2, 4, 6, 8, 10])
    assert got is not None and got.startswith('1-oxa'), got


@pytest.mark.unit
def test_multi_heteroatom_macrocycle_still_cites_every_locant():
    """Only the SOLE-heteroatom case loses its locant (12-crown-4 keeps all four)."""
    from orthonym.rules.heterocycles import _build_replacement_name
    got = _build_replacement_name(
        [(1, 'O'), (4, 'O'), (7, 'O'), (10, 'O')], 12, True, [])
    assert got == '1,4,7,10-tetraoxacyclododecane', got


@pytest.mark.unit
@pytest.mark.parametrize('smiles', [
    'C1CC2CC[IH]C2C1', 'C1C[IH]CC2(C1)CCCCC2',
    'C1CC2CC[AtH]C2C1', 'C1CC2CC[PoH]C2C1',
    'C1CC2CC[ZnH]C2C1', 'C1CC2CC[CdH]C2C1', 'C1CC2CC[Hg]C2C1',
])
def test_inadmissible_and_off_table_ring_atoms_still_refuse(smiles, namer):
    name = namer.name(smiles)
    assert _refused(name), f"expected a refusal, got {name!r}"
    for morpheme in ('ioda', 'astata', 'polona', 'carba', 'mercura'):
        assert morpheme not in (name or '')


@pytest.mark.unit
def test_senior_heteroatom_is_cited_first_even_when_it_holds_the_higher_locant():
    """P-23.3.1 citation order is NOT ascending-locant order.

    Here aluminium holds locant 2 and oxygen locant 7, yet ``oxa`` (rank 1) must
    be cited before ``alumina`` (rank 15). A locant-sorted implementation, or one
    that gave the new rows a rank above oxygen, would emit
    ``2-alumina-7-oxabicyclo[3.2.2]nonane``. OPSIN round-trip verified.
    """
    name = _namer_name('O1CC2CC[AlH]C1CC2')
    assert name == '7-oxa-2-aluminabicyclo[3.2.2]nonane', name
    assert name.index('oxa') < name.index('alumina')


@pytest.mark.unit
def test_boron_is_cited_before_aluminium():
    """B rank 14 < Al rank 15: the new rows go AFTER boron, per P-23.3.1."""
    name = _namer_name('C1CC2(CC[AlH]1)CC[BH]CC2')
    assert name == '3-bora-9-aluminaspiro[5.5]undecane', name


@pytest.mark.unit
@pytest.mark.parametrize('smiles', ['C1CC2(CC[InH]1)CC[TlH]CC2',
                                    'C1CC2(CC[TlH]1)CC[InH]CC2'])
def test_indium_thallium_citation_order_is_deterministic(smiles):
    """The concrete case the missing In/Tl priority ranks would have broken.

    Both SMILES describe the same molecule, and both orderings of the ring-atom
    set must produce the identical name. With In and Tl both falling through to
    priority 999 this depended on set iteration order.
    """
    assert _namer_name(smiles) == '3-inda-9-thallaspiro[5.5]undecane'


# ---------------------------------------------------------------------------
# 10. The divergence, behaviourally: same element, two contexts, two spellings.
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize('hw_smiles,hw_name,vb_smiles,vb_name', [
    ('C1CC[AlH]CC1', 'aluminane',
     '[AlH]1CC2CCC1CC2', '2-aluminabicyclo[2.2.2]octane'),
    ('C1CC[InH]CC1', 'indiginane',
     '[InH]1CC2CCC1CC2', '2-indabicyclo[2.2.2]octane'),
])
def test_same_element_spells_differently_per_context(
        hw_smiles, hw_name, vb_smiles, vb_name, namer):
    """``aluma``+HW stem vs ``alumina``+von Baeyer descriptor, from ONE element.

    This is the divergence made observable in emitted names rather than only in
    the tables: a six-membered monocycle is Hantzsch-Widman (Table 2.4) and a
    von Baeyer cage is Table 1.5. Both round-trip through OPSIN, whose own token
    files carry the same split (``hwHeteroAtoms.xml`` has ``aluma``/``indiga``;
    ``heteroAtoms.xml`` has ``alumina``/``inda``) -- an independent corroboration
    of the transcription.
    """
    assert namer.name(hw_smiles) == hw_name
    assert namer.name(vb_smiles) == vb_name


def _namer_name(smiles: str) -> str:
    """Name with the OPSIN jar absent, without taking the fixture as an arg.

    Duplicated rather than fixtured only where a test needs one call; the
    monkeypatching is identical to the ``namer`` fixture above.
    """
    import orthonym.namer as _namer_mod
    import orthonym.validation.opsin_roundtrip as _rt
    saved_find = _rt._find_opsin_jar
    saved_gate = _namer_mod._DISABLE_VALIDITY_GATE
    _rt._find_opsin_jar = lambda *a, **k: None
    _namer_mod._DISABLE_VALIDITY_GATE = True
    try:
        from orthonym import Orthonym
        return Orthonym(general_fallback=True,
                         _disable_opsin_validity_gate=True).name(smiles)
    finally:
        _rt._find_opsin_jar = saved_find
        _namer_mod._DISABLE_VALIDITY_GATE = saved_gate
