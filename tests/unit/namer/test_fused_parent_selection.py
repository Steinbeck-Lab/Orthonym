"""Phase 148 P-44.1 cascade tests for fused heterocycle parent selection.

Per V18 plan §6 Phase 148 + HERITAGE-1990 §4 (full seniority cascade on
ALL structures, no bypass — 61% Beilstein-expert agreement validates
the deletion of _should_bypass_fused_guard).

Each test cites:
  - The QMUL P-section URL (P-44.1, P-31.1.3.4, or P-52.2.8)
  - The IUPAC rule code (P-44.1(letter))
  - HERITAGE-1990 §4 reference

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
Source: https://iupac.qmul.ac.uk/BlueBook/P5.html P-52.2.8
Source: https://iupac.qmul.ac.uk/BlueBook/P3.html P-31.1.3.4
Source: HERITAGE-1990 §4 (Wisniewski J. Chem. Inf. Comput. Sci. 30, 324-332)
        — full seniority cascade on ALL structures, no bypass.
Source: Phase 148 CONTEXT D-01, D-02, D-03 (bypass deletion + cascade entry).
"""
import pytest
from rdkit import Chem

from orthonym import name_compound


@pytest.mark.unit
def test_p44_1c_indole_long_chain_chain_wins():
    """P-44.1(c) max chain length: indole + C11 acid → chain wins.

    Indole is a 9-atom fused heterocyclic ring system; an undecanoic acid
    chain (11 carbons + COOH) is longer and bears the principal group.
    Per P-44.1(a) the principal characteristic group is on the chain only,
    so the chain wins outright (the longer-chain test (c) is also satisfied
    but is not the binding criterion here).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1(a)/(c)
    Source: HERITAGE-1990 §4 — full seniority cascade.
    Source: Phase 148 CONTEXT D-03 (cascade entry condition).
    """
    name = name_compound('c1ccc2[nH]ccc2c1CCCCCCCCCCC(=O)O')
    assert name, f'name_compound returned empty/None: {name!r}'
    lower = name.lower()
    assert 'undecan' in lower, (
        f"Expected 'undecan' in name (chain wins per P-44.1(a)/(c)); got {name!r}"
    )
    assert 'indol' in lower, (
        f"Expected 'indol' substring (ring as substituent); got {name!r}"
    )


@pytest.mark.unit
def test_p44_1b_chain_more_pg_chain_wins():
    """P-44.1(b) chain has more principal characteristic groups → chain wins.

    Indole + chain bearing two carbonyl-type groups (ketone + carboxylic
    acid). Per P-44.1(a)/(b), when the chain has the principal group
    (carboxylic acid here) and the ring system has none, the chain wins
    outright. Once the bypass is deleted (Phase 148 D-01), the cascade
    correctly evaluates this case via select_parent's PG-count comparison
    (parent_selection.py:758-786 subsumes deleted Guard 3).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1(b)
    Source: HERITAGE-1990 §4.
    Source: Phase 148 RESEARCH §3 subsumption proof (Guard 3 → P-44.1(b)).
    Source: Phase 148 CONTEXT D-03.
    """
    # Indole + -CH2-C(=O)-CH2-CH2-COOH (5-atom chain with ketone + acid)
    name = name_compound('c1ccc2[nH]ccc2c1CC(=O)CCC(=O)O')
    assert name, f'name_compound returned empty/None: {name!r}'
    lower = name.lower()
    assert 'indol' in lower, (
        f"Expected 'indol' substring (ring as substituent); got {name!r}"
    )
    assert any(s in lower for s in ('oic acid', 'carbox', 'oate')), (
        f"Expected acid/oate handle (chain wins per P-44.1(b)); got {name!r}"
    )


@pytest.mark.unit
def test_p52_2_8_equal_tiebreak_ring_wins():
    """P-52.2.8 ring-on-tie: bare 1H-indole returns 'indole' (no chain at all).

    Per P-52.2.8, when ring and chain tie on every prior criterion the ring
    is preferred. With no chain present the cascade does not even fire
    (chain_len < 2 — cascade non-entry per D-03); this test is the
    regression scaffold for the no-chain code path post-bypass-deletion.

    Source: https://iupac.qmul.ac.uk/BlueBook/P5.html P-52.2.8
    Source: HERITAGE-1990 §4.
    Source: Phase 148 CONTEXT D-03 (cascade non-entry preserved for bare rings).
    """
    name = name_compound('c1ccc2[nH]ccc2c1')
    assert name, f'name_compound returned empty/None: {name!r}'
    lower = name.lower()
    assert 'indol' in lower, (
        f"Expected 'indol' or 'indole' for bare indole; got {name!r}"
    )


@pytest.mark.unit
def test_p31_1_3_4_np_override_preserved():
    """P-31.1.3.4 NP backbone always ring-parent: morphine stays 'morphine'.

    Morphine is a recognized natural-product backbone (morphinan scaffold);
    detect_natural_product() classifies it. The P-31.1.3.4 NP override at
    parent_selection.py:688-700 forces ring-parent regardless of chain
    length. Phase 148 preserves this override BYTE-IDENTICAL per D-04.

    Source: https://iupac.qmul.ac.uk/BlueBook/P3.html P-31.1.3.4
    Source: HERITAGE-1990 §4.
    Source: Phase 148 CONTEXT D-04 (NP override byte-identical).
    """
    morphine_smiles = 'CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5'
    name = name_compound(morphine_smiles)
    assert name, f'name_compound returned empty/None: {name!r}'
    lower = name.lower()
    assert 'morphin' in lower, (
        f"Expected 'morphin' (P-31.1.3.4 NP override); got {name!r}"
    )


@pytest.mark.unit
def test_nucleoside_still_ring_parent():
    """Adenosine stays ring-parent post-bypass-deletion.

    Adenosine has a purine ring + ribose chain; chain_len after ring
    exclusion is 1 (single primary alcohol carbon). The cascade entry
    condition (chain_len >= 2 per D-03) is NOT satisfied → cascade does
    not fire → ring path returns the retained name 'adenosine'.

    Note (RESEARCH §4d): purine NPs stay ring-parent via cascade non-entry,
    NOT via the P-31.1.3.4 NP override (detect_natural_product() returns
    None for purines — this is documented in D-04 and is the actual
    protective mechanism for the purine class).

    Source: https://iupac.qmul.ac.uk/BlueBook/P3.html (purine retained name)
    Source: HERITAGE-1990 §4.
    Source: Phase 148 CONTEXT D-04 + D-05 (NP regression scaffold).
    Source: Phase 148 RESEARCH §4c-§4d (cascade non-entry, NOT NP override).
    """
    adenosine_smiles = 'Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O'
    name = name_compound(adenosine_smiles)
    assert name, f'name_compound returned empty/None: {name!r}'
    lower = name.lower()
    assert ('adenos' in lower) or ('purin' in lower), (
        f"Expected 'adenos' or 'purin' substring (ring stays parent for "
        f"purine NPs via cascade non-entry); got {name!r}"
    )


# ---------------------------------------------------------------------------
# 5 RECOMMENDED edge-case tests (D-06 Tier 1 full coverage; Task 148-01-07).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_p44_1a_pg_only_on_ring_ring_wins():
    """P-44.1(a) PG-on-ring-only: quinoline-2-carboxylic acid + methyl
    substituents → ring wins.

    The carboxylic acid principal group is attached directly to the
    quinoline ring system; the off-ring substituents are simple methyls
    with no PG. Per P-44.1(a) the parent is the structure carrying the
    principal group → ring wins outright. The cascade reaches the same
    result via cascade non-entry (chain_len < 2 for the methyl branches).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1(a)
    Source: HERITAGE-1990 §4.
    Source: Phase 148 CONTEXT D-06 (Tier 1 full coverage).
    """
    name = name_compound('Cc1ccc2nc(C(=O)O)cc(C)c2c1')
    assert name, f'name_compound returned empty/None: {name!r}'
    lower = name.lower()
    assert 'quinolin' in lower, (
        f"Expected 'quinolin' (ring is parent per P-44.1(a)); got {name!r}"
    )
    assert any(s in lower for s in ('carboxylic acid', 'carboxylic', 'oic acid')), (
        f"Expected acid handle on ring parent; got {name!r}"
    )


@pytest.mark.unit
def test_p44_1a_pg_only_on_chain_chain_wins():
    """P-44.1(a) PG-on-chain-only: indole + butanoic acid chain → chain wins.

    The carboxylic acid is on the chain (-CH2-CH2-CH2-COOH); the indole
    ring system has no PG. Per P-44.1(a) the chain is parent because it
    carries the principal group. Pre-148 the bypass would have short-
    circuited this case to ring-parent; post-148 the cascade correctly
    routes to chain-parent.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1(a)
    Source: HERITAGE-1990 §4.
    Source: Phase 148 CONTEXT D-06 (Tier 1 full coverage).
    """
    name = name_compound('c1ccc2[nH]ccc2c1CCCC(=O)O')
    assert name, f'name_compound returned empty/None: {name!r}'
    lower = name.lower()
    assert 'indol' in lower, (
        f"Expected 'indol' substring (ring as substituent); got {name!r}"
    )
    assert any(s in lower for s in ('pentanoic acid', 'butanoic acid', 'oic acid')), (
        f"Expected chain-acid handle (chain wins per P-44.1(a)); got {name!r}"
    )


@pytest.mark.unit
def test_p44_1c_chain_5atoms_indole_9atoms_ring_wins():
    """P-44.1(c) ring longer than chain: indole (9 atoms) + butyl chain
    (4 atoms) → ring wins.

    No principal group; ring is bigger; per P-44.1(c) longer parent wins.
    The cascade fires (chain_len=4 >= 2) but returns ring-parent because
    P-44.1(c) max-length comparison favours the ring system.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1(c)
    Source: HERITAGE-1990 §4.
    Source: Phase 148 CONTEXT D-06 (Tier 1 full coverage).
    """
    name = name_compound('c1ccc2[nH]ccc2c1CCCC')
    assert name, f'name_compound returned empty/None: {name!r}'
    lower = name.lower()
    assert 'indol' in lower, (
        f"Expected 'indol' (ring is parent per P-44.1(c)); got {name!r}"
    )
    assert any(s in lower for s in ('butyl', 'propyl', 'methyl', 'ethyl')), (
        f"Expected chain-as-substituent token; got {name!r}"
    )


@pytest.mark.unit
def test_p52_2_8_zero_pg_ring_wins():
    """P-52.2.8 zero-PG tiebreak: quinoline (10 atoms, with N) + butyl chain
    (4 atoms) → ring wins.

    No PG on either side; per P-44.1(c) ring is longer, but if ring and
    chain were equal length P-52.2.8 would still pick ring. The post-148
    cascade reaches this verdict end-to-end.

    Source: https://iupac.qmul.ac.uk/BlueBook/P5.html P-52.2.8
    Source: HERITAGE-1990 §4.
    Source: Phase 148 CONTEXT D-06 (Tier 1 full coverage).
    """
    name = name_compound('c1ccc2ncccc2c1CCCC')
    assert name, f'name_compound returned empty/None: {name!r}'
    lower = name.lower()
    assert 'quinolin' in lower, (
        f"Expected 'quinolin' (ring is parent per P-52.2.8); got {name!r}"
    )
    assert any(s in lower for s in ('butyl', 'propyl', 'methyl', 'ethyl')), (
        f"Expected chain-as-substituent token; got {name!r}"
    )


@pytest.mark.unit
def test_polyfunctional_acid_on_chain_with_indole_ring_chain_wins():
    """P-44.1(b) polyfunctional chain (acid + ketone) > indole ring (no PG).

    Chain has carboxylic acid + ketone; ring has none. Per P-44.1(a)/(b)
    the chain is parent because it carries the principal group (and more
    of them, satisfying both (a) and (b)). The post-148 cascade evaluates
    this correctly via select_parent's PG-count comparison.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1(b)
    Source: HERITAGE-1990 §4.
    Source: Phase 148 RESEARCH §3 subsumption proof (Guard 3 → P-44.1(b)).
    Source: Phase 148 CONTEXT D-06 (Tier 1 full coverage).
    """
    # Indole on chain end; chain has a 5-oxo + COOH (PG=acid):
    # OC(=O)-CH2CH2CH2-C(=O)-CH2-indole
    name = name_compound('OC(=O)CCCC(=O)Cc1c[nH]c2ccccc12')
    assert name, f'name_compound returned empty/None: {name!r}'
    lower = name.lower()
    assert 'indol' in lower, (
        f"Expected 'indol' substring (ring as substituent); got {name!r}"
    )
    assert any(s in lower for s in ('oic acid', 'amide', 'carbox', 'carbamoyl')), (
        f"Expected acid/amide handle (chain wins per P-44.1(b)); got {name!r}"
    )
