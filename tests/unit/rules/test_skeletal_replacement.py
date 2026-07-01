"""
Unit tests for skeletal replacement ("a") nomenclature.

Tests IUPAC P-15.4 replacement naming for chains with embedded heteroatoms.
"""

import pytest
from rdkit import Chem

from orthonym.rules.skeletal_replacement import (
    try_skeletal_replacement_name,
    REPLACEMENT_TERMS,
)
from orthonym import name_compound


# ============================================================================
# Module-level tests
# ============================================================================

@pytest.mark.unit
class TestReplacementTerms:
    """Test the REPLACEMENT_TERMS table."""

    def test_oxa_present(self):
        assert REPLACEMENT_TERMS['O'] == 'oxa'

    def test_aza_present(self):
        assert REPLACEMENT_TERMS['N'] == 'aza'

    def test_thia_present(self):
        assert REPLACEMENT_TERMS['S'] == 'thia'

    def test_selena_present(self):
        assert REPLACEMENT_TERMS['Se'] == 'selena'

    def test_phospha_present(self):
        assert REPLACEMENT_TERMS['P'] == 'phospha'

    def test_sila_present(self):
        assert REPLACEMENT_TERMS['Si'] == 'sila'

    def test_arsa_present(self):
        assert REPLACEMENT_TERMS['As'] == 'arsa'

    def test_stiba_present(self):
        assert REPLACEMENT_TERMS['Sb'] == 'stiba'

    def test_bisma_present(self):
        assert REPLACEMENT_TERMS['Bi'] == 'bisma'

    def test_germa_present(self):
        assert REPLACEMENT_TERMS['Ge'] == 'germa'

    def test_stanna_present(self):
        assert REPLACEMENT_TERMS['Sn'] == 'stanna'

    def test_plumba_present(self):
        assert REPLACEMENT_TERMS['Pb'] == 'plumba'

    def test_total_element_count(self):
        assert len(REPLACEMENT_TERMS) == 14

    def test_tellura_present(self):
        assert REPLACEMENT_TERMS['Te'] == 'tellura'

    def test_bora_present(self):
        assert REPLACEMENT_TERMS['B'] == 'bora'


# ============================================================================
# Phase 154.A D-03: PIN-trigger function (P-15.4.1.2)
# ============================================================================

@pytest.mark.unit
class TestPinTrigger:
    """D-03: strict IUPAC P-15.4.1.2 PIN trigger function tests.

    Tests `_qualifies_for_pin_skeletal_replacement(backbone, mol)` which
    replaces the legacy single-hetero chain-len < 6 inline reject with
    explicit branch labels per IUPAC Blue Book P-15.4.1.2.

    Source: 154-CONTEXT.md D-03; 154-RESEARCH.md §3.2; 154-AUDIT-A.md §7.
    Source: IUPAC Blue Book 2013 P-15.4.1.2.
    """

    @pytest.mark.parametrize("smiles,backbone_indices,expected_qualifies,expected_branch", [
        # Branch (a): >= 4 same-kind heteroatoms
        # COCOCOCOC: positions 0=C, 1=O, 2=C, 3=O, 4=C, 5=O, 6=C, 7=O, 8=C
        # Embedded (skip 0 and 8): O O O O at positions 1,3,5,7 -- 4 same-kind
        ("COCOCOCOC", [0, 1, 2, 3, 4, 5, 6, 7, 8], True, ">=4-same-kind"),
        # Branch (b): >= 3 mixed-kind heteroatoms
        # CSCNCOC: 0=C, 1=S, 2=C, 3=N, 4=C, 5=O, 6=C  (S+N+O = 3 distinct)
        ("CSCNCOC", [0, 1, 2, 3, 4, 5, 6], True, ">=3-mixed-kind"),
        # Single hetero long chain (legacy gate-5 accept):
        # CCCOCCCC: 0..7, embedded O at index 3, len=8 >= 6
        ("CCCOCCCC", [0, 1, 2, 3, 4, 5, 6, 7], True, "single-hetero-long-chain"),
        # Single hetero short chain (legacy gate-5 reject):
        # COC: 0=C, 1=O, 2=C, embedded O at index 1, len=3 < 6
        ("COC", [0, 1, 2], False, "single-hetero-short-chain"),
        # No heteroatoms in embedded positions:
        # CCCCC: 0=C, 1=C, 2=C, 3=C, 4=C
        ("CCCCC", [0, 1, 2, 3, 4], False, "no-heteroatoms"),
    ])
    def test_branch_assignment(self, smiles, backbone_indices, expected_qualifies, expected_branch):
        from orthonym.rules.skeletal_replacement import _qualifies_for_pin_skeletal_replacement
        mol = Chem.MolFromSmiles(smiles)
        qualifies, branch = _qualifies_for_pin_skeletal_replacement(backbone_indices, mol)
        assert qualifies == expected_qualifies, (
            f"smiles={smiles}, expected_qualifies={expected_qualifies}, got={qualifies}, branch={branch}"
        )
        assert branch == expected_branch, (
            f"smiles={smiles}, expected_branch={expected_branch}, got={branch}"
        )


@pytest.mark.unit
class TestPinTriggerEndToEnd:
    """D-03 end-to-end: confirm helper is wired into try_skeletal_replacement_name.

    Verifies the helper is called from the gate-5 position in
    `try_skeletal_replacement_name` (replacing the inline reject) and that
    the existing happy-path semantics are preserved.
    """

    @pytest.mark.parametrize("smiles,expected_name", [
        # branch (a) >= 4 same-kind oxygens
        ("COCOCOCOC", "2,4,6,8-tetraoxanonane"),
        # two-hetero accept path with terminal-OH suffix
        ("OCCOCCOCC", "3,6-dioxaoctan-1-ol"),
    ])
    def test_accepts(self, smiles, expected_name):
        from orthonym.rules.skeletal_replacement import try_skeletal_replacement_name
        mol = Chem.MolFromSmiles(smiles)
        assert try_skeletal_replacement_name(mol) == expected_name

    @pytest.mark.parametrize("smiles", [
        "COC",     # single-hetero-short-chain reject
        "CCC",     # no-heteroatoms reject
        "CCCCC",   # no-heteroatoms reject (longer)
    ])
    def test_rejects(self, smiles):
        from orthonym.rules.skeletal_replacement import try_skeletal_replacement_name
        mol = Chem.MolFromSmiles(smiles)
        assert try_skeletal_replacement_name(mol) is None


# ============================================================================
# Unsaturated large heterocyclic ring replacement naming
# ============================================================================

@pytest.mark.unit
class TestUnsaturatedCyclicReplacement:
    """Test unsaturated large heterocyclic ring replacement naming."""

    def test_oxacycloheptadiene(self):
        """7-member ring with O and 2 double bonds."""
        mol = Chem.MolFromSmiles('C1=CC=COCC1')
        result = try_skeletal_replacement_name(mol)
        assert result is not None
        assert 'oxa' in result
        assert 'diene' in result

    def test_oxacyclooctatriene(self):
        """8-member ring with O and 3 double bonds."""
        mol = Chem.MolFromSmiles('O1C=CC=CC=CC1')
        result = try_skeletal_replacement_name(mol)
        assert result is not None
        assert 'oxa' in result
        assert 'triene' in result

    def test_oxacyclooctadiene(self):
        """8-member ring with O and 2 double bonds."""
        mol = Chem.MolFromSmiles('C1=CC=COCCC1')
        result = try_skeletal_replacement_name(mol)
        assert result is not None
        assert 'oxa' in result
        assert 'diene' in result

    def test_saturated_still_works(self):
        """Regression: saturated dioxacyclononane unchanged."""
        mol = Chem.MolFromSmiles('C1COCCOCCC1')
        result = try_skeletal_replacement_name(mol)
        assert result is not None
        assert 'ane' in result
        assert 'ene' not in result

    def test_aromatic_6member_not_replacement(self):
        """Aromatic rings should NOT get replacement naming (6-member < 7)."""
        mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine, 6-member aromatic
        result = try_skeletal_replacement_name(mol)
        assert result is None

    def test_no_double_vowel(self):
        """Name should not have double vowels at prefix-suffix junction."""
        mol = Chem.MolFromSmiles('C1=CC=COCCC1')  # 8-member, O, 2 DB
        result = try_skeletal_replacement_name(mol)
        assert result is not None
        # Should not contain "octaadiene" or "octaene" - should be "octadiene"
        assert 'aa' not in result
        assert 'ae' not in result


# ============================================================================
# Basic dioxa chains
# ============================================================================

@pytest.mark.unit
class TestDioxaChains:
    """Test replacement naming for chains with embedded oxygen atoms."""

    def test_dioxahexane(self):
        """R4 / P-12.1 / P-63.2.4: COCCOC has 2 embedded O-ethers, no terminal -ol.
        The substitutive PIN is '1,2-dimethoxyethane', so try_skeletal_replacement_name
        returns None (hands off to the substitutive namer).  Updated from the old
        '2,5-dioxahexane' assertion which was pre-R4 behaviour."""
        mol = Chem.MolFromSmiles('COCCOC')
        assert try_skeletal_replacement_name(mol) is None

    def test_dioxaoctane(self):
        """R4 / P-12.1 / P-63.2.4: CCOCCOCC has 2 embedded O-ethers, no terminal -ol.
        The substitutive PIN is '1,2-diethoxyethane', so try_skeletal_replacement_name
        returns None.  Updated from the old '3,6-dioxaoctane' assertion (pre-R4)."""
        mol = Chem.MolFromSmiles('CCOCCOCC')
        assert try_skeletal_replacement_name(mol) is None

    def test_trioxanonane(self):
        """COCCOCCOC -> 2,5,8-trioxanonane"""
        mol = Chem.MolFromSmiles('COCCOCCOC')
        assert try_skeletal_replacement_name(mol) == '2,5,8-trioxanonane'

    def test_trioxaundecane(self):
        """CCOCCOCCOCC -> 3,6,9-trioxaundecane"""
        mol = Chem.MolFromSmiles('CCOCCOCCOCC')
        assert try_skeletal_replacement_name(mol) == '3,6,9-trioxaundecane'


# ============================================================================
# Aza chains
# ============================================================================

@pytest.mark.unit
class TestAzaChains:
    """Test replacement naming for chains with embedded nitrogen atoms."""

    def test_azahexane(self):
        """P-62.2.2: acyclic amine N bonded only to C → substitutive preferred.
        CCNCCC: the amine gate (Gate 2c) returns None so skeletal replacement
        does NOT produce '3-azahexane'; the substitutive handler gives
        'N-ethylpropan-1-amine' instead (tested in TestEndToEnd.test_azahexane_e2e)."""
        mol = Chem.MolFromSmiles('CCNCCC')
        assert try_skeletal_replacement_name(mol) is None

    def test_diazaheptane(self):
        """P-62.2.2: acyclic N–C–N chain still blocked by amine gate (Gate 2c).
        CNCCNCC: both N atoms are bonded only to C → Gate 2c fires → None.
        The substitutive multi-amine form is not yet implemented (fail-closed)."""
        mol = Chem.MolFromSmiles('CNCCNCC')
        assert try_skeletal_replacement_name(mol) is None

    def test_single_n_5atom_not_replacement(self):
        """CCNCC -> None (single N, chain = 5, below threshold of 6)"""
        mol = Chem.MolFromSmiles('CCNCC')
        assert try_skeletal_replacement_name(mol) is None


# ============================================================================
# Thia chains
# ============================================================================

@pytest.mark.unit
class TestThiaChains:
    """Test replacement naming for chains with embedded sulfur atoms."""

    def test_dithiahexane(self):
        """CSCCSC -> 2,5-dithiahexane"""
        mol = Chem.MolFromSmiles('CSCCSC')
        assert try_skeletal_replacement_name(mol) == '2,5-dithiahexane'


# ============================================================================
# Mixed heteroatom chains
# ============================================================================

@pytest.mark.unit
class TestMixedHeteroatomChains:
    """Test replacement naming for chains with different heteroatom types."""

    def test_oxa_aza_octane(self):
        """P-62.2.2: mixed oxa+aza chain — amine gate (Gate 2c) blocks skeletal replacement.
        COCCNCCC: the N is bonded only to C atoms → Gate 2c fires → None.
        The skeletal '2-oxa-5-azaoctane' is NOT the PIN; a substitutive
        mixed-heteroatom form is not yet assembled (fail-closed → None)."""
        mol = Chem.MolFromSmiles('COCCNCCC')
        assert try_skeletal_replacement_name(mol) is None


# ============================================================================
# Boundary cases -- should return None
# ============================================================================

@pytest.mark.unit
class TestBoundaryCases:
    """Test cases that should NOT trigger skeletal replacement."""

    def test_dimethyl_ether_too_short(self):
        """COC -> None (3-atom chain, single O, too short)"""
        mol = Chem.MolFromSmiles('COC')
        assert try_skeletal_replacement_name(mol) is None

    def test_diethyl_ether_too_short(self):
        """COCC -> None (4-atom chain, single O, too short)"""
        mol = Chem.MolFromSmiles('COCC')
        assert try_skeletal_replacement_name(mol) is None

    def test_ethoxyethane_single_o_5atom(self):
        """CCOCC -> None (5-atom chain, single O, uses substitutive naming)"""
        mol = Chem.MolFromSmiles('CCOCC')
        assert try_skeletal_replacement_name(mol) is None

    def test_ethanol_terminal_oh(self):
        """CCO -> None (terminal OH = alcohol)"""
        mol = Chem.MolFromSmiles('CCO')
        assert try_skeletal_replacement_name(mol) is None

    def test_acetic_acid_priority_fg(self):
        """CC(=O)O -> None (carboxylic acid is priority FG)"""
        mol = Chem.MolFromSmiles('CC(=O)O')
        assert try_skeletal_replacement_name(mol) is None

    def test_benzene_cyclic(self):
        """c1ccccc1 -> None (cyclic molecule)"""
        mol = Chem.MolFromSmiles('c1ccccc1')
        assert try_skeletal_replacement_name(mol) is None

    def test_cyclohexane_cyclic(self):
        """C1CCCCC1 -> None (cyclic molecule)"""
        mol = Chem.MolFromSmiles('C1CCCCC1')
        assert try_skeletal_replacement_name(mol) is None

    def test_acetaldehyde_priority_fg(self):
        """CC=O -> None (aldehyde is priority FG)"""
        mol = Chem.MolFromSmiles('CC=O')
        assert try_skeletal_replacement_name(mol) is None

    def test_acetone_priority_fg(self):
        """CC(=O)C -> None (ketone is priority FG)"""
        mol = Chem.MolFromSmiles('CC(=O)C')
        assert try_skeletal_replacement_name(mol) is None

    def test_methylamine_terminal_nh2(self):
        """CN -> None (terminal NH2 = amine)"""
        mol = Chem.MolFromSmiles('CN')
        assert try_skeletal_replacement_name(mol) is None

    def test_branched_chain_rejected(self):
        """COCCOCC(C)C -> None (has substituent branch)"""
        mol = Chem.MolFromSmiles('COCCOCC(C)C')
        assert try_skeletal_replacement_name(mol) is None

    def test_none_molecule(self):
        """None mol -> None"""
        assert try_skeletal_replacement_name(None) is None

    def test_pure_hydrocarbon(self):
        """CCCCCC -> None (no heteroatoms)"""
        mol = Chem.MolFromSmiles('CCCCCC')
        assert try_skeletal_replacement_name(mol) is None

    def test_nitrile_priority_fg(self):
        """CC#N -> None (nitrile is priority FG)"""
        mol = Chem.MolFromSmiles('CC#N')
        assert try_skeletal_replacement_name(mol) is None

    def test_thiol_terminal(self):
        """CCS -> None (terminal SH = thiol)"""
        mol = Chem.MolFromSmiles('CCS')
        assert try_skeletal_replacement_name(mol) is None


# ============================================================================
# End-to-end via name_compound
# ============================================================================

@pytest.mark.unit
class TestEndToEnd:
    """Test skeletal replacement via the full name_compound pipeline."""

    def test_dioxahexane_e2e(self):
        """R4 / P-12.1 / P-63.2.4: COCCOC (2 embedded O-ethers, no -ol suffix) is
        named substitutively as '1,2-dimethoxyethane', NOT skeletal '2,5-dioxahexane'.
        Updated from pre-R4 assertion."""
        assert name_compound('COCCOC') == '1,2-dimethoxyethane'

    def test_azahexane_e2e(self):
        """P-62.2.2: CCNCCC → substitutive 'N-ethylpropan-1-amine', not '3-azahexane'.
        Amine Gate 2c blocks skeletal replacement; substitutive handler produces the PIN."""
        assert name_compound('CCNCCC') == 'N-ethylpropan-1-amine'

    def test_dioxaoctane_e2e(self):
        """R4 / P-12.1 / P-63.2.4: CCOCCOCC (2 embedded O-ethers, no -ol suffix) is
        named substitutively as '1,2-diethoxyethane', NOT skeletal '3,6-dioxaoctane'.
        Updated from pre-R4 assertion."""
        assert name_compound('CCOCCOCC') == '1,2-diethoxyethane'

    def test_trioxanonane_e2e(self):
        """name_compound('COCCOCCOC') -> '2,5,8-trioxanonane'"""
        assert name_compound('COCCOCCOC') == '2,5,8-trioxanonane'

    def test_mixed_oxa_aza_e2e(self):
        """P-62.2.2: COCCNCCC — amine Gate 2c blocks skeletal '2-oxa-5-azaoctane'.
        Substitutive handler names the amine substitutively:
        'N-2-methoxyethylmethoxypropan-1-amine' (methoxy substituents on propan-1-amine)."""
        assert name_compound('COCCNCCC') == 'N-2-methoxyethylmethoxypropan-1-amine'

    def test_coc_not_replacement(self):
        """COC should NOT produce a replacement name."""
        result = name_compound('COC')
        assert 'oxa' not in result
        assert result == 'methoxymethane'

    def test_ethanol_not_replacement(self):
        """CCO should still be 'ethanol'."""
        assert name_compound('CCO') == 'ethanol'

    def test_dithiahexane_e2e(self):
        """name_compound('CSCCSC') -> '2,5-dithiahexane'"""
        assert name_compound('CSCCSC') == '2,5-dithiahexane'


# ============================================================================
# Phase 154.A D-07: cyclic skeletal replacement size-class regression tests
# ============================================================================

@pytest.mark.unit
class TestCyclicReplacement:
    """D-07: confirm _try_cyclic_replacement_name covers >= 7-member hetero rings.

    Pulls the audit-verified fixtures from 154-AUDIT-A.md §6 (4 corpus +
    2 Blue Book = 6 cyclic-large fixtures spanning ring sizes 9, 12, 14, 18).
    These ALL pass OPSIN-RT InChI L1 match per the audit; this regression
    guard ensures `_try_cyclic_replacement_name` never silently regresses
    on these size classes.

    Rings 4-6 stay routed to Hantzsch-Widman per P-22.2.1 (verified by
    test_hantzsch_widman.py); the 4 small-ring reject tests here confirm
    gate 1 at skeletal_replacement.py:115-117 keeps rejecting <7-member.

    Source: 154-CONTEXT.md D-07; 154-RESEARCH.md §3.6; 154-AUDIT-A.md §6.
    """

    @pytest.mark.parametrize("smiles,expected_name,ring_size", [
        # Audit-verified Blue Book + corpus examples per 154-AUDIT-A.md §6
        # Blue Book fixtures (BB_4, BB_7)
        ("O1CCOCCOCC1", "1,4,7-trioxacyclononane", 9),
        ("O1CCOCCOCCOCC1", "1,4,7,10-tetraoxacyclododecane", 12),
        # Corpus fixtures (audit §6 rows 3-6)
        ("C1CNCCNCCCNCCNC1", "1,4,8,11-tetraazacyclotetradecane", 14),
        ("C1CNCCNCCN1", "1,4,7-triazacyclononane", 9),
        ("C1COCCOCCNCCOCCOCCN1",
         "1,10-diaza-4,7,13,16-tetraoxacyclooctadecane", 18),
        ("C1COCCOCCOCCOCCOCCO1",
         "1,4,7,10,13,16-hexaoxacyclooctadecane", 18),
    ])
    def test_cyclic_replacement_size_class(self, smiles, expected_name, ring_size):
        from orthonym.rules.skeletal_replacement import try_skeletal_replacement_name
        mol = Chem.MolFromSmiles(smiles)
        result = try_skeletal_replacement_name(mol)
        assert result == expected_name, (
            f"ring_size={ring_size}, smiles={smiles}, "
            f"expected={expected_name!r}, got={result!r}"
        )

    @pytest.mark.parametrize("smiles", [
        # 4-6-member heterocycles MUST route to Hantzsch-Widman, NOT
        # _try_cyclic_replacement_name. Confirm gate 1 at lines 115-117 rejects.
        "C1CCNCC1",  # piperidine (6-mem -- HW owns)
        "C1CNCCO1",  # morpholine (6-mem -- HW owns)
        "C1CCOC1",   # tetrahydrofuran (5-mem -- HW owns)
        "C1COC1",    # oxetane (4-mem -- HW owns)
    ])
    def test_small_rings_rejected_by_gate1(self, smiles):
        """4-6 member heterocycles: gate 1 rejects -- HW handler owns."""
        from orthonym.rules.skeletal_replacement import try_skeletal_replacement_name
        mol = Chem.MolFromSmiles(smiles)
        result = try_skeletal_replacement_name(mol)
        # For 4-6 member rings, expect None (gate 1 reject for ring < 7).
        assert result is None, (
            f"gate 1 should reject ring size <7: smiles={smiles}, got {result!r}"
        )


@pytest.mark.unit
class TestTerminalGroup14Gate:
    """v23 Phase 8 (P-68.2.1.1, E2-owned terminal-atom gate): a TERMINAL Group-14
    backbone atom (Si/Ge/Sn/Pb) is NOT a skeletal-replacement chain atom — it
    would be silently counted as a carbon of the alkane stem (structure loss).
    Fail-closed; INTERIOR Group-14 atoms and ether/replacement chains are
    UNAFFECTED."""

    @pytest.mark.parametrize("smiles", [
        "[SiH3]O[SiH2]O[SiH3]",   # trisiloxane: was '2,4-dioxa-3-silapentane' (=> dimethoxysilane)
        "[SiH3]O[SiH3]",          # disiloxane
        "[GeH3]O[GeH3]",          # digermoxane (terminal Ge)
    ])
    def test_terminal_group14_fail_closed(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        assert try_skeletal_replacement_name(mol) is None

    @pytest.mark.parametrize("smiles,expected", [
        # INTERIOR Group-14 stays a skeletal replacement (terminal atoms are C).
        ("C[SiH2]CC[SiH2]CC[SiH2]CC[SiH2]C", "2,5,8,11-tetrasiladodecane"),
        # R4 / P-63.2.4: 2-O diether without terminal -ol -> substitutive (None from skeletal).
        # COCCOC -> None (substitutive gives '1,2-dimethoxyethane').  Updated pre-R4 '2,5-dioxahexane'.
        # 3-O triether stays skeletal (>= 3 O falls through R4 gate).
        ("COCCOCCOC", "2,5,8-trioxanonane"),
    ])
    def test_interior_and_ether_unaffected(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert try_skeletal_replacement_name(mol) == expected
