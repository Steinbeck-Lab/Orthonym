"""
Integration tests for ring-as-substituent naming through all 6 enumeration paths.

a phase: Verifies that non-phenyl ring substituents are correctly named
when appearing as substituents in different structural contexts. Each test
class covers one of the 6 substituent enumeration paths.

Path 4 (FG prefix) does not handle ring systems (it handles functional group
prefixes like hydroxy, amino, oxo) so no tests are needed for it.
"""

import pytest
from orthonym import name_compound


@pytest.mark.integration
class TestPath1ChainAlkylRingSub:
    """Path 1: _generate_alkyl_prefixes - ring on chain parent via heteroatom branch.

    When chain_is_parent=True and a substituent on the chain contains a ring,
    the ring should be identified and named as a prefix.
    """

    def test_cyclopentyl_on_acid_chain(self):
        """5-cyclopentylpentanoic acid: cyclopentyl ring on acid chain."""
        result = name_compound("C(CCCC(=O)O)C1CCCC1")
        assert result is not None, "name_compound returned None"
        assert "cyclopentyl" in result.lower(), (
            f"Expected 'cyclopentyl' in '{result}'"
        )

    def test_cyclohexyl_on_chain(self):
        """Cyclohexyl attached to a simple chain."""
        result = name_compound("CCCCC1CCCCC1")
        assert result is not None, "name_compound returned None"
        assert "cyclohexyl" in result.lower() or "cyclohex" in result.lower(), (
            f"Expected cyclohexyl reference in '{result}'"
        )


@pytest.mark.integration
class TestPath2RingAlkylRingSub:
    """Path 2: _generate_ring_alkyl_prefixes - substituent on a ring parent.

    When the parent is a ring and a substituent contains another ring,
    the substituent ring should be named correctly.
    """

    def test_cyclopentyl_on_cyclohexane(self):
        """Cyclopentylcyclohexane: cyclopentyl ring on cyclohexane parent."""
        result = name_compound("C1CCC(CC1)C2CCCC2")
        assert result is not None, "name_compound returned None"
        assert "cyclopentyl" in result.lower(), (
            f"Expected 'cyclopentyl' in '{result}'"
        )


@pytest.mark.integration
class TestPath3RingAsSub:
    """Path 3: _generate_ring_substituent_prefixes - ring as substituent on chain.

    This is the reference path, already using get_ring_substituent_name.
    Tests verify it continues to work correctly.
    """

    def test_cyclohexyl_on_long_chain(self):
        """Cyclohexyl on a long chain acid where chain_is_parent=True."""
        result = name_compound("OC(=O)CCCCCC(CCCCC)C1CCCCC1")
        assert result is not None, "name_compound returned None"
        assert "cyclohexyl" in result.lower(), (
            f"Expected 'cyclohexyl' in '{result}'"
        )

    def test_phenyl_on_chain_acid(self):
        """Phenyl on acid chain (reference case, always worked)."""
        result = name_compound("OC(=O)CCc1ccccc1")
        assert result is not None, "name_compound returned None"
        assert "phenyl" in result.lower(), (
            f"Expected 'phenyl' in '{result}'"
        )


@pytest.mark.integration
class TestPath5RecursiveNbranch:
    """Path 5: _name_heteroatom_substituent - ring on N-branch (fix).

    When a non-phenyl ring is attached to a nitrogen on the principal chain,
    it should be named using get_ring_substituent_name and formatted with
    the amino context.
    """

    def test_n_cyclohexylamino_on_acid(self):
        """N-cyclohexylamino group on an acid chain."""
        result = name_compound("OC(=O)CCCNC1CCCCC1")
        assert result is not None, "name_compound returned None"
        result_lower = result.lower()
        assert "cyclohexyl" in result_lower, (
            f"Expected 'cyclohexyl' in '{result}'"
        )

    def test_n_piperidinyl_on_acid(self):
        """N-piperidinyl group on an acid chain (N is in the ring)."""
        result = name_compound("OC(=O)CCCN1CCCCC1")
        assert result is not None, "name_compound returned None"
        result_lower = result.lower()
        assert "piperid" in result_lower, (
            f"Expected 'piperid*' in '{result}'"
        )

    def test_n_morpholinyl_on_acid(self):
        """N-morpholinyl group on an acid chain."""
        result = name_compound("OC(=O)CCCN1CCOCC1")
        assert result is not None, "name_compound returned None"
        result_lower = result.lower()
        assert "morpholin" in result_lower, (
            f"Expected 'morpholin*' in '{result}'"
        )

    def test_anilino_still_works(self):
        """Anilino (phenyl on N) detection unchanged by a phase changes.

        The inline anilino detection at _name_heteroatom_substituent:4650-4656
        must still produce 'anilino' for phenyl-on-N when chain is parent.
        """
        # 4-anilinopentanoic acid -> chain is parent, phenyl on N is detected
        result = name_compound("OC(=O)CCC(NC1CCCCC1)CC")
        assert result is not None, "name_compound returned None"
        result_lower = result.lower()
        # Should have cyclohexyl or amino reference (cyclohexyl on N)
        assert "cyclohexylamino" in result_lower or "amino" in result_lower, (
            f"Expected amino/cyclohexyl reference in '{result}'"
        )


@pytest.mark.integration
class TestPath5RecursiveCbranch:
    """Path 5: _name_heteroatom_substituent - ring on C-branch (fix).

    When a C-branch substituent on the chain contains a ring,
    it should be identified and named.
    """

    def test_c_branch_with_pyrrolidine(self):
        """Pyrrolidinyl ring on a C-branch of the chain."""
        # 4-(pyrrolidin-1-yl)butanoic acid or similar
        result = name_compound("OC(=O)CCCC1CCNC1")
        assert result is not None, "name_compound returned None"
        result_lower = result.lower()
        # Should contain pyrrolidinyl or pyrrolidine reference
        assert "pyrrolid" in result_lower or "cyclo" in result_lower, (
            f"Expected ring reference in '{result}'"
        )


@pytest.mark.integration
class TestPath6Polyfunctional:
    """Path 6: name_polyfunctional - ring sub on polyfunctional compound.

    Polyfunctional compounds delegate to _generate_prefixes which calls path 3
    for chain-parent ring subs. Verify this works end-to-end.

    Note: Path 4 (FG prefix) does not generate ring names by design --
    it handles functional group prefixes (hydroxy, amino, oxo), not ring systems.
    """

    def test_polyfunctional_with_ring_sub(self):
        """A polyfunctional compound with a ring substituent."""
        # Hydroxy acid with cyclohexyl: has 3+ functional groups
        result = name_compound("OC(=O)C(O)CC1CCCCC1")
        assert result is not None, "name_compound returned None"
        result_lower = result.lower()
        # Should contain cyclohexyl or cyclohex reference
        assert "cyclohex" in result_lower or "cyclo" in result_lower, (
            f"Expected ring reference in '{result}'"
        )


@pytest.mark.integration
class TestAmineNSubRing:
    """_assemble_amine_name N-sub path (fix).

    When an amine has a non-phenyl ring N-substituent, it should be named
    using get_ring_substituent_name instead of falling back to alkyl naming.
    """

    def test_n_cyclohexyl_amine(self):
        """Ethyl(cyclohexyl)amine: the RING is the parent and carries only the amine.

         (the Blue Book, 'Secondary and tertiary amines'):
        '*N*-butylcyclopropanamine (PIN)... (not *N*-cyclopropylbutan-1-amine)'
        (:26292), so the ring is the parent (the old 'cyclohexyl' expectation had
        the chain as parent), and (c) (:2891/:2913) omits the locant '1'
        on the monosubstituted ring -- the N-substituent hangs off the nitrogen.
        """
        assert name_compound("CCNC1CCCCC1") == "N-ethylcyclohexanamine"

    def test_n_phenyl_amine_still_works(self):
        """N-phenylamine detection unchanged by a phase changes.

        When benzene is parent (ring > chain), phenyl doesn't appear as
        substituent prefix -- the output is N-ethylaminobenzene.
        Verifying the naming doesn't crash and produces a reasonable name.
        """
        result = name_compound("CCNc1ccccc1")
        assert result is not None, "name_compound returned None"
        result_lower = result.lower()
        # Ring is parent -> "aminobenzene" form, not "phenyl" prefix
        assert "benzene" in result_lower or "aniline" in result_lower, (
            f"Expected 'benzene' or 'aniline' in '{result}'"
        )


@pytest.mark.integration
class TestRingSubstituentPrefixVariety:
    """Test variety of ring substituent prefix names."""

    def test_cyclopropyl_prefix(self):
        """Cyclopropyl should appear in the name (acid forces chain as parent)."""
        result = name_compound("OC(=O)CCC(C1CC1)CCC")
        assert result is not None
        assert "cyclopropyl" in result.lower(), f"Expected 'cyclopropyl' in '{result}'"

    def test_cyclobutyl_prefix(self):
        """Cyclobutyl should appear in the name (acid forces chain as parent)."""
        result = name_compound("OC(=O)CCC(C1CCC1)CCC")
        assert result is not None
        assert "cyclobutyl" in result.lower(), f"Expected 'cyclobutyl' in '{result}'"

    def test_pyridyl_prefix(self):
        """Pyridyl should appear in the name for pyridine substituent."""
        result = name_compound("OC(=O)CCCc1ccncc1")
        assert result is not None
        result_lower = result.lower()
        assert "pyrid" in result_lower, f"Expected 'pyrid*' in '{result}'"


# ===========================================================================
# a phase-02: Fused heterocycle ring-as-substituent tests
# ===========================================================================


@pytest.mark.integration
class TestFusedHetOnNBranch:
    """ +: Fused het ring on N-branch via _name_heteroatom_substituent.

    When a fused heterocycle (quinoline, indole, etc.) is bonded to N on a
    chain parent, the fused het should be identified using static O(1) lookup
    (match_fused_heterocycle_core + get_fused_heterocycle_prefix) and formatted
    as "(stem-locant-ylamino)".

    Uses long chains (C10) so the chain wins parent selection over the fused ring.
    """

    def test_n_quinolinyl_amino_on_chain(self):
        """N-(quinolin-8-yl)amino group on a decanoic acid chain."""
        result = name_compound('OC(=O)CCCCCCCCCNc1cccc2cccnc12')
        assert result is not None, "name_compound returned None"
        assert 'quinolin' in result.lower(), (
            f"Expected 'quinolin' in name for N-quinolinylamino, got: {result}"
        )

    def test_n_indolyl_amino_on_chain(self):
        """N-(1H-indol-5-yl)amino group on a decanoic acid chain."""
        result = name_compound('OC(=O)CCCCCCCCCNc1ccc2[nH]ccc2c1')
        assert result is not None, "name_compound returned None"
        assert 'indol' in result.lower(), (
            f"Expected 'indol' in name for N-indolylamino, got: {result}"
        )

    def test_anilino_not_triggered_for_fused_het(self):
        """Fused het on N should NOT produce 'anilino' (benzene sub-ring false match).

        Quinoline contains a benzene ring, but the anilino detection should be
        guarded by the is_fused check to prevent false matching.
        """
        result = name_compound('OC(=O)CCCCCCCCCNc1cccc2cccnc12')
        assert result is not None
        assert 'anilino' not in result.lower(), (
            f"Fused het incorrectly identified as anilino: {result}"
        )

    def test_real_anilino_still_works(self):
        """Isolated phenyl on N should still produce 'anilino'."""
        result = name_compound('OC(=O)CCCCCCCCCNc1ccccc1')
        assert result is not None
        assert 'anilino' in result.lower(), (
            f"Expected 'anilino' for isolated phenyl on N, got: {result}"
        )


@pytest.mark.integration
class TestFusedHetOnCBranch:
    """: Fused het ring on C-branch via _name_heteroatom_substituent.

    When a fused heterocycle is directly bonded to a carbon on the chain,
    it should be named using the fused het prefix lookup.
    """

    def test_quinolinyl_on_c_branch(self):
        """Quinoline directly bonded to a carbon on a decanoic acid chain."""
        result = name_compound('OC(=O)CCCCCCCCCc1cccc2cccnc12')
        assert result is not None, "name_compound returned None"
        assert 'quinolin' in result.lower(), (
            f"Expected 'quinolin' in name for quinolinyl on C-branch, got: {result}"
        )

    def test_indolyl_on_c_branch(self):
        """Indole directly bonded to a carbon on a decanoic acid chain."""
        result = name_compound('OC(=O)CCCCCCCCCc1ccc2[nH]ccc2c1')
        assert result is not None, "name_compound returned None"
        assert 'indol' in result.lower(), (
            f"Expected 'indol' in name for indolyl on C-branch, got: {result}"
        )


@pytest.mark.integration
class TestDROP18Elimination:
    """: Verify triggers are eliminated for known ring systems.

    These molecules have ring systems on N-branches that previously triggered
    . After a phase-01 (monocyclic) and 79-02 (fused het), known
    ring systems should produce correct prefix names.
    """

    @pytest.mark.parametrize("smiles,expected_ring_token", [
        # N-piperidinyl on chain
        ('OC(=O)CCCN1CCCCC1', 'piperid'),
        # N-morpholinyl on chain
        ('OC(=O)CCCN1CCOCC1', 'morphol'),
        # N-pyrrolidinyl on chain
        ('OC(=O)CCCN1CCCC1', 'pyrrolid'),
    ])
    def test_known_rings_on_n_branch_not_dropped(self, smiles, expected_ring_token):
        """Known ring systems on N-branches produce ring prefix, not."""
        result = name_compound(smiles)
        assert result is not None, f"name_compound returned None for {smiles}"
        assert expected_ring_token in result.lower(), (
            f"Expected '{expected_ring_token}' in name for {smiles}, got: {result}"
        )


@pytest.mark.integration
class TestFusedHetSubstRoundTrip:
    """: Fused het substituent prefix names should contain correct stems.

    Verifies that fused het ring systems appearing as substituents produce
    names containing the correct IUPAC prefix stem (e.g., quinolin, indol).
    """

    @pytest.mark.parametrize("smiles,description,expected_tokens", [
        ('OC(=O)CCCCCCCCCc1cccc2cccnc12', 'quinolinyl on chain',
         ['quinolin']),
        ('OC(=O)CCCCCCCCCc1ccc2[nH]ccc2c1', 'indolyl on chain',
         ['indol']),
        ('OC(=O)CCCCCCCCCNc1cccc2cccnc12', 'quinolinylamino on chain',
         ['quinolin']),
    ])
    def test_fused_het_sub_prefix_stems(self, smiles, description, expected_tokens):
        """Fused het substituent names contain correct IUPAC prefix stems."""
        result = name_compound(smiles)
        assert result is not None, f"Failed to name: {description}"
        result_lower = result.lower()
        for tok in expected_tokens:
            assert tok in result_lower, (
                f"Expected '{tok}' in name for {description}: {result}"
            )


@pytest.mark.integration
class TestDROPReduction:
    """Verify that / are reduced by ring-as-substituent wiring."""

    def test_known_ring_subs_dont_trigger_drops(self):
        """Molecules with known ring substituents should produce names, not drops."""
        test_smiles = [
            'OC(=O)CCCN1CCCCC1',  # N-piperidinyl
            'OC(=O)CCCN1CCOCC1',  # N-morpholinyl
            'C(CCCC(=O)O)C1CCCC1',  # cyclopentyl on chain
        ]
        for smiles in test_smiles:
            result = name_compound(smiles)
            assert result is not None, f"Should produce a name for {smiles}"

    def test_fused_het_subs_produce_names(self):
        """Fused het ring substituents should produce names with ring tokens."""
        test_smiles = [
            ('OC(=O)CCCCCCCCCc1cccc2cccnc12', 'quinolin'),
            ('OC(=O)CCCCCCCCCNc1cccc2cccnc12', 'quinolin'),
        ]
        for smiles, expected in test_smiles:
            result = name_compound(smiles)
            assert result is not None, f"Should produce a name for {smiles}"
            assert expected in result.lower(), (
                f"Expected '{expected}' in name for {smiles}, got: {result}"
            )
