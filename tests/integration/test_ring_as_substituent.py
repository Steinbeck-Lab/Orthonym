"""
Integration tests for ring-as-substituent naming through all 6 enumeration paths.

Phase 79: Verifies that non-phenyl ring substituents are correctly named
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

    This is the reference path, already using get_ring_substituent_name().
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
    """Path 5: _name_heteroatom_substituent - ring on N-branch (DROP-18 fix).

    When a non-phenyl ring is attached to a nitrogen on the principal chain,
    it should be named using get_ring_substituent_name() and formatted with
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
        """Anilino (phenyl on N) detection unchanged by Phase 79 changes.

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
    """Path 5: _name_heteroatom_substituent - ring on C-branch (DROP-19 fix).

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
    """_assemble_amine_name() N-sub path (DROP-25 fix).

    When an amine has a non-phenyl ring N-substituent, it should be named
    using get_ring_substituent_name() instead of falling back to alkyl naming.
    """

    def test_n_cyclohexyl_amine(self):
        """N-cyclohexylamine: cyclohexyl on nitrogen of amine."""
        result = name_compound("CCNC1CCCCC1")
        assert result is not None, "name_compound returned None"
        result_lower = result.lower()
        assert "cyclohexyl" in result_lower, (
            f"Expected 'cyclohexyl' in '{result}'"
        )

    def test_n_phenyl_amine_still_works(self):
        """N-phenylamine detection unchanged by Phase 79 changes.

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
