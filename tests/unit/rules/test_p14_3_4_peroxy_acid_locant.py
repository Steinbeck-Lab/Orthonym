"""P-14.3.4 — the ring peroxy-acid suffix locant (v29 Phase C tranche B, Task 2).

We shipped ``cyclohexane-1-carboperoxoic acid`` against the verbatim Blue Book PIN.

AUTHORITY, all verified verbatim in ``BlueBookV2/BlueBookV2.md``:

* ``:30184`` — ``cyclohexanecarboperoxoic acid (PIN)``
* ``:30178`` — ``C6H5-CO-OOH benzenecarboperoxoic acid (PIN) peroxybenzoic acid
  perbenzoic acid`` (and ``:29797`` prints the same pair the other way round)
* ``:30174`` ``ethaneperoxoic acid (PIN)``, ``:30176`` ``hexaneperoxoic acid (PIN)``
* the licence is **P-14.3.4.2(c)**, ``:2891`` *"The locant '1' is omitted:"* + ``:2913``
  *"(c) in monosubstituted homogeneous monocyclic rings;"*
* the answer to "but the suffix spans several heavy atoms" is **P-14.3.4.3**, ``:2939``
  *"The locant is omitted in monosubstituted symmetrical parent hydrides or parent
  compounds where there is only one kind of substitutable hydrogen."* — whose own
  example block contains ``:2949`` ``pyrazinecarboxylic acid (PIN)``, a multi-atom acid
  suffix on a monocycle with the locant omitted.
* it is **NOT** P-14.3.4.1 (``:2877``), which is scoped to *"mono- and dicarboxylic acids
  derived from ACYCLIC hydrocarbons"* and enumerates its derivative classes — cyclohexane
  is not acyclic and peroxy acids are not in the list.

★ ROOT CAUSE was a ONE-ROW DATA ASYMMETRY between two parallel tables, not the assembler
and not tranche A's ring predicate. ``peroxy_acid`` was in
``naming_utils.TERMINAL_FG_TYPES`` but missing from ``composer.TERMINAL_GROUPS``, so a
peroxy acid took the NON-terminal branch of ``_handler_shared._generate_suffix``
(which cites the locant) instead of the TERMINAL branch (which already withholds it —
the reason ``cyclohexanecarboxylic acid`` and ``cyclohexanecarbaldehyde`` are correct).

The plan had said to build a licence in ``_assemble_fragments``. Measured: the ``1`` is
already baked into the fragment before the assembler sees it, so the assembler was the
wrong site — the fifth off-path choke point in this phase.
"""
import pytest

from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


class TestTheTwoTablesAgree:
    def test_terminal_group_tables_are_consistent(self):
        """The invariant the defect violated: neither table may hold a terminal FG the
        other lacks. Kept as a test so the next divergence fails here, loudly, instead
        of surfacing as one mis-spelled name years later."""
        from orthonym.assembly.composer import TERMINAL_GROUPS
        from orthonym.assembly.naming_utils import TERMINAL_FG_TYPES

        missing_here = set(TERMINAL_FG_TYPES) - set(TERMINAL_GROUPS)
        missing_there = set(TERMINAL_GROUPS) - set(TERMINAL_FG_TYPES)

        # Five classes remain deliberately absent from TERMINAL_GROUPS: they have
        # substitutable suffix nitrogens, and the terminal branch's per-scope check
        # cannot see a substituent on the suffix heteroatom (the hole that shipped
        # `N-hydroxycyclohexanimine`). Documented, not accidental.
        assert missing_here == {
            "hydrazidine", "hydrazonamide", "hydrazonic_acid",
            "imidic_acid", "thiohydrazide",
        }, f"terminal-group tables diverged: {sorted(missing_here)}"
        assert missing_there == set(), sorted(missing_there)

    def test_peroxy_acid_is_terminal_in_both(self):
        from orthonym.assembly.composer import TERMINAL_GROUPS
        from orthonym.assembly.naming_utils import TERMINAL_FG_TYPES
        assert "peroxy_acid" in TERMINAL_FG_TYPES
        assert "peroxy_acid" in TERMINAL_GROUPS


class TestLicensedOmission:
    def test_cyclohexanecarboperoxoic_acid(self, namer):
        """BB:30184 verbatim `cyclohexanecarboperoxoic acid (PIN)`."""
        assert namer.name("OOC(=O)C1CCCCC1") == "cyclohexanecarboperoxoic acid"

    @pytest.mark.parametrize("smiles,expected", [
        ("OOC(=O)C1CCC1", "cyclobutanecarboperoxoic acid"),
        ("OOC(=O)C1CCCC1", "cyclopentanecarboperoxoic acid"),
        ("OOC(=O)C1CCCCCC1", "cycloheptanecarboperoxoic acid"),
    ])
    def test_other_ring_sizes_follow_the_same_rule(self, namer, smiles, expected):
        """The class is OPEN in ring size — a fix keyed on cyclohexane would be wrong
        on the complement by construction."""
        assert namer.name(smiles) == expected


class TestDenyByDefault:
    """P-14.3.3 (:2869) is deny-by-default: one essential locant in a scope restores
    every locant in that scope. Each row asserts the WHOLE name (invariant 11)."""

    def test_ring_substituent_restores_the_locant(self, namer):
        assert namer.name("OOC(=O)C1CCC(C)CC1") == \
            "4-methylcyclohexane-1-carboperoxoic acid"

    def test_two_suffixes_restore_the_locants(self, namer):
        assert namer.name("OOC(=O)C1CCCCC1C(=O)OO") == \
            "cyclohexane-1,2-dicarboperoxoic acid"

    @pytest.mark.parametrize("smiles,expected,why", [
        ("OOC(=O)CCCCC", "hexaneperoxoic acid", "acyclic, BB:30176 — unaffected"),
        ("OOC(=O)C", "ethaneperoxoic acid", "acyclic, BB:30174 — unaffected"),
        ("OC(=O)C1CCCCC1", "cyclohexanecarboxylic acid", "sibling carb- suffix"),
        ("N#CC1CCCCC1", "cyclohexanecarbonitrile", "BB:34728"),
        ("O=CC1CCC1", "cyclobutanecarbaldehyde", "sibling carb- suffix"),
        ("NC(=O)C1CCCCC1", "cyclohexanecarboxamide", "sibling carb- suffix"),
        ("O=C1CCCCC1", "cyclohexanone", "tranche A must not shift"),
        ("OC1CCCC1", "cyclopentanol", "tranche A must not shift"),
        ("ON=C1CCCCC1", "N-hydroxycyclohexan-1-imine",
         "the essential-N guard — the oxime that failed tranche A's gate"),
        ("OCCCl", "2-chloroethan-1-ol",
         "THE headline guard: BB:2869 names this omission as not allowed"),
    ])
    def test_controls_unchanged(self, namer, smiles, expected, why):
        assert namer.name(smiles) == expected, why


class TestKnownAdjacentDefects:
    """Found by this task's own control rows, recorded rather than silently left.

    Both are the INVERSE of Phase C — a locant that P-14.3.3 makes essential is being
    OMITTED — and both are pre-existing (measured identical before and after the
    ``peroxy_acid`` fix, so this task did not cause them).
    """

    @pytest.mark.xfail(strict=True, reason=(
        "Pre-existing: we emit the deprecated `perbenzoic acid`. BB:30178 and BB:29797 "
        "both print `benzenecarboperoxoic acid (PIN)` beside it, and BB:3009 states "
        "\"The prefix 'per-' is no longer recommended.\""))
    def test_benzenecarboperoxoic_acid(self, namer):
        assert namer.name("OOC(=O)c1ccccc1") == "benzenecarboperoxoic acid"

    @pytest.mark.xfail(strict=True, reason=(
        "Pre-existing P-14.3.3 UNDER-citation: the essential 4-methyl locant must "
        "restore the suffix locant. Its siblings already do — measured, "
        "`4-methylcyclohexane-1-carboxylic acid` and "
        "`4-methylcyclohexane-1-carbaldehyde` are correct, so nitrile and amide are "
        "inconsistent with them. BB:5822/:23198 print "
        "`4,4'-oxydi(cyclohexane-1-carboxylic acid) (PIN)`, a substituted ring "
        "carboxylic acid citing its 1."))
    def test_substituted_ring_carbonitrile_cites_its_locant(self, namer):
        assert namer.name("N#CC1CCC(C)CC1") == "4-methylcyclohexane-1-carbonitrile"

    @pytest.mark.xfail(strict=True, reason="Same class as the carbonitrile above.")
    def test_substituted_ring_carboxamide_cites_its_locant(self, namer):
        assert namer.name("NC(=O)C1CCC(C)CC1") == "4-methylcyclohexane-1-carboxamide"
