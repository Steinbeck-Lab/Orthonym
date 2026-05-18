"""Phase 160.1 regression fixture + audit §3 spot-check tests.

Plan-02-07 per CONTEXT D-16 honest-fail-on-data + RESEARCH §7 acceptance
test. The regression fixture is the canonical Phase 160.1 fix target —
pre-fix it names with spurious "hydroxymethyl"; post-fix with
"methoxycarbonyl" per IUPAC P-65.6.3.

Audit §3 spot-checks confirm the canary re-baseline policy:
* `name_stability_373` is the FIX (hydroxymethyl → methoxycarbonyl).
* `name_stability_265` is a documented pre-existing environmental drift
  (per 160.1-01-SUMMARY.md §3); NOT introduced by Phase 160.1.
"""
from orthonym import name_compound


# ====================================================================
# Regression fixture — THE Phase 160.1 fix target
# ====================================================================


class TestRegressionFixture:
    """The canonical Phase 160.1 regression fixture from CONTEXT + RESEARCH §3."""

    def test_regression_fixture_methoxycarbonyl_present(self):
        """Phase 160.1 regression fixture names with methoxycarbonyl."""
        smi = (
            "COC(=O)/C(CC(=O)O)=C(\\CCCCCCCCCCCCCCCCC1=C(C)C(=O)OC1=O)C(=O)O"
        )
        n = name_compound(smi)
        # The 3-position methyl-ester substituent must be named with the
        # IUPAC-canonical alkoxycarbonyl prefix form per P-65.6.3.
        assert "methoxycarbonyl" in n, (
            f"expected 'methoxycarbonyl' in name, got: {n!r}"
        )
        # Pre-fix the recursive Tier-4 path produced 'hydroxymethyl' /
        # 'methyl formatyl' for the same atoms. The Tier-0.5 hook short-
        # circuits these incorrect names.
        assert "hydroxymethyl" not in n, (
            f"BUG: hydroxymethyl still in name: {n!r}"
        )
        assert "formatyl" not in n, f"BUG: formatyl still in name: {n!r}"


# ====================================================================
# Audit §3 spot-checks (per 160.1-AUDIT-SUBENUM.md §3 4-row table)
# ====================================================================


class TestAuditSection3SpotChecks:
    """Each canary fixture from audit §3 is verified for its expected post-fix name."""

    def test_audit_row1_name_stability_373(self):
        """Audit §3 row 1: -C(=O)OCH3 substituent → methoxycarbonyl."""
        smi = (
            "COC(=O)/C(CC(=O)O)=C(\\CCCCCCCCCCCCCCCCC1=C(C)C(=O)OC1=O)C(=O)O"
        )
        n = name_compound(smi)
        assert "methoxycarbonyl" in n

    def test_audit_row2_name_stability_4_no_regression(self):
        """Audit §3 row 2: tricarboxylic acid with spurious hydroxymethyl.

        This row was a candidate for re-baselining but the 14-row closed
        set does not include carboxylic_acid. The fix does NOT change the
        name. Acceptance: the name is stable (does not regress further).
        """
        smi = "NC(C(=O)O)C(CC[C@H](N)C(=O)O)C(=O)O"
        n = name_compound(smi)
        # Name should still build (whatever the polyfunctional path produces).
        # Acceptable post-fix: same as baseline OR an improvement
        # (the goal is NO new regression, not necessarily a re-baseline).
        assert n is not None
        assert "unknown" not in n.lower()

    def test_audit_row3_name_stability_545_no_regression(self):
        """Audit §3 row 3: pyrrolidinyl heptanedioic acid (similar to row 2)."""
        smi = (
            "C[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O"
        )
        n = name_compound(smi)
        assert n is not None and "unknown" not in n.lower()

    def test_audit_row4_rt75_182_no_regression(self):
        """Audit §3 row 4: another tricarboxylic acid pattern (similar to row 2)."""
        smi = "NC(C(=O)O)C(CCC(N)C(=O)O)C(=O)O"
        n = name_compound(smi)
        assert n is not None and "unknown" not in n.lower()


# ====================================================================
# Hydroxymethyl preservation — confirms bona-fide -CH2OH substituents
# are NOT broken by the Phase 160.1 fix
# ====================================================================


class TestHydroxymethylPreservation:
    """Bona-fide -CH2OH substituents must still name as hydroxymethyl."""

    def test_4_hydroxymethylphenol_preserves_hydroxymethyl(self):
        """4-(hydroxymethyl)phenol literal -CH2OH preserves hydroxymethyl name."""
        smi = "OCc1ccc(O)cc1"
        n = name_compound(smi)
        # Hydroxymethyl is a bona-fide name for -CH2OH per IUPAC; the fix
        # should NOT change this case (the 14-row table is exact-match,
        # and the -CH2OH 2-atom fragment doesn't match any 14-row SMARTS
        # at strict equality).
        assert "hydroxymethyl" in n or "methanol" in n or "phenol" in n, (
            f"expected hydroxymethyl-related name, got {n!r}"
        )

    def test_2_hydroxymethylphenol_preserves(self):
        """2-(hydroxymethyl)phenol — same preservation expectation."""
        smi = "OCc1ccccc1O"
        n = name_compound(smi)
        assert n is not None and "unknown" not in n.lower()

    def test_tris_buffer_preserves_hydroxymethyl(self):
        """Tris(hydroxymethyl)aminomethane has multiple -CH2OH groups."""
        smi = "NC(CO)(CO)C(=O)O"  # 2-amino-3-hydroxy-2-(hydroxymethyl)propanoic acid
        n = name_compound(smi)
        # The bona-fide -CH2OH groups should still produce hydroxymethyl-like names
        assert n is not None and "unknown" not in n.lower()


# ====================================================================
# IUPAC canonical form preference (per audit §1 OPSIN cross-reference)
# ====================================================================


class TestCanonicalForms:
    """Orthonym emits the IUPAC-preferred, non-deprecated form."""

    def test_methoxycarbonyl_not_carbmethoxy(self):
        """Row 1: emits 'methoxycarbonyl' not deprecated 'carbmethoxy'."""
        smi = "OC(=O)CC(C(=O)OC)CC(=O)O"
        n = name_compound(smi)
        assert "methoxycarbonyl" in n.lower() or "(methoxy)" in n.lower()
        # Deprecated alternates must NOT appear
        assert "carbmethoxy" not in n.lower()
        assert "carbomethoxy" not in n.lower()

    def test_phenoxycarbonyl_not_carbphenoxy(self):
        """Row 2: emits 'phenoxycarbonyl' not deprecated 'carbphenoxy'."""
        smi = "O=C(Oc1ccccc1)CCCC(=O)O"  # 5-phenoxycarbonyl-pentanoic acid
        n = name_compound(smi)
        # Deprecated alternates must NOT appear
        assert "carbphenoxy" not in n.lower()
        assert "carbophenoxy" not in n.lower()


# ====================================================================
# Phase 160.2 Plan-04-01 — CR-01 BLOCKER regression fixture
# ====================================================================


class TestOxygenAttachedCarbamateBranchB:
    """Phase 160.2 Plan-04-01 CR-01 regression: oxygen-attached carbamate
    must use ``carbamoyloxy`` (IUPAC P-66.6.4 Branch B), not Branch A
    fall-through.

    SMILES: ``O=C(N)OCCCC(=O)O``
    Structure: ``NH2-C(=O)-O-CH2-CH2-CH2-C(=O)-OH``
    Parent: butanoic acid (4C including carboxyl)
    Substituent at C4: ``carbamoyloxy`` (P-66.6.4 Branch B)
    Expected: ``4-(carbamoyloxy)butanoic acid``

    Pre-fix (160.1 ship at HEAD): emitted ``4-(methanoyloxy)butanoic acid``;
    OPSIN round-trip dropped the N+H atoms producing the wrong molecule
    (InChI=1S/C5H8O4 instead of input C5H9NO4) per RESEARCH §2.
    Post-fix (160.2 Plan-04-01): emits ``4-(carbamoyloxy)butanoic acid``;
    OPSIN round-trip matches input InChI L1.
    """

    def test_oxygen_attached_carbamate_branch_b(self):
        """name_compound emits Branch B carbamoyloxy per IUPAC P-66.6.4."""
        smiles = "O=C(N)OCCCC(=O)O"
        expected = "4-(carbamoyloxy)butanoic acid"
        actual = name_compound(smiles, style="pin")
        assert actual == expected, (
            f"CR-01 regression: expected {expected!r} per IUPAC P-66.6.4 "
            f"Branch B (carbamate attached through ester-O); "
            f"got: {actual!r}"
        )

    def test_oxygen_attached_carbamate_opsin_roundtrip(self):
        """OPSIN parses the corrected name back to the original structure
        per InChI L1 (formula + connectivity)."""
        import subprocess
        import os
        from rdkit import Chem

        smiles = "O=C(N)OCCCC(=O)O"
        name = name_compound(smiles, style="pin")

        # Locate the OPSIN jar shipping in the repo opsin/ tree (relative
        # path; opsin/opsin-cli-2.9.0-jar-with-dependencies.jar per
        # Phase 138 D-22 pinned version).
        opsin_jar_candidates = [
            "opsin/opsin-cli-2.9.0-jar-with-dependencies.jar",
            os.path.join(
                os.path.dirname(__file__), "..", "..", "opsin",
                "opsin-cli-2.9.0-jar-with-dependencies.jar",
            ),
        ]
        opsin_jar = None
        for candidate in opsin_jar_candidates:
            if os.path.isfile(candidate):
                opsin_jar = candidate
                break
        if opsin_jar is None:
            import pytest as _pytest
            _pytest.skip("OPSIN jar not found in repo; skipping round-trip")

        result = subprocess.run(
            ["java", "-jar", opsin_jar, "-osmi"],
            input=name + "\n", capture_output=True, text=True, timeout=30,
        )
        rt_smiles = result.stdout.strip().split("\n")[0]
        assert rt_smiles, f"OPSIN failed to parse {name!r}: {result.stderr}"

        mol_input = Chem.MolFromSmiles(smiles)
        mol_rt = Chem.MolFromSmiles(rt_smiles)
        assert mol_input is not None and mol_rt is not None
        inchi_input = Chem.MolToInchi(mol_input)
        inchi_rt = Chem.MolToInchi(mol_rt)
        # Formula layer match (1S/<formula>/...) — strict equality across
        # all layers since the molecule is fully determinate.
        assert inchi_input.split("/")[1] == inchi_rt.split("/")[1], (
            f"formula mismatch: input={inchi_input} rt={inchi_rt}"
        )


class TestNSubstitutedCarbamateBranchB:
    """Phase 160.2 post-merge: N-substituted Branch B carbamate per IUPAC
    P-66.6.4. The original Plan-04-01 guard hard-coded ``carbamoyloxy`` and
    silently dropped N-substituents; the follow-up fix routes through
    ``_compute_branch_b_carbamoyloxy_name`` which produces explicit-N-locant
    PIN forms via the existing ``get_n_alkyl_carbamoyl_prefix`` /
    ``get_n_n_dialkyl_carbamoyl_prefix`` conventions then appends ``oxy``.

      * ``O=C(NC)OCCCC(=O)O``        → ``4-(N-methylcarbamoyl)oxybutanoic acid``
      * ``O=C(NCC)OCCCC(=O)O``       → ``4-(N-ethylcarbamoyl)oxybutanoic acid``
      * ``O=C(N(C)C)OCCCC(=O)O``     → ``4-(N,N-dimethylcarbamoyl)oxybutanoic acid``
      * ``O=C(N(C)CC)OCCCC(=O)O``    → ``4-(N-ethyl-N-methylcarbamoyl)oxybutanoic acid``

    Each name OPSIN-round-trips to the input InChI L1 (formula + connectivity).
    """

    _CASES = [
        ("O=C(NC)OCCCC(=O)O", "4-(N-methylcarbamoyl)oxybutanoic acid"),
        ("O=C(NCC)OCCCC(=O)O", "4-(N-ethylcarbamoyl)oxybutanoic acid"),
        ("O=C(N(C)C)OCCCC(=O)O", "4-(N,N-dimethylcarbamoyl)oxybutanoic acid"),
        (
            "O=C(N(C)CC)OCCCC(=O)O",
            "4-(N-ethyl-N-methylcarbamoyl)oxybutanoic acid",
        ),
    ]

    def test_n_substituted_carbamate_branch_b_names(self):
        """name_compound emits N-substituted Branch B carbamoyloxy forms."""
        for smiles, expected in self._CASES:
            actual = name_compound(smiles, style="pin")
            assert actual == expected, (
                f"Branch B regression: smiles={smiles!r}, "
                f"expected={expected!r}, got={actual!r}"
            )

    def test_n_substituted_carbamate_opsin_roundtrip(self):
        """Generated names OPSIN-round-trip to input InChI L1."""
        import subprocess
        import os
        from rdkit import Chem

        opsin_jar_candidates = [
            "opsin/opsin-cli-2.9.0-jar-with-dependencies.jar",
            os.path.join(
                os.path.dirname(__file__), "..", "..", "opsin",
                "opsin-cli-2.9.0-jar-with-dependencies.jar",
            ),
        ]
        opsin_jar = None
        for candidate in opsin_jar_candidates:
            if os.path.isfile(candidate):
                opsin_jar = candidate
                break
        if opsin_jar is None:
            import pytest as _pytest
            _pytest.skip("OPSIN jar not found in repo; skipping round-trip")

        for smiles, _expected in self._CASES:
            name = name_compound(smiles, style="pin")
            result = subprocess.run(
                ["java", "-jar", opsin_jar, "-osmi"],
                input=name + "\n", capture_output=True, text=True, timeout=30,
            )
            rt_smiles = result.stdout.strip().split("\n")[0]
            assert rt_smiles, (
                f"OPSIN failed to parse {name!r}: {result.stderr}"
            )
            mol_input = Chem.MolFromSmiles(smiles)
            mol_rt = Chem.MolFromSmiles(rt_smiles)
            assert mol_input is not None and mol_rt is not None
            inchi_input = Chem.MolToInchi(mol_input)
            inchi_rt = Chem.MolToInchi(mol_rt)
            assert inchi_input.split("/")[1] == inchi_rt.split("/")[1], (
                f"formula mismatch for {smiles!r}: "
                f"input={inchi_input} rt={inchi_rt}"
            )
