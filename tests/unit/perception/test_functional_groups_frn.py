"""Phase 163 Tier FRN-A..E perception SMARTS unit tests.

Asserts the 16 new SMARTS patterns shipped in Plan-02 (commits 163-02-01..05)
detect their target chalcogen-replacement functional groups AND the 18 new
suppression-map entries correctly suppress cross-pattern false-positives.

Test pyramid composition per CONTEXT D-12 + RESEARCH §8.1:
- Section A: per-pattern positive detection (16-22 tests)
- Section B: suppression-map verification (8-10 tests)
- Section C: edge cases + no-regression on existing patterns (5+ tests)

Total: >= 30 tests.

References:
- src/orthonym/perception/functional_groups.py FUNCTIONAL_GROUP_SMARTS dict
  + _resolve_fg_collisions
- 163-AUDIT-FRN.md § 2 SMARTS catalog
- 163-RESEARCH.md §3 (SMARTS spec) + §8.1 (test pyramid spec)
"""
import pytest
from rdkit import Chem
from orthonym.perception.functional_groups import (
    FUNCTIONAL_GROUP_SMARTS,
    detect_functional_groups,
)

# Alias to mirror CONTEXT/RESEARCH/AUDIT prose naming convention.
# The actual source identifier is FUNCTIONAL_GROUP_SMARTS (per Plan-02 SUMMARY
# Decision 2); audit-time prose used the idealized name PATTERNS.
PATTERNS = FUNCTIONAL_GROUP_SMARTS


@pytest.mark.unit
class TestFRNSelenoicAcidDetection:
    """Tier FRN-A — Chalcogen-on-acid SMARTS detection (6 tests)."""

    def test_selenoic_Se_acid_detected(self):
        """R-C(=O)-SeH (propaneselenoic Se-acid; P-65.3; AUDIT § 2.1)."""
        mol = Chem.MolFromSmiles("CCC(=O)[SeH]")
        groups = detect_functional_groups(mol)
        assert "selenoic_Se_acid" in groups

    def test_selenoic_O_acid_detected(self):
        """R-C(=Se)-OH (propaneselenoic O-acid; P-65.3 parallel)."""
        mol = Chem.MolFromSmiles("CCC(=[Se])O")
        groups = detect_functional_groups(mol)
        assert "selenoic_O_acid" in groups

    def test_diselenoic_acid_detected(self):
        """R-C(=Se)-SeH (propanediselenoic acid; P-65.3 parallel)."""
        mol = Chem.MolFromSmiles("CCC(=[Se])[SeH]")
        groups = detect_functional_groups(mol)
        assert "diselenoic_acid" in groups

    def test_telluroic_Te_acid_detected(self):
        """R-C(=O)-TeH (propanetelluroic Te-acid; P-65.3 parallel)."""
        mol = Chem.MolFromSmiles("CCC(=O)[TeH]")
        groups = detect_functional_groups(mol)
        assert "telluroic_Te_acid" in groups

    def test_telluroic_O_acid_detected(self):
        """R-C(=Te)-OH (propanetelluroic O-acid; P-65.3 parallel)."""
        mol = Chem.MolFromSmiles("CCC(=[Te])O")
        groups = detect_functional_groups(mol)
        assert "telluroic_O_acid" in groups

    def test_ditelluroic_acid_detected(self):
        """R-C(=Te)-TeH (propaneditelluroic acid; P-65.3 parallel)."""
        mol = Chem.MolFromSmiles("CCC(=[Te])[TeH]")
        groups = detect_functional_groups(mol)
        assert "ditelluroic_acid" in groups


@pytest.mark.unit
class TestFRNChalcogenAmideDetection:
    """Tier FRN-B — Chalcogen-on-amide SMARTS detection (7 tests; single-permissive [NX3])."""

    def test_thioamide_detected_primary(self):
        """R-C(=S)-NH2 (P-66.1.4.1.1)."""
        mol = Chem.MolFromSmiles("CCC(N)=S")
        assert "thioamide" in detect_functional_groups(mol)

    def test_thioamide_detected_N_sub(self):
        """R-C(=S)-NHR' (N-substituted)."""
        mol = Chem.MolFromSmiles("CCC(=S)NC")
        assert "thioamide" in detect_functional_groups(mol)

    def test_thioamide_detected_NN_disub(self):
        """R-C(=S)-NR'R'' (N,N-disubstituted)."""
        mol = Chem.MolFromSmiles("CCC(=S)N(C)C")
        assert "thioamide" in detect_functional_groups(mol)

    def test_selenoamide_detected_primary(self):
        """R-C(=Se)-NH2."""
        mol = Chem.MolFromSmiles("CCC(N)=[Se]")
        assert "selenoamide" in detect_functional_groups(mol)

    def test_selenoamide_detected_N_sub(self):
        """R-C(=Se)-NHR'."""
        mol = Chem.MolFromSmiles("CCC(=[Se])NC")
        assert "selenoamide" in detect_functional_groups(mol)

    def test_telluroamide_detected_primary(self):
        """R-C(=Te)-NH2."""
        mol = Chem.MolFromSmiles("CCC(N)=[Te]")
        assert "telluroamide" in detect_functional_groups(mol)

    def test_telluroamide_detected_N_sub(self):
        """R-C(=Te)-NHR'."""
        mol = Chem.MolFromSmiles("CCC(=[Te])NC")
        assert "telluroamide" in detect_functional_groups(mol)


@pytest.mark.unit
class TestFRNChalcogenCarbonylDetection:
    """Tier FRN-C — Chalcogen-on-aldehyde/ketone SMARTS detection (4 tests)."""

    def test_selenoaldehyde_detected(self):
        """R-C(=Se)H (selenal; P-66.6.3 parallel)."""
        mol = Chem.MolFromSmiles("CCC=[Se]")
        assert "selenoaldehyde" in detect_functional_groups(mol)

    def test_telluroaldehyde_detected(self):
        """R-C(=Te)H (tellural; P-66.6.3 parallel)."""
        mol = Chem.MolFromSmiles("CCC=[Te]")
        assert "telluroaldehyde" in detect_functional_groups(mol)

    def test_selenoketone_detected(self):
        """R-C(=Se)-R' (selone; P-66.6.3 parallel)."""
        mol = Chem.MolFromSmiles("CC(=[Se])C")
        assert "selenoketone" in detect_functional_groups(mol)

    def test_telluroketone_detected(self):
        """R-C(=Te)-R' (tellone; P-66.6.3 parallel)."""
        mol = Chem.MolFromSmiles("CC(=[Te])C")
        assert "telluroketone" in detect_functional_groups(mol)


@pytest.mark.unit
class TestFRNIminoesterDetection:
    """Tier FRN-D — Iminoester SMARTS detection (4 tests; baseline scope)."""

    def test_iminoester_detected_methyl_propanimidate(self):
        """R-C(=NH)-O-R' (P-65.1.7; "alkyl alkanimidate")."""
        mol = Chem.MolFromSmiles("CCC(=N)OC")
        assert "iminoester" in detect_functional_groups(mol)

    def test_iminoester_detected_methyl_benzimidate(self):
        """Ar-C(=NH)-O-R' (aryl variant)."""
        mol = Chem.MolFromSmiles("COC(=N)c1ccccc1")
        assert "iminoester" in detect_functional_groups(mol)

    def test_iminoester_N_substituted_out_of_baseline_scope(self):
        """AUDIT DECISION (AUDIT-FRN § 2.4): N-sub iminoesters deferred to 163.1."""
        mol = Chem.MolFromSmiles("CCC(=NC)OC")
        assert "iminoester" not in detect_functional_groups(mol)

    def test_iminoester_imidic_acid_free_form_deferred(self):
        """CONTEXT line 120: R-C(=NH)-OH (free imidic acid) deferred to 163.1."""
        mol = Chem.MolFromSmiles("CCC(=N)O")
        assert "iminoester" not in detect_functional_groups(mol)


@pytest.mark.unit
class TestFRNChalcogenEsterDetection:
    """Tier FRN-E — Chalcogen-ester SMARTS detection (2 tests)."""

    def test_selenoester_detected(self):
        """R-C(=O)-Se-R' (Se-alkyl alkaneselenoate; P-65.6 parallel)."""
        mol = Chem.MolFromSmiles("CCC(=O)[Se]C")
        assert "selenoester" in detect_functional_groups(mol)

    def test_telluroester_detected(self):
        """R-C(=O)-Te-R' (Te-alkyl alkanetelluroate; P-65.6 parallel)."""
        mol = Chem.MolFromSmiles("CCC(=O)[Te]C")
        assert "telluroester" in detect_functional_groups(mol)


@pytest.mark.unit
class TestFRNSuppressionMaps:
    """Suppression-map verification (8 tests per RESEARCH §8.1 Section B)."""

    def test_selenoic_Se_acid_suppresses_selenol(self):
        """[SeH] in C(=O)[SeH] must NOT also match selenol."""
        mol = Chem.MolFromSmiles("CCC(=O)[SeH]")
        groups = detect_functional_groups(mol)
        assert "selenol" not in groups

    def test_thioamide_suppresses_amine(self):
        """N in C(=S)N must NOT also match primary_amine."""
        mol = Chem.MolFromSmiles("CCC(N)=S")
        groups = detect_functional_groups(mol)
        assert "primary_amine" not in groups

    def test_iminoester_suppresses_imine(self):
        """=NH in C(=NH)OC must NOT also match imine."""
        mol = Chem.MolFromSmiles("CCC(=N)OC")
        groups = detect_functional_groups(mol)
        assert "imine" not in groups

    def test_iminoester_suppresses_ether(self):
        """OC in C(=NH)OC must NOT also match ether."""
        mol = Chem.MolFromSmiles("CCC(=N)OC")
        groups = detect_functional_groups(mol)
        assert "ether" not in groups

    def test_real_selenol_still_detected(self):
        """Pure selenol (no chalcogen acid) STILL matches selenol."""
        mol = Chem.MolFromSmiles("CC[SeH]")
        groups = detect_functional_groups(mol)
        assert "selenol" in groups

    def test_real_amine_still_detected_no_thioamide_overbroadening(self):
        """Risk A mitigation: plain amine does NOT spuriously match thioamide."""
        mol = Chem.MolFromSmiles("CCN")
        groups = detect_functional_groups(mol)
        assert "thioamide" not in groups
        assert "primary_amine" in groups

    def test_real_ester_still_detected_no_iminoester_overlap(self):
        """Risk B mitigation: plain ester does NOT spuriously match iminoester."""
        mol = Chem.MolFromSmiles("CCC(=O)OC")
        groups = detect_functional_groups(mol)
        assert "iminoester" not in groups
        assert "ester" in groups

    def test_real_thioester_still_detected(self):
        """No regression on existing thioester SMARTS after FRN-E additions."""
        mol = Chem.MolFromSmiles("CCC(=O)SC")
        groups = detect_functional_groups(mol)
        assert "thioester" in groups


@pytest.mark.unit
class TestFRNEdgeCasesAndNoRegression:
    """Edge cases + no regression on existing patterns (6 tests)."""

    def test_existing_thioic_S_acid_unchanged(self):
        """No regression: existing thioic_S_acid SMARTS still works."""
        mol = Chem.MolFromSmiles("CCC(=O)S")
        assert "thioic_S_acid" in detect_functional_groups(mol)

    def test_existing_thioketone_unchanged(self):
        """No regression: existing thioketone SMARTS still works."""
        mol = Chem.MolFromSmiles("CC(=S)C")
        assert "thioketone" in detect_functional_groups(mol)

    def test_existing_carboxylic_acid_unchanged(self):
        """No regression: existing carboxylic_acid SMARTS still works."""
        mol = Chem.MolFromSmiles("CCC(=O)O")
        assert "carboxylic_acid" in detect_functional_groups(mol)

    def test_all_16_FRN_smarts_compile(self):
        """All 16 new FRN SMARTS compile cleanly via Chem.MolFromSmarts."""
        frn_keys = [
            'selenoic_Se_acid', 'selenoic_O_acid', 'diselenoic_acid',
            'telluroic_Te_acid', 'telluroic_O_acid', 'ditelluroic_acid',
            'thioamide', 'selenoamide', 'telluroamide',
            'selenoaldehyde', 'telluroaldehyde', 'selenoketone', 'telluroketone',
            'iminoester', 'selenoester', 'telluroester',
        ]
        for key in frn_keys:
            assert key in PATTERNS, f"missing FRN SMARTS: {key}"
            patt = Chem.MolFromSmarts(PATTERNS[key])
            assert patt is not None, \
                f"SMARTS for {key} failed to compile: {PATTERNS[key]}"

    def test_predicate_purity_perception_no_mutation(self):
        """D-07 invariant: detect_functional_groups does not mutate the input mol."""
        smiles = "CCC(N)=S"
        mol1 = Chem.MolFromSmiles(smiles)
        n_atoms_before = mol1.GetNumAtoms()
        n_bonds_before = mol1.GetNumBonds()
        _ = detect_functional_groups(mol1)
        assert mol1.GetNumAtoms() == n_atoms_before
        assert mol1.GetNumBonds() == n_bonds_before

    def test_chalcogen_acid_distinct_atomic_numbers(self):
        """Risk I: SMARTS use chalcogen-specific atomic numbers; cannot match thio*.

        FRN-A SeH/Se SMARTS use [SeX2H1]/[SeX1] (atomic 34); FRN-A TeH/Te
        SMARTS use [TeX2H1]/[TeX1] (atomic 52). These cannot match the
        v18 thio* SMARTS that use [SX2H1]/[SX1] (atomic 16). Empirical
        proof: a plain dithioic acid does NOT match any FRN-A SMARTS.
        """
        mol = Chem.MolFromSmiles("CCC(=S)S")  # dithioic acid (v18 thio*)
        groups = detect_functional_groups(mol)
        # The 6 chalcogen-acid keys must NOT spuriously match
        for k in ['selenoic_Se_acid', 'selenoic_O_acid', 'diselenoic_acid',
                  'telluroic_Te_acid', 'telluroic_O_acid', 'ditelluroic_acid']:
            assert k not in groups, f"FRN-A SMARTS {k} spuriously matched thio* SMILES"
        # The existing dithioic_acid IS matched (no regression)
        assert "dithioic_acid" in groups
