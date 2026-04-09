"""
Decomposition regression canary tests (Phase 39, Plan 04).

Verifies that molecules previously producing correct IUPAC names
continue to produce the EXACT same names after the decomposition
engine is integrated into namer.py.

The decomposition engine includes a quality gate that should prevent
it from activating on well-named molecules. These canary tests detect
quality gate failures: if any canary produces a different name,
the gate has a bug.

Sources for canary molecules:
- ROUNDTRIP_VERIFIED from test_roundtrip_regression.py (OPSIN-confirmed)
- CORE_NAMING from test_roundtrip_regression.py (manually verified)
- Additional ester/amide-containing molecules (highest regression risk)
"""

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# Section 1: Basic foundations (no ester/amide bonds)
# These should NEVER be affected by decomposition.
# ---------------------------------------------------------------------------

BASIC_CANARIES = [
    ("C", "methane"),
    ("CC", "ethane"),
    ("CCC", "propane"),
    ("CCCC", "butane"),
    ("CCO", "ethanol"),
    ("CCCO", "propan-1-ol"),
    ("CC(O)C", "propan-2-ol"),
    ("CC=O", "acetaldehyde"),
    ("CCC=O", "propanal"),
    ("CC(=O)C", "propan-2-one"),
    ("CC(=O)O", "acetic acid"),
    ("CCC(=O)O", "propanoic acid"),
    ("CCCC(=O)O", "butanoic acid"),
    ("C=C", "ethene"),
    ("C#C", "acetylene"),  # retained name (P-31.1.2.1 PIN)
    ("c1ccccc1", "benzene"),
    ("Cc1ccccc1", "toluene"),
    ("Oc1ccccc1", "phenol"),
    ("Nc1ccccc1", "aniline"),
    ("OC(=O)c1ccccc1", "benzoic acid"),
    ("C1CCCCC1", "cyclohexane"),
    ("c1ccncc1", "pyridine"),
    ("c1ccoc1", "furan"),
    ("c1cc[nH]c1", "pyrrole"),
    ("C1CCNCC1", "piperidine"),
    ("CS", "methanethiol"),
    ("OCCO", "ethylene glycol"),
    ("OCC(O)CO", "glycerol"),
    ("C/C=C/C", "(2E)-but-2-ene"),
    ("C/C=C\\C", "(2Z)-but-2-ene"),
]


class TestBasicCanaries:
    """Basic molecules with no ester/amide bonds.
    Decomposition should NEVER even consider these."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected",
        BASIC_CANARIES,
        ids=[f"basic-{i}" for i in range(len(BASIC_CANARIES))],
    )
    def test_basic_canary(self, smiles, expected):
        name = name_compound(smiles)
        assert name == expected, (
            f"DECOMPOSITION REGRESSION: {smiles}\n"
            f"  Expected: {expected}\n"
            f"  Got:      {name}"
        )


# ---------------------------------------------------------------------------
# Section 2: Ester-containing molecules that name correctly
# These are the highest regression risk: they have cleavable ester bonds
# but the quality gate should leave them alone.
# ---------------------------------------------------------------------------

ESTER_CANARIES = [
    # Simple esters (existing pipeline handles well)
    ("CC(=O)OC", "methyl acetate"),
    ("CC(=O)OCC", "ethyl acetate"),
    ("CCOC(=O)c1ccccc1", "ethyl benzoate"),
    ("CCCCOC(=O)c1ccccc1", "butyl benzoate"),
    ("CC(C)OC(=O)c1ccccc1", "propan-2-yl benzoate"),
    ("CCCC(=O)OCCC", "propyl butanoate"),
    ("CC(=O)OCCCC", "butyl acetate"),
    ("CCCCCCCC(=O)OC", "methyl octanoate"),
    ("CC(=O)OC(C)C", "propan-2-yl acetate"),
    ("CCCCCCCCCCCCCCCC(=O)OC", "methyl palmitate"),
    # Ester on aromatic (acetyloxy pattern)
    ("CC(=O)Oc1ccccc1", "acetyloxybenzene"),
    # Formate esters
    ("O=COCC", "ethyl formate"),
    # NP ester
    (
        "CC(=O)O[C@H]1CC[C@@H]2[C@@]1(C)CC[C@H]1[C@@H]2CCC2=CC(=O)CC[C@@]12C",
        "(8S,9S,10S,13R,14S,17S)-3-oxoandrost-4-en-17-yl acetate",
    ),
]


class TestEsterCanaries:
    """Molecules with ester bonds that already name correctly.
    The quality gate must NOT allow decomposition to change these."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected",
        ESTER_CANARIES,
        ids=[f"ester-{i}" for i in range(len(ESTER_CANARIES))],
    )
    def test_ester_canary(self, smiles, expected):
        name = name_compound(smiles)
        assert name == expected, (
            f"DECOMPOSITION REGRESSION (ester): {smiles}\n"
            f"  Expected: {expected}\n"
            f"  Got:      {name}\n"
            f"  The quality gate may have incorrectly triggered decomposition."
        )


# ---------------------------------------------------------------------------
# Section 3: Amide-containing molecules that name correctly
# Also high regression risk.
# ---------------------------------------------------------------------------

AMIDE_CANARIES = [
    # Simple amides
    ("CC(=O)N", "acetamide"),
    ("CC#N", "acetonitrile"),
    ("CC(=O)Nc1ccccc1", "N-phenylacetamide"),
    ("CC(=O)N(C)C", "N,N-dimethylacetamide"),
    ("CC(=O)NCC", "N-ethylacetamide"),  # Note: existing pipeline name
    ("CCCC(=O)NCC", "N-ethylbutanamide"),
    ("CC(=O)NCCC", "N-propylacetamide"),
    # Amino acids (contain amide-like bonds)
    ("NCC(=O)O", "glycine"),
    ("CC(N)C(=O)O", "alanine"),
    # Peptides (contain amide bonds, handled by peptide route)
    ("NCC(=O)NCC(=O)O", "glycylglycine"),
]


class TestAmideCanaries:
    """Molecules with amide bonds that already name correctly.
    The quality gate must NOT allow decomposition to change these."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected",
        AMIDE_CANARIES,
        ids=[f"amide-{i}" for i in range(len(AMIDE_CANARIES))],
    )
    def test_amide_canary(self, smiles, expected):
        name = name_compound(smiles)
        assert name == expected, (
            f"DECOMPOSITION REGRESSION (amide): {smiles}\n"
            f"  Expected: {expected}\n"
            f"  Got:      {name}\n"
            f"  The quality gate may have incorrectly triggered decomposition."
        )


# ---------------------------------------------------------------------------
# Section 4: Round-trip verified molecules (from OPSIN confirmation)
# These are the gold standard -- OPSIN confirmed the names.
# ---------------------------------------------------------------------------

ROUNDTRIP_CANARIES = [
    ("CCN(CC)CC", "triethylamine"),
    ("CCCCCCCCCCCC/C=C/C(=O)O", "(2E)-pentadec-2-enoic acid"),
    ("CCCCCCC#CCCCC(=O)O", "dodec-5-ynoic acid"),
    ("CCCCCCCCCC/C=C/CCCCCCCCCC(=O)O", "(11E)-docos-11-enoic acid"),
    ("CCCCCCCCCCCCCC(O)CC", "hexadecan-3-ol"),
    ("C/C=C/CCCCCCCCC", "(2E)-dodec-2-ene"),
    ("Cc1ccc(N)cc1N", "2,4-diamino-1-methylbenzene"),
    ("OCc1ccc(O)cc1", "4-(hydroxymethyl)phenol"),  # ASML-13: phenol suffix routing
    ("c1ccc2ccccc2c1", "naphthalene"),
    # Fatty acid with Z geometry
    ("CCCCCCCC/C=C\\CCCCCCCC(=O)O", "(9Z)-octadec-9-enoic acid"),
]


class TestRoundTripCanaries:
    """Molecules confirmed by OPSIN round-trip that must not regress."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected",
        ROUNDTRIP_CANARIES,
        ids=[f"rt-{i}" for i in range(len(ROUNDTRIP_CANARIES))],
    )
    def test_roundtrip_canary(self, smiles, expected):
        name = name_compound(smiles)
        assert name == expected, (
            f"DECOMPOSITION REGRESSION (roundtrip): {smiles}\n"
            f"  Expected: {expected}\n"
            f"  Got:      {name}"
        )


# ---------------------------------------------------------------------------
# Section 5: Phase 139.1 Plan 03 -- recovered compounds via DECO-22/25
# Ester threshold lowered to 2 (enables diester decomposition) and
# partial assembly (recovers names when 2/3+ fragments succeed).
# ---------------------------------------------------------------------------

DECO_22_25_CANARIES = [
    # Diester decomposition (DECO-22: ester threshold 2)
    ("COC(=O)CC(=O)OC", "dimethyl propanedioate"),
    ("CCOC(=O)CCC(=O)OCC", "diethyl butanedioate"),
    ("COC(=O)CCCCC(=O)OC", "dimethyl hexanedioate"),
    ("COC(=O)CC(=O)OCC", "ethyl methyl propanedioate"),
    ("COC(=O)CCCC(=O)OC", "dimethyl pentanedioate"),
    # Triester via multi-bond ester cleavage
    ("CC(=O)OCC(COC(C)=O)OC(C)=O", "1,2,3-tris(acetyloxy)propane"),
]


class TestDeco22And25Canaries:
    """Compounds recovered by DECO-22 (ester threshold 2) and DECO-25 (partial assembly).
    These must not regress in future phases."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected",
        DECO_22_25_CANARIES,
        ids=[f"deco-{i}" for i in range(len(DECO_22_25_CANARIES))],
    )
    def test_deco_canary(self, smiles, expected):
        name = name_compound(smiles)
        assert name == expected, (
            f"DECOMPOSITION REGRESSION (DECO-22/25): {smiles}\n"
            f"  Expected: {expected}\n"
            f"  Got:      {name}\n"
            f"  DECO-22 (ester threshold 2) or DECO-25 (partial assembly) may have regressed."
        )
