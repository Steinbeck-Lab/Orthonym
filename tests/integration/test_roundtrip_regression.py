"""
Permanent round-trip regression test suite (Plan 15-08).

Contains compounds that produce verified correct IUPAC names,
confirmed via OPSIN round-trip (name -> OPSIN -> canonical SMILES match)
and/or manual verification against IUPAC 2013 rules.

These tests MUST NEVER regress. Any failure indicates a naming rule
was accidentally broken by later changes.

Coverage:
- Simple acyclic (alkanes, alkenes, alkynes)
- Functional groups (alcohols, aldehydes, ketones, acids)
- Aromatic (benzene derivatives, naphthalene)
- Heterocyclic (pyridine, furan, pyrrole, thiophene)
- Stereochemistry (E/Z, R/S)
- Polycyclic (von Baeyer, steroids)
- Retained names (benzene, phenol, aniline, etc.)
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Section 1: Round-trip verified compounds (OPSIN confirmed)
# These produced names that OPSIN parsed back to the original SMILES.
# ---------------------------------------------------------------------------

ROUNDTRIP_VERIFIED = [
    # Acyclic functional compounds
    ("CCCCC/C=C\\CC/C=C/C=O", "(2E,6Z)-dodeca-2,6-dienal"),
    ("CCCCCCCCCCC/C=C/CC/C=C/C(=O)O", "(2E,6E)-octadeca-2,6-dienoic acid"),
    ("C[C@H](O)C(=O)CC(=O)C(=O)O", "(5S)-5-hydroxy-2,4-dioxohexanoic acid"),
    ("CCN(CC)CC", "triethylamine"),
    ("CCCCCCCCCCCC/C=C/C(=O)O", "(2E)-pentadec-2-enoic acid"),
    ("CCCCCCC#CCCCC(=O)O", "dodec-5-ynoic acid"),
    ("CC/C=C\\C/C=C\\C/C=C\\CC#CCCCCC(=O)O", "(9Z,12Z,15Z)-octadeca-9,12,15-trien-6-ynoic acid"),
    ("CCCCCCCCCCCCCCCCCCOCC(O)CO", "3-octadecyloxy-2-hydroxypropan-1-ol"),
    ("CC/C(C)=C\\CC/C(C)=C/CC(C)C(C)CC=O", "(6E,10Z)-3,4,7,11-tetramethyltrideca-6,10-dienal"),
    ("O=C(O)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCO", "31-hydroxyhentriacontanoic acid"),
    ("CCCCCCCCCC/C=C/CCCCCCCCCC(=O)O", "(11E)-docos-11-enoic acid"),
    ("CC/C=C/C=C\\CCCCCC(=O)O", "(7Z,9E)-dodeca-7,9-dienoic acid"),
    ("CCCCCCCCCCCCCC(O)CC", "hexadecan-3-ol"),
    ("CCCCCCCCCCCCCC=CC(O)C(N)CO", "2-amino-3-hydroxyoctadec-4-en-1-ol"),
    ("OC/C=C/C#CC#C/C=C/C=C/C(O)CCO", "(2E,8E,10E)-12-hydroxytetradeca-2,8,10-trien-4,6-diyne-1,14-diol"),
    ("CCCCCCCCCCCCCCC(O)[C@@H](O)[C@@H](N)CO", "(2S,3S)-2-amino-3,4-dihydroxyoctadecan-1-ol"),
    (
        "CCCCCC(O)CC(=O)CCCCCC(O)CC(=O)CCCCCC(O)CC(=O)CCCCCC(O)CC(O)CC(C)O",
        "6,14,22,30,32,34-hexahydroxypentatriacontane-8,16,24-trione",
    ),

    # Acyclic hydrocarbon
    ("C/C=C/CCCCCCCCC", "(2E)-dodec-2-ene"),

    # Aromatic compounds
    ("Cc1ccc(N)cc1N", "2,4-diamino-1-methylbenzene"),
    ("OCc1ccc(O)cc1", "1-hydroxy-4-hydroxymethylbenzene"),
    ("CN(C)c1ccc(N)cc1", "1-amino-4-(N,N-dimethylamino)benzene"),

    # Fused aromatic
    ("O=C(O)Cc1c[nH]c2ccc(Cl)cc12", "3-carboxymethyl-5-chloro-1H-indole"),

    # Heterocyclic
    ("C1=CN1", "azirene"),

    # Polycyclic (von Baeyer)
    (
        "CC1(C)CC=C[C@]2(C)OO[C@@H]3C[C@@]12CC[C@H]3O",
        "(1S,4R,5R,8S)-1,9,9-trimethyl-2,3-dioxa-tricyclo[6.4.0.1(4,8)]tridec-11-en-5-ol",
    ),
    (
        "CC1(C)C(O)C(O)CC2(C)C1CCC13CC(CCC21)C1(C)OC31",
        "5,5,9,14-tetramethyl-15-oxa-pentacyclo[8.6.0.1(1,13).0(4,9).0(14,16)]heptadecan-6,7-diol",
    ),
    (
        "CC1(C)CCC2(C(=O)O)CCC3(C)C(=CCC4C5(C)CC(O)C(=O)C(C)(C)C5CCC43C)C2C1",
        "8-hydroxy-1,2,6,6,10,17,17-heptamethyl-7-oxo-pentacyclo[12.8.0.0(15,20).0(2,11).0(5,10)]docos-13-ene-20-carboxylic acid",
    ),

    # Steroid (natural product)
    (
        "CC(C)C(=O)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4=CCCC[C@]4(C)[C@H]3CC[C@]12C",
        "cholest-4-en-24-one",
    ),
]


# ---------------------------------------------------------------------------
# Section 2: Core naming foundations (manually verified)
# These are fundamental IUPAC naming rules that must always work.
# ---------------------------------------------------------------------------

CORE_NAMING = [
    # Alkanes
    ("C", "methane"),
    ("CC", "ethane"),
    ("CCC", "propane"),
    ("CCCC", "butane"),
    ("CCCCC", "pentane"),

    # Alcohols
    ("CCO", "ethanol"),
    ("CCCO", "propan-1-ol"),
    ("CC(O)C", "propan-2-ol"),

    # Aldehydes
    ("CC=O", "acetaldehyde"),
    ("CCC=O", "propanal"),

    # Ketones
    ("CC(=O)C", "acetone"),

    # Carboxylic acids
    ("CC(=O)O", "acetic acid"),
    ("CCC(=O)O", "propanoic acid"),
    ("CCCC(=O)O", "butanoic acid"),

    # Alkenes and alkynes
    ("C=C", "ethene"),
    ("CC=CC", "but-2-ene"),
    ("C#C", "ethyne"),

    # Nitriles and amides
    ("CC#N", "acetonitrile"),
    ("CC(=O)N", "acetamide"),

    # Halogenated
    ("CCl", "chloromethane"),
    ("CCBr", "bromoethane"),

    # Cycloalkanes
    ("C1CC1", "cyclopropane"),
    ("C1CCC1", "cyclobutane"),
    ("C1CCCC1", "cyclopentane"),
    ("C1CCCCC1", "cyclohexane"),

    # Retained aromatic names
    ("c1ccccc1", "benzene"),
    ("Cc1ccccc1", "toluene"),
    ("Oc1ccccc1", "phenol"),
    ("Nc1ccccc1", "aniline"),
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("OC(=O)c1ccccc1", "benzoic acid"),

    # Retained heterocyclic names
    ("c1ccncc1", "pyridine"),
    ("c1ccoc1", "furan"),
    ("c1cc[nH]c1", "pyrrole"),
    ("c1ccsc1", "thiophene"),

    # Simple ethers
    ("COC", "methoxymethane"),

    # Esters (retained)
    ("CC(=O)OC", "methyl acetate"),
    ("CC(=O)OCC", "ethyl acetate"),

    # Amino acids (retained)
    ("NCC(=O)O", "glycine"),
    ("CC(N)C(=O)O", "alanine"),

    # Cycloalkenes
    ("C1=CCCCC1", "cyclohexene"),

    # Saturated heterocycles (retained)
    ("C1CCNCC1", "piperidine"),
    ("C1CCOCC1", "tetrahydropyran"),
    ("C1CCOC1", "tetrahydrofuran"),

    # Stereochemistry
    ("C/C=C/C", "(2E)-but-2-ene"),
    ("C/C=C\\C", "(2Z)-but-2-ene"),

    # Sulfur compounds
    ("CS", "methanethiol"),
    ("CS(=O)(=O)C", "dimethyl sulfone"),

    # Retained names
    ("OCCO", "ethylene glycol"),
    ("OCC(O)CO", "glycerol"),
]


class TestRoundTripRegression:
    """Tests from OPSIN round-trip verified compounds.

    Each compound generated a name that OPSIN parsed back to
    the exact original canonical SMILES. These MUST NOT regress.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_name", ROUNDTRIP_VERIFIED,
                             ids=[f"rt-{i}" for i in range(len(ROUNDTRIP_VERIFIED))])
    def test_roundtrip_verified(self, smiles, expected_name):
        """Verify round-trip confirmed names never regress."""
        name = name_compound(smiles)
        assert name == expected_name, (
            f"REGRESSION: {smiles}\n"
            f"  Expected: {expected_name}\n"
            f"  Got:      {name}"
        )


class TestCoreNamingRegression:
    """Tests for fundamental IUPAC naming rules.

    These cover the most basic compound classes and retained names
    that form the foundation of all naming. Failures here indicate
    critical regressions.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_name", CORE_NAMING,
                             ids=[f"core-{i}" for i in range(len(CORE_NAMING))])
    def test_core_naming(self, smiles, expected_name):
        """Verify core naming foundations never regress."""
        name = name_compound(smiles)
        assert name == expected_name, (
            f"REGRESSION: {smiles}\n"
            f"  Expected: {expected_name}\n"
            f"  Got:      {name}"
        )


class TestNamingNeverCrashes:
    """Smoke tests for compounds that previously caused crashes.

    These don't check exact names, just that name_compound() returns
    a string without raising an exception.
    """

    CRASH_PRONE_SMILES = [
        # Ion/salt compounds (previously caused recursion)
        "[NH4+]",
        "[Na+].[Cl-]",
        "CC(=O)[O-]",
        # Complex polycyclic
        "C1CC2CCCC(C1)C2",
        # Large ring
        "C1CCCCCCCCCCCCCCCCCCCC1",
        # Fused 3-ring (anthracene)
        "c1ccc2cc3ccccc3cc2c1",
        # Bridged bicyclic
        "C1CC2CCC1C2",
    ]

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles", CRASH_PRONE_SMILES)
    def test_no_crash(self, smiles):
        """Verify naming never crashes (may return fallback name)."""
        try:
            name = name_compound(smiles)
            assert isinstance(name, str), f"Expected string, got {type(name)}"
            assert len(name) > 0, f"Empty name for {smiles}"
        except RecursionError:
            pytest.fail(f"RecursionError for {smiles} - infinite loop in naming")
        except Exception as e:
            # Allow other exceptions but document them
            pytest.fail(f"Unexpected crash for {smiles}: {type(e).__name__}: {e}")
