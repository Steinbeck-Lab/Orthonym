"""
Benzene handler accuracy test suite from benchmark failures.

Tests extracted from ChEBI 500 benchmark diagnostics (Phase 142, D-02).
Covers retained-base name routing for benzaldehyde, phenol, benzoic acid,
aniline, and acetophenone compound classes.

IUPAC 2013 Rules:
- P-66.6.3.1.1: benzaldehyde is an IUPAC retained name for benzenecarbaldehyde
- P-65.1.2.1: benzoic acid retained name for benzenecarboxylic acid
- P-63.1.1.1: phenol retained name for benzenol
- P-62.2.3.1: aniline retained name for benzenamine
- P-66.6.4.1.1: acetophenone retained name for 1-phenylethan-1-one

All expected names verified against OPSIN 2.9.0 round-trip (parse to SMILES).
"""

import pytest
from orthonym.namer import name_compound


# === Benzaldehyde retained base (Phase 142-01 fix) ===
# These previously produced "benzenecarbaldehyde" instead of "benzaldehyde".
# Fixed by adding carbaldehyde routing in _assemble_benzene_with_suffix().

BENZALDEHYDE_CASES = [
    # Unsubstituted benzaldehyde (retained name lookup)
    ("O=Cc1ccccc1", "benzaldehyde"),
    # Halogenated benzaldehydes
    ("O=Cc1ccc(Cl)cc1", "4-chlorobenzaldehyde"),
    ("O=Cc1ccc(F)cc1", "4-fluorobenzaldehyde"),
    ("O=Cc1ccc(Br)cc1", "4-bromobenzaldehyde"),
    # Nitro substituted
    ("O=Cc1ccc([N+](=O)[O-])cc1", "4-nitrobenzaldehyde"),
    # Amino substituted
    ("O=Cc1ccc(N)cc1", "4-aminobenzaldehyde"),
    # Hydroxyl substituted (aldehyde > hydroxyl in seniority, so OH is prefix)
    ("O=Cc1ccccc1O", "2-hydroxybenzaldehyde"),
    ("O=Cc1ccc(O)c(O)c1", "3,4-dihydroxybenzaldehyde"),
    ("O=Cc1cc(O)c(O)c(O)c1", "3,4,5-trihydroxybenzaldehyde"),
    # Alkyl substituted. The xfail here (substituent named "isopropyl" instead of
    # "propan-2-yl") was STALE -- it XPASSed on 2026-07-28, so the assertion is live now.
    # It was non-strict, which is why it sat unnoticed after the underlying fix landed.
    ("CC(C)c1ccc(C=O)cc1", "4-(propan-2-yl)benzaldehyde"),
]

# === Phenol, benzoic acid, and other retained bases (regression checks) ===

RETAINED_BASE_REGRESSION_CASES = [
    # Phenol retained base
    ("Oc1ccc(Cl)cc1", "4-chlorophenol"),
    ("Nc1ccc(O)cc1", "4-aminophenol"),
    # Benzoic acid retained base
    ("OC(=O)c1ccc(Cl)c(Cl)c1", "3,4-dichlorobenzoic acid"),
    # Aniline retained base
    ("Nc1ccccc1", "aniline"),
    # Acetophenone: Wave2 T1d de-headlined to the substitutive PIN (P-64.2.1.2)
    ("CC(=O)c1ccccc1", "1-phenylethan-1-one"),
    ("CC(=O)c1ccc(O)cc1", "4-hydroxyacetophenone"),
]

# === Multi-OH benzene: systematic naming ===

MULTI_OH_CASES = [
    # Per IUPAC 2013 P-63.1.1.1, benzene-X,Y-diol is the preferred IUPAC name.
    # ✅ FIXED 2026-07-28, v29 Phase C Task 10 -- these two xfails XPASSed and are now
    # live assertions. They had recorded exactly the right diagnosis ("returns retained
    # name ... instead of systematic PIN") and sat non-strict, so nothing announced the
    # fix. `hydroquinone` and `pyrogallol` now carry `pin: false` rows in
    # data/iupac_2013_pin_list.json: BB:26806/:26808 print `hydroquinone` /
    # `benzene-1,4-diol (PIN)`, and `pyrogallol` appears nowhere in the Blue Book at all,
    # so the systematic form governs by default.
    # ⚠ Denying `hydroquinone` first UNMASKED `quinol`, a second non-PIN synonym for the
    # same structure -- invariant 11 -- which needed its own row.
    ("Oc1ccc(O)cc1", "benzene-1,4-diol"),
    ("Oc1cccc(O)c1O", "benzene-1,2,3-triol"),
]

# Combine all cases for the main parametrized test
BENZENE_ACCURACY_CASES = (
    BENZALDEHYDE_CASES
    + RETAINED_BASE_REGRESSION_CASES
    + MULTI_OH_CASES
)


@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected", BENZENE_ACCURACY_CASES)
def test_benzene_handler_accuracy(smiles, expected):
    """Test benzene handler produces correct retained-base IUPAC names.

    Covers benzaldehyde, phenol, benzoic acid, aniline, and acetophenone
    compound classes. Expected names verified against OPSIN 2.9.0.
    """
    result = name_compound(smiles)
    assert result == expected, f"Got '{result}' for {smiles}"
