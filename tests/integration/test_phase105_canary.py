"""Phase 105 canary tests: freeze compound names before assembly fixes.

These tests lock down the CURRENT output of compounds relevant to Phase 105
(FIX-02 BUG-B guard, FIX-03 format, FIX-09 NP steroid). If a Phase 105 fix
changes the output, the canary must be updated deliberately (not silently).

Canary compounds exercise:
- BUG-B guard: FGs (hydroxy, amino, halogen) on small substituent branches
- Polyfunctional prefix completeness: multiple FGs on chain compounds
- Ring polyfunctional: FGs on ring + FGs on branches
- Ring prefix conversion: heterocyclic ring substituents (Plan 03)
- Unsaturation format: 'a' euphonic connector (Plan 02)
- IUPAC P-59.1(a): all non-principal groups as prefixes
"""

import pytest
from orthonym.namer import name_compound

# Phase 105 canary compounds: frozen CURRENT output
PHASE105_CANARY = [
    # -- BUG-B guard: hydroxy/amino/halogen on small substituent branches --
    ("OCC(CCC)C(=O)O", "2-(hydroxymethyl)pentanoic acid"),
    ("NCC(CCC)C(=O)O", "2-(aminomethyl)pentanoic acid"),
    ("FCC(CC)C(=O)O", "2-(fluoromethyl)butanoic acid"),
    ("ClCC(CCCC)C(=O)O", "2-(chloromethyl)hexanoic acid"),
    # -- Polyfunctional chain compounds --
    ("OCC(CC(=O)O)CC=O", "3-(hydroxymethyl)-5-oxopentanoic acid"),
    ("OC(CC)CC(=O)O", "3-hydroxypentanoic acid"),
    ("NC(CC)C(=O)O", "butyrine"),
    ("OC(CC=O)CC(=O)O", "3-hydroxy-5-oxopentanoic acid"),
    # -- Ring polyfunctional --
    ("OC1CCCCC1C(=O)O", "2-hydroxycyclohexan-1-carboxylic acid"),
    # -- Multi-FG complexity --
    ("NCC(CC(N)C(=O)O)C(=O)O", "4-amino-2-(aminomethyl)pentanedioic acid"),
    # -- Ring prefix conversion (Plan 03): heterocyclic -ane ring names --
    ("O=C(O)CC1CCOCC1", "2-oxanylethanoic acid"),
    ("O=C(O)CC1CCNCC1", "2-piperidinylethanoic acid"),
    ("O=C(O)CC1CCCO1", "2-oxolanylethanoic acid"),
    # -- Unsaturation format (Plan 02): 'a' euphonic connector --
    ("CC/C=C\\C/C=C\\C/C=C\\CCCCCCCC(=O)O",
     "(9Z,12Z,15Z)-octadeca-9,12,15-trienoic acid"),
]

_CANARY_IDS = [
    name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")[:50]
    for _, name in PHASE105_CANARY
]


@pytest.mark.parametrize("smiles,expected", PHASE105_CANARY, ids=_CANARY_IDS)
def test_phase105_canary(smiles, expected):
    """Phase 105 name-stability canary: output must not change unexpectedly."""
    result = name_compound(smiles)
    assert result == expected, (
        f"PHASE 105 CANARY REGRESSION: {smiles}\n"
        f"  Expected: {expected}\n"
        f"  Got: {result}"
    )
