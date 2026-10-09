"""a phase canary tests: freeze compound names before assembly fixes.

These tests lock down the CURRENT output of compounds relevant to a phase
(BUG-B guard, format, NP steroid). If a a phase fix
changes the output, the canary must be updated deliberately (not silently).

Canary compounds exercise:
- BUG-B guard: FGs (hydroxy, amino, halogen) on small substituent branches
- Polyfunctional prefix completeness: multiple FGs on chain compounds
- Ring polyfunctional: FGs on ring + FGs on branches
- Ring prefix conversion: heterocyclic ring substituents (Plan 03)
- Unsaturation format: 'a' euphonic connector (Plan 02)
- IUPAC (a): all non-principal groups as prefixes
"""

import pytest
from orthonym.namer import name_compound

# a phase canary compounds: frozen CURRENT output
PHASE105_CANARY = [
    # -- BUG-B guard: hydroxy/amino/halogen on small substituent branches --
    ("OCC(CCC)C(=O)O", "2-(hydroxymethyl)pentanoic acid"),
    ("NCC(CCC)C(=O)O", "2-(aminomethyl)pentanoic acid"),
    ("FCC(CC)C(=O)O", "2-(fluoromethyl)butanoic acid"),
    ("ClCC(CCCC)C(=O)O", "2-(chloromethyl)hexanoic acid"),
    # -- Polyfunctional chain compounds --
    ("OCC(CC(=O)O)CC=O", "3-(hydroxymethyl)-5-oxopentanoic acid"),
    ("OC(CC)CC(=O)O", "3-hydroxypentanoic acid"),
    # 'butyrine' is not a retained amino-acid name: 'Systematic substitutive names are
    # given to homologues of glycine and alanine, for example 2-aminobutanoic acid'
    #, the Blue Book). The id keeps the original spelling (stable row identity).
    pytest.param("NC(CC)C(=O)O", "2-aminobutanoic acid", id="butyrine"),
    ("OC(CC=O)CC(=O)O", "3-hydroxy-5-oxopentanoic acid"),
    # -- Ring polyfunctional --
    # 'cyclohexane-1-carboxylic acid': the ring name keeps its 'e' before the
    # consonant of 'carboxylic acid' (a), the Blue Book; 'cyclohexane-1-carboxylic
    # acid', the Blue Book). The id keeps the original spelling.
    pytest.param("OC1CCCCC1C(=O)O", "2-hydroxycyclohexane-1-carboxylic acid",
                 id="2-hydroxycyclohexan-1-carboxylic_acid"),
    # -- Multi-FG complexity --
    # (g) (the Blue Book): {2,4} ties, and 'amino' is cited before
    # 'aminomethyl':3448), so amino takes 2 (d3c164db2, plan ruling
    # R26; the old snapshot '4-amino-2-(aminomethyl)...' numbered from the other end).
    ("NCC(CC(N)C(=O)O)C(=O)O", "2-amino-4-(aminomethyl)pentanedioic acid"),
    # -- Ring prefix conversion (Plan 03): heterocyclic -ane ring names --
    # Acetic acid is the retained PIN and takes substituents, the Blue Book), the
    # ring is a substituent prefix with a locant, enclosed: '(1,3-thiazol-2-yl)acetic
    # acid (PIN)', the Blue Book. The ids keep the original spellings.
    pytest.param("O=C(O)CC1CCOCC1", "(oxan-4-yl)acetic acid",
                 id="2-oxanylethanoic_acid"),
    pytest.param("O=C(O)CC1CCNCC1", "(piperidin-4-yl)acetic acid",
                 id="2-piperidinylethanoic_acid"),
    pytest.param("O=C(O)CC1CCCO1", "(oxolan-2-yl)acetic acid",
                 id="2-oxolanylethanoic_acid"),
    # -- Unsaturation format (Plan 02): 'a' euphonic connector --
    ("CC/C=C\\C/C=C\\C/C=C\\CCCCCCCC(=O)O",
     "(9Z,12Z,15Z)-octadeca-9,12,15-trienoic acid"),
]

_CANARY_IDS = [
    # a pytest.param row carries its own stable id
    row.id if hasattr(row, "id")
    else row[1].replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")[:50]
    for row in PHASE105_CANARY
]


@pytest.mark.parametrize("smiles,expected", PHASE105_CANARY, ids=_CANARY_IDS)
def test_phase105_canary(smiles, expected):
    """a phase name-stability canary: output must not change unexpectedly."""
    result = name_compound(smiles)
    assert result == expected, (
        f"PHASE 105 CANARY REGRESSION: {smiles}\n"
        f"  Expected: {expected}\n"
        f"  Got: {result}"
    )
