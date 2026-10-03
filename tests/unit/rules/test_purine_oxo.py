"""Unit tests for the substituted purine-2,6-dione naming engine.

Every structure here is OPSIN-authoritative (canonical SMILES obtained by parsing
the accepted name through OPSIN), and every expected PIN round-trips to that
structure (verified 2026-07-17). The engine names N-/C8-substituted
purine-2,6-diones systematically retained purine parent + fixed
numbering + dione), and FAILS CLOSED (returns None) on
anything outside that scope so a wrong name is never emitted.
"""
import pytest
from rdkit import Chem

from orthonym.rules.purine_oxo import name_purine_26_dione

pytestmark = pytest.mark.unit

# --- OPSIN-authoritative structures ------------------------------------------
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
THEOBROMINE = "Cn1cnc2c1c(=O)[nH]c(=O)n2C"
THEOPHYLLINE = "Cn1c(=O)c2[nH]cnc2n(C)c1=O"
PARAXANTHINE = "Cn1c(=O)[nH]c2ncn(C)c2c1=O"
X1_METHYL = "Cn1c(=O)[nH]c2nc[nH]c2c1=O"
X3_METHYL = "Cn1c(=O)[nH]c(=O)c2[nH]cnc21"
X7_METHYL = "Cn1cnc2[nH]c(=O)[nH]c(=O)c21"
X1ET_37DIMETHYL = "CCn1c(=O)c2c(ncn2C)n(C)c1=O"
X8BR_CAFFEINE = "Cn1c(=O)c2c(nc(Br)n2C)n(C)c1=O"

BARE_XANTHINE = "O=c1[nH]c(=O)c2[nH]cnc2[nH]1"
X8_CARBOXY = "Cn1c(=O)c2c(nc(C(=O)O)n2C)n(C)c1=O"
X8_HYDROXY = "Cn1c(=O)c2c(nc(O)n2C)n(C)c1=O"
# purine-2,6,8-TRIONE family (uric acid + N-methyls): the C8=O is part of a
# DIFFERENT parent (-2,6,8-trione). The 2,6-dione engine must NOT drop it and
# emit a dione name — that would name a different molecule. Fail closed.
TETRAMETHYLURIC = "Cn1c(=O)c2c(n(C)c1=O)n(C)c(=O)n2C"
DIMETHYLURIC_8OXO = "Cn1c(=O)c2[nH]c(=O)[nH]c2n(C)c1=O"


@pytest.mark.parametrize("smiles,expected", [
    (CAFFEINE,        "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione"),
    (THEOBROMINE,     "3,7-dimethyl-3,7-dihydro-1H-purine-2,6-dione"),
    (THEOPHYLLINE,    "1,3-dimethyl-3,7-dihydro-1H-purine-2,6-dione"),
    (PARAXANTHINE,    "1,7-dimethyl-3,7-dihydro-1H-purine-2,6-dione"),
    (X1_METHYL,       "1-methyl-3,7-dihydro-1H-purine-2,6-dione"),
    (X3_METHYL,       "3-methyl-3,7-dihydro-1H-purine-2,6-dione"),
    (X7_METHYL,       "7-methyl-3,7-dihydro-1H-purine-2,6-dione"),
    (X1ET_37DIMETHYL, "1-ethyl-3,7-dimethyl-3,7-dihydro-1H-purine-2,6-dione"),
    (X8BR_CAFFEINE,   "8-bromo-1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione"),
    # N9-substituted isomers: the saturated 5-ring N is N9, so the added-H form
    # is 3,9-dihydro-1H (NOT the family-typical 3,7-). Constitutionally fixed
    # (the substituent cannot move), so this is the definitive PIN.
    ("Cn1cnc2c(=O)[nH]c(=O)[nH]c21",       "9-methyl-3,9-dihydro-1H-purine-2,6-dione"),
    ("Cn1c(=O)c2ncn(C)c2n(C)c1=O",         "1,3,9-trimethyl-3,9-dihydro-1H-purine-2,6-dione"),
])
def test_named_purine_diones(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert name_purine_26_dione(mol) == expected


@pytest.mark.parametrize("label,smiles", [
    ("benzene (no purine)", "c1ccccc1"),
    ("pyridine (no purine)", "c1ccncc1"),
    ("adenine (6-amino, not a dione)", "Nc1ncnc2nc[nH]c12"),
    ("8-carboxy (suffix group out of scope)", X8_CARBOXY),
    ("8-hydroxy (functional group out of scope)", X8_HYDROXY),
    ("1,3,7,9-tetramethyluric acid (2,6,8-trione, not dione)", TETRAMETHYLURIC),
    ("1,3-dimethyluric-acid-like (2,6,8-trione)", DIMETHYLURIC_8OXO),
    # Unsaturated (alkenyl) substituents: the SHARED substituent namer mis-numbers
    # BRANCHED alkenyls (prenyl -> "2-methylbut-2-enyl", a different molecule), so
    # the engine conservatively declines ANY aliphatic C=C/C#C-bearing substituent
    # rather than emit a name it cannot guarantee. Fail closed.
    ("7-prenylcaffeine-like (branched alkenyl)", "CC(C)=CCn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("7-methallyl (branched alkenyl)", "C=C(C)Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("7-but-3-enyl (straight alkenyl, declined conservatively)", "C=CCCn1cnc2c1c(=O)n(C)c(=O)n2C"),
])
def test_fail_closed_returns_none(label, smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert name_purine_26_dione(mol) is None, label


def test_bare_parent_is_named_on_purine():
    # 'xanthine' does not occur in the Blue Book (0 hits); "the PIN is 7H-purine"
    # (the Blue Book), so the bare dione gets the parent name (it used to
    # decline and defer to the retained 'xanthine').
    assert name_purine_26_dione(Chem.MolFromSmiles(BARE_XANTHINE)) == \
        "3,7-dihydro-1H-purine-2,6-dione"


def test_none_input():
    assert name_purine_26_dione(None) is None
