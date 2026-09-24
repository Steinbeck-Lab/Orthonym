""" a phase — LOW-risk retained-parent PIN spelling / catalog / gold defects.

Four groups, all named by an exact canonical-SMILES entry in ``ALL_RETAINED_NAMES``
consumed on the PIN path at ``routing/dispatch_table.py:1488``
(``_handle_retained_name``, RETAINED_NAME). The hand-curated entries in
``data/retained_names.py`` win the merge over the OPSIN aryl-group import
(``_HAND_CURATED`` overrides ``_OPSIN_NAMES``), which supplies the As/P stems
WITHOUT the terminal ``e``.

Blue Book citations (``the Blue Book Blue Book``):

Group 1 — gold-defect (engine already emits the PIN; the bb gold dropped the
indicated hydrogen, corrected in the report + baseline, ZERO engine change):
  * 1H-indole the Blue Book "(1H-isomer shown; the PIN is 1H-indole)"
  * 2H-isoindole the Blue Book "(2H-isomer shown; the PIN is 2H-isoindole)"
  * 4H-quinolizine the Blue Book "(4H-isomer shown; the PIN is 4H-quinolizine)"

Group 2 — terminal-``e`` retained-name spelling:
  * arsindole the Blue Book "arsindole (PIN)"
  * isoarsindole the Blue Book "isoarsindole (PIN)"
  * phosphindole the Blue Book "phosphindole (PIN)"
  * isophosphindole the Blue Book "isophosphindole (PIN)"

Group 3 — catalog adds for von-Baeyer-emitting retained parents:
  * octalene the Blue Book "pentalene (PIN) octalene (PIN)" /
  * arsindolizine the Blue Book "arsindolizine (PIN)"
  * phosphindolizine the Blue Book "phosphindolizine (PIN)"
  * 4H-phosphinolizine the Blue Book Table 2.9 shorthand "phosphinolizine (PIN)"; the
                       6+6 QUINOLIZINE analogue carries indicated H like its N
                       parent (the Blue Book "the PIN is 4H-quinolizine";
                       the Blue Book + the Blue Book). a review.

Group 4 — retained nitrile names:
  * formonitrile the Blue Book "HCN formonitrile(PIN)... hydrogen cyanide"
  * oxalonitrile the Blue Book "NC-CN oxalonitrile (PIN) ethanedinitrile"
"""
import shutil
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.data import ALL_RETAINED_NAMES
from tests.support.jars import jar_or_none

_OPSIN_JAR = jar_or_none()
_OPSIN_OK = shutil.which("java") is not None and _OPSIN_JAR is not None

# Group 1 — gold-defect rows: engine already emits the indicated-H PIN.
_GROUP1 = [
    ("c1ccc2[nH]ccc2c1", "1H-indole"),       # the Blue Book
    ("c1ccc2c[nH]cc2c1", "2H-isoindole"),     # the Blue Book
    ("C1=CCN2C=CC=CC2=C1", "4H-quinolizine"),  # the Blue Book
]

# Groups 2, 3, 4 — engine-changed entries (hand-curated retained-name catalog).
_ENGINE_CHANGED = [
    ("C1=Cc2ccccc2[AsH]1", "arsindole"),        # G2, the Blue Book
    ("C1=c2ccccc2=C[AsH]1", "isoarsindole"),     # G2, the Blue Book
    ("c1ccc2[pH]ccc2c1", "phosphindole"),        # G2, the Blue Book
    ("c1ccc2c[pH]cc2c1", "isophosphindole"),     # G2, the Blue Book
    ("c1cccc2ccccccc-2cc1", "octalene"),         # G3, the Blue Book
    ("C1=CC2=CC=C[As]2C=C1", "arsindolizine"),   # G3, the Blue Book
    ("c1ccp2cccc2c1", "phosphindolizine"),       # G3, the Blue Book
    ("C1=CCP2C=CC=CC2=C1", "4H-phosphinolizine"),  # G3, the Blue Book (6+6 quinolizine analogue; a review)
    ("C#N", "formonitrile"),                     # G4, the Blue Book
    ("N#CC#N", "oxalonitrile"),                  # G4, the Blue Book
]

_ALL = _GROUP1 + _ENGINE_CHANGED


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", _ALL)
def test_key_is_canonical(smiles, expected):
    """Each SMILES used as a catalog key is already RDKit-canonical (else it misses)."""
    assert Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) == smiles


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", _ENGINE_CHANGED)
def test_retained_catalog_entry_present(smiles, expected):
    """The hand-curated retained-name catalog holds the exact PIN (wins the merge)."""
    assert ALL_RETAINED_NAMES.get(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", _ALL)
def test_name_compound_emits_pin(smiles, expected):
    """End-to-end: name_compound emits the Blue-Book PIN (not the OPSIN stem
    without ``e``, not a von Baeyer name, not the retained functional-class name)."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
def test_neighbor_nitriles_unchanged():
    """ substituted/dinitrile neighbors keep their systematic/retained names;
    the bare-parent ``C#N`` -> formonitrile fix must not touch composed forms."""
    assert name_compound("CC#N") == "acetonitrile"          # substituted
    assert name_compound("c1ccccc1C#N") == "benzonitrile"    # retained aromatic
    assert name_compound("N#CC(=O)C#N") == "carbonyl dicyanide"


@pytest.mark.unit
def test_neighbor_small_molecules_unchanged():
    """Retained-name siblings around the C#N/N#CC#N edits are untouched."""
    assert name_compound("O=C=O") == "carbon dioxide"
    assert name_compound("O=S=O") == "sulfur dioxide"
    assert name_compound("N#N") == "dinitrogen"


@pytest.mark.roundtrip
@pytest.mark.skipif(not _OPSIN_OK, reason="OPSIN JAR / Java runtime unavailable")
@pytest.mark.parametrize("smiles,expected", _ALL)
def test_pin_roundtrips_to_input(smiles, expected):
    """0-wrong: the emitted PIN parses back (OPSIN) to the input structure."""
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

    rt = opsin_roundtrip_check(smiles, expected)
    assert rt["passed"], f"{expected!r} did not round-trip: {rt}"
