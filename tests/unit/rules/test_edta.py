"""EDTA-type polyaminopolycarboxylic acids — multiplicative nomenclature.

A pair of trivalent-nitrogen hubs joined by an unbranched alkanediyl chain,
each hub bearing two identical acyclic functional-parent arms, is named by
multiplicative nomenclature over the composite central group
``{alkane}-1,m-diyldinitrilo``.

Blue Book authority (name strings verbatim-verified against the Blue Book; note
the BB attaches NO explicit ``(PIN)`` tag to the EDTA name in the source — its
PIN status is DERIVED from the two rules below, see the note at the example):

  * (the Blue Book) "MULTIPLICATION OF IDENTICAL SENIOR PARENT
    STRUCTURES": "Multiplicative nomenclature is senior to substitutive
    nomenclature for generating preferred IUPAC names to express multiple
    occurrences of identical senior parent structures, other than alkanes..."
  * (the Blue Book): "When two or more parent structures...
    satisfy the requirements for multiplicative nomenclature (see, the
    structure chosen as the parent structure to be multiplied is the more
    numerous." Its worked example (the Blue Book) is EDTA itself:
    ``2,2',2'',2'''-(ethane-1,2-diyldinitrilo)tetraacetic acid`` — the
    multiplicative form (FOUR acetic acid units), preferred over the
    glycine-based name ``N,N'-(ethane-1,2-diyl)bis[N-(carboxymethyl)glycine]``
    (two units) by 's more-numerous-parent rule and 's
    multiplicative>substitutive seniority. Cross-cited at the Blue Book
    with "(see ". NB: the BB attaches no explicit ``(PIN)`` tag to
    either name in any of the four occurrences (:7669/:21586/:29809/:29970);
    the tetraacetic form is the PIN by this DERIVATION, not by a verbatim label.
  * The 3-arm single-hub analogue ``2,2',2''-nitrilotriacetic acid``
    (the Blue Book,:2648) already ships via the sibling nitrilo handler;
    it and iminodiacetic acid are asserted here as class boundaries.

Every emission is round-tripped through OPSIN to the input full InChIKey; a
name that does not describe the input structure must never ship (0-wrong).
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.validation.dual_validator import _find_opsin_jar, _parse_name_with_opsin

_OPSIN = _find_opsin_jar()


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparseable input SMILES: {smiles}"
    return Chem.MolToInchiKey(mol)


# EDTA and a second, longer-bridge member (PDTA) prove the CLASS — two distinct
# alkanediyl bridge lengths — not a single molecule.
_EDTA_CLASS = [
    # EDTA (ethane-1,2-diyl bridge) — the BB worked example.
    ("OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O",
     "2,2',2'',2'''-(ethane-1,2-diyldinitrilo)tetraacetic acid"),
    # PDTA (propane-1,3-diyl bridge) — same class, different bridge length.
    ("OC(=O)CN(CC(=O)O)CCCN(CC(=O)O)CC(=O)O",
     "2,2',2'',2'''-(propane-1,3-diyldinitrilo)tetraacetic acid"),
]

# Single-hub siblings that must keep working (regression anchors for the
# adjacent nitrilo / azanediyl handlers this fix sits beside).
_SIBLINGS = [
    ("OC(=O)CN(CC(=O)O)CC(=O)O", "2,2',2''-nitrilotriacetic acid"),
    ("OC(=O)CNCC(=O)O", "2,2'-azanediyldiacetic acid"),
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", _EDTA_CLASS)
def test_edta_class_exact_name(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", _SIBLINGS)
def test_sibling_members_unchanged(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.roundtrip
@pytest.mark.skipif(_OPSIN is None, reason="OPSIN jar not found")
@pytest.mark.parametrize("smiles,expected", _EDTA_CLASS + _SIBLINGS)
def test_edta_class_opsin_roundtrip(smiles, expected):
    """The emitted name must round-trip through OPSIN to the input InChIKey."""
    name = name_compound(smiles)
    assert name == expected
    opsin_smiles = _parse_name_with_opsin(name, opsin_jar=_OPSIN)
    assert opsin_smiles is not None, f"OPSIN could not parse: {name}"
    assert _inchikey(opsin_smiles) == _inchikey(smiles), (
        f"round-trip mismatch for {name}: "
        f"{_inchikey(opsin_smiles)} != {_inchikey(smiles)}"
    )
