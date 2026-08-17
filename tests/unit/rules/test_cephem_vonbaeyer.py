"""v33 Phase 3 residual / Phase-8 enabler: the unsaturated CEPHEM (ceph-3-em)
von Baeyer core.

Root cause (SPY, measured): the SATURATED penam core already names correctly
(``3,3-dimethyl-7-oxo-4-thia-1-azabicyclo[3.2.0]heptane-2-carboxylic acid``),
but the ceph-3-em acid core (``OC(=O)C1=CCSC2CC(=O)N12``) is NOT a producer
abstention -- ``_assemble_complete_bicyclo_name`` DOES build a candidate,
``7-oxo-5-thia-1-azabicyclo[4.2.0]oct-2-ene-2-carboxylic acid``. SELF-01
correctly rejects it because OPSIN parses that string to a DIFFERENT molecule
(the oxo group lands one ring atom off). The bug is in
``rules/bicyclo.py::_legacy_bicyclo_numbering``: its secondary (second-longest)
bridge is numbered in the SAME direction as the longest bridge (near the
START bridgehead first), but von Baeyer numbering requires the secondary
bridge to be numbered continuing FROM the second (far) bridgehead BACK toward
the first (see the docstring's own norbornane example: "Second longest
(2 atoms): 4 -> 5 -> 6 -> 1", i.e. from bridgehead 4 back to bridgehead 1).
``_enumerate_bicyclo_numberings`` already implements this correctly (it
explicitly reverses that segment); ``_legacy_bicyclo_numbering`` does not.

For most existing molecules this invisible bug never surfaces: either the
enumerate-based candidate already wins the numbering tie-break on an earlier
P-14.4 tier (heteroatoms / principal-group suffix / ene), or the secondary
bridge carries no distinguishing substituent so which specific atom gets
locant N vs N+1 does not change the emitted name. The ceph-3-em core is the
first case on record where (a) legacy ties the correct candidate on
heteroatoms/suffix/ene *and* (b) the secondary bridge carries a
distinguishing oxo substituent -- so the wrong-direction legacy numbering
wins the tie-break on the substituent-locant tier and ships a candidate that
denotes a different molecule.
"""

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse


def _namer() -> Orthonym:
    return Orthonym(style="pin")


def _abstains(name) -> bool:
    return name is None or is_failure_name(name)


def _inchikey(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return Chem.MolToInchiKey(mol)


def _rt_inchikey(name: str):
    smi = opsin_parse(name)
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(mol) if mol is not None else None


CEPHEM_ACID = "OC(=O)C1=CCSC2CC(=O)N12"
CEPHEM_EXPECTED = "8-oxo-5-thia-1-azabicyclo[4.2.0]oct-2-ene-2-carboxylic acid"


@pytest.mark.opsin_gate
def test_cephem_acid_core_names_correctly(opsin_gate):
    """The ceph-3-em-4-carboxylic acid core round-trips to the correct
    numbering: 8-oxo (NOT 7-oxo), oct-2-ene (NOT oct-3-ene/oct-4-ene)."""
    emitted = _namer().name(CEPHEM_ACID)
    assert not _abstains(emitted), f"{CEPHEM_ACID} abstained: {emitted!r}"
    assert _rt_inchikey(emitted) == _inchikey(CEPHEM_ACID), (
        f"{CEPHEM_ACID} -> {emitted!r} does not round-trip to the input molecule")
    assert emitted == CEPHEM_EXPECTED, (
        f"spelling changed: {emitted!r} != {CEPHEM_EXPECTED!r}")


def test_cephem_acid_core_is_not_seven_oxo():
    """Regression pin on the exact SELF-01-caught defect: the wrong-direction
    legacy numbering placed the oxo group at locant 7, one ring atom off from
    the carbonyl carbon it actually names (a DIFFERENT molecule)."""
    emitted = _namer().name(CEPHEM_ACID)
    assert emitted is not None
    assert "7-oxo" not in emitted, f"reverted to the wrong-direction defect: {emitted!r}"


# --------------------------------------------------------------------------
# Regression: existing von Baeyer / bicyclo names must stay byte-identical.
# --------------------------------------------------------------------------

VONBAEYER_REGRESSION = [
    ("C1CC2CCC1C2", "norbornane"),
    ("O=C1CC2CCC1C2", "bicyclo[2.2.1]heptan-2-one"),
    ("C1CC2CCCC1CC2", "bicyclo[3.2.2]nonane"),
    ("C1=CC2CC1CC2", "bicyclo[2.2.1]hept-2-ene"),
    ("C1CC2CCCCC2C1", "octahydro-1H-indene"),
    # the saturated penam core (WORKS today) -- the acid-ester-anion producer's
    # neighbour, and the molecule that proves this fix does not regress the
    # already-correct saturated beta-lactam case.
    ("OC(=O)C1N2C(=O)CC2SC1(C)C",
     "3,3-dimethyl-7-oxo-4-thia-1-azabicyclo[3.2.0]heptane-2-carboxylic acid"),
]


@pytest.mark.parametrize("smiles,expected", VONBAEYER_REGRESSION)
def test_existing_vonbaeyer_names_are_byte_identical(smiles, expected):
    emitted = _namer().name(smiles)
    assert emitted == expected, (
        f"{smiles}: von Baeyer regression -- {emitted!r} != {expected!r}")


# --------------------------------------------------------------------------
# A decorated cephem: the fix generalises to a real substituent (3-methyl).
# --------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_3_methyl_cephem_names_correctly(opsin_gate):
    """A 3-methyl-substituted ceph-3-em acid (the fix must not be a
    single-molecule special case): the ring double bond and the extra
    substituent must not perturb the corrected oct-2-ene / 8-oxo numbering."""
    smiles = "OC(=O)C1=C(C)CSC2CC(=O)N12"
    expected = "3-methyl-8-oxo-5-thia-1-azabicyclo[4.2.0]oct-2-ene-2-carboxylic acid"
    emitted = _namer().name(smiles)
    assert not _abstains(emitted), f"{smiles} abstained: {emitted!r}"
    assert _rt_inchikey(emitted) == _inchikey(smiles), (
        f"{smiles} -> {emitted!r} does not round-trip to the input molecule")
    assert emitted == expected, f"spelling changed: {emitted!r} != {expected!r}"


@pytest.mark.opsin_gate
def test_full_cephalosporin_with_acylamino_still_abstains_not_forced(opsin_gate):
    """A full cephalosporin core (7-acetamido + 3-methyl analogue) currently
    still abstains -- an UNRELATED, pre-existing substituent-naming defect
    (the -NHC(=O)CH3 acylamino group on the secondary-bridge ring carbon is
    mis-decomposed as a plain 'ethyl' substituent, an atom-drop that SELF-01/
    OPSIN suppresses with the gate on -- the raw, gate-off candidate is
    '7-ethyl-8-oxo-...', a WRONG constitution, not this fix's numbering bug).
    Documented per instructions rather than forced: this ticket's scope is
    the ring numbering/unsaturation interaction, not the acylamino
    substituent namer."""
    smiles = "OC(=O)C1=CCSC2C(NC(=O)C)C(=O)N12"
    emitted = _namer().name(smiles)
    assert _abstains(emitted), (
        f"{smiles} now emits {emitted!r} -- if this now round-trips, replace "
        f"this abstention pin with a positive assertion")
