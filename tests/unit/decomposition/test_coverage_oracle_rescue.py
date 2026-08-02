"""Task Z3 -- a character count must not be the verdict on whether a name
covers a molecule.

`decomposition/engine.py` decided "is this name good enough to skip
decomposition?" with several tests on the LENGTH OF THE NAME STRING.  A
character count is anti-correlated with coverage: the correct `cholesterol`
scores 0.393 chars/HA while `2-amino-2-(methylamino)acetamide` -- a name that
INVENTS atoms -- scores 5.333.  Measured before the fix, the predicates refused
29 names that were each exactly right for the molecule they were asked about,
including PINs marked `(PIN)` in Blue Book Table 2.7.

Those counts are now PRE-FILTERS: a name that fails one is measured against the
real oracle (`validation/atom_coverage.py`, constitution by InChIKey skeleton)
and kept only if the measurement proves it denotes the molecule exactly.

The two halves of this file are equally load-bearing.  Rescuing correct names
is worthless if partial names are rescued too, so every positive here is paired
with a NEGATIVE CONTROL lifted from the repo's own pre-existing fixtures.
"""

import pytest
from rdkit import Chem

from orthonym.decomposition.engine import (
    _coverage_is_adequate,
    _name_covers_molecule,
    _name_is_proven_complete,
    _name_quality_is_acceptable,
)
from orthonym.validation.atom_coverage import find_opsin_jar

pytestmark = pytest.mark.skipif(
    find_opsin_jar() is None,
    reason="the coverage oracle needs the OPSIN jar; without it it fails "
           "CLOSED and no rescue happens (which is the designed behaviour, "
           "but makes these assertions vacuous)",
)


# --------------------------------------------------------------------------
# Correct names that a character count refused.  SMILES are OPSIN's own parse
# of the name, so the (name, molecule) pairing is correct by construction.
# --------------------------------------------------------------------------

# (name, SMILES, heavy atoms, why the name is right)
CORRECT_BUT_SHORT = [
    # Blue Book P-25.1.1 "Retained names for hydrocarbons used for parent ring
    # components and as attached ring components", Table 2.7 "Retained names
    # for hydrocarbon parent ring components in descending order of
    # seniority" -- each of these is marked (PIN) there.
    # SMILES are OPSIN's own parse of the name, canonicalised by RDKit, so the
    # pairing cannot be a hand-transcription error.  (It was: a hand-written
    # `chrysene` SMILES was a different C18H12 isomer, and
    # test_correct_short_name_is_proven_complete caught it.)
    ("pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34", 16),
    ("chrysene", "c1ccc2c(c1)ccc1c3ccccc3ccc21", 18),
    ("perylene", "c1cc2cccc3c4cccc5cccc(c(c1)c23)c54", 20),
    ("coronene", "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61", 24),
    ("picene", "c1ccc2c(c1)ccc1c2ccc2c3ccccc3ccc21", 22),
    # Systematic PINs.  An unbranched parent hydride or acid carries NO
    # locants, which is what the no-digits/no-hyphens guard punished.
    ("henicosane", "CCCCCCCCCCCCCCCCCCCCC", 21),
    ("docosane", "CCCCCCCCCCCCCCCCCCCCCC", 22),
    ("icosanoic acid", "CCCCCCCCCCCCCCCCCCCC(=O)O", 22),
    ("docosanedioic acid", "OC(=O)CCCCCCCCCCCCCCCCCCCCC(=O)O", 26),
]


def _mol(smiles, expect_ha):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"fixture SMILES did not parse: {smiles}"
    assert mol.GetNumHeavyAtoms() == expect_ha, (
        f"fixture drift: expected {expect_ha} heavy atoms, "
        f"got {mol.GetNumHeavyAtoms()} for {smiles}"
    )
    return mol


@pytest.mark.unit
@pytest.mark.parametrize("name,smiles,ha", CORRECT_BUT_SHORT)
def test_correct_short_name_is_proven_complete(name, smiles, ha):
    """The oracle recognises each fixture as denoting its own molecule.

    This is the precondition for every rescue below; if it fails, the rest of
    the file is testing nothing.
    """
    assert _name_is_proven_complete(name, _mol(smiles, ha)) is True


@pytest.mark.unit
@pytest.mark.parametrize("name,smiles,ha", CORRECT_BUT_SHORT)
def test_correct_short_name_survives_the_quality_gate(name, smiles, ha):
    """A correct name is not sent to decomposition for being short."""
    assert _name_quality_is_acceptable(name, _mol(smiles, ha)) is True


@pytest.mark.unit
@pytest.mark.parametrize("name,smiles,ha", CORRECT_BUT_SHORT)
def test_correct_short_name_counts_as_adequate_coverage(name, smiles, ha):
    """`_coverage_is_adequate` compares `len(name) * bonus` with
    `int(heavy_atoms * threshold)` -- a character count written WITHOUT a
    division, which is why greps for `len(name) /` and for `heavy_atoms //`
    both miss it.  It gets the same treatment as its siblings.
    """
    assert _coverage_is_adequate(name, _mol(smiles, ha)) is True


# --------------------------------------------------------------------------
# NEGATIVE CONTROLS.  Every one of these is a partial name the guards were
# built to catch, taken from the pre-existing fixtures in
# tests/unit/decomposition/test_quality_gate.py.  Rescuing any of them would
# mean the oracle had become a rubber stamp.
# --------------------------------------------------------------------------

PARTIAL_NAMES = [
    # (name, SMILES, which predicate the repo already asserts rejects it)
    ("5-chloroquinoline", "CCCCCCCCCCCCC(=O)Oc1ccc(Cl)c2ncccc12", 26),
    ("(22E)-stigmasta-7,22-diene",
     "CC(C)C(C)CC=CC(C)C1CCC2C3=CCC4CC(OC5OC(CO)C(O)C(O)C5O)CCC4(C)C3CCC12C", 41),
    ("hexadecan-1-ol", "CCCCCCCCCCCC(=O)OCCCCOP(=O)(O)OCCCC", 27),
    ("hexadecan-3-ol",
     "CCCCCCCCCC(=O)NCCCC(=O)OCC(NC(=O)CCCCC)CC(=O)OCCCCCCCCCC", 42),
    # The guard's own stated target: a retained ring name for a much larger
    # ester.  Named in the no-digits/no-hyphens comment as the thing it exists
    # to catch, so it must stay caught.
    ("benzene", "CCCCCCCCCCCCC(=O)Oc1ccc(Cl)c2ncccc12", 26),
]


@pytest.mark.unit
@pytest.mark.parametrize("name,smiles,ha", PARTIAL_NAMES)
def test_partial_name_is_not_proven_complete(name, smiles, ha):
    """A name that does not cover the molecule is not rescuable."""
    assert _name_is_proven_complete(name, _mol(smiles, ha)) is False


@pytest.mark.unit
@pytest.mark.parametrize("name,smiles,ha", PARTIAL_NAMES)
def test_partial_name_still_rejected(name, smiles, ha):
    """The rescue is one-directional: it never turns a rejection into an
    acceptance for a name that fails the measurement."""
    assert _name_quality_is_acceptable(name, _mol(smiles, ha)) is False


@pytest.mark.unit
def test_name_covers_molecule_still_rejects_partial_ring_name():
    """The `len(name) / 1.5 / heavy_atoms < 0.45` site keeps its true positive."""
    mol = _mol("CCCCCCCCCCCCC(=O)Oc1ccc(Cl)c2ncccc12", 26)
    assert _name_covers_molecule("5-chloroquinoline", mol) is False


# --------------------------------------------------------------------------
# The oracle itself
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_oracle_refuses_a_name_that_invents_atoms():
    """Atom GAIN, not just atom drop.

    `CNCC(=O)N` named `2-amino-2-(methylamino)acetamide` is the case that made
    the old count-based validator unsound -- it scored 5.333 chars/HA, the
    HIGHEST score in the whole audit, while describing a different molecule.
    """
    mol = Chem.MolFromSmiles("CNCC(=O)N")
    assert _name_is_proven_complete("2-amino-2-(methylamino)acetamide", mol) is False


@pytest.mark.unit
def test_oracle_distinguishes_same_formula_different_constitution():
    """sarcosine vs alanine share molecular formula AND heavy-atom element
    multiset and are different molecules, so nothing weaker than constitution
    can separate them."""
    sarcosine = Chem.MolFromSmiles("CNCC(=O)O")
    assert _name_is_proven_complete("2-aminopropanoic acid", sarcosine) is False
    assert _name_is_proven_complete("2-(methylamino)acetic acid", sarcosine) is True


@pytest.mark.unit
def test_oracle_is_disabled_by_the_rollback_lever(monkeypatch):
    """Turning the lever off restores pre-Z3 behaviour exactly, because the
    only thing the oracle can do is turn a REJECT into an ACCEPT."""
    from orthonym.decomposition import engine

    mol = _mol("CCCCCCCCCCCCCCCCCCCCC", 21)
    engine._PROVEN_COMPLETE_CACHE.clear()
    monkeypatch.setenv("ORTHONYM_DECOMP_COVERAGE_ORACLE", "0")
    assert _name_is_proven_complete("henicosane", mol) is False
    assert _name_quality_is_acceptable("henicosane", mol) is False

    engine._PROVEN_COMPLETE_CACHE.clear()
    monkeypatch.setenv("ORTHONYM_DECOMP_COVERAGE_ORACLE", "1")
    assert _name_quality_is_acceptable("henicosane", mol) is True


@pytest.mark.unit
def test_oracle_result_is_cached_per_name_and_molecule():
    """The oracle costs an OPSIN parse, so it must be asked once per pair.

    Asserted by observing the cache, not by timing: a timing assertion would
    be flaky and would not prove the cache is the reason.
    """
    from orthonym.decomposition import engine

    mol = _mol("CCCCCCCCCCCCCCCCCCCCCC", 22)
    engine._PROVEN_COMPLETE_CACHE.clear()
    assert _name_is_proven_complete("docosane", mol) is True
    key = ("docosane", Chem.MolToSmiles(mol))
    assert engine._PROVEN_COMPLETE_CACHE[key] is True
    size = len(engine._PROVEN_COMPLETE_CACHE)
    assert _name_is_proven_complete("docosane", mol) is True
    assert len(engine._PROVEN_COMPLETE_CACHE) == size, (
        "a repeat call added a cache entry, so it re-ran the oracle"
    )


@pytest.mark.unit
def test_oracle_fails_closed_on_junk_input():
    """No name, no molecule, and an unparseable name all yield False."""
    mol = _mol("CCCCCCCCCCCCCCCCCCCCC", 21)
    assert _name_is_proven_complete("", mol) is False
    assert _name_is_proven_complete("henicosane", None) is False
    assert _name_is_proven_complete("zzqqxx-not-a-name", mol) is False
