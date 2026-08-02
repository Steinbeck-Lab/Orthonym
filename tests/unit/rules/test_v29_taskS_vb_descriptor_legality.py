"""v29 Task S — a von Baeyer descriptor is legal only if it REBUILDS the cage.

The audit these tests cover used to reconstruct from
``PolycyclicDescriptor.bridge_info_list`` (internal bookkeeping) instead of from
the descriptor string that actually gets spelled. Measured over 8,201 enumerated
cages (N<=12) and 1,623 corpus cages, that was wrong in both directions:

* it refused **556** enumerated / **14** corpus cages whose emitted name was
  correct (``BridgeInfo.atoms`` for a secondary bridge is stored in the opposite
  order to the bond path, because P-23.2.6.3 numbers it from the *higher*
  bridgehead), and
* it could pass a descriptor that does not describe the cage at all.

The replacement rebuilds the skeleton from the STRING using the Blue Book's own
numbering rules and requires set-equality with the molecule's cage bonds.

No OPSIN here: both downstream gates are documented FAIL-OPEN without Java
(``namer.py:588``, ``:608``), which is exactly why this Java-free floor exists.
Every expected value below was computed and cross-checked against OPSIN
(319/319 agreement on a 400-descriptor sample) before being asserted.
"""
import pytest
from rdkit import Chem

from orthonym.rules.polycyclic import VonBaeyerAnalyzer
from orthonym.rules.vonbaeyer_universal import (
    audit_von_baeyer_descriptor,
    parse_von_baeyer_descriptor,
    reconstruct_von_baeyer_skeleton,
)


def _cage(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    return mol, ring, VonBaeyerAnalyzer().analyze(mol, ring)


def _rebuild_smiles(descriptor):
    """The molecule the descriptor STRING denotes, as canonical SMILES."""
    total, edges = reconstruct_von_baeyer_skeleton(descriptor)
    rw = Chem.RWMol()
    for _ in range(total):
        rw.AddAtom(Chem.Atom(6))
    for a, b in edges:
        rw.AddBond(a - 1, b - 1, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


# --------------------------------------------------------------------------
# The reconstructor is the oracle -- it must rebuild the Blue Book's own PINs.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("descriptor,smiles", [
    # P-23.2.3 example (:9589)
    ("bicyclo[2.2.1]", "C1CC2CCC1C2"),                        # norbornane
    ("bicyclo[3.2.1]", "C1CC2CCC(C1)C2"),
    ("bicyclo[4.4.0]", "C1CCC2CCCCC2C1"),                     # decalin
    # P-23.2.5.1 / P-23.2.6 -- secondary bridges
    ("tricyclo[3.3.1.1^3,7]", "C1C2CC3CC1CC(C2)C3"),          # adamantane
    ("tricyclo[2.2.1.0^2,6]", "C1C2CC3C1C3C2"),               # nortricyclene
    ("tricyclo[4.4.0.0^3,8]", "C1CC2CC3CCC2CC13"),            # twistane
    ("pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]", "C12C3C4C1C1C2C3C41"),   # cubane
    # secondary bridge attached AT a main bridgehead (P-23.2.5.2 example :9635)
    ("tricyclo[9.3.3.1^1,11]", "C1CCCCC23CCCC(CCCC1)(CCC2)C3"),
])
def test_reconstructor_rebuilds_blue_book_pins(descriptor, smiles):
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert _rebuild_smiles(descriptor) == expected


def test_reconstructor_atom_and_bond_counts():
    """Sizes are fixed by P-23.2.6.1.4 (bracket sum + 2 = skeletal atoms).

    Cited as P-23.2.6.1.1 until v29 Task S2; that rule (``:9645``) fixes the
    ring-count WORD, not the atom count. The atom count is ``:9651``.
    """
    assert reconstruct_von_baeyer_skeleton("bicyclo[2.2.1]")[0] == 7
    assert len(reconstruct_von_baeyer_skeleton("bicyclo[2.2.1]")[1]) == 8
    # adamantane: 10 atoms, 12 bonds -> circuit rank 3 = 'tricyclo'
    total, edges = reconstruct_von_baeyer_skeleton("tricyclo[3.3.1.1^3,7]")
    assert (total, len(edges)) == (10, 12)
    # cubane: 8 atoms, 12 bonds -> circuit rank 5 = 'pentacyclo'
    total, edges = reconstruct_von_baeyer_skeleton(
        "pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]")
    assert (total, len(edges)) == (8, 12)


# --------------------------------------------------------------------------
# Parsing: both typographies, and fail-closed on anything malformed.
# --------------------------------------------------------------------------
def test_parse_accepts_both_secondary_typographies():
    """``_build_descriptor`` has emitted both ``2^3,7`` and ``2(3,7)``."""
    caret = parse_von_baeyer_descriptor("tricyclo[3.3.1.1^3,7]")
    paren = parse_von_baeyer_descriptor("tricyclo[3.3.1.1(3,7)]")
    assert caret == paren == ([3, 3, 1], [(1, 3, 7)])


@pytest.mark.parametrize("descriptor", [
    "bicyclo[2.2]",              # fewer than three primary numbers
    "bicyclo[1.2.3]",            # not in descending order (P-23.2.2)
    "tricyclo[3.3.1.x^3,7]",     # non-numeric bridge length
    "tricyclo[3.3.1.1^3]",       # secondary bridge missing an attachment
    "adamantane",                # not a descriptor at all
    "",
])
def test_malformed_descriptors_fail_closed(descriptor):
    assert reconstruct_von_baeyer_skeleton(descriptor) is None
    mol, ring, _ = _cage("C1CC2CCC1C2")
    assert audit_von_baeyer_descriptor(mol, ring, {}, descriptor) is False


# --------------------------------------------------------------------------
# The audit ACCEPTS every correct analysis.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected_descriptor", [
    ("C1CC2CCC1C2", "bicyclo[2.2.1]"),
    ("C1C2CC3CC1CC(C2)C3", "tricyclo[3.3.1.1^3,7]"),
    ("C12C3C4C1C5C4C3C25", "pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]"),
])
def test_audit_accepts_correct_analysis(smiles, expected_descriptor):
    mol, ring, desc = _cage(smiles)
    assert desc.descriptor_string == expected_descriptor
    assert audit_von_baeyer_descriptor(
        mol, ring, desc.numbering, desc.descriptor_string) is True


def test_audit_accepts_secondary_bridge_numbered_from_higher_bridgehead():
    """REGRESSION for the 556/14 false rejections.

    P-23.2.6.3 (``BlueBookV2.md:9711``) "Numbering of secondary bridges":
    *"Each atom of a secondary bridge is numbered starting with the atom next to
    the higher numbered bridgehead."*  The analyzer numbers it that way, but its
    ``BridgeInfo.atoms`` records the reverse order, so the previous audit walked
    a non-bonded pair and refused this perfectly correct cage.
    """
    mol, ring, desc = _cage("C1CC23CCC(C1)(CC2)C3")
    assert desc.descriptor_string == "tricyclo[3.2.1.2^1,5]"
    assert audit_von_baeyer_descriptor(
        mol, ring, desc.numbering, desc.descriptor_string) is True


# --------------------------------------------------------------------------
# The audit REJECTS a descriptor that does not account for the whole cage.
# --------------------------------------------------------------------------
def test_audit_rejects_bridge_dropping_descriptor():
    """The live defect this floor exists for.

    ``_analyze_impl`` detects the P-23.2.6.1.4 violation (``:9651`` -- the
    bracket sum + 2 must equal the alkane stem) and used to only ``logger.error``
    it, while ``analyze``'s fallback branch ran no validity check at all. The
    result is a name whose brackets account for 11 atoms while its own stem says
    12 -- and with both OPSIN gates off (their documented no-Java state) it
    shipped. v29 Task S2 wired this audit into ``analyze`` and the three
    name-producers, so that molecule now abstains; the audit's own verdict,
    asserted here, is unchanged.
    """
    mol, ring, desc = _cage("C1C2CC1C1CCC3(C2)CC1C3")
    assert len(ring) == 12
    assert desc.descriptor_string == "tetracyclo[5.1.1.2^3,6]"
    # the string denotes an 11-atom cage: 5+1+1+2 bridge atoms + 2 bridgeheads
    assert reconstruct_von_baeyer_skeleton(desc.descriptor_string)[0] == 11
    assert audit_von_baeyer_descriptor(
        mol, ring, desc.numbering, desc.descriptor_string) is False


def test_audit_rejects_wrong_size_descriptor():
    """A well-formed descriptor for a DIFFERENT cage size is refused."""
    mol, ring, desc = _cage("C1CC2CCC1C2")          # 7 atoms
    assert audit_von_baeyer_descriptor(
        mol, ring, desc.numbering, "bicyclo[2.2.2]") is False   # 8 atoms


@pytest.mark.parametrize("wrong_word", [
    "bicyclo[3.3.1.1^3,7]",       # 3 rings spelled as 2
    "tetracyclo[3.3.1.1^3,7]",    # 3 rings spelled as 4
    "pentacyclo[3.3.1.1^3,7]",
])
def test_audit_rejects_wrong_ring_count_word(wrong_word):
    """The ring-count word is part of the name and must match the skeleton.

    P-23.1.9, under P-23.1 "DEFINITIONS AND TERMINOLOGY"
    (``BlueBookV2.md:9558``): *"A 'polycyclic system' contains a number of rings
    equal to the minimum number of scissions required to convert the system into
    an acyclic skeleton. The number of rings is indicated by the nondetachable
    prefix 'bicyclo' (not dicyclo), 'tricyclo', 'tetracyclo', etc."*

    Found by mutation testing: with the bracket body left correct, every other
    clause of the audit passed, so ``tetracyclo[3.3.1.1^3,7]decane`` for
    adamantane was accepted before this check existed.
    """
    mol, ring, desc = _cage("C1C2CC3CC1CC(C2)C3")
    assert desc.descriptor_string == "tricyclo[3.3.1.1^3,7]"
    assert audit_von_baeyer_descriptor(
        mol, ring, desc.numbering, wrong_word) is False


def test_audit_rejects_non_bijective_numbering():
    """Two cage atoms sharing a locant cannot be read against the descriptor."""
    mol, ring, desc = _cage("C1CC2CCC1C2")
    bad = dict(desc.numbering)
    keys = sorted(bad)
    bad[keys[0]] = bad[keys[1]]
    assert audit_von_baeyer_descriptor(
        mol, ring, bad, desc.descriptor_string) is False


def test_audit_rejects_incomplete_numbering():
    mol, ring, desc = _cage("C1CC2CCC1C2")
    partial = dict(desc.numbering)
    partial.pop(sorted(partial)[0])
    assert audit_von_baeyer_descriptor(
        mol, ring, partial, desc.descriptor_string) is False
