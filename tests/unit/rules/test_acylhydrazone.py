"""Acylhydrazones R'2C=N-NH-C(=O)-R -> N'-({ylidene}){acyl}hydrazide (PIN).

Governing rules (the Blue Book):
- (the Blue Book) "Substituted hydrazides": substituents on the nitrogen
  atoms of a hydrazide take the locants 'N' for -NH- and 'N'' for -NH2. The
  =CR'2 group replaces the two H of the terminal (former -NH2) nitrogen, so it
  is an N'-ylidene substituent.
- (the Blue Book-38962) gives the verbatim (PIN) for exactly this class:
      CH3-CO-NH-N=CH-N=N-C6H5
      N'-[(phenyldiazenyl)methylidene]acetohydrazide (PIN)
  The acyl group makes the hydrazide the senior parent carboxamide
  family), so an acylhydrazone is NOT named as an 'ylidene'-hydrazine
  (that is the bare-hydrazone class, nor as a functional-class
  hydrazone.

The class is OPEN over the ylidene R' (any aldehyde/ketone-derived fragment the
substituent pipeline names as an '-ylidene') and over an UNsubstituted acyl-R
hydrazide parent (acetohydrazide, benzohydrazide,...).

Every assertion pins the exact derived PIN AND round-trips the emitted name
through OPSIN back to the input's full InChIKey (0-wrong: the name must denote
the input molecule, not merely parse).
"""
import pytest
from rdkit import Chem
from orthonym.namer import name_compound
from orthonym.validation.opsin_roundtrip import opsin_parse


def _inchikey(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(m) if m else None


def _assert_pin_and_roundtrip(smiles, expected_pin):
    """Name the input, assert the exact PIN, and OPSIN-round-trip the emitted
    name to the input's full InChIKey."""
    name = name_compound(smiles, style="pin")
    assert name == expected_pin, f"{smiles}: got {name!r}, want {expected_pin!r}"
    opsin_smiles = opsin_parse(name)
    assert opsin_smiles, f"OPSIN could not parse emitted name {name!r}"
    assert _inchikey(opsin_smiles) == _inchikey(smiles), (
        f"round-trip mismatch for {name!r}: "
        f"{_inchikey(opsin_smiles)} != {_inchikey(smiles)}")


@pytest.mark.unit
class TestAcylhydrazone:
    def test_acetaldehyde_acetylhydrazone(self):
        # CH3-CO-NH-N=CH-CH3 (C4H8N2O, formula-verified by the lead)
        _assert_pin_and_roundtrip("CC=NNC(=O)C", "N'-ethylideneacetohydrazide")

    def test_ethylidene_benzohydrazide(self):
        # aryl acyl, alkyl ylidene: CH3-CH=N-NH-CO-C6H5
        _assert_pin_and_roundtrip("CC=NNC(=O)c1ccccc1",
                                  "N'-ethylidenebenzohydrazide")

    def test_benzylidene_acetohydrazide(self):
        # aryl ylidene, alkyl acyl: C6H5-CH=N-NH-CO-CH3
        _assert_pin_and_roundtrip("O=C(C)NN=Cc1ccccc1",
                                  "N'-benzylideneacetohydrazide")

    def test_benzylidene_benzohydrazide(self):
        # both aryl -> proves the class, not one molecule
        _assert_pin_and_roundtrip("O=C(NN=Cc1ccccc1)c1ccccc1",
                                  "N'-benzylidenebenzohydrazide")

    def test_propan_2_ylidene_acetohydrazide(self):
        # ketone-derived ylidene (complex, parenthesised): (CH3)2C=N-NH-CO-CH3
        _assert_pin_and_roundtrip("CC(C)=NNC(=O)C",
                                  "N'-(propan-2-ylidene)acetohydrazide")

    def test_bare_hydrazide_unchanged(self):
        # the parent hydrazide must still name (no regression on the acyl side)
        assert name_compound("NNC(=O)C", style="pin") == "acetohydrazide"

    def test_bare_hydrazone_unchanged(self):
        # the bare hydrazone must still name (no regression on the ylidene side)
        assert name_compound("CC=NN", style="pin") == "ethylidenehydrazine"
