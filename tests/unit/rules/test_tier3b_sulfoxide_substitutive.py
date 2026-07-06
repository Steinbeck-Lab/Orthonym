"""Wave2 T3b — P-63.6 substitutive sulfoxide/sulfone PINs + conservation.

Functional-class 'dimethyl sulfoxide' style names are general nomenclature
only; the PIN is substitutive with sulfinyl/sulfonyl prefixes built on the
ACID STEM ('methanesulfinyl', from methanesulfinic acid), the senior R as
parent, and the multiplicative form for symmetric diaryls (BB P-63.6:
'(methanesulfinyl)methane' 46154, '1-(ethanesulfinyl)butane' 28094,
"(ethanesulfonyl)ethane" + "1,1'-sulfinyldibenzene" 28110-28115;
'Multiplication of acyclic hydrocarbons is not permitted').

Mechanism: sulfoxide/sulfone joined _PREFIX_ONLY_PRINCIPAL (no suffix form
exists — claiming the PCG starved the polyfunctional path); the dedicated
handlers fire on FG-present + no-PCG, decline when the R-SOx-R' unit does
not cover the whole molecule (the old functional-class alkyl walk silently
dropped atoms beyond a heteroatom: 'CSCCS(=O)C' -> 'ethyl methyl sulfoxide'
with -S-CH3 lost), and build the substitutive PIN through the strict
_classify_oxide_side conservation classifier (linear terminal saturated
all-C chain / plain benzene / plain cycloalkane — else fall back or fail).
"""

import pytest
from rdkit import Chem

from orthonym import name_compound


@pytest.mark.unit
class TestSubstitutivePins:
    @pytest.mark.parametrize("smiles,expected", [
        ("CS(=O)C", "(methanesulfinyl)methane"),          # BB 46154 verbatim
        ("CS(=O)(=O)C", "(methanesulfonyl)methane"),
        ("CCS(=O)CCCC", "1-(ethanesulfinyl)butane"),      # BB 28094 verbatim
        ("CCS(=O)(=O)CC", "(ethanesulfonyl)ethane"),      # BB 28115 verbatim
        ("CS(=O)c1ccccc1", "(methanesulfinyl)benzene"),
        ("CS(=O)C1CCCCC1", "(methanesulfinyl)cyclohexane"),
    ])
    def test_substitutive(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("O=S(c1ccccc1)c1ccccc1", "1,1'-sulfinyldibenzene"),    # BB 28110
        ("O=S(=O)(c1ccccc1)c1ccccc1", "1,1'-sulfonyldibenzene"),
    ])
    def test_symmetric_diaryl_multiplicative(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestPolyfunctionalAcidStemPrefix:
    """Non-principal sulfoxide/sulfone: acid-stem prefix with enclosing
    marks (BB 28150 verbatim analog '2-(methanesulfonyl)ethan-1-ol')."""

    @pytest.mark.parametrize("smiles,expected", [
        ("CS(=O)CCO", "2-(methanesulfinyl)ethan-1-ol"),
        ("OCCS(C)(=O)=O", "2-(methanesulfonyl)ethan-1-ol"),
        ("CS(=O)CC(=O)O", "2-(methanesulfinyl)ethanoic acid"),
        ("CS(=O)CCN", "2-(methanesulfinyl)ethan-1-amine"),
    ])
    def test_prefix_form(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.fixture
def _validity_gate_on(monkeypatch):
    """The suite's autouse fixture disables the OPSIN validity gate so RAW
    output is asserted. The conservation cases below are ABOUT the gate
    suppressing a wrong RAW name to 'unknown' in production — re-enable it
    (the test_opsin_validity_gate.py pattern)."""
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    yield


@pytest.mark.unit
class TestConservation:
    def test_beyond_unit_atoms_decline(self, _validity_gate_on):
        # -S-CH3 beyond the sulfoxide unit: the FC walk used to emit
        # 'ethyl methyl sulfoxide' (a different molecule). Now fail-closed;
        # the PIN 1-(methanesulfinyl)-2-(methylsulfanyl)ethane (BB 18284)
        # needs the thioether principal-group demotion — documented T3b
        # deferral (same prefix-only debt as ether).
        assert name_compound("CSCCS(=O)C") == "unknown organic compound"

    def test_benzyl_side_declines(self, _validity_gate_on):
        # A benzyl side is not an honestly-nameable FC side (the old walk
        # flattened it to 'heptyl'); substitutive needs the benzene-path
        # S-branch namer (deferred) -> deterministic unknown.
        assert name_compound("CS(=O)(=O)Cc1ccccc1") == "unknown organic compound"

    def test_classifier_shapes(self):
        from orthonym.rules.sulfur import _classify_oxide_side
        mol = Chem.MolFromSmiles("CCS(=O)CCCC")
        s = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'S')
        sides = sorted(
            _classify_oxide_side(mol, n.GetIdx(), s)[0]
            for n in mol.GetAtomWithIdx(s).GetNeighbors()
            if n.GetSymbol() == 'C'
        )
        assert sides == ['butane', 'ethane']
        # branched side -> None
        mol2 = Chem.MolFromSmiles("CC(C)S(=O)C")
        s2 = next(a.GetIdx() for a in mol2.GetAtoms() if a.GetSymbol() == 'S')
        branched = next(
            n.GetIdx() for n in mol2.GetAtomWithIdx(s2).GetNeighbors()
            if n.GetSymbol() == 'C' and n.GetDegree() > 1
        )
        assert _classify_oxide_side(mol2, branched, s2) is None

    def test_fc_coverage_guard(self):
        from orthonym.rules.sulfur import chalcogen_oxide_fc_covers_molecule
        mol = Chem.MolFromSmiles("CSCCS(=O)C")
        match = mol.GetSubstructMatch(
            Chem.MolFromSmarts("[SX3](=[OX1])([#6])[#6]"))
        # Full-BFS sides do cover everything here; the decline comes from
        # the substitutive/FC side classifiers. The guard's job is the
        # simple-molecule contract:
        mol2 = Chem.MolFromSmiles("CS(=O)C")
        match2 = mol2.GetSubstructMatch(
            Chem.MolFromSmarts("[SX3](=[OX1])([#6])[#6]"))
        assert chalcogen_oxide_fc_covers_molecule(mol2, match2)
        assert chalcogen_oxide_fc_covers_molecule(mol, match)

    def test_needs_brackets_acid_stems(self):
        from orthonym.assembly.naming_utils import needs_brackets
        assert needs_brackets("methanesulfinyl")
        assert needs_brackets("methanesulfonyl")
        assert needs_brackets("benzenesulfinyl")
        assert needs_brackets("cyclohexanesulfinyl")
        assert not needs_brackets("sulfinyl")

    def test_acid_stem_prefix_builder(self):
        from orthonym.assembly.substituent_prefix_forms import (
            get_sulfinyl_prefix,
        )
        mol = Chem.MolFromSmiles("CS(=O)CC(=O)O")
        match = mol.GetSubstructMatch(
            Chem.MolFromSmarts("[SX3](=[OX1])([#6])[#6]"))
        chain = [a.GetIdx() for a in mol.GetAtoms()
                 if a.GetSymbol() == 'C' and not a.IsInRing()
                 and a.GetIdx() not in match[:2]]
        result = get_sulfinyl_prefix(mol, match, None)
        assert result == "methanesulfinyl"


@pytest.mark.unit
class TestTrivialAndControls:
    def test_sulfide_functional_class_unchanged(self):
        assert name_compound("CSC") == "dimethyl sulfide"
        assert name_compound("CSCC") == "ethyl methyl sulfide"

    def test_sulfonic_acid_unchanged(self):
        assert name_compound("CS(=O)(=O)O") == "methanesulfonic acid"

    def test_sulfonamide_unchanged(self):
        assert name_compound("CCS(N)(=O)=O") == "ethanesulfonamide"
