"""Tests for anhydride naming (19-02).

Tests functional class naming for anhydrides following IUPAC conventions:
- Symmetric acyclic: acetic anhydride (retained acid name, PIN per
- Mixed/asymmetric: acetic propanoic anhydride (alphabetical)
- Cyclic (from diacids): butanedioic anhydride
- Consumed-atom filtering: anhydride atoms removed from ester FG detection
"""
import pytest
from orthonym import name_compound


class TestSymmetricAnhydrides:
    """Test symmetric acyclic anhydride naming."""

    def test_ethanoic_anhydride(self):
        """Acetic anhydride - simplest symmetric (retained acid PIN."""
        result = name_compound("CC(=O)OC(=O)C")
        assert result == "acetic anhydride"

    def test_propanoic_anhydride(self):
        """Propanoic anhydride - C3 symmetric."""
        result = name_compound("CCC(=O)OC(=O)CC")
        assert result == "propanoic anhydride"

    def test_butanoic_anhydride(self):
        """Butanoic anhydride - C4 symmetric."""
        result = name_compound("CCCC(=O)OC(=O)CCC")
        assert result == "butanoic anhydride"

    def test_methanoic_anhydride(self):
        """Formic anhydride - C1 symmetric (retained acid PIN."""
        result = name_compound("O=COC=O")
        assert result == "formic anhydride"


class TestMixedAnhydrides:
    """Test mixed/asymmetric acyclic anhydride naming."""

    def test_ethanoic_propanoic_anhydride(self):
        """Acetic propanoic anhydride - alphabetical order (acetic retained)."""
        result = name_compound("CC(=O)OC(=O)CC")
        assert result == "acetic propanoic anhydride"

    def test_butanoic_propanoic_anhydride(self):
        """Butanoic propanoic anhydride - mixed C4+C3."""
        result = name_compound("CCC(=O)OC(=O)CCC")
        assert result == "butanoic propanoic anhydride"


class TestThioSelenoAcylAnhydrides:
    """D: acyclic thio-/seleno-ACYL anhydrides R-C(=X)-Y-C(=X)-R'.

    The acyl chalcogen (=O/=S/=Se) fixes each acid component's affix
    ('...thioic'/'...selenoic';; components are cited in
    ALPHABETICAL order; the BRIDGE chalcogen (O/S/Se/Te) fixes the
    class word ('anhydride'/'thioanhydride'/'selenoanhydride';.
    All names OPSIN-round-trip to the input structure.
    """

    def test_propanethioic_anhydride(self):
        """ symmetric thioacyl, O bridge: C(=S)-O-C(=S)."""
        result = name_compound("CCC(=S)OC(=S)CC")
        assert result == "propanethioic anhydride"

    def test_benzenecarbothioic_anhydride(self):
        """ symmetric ring thioacyl (benzoyl =S), O bridge."""
        result = name_compound("c1ccccc1C(=S)OC(=S)c1ccccc1")
        assert result == "benzenecarbothioic anhydride"

    def test_ethanethioic_propanoic_anhydride(self):
        """ mixed acyl (=S + =O), O bridge -> 'anhydride';
        alphabetical order 'ethanethioic' < 'propanoic'."""
        result = name_compound("CC(=S)OC(=O)CC")
        assert result == "ethanethioic propanoic anhydride"

    def test_ethanethioic_propanethioic_anhydride(self):
        """ both acyls =S, O bridge -> 'anhydride'."""
        result = name_compound("CC(=S)OC(=S)CC")
        assert result == "ethanethioic propanethioic anhydride"

    def test_ethanethioic_propanethioic_thioanhydride(self):
        """ both acyls =S, S bridge -> 'thioanhydride'."""
        result = name_compound("CC(=S)SC(=S)CC")
        assert result == "ethanethioic propanethioic thioanhydride"

    def test_ethanethioic_propanoic_thioanhydride(self):
        """ mixed acyl (=S + =O), S bridge -> 'thioanhydride'."""
        result = name_compound("CC(=S)SC(=O)CC")
        assert result == "ethanethioic propanoic thioanhydride"

    def test_acetic_propanethioic_selenoanhydride(self):
        """ Se bridge -> 'selenoanhydride'; the =O acetic component
        keeps its retained name, the =S propanethioic its thioic affix."""
        result = name_compound("CC(=O)[Se]C(=S)CC")
        assert result == "acetic propanethioic selenoanhydride"


class TestCyclicAnhydrides:
    """Test cyclic anhydride naming (from dicarboxylic acids)."""

    def test_butanedioic_anhydride(self):
        """Succinic anhydride - 5-membered ring. D-FOLLOWON item 6: PIN is the
        heterocyclic-pseudoketone dione method 1), not the non-PIN
        functional-class 'butanedioic anhydride'."""
        result = name_compound("O=C1CCC(=O)O1")
        assert result == "oxolane-2,5-dione"

    def test_pentanedioic_anhydride(self):
        """Glutaric anhydride - 6-membered ring. PIN = oxane-2,6-dione (item 6)."""
        result = name_compound("O=C1CCCC(=O)O1")
        assert result == "oxane-2,6-dione"

    def test_cyclic_anhydride_not_lactone(self):
        """Cyclic anhydrides must NOT be mis-named as a (mono)lactone by the
        lactone handler — the dione names BOTH ring carbonyls."""
        # O=C1CCC(=O)O1 has two C=O in the ring -> oxolane-2,5-dione (a dione),
        # NOT the single-carbonyl lactone oxolan-2-one.
        result = name_compound("O=C1CCC(=O)O1")
        assert result == "oxolane-2,5-dione"
        assert "-2-one" not in result  # not the mono-lactone oxolan-2-one


class TestConsumedAtomFilteringAnhydride:
    """Test that anhydride consumed-atom filtering removes ester matches."""

    def test_anhydride_filters_ester(self):
        """Anhydride atoms must not double-name as an ester. The collision
        resolver inside detect_functional_groups already suppresses the ester
        sub-read on the anhydride atoms (the ('anhydride', [...,'ester',...])
        rule), so ester is absent from detection; the consumed-atom filter then
        also leaves anhydride owning its atoms."""
        from rdkit import Chem
        from orthonym.perception.functional_groups import detect_functional_groups
        from orthonym.namer import _filter_consumed_fg_atoms

        mol = Chem.MolFromSmiles("CC(=O)OC(=O)C")
        fgs = detect_functional_groups(mol)
        assert "anhydride" in fgs
        assert "ester" not in fgs  # ester sub-read suppressed at detection

        filtered = _filter_consumed_fg_atoms(fgs)
        assert "ester" not in filtered  # still no ester after filtering
        assert "anhydride" in filtered  # anhydride still present


class TestDicarbonicDihalidePseudohalide:
    """ a phase: dicarbonic dihalides / dipseudohalides,
    X-CO-O-CO-Y -> 'dicarbonic <class word(s)>'."""

    def test_dicarbonic_dichloride(self):
        # the Blue Book 'Cl-CO-O-CO-Cl dicarbonic dichloride (PIN)'.
        assert name_compound("ClC(=O)OC(=O)Cl") == "dicarbonic dichloride"

    def test_dicarbonic_bromide_chloride(self):
        # the Blue Book mixed halide, alphabetical order.
        assert name_compound("ClC(=O)OC(=O)Br") == "dicarbonic bromide chloride"

    def test_dicarbonic_diisocyanate(self):
        # the Blue Book 'OCN-CO-O-CO-NCO dicarbonic diisocyanate (PIN)'.
        assert name_compound("O=C=NC(=O)OC(=O)N=C=O") == "dicarbonic diisocyanate"
