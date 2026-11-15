"""Atom-by-atom comparison of a drawing with a name's read-back (``validation/depiction_match.py``).

The standard InChIKey does not see the hydrogens of an over-valent halogen (``C1CC[Cl]CC1`` and ``C1CC[ClH]CC1``
share one key, and so do ``C[Cl-]1CCCCC1`` and ``C[ClH-]1CCCCC1``). Hydrogen counts follow OpenSMILES: an
unbracketed halogen beyond valence 1 has none, which RDKit does not give an unbracketed iodine with two bonds.
"""
import pytest
from rdkit import Chem

from orthonym.validation.depiction_match import _relaxed, depiction_matches


@pytest.mark.parametrize("drawn, read", [
    ("C1CC[Cl]CC1", "Cl1CCCCC1"),             # OPSIN's '1λ3-chlorinan-1-yl'
    ("CC[Cl-]CC", "CC[Cl-]CC"),               # 'diethylchloranuide'
    ("FCl(F)F", "FCl(F)F"),                   # 'trifluoro-λ3-chlorane'
    ("C1CC[I]CC1", "I1CCCCC1"),               # OPSIN's '1λ3-iodinan-1-yl'
    ("C1=CC=CC=C1", "c1ccccc1"),
])
def test_the_same_drawing_matches(drawn, read):
    assert depiction_matches(_relaxed(drawn), read)
    assert depiction_matches(_relaxed(drawn), read, halogens_only=True)


@pytest.mark.parametrize("drawn, read", [
    ("C1CC[Cl]CC1", "[ClH]1CCCCC1"),          # one key, one hydrogen more on chlorine ('1λ3-chlorinane')
    ("C[Cl-]1CCCCC1", "C[ClH-]1CCCCC1"),      # one key, one hydrogen more on the chloranuide
    ("C1CC[IH]CC1", "I1CCCCC1"),              # RDKit would give the unbracketed iodine one hydrogen
    ("CCO", "CCN"),
])
def test_another_drawing_does_not_match(drawn, read):
    if drawn != "CCO":
        assert Chem.MolToInchiKey(_relaxed(drawn)) == Chem.MolToInchiKey(_relaxed(read))
    assert not depiction_matches(_relaxed(drawn), read)
    assert not depiction_matches(_relaxed(drawn), read, halogens_only=True)


def test_halogens_only_ignores_how_the_rest_is_drawn():
    # an amide drawn as its iminol tautomer differs off the halogen: the caller's full key decides the rest
    assert depiction_matches(_relaxed("CC(=O)N.C[Cl-]C"), "CC(O)=N.C[Cl-]C", halogens_only=True)
    assert not depiction_matches(_relaxed("CC(=O)N.C[Cl-]C"), "CC(O)=N.C[Cl-]C")
