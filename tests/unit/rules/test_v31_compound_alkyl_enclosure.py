""" lever B — enclosing marks for a compound FG-prefix + alkyl.

A SUBSTITUTED alkyl ('hydroxymethyl', 'cyanomethyl', 'carboxymethyl',
'aminomethyl') is a COMPOUND substituent and takes enclosing marks even though it
carries no locant/hyphen. Two divergent enclosure predicates missed this class
(they keyed on digits/hyphens only), so it was cited BARE:

* N-substituent path -> 'N-cyanomethylacetamide' (want 'N-(cyanomethyl)...')
* ring-decoration path -> 'N-(4-hydroxymethylphenyl)...'
                                       (want 'N-[4-(hydroxymethyl)phenyl]...')

Fixed by teaching BOTH ``naming_utils.is_complex_substituent`` and
``ring_substituents.decorated_ring_substituent_name``'s local ``_is_complex_prefix``
the same ``_COMPOUND_FG_PREFIXES`` + alkyl-root recognition that
``naming_utils.needs_brackets`` already used.

Blue Book PIN authority: '2-(hydroxymethyl)benzene-1,4-diol' (the Blue Book),
'6-(hydroxymethyl)oxane-2,3,4-triol' (:2688), '4-chloro-2-(hydroxymethyl)-5-oxohexyl'
(:25293), '(cyanomethyl)' (:33087). All names below are RT-exact.
"""
from __future__ import annotations

import pytest

from orthonym.namer import name_compound
from orthonym.assembly.naming_utils import (
    is_complex_substituent, needs_brackets, enclose_if_compound,
)


@pytest.mark.parametrize("name", [
    "hydroxymethyl", "cyanomethyl", "carboxymethyl", "aminomethyl",
    "nitromethyl", "aminoethyl", "hydroxyethyl",
])
def test_compound_alkyl_is_complex_and_needs_brackets(name):
    # both predicates now agree this class is compound
    assert is_complex_substituent(name) is True
    assert needs_brackets(name) is True
    assert enclose_if_compound(name) == f"({name})"


@pytest.mark.parametrize("name", ["methyl", "ethyl", "phenyl", "benzyl", "chloro"])
def test_simple_substituents_stay_bare(name):
    assert is_complex_substituent(name) is False
    assert enclose_if_compound(name) == name


@pytest.mark.parametrize("smiles,expected", [
    # N-substituent path
    ("CC(=O)NCC#N", "N-(cyanomethyl)acetamide"),
    # ring-decoration path: inner  forces outer  escalation
    ("CC(=O)Nc1ccc(CO)cc1", "N-[4-(hydroxymethyl)phenyl]acetamide"),
    # ring decorations on non-amide parents (BB-shaped)
    ("OCc1ccccc1C(=O)O", "2-(hydroxymethyl)benzoic acid"),
    ("N#Cc1ccc(CO)cc1", "4-(hydroxymethyl)benzonitrile"),
    ("OCc1ccc(CN)cc1", "[4-(aminomethyl)phenyl]methanol"),
    # already-locanted compound alkyl unchanged (was already enclosed)
    ("CC(=O)Nc1ccc(C(C)O)cc1", "N-[4-(1-hydroxyethyl)phenyl]acetamide"),
])
def test_compound_alkyl_enclosed_in_full_names(smiles, expected):
    assert name_compound(smiles) == expected
