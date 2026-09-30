"""Merged N + numeric prefix lists on amides — / /.

THE TWO DEFECTS THIS CLOSES (they are NOT one class)
----------------------------------------------------
``benzene.py::_name_substituted_benzamide`` spelled the N-substituents into
their own string and glued it in front of the ring-prefix string. That produced
two independent errors, and BOTH round-tripped cleanly through OPSIN to the
correct InChIKey -- a round trip proves the STRUCTURE, never the SPELLING:

* **MERGE** ``CNC(=O)c1ccc(C)cc1`` -> ``N-methyl-4-methylbenzamide``
  PIN ``N,4-dimethylbenzamide``
* **ORDER** ``CNC(=O)c1cccc(Cl)c1`` -> ``N-methyl-3-chlorobenzamide``
  PIN ``3-chloro-N-methylbenzamide``

`chloro` and `methyl` are DIFFERENT prefix names and can never merge, so the
second is purely an alphanumerical ORDER defect. Keeping them apart matters:
the merge fix alone would not have repaired it.

A third error surfaced only once the first two were fixed: the joiner
hyphenated before a DIGIT only, so an italic locant glued on as
``3-chloroN-methylbenzamide`` -- which OPSIN also accepted.

BLUE BOOK AUTHORITY (each pointer re-opened with `sed -n '<N>p'` at write time)
------------------------------------------------------------------------------
**** (``the Blue Book``) -- OCR-mangled in the dump (`P"16.3.3`,
`!` for spaces), so it is quoted from the surrounding block:

    "The basic numerical prefixes 'di', 'tri', 'tetra', etc. are used to
    indicate a multiplicity of:... (b) simple substituent prefixes, including
    parent hydrides with 'ene' and 'yne' endings (without locants)"

with ``dimethyl`` printed in clause (b)'s own example list (``:7067`` ff).
Multiplicity is therefore a property of the substituent NAME, not of which atom
carries it -- an ``N``-methyl and a ring 4-methyl are ONE group of two.

**** "Citation of locants" (``:2869``) is deny-by-default: "if any
locants are essential... then all locants must be cited". So the merged set is
cited whole: ``N,4-``.

The Blue Book's own MIXED italic/numeral sets fix the rendering and the
intra-set order (italic letters lead):

* ``:42213`` ``*N*,*N*,*N*,1-tetramethylquinolin-1-ium-3-aminium (PIN)``
* ``:42460`` ``*N*,1,4-triphenyl-1*H*-1,2,4-triazol-4-ium-3-aminide (PIN)``

And within itself, ``:32879``
``*N*,4-dimethyl-*N*-(3-methylphenyl)benzamide (PIN)`` is the merge, while
``:32881`` ``3-chloro-*N*-(2-chlorophenyl)naphthalene-2-sulfonamide (PIN)``
is the order -- a chalcogen-acid amide citing the parent's ``3-chloro`` BEFORE
the N-prefix.
"""
from __future__ import annotations

import pytest

from orthonym.assembly.naming_utils import (
    format_substituent_prefix, locant_sort_key, starts_with_locant,
)
from orthonym.namer import name_compound


# --------------------------------------------------------------------------
# The canonical formatter reproduces the Blue Book's mixed sets byte-exactly.
# It was never broken -- the N-substituents were simply never put in its bucket.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,locants,count,expected", [
    ("methyl", ["N", 4], 2, "N,4-dimethyl"),            #:32879
    ("methyl", ["N", "N", "N", 1], 4, "N,N,N,1-tetramethyl"),  #:42213
    ("phenyl", ["N", 1, 4], 3, "N,1,4-triphenyl"),      #:42460
    ("methyl", ["N"], 1, "N-methyl"),
    ("propan-2-yl", ["N"], 1, "N-(propan-2-yl)"),
])
def test_formatter_renders_mixed_locant_sets(name, locants, count, expected):
    assert format_substituent_prefix(name, locants, count) == expected


def test_locant_sort_key_puts_italic_before_numeral():
    """:42213 `N,N,N,1-tetramethyl` -- italic letters lead, numerals ascend."""
    assert sorted([4, "N"], key=locant_sort_key) == ["N", 4]
    assert sorted([1, "N", "N", "N"], key=locant_sort_key) == ["N", "N", "N", 1]
    assert sorted([4, 1, "N"], key=locant_sort_key) == ["N", 1, 4]
    # A naive sort cannot do this at all -- that is why the key exists.
    with pytest.raises(TypeError):
        sorted([4, "N"])


@pytest.mark.parametrize("prefix,expected", [
    ("4-chloro", True), ("N-methyl", True), ("N,4-dimethyl", True),
    ("N,N-dimethyl", True), ("1,2,4-trimethyl", True),
    ("chloro", False), ("methyl", False), ("", False),
    # fails SAFE: an unrecognised head is not treated as a locant
    ("tert-butyl", False),
])
def test_starts_with_locant(prefix, expected):
    assert starts_with_locant(prefix) is expected


# --------------------------------------------------------------------------
# DEFECT 1 — MERGE. Identical prefix names at N and at a numbered position are
# ONE group with ONE multiplying prefix over the whole locant set.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("CNC(=O)c1ccc(C)cc1",   "N,4-dimethylbenzamide"),
    ("CN(C)C(=O)c1ccc(C)cc1", "N,N,4-trimethylbenzamide"),
])
def test_identical_prefixes_merge_across_n_and_ring(smiles, expected):
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------
# DEFECT 2 — ORDER. Different prefix names cannot merge; they are cited in
# alphanumerical order, which is NOT "N first" (:32881).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("CNC(=O)c1cccc(Cl)c1",  "3-chloro-N-methylbenzamide"),
    ("CNC(=O)c1ccc(Cl)cc1",  "4-chloro-N-methylbenzamide"),
    ("CNC(=O)c1ccc(O)cc1",   "4-hydroxy-N-methylbenzamide"),
    ("CC(C)NC(=O)c1ccc(C)cc1", "4-methyl-N-(propan-2-yl)benzamide"),
])
def test_different_prefixes_are_alphanumerically_ordered(smiles, expected):
    assert name_compound(smiles) == expected


def test_italic_locant_is_not_glued_to_the_previous_prefix():
    """Regression: the joiner hyphenated before a DIGIT only, so an italic
    locant produced `3-chloroN-methylbenzamide` -- which OPSIN accepted."""
    out = name_compound("CNC(=O)c1cccc(Cl)c1")
    assert "chloroN" not in out
    assert "-N-methyl" in out


# --------------------------------------------------------------------------
# Unchanged shapes — the merge must not disturb the cases that were correct.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("CNC(=O)c1ccccc1",  "N-methylbenzamide"),
    ("CN(C)C(=O)c1ccccc1", "N,N-dimethylbenzamide"),
    ("Cc1ccc(cc1)C(=O)N", "4-methylbenzamide"),
    ("Clc1ccc(cc1)C(=O)N", "4-chlorobenzamide"),
    ("Cc1ccc(cc1)Cl",     "1-chloro-4-methylbenzene"),
    ("Cc1cc(C)ccc1C",     "1,2,4-trimethylbenzene"),
])
def test_unchanged_shapes(smiles, expected):
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------
# KNOWN REMAINING GAP — the CHAIN amide producer is a different site.
#
# `composer.py::_assemble_amide_name` (~:5656) receives `base_name` with the
# N-prefix ALREADY spelled into it, so the N-substituents are not available as
# structured data at the point the numeric prefixes are ordered, and they cannot
# merge. Fixing it means changing what `name_amide` returns -- a refactor in
# composer.py, which was outside this task's file surface.
#
# strict=True so this self-clears loudly the moment that site is fixed, rather
# than silently asserting a defect.
# --------------------------------------------------------------------------
# The chain-amide site merges an N-substituent with an acyl prefix of the same
# name (composer._amide_with_identical_prefixes_merged, the class-2 grouping) and,
# since a performance pass, cites every N-substituent in the one alphanumerical series.
def test_chain_amide_merges_too():
    assert name_compound("CC(C)C(=O)NC") == "N,2-dimethylpropanamide"
