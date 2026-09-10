"""Identical substituents on a heterocycle collapse into ONE multiplying prefix.

THE DEFECT
----------
``Cn1ccnc1C`` was named ``1-methyl-2-methyl-1H-imidazole`` -- the prefix ``methyl``
spelled twice -- instead of ``1,2-dimethyl-1H-imidazole``. ``Cn1c(C)nc(C)c1C`` was
named ``1-methyl-2,4,5-trimethyl-1H-imidazole``, which is the tell: the three
*ring-carbon* methyls DID collapse into ``2,4,5-trimethyl``, so the multiplier
machinery works. Only the *ring-nitrogen* methyl stood outside it.

THE RULE
--------
* **P-16.3.3** "The basic numerical prefixes 'di', 'tri', 'tetra', etc. are used to
  indicate a multiplicity of:" (``the Blue Book Blue Book``), clause **(b)**
  (``:7067``): "simple substituent prefixes, including parent hydrides with 'ene' and
  'yne' endings (without locants)", whose own example list prints ``dimethyl`` (``:7069``),
  ``diethenyl`` (``:7071``) and ``dibromo`` (``:7073``). ``methyl`` and ``ethyl`` are
  simple unsubstituted prefixes -- **P-16.3.2 (a)** (``:7033``): "Simple components are
  ... unsubstituted prefixes, such as ethyl or *tert*-butyl... All of these are
  multiplied by the multiplicative prefixes 'di', 'tri', etc." -- so the multiplier is
  ``di``/``tri``/``tetra``, not ``bis``/``tris`` (**P-16.3.2 (c)**, ``:7035``, reserves
  those for a *substituted* component).
* **P-14.3.3** "Citation of locants" (``:2867``), sentence ``:2869``: "if any locants
  are essential for defining the structure of the parent structure or of a unit of
  structure as defined by its appropriate enclosing marks, then **all** locants must be
  cited for the parent structure or that structural unit." Deny-by-default: the
  collapsed prefix cites the whole locant set, ``1,2-``.
* **P-14.3.2** "Position of locants" (``:2857``), sentence ``:2859``: "Locants (numerals
  and/or letters) are placed immediately before that part of the name to which they
  relate" -- one locant set, immediately before the one multiplied prefix.

THE N/C CRUX -- a ring nitrogen's locant is an ORDINARY ARABIC NUMERAL
----------------------------------------------------------------------
Mixing a ring-N locant with ring-C locants changes **nothing**, because a ring nitrogen
that is part of the ring numbering does not take an italic ``N`` locant at all -- it
takes its ring number. The Blue Book draws that line explicitly:

* **P-66.1.3** "'Hidden' amides" (``:33125``): naming an acyl group as a substituent on
  a heterocyclic ring nitrogen "is allowed but only in general nomenclature", because
  "preferred IUPAC names are constructed" as pseudoketones -- on the numbered ring.
  ``:33129`` prints ``1-(piperidin-1-yl)ethan-1-one (PIN)`` against ``1-acetylpiperidine``.
* **P-66.1.5.1** "Lactams and lactims" (``:33224``): of its two methods, "(1) as
  heterocyclic pseudoketones" is the one that "generates preferred IUPAC names".
* The ``(PIN)`` examples spell the ring N as a numeral and demote the italic form to the
  general name: ``:27249`` ``pyrrolidine-1,2-diol (PIN)`` (alternatives
  ``1-hydroxypyrrolidin-2-ol``, ``*N*-hydroxypyrrolidin-2-ol``); ``:40645``
  ``2,5-dioxopyrrolidin-1-yl (PIN)``; ``:33847`` ``1-bromopyrrolidine-2,5-dione (PIN)``.

.. warning::

   This docstring used to rest the crux on **P-65.2.3.1.2.1** (``:31041``) alone. The
   quote was verbatim, but the emphasis bolded *around* "amide linkages" and so
   reconstructed the very elision that inverts the sentence -- and the section governs
   **superscripted** locants (*N*\\:sup:`2`, *N*\\:sup:`3`) under P-65.2.3, "di-, tri-,
   tetra-, and polycarbonic acids", not ring nitrogens. Its neighbour **P-65.2.3.1.4**
   (``:31107``) states the convention cleanly -- italic letter locants "are used to
   designate substitution on nitrogen atoms that are not amide linkages for which
   numerical locants are used" -- but carries the same chapter scope, so it corroborates
   rather than governs. Corrected 2026-08-02; do not re-derive this from P-65.2.3.

And the Blue Book prints multiplied prefixes over exactly such mixed locant sets:

* ``:42460`` ``*N*,1,4-triphenyl-1*H*-1,2,4-triazol-4-ium-3-aminide (PIN)`` -- ``1`` and
  ``4`` are the triazole *ring nitrogens* and ``N`` the exocyclic one, all three phenyls
  under ONE ``triphenyl``; the same line's alternative prints ``1,4-diphenyl-`` over two
  ring nitrogens.
* ``:42213`` ``*N*,*N*,*N*,1-tetramethylquinolin-1-ium-3-aminium (PIN)`` -- ring-N
  numeral ``1`` collapsed together with three italic ``N``s under one ``tetramethyl``.
* ``:24914`` ``1,3-dimethyltriazane-2-carboxylic acid (PIN)`` -- ``1`` and ``3`` are both
  nitrogens, cited as plain numerals under one ``dimethyl``.
* ``:41122`` ``1-methoxy-1,3-dimethyl-1*H*-1-benzoborol-1-uide (PIN)`` -- locant ``1`` is
  the ring *boron* and ``3`` a ring carbon: one ``dimethyl`` over a heteroatom/carbon
  mixed set.

THE ROOT CAUSE -- the collapse is NOT absent; the PARTITION is wrong
-------------------------------------------------------------------
``name_substituted_heterocycle`` bucketed prefixes into **two** name-keyed dicts,
``n_groups`` and ``c_groups``, split by ``is_on_nitrogen`` -- which is a *ring-atom*
test (``heterocycles.py:2060``, ``ring_atom.GetSymbol() == 'N'``). Each dict then took
its own ``count = len(locants)``, so a methyl on the ring N and a methyl on a ring C were
two groups of one instead of one group of two. Both formatters already delegate to the
shared ``naming_utils.multiplied_component``; nothing about the multiplier was broken.

The sibling producers already do it right and are the model for the fix:
``rules/fused_rings.py`` leaves its ``n_substituents`` bucket EMPTY and routes ring-N
locants as ordinary numerals (measured: ``_format_n_prefix`` records 0 calls for
``1,2-dimethyl-1H-benzimidazole``), and ``rules/benzene.py`` uses one flat name-keyed
bucket. So the fix is to merge a ring-N group into the shared numeric bucket **whenever
every one of its locants is a number**, leaving the locant-less italic-``N`` fallback
exactly as it was.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.heterocycles import (
    _format_n_substituent,
    get_heterocycle_substituents,
    name_heterocycle,
    name_substituted_heterocycle,
    orient_heterocycle_with_substituents,
)


def produce(smiles: str):
    """Call the PRODUCER directly -- no OPSIN gate in the loop.

    Asserting through ``name_compound`` would let the validity gate suppress a
    mutant before any assertion saw it; these tests bind the generator itself.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    ring = mol.GetRingInfo().AtomRings()[0]
    ring_set = set(ring)
    sub_positions = {
        i for i in ring
        if any(n.GetIdx() not in ring_set
               for n in mol.GetAtomWithIdx(i).GetNeighbors())
    }
    oriented, a2l = orient_heterocycle_with_substituents(mol, ring, sub_positions)
    subs = get_heterocycle_substituents(mol, ring, oriented, a2l)
    parent = name_heterocycle(mol, ring)
    return name_substituted_heterocycle(mol, ring, parent, subs, a2l)


# ---------------------------------------------------------------------------
# 1. The defect -- a ring-N substituent joins its identical ring-C twin
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    # --- the reported probes, aromatic retained azoles ---
    ("Cn1ccnc1C",       "1,2-dimethyl-1H-imidazole"),
    ("Cn1c(C)ncc1",     "1,2-dimethyl-1H-imidazole"),
    ("Cn1cc(C)cn1",     "1,4-dimethyl-1H-pyrazole"),
    ("CCn1cc(CC)cn1",   "1,4-diethyl-1H-pyrazole"),
    # --- the rest of the retained-azole family ---
    ("Cn1cccc1C",       "1,2-dimethyl-1H-pyrrole"),
    ("Cn1ccc(C)c1",     "1,3-dimethyl-1H-pyrrole"),
    ("Cn1cnc(C)n1",     "1,3-dimethyl-1H-1,2,4-triazole"),
    ("Cn1nnnc1C",       "1,5-dimethyl-1H-tetrazole"),
    # --- SATURATED heterocycles: the same defect, NOT new surface ---
    ("CN1CCCC1C",       "1,2-dimethylpyrrolidine"),
    ("CN1CCCCC1C",      "1,2-dimethylpiperidine"),
    ("CN1CCC(C)CC1",    "1,4-dimethylpiperidine"),
    ("CCN1CCCC1CC",     "1,2-diethylpyrrolidine"),
    # morpholine also had its locants out of ascending order ('4-methyl-3-methyl')
    ("CN1C(C)COCC1",    "3,4-dimethylmorpholine"),
])
def test_ring_n_substituent_collapses_with_identical_ring_c(smiles, expected):
    assert produce(smiles) == expected


@pytest.mark.unit
def test_multiplicity_is_recounted_not_merely_relocanted():
    """tri + 1 must become tetra, not stay 'tri'.

    ``1-methyl-2,4,5-trimethyl-1H-imidazole`` is the case that proves the C-side
    collapse already worked; merging the N methyl has to raise the multiplier.
    """
    assert produce("Cn1c(C)nc(C)c1C") == "1,2,4,5-tetramethyl-1H-imidazole"


# ---------------------------------------------------------------------------
# 2. Controls -- these already worked and must stay byte-identical
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    # C-substituents only: the collapse that already worked
    ("Cc1cc(C)[nH]c1",  "2,4-dimethyl-1H-pyrrole"),
    ("Cc1ncc(C)[nH]1",  "2,5-dimethyl-1H-imidazole"),
    # BOTH substituents on ring nitrogens: one bucket already, count 2
    ("CN1CCN(C)CC1",    "1,4-dimethylpiperazine"),
    # a single N-substituent: nothing to collapse
    ("CN1CCCC1",        "1-methylpyrrolidine"),
    # the P-14.3.4.3 locant-omission licence must NOT be unmasked by the merge
    ("CN1CCOCC1",       "4-methylmorpholine"),
])
def test_already_correct_names_are_unchanged(smiles, expected):
    assert produce(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("CN1CCCC1CC",   "2-ethyl-1-methylpyrrolidine"),
    ("Cn1ccnc1CC",   "2-ethyl-1-methyl-1H-imidazole"),
    ("CCn1cc(C)cn1", "1-ethyl-4-methyl-1H-pyrazole"),
    ("Clc1cccn1C",   "2-chloro-1-methyl-1H-pyrrole"),
])
def test_non_identical_substituents_must_not_collapse(smiles, expected):
    """Different prefixes stay separate and alphabetised -- P-16.3.3 multiplies
    a multiplicity of the SAME component only."""
    assert produce(smiles) == expected


@pytest.mark.unit
def test_italic_n_fallback_is_untouched():
    """A ring N with no numeric locant keeps the italic-'N' citation.

    P-66.1.3 (the Blue Book) / P-66.1.5.1 (the Blue Book) -- a ring N that IS numbered takes
    its numeral in a PIN, so the italic form is left to a nitrogen that receives no
    arabic number. The merge is conditioned on every locant being a number, so this
    branch must be unreachable from it. (Not P-65.2.3.1.2.1 -- see the module
    docstring's warning; that section is polycarbonic-acid superscripts.)
    """
    assert _format_n_substituent("methyl", 1, locants=None) == "N-methyl"
    assert _format_n_substituent("methyl", 2, locants=None) == "N,N-dimethyl"
    assert _format_n_substituent("methyl", 2, locants=[1, 4]) == "1,4-dimethyl"


# ---------------------------------------------------------------------------
# 3. Cross-path controls -- other producers must not move (CLAUDE.md #9)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    # carbocyclic producer (rules/benzene.py) -- the reference implementation
    ("Cc1ccc(C)cc1",         "1,4-dimethylbenzene"),
    ("Clc1ccc(Cl)cc1",       "1,4-dichlorobenzene"),
    ("CC(C)c1ccc(C(C)C)cc1", "1,4-di(propan-2-yl)benzene"),
    # fused producer (rules/fused_rings.py) -- already collapses ring-N + ring-C
    ("Cn1c(C)nc2ccccc21",    "1,2-dimethyl-1H-benzimidazole"),
    ("Cn1cc(C)c2ccccc21",    "1,3-dimethyl-1H-indole"),
])
def test_other_producers_are_byte_identical(smiles, expected):
    result = name_compound(smiles)
    assert getattr(result, "name", result) == expected
