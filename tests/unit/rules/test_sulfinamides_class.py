"""Sulfinamides -SO-NH2 — perception, seniority and suffix/prefix wiring.

Blue Book authority
-------------------
**** "Sulfonamides, sulfinamides, and related selenium and tellurium
amides" (``the Blue Book Blue Book``) — the section heading names the
class, and the suffix table at ``:32746`` reads:

    "Sulfonamides, sulfinamides, and the analogous selenium and tellurium amides
    are named substitutively using the following suffixes: -SO2-NH2 sulfonamide
    (preselected suffix) **-SO-NH2 sulfinamide (preselected suffix)**..."

**Table 6.1 item 24** (``:18782``) ``| 24. Sulfinamides | -SO-NH2 | sulfinamide |``
places it below sulfonamide (19) / sulfonimidamide (20) and above
sulfinimidamide (25).

Worked ``(PIN)`` examples, each re-opened at write time:

* ``butane-2-sulfinamide (PIN)`` (``:32754``) — a 4-carbon chain CITES its
  suffix locant.
* ``N-hydroxypropane-1-sulfinamide (PIN)`` (``:31236``) — 3 carbons, locant cited.
* ``3-[(aminosulfinyl)oxy]propanoic acid (PIN)`` (``:36500``), printed against
  "[not 3-(sulfinamidoyloxy)propanoic acid; the name sulfinamidic acid is not an
  approved name]" — which is also why the SMARTS must exclude H2N-S(=O)-OH.
* ``aminosulfinyl* (not sulfinamoyl) | H2N-S(O)- | `` (``:55485``) —
  the prefix form is ``aminosulfinyl``; ``sulfinamoyl`` is named as NOT preferred.

These tests assert at the DATA/PRODUCER level (the SMARTS, ``SENIORITY_ORDER``,
``SUFFIX_FORMS``, ``PREFIX_FORMS``) rather than through ``name_compound``: the
OPSIN self-consistency gate can suppress a wrong name before an assertion could
see it, which is how earlier mutations survived a green suite in this tree.
End-to-end emission for this class is covered by ``test_sulfinamide_emission``.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym.assembly.naming_utils import _ETHANE_SUFFIX_ELIDE_FGS
from orthonym.perception.functional_groups import FUNCTIONAL_GROUP_SMARTS
from orthonym.rules.seniority import PREFIX_FORMS, SENIORITY_ORDER, SUFFIX_FORMS

SULFINAMIDES = ("primary_sulfinamide", "secondary_sulfinamide", "tertiary_sulfinamide")


def _matches(key: str, smiles: str) -> bool:
    patt = Chem.MolFromSmarts(FUNCTIONAL_GROUP_SMARTS[key])
    assert patt is not None, f"{key} SMARTS failed to compile"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"bad test SMILES {smiles}"
    return mol.HasSubstructMatch(patt)


# ---------------------------------------------------------------------------
# 1. Perception — the three buckets, and the split that keeps a carbon alive
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("key", SULFINAMIDES)
def test_sulfinamide_smarts_are_registered_and_compile(key):
    assert key in FUNCTIONAL_GROUP_SMARTS
    assert Chem.MolFromSmarts(FUNCTIONAL_GROUP_SMARTS[key]) is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    "key,smiles",
    [
        ("primary_sulfinamide", "CS(=O)N"),
        ("primary_sulfinamide", "c1ccccc1S(=O)N"),
        ("secondary_sulfinamide", "CS(=O)NC"),
        ("tertiary_sulfinamide", "CS(=O)N(C)C"),
    ],
)
def test_sulfinamide_bucket_matches_its_own_substitution(key, smiles):
    assert _matches(key, smiles)


@pytest.mark.unit
def test_primary_bucket_does_not_swallow_n_substituted_forms():
    """The primary/secondary/tertiary split is LOAD-BEARING, not cosmetic.

    A single permissive ``[NX3]`` bucket would route ``CS(=O)NC`` down the
    generic suffix path and silently DROP the N-methyl carbon — the defect
    recorded in ``rules/sulfonamides.py``. With the split, the N-substituted
    forms have no producer and therefore fail CLOSED instead of shipping a name
    that is missing an atom.
    """
    assert not _matches("primary_sulfinamide", "CS(=O)NC")
    assert not _matches("primary_sulfinamide", "CS(=O)N(C)C")


@pytest.mark.unit
def test_carbon_guard_excludes_the_unapproved_sulfinamidic_acid():
    """the Blue Book — "the name sulfinamidic acid is not an approved name".

    ``H2N-S(=O)-OH`` has no S-C bond, so the ``$([SX3][#6])`` guard must reject
    it; without the guard the pattern would claim it and invent a parent.
    """
    for key in SULFINAMIDES:
        assert not _matches(key, "NS(=O)O")


@pytest.mark.unit
def test_sulfinamide_is_disjoint_from_its_nearest_rivals():
    """SX3-vs-SX4 and the N/O split keep this class from stealing others' matches."""
    rivals = {
        "sulfonamide": "CS(=O)(=O)N",       # SX4 — one more oxygen
        "sulfinic_acid": "CS(=O)O",          # O, not N
        "sulfoxide": "CS(=O)C",              # C, not N
        "sulfinyl_halide": "CS(=O)Cl",
    }
    for label, smi in rivals.items():
        for key in SULFINAMIDES:
            assert not _matches(key, smi), f"{key} wrongly matched {label} ({smi})"


@pytest.mark.unit
def test_sulfinohydrazide_and_n_hydroxy_are_not_claimed():
    """N bearing N or O is a different class; neither may be misnamed as sulfinamide."""
    assert not _matches("primary_sulfinamide", "CS(=O)NN")
    assert not _matches("primary_sulfinamide", "CS(=O)NO")


# ---------------------------------------------------------------------------
# 2. Seniority — a SMARTS absent from SENIORITY_ORDER can never be principal
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("key", SULFINAMIDES)
def test_sulfinamide_is_in_the_seniority_order(key):
    """``seniority.py`` iterates the ORDER LIST, not the perception dict.

    An FG present only in ``FUNCTIONAL_GROUP_SMARTS`` is inert — it can never be
    selected as the principal characteristic group. This is the "presence in a
    table is not evidence the table is reached" trap; the row is what makes the
    SMARTS live.
    """
    assert key in SENIORITY_ORDER


@pytest.mark.unit
def test_table_6_1_ordering_sulfonimidamide_then_sulfinamide_then_sulfinimidamide():
    """Table 6.1 (the Blue Book): item 20 > item 24 > item 25."""
    order = SENIORITY_ORDER
    assert order.index("sulfonimidamide") < order.index("primary_sulfinamide")
    for key in SULFINAMIDES:
        assert order.index(key) < order.index("sulfinimidamide")


# ---------------------------------------------------------------------------
# 3. Suffix / prefix wiring
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("key", SULFINAMIDES)
def test_suffix_word_is_sulfinamide_for_both_chain_and_ring(key):
    """BB:32746 — one word for either parent: methanesulfinamide / benzenesulfinamide."""
    assert SUFFIX_FORMS[key] == ("sulfinamide", "sulfinamide")


@pytest.mark.unit
@pytest.mark.parametrize("key", SULFINAMIDES)
def test_prefix_form_is_aminosulfinyl_not_sulfinamoyl(key):
    """the Blue Book prints ``aminosulfinyl* (not sulfinamoyl)``.

    The prefix is required so a DEMOTED sulfinamide is not dropped by the
    ``no_fg_prefix_form`` skip when a senior group takes the suffix — the
    difference between ``3-(aminosulfinyl)propanoic acid`` and a silently lost
    functional group.
    """
    assert PREFIX_FORMS[key] == "aminosulfinyl"
    assert PREFIX_FORMS[key] != "sulfinamoyl"


@pytest.mark.unit
@pytest.mark.parametrize("key", SULFINAMIDES)
def test_two_carbon_parent_elides_the_suffix_locant(key):
    """``ethanesulfinamide``, not ``ethane-1-sulfinamide``.

    The symmetric 2-carbon parent has only one distinguishable position, so
     lets the locant go. A 3+ carbon chain KEEPS it, which the Blue
    Book's own PINs require: ``butane-2-sulfinamide`` (the Blue Book) and
    ``N-hydroxypropane-1-sulfinamide`` (the Blue Book).
    """
    assert key in _ETHANE_SUFFIX_ELIDE_FGS
