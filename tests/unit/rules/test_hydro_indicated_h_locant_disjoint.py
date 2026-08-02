"""A hydro locant and an indicated-hydrogen locant may never be the same locant.

Blue Book P-14.4 "NUMBERING" (``BlueBookV2/BlueBookV2.md:3219``) ranks the
structural features that compete for low locants, in decreasing seniority:

    "(b) indicated hydrogen for unsubstituted compounds" (``:3246``)
    ...
    "(e) saturation/unsaturation: (i) low locants are given to hydro/dehydro
     prefixes ... and 'ene' and 'yne' endings" (``:3288``/``:3289``)

P-31.2.2 "General methodology" (``:16878``) states the consequence outright,
in the LAST two sentences of its paragraph (``:16880``):

    "Indicated hydrogen atoms have priority over 'hydro' prefixes for low
     locants. If indicated hydrogen atoms are present in a name, the 'hydro'
     prefixes precede them."

The two locant sets are therefore not merely ordered but DISJOINT, and that is
structural rather than stylistic: an indicated-hydrogen atom carries no double
bond in the mancude parent, so it has no double bond to lose and can never be a
hydro position.  Every (PIN) example in P-31.2.2 / P-31.2.3.1 obeys it --
``4,5-dihydro-3H-azepine`` (``:16888``), ``3,4-dihydro-2H-pyrrole``
(``:16896``), ``2,7-dihydro-1H-azepine`` (``:16920``),
``2,3-dihydro-1H-phosphole`` (``:16924``).

Regression guarded here (Task AA).  ``_aromatizable_hydro_name`` identified the
indicated-hydrogen atom from the MOLECULE, as "any ring N bearing an H".  That
guess cannot tell two N-H apart, so for every 2-N parent whose hydro form
saturates the second nitrogen the tie-break went slack and the hydro criterion
-- ranked below it -- was free to put locant 1 on a HYDRO nitrogen:

    N1NC=CC1      -> '1,3-dihydro-1H-pyrazole'   (locant 1 cited twice)
    N1CNC=C1      -> '1,2-dihydro-1H-imidazole'  (locant 1 cited twice)
    C1=CNCCCCNC1  -> '1,2,6,7,8,9-hexahydro-1H-1,5-diazonine'

OPSIN cannot see a defect of this shape -- it is a spelling contradiction, not
a structural one -- so these assertions are made at the PRODUCER
(``name_heterocycle``), never via a round-trip.

The same fix repaired a strictly worse class: 8- and 12-membered rings whose
emitted hydro COUNT disagreed with the structure, e.g. ``N1NC=CC=CCC1`` named
'3,4-dihydro-1,2-diazocine' when the molecule is two double bonds short of the
mancude parent, not one.  Those names denoted a DIFFERENT molecule.
"""

import re

import pytest
from rdkit import Chem

from orthonym.rules.heterocycles import name_heterocycle

_HYDRO = re.compile(
    r'^(?P<locs>[\d,]+)-(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)hydro-?'
    r'(?P<ih>(?:\d+H,)*\d+H-)?')


def _name(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"bad test SMILES: {smiles}"
    rings = mol.GetRingInfo().AtomRings()
    assert len(rings) == 1, f"test SMILES must be monocyclic: {smiles}"
    return name_heterocycle(mol, list(rings[0]))


def _locant_sets(name):
    """(hydro locants, indicated-hydrogen locants) parsed out of a hydro name."""
    m = _HYDRO.match(name)
    assert m, f"not a hydro name: {name}"
    hydro = {int(x) for x in m.group('locs').split(',')}
    ih = {int(x) for x in re.findall(r'\d+', m.group('ih') or '')}
    return hydro, ih


# --------------------------------------------------------------------------
# The Blue Book's own worked (PIN) examples. These passed before the fix too --
# they are here so a future change cannot buy the disjointness invariant by
# breaking the names that motivated it.
# --------------------------------------------------------------------------
BB_PIN_EXAMPLES = [
    ('C1C=CC=CN1', '1,2-dihydropyridine', 16914),
    ('N1=CCCCC=C1', '4,5-dihydro-3H-azepine', 16888),
    ('C1CC=NC1', '3,4-dihydro-2H-pyrrole', 16896),
    ('N1CC=CC=CC1', '2,7-dihydro-1H-azepine', 16920),
    ('P1CCC=C1', '2,3-dihydro-1H-phosphole', 16924),
    ('S1C=CNCCC1', '4,5,6,7-tetrahydro-1,4-thiazepine', 16918),
]


@pytest.mark.parametrize('smiles,expected,bb_line', BB_PIN_EXAMPLES)
def test_blue_book_pin_examples(smiles, expected, bb_line):
    assert _name(smiles) == expected


# --------------------------------------------------------------------------
# The defect class: a parent with TWO N-H, where the second N is a hydro
# position. Locant 1 belongs to the indicated hydrogen, so the hydro set must
# start at 2.
# --------------------------------------------------------------------------
TWO_NH_PARENTS = [
    ('N1NC=CC1', '2,5-dihydro-1H-pyrazole'),
    ('C1=CNNC1', '2,3-dihydro-1H-pyrazole'),
    ('N1CNC=C1', '2,3-dihydro-1H-imidazole'),
    ('C1=CNCN1', '2,3-dihydro-1H-imidazole'),
    ('C1=CNCCCCNC1', '4,5,6,7,8,9-hexahydro-1H-1,5-diazonine'),
    ('C1=CC=CNCNC=C1', '2,3-dihydro-1H-1,3-diazonine'),
    # The parent's indicated hydrogen sits on the N adjacent to the surviving
    # C=C, so it takes locant 1 and the hydro set runs 2..7 -- NOT 4..9, which
    # is what the 1,5-isomer above gets. Verified by reading the parent
    # (1H-1,2-diazonine) and the atom its indicated hydrogen maps back to.
    ('C1=CNNCCCCC1', '2,3,4,5,6,7-hexahydro-1H-1,2-diazonine'),
]


@pytest.mark.parametrize('smiles,expected', TWO_NH_PARENTS)
def test_two_nh_parent_keeps_locant_one_for_indicated_hydrogen(smiles, expected):
    assert _name(smiles) == expected


@pytest.mark.parametrize('smiles,_expected', TWO_NH_PARENTS)
def test_hydro_and_indicated_hydrogen_locants_are_disjoint(smiles, _expected):
    hydro, ih = _locant_sets(_name(smiles))
    assert ih, 'this family must carry indicated hydrogen'
    assert not (hydro & ih), (
        f'{_name(smiles)!r} cites {sorted(hydro & ih)} as BOTH a hydro '
        'position and indicated hydrogen (P-14.4(b) vs (e), P-31.2.2:16880)')


# --------------------------------------------------------------------------
# The structure-wrong sibling: the emitted hydro COUNT must equal twice the
# number of double bonds the molecule is short of its mancude parent. These
# names previously denoted a different molecule.
# --------------------------------------------------------------------------
WRONG_COUNT_REGRESSIONS = [
    ('N1NC=CC=CCC1', '1,2,3,4-tetrahydro-1,2-diazocine'),
    ('N1NC=CCC=CC1', '1,2,3,6-tetrahydro-1,2-diazocine'),
    ('N1NC=CCCCC1', '1,2,3,4,5,6-hexahydro-1,2-diazocine'),
    ('N1CCNC=CC=C1', '1,2,3,4-tetrahydro-1,4-diazocine'),
    ('N1C=CNCCCC1', '1,4,5,6,7,8-hexahydro-1,4-diazocine'),
]


@pytest.mark.parametrize('smiles,expected', WRONG_COUNT_REGRESSIONS)
def test_hydro_count_matches_the_structure(smiles, expected):
    assert _name(smiles) == expected


@pytest.mark.parametrize('smiles,_expected', WRONG_COUNT_REGRESSIONS)
def test_hydro_count_equals_double_bonds_lost(smiles, _expected):
    """hydro positions == 2 x (mancude parent double bonds - molecule's)."""
    from orthonym.rules.heterocycles import (
        _macrocycle_ordered_ring, _mancude_max_matching)

    mol = Chem.MolFromSmiles(smiles)
    ring_set = set(mol.GetRingInfo().AtomRings()[0])
    ordered = _macrocycle_ordered_ring(mol, ring_set)
    max_match, _ = _mancude_max_matching(mol, ordered)
    have = sum(1 for b in mol.GetBonds()
               if b.GetBeginAtomIdx() in ring_set and b.GetEndAtomIdx() in ring_set
               and b.GetBondTypeAsDouble() >= 2.0)
    hydro, _ih = _locant_sets(_name(smiles))
    assert len(hydro) == 2 * (max_match - have)


# --------------------------------------------------------------------------
# Rings of eleven and more members are named by skeletal replacement with
# 'ene' endings (P-22.2.3, :8482), never by hydro prefixes stacked on an
# already-ene-specified replacement parent.
# --------------------------------------------------------------------------
LARGE_RING_REPLACEMENT = [
    ('O1NC=CCCCCCCCC1', '1-oxa-2-azacyclododec-3-ene'),
    ('O1NCCCCCCCC=CC1', '1-oxa-2-azacyclododec-10-ene'),
]


@pytest.mark.parametrize('smiles,expected', LARGE_RING_REPLACEMENT)
def test_large_rings_use_replacement_with_ene_not_hydro(smiles, expected):
    got = _name(smiles)
    assert got == expected
    assert 'hydro' not in got
