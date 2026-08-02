"""P-22.2.7 lambda heteromonocycles: mancude parents with indicated hydrogen on
a CARBON, and hydro forms.

Task AA2.  Both classes used to abstain at producer level -- 707 rings out of
an enumeration of 27,687 bare heteromonocycles (sizes 3-14 x N/O/S/O+N/S+N/N+N
at every heteroatom position x every independent edge set of the ring), all 707
falling on a single ``is any saturated ring atom a carbon?`` test.  That one
code branch merged two different nomenclature classes; the Blue Book marks both
(PIN).

Every expected value here is derived from the cited rule FIRST and only then
compared with output.  Where a name is checkable by round-trip it has been
checked (OPSIN 2.9.0, 707/707 on the full class), but a round-trip cannot
choose between two numberings of the same molecule, so the locant-bearing
assertions carry their derivation in the docstring.
"""
import pytest
from rdkit import Chem

from orthonym.namer import name_compound
from orthonym.rules.heterocycles import (
    _apply_retained_stem,
    _mancude_bond_eligible,
    _MANCUDE_RETAINED_STEM,
    name_heterocycle,
    orient_heterocycle,
)


def _ring_name(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return name_heterocycle(mol, mol.GetRingInfo().AtomRings()[0])


# --------------------------------------------------------------------------
# 1. The Blue Book's own (PIN) examples for the class
# --------------------------------------------------------------------------

# P-22.2.7 "Heteromonocyclic hydrides having heteroatoms with nonstandard
# bonding numbers." (BlueBookV2.md:9156); P-22.2.7.1 at :9158.
BB_PIN_MANCUDE = [
    # 3H- puts the indicated hydrogen on a CARBON.  This is the case the old
    # guard refused, while emitting its 1H- companion perfectly well.
    ("[SH]1=CCC=C1", "3H-1lambda4-thiophene"),    # :9171
    ("[SH2]1C=CC=C1", "1H-1lambda4-thiophene"),   # :9167
    ("[SH2]1C=CC=CC=C1", "1H-1lambda4-thiepine"),  # :9496
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", BB_PIN_MANCUDE)
def test_mancude_lambda_parent_pin(smiles, expected):
    assert _ring_name(smiles) == expected
    assert name_compound(smiles) == expected


@pytest.mark.unit
def test_hydro_lambda_ring_matches_the_bluebook_parent_hydride():
    """The ring skeleton of ``3,4,5,6-tetrahydro-1lambda4,2-thiazin-1-ol
    (PIN)``, P-66.1.5.2.3 (BlueBookV2.md:33282), example at :33292.

    Stripping the 1-ol leaves the parent hydride, whose hydro locants, lambda
    citation and heteroatom locants must be spelled exactly as the Blue Book
    spells them in the substituted name.
    """
    assert _ring_name("[SH]1=NCCCC1") == "3,4,5,6-tetrahydro-1lambda4,2-thiazine"


# --------------------------------------------------------------------------
# 2. P-14.4(b): indicated hydrogen takes the LOWEST locant
# --------------------------------------------------------------------------

# The mancude lambda branch used to inherit orient_heterocycle's numbering,
# which ranks heteroatoms but not indicated hydrogen, so where the heteroatom
# criteria tied it chose arbitrarily.  5 of the 60 mancude lambda rings in the
# enumeration came out non-minimal.  Neither OPSIN nor an InChIKey can see
# this: the two numberings describe the same molecule, and the whole class
# round-tripped 707/707 while these were still wrong.
P14_4B_MINIMAL_IH = [
    ("C1=CC[SH]=C1", "2H-1lambda4-thiophene"),        # was 5H-
    ("C1=CCC=C[SH]=C1", "4H-1lambda4-thiepine"),      # was 5H-
    ("C1=CCC=[SH]C=C1", "3H-1lambda4-thiepine"),      # was 6H-
    ("C1=CC=CC[SH]=CC=C1", "2H-1lambda4-thionine"),   # was 9H-
    ("C1=CCC=CC=[SH]C=C1", "5H-1lambda4-thionine"),   # was 6H-
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", P14_4B_MINIMAL_IH)
def test_indicated_hydrogen_takes_lowest_locant(smiles, expected):
    # P-14.4 "NUMBERING" (BlueBookV2.md:3219): "low locants are assigned to
    # them in the following decreasing order of seniority ... (b) indicated
    # hydrogen for unsubstituted compounds" (:3246).
    assert _ring_name(smiles) == expected


# --------------------------------------------------------------------------
# 3. Retained stems -- whole stem only, never a suffix
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_retained_stem_is_whole_stem_not_suffix():
    """A suffix match would rewrite the Blue Book's own names.

    ``1lambda4,3-dithiole (PIN)`` is at BlueBookV2.md:9527 and must survive;
    an ``endswith('thiole')`` test -- which the lambda producer used to carry
    -- turns it into the non-existent '...dithiophene'.  ``1,3-oxazole``
    (:8134) would likewise become '1,3-furan'.
    """
    assert _apply_retained_stem("1,2-dithiole") == "1,2-dithiole"
    assert _apply_retained_stem("1,3-dithiole") == "1,3-dithiole"
    assert _apply_retained_stem("1,3-oxazole") == "1,3-oxazole"
    assert _apply_retained_stem("1,4-dioxine") == "1,4-dioxine"
    assert _apply_retained_stem("1lambda4,2-thiazine") == "1lambda4,2-thiazine"


@pytest.mark.unit
def test_retained_stem_applies_to_every_row_of_the_one_table():
    # The table had drifted into three partial hand-copies; these are the rows
    # the copies were missing.  P-22.2.1 Table 2.2: pyran/thiopyran/
    # selenopyran/telluropyran :8141, pyrrole :8163, pyridine :8157.
    assert _apply_retained_stem("azole") == "pyrrole"
    assert _apply_retained_stem("2H-azole") == "2H-pyrrole"
    assert _apply_retained_stem("1lambda4-thiine") == "1lambda4-thiopyran"
    for stem, retained in _MANCUDE_RETAINED_STEM.items():
        assert _apply_retained_stem(stem) == retained
        assert _apply_retained_stem("7-" + stem) == "7-" + retained


@pytest.mark.unit
def test_mancude_five_and_six_ring_stems_are_retained_names():
    # Regression locks for the three names the shared table changed.
    assert _ring_name("C1=CCN=C1") == "2H-pyrrole"
    assert _ring_name("C1=CN=CC1") == "3H-pyrrole"
    assert _ring_name("C1=CC=[SH]C=C1") == "1lambda4-thiopyran"


# --------------------------------------------------------------------------
# 4. Lambda-aware double-bond eligibility
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_lambda_sulfur_is_double_bond_eligible_but_standard_sulfur_is_not():
    """P-22.2.7.1 (:9158) + the 1lambda6-thiopyran note at :9513 -- "this
    heteromonocycle has the maximum number of double bonds and one double bond
    at every position; hence, no indicated hydrogen is cited for the sulfur".
    A divalent sulfur can hold no ring double bond; a lambda-4 sulfur can.
    """
    lam = Chem.MolFromSmiles("[SH]1=CCC=C1")     # lambda-4 S
    std = Chem.MolFromSmiles("c1ccsc1")          # thiophene, divalent S
    for mol, expect_s_eligible in ((lam, True), (std, False)):
        ordered, _ = orient_heterocycle(mol, mol.GetRingInfo().AtomRings()[0])
        elig = _mancude_bond_eligible(mol, ordered)
        s_pos = [p for p, i in enumerate(ordered)
                 if mol.GetAtomWithIdx(i).GetSymbol() == 'S']
        assert elig[s_pos[0]] is expect_s_eligible


@pytest.mark.unit
@pytest.mark.parametrize("smiles", [
    "c1ccsc1", "c1ccncc1", "C1CCOC1", "C1=CCNC1", "C1CCOCC1",
    "c1ccoc1", "C1=CCOC1", "C1CNCCN1", "c1cn[nH]c1",
])
def test_eligibility_unchanged_without_a_lambda_atom(smiles):
    """Outside the lambda class the helper must reproduce the old element test
    verbatim -- an atom whose bonding number is standard is judged by
    ``symbol not in {O,S,Se,Te}``, exactly as before.
    """
    mol = Chem.MolFromSmiles(smiles)
    ordered, _ = orient_heterocycle(mol, mol.GetRingInfo().AtomRings()[0])
    divalent = frozenset({'O', 'S', 'Se', 'Te'})
    old = [mol.GetAtomWithIdx(i).GetSymbol() not in divalent for i in ordered]
    assert _mancude_bond_eligible(mol, ordered) == old


# --------------------------------------------------------------------------
# 5. Fail-closed contract
# --------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles", [
    "C1=C[SH]=CS1",        # 2 S, one lambda-4
    "S1[SH]=CC=C1",
    "C1=C[SH]=CC=CS1",
    "[SH]1=CCCCS1",
])
def test_two_same_element_heteroatoms_still_fail_closed(smiles):
    """P-22.2.7.2 (BlueBookV2.md:9515) -- "If a further choice is needed
    between two or more of the same skeletal atom with different bonding
    numbers, the lower locant is assigned in order of the decreasing value of
    the bonding number" -- is not implemented, so these must refuse.  The new
    delegation to the hydro namer must not open a hole here, which is why the
    guard sits ahead of it AND is repeated inside _mancude_hydro_name.
    """
    assert _ring_name(smiles) is None


@pytest.mark.unit
def test_hydro_and_indicated_hydrogen_locants_are_disjoint():
    """An indicated-hydrogen atom holds no double bond in the parent, so it has
    none to lose and can never also be a hydro position.
    """
    import re
    for smiles in ("[SH2]1CCC=C1", "[SH]1=NCCCC1", "C1=[SH]CC1",
                   "C1=CCC[SH]=C1", "C1=CC=[SH]CCC=C1"):
        name = _ring_name(smiles)
        assert name, smiles
        hydro = re.match(r'^([\d,]+)-(?:di|tri|tetra|penta|hexa)hydro', name)
        ih = re.search(r'-((?:\d+H,)*\d+H)-', '-' + name)
        if hydro and ih:
            h = {int(x) for x in hydro.group(1).split(',')}
            i = {int(x) for x in re.findall(r'\d+', ih.group(1))}
            assert not (h & i), f"{smiles} -> {name}"


@pytest.mark.unit
def test_lambda_ring_never_loses_its_lambda():
    """The hydro namer must carry the lambda through to the stem.  Dropping it
    names a different molecule -- the standard-valence parent -- which is the
    failure mode the whole lambda branch exists to prevent.
    """
    for smiles in ("[SH2]1CCC=C1", "[SH]1=NCCCC1", "C1=[SH]CC1",
                   "C1=CCC[SH]=C1", "C1=CC=CCC[SH]=CC=C1"):
        name = _ring_name(smiles)
        assert name and "lambda4" in name, f"{smiles} -> {name}"
