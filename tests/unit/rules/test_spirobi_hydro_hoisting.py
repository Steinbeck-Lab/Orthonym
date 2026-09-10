""" -- hydro / indicated-hydrogen hoisting for 'spirobi' components.

This is the build that unblocked the indane / indoline / isoindoline rename
, ``the Blue Book``). ``data/fused_heterocycles.py``'s ``name`` field feeds
TWO consumers: standalone naming, and ``rules/spiro.py::_name_spirobi_core``,
which used to embed that string verbatim as the spiro component. Renaming the row
alone therefore produced ``1,2'-spirobi[2,3-dihydro-1H-indene]``, and the obvious
shortcut ``1,2'-spirobi[1H-indene]`` denotes the UNSATURATED molecule -- a wrong
STRUCTURE, not a wrong spelling. The gate rejected the first attempt.

**** (``the Blue Book``), verbatim:

    Where appropriate the maximum number of noncumulative double bonds is added
    (i.e., the system is made mancude) AFTER CONSTRUCTION OF THE COMPLETE
    SKELETON. Indicated hydrogen of individual components is not cited.
    No indicated hydrogen is cited when none is present in the spiro system. If
    indicated hydrogen is needed, it is cited in front of the spiro atom locants.

Two consequences are asserted here:

  1. The bracket holds the MANCUDE component; saturation is hoisted outside it.
     Template ``the Blue Book``:
     ``1,3'-dihydro-3H-1lambda6,1'-spirobi[[2,1]benzoxathiole] (PIN)``.
     Every one of the ~20 'spirobi' examples in the Blue Book cites indicated
     hydrogen OUTSIDE the bracket; not one cites it inside.

  2. Because the skeleton is made mancude as a WHOLE, the hydro locants are NOT
     the component's own. For ``1,2'-spirobi[indene]`` the spiro atom is sp3 by
     construction and the remaining eight atoms of each half admit a PERFECT
     matching, so no indicated hydrogen is needed and the hydro positions are
     2,3 (unprimed) and 1',3' (primed) -- NOT 2,3,2',3'.

Locant order is **** (``the Blue Book``): *"Primed locants are placed
immediately after the corresponding unprimed locants in a set arranged in
ascending order"*, i.e. ``1 < 1' < 2 < 2' < 3 < 3'`` -- NOT every unprimed before
every primed. Confirmed inside the spirobi section itself by ``the Blue Book``
``2-phospha-3,3'-spirobi[bicyclo[3.3.1]nonane]-6',7-diene (PIN)``, which cites
``6'`` BEFORE ``7``, and by ``the Blue Book`` ``1'H,2H-1,2'-spirobi[azulene] (PIN)``.

★ Every assertion is on the WHOLE emitted name (session a project rule): removing a
wrong output can unmask a worse generator, so a substring check would not show
that the hydro prefix landed in the right place.
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.rules.spiro import (
    _can_bear_ring_double_bond,
    _is_saturated_in_component,
    _mancude_max_matching,
    _spirobi_locant_display,
    _spirobi_locant_key,
    _spirobi_component_saturation,
)


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# --------------------------------------------------------------------------
# The two molecules the rename regressed, as WHOLE names.
# --------------------------------------------------------------------------

class TestSpirobiHydroHoisting:
    """The gold rows DD7-spiro-1 and P13-SPIROBI-02."""

    @pytest.mark.parametrize("smiles,expected", [
        # DD7-spiro-1: ASYMMETRIC -- spiro atom at indene position 1 of one half
        # and 2' of the other, so hydro at 2,3 (unprimed) and 1',3' (primed).
        ("C1Cc2ccccc2C13Cc1ccccc1C3",
         "1',2,3,3'-tetrahydro-1,2'-spirobi[indene]"),
        # P13-SPIROBI-02: symmetric, spiro at position 2 of both halves.
        ("c1ccc2c(c1)CC1(C2)Cc2ccccc2C1",
         "1,1',3,3'-tetrahydro-2,2'-spirobi[indene]"),
    ])
    def test_whole_name(self, namer, smiles, expected):
        assert namer.name(smiles) == expected

    @pytest.mark.parametrize("smiles", [
        "C1Cc2ccccc2C13Cc1ccccc1C3",
        "c1ccc2c(c1)CC1(C2)Cc2ccccc2C1",
    ])
    def test_no_saturation_inside_the_bracket(self, namer, smiles):
        """/: the bracket holds the MANCUDE ring system only."""
        got = namer.name(smiles)
        bracket = got[got.index("[") + 1:got.rindex("]")]
        assert bracket == "indene", f"bracket must be mancude, got {bracket!r} in {got!r}"
        assert "hydro" not in bracket, got
        assert "H-" not in bracket, got

    @pytest.mark.parametrize("smiles", [
        "C1Cc2ccccc2C13Cc1ccccc1C3",
        "c1ccc2c(c1)CC1(C2)Cc2ccccc2C1",
    ])
    def test_name_denotes_the_SATURATED_molecule(self, namer, smiles):
        """★ The specific hazard: a correctly-spelled name for the wrong compound.

        ``1,2'-spirobi[1H-indene]`` -- the shortcut -- is the unsaturated molecule.
        Round-trip the emitted name through OPSIN and require the ORIGINAL
        structure back, so a wrong-structure spelling cannot pass this file.
        """
        from orthonym.validation.opsin_roundtrip import opsin_parse
        got = namer.name(smiles)
        opsin_smiles = opsin_parse(got)
        assert opsin_smiles, f"OPSIN could not parse {got!r}"
        assert Chem.CanonSmiles(opsin_smiles) == Chem.CanonSmiles(smiles), (
            f"{got!r} denotes a DIFFERENT molecule: "
            f"{Chem.CanonSmiles(opsin_smiles)} != {Chem.CanonSmiles(smiles)}")

    def test_hydro_locant_order_is_p14_3_5_not_unprimed_first(self, namer):
        """``1',2,3,3'`` and not ``2,3,1',3'``.

         (``the Blue Book``) interleaves: 1 < 1' < 2 < 2' < 3 < 3'. The
        superseded gold value used the unprimed-first ordering, which the Blue Book
        (``-6',7-diene``) contradicts.
        """
        got = namer.name("C1Cc2ccccc2C13Cc1ccccc1C3")
        locants = got.split("-tetrahydro")[0]
        assert locants == "1',2,3,3'", got
        assert locants != "2,3,1',3'", "unprimed-first ordering violates P-14.3.5"


class TestFullyMancudeComponentsUnchanged:
    """The mancude spirobi names must stay byte-identical -- the hoisting path is
    gated on a component atom that COULD bear a ring double bond but does not, so
    a divalent ring O (which never can) must not drag a molecule onto it."""

    @pytest.mark.parametrize("smiles,expected", [
        # Component already mancude, spiro atom AT the indicated-H position.
        ("C1=Cc2ccccc2C13C=Cc1ccccc13", "1,1'-spirobi[indene]"),
        # Divalent ring O atoms are sp3 but can never be doubly bonded: they must
        # attract neither a hydro prefix nor indicated hydrogen.
        ("O1S2(OC3=C1C=CC=C3)OC3=C(O2)C=CC=C3",
         "2λ4,2'-spirobi[[1,3,2]benzodioxathiole]"),
        ("c1ccc2c(c1)OS1(O2)Oc2ccccc2O1",
         "2λ4,2'-spirobi[[1,3,2]benzodioxathiole]"),
        # Owned by _name_spiro_vonbaeyer_core, not the spirobi core -- pinned so a
        # change of dispatch shows up here.
        ("c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc21", "9,9'-spirobi[fluorene]"),
        ("C1CC2CC1CC21CC2CCC1C2", "2,2'-spirobi[bicyclo[2.2.1]heptane]"),
        ("C1CC2CCC1C21C2CCC1CC2", "7,7'-spirobi[bicyclo[2.2.1]heptane]"),
        ("C1CC2CCC1CC21CC2CCC1CC2", "2,2'-spirobi[bicyclo[2.2.2]octane]"),
        ("C12CC3(CC(C=CC1)C2)CC2CC=CC(C3)C2",
         "3,3'-spirobi[bicyclo[3.3.1]nonane]-6,6'-diene"),
    ])
    def test_unchanged(self, namer, smiles, expected):
        assert namer.name(smiles) == expected


# --------------------------------------------------------------------------
# The primitives, validated against Blue Book spirobi PINs.
# --------------------------------------------------------------------------

class TestLocantOrderP1435:
    """``_spirobi_locant_key`` implements (``the Blue Book``) exactly."""

    def test_primed_sorts_immediately_after_its_own_unprimed(self):
        order = [_spirobi_locant_key(1, 0), _spirobi_locant_key(1, 1),
                 _spirobi_locant_key(2, 0), _spirobi_locant_key(2, 1),
                 _spirobi_locant_key(3, 0), _spirobi_locant_key(3, 1)]
        assert order == sorted(order)

    def test_primed_one_sorts_BEFORE_unprimed_two(self):
        """The discriminating case: the Blue Book cites ``1'H`` before ``2H``, and
        the Blue Book cites ``6'`` before ``7``."""
        assert _spirobi_locant_key(1, 1) < _spirobi_locant_key(2, 0)

    def test_lettered_fusion_locant_sorts_after_its_number(self):
        """: ``4a`` and ``4'a`` follow the plain ``4``, before ``5``."""
        assert (_spirobi_locant_key(4, 0) < _spirobi_locant_key(4, 1)
                < _spirobi_locant_key('4a', 0) < _spirobi_locant_key('4a', 1)
                < _spirobi_locant_key(5, 0))

    def test_prime_is_rendered_after_the_NUMBER_not_the_whole_locant(self):
        """ spells the primed fusion locant ``4'a`` and states
        explicitly ``(not 4a')``."""
        assert _spirobi_locant_display(4, 1) == "4'"
        assert _spirobi_locant_display('4a', 1) == "4'a"
        assert _spirobi_locant_display('4a', 1) != "4a'"
        assert _spirobi_locant_display('4a', 0) == "4a"
        assert _spirobi_locant_display(3, 0) == "3"


class TestDoubleBondEligibility:
    """A mancude system never doubly bonds an atom with no spare valence."""

    @pytest.mark.parametrize("smiles,symbol,expected", [
        # thiophene S / furan O: two ring bonds, valence 2 -> no spare
        ("c1ccsc1", "S", False),
        ("c1ccoc1", "O", False),
        # pyrrole N: two ring bonds, valence 3 -> spare (1H-pyrrole)
        ("c1cc[nH]c1", "N", True),
        # ring carbon with two ring bonds -> spare
        ("c1ccccc1", "C", True),
    ])
    def test_spare_valence(self, smiles, symbol, expected):
        mol = Chem.MolFromSmiles(smiles)
        ring = set(mol.GetRingInfo().AtomRings()[0])
        idx = next(i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == symbol)
        assert _can_bear_ring_double_bond(mol, idx, ring) is expected

    def test_fusion_carbon_with_three_ring_bonds_still_has_spare_valence(self):
        """Carbon valence 4, three ring bonds -> one double bond available; this
        is what lets the benzo ring of indene stay aromatic."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")   # naphthalene
        ring = set().union(*[set(r) for r in mol.GetRingInfo().AtomRings()])
        fusion = [i for i in ring if sum(
            1 for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in ring) == 3]
        assert len(fusion) == 2
        for idx in fusion:
            assert _can_bear_ring_double_bond(mol, idx, ring) is True


class TestMancudeMatching:
    """``_mancude_max_matching`` maximises pairs, then minimises the unmatched
    locant set."""

    def test_even_path_is_perfectly_matched(self):
        adj = {1: [2], 2: [1, 3], 3: [2, 4], 4: [3]}
        key = {n: _spirobi_locant_key(n) for n in adj}
        pairs, unmatched = _mancude_max_matching(adj, list(adj), key)
        assert len(pairs) == 2 and unmatched == set()

    def test_odd_path_leaves_the_LOWEST_locant_unmatched(self):
        adj = {1: [2], 2: [1, 3], 3: [2]}
        key = {n: _spirobi_locant_key(n) for n in adj}
        pairs, unmatched = _mancude_max_matching(adj, list(adj), key)
        assert len(pairs) == 1
        assert unmatched == {1}, "the indicated-H position must take the low locant"

    def test_isolated_node_is_forced_unmatched(self):
        """the Blue Book ``2'H,3H-2,3'-spirobi[[1]benzothiophene]``: with the spiro
        atom at 3', C2' has only the divalent S left as a neighbour, so it is
        isolated in the eligible subgraph and MUST carry the indicated hydrogen."""
        adj = {2: [], 4: [5], 5: [4]}
        key = {n: _spirobi_locant_key(n) for n in adj}
        pairs, unmatched = _mancude_max_matching(adj, list(adj), key)
        assert len(pairs) == 1 and unmatched == {2}

    def test_result_is_independent_of_node_iteration_order(self):
        adj = {1: [2], 2: [1, 3], 3: [2, 4], 4: [3]}
        key = {n: _spirobi_locant_key(n) for n in adj}
        a = _mancude_max_matching(adj, [1, 2, 3, 4], key)
        b = _mancude_max_matching(adj, [4, 3, 2, 1], key)
        assert sorted(map(sorted, a[0])) == sorted(map(sorted, b[0]))
        assert a[1] == b[1]


class TestComponentSaturationAgainstBlueBookPINs:
    """★ The derivation, validated on Blue Book spirobi PINs BEFORE it is trusted.

    ``_spirobi_component_saturation`` reports ``(hydro, indicated_h)`` for ONE
    half. Each case below is read off a printed (PIN) in the Blue Book, so a
    change in the algorithm that breaks the rule breaks this test.
    """

    # Fusion locants cannot be written as an atom map number, so they travel as
    # 31 -> '3a', 41 -> '4a', 71 -> '7a', 81 -> '8a'.
    _FUSION = {31: '3a', 41: '4a', 71: '7a', 81: '8a'}

    def _report(self, mapped_smiles, spiro_locant):
        """Report ``(hydro, indicated_h)`` for ONE spirobi half.

        The locants travel as ATOM MAP NUMBERS in the SMILES, so the numbering
        cannot silently misalign with RDKit's atom order -- an earlier version of
        this harness hand-wrote index->locant dicts and fed a fully MANCUDE
        fragment where the real spiro half is partly saturated, which made the
        fail-closed guard fire and looked like a code bug.

        ``mapped_smiles`` must be the component AS IT APPEARS IN THE SPIRO
        SYSTEM: the spiro atom and any indicated-hydrogen position are sp3.
        """
        mol = Chem.MolFromSmiles(mapped_smiles)
        assert mol is not None, mapped_smiles
        comp = set().union(*[set(r) for r in mol.GetRingInfo().AtomRings()])
        numbering = {}
        for atom in mol.GetAtoms():
            num = atom.GetAtomMapNum()
            assert num, f"every atom needs a locant map: idx {atom.GetIdx()}"
            numbering[atom.GetIdx()] = self._FUSION.get(num, num)
        assert set(numbering) == comp, "all atoms must be ring atoms"
        assert len(set(map(str, numbering.values()))) == len(numbering), \
            "locants must be unique"
        spiro = next(i for i, loc in numbering.items() if loc == spiro_locant)
        return _spirobi_component_saturation(mol, comp, spiro, numbering)

    def test_naphthalene_spiro_at_2_needs_one_indicated_hydrogen(self):
        """the Blue Book ``1H,1'H-2,2'-spirobi[naphthalene] (PIN)``.

        The half is a naphthalene skeleton whose C2 is the spiro atom. Nine
        non-spiro atoms is ODD, so one cannot be matched and the lowest locant
        takes the indicated hydrogen -> ``1H``. No hydro: the spiro system IS the
        mancude system, so nothing was added.
        """
        # 1,2-dihydronaphthalene = the naphthalene half with C1 and C2 both sp3.
        half = ("[CH2:1]1[CH2:2][CH:3]=[CH:4][c:41]2[cH:5][cH:6][cH:7]"
                "[cH:8][c:81]12")
        report = self._report(half, 2)
        assert report is not None
        hydro, indicated = report
        assert indicated == {1}, f"BB:10158 cites 1H: got {indicated}"
        assert hydro == set(), f"nothing was added, so no hydro: {hydro}"

    def test_indene_spiro_at_1_needs_neither(self):
        """the Blue Book ``1,1'-spirobi[indene] (PIN)`` -- no hydro, no indicated H,
        because the spiro atom itself occupies indene's only sp3 position."""
        half = ("[CH2:1]1[CH:2]=[CH:3][c:31]2[cH:4][cH:5][cH:6][cH:7]"
                "[c:71]12")
        assert self._report(half, 1) == (set(), set())

    def test_indane_spiro_at_1_gives_hydro_2_3(self):
        """The unprimed half of DD7-spiro-1: hydro at 2,3 and NO indicated H,
        because the eight non-spiro atoms admit a perfect matching."""
        half = ("[CH2:1]1[CH2:2][CH2:3][c:31]2[cH:4][cH:5][cH:6][cH:7]"
                "[c:71]12")
        report = self._report(half, 1)
        assert report is not None
        hydro, indicated = report
        assert hydro == {2, 3}, hydro
        assert indicated == set(), (
            "a perfect matching exists, so P-24.3.2 cites NO indicated hydrogen")

    def test_indane_spiro_at_2_gives_hydro_1_3_not_2_3(self):
        """★ The primed half of DD7-spiro-1 -- the exact case the superseded gold
        value got wrong by reusing the component's OWN ``2,3-dihydro`` pattern
        instead of re-deriving against the assembled mancude skeleton."""
        half = ("[CH2:1]1[CH2:2][CH2:3][c:31]2[cH:4][cH:5][cH:6][cH:7]"
                "[c:71]12")
        report = self._report(half, 2)
        assert report is not None
        hydro, indicated = report
        assert hydro == {1, 3}, f"must be 1,3 -- NOT the component's own 2,3: {hydro}"
        assert indicated == set(), indicated

    def test_benzothiophene_spiro_at_2_puts_indicated_h_at_3_not_on_sulfur(self):
        """the Blue Book ``3H,3'H-2,2'-spirobi[[1]benzothiophene] (PIN)``.

        The divalent ring S is ineligible for a double bond, leaving SEVEN
        eligible carbons (odd), so one indicated hydrogen is required and the
        lowest eligible locant, 3, takes it. It must NOT land on the sulfur, and
        the sulfur must not attract a hydro prefix either.
        """
        # 2,3-dihydro-1-benzothiophene = the half with C2 (spiro) and C3 sp3.
        half = ("[S:1]1[CH2:2][CH2:3][c:31]2[cH:4][cH:5][cH:6][cH:7]"
                "[c:71]12")
        report = self._report(half, 2)
        assert report is not None
        hydro, indicated = report
        assert indicated == {3}, f"BB:10160 cites 3H: got {indicated}"
        assert 1 not in indicated, "the divalent S can never carry indicated hydrogen"
        assert hydro == set(), f"the sulfur must not attract a hydro prefix: {hydro}"


class TestFailClosed:
    """The build must decline rather than guess.

    ⚠ Mutation-testing note, recorded so the coverage here is not over-read:
    deleting the ``unmatched <= real_sp3`` guard alone SURVIVES, because the hydro
    parity check refuses the same witness. The DOUBLE deletion is killed by
    ``test_declines_when_indicated_h_would_land_on_an_unsaturated_atom`` below. See
    the comment at that guard in ``spiro.py`` for why an isolating witness does not
    appear to exist.
    """

    def test_charged_skeleton_atom_is_not_eligible(self):
        """Valence bookkeeping for charged skeletons is not derived, so the
        predicate must never over-report eligibility."""
        mol = Chem.MolFromSmiles("c1cc[n+](C)cc1")
        ring = set(mol.GetRingInfo().AtomRings()[0])
        n = next(i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == 'N')
        assert _can_bear_ring_double_bond(mol, n, ring) is False

    def test_declines_when_indicated_h_would_land_on_an_unsaturated_atom(self):
        """★ A real witness for the fail-closed guard, not an escape hatch.

        This component is a naphthalene skeleton whose spiro atom is C2 and whose
        only other sp3 atom is C3. Nine eligible non-spiro atoms is ODD, so the
        mancude form must leave one unmatched, and the lowest-locant choice is C1
        -- but C1 is NOT sp3 here. Naming it would require the joint
        indicated-H/hydro locant optimisation (the correct answer being a ``3H``
        form), which is deliberately NOT derived, so the report must DECLINE
        rather than emit a name for a structure that does not exist.

        Locants travel as atom map numbers; 41 -> '4a', 81 -> '8a'.
        """
        mapped = ("[CH:1]1=[C:81]2[CH:8]=[CH:7][CH:6]=[CH:5][C:41]2=[CH:4]"
                  "[CH2:3][CH2:2]1")
        mol = Chem.MolFromSmiles(mapped)
        assert mol is not None
        fusion = {41: '4a', 81: '8a'}
        comp = set().union(*[set(r) for r in mol.GetRingInfo().AtomRings()])
        numbering = {a.GetIdx(): fusion.get(a.GetAtomMapNum(), a.GetAtomMapNum())
                     for a in mol.GetAtoms()}
        assert set(numbering) == comp
        spiro = next(i for i, loc in numbering.items() if loc == 2)
        # Precondition of the witness: C1 really is unsaturated, C3 really is sp3.
        c1 = next(i for i, loc in numbering.items() if loc == 1)
        c3 = next(i for i, loc in numbering.items() if loc == 3)
        assert not _is_saturated_in_component(mol, c1, comp), "witness broken: C1"
        assert _is_saturated_in_component(mol, c3, comp), "witness broken: C3"
        assert _spirobi_component_saturation(mol, comp, spiro, numbering) is None
