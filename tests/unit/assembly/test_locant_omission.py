"""P-14.3.4 locant-omission licence primitives (v29 Phase C tranche B, Task 1).

``orthonym.assembly.locant_omission`` is the ONE place the Blue Book's omission
licences are decided. It is pure: RDKit mols in, booleans out.

Governing rule chain, verbatim from ``BlueBookV2/BlueBookV2.md``:

``P-14.3.3`` "Citation of locants" (``:2869``) is **DENY BY DEFAULT** —

    "In preferred IUPAC names, if any locants are essential for defining the structure
     of the parent structure or of a unit of structure as defined by its appropriate
     enclosing marks, then all locants must be cited for the parent structure or that
     structural unit."

``P-14.3.4`` then grants narrow licences. The three modelled here:

``P-14.3.4.5`` (``:3007``)

    "All locants are omitted in compounds or substituent groups in which all
     substitutable positions are completely substituted or modified, for example, by
     hydro, in the same way. Except for hydrogen atoms attached to chalcogen atoms,
     such as in acids, alcohols, and to the carbon atoms of formyl groups (aldehydes),
     all hydrogen atoms are considered substitutable."

with the counter-clause (``:3009``)

    "In case of partial substitution or modification, all numerical prefixes must be
     indicated. The prefix 'per-' is no longer recommended."

``P-14.3.4.3`` (``:2939``)

    "The locant is omitted in monosubstituted symmetrical parent hydrides or parent
     compounds where there is only one kind of substitutable hydrogen."

``P-14.3.4.6`` (``:3031``)

    "All locants are omitted for parent compounds when all substitutable hydrogen atoms
     have the same locant."   [example ``:3037`` ``difluoroacetic acid (PIN)``]

★ THE SELF-VALIDATING BOUNDARY PAIR — the reason these primitives count HYDROGENS and
not positions:

    ``:7625``   ``benzenehexol (PIN, P-63.1.2) (not benzenehexaol)``        -> OMITS
    ``:54823``  "Inositols, cyclohexane-1,2,3,4,5,6-hexols, are a specific
                 group of cyclitols."                                      -> RETAINS

Same six OH, same ring size. A benzene ring carbon has ONE substitutable H, so six OH
completely substitute it (L5 fires). A cyclohexane ring carbon has TWO, so six OH is
PARTIAL substitution and ``:3009`` restores every locant. A predicate that counts
positions gets this pair wrong; a predicate that counts hydrogens gets it right.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.locant_omission import (
    l3_one_kind_of_substitutable_h,
    l5_uniform_complete,
    l6_all_substitutable_h_share_one_locant,
    scope_forces_locants,
    substitutable_h_count,
    substitutable_positions,
)


def _mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, smiles
    return m


# --------------------------------------------------------------------------- #
# 1. substitutable_h_count -- the ':3007' carve-outs, verbatim                 #
# --------------------------------------------------------------------------- #
class TestSubstitutableHCount:
    def test_benzene_ring_carbon_has_one(self):
        m = _mol("c1ccccc1")
        assert [substitutable_h_count(m, i) for i in range(6)] == [1] * 6

    def test_cyclohexane_ring_carbon_has_two(self):
        """★ the other half of the boundary pair."""
        m = _mol("C1CCCCC1")
        assert [substitutable_h_count(m, i) for i in range(6)] == [2] * 6

    def test_methyl_carbon_has_three(self):
        m = _mol("CC")
        assert substitutable_h_count(m, 0) == 3

    @pytest.mark.parametrize("smiles,idx,element", [
        ("CO", 1, "O -- alcohol"),
        ("CC(=O)O", 3, "O -- acid"),
        ("CS", 1, "S -- thiol"),
        ("C[SeH]", 1, "Se"),
        ("C[TeH]", 1, "Te"),
    ])
    def test_chalcogen_h_is_not_substitutable(self, smiles, idx, element):
        """':3007' -- 'Except for hydrogen atoms attached to chalcogen atoms'."""
        m = _mol(smiles)
        assert m.GetAtomWithIdx(idx).GetTotalNumHs() >= 1, "witness has no H at all"
        assert substitutable_h_count(m, idx) == 0, element

    def test_aldehyde_formyl_h_is_not_substitutable(self):
        """':3007' -- 'and to the carbon atoms of formyl groups (aldehydes)'."""
        m = _mol("CC=O")           # acetaldehyde
        assert m.GetAtomWithIdx(1).GetTotalNumHs() == 1
        assert substitutable_h_count(m, 1) == 0
        assert substitutable_h_count(m, 0) == 3, "the methyl is still substitutable"

    def test_amide_nh_is_substitutable(self):
        """N is not a chalcogen -- 'methylurea' (:2943) proves urea N-H substitutable."""
        m = _mol("NC(N)=O")        # urea
        assert substitutable_h_count(m, 0) == 2
        assert substitutable_h_count(m, 2) == 2

    def test_carboxamide_carbon_is_not_a_formyl_carbon(self):
        """Derivation F1 finding 5: the carve-out is NARROW -- one H, one C=O, and no
        single-bonded O/N/S neighbour. An amide/acid carbon has such a neighbour."""
        m = _mol("CC(=O)N")
        assert substitutable_h_count(m, 1) == 0, "no H there anyway"
        # formic acid: carbon has 1 H, one C=O, AND a single-bonded O -> not formyl
        f = _mol("OC=O")
        assert f.GetAtomWithIdx(1).GetTotalNumHs() == 1
        assert substitutable_h_count(f, 1) == 1, (
            "formic-acid carbon must stay substitutable: the narrow carve-out "
            "requires NO single-bonded O/N/S neighbour"
        )

    def test_formaldehyde_carbon_is_not_carved_out(self):
        """Two H, so not a 'formyl group' under the narrow definition."""
        m = _mol("C=O")
        assert substitutable_h_count(m, 0) == 2

    def test_no_hydrogen_is_zero(self):
        m = _mol("Cc1c(C)c(C)c(C)c(C)c1C")   # ring carbons fully substituted
        ring = [a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic()]
        assert all(substitutable_h_count(m, i) == 0 for i in ring)


class TestSubstitutablePositions:
    def test_benzene_all_six(self):
        assert substitutable_positions(_mol("c1ccccc1")) == frozenset(range(6))

    def test_ethanol_excludes_the_hydroxy_oxygen(self):
        m = _mol("CCO")
        assert substitutable_positions(m) == frozenset({0, 1})

    def test_acetaldehyde_excludes_the_formyl_carbon(self):
        assert substitutable_positions(_mol("CC=O")) == frozenset({0})

    def test_returns_a_frozenset(self):
        assert isinstance(substitutable_positions(_mol("CC")), frozenset)


# --------------------------------------------------------------------------- #
# 2. l5_uniform_complete -- P-14.3.4.5 + the ':3009' counter-clause            #
# --------------------------------------------------------------------------- #
class TestL5UniformComplete:
    def test_benzene_six_identical_is_licensed(self):
        """':7625' benzenehexol (PIN)."""
        m = _mol("c1ccccc1")
        assert l5_uniform_complete(m, decoration_of={i: "ol" for i in range(6)}) is True

    def test_the_boundary_cyclohexane_six_ol_is_partial(self):
        """★ ':54823' cyclohexane-1,2,3,4,5,6-hexols RETAIN every locant.

        12 substitutable H, 6 decorated => partial => ':3009'. If this ever returns
        True the predicate is counting POSITIONS instead of HYDROGENS.
        """
        m = _mol("C1CCCCC1")
        assert l5_uniform_complete(m, decoration_of={i: "ol" for i in range(6)}) is False

    def test_cyclohexane_is_licensed_only_when_all_twelve_h_go(self):
        m = _mol("C1CCCCC1")
        assert l5_uniform_complete(
            m, decoration_of={i: "F" for i in range(6)},
            counts={i: 2 for i in range(6)},
        ) is True

    def test_partial_substitution_denies(self):
        m = _mol("c1ccccc1")
        assert l5_uniform_complete(m, decoration_of={i: "ol" for i in range(5)}) is False

    def test_heterogeneous_complete_substitution_denies(self):
        """':3007' says 'in the same way'. Complete but mixed => keep all locants."""
        m = _mol("c1ccccc1")
        dec = {i: "methyl" for i in range(5)}
        dec[5] = "chloro"
        assert l5_uniform_complete(m, decoration_of=dec) is False

    def test_complete_hydro_modification_is_licensed(self):
        """':3013' decahydronaphthalene (PIN) -- 'modified, for example, by hydro'."""
        m = _mol("c1ccc2ccccc2c1")
        dec = {i: "hydro" for i in range(m.GetNumAtoms())}
        assert l5_uniform_complete(m, decoration_of=dec) is True

    def test_partial_hydro_denies(self):
        """1,2,3,4-tetrahydronaphthalene keeps its locants."""
        m = _mol("c1ccc2ccccc2c1")
        subs = sorted(substitutable_positions(m))
        dec = {i: "hydro" for i in subs[:4]}
        assert l5_uniform_complete(m, decoration_of=dec) is False

    @pytest.mark.parametrize("dec", [{}, None])
    def test_no_decoration_denies(self, dec):
        assert l5_uniform_complete(_mol("c1ccccc1"), decoration_of=dec) is False

    def test_no_substitutable_position_denies(self):
        """Fail toward retaining: nothing to 'completely substitute'."""
        m = _mol("ClC(Cl)(Cl)Cl")
        assert substitutable_positions(m) == frozenset()
        assert l5_uniform_complete(m, decoration_of={0: "chloro"}) is False

    def test_none_mol_denies(self):
        assert l5_uniform_complete(None, decoration_of={0: "ol"}) is False

    def test_out_of_range_index_denies(self):
        m = _mol("c1ccccc1")
        dec = {i: "ol" for i in range(6)}
        dec[99] = "ol"
        assert l5_uniform_complete(m, decoration_of=dec) is False

    def test_empty_decoration_kind_denies(self):
        m = _mol("c1ccccc1")
        assert l5_uniform_complete(
            m, decoration_of={i: "" for i in range(6)}) is False
        assert l5_uniform_complete(
            m, decoration_of={i: None for i in range(6)}) is False


# --------------------------------------------------------------------------- #
# 3. l3_one_kind_of_substitutable_h -- P-14.3.4.3 (':2939')                    #
# --------------------------------------------------------------------------- #
class TestL3OneKind:
    @pytest.mark.parametrize("smiles,expected,why", [
        ("c1ccccc1", True, "benzene: one orbit"),
        ("C1CCCCC1", True, "cyclohexane: one orbit"),
        ("NC(N)=O", True, "urea -- ':2943' methylurea (PIN)"),
        ("c1cnccn1", True, "pyrazine -- ':2949' pyrazinecarboxylic acid (PIN)"),
        ("CCO", False, "ethanol: CH3 and CH2 are different kinds"),
        ("Cc1ccccc1", False, "toluene: methyl + 3 ring orbits"),
        ("c1ccc2ccccc2c1", False, "naphthalene: alpha and beta differ"),
        ("CCC", False, "propane: CH3 vs CH2"),
    ])
    def test_one_kind(self, smiles, expected, why):
        assert l3_one_kind_of_substitutable_h(_mol(smiles)) is expected, why

    def test_no_substitutable_h_denies(self):
        assert l3_one_kind_of_substitutable_h(_mol("ClC(Cl)(Cl)Cl")) is False

    def test_none_mol_denies(self):
        assert l3_one_kind_of_substitutable_h(None) is False


# --------------------------------------------------------------------------- #
# 4. l6_all_substitutable_h_share_one_locant -- P-14.3.4.6 (':3031')           #
# --------------------------------------------------------------------------- #
class TestL6OneLocant:
    def test_acetic_acid_is_licensed(self):
        """':3037' difluoroacetic acid (PIN) (not 2,2-difluoroacetic acid).

        Acetic acid's only substitutable H are the three on C-2 (the acid OH is a
        chalcogen H, excluded by ':3007').
        """
        m = _mol("CC(=O)O")                      # idx0 = CH3, idx1 = C, idx3 = OH
        assert l6_all_substitutable_h_share_one_locant(m, {0: 2, 1: 1, 3: 1}) is True

    def test_propanoic_acid_is_not_licensed(self):
        m = _mol("CCC(=O)O")                     # idx0 = C3, idx1 = C2
        assert l6_all_substitutable_h_share_one_locant(
            m, {0: 3, 1: 2, 2: 1, 4: 1}) is False

    def test_missing_locant_denies(self):
        """Fail-closed: an unmapped substitutable atom cannot be cleared."""
        m = _mol("CC(=O)O")
        assert l6_all_substitutable_h_share_one_locant(m, {1: 1}) is False

    def test_no_substitutable_h_denies(self):
        assert l6_all_substitutable_h_share_one_locant(
            _mol("ClC(Cl)(Cl)Cl"), {0: 1}) is False

    @pytest.mark.parametrize("mol_arg,loc", [(None, {0: 1}), ("CC", None)])
    def test_none_input_denies(self, mol_arg, loc):
        m = None if mol_arg is None else _mol(mol_arg)
        assert l6_all_substitutable_h_share_one_locant(m, loc) is False


# --------------------------------------------------------------------------- #
# 5. scope_forces_locants -- P-14.3.3 (':2869'), the deny-default itself       #
# --------------------------------------------------------------------------- #
_CLEAN = dict(
    prefix_locants=(), suffix_locants=(), stereo_text="",
    has_indicated_h=False, has_isotope=False, is_multiplicative=False,
    is_ring_assembly=False, has_skeletal_replacement=False,
)


class TestScopeForcesLocants:
    def test_clean_scope_does_not_force(self):
        assert scope_forces_locants(**_CLEAN) is False

    def test_numeric_locants_alone_do_not_force(self):
        """The numeric locants are exactly what a licence may omit."""
        assert scope_forces_locants(
            **{**_CLEAN, "prefix_locants": (1, 2), "suffix_locants": (1,)}) is False

    @pytest.mark.parametrize("flag", [
        "has_indicated_h", "has_isotope", "is_multiplicative",
        "is_ring_assembly", "has_skeletal_replacement",
    ])
    def test_each_hard_override_forces(self, flag):
        """Global constraint 10 -- the hard overrides beat every licence."""
        assert scope_forces_locants(**{**_CLEAN, flag: True}) is True

    def test_stereodescriptor_forces(self):
        assert scope_forces_locants(**{**_CLEAN, "stereo_text": "(1R,2S)-"}) is True

    @pytest.mark.parametrize("loc", ["N", "N1", "1'", "O"])
    def test_letter_locant_forces(self, loc):
        """A letter locant is essential and cannot be omitted -- 'N-methylurea'."""
        assert scope_forces_locants(**{**_CLEAN, "prefix_locants": (loc,)}) is True
        assert scope_forces_locants(**{**_CLEAN, "suffix_locants": (loc,)}) is True

    @pytest.mark.parametrize("field", sorted(_CLEAN))
    def test_none_input_forces(self, field):
        """Verbatim from the brief: 'unknown/None inputs must return True'."""
        assert scope_forces_locants(**{**_CLEAN, field: None}) is True
