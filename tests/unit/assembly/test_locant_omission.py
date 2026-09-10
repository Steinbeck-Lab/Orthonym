"""P-14.3.4 locant-omission licence primitives (Phase C tranche B, Task 1).

``orthonym.assembly.locant_omission`` is the ONE place the Blue Book's omission
licences are decided. It is pure: RDKit mols in, booleans out.

Governing rule chain, verbatim from ``the Blue Book Blue Book``:

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
     have the same locant." [example ``:3037`` ``difluoroacetic acid (PIN)``]

★ THE SELF-VALIDATING BOUNDARY PAIR — the reason these primitives count HYDROGENS and
not positions:

    ``:7625`` ``benzenehexol (PIN, P-63.1.2) (not benzenehexaol)`` -> OMITS
    ``:54823`` "Inositols, cyclohexane-1,2,3,4,5,6-hexols, are a specific
                 group of cyclitols." -> RETAINS

Same six OH, same ring size. A benzene ring carbon has ONE substitutable H, so six OH
completely substitute it (L5 fires). A cyclohexane ring carbon has TWO, so six OH is
PARTIAL substitution and ``:3009`` restores every locant. A predicate that counts
positions gets this pair wrong; a predicate that counts hydrogens gets it right.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.locant_omission import (
    forced_locant_scope,
    isotopic_naming_scope,
    l3_one_kind_of_substitutable_h,
    l4_no_isomer_by_relocation,
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
# 1. substitutable_h_count -- the ':3007' carve-outs, verbatim #
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
# 2. l5_uniform_complete -- P-14.3.4.5 + the ':3009' counter-clause #
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
# 3. l3_one_kind_of_substitutable_h -- P-14.3.4.3 (':2939') #
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
# 4. l6_all_substitutable_h_share_one_locant -- P-14.3.4.6 (':3031') #
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
# 5. scope_forces_locants -- P-14.3.3 (':2869'), the deny-default itself #
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
        """A letter locant in scope is essential and cannot be omitted by this
        generic path -- 'N-methylthiourea', 'N,N'-dimethylurea'. (The MONO urea
        omission is a separate composer-level rule, P-14.3.4.3, that never lets a
        letter locant reach this scope.)"""
        assert scope_forces_locants(**{**_CLEAN, "prefix_locants": (loc,)}) is True
        assert scope_forces_locants(**{**_CLEAN, "suffix_locants": (loc,)}) is True

    @pytest.mark.parametrize("field", sorted(_CLEAN))
    def test_none_input_forces(self, field):
        """Verbatim from the brief: 'unknown/None inputs must return True'."""
        assert scope_forces_locants(**{**_CLEAN, field: None}) is True


# --------------------------------------------------------------------------- #
# 6. l4_no_isomer_by_relocation -- P-14.3.4.4 (':2953'), the ISOMER-COUNT #
# licence. Phase C Task 11. #
# --------------------------------------------------------------------------- #
# §**P-14.3.4.4** (``:2953``), verbatim:
#
# "Locants are omitted when no isomer can be generated by moving suffixes
# and/or prefixes (if any) from their position to another or by interchanging
# them between two different positions."
#
# Every row below is an example the Blue Book PRINTS for this rule (or, for the
# polysulfanes, in §**P-68.4.1.1** "Compounds with three or more contiguous
# identical chalcogen atoms are treated as parent hydrides in substitutive
# nomenclature", whose whole example block BB 39333-39343 is locant-free).
#
# ★ The two negatives at BB 2995 / BB 2999 are the rows that make this a real
# test rather than a symmetry check: BB 2995 needs INTERCHANGE of two different
# prefixes (a single-orbit test omits there and is wrong), and BB 2999's stated
# reason is *"another isomer is generated by moving the Cl atom to the other Si
# atom"* -- CO-LOCATION of both prefixes on ONE position, which a test that
# assigns at most one decoration per atom never enumerates.

def _l4(smiles, parent_symbols, n_locants=1, **over):
    """Run the licence on ``smiles`` with the parent taken as every atom whose
    element is in ``parent_symbols`` (a ``*`` dummy stands for a free valence)."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit rejected {smiles!r}"
    parent = [a.GetIdx() for a in mol.GetAtoms()
              if a.GetSymbol() in parent_symbols]
    assert parent, f"no parent atoms matched {parent_symbols!r} in {smiles!r}"
    kw = dict(prefix_locants=list(range(1, n_locants + 1)), suffix_locants=[],
              stereo_text="", has_indicated_h=False, has_isotope=False)
    kw.update(over)
    return l4_no_isomer_by_relocation(mol, parent, **kw)


class TestL4BlueBookPositives:
    """Rows the Blue Book prints WITHOUT locants."""

    @pytest.mark.parametrize("smiles,parent,n,bb", [
        ("CSSS", {"S"}, 1, ":39335 CH3-S-S-SH methyltrisulfane (PIN)"),
        ("CSSSS", {"S"}, 1, "homologue -- methyltetrasulfane"),
        ("CSSSC", {"S"}, 2, ":39339 CH3-S-S-S-CH3 dimethyltrisulfane (PIN)"),
        ("COOOC", {"O"}, 2, ":39337 dimethyltrioxidane (PIN)"),
        ("C[Se][Se][Se]c1ccccc1", {"Se"}, 2,
         ":39341 methyl(phenyl)triselane (PIN) -- HETEROGENEOUS interchange"),
        ("CSSBr", {"S"}, 2, ":2979 (not 1-bromo-2-methyldisulfane)"),
        ("CC=NN*", {"N", "*"}, 1,
         ":2959 ethylidenehydrazinyl (not 2-ethylidenehydrazin-1-yl)"),
        ("C#[Si][Si]#CC", {"Si"}, 2, ":2963 ethylidyne(methylidyne)disilane"),
        ("c1ccccc1C=[SiH][Si](=Cc1ccccc1)*", {"Si", "*"}, 2,
         ":2967 dibenzylidenedisilanyl"),
        ("[SiH2]=NNCl", {"N"}, 2, ":2975 chloro(silylidene)hydrazine"),
        ("O=C=C*", {"C", "*"}, 1, ":2983 oxoethenyl"),
        ("ClN=N*", {"N", "*"}, 1, ":2987 chlorodiazenyl"),
    ])
    def test_licence_fires(self, smiles, parent, n, bb):
        assert _l4(smiles, parent, n) is True, bb


class TestL4BlueBookNegatives:
    """``:2991`` -- "In the following examples locants are needed." """

    def test_interchange_of_two_different_prefixes_makes_an_isomer(self):
        """``:2995`` ``1-ethylidene-2-propylidenedisilan-1-yl``.

        The BB's own reason: *"by interchanging the two substituent groups another
        isomer is generated, i.e., 2-ethylidene-1-propylidenedisilan-1-yl"*. Both
        silicons carry hydrogen and lie in DIFFERENT orbits (only one bears the
        free valence), so a heterogeneous multiset must be tested by interchange.
        """
        assert _l4("CCC=[SiH][Si](=CC)*", {"Si", "*"}, 2) is False

    def test_co_location_on_one_position_makes_an_isomer(self):
        """``:2999`` ``1-chloro-2-ethylidenedisilane (PIN)``.

        The BB's own reason: *"another isomer is generated by moving the Cl atom to
        the other Si atom, i.e., 1-chloro-1-ethylidenedisilane"*. Disilane IS
        symmetric, so every one-decoration-per-atom placement agrees; the isomer
        appears only when BOTH decorations sit on the SAME silicon, which its three
        hydrogens permit (2 for the ethylidene + 1 for the chloro).
        """
        assert _l4("CC=[SiH][SiH2]Cl", {"Si"}, 2) is False

    def test_two_hydrogen_bearing_positions_in_different_orbits(self):
        """``:3003`` ``2-chloroethen-1-yl`` -- C1 carries the free valence."""
        assert _l4("ClC=C*", {"C", "*"}, 1) is False


class TestL4TheHydrogenCountBoundary:
    """★ The pair the whole rule turns on, and it is a hydrogen COUNT.

    Trisulfane ``HS-S-SH`` and triazane ``H2N-NH-NH2`` are the same length and the
    same shape. The middle sulfur has **no** hydrogen, so S1/S3 are the only
    placements and the licence fires; the middle nitrogen **has** one, so
    ``2-methyltriazane`` and ``1,1-dimethyltriazane`` are real isomers and the
    licence must deny. Nothing about chalcogens enters -- see
    :class:`TestL4DoesNotReuseSubstitutablePositions`.
    """

    def test_trisulfane_omits(self):
        assert _l4("CSSS", {"S"}, 1) is True

    def test_triazane_cites(self):
        """``1-methyltriazane``: moving the methyl to N2 gives 2-methyltriazane."""
        assert _l4("CNNN", {"N"}, 1) is False

    def test_dimethyltriazane_cites(self):
        """``1,3-dimethyltriazane``: ``1,1-`` is a legal placement and an isomer."""
        assert _l4("CNNNC", {"N"}, 2) is False


class TestL4DoesNotReuseSubstitutablePositions:
    """⚠ The single most important design point of the predicate.

    ``substitutable_positions()`` applies ``:3007``'s carve-out (*"Except for
    hydrogen atoms attached to chalcogen atoms..."*), which is a sentence of
    P-14.3.4.5 and has no counterpart in P-14.3.4.4. Every hydrogen a polysulfane
    has is on a sulfur, so routing L4 through that helper would make the entire
    family deny by construction. These two assertions pin the divergence, so a
    later "simplification" that reuses the sibling helper fails here loudly
    instead of silently reverting BB 39335.
    """

    def test_the_sibling_helper_sees_no_position_at_all(self):
        mol = Chem.MolFromSmiles("CSSS")
        s_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S"}
        assert substitutable_positions(mol) & s_atoms == frozenset()

    def test_and_all_three_sibling_licences_therefore_deny(self):
        parent = Chem.MolFromSmiles("SSS")          # trisulfane, the parent hydride
        assert substitutable_positions(parent) == frozenset()
        assert l3_one_kind_of_substitutable_h(parent) is False
        assert l5_uniform_complete(
            parent, decoration_of={0: "methyl"}) is False
        assert l6_all_substitutable_h_share_one_locant(
            parent, {0: 1, 1: 2, 2: 3}) is False

    def test_but_l4_fires(self):
        assert _l4("CSSS", {"S"}, 1) is True


class TestL4IsAFunctionOfTheStructureNotTheSPELLING:
    """★ Found by chasing a mutation SURVIVOR, and it is a real defect class.

    Deleting the ``SetNoImplicit``/``SetNumExplicitHs`` hydrogen freeze from
    ``_l4_relocated_key`` survived every other test here. Investigating the survival
    showed why, and that the freeze is load-bearing: RDKit gives a BRACKET atom
    (``[SH]``, ``[SH0]``) ``NoImplicit=True`` and a fixed explicit-hydrogen count, so
    after the relocation surgery its hydrogens are no longer recomputed from valence.
    Without the freeze the mutant measured::

        CSSS -> True (implicit hydrogens, recomputed correctly)
        CSS[SH] -> False SAME MOLECULE, bracket spelling
        C[SH0]SS -> False SAME MOLECULE, bracket spelling

    -- three spellings of one compound, two different licence answers, i.e. the name
    would depend on how the input SMILES was written. It fails in the safe direction
    (a lost licence, never a wrong name), which is exactly why nothing else caught it.

    All three canonicalize to ``CSSS``; a predicate that is a function of the
    STRUCTURE must return one answer for all of them.
    """

    @pytest.mark.parametrize("smiles", ["CSSS", "CSS[SH]", "C[SH0]SS"])
    def test_one_answer_for_every_spelling_of_methyltrisulfane(self, smiles):
        assert Chem.CanonSmiles(smiles) == "CSSS", "witness is not the same molecule"
        assert _l4(smiles, {"S"}, 1) is True

    @pytest.mark.parametrize("smiles", ["CSSSC", "C[SH0][SH0][SH0]C"])
    def test_one_answer_for_every_spelling_of_dimethyltrisulfane(self, smiles):
        assert Chem.CanonSmiles(smiles) == "CSSSC", "witness is not the same molecule"
        assert _l4(smiles, {"S"}, 2) is True

    @pytest.mark.parametrize("smiles", ["CNNN", "CN[NH]N", "C[NH]NN"])
    def test_the_boundary_is_spelling_independent_too(self, smiles):
        """The DENY side must be stable across spellings as well."""
        assert Chem.CanonSmiles(smiles) == "CNNN", "witness is not the same molecule"
        assert _l4(smiles, {"N"}, 1) is False


class TestL4DeniesByDefault:
    """Deny-by-default, per the module docstring and ARCH-a."""

    def test_forced_locant_scope_declines(self):
        """ARCH-a ambient declaration 1 (P-14.3.3, ``:2869``)."""
        with forced_locant_scope("test"):
            assert _l4("CSSS", {"S"}, 1) is False

    def test_isotopic_naming_scope_declines(self):
        """ARCH-a ambient declaration 2 -- P-82.6.1.1 (``:44180``).

        The isotope path names an isotope-STRIPPED skeleton, so no structural test
        inside can see the label; this ambient flag is the only signal left.
        """
        with isotopic_naming_scope():
            assert _l4("CSSS", {"S"}, 1) is False

    def test_structural_isotope_declines(self):
        assert _l4("CSSS", {"S"}, 1, has_isotope=True) is False

    def test_stereodescriptor_declines(self):
        assert _l4("CSSS", {"S"}, 1, stereo_text="(R)-") is False

    @pytest.mark.parametrize("flag", [
        "has_indicated_h", "is_multiplicative", "is_ring_assembly",
        "has_skeletal_replacement",
    ])
    def test_each_hard_override_declines(self, flag):
        assert _l4("CSSS", {"S"}, 1, **{flag: True}) is False

    def test_letter_locant_declines(self):
        assert _l4("CSSS", {"S"}, 1, prefix_locants=["N"]) is False

    def test_defined_stereochemistry_declines(self):
        """Relocating a bond cannot be shown to preserve a descriptor."""
        mol = Chem.MolFromSmiles("C[C@H](Cl)SSS")
        parent = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S"]
        assert l4_no_isomer_by_relocation(
            mol, parent, prefix_locants=[1], suffix_locants=[], stereo_text="",
            has_indicated_h=False, has_isotope=False) is False

    def test_none_mol_declines(self):
        assert l4_no_isomer_by_relocation(
            None, [0], prefix_locants=[1], suffix_locants=[], stereo_text="",
            has_indicated_h=False, has_isotope=False) is False

    def test_none_parent_declines(self):
        assert l4_no_isomer_by_relocation(
            Chem.MolFromSmiles("CSSS"), None, prefix_locants=[1],
            suffix_locants=[], stereo_text="", has_indicated_h=False,
            has_isotope=False) is False

    def test_out_of_range_parent_declines(self):
        assert l4_no_isomer_by_relocation(
            Chem.MolFromSmiles("CSSS"), [99], prefix_locants=[1],
            suffix_locants=[], stereo_text="", has_indicated_h=False,
            has_isotope=False) is False

    def test_undecorated_parent_declines(self):
        """No decoration means no locant to omit."""
        assert _l4("SSS", {"S"}, 0) is False

    def test_ring_fused_to_the_parent_declines(self):
        """A component joined by TWO bonds is not something ':2953' can 'move'."""
        mol = Chem.MolFromSmiles("C1CSSS1")          # a ring THROUGH the S chain
        parent = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S"]
        assert l4_no_isomer_by_relocation(
            mol, parent, prefix_locants=[1], suffix_locants=[], stereo_text="",
            has_indicated_h=False, has_isotope=False) is False
