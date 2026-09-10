"""§ (the Blue Book) — the locant of a lone substitution on a symmetric parent.

 Phase C Task 3. Covers the rule verbatim::

    "The locant is omitted in monosubstituted symmetrical parent hydrides or parent
     compounds where there is only one kind of substitutable hydrogen."

and its five printed positives (``:2943`` ``methylurea``, ``:2945``
``chlorodisiloxane``, ``:2947`` ``chlorocoronene``, ``:2949``
``pyrazinecarboxylic acid``, ``:2951`` ``chloropropanedioic acid``) — note that
``pyrazinecarboxylic acid`` is a **suffix** substitution and ``chloropropanedioic
acid`` a **prefix** one, which is why the licence is wired at two joins.

★ THE BOUNDARY THIS WHOLE TASK TURNS ON — and why the predicate tests below are the
real guard rather than a formality. Two rows sit in the example block of a DIFFERENT
sub-rule, § (``:2877``), and read as a flat contradiction::

    :2883 HOOC-CH2-CH(Cl)-COOH chlorobutanedioic acid (PIN) locant OMITTED
    :2887 H2N-CO-CH(CH3)-CO-NH2 2-methylpropanediamide (PIN) locant KEPT

They are reconciled by splitting the two rules: ** withdraws only the
TERMINAL (suffix) locants** — that is what makes both of them ``-dioic acid`` /
``-diamide`` rather than ``-1,4-dioic acid`` / ``-1,3-diamide``, and it says nothing
about substituent locants — and ** then decides the substituent locant** on
the substitutable-hydrogen test. Propanedioic acid's two acid O-H are on a chalcogen
and are NOT substitutable (``:3007``), leaving C2 as the only kind, so the licence
fires. Propane**diamide** has C2 *and* two amide N-H, which are neither chalcogen nor
formyl H and therefore DO count, giving two kinds, so it is denied. ``:2889``
``N1,N3-dimethylpropanediamide (PIN)`` proves independently that an amide N-H is
substitutable. Nothing about chain length; nothing about the substituent.

⚠ WHY ``2-methylpropanediamide`` IS ASSERTED TWICE. It is the row that proves an L3
wiring has not over-stripped the ``-diamide`` family, but Orthonym emits
``unknown organic compound`` for it today (measured 2026-07-29 — a coverage gap, not a
locant defect). An end-to-end assertion of the form "the output is not
``chloropropanediamide``" is therefore **vacuously green**: the abstention string
satisfies it. So it is asserted at the **predicate** level, which is the live guard,
AND as an ``xfail(strict=True)`` end-to-end, so the day the diamide becomes nameable
the XPASS forces someone to check the spelling.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.locant_omission import (
    _one_substituent_removed,
    forced_locant_scope,
    isotopic_naming_scope,
    l3_locant_omitted_for_parent_atoms,
    l3_monosubstituted_locant_omitted,
    l3_one_kind_of_substitutable_h,
)


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"test fixture SMILES did not parse: {smiles}"
    return m


def _atoms_except(mol, *symbols):
    """Parent-atom indices = everything that is not one of ``symbols``.

    Used instead of hardcoded indices so the tests state WHICH atom is the
    substitution rather than encoding RDKit's atom ordering.
    """
    drop = set(symbols)
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() not in drop]


# --------------------------------------------------------------------------- #
# 1. THE PREDICATE — the C4 pair, and the orbit boundary either side of it #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "smiles,label,expected",
    [
        # ★ THE PAIR. These two rows ARE the rule; everything else follows.
        ("OC(=O)CC(=O)O", "propanedioic acid — C2 only (acid O-H excluded, :3007)", True),
        ("NC(=O)C(C)C(=O)N", "2-methylpropanediamide — C2, CH3 and N-H", False),
        # The undecorated diamide, so the denial cannot be blamed on the methyl.
        ("NC(=O)CC(=O)N", "propanediamide — C2 AND amide N-H = two kinds", False),
        #:2883 — equivalent by symmetry rather than by being a lone position.
        ("OC(=O)CCC(=O)O", "butanedioic acid — C2/C3 one orbit", True),
        # ★ The next homologue DENIES, and no rule about diacids says so: pentanedioic
        # acid's C2/C4 are one orbit but C3 is another. This is the row that proves the
        # predicate is measuring orbits and not "is a diacid".
        ("OC(=O)CCCC(=O)O", "pentanedioic acid — C2/C4 vs C3 = two orbits", False),
        #:2949's parent hydride, and the three rings it must be distinguished from.
        ("c1cnccn1", "pyrazine — four equivalent CH", True),
        ("c1ccncc1", "pyridine — C2/C6, C3/C5, C4 = three orbits", False),
        ("C1CCNCC1", "piperidine — N-H, C2/C6, C3/C5, C4 = four orbits", False),
        ("C1CNCCN1", "piperazine — N-H and CH2 = two orbits", False),
        #:2943's parent compound.
        ("NC(=O)N", "urea — two equivalent NH2", True),
        # Carbocyclic controls.
        ("C1CCCCC1", "cyclohexane", True),
        ("c1ccccc1", "benzene", True),
        # An all-chalcogen parent has NO substitutable hydrogen at all, so the licence
        # denies by construction. ⚠ Do not "fix" this to make `methyltrisulfane` work:
        # the chalcogen exclusion is load-bearing for `chloropropanedioic acid` above,
        # and trisulfane is licensed by, a different (unimplemented) rule.
        ("SSS", "trisulfane — every H on sulfur", False),
    ],
)
def test_l3_orbit_predicate(smiles, label, expected):
    assert l3_one_kind_of_substitutable_h(_mol(smiles)) is expected, label


# --------------------------------------------------------------------------- #
# 2. "MONOSUBSTITUTED" IS PROVEN STRUCTURALLY, NOT ASSUMED #
# --------------------------------------------------------------------------- #
def test_one_substituent_removed_counts_components_not_atoms():
    """Three fluorines are THREE substituents, not one — the shape that would
    otherwise let `2,2,3,3,3-pentafluoropropanoic acid` reach an L3 licence."""
    mono = _mol("OC(=O)C(Cl)C(=O)O")
    assert _one_substituent_removed(mono, frozenset(_atoms_except(mono, "Cl"))) is True

    tri = _mol("OC(=O)CC(F)(F)F")
    assert _one_substituent_removed(tri, frozenset(_atoms_except(tri, "F"))) is False


def test_one_substituent_removed_rejects_nothing_and_everything():
    m = _mol("OC(=O)CC(=O)O")
    # No substitution at all -> there is no locant to omit.
    assert _one_substituent_removed(m, frozenset(range(m.GetNumAtoms()))) is False
    # An empty parent is not a parent.
    assert _one_substituent_removed(m, frozenset()) is False


def test_one_substituent_removed_rejects_a_doubly_bonded_group():
    """A group attached twice is a fused/bridging unit, which:2939 does not licence.
    Cyclohexane fused to the chain would present TWO attachment bonds."""
    m = _mol("O=C(O)C1CCC(C(=O)O)CC1")
    ring = set(m.GetRingInfo().AtomRings()[0])
    assert _one_substituent_removed(m, frozenset(ring)) is False


def test_one_substituent_removed_needs_the_connected_component_check():
    """★ Mutation-derived. The attachment-bond count ALONE is not enough, and the
    trifluoro witness above cannot show it: three fluorines fail the bond count first,
    so the component check has no independent witness there and a mutant that deletes
    it survives.

    This is the shape that needs it — exactly ONE attachment bond but TWO components
    outside the parent: a hydrate, where the second component is bonded to nothing.
    Without the component check the water would be absorbed into "the one substituent"
    and a locant would be omitted from a molecule that has an unaccounted fragment.
    """
    m = _mol("OC(=O)C(Cl)C(=O)O.O")
    acid = max(Chem.GetMolFrags(m), key=len)
    chlorine = {a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "Cl"}
    parent = frozenset(i for i in acid if i not in chlorine)
    # Precondition, asserted so the witness cannot silently stop being one: exactly
    # one bond crosses the parent boundary, and two components lie outside it.
    outside = [i for i in range(m.GetNumAtoms()) if i not in parent]
    crossings = sum(
        1 for b in m.GetBonds()
        if (b.GetBeginAtomIdx() in parent) != (b.GetEndAtomIdx() in parent)
    )
    assert crossings == 1, "witness precondition: one attachment bond"
    assert len(outside) == 2, "witness precondition: Cl plus a detached water O"
    assert _one_substituent_removed(m, parent) is False


# --------------------------------------------------------------------------- #
# 3. THE LICENCE ENTRY POINT, INCLUDING BOTH AMBIENT SCOPES #
# --------------------------------------------------------------------------- #
def _acid_licence():
    m = _mol("OC(=O)C(Cl)C(=O)O")
    return m, _atoms_except(m, "Cl")


def test_licence_fires_on_the_printed_positive():
    m, parent = _acid_licence()
    assert l3_locant_omitted_for_parent_atoms(
        m, parent, prefix_locants=[2], suffix_locants=[],
        parent_cites_locants=False, stereo_text="",
    ) is True


@pytest.mark.parametrize(
    "kwargs,why",
    [
        ({"suffix_locants": [1, 3]}, "a suffix locant still cited in the scope"),
        ({"parent_cites_locants": True}, "parent name already cites a locant"),
        ({"stereo_text": "(2R)"}, "a stereodescriptor needs a locant"),
        ({"is_multiplicative": True}, "multiplicative names always cite"),
        ({"is_ring_assembly": True}, "ring assemblies always cite"),
        ({"has_skeletal_replacement": True}, "heteroatom locants are essential"),
        ({"prefix_locants": ["N"]}, "a LETTER locant is always essential"),
        ({"prefix_locants": [2, 3]}, "two locants is not 'monosubstituted'"),
        ({"prefix_locants": []}, "no locant cited -> nothing to omit"),
    ],
)
def test_licence_declines_on_each_essential_locant_route(kwargs, why):
    m, parent = _acid_licence()
    call = dict(prefix_locants=[2], suffix_locants=[],
                parent_cites_locants=False, stereo_text="")
    call.update(kwargs)
    assert l3_locant_omitted_for_parent_atoms(m, parent, **call) is False, why


def test_licence_declines_inside_forced_locant_scope():
    """ as an ambient scope. The positive above is re-asserted OUTSIDE the
    scope in the same test so a broken licence cannot make this vacuously green."""
    m, parent = _acid_licence()
    call = dict(prefix_locants=[2], suffix_locants=[],
                parent_cites_locants=False, stereo_text="")
    assert l3_locant_omitted_for_parent_atoms(m, parent, **call) is True
    with forced_locant_scope("test"):
        assert l3_locant_omitted_for_parent_atoms(m, parent, **call) is False


def test_licence_declines_inside_isotopic_naming_scope():
    """⚠ ``locants_are_forced`` ALONE IS NOT ENOUGH — measured in Task 5a, where the
    isotope decorator enters the forced scope only conditionally and a licence wired on
    the forced flag alone emptied a scope that still carried a ``(13C1)`` descriptor
    , ``:44180``). This licence empties its scope of ALL locants, so it must
    decline on the weaker declaration too."""
    m, parent = _acid_licence()
    call = dict(prefix_locants=[2], suffix_locants=[],
                parent_cites_locants=False, stereo_text="")
    assert l3_locant_omitted_for_parent_atoms(m, parent, **call) is True
    with isotopic_naming_scope():
        assert l3_locant_omitted_for_parent_atoms(m, parent, **call) is False


@pytest.mark.parametrize("n_subs", [0, 2, 3, True, None, "1"])
def test_core_licence_requires_exactly_one_substitution(n_subs):
    """★ Mutation-derived. ``l3_monosubstituted_locant_omitted`` is exported and its
    ``n_substitutions`` guard is unreachable through the structural front end (which
    always proves and passes 1), so a mutant deleting it survived. Asserted directly
    here — including ``True``, which ``isinstance(True, int)`` would otherwise let
    through as 1.
    """
    parent = _mol("OC(=O)CC(=O)O")            # propanedioic acid: L3 says True at n=1
    assert l3_monosubstituted_locant_omitted(
        parent, n_substitutions=n_subs, prefix_locants=[2], suffix_locants=[],
        parent_cites_locants=False, stereo_text="", has_indicated_h=False,
        has_isotope=False,
    ) is False
    #...and the same call with n_substitutions=1 DOES fire, so the above is not
    # vacuously green.
    assert l3_monosubstituted_locant_omitted(
        parent, n_substitutions=1, prefix_locants=[2], suffix_locants=[],
        parent_cites_locants=False, stereo_text="", has_indicated_h=False,
        has_isotope=False,
    ) is True


def test_core_licence_declines_on_indicated_hydrogen():
    """``has_indicated_h`` reaches ``scope_forces_locants``; the structural front end
    hardcodes False, so this is asserted on the core function."""
    parent = _mol("OC(=O)CC(=O)O")
    assert l3_monosubstituted_locant_omitted(
        parent, n_substitutions=1, prefix_locants=[2], suffix_locants=[],
        parent_cites_locants=False, stereo_text="", has_indicated_h=True,
        has_isotope=False,
    ) is False
    assert l3_monosubstituted_locant_omitted(
        parent, n_substitutions=1, prefix_locants=[2], suffix_locants=[],
        parent_cites_locants=False, stereo_text="", has_indicated_h=False,
        has_isotope=True,
    ) is False


def test_licence_declines_when_the_parent_is_the_wrong_atom_set():
    """Feeding the ring only, for a molecule whose parent compound includes the acid,
    measures a different molecule. Fail-closed rather than answer the wrong question."""
    m = _mol("OC(=O)C(Cl)C(=O)O")
    assert l3_locant_omitted_for_parent_atoms(
        m, [999], prefix_locants=[2], suffix_locants=[],
        parent_cites_locants=False, stereo_text="",
    ) is False
    assert l3_locant_omitted_for_parent_atoms(
        None, [0], prefix_locants=[2], suffix_locants=[],
        parent_cites_locants=False, stereo_text="",
    ) is False


# --------------------------------------------------------------------------- #
# 4. END TO END — the three targets, across BOTH joins #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "smiles,expected,citation",
    [
        # Ring-SUFFIX join (rules/heterocycles.py).:2949 verbatim.
        ("OC(=O)c1cnccn1", "pyrazinecarboxylic acid", ":2949"),
        # Substituent-PREFIX join (assembly/handlers/_handler_shared.py).:2951 verbatim.
        ("OC(=O)C(Cl)C(=O)O", "chloropropanedioic acid", ":2951"),
        #:2883 verbatim — L1 takes the suffix locants, L3 the substituent one.
        ("OC(=O)CC(Cl)C(=O)O", "chlorobutanedioic acid", ":2883"),
        # Fused-PAH PREFIX join (rules/polycyclics.py).:2947 verbatim — coronene's
        # twelve CH are ONE orbit .
        ("Clc1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61", "chlorocoronene", ":2947"),
        # Retained-parent urea join (assembly/composer.py).:2943 verbatim — urea's
        # four N-H are ONE orbit, so the monosubstituted italic-N locant is omitted
        # . NOT the chalcogen analogues -> see the negatives below.
        ("CNC(=O)N", "methylurea", ":2943"),
    ],
)
def test_targets(namer, smiles, expected, citation):
    """The WHOLE name is asserted (session invariant 11). Four times in v29 a change
    that stopped a bad path emitted something WORSE — a dropped atom, a lost pair of
    enclosing marks — and only a whole-string assertion can see that."""
    assert namer.name(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected,why",
    [
        # The licence is a structural predicate, not a table of three molecules: it
        # generalises across the substituent and across the suffix. All measured.
        ("OC(=O)C(Br)C(=O)O", "bromopropanedioic acid", "same licence, other halogen"),
        ("N#CC(Cl)C#N", "chloropropanedinitrile", ":2877 covers nitriles too"),
        ("N#Cc1cnccn1", "pyrazinecarbonitrile", "same ring, other suffix"),
        ("NC(=O)c1cnccn1", "pyrazinecarboxamide", "same ring, other suffix"),
        ("Oc1cnccn1", "pyrazinol", "same ring, vowel-initial suffix (elision holds)"),
        ("Sc1cnccn1", "pyrazinethiol", "same ring, consonant-initial suffix"),
        ("Nc1cnccn1", "pyrazinamine", "same ring, 'e' elided before 'amine'"),
    ],
)
def test_class_generalises(namer, smiles, expected, why):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # ★ MUTATION-DERIVED WITNESSES. Deleting the N-substituent guard at the ring
        # -suffix join survived the first test round — these are the rows that kill it.
        # The italic-N locant is ESSENTIAL, and it is prepended AFTER the suffix join,
        # so the guard has to be consulted at the join or (:2869) is violated:
        # one essential locant in the scope restores EVERY locant in it. Without the
        # guard these become 'N-methylpyrazinamine' / 'N,N-dimethylpyrazinamine' /
        # 'N-methylpyrazinecarboxamide'.
        ("CNc1cnccn1", "N-methylpyrazin-2-amine"),
        ("CN(C)c1cnccn1", "N,N-dimethylpyrazin-2-amine"),
        ("CNC(=O)c1cnccn1", "N-methylpyrazine-2-carboxamide"),
    ],
)
def test_an_essential_N_locant_restores_the_ring_suffix_locant(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --------------------------------------------------------------------------- #
# 5. THE NEGATIVES. Each was measured at HEAD before the change; none may move. #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "smiles,expected,why",
    [
        # --- the diamide family, the class L3 must NOT over-strip ---
        ("NC(=O)CC(=O)N", "propanediamide", "unsubstituted parent, nothing to omit"),
        ("CNC(=O)CC(=O)NC", "N1,N3-dimethylpropanediamide",
         ":2889 — letter locants, and TWO substituents"),
        # --- the orbit boundary: one carbon longer / shorter and the licence dies ---
        ("OC(=O)C(Cl)CCC(=O)O", "2-chloropentanedioic acid", "C3 is its own orbit"),
        ("CC(Cl)C(=O)O", "2-chloropropanoic acid", "C2 and C3 are different orbits"),
        ("ClCCC(=O)O", "3-chloropropanoic acid", "ditto"),
        ("OC(=O)C(Cl)C(Cl)C(=O)O", "2,3-dichlorobutanedioic acid",
         "TWO substituents — not 'monosubstituted'"),
        # --- the ring boundary,:34728 vs:34730 ---
        ("OC(=O)C1CCCCC1", "cyclohexanecarboxylic acid",
         "already correct via P-14.3.4.2(c); the in-handler control"),
        ("N#CC1CCCCC1", "cyclohexanecarbonitrile", ":34728 carbocycle omits"),
        ("N#CN1CCCCC1", "piperidine-1-carbonitrile",
         "★ :34730 — same suffix, HETEROcycle KEEPS. Four orbits."),
        ("OC(=O)c1ccncc1", "pyridine-4-carboxylic acid", "three orbits"),
        # --- a parent name that already cites locants restores them all ---
        ("OC(=O)C1COCCO1", "1,4-dioxane-2-carboxylic acid",
         "heteroatom locant set in the parent name"),
        ("OC(=O)c1ncncn1", "1,3,5-triazine-2-carboxylic acid", "ditto"),
        ("OC(=O)c1cc[nH]c1", "1H-pyrrole-3-carboxylic acid", "indicated hydrogen"),
        # --- urea OMITS (see the positives above,:2943); its CHALCOGEN analogues
        # and its disubstituted forms KEEP the letter locant ---
        ("NC(=O)N", "urea", "the retained parent"),
        ("CNC(=S)N", "N-methylthiourea",
         "chalcogen analogue keeps its letter locant — P-66.1.6.1.3.1, :33451 "
         "N-(butan-2-yl)selenourea (PIN); one-orbit but a MORE SPECIFIC rule"),
        ("CNC(=O)NC", "N,N'-dimethylurea",
         "TWO substituents — not 'monosubstituted' (:33327)"),
        # --- / L6 neighbours, none of which L3 may disturb ---
        ("OC(=O)C(F)(F)F", "trifluoroacetic acid", "L6, :3037"),
        ("OC(=O)CC(F)(F)F", "3,3,3-trifluoropropanoic acid", "partial -> :3009 retains"),
        ("OC(=O)C(F)C(F)(F)F", "2,3,3,3-tetrafluoropropanoic acid", "ditto"),
        ("NC(=O)C(F)(F)C(F)(F)F", "2,2,3,3,3-pentafluoropropanamide",
         "★ amide N-H are substitutable and undecorated -> partial"),
        ("Oc1c(O)c(O)c(O)c(O)c1O", "benzenehexol", "L5 uniform complete"),
        ("Cc1c(C)c(C)c(C)c(C)c1C", "hexamethylbenzene", "ditto"),
        ("Clc1ccccc1C(F)(F)C(F)(F)F", "1-chloro-2-(pentafluoroethyl)benzene",
         "L5 inside an enclosing-mark scope; outer locants survive"),
        ("FC(F)(F)C(F)(F)C1CCCCC1", "(pentafluoroethyl)cyclohexane", "ditto"),
        ("FC(F)(F)C(F)(F)C(F)(F)c1ccccc1", "(heptafluoropropyl)benzene",
         "WITH its parentheses — a v29 change once lost exactly these"),
        ("Oc1c(O)c(O)c(O)c(O)c1Cl", "6-chlorobenzene-1,2,3,4,5-pentol", "P-14.4(c)"),
        ("Cc1c(C)c(C)c(C)c(C)c1Cl", "1-chloro-2,3,4,5,6-pentamethylbenzene",
         "no principal characteristic group -> (c) vacuous"),
        ("OC1CCCCC1", "cyclohexanol", "P-14.3.4.2(c)"),
        ("CCO", "ethanol", "P-14.3.4.2(a)"),
        # --- not this task: L4 owns the polysulfanes ---
        # SHIPPED by Task 11 (2026-07-30): L4 is implemented and this row is now
        # BB 39335's verbatim PIN. It stays in THIS file's negative list because
        # the point it makes is unchanged -- L3 must not be widened to reach it,
        # and no L3 change may alter it. (:3007's chalcogen carve-out still makes
        # trisulfane's substitutable set EMPTY, so all of L3/L5/L6 deny here.)
        ("CSSS", "methyltrisulfane",
         "licensed by P-14.3.4.4, NOT by L3: every H is on a chalcogen, so "
         "substitutable_positions is empty and L3 denies by construction"),
        # --- not this task: L5 parent scope, SHIPPED by Task 5b (2026-07-30) ---
        ("OC(=O)C(F)(F)C(F)(F)C(F)(F)F", "heptafluorobutanoic acid",
         "P-14.3.4.5 (:3017 verbatim PIN) -- a DIFFERENT licence, which must not be "
         "reached by widening L3: L3 needs exactly ONE cited locant and this cites "
         "seven"),
    ],
)
def test_negatives_unchanged(namer, smiles, expected, why):
    assert namer.name(smiles) == expected


# --------------------------------------------------------------------------- #
# 5b. THE HETEROATOM-H BOUNDARY OF THE UREA LICENCE (-fix) #
# --------------------------------------------------------------------------- #
# omits the italic-N locant only while the parent keeps "only one kind
# of substitutable hydrogen". Task 7B2 wired the licence for a CARBON substituent
# (methylurea, the Blue Book) but over-fired on a substituent that carries its OWN
# substitutable heteroatom-H: N=C(N)NC(N)=O was mis-named `carbamimidoylurea`.
# Carbamimidoyl (H2N-C(=NH)-) bears an imino and an amino N-H, which are a SECOND
# kind of substitutable hydrogen, so the locant MUST be cited — the Blue Book verbatim
# `N-carbamimidoylurea (PIN)` (also `N-carbamimidoylformamide`:34290,
# `N-carbamimidoylacetamide`:34294). The omit rows are re-asserted alongside the
# keep row so the keep assertion is not vacuously green (session a project rule).
@pytest.mark.parametrize(
    "smiles,expected,why",
    [
        # ★ THE FIX. Substituent carries its own N-H -> a second kind -> KEEP the N-.
        ("N=C(N)NC(N)=O", "N-carbamimidoylurea",
         "carbamimidoyl bears imino + amino N-H -> two kinds (:34292 verbatim PIN)"),
        # The gains Task 7B2 shipped, which the narrowed guard must NOT undo: a plain
        # or halogenated hydrocarbyl substituent has no heteroatom-H -> one kind -> OMIT.
        ("CNC(=O)N", "methylurea", "methyl: no heteroatom-H (:2943)"),
        ("FC(F)(F)NC(=O)N", "(trifluoromethyl)urea", "CF3: F carries no H"),
        ("CCNC(=O)N", "ethylurea", "ethyl: no heteroatom-H"),
    ],
)
def test_urea_locant_kept_when_substituent_has_heteroatom_h(namer, smiles, expected, why):
    """The WHOLE name is asserted (session a project rule)."""
    assert namer.name(smiles) == expected


# --------------------------------------------------------------------------- #
# 6. THE MANDATORY NEGATIVE THAT IS NOT NAMEABLE — asserted as strict xfail #
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=True,
    reason="coverage gap measured 2026-07-29 AND re-measured after this change: "
           "'NC(=O)C(C)C(=O)N' emits 'unknown organic compound' (OPSIN-UNPARSEABLE). "
           "The live guard for this row is test_l3_orbit_predicate, which asserts the "
           "predicate is False. When this XPASSes, confirm the locant SURVIVED — an "
           "L3 wiring that over-strips would emit 'methylpropanediamide' here.",
)
def test_2_methylpropanediamide_keeps_its_locant_when_it_becomes_nameable(namer):
    assert namer.name("NC(=O)C(C)C(=O)N") == "2-methylpropanediamide"
