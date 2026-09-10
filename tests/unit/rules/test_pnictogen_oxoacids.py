""" organo-oxoacids of the heavier pnictogens (As, Sb).

An arsonic acid is NOT an organometallic -- it is a oxoacid, the exact
analogue of a phosphonic acid. The Blue Book gives all four as *preselected
names* (BB L36051-36054):

    AsH(O)(OH)2 arsonic acid AsH2(O)OH arsinic acid
    SbH(O)(OH)2 stibonic acid SbH2(O)OH stibinic acid

and gives the substituent-prefix PIN forms verbatim:

    CH3-P(O)(OH)2 methylphosphonic acid (PIN) BB L36062
    CH3-CH2-SbH(O)OH ethylstibinic acid (PIN) BB L36064
    C6H5-As(CH3)(O)OH methyl(phenyl)arsinic acid (PIN) BB L36066
    (4-acetamido-3-methylphenyl)arsonic acid (PIN) BB L33010

Bismuth has NO oxoacid analogue in the Blue Book (there is no "bismuthonic
acid"), so the family is bounded to P / As / Sb -- matching the existing
``_PNICTOGEN_ACID_STEM`` table in ``rules.phosphorus``.

Every expected name in this module was verified OPSIN-exact (name -> SMILES ->
canonical SMILES == canonical SMILES of the input) before being written.
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.perception.functional_groups import detect_functional_groups


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.fixture
def gated_namer(monkeypatch):
    """A namer with the OPSIN validity gate ON (production behaviour).

    ``tests/conftest.py`` has an autouse fixture that disables the gate for the
    whole suite so tests can assert RAW producer output. Every assertion in this
    module about the CONTRACT -- what may and may not leave the engine -- is
    about what actually SHIPS, so it must re-enable the gate; otherwise the test
    asserts an ungated candidate string and proves nothing about production.
    (Before, without this, the benzyl case asserted against the raw
    'methylbenzene' candidate rather than the abstention.) Tests that must hold
    even with no JRE take the opposite tack and disable the gate explicitly --
    see ``test_benzyl_case_names_correctly_even_without_the_opsin_gate``.
    """
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    return Orthonym()


# --------------------------------------------------------------------------
# 1. Perception: the four new functional groups are detected
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,fg", [
    ("C[As](=O)(O)O", "arsonic_acid"),
    ("CC[As](=O)(O)O", "arsonic_acid"),
    ("c1ccccc1[As](=O)(O)O", "arsonic_acid"),
    ("C[As](C)(=O)O", "arsinic_acid"),
    ("C[Sb](=O)(O)O", "stibonic_acid"),
    ("C[Sb](C)(=O)O", "stibinic_acid"),
])
def test_pnictogen_oxoacid_functional_group_detected(smiles, fg):
    mol = Chem.MolFromSmiles(smiles)
    assert fg in detect_functional_groups(mol), (
        f"{smiles} should perceive {fg}"
    )


# --------------------------------------------------------------------------
# 2. The carbon guard: a FREE inorganic oxoacid must NOT be claimed
# (arsoric acid As(O)(OH)3 / stiboric acid are separate preselected names)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,fg", [
    ("O[As](=O)(O)O", "arsonic_acid"),    # arsoric acid -- no C-As bond
    ("O[Sb](=O)(O)O", "stibonic_acid"),   # stiboric acid -- no C-Sb bond
])
def test_free_inorganic_oxoacid_is_not_an_organo_oxoacid(smiles, fg):
    mol = Chem.MolFromSmiles(smiles)
    assert fg not in detect_functional_groups(mol), (
        f"{smiles} is a free inorganic oxoacid and must not match {fg}"
    )


# --------------------------------------------------------------------------
# 3. End-to-end PIN names (all OPSIN-verified against the input structure)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # -onic (one C, two OH) -- BB L36051 / L36054
    ("C[As](=O)(O)O",            "methylarsonic acid"),
    ("CC[As](=O)(O)O",           "ethylarsonic acid"),
    ("CCC[As](=O)(O)O",          "propylarsonic acid"),
    ("c1ccccc1[As](=O)(O)O",     "phenylarsonic acid"),
    ("C[Sb](=O)(O)O",            "methylstibonic acid"),
    ("CC[Sb](=O)(O)O",           "ethylstibonic acid"),
    # -inic (two C, one OH) -- BB L36052 / L36054
    ("C[As](C)(=O)O",            "dimethylarsinic acid"),
    ("c1ccccc1[As](=O)(O)c1ccccc1", "diphenylarsinic acid"),
    ("C[Sb](C)(=O)O",            "dimethylstibinic acid"),
    # BB L36066 verbatim PIN: C6H5-As(CH3)(O)OH
    ("c1ccccc1[As](C)(=O)O",     "methyl(phenyl)arsinic acid"),
])
def test_pnictogen_oxoacid_pin_name(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --------------------------------------------------------------------------
# 4. The phosphorus sibling is unchanged (this build generalises its code path)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("CP(=O)(O)O",             "methylphosphonic acid"),
    ("CCP(=O)(O)O",            "ethylphosphonic acid"),
    ("CCCP(=O)(O)O",           "propylphosphonic acid"),
    ("OP(=O)(O)c1ccccc1",      "phenylphosphonic acid"),
    ("CP(C)(=O)O",             "dimethylphosphinic acid"),
    ("c1ccccc1P(C)(=O)O",      "methyl(phenyl)phosphinic acid"),
])
def test_phosphorus_oxoacid_names_unchanged(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --------------------------------------------------------------------------
# 4b. DI-ARYL pnictogen -inic acids: the acid is SENIOR to the ring, so
# the two identical carbocycles are cited as ``diphenyl``/``dicyclohexyl``
# prefixes on the acid parent.x -inic acid). They must NOT be
# named multiplicatively with the acid demoted to a ``(hydroxy...oryl)``
# bridge (``1,1'-(hydroxyarsoryl)dibenzene``) — a right-molecule, wrong-PIN
# regression that surfaced once the acyl-prefix bridge
# generalised from phosphoryl to arsoryl/stiboryl (rules/multiplicative.py).
# The multiplicative PIN is correctly retained ONLY when the ring units bear
# a senior PCG of their own (4,4'-(hydroxyarsoryl)dibenzoic acid).
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("c1ccccc1[As](=O)(O)c1ccccc1",         "diphenylarsinic acid"),
    ("c1ccccc1[P](=O)(O)c1ccccc1",          "diphenylphosphinic acid"),
    ("c1ccccc1[Sb](=O)(O)c1ccccc1",         "diphenylstibinic acid"),
    # whole class: two identical SIMPLE carbocycles, not only benzene
    ("C1CCCCC1[As](=O)(O)C1CCCCC1",         "dicyclohexylarsinic acid"),
])
def test_diaryl_pnictogen_inic_acid_is_parent_not_multiplicative(
        namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # ring units carry a senior CARBOXYLIC acid -> the acyl bridge is correct
    ("O[As](=O)(C1=CC=C(C(=O)O)C=C1)C1=CC=C(C(=O)O)C=C1",
     "4,4'-(hydroxyarsoryl)dibenzoic acid"),
    ("COP(=O)(C1=CC=C(C(=O)O)C=C1)C1=CC=C(C(=O)O)C=C1",
     "4,4'-(methoxyphosphoryl)dibenzoic acid"),
])
def test_pnictogen_acyl_bridge_multiplicative_retained_on_senior_ring_pcg(
        namer, smiles, expected):
    assert namer.name(smiles) == expected


# --------------------------------------------------------------------------
# 5. A complex organyl is NAMED, and named CORRECTLY.
#
# a phase (2026-07-26) lifted the fail-closed contract this section used
# to encode. The refusal was never a nomenclature rule: it was an artefact
# of the handler-private carbon-skeleton walker, which miscounts a ring or a
# branch as a linear chain (benzyl -> 'heptyl', cyclohexyl -> 'hexyl'). The
# guard was right to refuse ITS OWN walker's output; the walker was the
# defect. The organyl now goes through the shared substituent chokepoint
# (``assembly.substituent_enumerator.name_substituent``), so these molecules
# are named instead of refused.
#
# THE SAFETY PROPERTY IS UNCHANGED AND STILL ASSERTED BELOW: no emission may
# carry a fabricated constitution. Only the mechanism moved, from "refuses"
# to "names correctly", and every assertion below therefore pins the EXACT
# string rather than merely "something was produced" -- a test that only
# checked for the absence of a failure marker would pass on a wrong name.
#
# Blue Book authority for the names asserted here:
# * (BB 16272): benzyl is a retained PREFERRED prefix, cited bare
# -- '2-benzylpyridine' (PIN), BB 16280.
# * (BB 16270/16286): 'tert-butyl' likewise, cited bare --
# '*tert*-butyldi(methyl)phosphane' (PIN); (b)/(d) 'N-tert-butyl' NOT
# the enclosed form (BB 3465 cites it bare).
# * / BB L36066: for a mononuclear parent the FIRST cited group
# takes no marks and each subsequent one is enclosed --
# 'methyl(phenyl)arsinic acid' (PIN).
# * BB 35461: 'C2H5-P(O)(OH)2 ethylphosphonic acid (PIN) (not
# ethanephosphonic acid)' -- the organyl-prefix form IS the PIN.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("C1=CC=C(C=C1)C[As](=O)(O)O", "benzylarsonic acid"),
    ("C1=CC=C(C=C1)CP(=O)(O)O",    "benzylphosphonic acid"),
])
def test_complex_organyl_is_named_correctly(gated_namer, smiles, expected):
    """The user-reported abstention, now named -- and named EXACTLY right.

    Asserting the exact string is the point: "a name was produced" would be
    satisfied by a fabricated 'heptylarsonic acid' just as well. Run with the
     OPSIN gate ON, so this is what actually SHIPS.
    """
    from orthonym.errors import is_failure_name
    name = gated_namer.name(smiles)
    assert not is_failure_name(name), f"{smiles} must now be named, got {name!r}"
    assert name == expected


@pytest.mark.parametrize("rules_fn,fg,smiles,expected", [
    ("name_arsonic_acid", "arsonic_acid", "C1=CC=C(C=C1)C[As](=O)(O)O",
     "benzylarsonic acid"),
    # / (b)/(d) (BB 16286): the retained italicized prefix is BARE.
    ("name_arsonic_acid", "arsonic_acid", "CC(C)(C)[As](=O)(O)O",
     "tert-butylarsonic acid"),
    ("name_stibonic_acid", "stibonic_acid", "C1=CC=C(C=C1)C[Sb](=O)(O)O",
     "benzylstibonic acid"),
    ("name_arsinic_acid", "arsinic_acid", "C1=CC=C(C=C1)C[As](C)(=O)O",
     "benzyl(methyl)arsinic acid"),
    ("name_stibinic_acid", "stibinic_acid", "C1=CC=C(C=C1)C[Sb](C)(=O)O",
     "benzyl(methyl)stibinic acid"),
])
def test_producer_itself_returns_the_exact_right_string(rules_fn, fg, smiles, expected):
    """The NAMER must return the right string -- not merely be repaired downstream.

    Asserting only the shipped string is too weak: the OPSIN gate would
    mask a producer that fabricates a wrong name (a mutation that made the
    namer emit 'methylarsonic acid' for benzylarsonic acid survived a
    shipped-output-only test). The OPSIN gate also fails OPEN when no JRE is
    present, so the producer level is where the real safety property lives.

    That rationale is why this test still exists after the fail-closed contract
    was lifted: what it guards is not the refusal but the fact that the
    producer, unaided, is the thing that has to be right.
    """
    from orthonym.rules import phosphorus
    mol = Chem.MolFromSmiles(smiles)
    matches = detect_functional_groups(mol).get(fg, [])
    assert matches, f"{smiles} should still perceive {fg}"
    assert getattr(phosphorus, rules_fn)(mol, matches[0]) == expected


@pytest.mark.parametrize("rules_fn,fg,smiles,wrong_token,expected", [
    # benzyl (7 C, one in a ring) was counted as a LINEAR heptyl chain
    ("name_phosphinic_acid", "phosphinic_acid", "C1=CC=C(C=C1)CP(C)(=O)O",
     "heptyl", "benzyl(methyl)phosphinic acid"),
    ("name_arsinic_acid", "arsinic_acid", "C1=CC=C(C=C1)C[As](C)(=O)O",
     "heptyl", "benzyl(methyl)arsinic acid"),
    ("name_stibinic_acid", "stibinic_acid", "C1=CC=C(C=C1)C[Sb](C)(=O)O",
     "heptyl", "benzyl(methyl)stibinic acid"),
    # cyclohexyl (6 C, all in a ring) was counted as a LINEAR hexyl chain
    ("name_phosphinic_acid", "phosphinic_acid", "C1CCCCC1P(C)(=O)O",
     "hexyl", "cyclohexyl(methyl)phosphinic acid"),
    ("name_arsinic_acid", "arsinic_acid", "C1CCCCC1[As](C)(=O)O",
     "hexyl", "cyclohexyl(methyl)arsinic acid"),
])
def test_cyclic_substituent_is_never_renamed_as_a_linear_chain(
    rules_fn, fg, smiles, wrong_token, expected,
):
    """REGRESSION (pre-existing accuracy bug found in Phase B) -- STRENGTHENED.

    ``_characterize_substituent`` counts the carbons of a non-aromatic subtree
    as a linear chain, so a cyclic substituent was silently renamed to a WRONG
    CONSTITUTION: benzyl(methyl)phosphinic acid -> 'heptyl(methyl)phosphinic
    acid', cyclohexyl(methyl) -> 'hexyl(methyl)'.

    Phase B stopped that by REFUSING. a phase stops it by naming the ring
    correctly through the shared chokepoint. The guarantee is the same and is
    now checked from BOTH sides: the fabricated linear token must be absent AND
    the correct name must be present. Only the second half would let a
    'heptyl'-free but still wrong name through; only the first half would be
    satisfied by an abstention.

    The 'hexyl' probe strips 'cyclohexyl' first -- the correct name legitimately
    contains that substring, and a naive ``'hexyl' not in name`` would be
    vacuously unfalsifiable here.
    """
    from orthonym.rules import phosphorus
    mol = Chem.MolFromSmiles(smiles)
    matches = detect_functional_groups(mol).get(fg, [])
    assert matches, f"{smiles} should perceive {fg}"
    result = getattr(phosphorus, rules_fn)(mol, matches[0])
    assert result is not None, f"{rules_fn} must now name {smiles}"
    assert wrong_token not in result.replace("cyclohexyl", ""), (
        f"{rules_fn} fabricated a linear '{wrong_token}' chain for {smiles}: "
        f"{result!r}"
    )
    assert result == expected


@pytest.mark.parametrize("smiles,expected", [
    ("C1=CC=C(C=C1)C[As](=O)(O)O", "benzylarsonic acid"),
    ("C1=CC=C(C=C1)C[Sb](=O)(O)O", "benzylstibonic acid"),
])
def test_benzyl_case_names_correctly_even_without_the_opsin_gate(
    monkeypatch, smiles, expected,
):
    """Correctness here must not depend on Java being installed.

    The OPSIN validity gate fails OPEN when no JAR is present, so a guarantee
    that only holds inside the gate is not a guarantee. Before Phase B the
    engine emitted the constitutionally-WRONG 'methylbenzene' for benzylarsonic
    acid and relied entirely on to suppress it.

    Naming CORRECTLY with the gate disabled is the strongest form of that
    contract -- stronger than the refusal this test used to assert, because a
    refusal is also what a broken producer yields. The historical wrong tokens
    stay asserted so the specific old leak cannot come back.
    """
    import orthonym.namer as _namer
    from orthonym.errors import is_failure_name
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", True, raising=False)
    name = Orthonym().name(smiles)
    assert not is_failure_name(name), f"ungated output must name, got {name!r}"
    assert "methylbenzene" not in name and "toluene" not in name
    assert name == expected


@pytest.mark.parametrize("smiles,element,shipped", [
    ("OCCP(=O)(O)O",    "phosphon", "2-hydroxyethane-1-phosphonic acid"),
    # NOTE the elision difference from its P and Sb siblings ('ethan-1-' vs
    # 'ethane-1-'): a separate pre-existing defect in the generic assembler,
    # pinned here so the PIN upgrade below cannot quietly paper over it.
    ("OCC[As](=O)(O)O", "arsonic",  "2-hydroxyethan-1-arsonic acid"),
    ("OCC[Sb](=O)(O)O", "stibonic", "2-hydroxyethane-1-stibonic acid"),
])
def test_heteroatom_organyl_defers_to_the_generic_path_on_every_element(
    gated_namer, smiles, element, shipped,
):
    """A substituent the namer cannot prove defers, it does not abstain.

    ``organyl_prefix_name`` is bounded to C/H/halogen fragments, so a
    heteroatom-bearing organyl still returns None; the handler defers and the
    GENERIC suffix assembler emits the parent-hydride-stem form
    (``2-hydroxyethane-1-phosphonic acid``, ``2-hydroxyethan-1-arsonic acid``).
    Those are structure-correct and OPSIN-exact, and they are pre-existing
    shipped behaviour on phosphorus, so this test pins that As/Sb behave the SAME
    as P rather than diverging in either direction.

    NOT THE PIN -- caveat CONFIRMED against the Blue Book (, 2026-07-26).
    The earlier version of this docstring asserted the PIN without a citation;
    the citations exist and are stronger than claimed:

      * BB 35461 (verbatim): ``C2H5-P(O)(OH)2 ethylphosphonic acid (PIN) (not
        ethanephosphonic acid)`` -- the Blue Book explicitly REJECTS the
        parent-hydride-stem shape that these three names use.
      * BB 36540 (verbatim): ``[2-(methoxysulfonyl)phenyl]phosphonic acid (PIN)
        (an acid is senior to an ester)`` -- a HETEROATOM-bearing organyl cited
        as an enclosed prefix on a phosphonic acid, with 's seniority
        reasoning spelled out. So ``(2-hydroxyethyl)phosphonic acid`` is the PIN.
      * BB 36540 also gives ``(HO)2P(O)-CH2-COOH -> phosphonoacetic acid (PIN)``
        and BB 36542 ``(4-arsonobutyl)phosphonic acid (PIN) (a phosphorus acid is
        senior to an arsenic acid)``: when the organyl carries an acid SENIOR to
        the hub, the hub becomes the ``phosphono``/``arsono`` prefix instead.
        Orthonym already emits ``phosphonoacetic acid`` correctly by deferring,
        which is exactly why the guard must keep refusing that shape.

    NOT ACHIEVABLE BY WIDENING THIS GUARD -- MEASURED, not assumed. The obvious
    reading is that the C/H/halogen bound is what blocks the PIN, and it is
    wrong. Probed directly :

      * With ``_fragment_atoms`` widened to admit heteroatoms, the producer
        returns the PIN outright: ``name_phosphonic_acid`` -> '(2-hydroxyethyl)
        phosphonic acid', ``name_arsonic_acid`` -> '(2-hydroxyethyl)arsonic
        acid'. So the guard and the chokepoint are BOTH already capable.
      * But for these molecules the producer is invoked **zero times** --
        the shipped '2-hydroxyethane-1-phosphonic acid' comes from a different
        path entirely. Widening the guard therefore changes nothing here; a
        mutation test that widened it left this whole module green.

    The blocker is UPSTREAM, in which handler claims a polyfunctional oxoacid --
    the secondary blocker recorded in the a phase design (``p44_scorer``
    Pre-empt 2 / ``parent_selection.SKELETAL_SUFFIX_PGS``), assigned to a phase.
    Closing it is a DISPATCH change, so it needs the full gate, not this
    boundary. The guard-side half is also still required and is specified: a
    hub-seniority parameter admitting a heteroatom fragment iff every
    characteristic group lying ENTIRELY inside it is strictly junior to the hub's
    own acid (``rules.seniority.compare_seniority`` already ranks phosphonic 26 <
    phosphinic 27 < arsonic 28 < arsinic 29 < stibonic 30 < stibinic 31 against
    primary_alcohol 93), which is what keeps ``phosphonoacetic acid`` deferring.

    Until then this test pins current behaviour EXACTLY, so
    that the upgrade has to come here and replace these strings deliberately --
    the old ``element in name`` check would have passed unchanged on
    ``(2-hydroxyethyl)phosphonic acid`` and hidden the very change it describes.
    """
    name = gated_namer.name(smiles)
    assert element in name, f"{smiles} -> {name!r}"
    assert name == shipped, (
        f"{smiles}: shipped non-PIN form moved. If this is the "
        f"(2-hydroxyethyl)... PIN upgrade described above, update the expectation "
        f"and the docstring together; got {name!r}"
    )


@pytest.mark.parametrize("smiles", [
    "CCP(=O)O",     # ethylphosphinic acid (P sibling)
    "CC[As](=O)O",  # ethylarsinic acid (As)
    "CC[Sb](=O)O",  # ethylstibinic acid (Sb, BB L36064 PIN)
])
def test_monosubstituted_inic_acid_is_a_symmetric_shared_gap(gated_namer, smiles):
    """AUDITED PRE-EXISTING GAP, deliberately left symmetric.

    BB L36064 gives ``ethylstibinic acid`` as a PIN, so a MONO-substituted
    -inic acid (one C + one H on the central atom) is a real class member.
    The shipped phosphorus path has never covered it -- ``CCP(=O)O`` perceives
    no functional group and fails closed -- and this phase mirrors that scope
    rather than making As/Sb silently better than P. Widening it must be done
    for all three elements at once.

    This test exists to PIN the symmetry: if any one element starts naming
    mono-substituted -inic acids, all three must.
    """
    from orthonym.errors import is_failure_name
    assert is_failure_name(gated_namer.name(smiles))


def test_complex_organyl_never_named_as_the_bare_parent(gated_namer):
    """The specific historical leak: dropping the ring and naming the rest.

    Kept verbatim through the contract change -- the leak it guards
    (emitting the benzyl fragment's own parent and silently discarding the
    arsonic acid) is an atom-dropping WRONG name whether the handler refuses or
    names, so the guard is orthogonal to which of the two it does. The exact
    name is pinned as well now that there is one.
    """
    name = gated_namer.name("C1=CC=C(C=C1)C[As](=O)(O)O")
    assert "methylbenzene" not in name
    assert "toluene" not in name
    assert name == "benzylarsonic acid"


# --------------------------------------------------------------------------
# 6. Element routing: genuine organometallics and organo-pnictogen hydrides
# are NOT stolen by the new acid path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("C[As](C)C",       "trimethylarsane"),
    ("C[Sb](C)C",       "trimethylstibane"),
    ("CC[As](CC)CC",    "triethylarsane"),
    ("Cl[As](Cl)Cl",    "trichloroarsane"),
    ("Br[Sb](Br)Br",    "tribromostibane"),
])
def test_organo_pnictogen_hydrides_still_named(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize("smiles", [
    "[cH-]1cccc1.[cH-]1cccc1.[Fe+2]",   # ferrocene
    "C[Li]",                            # methyllithium
    "[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Fe]",  # Fe(CO)5
])
def test_genuine_organometallics_still_route_to_p69(smiles):
    """The ORGANOMETALLIC@50 predicate must still claim real organometallics."""
    from orthonym.routing.dispatch_table import DISPATCH_TABLE, StoutClass
    mol = Chem.MolFromSmiles(smiles)
    entry = DISPATCH_TABLE[StoutClass.ORGANOMETALLIC]
    assert entry.predicate(mol, smiles, Chem.CanonSmiles(smiles), None) is True


@pytest.mark.parametrize("smiles", [
    "C[As](=O)(O)O",
    "C[Sb](=O)(O)O",
    "C[As](C)(=O)O",
])
def test_pnictogen_oxoacids_are_not_organometallic(smiles):
    """An arsonic acid is a oxoacid, never a organometallic."""
    from orthonym.routing.dispatch_table import DISPATCH_TABLE, StoutClass
    mol = Chem.MolFromSmiles(smiles)
    entry = DISPATCH_TABLE[StoutClass.ORGANOMETALLIC]
    assert entry.predicate(mol, smiles, Chem.CanonSmiles(smiles), None) is False


# --------------------------------------------------------------------------
# 7. Honesty of the descriptive fallback: a carbon-bearing arsenic compound
# is not "inorganic" -- and, since, is not a fallback at all
# --------------------------------------------------------------------------

def test_organo_arsenic_compound_is_named_not_labelled(gated_namer):
    """C7H9AsO3 bearing a benzyl group is an ORGANO-arsenic compound.

    RE-DERIVED . This test's premise -- that the molecule FAILS and
    therefore gets a descriptive label -- is gone: the producer now names
    it. Its guarantee is re-expressed at full strength.

    The dropped assertion was ``"arsenic" in name``, which asserted that the
    descriptive FALLBACK said 'arsenic compound' rather than 'inorganic
    compound'. It cannot be kept: there is no fallback string to inspect, and
    the real name legitimately says 'arsonic', not 'arsenic'. What the old
    assertion was protecting -- that a carbon-bearing arsenic compound is never
    mislabelled -- is now protected more strongly by pinning the actual name and
    by asserting no descriptive-label vocabulary appears at all.
    """
    from orthonym.errors import is_failure_name
    name = gated_namer.name("C1=CC=C(C=C1)C[As](=O)(O)O")
    assert not is_failure_name(name), f"must be named, got {name!r}"
    assert name == "benzylarsonic acid"
    for label in ("inorganic", "compound", "not supported", "unknown"):
        assert label not in name, (
            f"a named organo-arsenic compound must carry no descriptive label "
            f"({label!r}): {name!r}"
        )


def test_carbon_free_antimony_fallback_label_is_preserved():
    """The no-carbon case is genuinely inorganic -- canary must not move."""
    from orthonym.errors import classify_failure_limit
    mol = Chem.MolFromSmiles("[O]=[Sb]([O-])([O-])[OH]")
    assert classify_failure_limit(mol).message == "antimony compound (not supported)"
