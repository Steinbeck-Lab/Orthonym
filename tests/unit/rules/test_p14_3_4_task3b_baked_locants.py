"""§ where the locant was BAKED INTO THE PREFIX TEXT before any licence ran.

 Phase C Task 3b. Five defects, one shared root cause: a producer that renders
``{locants}-{prefix}`` into a string, so the licence downstream has nothing
left to withhold. Measured at HEAD `` with a trace validated on two known
positives (``chloropropanedioic acid``, ``chlorobutanedioic acid`` -> licence True)
and a known negative (``ethanol`` -> zero calls at every site):

====== ========================================= ==================================
defect producer that baked the locant HEAD -> required
====== ========================================= ==================================
A ``composer._generate_alkyl_prefixes`` ``2-methylpropanedioic acid`` ->
                                                   ``methylpropanedioic acid``
B ``heterocycles._format_c_substituent`` ``2-chloropyrazine`` ->
                                                   ``chloropyrazine``
C ``polyfunctional`` FG-prefix loop ``2-aminopropanedioic acid`` ->
                                                   ``aminopropanedioic acid``
D ``_name_ether_substituted_chain`` hand-rolled private licence
E same ``bis(methoxy)`` -> ``dimethoxy``
====== ========================================= ==================================

(User ruling D1, 2026-10-02: defect B's pyrazine rows now cite the locant again -- a
prefix on a mancude heteromonocycle, '2-[(pyridin-3-yl)oxy]pyrazine (PIN)' the Blue Book;
the licence keeps the saturated one-orbit ring, 'phenyloxirane (PIN)' the Blue Book.)

★ WHY THE GUARD ROWS ARE THE REAL TEST. For defect A the licence at HEAD refused at a
"prefix text begins with a digit" guard, so ``2-methylpentanedioic acid`` was right by
ACCIDENT -- the guard refused every alkyl prefix, correct and incorrect alike. After
the fix that row is decided by the orbit predicate instead, and it must still keep its
locant: pentanedioic acid's C2/C4 are one orbit and C3 is another, so there is more
than one kind of substitutable hydrogen. Same for defect B: pyridine's CH are THREE
orbit classes (2/6, 3/5, 4), which is the only thing separating ``chloropyrazine``
from ``4-chloropyridine``. A "symmetric heterocycle => omit" rule passes the target
rows and fails these.

Blue Book evidence, all verified by line:

* ``:2947`` ``chlorocoronene (PIN)`` -- L3's own printed PREFIX positive on a
  symmetrical ring parent (defect B's exact shape).
* ``:2951`` ``chloropropanedioic acid (PIN) chloromalonic acid`` and ``:2883``
  ``chlorobutanedioic acid (PIN)`` -- defects A and C's shape. ``:4973``
  ``propanedioic acid (PIN) malonic acid`` confirms the systematic parent is the PIN,
  and ``tartronic``/``aminomalonic`` appear ZERO times in the Blue Book, so no
  retained name pre-empts defect C's two targets.
* ``:29834`` ``2,3-dihydroxybutanedioic acid (PIN)`` -- the DIsubstituted control:
  L3 requires monosubstitution, so this row must keep both locants.
* ``:16408`` ``diphenylmethyl (preferred prefix) (not benzhydryl)`` + ``:55912``
  (``(C6H5)2CH-``, -- defect D. A ``methyl`` substituent group with 2 of
  its 3 substitutable hydrogens replaced is PARTIAL substitution, so does
  not apply; the locant-free preferred prefix is licensed by ****
  (``:3031``) because all three hydrogens share locant 1. ⇒ The brief's claim that
  ``:3009`` REQUIRES a locant here is refuted by the Blue Book's own preferred prefix.
* ``:5098`` ``1,1-dimethoxypropane (PIN)`` vs ``:35344``
  ``1,1-bis(methylsulfanyl)pentane (PIN)`` -- defect E's boundary pair: one skeleton
  shape, ``di`` + bare stem for the CONTRACTED ``methoxy`` (``:17958``, verbatim
  "from a contracted name, for example 'methoxy'"), ``bis(...)`` for the uncontracted
  ``methylsulfanyl``. ``bis(methoxy)`` occurs ZERO times in the Blue Book.
"""
import re
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.locant_omission import l3_one_kind_of_substitutable_h

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"test fixture SMILES did not parse: {smiles}"
    return m


# --------------------------------------------------------------------------- #
# 1. THE PREDICATE decides the guards -- orbits, never a molecule class #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "smiles,label,expected",
    [
        # Defect A / C targets: the parent COMPOUND, both -COOH included.
        ("OC(=O)CC(=O)O", "propanedioic acid -- C2 only (:3007 excludes acid O-H)", True),
        ("OC(=O)CCC(=O)O", "butanedioic acid -- C2/C3 one orbit (:2883)", True),
        # ★ THE DEFECT-A GUARD. No rule about diacids produces this answer.
        ("OC(=O)CCCC(=O)O", "pentanedioic acid -- C2/C4 vs C3 = two orbits", False),
        ("OC(=O)CCCCCC(=O)O", "heptanedioic acid -- three orbits", False),
        # Defect B target and guards.
        ("c1cnccn1", "pyrazine -- four equivalent CH", True),
        ("c1ccncc1", "pyridine -- C2/C6, C3/C5, C4 = THREE orbits", False),
        ("c1cnccn1".replace("cnccn", "cncnc"), "pyrimidine -- three orbits", False),
        ("C1CCNCC1", "piperidine -- four orbits", False),
        # Defect C's own guard: a monoacid's C2/C3 differ.
        ("OC(=O)CC", "propanoic acid -- C2 (2H) vs C3 (3H)", False),
    ],
)
def test_orbit_predicate_decides_every_guard(smiles, label, expected):
    assert l3_one_kind_of_substitutable_h(_mol(smiles)) is expected, label


# --------------------------------------------------------------------------- #
# 2. DEFECT A -- alkyl prefixes on a chain parent (composer) #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("OC(=O)C(C)C(=O)O", "methylpropanedioic acid"),
        ("OC(=O)CC(C)C(=O)O", "methylbutanedioic acid"),
        ("OC(=O)C(CC)C(=O)O", "ethylpropanedioic acid"),
        ("OC(=O)C(CC)CC(=O)O", "ethylbutanedioic acid"),
    ],
)
def test_defect_a_alkyl_prefix_locant_omitted(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_defect_a_root_cause_pair_same_parent_two_producers():
    """★ THE ROOT CAUSE IN ONE PAIR, on a parent neither the brief nor the Blue Book
    example block mentions -- found by the blast-radius sweep.

    ``propanedinitrile`` has exactly one kind of substitutable hydrogen (C2; a nitrile
    carries no N-H, unlike the ``-diamide`` family whose amide N-H give it a second
    kind and keep its locant). At HEAD the two rows below DISAGREED::

        N#CC(Cl)C#N chloropropanedinitrile <- `chloro` arrives as bare text
        N#CC(C)C#N 2-methylpropanedinitrile <- `2-methyl` arrives BAKED

    Same molecule class, same parent, same one-prefix-one-locant scope; the only
    difference was which producer rendered the prefix. That is the defect, and it is
    why the fix had to be in the THREADING and not in a spelling rule."""
    n = Orthonym()
    assert n.name("N#CC(Cl)C#N") == "chloropropanedinitrile"
    assert n.name("N#CC(C)C#N") == "methylpropanedinitrile"
    assert n.name("N#CC(C)CC#N") == "methylbutanedinitrile"
    # And the family that must NOT move: an amide N-H IS substitutable (the Blue Book
    # `N1,N3-dimethylpropanediamide (PIN)`), so the diamide keeps its locant.
    assert n.name("CNC(=O)CC(=O)NC") == "N1,N3-dimethylpropanediamide"


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # ★ THE GUARD. Was right by accident at HEAD (refused at the digit guard);
        # is now decided by the orbit predicate.
        ("OC(=O)CCC(C)C(=O)O", "2-methylpentanedioic acid"),
        ("OC(=O)CCCCC(C)C(=O)O", "2-methylheptanedioic acid"),
        # Not a diacid at all -- the alkyl prefix path must be untouched here.
        ("OC(=O)C(C)C", "2-methylpropanoic acid"),
        ("CC(C)C", "2-methylpropane"),
        ("CCCC(C)C", "2-methylpentane"),
    ],
)
def test_defect_a_guards_keep_their_locant(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --------------------------------------------------------------------------- #
# 3. DEFECT B -- C-substituent prefixes on a heterocycle #
# --------------------------------------------------------------------------- #
# User ruling D1 (2026-10-02): a prefix on a MANCUDE heteromonocycle cites its locant
# even with one kind of substitutable hydrogen -- (the Blue Book) prints
# '(1) 2-[(pyridin-3-yl)oxy]pyrazine (PIN)' (the Blue Book). The licence still fires for the
# saturated one-orbit ring (the oxirane pair below) and for the pyrazine SUFFIX.
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("Clc1cnccn1", "2-chloropyrazine"),
        ("Cc1cnccn1", "2-methylpyrazine"),
    ],
)
def test_defect_b_mancude_heteroring_prefix_cites_its_locant(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # ★ A PRINTED BLUE BOOK PIN the brief did not mention, found by the targeted
        # blast-radius sweep: the Blue Book's table row is
        # `| 7 | oxirane (PIN) | substitutive | phenyloxirane (PIN) | |`
        # -- a MONOsubstituted oxirane with the locant omitted. HEAD emitted
        # `2-phenyloxirane`, so defect B's fix corrects a row against the Blue Book's
        # own PIN. Oxirane's two ring CH2 are one orbit.
        ("c1ccccc1C1CO1", "phenyloxirane"),
        ("ClC1CO1", "chlorooxirane"),
        ("CC1CO1", "methyloxirane"),
        #...and its printed DIsubstituted counterpart, the Blue Book
        # `2-ethyl-2-methyloxirane (PIN)`, which must KEEP both locants because L3
        # requires monosubstitution. The two rows together are a boundary pair.
        ("CCC1(C)CO1", "2-ethyl-2-methyloxirane"),
    ],
)
def test_defect_b_oxirane_printed_bluebook_pair(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # ★ THE PAIR that separates the orbit predicate from "symmetric ring".
        ("Clc1ccncc1", "4-chloropyridine"),
        ("Clc1ccccn1", "2-chloropyridine"),
        ("Clc1cccnc1", "3-chloropyridine"),
        ("Cc1ccncc1", "4-methylpyridine"),
        ("Clc1ncccn1", "2-chloropyrimidine"),
        # DIsubstituted pyrazine: one orbit, but not MONOsubstituted.
        ("Cc1cnc(C)cn1", "2,5-dimethylpyrazine"),
        ("Clc1ccc(Cl)cn1", "2,5-dichloropyridine"),
        # An N-substituent carries an essential ring-N locant.
        ("CN1CCCC1", "1-methylpyrrolidine"),
        # The ring-SUFFIX sibling licence must be unchanged in both directions.
        ("OC(=O)c1cnccn1", "pyrazinecarboxylic acid"),
        ("N#CN1CCCCC1", "piperidine-1-carbonitrile"),
        ("N#Cc1ccncc1", "pyridine-4-carbonitrile"),
        ("OC(=O)c1ccncc1", "pyridine-4-carboxylic acid"),
    ],
)
def test_defect_b_guards_keep_their_locant(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --------------------------------------------------------------------------- #
# 4. DEFECT C -- the THIRD handler (polyfunctional) #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("OC(=O)C(N)C(=O)O", "aminopropanedioic acid"),
        ("OC(=O)C(O)C(=O)O", "hydroxypropanedioic acid"),
        ("OC(=O)C(F)C(=O)O", "fluoropropanedioic acid"),
        ("OC(=O)C(O)CC(=O)O", "hydroxybutanedioic acid"),
    ],
)
def test_defect_c_polyfunctional_prefix_locant_omitted(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [
        #:29834 verbatim (PIN) -- DIsubstituted, so L3 (monosubstituted) denies.
        ("OC(=O)C(O)C(O)C(=O)O", "2,3-dihydroxybutanedioic acid"),
        # A monoacid parent: C2 and C3 are different orbits.
        ("OC(=O)C(O)C", "2-hydroxypropanoic acid"),
        # The retained acetic-acid path must be untouched.
        ("OCC(=O)O", "hydroxyacetic acid"),
        ("N#CCC(=O)O", "cyanoacetic acid"),
        ("SCC(=O)O", "sulfanylacetic acid"),
        # Retained amino-acid names must still pre-empt the systematic form -- for an
        # input that defines the configuration they imply. A bare retained name asserts
        # the L configuration 'The stereodescriptors D and L',
        # the Blue Book: 'The stereodescriptor xi indicates unknown
        # configuration'; OPSIN reads 'aspartic acid' as the L isomer, full InChIKey
        # an InChIKey), so the stereo-free input takes the systematic
        # name (9292c013d, 6ed78b1e4), and with a single substituted position the
        # locant is omitted like 'chlorobutanedioic acid (PIN)',:2883).
        ("NCC(=O)O", "glycine"),
        ("OC(=O)[C@@H](N)CC(=O)O", "L-aspartic acid"),
        ("OC(=O)C(N)CC(=O)O", "aminobutanedioic acid"),
    ],
)
def test_defect_c_guards(namer, smiles, expected):
    assert namer.name(smiles) == expected


# --------------------------------------------------------------------------- #
# 5. DEFECTS D + E -- the ether-substituted chain #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # E: `di`, not `bis`, for the contracted `methoxy` (:5098 /:17958).
        ("COC(OC)c1ccccc1", "(dimethoxymethyl)benzene"),
        ("COC(OC)C1CCCCC1", "(dimethoxymethyl)cyclohexane"),
        ("COC(OC)(OC)c1ccccc1", "(trimethoxymethyl)benzene"),
        # E, the other side of the boundary: the UNCONTRACTED `methylsulfanyl` is a
        # substituted prefix and keeps `bis(...)` per:35344 / (a). At HEAD
        # this emitted `bis((methylsulfanyl))methylbenzene` -- double-enclosed, and
        # rejected by the OPSIN grammar check. The outer marks were then missing
        # (`bis(methylsulfanyl)methylbenzene`, a `rules/benzene.py` defect this row
        # was asserted as-emitted to expose): the monosubstituted benzene branch now
        # encloses a compound prefix by enclose_if_compound (bridged fused S3, fix
        # a performance pass), (the Blue Book) with the next mark of
        # (:7446), as in 'bis[bis(trimethylsilyl)methyl]stannanol (PIN)' (:38204).
        ("CSC(SC)c1ccccc1", "[bis(methylsulfanyl)methyl]benzene"),
    ],
)
def test_defect_e_multiplier_and_enclosure(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # A ONE-carbon backbone omits, all H at locant 1).
        ("COCc1ccccc1", "(methoxymethyl)benzene"),
        # A TWO-carbon backbone CITES: `*CC` has substitutable H at locants 1 and 2,
        # so does not fire and is partial.
        ("COCCc1ccccc1", "(2-methoxyethyl)benzene"),
        ("COCC1CCCCC1", "(methoxymethyl)cyclohexane"),
    ],
)
def test_defect_d_backbone_length_boundary(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_defect_d_licence_is_the_predicate_not_the_backbone_length():
    """The replaced condition was ``len(backbone) > 1``. The licence that actually
    licenses the one-carbon case is measured on the substituent's PARENT
    HYDRIDE, and it must AGREE with the old answer on both lengths -- that agreement
    is the evidence the old condition was right for the wrong reason, and the reason
    it may be replaced without moving a name."""
    from orthonym.assembly.locant_omission import (
        l6_all_substitutable_h_share_one_locant,
    )
    from orthonym.assembly.substituent_naming import _l5_chain_parent_hydride

    for chain_len, expected in ((1, True), (2, False), (3, False)):
        hydride = _l5_chain_parent_hydride(chain_len, 1)
        assert hydride is not None
        a2l = {a.GetIdx(): a.GetIdx() + 1 for a in hydride.GetAtoms()}
        assert l6_all_substitutable_h_share_one_locant(hydride, a2l) is expected, (
            f"chain_len={chain_len}")


# --------------------------------------------------------------------------- #
# 6. THE TWO RENDERERS MUST AGREE (the divergence a print-time flag caused) #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "smiles,expected_name",
    [
        # ★ THE TWO ROWS THAT FAIL AT HEAD. Measured 2026-07-30 in a worktree at
        # c8d80c07: both report class_id 'coarse_fallback'.
        ("OC(=O)C(Cl)C(=O)O", "chloropropanedioic acid"),
        ("OC(=O)CC(Cl)C(=O)O", "chlorobutanedioic acid"),
        # Newly licensed by this task -- must land in the same state, not merely agree.
        ("OC(=O)C(C)C(=O)O", "methylpropanedioic acid"),
        ("OC(=O)CC(C)C(=O)O", "methylbutanedioic acid"),
        # The guard: agreement must hold while the locant is KEPT.
        ("OC(=O)CCC(C)C(=O)O", "2-methylpentanedioic acid"),
        # The sibling, which has applied its licence to the fragments
        # since Task 5b and therefore already agreed at HEAD.
        ("OC(=O)C(F)(F)C(F)(F)F", "pentafluoropropanoic acid"),
    ],
)
def test_licensed_rows_keep_their_structured_tree(smiles, expected_name):
    """The licence must reach the NAME TREE, not just the legacy string.

    ``general_acyclic`` is the sole member of ``SERIALIZER_PRODUCTION_CLASSES``, so the
    name-tree serializer IS its production composition site. The licence used to be
    applied by a print-time flag threaded into the legacy assembler only, so the two
    renderers disagreed and ``composer._serializer_flip_or_name`` silently kept the
    legacy string.

    ⚠ THE OBSERVABLE IS THE CLASS_ID, NOT THE SERIALIZED STRING, and getting that
    wrong is how this test was blind on first writing. ``namer.py:2166`` holds a
    staleness guard that REPLACES any tree failing ``name_tree_to_string(tree) ==
    name`` with a synthetic ``class_id="coarse_fallback"`` node -- so an assertion of
    that equality is true **by construction for every SMILES** and can never fail.
    What the divergence actually destroys is the structured tree: measured at HEAD,
    ``chloropropanedioic acid`` and ``chlorobutanedioic acid`` come back as
    ``coarse_fallback`` (losing the parent/suffix/prefix fields, the per-substring
    scorer's input and ``--dump-tree``), while the same molecules' undivergent
    siblings keep ``general_acyclic``. Asserting the class is what fails at HEAD.
    """
    from orthonym import name_with_tree
    from orthonym.assembly.name_tree_to_string import (
        SERIALIZER_PRODUCTION_CLASSES,
        name_tree_to_string,
    )

    result = name_with_tree(smiles)
    assert result.name == expected_name
    tree = getattr(result, "tree", None)
    assert tree is not None
    assert getattr(tree, "class_id", "") in SERIALIZER_PRODUCTION_CLASSES, (
        f"{smiles}: tree degraded to {getattr(tree, 'class_id', None)!r} -- the "
        f"legacy assembler and the name-tree serializer disagree on this row")
    # Entailed by the staleness guard above, kept as an explicit statement of the
    # property the class_id assertion is standing in for.
    assert name_tree_to_string(tree, style="pin") == result.name


# --------------------------------------------------------------------------- #
# 7. THE THIRD SCOPE CHECK IS CONSULTED AT ALL THREE SITES #
# --------------------------------------------------------------------------- #
def test_fragment_boundary_observation_reads_the_visited_set():
    """The observation itself: empty visited set => this IS the whole molecule."""
    from orthonym.assembly.fragment_naming import _get_visited
    from orthonym.assembly.handlers._handler_shared import (
        locant_scope_is_a_name_component,
    )

    visited = _get_visited()
    assert not visited, "test precondition: no nested naming in progress"
    assert locant_scope_is_a_name_component() is False
    visited.add("O=C(O)C(Cl)C(=O)O")
    try:
        assert locant_scope_is_a_name_component() is True
    finally:
        visited.discard("O=C(O)C(Cl)C(=O)O")


@pytest.mark.parametrize(
    "smiles,licensed,cited",
    [
        # general_acyclic (defect A + the pre-existing chloro rows)
        ("OC(=O)C(C)C(=O)O", "methylpropanedioic acid", "2-methylpropanedioic acid"),
        ("OC(=O)C(Cl)C(=O)O", "chloropropanedioic acid", "2-chloropropanedioic acid"),
        # heterocycle (defect B; the saturated one-orbit ring, the Blue Book -- a prefix on
        # mancude pyrazine cites its locant, user ruling D1)
        ("ClC1CO1", "chlorooxirane", "2-chlorooxirane"),
        # polyfunctional (defect C)
        ("OC(=O)C(O)C(=O)O", "hydroxypropanedioic acid", "2-hydroxypropanedioic acid"),
    ],
)
def test_every_site_consults_the_fragment_boundary_observation(
    namer, monkeypatch, smiles, licensed, cited,
):
    """ licences that empty a scope of ALL its locants must consult THREE
    things, and a licence consulting two of three is broken on the third's whole class
    (`internal notes`).
    The other two are ambient ContextVars checked inside ``locant_omission``; this one
    is the read-only fragment-boundary observation.

    ⚠ This is a REACHABILITY test, not a mutation-coverage test. Forcing the
    observation True must bring every locant back at all three sites -- if a site did
    not consult it, its row would still elide and this would fail. Written this way
    because no corpus molecule was found that reaches these three handlers *through*
    the decomposition engine (probed: five acyl-piperidine / ring-amide shapes, all
    abstain today), so the guard has no end-to-end witness and a plain mutation of the
    guard line SURVIVES. Monkeypatching the observation is the honest substitute.
    """
    from orthonym.assembly.handlers import _handler_shared

    assert namer.name(smiles) == licensed, "baseline: the licence fires here"
    monkeypatch.setattr(
        _handler_shared, "_naming_call_produces_a_name_component", lambda: True)
    monkeypatch.setattr(
        _handler_shared, "locant_scope_is_a_name_component", lambda: True)
    assert Orthonym().name(smiles) == cited


# --------------------------------------------------------------------------- #
# 8. TRIPWIRES on the two things this task removed #
# --------------------------------------------------------------------------- #
def test_no_private_multiplier_table_in_the_ether_chain_producer():
    """``_name_ether_substituted_chain`` carried one of the 27 divergent multiplier
    tables (`internal notes`),
    hardcoded to ``bis``/``tris`` so it could never emit ``di``. The multiplier now
    comes from the shared ``multiplied_component``. This fails if a private table
    reappears in that function."""
    src = (PROJECT_ROOT / "src/orthonym/assembly/substituent_naming.py").read_text()
    start = src.index("def _name_ether_substituted_chain(")
    end = src.index("\ndef ", start + 1)
    # CODE only: the function's own comments discuss the removed table by name, and
    # matching those would make this tripwire fire on its own documentation.
    body = "\n".join(
        line for line in src[start:end].splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "_MULT" not in body, "a private multiplier table came back"
    assert "multiplied_component" in body, (
        "the shared P-16.3 multiplier is no longer used")
    assert not re.search(r"cite_locants\s*=\s*len\(", body), (
        "the hand-rolled `cite_locants = len(backbone) > 1` licence came back")


def test_assemble_fragments_has_no_print_time_locant_flag():
    """The licence must not be re-introduced as a print-time parameter of
    ``_assemble_fragments``: that renderer is only ONE of the two reading the fragment
    list, so a flag there cannot reach the name-tree serializer."""
    import inspect

    from orthonym.assembly.handlers._handler_shared import _assemble_fragments

    params = inspect.signature(_assemble_fragments).parameters
    assert "l3_omit_prefix_locant" not in params, (
        "the print-time flag is back; apply the licence to the FRAGMENTS instead")
