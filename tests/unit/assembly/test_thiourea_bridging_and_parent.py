"""Residue R3: a BRIDGING thiourea core was spelled twice and named as a
monovalent prefix, producing a different molecule.

`CC(C)(C)NC(=S)NC1(CCCCC1)N=NC(C)(C)C` emitted

    1-tert-butyldiazenyl-1-(2-(carbamothioylamino)-2-carbamothioylamino-2-
    methylpropyl)cyclohexane

which spells the single thiourea unit TWICE and renders the ring->N bond as
ring->C. Traced by trace (not by reading): the substituent fragment handed to
``_check_substituent_prefix_form`` was the bare 4-atom core ``{N,C,S,N}`` with
**two** bonds leaving it -- one to the tert-butyl carbon, one to the ring
carbon. ``get_substituent_prefix_form`` then returned the STATIC string
``PREFIX_FORMS['thiourea']`` = ``carbamothioylamino``, which is the monovalent
group ``H2N-CS-NH-``. Naming a two-attachment bridge with a one-attachment
prefix orphans everything past the distal nitrogen; the orphan was then
re-attached elsewhere, inventing a C-C bond, and the core was consumed a second
time from the other direction.

The oxo sibling never had the defect because ``urea`` dispatches to a DYNAMIC
generator (``get_n_substituted_carbamoylamino_prefix``) whose
``_urea_distal_n`` returns None when both nitrogens are substituted -- measured
12/12 None on the urea analogue, which then correctly re-parents onto urea.
Thiourea never got that upgrade.

Blue Book (quotations verified in ``the Blue Book Blue Book``, cross-checked
online at <https://iupac.qmul.ac.uk/BlueBook/>):

* ** "Chalcogen analogues of urea and isourea"** (:33437), section
  **** (:33439): *"Chalcogen analogues of urea are named by
  functional replacement nomenclature using the prefixes 'thio', 'seleno', and
  'telluro'. Preferred IUPAC names use the letter locants N, and N'. Numerical
  locants may be used for thiourea in general nomenclature."* And (:33446)
  *"Numerical locants are no longer used for thiourea in the IUPAC preferred
  name."* So a `1,3-...thiourea` spelling is general nomenclature, NOT the PIN.
* The class exemplar, a `(PIN)` example of an N-substituted chalcogen urea
  named on the RETAINED PARENT (:33451): **`N-(butan-2-yl)selenourea (PIN)`**.
* ** "Seniority order for classes"**, Table 4.1: amides are class **11**
  (:18184), diazenes/azanes class **21** (:18197), carbon rings and chains
  class **40** (:18216). **** (:18875) selects the parent by that
  order. The "a ring outranks a chain" licence is gated on sameness of class
  -- **** (:24096) *"Within the same heteroatom class... a ring is
  always selected as the parent hydride"* -- so it cannot promote cyclohexane
  over the thiourea. **** (:33320) ranks urea *"as an amide of
  carbonic acid"*, and makes thiourea its retained chalcogen
  analogue. => **the thiourea is the parent; the ring is a substituent.**
* **** (:33487) enumerates only the UNSUBSTITUTED prefix
  ``carbamothioylamino`` (:33489) with the example
  ``3-(carbamothioylamino)propanoic acid (PIN)`` (:33501). No substituted-
  distal-N row exists anywhere in the book (grep proven against known
  positives). The substituted form is therefore derived across the
   chalcogen-replacement relationship from the oxo `(PIN)`
  example ``2-[(methylcarbamoyl)amino]naphthalene-1-carboxylic acid`` (:33354).

Every expected name below was round-tripped through OPSIN 2.9.0 to an
InChIKey identical to the input's before being asserted here.
"""

import pytest
from rdkit import Chem

import orthonym.namer as _namer
from orthonym import Orthonym


R3_SMILES = "CC(C)(C)NC(=S)NC1(CCCCC1)N=NC(C)(C)C"


@pytest.fixture(scope="module")
def namer():
    # Judge the GENERATOR, not the gate: with the validity gate on,
    # merely suppresses the wrong name instead of preventing its construction.
    _namer._DISABLE_VALIDITY_GATE = True
    return Orthonym(style="pin")


def _name(nm, smiles):
    r = nm.name_tiered(smiles)
    return r.get("name") if isinstance(r, dict) else r


def _formula(smiles):
    from rdkit.Chem import rdMolDescriptors
    return rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles))


# ---------------------------------------------------------------------------
# 1. The defect itself: one thiourea unit, spelled once.
# ---------------------------------------------------------------------------

def test_r3_thiourea_unit_is_not_spelled_twice(namer):
    """THE DEFECT. The single N-CS-N unit must contribute one thiourea morpheme.

    Judged with the validity gate OFF, so this measures the GENERATOR: the
    duplication has to stop being constructed, not merely be suppressed.
    """
    name = _name(namer, R3_SMILES)
    assert name is not None
    # 'carbamothioylamino' is the monovalent H2N-CS-NH- prefix. The molecule
    # holds ONE thiourea carbon; two occurrences means two thiourea units.
    assert name.count("carbamothioylamino") <= 1, name


def test_r3_abstains_rather_than_emitting_a_wrong_constitution():
    """R3 must produce NO name in production rather than a wrong molecule.

    Its verified PIN is

        N-tert-butyl-N'-[1-(tert-butyldiazenyl)cyclohexyl]thiourea

    (OPSIN 2.9.0 -> FJFCAHBBIQSNCA-UHFFFAOYSA-N, C15H30N4S, identical to the
    input). Reaching it needs the N'-substituent ``1-(tert-butyldiazenyl)
    cyclohexyl`` to be nameable, and this tree cannot yet name ANY
    1,1-disubstituted cycloalkyl substituent -- see the blocker test below.
    Until that gap closes the honest outcome is abstention, so
    ``_try_name_thiourea`` fails closed on an un-nameable N-substituent
    instead of dropping it.
    """
    from orthonym import Orthonym as _OS
    import orthonym.namer as _n
    saved = getattr(_n, "_DISABLE_VALIDITY_GATE", False)
    _n._DISABLE_VALIDITY_GATE = False
    try:
        row = _OS(style="pin").name_tiered(R3_SMILES)
    finally:
        _n._DISABLE_VALIDITY_GATE = saved
    # PIN tier spells an abstention as the sentinel + gate_outcome='suppressed';
    # what must never happen is a real name describing a different molecule.
    assert row.get("gate_outcome") == "suppressed", row
    assert row.get("name") in (None, "", "unknown organic compound"), row.get("name")
    assert row.get("is_pin") is False, row
    assert _formula(R3_SMILES) == "C15H30N4S"


def _features_for(smiles):
    """The MolecularFeatures the dispatcher hands the thiourea handler.

    Captured through the handler's own late-bound import so the object under
    test is the real one, and filtered to the WHOLE molecule (the pipeline also
    names sub-fragments through the same builder).
    """
    import orthonym.assembly.composer as C
    from orthonym import Orthonym as _OS
    n_atoms = Chem.MolFromSmiles(smiles).GetNumAtoms()
    grabbed = []
    orig = C._try_name_thiourea

    def spy(f):
        if getattr(f, "mol", None) is not None and f.mol.GetNumAtoms() == n_atoms:
            grabbed.append(f)
        return orig(f)

    C._try_name_thiourea = spy
    try:
        _OS(style="pin").name_tiered(smiles)
    finally:
        C._try_name_thiourea = orig
    return grabbed[0] if grabbed else None


def test_try_name_thiourea_refuses_r3_instead_of_dropping_the_ring_half():
    """GENERATOR-level. The gate must not be what saves us here.

    R3's N'-substituent is un-nameable. If the builder SKIPPED it (the way the
    urea builder does) it would emit ``N-tert-butylthiourea`` for a
    20-heavy-atom molecule — dropping the entire cyclohexyl/diazenyl half. The
    validity gate would suppress that, which is exactly why this assertion is
    made against the builder and not against the pipeline's output.
    """
    from orthonym.assembly.composer import _try_name_thiourea
    features = _features_for(R3_SMILES)
    assert features is not None, "the thiourea builder was never reached"
    assert _try_name_thiourea(features) is None


def test_name_r_group_never_spells_a_ring_as_a_straight_chain():
    """GENERATOR-level guard on the count anti-pattern.

    ``_name_r_group`` used to hand a ring-bearing fragment's carbon TALLY to
    ``get_alkyl_name``, which spells an ACYCLIC chain: the R3 N'-branch (6 ring
    + 4 chain carbons) came back ``'decyl'``, opening the ring into a chain — a
    different molecule that only the validity gate caught.
    """
    from orthonym.assembly.composer import _name_r_group
    mol = Chem.MolFromSmiles(R3_SMILES)
    core = {4, 5, 6, 7}                      # the N-CS-N core; atom 7 is the N'
    result = _name_r_group(mol, 8, core)     # the cyclohexyl attachment carbon
    # Either it declines, or it names the ring AS a ring — never a chain stem.
    assert result is None or "cyclo" in result, result


def test_blocker_1_1_disubstituted_cycloalkyl_substituent_is_unnameable():
    """``name_substituent_fragment`` (the Tier-4 recursive namer) declines a
    gem-disubstituted ring fragment.

    ⚠ CORRECTION (2026-08-02). This test's original docstring claimed the gap
    was general -- *"NOT thiourea-specific -- 1-methylcyclohexyl fails too"* --
    and that R3's PIN would be reachable the day it started failing. Both
    claims are REFUTED by measurement:

    * ``(1-methylcyclohexyl)methanol``, ``1-methylcyclohexan-1-amine``,
      ``1,1-dimethylcyclohexane`` and ``1-methylcyclohexane-1-carboxylic acid``
      all named correctly even when this test was written, so the gem-
      disubstituted cycloalkyl substituent was never globally unnameable.
    * ``N-(1-methylcyclohexyl)acetamide`` and ``N-(1-methylcyclohexyl)thiourea``
      now name (``), yet THIS assertion still holds -- because
      ``name_substituent_fragment`` is a DIFFERENT tier from the Tier-1.6
      ``decorated_ring_substituent_name`` that was actually repaired. A test
      going green is not evidence that the tier it exercises was the blocker.

    R3 remains unreachable for a further, unrelated reason: its ring decoration
    is a ``tert-butyldiazenyl`` group, which is outside the supported table in
    ``_ring_atom_simple_substituents`` (oxo, cyano, carboxy, halogen, hydroxy,
    alkoxy, amino, nitro, unbranched alkyl, alkoxycarbonyl). It abstains; it
    does not emit a wrong constitution.
    """
    from orthonym.assembly.substituent_naming import name_substituent_fragment
    m = Chem.MolFromSmiles("NC(=S)NC1(C)CCCCC1")          # 1-methylcyclohexyl
    assert name_substituent_fragment(m, list(range(4, 11)), 4, [0, 1, 2, 3]) is None
    # the unsubstituted ring is fine, which is why the plain cyclohexyl row passes
    m2 = Chem.MolFromSmiles("CC(C)(C)NC(=S)NC1CCCCC1")
    assert name_substituent_fragment(
        m2, list(range(8, 14)), 8, [4, 5, 6, 7]) == "cyclohexyl"


# ---------------------------------------------------------------------------
# 2. The fail-closed invariant at the producer, stated structurally.
# A bridging core has a bond leaving it from an atom other than attach_idx;
# a monovalent prefix may not name it. (Not an atom COUNT -- the test walks
# the actual bonds crossing the fragment boundary.)
# ---------------------------------------------------------------------------

def _bridging_thiourea_core(smiles):
    """Return (mol, core_atom_set, attach_idx) for the 4-atom N-CS-N core."""
    mol = Chem.MolFromSmiles(smiles)
    q = Chem.MolFromSmarts("[NX3][CX3](=[SX1])[NX3]")
    match = mol.GetSubstructMatches(q)[0]
    core = set(match)
    return mol, core, match[0]


def test_bridging_core_has_two_exits_and_must_not_get_a_monovalent_prefix():
    from orthonym.assembly.substituent_prefix_forms import (
        _check_substituent_prefix_form,
    )
    mol, core, attach = _bridging_thiourea_core("CC(C)(C)NC(=S)NC1CCCCC1")

    # structure proof: enumerate the bonds that actually cross the boundary
    exits = [
        (a, nbr.GetIdx())
        for a in core
        for nbr in mol.GetAtomWithIdx(a).GetNeighbors()
        if nbr.GetIdx() not in core
    ]
    assert len(exits) == 2, exits          # tert-butyl C and ring C

    # A two-exit bridge is not a monovalent substituent -> fail closed.
    assert _check_substituent_prefix_form(mol, core, attach) is None


def test_terminal_core_with_one_exit_still_gets_carbamothioylamino():
    """The Blue Book's own example must keep working:33501)."""
    from orthonym.assembly.substituent_prefix_forms import (
        _check_substituent_prefix_form,
    )
    mol, core, _ = _bridging_thiourea_core("NC(=S)NCCC(=O)O")
    exits = [
        (a, nbr.GetIdx())
        for a in core
        for nbr in mol.GetAtomWithIdx(a).GetNeighbors()
        if nbr.GetIdx() not in core
    ]
    assert len(exits) == 1, exits
    attach = exits[0][0]
    assert _check_substituent_prefix_form(mol, core, attach) == "carbamothioylamino"


# ---------------------------------------------------------------------------
# 3. The class: thiourea prefix with a SUBSTITUTED distal nitrogen.
# Derived across from the oxo (PIN) example at:33354.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # unsubstituted distal N -- the BB's own (PIN) example, must not regress
    ("NC(=S)NCCC(=O)O", "3-(carbamothioylamino)propanoic acid"),
    # mono-substituted distal N
    ("CNC(=S)NCCC(=O)O", "3-[(methylcarbamothioyl)amino]propanoic acid"),
    ("CC(C)(C)NC(=S)NCCC(=O)O",
     "3-[(tert-butylcarbamothioyl)amino]propanoic acid"),
    # di-substituted distal N
    ("CN(C)C(=S)NCCC(=O)O", "3-[(dimethylcarbamothioyl)amino]propanoic acid"),
])
def test_thiourea_prefix_family(namer, smiles, expected):
    assert _name(namer, smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    ("NC(=S)NCCC(=O)O", "carbamothioylamino"),
    ("CNC(=S)NCCC(=O)O", "(methylcarbamothioyl)amino"),
    ("CC(C)(C)NC(=S)NCCC(=O)O", "(tert-butylcarbamothioyl)amino"),
    ("CN(C)C(=S)NCCC(=O)O", "(dimethylcarbamothioyl)amino"),
    # An ARYL distal substituent: the generator is correct, but the
    # polyfunctional prefix loop does not yet route an aromatic-substituted
    # distal N to it, so the whole molecule still abstains (measured
    # UNNAMEABLE both before and after this change -- no regression, and the
    # gate catches the drop). Asserted at the generator so the class is
    # provably built even where the routing has not caught up.
    ("c1ccccc1NC(=S)NCCC(=O)O", "(phenylcarbamothioyl)amino"),
])
def test_thiourea_prefix_generator_is_atoms_aware(smiles, expected):
    from orthonym.assembly.substituent_prefix_forms import (
        get_n_substituted_carbamothioylamino_prefix as gen,
    )
    mol = Chem.MolFromSmiles(smiles)
    q = Chem.MolFromSmarts("[NX3][CX3](=[SX1])[NX3]")
    match = mol.GetSubstructMatches(q)[0]
    assert gen(mol, match, None) == expected


def test_thiourea_prefix_generator_fails_closed_on_a_bridging_core():
    """Both nitrogens substituted -> ambiguous orientation -> None.

    Exactly the oxo sibling's behaviour, measured 12/12 None on the urea
    analogue of the same molecule.
    """
    from orthonym.assembly.substituent_prefix_forms import (
        get_n_substituted_carbamothioylamino_prefix as gen,
    )
    mol = Chem.MolFromSmiles("CC(C)(C)NC(=S)NC1CCCCC1")
    q = Chem.MolFromSmarts("[NX3][CX3](=[SX1])[NX3]")
    assert gen(mol, mol.GetSubstructMatches(q)[0], None) is None


def test_selenium_and_tellurium_prefixes_fail_closed():
    """ (:33487) enumerates only the SULFUR prefix.

    No ``carbamoselenoyl``/``carbamotelluroyl`` spelling exists anywhere in the
    Blue Book, so the prefix path refuses rather than inventing one; the Se/Te
    RETAINED PARENT is built instead (:33451 ``N-(butan-2-yl)selenourea (PIN)``).
    """
    from orthonym.assembly.substituent_prefix_forms import (
        get_n_substituted_carbamothioylamino_prefix as gen,
    )
    q = Chem.MolFromSmarts("[NX3][CX3](=[S,Se,Te;X1])[NX3]")
    for smiles in ("NC(=[Se])NCCC(=O)O", "NC(=[Te])NCCC(=O)O"):
        mol = Chem.MolFromSmiles(smiles)
        assert gen(mol, mol.GetSubstructMatches(q)[0], None) is None


# ---------------------------------------------------------------------------
# 4. The class: thiourea as RETAINED PARENT with N / N' letter locants.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("NC(=S)N", "thiourea"),                                # must not regress
    ("CNC(=S)N", "N-methylthiourea"),
    ("CC(C)(C)NC(=S)N", "N-tert-butylthiourea"),
    ("NC(=S)Nc1ccccc1", "N-phenylthiourea"),
    ("CNC(=S)NC", "N,N'-dimethylthiourea"),
    ("CC(C)(C)NC(=S)NC1CCCCC1", "N-tert-butyl-N'-cyclohexylthiourea"),
])
def test_thiourea_parent_family(namer, smiles, expected):
    assert _name(namer, smiles) == expected


def test_selenourea_pin_example_from_the_blue_book(namer):
    """ (:33451): ``N-(butan-2-yl)selenourea (PIN)``."""
    assert _name(namer, "CCC(C)NC(=[Se])N") == "N-(butan-2-yl)selenourea"


@pytest.mark.parametrize("smiles,expected", [
    ("NC(=[Se])N", "selenourea"),
    ("NC(=[Te])N", "tellurourea"),
])
def test_chalcogen_urea_retained_parents(namer, smiles, expected):
    assert _name(namer, smiles) == expected


# ---------------------------------------------------------------------------
# 5. The oxo sibling must be BYTE-IDENTICAL (CLAUDE.md #9: check what is
# emitted afterwards, not merely that the bad path stopped firing).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("NC(=O)N", "urea"),
    # Monosubstituted urea omits the italic-N locant,:2943);
    # disubstituted forms keep both letter locants.
    ("CNC(=O)N", "methylurea"),
    ("CC(C)(C)NC(=O)N", "tert-butylurea"),
    ("CNC(=O)NC", "N,N'-dimethylurea"),
    ("CC(C)(C)NC(=O)NC1CCCCC1", "N-cyclohexyl-N'-tert-butylurea"),
    ("NC(=O)NCCC(=O)O", "3-(carbamoylamino)propanoic acid"),
    ("CNC(=O)NCCC(=O)O", "3-[(methylcarbamoyl)amino]propanoic acid"),
    ("CN(C)C(=O)NCCC(=O)O", "3-[(dimethylcarbamoyl)amino]propanoic acid"),
])
def test_urea_path_unchanged(namer, smiles, expected):
    assert _name(namer, smiles) == expected
