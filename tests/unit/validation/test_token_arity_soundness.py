"""``token_arity`` must never be CONFIDENT AND WRONG.

That is the module's whole contract: a confident answer feeds proof P6, which
compares it against ``len(binding.atom_ids)`` and raises an ``error``-severity
``ARITY_MISMATCH`` on disagreement. A confidently WRONG answer therefore makes
P6 reject a CORRECT name -- and it did. Reproduced before this suite existed:

======================  ========  =======  =====================================
token                   answered  truth    why
======================  ========  =======  =====================================
``benzoyl``             6         8        only tiling was the FUSION prefix
                                           ``benzo`` + ``yl``; no ``benz`` acyl
                                           stem exists to contradict it
``diazenyl``            0         2        only tiling was ``di|az|en|yl`` -- a
                                           replacement prefix qualifying nothing
``diazenylidene``       0         2        same
``benzoic acid``        8         9        ``benzo`` + ``ic acid``
``benzoate``            8         9        ``benzo`` + ``ate``
``oxalic acid``         3         6        ``ox`` + ``al`` + ``ic acid``
``phosphate``           2         5        ``phosph`` + ``ate``
``alanylalanine``       7         11       ``al`` (a SUFFIX) read token-initially
======================  ========  =======  =====================================

``diazenyl`` is the one that bit: it is the prefix of three Blue Book PINs in
``, so P6 failed
``3-diazenylpropanoic acid``, ``8-diazenyloctanoic acid`` and
``3-diazenyl-3-methylbutanoic acid`` with an ``error``. See
``test_p6_no_longer_rejects_the_diazenyl_gold_pins``.

The suite has three layers:

1. **The eight reproduced defects**, with truths established independently.
2. **The whole swept corpus** (``data/token_arity_truths.json``, 4447 tokens
   whose atom count was established from OPSIN's structure for the text rather
   than from the naming tables the oracle reads). This is the layer that would
   have caught the defect; the eight tokens above are only its entry point.
3. **One test per ROOT CAUSE**, each pinning the mechanism by name so that a
   test cannot go green because some unrelated refusal happened to fire.

Plus tripwires: a new morpheme category or a new derivation source has not been
swept against an independent structure oracle, so it must not slip in silently.
"""
import json
from pathlib import Path

import pytest

from orthonym.validation.binding_spine import (BindingKind, BindingSpine,
                                                SpineBinding, verify_spine)
from orthonym.validation.name_morphemes import (lexicon_entries,
                                                 lexicon_sources, token_arity)

pytestmark = pytest.mark.unit

_TRUTHS = Path(__file__).parent / "data" / "token_arity_truths.json"

#: Every binding kind whose tokens reach ``token_arity`` in ``_p6_arity``.
_KINDS = tuple(BindingKind)


def _fixture():
    with open(_TRUTHS) as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Layer 1: the eight reproduced confident-wrong tokens
# ---------------------------------------------------------------------------

#: (token, heavy atoms the text really spells). Every count established from the
#: structure, not from the oracle's tables: benzoyl C6H5-CO- = 6 ring C + the
#: carbonyl C + its O = 8; diazenyl -N=N-H = 2 N; benzoic acid C6H5-COOH = 9;
#: oxalic acid HOOC-COOH = 2 C + 4 O = 6; phosphate PO4 = 5; Ala-Ala = 11.
_REPRODUCED_DEFECTS = [
    ("benzoyl", 8),
    ("diazenyl", 2),
    ("diazenylidene", 2),
    ("benzoic acid", 9),
    ("benzoate", 9),
    ("oxalic acid", 6),
    ("phosphate", 5),
    ("alanylalanine", 11),
]

#: Tokens verified CORRECT at the same time. They are here so a fix cannot pay
#: for the eight above by refusing everything: these must stay confident.
_MUST_STAY_CONFIDENT = [
    ("benzamide", 9),
    ("carbonimidoyl", 2),
    ("methyl", 1),
    ("ethyl", 2),
    ("phenyl", 6),
]


@pytest.mark.parametrize("token,truth", _REPRODUCED_DEFECTS)
@pytest.mark.parametrize("kind", _KINDS)
def test_reproduced_defects_are_never_confidently_wrong(token, truth, kind):
    estimate = token_arity(token, kind)
    assert (not estimate.confident) or estimate.heavy_atoms == truth, (
        f"{token!r} ({kind}): CONFIDENTLY answered {estimate.heavy_atoms}, "
        f"truth {truth} ({estimate.basis})")


@pytest.mark.parametrize("token,truth", _MUST_STAY_CONFIDENT)
def test_correct_answers_are_not_paid_for_with_refusals(token, truth):
    estimate = token_arity(token, BindingKind.PREFIX)
    assert estimate.confident, f"{token!r} lost its answer: {estimate.basis}"
    assert estimate.heavy_atoms == truth


# ---------------------------------------------------------------------------
# Layer 2: the whole swept corpus
# ---------------------------------------------------------------------------

def test_no_swept_token_is_confidently_wrong():
    """The deliverable check: 4447 independently counted tokens x every kind.

    Reported in bulk rather than parametrised so a regression shows the WHOLE
    damage at once -- the original defect was eight tokens across four unrelated
    morpheme families, and seeing one of them says nothing about the others.
    """
    truths = _fixture()["truths"]
    wrong = []
    for token, truth in truths.items():
        for kind in _KINDS:
            estimate = token_arity(token, kind)
            if estimate.confident and estimate.heavy_atoms != truth:
                wrong.append(f"{token!r} ({kind.value}): said "
                             f"{estimate.heavy_atoms}, truth {truth} "
                             f"[{estimate.basis}]")
    assert not wrong, (
        f"{len(wrong)} confident-and-wrong answer(s) over {len(truths)} swept "
        f"tokens:\n  " + "\n  ".join(sorted(wrong)[:40]))


def test_the_sweep_fixture_is_actually_populated():
    """Guards the test above from passing by having nothing to check.

    A fixture that failed to load, or one narrowed to the eight known tokens,
    would make ``test_no_swept_token_is_confidently_wrong`` vacuously green --
    which is how this milestone shipped a test that passed under the opposite of
    its own rule.
    """
    fixture = _fixture()
    truths = fixture["truths"]
    assert len(truths) > 4000, f"only {len(truths)} swept tokens"
    assert all(isinstance(v, int) and v >= 0 for v in truths.values())
    # The corpus must contain the tokens that motivated it.
    for token, truth in _REPRODUCED_DEFECTS:
        if token in truths:
            assert truths[token] == truth, (
                f"fixture disagrees with the reproduced truth for {token!r}")
    assert "OPSIN" in fixture["_provenance"]


def test_the_sweep_corpus_still_answers_a_useful_share():
    """A refusal-everything 'fix' would satisfy soundness and destroy P6.

    P6's teeth are ``arity_confident_atom_frac``: with no confident token
    nothing is corroborated and the proof reports PROOF_UNSUBSTANTIATED. Measured
    at 52.7% of (token, kind) pairs over the fixed sweep list when this landed
    (up from 48.2% before, because the same change also added terminal-'e'
    elision and fixed two multiplier-scope bugs). The floor is set well below
    that to pin the ORDER OF MAGNITUDE, not the exact number.
    """
    truths = _fixture()["truths"]
    confident = sum(1 for token in truths
                    if token_arity(token, BindingKind.PREFIX).confident)
    share = confident / len(truths)
    assert share > 0.35, (
        f"only {confident}/{len(truths)} = {share:.1%} of swept tokens get a "
        f"confident answer; P6 has no teeth left")


# ---------------------------------------------------------------------------
# Layer 3: one test per root cause, pinning the mechanism
# ---------------------------------------------------------------------------

#: (token, kind, fragment of the refusal reason). The reason fragment is what
#: makes these tests falsifiable: without it a test would pass whenever ANY
#: refusal fired, including one from an unrelated rule, and the rule under test
#: could be deleted with the test still green.
_ROOT_CAUSE_REFUSALS = [
    # Fusion is not addition: benzo+pyran spells 10, not 12, because ortho-fusion
    # shares the two atoms of the fusion bond. Refusing the whole class also
    # removes the benzo+yl / benzo+ate mis-tilings.
    ("benzoyl", BindingKind.PREFIX, "fusion prefix"),
    ("benzopyran", BindingKind.PARENT, "fusion prefix"),
    # A skeletal replacement prefix contributes 0 only because a stem beside it
    # already counted the atom it replaces (P-15.4).
    ("diazenyl", BindingKind.PREFIX, "qualifies no skeleton"),
    ("phosphate", BindingKind.SUFFIX, "qualifies no skeleton"),
    ("azide", BindingKind.PREFIX, "qualifies no skeleton"),
    # A characteristic-group suffix closes the name (P-14.2).
    ("alanylalanine", BindingKind.SUFFIX, "is not final"),
    # ... and attaches to a parent hydride, not to a molecule that already
    # carries the group ('phosphoramid' already holds the acid oxygens) nor to a
    # substituent prefix (where the accounting is functional REPLACEMENT).
    ("nitroformic acid", BindingKind.SUFFIX, "no parent hydride"),
    ("ethanimidohydrazide", BindingKind.SUFFIX, "no parent hydride"),
    # The bare chain form of an acid suffix needs a CHAIN parent; a ring takes
    # the carb- form (P-65.1.1).
    ("ethylideneazinic acid", BindingKind.SUFFIX, "no chain parent hydride"),
    # Two skeletons cannot be juxtaposed with no attachment affix between them.
    ("methylcyclohexanecarbohydrazide", BindingKind.PREFIX, "juxtaposed"),
    ("ethanethioamide", BindingKind.SUFFIX, "juxtaposed"),
    # A multiplier that multiplies nothing in a token that spells no atoms.
    ("heptaene", BindingKind.PARENT, "spells no atoms yet carries a multiplier"),
]


@pytest.mark.parametrize("token,kind,reason", _ROOT_CAUSE_REFUSALS)
def test_each_root_cause_refuses_for_its_own_stated_reason(token, kind, reason):
    estimate = token_arity(token, kind)
    assert not estimate.confident, (
        f"{token!r} is confident again at {estimate.heavy_atoms} "
        f"({estimate.basis})")
    assert reason in estimate.basis, (
        f"{token!r} was refused, but for the wrong reason -- expected "
        f"{reason!r}, got {estimate.basis!r}. A refusal from an unrelated rule "
        f"would let the rule under test be deleted with this test still green.")


#: Morphemes a derivation screen must keep OUT of the lexicon, because the
#: shipped entry's own name contradicts its own SMILES.
_SCREENED_OUT = [
    # A '.' SMILES describes several disconnected species, so its total is not
    # one morpheme's arity ('inosinylyl' summed to 22 for a 19-atom group).
    ("inosinylyl", "disconnected SMILES"),
    # A purely hydrogen fragment has no heavy atoms; its apparent 1 is only
    # RDKit declining to merge a lone [H] ('dihydrogen' answered 1).
    ("hydrogen", "all-hydrogen SMILES"),
    # The name asserts a hydrogen isotope the SMILES does not carry, so the
    # atoms are invisible in the implicit hydrogen count.
    ("borodeuteride", "isotope the SMILES lacks"),
    ("borotritide", "isotope the SMILES lacks"),
    # The name's Hantzsch-Widman ending asserts a ring the SMILES does not
    # close ('aluminane' shipped as [AlH], one atom, for a six-membered ring).
    ("aluminane", "HW ring the SMILES lacks"),
    ("iodinane", "HW ring the SMILES lacks"),
]


@pytest.mark.parametrize("morpheme,why", _SCREENED_OUT)
def test_self_contradictory_table_entries_never_become_morphemes(morpheme, why):
    assert morpheme not in lexicon_entries(), (
        f"{morpheme!r} is back in the lexicon; the {why} screen is not firing")


#: Values the fix newly gets RIGHT. Regression guards in the other direction: a
#: later tightening must not quietly turn these back into wrong answers.
_NEWLY_CORRECT = [
    # An internal multiplier genuinely multiplies: butane + TWO -oyl = 6 (was 5,
    # because only the FOLLOWING side of the multiplier was checked).
    ("butanedioyl", 6),
    ("hexanedioic acid", 10),
    # A whole-word substituent morpheme ends the multiplied group: two phenyls
    # on a methanone = 14 (was 8, the multiplier swallowing the whole tail).
    ("diphenylmethanone", 14),
    ("diphenylethenone", 15),
    # Terminal-'e' elision of a named ring (P-16.7.1(a)).
    ("pyridinyl", 6),
    ("cyclohexanol", 7),
    ("propanoic acid", 5),
]


@pytest.mark.parametrize("token,truth", _NEWLY_CORRECT)
def test_answers_the_fix_newly_gets_right(token, truth):
    estimate = token_arity(token, BindingKind.SUFFIX)
    assert estimate.confident, f"{token!r}: {estimate.basis}"
    assert estimate.heavy_atoms == truth, estimate.basis


def test_elided_hw_stems_stay_distinctive():
    """Pins ``_MIN_ELIDED`` directly, because the corpus no longer can.

    The 2-character elision of the HW stem ``ine`` produced a bare ``in`` that
    matched inside four Blue Book acid names and planted a spurious six-membered
    ring in each ("benzeneseleninic acid" -> benzene+selen+IN+ic acid = a
    confident 14 for a 9-atom acid). R3c now catches that family by a second
    route, so lowering the floor does NOT reintroduce a confident-wrong answer in
    the swept corpus -- which means the corpus cannot test this rule. Asserting
    the property itself is what keeps it falsifiable rather than decorative.

    The rule governs the DERIVED elisions only. The shipped full stems
    (``ane``, ``ine``, ``ole``, ``ete``) are short too, and they stay: they are
    what the HW table itself says, and they are reachable only straight after a
    replacement prefix.
    """
    from orthonym.data.hw_stems import HW_STEMS
    from orthonym.validation.name_morphemes import _MIN_ELIDED, _hw_stems

    table = _hw_stems()
    shipped = {stem.strip().lower()
               for variants in HW_STEMS.values() for stem in variants.values()
               if isinstance(stem, str)}
    too_short, elided_long = [], []
    for stem in shipped:
        if not stem.endswith("e"):
            continue
        elided = stem[:-1]
        if elided in shipped:
            continue        # coincides with a stem the table ships outright
        if len(elided) < _MIN_ELIDED:
            if elided in table:
                too_short.append(elided)
        elif elided in table:
            elided_long.append(elided)

    assert not too_short, (
        f"derived Hantzsch-Widman elisions {sorted(too_short)} are shorter than "
        f"{_MIN_ELIDED} characters. This path bypasses _Lexicon.add's screening, "
        f"so a short stem matches inside unrelated words and manufactures a ring "
        f"reading the lexicon cannot contradict -- a bare 'in' from 'ine' put a "
        f"spurious six-membered ring inside four Blue Book acid names.")
    assert elided_long, (
        "no HW stem is elided at all any more; 'oxan-4-yl' and friends can no "
        "longer be read, so this test would pass vacuously")


def test_narrowing_the_lexicon_by_kind_cannot_hide_a_reading():
    """Regression: the PREFIX path used to be less safe than the SUFFIX path.

    ``ethaneselenol`` has two readings -- the ``-ol`` suffix (3 atoms, right) and
    a spurious Hantzsch-Widman ``selen|ol`` ring (7). The SUFFIX lexicon saw
    both and refused; the narrowed non-suffix lexicon saw only the ring one and
    certified 7. Removing candidate readings can only manufacture false
    confidence, so every kind now enumerates the same lexicon.
    """
    for kind in _KINDS:
        estimate = token_arity("ethaneselenol", kind)
        assert (not estimate.confident) or estimate.heavy_atoms == 3, (
            f"{kind}: confidently answered {estimate.heavy_atoms} "
            f"({estimate.basis})")


# ---------------------------------------------------------------------------
# P6 on the three Blue Book PINs the defect broke
# ---------------------------------------------------------------------------

#: (SMILES, PIN, bindings). Blue Book P-68.3.1.3.1 / P-35.2.2; gold rows
#: W2-NPREF-DIAZENYL-P68, W2-NPREF-DIAZENYL, W2-NPREF-DIAZENYL-SORT in
#: 
_DIAZENYL_GOLD = [
    ("N=NCCC(=O)O", "3-diazenylpropanoic acid",
     [("diazenyl", BindingKind.PREFIX, [0, 1]),
      ("propan", BindingKind.PARENT, [2, 3, 4]),
      ("oic acid", BindingKind.SUFFIX, [5, 6])]),
    ("N=NCCCCCCCC(=O)O", "8-diazenyloctanoic acid",
     [("diazenyl", BindingKind.PREFIX, [0, 1]),
      ("octan", BindingKind.PARENT, list(range(2, 10))),
      ("oic acid", BindingKind.SUFFIX, [10, 11])]),
    ("CC(C)(N=N)CC(=O)O", "3-diazenyl-3-methylbutanoic acid",
     [("diazenyl", BindingKind.PREFIX, [3, 4]),
      ("methyl", BindingKind.PREFIX, [2]),
      ("butan", BindingKind.PARENT, [0, 1, 5, 6]),
      ("oic acid", BindingKind.SUFFIX, [7, 8])]),
]


@pytest.mark.parametrize("smiles,name,bindings", _DIAZENYL_GOLD,
                         ids=[row[1] for row in _DIAZENYL_GOLD])
def test_p6_no_longer_rejects_the_diazenyl_gold_pins(smiles, name, bindings):
    """These three CORRECT names used to fail P6 with an error-severity finding.

    Before: ``ok=False``, ``ARITY_MISMATCH`` -- "token 'diazenyl' claims 2 heavy
    atom(s) but its morphemes spell 0 (di+az+en+yl)". After: ``ok=True`` with an
    info-severity ``ARITY_UNVERIFIED``, which is the honest verdict: the oracle
    cannot count the token, so it declines to instead of certifying a wrong
    count. Unverified is always acceptable; confidently wrong never is.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    spine = BindingSpine(roots=tuple(
        SpineBinding(token=token, kind=kind, atom_ids=frozenset(atoms))
        for token, kind, atoms in bindings))
    proof = verify_spine(mol, spine, name, mode="audit")
    mismatches = [f for f in proof.findings if f.code == "ARITY_MISMATCH"]
    assert not mismatches, (
        f"P6 still rejects the PIN {name!r}: "
        + "; ".join(f.detail for f in mismatches))
    errors = [f for f in proof.findings if f.severity == "error"]
    assert not errors, "; ".join(f"{f.code}: {f.detail}" for f in errors)
    assert proof.ok


# ---------------------------------------------------------------------------
# Tripwires
# ---------------------------------------------------------------------------

#: Every category ``_well_formed`` knows how to police. A NEW one would sit in
#: the lexicon unpoliced by every positional rule, which is precisely how a
#: fusion prefix came to be summed like a stem.
_POLICED_CATEGORIES = frozenset(
    {"stem", "subst", "attach", "affix", "repl", "fuse"})


def test_every_lexicon_entry_carries_real_arity_data():
    """No morpheme may reach the table without a usable atom count."""
    offenders = [
        (key, atoms, category)
        for key, (atoms, category, _suffix_only) in lexicon_entries().items()
        if not isinstance(atoms, int) or isinstance(atoms, bool) or atoms < 0
        or not isinstance(category, str) or not category]
    assert not offenders, f"morphemes without arity data: {offenders[:20]}"


def test_a_new_morpheme_category_cannot_arrive_unpoliced():
    """TRIPWIRE. Fails when the lexicon grows a category ``_well_formed`` does
    not police, so that whoever adds it must extend the positional rules (and
    re-run the arity sweep) rather than inheriting silent summation."""
    present = {category for _atoms, category, _s in lexicon_entries().values()}
    unpoliced = present - _POLICED_CATEGORIES
    assert not unpoliced, (
        f"new morpheme category/ies {sorted(unpoliced)} are in the lexicon but "
        f"no _well_formed rule constrains where they may appear. Extend "
        f"_well_formed, re-run the arity sweep, then add them to "
        f"_POLICED_CATEGORIES.")


def test_a_new_derivation_source_cannot_arrive_unswept():
    """TRIPWIRE. Fails when ``_lexicon`` gains a morpheme source, because its
    entries have not been through ``_entry_atoms``' screens nor swept against an
    independent structure oracle."""
    assert lexicon_sources() == (
        "_STRUCTURAL_AFFIXES",
        "_add_suffix_morphemes",
        "_add_replacement_prefixes",
        "_add_chain_stems",
        "_add_group_tables",
        "_add_retained_names",
        "_add_ring_derived",
        "_add_elided_stems",
    ), ("the lexicon's morpheme sources changed. Re-run the arity sweep over "
        "the new source, confirm 0 confident-and-wrong, refresh "
        "data/token_arity_truths.json, then update this tripwire.")
