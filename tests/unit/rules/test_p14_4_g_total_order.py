""" is an ORDERED CASCADE, and it must terminate in the STRUCTURE.

 Phase C Task 9b. Task 9 (``) implemented criterion **(c)** and left
the cascade to fall through to RDKit's ring-atom enumeration order -- i.e. to the
input SMILES. Consequence, one command:

    scripts/diagnose.py "c1(O)c(Cl)cccc1C" "Cc1cccc(Cl)c1O"
      c1(O)c(Cl)cccc1C -> 6-chloro-2-methylphenol
      Cc1cccc(Cl)c1O -> 2-chloro-6-methylphenol # THE SAME MOLECULE

Which one is right, derived at source
-------------------------------------
§** "NUMBERING"** (``the Blue Book Blue Book``): *"When several
structural features appear in cyclic and acyclic compounds, low locants are
assigned to them in the following decreasing order of seniority:"*

    :3227 (a) fixed numbering... -- benzene has none
    :3246 (b) indicated hydrogen... -- benzene has none
    :3256 (c) principal characteristic groups and free valences (suffixes);
    :3270 (d) 'added indicated hydrogen' -- benzene has none
    :3288 (e) saturation/unsaturation -- benzene has no choice
    :3301 (f) detachable alphabetized prefixes, all considered together in a
           series of increasing numerical order;
    :3307 (g) lowest locants for the substituent cited first as a prefix in the
           name;
    :3320 (h) nonstandard valence state -- all carbons

After (c) fixes the ``-ol`` at locant 1, BOTH orientations give the prefix set
{2,6}, so **(f)** ties (worked at ``:3305``: *"the locant set '4,5,8' is lower than
'4,7,8'"*) and **(g)** decides. ``chloro`` is cited before ``methyl``, so chloro
takes the lower locant: the PIN is ``2-chloro-6-methylphenol``.

The decisive worked examples for (g):
  * ``:3315`` ``4-methyl-5-nitrooctanedioic acid (PIN)``
  * ``:3317`` ``1-methyl-4-nitronaphthalene (PIN) (not 4-methyl-1-nitronaphthalene)``

and (g) is stated a SECOND time, with a benzene example, as §** "Low
locants are assigned to the prefix cited first in the name"** (``:26085``):
  * ``1-bromo-2-chloroethane (PIN)``
  * ``:26094`` ``1-azido-4-isocyanatobenzene (PIN)``

So this was an **accuracy** defect as much as a determinism one, and neither the
gold set nor ``scripts/determinism_eval.py`` could see it: the gate has 27 aryl-OH
probes and not one mirror pair.
"""

import random

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules import benzene as benzene_rules
from orthonym.rules.benzene import (
    benzene_prefix_citation_locants,
    benzene_prefix_suffix_promotion,
    get_benzene_substituents,
    _canonical_orbit_key,
)

pytestmark = pytest.mark.unit


N_RESPELL = 7
_RESPELL_SEED = 20260729


def _respellings(smiles):
    """The canonical spelling plus ``N_RESPELL`` seeded atom-order permutations.

    A determinism defect that only SOME spellings reveal is the failure mode here
    (the live one splits 8/4 over 12 spellings), so a single alternative spelling
    is not enough evidence.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    out = [Chem.MolToSmiles(mol)]
    rng = random.Random(_RESPELL_SEED)
    for _ in range(N_RESPELL):
        perm = list(range(mol.GetNumAtoms()))
        rng.shuffle(perm)
        out.append(
            Chem.MolToSmiles(Chem.RenumberAtoms(mol, perm), canonical=False)
        )
    assert len(out) == N_RESPELL + 1
    return out


# --------------------------------------------------------------------------
# (g), through the emitted NAME. Session a project rule: every row asserts the
# COMPLETE name -- moving a locant has twice this phase unmasked something worse
# (a lost enclosing mark, a fabricated morpheme), so "the locant moved" is not a
# pass condition.
# --------------------------------------------------------------------------

# Each row: the molecule, and the ONE name every spelling of it must produce.
G_TARGETS = [
    # ★ THE live defect. (c) puts -ol at 1; (f) ties at {2,6}; (g) gives chloro
    # (cited first) the 2. Both source spellings are listed because the pair is
    # what made the defect visible at all.
    ("c1(O)c(Cl)cccc1C", "2-chloro-6-methylphenol"),
    ("Cc1cccc(Cl)c1O", "2-chloro-6-methylphenol"),
    ("Clc1cccc(C)c1O", "2-chloro-6-methylphenol"),
    # fluoro < methyl
    ("Cc1cccc(F)c1O", "2-fluoro-6-methylphenol"),
    # chloro < fluoro
    ("Oc1c(F)cccc1Cl", "2-chloro-6-fluorophenol"),
    # ★ (g) across a meta pair -- the mirror maps 3<->5, so (f) ties at {3,5}.
    # Was '5-chloro-3-methylphenol', which cites chloro at the HIGHER locant.
    ("Cc1cc(O)cc(Cl)c1", "3-chloro-5-methylphenol"),
    ("Cc1cc(O)cc(F)c1", "3-fluoro-5-methylphenol"),
    ("Oc1cc(F)cc(Cl)c1", "3-chloro-5-fluorophenol"),
    # ★ (g) with NO suffix at all and a multiplied prefix: {1,3,5} either way, so
    # (g) must give the two chloro the {1,3} and leave methyl the 5.
    ("Cc1cc(Cl)cc(Cl)c1", "1,3-dichloro-5-methylbenzene"),
    ("Cc1cc(F)cc(F)c1", "1,3-difluoro-5-methylbenzene"),
    ("Fc1cc(Cl)cc(Cl)c1", "1,3-dichloro-5-fluorobenzene"),
    ("Cc1cc(F)cc(Cl)c1", "1-chloro-3-fluoro-5-methylbenzene"),
    # ★ Five hydroxy + one functional-class prefix: (f) is {1..6} in every
    # orientation, so only (g) can act -- hydroxy is cited before isocyanato and
    # must hold {1,2,3,4,5}. Emitted three DIFFERENT names before this.
    ("O=C=Nc1c(O)c(O)c(O)c(O)c1O",
     "1,2,3,4,5-pentahydroxy-6-isocyanatobenzene"),
]


@pytest.mark.parametrize("smiles,expected", G_TARGETS)
def test_g_decides_and_every_spelling_agrees(smiles, expected):
    """(g)/: one molecule, one PIN, whatever the input SMILES."""
    names = [name_compound(s) for s in _respellings(smiles)]
    assert set(names) == {expected}, sorted(set(names))


# The BB's own (g) example transposed onto benzene, where we can actually run it:
#:3317 is 1-methyl-4-nitronaphthalene (PIN) (not 4-methyl-1-nitronaphthalene).
# methyl is cited before nitro, so methyl takes locant 1.
BB_G_ANALOGUES = [
    ("Cc1ccc([N+](=O)[O-])cc1", "1-methyl-4-nitrobenzene"),
    ("Cc1cccc([N+](=O)[O-])c1", "1-methyl-3-nitrobenzene"),
    ("Cc1ccccc1[N+](=O)[O-]", "1-methyl-2-nitrobenzene"),
    #:26085's own example is 1-bromo-2-chloroethane; on a ring, bromo < chloro.
    ("Brc1ccccc1Cl", "1-bromo-2-chlorobenzene"),
]


@pytest.mark.parametrize("smiles,expected", BB_G_ANALOGUES)
def test_bb_g_worked_examples_on_benzene(smiles, expected):
    """The prefix cited first takes the lower locant (the Blue Book, the Blue Book)."""
    names = [name_compound(s) for s in _respellings(smiles)]
    assert set(names) == {expected}, sorted(set(names))


# --------------------------------------------------------------------------
# (g) must use the / alphanumerical key, not raw string order.
# Witness derived after a MUTATION SURVIVED: replacing ``alpha_sort_key`` with a
# plain ``sorted`` in the (g) key was invisible to every other test here.
#
# §**** (``the Blue Book Blue Book``): *"When an alphanumerical
# ordering is required and Roman letters do not permit a decision for the order of
# citation, italicized letters are considered."* So the italicized ``tert`` is NOT
# part of the primary comparison -- the BB's own example cites
# ``3-tert-butyl-1-(1-methylpropyl)benzene`` (butyl before methylpropyl), and lists
# ``1-sec-butyl-3-tert-butylbenzene`` as the form where the Roman letters ARE
# identical so the italics decide. ``alpha_sort_key`` implements this; raw string
# order does not ('c' < 't').
#
# ⚠ NOTE FOR THE READER: ``CLAUDE.md`` states "INCLUDE for alphabetization: iso-,
# neo-, cyclo-, sec-, tert-". That is right for the nonitalic iso/neo/cyclo and
# WRONG for the italicized sec-/tert-, per above.
# --------------------------------------------------------------------------

ALPHA_KEY_WITNESSES = [
    # 'butyl' < 'chloro', so tert-butyl is cited first AND takes locant 2.
    # A raw string sort ('chloro' < 'tert-butyl') would give chloro the 2.
    ("CC(C)(C)c1cccc(Cl)c1O", "2-tert-butyl-6-chlorophenol"),
    # Same discrimination with no suffix at all: tert-butyl takes locant 1 and the
    # two chloro take {3,5}. Raw string order would hand the chloro pair {1,3}.
    ("CC(C)(C)c1cc(Cl)cc(Cl)c1", "1-tert-butyl-3,5-dichlorobenzene"),
    # ★ CONTROL: 'bromo' < 'butyl' in BOTH orders, so this row does NOT
    # discriminate the two keys. It is here to prove the two rows above are not
    # simply "tert-butyl always wins".
    ("CC(C)(C)c1cccc(Br)c1O", "2-bromo-6-tert-butylphenol"),
]


@pytest.mark.parametrize("smiles,expected", ALPHA_KEY_WITNESSES)
def test_g_uses_the_iupac_alphanumerical_key(smiles, expected):
    """: the italicized ``tert`` is not part of the Roman-letter order."""
    names = [name_compound(s) for s in _respellings(smiles)]
    assert set(names) == {expected}, sorted(set(names))


def test_the_alpha_key_and_raw_string_order_really_disagree_here():
    """Proves the witnesses above are load-bearing rather than incidental.

    If ``alpha_sort_key`` ever stopped differing from ``sorted`` on this pair,
    the rows above would silently stop testing anything.
    """
    from orthonym.assembly.naming_utils import alpha_sort_key

    pair = ["tert-butyl", "chloro"]
    assert sorted(pair, key=alpha_sort_key) == ["tert-butyl", "chloro"]
    assert sorted(pair) == ["chloro", "tert-butyl"]
    #...and they AGREE on the control pair.
    control = ["bromo", "tert-butyl"]
    assert sorted(control, key=alpha_sort_key) == sorted(control) == control


# --------------------------------------------------------------------------
# The tie-break primitives, pinned directly rather than only through a name.
# --------------------------------------------------------------------------

def _ring_and_subs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    ring = next(
        r for r in mol.GetRingInfo().AtomRings()
        if len(r) == 6 and all(
            mol.GetAtomWithIdx(i).GetIsAromatic()
            and mol.GetAtomWithIdx(i).GetSymbol() == "C" for i in r
        )
    )
    return mol, ring, get_benzene_substituents(mol, ring)


def test_citation_locants_orders_by_alpha_not_by_ring_order():
    """(g)'s key is per-prefix locants in ALPHABETICAL citation order."""
    mol, ring, subs = _ring_and_subs("Cc1cccc(Cl)c1O")
    # Hang the ring so the methyl-bearing atom is locant 2 and chloro is 6.
    oh_atom = next(
        a for a in ring
        if any(n.GetSymbol() == "O" for n in mol.GetAtomWithIdx(a).GetNeighbors())
    )
    order = list(ring)
    order = order[order.index(oh_atom):] + order[:order.index(oh_atom)]

    key_fwd = benzene_prefix_citation_locants(order, subs, {oh_atom})
    key_rev = benzene_prefix_citation_locants(
        [order[0]] + list(reversed(order[1:])), subs, {oh_atom}
    )
    # chloro is cited first, so its locant is the FIRST element of the key.
    assert len(key_fwd) == 2 and len(key_rev) == 2, (key_fwd, key_rev)
    assert key_fwd != key_rev, "the two directions must be distinguishable"
    assert min(key_fwd, key_rev)[0] == (2,), (key_fwd, key_rev)


def test_citation_locants_excludes_the_pcg_but_keeps_its_n_prefix():
    """The suffix is (c)'s business; its italic-N prefix is still a cited prefix.

    Regression witness: with the N-substituents omitted, two suffix instances that
    differ ONLY in their N-substituent produce an EMPTY (g) key, the tie falls to
    the canonical last resort, and ``Nc1ccc(Nc2ccc(Nc3ccccc3)cc2)cc1`` came out
    with ``(4-aminophenyl)`` on the HIGHER nitrogen.
    """
    smiles = "Nc1ccc(Nc2ccc(Nc3ccccc3)cc2)cc1"
    mol = Chem.MolFromSmiles(smiles)
    ring = next(
        r for r in mol.GetRingInfo().AtomRings()
        if len(r) == 6 and len(get_benzene_substituents(mol, r)) == 2
        and all(
            (s.get("amine_candidate") or {}).get("n_substituents")
            for subs in get_benzene_substituents(mol, r).values() for s in subs
        )
    )
    subs = get_benzene_substituents(mol, ring)
    pcg = set(subs)
    key = benzene_prefix_citation_locants(list(ring), subs, pcg)
    # Both ring atoms are the PCG, so the ONLY entries can be the two N-prefixes.
    assert len(key) == 2, key
    assert all(len(locs) == 1 for locs in key), key


def test_citation_locants_ignores_is_suffix_when_no_pcg_supplied():
    """With no PCG set every ``is_suffix`` substituent IS the suffix, not a prefix."""
    mol, ring, subs = _ring_and_subs("Cc1ccc(C(=O)O)cc1")
    assert any(
        s.get("is_suffix") for subs_at in subs.values() for s in subs_at
    ), "witness needs an is_suffix substituent to be meaningful"
    key = benzene_prefix_citation_locants(list(ring), subs, None)
    assert len(key) == 1, key      # methyl only; the acid is excluded


def test_canonical_orbit_key_is_invariant_under_respelling():
    """The last resort must be a STRUCTURE invariant, not an atom-order artefact.

    If this key varied with the input spelling it would be no better than the
    enumeration order it replaced.
    """
    smiles = "Cc1cccc(Cl)c1O"
    mol = Chem.MolFromSmiles(smiles)
    ring = next(
        r for r in mol.GetRingInfo().AtomRings() if len(r) == 6
    )
    base = _canonical_orbit_key(mol, list(ring))
    assert base and len(base) == 6, base

    rng = random.Random(7)
    for _ in range(N_RESPELL):
        perm = list(range(mol.GetNumAtoms()))
        rng.shuffle(perm)
        mol2 = Chem.RenumberAtoms(mol, perm)
        # perm[i] of the original becomes atom i of mol2.
        old_to_new = {old: new for new, old in enumerate(perm)}
        ring2 = [old_to_new[a] for a in ring]
        assert _canonical_orbit_key(mol2, ring2) == base


def test_canonical_orbit_key_fails_closed_on_a_bad_molecule():
    """An empty key = no discrimination; it must never raise into the namer."""
    assert _canonical_orbit_key(None, [0, 1, 2]) == ()


# --------------------------------------------------------------------------
# The SHARED promotion authority (Task 9b defect C-2).
# --------------------------------------------------------------------------

def test_promotion_authority_promotes_hydroxy_when_nothing_senior_holds_the_slot():
    assert benzene_prefix_suffix_promotion(
        set(), {"hydroxy", "chloro"}, {},
    ) == ("ol", frozenset({"hydroxy"}))


def test_promotion_authority_demotes_a_junior_suffix_but_not_a_senior_one():
    """: -ol outranks -thiol, and a carboxylic acid outranks -ol."""
    assert benzene_prefix_suffix_promotion(
        {"thiol"}, {"hydroxy"}, {},
    ) == ("ol", frozenset({"hydroxy"}))
    assert benzene_prefix_suffix_promotion(
        {"carboxylic acid"}, {"hydroxy"}, {},
    ) == (None, frozenset())


def test_promotion_authority_promotes_an_amine_only_when_nothing_holds_the_slot():
    assert benzene_prefix_suffix_promotion(
        set(), {"amino"}, {}, has_amine_candidates=True,
    ) == ("amine", frozenset())
    # An -ol takes precedence and is mutually exclusive with the amine promotion.
    assert benzene_prefix_suffix_promotion(
        set(), {"hydroxy", "amino"}, {}, has_amine_candidates=True,
    ) == ("ol", frozenset({"hydroxy"}))
    # A junior thiol suffix blocks the AMINE promotion (it is not junior to -amine).
    assert benzene_prefix_suffix_promotion(
        {"thiol"}, {"amino"}, {}, has_amine_candidates=True,
    ) == (None, frozenset())


@pytest.mark.parametrize("fg", sorted(benzene_rules._FUNCTIONAL_CLASS_FGS))
def test_promotion_authority_blocks_on_every_functional_class_key(fg):
    """Each surviving blacklist key must actually suppress the promotion.

    Parametrised over the frozenset itself so a key added later cannot join it
    untested -- the dead ``'azide'`` key got in exactly that way.
    """
    assert benzene_prefix_suffix_promotion(
        set(), {"hydroxy"}, {fg: [(0,)]},
    ) == (None, frozenset())


def test_functional_class_keys_all_exist_in_the_detector_registry():
    """★ Step 3: presence in a table is NOT evidence the table is reached.

    ``_FUNCTIONAL_CLASS_FGS`` carried ``'azide'`` and ``'selenocyanate'``, neither
    of which ``detect_functional_groups`` can ever emit (it emits ``'azido'``, and
    has no selenocyanate pattern), so those entries were inert while the comment
    beside them claimed azides blocked the promotion. This asserts the invariant
    the module now enforces at import.
    """
    from orthonym.perception.functional_groups import FUNCTIONAL_GROUP_SMARTS

    dead = sorted(
        benzene_rules._FUNCTIONAL_CLASS_FGS - set(FUNCTIONAL_GROUP_SMARTS)
    )
    assert dead == [], dead
    # And the import-time guard really fires, rather than being decorative.
    with pytest.raises(AssertionError, match="can never fire"):
        original = benzene_rules._FUNCTIONAL_CLASS_FGS
        try:
            benzene_rules._FUNCTIONAL_CLASS_FGS = frozenset(
                original | {"not_a_detector_key"}
            )
            benzene_rules._assert_registry_keys()
        finally:
            benzene_rules._FUNCTIONAL_CLASS_FGS = original


def test_azido_does_not_block_the_ol_promotion_p61_7():
    """§ AZIDES (the Blue Book): azido is a substitutive PREFIX and gives PINs.

    ``:25997`` ``3-azidonaphthalene-2-sulfonic acid (PIN)`` shows azido cited as a
    detachable prefix while a SUFFIX governs the parent, so an azide cannot stop
    the -OH from being the principal characteristic group. The dead ``'azide'``
    key was producing this right answer for the wrong reason; removing it keeps the
    answer and makes the reason real.
    """
    assert benzene_prefix_suffix_promotion(
        set(), {"hydroxy"}, {"azido": [(0,)]},
    ) == ("ol", frozenset({"hydroxy"}))
    names = [
        name_compound(s)
        for s in _respellings("Oc1c(O)c(O)c(O)c(O)c1N=[N+]=[N-]")
    ]
    assert set(names) == {"6-azidobenzene-1,2,3,4,5-pentol"}, sorted(set(names))


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN DEFECT, derived not guessed: BB:1710(p) records that -N=C=O "
           "'and its chalcogen analogues ... have been added to the list of "
           "characteristic groups that are ALWAYS cited as prefixes'; :26003 "
           "repeats it and :26014 gives '4-isocyanatobenzene-1-sulfonyl chloride "
           "(PIN)' -- isocyanato as a prefix WITH a suffix on the parent. A group "
           "that can never be a suffix cannot outrank -ol, so the PIN is "
           "'6-isocyanatobenzene-1,2,3,4,5-pentol'. Fixing it changes the benzene "
           "SUFFIX-selection layer, which is outside Task 9b's numbering scope. "
           "MEASURED exposure: of the 1037 benzene-containing molecules in the "
           "three corpora, 4 carry a functional-class blocker and only ONE also "
           "carries a promotable -OH (N#CSc1ccc(O)cc1), which already refuses "
           "('unknown organic compound') for an unrelated upstream reason -- so "
           "the live corpus exposure of this guard is ZERO.",
)
def test_isocyanato_should_not_block_the_ol_promotion_p61_9():
    # The mono case, where the consequence is easiest to read.
    assert name_compound("Oc1ccc(N=C=O)cc1") == "4-isocyanatophenol"
    #...and the penta case from the Task 9 target family.
    assert (
        name_compound("Oc1c(O)c(O)c(O)c(O)c1N=C=O")
        == "6-isocyanatobenzene-1,2,3,4,5-pentol"
    )


# --------------------------------------------------------------------------
# SPIES. A mutation-killed test proves a test is sensitive to the code it CALLS,
# not that production ever calls it. Each trace below is validated on >=2 known
# positives and >=1 known negative in the same run, so a trace that recorded
# nothing could not pass.
# --------------------------------------------------------------------------

def test_production_reaches_the_g_tier(monkeypatch):
    """``benzene_prefix_citation_locants`` is on the live path, not just correct."""
    calls = []
    original = benzene_rules.benzene_prefix_citation_locants

    def recording(oriented, substituents, principal_group_positions=None):
        result = original(oriented, substituents, principal_group_positions)
        calls.append(result)
        return result

    monkeypatch.setattr(
        benzene_rules, "benzene_prefix_citation_locants", recording
    )

    # positive 1: (f) ties at {2,6}, so (g) MUST be consulted
    assert name_compound("Cc1cccc(Cl)c1O") == "2-chloro-6-methylphenol"
    assert calls, "the (g) tier was never reached for the mirror-pair phenol"
    # positive 2: a different shape (no suffix, multiplied prefix), {1,3,5} tie
    calls.clear()
    assert name_compound("Cc1cc(Cl)cc(Cl)c1") == "1,3-dichloro-5-methylbenzene"
    assert calls, "the (g) tier was never reached for the 1,3,5 pattern"
    # negative: monosubstituted short-circuits before any tie-break at all
    calls.clear()
    assert name_compound("Clc1ccccc1") == "chlorobenzene"
    assert not calls, calls


def test_orient_benzene_g_is_load_bearing_where_only_it_can_act(monkeypatch):
    """Neutralise ``orient_benzene``'s (g) and the prefix-only path regresses.

    Witness derived from the FAILURE MODE, and chosen by MEASUREMENT: these two
    molecules never reach ``_renumber_relative_to`` (no suffix re-anchoring), so
    ``orient_benzene``'s (g) is the only tier that can order their prefixes.
    """
    before = {
        "Cc1cc(Cl)cc(Cl)c1": name_compound("Cc1cc(Cl)cc(Cl)c1"),
        "O=C=Nc1c(O)c(O)c(O)c(O)c1O":
            name_compound("O=C=Nc1c(O)c(O)c(O)c(O)c1O"),
    }
    assert before == {
        "Cc1cc(Cl)cc(Cl)c1": "1,3-dichloro-5-methylbenzene",
        "O=C=Nc1c(O)c(O)c(O)c(O)c1O":
            "1,2,3,4,5-pentahydroxy-6-isocyanatobenzene",
    }, before

    monkeypatch.setattr(
        benzene_rules, "benzene_prefix_citation_locants", lambda *a, **k: ()
    )
    after = {s: name_compound(s) for s in before}
    assert after != before, (
        "neutralising orient_benzene's (g) changed nothing, so it is not what "
        "produces these names and this test proves nothing: %s" % after
    )
    # And specifically: the multiplied prefix loses its low locant set.
    assert after["Cc1cc(Cl)cc(Cl)c1"] == "3,5-dichloro-1-methylbenzene", after


def test_the_mirror_pair_is_protected_by_BOTH_g_tiers(monkeypatch):
    """★ The pair is protected REDUNDANTLY -- so a single mutation survives.

    Measured, and it is the reason a one-site mutation test here would have been
    green-but-blind: ``orient_benzene``'s (g) and ``_renumber_relative_to``'s (g)
    each produce the PIN on their own, so disabling either leaves the other to do
    it. With BOTH disabled the pair collapses onto the NON-PIN
    ``6-chloro-2-methylphenol`` -- both spellings agree, but on the name (g)
    forbids. (It no longer SPLITS, because the split needed the crude "position 1
    goes to the alphabetically first substituent at position 1" tier that (g)
    replaced; that tier's key was constant across every survivor, so its stable
    sort simply kept the first candidate in enumeration order.)
    """
    pair = ("c1(O)c(Cl)cccc1C", "Cc1cccc(Cl)c1O")
    assert {name_compound(s) for s in pair} == {"2-chloro-6-methylphenol"}

    monkeypatch.setattr(
        benzene_rules, "benzene_prefix_citation_locants", lambda *a, **k: ()
    )
    original_renumber = benzene_rules._renumber_relative_to

    def renumber_without_g(groups, reference_locant):
        """The pre-Task-9b body: (f) only, ties keep ``direction = 1``."""
        from collections import defaultdict
        best_groups, best_locants = None, None
        for direction in (1, -1):
            converted = defaultdict(list)
            for name, locants in groups.items():
                for old in locants:
                    new = ((old - reference_locant) * direction % 6) + 1
                    converted[name].append(6 if new == 1 else new)
            for name in converted:
                converted[name].sort()
            all_locants = sorted(
                loc for locs in converted.values() for loc in locs
            )
            if best_locants is None or all_locants < best_locants:
                best_locants, best_groups = all_locants, dict(converted)
        return best_groups or {}

    monkeypatch.setattr(
        benzene_rules, "_renumber_relative_to", renumber_without_g
    )
    got = {name_compound(s) for s in pair}
    monkeypatch.setattr(
        benzene_rules, "_renumber_relative_to", original_renumber
    )
    assert got == {"6-chloro-2-methylphenol"}, (
        "with BOTH (g) tiers removed the pair must land on the NON-PIN "
        "'6-chloro-2-methylphenol'. If it still produces the PIN, neither tier is "
        "what fixes it and this test proves nothing: %s" % got
    )


def test_production_reaches_the_canonical_last_resort(monkeypatch):
    """The last resort is wired in -- and it is only ever a tie-break.

    Validated on two positives (molecules whose (c)/(f)/(g) genuinely tie because
    the ring is symmetric) and one negative (monosubstituted, no tie-break at all).
    """
    calls = []
    original = benzene_rules._canonical_orbit_key

    def recording(mol, oriented):
        calls.append(tuple(oriented))
        return original(mol, oriented)

    monkeypatch.setattr(benzene_rules, "_canonical_orbit_key", recording)

    assert name_compound("Clc1ccc(Cl)cc1") == "1,4-dichlorobenzene"
    assert calls, "the last resort was never reached for para-dichlorobenzene"
    calls.clear()
    assert name_compound("Cc1ccc(C)cc1") == "1,4-dimethylbenzene"
    assert calls, "the last resort was never reached for para-xylene"
    calls.clear()
    assert name_compound("Clc1ccccc1") == "chlorobenzene"
    assert not calls, calls


def test_renumber_relative_to_applies_g_not_the_inherited_direction():
    """The suffix paths' direction choice obeys (g), not ``direction = 1``."""
    from orthonym.rules.benzene import _renumber_relative_to

    # phenol at locant 1 (reference 1); chloro at 2, methyl at 6 in one direction
    # and swapped in the other. (f) ties at [2, 6]; chloro is cited first.
    got = _renumber_relative_to({"chloro": [6], "methyl": [2]}, 1)
    assert got == {"chloro": [2], "methyl": [6]}, got
    # Already (g)-optimal: unchanged.
    got = _renumber_relative_to({"chloro": [2], "methyl": [6]}, 1)
    assert got == {"chloro": [2], "methyl": [6]}, got
    # (f) still outranks (g): [2, 3] beats [5, 6] even though it puts methyl low.
    got = _renumber_relative_to({"chloro": [3], "methyl": [2]}, 1)
    assert got == {"chloro": [3], "methyl": [2]}, got


def test_preferred_benzene_parent_ring_is_exercised_and_anchors(monkeypatch):
    """★ The third 'unified' anchor call site, which had NO test of its own.

    ``_preferred_benzene_parent_ring`` is only reached for MULTI-benzene molecules,
    so a name-level test elsewhere cannot cover it. Spy validated on two positives
    and one negative (a single-ring molecule never reaches it).
    """
    from orthonym.rules import benzene as br

    seen = []
    original = br.principal_group_ring_atoms

    def recording(mol, ring_atoms, substituents, **kw):
        result = original(mol, ring_atoms, substituents, **kw)
        seen.append((tuple(ring_atoms), frozenset(result)))
        return result

    monkeypatch.setattr(br, "principal_group_ring_atoms", recording)

    # positive 1: two candidate rings (measured), so the anchor is consulted twice
    mol = Chem.MolFromSmiles("Clc1ccccc1-c1ccccc1O")
    assert len(br._benzene_parent_candidates(mol)) == 2
    assert br._preferred_benzene_parent_ring(mol) is not None
    assert len(seen) == 2, (
        "expected the anchor once per candidate ring: %s" % (seen,)
    )
    #...and it actually anchored the -OH ring rather than returning nothing.
    assert any(pcg for _ring, pcg in seen), seen

    # positive 2: a different two-candidate shape
    seen.clear()
    mol = Chem.MolFromSmiles("Oc1ccccc1Cc1ccccc1O")
    assert len(br._benzene_parent_candidates(mol)) == 2
    assert br._preferred_benzene_parent_ring(mol) is not None
    assert len(seen) == 2, seen

    # negative: no benzene ring at all, so there is no candidate set and the
    # anchor site is never entered -- a trace that recorded unconditionally fails here.
    seen.clear()
    assert br._benzene_parent_candidates(Chem.MolFromSmiles("OC1CCCCC1")) == []
    assert br._preferred_benzene_parent_ring(Chem.MolFromSmiles("OC1CCCCC1")) is None
    assert not seen, seen


def test_parent_selection_hint_needs_no_hand_assigned_principal_group():
    """★ Step 4: ``compute_features`` leaves ``principal_group`` None.

    The Task 9 test concealed this by ASSIGNING the attribute itself, so the
    anchor looked wired while being a no-op for all four external callers
    (rules/ring_chalcogen_oxide.py:277, rules/multiplicative.py:1795 and:2454,
    rules/ions.py:2240). Nothing is assigned here.
    """
    from orthonym.namer import (
        compute_features, _build_ring_info_for_parent_selection,
    )

    smiles = "Oc1c(O)c(O)c(O)c(O)c1Cl"
    mol = Chem.MolFromSmiles(smiles)
    features = compute_features(mol, smiles)
    assert getattr(features, "principal_group", None) is None, (
        "compute_features now sets principal_group -- this test's premise is "
        "stale and the fallback it guards may be dead"
    )

    info = _build_ring_info_for_parent_selection(features)
    assert info and info.get("iupac_locants"), info
    hint = info["iupac_locants"]

    oh_locants, cl_locants = set(), set()
    for atom_idx, locant in hint.items():
        symbols = {
            n.GetSymbol() for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        }
        if "O" in symbols:
            oh_locants.add(locant)
        if "Cl" in symbols:
            cl_locants.add(locant)
    # Must agree with the emitted '6-chlorobenzene-1,2,3,4,5-pentol'.
    assert oh_locants == {1, 2, 3, 4, 5}, hint
    assert cl_locants == {6}, hint
    assert name_compound(smiles) == "6-chlorobenzene-1,2,3,4,5-pentol"


# --------------------------------------------------------------------------
# The anchor must NOT fire for the 74 seniority names that are never promoted
# (Task 9b defect C-2).
# --------------------------------------------------------------------------

def test_anchor_does_not_fire_for_a_seniority_name_that_is_never_promoted():
    """★ C-2 pinned at the predicate.

    76 of the 136 seniority entries have BOTH a ring suffix and a prefix form, and
    the anchor used to fire for every one of them ("the principal group has a
    prefix and it is on this ring"). Only TWO are ever promoted
    (``hydroxy -> -ol`` and the amine -> aniline), so for the other 74 criterion
    (c) minimised the locant of a group the name still spells as a detachable
    prefix -- inverting (f) and (g).
    """
    from orthonym.rules.benzene import principal_group_ring_atoms
    from orthonym.rules.seniority import get_prefix, get_suffix

    # 'ketone' has ring suffix '-one' and prefix 'oxo': promotion-ELIGIBLE by the
    # tables, never promoted by name_substituted_benzene.
    assert get_suffix("ketone", is_ring=True) and get_prefix("ketone") == "oxo"

    mol, ring, subs = _ring_and_subs("Cc1ccccc1Cl")
    # Pretend the molecule-level principal group is the ketone and that an 'oxo'
    # prefix sits on the ring: the OLD code returned that atom, the new code
    # refuses because nothing promotes 'oxo'.
    faked = {
        atom: [dict(s) for s in slist] for atom, slist in subs.items()
    }
    first_atom = sorted(faked)[0]
    faked[first_atom][0]["name"] = "oxo"
    assert principal_group_ring_atoms(
        mol, ring, faked, principal_group="ketone", detected_fgs={},
    ) == set(), "the anchor fired for a group the name never promotes"

    # Control: with the promoted pair it DOES fire, so the assertion above is not
    # vacuous.
    faked[first_atom][0]["name"] = "hydroxy"
    assert principal_group_ring_atoms(
        mol, ring, faked, principal_group="phenol", detected_fgs={},
    ) == {first_atom}
