"""Leads item 29: the name of a cation does not depend on where its SMILES draws the charge.

'Cc1c[nH]c[nH+]1' and 'Cc1c[nH+]c[nH]1' are one species (the formal +1 sits on either NH nitrogen,
every atom keeps its hydrogens, one standard InChIKey) and named '4-methyl-1H-imidazol-3-ium' and
'5-methyl-1H-imidazol-3-ium', both labelled pin_verified; the second is not the PIN. Likewise
'CCCCn1cc[n+](C)c1' and 'CCCC[n+]1ccn(C)c1' gave '1-butyl-3-methyl-' and '3-butyl-1-methyl-1,3-diazol-
3-ium', and the PIN tier emitted for one drawing of '1,3,8-trimethyl[1,2,4]triazolo[4,3-b]pyridazin-1-ium'
and of N,N,N',N'-tetrapropylguanidinium and abstained for the other.

``namer._charge_forms`` enumerates the charge-only resonance forms of an input (a +1 moved between
nitrogen atoms, no hydrogen moved, one standard InChIKey) and ``Orthonym._name_with_form_choice`` names
each as an outermost ``name`` call: the highest label, then the lower locants by (b) indicated
hydrogen, (c) the ionic suffix, (e) the hydro prefixes, (f) the set of the prefix locants, (g) the
prefix cited first, then the first form by canonical SMILES. The form the caller drew is one of them, so
the result is never lower than the drawing's.

 'NUMBERING' (the Blue Book): 'low locants are assigned to them in the following decreasing
order of seniority' (:3221); (c) 'principal characteristic groups and free valences (suffixes)' (:3256);
(f) 'detachable alphabetized prefixes, all considered together in a series of increasing numerical order'
(:3301); (g) 'lowest locants for the substituent cited first as a prefix in the name' (:3307).
'General rule for systematically naming cationic centers in parent hydrides' (:41366), example
'1H-imidazol-3-ium (PIN)' (:41396). Every name is read back by OPSIN's own StdInChIKey against RDKit's key
of the input.
"""
import random

import pytest
from rdkit import Chem

from orthonym import Orthonym, name_compound, namer
from orthonym.cli import _emit_tier_flags
from tests.unit.namer.leads_l1_support import opsin_key, rdkit_key

pytestmark = pytest.mark.opsin_gate

#: (the two drawings of one species, the PIN, which the PIN tier builds for both)
PINS = [
    ("Cc1c[nH]c[nH+]1", "Cc1c[nH+]c[nH]1", "4-methyl-1H-imidazol-3-ium"),
    ("Clc1ccc2[nH]c[nH+]c2c1", "Clc1ccc2[nH+]c[nH]c2c1", "5-chloro-1H-1,3-benzimidazol-3-ium"),
    ("CCc1[nH+]c(C)c[nH]1", "CCc1[nH]c(C)c[nH+]1", "2-ethyl-4-methyl-1H-imidazol-3-ium"),
    ("Cc1cc[nH][nH+]1", "Cc1cc[nH+][nH]1", "3-methyl-1H-pyrazol-2-ium"),
    ("Cc1ccnn2c(C)n[n+](C)c12", "Cc1ccn[n+]2c(C)nn(C)c12",
     "1,3,8-trimethyl[1,2,4]triazolo[4,3-b]pyridazin-1-ium"),
    ("CCCN(CCC)C(=[N+](CCC)CCC)N", "CCCN(CCC)C(N(CCC)CCC)=[NH2+]",
     "N,N,N',N'-tetrapropylguanidinium"),
    ("CNC(=[NH2+])NCc1ccccc1", "C[NH+]=C(N)NCc1ccccc1", "N-benzyl-N'-methylguanidinium"),
]
#: the N-alkyl azolium pair: no PIN is built (the PIN tier declines); the wider tiers name it
IONIC_LIQUID = ("CCCCn1cc[n+](C)c1", "CCCC[n+]1ccn(C)c1")
WIDER = ["valid", "complete", "best-effort"]


def _engine(tier):
    return (Orthonym(style="pin") if tier == "pin"
            else Orthonym(style="pin", **_emit_tier_flags(tier)))


@pytest.mark.parametrize("a, b, pin", PINS)
def test_the_pin_reads_back_as_both_drawings(a, b, pin):
    assert rdkit_key(a) == rdkit_key(b)                 # one species, one standard InChIKey
    assert opsin_key(pin) == rdkit_key(a), pin


@pytest.mark.parametrize("tier", ["pin", *WIDER])
@pytest.mark.parametrize("a, b, pin", PINS)
def test_both_drawings_get_the_pin_labelled_pin_verified(a, b, pin, tier):
    engine = _engine(tier)
    for smiles in (a, b):
        row = engine.name_tiered(smiles)
        assert (row["name"], row["tier"]) == (pin, "pin_verified"), (tier, smiles, row)
        assert row["is_pin"] is True


@pytest.mark.parametrize("a, b, pin", PINS[:3])
def test_every_public_entry_gives_the_one_name(a, b, pin):
    o = Orthonym(style="pin")
    for smiles in (a, b):
        assert o.name(smiles) == pin
        assert name_compound(smiles) == pin
        assert o.name_with_confidence(smiles)["name"] == pin
        assert o.name_with_tree(smiles).name == pin


def test_random_atom_orders_of_both_drawings_give_the_pin():
    smiles, pin = "Cc1c[nH]c[nH+]1", "4-methyl-1H-imidazol-3-ium"
    mol = Chem.MolFromSmiles(smiles)
    state = random.getstate()
    random.seed(3)
    orders = {Chem.MolToSmiles(mol, doRandom=True) for _ in range(40)}
    random.setstate(state)
    assert len(orders) >= 6
    engine = Orthonym(style="pin")
    for order in sorted(orders):
        assert engine.name(order) == pin, order


@pytest.mark.parametrize("tier", WIDER)
def test_the_ionic_liquid_cation_gets_one_name_by_the_lower_locants(tier):
    # (g) (:3307): the substituent cited first as a prefix, butyl, gets the lowest locant
    a, b = IONIC_LIQUID
    rows = [_engine(tier).name_tiered(s) for s in (a, b)]
    assert rows[0]["name"] == rows[1]["name"] and rows[0]["tier"] == rows[1]["tier"]
    assert rows[0]["name"].startswith("1-butyl-3-methyl-1"), rows[0]
    assert rows[0]["tier"] == "systematic_verified"
    assert opsin_key(rows[0]["name"]) == rdkit_key(a)


#: ring cations with two prefixes whose first-cited one has the higher locant in the numbering that
#: has the lower locant set: (f) decides before (g)
BENZIMIDAZOLIUMS = [
    ("Brc1cc(Cl)c2[nH]c[nH+]c2c1", "Brc1cc(Cl)c2[nH+]c[nH]c2c1",
     "6-bromo-4-chloro-1H-1,3-benzimidazol-3-ium"),
    ("Clc1cc(C)c2[nH]c[nH+]c2c1", "Clc1cc(C)c2[nH+]c[nH]c2c1",
     "6-chloro-4-methyl-1H-1,3-benzimidazol-3-ium"),
]


@pytest.mark.parametrize("tier", ["pin", *WIDER])
@pytest.mark.parametrize("a, b, pin", BENZIMIDAZOLIUMS)
def test_the_prefix_locant_set_is_compared_before_the_first_cited_prefix(a, b, pin, tier):
    # 'NUMBERING' (the Blue Book), (f) 'detachable alphabetized prefixes, all considered
    # together in a series of increasing numerical order' (:3301), then (g) 'lowest locants for the
    # substituent cited first as a prefix in the name' (:3307): the locant set 4,6 is lower than 5,7
    # (as in the example of (f),:3305 '5-bromo-8-hydroxy-4-methylazulene-2-carboxylic acid (PIN)
    # (the locant set '4,5,8' is lower than '4,7,8')'). Mutation: compare the locants in the order
    # they are cited only, and both drawings give '5-bromo-7-chloro-' / '5-chloro-7-methyl-'.
    assert rdkit_key(a) == rdkit_key(b)
    assert opsin_key(pin) == rdkit_key(a), pin
    for smiles in (a, b):
        row = _engine(tier).name_tiered(smiles)
        assert (row["name"], row["tier"]) == (pin, "pin_verified"), (tier, smiles, row)
        assert row["is_pin"] is True


def test_the_ionic_liquid_cation_is_declined_by_the_default_tier_for_both_drawings():
    rows = [_engine("pin").name_tiered(s) for s in IONIC_LIQUID]
    assert [r["tier"] for r in rows] == ["abstain", "abstain"]
    assert rows[0]["limit_code"] == rows[1]["limit_code"]


def test_the_two_forms_of_a_zwitterion_get_one_name_at_the_wider_tiers():
    # glycocyamine: the +1 on the unsubstituted or on the substituted guanidinium nitrogen
    a, b = "NC(=[NH2+])NCC(=O)[O-]", "NC(N)=[NH+]CC(=O)[O-]"
    for tier in WIDER:
        rows = [_engine(tier).name_tiered(s) for s in (a, b)]
        assert (rows[0]["name"], rows[0]["tier"]) == (rows[1]["name"], rows[1]["tier"]), tier


# ---- the form enumeration -------------------------------------------------------------------

def test_the_forms_of_an_imidazolium_and_the_form_the_caller_drew():
    forms = namer._charge_forms("Cc1c[nH]c[nH+]1")
    strings = [f[0] for f in forms]
    canon = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in strings]
    assert canon == sorted(canon) and len(set(canon)) == 2
    # the form the caller drew is its own string (order None); the other is RDKit's
    assert [f[1] is None for f in forms].count(True) == 1
    assert "Cc1c[nH]c[nH+]1" in strings
    # both drawings of one species give the same forms, in the same canonical order
    other = namer._charge_forms("Cc1c[nH+]c[nH]1")
    assert [Chem.MolToSmiles(Chem.MolFromSmiles(f[0])) for f in other] == canon
    assert sorted(Chem.MolToSmiles(Chem.MolFromSmiles(f[0])) for f in other) == sorted(canon)


@pytest.mark.parametrize("smiles", [
    "c1cc[nH+]cc1",                       # one form
    "C[N+](=O)[O-]", "c1ccccc1[N+](=O)[O-]",      # the charge sits on the oxygen
    "C[N+](C)(C)C", "C[NH3+]",            # no conjugation
    "C[n+]1ccccc1",                       # one nitrogen
    "CCO",                                # no charge
    "C[C@H](O)CC |&1:1|",                 # a CXSMILES string: the stereo group is read first
    "C[S@](=O)CC.CC(=[NH+]C)N",           # a lone-pair stereocentre
])
def test_inputs_with_fewer_than_two_forms_or_out_of_bounds_are_named_as_drawn(smiles):
    assert namer._charge_forms(smiles) is None


def test_the_iminium_depiction_of_an_aromatic_amine_is_no_form_of_its_species():
    # '[NH2+]=c1cc[nH]cc1' is a resonance form of 4-aminopyridinium that moves the charge through
    # the ring, not between two nitrogens of one amidinium or azolium unit: that class is named
    # as it is drawn, as before (the iminium drawing is 'pyridin-4(1H)-iminium' pin_verified
    # where the conventional drawing is declined; offering it to the conventional drawing would
    # label a name the PIN that no one checked)
    for smiles in ("Nc1cc[nH+]cc1", "[NH2+]=c1cc[nH]cc1", "Nc1cccc[nH+]1",
                   "CN(C)c1cc[nH+]cc1", "C[n+]1ccc(N)cc1", "C[n+]1c(N)cccc1"):
        assert namer._charge_forms(smiles) is None, smiles
    o = Orthonym(style="pin")
    assert o.name("[NH2+]=c1cc[nH]cc1") == "pyridin-4(1H)-iminium"


def test_the_enumeration_is_bounded_and_no_hydrogen_moves():
    # 20 amidinium groups: more forms than the bound, named as drawn
    many = ".".join(["CC(=[NH+]C)N"] * 20)
    assert namer._charge_forms(many) is None
    # '1-methyl-1H-benzimidazol-1-ium' (H on N3, +1 on N1 methyl...) and its tautomer-like
    # H-moved isomer are different species: no form moves a hydrogen
    for smiles in ("C[n+]1c[nH]c2ccccc21", "Cn1c[nH+]c2ccccc21"):
        forms = namer._charge_forms(smiles)
        assert forms is not None
        for string, _order in forms:
            mol = Chem.MolFromSmiles(string)
            assert sorted(a.GetTotalNumHs() for a in mol.GetAtoms()) == sorted(
                a.GetTotalNumHs() for a in Chem.MolFromSmiles(smiles).GetAtoms())
            assert Chem.MolToInchiKey(mol) == Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


# ---- the choice -------------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", [
    ("4-methyl-1H-imidazol-3-ium", "5-methyl-1H-imidazol-3-ium"),
    ("1-butyl-3-methyl-1,3-diazol-3-ium", "3-butyl-1-methyl-1,3-diazol-3-ium"),
    ("2-ethyl-4-methyl-1H-imidazol-3-ium", "2-ethyl-5-methyl-1H-imidazol-3-ium"),
    ("N-benzyl-N'-methylguanidinium", "N'-benzyl-N-methylguanidinium"),
    ("3-methyl-1H-pyrazol-2-ium", "5-methyl-1H-pyrazol-2-ium"),
    ("4-methyl-1H-imidazol-3-ium", "4-methyl-3H-imidazol-1-ium"),
    # (f) 'detachable alphabetized prefixes, all considered together in a series of increasing
    # numerical order' (:3301) before (g) the first cited (:3307): the set 4,6 is lower than 5,7
    # although the first cited prefix, bromo, has the higher locant in the first
    ("6-bromo-4-chloro-1H-1,3-benzimidazol-3-ium", "5-bromo-7-chloro-1H-1,3-benzimidazol-3-ium"),
    ("6-chloro-4-methyl-1H-1,3-benzimidazol-3-ium", "5-chloro-7-methyl-1H-1,3-benzimidazol-3-ium"),
    # (g) decides only when the sets are equal: the same set 4,5 with the first cited at 4
    ("4-bromo-5-chloro-1H-1,3-benzimidazol-3-ium", "5-bromo-4-chloro-1H-1,3-benzimidazol-3-ium"),
    # (e)(i) 'low locants are given to hydro/dehydro prefixes' (:3289) before (f): the hydro locants
    # 2,3 are lower than 4,5 although the methyl has the higher locant in the first (a comparison of
    # the roles, not a pair of names of one compound)
    ("5-methyl-2,3-dihydro-1H-imidazol-3-ium", "2-methyl-4,5-dihydro-1H-imidazol-3-ium"),
])
def test_the_lower_locants_by_p14_4(a, b):
    assert namer._lower_locants(a, b) == -1 and namer._lower_locants(b, a) == 1
    assert namer._lower_locants(a, a) == 0


def test_names_that_are_not_one_skeleton_are_not_compared():
    assert namer._lower_locants("4-methyl-1H-imidazol-3-ium", "1-methyl-1H-imidazole") == 0
    assert namer._lower_locants("(carbamimidoylamino)acetic acid",
                                "[(diaminomethylidene)amino]acetic acid") == 0


def test_the_choice_never_gives_a_lower_label_than_the_drawing(monkeypatch):
    # the drawing's own run is one of the candidates: when the other form is declined, the
    # drawn form's name stands, and the result is still one name for both drawings
    calls = []
    original = Orthonym._name_with_form_choice

    def spy(self, fn, forms, args, kwargs):
        calls.append([f[0] for f in forms])
        return original(self, fn, forms, args, kwargs)

    monkeypatch.setattr(Orthonym, "_name_with_form_choice", spy)
    a, b, pin = PINS[4]                      # the drawing that the PIN tier used to decline: b
    assert Orthonym(style="pin").name(b) == pin
    assert Orthonym(style="pin").name(a) == pin
    assert len(calls) == 2 and all(len(c) == 2 for c in calls)


def test_a_single_form_input_does_not_enter_the_choice(monkeypatch):
    def boom(self, fn, forms, args, kwargs):
        raise AssertionError("the form choice ran for a single-form input")

    monkeypatch.setattr(Orthonym, "_name_with_form_choice", boom)
    assert Orthonym(style="pin").name("c1cc[nH+]cc1") == "pyridin-1-ium"
    assert Orthonym(style="pin").name("CCO") == "ethanol"


def test_a_form_run_keeps_the_scope_that_rekeys_atom_maps():
    # the atom map of the string a form names is re-keyed to the caller's atoms through the
    # composed atom order, as for the other rewrites of the input scope
    caller = "Cc1c[nH+]c[nH]1"
    seen = []

    def fn(self, string, *args, **kwargs):
        seen.append((string, namer._LP_INPUT.rewrite))
        return "4-methyl-1H-imidazol-3-ium"

    named, opened = namer._lone_pair_input_enter(caller)
    try:
        forms = namer._charge_forms(named)
        Orthonym(style="pin")._name_with_form_choice(fn, forms, (named,), {})
        given = Chem.MolFromSmiles(caller)
        for string, rewrite in seen:
            if string == named:
                assert rewrite is None
                continue
            assert rewrite[0] == string
            out = Chem.MolFromSmiles(string)
            rekeyed = namer._lone_pair_caller_atoms({k: k for k in range(out.GetNumAtoms())})
            assert sorted(rekeyed) == list(range(given.GetNumAtoms()))
            for caller_idx, named_idx in rekeyed.items():
                assert (given.GetAtomWithIdx(caller_idx).GetSymbol()
                        == out.GetAtomWithIdx(named_idx).GetSymbol())
    finally:
        namer._lone_pair_input_exit(opened)


# an amidinium arm: the strict path names '2-[(diaminomethylideneazaniumyl)methyl]butanedioate' and
# records the unmarked 'azaniumyl' as not the PIN, so the default tier declines it (NO_VERIFIED_PIN)
# and the wider tiers keep the name (tests/unit/rules/test_aminium_polyacid.py)
AMIDINIUM_ARM = ("NC(N)=[NH+]CC(CC(=O)[O-])C(=O)[O-]", "[NH2+]=C(N)NCC(CC(=O)[O-])C(=O)[O-]")


def test_the_forms_of_an_amidinium_arm_are_two():
    forms = namer._charge_forms(AMIDINIUM_ARM[0])
    assert forms is not None and len(forms) == 2
    assert rdkit_key(AMIDINIUM_ARM[0]) == rdkit_key(AMIDINIUM_ARM[1])


@pytest.mark.parametrize("smiles", AMIDINIUM_ARM)
def test_a_decline_keeps_its_reason_and_is_never_certified_for_either_drawing(smiles):
    # one drawing's other form names nothing at all (UNNAMEABLE): the decline that says a name
    # exists below the PIN (NO_VERIFIED_PIN) stands over it, for either drawing; and the run of
    # the winner that follows the other form's run must not label the name the PIN (a form's run
    # shares the memo of ``name_tiered``'s scope unless it has one of its own)
    row = Orthonym(style="pin").name_tiered(smiles)
    assert (row["tier"], row["limit_code"], row["is_pin"]) == ("abstain", "NO_VERIFIED_PIN", False), row
    assert Orthonym(style="pin").name(smiles) == row["name"]


@pytest.mark.parametrize("tier", WIDER)
def test_the_wider_tiers_give_both_drawings_of_an_amidinium_arm_one_name(tier):
    rows = [_engine(tier).name_tiered(s) for s in AMIDINIUM_ARM]
    assert rows[0]["name"] == rows[1]["name"], rows
    assert rows[0]["tier"] == rows[1]["tier"] != "pin_verified", rows
    assert opsin_key(rows[0]["name"]) == rdkit_key(AMIDINIUM_ARM[0]), rows[0]
    if tier == "best-effort":
        # the name the strict path gives the second drawing, which main gave that drawing alone
        assert rows[0]["name"] == "2-({[amino(azaniumylidene)methyl]amino}methyl)butanedioate", rows


def test_each_form_is_named_in_a_memo_scope_of_its_own():
    # mutation: with the sandbox taken out (the form runs sharing the scope of the enclosing
    # ``name_tiered``) the second run reads what the first memoised, and the count is 1
    from orthonym.assembly import memo
    forms = namer._charge_forms("Cc1c[nH+]c[nH]1")
    assert forms is not None and len(forms) == 2
    computed = []

    def fn(self, string, *args, **kwargs):
        memo.cache_or_compute("leads_l1_form_probe", ("k",), lambda: computed.append(string) or 1)
        return "4-methyl-1H-imidazol-3-ium"

    token = memo.push_scope()
    try:
        Orthonym(style="pin")._name_with_form_choice(fn, forms, ("Cc1c[nH+]c[nH]1",), {})
        assert memo.cache_or_compute("leads_l1_form_probe", ("k",), lambda: computed.append("outer") or 2) == 2
    finally:
        memo.pop_scope(token)
    assert sorted(computed) == sorted([f[0] for f in forms] + ["outer"]), computed


def test_a_dipole_is_not_a_charge_form_and_is_named_as_drawn():
    # an azide moves a -1 between its resonance structures: no form, so the choice never runs;
    # the acyl azide was named by the form whose canonical SMILES comes first, as a phenol with an
    # 'azido-oxoethyl' prefix, labelled pin_verified like the PIN it replaced
    acyl_azide = "[N-]=[N+]=NC(=O)Cc1ccc(O)c([N+](=O)[O-])c1"
    pin = "(4-hydroxy-3-nitrophenyl)acetyl azide"
    for smiles in ("CN=[N+]=[N-]", "C[N-][N+]#N", acyl_azide, "C=[N+]=[N-]"):
        assert namer._charge_forms(smiles) is None, smiles
    assert opsin_key(pin) == rdkit_key(acyl_azide), pin
    for tier in ("pin", *WIDER):
        row = _engine(tier).name_tiered(acyl_azide)
        assert (row["name"], row["tier"]) == (pin, "pin_verified"), (tier, row)


def _stub_choice(monkeypatch, labels, drawn):
    """The form choice over the two forms of ``drawn`` (an imidazolium), the run of each form
    giving ``names[form string]`` with the label ``labels[form string]``."""
    forms = namer._charge_forms(drawn)
    assert forms is not None and len(forms) == 2

    def tier_row(self, smiles, name, prov):
        return {"tier": labels[smiles]}, True

    monkeypatch.setattr(Orthonym, "_tier_row_and_pin_form", tier_row)
    return Orthonym(style="pin"), forms


def _choose(monkeypatch, drawn, names, label):
    labels = {string: label for string in names}
    engine, forms = _stub_choice(monkeypatch, labels, drawn)

    def fn(self, string, *args, **kwargs):
        return names[string]

    return engine._name_with_form_choice(fn, forms, (drawn,), {})


def test_a_tie_no_rule_decides_between_pin_labels_keeps_the_drawings_name(monkeypatch):
    # the drawing whose canonical SMILES comes second, so that the canonical order and the
    # drawing disagree
    drawn, other = "Cc1c[nH]c[nH+]1", "Cc1c[nH+]c[nH]1"
    assert namer._charge_forms(drawn)[0][0] == other
    names = {drawn: "name that the drawing gave", other: "another name, not the same skeleton"}
    assert _choose(monkeypatch, drawn, names, "pin_verified") == names[drawn]
    # the canonical order decides the same tie when the labels are below the PIN
    assert _choose(monkeypatch, drawn, names, "systematic_verified") == names[other]
    # and the lower locants decide a tie between names of one skeleton, whichever the drawing
    names = {drawn: "5-methyl-1H-imidazol-3-ium", other: "4-methyl-1H-imidazol-3-ium"}
    assert _choose(monkeypatch, drawn, names, "pin_verified") == "4-methyl-1H-imidazol-3-ium"
    names = {drawn: "4-methyl-1H-imidazol-3-ium", other: "5-methyl-1H-imidazol-3-ium"}
    assert _choose(monkeypatch, drawn, names, "pin_verified") == "4-methyl-1H-imidazol-3-ium"


# ---- the cost of the choice is bounded by the input, not by the clock ---------------------------

#: Each form is a whole naming run, and a run of a peptide with arginines takes seconds (measured at
#: the best-effort tier on this tree: 39 heavy atoms x 4 forms 6 s, 51 x 4 15 s, 62 x 5 19 s). The
#: bound is decided before any form is named, from the input alone (heavy atoms x forms); a wall-clock
#: cutoff made the set of forms named, and with it the name, depend on the load of the machine.
PEPTIDE_INSIDE = ("C[C@H](NC(=O)[C@@H](N)CCCNC(N)=[NH2+])C(=O)N[C@@H](Cc1ccccc1)C(=O)"
                  "N[C@@H](CCCNC(N)=[NH2+])C(=O)O")                      # 39 x 4 = 156
PEPTIDE_OUTSIDE = ("NC(=[NH2+])NCCC[C@H](NC(=O)[C@@H](NC(=O)[C@@H](N)Cc1ccc(O)cc1)CC(C)C)C(=O)"
                   "N[C@@H](CCCNC(N)=[NH2+])C(=O)N[C@@H](CC(C)C)C(=O)O")  # 51 x 4 = 204


def test_the_work_of_the_choice_is_bounded_before_any_form_is_named(monkeypatch):
    inside = namer._charge_forms(PEPTIDE_INSIDE)
    assert inside is not None and len(inside) == 4
    assert Chem.MolFromSmiles(PEPTIDE_INSIDE).GetNumHeavyAtoms() * len(inside) <= namer._FORMS_MAX_WORK
    # over the bound the input is named as drawn (no choice), whatever the speed of the machine
    assert namer._charge_forms(PEPTIDE_OUTSIDE) is None
    # mutation: with the bound out of the way the same input has forms (4 of them)
    monkeypatch.setattr(namer, "_FORMS_MAX_WORK", 10 ** 6)
    namer._charge_forms.cache_clear()
    try:
        assert len(namer._charge_forms(PEPTIDE_OUTSIDE)) == 4
    finally:
        monkeypatch.undo()
        namer._charge_forms.cache_clear()


def test_the_form_chosen_does_not_depend_on_how_long_the_forms_take(monkeypatch):
    # a clock that has run far past any limit after every reading: the forms named, and the name
    # chosen, are the same as with a clock that stands still (the lower locants win although that
    # form is named second)
    import itertools
    import time
    drawn, other = "Cc1c[nH]c[nH+]1", "Cc1c[nH+]c[nH]1"
    names = {drawn: "5-methyl-1H-imidazol-3-ium", other: "4-methyl-1H-imidazol-3-ium"}
    ticks = itertools.count(0, 1000)
    monkeypatch.setattr(time, "monotonic", lambda: float(next(ticks)))
    assert _choose(monkeypatch, drawn, names, "pin_verified") == "4-methyl-1H-imidazol-3-ium"
    assert _choose(monkeypatch, other, names, "pin_verified") == "4-methyl-1H-imidazol-3-ium"
