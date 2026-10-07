"""Lane L2 proper fix: a writer that gives a general spelling records it as never part of
a PIN (``metrics.provenance.record_non_pin_fragment``); no pattern over the finished name
decides the label.

* (the Blue Book) 'benzoyl (preferred prefix)... oxo(phenyl)methyl';
   (:30628) '... cyclohexyl(oxo)methyl';
* (:30851,:30853): concatenated '(azetidin-1-yl)carbonyl' is method (2);
* (:32991,:32998): the amido prefix (method (1)) is the PIN form;
* (:6465), (:23348): an 'a' chain the book does not allow.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.book_prefixes import acyl_group_prefix
from orthonym.cli import _emit_tier_flags
from orthonym.metrics import provenance as prov
from orthonym.rules.pin_vocabulary import non_pin_vocabulary
from tests.support.rt_assert import name_is_rt_exact


def _records(fn):
    prov.clear_provenance()
    out = fn()
    return out, tuple(prov.get_provenance()["non_pin_fragments"])


def test_the_acyl_fallback_records_r_oxo_methyl(monkeypatch):
    import orthonym.assembly.substituent_naming as sn
    monkeypatch.setattr(sn, "acyl_prefix_from_branch", lambda *a, **k: None)
    mol = Chem.MolFromSmiles("c1ccccc1C(=O)C1CCCC1")
    acyl = set(range(6, mol.GetNumAtoms()))
    out, rec = _records(lambda: acyl_group_prefix(mol, 6, 5, acyl, lambda a, r: "cyclopentyl"))
    assert out == "cyclopentyl(oxo)methyl"
    assert "cyclopentyl(oxo)methyl" in rec


def test_a_verified_acyl_name_is_not_recorded():
    mol = Chem.MolFromSmiles("c1ccccc1C(=O)C1CCCC1")
    acyl = set(range(6, mol.GetNumAtoms()))
    out, rec = _records(lambda: acyl_group_prefix(mol, 6, 5, acyl, lambda a, r: "cyclopentyl"))
    assert out == "cyclopentanecarbonyl"
    assert not any("carbonyl" in f or "(oxo)" in f for f in rec)


def test_the_label_patterns_are_gone():
    for name in ("cyclohexyl(oxo)methyl", "4-[(azetidin-1-yl)carbonyl]phenol",
                 "4-[(cyclohexanecarbonyl)amino]phenol"):
        assert non_pin_vocabulary(name) is None, name


def _name_and_records(monkeypatch, smiles, tier):
    seen = []
    orig = prov.record_non_pin_fragment

    def spy(fragment):
        seen.append(fragment)
        return orig(fragment)
    monkeypatch.setattr(prov, "record_non_pin_fragment", spy)
    eng = Orthonym() if tier == "pin" else Orthonym(style="pin", **_emit_tier_flags(tier))
    return eng.name_tiered(smiles), seen


# the rows a deleted pattern alone decided (study-r1, study-audit pv_out.jsonl)
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,form", [
    ("C[C@@H](O)[C@H](NC(=O)[C@@H]1CC[C@](O)(CN)C(=O)N1)C(=O)O", "(oxo)methyl"),
    ("C[N+]1=CC=C(C=C1)CNC(=O)C2=CC3=CC=CC=C3N2CC#CC4=CC=C(C=C4)C(=N)N", "(oxo)methyl"),
    ("C1CS(=O)(=O)CC1N(C(=O)C23CC4CC(C2)CC(C4)C3)NC(=O)C56CC7CC(C5)CC(C7)C6",
     "carbonyl)amino"),
])
def test_the_writer_records_the_general_spelling_it_ships(monkeypatch, smiles, form):
    row, seen = _name_and_records(monkeypatch, smiles, "best-effort")
    name = row.get("name")
    assert name and name_is_rt_exact(name, smiles), row
    assert row.get("tier") != "pin_verified" and row.get("tier") != "pin_unverified", row
    if form in name:
        assert any(form in f and f in name for f in seen), (name, seen)


# properfix a performance pass (controller): the chalcogen-acyl general spelling
# ('R(sulfanylidene)methyl', '(hydroxy)(sulfanylidene)methyl', 'oxo(sulfanyl)
# methyl' and their Se/Te analogues) must record itself as not the PIN
#, the Blue Book 'benzene-1,2-dicarbodithioic acid (PIN)'),
# the SAME fact Task 5 already records for the plain-oxo case -- the acid-
# suffix path is the PIN, never this writer's systematic spelling. The PIN
# controls alongside it (plain oxo/carbamoyl/carbamothioyl/benzoyl/acetyl acyl
# groups, and the two acetic-acid rows / name as the
# PIN already) must stay pin_verified -- the demotion is scoped to the
# chalcogen-acyl case, not every acyl-substituted benzoic/acetic acid.
MC2_ABSTAIN_ROWS = [
    ("S=C(S)c1ccccc1C(=S)S", "1,2-bis[sulfanyl(sulfanylidene)methyl]benzene"),
    ("OC(=O)c1ccccc1C(=S)S", "2-[sulfanyl(sulfanylidene)methyl]benzoic acid"),
]
MC2_PIN_CONTROL_ROWS = [
    ("O=Cc1ccc(C(=O)O)cc1", "4-formylbenzoic acid"),
    ("NC(=O)c1ccc(C(=O)O)cc1", "4-carbamoylbenzoic acid"),
    ("NC(=S)c1ccc(C(=O)O)cc1", "4-carbamothioylbenzoic acid"),
    ("O=C(c1ccccc1)c1ccc(C(=O)O)cc1", "4-benzoylbenzoic acid"),
    ("CC(=O)c1ccc(C(=O)O)cc1", "4-acetylbenzoic acid"),
    ("O=C(C(=O)O)c1ccccc1", "oxo(phenyl)acetic acid"),
    ("OC(C(=O)O)=S", "hydroxy(sulfanylidene)acetic acid"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,systematic_name", MC2_ABSTAIN_ROWS,
                          ids=[r[1] for r in MC2_ABSTAIN_ROWS])
def test_the_chalcogen_acyl_general_spelling_is_never_labelled_pin(
    smiles, systematic_name,
):
    pin_row = Orthonym().name_tiered(smiles)
    assert pin_row.get("tier") == "abstain", pin_row
    be_row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert be_row.get("name") == systematic_name, be_row
    assert be_row.get("tier") == "systematic_verified", be_row
    assert name_is_rt_exact(systematic_name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin_name", MC2_PIN_CONTROL_ROWS,
                          ids=[r[1] for r in MC2_PIN_CONTROL_ROWS])
def test_the_mc2_demotion_does_not_touch_the_pin_controls(smiles, pin_name):
    row = Orthonym().name_tiered(smiles)
    assert (row.get("name"), row.get("tier")) == (pin_name, "pin_verified"), row
    assert name_is_rt_exact(pin_name, smiles)


# properfix a performance pass (review-2 M7): R = H (formyl) is the SAME acyl-
# on-nitrogen class as 'acetyl'/'benzoyl' (the Blue Book '4-formamidobenzoic
# acid (PIN)'), so an N,N-disubstituted amide with a formyl branch records its
# compound-prefix spelling as never the PIN exactly as the other acyl branches do.
# Before this fix formyl was invisible to the structural test, so this row's label
# fell to 'pin_unverified' (neither a certified PIN nor an honestly-labelled
# systematic name) instead of 'systematic_verified'.
@pytest.mark.opsin_gate
def test_a_formyl_branch_on_a_disubstituted_amide_is_recorded_too():
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(
        "CC(=O)CC(=O)OCCN(C)C=O")
    assert row.get("name") == "2-[formyl(methyl)amino]ethyl 3-oxobutanoate", row
    assert row.get("tier") == "systematic_verified", row
    assert name_is_rt_exact(row["name"], "CC(=O)CC(=O)OCCN(C)C=O")


TRITHIA = "C(NC(=O)N)OC(=S)SSSC(=S)OCNC(=O)N"


def test_an_unlicensed_a_chain_is_recorded_by_the_chain_writer():
    from orthonym.assembly.book_prefixes import mechanical_forms
    from orthonym.rules.terminal_fragment import terminal_fragment_name
    m = Chem.MolFromSmiles(TRITHIA)
    frag = {0} | set(range(5, m.GetNumAtoms()))
    with mechanical_forms():
        res, rec = _records(lambda: terminal_fragment_name(m, frag, 0))
    assert res.name == ("11-amino-3,7-disulfanylidene-2,8,12-trioxa-4,5,6-trithia-"
                        "10-azadodec-11-en-1-yl")
    assert "2,8,12-trioxa-4,5,6-trithia-10-azadodec" in rec
    m2 = Chem.MolFromSmiles("CCOCc1ccccc1")
    with mechanical_forms():
        res2, rec2 = _records(lambda: terminal_fragment_name(m2, {0, 1, 2, 3}, 3))
    assert res2.name == "2-oxabutyl" and "2-oxabut" in rec2


def test_a_licensed_a_chain_is_not_recorded():
    from orthonym.rules.terminal_fragment import terminal_fragment_name
    m = Chem.MolFromSmiles("COCCOCCOCCOCc1ccccc1")
    res, rec = _records(lambda: terminal_fragment_name(m, set(range(12)), 11))
    assert res.name == "2,5,8,11-tetraoxadodecyl"
    assert not any("tetraoxadodec" in f for f in rec)


def test_the_floor_records_an_unlicensed_a_chain():
    from orthonym.assembly.book_prefixes import mechanical_forms
    from orthonym.assembly.universal_substituent import name_universal_substitutive
    with mechanical_forms():
        res, rec = _records(lambda: name_universal_substitutive(
            Chem.MolFromSmiles("CCOCCOCC")))
    assert res.name == "3,6-dioxaoctane"
    assert "3,6-dioxaoct" in rec


@pytest.mark.opsin_gate
def test_the_urea_with_an_unlicensed_a_chain_is_never_labelled_a_pin():
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(TRITHIA)
    assert row.get("name") and name_is_rt_exact(row["name"], TRITHIA), row
    assert row["tier"] == "systematic_verified", row
