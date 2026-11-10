"""The free acid of an ester side is named once per naming, and a memo hit notes the book spellings of that naming
again (program 22, 2026-11-10). The polyol-ester routes asked the free acid of one hemi-carbonate side 22 times in
one naming; ``rules.free_acid_names.free_acid_name`` now keeps the first name. The enclosing ``watch_book_forms`` box decides
whether the molecule is named again with the mechanical spellings, so a hit must note what the fresh naming noted."""
import pytest

pytestmark = pytest.mark.opsin_gate


def test_a_memo_hit_notes_the_book_spellings_again(monkeypatch):
    """A stand-in naming that gives a ring spelling: the first call (fresh) and the second (a hit) both note it in
    the box that encloses them."""
    import orthonym.rules.free_acid_names as esters
    from orthonym.assembly.book_prefixes import note_book_form, watch_book_forms
    from orthonym.assembly.memo import pop_scope, push_scope
    runs = []

    def stand_in(acid_smi):
        runs.append(acid_smi)
        note_book_form("ring")
        return "pyridine-4-carboxylic acid"

    monkeypatch.setattr(esters, "_free_acid_name_once", stand_in)
    token = push_scope()
    try:
        with watch_book_forms() as first:
            a = esters.free_acid_name("OC(=O)c1ccncc1")
        with watch_book_forms() as second:
            b = esters.free_acid_name("OC(=O)c1ccncc1")
    finally:
        pop_scope(token)
    assert a == b == "pyridine-4-carboxylic acid" and len(runs) == 1
    assert first == second == [True, {"ring"}]


def test_the_spiro_carbonate_names_its_free_acid_once(monkeypatch):
    import orthonym.rules.free_acid_names as esters
    from orthonym import name_compound
    calls = []
    once = esters._free_acid_name_once
    monkeypatch.setattr(esters, "_free_acid_name_once", lambda s: calls.append(s) or once(s))
    name_compound("CCOC(=O)OC1=C(c2cc(C)ccc2C)C(=O)N[C@]12CC[C@@H](OC)CC2")
    assert calls and max(calls.count(s) for s in set(calls)) < 22
