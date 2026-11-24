"""A large mixture part that only the top-level producers name is named as a compound of its own, once every other
part has a name.

 "Mixed organic – inorganic adducts" (the Blue Book): "names are formed by citing the names of individual
compounds". The last mixture producer puts such a part aside and retries it in the fresh frame only when no other
part is left unnamed (``rules.adducts._component_adduct_words_and_name``), so no large retry is spent on a mixture
that cannot be named anyway.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact


def test_a_large_plain_part_is_put_aside_and_a_small_or_exotic_one_is_not():
    from orthonym.rules.adducts import _large_plain_component as large
    assert large("CCCCCCCCCCc1ccc(CCCCCCCCCC)c(OP(=S)([O-])S)c1")   # 30 heavy atoms, no exotic centre
    assert not large("CC(=O)[O-]")                                   # small: the capped retry already names it
    assert not large("CCCC[Sn+](CCCC)CCCC" + "C" * 6)                # an exotic charged centre: retried at once


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("Cc1cccc(P(S[Na])c2cccc(C)c2)c1", "[bis(3-methylphenyl)phosphanyl]sulfanide—sodium(1+) (1/1)"),
    ("Cc1cccc(P(S[K])c2cccc(C)c2)c1", "[bis(3-methylphenyl)phosphanyl]sulfanide—potassium(1+) (1/1)"),
])
def test_a_large_part_only_the_top_level_names(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


def test_no_retry_for_the_pieces_of_one_molecule(monkeypatch):
    """The hydrazone cut names two pieces of one molecule as a mixture in a nested ``name``: no large retry there,
    so that producer keeps declining and the molecule keeps its name."""
    from orthonym.rules import adducts
    seen = []
    real = adducts._component_word

    def word(smi, style):
        seen.append(getattr(adducts._FRESH_FRAME, "uncapped", False))
        return real(smi, style)
    monkeypatch.setattr(adducts, "_component_word", word)
    smiles = "C1C2C/C(=N\\NC(=O)CCCCCC(=O)N/N=C\\3/C4CC(C3)CC4)/C(C1)C2"
    r = name_best_effort(smiles)
    assert seen and not any(seen), seen
    assert r["name"] == ("(1E,2E)-2-({7-[(bicyclo[2.2.1]heptan-2-ylidene)hydrazinyl]-1,7-dioxoheptyl}hydrazinylidene)"
                         "bicyclo[2.2.1]heptane"), r


def test_the_flag_is_restored_after_an_exception(monkeypatch):
    from orthonym.rules import adducts
    from rdkit import Chem
    calls = []

    def word(smi, style):
        calls.append(smi)
        if len(calls) > 1:
            raise RuntimeError("boom")
        return None
    monkeypatch.setattr(adducts, "_component_word", word)
    monkeypatch.setattr(adducts, "_large_plain_component", lambda smi: True)
    monkeypatch.setattr("orthonym.assembly.fragment_naming.name_scope_depth", lambda: 1)
    adducts._FRESH_FRAME.uncapped = False
    comps = [("CCCCCCCCCCCCCCCCCC(=O)O", 1), ("O", 1)]
    mols = {s: Chem.MolFromSmiles(s) for s, _ in comps}
    with pytest.raises(RuntimeError):
        adducts._component_adduct_words_and_name(comps, mols, Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCC(=O)O.O"), "pin")
    assert adducts._FRESH_FRAME.uncapped is False
