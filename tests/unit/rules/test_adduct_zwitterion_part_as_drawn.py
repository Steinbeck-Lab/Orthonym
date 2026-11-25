"""A zwitterion part of an ionic mixture is named as drawn when that name reproduces the drawing.

 "Ionic centers in the same parent structure" (the Blue Book): zwitterionic compounds "are not considered
as neutral compounds"; "Mixed organic – inorganic adducts" (:4667): the mixture cites the names of its
components. The part alone takes its neutral-form name at best-effort (``salts.name_zwitterion``'s fallback, the same
standard InChIKey), and inside the mixture that name puts a hydron where the drawing has none, which the namer's exit
gate refuses. ``rules.adducts._zwitterion_drawn_name`` renames such a part once in a fresh frame with that fallback off
(``salts.NO_NEUTRAL_ZWITTERION_FALLBACK``); the name ships only on the exact depiction check, and a name that ships
today (its hydrons where drawn) is kept.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

LITHIUM_QUINOLINIUM_OLATE = "[Li+].CC(C)(C)C(C1=[NH+]C2=C(C=CC=C2[O-])C=C1)C(C)(C)C"


@pytest.mark.opsin_gate
def test_the_zwitterion_part_is_named_as_drawn():
    r = name_best_effort(LITHIUM_QUINOLINIUM_OLATE)
    assert r["tier"] == "systematic_verified" and not r["is_pin"], r
    assert "-ium-" in r["name"] and "olate" in r["name"] and "lithium(1+)" in r["name"], r
    assert name_is_rt_exact(r["name"], LITHIUM_QUINOLINIUM_OLATE)


def test_the_switch_skips_the_neutral_form_fallback(monkeypatch):
    from rdkit import Chem
    from orthonym.metrics.provenance import best_effort_ctx
    from orthonym.rules import salts
    called = []
    monkeypatch.setattr(salts, "_name_general_zwitterion", lambda mol, style: called.append(1) or "neutral form")
    monkeypatch.setattr(salts, "_is_amino_acid_zwitterion", lambda mol: False)
    mol = Chem.MolFromSmiles("CC(C)(C)C(C1=[NH+]C2=C(C=CC=C2[O-])C=C1)C(C)(C)C")
    token = best_effort_ctx.set(True)
    flag = salts.NO_NEUTRAL_ZWITTERION_FALLBACK.set(True)
    try:
        assert salts.name_zwitterion(mol) == ""
    finally:
        salts.NO_NEUTRAL_ZWITTERION_FALLBACK.reset(flag)
        best_effort_ctx.reset(token)
    assert not called


def test_a_plain_salt_and_a_lone_zwitterion_keep_their_names():
    assert name_best_effort("CC(=O)[O-].[Na+]")["name"] == "sodium acetate"
    assert name_best_effort("[NH3+]CC(=O)[O-]")["name"] == "azaniumylacetate"


@pytest.mark.opsin_gate
def test_only_at_best_effort_and_the_part_alone_keeps_its_name():
    """The switch only acts at best-effort (``salts.name_zwitterion`` has no neutral fallback below it), so the rename
    runs at best-effort only: the complete tier declines the mixture as before. The part named alone afterwards keeps
    its neutral-form name: the switch does not outlive the mixture's naming."""
    from orthonym import cli
    complete = cli._tiered_namer("pin", "complete").name_tiered(LITHIUM_QUINOLINIUM_OLATE)
    assert complete["tier"] == "abstain", complete
    name_best_effort(LITHIUM_QUINOLINIUM_OLATE)
    part = name_best_effort("CC(C)(C)C(C1=[NH+]C2=C(C=CC=C2[O-])C=C1)C(C)(C)C")
    assert part["name"].endswith("quinolin-8-ol"), part
