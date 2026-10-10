"""Leads program L6, item 31: the divalent tin and lead radicals take the Group-14 radical names.

RDKit gives the bracket atom of 'CC[Sn]CC' two radical electrons, as it does the bracket atom
of 'CC[Ge]CC' and 'CC[Si]CC', so one input state is named by two rules: germanium and silicon
reach the radical route of ``charged_router.route_charged`` ('dimethylgermylidene'), while tin
and lead are metals to ``ions._has_metal`` and were refused at its step 1. The PIN tier then
abstained ('tin compound (not supported)') and the wider tiers named the lambda-convention
skeletal chain ('3λ2-stannapentane'), which is general nomenclature only.

Blue Book, 'Specific method and retained names' (the Blue Book): "A radical
formally derived by the removal of two hydrogen atom from one skeletal atom of a mononuclear
parent hydride of an element of Group 14,... is named by replacing the 'ane' ending of the
systematic name of the parent hydride by the suffix '-ylidene' or '-diyl'. The suffix
'-ylidyne' or '-triyl' is used to name radicals formally derived by the removal of three
hydrogen atoms from a mononuclear parent hydride of an element of Group 14" (:40476); examples
':40484' 'silylidene (preselected name)', ':40490' 'benzylsilylidene (PIN)'.
'Monovalent radicals' (:40412), (:40414): the same replacement by 'yl' for one
hydrogen atom. The lambda form is general nomenclature only: 'The λ-convention'
(:40560): "This method is only for general nomenclature" (:40562); 'dichloromethylidene (PIN)'
(:40566). 'stannyl (preselected prefix)' (:38065), 'plumbyl (preselected prefix)' (:38069).

Every expected name is read back by OPSIN to the input's full InChIKey and radical profile
(``tests.support.rt_assert.name_is_rt_exact``).

Mutation check (run once, recorded in internal notes): making
``charged_router._group14_metal_radical_element`` return None (tin behind the gate again) fails
every pin test and the unit test of the helper below.
"""
import pytest
from rdkit import Chem

import orthonym.rules.charged_router as cr
from orthonym.assembly import fragment_naming as fn
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

# (SMILES, PIN) -- every PIN below is read back by OPSIN to the input's full key and radicals.
PINS = [
    ("CC[Sn]CC", "diethylstannylidene"),
    ("Cl[Sn]Cl", "dichlorostannylidene"),
    ("C[Pb]C", "dimethylplumbylidene"),
    ("C[SnH]C", "dimethylstannyl"),
    ("[SnH2]", "stannylidene"),
    ("C[Sn](C)C", "trimethylstannyl"),
    ("CC[Sn]", "ethylstannylidyne"),
    ("c1ccccc1[Sn]c1ccccc1", "diphenylstannylidene"),
    ("Cl[Sn]CC", "chloro(ethyl)stannylidene"),
    # the controls with the same input state: silicon and germanium, unchanged
    ("C[Ge]C", "dimethylgermylidene"),
    ("C[Si]C", "dimethylsilylidene"),
    ("Cl[Ge]Cl", "dichlorogermylidene"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", PINS, ids=[s for s, _ in PINS])
def test_the_group14_radical_is_named_on_its_parent_hydride(smiles, pin):
    """The default tier and the breadth tier both give the name, labelled PIN."""
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["CC[Sn]CC", "C[Pb]C", "Cl[Sn]Cl"])
def test_the_lambda_chain_is_no_longer_the_name(smiles):
    """Before: the wider tiers named '3λ2-stannapentane' / '2λ2-plumbapropane' /
    '1,1-dichloro-1-stannamethan-1-ylidene' (general nomenclature,:40562)."""
    for res in (name_default(smiles), name_breadth(smiles)):
        name = res.get("name") or ""
        assert "λ" not in name and "stannamethan" not in name and "stannapropan" not in name, res


# ---------------------------------------------------------------------------
# What the route leaves alone
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "CCCCCCCC[Sn]CCCCCCCC.CC(CC(=O)O)CC(C)(C)C.CC(CC(=O)O)CC(C)(C)C",   # a adduct
    "CCC[Sn]CCC.[Br-].[Br-]",                                           # a salt
    "C[Sn]C.CC(=O)O",
])
def test_a_tin_piece_of_a_mixture_keeps_its_component_name(smiles):
    """A mixture is named by its components, the Blue Book) and a radical word
    cannot stand in a multi-component name: OPSIN reads none of
    '3,5,5-trimethylhexanoic acid—dioctylstannylidene (2/1)'. These names exist at the breadth
    tier before the change and must keep existing (best-effort never goes down)."""
    res = name_breadth(smiles)
    name = res.get("name")
    assert name and res.get("tier") != "abstain", res
    assert "stannylidene" not in name, name
    assert name_is_rt_exact(name, smiles), name


@pytest.mark.opsin_gate
def test_the_tin_atom_of_a_phosphonate_is_not_a_hydride_radical():
    """The tin atom is a substituent of a bisphosphonate: its neutral parent is the
    phosphonate, not 'stannane', so the '-ylidene' replacement of does not apply."""
    smiles = "CC(C)OP(=O)(C(C[Sn]Cl)P(=O)(OC(C)C)OC(C)C)OC(C)C"
    res = name_breadth(smiles)
    name = res.get("name")
    assert name and "stannylidene" not in name, res
    assert name_is_rt_exact(name, smiles), name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["CC[Sn+](CC)CC", "[Sn]", "[Pb]"])
def test_ionic_and_bare_atoms_stay_out_of_scope(smiles):
    res = name_default(smiles)
    assert res.get("tier") == "abstain", res


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("C[Sn](C)(C)C", "tetramethylstannane"),
    ("C[SnH2]C", "dimethylstannane"),
])
def test_a_closed_shell_organotin_is_no_radical(smiles, name):
    assert_pin_at_both_tiers(smiles, name)


# ---------------------------------------------------------------------------
# The gate of the route (a unit test of its decision)
# ---------------------------------------------------------------------------

def _element(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return cr._group14_metal_radical_element(mol)


def test_the_gate_lets_through_only_a_neutral_single_tin_or_lead_radical_centre():
    assert _element("CC[Sn]CC") == "Sn"
    assert _element("C[Pb]C") == "Pb"
    assert _element("C[SnH]C") == "Sn"
    assert _element("[SnH2]") == "Sn"
    # closed-shell tin: no radical electron
    assert _element("C[Sn](C)(C)C") is None
    assert _element("C[SnH2]C") is None
    # silicon and germanium are in the allowlist and never needed the exception
    assert _element("C[Ge]C") is None
    assert _element("C[Si]C") is None
    # ions, mixtures, a second metal, a second radical site
    assert _element("CC[Sn+](CC)CC") is None
    assert _element("CC[Sn]CC.[Br-].[Br-]") is None
    assert _element("C[Sn]C.CC(=O)O") is None
    assert _element("C[Sn][Sn]C") is None
    assert _element("C[Sn]C[CH2]") is None
    # a bare atom is no organic parent
    assert _element("[Sn]") == "Sn"      # the gate passes it; the route declines it (below)
    # a metal other than tin and lead stays a metal
    assert _element("C[Zn]C") is None


def test_the_gate_applies_only_to_the_callers_whole_input(monkeypatch):
    """A tin piece of a mixture is a component (a nested ``name``, the salt writer): the
    gate declines, as the radical name cannot join a multi-component name. The caller's
    string is ``namer._caller_input`` while the outermost ``name`` runs; the depth of the
    ``name`` stack does not tell (the default tier names the caller's molecule a second
    time one level down)."""
    import orthonym.namer as namer
    assert _element("CC[Sn]CC") == "Sn"                    # no name running: the molecule given
    monkeypatch.setattr(namer._CALLER_INPUT, "smiles", "CC[Sn]CC", raising=False)
    assert _element("CC[Sn]CC") == "Sn"                    # the caller's own molecule
    fn.enter_name_scope()
    fn.enter_name_scope()                                  # a second level, as the promotion re-run
    try:
        assert _element("CC[Sn]CC") == "Sn"
    finally:
        fn.exit_name_scope()
        fn.exit_name_scope()
    monkeypatch.setattr(namer._CALLER_INPUT, "smiles", "CC[Sn]CC.[Br-].[Br-]", raising=False)
    assert _element("CC[Sn]CC") is None                    # a piece of the caller's mixture
    monkeypatch.setattr(namer._CALLER_INPUT, "smiles", "not a smiles", raising=False)
    assert _element("CC[Sn]CC") is None                    # unreadable: the piece keeps its naming


def test_the_route_declines_a_tin_atom_that_is_not_its_own_parent_hydride():
    """The neutral name of '[Sn]' is not a stannane that a suffix can replace: the route
    returns '' (the caller's other producers decide), it never emits a radical name for it."""
    assert cr.route_charged(Chem.MolFromSmiles("[Sn]"), "pin") == ""
