"""A carbon-substituted N+ cited as '...azaniumyl' / '...azanium' is never labelled a PIN.

Rule. A carbon group on N+ makes the cation an AMINE cation, named by the cationic
suffix 'aminium', the Blue Book; '*N*,*N*,*N*-trimethylmethanaminium
(PIN)':41438). Its PIN prefix is built on that cation name: method (1) "all
prefix names are formed by adding the suffixes 'yl', 'ylidene', etc. to the cation name"
(:42296), "Method (1) leads to preferred IUPAC names" (:42299) -- hence
'(*N*,*N*-dimethylmethanaminiumyl)acetate (PIN)':42473) and
'2-(*N*,*N*-dimethylmethanaminiumyl)propan-2-ide (PIN)':42517). The
round-trip parser reads none of the '-aminiumyl' forms, so those PINs cannot be
verified: the engine keeps the verified azanium name and labels it general-tier.

A cation also outranks every neutral class: '6 Cations' before
'7 Acids',:18169-18170), so '3-(methylazaniumyl)propanoic acid' (an azaniumyl prefix on
an acid parent) is not a PIN either.

Two quaternary N+ on the ends of one chain: the multiplicative 'hexane-1,6-diylbis(
trimethylazanium)' is the same kind of name (see test_bis_quaternary_aminium_pin.py for
the substitutive PIN).

The label is name-scoped (``record_non_pin_fragment``): a prefix built by a speculative
sort key or a discarded candidate cannot demote a name that does not contain it.
"""
import pytest

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

_PLASMALOGEN = "CCCCCCCCCCCCCC/C=C\\OC[C@H](COP(=O)(O)OCC[N+](C)(C)C)O"
_PC = "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP([O-])(=O)OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC"


def _row(smiles, namer=None):
    with jvm_slots(1, purpose="test-n-cation-labels"):
        return (namer or Orthonym(style="pin")).name_tiered(smiles)


# The verified azanium name ships unchanged, labelled general-tier (never pin_verified).
@pytest.mark.parametrize("smiles,expected", [
    ("C[N+](C)(C)CC(=O)[O-]", "(trimethylazaniumyl)acetate"),                 #:42473
    ("CC[N+](CC)(CC)CC(=O)[O-]", "(triethylazaniumyl)acetate"),
    ("C[N+](C)(C)[C-](C)C", "2-(trimethylazaniumyl)propan-2-ide"),            #:42517
    ("C[N+](C)(C)CCCC(=O)[O-]", "4-(trimethylazaniumyl)butanoate"),
    ("C[N+](C)(C)CCOS(=O)(=O)[O-]", "2-(trimethylazaniumyl)ethyl sulfate"),
    (_PC, "[(2R)-2,3-bis(hexadecanoyloxy)propyl] 2-(trimethylazaniumyl)ethyl phosphate"),
    (_PLASMALOGEN, "(2R)-1-{[(1Z)-hexadec-1-en-1-yl]oxy}-3-({[2-(trimethylazaniumyl)"
                   "ethoxy]hydroxyphosphoryl}oxy)propan-2-ol"),
    #: the cation is principal; an azaniumyl prefix on a neutral parent is not a PIN.
    # With an oxoacid on the cation the aminium-principal PIN is not built (the charged
    # router's zwitterion step ionizes that acid suffix), so the demoted swap ships.
    ("C[NH2+]CCC(=O)O", "3-(methylazaniumyl)propanoic acid"),
    ("C[NH+](C)CCC(=O)O", "3-(dimethylazaniumyl)propanoic acid"),
    ("[NH3+]CCC(=O)O", "3-azaniumylpropanoic acid"),
])
def test_unverifiable_pin_form_ships_demoted(smiles, expected):
    row = _row(smiles)
    assert row["name"] == expected
    assert row["tier"] not in ("pin_verified", "abstain") and not row["is_pin"]
    assert name_is_rt_exact(expected, smiles)


# Controls: nothing here cites a carbon-substituted N+ as azanium(yl).
@pytest.mark.parametrize("smiles,expected", [
    ("[NH3+]CC(=O)[O-]", "glycine"),
    # wp7 change-asserted-value: (the Blue Book) '2-chloroethan-1-ol is the
    # PIN'; '2-aminoethan-1-aminium chloride (PIN)' (:43572). Was '...trimethylethanaminium'
    # (the elided locant). OPSIN 2.9.0 full-InChIKey exact.
    ("C[N+](C)(C)CCO", "2-hydroxy-N,N,N-trimethylethan-1-aminium"),
    ("C[N+](C)(C)C", "N,N,N-trimethylmethanaminium"),
    ("C[NH3+]", "methanaminium"),
    ("OCC(CO)(CO)[NH3+]", "1,3-dihydroxy-2-(hydroxymethyl)propan-2-aminium"),
    # an aminium-principal name whose molecule ALSO reaches the choline lipid
    # assembler (whose candidate the gate rejects): not demoted by that attempt
    ("CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC[N+](C)(C)C)OC(=O)C",
     "2-({[2-(acetyloxy)-3-(octadecanoyloxy)propoxy]hydroxyphosphoryl}oxy)-"
     "N,N,N-trimethylethan-1-aminium"),  # wp7::2869; was '...ethanaminium'
])
def test_controls_stay_pin_verified(smiles, expected):
    row = _row(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified" and row["is_pin"]


# wp7: without an oxoacid on the cation the swap is replaced by the aminium-principal
# PIN, '6 Cations' before the neutral classes, the Blue Book; the
# charged-router re-derivation through the systematic amine parent, RT-gated). Was the
# demoted swap ('2-(diethylazaniumyl)ethan-1-ol'). Each name OPSIN 2.9.0 full-InChIKey
# exact (fresh java run outside the engine).
@pytest.mark.parametrize("smiles,expected", [
    ("CC[NH+](CC)CCO", "N,N-diethyl-2-hydroxyethan-1-aminium"),
    ("[NH3+]CCCCC=O", "5-oxopentan-1-aminium"),
    ("C[NH2+]CCC#N", "2-cyano-N-methylethan-1-aminium"),
    ("C[NH2+]CCC(=O)NC", "N-methyl-3-(methylamino)-3-oxopropan-1-aminium"),
    ("C[NH2+]CCC(=O)NCC", "3-(ethylamino)-N-methyl-3-oxopropan-1-aminium"),
    ("CCCCCCCCC(C)(C)NC(=O)C[NH+](CC)CC",
     "N,N-diethyl-2-[(2-methyldecan-2-yl)amino]-2-oxoethan-1-aminium"),
])
def test_cation_principal_pin_replaces_the_swap(smiles, expected):
    row = _row(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified" and row["is_pin"], row
    assert name_is_rt_exact(expected, smiles)


def test_multiplicative_fallback_unit_records_its_fragment():
    # the multiplicative builder stays as the general-tier fallback; its onium
    # unit is recorded as a non-PIN fragment, so a name carrying it is demoted
    from rdkit import Chem

    from orthonym.metrics.provenance import (
        clear_provenance,
        get_provenance,
        name_carries_non_pin_part,
    )
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.ions import emit_bis_quaternary_ammonium
    mol = Chem.MolFromSmiles("C[N+](C)(C)CCCCCC[N+](C)(C)C")
    clear_provenance()
    name = emit_bis_quaternary_ammonium(mol, get_ion_sites(mol)["cations"])
    assert name == "hexane-1,6-diylbis(trimethylazanium)"
    prov = get_provenance()
    assert "trimethylazanium" in prov["non_pin_fragments"]
    assert name_carries_non_pin_part(prov, name)
    assert not name_carries_non_pin_part(
        prov, "N1,N1,N1,N6,N6,N6-hexamethylhexane-1,6-bis(aminium)")
    clear_provenance()
    assert get_provenance()["non_pin_fragments"] == ()


def test_label_is_stable_in_one_warm_instance():
    # memo / cache hits must not lose (or leak) the label across calls
    namer = Orthonym(style="pin")
    seq = [(_PC, False), (_PLASMALOGEN, False), (_PLASMALOGEN, False),
           ("C[N+](C)(C)CC(=O)[O-]", False), ("C[N+](C)(C)CC(=O)[O-]", False),
           ("[NH3+]CC(=O)[O-]", True), ("C[N+](C)(C)CCO", True), (_PC, False),
           ("CCO", True)]
    for smiles, is_pin in seq:
        row = _row(smiles, namer)
        assert row["is_pin"] is is_pin, (smiles, row["name"], row["tier"])
        assert (row["tier"] == "pin_verified") is is_pin, (smiles, row["tier"])


def test_replay_merges_recorded_fragments():
    # a replay-memo hit must ADD the fresh call's fragments, never overwrite the
    # ones recorded since (an overwrite would drop a later fragment)
    from orthonym.assembly import nested_memo
    from orthonym.metrics import provenance as pv
    pv.clear_provenance()
    pv.record_non_pin_fragment("a")
    pv.record_non_pin_fragment("b")
    nested_memo._apply({"non_pin_fragments": ("a", "c")}, (0, 0, 0))
    assert pv.get_provenance()["non_pin_fragments"] == ("a", "b", "c")
    pv.clear_provenance()


# (the Blue Book) on the mononuclear onium centre: first cited bare,
# second and further enclosed, multiplier outside; order ignores 'tert-';
# the whole prefix then takes the next enclosing mark,:7446).
@pytest.mark.parametrize("smiles,expected,is_pin", [
    ("C[P+](c1ccccc1)(c1ccccc1)CC(=O)[O-]",
     "[methyldi(phenyl)phosphaniumyl]acetate", True),              #:42468 form
    ("CC[S+](C)CC(=O)[O-]", "[ethyl(methyl)sulfaniumyl]acetate", True),  #:41777 form
    ("CC[P+](C)(C)[C-](C)C", "2-[ethyldi(methyl)phosphaniumyl]propan-2-ide", True),
    ("CC[N+](C)(C)CC(=O)[O-]", "[ethyldi(methyl)azaniumyl]acetate", False),
    ("CC(C)(C)[N+](C)(C)CC(=O)[O-]", "[tert-butyldi(methyl)azaniumyl]acetate", False),
    ("CCC[N+](C)(C)CC(=O)[O-]", "[dimethyl(propyl)azaniumyl]acetate", False),
])
def test_onium_prefix_enclosing_marks(smiles, expected, is_pin):
    row = _row(smiles)
    assert row["name"] == expected and row["is_pin"] is is_pin
    assert (row["tier"] == "pin_verified") is is_pin
    assert name_is_rt_exact(expected, smiles)
