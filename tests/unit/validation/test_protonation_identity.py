"""Protonation-site identity (fix a performance pass, wp7; whole-branch verification panel RISK 1).

The standard InChI moves an onium atom's hydrons into one mobile /p layer, so the full
InChIKey cannot tell protonation isomers apart. Five names shipped at pin_verified with
the '-ium' on the WRONG nitrogen, and every gate ('s full-key shortcut, the
charged-router RT helpers, tests/support/rt_assert) accepted them:

    C[NH2+]CCC(=O)NC 'N-methyl-3-(methylamino)propanamidium'
    C[NH2+]CCC(=O)NCC 'N-ethyl-3-(methylamino)propanamidium'
    C[NH2+]CCC#N '3-(methylamino)propanenitrilium'
    C[NH+](C)CCN '2-(dimethylamino)ethan-1-aminium'
    [NH3+]CC(=O)NCC(=O)O 'glycylglycinium'

 (the Blue Book) / Table 7.4 (:41417): 'amidium', 'nitrilium', 'aminium'
are the cationic forms OF those suffix groups, so each name above denotes a protonation
isomer of the input (independent OPSIN 2.9.0 parse, e.g. 'N-methyl-3-(methylamino)-
propanamidium' -> CNCCC(=O)[NH2+]C). With the charge elsewhere the cation is the
principal group, '6 Cations' before the neutral classes,:18169).
"""
import pytest

from orthonym.validation.protonation_identity import protonation_site_verdict

pytestmark = pytest.mark.unit

# (input, OPSIN 2.9.0 parse of the old wrong name)
_WRONG_SITE = [
    ("C[NH2+]CCC(=O)NC", "C[NH2+]C(CCNC)=O"),
    ("C[NH2+]CCC(=O)NCC", "C(C)[NH2+]C(CCNC)=O"),
    ("C[NH2+]CCC#N", "CNCCC#[NH+]"),
    ("C[NH+](C)CCN", "CN(CC[NH3+])C"),
    ("[NH3+]CC(=O)NCC(=O)O", "NCC(=O)[NH2+]CC(=O)O"),
    ("Nc1cc[nH+]cc1", "[NH3+]c1ccncc1"),   # 'pyridin-4-aminium' for the ring-N cation
]

_OLD_WRONG_NAMES = {
    "C[NH2+]CCC(=O)NC": "N-methyl-3-(methylamino)propanamidium",
    "C[NH2+]CCC(=O)NCC": "N-ethyl-3-(methylamino)propanamidium",
    "C[NH2+]CCC#N": "3-(methylamino)propanenitrilium",
    "C[NH+](C)CCN": "2-(dimethylamino)ethan-1-aminium",
    "[NH3+]CC(=O)NCC(=O)O": "glycylglycinium",
    "Nc1cc[nH+]cc1": "pyridin-4-aminium",
}


@pytest.mark.parametrize("smiles,parsed", _WRONG_SITE)
def test_wrong_protonation_site_is_a_mismatch(smiles, parsed):
    from rdkit import Chem
    # the trap: the full standard InChIKey is equal
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        Chem.MolToInchiKey(Chem.MolFromSmiles(parsed))
    assert protonation_site_verdict(smiles, parsed) == "mismatch"


@pytest.mark.parametrize("smiles,parsed", [
    # resonance drawings of one charge-delocalised cation are the same species
    ("CC(=[NH2+])NC", "CC(N)=[NH+]C"),
    ("Nc1cc[nH+]cc1", "[NH2+]=C1C=CNC=C1"),
    ("NC(=[NH2+])NC", "NC(N)=[NH+]C"),
    # the same drawing
    ("C[NH3+]", "C[NH3+]"),
    ("NCC[NH3+].[Cl-]", "[Cl-].[NH3+]CCN"),
])
def test_same_species_is_ok(smiles, parsed):
    assert protonation_site_verdict(smiles, parsed) == "ok"


@pytest.mark.parametrize("smiles,parsed", [
    # a salt named as its neutral acid-base form, and a zwitterion named as the
    # neutral amino acid:54518): not a protonation-SITE question
    ("C[NH3+].[Cl-]", "CN.Cl"),
    ("[NH3+]CC(=O)[O-]", "NCC(=O)O"),
    # no protonated heavy atom at all
    ("CCO", "CCO"),
    ("C[N+](C)(C)C", "C[N+](C)(C)C"),
])
def test_out_of_scope_is_na(smiles, parsed):
    assert protonation_site_verdict(smiles, parsed) == "n/a"


def test_self_consistency_verdict_no_longer_short_circuits():
    """'s full-InChIKey shortcut accepted every protonation isomer."""
    from orthonym.namer import _self_consistency_verdict
    for smiles, parsed in _WRONG_SITE:
        assert _self_consistency_verdict(smiles, parsed) == "mismatch", smiles
    assert _self_consistency_verdict("CC(=[NH2+])NC", "CC(N)=[NH+]C") == "ok"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", list(_OLD_WRONG_NAMES))
def test_wrong_site_name_never_ships(smiles):
    """Gate on: the protonation-isomer name is never shipped; the PIN tier ships an
    exact name (at most a demoted one) or fails closed, and best-effort names the
    molecule exactly (breadth)."""
    from orthonym import Orthonym
    from tests.support.rt_assert import assert_tier_contract, name_is_rt_exact
    assert not name_is_rt_exact(_OLD_WRONG_NAMES[smiles], smiles)
    assert_tier_contract(smiles)
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] != _OLD_WRONG_NAMES[smiles], r
    if r["tier"] == "pin_verified":
        assert name_is_rt_exact(r["name"], smiles), r


@pytest.mark.parametrize("smiles", [
    "C[NH2+]CCC(=O)NC", "C[NH2+]CCC#N", "C=CC(=O)NCCC[N+](C)(C)C"])
def test_suffix_swap_declines_off_the_suffix_nitrogen(smiles):
    """Producer: the amidium/nitrilium/aminium suffix swap (composer
    _try_ion_aspect_composition) runs only when every cationic atom is a suffix
    nitrogen; it used to append 'ium' by suffix text alone."""
    from orthonym import Orthonym
    from orthonym.assembly.composer import _try_ion_aspect_composition
    from rdkit import Chem
    namer = Orthonym(style="pin", _disable_opsin_validity_gate=True)
    mol = Chem.MolFromSmiles(smiles)
    can = Chem.MolToSmiles(mol)
    features = namer._perceive(mol, can, can)
    namer._classify(features)
    assert _try_ion_aspect_composition(features, "pin") is None


@pytest.mark.opsin_gate
def test_diamine_monocation_takes_the_aminium_pin():
    """The suffix swap used to turn NCC[NH3+] into 'ethan-1-aminium' (the neutral
    NH2 was the suffix; the other nitrogen was dropped), which the gate voided, so
    the PIN tier abstained. Declining the swap lets the cation path name it:
    '2-aminoethan-1-aminium chloride (PIN)' (the Blue Book) is this cation's
    salt."""
    from orthonym import Orthonym
    from tests.support.rt_assert import name_is_rt_exact
    r = Orthonym(style="pin").name_tiered("NCC[NH3+]")
    assert r["name"] == "2-aminoethan-1-aminium" and r["tier"] == "pin_verified", r
    assert name_is_rt_exact(r["name"], "NCC[NH3+]")
    assert Orthonym(style="pin").name_tiered("CC[NH3+]")["name"] == "ethanaminium"
