"""A drawing whose metal-carbon bonds are broken into carbon radicals beside the one bond-free metal atom
(``[Li].CCC[CH2]``, ``[CH]1CC1.[Br-].[Mg+]``) is named through its bonded drawing (``RECONNECT_RADICAL``,
``assembly.metal_offer.metal_radical_joined_spelling_and_order``).

Standard InChI ignores metal-ligand bonds and radicals, so both drawings have one standard InChIKey; the bonded
drawing is offered only when an independent count of atoms, charges and radical electrons agrees
(``perception.metals.atom_and_charge_ledger_ok`` with ``moved_radical``). "Structures for Lewis
'adducts'" (the Blue Book): "One of the fundamental principles of organic chemical nomenclature is to name
the structure that is drawn", so the name of the bonded drawing is labelled below the PIN.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.metal_offer import metal_radical_joined_spelling_and_order
from orthonym.perception.metals import atom_and_charge_ledger_ok, radical_join_sites
from tests.support.rt_assert import name_best_effort, name_is_rt_exact


@pytest.mark.parametrize("smiles,joined", [
    ("[Li].CCC[CH2]", "CCCC[Li]"),
    ("[CH]1CC1.[Br-].[Mg+]", "[Mg+]C1CC1.[Br-]"),
    ("CCC[CH2].CCC[CH2].CCC[CH2].C=[CH].[Sn]", "C=C[Sn](CCCC)(CCCC)CCCC"),
])
def test_bonded_drawing(smiles, joined):
    written, _order = metal_radical_joined_spelling_and_order(smiles)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(written)) == Chem.MolToSmiles(Chem.MolFromSmiles(joined))


@pytest.mark.parametrize("smiles", [
    "[Li].C[CH2].C[CH2]",          # lithium(2+) would be the cation: no valence of the table (the key is the same)
    "[Sn].[CH2]",                  # two radical electrons: a single or a double bond, one key
    "[N].[Y]",                     # a radical that is no carbon
    "[Li].[Na].CCC[CH2]",          # two metals: the pairing is a choice
    "[ZnH].[CH3].[CH3]",           # the hydrogen of the metal counts: zinc(3+) is no valence of the table
    "CCCC[Li]",                    # already bonded
    "CCC[CH2]",                    # no metal
])
def test_no_bonded_drawing(smiles):
    assert metal_radical_joined_spelling_and_order(smiles) is None


def test_ledger_takes_one_radical_electron_per_bond():
    drawn = Chem.MolFromSmiles("[Li].CCC[CH2]")
    sites = radical_join_sites(drawn)
    bonded = Chem.RWMol(drawn)
    metal, carbon = sites[0]
    bonded.AddBond(metal, carbon, Chem.BondType.SINGLE)
    bonded.GetAtomWithIdx(carbon).SetNumRadicalElectrons(0)
    bonded.GetAtomWithIdx(carbon).SetNoImplicit(True)
    bonded.GetAtomWithIdx(metal).SetNumRadicalElectrons(0)
    bonded = bonded.GetMol()
    Chem.SanitizeMol(bonded)
    assert atom_and_charge_ledger_ok(bonded, drawn, sites, 0, moved_radical=True)
    # read as a heterolytic join the count fails: the radical electron is not accounted for
    assert not atom_and_charge_ledger_ok(bonded, drawn, sites, 0)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("[Li].CCC[CH2]", "butyllithium"),
    ("[CH]1CC1.[Br-].[Mg+]", "cyclopropylmagnesium bromide"),
    ("CCC[CH2].CCC[CH2].CCC[CH2].C=[CH].[Sn]", "tributyl(ethenyl)stannane"),
    ("C=C1C=CC=C[CH]1.[Cl-].[Hg+]", "(6-methylidenecyclohexa-2,4-dien-1-yl)mercury chloride"),
])
def test_named_through_the_bonded_drawing(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected and res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    # an organyl ligand of a Group 1 or 2 metal is named by the substituent composer, as for the Group 12 metals
    ("Br[Mg]C1CCCCC1", "cyclohexylmagnesium bromide"),
    ("[Li]C1CCCCC1", "cyclohexyllithium"),
])
def test_group_1_2_organyl_ligand(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res


@pytest.mark.parametrize("smiles,parsed,verdict", [
    ("[Li].CCC[CH2]", "CCCC[Li]", "ok"),             # judged against the bonded drawing
    ("[Li].CCC[CH2]", "[Li].CCC[CH2]", "mismatch"),  # a radical parse is not the bonded drawing
    ("CCC[CH2]", "CCCC", "mismatch"),                # no metal: the strict rule
    ("[Li].C[CH2].C[CH2]", "CC[Li]CC", "mismatch"),  # no join (no valence of the table): the strict rule
    # the key does not see on which carbon of an allyl group the metal sits: the organic parts must be one graph
    ("[Li].C=C[CH]C", "[Li]C(C)C=C", "ok"),
    ("[Li].C=C[CH]C", "[Li]CC=CC", "mismatch"),
])
def test_radical_rule_exemption_is_the_joined_sites_only(smiles, parsed, verdict):
    from orthonym.validation.radical_identity import _radical_identity_verdict_impl
    assert _radical_identity_verdict_impl(smiles, parsed) == verdict
