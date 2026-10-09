"""Item 12a (task 1): a ring carbonyl of a mancude ring is an 'oxo' prefix on a hydro parent.

* (the Blue Book, "Heteromonocyclic hydrides named by skeletal replacement ('a')
  nomenclature"): "Mancude and saturated heteromonocyclic compounds with up to and including ten
  ring members are named by the extended Hantzsch-Widman system"; (:23682) makes
  those names the preferred ones. So never '1-oxacyclohexa-2,4-dien-2-yl'.
* (:24868, "Prefix nomenclature"; Example 5,:25477): '1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl
  (preferred prefix)'; (:43483, "'Added indicated hydrogen'"):
  '1-ethyl-2-oxo-1,2-dihydropyridin-1-ium-1-yl'. (b) (:3246) puts indicated hydrogen
  before the free valence (c,:3256).
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.book_prefixes import mechanical_forms
from orthonym.assembly.universal_substituent import name_universal_substitutive
from orthonym.cli import _emit_tier_flags
from orthonym.rules.monocycle_forms import monocycle_form
from orthonym.rules.terminal_ring import terminal_ring_name

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from name_quality import forms as F  # noqa: E402
from tests.support.rt_assert import name_is_rt_exact  # noqa: E402


def _ring_and_fv(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ring = next(r for r in mol.GetRingInfo().AtomRings()
                if any(mol.GetBondBetweenAtoms(1, a) for a in r))
    fv = next(a for a in ring if mol.GetBondBetweenAtoms(1, a))
    return mol, list(ring), fv


def _form(smiles, **kw):
    mol, ring, fv = _ring_and_fv(smiles)
    branches = [a for a in ring for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                if nb.GetIdx() not in ring and nb.GetIdx() != 1]
    return fv, monocycle_form(mol, ring, fv, branches, **kw)


@pytest.mark.parametrize("smiles,parent,prefix", [
    ("OCc1ccc(=O)[nH]c1", "1,6-dihydropyridine", "1,6-dihydropyridin-3-yl"),
    ("OCc1ccc(=O)oc1", "2H-pyran", "2H-pyran-5-yl"),
    ("OCC1=CC(=O)OC=C1", "2H-pyran", "2H-pyran-4-yl"),
    ("OCn1ccc(=O)[nH]c1=O", "1,2,3,4-tetrahydropyrimidine", "1,2,3,4-tetrahydropyrimidin-1-yl"),
    ("OCc1c[nH]c(=O)[nH]c1=O", "1,2,3,4-tetrahydropyrimidine", "1,2,3,4-tetrahydropyrimidin-5-yl"),
    ("OCc1nc(=O)[nH]cc1", "1,2-dihydropyrimidine", "1,2-dihydropyrimidin-4-yl"),
])
def test_a_ring_carbonyl_of_a_mancude_ring_is_an_oxo_prefix_on_a_hydro_parent(smiles, parent, prefix):
    fv, form = _form(smiles)
    assert form is not None
    assert (form.parent, form.prefix(fv)) == (parent, prefix)


@pytest.mark.parametrize("smiles", [
    "OC=c1ccc(=C)cc1",            # RDKit calls this ring aromatic: it is not benzene (a quinodimethane)
    "OC=C1C=CC=CC=C1",            # heptafulvene: a carbocycle with a ring double bond outside it
])
def test_a_carbocycle_with_a_ring_double_bond_outside_it_is_not_benzene(smiles):
    assert _form(smiles)[1] is None


@pytest.mark.parametrize("smiles,expected", [
    ("COC(=O)CCc1ccc(=O)[nH]c1C(C)(C)C", "6-tert-butyl-5-(3-methoxy-3-oxopropyl)-2-oxo-1,2-dihydropyridine"),
    ("CC(C)n1ccc(=O)[nH]c1=O", "1-(1-methylethyl)-2,4-dioxo-1,2,3,4-tetrahydropyrimidine"),
    ("COc1ccc(=S)[nH]c1CC=C", "5-methoxy-6-(prop-2-en-1-yl)-2-sulfanylidene-1,2-dihydropyridine"),
    ("CC1CCC(=O)N1c1ccc(=O)oc1", "5-(5-methyl-2-oxopyrrolidin-1-yl)-2-oxo-2H-pyran"),
])
def test_floor_names_an_oxo_ring_by_its_hydro_parent(smiles, expected):
    name = name_universal_substitutive(Chem.MolFromSmiles(smiles)).name
    assert name == expected
    assert name_is_rt_exact(name, smiles)


def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


#: molecules whose best-effort name was written by 'a' replacement at the base
E2E = [
    ("CC(=O)N1CCN(c2ccc(=O)oc2)CC1", "1-acetyl-4-(2-oxo-2H-pyran-5-yl)piperazine"),
    ("COc1cc([C@H](Cc2ccccc2)NC(C)=O)oc(=O)c1",
     "N-[(1S)-1-(4-methoxy-2-oxo-2H-pyran-6-yl)-2-phenylethyl]acetamide"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", E2E)
def test_wider_tier_rings_end_to_end(smiles, expected):
    row = _row(smiles, "best-effort")
    assert row["name"] == expected, row
    assert name_is_rt_exact(expected, smiles)
    assert not {"MONO", "BENZ"} & set(F.detect(expected))


def test_terminal_ring_name_gives_the_oxo_ring_its_book_name():
    mol = Chem.MolFromSmiles("CC(=O)c1ccc(=O)oc1")
    ring = [a for a in mol.GetRingInfo().AtomRings()[0]]
    fv = next(a for a in ring if mol.GetBondBetweenAtoms(1, a))
    tr = terminal_ring_name(mol, ring, fv)
    assert tr is not None and tr.name == "2H-pyran-5-yl"
    with mechanical_forms():
        assert "oxacyclohexa" in terminal_ring_name(mol, ring, fv).name


# (the Blue Book, "All mancude rings and ring systems are named by method (2)"
# text): "When no hydrogen atoms are present or when an 'ylidene' type substituent group is
# needed, it is necessary to use 'added [indicated] hydrogen'"; (:24703) 'pyridin-1(2H)-yl
# (preferred prefix)'. The floor's hydro-prefix spelling of such a prefix is valid, not the PIN.
@pytest.mark.parametrize("smiles,locant,ring_atom_is_c", [
    ("OCn1ccc(=O)[nH]c1=O", 1, False),      # uracil N1: no hydrogen to substitute
    ("OCn1ccc(=O)cc1", 1, False),           # 4-pyridone N1
])
def test_a_prefix_on_a_ring_nitrogen_of_a_hydro_ring_is_recorded_as_not_the_pin(
        smiles, locant, ring_atom_is_c):
    from orthonym.metrics import provenance as pv
    fv, form = _form(smiles)
    assert form.fv_hydro_locant == locant and form.fv_has_hydrogen is False
    token = pv._NON_PIN_LABELS.set(())
    try:
        prefix = form.prefix(fv)
        assert prefix in pv._NON_PIN_LABELS.get()
    finally:
        pv._NON_PIN_LABELS.reset(token)


def test_a_prefix_on_the_indicated_hydrogen_atom_or_a_ring_carbon_is_not_recorded():
    from orthonym.metrics import provenance as pv
    for smiles in ("OCC1=CC(=O)OC=C1", "OCc1ccc(=O)[nH]c1", "OCn1cc[nH]c1=O"):
        fv, form = _form(smiles)
        token = pv._NON_PIN_LABELS.set(())
        try:
            form.prefix(fv)
            assert pv._NON_PIN_LABELS.get() == (), smiles
        finally:
            pv._NON_PIN_LABELS.reset(token)


@pytest.mark.opsin_gate
def test_a_nucleoside_row_with_a_hydro_uracil_prefix_is_not_labelled_a_pin():
    smiles = "COC1C(O)C(n2ccc(=O)[nH]c2=O)OC1C(OC1OC(C(=O)N)=CC(O)C1O)C(N)=O"
    row = _row(smiles, "best-effort")
    assert "tetrahydropyrimidin-1-yl" in row["name"], row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
