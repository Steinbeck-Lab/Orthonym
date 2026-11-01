"""The natural-product route also names substituted free-base porphyrins (rules/porphyrin.py).

LABEL. NPL 2026-11-01 (user decision 2026-10-31): INTRODUCTION (the Blue Book) says "Preferred IUPAC
names (PINs) are not identified for the compounds in this Chapter", and (``porphyrin``, Table 10.1 (d))
is in that chapter. So a porphyrin name is no PIN: the default tier declines it (NO_VERIFIED_PIN) and the
best-effort tier gives the same name labelled ``systematic_verified`` (``is_pin`` False), read back by OPSIN.

The molecules are constructed from names (tests/fixtures/porphyrin/*.tsv); u1 and u2 are user molecules
(tests/fixtures/resonance/user_molecules.tsv).
"""
import csv
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from tests.support.default_tier import assert_default_tier_declines
from tests.support.pin_tiers import name_breadth, name_default
from tests.support.resonance_rows import rows as resonance_rows
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "porphyrin"


def _table(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


SUBSTITUTED = {r["id"]: r for r in _table("substituted_porphyrins.tsv")}
DECLINED_TODAY = _table("declined_today.tsv")
BARE = "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3"
U1 = next(r for r in resonance_rows("user_molecules.tsv") if r["id"] == "u1")["smiles"]
U2 = next(r for r in resonance_rows("user_molecules.tsv") if r["id"] == "u2")["smiles"]
U2_FREE_BASE = U2.split(".")[0]


@pytest.mark.parametrize("row_id", ["5,10,15,20-tetraphenylporphyrin", "2,3,7,8,12,13,17,18-octaethylporphyrin",
                                    "2-methyl-5-phenylporphyrin"])
def test_the_name_and_label_at_both_tiers(row_id):
    """The default tier declines it; best-effort gives the name, ``systematic_verified``, read back by OPSIN."""
    row = SUBSTITUTED[row_id]
    assert_default_tier_declines(row["smiles"])
    result = name_breadth(row["smiles"])
    assert result.get("name") == row["expected_name"], result
    assert result.get("tier") == "systematic_verified", result      # NPL 2026-11-01:, no PIN for the chapter
    assert name_is_rt_exact(result["name"], row["smiles"]), result


def test_the_bare_parent_is_unchanged():
    """The exact lookup (data/natural_products.py) runs first and keeps its own label."""
    # NPL 2026-11-01: INTRODUCTION (the Blue Book): no PIN for the parents; the default tier declines.
    assert_default_tier_declines(BARE)
    result = name_breadth(BARE)
    assert (result.get("name"), result.get("tier")) == ("porphyrin", "systematic_verified"), result
    assert result.get("is_pin") is False


def test_the_hook_asks_the_catalogue_before_the_porphyrin_producer():
    from orthonym.routing.dispatch_table import _natural_product_name
    assert _natural_product_name(Chem.MolFromSmiles(BARE)) == "porphyrin"
    tpp = SUBSTITUTED["5,10,15,20-tetraphenylporphyrin"]
    assert _natural_product_name(Chem.MolFromSmiles(tpp["smiles"])) == tpp["expected_name"]
    assert _natural_product_name(Chem.MolFromSmiles("CCO")) is None


# Two of the declined molecules do change, each by a mechanism other than the producer, each exact:
# * the 13C-labelled methyl row: the producer declines it (``isotope``), but the isotope layer of the
# namer names the unlabelled molecule and re-anchors the label, so the name reaches the route through
# its unlabelled graph: both tiers now name it (before: ``UNNAMEABLE`` and a von Baeyer name);
# * the neutral porphyrin with a free [Zn+2]: the whole molecule is declined (``fragments``, default tier
# unchanged), but the best-effort composite names its ligand component on its own, and that component
# is a free-base porphyrin (risk R4 of the plan): the ligand part of ``<ligand>--zinc(2+) (1/1)`` is the
# porphyrin name now, and the composite reads back to the full InChIKey of the input.
CHANGES = {
    "isotope_13C_methyl": {"default": "5-(13C)methyl-10,15,20-triphenylporphyrin",
                           "best_effort": "5-(13C)methyl-10,15,20-triphenylporphyrin"},
    "TPP_plus_free_Zn2+": {"best_effort": "5,10,15,20-tetraphenylporphyrin\u2014zinc(2+) (1/1)"},
}


# The palladium rows: palladium is admitted to every mechanism of zinc since 2026-10-30
# (``data.metal_scope.BONDED_METAL_VALENCES``), so the best-effort tier names the dichloropalladium pyridyl
# porphyrin through its ionic drawing (and the dative drawing through its split drawing); the default tier
# keeps its output.
PALLADIUM_ROWS = {
    "TPP_pyridyl_PdCl2_covalent": "5,10,15-triphenyl-20-(pyridin-4-yl)porphyrin\u2014chloride\u2014palladium(2+) (1/2/1)",
    "TPP_pyridyl_PdCl2_dative": "5,10,15-triphenyl-20-(pyridin-4-yl)porphyrin\u2014chloride\u2014palladium(2+) (1/2/1)",
}


@pytest.mark.parametrize("row", [r for r in DECLINED_TODAY if r["id"] in PALLADIUM_ROWS], ids=lambda r: r["id"])
def test_the_palladium_rows_are_named_at_best_effort(row):
    from tests.support.rt_assert import smiles_without_dative_bonds
    default = name_default(row["smiles"])
    assert (default.get("tier"), default.get("limit_code") or "", default.get("name") or "") == (
        row["default_tier"], row["default_limit_code"], row["default_name"]), default
    breadth = name_breadth(row["smiles"])
    assert (breadth.get("name"), breadth.get("tier")) == (PALLADIUM_ROWS[row["id"]], "systematic_verified"), breadth
    assert name_is_rt_exact(breadth["name"], smiles_without_dative_bonds(row["smiles"]) or row["smiles"]), breadth


@pytest.mark.parametrize("row", [r for r in DECLINED_TODAY if r["id"] not in CHANGES and r["id"] not in PALLADIUM_ROWS],
                         ids=lambda r: r["id"])
def test_declined_rows_keep_todays_output(row):
    """Every class the producer declines keeps the output it had before this route change: the label and
    the limit code at the default tier, the name byte for byte at best-effort. The two rows that change are
    in ``CHANGES``."""
    default = name_default(row["smiles"])
    assert (default.get("tier"), default.get("limit_code") or "", default.get("name") or "") == (
        row["default_tier"], row["default_limit_code"], row["default_name"]), default
    breadth = name_breadth(row["smiles"])
    assert (breadth.get("tier"), breadth.get("limit_code") or "", breadth.get("name") or "") == (
        row["best_effort_tier"], row["best_effort_limit_code"], row["best_effort_name"]), breadth


@pytest.mark.parametrize("row", [r for r in DECLINED_TODAY if r["id"] in CHANGES], ids=lambda r: r["id"])
def test_the_two_declined_rows_that_change_are_named_exactly(row):
    change = CHANGES[row["id"]]
    default = name_default(row["smiles"])
    if "default" in change:
        # NPL 2026-11-01: INTRODUCTION (the Blue Book): no PIN for a parent, so the default tier
        # declines it; best-effort names it, ``systematic_verified``
        assert_default_tier_declines(row["smiles"])
    else:   # the default tier keeps its output
        assert (default.get("tier"), default.get("limit_code") or "", default.get("name") or "") == (
            row["default_tier"], row["default_limit_code"], row["default_name"]), default
    breadth = name_breadth(row["smiles"])
    assert breadth.get("name") == change["best_effort"], breadth
    assert name_is_rt_exact(breadth["name"], row["smiles"]), breadth
    assert breadth.get("name") != row["best_effort_name"]      # it is a change, against the recorded name


def test_u2_free_base_at_both_tiers():
    expected = "2-(azidomethyl)-5,10,15,20-tetraphenylporphyrin"
    # NPL 2026-11-01: INTRODUCTION (the Blue Book): the default tier declines a parent's name.
    assert_default_tier_declines(U2_FREE_BASE)
    result = name_breadth(U2_FREE_BASE)
    assert (result.get("name"), result.get("tier")) == (expected, "systematic_verified"), result
    assert name_is_rt_exact(expected, U2_FREE_BASE)


def test_u2_with_its_free_zinc_ion():
    """The drawing is a neutral free-base porphyrin plus a bare [Zn+2]. The default tier names nothing for it
    (UNNAMEABLE: no producer of that tier names the whole drawing); the best-effort name is the composite ``<ligand>--zinc(2+) (1/1)`` and it must read back
    exactly: a composite that did not would turn a name into an abstention."""
    default = name_default(U2)
    assert default.get("name") == "zinc compound (not supported)", default      # no producer of the default tier names it
    assert default.get("limit_code") == "UNNAMEABLE", default
    breadth = name_breadth(U2)
    assert breadth.get("name") and name_is_rt_exact(breadth["name"], U2), breadth


def test_u1_default_tier_and_round_trip():
    # NPL 2026-11-01: INTRODUCTION (the Blue Book): the default tier declines; best-effort names it.
    assert_default_tier_declines(U1)
    result = name_breadth(U1)
    name = result.get("name") or ""
    assert result.get("tier") == "systematic_verified", result
    assert name == "5,10,15,20-tetrakis(3-azidophenyl)porphyrin", name
    assert name_is_rt_exact(name, U1), name


def test_u1_azide_is_azido_at_both_tiers():
    """The aryl azide is the prefix ``azido``; the ylide spelling ``triaz-1,2-dien-2-ium-1-idyl`` cites
    locants of the capped molecule and is not made as a prefix."""
    expected = "5,10,15,20-tetrakis(3-azidophenyl)porphyrin"
    assert name_breadth(U1).get("name") == expected
    assert_default_tier_declines(U1)    # NPL 2026-11-01: INTRODUCTION (the Blue Book)


def test_no_new_class_in_the_dispatch_table():
    """The route is the natural-product class; no class was added for the porphyrin producer."""
    from orthonym.routing.dispatch_table import DISPATCH_TABLE, StoutClass
    assert not [member for member in StoutClass if "PORPHYRIN" in member.name]
    assert set(DISPATCH_TABLE) == set(StoutClass)
    assert Orthonym().name_tiered("CCO").get("name") == "ethanol"
