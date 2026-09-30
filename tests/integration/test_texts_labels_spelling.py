"""Texts, labels and spelling (user decisions of 2026-09-29), the two engine classes.

A. Carbon-free compounds carry a preselected name at most, never a PIN.
    PREFERRED IUPAC NAMES (the Blue Book): "the label 'PIN' is added to
   the names of compounds whose parent hydride contains at least one of the
   following elements: B, Si, Ge, Sn, Pb, N, P, As, Sb, Bi, O, S, Se, Te, Po, F, Cl,
   Br, I, At, and that also contain at least one carbon atom in their structure";
   :2058 the PIN rules for such compounds "that do not contain carbon... will be
   discussed in a further publication... the label 'preselected name' is added to
   appropriate names"; PRESELECTED NAMES (:2062); '(HO)3PO phosphoric acid
   (preselected name)' (:2076), 'sulfuric acid (preselected name)' (:35449),
   'tetrachlorosilane (preselected name)' (:35758). The Blue Book labels no
   carbon-free structure '(PIN)'. The tier label, however, follows the paper's
   measured run (user decision 2026-09-30, replacing the 2026-09-29 label
   systematic_verified): the label of the naming path, 'sodium chloride'
   pin_verified -- the paper, Methods, "Tiers": "The tier labels describe how the
   engine built a name". The name is unchanged. A carbon-containing compound keeps
   its label.

B. A multiplied organic cation in a salt takes 'bis', 'tris',... when it carries a
   cumulative cationic suffix or is a substituted parent cation, and every
   multiplied ion takes the nesting mark. (:43564)
   'bis(methanaminium) sulfate (PIN)'; (1) (:41431) "These cationic
   suffixes are used with the multiplying prefixes 'bis', 'tris', etc. to denote
   multiplicity"; (c) (:7110); (c) (:7035) "any component which is
   substituted automatically requires use of the multiplicative forms 'bis',
   'tris', etc."; (c) (:7104) a component beginning with a multiplicative
   prefix is enclosed; (:7446) nesting order {[({})]}. A retained name
   plus 'ium' ('anilinium') and a metal cation ('disodium carbonate (PIN)',:31579)
   keep 'di'; so does a simple anion ('calcium diacetate (PIN)',:31571).

Every shipped name is read back by a FRESH OPSIN call outside the engine
(``tests.support.rt_assert.assert_full_rt``, full InChIKey). Fixtures are Blue Book
rows, minimal analogues and one milestone1500 row, never a holdout split rows (checked by
full InChIKey and connectivity against eval/splits/a holdout split.json).
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import assert_full_rt
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "C/C=C/C=C/C(=O)[O-].C/C=C/C=C/C(=O)[O-].C/C=C/C=C/C(=O)[O-].[Al+3]",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]


def _row(smiles, tier):
    if tier == "pin" and smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    if tier == "pin":
        return Orthonym(style="pin").name_tiered(smiles)
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


# ---------------------------------------------------------------------------
# A. Carbon-free compounds: preselected names; the label of the naming path
# ---------------------------------------------------------------------------

CARBON_FREE = [
    ("OS(=O)(=O)O", "sulfuric acid"),                    # (preselected name):35449
    ("N", "ammonia"),
    ("[O-]S(=O)O", "hydrogen sulfite"),
    ("Cl[Si](Cl)(Cl)Cl", "tetrachlorosilane"),           # (preselected name):35758
    ("F[P-](F)(F)(F)(F)F", "hexafluoro-λ5-phosphanuide"),
    ("O=P(O)(O)O", "phosphoric acid"),                   # (preselected name):2076
    ("P", "phosphane"),                                  # BB row 12.1, 'preselected'
    ("[NH4+]", "azanium"),                               # (preselected name):41373
]


@pytest.mark.parametrize("smiles,expected", CARBON_FREE)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_a_carbon_free_compound_keeps_the_label_of_its_naming_path(smiles, expected,
                                                                   tier):
    row = _row(smiles, tier)
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert row["verified"] == "opsin", row
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles,expected", [
    ("[Na+].OC([O-])=O", "sodium hydrogen carbonate"),               # (PIN):31623
    ("CC[NH+](CC)CC.OS(=O)(=O)[O-]",
     "N,N-diethylethanaminium hydrogen sulfate"),                    # (PIN):43566
    ("C[NH3+].[Cl-]", "methanaminium chloride"),                     #:4712
    ("C", "methane"),
])
def test_a_a_compound_with_carbon_keeps_its_pin_label(smiles, expected):
    row = _row(smiles, "pin")
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert_full_rt(expected, smiles)


# ---------------------------------------------------------------------------
# B. Multiplied ions in salts
# ---------------------------------------------------------------------------

MULTIPLIED = [
    # the Blue Book PIN,:43564) and its hydrogen-anion sibling
    ("C[NH3+].C[NH3+].[O-]S(=O)(=O)[O-]", "bis(methanaminium) sulfate"),
    ("C[NH3+].C[NH3+].OP(=O)([O-])[O-]", "bis(methanaminium) hydrogen phosphate"),
    # other cumulative cationic suffixes (Table 7.4, (1))
    ("CC[NH3+].CC[NH3+].CC[NH3+].[O-]P(=O)([O-])[O-]", "tris(ethanaminium) phosphate"),
    ("C1CCC(CC1)[NH3+].C1CCC(CC1)[NH3+].[O-]S(=O)(=O)[O-]",
     "bis(cyclohexanaminium) sulfate"),
    ("CC=[NH2+].CC=[NH2+].[O-]S(=O)(=O)[O-]", "bis(ethaniminium) sulfate"),
    ("CC(=O)[NH3+].CC(=O)[NH3+].[O-]S(=O)(=O)[O-]", "bis(acetamidium) sulfate"),
    ("CC#[NH+].CC#[NH+].[O-]S(=O)(=O)[O-]", "bis(acetonitrilium) sulfate"),
    # substituted parent cations (c))
    ("C[S+](C)C.C[S+](C)C.[O-]S(=O)(=O)[O-]", "bis(trimethylsulfanium) sulfate"),
    ("c1ccc(cc1)[I+]c1ccccc1.c1ccc(cc1)[I+]c1ccccc1.[O-]S(=O)(=O)[O-]",
     "bis(diphenyliodanium) sulfate"),
    # anions that begin with a multiplying prefix (c), (c))
    ("[Ca+2].[O-]C(=O)C(F)(F)F.[O-]C(=O)C(F)(F)F", "calcium bis(trifluoroacetate)"),
    ("[Ca+2].[O-]C(=O)C(Cl)Cl.[O-]C(=O)C(Cl)Cl", "calcium bis(dichloroacetate)"),
    # nesting of the enclosing mark
    ("[Ca+2].CC(C)Cc1ccc(cc1)C(C)C([O-])=O.CC(C)Cc1ccc(cc1)C(C)C([O-])=O",
     "calcium bis{2-[4-(2-methylpropyl)phenyl]propanoate}"),
]


@pytest.mark.parametrize("smiles,expected", MULTIPLIED)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_b_multiplied_ion_spelling(smiles, expected, tier):
    row = _row(smiles, tier)
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert_full_rt(expected, smiles)


def test_b_milestone1500_aluminium_salt_takes_brackets():
    # milestone1500 row; (:7478) stereodescriptor parentheses count,
    # (:7509) consecutive marks of one level take the next. Aluminium:
    # no PIN status (the Blue Book), so systematic_verified.
    smiles = "C/C=C/C=C/C(=O)[O-].C/C=C/C=C/C(=O)[O-].C/C=C/C=C/C(=O)[O-].[Al+3]"
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == "aluminium tris[(2E,4E)-hexa-2,4-dienoate]", (tier, row)
        assert row["tier"] == "systematic_verified", (tier, row)
    assert_full_rt("aluminium tris[(2E,4E)-hexa-2,4-dienoate]", smiles)


@pytest.mark.parametrize("smiles,expected", [
    # retained name + 'ium' (1)) keeps 'di' (d))
    ("[NH3+]c1ccccc1.[NH3+]c1ccccc1.[O-]S(=O)(=O)[O-]", "dianilinium sulfate"),
    ("NC(N)=[NH2+].NC(N)=[NH2+].[O-]C([O-])=O", "diguanidinium carbonate"),
    # a simple anion and a metal cation keep 'di'
    ("CC(=O)[O-].CC(=O)[O-].[Ca+2]", "calcium diacetate"),           # (PIN):31571
    ("[Mg+2].CS(=O)(=O)[O-].CS(=O)(=O)[O-]", "magnesium dimethanesulfonate"),
    ("[Na+].[Na+].[O-]C([O-])=O", "disodium carbonate"),             # (PIN):31579
])
def test_b_simple_ions_keep_di(smiles, expected):
    row = _row(smiles, "pin")
    assert row["name"] == expected, row
    assert_full_rt(expected, smiles)
