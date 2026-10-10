""" mixed-type multi-anion naming (a phase).

A skeleton carrying TWO anionic centres of DIFFERENT acid classes (e.g. a
carboxylate + a sulfonate) must name the SENIOR class as the parent anion
suffix: carboxylic acid senior to sulfonic acid in the
class-seniority order — the same rule that gives the BB's own
"3-oxidonaphthalene-2-carboxylate (PIN) (carboxylate senior to olate)") and
cite every OTHER (junior) anionic centre by its anionic substituent prefix
: 'sulfonato' for -SO2-O-, 'phosphonato' for -P(O)(O-)2 — BB
:41211/:41213), never the neutral prefix ('sulfo'/'phosphono') — a neutral
prefix silently drops the charge and denotes a DIFFERENT (mono-anion)
molecule, which the gate correctly rejects (measured: '4-sulfobenzoate'
-> 'unknown organic compound' before this fix).

All targets below are verified round-trip-exact (OPSIN 2.9.0 + InChIKey) by
the implementing session; see the report at
internal notes
"""
import pytest
from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # 4-sulfobenzoate dianion: carboxylate parent (senior), sulfonate junior
    # prefix. RT-verified: OPSIN('4-sulfonatobenzoate') round-trips to the
    # identical dianion (InChIKey match), the neutral-prefix form
    # ('4-sulfobenzoate') round-trips to the MONO-anion (different molecule).
    ("O=C([O-])c1ccc(S(=O)(=O)[O-])cc1", "4-sulfonatobenzoate"),
    # aliphatic carboxylate+sulfonate dianion (3-sulfopropanoic acid, fully
    # deprotonated). RT-verified round-trip-exact.
    ("[O-]C(=O)CCS(=O)(=O)[O-]", "3-sulfonatopropanoate"),
    # 2-carbon case: locant omitted — unambiguous with only one
    # non-C1 position). RT-verified round-trip-exact.
    ("[O-]C(=O)CS(=O)(=O)[O-]", "sulfonatoacetate"),
])
def test_integration_mixed_dianion_names(namer, smi, expected):
    assert _dt_obj_name(namer, smi) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # mono mixed (already worked pre-fix) — must stay unchanged.
    ("OC(=O)CCS(=O)(=O)[O-]", "2-carboxyethane-1-sulfonate"),
    # dicarboxylate (already worked pre-fix) — must stay unchanged.
    ("[O-]C(=O)CCC(=O)[O-]", "butanedioate"),
    # acid-ester anion (a phase Slice A) — must stay unchanged.
    ("CCCCCCCCCCCCOS(=O)(=O)[O-]", "dodecyl sulfate"),
    # zwitterion (unrelated path) — must stay unchanged.
    ("C[N+](C)(C)CCC(=O)[O-]", "3-(trimethylazaniumyl)propanoate"),
    # plain sulfonate / methanesulfonate — must stay unchanged.
    ("C1=CC=CC=C1S(=O)(=O)[O-]", "benzenesulfonate"),
    ("CS(=O)(=O)[O-]", "methanesulfonate"),
])
def test_integration_regressions_unchanged(namer, smi, expected):
    assert _dt_obj_name(namer, smi) == expected


@pytest.mark.opsin_gate
def test_integration_failclosed_never_wrong(namer):
    # A shape the new machinery cannot yet name correctly must abstain to the
    # honest sentinel, never emit the neutral-prefix ('...sulfo...' /
    # '...phosphono...') form that denotes a different (mono-anion) molecule.
    out = _dt_obj_name(namer, "O=P([O-])([O-])c1ccc(C(=O)[O-])cc1")  # triple mixed anion
    assert "sulfo" not in out
    assert "phosphono" not in out


# ---------------------------------------------------------------------------
# fix a performance pass (wp1-zero-wrong): an S/P-oxoacid parent anion with junior
# anionic chalcogen centres (-O- / -S-).
#
# "ANIONIC CENTERS IN BOTH PARENT COMPOUNDS AND SUBSTITUENT GROUPS"
# (the Blue Book): "one anion must be chosen as the parent anion and the
# other expressed as anionic substituent group(s)". "Prefixes for
# anionic chalcogens": "–O– oxido (preselected prefix)" (:41223), "–S– sulfido
# (preselected prefix)" (:41225). "CHOICE OF AN ANIONIC PARENT
# STRUCTURE" (:41261): (a) "parent with the maximum number of anionic centers,
# including anionic suffixes" (:41265); (d) "N > P >... > O > S" (:41281);
# (e) the suffix seniority of (:41289), as in "3-oxidonaphthalene-2-
# carboxylate (PIN) (carboxylate senior to olate)" (:41295). Salt words:
# "Salts" (:31563) "Neutral salts of acids are named by citing the
# name of the cation(s) followed by the name of the anion".
#
# The first four rows shipped a WRONG molecule as pin_verified before this fix
# ('trisodium 5-hydroxybenzene-1,3-disulfonate': OPSIN parses a dianion + 3 Na+,
# net +1). Every expected name below is OPSIN 2.9.0 full-InChIKey exact
# (independent batch call; TRIAGE.md ' fix a performance pass -- wp1-zero-wrong').
# ---------------------------------------------------------------------------
from tests.support.rt_assert import name_is_rt_exact  # noqa: E402
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
    "CC([O-])CC(C)CC([O-])=O.[Na+].[Na+]",
    "C[N+](C)(C)CCC(=O)[O-]",
    "Cc1ccc([O-])c(C([O-])=O)c1.[Na+].[Na+]",
    "[O-]CC(C)C([O-])=O.[Na+].[Na+]",
    # '[O-]CCP(=O)([O-])[O-].[Na+].[Na+].[Na+]' left this set (leads program L4, item 30b): the
    # functional-parent form is built now, see test_chain_phosphonate_is_the_functional_
    # parent_form.
    "[O-]c1cc(C)cc(C([O-])=O)c1.[Na+].[Na+]",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_obj_name(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return namer_obj.name(smiles)


def _dt_obj_row(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return namer_obj.name_tiered(smiles)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)


_JUNIOR_CHALCOGEN_PINS = [
    # previously WRONG molecule at pin_verified
    ("[O-]c1cc(cc(c1)S([O-])(=O)=O)S([O-])(=O)=O.[Na+].[Na+].[Na+]",
     "trisodium 5-oxidobenzene-1,3-disulfonate"),
    ("[O-]c1cc(cc(c1)S([O-])(=O)=O)S([O-])(=O)=O.[K+].[K+].[K+]",
     "tripotassium 5-oxidobenzene-1,3-disulfonate"),
    ("[O-]c1cc(cc(c1)S([O-])(=O)=O)S([O-])(=O)=O.[Na+].[Ca+2]",
     "calcium sodium 5-oxidobenzene-1,3-disulfonate"),
    ("[O-]c1ccc(cc1)P(=O)([O-])[O-].[Na+].[Na+].[Na+]",
     "trisodium (4-oxidophenyl)phosphonate"),
    # previously abstained at the PIN tier
    ("[O-]S(=O)(=O)CC[O-].[Na+].[Na+]", "disodium 2-oxidoethane-1-sulfonate"),
    ("[S-]CCS(=O)(=O)[O-].[Na+].[Na+]", "disodium 2-sulfidoethane-1-sulfonate"),
    ("[O-]c1ccc(cc1)S([O-])(=O)=O.[Na+].[Na+]",
     "disodium 4-oxidobenzene-1-sulfonate"),
    ("[O-]CCS([O-])=O.[Na+].[Na+]", "disodium 2-oxidoethane-1-sulfinate"),
    ("[O-]S(=O)(=O)CC[O-].[Ca+2]", "calcium 2-oxidoethane-1-sulfonate"),
    ("[O-]c1ccc(cc1)S([O-])(=O)=O.[K+].[K+]",
     "dipotassium 4-oxidobenzene-1-sulfonate"),
    # the bare dianion (was 'unknown organic compound' at the PIN tier)
    ("[O-]S(=O)(=O)CC[O-]", "2-oxidoethane-1-sulfonate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", _JUNIOR_CHALCOGEN_PINS)
def test_junior_anionic_chalcogen_is_oxido_or_sulfido(namer, smi, expected):
    res = _dt_obj_row(namer, smi)
    assert res["name"] == expected
    assert res["tier"] == "pin_verified"
    assert name_is_rt_exact(expected, smi)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", [s for s, _ in _JUNIOR_CHALCOGEN_PINS[:4]])
def test_mixed_multi_anion_never_drops_a_charge(namer, smi):
    # the charge-dropping neutral prefix must never ship, at any tier
    for tier_namer in (namer, _best_effort()):
        out = tier_namer.name(smi)
        assert "hydroxy" not in out, out
        assert not name_is_rt_exact(out.replace("oxido", "hydroxy"), smi)


def _best_effort():
    from orthonym.cli import _emit_tier_flags
    return Orthonym(style="pin", **_emit_tier_flags("best-effort"))


@pytest.mark.opsin_gate
def test_p72_7a_boundary_is_not_pin_verified(namer):
    # Two olate centres vs one sulfonate: (a) makes the bis(olate) the
    # parent ('trisodium 3-sulfonatopropane-1,2-bis(olate)'); both that name and
    # 'trisodium 2,3-dioxidopropane-1-sulfonate' round-trip, so only the rule can
    # choose. The oxoacid-parent producer must decline (never pin_verified).
    res = _dt_obj_row(namer, "[O-]CC([O-])CS(=O)(=O)[O-].[Na+].[Na+].[Na+]")
    assert res["tier"] != "pin_verified"
    assert "dioxidopropane" not in (res["name"] or "")


@pytest.mark.opsin_gate
def test_chain_phosphonate_is_the_functional_parent_form(namer):
    # RESOLVED (leads program, lane L4, item 30b). This row pinned the defect: the chain
    # phosphonic acid came from the neutral namer's suffix form
    # ('trisodium 2-oxidoethane-1-phosphonate', right molecule, not the PIN) and was demoted.
    # The polyfunctional suffix assembly now offers the namer first, so the PIN is
    # built: the Blue Book "Substitution of mononuclear noncarbon oxoacids with hydrogen
    # atoms attached to the central atom (substitutable hydrogen)",:35461 "ethylphosphonic acid
    # (PIN) (not ethanephosphonic acid)" -> 'trisodium (2-oxidoethyl)phosphonate' (OPSIN reads
    # it back to the full InChIKey, asserted below; the suffix form reads back too).
    smi = "[O-]CCP(=O)([O-])[O-].[Na+].[Na+].[Na+]"
    res = _dt_obj_row(namer, smi)
    assert res["name"] == "trisodium (2-oxidoethyl)phosphonate"
    assert name_is_rt_exact(res["name"], smi)
    assert res["tier"] == "pin_verified"
    assert res["is_pin"] is True
    assert name_is_rt_exact("trisodium 2-oxidoethane-1-phosphonate", smi)


# (the Blue Book) "Simple prefixes... are arranged
# alphabetically"; (g) (:3306) "lowest locants for the substituent cited
# first as a prefix in the name", e.g. "1-methyl-4-nitronaphthalene (PIN) (not
# 4-methyl-1-nitronaphthalene)" (:3317). The 'hydroxy' -> 'oxido' swap moves
# the prefix past 'methyl' without re-ordering it, so these names are the right
# molecule but NOT the PIN (PINs: disodium 3-methyl-5-oxidobenzoate, 3-methyl-
# 5-oxidohexanoate, 5-methyl-2-oxidobenzoate, 2-methyl-3-oxidopropanoate; all
# RT-exact). Until the producer re-orders them, they must not be pin_verified.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", [
    "[O-]c1cc(C)cc(C([O-])=O)c1.[Na+].[Na+]",
    "CC([O-])CC(C)CC([O-])=O.[Na+].[Na+]",
    "Cc1ccc([O-])c(C([O-])=O)c1.[Na+].[Na+]",
    "[O-]CC(C)C([O-])=O.[Na+].[Na+]",
])
def test_rank_changing_oxido_swap_is_not_pin_verified(namer, smi):
    res = _dt_obj_row(namer, smi)
    assert name_is_rt_exact(res["name"], smi)
    assert res["tier"] != "pin_verified"
    assert res["is_pin"] is False


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # no sibling sorts between 'hydroxy' and 'oxido': the swap keeps the PIN
    ("[O-]c1ccc(Cl)cc1C([O-])=O.[Na+].[Na+]", "disodium 5-chloro-2-oxidobenzoate"),
    ("[O-]c1cc2ccccc2cc1C([O-])=O.[Na+].[Na+]",
     "disodium 3-oxidonaphthalene-2-carboxylate"),
    ("[O-]c1ccccc1C([O-])=O.[Na+].[Na+]", "disodium 2-oxidobenzoate"),
])
def test_rank_preserving_oxido_swap_stays_pin_verified(namer, smi, expected):
    res = _dt_obj_row(namer, smi)
    assert res["name"] == expected
    assert res["tier"] == "pin_verified"


def test_rank_guard_unit():
    from orthonym.rules.ions import _prefix_swap_may_change_rank as g
    assert g("3-oxido-5-methylbenzoate", "hydroxy", "oxido")
    assert g("2-[(2-oxidoethyl)]-5-nitrobenzoate", "hydroxy", "oxido")
    assert not g("5-chloro-2-oxidobenzoate", "hydroxy", "oxido")
    assert not g("3-oxidonaphthalene-2-carboxylate", "hydroxy", "oxido")
    assert not g("5-oxidobenzene-1,3-disulfonate", "hydroxy", "oxido")
    assert not g("(4-oxidophenyl)phosphonate", "hydroxy", "oxido")
    # a multiplied sibling is alphabetized without its multiplier
    assert g("2-oxido-3,5-dimethylbenzoate", "hydroxy", "oxido")
    assert not g("2-sulfido-4-sulfonatobenzoate", "sulfanyl", "sulfido")
