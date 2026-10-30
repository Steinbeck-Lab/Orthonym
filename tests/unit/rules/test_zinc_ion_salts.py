"""Zinc salts and adducts of organic ions are composed from their ions (M2), and the label of a
salt of a carbanion with a Group 3-12 metal cation.

``perception.metals.assembly_has_out_of_scope_metal`` reads the scope table
``data/metal_scope.py``. A metal admitted to ``ION_SALTS`` lets the best-effort tier compose a
salt of its cation with organic ions (``rules.salts.name_salt``) and an adduct
(``rules.adducts.name_adduct``) from the names of the ions, as it does for sodium. A metal also
admitted to ``CARBANION_SALTS`` is composed beside a carbanion or a hydride. Zinc is admitted to
both. Every name is read back by a fresh OPSIN call to the full standard InChIKey of the input
and by the unchanged validity gate of the engine; a near-miss name is rejected by it.

Labels. INTRODUCTION (the Blue Book): "neither preferred IUPAC names or preselected
names (see for organometallic compounds involving the transition elements (including the
Group 3 elements) and Groups 1 and 2 elements, except for 'ocene' compounds, are noted";
the heading of puts Groups 3 through 12 together; (:40173) calls
"sodium ethenide" "also acceptable; it is a compositional name analogous to sodium chloride".
A salt of a carbanion with a Group 3-12 metal cation is therefore a compositional name of an
organometallic compound and is systematic_verified, not pin_verified (user answer 3 of
2026-10-13). The ocene names "Ocenes": "(PIN)") and the eta-hapto additive names of the
organometallic assembler keep their labels; carbon-free salts and the salts of acids, alcohols and
thiols keep theirs (the named exceptions of section 3 of the plan).
"""
import csv

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.rules.metal_labels import is_carbanion_salt_of_group_3_12_metal
from tests.support.metal_baselines import baseline_path, declared_bonded_metal_moves, identity_view, read_rows
from tests.support.rt_assert import assert_full_rt, name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

EM_DASH = "—"
PREDICATE = is_carbanion_salt_of_group_3_12_metal


def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


def _panel_smiles(panel_id, drawing):
    with open(baseline_path("zinc").parent / "panel_pairs.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["id"] == panel_id and r["drawing"] == drawing:
                return r["smiles"]
    raise AssertionError((panel_id, drawing))


# --------------------------------------------------------------------------- composition

SALTS = [
    "[O-][n+]1ccccc1[S-].[O-][n+]1ccccc1[S-].[Zn+2]",              # zinc pyrithione, ionic drawing
    "CC(=CC(C)=O)[O-].CC(=CC(C)=O)[O-].[Zn+2]",                    # zinc acetylacetonate, ionic drawing
    _panel_smiles("E1", "ionic"),                                  # zinc tetraphenylporphyrin, ionic drawing
]


@pytest.mark.parametrize("smiles", SALTS)
def test_organic_ion_salts_of_zinc_are_composed_as_salts(smiles):
    """The adduct form 'ligand' + em dash + 'zinc(2+) (2/1)' is the form of a metal the table does
    not admit; zinc is admitted, so its salts of organic anions are named as salts, as sodium's are."""
    row = _row(smiles, "best-effort")
    assert row["name"].startswith("zinc ") and EM_DASH not in row["name"], row["name"]
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False
    assert_full_rt(row["name"], smiles)
    assert _row(smiles, "pin")["tier"] == "abstain"                # the default tier is not widened


def test_other_metals_keep_the_adduct_form_of_organic_ion_salts():
    """Technetium and plutonium are not in the table: the adduct form of the start commit, byte for byte (platinum,
    the other metal of the start commit, is admitted since 2026-10-30, copper and nickel since lane B2: their salts are
    named from their ions, ``test_bonded_metal_admissions.py``)."""
    for smiles, name in [
        ("[O-][n+]1ccccc1[S-].[O-][n+]1ccccc1[S-].[Tc+2]", f"1-oxido-2-sulfidopyridin-1-ium{EM_DASH}technetium(2+) (2/1)"),
        ("[O-][n+]1ccccc1[S-].[O-][n+]1ccccc1[S-].[Pu+2]", f"1-oxido-2-sulfidopyridin-1-ium{EM_DASH}plutonium(2+) (2/1)"),
        ("CC(=CC(C)=O)[O-].CC(=CC(C)=O)[O-].[Tc+2]", f"2-oxido-4-oxopent-2-ene{EM_DASH}technetium(2+) (2/1)"),
        ("CC(=CC(C)=O)[O-].CC(=CC(C)=O)[O-].[Pu+2]", f"2-oxido-4-oxopent-2-ene{EM_DASH}plutonium(2+) (2/1)"),
    ]:
        row = _row(smiles, "best-effort")
        assert (row["name"], row["tier"], row["is_pin"]) == (name, "systematic_verified", False), smiles
        assert _row(smiles, "pin")["limit_code"] == "UNSUPPORTED_ELEMENT"


def test_platinum_carbanide_stays_out():
    """The platinum(II)-carbanide hazard of -1b: a true metal with a carbanion is never
    rendered as a salt of a best-effort-named organic ion unless the table admits it. The names are
    those of the start commit."""
    best = _row("[Pt+2].[CH3-].[CH3-]", "best-effort")
    assert (best["name"], best["tier"], best["is_pin"]) == ("platinum(II) methanide", "systematic_verified", False)
    default = _row("[Pt+2].[CH3-].[CH3-]", "pin")
    assert default["tier"] == "abstain" and default["limit_code"] == "NO_VERIFIED_PIN"


def test_a_wrong_salt_name_is_rejected_by_the_gate(monkeypatch):
    """H3: OPSIN accepts near-miss names ('zinc dioctanoate' reads as another structure than zinc
    stearate). The read-back is the only guard: a salt name that is not the structure's never ships."""
    stearate = _panel_smiles("C4", "ionic")
    assert name_is_rt_exact("zinc dioctadecanoate", stearate) is True
    assert name_is_rt_exact("zinc dioctanoate", stearate) is False        # the negative control
    import orthonym.rules.salts as salts
    real = salts.name_salt

    def near_miss(mol, *args, **kwargs):
        if any(a.GetSymbol() == "Zn" for a in mol.GetAtoms()):
            return "zinc dioctanoate"
        return real(mol, *args, **kwargs)

    monkeypatch.setattr(salts, "name_salt", near_miss)
    for tier in ("pin", "best-effort"):
        row = _row(stearate, tier)
        assert row["name"] != "zinc dioctanoate", tier
        if row["tier"] != "abstain":
            assert_full_rt(row["name"], stearate)


# --------------------------------------------------------------------------- labels

CARBANION_SALTS_OF_ZINC = [
    ("[CH2-]C.[CH2-]C.[Zn+2]", "zinc bis(ethan-1-ide)"),
    ("[c-]1ccccc1.[c-]1ccccc1.[Zn+2]", "zinc dibenzenide"),
    ("[CH3-].[CH3-].[Zn+2]", "zinc bis(methanide)"),
]


def test_each_ion_of_a_salt_is_named_with_the_hang_budgets_of_one_compound(monkeypatch):
    """The ions of one salt are compounds of their own, "Neutral salts of acids are named
    by citing the name of the cation(s) followed by the name of the anion"), as the components of an
    adduct are: ``salts._name_organic_ion_best_effort`` names an ion with the budgets of one
    compound, whatever is left of the budgets of the whole assembly, and puts the enclosing ones back.
    Two macrocyclic anions that shared one budget ran it out in the second and abstained the drawing."""
    from orthonym.assembly import fragment_naming
    from orthonym.rules import salts
    guard = fragment_naming._fragment_guard
    names = ("name_call_depth", "perf_budget", "analysis_budget")
    absent = object()
    saved = {n: getattr(guard, n, absent) for n in names}
    seen = []

    def name_of_one_ion(self, smiles, *args, **kwargs):
        seen.append((guard.perf_budget, guard.analysis_budget))
        return "pyrrol-1-ide"

    monkeypatch.setattr(Orthonym, "name", name_of_one_ion)
    guard.name_call_depth, guard.perf_budget, guard.analysis_budget = 1, 7, 3     # budgets nearly spent
    try:
        word = salts._name_organic_ion_best_effort(
            "c1cc[n-]c1", style="general", general_fallback_unverified=True)
        after = (guard.perf_budget, guard.analysis_budget)
    finally:
        for n, value in saved.items():
            delattr(guard, n) if value is absent else setattr(guard, n, value)
    assert word == "pyrrol-1-ide"
    assert seen == [(fragment_naming._PERF_BUDGET, fragment_naming._ANALYSIS_CALL_BUDGET)]
    assert after == (7, 3)


@pytest.mark.parametrize("smiles, name", CARBANION_SALTS_OF_ZINC)
def test_carbanion_salts_of_zinc_are_not_pin_labelled(smiles, name):
    """ notes no PIN for organometallic compounds of the transition elements (the heading of
     puts Group 12 with them); calls the salt name 'a compositional name'. The wider
    tiers keep the name, the default tier declines it. The name is unchanged."""
    row = _row(smiles, "best-effort")
    assert row["name"] == name
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False
    assert_full_rt(row["name"], smiles)
    default = _row(smiles, "pin")
    assert default["tier"] == "abstain" and default["limit_code"] == "NO_VERIFIED_PIN"
    assert Orthonym(style="pin").name(smiles, on_failure="label") == "unknown organic compound"


def test_carbon_free_and_acid_salts_keep_their_label():
    """The named exceptions of section 3: carbon-free salts (the convention of 2026-09-30, kept for
    every metal alike by user answer 3) and the salts of acids, alcohols and thiols (the pattern of
     and, ASSUMED for zinc: the book prints no zinc example)."""
    for smiles, name in [
        ("[Cl-].[Cl-].[Zn+2]", "zinc dichloride"),
        ("CC(=O)[O-].CC(=O)[O-].[Zn+2]", "zinc diacetate"),
        ("CC[O-].CC[O-].[Zn+2]", "zinc diethoxide"),
        ("[Na+].[Cl-]", "sodium chloride"),
    ]:
        for tier in ("pin", "best-effort"):
            row = _row(smiles, tier)
            assert (row["name"], row["tier"], row["is_pin"]) == (name, "pin_verified", True), (smiles, tier)


def test_cyanide_is_not_a_carbanion_for_the_label():
    """Cyanide is the anion of hydrogen cyanide, an acid: 'zinc dicyanide' stays in the first class
    of section 3 (ASSUMED, listed in ZN11)."""
    for tier in ("pin", "best-effort"):
        row = _row("[C-]#N.[C-]#N.[Zn+2]", tier)
        assert (row["name"], row["tier"], row["is_pin"]) == ("zinc dicyanide", "pin_verified", True), tier


OTHER_METALS = [   # constructed carbanion salts of other Group 3-12 metals, names of the start commit
    ("[Cu+2].[c-]1ccccc1.[c-]1ccccc1", "copper(II) benzenide"),
    ("[Ni+2].[c-]1ccccc1.[c-]1ccccc1", "nickel(II) benzenide"),
    ("[Cd+2].[c-]1ccccc1.[c-]1ccccc1", "cadmium dibenzenide"),
    ("[Hg+2].[c-]1ccccc1.[c-]1ccccc1", "mercury(II) benzenide"),
    ("[Sc+3].[c-]1ccccc1.[c-]1ccccc1.[c-]1ccccc1", "scandium tribenzenide"),
]


@pytest.mark.parametrize("smiles, name", OTHER_METALS)
def test_carbanion_salts_of_other_group_3_12_metals_follow_the_rule(smiles, name):
    """The rule is the user's wording: every Group 3-12 metal alike. The name is the one of the
    start commit (pin_verified there); only the label moves."""
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        if tier == "pin":
            assert row["tier"] == "abstain" and row["limit_code"] == "NO_VERIFIED_PIN", smiles
        else:
            assert (row["name"], row["tier"], row["is_pin"]) == (name, "systematic_verified", False), smiles


def test_the_declared_list_is_the_set_of_rows_the_rule_moves():
    """``declared_demotions_<START>.tsv`` (constructed molecules) lists, row by row, the rows of the
    start-commit baselines whose label the rule moves out of pin_verified: exactly the rows for which
    the predicate holds, at the default tier (it declines), at the valid tier and at the best-effort tier
    (``systematic_verified``). Nothing else is declared, and no row of the other-metals panel is."""
    declared_path = baseline_path("zinc").parent / "declared_demotions_75b4ba386.tsv"
    with open(declared_path, newline="") as f:
        declared = {(r["smiles"], r["tier_req"]): r["to_tier"] for r in csv.DictReader(f, delimiter="\t")}
    expected = {}
    for kind in ("zinc", "other"):
        for row in read_rows(baseline_path(kind)):
            if row["tier"] in ("pin_verified", "pin_unverified") and PREDICATE(row["smiles"], row["name"]):
                expected[(row["smiles"], row["tier_req"])] = "abstain" if row["tier_req"] == "pin" else "systematic_verified"
                if row["tier_req"] == "pin":                    # the valid tier names the same salt as the default tier
                    expected[(row["smiles"], "valid")] = "systematic_verified"
    assert declared == expected
    assert declared, "the panel holds the ionic diethylzinc and diphenylzinc"
    assert not [k for k in declared if k in {(r["smiles"], r["tier_req"]) for r in read_rows(baseline_path("other"))}]


# --------------------------------------------------------------------------- ocenes and eta-hapto names

OCENE_AND_CYMANTRENE = [
    # smiles, (default name, tier, is_pin), (best-effort name, tier, is_pin): measured at the start commit
    ("[cH-]1cccc1.[cH-]1cccc1.[Fe+2]",
     ("ferrocene", "pin_verified", True), ("ferrocene", "pin_verified", True)),
    ("[Ru+2].c1cc[cH-]c1.c1cc[cH-]c1",
     ("ruthenocene", "pin_verified", True), ("ruthenocene", "pin_verified", True)),
    ("Cc1c(C)c(C)[c-](C)c1C.Cc1c(C)c(C)[c-](C)c1C.[Fe+2]",
     ("decamethylferrocene", "pin_verified", True), ("decamethylferrocene", "pin_verified", True)),
    ("[Mn+].c1cc[cH-]c1.[C-]#[O+].[C-]#[O+].[C-]#[O+]",
     ("tricarbonyl(η⁵-cyclopentadienyl)manganese", "pin_unverified", False),
     (f"cyclopenta-2,4-dien-1-ide{EM_DASH}1-oxaeth-1-yn-1-ium-2-ide{EM_DASH}manganese(1+) (1/3/1)",
      "systematic_verified", False)),
]


@pytest.mark.parametrize("smiles, default, best_effort", OCENE_AND_CYMANTRENE)
def test_ionic_ocenes_and_cymantrene_keep_their_start_label_and_tier(smiles, default, best_effort):
    """ (:39735) "except for 'ocene' compounds" and "Ocenes" (:40129; the examples
    carry "(PIN)",:40133): the ocene names keep PIN status. The eta-hapto name of cymantrene is a
    target row of the PIN gate and keeps its label. Each holds a Group 3-12 metal, no metal-carbon
    bond and a carbanion carbon with no positive neighbour, so only clauses (e) and (f) of the
    predicate keep it out of the rule."""
    for tier, expected in (("pin", default), ("best-effort", best_effort)):
        row = _row(smiles, tier)
        assert (row["name"], row["tier"], row["is_pin"]) == expected, (smiles, tier)


BB_OCENE_ROWS = [   # Ocenes, the Blue Book,:40135,:40139: the structure and its (PIN)
    ("CC(=O)[c-]1cccc1.CC(=O)[c-]1cccc1.[Fe+2]", "1,1'-(ferrocene-1,1'-diyl)di(ethan-1-one)"),
    ("OCC[c-]1cccc1.[Os+2].c1cc[cH-]c1", "2-(osmocen-1-yl)ethanol"),
    ("C[c-]1cccc1.C[c-]1cccc1.[Ru+2].[Ru+2].c1cc[c-](CC[c-]2cccc2)c1", "1,1''-(ethane-1,2-diyl)bis(1'-methylruthenocene)"),
]


def test_the_predicate_excludes_ocene_and_eta_hapto_names():
    """The function called directly: False for the four molecules with their names and for the
    three ionic ocene rows of the Blue Book with their (PIN) names; True for the salt of a carbanion
    of zinc, so the exclusions are not a blanket False."""
    for smiles, default, best_effort in OCENE_AND_CYMANTRENE:
        assert PREDICATE(smiles, default[0]) is False, default[0]
        if default[0] == best_effort[0]:
            assert PREDICATE(smiles, best_effort[0]) is False, best_effort[0]
    # The best-effort name of cymantrene is an adduct name (em dash): no eta descriptor, so the predicate
    # holds for it. It moves nothing: the row is systematic_verified already (``_is_metal_adduct_without_pin``,
    #, and the demotion reads pin_verified and pin_unverified rows only.
    cymantrene = OCENE_AND_CYMANTRENE[-1]
    assert PREDICATE(cymantrene[0], cymantrene[2][0]) is True
    assert cymantrene[2][1:] == ("systematic_verified", False)
    for smiles, name in BB_OCENE_ROWS:
        assert PREDICATE(smiles, name) is False, name
    assert PREDICATE("[CH2-]C.[CH2-]C.[Zn+2]", "zinc bis(ethan-1-ide)") is True
    # a name of the organometallic assembler with an eta descriptor is outside the rule (clause (f))
    assert PREDICATE("[Mn+].c1cc[cH-]c1.[C-]#[O+].[C-]#[O+].[C-]#[O+]",
                     "tricarbonyl(η⁵-cyclopentadienyl)manganese") is False


@pytest.mark.parametrize("smiles, name, expected", [
    ("[CH2-]C.[CH2-]C.[Zn+2]", "zinc bis(ethan-1-ide)", True),            # Group 12
    ("[c-]1ccccc1.[c-]1ccccc1.[Cu+2]", "copper(II) benzenide", True),     # Group 11
    ("[c-]1ccccc1.[c-]1ccccc1.[c-]1ccccc1.[Sc+3]", "scandium tribenzenide", True),     # Group 3
    ("[c-]1ccccc1.[c-]1ccccc1.[c-]1ccccc1.[La+3]", "lanthanum tribenzenide", True),    # a lanthanide
    ("[Na+].C[CH2-]", "sodium ethan-1-ide", False),                       # Group 1: its own rule
    ("[Ca+2].[CH3-].[CH3-]", "calcium bis(methanide)", False),            # Group 2
    ("[Sn+2].[CH3-].[CH3-]", "tin dimethanide", False),                   # Group 14:
    ("CC[Zn]CC", "diethylzinc", False),                                   # a metal-carbon bond: the other predicate
    ("[C-]#N.[C-]#N.[Zn+2]", "zinc dicyanide", False),                    # cyanide: the anion of an acid
    ("CC(=O)[O-].CC(=O)[O-].[Zn+2]", "zinc diacetate", False),            # no carbanion
    ("[Zn+2].[H-].[H-]", "zinc dihydride", False),                        # a hydride is no carbanion carbon
    ("C[N+](C)(C)[CH-]C(=O)[O-].[Zn+2]", "x", False),                      # a carbon with a positive neighbour (ylide)
    ("[Cl-].[Cl-].[Zn+2]", "zinc dichloride", False),
    ("", "x", False),
    ("not a smiles", "x", False),
])
def test_the_predicate_clauses(smiles, name, expected):
    assert PREDICATE(smiles, name) is expected


def test_the_coordination_list_names_are_not_demoted():
    """Clause (d): a name of the exact-structure list keeps the label of its path (user decision
    2026-09-30). The predicate is False for a name that ``_coordination_list_name`` accepts."""
    import orthonym.namer as nm
    original = nm._coordination_list_name
    try:
        nm._coordination_list_name = lambda smiles, name: True
        assert PREDICATE("[CH2-]C.[CH2-]C.[Zn+2]", "zinc bis(ethan-1-ide)") is False
    finally:
        nm._coordination_list_name = original
    assert PREDICATE("[CH2-]C.[CH2-]C.[Zn+2]", "zinc bis(ethan-1-ide)") is True


# --------------------------------------------------------------------------- other metals

def _salt_rows_of_other_metals():
    """The rows of the other-metals baseline with several fragments and a true metal other than zinc:
    the rows where the scope function runs for a metal the table does not admit."""
    from orthonym.perception.metals import _TRUE_METAL_SYMBOLS_FOR_VETO
    rows = []
    for row in read_rows(baseline_path("other")):
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None or "." not in row["smiles"]:
            continue
        if any(a.GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO and a.GetSymbol() != "Zn" for a in mol.GetAtoms()):
            rows.append(row)
    return rows


def test_other_metals_are_not_moved():            # reads the baseline fixture (constructed molecules)
    rows = _salt_rows_of_other_metals()
    assert len(rows) >= 30, len(rows)
    declared = baseline_path("zinc").parent / "declared_demotions_75b4ba386.tsv"
    with open(declared, newline="") as f:
        declared_keys = {(r["smiles"], r["tier_req"]) for r in csv.DictReader(f, delimiter="\t")}
    # The salts of an acid, an alcohol or a thiol take the element name of the cation and the multiplied anion, with no
    # oxidation number: Salts (the Blue Book), 'germanium tetraacetate (PIN)' (:31581); (:28178),
    # '(C6H5-O-)4 Pb4+ lead tetraphenoxide (PIN)' (:28196). The rows below are respelled so, with the label they had
    # (``declared_respellings_pinc.tsv``): the rule is the book's, not a change of the metals the table admits.
    with open(baseline_path("zinc").parent / "declared_respellings_pinc.tsv", newline="") as f:
        respelled = {(r["smiles"], r["tier_req"]): (r["from_name"], r["to_name"])
                     for r in csv.DictReader(f, delimiter="\t")}
    # The neutral metal atoms beside ions that the table admits to ZERO_VALENT (user decision 2026-10-30) are named
    # at the best-effort tier with the oxidation number zero ('iridium(0)'): the rows below gain that name.
    with open(baseline_path("zinc").parent / "declared_gains_zero_valent.tsv", newline="") as f:
        gains = {(r["smiles"], r["tier_req"]): (r["to_name"], r["to_tier"]) for r in csv.DictReader(f, delimiter="\t")}
    # The metals admitted to every mechanism since 2026-10-30 (``BONDED_METAL_VALENCES``) move as declared there.
    bonded_moves = declared_bonded_metal_moves()
    moved = []
    namers = {tier: Orthonym(style="pin", **_emit_tier_flags(tier)) for tier in ("pin", "best-effort")}
    for row in rows:
        if (row["smiles"], row["tier_req"]) in declared_keys:
            continue
        if (row["smiles"], row["tier_req"]) in bonded_moves:
            now = namers[row["tier_req"]].name_tiered(row["smiles"])
            assert identity_view(now) == bonded_moves[(row["smiles"], row["tier_req"])], row["smiles"]
            if now["name"] and now["tier"] != "abstain":
                assert_full_rt(now["name"], row["smiles"])
            continue
        if (row["smiles"], row["tier_req"]) in gains:
            now = namers[row["tier_req"]].name_tiered(row["smiles"])
            assert (now["name"], now["tier"]) == gains[(row["smiles"], row["tier_req"])], row["smiles"]
            assert_full_rt(now["name"], row["smiles"])
            continue
        now = namers[row["tier_req"]].name_tiered(row["smiles"])
        key = (row["smiles"], row["tier_req"])
        if key in respelled:
            assert respelled[key][0] == row["name"], key
            assert identity_view(now) == (respelled[key][1],) + tuple(identity_view(row))[1:], (key, identity_view(now))
            continue
        if identity_view(now) != identity_view(row):
            moved.append((row["smiles"][:60], row["tier_req"], identity_view(row), identity_view(now)))
    assert moved == []
