""" Milestone B3 -- charged-acyclic FG + inorganic classification.

Covers four trace-pinpointed fixes (`internal notes`):

1. Inorganic-fragment honesty floor (``errors.py::classify_failure_limit``):
   a carbon-free bare ion must abstain to the honest "... (not supported)"
   message, never the "unknown organic compound" sentinel.
2. Unblocking the already-correct inorganic oxoanion namer: a carbon-free
   whole-molecule guard in ``perception/ions.py::_get_internal_charge_atoms``
   (nitrate's own charge centres were mis-marked "internal"), plus a
   ``name_anion`` fallback wired into ``routing/dispatch_table.py``'s
   ``_handle_poly_anion`` and ``_is_mixed_sign_zwitterion`` /
   ``_handle_mixed_sign_zwitterion`` (the mixed +/- single-ion oxoanion
   shape -- chlorite, nitrate), plus an ``'ite'`` gap fix in
   ``perception/structure_conservation.py``'s ionic-suffix recognizer
   (chlorite's hypervalent-halogen charge-conservation veto was rejecting
   the correct name because "-ite" was missing from the suffix regex).
3. Inorganic retained-name table fixes/additions (``data/ion_retained_names.py``):
   chlorite/chlorate swap, a malformed nitrite key, bisulfite (as
   ``hydrogen sulfite``), phosphonate dianion.
4. Substitutive nitramide/N-nitro producer (``rules/nitramide.py``), wired
   into dispatch as ``NITRAMIDE_SUBSTITUTED``.

Every target name below is OPSIN-round-trip-verified (full InChIKey match)
by the implementing session.
"""
import pytest

from orthonym import Orthonym
from orthonym.errors import is_failure_name
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
    "O=[N+]([O-])NCN[N+](=O)[O-]",
    "OCC(O[N+](=O)[O-])CO[N+](=O)[O-]",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_obj_name(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES:
        return _declined_pin_row(smiles)["name"]
    return namer_obj.name(smiles)


def _dt_obj_row(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES:
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



@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# ---------------------------------------------------------------------------
# Task 1: inorganic-fragment honesty floor
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate  # needs the REAL OPSIN validity gate enabled: with it
# off, the raw retained-table hit ('sulfide') is returned unsuppressed and
# never reaches classify_failure_limit at all -- this test is about what the
# gate does to an OPSIN-unparseable-standing-alone retained word, not about
# the retained lookup itself.
@pytest.mark.parametrize("smi", [
    # "[S-2]" left this list in breadth job 3: the bare ion is 'sulfanediide' by the
    # parent-hydride rule (the Blue Book-40902), asserted in
    # tests/unit/rules/test_breadth3_m15_small_ions.py; the salt word 'sulfide'
    # still names no bare ion
    "[H+]",              # bare proton
    "O=[SH][O-]",        # HO2S- -- no retained-table row (not the same species as bisulfite)
])
def test_carbon_free_ion_honest_abstain(namer, smi):
    out = _dt_obj_name(namer, smi)
    assert out != "unknown organic compound", (
        f"{smi} should abstain honestly, not the organic sentinel: {out!r}")
    assert "not supported" in out
    assert is_failure_name(out)


@pytest.mark.opsin_gate  # same reasoning as above: needs the real gate on
def test_organic_unnameable_still_uses_organic_sentinel(namer):
    # A carbon-BEARING molecule that fails to name must still use the
    # pre-existing organic sentinel -- Task 1 must not over-broaden.
    out = _dt_obj_name(namer, "CCCCCCCCCCCCOP(=S)([O-])[O-]")
    assert out == "unknown organic compound"


# ---------------------------------------------------------------------------
# Task 2 + 3: unblocked inorganic oxoanion naming (dispatch + SMARTS + data)
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("O=[N+]([O-])[O-]", "nitrate"),
    ("O=S([O-])[O-]", "sulfite"),
    ("[O-][Cl+][O-]", "chlorite"),          # was wrongly 'chlorate' in the table
    ("[O-]N=O", "nitrite"),                 # old table key was a malformed N+ form
    ("O=S([O-])O", "hydrogen sulfite"),     # genuine missing row; two words
    ("O=[PH]([O-])[O-]", "phosphonate"),    # genuine missing row
])
def test_inorganic_oxoanion_names(namer, smi, expected):
    assert _dt_obj_name(namer, smi) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("C[N+](=O)[O-]", "nitromethane"),
    ("c1ccccc1[N+](=O)[O-]", "nitrobenzene"),
    # alkyl nitrate ester: (the Blue Book-35918, 'Esters of
    # mononuclear noncarbon oxoacids'), "Alkyl groups, aryl groups, etc. are cited
    # as separate words... followed by the name of the appropriate anion";
    # sibling 'pentyl nitrite (PIN)' (:35922)
    ("CCO[N+](=O)[O-]", "ethyl nitrate"),
    ("OCC(O[N+](=O)[O-])CO[N+](=O)[O-]", "2,3-bis(nitrooxy)propan-1-ol"),
])
def test_organic_nitro_regression(namer, smi, expected):
    # Task 2's carbon-free guard + SMARTS narrowing must NEVER touch a real
    # organic nitro compound or a nitrate-ester substituent.
    assert _dt_obj_name(namer, smi) == expected


@pytest.mark.unit
def test_looks_like_ionic_name_recognises_ite_suffix():
    from orthonym.perception.structure_conservation import looks_like_ionic_name
    assert looks_like_ionic_name("chlorite") is True
    assert looks_like_ionic_name("nitrite") is True
    assert looks_like_ionic_name("sulfite") is True


# ---------------------------------------------------------------------------
# Task 4: substitutive nitramide / N-nitro producer
# ---------------------------------------------------------------------------

# a phase (task 11C1, / the Blue Book "(chloromethyl)(methyl)nitramide
# (PIN)"): the amide N of the `nitramide` functional parent is its ONLY
# substitutable position, so the N-locant is OMITTED and the
# substituent prefixes are enclosed per the mononuclear single-attachment rule
# (the Blue Book). The `N,N'-dinitromethanediamine` row keeps its locants
# -- it is the TWO-different-amide-nitrogens shape, where they disambiguate.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("O=[N+]([O-])NCO", "(hydroxymethyl)nitramide"),
    ("O=[N+]([O-])N(CO)CO", "bis(hydroxymethyl)nitramide"),
    ("CN(CCl)[N+](=O)[O-]", "(chloromethyl)(methyl)nitramide"),
    ("O=[N+]([O-])NCN[N+](=O)[O-]", "N,N'-dinitromethanediamine"),
])
def test_substituted_nitramide_names(namer, smi, expected):
    assert _dt_obj_name(namer, smi) == expected


@pytest.mark.unit
def test_plain_nitramide_unchanged(namer):
    # The exact-whole-molecule retained-table hit must still win (the new
    # producer declines on 0 substituents).
    assert _dt_obj_name(namer, "O=[N+]([O-])N") == "nitramide"


@pytest.mark.unit
def test_nitramide_producer_declines_outside_its_class():
    from rdkit import Chem
    from orthonym.rules.nitramide import name_substituted_nitramide

    for smi in [
        "C[N+](=O)[O-]",          # nitromethane -- nitro on carbon, not nitrogen
        "c1ccccc1[N+](=O)[O-]",   # nitrobenzene
        "CCO[N+](=O)[O-]",        # nitrate ester (O-attached), not an N-amide
        "O=[N+]([O-])N",          # plain nitramide -- owned by the retained table
    ]:
        mol = Chem.MolFromSmiles(smi)
        assert name_substituted_nitramide(mol) is None


# ---------------------------------------------------------------------------
# Full round-trip verification (real OPSIN 2.9.0)
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
def test_all_targets_round_trip_exact():
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

    pairs = [
        ("O=[N+]([O-])[O-]", "nitrate"),
        ("O=S([O-])[O-]", "sulfite"),
        ("[O-][Cl+][O-]", "chlorite"),
        ("[O-]N=O", "nitrite"),
        ("O=S([O-])O", "hydrogen sulfite"),
        ("O=[PH]([O-])[O-]", "phosphonate"),
        ("C[N+](=O)[O-]", "nitromethane"),
        ("c1ccccc1[N+](=O)[O-]", "nitrobenzene"),
        ("CCO[N+](=O)[O-]", "nitrooxyethane"),
        ("O=[N+]([O-])NCO", "(hydroxymethyl)nitramide"),
        ("O=[N+]([O-])N(CO)CO", "bis(hydroxymethyl)nitramide"),
        ("CN(CCl)[N+](=O)[O-]", "(chloromethyl)(methyl)nitramide"),
        ("O=[N+]([O-])NCN[N+](=O)[O-]", "N,N'-dinitromethanediamine"),
    ]
    for smi, name in pairs:
        result = opsin_roundtrip_check(smi, name)
        assert result["passed"], f"{name!r} vs {smi!r}: {result}"
