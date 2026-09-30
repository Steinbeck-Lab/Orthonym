""" a phase: diacylglycerol-shape ester-vs-hydroxy parent-selection fix.

the Blue Book "General compound classes listed in decreasing
order of seniority" (:18158+): class 9 Esters -- "functional class names are
given to noncyclic esters" -- outranks class 17 Hydroxy compounds. A
partially-esterified acyclic polyol carrying >=2 DIFFERENT noncyclic esters
plus a free -OH (the diacylglycerol shape) must therefore name by the SENIOR
ester (functional-class '<yl> <acid>oate'), with the junior ester demoted to
an 'acyloxy' prefix and the free -OH to 'hydroxy' -- never by the junior
'-ol' suffix with BOTH esters demoted.

Bug + derivation: internal notes.
Fix site: `rules/esters.py::name_polyfunctional_diester_free_hydroxy`, wired
into `rules/polyfunctional.py::name_polyfunctional` just before the legacy
 ester-demotion fallback.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse
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
    "C(CCCCCCC/C=C\\CCCCC)(=O)OC[C@@H](OC(CCCCCCC/C=C\\CCCCCCCC)=O)CO",
    "CC(=O)OCC(O)COC(=O)CCCCC",
    "CCCCC/C=C\\CCCCCCCC(=O)OC[C@H](CO)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


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


pytestmark = pytest.mark.opsin_gate


def _rt_matches(smiles: str, name: str) -> bool:
    """True iff OPSIN parses `name` back to the SAME molecule as `smiles`
    (canonical InChIKey match) -- the project's 0-wrong oracle."""
    opsin_smi = opsin_parse(name)
    if not opsin_smi:
        return False
    a = Chem.MolFromSmiles(smiles)
    b = Chem.MolFromSmiles(opsin_smi)
    if a is None or b is None:
        return False
    return inchi.MolToInchiKey(a) == inchi.MolToInchiKey(b)


# The user-found witness: a 1,2-diacylglycerol with two DIFFERENT unsaturated
# acyl groups (C15 pentadecenoyl + C18 octadecenoyl) plus a free -OH.
WITNESS = "CCCCC/C=C\\CCCCCCCC(=O)OC[C@H](CO)OC(=O)CCCCCCC/C=C\\CCCCCCCC"
WITNESS_NAME = (
    "(2S)-1-hydroxy-3-[(9Z)-pentadec-9-enoyloxy]propan-2-yl "
    "(9Z)-octadec-9-enoate"
)
WITNESS_INCHIKEY = "OALWFBDEUFFROY-KBMTVBSSSA-N"


def test_diacylglycerol_unsaturated_asymmetric_names_by_senior_ester():
    """The senior (C18, more carbons -- acid stays the '...oate'
    parent; the junior C15 acid demotes to an acyloxy prefix; the free -OH
    is a hydroxy prefix. Before the fix this named
    '(2S)-2-[(9Z)-octadec-9-enoyloxy]-1-[(9Z)-pentadec-9-enoyloxy]propan-3-ol'
    -- the junior hydroxy class promoted to principal, inverting."""
    assert inchi.MolToInchiKey(Chem.MolFromSmiles(WITNESS)) == WITNESS_INCHIKEY

    name = _dt_name(WITNESS)
    assert name == WITNESS_NAME, name
    assert _rt_matches(WITNESS, name), name


def test_diacylglycerol_order_invariant():
    """The identical molecule, atoms visited in a different SMILES order,
    must name IDENTICALLY -- the fix must not depend on which ester the
    perception layer happens to match first (or the atom-index accident of
    which alkyl attachment atom is 'first')."""
    alt_smiles = (
        "C(CCCCCCC/C=C\\CCCCC)(=O)OC[C@@H]"
        "(OC(CCCCCCC/C=C\\CCCCCCCC)=O)CO"
    )
    assert inchi.MolToInchiKey(Chem.MolFromSmiles(alt_smiles)) == WITNESS_INCHIKEY

    name = _dt_name(alt_smiles)
    assert name == WITNESS_NAME, name


def test_1_3_diacylglycerol_different_acyls_names_by_senior_ester():
    """A 1,3-diacylglycerol with two DIFFERENT acyl groups (acetate C2 +
    hexanoate C6) and undefined stereo at the (real, but input-undefined)
    C2 centre. The senior (hexanoate) acid is the parent; acetate demotes to
    an acyloxy prefix; the free -OH is a hydroxy prefix. This exercises the
    TERMINAL-attachment (k=1, plain 'propyl', no 'an-k-yl' locant) branch of
    the new numbering, complementing the witness's non-terminal (k=2,
    'propan-2-yl') branch -- 's own worked example
    ('2-(acetyloxy)ethyl methyl butanedioate (PIN)') is exactly this shape."""
    smi = "CC(=O)OCC(O)COC(=O)CCCCC"
    name = _dt_name(smi)
    assert name == "3-(acetyloxy)-2-hydroxypropyl hexanoate", name
    assert _rt_matches(smi, name), name


def test_1_3_diacylglycerol_identical_acyls_stays_multiplicative_diyl_form():
    """Regression lock, NOT a target of this fix: two IDENTICAL acyl groups
    (hexanoate + hexanoate) + a free -OH is 's own symmetric
    multiplicative form ('di<acid>oate' on a decorated diyl group), produced
    by the pre-existing `rules.lipids.name_lipid` path -- it never reaches
    `name_polyfunctional_diester_free_hydroxy` at all (that function
    correctly declines on the tied acid length). Locks that this molecule's
    name did not change as a side effect of wiring the new path into
    `name_polyfunctional`."""
    smi = "CCCCCC(=O)OCC(O)COC(=O)CCCCC"
    name = _dt_name(smi)
    assert name == "2-hydroxypropane-1,3-diyl dihexanoate", name
    assert _rt_matches(smi, name), name


class TestRegressionLockAlreadyCorrectCases:
    """The 4 already-correct ester(+hydroxy) shapes cited in
    PHASE6-DIESTER-OL-SENIORITY.md as 'must stay byte-identical' -- none of
    these has >=2 DIFFERENT noncyclic esters, so none is in this fix's
    scope, but all pass through the SAME `name_polyfunctional` function the
    fix edited."""

    def test_monoester_monool_fragment(self):
        # PIN per R5: "acetic acid (PIN)" (the Blue Book); an ester
        # takes the acid's PIN anion word, "ethyl acetate (PIN)" (:31667) -- the same
        # word as test_monoester_free_diol below. OPSIN RT exact. (Was 'ethanoate':
        # the polyfunctional ester path named the acid analog without the retained
        # name.)
        assert Orthonym().name("CC(=O)OCCO") == "2-hydroxyethyl acetate"

    def test_monoester_free_diol(self):
        assert Orthonym().name("CC(=O)OCC(O)CO") == "2,3-dihydroxypropyl acetate"

    def test_monoacylglycerol(self):
        assert Orthonym().name("CCCCCCCCCCCCCCCC(=O)OCC(O)CO") == \
            "2,3-dihydroxypropyl hexadecanoate"

    def test_symmetric_diester_no_free_hydroxyl(self):
        assert Orthonym().name("CC(=O)OCCOC(C)=O") == "ethane-1,2-diyl diacetate"
