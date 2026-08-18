"""v33 Phase 6: diacylglycerol-shape ester-vs-hydroxy parent-selection fix.

BlueBookV2.md P-41 Table 4.1 "General compound classes listed in decreasing
order of seniority" (:18158+): class 9 Esters -- "functional class names are
given to noncyclic esters" -- outranks class 17 Hydroxy compounds. A
partially-esterified acyclic polyol carrying >=2 DIFFERENT noncyclic esters
plus a free -OH (the diacylglycerol shape) must therefore name by the SENIOR
ester (functional-class '<yl> <acid>oate'), with the junior ester demoted to
an 'acyloxy' prefix and the free -OH to 'hydroxy' -- never by the junior
'-ol' suffix with BOTH esters demoted.

Bug + derivation: .
Fix site: `rules/esters.py::name_polyfunctional_diester_free_hydroxy`, wired
into `rules/polyfunctional.py::name_polyfunctional` just before the legacy
EL-02 ester-demotion fallback.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse

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
    """The senior (C18, more carbons -- P-44.3) acid stays the '...oate'
    parent; the junior C15 acid demotes to an acyloxy prefix; the free -OH
    is a hydroxy prefix. Before the fix this named
    '(2S)-2-[(9Z)-octadec-9-enoyloxy]-1-[(9Z)-pentadec-9-enoyloxy]propan-3-ol'
    -- the junior hydroxy class promoted to principal, inverting P-41."""
    assert inchi.MolToInchiKey(Chem.MolFromSmiles(WITNESS)) == WITNESS_INCHIKEY

    name = Orthonym().name(WITNESS)
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

    name = Orthonym().name(alt_smiles)
    assert name == WITNESS_NAME, name


def test_1_3_diacylglycerol_different_acyls_names_by_senior_ester():
    """A 1,3-diacylglycerol with two DIFFERENT acyl groups (acetate C2 +
    hexanoate C6) and undefined stereo at the (real, but input-undefined)
    C2 centre. The senior (hexanoate) acid is the parent; acetate demotes to
    an acyloxy prefix; the free -OH is a hydroxy prefix. This exercises the
    TERMINAL-attachment (k=1, plain 'propyl', no 'an-k-yl' locant) branch of
    the new numbering, complementing the witness's non-terminal (k=2,
    'propan-2-yl') branch -- P-65.6.3.3.4.2's own worked example
    ('2-(acetyloxy)ethyl methyl butanedioate (PIN)') is exactly this shape."""
    smi = "CC(=O)OCC(O)COC(=O)CCCCC"
    name = Orthonym().name(smi)
    assert name == "3-(acetyloxy)-2-hydroxypropyl hexanoate", name
    assert _rt_matches(smi, name), name


def test_1_3_diacylglycerol_identical_acyls_stays_multiplicative_diyl_form():
    """Regression lock, NOT a target of this fix: two IDENTICAL acyl groups
    (hexanoate + hexanoate) + a free -OH is P-65.6.3.3.4.1's own symmetric
    multiplicative form ('di<acid>oate' on a decorated diyl group), produced
    by the pre-existing `rules.lipids.name_lipid` path -- it never reaches
    `name_polyfunctional_diester_free_hydroxy` at all (that function
    correctly declines on the tied acid length). Locks that this molecule's
    name did not change as a side effect of wiring the new path into
    `name_polyfunctional`."""
    smi = "CCCCCC(=O)OCC(O)COC(=O)CCCCC"
    name = Orthonym().name(smi)
    assert name == "2-hydroxypropane-1,3-diyl dihexanoate", name
    assert _rt_matches(smi, name), name


class TestRegressionLockAlreadyCorrectCases:
    """The 4 already-correct ester(+hydroxy) shapes cited in
    PHASE6-DIESTER-OL-SENIORITY.md as 'must stay byte-identical' -- none of
    these has >=2 DIFFERENT noncyclic esters, so none is in this fix's
    scope, but all pass through the SAME `name_polyfunctional` function the
    fix edited."""

    def test_monoester_monool_fragment(self):
        assert Orthonym().name("CC(=O)OCCO") == "2-hydroxyethyl ethanoate"

    def test_monoester_free_diol(self):
        assert Orthonym().name("CC(=O)OCC(O)CO") == "2,3-dihydroxypropyl acetate"

    def test_monoacylglycerol(self):
        assert Orthonym().name("CCCCCCCCCCCCCCCC(=O)OCC(O)CO") == \
            "2,3-dihydroxypropyl hexadecanoate"

    def test_symmetric_diester_no_free_hydroxyl(self):
        assert Orthonym().name("CC(=O)OCCOC(C)=O") == "ethane-1,2-diyl diacetate"
