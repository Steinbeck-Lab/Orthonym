"""Wave2 — substitutive sulfoxide/sulfone PINs + conservation.

Functional-class 'dimethyl sulfoxide' style names are general nomenclature
only; the PIN is substitutive with sulfinyl/sulfonyl prefixes built on the
ACID STEM ('methanesulfinyl', from methanesulfinic acid), the senior R as
parent, and the multiplicative form for symmetric diaryls (BB:
'(methanesulfinyl)methane' 46154, '1-(ethanesulfinyl)butane' 28094,
"(ethanesulfonyl)ethane" + "1,1'-sulfinyldibenzene" 28110-28115;
'Multiplication of acyclic hydrocarbons is not permitted').

Mechanism: sulfoxide/sulfone joined _PREFIX_ONLY_PRINCIPAL (no suffix form
exists — claiming the PCG starved the polyfunctional path); the dedicated
handlers fire on FG-present + no-PCG, decline when the R-SOx-R' unit does
not cover the whole molecule (the old functional-class alkyl walk silently
dropped atoms beyond a heteroatom: 'CSCCS(=O)C' -> 'ethyl methyl sulfoxide'
with -S-CH3 lost), and build the substitutive PIN through the strict
_classify_oxide_side conservation classifier (linear terminal saturated
all-C chain / plain benzene / plain cycloalkane — else fall back or fail).
"""

import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import name_compound


@pytest.mark.unit
class TestSubstitutivePins:
    @pytest.mark.parametrize("smiles,expected", [
        ("CS(=O)C", "(methanesulfinyl)methane"),          # BB 46154 verbatim
        ("CS(=O)(=O)C", "(methanesulfonyl)methane"),
        ("CCS(=O)CCCC", "1-(ethanesulfinyl)butane"),      # BB 28094 verbatim
        ("CCS(=O)(=O)CC", "(ethanesulfonyl)ethane"),      # BB 28115 verbatim
        ("CS(=O)c1ccccc1", "(methanesulfinyl)benzene"),
        ("CS(=O)C1CCCCC1", "(methanesulfinyl)cyclohexane"),
    ])
    def test_substitutive(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("O=S(c1ccccc1)c1ccccc1", "1,1'-sulfinyldibenzene"),    # BB 28110
        ("O=S(=O)(c1ccccc1)c1ccccc1", "1,1'-sulfonyldibenzene"),
    ])
    def test_symmetric_diaryl_multiplicative(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestPolyfunctionalAcidStemPrefix:
    """Non-principal sulfoxide/sulfone: acid-stem prefix with enclosing
    marks (BB 28150 verbatim analog '2-(methanesulfonyl)ethan-1-ol')."""

    @pytest.mark.parametrize("smiles,expected", [
        ("CS(=O)CCO", "2-(methanesulfinyl)ethan-1-ol"),
        ("OCCS(C)(=O)=O", "2-(methanesulfonyl)ethan-1-ol"),
        # CORRECTED 2026-08-02. Asserted `2-(methanesulfinyl)ethanoic acid`
        # until now -- an over-generalisation from the `ethan-1-ol` /
        # `ethan-1-amine` siblings above. Acetic acid does NOT behave like
        # ethanol here: it is a RETAINED name that keeps its retained form
        # under substitution AND drops the locant.
        # the Blue Book `CH3-COOH acetic acid (PIN) ethanoic acid`
        # -- `ethanoic acid` is the non-PIN alternative.
        # the Blue Book `difluoroacetic acid (PIN) (not 2,2-difluoroacetic acid)`
        # -- the substituted form keeps `acetic acid` and the
        # locant is explicitly marked "not".
        # the Blue Book `sulfanylacetic acid (PIN)` -- a SULFUR substituent on
        # acetic acid, the direct analogue of this row.
        # the Blue Book `(1H-indol-1-yl)acetic acid (PIN)`,.
        # The old value was thus non-PIN on both counts. Both spellings parse
        # to the same molecule under OPSIN, so this is a spelling correction,
        # not a structural one.
        ("CS(=O)CC(=O)O", "(methanesulfinyl)acetic acid"),
        ("CS(=O)CCN", "2-(methanesulfinyl)ethan-1-amine"),
    ])
    def test_prefix_form(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.fixture
def _validity_gate_on(monkeypatch):
    """The suite's autouse fixture disables the OPSIN validity gate so RAW
    output is asserted. The conservation cases below are ABOUT the gate
    suppressing a wrong RAW name to 'unknown' in production — re-enable it
    (the test_opsin_validity_gate.py pattern)."""
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
    yield


@pytest.mark.unit
class TestConservation:
    def test_beyond_unit_atoms_are_named_not_dropped(self, _validity_gate_on):
        """The atoms beyond the sulfoxide unit survive — now positively.

        History: the old functional-class walk emitted `ethyl methyl sulfoxide`
        for this, silently losing the `-S-CH3` — a DIFFERENT molecule. made
        it fail closed (`unknown organic compound`) and this test asserted that
        abstention, recording the real PIN as a deferral in its own comment:
        "the PIN 1-(methanesulfinyl)-2-(methylsulfanyl)ethane (BB 18284) needs
        the thioether principal-group demotion — documented deferral".

        CORRECTED 2026-08-02: that deferral has since closed and the producer
        now emits exactly the PIN the comment named. Verified verbatim at
        ``the Blue Book``::

            CH3-S-CH2-CH2-SO-CH3
            1-(methanesulfinyl)-2-(methylsulfanyl)ethane (PIN)
            (C > sulfide and sulfoxide)

        The conservation intent is unchanged and is now asserted POSITIVELY —
        naming every atom is a strictly stronger guarantee than declining to
        name any. Confirmed pre-existing (identical at the session-start commit
        ), so this is a stale expectation, not a regression.
        """
        assert name_compound("CSCCS(=O)C") == (
            "1-(methanesulfinyl)-2-(methylsulfanyl)ethane"
        )

    def test_benzyl_side_names_substitutively(self, _validity_gate_on):
        # A benzyl side was "not an honestly-nameable FC side" and this pinned
        # the abstention, recording "substitutive needs the benzene-path S-branch
        # namer (deferred)". That deferral has since closed: the benzene parent
        # now names the -CH2-S(=O)(=O)-CH3 substituent substitutively as
        # ``[(methanesulfonyl)methyl]benzene``. Confirmed pre-existing (identical
        # at HEAD via scripts/an A/B check), so this is a stale expectation, not a
        # regression. OPSIN round-trips to the input InChIKey (change-asserted-
        # value verified).
        name = name_compound("CS(=O)(=O)Cc1ccccc1")
        assert name == "[(methanesulfonyl)methyl]benzene"
        from orthonym.validation.opsin_roundtrip import opsin_parse
        got = opsin_parse(name)
        assert got and (inchi.MolToInchiKey(Chem.MolFromSmiles(got))
                        == inchi.MolToInchiKey(Chem.MolFromSmiles(
                            "CS(=O)(=O)Cc1ccccc1")))

    def test_classifier_shapes(self):
        from orthonym.rules.sulfur import _classify_oxide_side
        mol = Chem.MolFromSmiles("CCS(=O)CCCC")
        s = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'S')
        sides = sorted(
            _classify_oxide_side(mol, n.GetIdx(), s)[0]
            for n in mol.GetAtomWithIdx(s).GetNeighbors()
            if n.GetSymbol() == 'C'
        )
        assert sides == ['butane', 'ethane']
        # branched side -> None
        mol2 = Chem.MolFromSmiles("CC(C)S(=O)C")
        s2 = next(a.GetIdx() for a in mol2.GetAtoms() if a.GetSymbol() == 'S')
        branched = next(
            n.GetIdx() for n in mol2.GetAtomWithIdx(s2).GetNeighbors()
            if n.GetSymbol() == 'C' and n.GetDegree() > 1
        )
        assert _classify_oxide_side(mol2, branched, s2) is None

    def test_fc_coverage_guard(self):
        from orthonym.rules.sulfur import chalcogen_oxide_fc_covers_molecule
        mol = Chem.MolFromSmiles("CSCCS(=O)C")
        match = mol.GetSubstructMatch(
            Chem.MolFromSmarts("[SX3](=[OX1])([#6])[#6]"))
        # Full-BFS sides do cover everything here; the decline comes from
        # the substitutive/FC side classifiers. The guard's job is the
        # simple-molecule contract:
        mol2 = Chem.MolFromSmiles("CS(=O)C")
        match2 = mol2.GetSubstructMatch(
            Chem.MolFromSmarts("[SX3](=[OX1])([#6])[#6]"))
        assert chalcogen_oxide_fc_covers_molecule(mol2, match2)
        assert chalcogen_oxide_fc_covers_molecule(mol, match)

    def test_needs_brackets_acid_stems(self):
        from orthonym.assembly.naming_utils import needs_brackets
        assert needs_brackets("methanesulfinyl")
        assert needs_brackets("methanesulfonyl")
        assert needs_brackets("benzenesulfinyl")
        assert needs_brackets("cyclohexanesulfinyl")
        assert not needs_brackets("sulfinyl")

    def test_acid_stem_prefix_builder(self):
        from orthonym.assembly.substituent_prefix_forms import (
            get_sulfinyl_prefix,
        )
        mol = Chem.MolFromSmiles("CS(=O)CC(=O)O")
        match = mol.GetSubstructMatch(
            Chem.MolFromSmarts("[SX3](=[OX1])([#6])[#6]"))
        chain = [a.GetIdx() for a in mol.GetAtoms()
                 if a.GetSymbol() == 'C' and not a.IsInRing()
                 and a.GetIdx() not in match[:2]]
        result = get_sulfinyl_prefix(mol, match, None)
        assert result == "methanesulfinyl"


@pytest.mark.unit
class TestCarbonPathMultipliedAcidGuard:
    """F6 : the carbon-path sulfone/sulfoxide PREFIX builder must NOT emit a
    corrupted 'multiplied acid' stem when the R' arm itself carries a SECOND
    S-oxo-acid group. For CS(=O)(=O)CCS(O)(=O)=O (CH3-SO2-CH2CH2-SO3H), capping
    the sulfone S with -OH forms 'ethane-1,2-disulfonic acid'; stripping the
    'sulfonic acid' suffix then corrupts to 'ethane-1,2-disulfonyl' -- asserting
    the whole ethane-1,2-diyl is a disulfonyl (a wrong constitution).

    The two soundness guards from 8afa533c (guard 1 = exactly one S-oxo-acid in the
    capped fragment; guard 2 = gate-independent OPSIN re-anchor), previously scoped
    to the N-parent sulfonamido branch, must now fire for the CARBON parent too.
    Fail closed (None) -> the caller drops to its own fail-closed handling; never
    the corrupted stem. Legit arms (saturated methyl / unsaturated allyl) unchanged.
    """

    def test_competing_sulfonic_acid_arm_fails_closed(self):
        from orthonym.assembly.substituent_prefix_forms import _acid_stem_oxide_prefix
        # CH3-SO2-CH2CH2-SO3H: sulfone S = idx1; SO3H-bearing arm carbon = idx4.
        mol = Chem.MolFromSmiles("CS(=O)(=O)CCS(O)(=O)=O")
        assert _acid_stem_oxide_prefix(mol, 4, 1, "sulfonyl") is None

    def test_plain_methyl_arm_unchanged(self):
        from orthonym.assembly.substituent_prefix_forms import _acid_stem_oxide_prefix
        # The CH3 side (idx0) of the same molecule is a clean methanesulfonyl.
        mol = Chem.MolFromSmiles("CS(=O)(=O)CCS(O)(=O)=O")
        assert _acid_stem_oxide_prefix(mol, 0, 1, "sulfonyl") == "methanesulfonyl"

    def test_unsaturated_arm_unchanged(self):
        from orthonym.assembly.substituent_prefix_forms import _acid_stem_oxide_prefix
        # Allyl methyl sulfone: the prop-2-ene arm still builds via the acid path
        # (guard 2's re-anchor accepts the clean 'prop-2-ene-1-sulfonic acid').
        m = Chem.MolFromSmiles("C=CCS(=O)(=O)C")
        s = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "S")
        assert _acid_stem_oxide_prefix(m, 2, s, "sulfonyl") == "prop-2-ene-1-sulfonyl"


@pytest.mark.unit
class TestTrivialAndControls:
    def test_sulfide_substitutive_pin(self):
        # (the Blue Book method 1 = PIN; the Blue Book): a sulfide's PIN is the
        # substitutive "(R'-sulfanyl)RH", not the functional-class "R R' sulfide".
        # (Migrated from the method-2 forms in the sulfanyl-vs-sulfide slice.)
        assert name_compound("CSC") == "(methylsulfanyl)methane"
        assert name_compound("CSCC") == "(methylsulfanyl)ethane"

    def test_sulfonic_acid_unchanged(self):
        assert name_compound("CS(=O)(=O)O") == "methanesulfonic acid"

    def test_sulfonamide_unchanged(self):
        assert name_compound("CCS(N)(=O)=O") == "ethanesulfonamide"


@pytest.mark.unit
class TestSeleniumTelluriumOxide:
    """-6I — Se/Te oxide analogues (the Blue Book "selenium and
    tellurium... named in the same way"; class names selenoxide/selenone,
    telluroxide/tellurone). The chemical-logic body is the SHARED, element-
    generic name_chalcogen_oxide_substitutive; only the prefix stem differs
    (seleninyl/selenonyl, tellurinyl/telluronyl)."""

    @pytest.mark.parametrize("smiles,expected", [
        # the Blue Book verbatim (diaryl multiplicative).
        ("O=[Se](=O)(c1ccccc1)c1ccccc1", "1,1'-selenonyldibenzene"),
        # the Blue Book verbatim (ring + chain).
        ("CC[Se](=O)c1ccccc1", "(ethaneseleninyl)benzene"),
        # symmetric dialkyl via the substitutive two-chain branch.
        ("C[Se](=O)C", "(methaneseleninyl)methane"),
        ("C[Se](=O)(=O)C", "(methaneselenonyl)methane"),
    ])
    def test_selenium(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # Te parses in OPSIN 2.9.0 and round-trips, so these emit (bonus breadth).
        ("O=[Te](=O)(c1ccccc1)c1ccccc1", "1,1'-telluronyldibenzene"),
        ("CC[Te](=O)c1ccccc1", "(ethanetellurinyl)benzene"),
        ("C[Te](=O)C", "(methanetellurinyl)methane"),
    ])
    def test_tellurium(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_selenium_roundtrips(self):
        """The emitted Se PIN parses back to the input structure under OPSIN."""
        from orthonym.validation.opsin_roundtrip import opsin_parse
        for smiles in ("O=[Se](=O)(c1ccccc1)c1ccccc1", "CC[Se](=O)c1ccccc1"):
            name = name_compound(smiles)
            got = opsin_parse(name)
            assert got and (inchi.MolToInchiKey(Chem.MolFromSmiles(got))
                            == inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)))

    def test_sulfur_analogue_unchanged(self):
        """Invariant 9: the S handlers are byte-identical after the Se/Te add."""
        assert name_compound("O=S(=O)(c1ccccc1)c1ccccc1") == "1,1'-sulfonyldibenzene"
        assert name_compound("CS(=O)c1ccccc1") == "(methanesulfinyl)benzene"
        assert name_compound("CS(=O)C") == "(methanesulfinyl)methane"
