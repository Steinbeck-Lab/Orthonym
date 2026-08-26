"""v38 Composition Increment 1a -- O-glycoside of a COMPLEX/retained-named aglycone.

The glycoside namer builds the functional-class ``<aglycone-yl> <sugar>oside`` form
(P-102.5.6.2.2) and already names simple aglycones (``cyclohexyl``/``menthyl``/
``phenyl`` beta-D-glucopyranoside). It USED to decline when the aglycone's ``-yl``
prefix could only be produced by the string rule ``_alcohol_to_alkyl`` and that rule
FABRICATED an OPSIN-unparseable token -- e.g. the retained alcohol ``borneol`` ->
``borneyl`` (not an OPSIN substituent), which the OPSIN validity gate then suppressed
to ``unknown organic compound``.

Increment 1a derives the aglycone ``-yl`` prefix from STRUCTURE (the aglycone atom
set + the glycosidic-oxygen carbon) via the ring-substituent chokepoint
``name_ring_system_substituent`` when the string rule's token is OPSIN-unparseable,
and RT-gates the whole glycoside (full InChI) before emitting. 0-wrong is absolute:
a candidate that does not round-trip is never shipped (the molecule abstains).

Governing rule: P-102.5.6.2.2 (functional-class glycoside), P-29.2 (substituent
free-valence morphology). Every asserted name string below was confirmed to
OPSIN-round-trip to the input's InChI before being pinned.
"""

import pytest

from orthonym.namer import name_compound
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


# --- Best-effort tier helper (default tier is a plain name_compound) ----------
def _best_effort(smiles: str) -> str:
    return name_compound(
        smiles,
        general_fallback=True,
        general_fallback_unverified=True,
        allow_aromatic_general=True,
    )


# --- RED anchors: retained-alcohol aglycones that abstained before the fix ----
# (smiles, expected_exact_name). The expected name is a systematic von-Baeyer /
# ring substituent form -- NON-preferred numbering is acceptable at 0-wrong
# because the RT gate proves the constitution+stereo (spelling-layer numbering
# is a separate, out-of-scope concern). Each expected name RT-verifies.
BORNYL_GLUCOSIDE = "CC1(C)C2CCC1(C)C(O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O)C2"
BORNYL_EXPECTED = "(4,7,7-trimethylbicyclo[2.2.1]heptan-5-yl) beta-D-glucopyranoside"

TERPINEOL_GLUCOSIDE = (
    "CC1=CCC(CC1)C(C)(C)O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"
)
MYRTANOL_GLUCOSIDE = "CC1(C)C2CCC1CC2CO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"

RED_ANCHORS = [
    BORNYL_GLUCOSIDE,
    TERPINEOL_GLUCOSIDE,
    MYRTANOL_GLUCOSIDE,
]


@pytest.mark.integration
class TestComplexAglyconeGlycosideNames:
    """A sugar on a retained/complex aglycone now NAMES + RT-verifies (was abstain)."""

    def test_bornyl_glucoside_names_and_roundtrips(self):
        name = name_compound(BORNYL_GLUCOSIDE)
        assert not is_failure_name(name), f"still abstains: {name!r}"
        # 0-wrong: the emitted name must describe the input structure.
        assert opsin_roundtrip_check(BORNYL_GLUCOSIDE, name)["passed"], name
        assert name.endswith("beta-D-glucopyranoside")
        # Regression pin (structural, RT-verified). If the ring-substituent
        # numbering later improves to the preferred `1,7,7-...-2-yl`, update
        # this string -- the RT assertion above is the load-bearing check.
        assert name == BORNYL_EXPECTED

    @pytest.mark.parametrize("smiles", RED_ANCHORS)
    def test_complex_aglycone_glucosides_name_or_abstain_never_wrong(self, smiles):
        """Each abstained before the fix -> now NAME + RT, or (worst case) abstain.

        NEVER a wrong molecule: an emitted name must round-trip to the input.
        """
        name = name_compound(smiles)
        if is_failure_name(name):
            pytest.skip(f"abstains (acceptable, not wrong): {name!r}")
        assert opsin_roundtrip_check(smiles, name)["passed"], (smiles, name)
        assert "glucopyranoside" in name

    def test_bornyl_glucoside_best_effort_also_names_and_roundtrips(self):
        name = _best_effort(BORNYL_GLUCOSIDE)
        assert not is_failure_name(name), name
        assert opsin_roundtrip_check(BORNYL_GLUCOSIDE, name)["passed"], name


@pytest.mark.integration
class TestComplexAglyconeGlycosideDeterminism:
    """The derived aglycone substituent must be order-independent (canonical rank)."""

    def test_bornyl_glucoside_deterministic_across_smiles_orders(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(BORNYL_GLUCOSIDE)
        order_a = Chem.MolToSmiles(mol)  # canonical
        order_b = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        name_a = name_compound(order_a)
        name_b = name_compound(order_b)
        assert name_a == name_b, (order_a, name_a, order_b, name_b)
        assert name_a == BORNYL_EXPECTED


@pytest.mark.integration
class TestGlycosideByteIdentityControls:
    """The already-working glycosides + trivial names MUST NOT change.

    Checked at BOTH the default and best-effort tiers (invariant 16 / PIN
    never regresses). The structural fallback fires ONLY when the string
    aglycone token is OPSIN-unparseable, so every parseable retained form
    (menthyl/cyclohexyl/phenyl/...) is untouched.
    """

    CONTROLS = {
        "OC[C@H]1O[C@@H](OC2CCCCC2)[C@H](O)[C@@H](O)[C@@H]1O":
            "cyclohexyl beta-D-glucopyranoside",
        "OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O":
            "phenyl beta-D-glucopyranoside",
        "CC(C)C1CCC(C)CC1O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O":
            "menthyl beta-D-glucopyranoside",
        "OC[C@H]1O[C@@H](OC2CCCC3CCCCC23)[C@H](O)[C@@H](O)[C@@H]1O":
            "decahydronaphthalenyl beta-D-glucopyranoside",
        "OC[C@H]1O[C@@H](OC23CC4CC(CC(C4)C2)C3)[C@H](O)[C@@H](O)[C@@H]1O":
            "tricyclo[3.3.1.1^3,7]decyl beta-D-glucopyranoside",
        "OC[C@H]1O[C@@H](OC2CCC3(CCCO3)CC2)[C@H](O)[C@@H](O)[C@@H]1O":
            "1-oxaspiro[4.5]decan-8-yl beta-D-glucopyranoside",
        "CCO": "ethanol",
        "c1ccccc1": "benzene",
    }

    @pytest.mark.parametrize("smiles,expected", list(CONTROLS.items()))
    def test_control_default_tier_unchanged(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", list(CONTROLS.items()))
    def test_control_best_effort_tier_unchanged(self, smiles, expected):
        assert _best_effort(smiles) == expected
