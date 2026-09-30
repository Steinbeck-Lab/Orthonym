""" Composition Increment 1a -- O-glycoside of a COMPLEX/retained-named aglycone.

The glycoside namer builds the functional-class ``<aglycone-yl> <sugar>oside`` form
 and already names simple aglycones (``cyclohexyl``/``menthyl``/
``phenyl`` β-D-glucopyranoside). It USED to decline when the aglycone's ``-yl``
prefix could only be produced by the string rule ``_alcohol_to_alkyl`` and that rule
FABRICATED an OPSIN-unparseable token -- e.g. the retained alcohol ``borneol`` ->
``borneyl`` (not an OPSIN substituent), which the OPSIN validity gate then suppressed
to ``unknown organic compound``.

Increment 1a derives the aglycone ``-yl`` prefix from STRUCTURE (the aglycone atom
set + the glycosidic-oxygen carbon) via the ring-substituent chokepoint
``name_ring_system_substituent`` when the string rule's token is OPSIN-unparseable,
and RT-gates the whole glycoside (full InChI) before emitting. 0-wrong is absolute:
a candidate that does not round-trip is never shipped (the molecule abstains).

Governing rule: (functional-class glycoside), (substituent
free-valence morphology). Every asserted name string below was confirmed to
OPSIN-round-trip to the input's InChI before being pinned.
"""

import pytest

from orthonym.namer import name_compound
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
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
    "OC[C@H]1O[C@@H](OC23CC4CC(CC(C4)C2)C3)[C@H](O)[C@@H](O)[C@@H]1O",
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


# whole-branch review (minor 2): every test here asserts an OPSIN round-trip,
# so mark the whole module opsin_gate — a Java-free run SKIPS these rather than
# reporting them as failures (parity with the CP2 / salt / test files).
pytestmark = pytest.mark.opsin_gate


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
# 2026-09-26 (wp7) change-asserted-value: was '(4,7,7-trimethylbicyclo[2.2.1]heptan-
# 5-yl) β-D-glucopyranoside'. With the OPSIN-import trivial 'borneol' out of the PIN lookup
# (no Blue Book PIN evidence; 0 BB hits) the aglycone is named from structure: the free
# valence takes the lowest locant the von Baeyer numbering allows (2 < 5,,
# exactly as the Blue Book numbers camphor, '(1R,4R)-1,7,7-trimethylbicyclo[2.2.1]heptan-
# 2-one' (the Blue Book). OPSIN 2.9.0 full-InChIKey exact.
BORNYL_EXPECTED = "1,7,7-trimethylbicyclo[2.2.1]heptan-2-yl β-D-glucopyranoside"

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
        name = _dt_name_compound(BORNYL_GLUCOSIDE)
        assert not is_failure_name(name), f"still abstains: {name!r}"
        # 0-wrong: the emitted name must describe the input structure.
        assert opsin_roundtrip_check(BORNYL_GLUCOSIDE, name)["passed"], name
        assert name.endswith("β-D-glucopyranoside")
        # Regression pin (structural, RT-verified). If the ring-substituent
        # numbering later improves to the preferred `1,7,7-...-2-yl`, update
        # this string -- the RT assertion above is the load-bearing check.
        assert name == BORNYL_EXPECTED

    @pytest.mark.parametrize("smiles", RED_ANCHORS)
    def test_complex_aglycone_glucosides_name_or_abstain_never_wrong(self, smiles):
        """Each abstained before the fix -> now NAME + RT, or (worst case) abstain.

        NEVER a wrong molecule: an emitted name must round-trip to the input.
        """
        name = _dt_name_compound(smiles)
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

    @staticmethod
    def _reordered_smiles(smiles):
        """Yield the canonical SMILES + several DETERMINISTIC atom-order variants.

        Determinism matters here: `Chem.MolToSmiles(mol, doRandom=True)` varies
        run-to-run, so a green pass would not be reproducible. Instead we renumber
        the atoms by fixed rotations (each a valid permutation) and write each in
        atom-index order (`canonical=False`), giving distinct-but-reproducible
        writings of the same molecule to feed back through the namer.
        """
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        n = mol.GetNumAtoms()
        out = [Chem.MolToSmiles(mol)]  # canonical
        for shift in (1, 3, 7, 13):  # fixed rotations -> reproducible orderings
            order = [(i + shift) % n for i in range(n)]
            out.append(
                Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
            )
        return out

    @pytest.mark.parametrize("smiles,expected", [
        (BORNYL_GLUCOSIDE, BORNYL_EXPECTED),
        # a second complex-aglycone witness (chain-rooted onto a bicyclic)
        (MYRTANOL_GLUCOSIDE, None),
    ])
    def test_glucoside_deterministic_across_atom_orders(self, smiles, expected):
        variants = self._reordered_smiles(smiles)
        assert len(variants) >= 5  # canonical + 4 deterministic reorderings
        names = [_dt_name_compound(v) for v in variants]
        # Every atom ordering yields the identical name (a project rule).
        assert len(set(names)) == 1, dict(zip(variants, names))
        assert not is_failure_name(names[0]), names[0]
        # And that name round-trips (0-wrong holds under every ordering).
        assert opsin_roundtrip_check(smiles, names[0])["passed"], names[0]
        if expected is not None:
            assert names[0] == expected


@pytest.mark.integration
class TestGlycosideByteIdentityControls:
    """The already-working glycosides + trivial names MUST NOT change.

    Checked at BOTH the default and best-effort tiers (a project rule / PIN
    never regresses). The structural fallback fires ONLY when the string
    aglycone token is OPSIN-unparseable, so every parseable retained form
    (menthyl/cyclohexyl/phenyl/...) is untouched.
    """

    CONTROLS = {
        "OC[C@H]1O[C@@H](OC2CCCCC2)[C@H](O)[C@@H](O)[C@@H]1O":
            "cyclohexyl β-D-glucopyranoside",
        "OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O":
            "phenyl β-D-glucopyranoside",
        # wp7 change-asserted-value: was 'menthyl β-D-glucopyranoside'; 'menthyl' came
        # from the OPSIN-import trivial 'menthol' (0 Blue Book hits, no PIN evidence), which
        # left the PIN lookup, so the aglycone is named from structure. OPSIN 2.9.0
        # full-InChIKey exact.
        "CC(C)C1CCC(C)CC1O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O":
            "5-methyl-2-(propan-2-yl)cyclohexyl β-D-glucopyranoside",
        # Breadth job 1 change-asserted-value: were 'decahydronaphthalenyl' and
        # 'tricyclo[3.3.1.1^3,7]decyl'. The alcohol-part prefix now keeps its free-
        # valence locant for a parent that is not a saturated chain or monocycle:
        # (the Blue Book, "Locants... are placed immediately before
        # that part of the name to which they relate";:2864 'naphthalen-2-yl
        # (preferred prefix)'), (c) (:2913, the locant '1' is omitted
        # only in monosubstituted homogeneous MONOcyclic rings) and
        # (:16374 '... tricyclo[3.3.1.1^3,7]decan-2-yl', the von Baeyer prefix keeps
        # its locant). OPSIN 2.9.0 full-InChIKey exact for both.
        "OC[C@H]1O[C@@H](OC2CCCC3CCCCC23)[C@H](O)[C@@H](O)[C@@H]1O":
            "decahydronaphthalen-1-yl β-D-glucopyranoside",
        "OC[C@H]1O[C@@H](OC23CC4CC(CC(C4)C2)C3)[C@H](O)[C@@H](O)[C@@H]1O":
            "tricyclo[3.3.1.1^3,7]decan-1-yl β-D-glucopyranoside",
        "OC[C@H]1O[C@@H](OC2CCC3(CCCO3)CC2)[C@H](O)[C@@H](O)[C@@H]1O":
            "1-oxaspiro[4.5]decan-8-yl β-D-glucopyranoside",
        "CCO": "ethanol",
        "c1ccccc1": "benzene",
    }

    @pytest.mark.parametrize("smiles,expected", list(CONTROLS.items()))
    def test_control_default_tier_unchanged(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", list(CONTROLS.items()))
    def test_control_best_effort_tier_unchanged(self, smiles, expected):
        assert _best_effort(smiles) == expected
