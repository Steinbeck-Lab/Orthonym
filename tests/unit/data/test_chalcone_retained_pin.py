""" — `chalcone`, the ONLY retained ketone name that is a PIN.

Authority (opened and read, not relayed):

    the Blue Book Blue Book — section heading ``### **** Retained
    names`` / ``****`` — "The name 'chalcone' is the only retained name
    as a preferred IUPAC name and is limited to **ring substitution only by
    characteristic groups lower than 'ketone'**. Chalcone refers only to the
    *trans*- or (*E*)- stereoisomer."

    the Blue Book Blue Book — ``chalcone (PIN) (2E)-1,3-diphenylprop-2-en-1-one``

Context that makes this a PIN rather than general nomenclature:
(:28307) closes the *general*-nomenclature ketone list (acetone, 1,4-benzoquinone,
naphthoquinone, anthraquinone, ketene, acetophenone, benzophenone) and ends
"Substitutive names, systematically constructed, are the preferred IUPAC names
for ketones". Chalcone is the single exception.

THE THREE CONSTRAINTS the rule imposes, each with a test below:

  1. (E) ONLY. "Chalcone refers only to the trans- or (E)- stereoisomer." The
     (Z) isomer and the stereo-UNSPECIFIED molecule must both keep the
     systematic name — emitting `chalcone` for an unspecified double bond would
     assert a configuration the input does not carry.
  2. Ring substitution ONLY, and only by characteristic groups JUNIOR to ketone.
  3. No substitution on the propenone chain.

IMPLEMENTATION NOTE — why an exact whole-molecule canonical-SMILES key enforces
all three constraints as a PROOF rather than an approximation:
``namer._name_impl`` computes ``canonical_smiles = Chem.MolToSmiles(mol,
canonical=True)``, which is ISOMERIC by default. The three stereo variants
therefore have three DISTINCT keys (measured, see ``test_stereo_variants_have_
distinct_canonical_keys``), and ANY substitution anywhere — ring or chain,
junior or senior — changes the key. So the retained name can only ever be
emitted for the bare (E) parent, and the Blue Book's own worked counter-example
``2',4'-dihydroxychalcone-4-carboxamide`` (BB:28305, "(not...)"; already
listed in ``data/bluebook_not_names.py:171``) is structurally unreachable.

SCOPE (deliberate, documented): substituted chalcones such as the BB's
``2',4'-dihydroxy-3,3'-dimethoxychalcone (PIN)`` (:28303) are NOT emitted. They
require primed-locant machinery across two rings plus a ketone-seniority
adjudication; building that wrong would emit a WRONG name, whereas falling
through emits the valid systematic name. See TaskL-report.md.
"""
import pytest
from rdkit import Chem

from orthonym.data import ALL_RETAINED_NAMES
from orthonym.data.bluebook_not_names import BLUEBOOK_NOT_NAMES
from orthonym.namer import Orthonym


# (E)-1,3-diphenylprop-2-en-1-one — the chalcone parent.
E_CHALCONE = "O=C(/C=C/c1ccccc1)c1ccccc1"
Z_ISOMER = r"O=C(/C=C\c1ccccc1)c1ccccc1"
NO_STEREO = "O=C(C=Cc1ccccc1)c1ccccc1"
RING_OH = "O=C(/C=C/c1ccc(O)cc1)c1ccccc1"
# The Blue Book's own counter-example: a carboxamide OUTRANKS ketone, so the
# chalcone name is forbidden and the systematic form is the PIN (BB:28305).
BB_COUNTEREXAMPLE = "O=C(/C=C/c1ccc(cc1)C(N)=O)c1ccc(O)cc1O"
CHAIN_SUBST = "O=C(/C=C/c1ccccc1)C"   # (3E)-4-phenylbut-3-en-2-one, not chalcone
SATURATED = "O=C(CCc1ccccc1)c1ccccc1"  # 1,3-diphenylpropan-1-one


def _canon(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


class TestChalconeRetainedKey:
    """The data row itself."""

    def test_e_chalcone_is_a_retained_pin(self):
        """BB :28299 — ``chalcone (PIN)  (2E)-1,3-diphenylprop-2-en-1-one``."""
        assert ALL_RETAINED_NAMES.get(_canon(E_CHALCONE)) == "chalcone"

    def test_z_isomer_is_not_keyed(self):
        """"Chalcone refers only to the trans- or (E)- stereoisomer" (:28297)."""
        assert _canon(Z_ISOMER) not in ALL_RETAINED_NAMES

    def test_stereo_unspecified_is_not_keyed(self):
        """An unspecified double bond must not be given an (E) assertion."""
        assert _canon(NO_STEREO) not in ALL_RETAINED_NAMES

    def test_stereo_variants_have_distinct_canonical_keys(self):
        """The load-bearing measurement: the retained key is stereo-DISCRIMINATING.

        If ``MolToSmiles`` collapsed E/Z, one dict row would silently name the
        (Z) isomer `chalcone` — a wrong structure. This test is the guard on
        that assumption, not an incidental check.
        """
        keys = {_canon(E_CHALCONE), _canon(Z_ISOMER), _canon(NO_STEREO)}
        assert len(keys) == 3, f"stereo variants collapsed to {keys}"


class TestChalconeEmission:
    """What actually SHIPS — presence in a table is not evidence it is reached."""

    def test_emits_chalcone(self, namer):
        assert namer.name(E_CHALCONE) == "chalcone"

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            (Z_ISOMER, "(2Z)-1,3-diphenylprop-2-en-1-one"),
            (NO_STEREO, "1,3-diphenylprop-2-en-1-one"),
            (RING_OH, "(2E)-3-(4-hydroxyphenyl)-1-phenylprop-2-en-1-one"),
            (CHAIN_SUBST, "(3E)-4-phenylbut-3-en-2-one"),
            (SATURATED, "1,3-diphenylpropan-1-one"),
        ],
    )
    def test_near_family_keeps_systematic_name(self, namer, smiles, expected):
        """Constraints 1-3: everything outside the bare (E) parent falls through."""
        assert namer.name(smiles) == expected

    @pytest.mark.parametrize(
        "smiles",
        [Z_ISOMER, NO_STEREO, RING_OH, BB_COUNTEREXAMPLE, CHAIN_SUBST, SATURATED],
    )
    def test_no_chalcone_token_leaks_into_the_family(self, namer, smiles):
        """The substitution limit (:28297) — no `...chalcone...` name may appear."""
        assert "chalcone" not in namer.name(smiles).lower()

    def test_bb_counterexample_is_never_the_forbidden_name(self, namer):
        """BB:28305 ``(not 2',4'-dihydroxychalcone-4-carboxamide)``.

        A carboxamide is SENIOR to ketone, so 's "lower than 'ketone'"
        limit forbids the chalcone name here.
        """
        emitted = namer.name(BB_COUNTEREXAMPLE).lower().strip()
        assert emitted != "2',4'-dihydroxychalcone-4-carboxamide"
        assert emitted not in BLUEBOOK_NOT_NAMES

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("CC(C)=O", "propan-2-one"),
            ("O=C1C=CC(=O)C=C1", "cyclohexa-2,5-diene-1,4-dione"),
            ("CC(=O)c1ccccc1", "1-phenylethan-1-one"),
            ("O=C(c1ccccc1)c1ccccc1", "diphenylmethanone"),
            ("O=C1CCCCC1", "cyclohexanone"),
        ],
    )
    def test_nearby_ketones_unregressed(self, namer, smiles, expected):
        """P-64.2.1.2: these stay SYSTEMATIC — chalcone is the only exception."""
        assert namer.name(smiles) == expected


class TestChalconeImpliesItsOwnStereo:
    """ (:28297): "Chalcone refers only to the trans- or (E)-
    stereoisomer." — so the retained name IS the descriptor for its one E/Z
    unit, and the stereo backstop's "name lacks descriptors" warning would be a
    FALSE POSITIVE. `_stereo_is_implied_by_name` is diagnostic-only (both
    branches at namer.py:182 return the name unchanged), so this changes no
    emitted name."""

    def test_chalcone_is_recognised_as_self_describing(self):
        from orthonym.namer import _stereo_is_implied_by_name

        assert _stereo_is_implied_by_name("chalcone") is True

    def test_leg_is_exact_not_a_substring(self):
        """A substituted chalcone may carry ADDITIONAL stereogenic units that do
        need descriptors, so the exemption must not swallow them."""
        from orthonym.namer import _stereo_is_implied_by_name

        for n in ("4-hydroxychalcone", "2',4'-dihydroxychalcone-4-carboxamide",
                  "chalcones", "dihydrochalcone"):
            assert _stereo_is_implied_by_name(n) is False, n


class TestChalconeUnderTheRealGate:
    """The default shipping path runs the OPSIN validity gate; chalcone must
    survive it (OPSIN 2.9.0 parses `chalcone` natively)."""

    @pytest.mark.opsin_gate
    def test_emits_chalcone_with_gate_enabled(self, namer):
        assert Orthonym().name(E_CHALCONE) == "chalcone"
