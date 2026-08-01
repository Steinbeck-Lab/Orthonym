"""Task E: the curated PIN deny-list must reach the amino-acid, natural-product
and trivial-acid surfaces, not only the retained-names dict.

Producer-level assertions only. Whole-molecule assertions are unsound in this
suite: ``conftest`` disables the OPSIN gate suite-wide, so ``name_tiered`` can
select a different producer than the CLI does. User-visible behaviour for these
names is verified through the CLI, recorded in
``.superpowers/sdd/v29-residue/TaskE-report.md``.

Blue Book basis (each row also carries its citation in
``data/iupac_2013_pin_list.json``):

* P-100 "INTRODUCTION" (BlueBookV2.md:50939), sentence :50943 -- "Preferred
  IUPAC names (PINs) are not identified for the compounds in this Chapter."
  So Chapter 10 names are *prescribed retained names*, never PINs, and a name
  absent from the P-103 tables is licensed by nothing.
* P-103.1.1.3 "Systematic substitutive names" (:54247), sentence :54251 --
  "When not denoted by a retained name, amino acids receive systematic
  substitutive names constructed by applying the principles, rules and
  conventions of substitutive nomenclature." Precedent at :54253: the names
  'norvaline' and 'norleucine' "are not recommended".
"""

import json
from pathlib import Path

import pytest


def _pin_list():
    import orthonym.data as data_pkg
    path = Path(data_pkg.__file__).parent / "iupac_2013_pin_list.json"
    with open(path) as fh:
        return json.load(fh)


# Names adjudicated in Task E: absent from (or explicitly deprecated by) the
# Blue Book AND with a measured, structurally correct systematic replacement.
GATED_AMINO_ACIDS = ["sarcosine", "taurine", "homotaurine", "statine",
                     "diaminopimelic acid", "abrine"]

GATED_RETAINED = ["nicotinamide", "picolinic acid", "vanillin",
                  "vanillic acid", "benzhydrol", "phloroglucinol", "durene",
                  "mesityl oxide"]

GATED_NATURAL_PRODUCTS = ["camphor"]


@pytest.mark.unit
class TestPinListRows:
    """Every gated name is a curated deny row carrying a Blue Book citation."""

    @pytest.mark.parametrize(
        "name", GATED_AMINO_ACIDS + GATED_RETAINED + GATED_NATURAL_PRODUCTS)
    def test_row_present_denied_and_cited(self, name):
        rows = [e for e in _pin_list()["entries"]
                if e["name"].lower() == name.lower()]
        assert rows, f"{name!r} has no row in iupac_2013_pin_list.json"
        row = rows[0]
        assert row.get("pin") is False, f"{name!r} must be pin:false"
        assert row.get("citation"), f"{name!r} must carry a Blue Book citation"


@pytest.mark.unit
class TestAminoAcidSurfaceIsGated:
    """data/amino_acids.py must consult the curated deny-list."""

    @pytest.mark.parametrize("name", GATED_AMINO_ACIDS)
    def test_denied_name_absent_from_pin_lookup(self, name):
        from orthonym.data.amino_acids import (
            STANDARD_AMINO_ACIDS, NON_STANDARD_AMINO_ACIDS,
        )
        live = set(STANDARD_AMINO_ACIDS.values()) | set(
            NON_STANDARD_AMINO_ACIDS.values())
        assert name not in live, (
            f"{name!r} is still reachable as a PIN-path amino-acid name")

    @pytest.mark.parametrize("name", GATED_AMINO_ACIDS)
    def test_denied_name_retained_for_general_nomenclature(self, name):
        """Rows are demoted, never deleted (--trivial / general keeps them)."""
        from orthonym.data.amino_acids import GENERAL_ONLY_AMINO_ACIDS
        assert name in set(GENERAL_ONLY_AMINO_ACIDS.values()), (
            f"{name!r} was deleted rather than demoted")

    @pytest.mark.parametrize("name", GATED_AMINO_ACIDS)
    def test_get_amino_acid_name_refuses_denied(self, name):
        """The lookup used by rules.amino_acids.name_amino_acid must not
        return a denied name for its own key."""
        from orthonym.data.amino_acids import (
            GENERAL_ONLY_AMINO_ACIDS, get_amino_acid_name,
        )
        keys = [k for k, v in GENERAL_ONLY_AMINO_ACIDS.items() if v == name]
        assert keys, f"no demoted key for {name!r}"
        for key in keys:
            assert get_amino_acid_name(key) != name


@pytest.mark.unit
class TestBlueBookRetainedAminoAcidsSurvive:
    """The `is_pin` field is NOT the gate.

    `` hard-codes ``"is_pin": False`` (lines 268,
    742, 847), so all 232 entries in ``amino_acids_opsin.py`` carry False. Gating
    on it would withdraw all 88 integrated names, including ``cystine`` -- a Blue
    Book Table 10.5 retained name (P-103.1.1.2). This test pins that trap shut.
    """

    @pytest.mark.parametrize("name", [
        "glycine", "alanine", "valine", "leucine", "isoleucine", "proline",
        "serine", "threonine", "cysteine", "methionine", "asparagine",
        "glutamine", "lysine", "arginine", "histidine", "phenylalanine",
        "tyrosine", "tryptophan", "aspartic acid", "glutamic acid",
        "cystine", "ornithine", "dopa",
    ])
    def test_retained_amino_acid_still_reachable(self, name):
        from orthonym.data.amino_acids import (
            STANDARD_AMINO_ACIDS, NON_STANDARD_AMINO_ACIDS,
        )
        live = set(STANDARD_AMINO_ACIDS.values()) | set(
            NON_STANDARD_AMINO_ACIDS.values())
        assert name in live, f"{name!r} was withdrawn -- over-broad gate"

    def test_uniform_is_pin_is_not_used_as_a_signal(self):
        """All 232 OPSIN amino-acid entries are is_pin:False, yet most survive."""
        from orthonym.data.opsin_imports.amino_acids_opsin import (
            OPSIN_AMINO_ACIDS,
        )
        from orthonym.data.amino_acids import NON_STANDARD_AMINO_ACIDS
        assert all(e.get("is_pin") is False for e in OPSIN_AMINO_ACIDS.values())
        assert len(NON_STANDARD_AMINO_ACIDS) > 50, (
            "gating on is_pin would leave almost nothing")


@pytest.mark.unit
class TestRetainedNameSurfaceIsGated:
    """The existing JSON deny already governs ALL_RETAINED_NAMES; these rows
    must land on the general side, exactly like glycerol/catechol."""

    @pytest.mark.parametrize("name", GATED_RETAINED)
    def test_absent_from_pin_dict(self, name):
        from orthonym.data import ALL_RETAINED_NAMES
        assert name not in set(ALL_RETAINED_NAMES.values())

    @pytest.mark.parametrize("name", GATED_RETAINED)
    def test_present_in_general_dict(self, name):
        from orthonym.data import GENERAL_RETAINED_NAMES
        assert name in set(GENERAL_RETAINED_NAMES.values()), (
            f"{name!r} was deleted rather than demoted")


@pytest.mark.unit
class TestNaturalProductSurfaceIsGated:
    """data/natural_products.py must consult the deny-list too.

    A runtime spy showed ``camphor`` was emitted by this surface, not by the
    retained-names dict where it is ALSO keyed -- so a deny row alone was inert
    until this module was wired. the contributor guide invariant 10: presence in a lookup
    table is not evidence the table is reached.
    """

    @pytest.mark.parametrize("name", GATED_NATURAL_PRODUCTS)
    def test_denied_name_withheld_from_pin_lookup(self, name):
        from orthonym.data.natural_products import (
            GENERAL_ONLY_NATURAL_PRODUCTS, get_natural_product_name,
        )
        keys = [k for k, v in GENERAL_ONLY_NATURAL_PRODUCTS.items() if v == name]
        assert keys, f"{name!r} was not withdrawn from the PIN lookup"
        for key in keys:
            assert get_natural_product_name(key) is None

    @pytest.mark.parametrize("name", GATED_NATURAL_PRODUCTS)
    def test_row_not_deleted(self, name):
        """Demoted, not deleted -- the table still carries the name."""
        from orthonym.data.natural_products import NATURAL_PRODUCT_DERIVATIVES
        assert name in set(NATURAL_PRODUCT_DERIVATIVES.values())

    def test_undenied_natural_products_survive(self):
        from orthonym.data.natural_products import get_natural_product_name
        assert get_natural_product_name(
            "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
        ) == "morphine"


@pytest.mark.unit
class TestSharedLoader:
    """One JSON, one loader. A silent load failure must not read as 'nothing
    is denied' -- that would make every gating test above pass vacuously."""

    def test_deny_sets_are_populated(self):
        from orthonym.data.pin_policy import PIN_DENY, PIN_DENY_HC, PIN_ALLOW
        assert len(PIN_DENY) > 150
        assert len(PIN_DENY_HC) > 150
        assert "glycerol" in PIN_DENY_HC
        assert "benzene" in PIN_ALLOW

    def test_data_package_reuses_the_shared_loader(self):
        import orthonym.data as pkg
        from orthonym.data import pin_policy
        assert pkg._PIN_DENY is pin_policy.PIN_DENY
        assert pkg._PIN_DENY_HC is pin_policy.PIN_DENY_HC
        assert pkg._PIN_ALLOW is pin_policy.PIN_ALLOW


@pytest.mark.unit
class TestNotOverGated:
    """Names the audit flagged that this task must NOT withdraw."""

    def test_tert_butylbenzene_is_a_pin_construction(self):
        """BlueBookV2.md:25717 '1,2-di-*tert*-butylbenzene (PIN)' and :3507
        '1-(butan-2-yl)-3-*tert*-butylbenzene (PIN)'. Raw grep gives 0 hits --
        the italic markup hides it -- so the audit read it as absent."""
        from orthonym.data import ALL_RETAINED_NAMES
        denied = {e["name"].lower() for e in _pin_list()["entries"]
                  if e.get("pin") is False}
        assert "tert-butylbenzene" not in denied
        assert ALL_RETAINED_NAMES.get("CC(C)(C)c1ccccc1") == "tert-butylbenzene"

    @pytest.mark.parametrize("name", [
        # measured fallback -> reason for not gating
        "creatine",           # 'unknown organic compound'
        "selenocystine",      # 'unknown organic compound'
        "tellurocystine",     # 'tellurium compound (not supported)'
        "lysopine",           # 'unknown organic compound'
        "saccharin",          # 'unknown organic compound'
        "triphenylmethane",   # 'unknown organic compound'
        "morphine",           # fallback uses 'morphin-7-ene', but the BB's own
                              # renderings (BB:2680, BB:52608) both use
                              # '7,8-didehydromorphinan' -- replacement unverified
        "glycocyamine",       # fallback 'guanidinoacetic acid' uses a prefix the
                              # BB deprecates for PINs (P-66.4.1.2.1.3)
    ])
    def test_no_verified_replacement_means_not_gated(self, name):
        """data/__init__.py:263 -- a deny row needs 'a Blue Book citation AND a
        verified replacement'. Each name here is genuinely non-Blue-Book, but its
        measured PIN-path fallback is an abstention, an unsupported-element
        refusal, or a name that is itself non-preferred. Gating them would trade
        a non-PIN name for a coverage loss, which is the failure mode recorded at
        data/__init__.py:258-262 ('methane' for N=C=N)."""
        denied = {e["name"].lower() for e in _pin_list()["entries"]
                  if e.get("pin") is False}
        assert name not in denied

    def test_guanidino_is_not_a_preferred_prefix(self):
        """Answers the audit's open question, and explains the glycocyamine row.

        P-66.4.1.2.1.3 (BB:34266): 'In the presence of a characteristic group
        having seniority over guanidine (see item 11 in P-41), the following
        prefixes are used. The prefix guanidino may be used in general
        nomenclature.' BB:34268 marks 'carbamimidoylamino (preferred prefix)'.
        The P-66 introduction is blunter still, item (g) at BB:1700: "The prefix
        'guanidino' is no longer acceptable in preferred IUPAC names but may be
        used in general nomenclature; the preferred prefix is
        'carbamimidoylamino'."

        Orthonym still emits it -- rules/seniority.py maps guanidine ->
        'guanidino'. That is a SEPARATE live defect, deliberately left to its own
        task: its sibling was already corrected in place (the same dict maps
        amidine -> 'carbamimidoyl' with the note '(was "amidino" - wrong per BB
        P-66.4.1.3.1)'), so the fix is in-class but out of scope here. This test
        pins the fact so the next session does not have to re-derive it.
        """
        from orthonym.rules.seniority import PREFIX_FORMS
        assert PREFIX_FORMS.get("guanidine") == "guanidino", (
            "guanidino prefix defect appears fixed -- retire this test and gate "
            "glycocyamine, whose fallback 'guanidinoacetic acid' then becomes "
            "'carbamimidoylaminoacetic acid'")
