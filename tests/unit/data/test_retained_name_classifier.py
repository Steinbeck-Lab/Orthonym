"""a phase - 3-signal AND classifier unit tests.

Per internal notes: classifier is a pure 3-signal AND function:
  Signal 1: _is_complete_name heuristic
  Signal 2: data/iupac_2013_pin_list.json allow-list (PIN authority)
  Signal 3: per-entry OPSIN round-trip via InChI L1 (Plan 02)

Promotion rule: (S1 OR S2) AND S3.
Provisional mode (Plan 01 before validator runs): (S1 OR S2) only.

Each test cites:
  - The QMUL P-section URL,,
  - The IUPAC rule code
  - a phase internal notes,, decisions

Source: https://iupac.qmul.ac.uk/BlueBook/P2.html
Source: 150-internal notes +.
"""
import json
from pathlib import Path

import pytest

from orthonym.data import (
    ALL_RETAINED_NAMES,
    _PIN_ALLOW,
    _PIN_DENY,
    _PROVISIONAL_MODE,
    _ROUNDTRIP_CACHE_PATH,
    _is_complete_name,
    _is_promotable,
)


PIN_LIST_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "orthonym"
    / "data"
    / "iupac_2013_pin_list.json"
)


class TestIsCompleteNameExtension:
    """Signal 1 (_is_complete_name) heuristic-acceptance tests.

    See https://iupac.qmul.ac.uk/BlueBook/P2.html,.
    Per internal notes: Signal 1 is the original (a phase) heuristic;
    Signals 2/3 augment but never replace it.
    """

    @pytest.mark.unit
    def test_is_complete_name_accepts_acetic_acid(self):
        """Names containing spaces are complete IUPAC PIN forms.

        Cites (substituent + functional class compound naming).
        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        a phase internal notes Signal 1.
        """
        assert _is_complete_name("acetic acid") is True

    @pytest.mark.unit
    def test_is_complete_name_accepts_pyridine(self):
        """The -ine ending marks a mancude heterocycle PIN form.

        Cites See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        a phase internal notes Signal 1.
        """
        assert _is_complete_name("pyridine") is True

    @pytest.mark.unit
    def test_is_complete_name_accepts_morpholine(self):
        """The -ine ending also marks saturated heterocycle PIN forms.

        Cites See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        a phase internal notes Signal 1.
        """
        assert _is_complete_name("morpholine") is True

    @pytest.mark.unit
    def test_is_complete_name_rejects_perimidin_stem(self):
        """OPSIN stem 'perimidin' (no -e) must not be promoted as a PIN.

        Cites RESEARCH section 4.2 + ADDITION 2 (stem-vs-complete forms).
        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        a phase internal notes Signal 1; resolved post-Plan-01 via
        scripts/import_opsin_xml.py:_complete_stem normalization.
        """
        assert _is_complete_name("perimidin") is False

    @pytest.mark.unit
    def test_is_complete_name_rejects_acridin_stem(self):
        """OPSIN stem 'acridin' (no -e) must not be promoted as a PIN.

        Cites RESEARCH section 4.2 + ADDITION 2.
        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        a phase internal notes Signal 1.
        """
        assert _is_complete_name("acridin") is False


class TestPINAllowList:
    """Signal 2 (_PIN_ALLOW frozenset) authority assertions.

    See https://iupac.qmul.ac.uk/BlueBook/P2.html.
    Per internal notes: data/iupac_2013_pin_list.json is the single source
    of truth for which retained names are PINs.
    """

    @pytest.mark.unit
    def test_pin_allow_contains_benzene(self):
        """Benzene is a retained PIN.

        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        a phase internal notes (PIN allow-list authority).
        """
        assert "benzene" in _PIN_ALLOW

    @pytest.mark.unit
    def test_pin_allow_contains_adamantane(self):
        """Adamantane is a retained von Baeyer PIN.

        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        a phase internal notes.
        """
        assert "adamantane" in _PIN_ALLOW

    @pytest.mark.unit
    def test_pin_allow_contains_furan_pyridine_morpholine(self):
        """Heterocycle retained PINs Tables 2.2 + 2.3).

        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        a phase internal notes.
        """
        for name in ("furan", "pyridine", "morpholine"):
            assert name in _PIN_ALLOW, f"missing PIN: {name}"


class TestPINDenyOverrides:
    """Signal 2 hard-deny semantics: non-PIN names that look complete.

    See https://iupac.qmul.ac.uk/BlueBook/P2.html.
    Per internal notes: explicit DENY in PIN list overrides Signal 1 even
    when _is_complete_name returns True. Per internal notes:
    _OPSIN_NON_PIN_EXCLUSIONS is now derived from _PIN_DENY (single
    source of truth).
    """

    @pytest.mark.unit
    def test_pin_deny_contains_mesitylene(self):
        """Mesitylene is NOT a PIN; PIN is 1,3,5-trimethylbenzene.

        Cites. See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        a phase internal notes.
        """
        assert "mesitylene" in _PIN_DENY

    @pytest.mark.unit
    def test_pin_deny_overrides_signal_1(self):
        """Even when Signal 1 says True, an explicit DENY rejects.

        Mesitylene ends in -ene (Signal 1 True) but PIN is
        1,3,5-trimethylbenzene per, so the classifier must
        return False.

        Cites. See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        a phase internal notes hard-deny override semantic.
        """
        # Sanity: Signal 1 actually returns True for "mesitylene"
        assert _is_complete_name("mesitylene") is True
        # But Signal 2 deny must override -> overall False
        assert _is_promotable("Cc1cc(C)cc(C)c1", "mesitylene") is False

    @pytest.mark.unit
    def test_pin_deny_contains_alane_quinuclidine_prismane(self):
        """Three deprecated/non-PIN entries from RESEARCH 6.4-6.5.

        - alane: deprecated; replacement alumane.
        - quinuclidine: PIN is 1-azabicyclo[2.2.2]octane.
        - prismane: deprecated; tetracyclo systematic equivalent.

        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        a phase internal notes.
        """
        for name in ("alane", "quinuclidine", "prismane"):
            assert name in _PIN_DENY, f"missing PIN-deny: {name}"


class TestThreeSignalGate:
    """Combined classifier gate semantics post-Plan-02 graduation.

    Plan 01 ran the classifier in PROVISIONAL mode (S1 OR S2) because
    Signal 3 is sourced from
    src/orthonym/data/opsin_imports/_phase150_validation.json which
    Plan 02's validator produces. Plan 02 ships that JSON, which
    graduates the gate to full (S1 OR S2) AND S3.

    These tests assert the post-graduation behaviour. See fix
    in test_promotable_provisional_mode_active for self-explanatory
    failure handling when the validation JSON is missing.

    See https://iupac.qmul.ac.uk/BlueBook/P2.html.
    Source: 150-internal notes + +.
    Source: internal notes +.
    """

    @pytest.mark.unit
    def test_promotable_benzene_passes_via_signal_2(self):
        """Benzene passes the gate via Signal 2 even in provisional mode.

        Cites. See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        a phase internal notes +.
        """
        assert _is_promotable("c1ccccc1", "benzene") is True

    @pytest.mark.unit
    def test_promotable_provisional_mode_active(self):
        """Plan 02 graduates the classifier out of provisional mode.

        Plan 01 left _phase150_validation.json absent on disk so the
        classifier ran in provisional mode (S1 OR S2). Plan 02 ships
        scripts/validate_retained_names.py and the validator JSON output
        at src/orthonym/data/opsin_imports/_phase150_validation.json,
        which graduates the gate to full (S1 OR S2) AND S3.

        Per internal notes +. Plan 02 acceptance criterion (Task
        02-03) requires _PROVISIONAL_MODE is False post-graduation.
        See https://iupac.qmul.ac.uk/BlueBook/P2.html.

        a phase REVIEW: emit a self-explanatory failure if
        the validation JSON is missing on disk (e.g., a fresh
        checkout that strips files starting with '_'). Without this
        the test fails as a bare 'False is False' assertion with no
        actionable next step.
        """
        if _PROVISIONAL_MODE:
            pytest.fail(
                "Phase 150 SC-3: classifier in PROVISIONAL mode means "
                f"{_ROUNDTRIP_CACHE_PATH} is missing or unreadable. "
                "Run scripts/validate_retained_names.py to regenerate."
            )
        assert _PROVISIONAL_MODE is False

    @pytest.mark.unit
    def test_promotable_signal_3_deferred_in_provisional(self):
        """In provisional mode (S1 OR S2), pyridine (S1 True) passes.

        Pyridine is a mancude heterocycle PIN:
        S1 True via -ine ending; S2 True via PIN allow-list. Either
        path is sufficient in provisional mode.

        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        a phase internal notes provisional-mode gate.
        """
        assert _is_promotable("c1ccncc1", "pyridine") is True


class TestJSONAllowListSchema:
    """data/iupac_2013_pin_list.json must conform to a fixed schema.

    See https://iupac.qmul.ac.uk/BlueBook/P2.html.
    Per internal notes +: every entry must carry a P-section citation
    so reviewers can audit each PIN-or-not classification against the
    Blue Book.
    """

    @pytest.mark.unit
    def test_pin_list_json_loads(self):
        """JSON file loads cleanly with required top-level keys.

        a phase internal notes.
        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        """
        with open(PIN_LIST_PATH) as f:
            data = json.load(f)
        for key in ("version", "entries", "total_entries"):
            assert key in data, f"missing top-level key: {key}"
        assert isinstance(data["entries"], list)
        assert len(data["entries"]) >= 57
        # a phase REVIEW: tighten schema to catch total_entries
        # drift; the previous '>= 57' threshold let the field disagree
        # with the actual array length silently.
        assert data["total_entries"] == len(data["entries"]), (
            f"total_entries={data['total_entries']} but "
            f"len(entries)={len(data['entries'])}"
        )

    @pytest.mark.unit
    def test_pin_list_entries_have_required_fields(self):
        """Every entry has name, smiles, pin, citation, note.

        a phase internal notes schema contract; single-source
        of truth requires citation traceability for every entry.
        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        """
        with open(PIN_LIST_PATH) as f:
            data = json.load(f)
        required = ("name", "smiles", "pin", "citation", "note")
        for i, entry in enumerate(data["entries"]):
            for field in required:
                assert field in entry, (
                    f"entry {i} ({entry.get('name', '?')}) "
                    f"missing field {field!r}"
                )
            assert isinstance(entry["pin"], bool), (
                f"entry {i} pin value not bool: {entry['pin']!r}"
            )
            assert entry["citation"].startswith("P-"), (
                f"entry {i} citation {entry['citation']!r} "
                f"does not start with P-"
            )
        # a phase REVIEW: classifier is name-keyed (the
        # _PIN_ALLOW / _PIN_DENY frozensets are built from
        # entry["name"].lower), so duplicate names would silently
        # collapse into a single classifier slot whose pin flag is
        # whichever entry the iterator visits last. Duplicate SMILES
        # are intentional (e.g., benzene + [6]annulene share
        # c1ccccc1) and not a bug; duplicate NAMES are a data error.
        names_lower = [e["name"].lower() for e in data["entries"]]
        dups = sorted({n for n in names_lower if names_lower.count(n) > 1})
        assert not dups, (
            f"duplicate names in PIN list: {dups}; classifier is "
            f"name-keyed so duplicates collapse silently."
        )


# ============================================================================
# a phase  — deny / gate-safety tests.
# Audit: docs/retained_name_conflicts.md § "a phase".
# ============================================================================


class TestErythreneDenied:
    """RED until Plan 167-02 adds 'erythrene' to _PIN_DENY.

    'erythrene' currently slips the gate because (S1 OR S2) AND S3 admits any
    OPSIN-recognised synonym that round-trips — round-trip != PIN. Explicit DENY
    wins (data/__init__.py:_is_promotable). See internal notes root cause.
    """

    @pytest.mark.unit
    def test_erythrene_in_pin_deny(self):
        assert "erythrene" in _PIN_DENY

    @pytest.mark.unit
    def test_is_promotable_rejects_erythrene(self):
        # Explicit DENY must override S1 ("hrene" ending) + S3 (OPSIN-input RT).
        assert _is_promotable("C=CC=C", "erythrene") is False


class TestHCGateSafetyA1:
    """A1 guard (regression): the deny-based fix must NOT drop genuine names.

    The A1 dropped-set audit (docs/retained_name_conflicts.md § a phase STEP B)
    found 217/296 hand-curated entries fail the FULL _is_promotable gate — which
    is exactly why Plan 02 uses DENY-based exclusion, not full-gate promotion.
    These genuine PIN/retained names must (a) never be in _PIN_DENY and (b) stay
    emittable in ALL_RETAINED_NAMES. Green now and after Plan 02 (regression guard).
    """

    # F-T9/DD6: 'biphenyl' is NO LONGER genuine — it is general-only (PIN
    # 1,1'-biphenyl) and is now correctly denied. Replaced with 'naphthalene', a true
    # retained PIN that must stay non-denied + emittable.
    #
    #: 'camphor' is NO LONGER genuine either, and is removed here for the
    # same reason biphenyl was. It is a KETONE, and the retained-ketone rule is a
    # closed list that excludes it, so this is a POSITIVE exclusion, not an absence
    # argument. "Retained names" (heading the Blue Book):
    # * (:28297) — "The name 'chalcone' is the only retained name as a
    # preferred IUPAC name". Camphor is not chalcone.
    # * (:28307) — the general-nomenclature retained set is exactly
    # acetone, 1,4-benzoquinone, naphthoquinone, anthraquinone, ketene,
    # acetophenone and benzophenone, closing with "Substitutive names,
    # systematically constructed, are the preferred IUPAC names for ketones".
    # Camphor is on neither list, so it has no standing at either level.
    # The Blue Book never CONSTRUCTS "camphor" as a name: it occurs twice in the whole
    # book, at:32500 (a DIFFERENT compound, "camphoric anhydride") and at:52646,
    # where it is the familiar label printed beside "(1R,4R)-bornan-2-one".
    # Replaced with 'toluene' so the guard keeps its width — a name the curated list
    # positively affirms (pin: true,, not merely one it fails to deny.
    # The POSITIVE record that camphor is now deliberately denied (and demoted, not
    # deleted) lives in tests/unit/data/test_pin_deny_amino_and_trivial.py
    #::TestNaturalProductSurfaceIsGated, so removing it here loses no coverage.
    GENUINE = ["benzaldehyde", "butanoic acid", "toluene", "naphthalene", "acetamide"]

    @pytest.mark.unit
    def test_genuine_names_not_denied(self):
        for name in self.GENUINE:
            assert name.lower() not in _PIN_DENY, (
                f"genuine retained/PIN name {name!r} must not be denied"
            )

    @pytest.mark.unit
    def test_genuine_names_still_emittable(self):
        values = set(ALL_RETAINED_NAMES.values())
        for name in self.GENUINE:
            assert name in values, (
                f"genuine retained/PIN name {name!r} dropped from ALL_RETAINED_NAMES"
            )
