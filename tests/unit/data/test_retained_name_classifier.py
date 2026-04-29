"""Phase 150 D-02 - 3-signal AND classifier unit tests.

Per CONTEXT D-02: classifier is a pure 3-signal AND function:
  Signal 1: _is_complete_name heuristic
  Signal 2: data/iupac_2013_pin_list.json allow-list (PIN authority)
  Signal 3: per-entry OPSIN round-trip via InChI L1 (Plan 02)

Promotion rule: (S1 OR S2) AND S3.
Provisional mode (Plan 01 before validator runs): (S1 OR S2) only.

Each test cites:
  - The QMUL P-section URL (P-22.1.3, P-22.2.1, P-23.7)
  - The IUPAC rule code
  - Phase 150 CONTEXT D-02, D-06, D-07 decisions

Source: https://iupac.qmul.ac.uk/BlueBook/P2.html
Source: 150-CONTEXT.md D-02 + D-06.
"""
import json
from pathlib import Path

import pytest

from orthonym.data import (
    ALL_RETAINED_NAMES,
    _PIN_ALLOW,
    _PIN_DENY,
    _PROVISIONAL_MODE,
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

    See https://iupac.qmul.ac.uk/BlueBook/P2.html (P-22.1.3, P-22.2.1).
    Per CONTEXT D-02: Signal 1 is the original (Phase 136) heuristic;
    Signals 2/3 augment but never replace it.
    """

    @pytest.mark.unit
    def test_is_complete_name_accepts_acetic_acid(self):
        """Names containing spaces are complete IUPAC PIN forms.

        Cites P-22.1.3 (substituent + functional class compound naming).
        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        Phase 150 CONTEXT D-02 Signal 1.
        """
        assert _is_complete_name("acetic acid") is True

    @pytest.mark.unit
    def test_is_complete_name_accepts_pyridine(self):
        """The -ine ending marks a mancude heterocycle PIN form.

        Cites P-22.2.1 Table 2.2. See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        Phase 150 CONTEXT D-02 Signal 1.
        """
        assert _is_complete_name("pyridine") is True

    @pytest.mark.unit
    def test_is_complete_name_accepts_morpholine(self):
        """The -ine ending also marks saturated heterocycle PIN forms.

        Cites P-22.2.1 Table 2.3. See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        Phase 150 CONTEXT D-02 Signal 1.
        """
        assert _is_complete_name("morpholine") is True

    @pytest.mark.unit
    def test_is_complete_name_rejects_perimidin_stem(self):
        """OPSIN stem 'perimidin' (no -e) must not be promoted as a PIN.

        Cites RESEARCH section 4.2 + ADDITION 2 (stem-vs-complete forms).
        See https://iupac.qmul.ac.uk/BlueBook/P2.html (P-22.2.1).
        Phase 150 CONTEXT D-02 Signal 1; resolved post-Plan-01 via
        :_complete_stem normalization.
        """
        assert _is_complete_name("perimidin") is False

    @pytest.mark.unit
    def test_is_complete_name_rejects_acridin_stem(self):
        """OPSIN stem 'acridin' (no -e) must not be promoted as a PIN.

        Cites RESEARCH section 4.2 + ADDITION 2.
        See https://iupac.qmul.ac.uk/BlueBook/P2.html (P-22.2.1 Table 2.2).
        Phase 150 CONTEXT D-02 Signal 1.
        """
        assert _is_complete_name("acridin") is False


class TestPINAllowList:
    """Signal 2 (_PIN_ALLOW frozenset) authority assertions.

    See https://iupac.qmul.ac.uk/BlueBook/P2.html.
    Per CONTEXT D-06: data/iupac_2013_pin_list.json is the single source
    of truth for which retained names are PINs.
    """

    @pytest.mark.unit
    def test_pin_allow_contains_benzene(self):
        """Benzene is a retained PIN (P-22.1.3).

        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        Phase 150 CONTEXT D-06 (PIN allow-list authority).
        """
        assert "benzene" in _PIN_ALLOW

    @pytest.mark.unit
    def test_pin_allow_contains_adamantane(self):
        """Adamantane is a retained von Baeyer PIN (P-23.7 Table 2.6).

        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        Phase 150 CONTEXT D-06.
        """
        assert "adamantane" in _PIN_ALLOW

    @pytest.mark.unit
    def test_pin_allow_contains_furan_pyridine_morpholine(self):
        """Heterocycle retained PINs (P-22.2.1 Tables 2.2 + 2.3).

        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        Phase 150 CONTEXT D-06.
        """
        for name in ("furan", "pyridine", "morpholine"):
            assert name in _PIN_ALLOW, f"missing PIN: {name}"


class TestPINDenyOverrides:
    """Signal 2 hard-deny semantics: non-PIN names that look complete.

    See https://iupac.qmul.ac.uk/BlueBook/P2.html.
    Per CONTEXT D-02: explicit DENY in PIN list overrides Signal 1 even
    when _is_complete_name returns True. Per CONTEXT CD-02:
    _OPSIN_NON_PIN_EXCLUSIONS is now derived from _PIN_DENY (single
    source of truth).
    """

    @pytest.mark.unit
    def test_pin_deny_contains_mesitylene(self):
        """Mesitylene is NOT a PIN; PIN is 1,3,5-trimethylbenzene.

        Cites P-22.1.3. See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        Phase 150 CONTEXT D-06.
        """
        assert "mesitylene" in _PIN_DENY

    @pytest.mark.unit
    def test_pin_deny_overrides_signal_1(self):
        """Even when Signal 1 says True, an explicit DENY rejects.

        Mesitylene ends in -ene (Signal 1 True) but PIN is
        1,3,5-trimethylbenzene per P-22.1.3, so the classifier must
        return False.

        Cites P-22.1.3. See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        Phase 150 CONTEXT D-02 hard-deny override semantic.
        """
        # Sanity: Signal 1 actually returns True for "mesitylene"
        assert _is_complete_name("mesitylene") is True
        # But Signal 2 deny must override -> overall False
        assert _is_promotable("Cc1cc(C)cc(C)c1", "mesitylene") is False

    @pytest.mark.unit
    def test_pin_deny_contains_alane_quinuclidine_prismane(self):
        """Three deprecated/non-PIN entries from RESEARCH 6.4-6.5.

        - alane: deprecated; replacement alumane (P-21.1.1.1).
        - quinuclidine: PIN is 1-azabicyclo[2.2.2]octane (P-23.7).
        - prismane: deprecated; tetracyclo systematic equivalent (P-23.7).

        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        Phase 150 CONTEXT D-06.
        """
        for name in ("alane", "quinuclidine", "prismane"):
            assert name in _PIN_DENY, f"missing PIN-deny: {name}"


class TestThreeSignalGate:
    """Combined classifier gate semantics in Provisional mode.

    See https://iupac.qmul.ac.uk/BlueBook/P2.html.
    Per CONTEXT D-02: provisional mode (Plan 01) uses (S1 OR S2) only;
    Signal 3 is deferred until Plan 02 round-trip validator runs.
    """

    @pytest.mark.unit
    def test_promotable_benzene_passes_via_signal_2(self):
        """Benzene passes the gate via Signal 2 even in provisional mode.

        Cites P-22.1.3. See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        Phase 150 CONTEXT D-02 + D-06.
        """
        assert _is_promotable("c1ccccc1", "benzene") is True

    @pytest.mark.unit
    def test_promotable_provisional_mode_active(self):
        """Plan 02 graduates the classifier out of provisional mode.

        Plan 01 left _phase150_validation.json absent on disk so the
        classifier ran in provisional mode (S1 OR S2). Plan 02 ships
         and the validator JSON output
        at src/orthonym/data/opsin_imports/_phase150_validation.json,
        which graduates the gate to full (S1 OR S2) AND S3.

        Per CONTEXT D-04 + D-07. Plan 02 acceptance criterion (Task
        02-03) requires _PROVISIONAL_MODE is False post-graduation.
        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        """
        assert _PROVISIONAL_MODE is False

    @pytest.mark.unit
    def test_promotable_signal_3_deferred_in_provisional(self):
        """In provisional mode (S1 OR S2), pyridine (S1 True) passes.

        Pyridine is a mancude heterocycle PIN (P-22.2.1 Table 2.2):
        S1 True via -ine ending; S2 True via PIN allow-list. Either
        path is sufficient in provisional mode.

        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        Phase 150 CONTEXT D-02 provisional-mode gate.
        """
        assert _is_promotable("c1ccncc1", "pyridine") is True


class TestJSONAllowListSchema:
    """data/iupac_2013_pin_list.json must conform to a fixed schema.

    See https://iupac.qmul.ac.uk/BlueBook/P2.html.
    Per CONTEXT D-06 + CD-02: every entry must carry a P-section citation
    so reviewers can audit each PIN-or-not classification against the
    Blue Book.
    """

    @pytest.mark.unit
    def test_pin_list_json_loads(self):
        """JSON file loads cleanly with required top-level keys.

        Phase 150 CONTEXT D-06.
        See https://iupac.qmul.ac.uk/BlueBook/P2.html.
        """
        with open(PIN_LIST_PATH) as f:
            data = json.load(f)
        for key in ("version", "entries", "total_entries"):
            assert key in data, f"missing top-level key: {key}"
        assert isinstance(data["entries"], list)
        assert len(data["entries"]) >= 57

    @pytest.mark.unit
    def test_pin_list_entries_have_required_fields(self):
        """Every entry has name, smiles, pin, citation, note.

        Phase 150 CONTEXT D-06 schema contract; CD-02 single-source
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
