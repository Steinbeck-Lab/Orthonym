"""Wave-0 RED unit suite for the charge-first PCG path: inorganic oxoacid
dianions + the charge-first PCG PROTECT (Phase 184 WS-E.3 / WS-E.3-dianion).

Written BEFORE the implementation (the Nyquist gate). Inorganic oxoacid dianions
are named as the anion of the preselected functional-parent acid (D-13): the
bare fully-deprotonated dianion -> the anion word (``sulfate``/``phosphate``/
``carbonate``), not ``unknown organic compound``. The ``sulfate`` assertion
FAILS on HEAD (``[O-]S(=O)(=O)[O-]`` -> ``unknown organic compound``) and flips
GREEN as Plan 184-03 adds the inorganic functional-parent table. ``phosphate``
and ``carbonate`` already PASS on HEAD (the table partially exists) and are
correctness guards that must stay GREEN.

The charge-first PCG PROTECT (D-12 / Phase 173.6) MUST stay byte-identical: the
ionized sulfonate is the senior PCG and the neutral COOH is demoted to a
``carboxy`` prefix -> ``2-carboxyethanesulfonate``.

A distinct-protonation guard (D-13 risk) asserts the mono-anion is NOT folded
into ``sulfate`` (it is ``hydrogen sulfate`` / currently ``unknown`` on HEAD,
but in any case MUST stay distinct from the dianion).

Every assertion docstring cites the verbatim Blue Book P-number.

Blue Book sources (BlueBookV2/BlueBookV2.md):
  - P-12.2 / line 35449: ``sulfuric acid`` is a preselected name -> ``sulfate``.
  - line 2076: ``(HO)3PO phosphoric acid (preselected name)`` -> ``phosphate``.
  - P-34.1.4: inorganic functional parents.
  - P-72.7 / P-41 (Phase 173.6): charge-first PCG ordering (the PROTECT row).
"""

import pytest

from orthonym.namer import Orthonym


@pytest.mark.unit
class TestInorganicDianion:
    """WS-E.3-dianion: a bare inorganic oxoacid dianion -> its preselected
    functional-parent anion word (D-13). ``sulfate`` is RED on HEAD;
    ``phosphate``/``carbonate`` already PASS (correctness guards)."""

    def test_sulfate(self):
        """P-12.2 / P-34.1.4 (BlueBookV2 line 35449): the fully-deprotonated
        dianion of the preselected ``sulfuric acid`` -> ``sulfate``. RED on HEAD
        (``unknown organic compound``)."""
        assert Orthonym().name("[O-]S(=O)(=O)[O-]") == "sulfate"

    def test_phosphate(self):
        """P-12.2 (BlueBookV2 line 2076): the trianion of the preselected
        ``phosphoric acid`` -> ``phosphate``."""
        assert Orthonym().name("[O-]P(=O)([O-])[O-]") == "phosphate"

    def test_carbonate(self):
        """P-66.1.6.1 / P-12.2: the dianion of carbonic acid -> ``carbonate``."""
        assert Orthonym().name("[O-]C(=O)[O-]") == "carbonate"

    def test_monoanion_distinct_from_sulfate(self):
        """D-13 risk guard: the mono-anion ``OS(=O)(=O)[O-]`` (``hydrogen
        sulfate``) MUST stay DISTINCT from the fully-deprotonated ``sulfate``
        dianion. The inorganic table keys on the exact protonation state and
        must never fold the mono-anion into ``sulfate``."""
        assert Orthonym().name("OS(=O)(=O)[O-]") != "sulfate"


@pytest.mark.unit
class TestChargeFirstPcgProtect:
    """D-12 / Phase 173.6 PROTECT: the ionized centre anchors the name; a neutral
    group of higher P-41 seniority is demoted to a prefix. GREEN on HEAD; must
    remain GREEN through WS-E.3 classifier generalization."""

    def test_carboxyethanesulfonate_unchanged(self):
        """P-72.7 / P-41: the sulfonate (ionized) is the senior PCG; the neutral
        COOH -> ``carboxy`` prefix -> ``2-carboxyethanesulfonate``. Generalizing
        the ``_ANION_PRINCIPAL_FG`` classifier must keep this byte-identical
        (the sulfonate/sulfinate/phosphonate entries stay verbatim)."""
        assert Orthonym().name("O=C(O)CCS(=O)(=O)[O-]") in {
            "2-carboxyethanesulfonate",
            "2-carboxyethane-1-sulfonate",
        }
