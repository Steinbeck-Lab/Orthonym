""" — spiro-VB 'a'-replacement prefix hyphen joiner.

PART 1 of the SP2 PIN-spelling pair. ``_spiro_vb_a_prefix`` (spiro.py) is the
THIRD spiro-context skeletal-replacement speller. It built each element's
``<locants>-<mult><stem>`` part correctly but joined the parts with ``''``
instead of ``'-'``, so a two-heteroatom prefix came out ``2-oxa6-aza`` instead
of the PIN ``2-oxa-6-aza``. The two sibling spellers
(``_build_hetero_prefix`` at spiro.py, ``build_replacement_prefix`` in
ring_replacement.py) both join with ``'-'``; this one had diverged.

 (a project rule) VERIFIED on HEAD in a fresh process: a trace on
``_spiro_vb_a_prefix`` returned ``'2-oxa6-aza'`` for ``O1CC2CNC1C23CCC3``, and
that string is prepended directly before ``spiro[`` in
``_name_spiro_vonbaeyer_core`` -> the emitted name. Both the pre-fix and
post-fix names round-trip through OPSIN to the same molecule (OPSIN is lenient
on the missing hyphen), so this is a pure PIN-spelling correction, 0 breadth.
"""
import pytest

from orthonym import errors
from orthonym.namer import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

# 2-oxa-6-azaspiro[bicyclo[2.2.1]heptane-7,1'-cyclobutane]: a monospiro of a
# bicyclo[2.2.1] cage (bearing O and N skeletal heteroatoms) with cyclobutane.
SMI = "O1CC2CNC1C23CCC3"
EXPECTED = "2-oxa-6-azaspiro[bicyclo[2.2.1]heptane-7,1'-cyclobutane]"


def _rt(smi: str, name) -> bool:
    if not name or errors.is_failure_name(name):
        return False
    r = opsin_roundtrip_check(smi, name)
    return bool(r.get("passed") and r.get("inchi_match"))


class TestSpiroVbReplacementHyphen:
    def test_prefix_joins_with_hyphen(self):
        """The two 'a'-prefix terms are separated by a hyphen (PIN spelling)."""
        name = Orthonym().name(SMI)
        assert name == EXPECTED, name
        # regression assert on the exact glitch string
        assert "oxa6-aza" not in name, name
        assert "2-oxa-6-aza" in name, name

    @pytest.mark.opsin_gate
    def test_corrected_name_round_trips(self):
        """0-wrong: the corrected spelling still describes the right molecule."""
        name = Orthonym().name(SMI)
        assert _rt(SMI, name), f"corrected name does not RT: {name!r}"
