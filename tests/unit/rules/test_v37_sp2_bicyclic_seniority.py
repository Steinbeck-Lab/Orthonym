"""SP2.2 -- element-seniority tiebreak on a heteroatom-locant-SET tie in a
BICYCLIC spiro von-Baeyer component (``spiro.py::_name_vonbaeyer_spiro_component``,
the ``_key`` selection loop).

This is the SPIRO sibling of SP1.5 (commit ``030cc1dd``), which fixed the same
set-vs-vector seniority bug for the NON-spiro ``bicyclo.py`` path. Before this fix
the spiro bicyclic ``_key`` tuple was ``(spiro_loc, het_SET, ene)`` with no
element-seniority tier, so a heteroatom locant-SET tie (e.g. ``{2,6}``) fell
through to candidate/atom-index order -- a SMILES-atom-order artifact. The two
atom-order permutations below are the SAME molecule and used to emit two different
names.

P-23.3.2.2 [BBv2:9789]: on a heteroatom locant-SET tie the senior element
(O > S > Se > Te > N > P > ...) takes the LOWER locant. Pure spelling/PIN-preference
fix; both spellings OPSIN-round-trip to the same structure, so 0-wrong holds either
way (asserted below).
"""
import pytest

from orthonym import name_compound
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

# One molecule, two SMILES atom orders. The bicyclo[2.2.1] O/N cage is spiro-fused
# at its C7 (1-atom bridge, on the mirror axis) to a cyclobutane, so O and N sit in
# the two mirror-symmetric 2-atom bridges: the heteroatom locant SET {2,6} ties and
# only element seniority can break it. Current emitted string (pre-fix) for _W_A is
# "6-oxa2-azaspiro[bicyclo[2.2.1]heptane-7,1'-cyclobutane]" (O wrongly at 6); _W_B
# already emits the O-at-2 form. (An unrelated pre-existing spelling glitch drops the
# hyphen after "oxa"/"aza"; this test deliberately checks the LOCANT assignment via
# substrings rather than baking in that separate defect.)
_W_A = "O1CC2CNC1C23CCC3"
_W_B = "N1CC2COC1C23CCC3"


def test_spiro_bicyclic_seniority_deterministic_across_atom_order():
    """Same molecule, two atom orders -> identical name (the core determinism bug)."""
    a = name_compound(_W_A)
    b = name_compound(_W_B)
    assert a == b, f"atom-order-dependent spiro-VB name: {a!r} != {b!r}"


def test_spiro_bicyclic_seniority_senior_o_gets_lower_locant():
    """O is senior to N (P-23.3.2.2) -> O takes locant 2, N locant 6."""
    name = name_compound(_W_A)
    assert "2-oxa" in name, name  # senior O at the lower locant
    assert "6-aza" in name, name  # N at the higher locant


@pytest.mark.parametrize("smi", [_W_A, _W_B])
def test_spiro_bicyclic_seniority_roundtrips(smi):
    """0-wrong guard: whichever spelling is chosen must round-trip to the input."""
    name = name_compound(smi)
    assert not name.startswith("unknown"), name
    assert opsin_roundtrip_check(smi, name)["passed"], name
