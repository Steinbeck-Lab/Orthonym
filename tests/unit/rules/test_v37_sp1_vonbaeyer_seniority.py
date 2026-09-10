""" -- von Baeyer element-seniority locant tiebreak.

For a mixed-heteroatom von Baeyer skeleton, the numbering selector
(``get_bicyclo_numbering._key_lists``/``_cmp`` in ``rules/bicyclo.py``) used to
break a heteroatom locant-SET tie ``{2, 6}`` by RDKit atom index -- a SMILES-
atom-order artifact -- instead of by element seniority. Governing rule:

     [BBv2:9789], verbatim at ``rules/ring_replacement.py:150-152``:
    "If there is still a choice, low locants are assigned in accord with the
    decreasing seniority order of heteroatoms
    O > S > Se > Te > N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B >..."

So on a set-tie the SENIOR element must take the LOWER locant. Seniority is
NOT alphabetical and NOT atomic number (note O > S >... > N).

RT-valid either way (each heteroatom lands at its locant regardless of which
gets 2 vs 6), so this is a pure spelling / PIN-preference fix; 0-wrong holds.
"""

from orthonym import name_compound


class TestVonBaeyerHeteroatomSeniorityLocant:
    """: the senior heteroatom takes the lower locant on a set-tie."""

    def test_oxygen_senior_to_nitrogen(self):
        # O > N: O must take locant 2 (was 6 -- atom-order artifact).
        assert name_compound("O1CC2CNC1C2") == "2-oxa-6-azabicyclo[2.2.1]heptane"

    def test_oxygen_senior_to_sulfur(self):
        # O > S: O must take locant 2 (was 6).
        assert name_compound("O1CC2CSC1C2") == "2-oxa-6-thiabicyclo[2.2.1]heptane"

    def test_oxygen_senior_to_silicon_unchanged(self):
        # O > Si: O already at 2 by accident of atom order -- must NOT regress.
        assert name_compound("[SiH2]1CC2COC1C2") == "2-oxa-6-silabicyclo[2.2.1]heptane"

    def test_atom_order_permutation_locks_artifact(self):
        # SAME molecule as O1CC2CNC1C2 (InChIKey JFLCTUFGJBISET) but N written
        # first in SMILES order. Before the fix this ordering happened to emit
        # the CORRECT name while O1CC2CNC1C2 emitted the wrong one -- proving the
        # defect is a pure atom-order artifact. Both orderings must now converge
        # on the seniority-correct spelling.
        assert name_compound("N1CC2COC1C2") == "2-oxa-6-azabicyclo[2.2.1]heptane"
        assert name_compound("N1CC2COC1C2") == name_compound("O1CC2CNC1C2")


class TestVonBaeyerControlsByteIdentical:
    """All-carbon / single-heteroatom von Baeyer names have no set-tie to break
    and MUST stay byte-identical (the PIN-never-regresses guard)."""

    def test_norbornane_unchanged(self):
        assert name_compound("C1CC2CCC1C2") == "norbornane"

    def test_bicyclo222octane_unchanged(self):
        assert name_compound("C1CC2CCC1CC2") == "bicyclo[2.2.2]octane"

    def test_single_oxa_unchanged(self):
        assert name_compound("O1CC2CCC1C2") == "2-oxabicyclo[2.2.1]heptane"

    def test_single_aza_unchanged(self):
        assert name_compound("N1CC2CCC1C2") == "2-azabicyclo[2.2.1]heptane"

    def test_single_thia_unchanged(self):
        assert name_compound("S1CC2CCC1C2") == "2-thiabicyclo[2.2.1]heptane"
