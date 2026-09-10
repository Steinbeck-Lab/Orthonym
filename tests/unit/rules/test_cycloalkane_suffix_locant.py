""".1 S3 — cycloalkane suffix-locant priority.

The principal characteristic group expressed as a suffix gets the lowest ring
locant FIRST tier (c)), before detachable prefixes. The legacy
`orient_cycloalkane` only knew substituted positions, so the alphabetic
tie-break handed locant 1 to the alkyl prefix: `1-octylcyclohexan-2-ol`
instead of the PIN `2-octylcyclohexan-1-ol`.

Tier order under test (V21-ALGORITHM-FIX-PLAN.md.1 step 3):
  (c) lowest locants to the suffix anchor(s) FIRST,
  (f) then lowest locants to the prefix-only set (EXCLUDING the suffix
      position) — NOT the combined set,
  (g) then lowest locant to the first-cited (alphabetical) prefix.

Negative protect: an exocyclic suffix carbon (-carbaldehyde, -carboxylic
acid) is NOT part of the ring and must not seize ring numbering (the FG
match's ring atom test: only a ring C directly bonded to the FG heteroatom).
"""

import pytest
from rdkit import Chem

from orthonym.namer import name_compound
from orthonym.rules.cycloalkanes import orient_cycloalkane


@pytest.mark.unit
class TestCycloalkaneSuffixLocantGold:
    """End-to-end gold targets (among-rings gold rows; OPSIN-verified PINs)."""

    def test_octylcyclohexanol(self):
        assert name_compound("OC1CCCCC1CCCCCCCC").strip() == "2-octylcyclohexan-1-ol"

    def test_pentylcyclohexanone(self):
        assert name_compound("O=C1CCCCC1CCCCC").strip() == "2-pentylcyclohexan-1-one"

    def test_hexylcyclohexanamine(self):
        assert name_compound("NC1CCCCC1CCCCCC").strip() == "2-hexylcyclohexan-1-amine"


@pytest.mark.unit
class TestCycloalkaneSuffixLocantProtect:
    """Regression guards — already-correct behavior that must not move."""

    def test_exocyclic_carbaldehyde_does_not_seize_numbering(self):
        # The CHO carbon is exocyclic: the ring-OH prefix takes 2, CHO stays
        # the C1 suffix anchor. Gold protect row (already green pre-S3).
        assert (
            name_compound("OC1CCCCC1C=O").strip()
            == "2-hydroxycyclohexane-1-carbaldehyde"
        )

    def test_plain_methylcyclohexane_unchanged(self):
        assert name_compound("CC1CCCCC1").strip() == "methylcyclohexane"

    def test_dimethylcyclohexane_unchanged(self):
        assert name_compound("CC1CCCCC1C").strip() == "1,2-dimethylcyclohexane"


@pytest.mark.unit
class TestOrientCycloalkanePGUnit:
    """Direct unit tests of orient_cycloalkane's principal_group_atoms path.

    substituent_positions mirrors the real caller (get_ring_substituents):
    the PG exocyclic heteroatom IS present as a substituent list.
    """

    def test_pg_gets_locant_one_direction_minimizes_prefix(self):
        # 2-ethylcyclohexan-1-ol topology: OH ring-C must be locant 1 and the
        # direction must put ethyl at 2 (not 6).
        mol = Chem.MolFromSmiles("OC1CCCCC1CC")
        ring = (1, 2, 3, 4, 5, 6)  # cyclic order; atom 6 adjacent to atom 1
        subs = {1: [[0]], 6: [[7, 8]]}
        oriented = orient_cycloalkane(mol, ring, subs, principal_group_atoms={1})
        assert oriented[0] == 1, "suffix anchor must take locant 1"
        assert oriented[1] == 6, "direction must give the prefix locant 2, not 6"

    def test_suffix_first_beats_combined_set_tie(self):
        # 3,5-dimethylcyclohexan-1-ol: the combined set is {1,3,5} whether
        # numbering starts at the OH carbon or at a methyl carbon — only the
        # suffix-FIRST tier (c) forces OH to locant 1 (legacy alphabetic
        # tie-break preferred 'methyl' at position 1).
        mol = Chem.MolFromSmiles("OC1CC(C)CC(C)C1")
        ring = (1, 2, 3, 5, 6, 8)  # cyclic order 1-2-3-5-6-8-1
        subs = {1: [[0]], 3: [[4]], 6: [[7]]}
        oriented = orient_cycloalkane(mol, ring, subs, principal_group_atoms={1})
        assert oriented[0] == 1, "suffix anchor must take locant 1 over methyls"
        # methyls land at {3,5} in either direction
        locants = sorted(oriented.index(a) + 1 for a in (3, 6))
        assert locants == [3, 5]

    def test_multiple_pg_atoms_lowest_set(self):
        # cyclohexane-1,2-diol topology: adjacent PG carbons take {1,2}.
        mol = Chem.MolFromSmiles("OC1CCCCC1O")
        ring = (1, 2, 3, 4, 5, 6)
        subs = {1: [[0]], 6: [[7]]}
        oriented = orient_cycloalkane(mol, ring, subs, principal_group_atoms={1, 6})
        pg_locants = sorted(oriented.index(a) + 1 for a in (1, 6))
        assert pg_locants == [1, 2]

    def test_alphabetical_tier_g_first_cited_prefix(self):
        # OH at C1 with ethyl and methyl on the two adjacent carbons: tiers
        # (c) and (f) tie ({2,6} both directions); tier (g) gives the
        # first-cited prefix (ethyl) locant 2.
        mol = Chem.MolFromSmiles("CCC1CCCC(C)C1O")
        ring = (2, 3, 4, 5, 6, 8)  # cyclic order 2-3-4-5-6-8-2
        subs = {2: [[1, 0]], 6: [[7]], 8: [[9]]}
        oriented = orient_cycloalkane(mol, ring, subs, principal_group_atoms={8})
        assert oriented[0] == 8, "suffix anchor must take locant 1"
        assert oriented[1] == 2, "ethyl (first-cited) must get locant 2"

    def test_no_pg_legacy_behavior_unchanged(self):
        # Without principal_group_atoms the legacy path is untouched: single
        # substituent becomes position 1.
        mol = Chem.MolFromSmiles("CC1CCCCC1")
        ring = (1, 2, 3, 4, 5, 6)
        subs = {1: [[0]]}
        oriented = orient_cycloalkane(mol, ring, subs)
        assert oriented[0] == 1
