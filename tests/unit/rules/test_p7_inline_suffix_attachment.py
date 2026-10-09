""" a phase / C2: the suffix locant of a phenol, aromatic amine or enol
is the CARBON's locant, never the heteroatom's.

Task 5 made ``general_engine._inline_suffix_locant`` refuse when the
characteristic atom is off the parent. It delegates atom choice to
``rules.parent_selection._pg_attachment_atoms``, which is driven by
``seniority.PG_ATTACHMENT_INDICES`` and defaults to SMARTS-match index 0.
``phenol`` (``[OX2H][cX3]``), ``aromatic_amine`` (``[NX3H2][cX3]``) and ``enol``
(``[OX2H][CX3]=[CX3]``) all lead with the HETEROATOM, so the default returned an
atom that is never on the ring -- the guard then refused correct, round-tripping
names.

Blue Book authority (heading + deciding sentence):

   (the Blue Book Blue Book) -- "Primary amines, R-NH2, are
  systematically named in the following ways: (1) by adding the suffix 'amine'
  to the name of the parent hydride". The suffix attaches to the PARENT
  HYDRIDE, whose skeletal atoms carry the locants; the -NH2 nitrogen is not a
  parent-hydride atom. Its own examples are ``quinolin-4-amine (PIN)`` and
  ``1-benzofuran-2-amine (PIN)`` -- the locant is the ring carbon.

   "Systematic names of alcohols, phenols, enols, and ynols"
  (:26826) -- "(1) substitutively, using the suffix 'ol'... When there is a
  choice for numbering, the starting point and the direction of numbering of a
  compound are chosen so as to give lowest locants to the 'ol' suffixes".
  Examples ``naphthalen-1-ol (PIN)`` (:26820) and ``2-nitrobenzene-1,3-diol
  (PIN)``: the cited locants are ring carbons -- the hydroxy oxygen has no
  skeletal locant at all.

   "Citation of locants" (:2869) -- "the name 2-chloroethan-1-ol is the
  PIN". The '1' designates the carbon bearing the -OH, not the oxygen.
"""

import pytest

from orthonym.rules.seniority import PG_ATTACHMENT_INDICES
from orthonym.rules.parent_selection import _pg_attachment_atoms


# Tier flags for the best-effort tier, mirroring scripts/measure_breadth.py's
# TIER_FLAGS['best-effort'] -- the only tier on which the guard is reachable.
BEST_EFFORT = dict(general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True)


class TestAttachmentIndices:
    """The three heteroatom-leading SMARTS get an explicit carbon index."""

    @pytest.mark.parametrize("fg", ["phenol", "aromatic_amine", "enol"])
    def test_heteroatom_leading_pg_points_at_index_1(self, fg):
        # [OX2H][cX3] / [NX3H2][cX3] / [OX2H][CX3]=[CX3] -- index 1 is the
        # carbon the suffix converts, above).
        assert PG_ATTACHMENT_INDICES[fg] == [1], (
            f"{fg} must cite its carbon, not its heteroatom"
        )

    def test_imine_is_explicitly_pinned_to_its_leading_carbon(self):
        # [CX3]=[NX2;...] already leads with the carbon, so [0] is a no-op
        # today -- pinned so a future SMARTS reorder cannot silently move the
        # locant onto the nitrogen.
        assert PG_ATTACHMENT_INDICES["imine"] == [0]

    def test_phenol_match_resolves_to_the_aromatic_carbon(self):
        # SMARTS match order is (O, c); atom ids here are arbitrary but
        # distinct so the assertion cannot pass by coincidence.
        assert _pg_attachment_atoms("phenol", (7, 3)) == [3]

    def test_aromatic_amine_match_resolves_to_the_aromatic_carbon(self):
        assert _pg_attachment_atoms("aromatic_amine", (8, 2)) == [2]

    def test_enol_match_resolves_to_the_hydroxy_bearing_carbon(self):
        # (O, C-OH, =C) -- index 1, not the oxygen and not the far alkene C.
        assert _pg_attachment_atoms("enol", (9, 4, 5)) == [4]


class TestSmartsOrderMatchesTheIndex:
    """The index is only correct if the shipped SMARTS really leads with the
    heteroatom -- assert that against RDKit rather than trusting the string."""

    @pytest.mark.parametrize("fg,smiles,het_sym,carbon_arom", [
        ("phenol", "Oc1ccccc1", "O", True),
        ("aromatic_amine", "Nc1ccccc1", "N", True),
        ("enol", "OC=C", "O", False),
    ])
    def test_index_1_of_a_real_match_is_the_carbon(
        self, fg, smiles, het_sym, carbon_arom
    ):
        from rdkit import Chem
        from orthonym.perception.functional_groups import (
            FUNCTIONAL_GROUP_SMARTS,
        )

        mol = Chem.MolFromSmiles(smiles)
        patt = Chem.MolFromSmarts(FUNCTIONAL_GROUP_SMARTS[fg])
        match = mol.GetSubstructMatch(patt)
        assert match, f"{fg} SMARTS did not match {smiles}"

        assert mol.GetAtomWithIdx(match[0]).GetSymbol() == het_sym
        chosen = _pg_attachment_atoms(fg, match)
        assert len(chosen) == 1
        atom = mol.GetAtomWithIdx(chosen[0])
        assert atom.GetSymbol() == "C", (
            f"{fg} attachment must be carbon, got {atom.GetSymbol()}"
        )
        assert atom.GetIsAromatic() is carbon_arom


class TestRecoveredName:
    """Whole-molecule: the guard must no longer destroy these names.

    Slow -- constructs the namer and (via retained_substitution) an OPSIN JVM.
    """

    @pytest.mark.slow
    @pytest.mark.opsin_gate
    def test_diaminothiadiazolopyrrole_is_named_again(self):
        # Regression row from the a phase 400-row re-measurement: Task 5's
        # guard turned this correct, InChIKey-exact name into an abstention.
        #
        # Requires the OPSIN validity gate (conftest disables it suite-wide).
        # With the gate OFF an earlier, malformed candidate ('azole-3,4-
        # diamine') survives and is emitted before the bicyclo producer is
        # reached -- so this assertion is only meaningful gate-ON, which is
        # also the configuration the breadth harness measures.
        from orthonym import Orthonym

        # Since cb3bf9850 (2026-10-08, "fused ring systems with ring heteroatoms are named
        # by fusion nomenclature ") this 5/5 ortho-fused bicycle is named by
        # fusion nomenclature, not as the von Baeyer '6-thia-1,7-diazabicyclo[3.3.0]octa-
        # 2,4,7-triene-3,4-diamine' this row used to pin (whose spelling had itself been
        # corrected for the (a) elision, 'triene' before the consonant-initial
        # 'diamine'). "Five-membered ring requirement" (the Blue Book):
        # "Fusion nomenclature gives preferred IUPAC names only to compounds having at least
        # two rings of at least five or more members... When fusion names are not allowed,
        # unsaturated von Baeyer ring system names are preferred IUPAC names" -- here the
        # fusion name is allowed (two five-membered rings), and the seniority order of
        # (:23843) ranks "fused ring systems > bridged fused systems > non-fused
        # bridged systems" (von Baeyer), so the fusion name is the preferred one.
        # OPSIN 2.9.0 reads 'pyrrolo[1,2-d][1,2,4]thiadiazole-6,7-diamine' back to the
        # input's full InChIKey (an InChIKey); the parent has no symmetry, so
        # its fusion numbering (and so the locants 6,7 OPSIN resolves to the two C-NH2 atoms)
        # does not depend on the substituents.
        name = Orthonym(**BEST_EFFORT).name("C1=C(C(=C2N1C=NS2)N)N")
        assert name == "pyrrolo[1,2-d][1,2,4]thiadiazole-6,7-diamine"

    @pytest.mark.slow
    def test_inline_suffix_locant_is_found_for_an_aromatic_amine(self):
        # Unit-level counterpart: _inline_suffix_locant must return a locant
        # (not None) once the attachment atom is the ring carbon.
        from orthonym.assembly.general_engine import _inline_suffix_locant

        # match = (N=8, c=2); the ring carbon 2 is on the parent, N is not.
        loc = _inline_suffix_locant(
            "aromatic_amine", (8, 2), {1, 2, 3, 4, 5},
            {1: 1, 2: 3, 3: 4, 4: 5, 5: 7},
        )
        assert loc == 3

    @pytest.mark.slow
    def test_inline_suffix_locant_is_found_for_a_phenol(self):
        from orthonym.assembly.general_engine import _inline_suffix_locant

        # match = (O=9, c=4); ring carbon 4 is on the parent, O is not.
        loc = _inline_suffix_locant(
            "phenol", (9, 4), {2, 4, 6},
            {2: 1, 4: 2, 6: 5},
        )
        assert loc == 2

    @pytest.mark.slow
    def test_inline_suffix_locant_still_refuses_a_genuinely_off_parent_group(
        self,
    ):
        # Task 5's fix must survive: when NEITHER atom of the match is on the
        # parent the guard still returns None.
        from orthonym.assembly.general_engine import _inline_suffix_locant

        assert _inline_suffix_locant(
            "phenol", (9, 4), {2, 6}, {2: 1, 6: 5}
        ) is None
