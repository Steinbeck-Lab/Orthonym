"""Wave-0 RED unit suite for the carbanion ``-ide`` emitter (Phase 184 WS-E.2).

These tests are written BEFORE the implementation (the Nyquist gate): they
assert the IUPAC-2013 Preferred IUPAC Names (PINs) for carbanions and MUST FAIL
(the assertion, not collection) on the pre-implementation HEAD. They flip to
GREEN as Plan 184-01 builds ``emit_parent_hydride_cumulative_suffix`` and wires
it into the carbanion path.

Decisions: 184-CONTEXT D-08 (new numbering call, P-72.2.2.1 lowest locant for
the ``-ide`` centre competing with unsaturation/substituents per P-31/P-14.4)
and D-09 (the gold targets + the ``methanide`` subsumption equivalence anchor).

Every assertion docstring cites the verbatim Blue Book P-number that governs it
(USER DIRECTIVE: every charged-class transform is cross-checked against the
Blue Book rule, with the citation in the test).

Blue Book sources (BlueBookV2/BlueBookV2.md):
  - Table 3.4 (line 17597): anion, loss of H+ -> suffix ``-ide``.
  - P-72.2.2.1 (lines 40900-40902): "locants identify positions of the negative
    charges"; the ``-ide`` centre takes the lowest locant.
  - line 42517: ``propan-2-ide`` worked example.
  - P-31.1.4: locant numbering; the suffix-type anion centre competes with the
    double bond for the low locant.

RED-by-design on HEAD (verified): ``CCC[CH-]CC`` -> ``hexane`` (charge dropped),
``CC(C)[CH-]C`` -> ``2-methylbutane``, ``CC([CH-])C`` -> ``unknown organic
compound``. The ``[CH3-]`` -> ``methanide`` anchor PASSES today (retained
lookup) and is the subsumption equivalence anchor (NOT xfail).
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym


@pytest.mark.unit
class TestCarbanionIde:
    """WS-E.2 carbanion ``-ide`` PIN targets (D-09). RED on HEAD by design."""

    def test_hexan_3_ide(self):
        """P-72.2.2.1: the carbanion centre takes the lowest locant over the
        re-found chain. CCC[CH-]CC is the hexan-3-ide carbanion (the negative
        carbon is C3; numbering from either end gives 3)."""
        assert Orthonym().name("CCC[CH-]CC") == "hexan-3-ide"

    def test_3_methylbutan_2_ide(self):
        """P-72.2.2.1 + P-31: the ``-ide`` centre wins the low locant (C2) and
        the methyl substituent is numbered consistently (C3). CC(C)[CH-]C is
        3-methylbutan-2-ide."""
        assert Orthonym().name("CC(C)[CH-]C") == "3-methylbutan-2-ide"

    def test_propan_2_ide(self):
        """BlueBookV2 line 42517: ``propan-2-ide`` is the worked example; the
        carbanion on the central carbon of propane takes locant 2.

        SMILES CORRECTED in Plan 184-01 (root-cause, not band-aid): the Wave-0
        scaffold used ``CC([CH-])C``, but that SMILES does NOT encode
        propan-2-ide. ``CC([CH-])C`` canonicalizes (RDKit) to ``[CH-]C(C)C`` —
        the ISObutyl divalent carbanion-radical (the ``[CH-]`` terminus carries a
        formal -1 AND a radical electron), whose chemically-correct PIN is
        ``2-methylpropan-1-ide`` and which the system fail-closes to ``unknown``
        (RDKit reads it as a carbanion+radical hybrid). The genuine propan-2-ide
        (isopropyl anion) is ``C[CH-]C`` — OPSIN-RT-confirmed (``-r``):
        ``propan-2-ide`` -> ``C[CH-]C`` (RDKit-canonical match). The cited Blue
        Book line 42517 worked example is exactly this isopropyl-anion structure.
        """
        assert Orthonym().name("C[CH-]C") == "propan-2-ide"

    def test_2_methylbutan_2_ide(self):
        """P-72.2.2.1: charge + methyl on the SAME carbon (D-09). The SMILES
        C[C-](C)CC is 2-methylbutan-2-ide: the anionic carbon bears a methyl and
        is part of a 4-carbon (butane) chain, charge at C2."""
        smi = "C[C-](C)CC"
        # Verify the intended structure: a carbanion (formal charge -1) on a
        # carbon that also carries a methyl branch, in a 4-carbon main chain.
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, "SMILES must parse"
        carbanions = [a for a in mol.GetAtoms()
                      if a.GetSymbol() == "C" and a.GetFormalCharge() == -1]
        assert len(carbanions) == 1, "exactly one carbanion centre expected"
        # the anionic carbon has 3 carbon neighbours (methyl + two chain carbons)
        c_neighbours = sum(1 for nb in carbanions[0].GetNeighbors()
                           if nb.GetSymbol() == "C")
        assert c_neighbours == 3
        assert Orthonym().name(smi) == "2-methylbutan-2-ide"

    def test_but_3_en_2_ide(self):
        """P-31.1.4: the suffix-type ``-ide`` centre is senior to the double bond
        for the low locant, and the ``-en`` ending precedes the cumulative
        ``-ide`` ending (Table 3.4). The carbanion at C2, double bond at C3 ->
        ``but-3-en-2-ide``.

        XFAIL FLIPPED + SMILES CORRECTED in Plan 184-01 (A2 resolved). The Wave-0
        scaffold used ``CC=C[CH-]C``, but that is a FIVE-carbon chain (it names
        the chemically-correct ``pent-3-en-2-ide``), not but-3-en-2-ide. The
        genuine but-3-en-2-ide (a 4-carbon chain) is ``C[CH-]C=C`` —
        OPSIN-RT-confirmed (``-r``): ``but-3-en-2-ide`` -> ``C[CH-]C=C``
        (RDKit-canonical match). On the corrected SMILES the system produces
        ``but-3-en-2-ide``: orient_chain criterion (a) gives the -ide centre
        locant 2 over the C=C at 3, and the ending order is ``en`` before ``ide``.
        """
        assert Orthonym().name("C[CH-]C=C") == "but-3-en-2-ide"

    def test_methanide_subsumption_anchor(self):
        """D-09 subsumption equivalence anchor (NOT RED): [CH3-] -> ``methanide``
        (single carbon, no locant). PASSES today via the retained-name lookup;
        the new primitive must reproduce it byte-identically. BlueBookV2 line
        41309: ``H3C- methanide (PIN)``."""
        assert Orthonym().name("[CH3-]") == "methanide"

    @pytest.mark.parametrize("smiles", [
        "[CH-]1CCCCC1",    # cyclohexanide (ring-member centre)
        "[CH-]1CCCC1",     # cyclopentanide (ring-member centre)
        "[CH-]CC1CCCCC1",  # exocyclic centre, ring otherwise absorbed into the chain
        "[CH-]C1CCCCC1",   # exocyclic centre adjacent to a ring
    ])
    def test_ring_carbanion_not_linearized(self, smiles):
        """CR-01 regression guard (Phase 184 deep code review).
        ``emit_parent_hydride_cumulative_suffix`` is an ACYCLIC parent-hydride
        primitive, but ``find_principal_chain`` linearizes a ring (it returns the
        ring carbons as a "chain"). Without the ring guards a cyclic/ring-bearing
        carbanion would be mis-named as a straight chain that round-trips to a
        DIFFERENT molecule (cyclohexanide -> ``hexan-1-ide``; ``[CH-]CC1CCCCC1``
        -> ``octan-1-yl``) — a structural error the parse-only validity gate
        cannot catch. The two guards (``center.IsInRing()`` + all-chain-atoms-
        acyclic) make the primitive decline these and fall through to legacy.
        This asserts the linearized acyclic ``hexan-*``/``octan-*``/``pentan-1-ide``
        names are NEVER produced for a ring substrate."""
        name = Orthonym().name(smiles)
        for forbidden in ("hexan-1-ide", "hexan-1-yl", "octan-1-ide",
                          "octan-1-yl", "pentan-1-ide"):
            assert forbidden not in name, (
                f"ring substrate {smiles!r} linearized to {name!r} "
                f"(contains forbidden acyclic fragment {forbidden!r})")
