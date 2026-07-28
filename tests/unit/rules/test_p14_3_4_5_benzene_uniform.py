"""P-14.3.4.5 -- uniform complete substitution on benzene (v29 Phase C tranche B, Task 1).

Governing rule chain, verbatim from ``BlueBookV2/BlueBookV2.md``:

``P-14.3.3`` "Citation of locants" (``:2869``) is **DENY BY DEFAULT** --

    "In preferred IUPAC names, if any locants are essential for defining the structure
     of the parent structure or of a unit of structure as defined by its appropriate
     enclosing marks, then all locants must be cited for the parent structure or that
     structural unit."

``P-14.3.4.5`` (``:3007``) grants the licence exercised here --

    "All locants are omitted in compounds or substituent groups in which all
     substitutable positions are completely substituted or modified, for example, by
     hydro, in the same way."

-- and its own counter-clause (``:3009``) is the tripwire --

    "In case of partial substitution or modification, all numerical prefixes must be
     indicated."

Verbatim ``(PIN)`` witnesses: ``:7625`` ``benzenehexol (PIN, P-63.1.2) (not
benzenehexaol)``; the free-valence analogue ``:3025`` ``benzenehexayl``. The three
prefix targets (``hexamethyl-`` / ``hexafluoro-`` / ``hexachlorobenzene``) are
**DERIVED** from ``:3007``, not verbatim rows -- derivation F7.

★ THE BOUNDARY, ``:54823``: "Inositols, cyclohexane-1,2,3,4,5,6-hexols, are a specific
group of cyclitols." The saturated analogue RETAINS every locant, because a cyclohexane
ring carbon has TWO substitutable H and six OH is therefore only PARTIAL substitution.
If ``OC1C(O)C(O)C(O)C(O)C1O`` ever loses its locants, the predicate is counting
positions instead of hydrogens.

MEASURED CODE PATH (validated call-spy, 2026-07-28 -- see task-pcB1-report.md):
  * the suffix target is joined at ``rules/benzene.py:3204`` inside
    ``_assemble_benzene_with_suffix`` -- CONFIRMED, 8 line hits for benzenehexol
    against 4 for the known positive ``benzene-1,2-diol``;
  * the three prefix targets are joined INLINE in ``name_substituted_benzene``
    (``:3080`` ``format_substituent_prefix`` -> ``:3085`` ``_join_benzene_prefixes``),
    **not** in ``_build_prefix_string_with_locants`` as the task brief stated -- that
    function and its sibling ``_build_prefix_string`` record ZERO hits for
    ``hexamethylbenzene`` AND zero for the prefix known positive
    ``1,4-dibromobenzene``.

Invariant 11: removing a locant can unmask something worse (in v29 a fail-closed prefix
turned a fabrication into a silent atom drop, four separate times). Every guard below
asserts the FULL emitted name, never merely that a locant vanished.
"""
import re

import pytest

from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _name(namer, smiles):
    return namer.name(smiles)


# --------------------------------------------------------------------------- #
# 1. The licence fires -- uniform complete substitution of the benzene ring    #
# --------------------------------------------------------------------------- #
class TestLicensedOmission:
    @pytest.mark.parametrize("smiles,expected,authority", [
        ("Oc1c(O)c(O)c(O)c(O)c1O", "benzenehexol", "verbatim BB:7625"),
        ("Cc1c(C)c(C)c(C)c(C)c1C", "hexamethylbenzene", "derived from BB:3007 (F7)"),
        ("Fc1c(F)c(F)c(F)c(F)c1F", "hexafluorobenzene", "derived from BB:3007 (F7)"),
        ("Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl", "hexachlorobenzene",
         "derived from BB:3007 (F7)"),
    ])
    def test_uniform_complete_benzene_omits_all_locants(
        self, namer, smiles, expected, authority
    ):
        assert _name(namer, smiles) == expected, authority


# --------------------------------------------------------------------------- #
# 2. The deny-default holds -- the mandatory tripwire set                      #
# --------------------------------------------------------------------------- #
class TestDenyByDefault:
    def test_the_boundary_cyclohexanehexol_keeps_every_locant(self, namer):
        """★ BB:54823. 12 substitutable H, 6 decorated => partial => BB:3009."""
        assert _name(namer, "OC1C(O)C(O)C(O)C(O)C1O") == \
            "cyclohexane-1,2,3,4,5,6-hexol"

    def test_partial_ring_substitution_keeps_the_pentol_locants(self, namer):
        """5 OH + 1 Cl on benzene: complete, but NOT 'in the same way' => BB:3009.

        Asserted structurally, NOT as an exact string: the current emission
        ``1-chlorobenzene-2,3,4,5,6-pentol`` is itself a non-PIN (defect N1 -- the
        principal characteristic group must get the lower locant set, so
        ``6-chlorobenzene-1,2,3,4,5-pentol`` is correct). Pinning the exact string
        would enshrine the non-PIN, which is the trap tranche A fell into.
        """
        got = _name(namer, "Oc1c(O)c(O)c(O)c(O)c1Cl")
        assert "pentol" in got, got
        assert re.search(r"\d(?:,\d){4}-pentol", got), \
            f"the five -ol locants must all be cited: {got!r}"
        assert "chloro" in got, f"the chlorine must not be dropped: {got!r}"

    def test_heterogeneous_complete_substitution_keeps_locants(self, namer):
        """All six positions substituted but not 'in the same way' (BB:3007)."""
        got = _name(namer, "Cc1c(C)c(C)c(C)c(C)c1Cl")
        assert got == "1-chloro-2,3,4,5,6-pentamethylbenzene", got

    @pytest.mark.parametrize("smiles,expected,why", [
        ("Oc1ccccc1O", "benzene-1,2-diol", "partial: 2 of 6"),
        ("Brc1ccc(Br)cc1", "1,4-dibromobenzene", "partial: 2 of 6"),
        ("Oc1ccccc1", "phenol", "retained name pre-empts (constraint 10)"),
        ("Oc1ccccc1Cl", "2-chlorophenol", "partial"),
        ("C1CCCc2ccccc12", "1,2,3,4-tetrahydronaphthalene", "partial hydro"),
        ("C1CCC2CCCCC2C1", "decahydronaphthalene",
         "complete hydro -- already correct, proves agreement with a shipped L5 case"),
        ("c1ccccc1-c1ccccc1", "1,1'-biphenyl", "ring assembly always cites"),
        ("c1ccccc1Oc1ccccc1", "1,1'-oxydibenzene", "multiplicative always cites"),
    ])
    def test_tripwire_names_are_unchanged(self, namer, smiles, expected, why):
        assert _name(namer, smiles) == expected, why


# --------------------------------------------------------------------------- #
# 3. Predicate level -- the benzene licence helper itself                      #
# --------------------------------------------------------------------------- #
class TestBenzeneLicenceHelper:
    """Direct tests of ``benzene._benzene_l5_uniform_licence``.

    The end-to-end tests above cannot distinguish "the licence declined" from "some
    other handler won", so the predicate is also pinned directly.
    """

    def _lic(self, smiles, suffixes, prefixes, stereo=()):
        from rdkit import Chem

        from orthonym.rules.benzene import _benzene_l5_uniform_licence
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        ring = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()][:6]
        return _benzene_l5_uniform_licence(
            mol, ring, suffixes, prefixes, stereo)

    def test_six_ol_licensed(self):
        assert self._lic("Oc1c(O)c(O)c(O)c(O)c1O",
                         {"ol": [1, 2, 3, 4, 5, 6]}, {}) is True

    def test_six_methyl_licensed(self):
        assert self._lic("Cc1c(C)c(C)c(C)c(C)c1C",
                         {}, {"methyl": [1, 2, 3, 4, 5, 6]}) is True

    def test_mixed_kinds_denied(self):
        assert self._lic("Cc1c(C)c(C)c(C)c(C)c1Cl",
                         {}, {"methyl": [1, 2, 3, 4, 5], "chloro": [6]}) is False

    def test_suffix_plus_prefix_denied(self):
        assert self._lic("Oc1c(O)c(O)c(O)c(O)c1Cl",
                         {"ol": [1, 2, 3, 4, 5]}, {"chloro": [6]}) is False

    def test_partial_denied(self):
        assert self._lic("Oc1ccccc1O", {"ol": [1, 2]}, {}) is False

    def test_stereodescriptor_denies(self):
        """P-14.3.3: one essential locant in the scope restores every locant."""
        assert self._lic("Cc1c(C)c(C)c(C)c(C)c1C",
                         {}, {"methyl": [1, 2, 3, 4, 5, 6]},
                         stereo=[{"locant": 1, "descriptor": "E"}]) is False

    def test_non_aromatic_six_ring_denied(self):
        """★ The licence is evaluated against BENZENE's parent hydride, so it must
        first prove the parent IS a benzene ring.

        Found by mutation testing: disabling the aromatic/carbon confirmation loop
        broke NO test, and the missing witness was the most important molecule in the
        whole class. A cyclohexane ring carbon has TWO substitutable H, so six
        identical substituents are only PARTIAL (BB:3009, BB:54823) -- but measured
        against benzene's parent hydride they look complete. Without this guard the
        predicate licenses ``hexamethylcyclohexane``, which is wrong.
        """
        from rdkit import Chem

        from orthonym.rules.benzene import _benzene_l5_uniform_licence
        mol = Chem.MolFromSmiles("CC1C(C)C(C)C(C)C(C)C1C")
        ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
        assert len(ring) == 6
        assert _benzene_l5_uniform_licence(
            mol, ring, {}, {"methyl": [1, 2, 3, 4, 5, 6]}, ()) is False

    def test_heteroatom_six_ring_denied(self):
        """Second witness for the same guard, for the OTHER reason it can fail:
        an aromatic six-ring that is not all-carbon. Pyridine's N is not a
        substitutable position at all, so benzene's parent hydride does not describe
        it."""
        from rdkit import Chem

        from orthonym.rules.benzene import _benzene_l5_uniform_licence
        mol = Chem.MolFromSmiles("Cc1c(C)c(C)nc(C)c1C")
        ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
        assert len(ring) == 6
        assert any(mol.GetAtomWithIdx(i).GetSymbol() == "N" for i in ring)
        assert _benzene_l5_uniform_licence(
            mol, ring, {}, {"methyl": [1, 2, 3, 4, 5, 6]}, ()) is False

    def test_non_six_ring_denied(self):
        from orthonym.rules.benzene import _benzene_l5_uniform_licence
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert _benzene_l5_uniform_licence(
            mol, [0, 1, 2, 3, 4], {}, {"methyl": [1, 2, 3, 4, 5]}) is False

    def test_duplicate_locant_denied(self):
        """Two decorations claiming one ring position -- fail closed."""
        assert self._lic("Cc1c(C)c(C)c(C)c(C)c1C",
                         {"ol": [1]}, {"methyl": [1, 2, 3, 4, 5, 6]}) is False

    def test_out_of_range_locant_denied(self):
        assert self._lic("Cc1c(C)c(C)c(C)c(C)c1C",
                         {}, {"methyl": [1, 2, 3, 4, 5, 7]}) is False

    def test_isotope_denies(self):
        """Constraint 10 -- isotopic labels (BB:44180) always cite locants."""
        assert self._lic("[13CH3]c1c(C)c(C)c(C)c(C)c1C",
                         {}, {"methyl": [1, 2, 3, 4, 5, 6]}) is False

    def test_none_mol_denied(self):
        from orthonym.rules.benzene import _benzene_l5_uniform_licence
        assert _benzene_l5_uniform_licence(
            None, [0, 1, 2, 3, 4, 5], {}, {"methyl": [1, 2, 3, 4, 5, 6]}) is False
