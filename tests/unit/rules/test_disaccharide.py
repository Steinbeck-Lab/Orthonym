"""RED unit/integration tests for the P-102.7 disaccharide regime (Phase 183, WSC-04).

The disaccharide assembler (`rules/oligosaccharides.name_disaccharide`, D-02)
reasons over the WHOLE multi-ring sugar structure: it detects the sugar units
and the inter-unit glycosidic bond, splits each unit (FragmentOnBonds + cap the
anomeric carbon with -OH), names each via the catalog / systematic-mono engine
(per-unit anomer from that unit's OWN ring, D-09 — never inherited), detects the
reducing end to choose the name shape (D-06: free hemiacetal -OH -> glycosylglycose
`-ose` parent; none free -> glycosyl glycoside `-oside`), derives the (1->n)
linkage locants (D-07, ASCII arrow per Pitfall 6), enforces the no-silent-drop
completeness invariant over the ORIGINAL mol's heavy atoms + the shared bridging
O (D-12, Pitfall 7), and fails closed (D-11) on anything out of scope (branched,
trisaccharide non-linear, C-glycoside, polymeric).

WAVE 0 CONTRACT (mirror tests/unit/rules/test_conjugate_controller.py): imports
of the not-yet-built `name_disaccharide` go INSIDE each test body, NOT at module
level, so `pytest --collect-only` succeeds while the engine is RED at run time
until Wave-2 (Plan 183-02) lands. `RDLogger.DisableLog("rdApp.*")` at module top.

Root-cause-only (the contributor guide): assertions are structural (unit recognition,
reducing-end detection, completeness invariant, fail-closed None) — the (1->4)
arrow and trailing-parent checks pin the emitted form, not a string transform.

Verified this session (OPSIN-RT True, ASCII descriptors + ASCII arrow):
  maltose -> alpha-D-glucopyranosyl-(1->4)-D-glucopyranose  (unspecified
             reducing end -> NO alpha/beta on the glucose parent, D-09)
  sucrose -> beta-D-fructofuranosyl alpha-D-glucopyranoside  (no free hemiacetal)
"""

import pytest
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Verified disaccharide SMILES (OPSIN-RT True this session)
# ---------------------------------------------------------------------------
# Maltose: alpha-(1->4) glucosyl-glucose; reducing C1 is C(O) (unspecified anomer).
MALTOSE_SMILES = (
    "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O)O[C@@H]2CO)"
    "[C@H](O)[C@@H](O)[C@@H]1O"
)
# Sucrose (canonical from retained_names.py:350): no free hemiacetal (both
# anomeric carbons in the glycosidic linkage) -> glycosyl glycoside.
SUCROSE_SMILES = (
    "OC[C@@H]1O[C@@](CO)(O[C@H]2[C@H](O)[C@@H](O)[C@@H](O)O[C@@H]2CO)"
    "[C@@H](O)[C@H]1O"
)

MALTOSE_EXPECTED = "alpha-D-glucopyranosyl-(1->4)-D-glucopyranose"
SUCROSE_EXPECTED = "beta-D-fructofuranosyl alpha-D-glucopyranoside"


@pytest.mark.integration
class TestDisaccharide:
    """WSC-04 P-102.7 disaccharide / oligosaccharide naming."""

    def test_glycosylglycose(self):
        """Maltose names as the glycosylglycose form (free hemiacetal, D-06)."""
        from orthonym.rules.oligosaccharides import name_disaccharide

        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        assert mol is not None
        assert name_disaccharide(mol) == MALTOSE_EXPECTED

    def test_glycosyl_glycoside(self):
        """Sucrose names as the glycosyl-glycoside form (no free hemiacetal, D-06)."""
        from orthonym.rules.oligosaccharides import name_disaccharide

        mol = Chem.MolFromSmiles(SUCROSE_SMILES)
        assert mol is not None
        assert name_disaccharide(mol) == SUCROSE_EXPECTED

    def test_unspecified_reducing_end_no_anomer(self):
        """Unspecified reducing-end anomer -> NO alpha/beta on the parent (D-09)."""
        from orthonym.rules.oligosaccharides import name_disaccharide

        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        name = name_disaccharide(mol)
        assert name is not None
        # The glucose parent trails as bare "-D-glucopyranose" (no invented anomer).
        assert name.endswith("-D-glucopyranose")
        assert "alpha-D-glucopyranose" not in name
        assert "beta-D-glucopyranose" not in name

    def test_linkage_arrow_glyph(self):
        """Linkage locant uses the ASCII (1->4) arrow matching the gold row (D-07)."""
        from orthonym.rules.oligosaccharides import name_disaccharide

        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        name = name_disaccharide(mol)
        assert name is not None
        assert "(1->4)" in name
        # Never the Unicode arrow (pin_strict_eval.normalize does not transliterate).
        assert "→" not in name

    def test_completeness_invariant(self):
        """No-silent-drop: an unclassifiable extra substituent -> None (D-12, Pitfall 7).

        A disaccharide-shaped molecule where one unit carries an extra heavy-atom
        substituent the engine cannot classify (here an O-allyl ether on a ring
        carbon) violates the completeness invariant over the ORIGINAL mol's heavy
        atoms (every atom must be consumed by exactly one named unit/prefix/suffix,
        reconciling the single shared bridging O) -> honest-fail to None.
        """
        from orthonym.rules.oligosaccharides import name_disaccharide

        # Maltose with an O-allyl ether on a glucose ring carbon (unclassified).
        decorated = Chem.MolFromSmiles(
            "OC[C@H]1O[C@H](O[C@H]2[C@H](OCC=C)[C@@H](O)[C@H](O)O[C@@H]2CO)"
            "[C@H](O)[C@@H](O)[C@@H]1O"
        )
        assert decorated is not None
        assert name_disaccharide(decorated) is None

    def test_disaccharide_fail_closed(self):
        """Fail-closed (D-11): branched / C-glycoside / non-linear -> None."""
        from orthonym.rules.oligosaccharides import name_disaccharide

        # C-glycoside (aglycone bonded by C, not the inter-unit glycosidic O) and
        # a non-sugar aromatic aglycone -> out of scope -> None.
        c_glycoside = Chem.MolFromSmiles(
            "OC[C@H]1O[C@@H](c2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O"
        )
        assert name_disaccharide(c_glycoside) is None

        # A non-sugar molecule -> None.
        non_sugar = Chem.MolFromSmiles("CCO")
        assert name_disaccharide(non_sugar) is None
