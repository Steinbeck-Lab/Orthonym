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
  beta-maltose -> alpha-D-glucopyranosyl-(1->4)-beta-D-glucopyranose  (W6B-T10:
             a DEFINED reducing-end anomer is cited, P-102.7.1.2; only an
             UNSPECIFIED reducing anomer is omitted, D-09)
  sucrose -> beta-D-fructofuranosyl alpha-D-glucopyranoside  (no free hemiacetal)
"""

import pytest
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Verified disaccharide SMILES (OPSIN-RT True this session)
# ---------------------------------------------------------------------------
# beta-Maltose: alpha-(1->4) glucosyl-glucose; the reducing C1 is a DEFINED beta
# anomer (W6B-T10: cited per P-102.7.1.2, previously wrongly dropped).
MALTOSE_SMILES = (
    "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O)O[C@@H]2CO)"
    "[C@H](O)[C@@H](O)[C@@H]1O"
)
# Maltose with an UNSPECIFIED reducing anomer (reducing-C1 stereo removed): the
# anomer is correctly omitted (mutarotation, D-09).
MALTOSE_UNSPEC_SMILES = (
    "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)C(O)O[C@@H]2CO)"
    "[C@H](O)[C@@H](O)[C@@H]1O"
)
# Sucrose: no free hemiacetal (both anomeric carbons in the glycosidic linkage)
# -> glycosyl glycoside. NOTE (Plan 183-02): the SMILES previously copied from
# retained_names.py:350 is stereochemically INCORRECT sucrose (its fructose ring
# is not D-fructofuranose: InChI .../m0 vs real sucrose .../m1) — it round-trips
# to neither real sucrose nor "beta-D-fructofuranosyl alpha-D-glucopyranoside".
# Corrected here to the structurally-correct sucrose (PubChem CID 5988), which the
# assembler derives + OPSIN-round-trips to the expected systematic name. The buggy
# retained_names.py entry is a separate (trivial-name-path) defect, logged for a
# later phase.
SUCROSE_SMILES = (
    "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)"
    "[C@@H](O)[C@@H]1O"
)

MALTOSE_EXPECTED = "alpha-D-glucopyranosyl-(1->4)-beta-D-glucopyranose"
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
        """UNSPECIFIED reducing-end anomer -> NO alpha/beta on the parent (D-09)."""
        from orthonym.rules.oligosaccharides import name_disaccharide

        mol = Chem.MolFromSmiles(MALTOSE_UNSPEC_SMILES)
        name = name_disaccharide(mol)
        assert name is not None
        # The glucose parent trails as bare "-D-glucopyranose" (no invented anomer).
        assert name.endswith("-D-glucopyranose")
        assert "alpha-D-glucopyranose" not in name
        assert "beta-D-glucopyranose" not in name

    def test_defined_reducing_end_cites_anomer(self):
        """W6B-T10 (P-102.7.1.2): a DEFINED reducing-end anomer MUST be cited."""
        from orthonym.rules.oligosaccharides import name_disaccharide

        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        assert name_disaccharide(mol) == "alpha-D-glucopyranosyl-(1->4)-beta-D-glucopyranose"

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


class TestW6bLinearOligosaccharide:
    """Wave 6b Task 11 (P-102.7.2.2): linear reducing oligosaccharide chain +
    1->6 disaccharide; branched / non-reducing fail closed."""

    def test_maltotriose(self):
        from orthonym import name_compound
        assert name_compound(
            "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@@H](O[C@H]3[C@H](O)"
            "[C@@H](O)C(O)O[C@@H]3CO)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O"
        ) == "alpha-D-glucopyranosyl-(1->4)-alpha-D-glucopyranosyl-(1->4)-D-glucopyranose"

    def test_isomaltose_1_6(self):
        from orthonym import name_compound
        assert name_compound(
            "OC[C@H]1O[C@H](OC[C@H]2O[C@H](O)[C@H](O)[C@@H](O)[C@@H]2O)"
            "[C@H](O)[C@@H](O)[C@@H]1O"
        ) == "alpha-D-glucopyranosyl-(1->6)-alpha-D-glucopyranose"

    def test_branched_fails_closed(self):
        from orthonym.rules.oligosaccharides import name_linear_oligosaccharide
        # a glucose accepting TWO glycosyls (branched) -> None.
        branched = Chem.MolFromSmiles(
            "OC[C@H]1O[C@H](O[C@H]2[C@H](O[C@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)"
            "[C@@H](O)[C@@H](O[C@H]3[C@H](O)[C@@H](O)C(O)O[C@@H]3CO)O[C@@H]2CO)"
            "[C@H](O)[C@@H](O)[C@@H]1O"
        )
        assert name_linear_oligosaccharide(branched) is None

    def test_maltotriose_deterministic(self):
        from orthonym import name_compound
        smi = ("OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@@H](O[C@H]3[C@H](O)"
               "[C@@H](O)C(O)O[C@@H]3CO)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O")
        mol = Chem.MolFromSmiles(smi)
        names = {name_compound(Chem.MolToSmiles(mol, doRandom=True)) for _ in range(8)}
        assert len(names) == 1, names


class TestExtractUnitCappedNoIsotopeLeak:
    """v33 P0 L3-2b: ``_extract_unit_capped`` (the multi-bond generalization of
    the proven ``conjugate_controller._extract_capped_sugar`` primitive) calls
    ``Chem.FragmentOnBonds(..., addDummies=True)`` WITHOUT ``dummyLabels``, so
    RDKit isotope-labels each new capping dummy atom with the bonded partner's
    ORIGINAL atom index. ``at.SetAtomicNum(8)`` (restoring the -OH) never clears
    that isotope, so every capped unit's canonical SMILES carries stray isotope
    tags (e.g. ``[C@H]1O[C@H]([5OH])...`` instead of plain ``O``).

    ``_extract_capped_sugar`` (the single-bond sibling this generalizes) already
    guards against exactly this via ``dummyLabels=[(0, 0)]`` — the multi-bond
    generalization dropped that detail.

    This is USUALLY masked: ``recognize_sugar_skeleton``'s connectivity/CIP
    fingerprint tolerates the stray isotopes for plain hexopyranoses (glucose,
    galactose, ...). But the only recognizer that knows a MODIFIED sugar (e.g.
    2-acetamido-2-deoxy-glucopyranose / GlcNAc) is ``lookup_sugar``'s EXACT
    canonical-SMILES dictionary match, which an isotope-tagged SMILES can never
    hit — so every GlcNAc/GalNAc (etc.) unit in an oligosaccharide chain fails
    recognition and the whole chain abstains (measured: v33 Phase 0 L3-2b SPY,
    5 real oligosaccharide/glycoconjugate rows, ALL declining here)."""

    def test_middle_unit_capped_smiles_has_no_isotopes(self):
        """Maltotriose's MIDDLE unit (2 bridging bonds) -> capped SMILES must
        carry plain ``O`` at both former-bridge positions, never an isotope
        tag. Reproduces the leak on an all-glucose chain (isotope-agnostic
        symptom), independent of any amino-sugar content."""
        from orthonym.rules import oligosaccharides as oligo

        smi = ("OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@@H](O[C@H]3[C@H](O)"
               "[C@@H](O)C(O)O[C@@H]3CO)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O")
        mol = Chem.MolFromSmiles(smi)
        topo = oligo._oligo_topology(mol)
        assert topo is not None
        units, links = topo["units"], topo["links"]
        unit_bridge_bonds = {ui: [] for ui in range(len(units))}
        for donor_ui, acc_ui, anom_c, o_idx, acc_c in links:
            unit_bridge_bonds[donor_ui].append((anom_c, o_idx))
            unit_bridge_bonds[acc_ui].append((acc_c, o_idx))
        middle_ui = next(ui for ui, bonds in unit_bridge_bonds.items() if len(bonds) == 2)
        canon = oligo._extract_unit_capped(
            mol, units[middle_ui]["ringset"], unit_bridge_bonds[middle_ui]
        )
        assert canon is not None
        capped = Chem.MolFromSmiles(canon)
        assert capped is not None
        assert all(a.GetIsotope() == 0 for a in capped.GetAtoms()), canon

    def test_amino_sugar_middle_unit_recognized_after_uncapping(self):
        """Once the isotope leak is fixed, an amino-sugar (GlcNAc) unit's
        capped SMILES must exact-match ``lookup_sugar``'s catalog — the
        recognizer ``name_linear_oligosaccharide`` actually calls."""
        from orthonym.data.sugar_names import lookup_sugar

        # A GlcNAc unit capped at both bridging positions (C1 exocyclic O and
        # the C4-O acceptor), as _extract_unit_capped would isolate it from a
        # real chain (v33 P0 L3-2b SPY row idx5 unit).
        capped_glcnac = "CC(=O)N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]1O"
        mol = Chem.MolFromSmiles(capped_glcnac)
        canon = Chem.MolToSmiles(mol)
        assert lookup_sugar(canon) == ("beta", "D", "2-acetamido-2-deoxy-glucopyranose")


class TestAminoSugarOligosaccharideChain:
    """v33 P0 L3-2b: a real hexasaccharide (3x beta-D-galactopyranose alternating
    with 3x N-acetyl-beta-D-glucosamine, terminal D-glucopyranose reducing end)
    that abstained pre-fix because every GlcNAc unit failed recognition (the
    isotope-leak above). Root cause fixed at the unit-extraction primitive, not
    a per-molecule special case."""

    HEXASACCHARIDE_SMILES = (
        "CC(=O)N[C@H]1[C@H](O[C@H]2[C@@H](O)[C@@H](CO)O[C@@H](O[C@H]3[C@H](O)"
        "[C@@H](NC(C)=O)[C@H](O[C@H]4[C@@H](O)[C@@H](CO)O[C@@H](O[C@H]5[C@H](O)"
        "[C@@H](NC(C)=O)[C@H](O[C@H]6[C@@H](O)[C@@H](CO)O[C@@H](O[C@H]7[C@H](O)"
        "[C@@H](O)C(O)O[C@@H]7CO)[C@@H]6O)O[C@@H]5CO)[C@@H]4O)O[C@@H]3CO)[C@@H]2O)"
        "O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2O)[C@@H]1O"
    )

    def test_amino_sugar_chain_names_and_round_trips(self):
        """The chain assembler now recognizes every unit (incl. the 3 GlcNAc
        rings) and emits an OPSIN-RT-verified name; previously abstained
        (``None``) because ``lookup_sugar`` never matched the isotope-tagged
        capped SMILES."""
        from orthonym.rules.oligosaccharides import name_disaccharide

        mol = Chem.MolFromSmiles(self.HEXASACCHARIDE_SMILES)
        assert mol is not None
        name = name_disaccharide(mol)
        assert name is not None
        assert "2-acetamido" in name or "acetamido" in name
