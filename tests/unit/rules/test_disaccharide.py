"""RED unit/integration tests for the disaccharide regime (a phase, -04).

The disaccharide assembler (`rules/oligosaccharides.name_disaccharide`,)
reasons over the WHOLE multi-ring sugar structure: it detects the sugar units
and the inter-unit glycosidic bond, splits each unit (FragmentOnBonds + cap the
anomeric carbon with -OH), names each via the catalog / systematic-mono engine
(per-unit anomer from that unit's OWN ring, — never inherited), detects the
reducing end to choose the name shape (: free hemiacetal -OH -> glycosylglycose
`-ose` parent; none free -> glycosyl glycoside `-oside`), derives the (1->n)
linkage locants (, ASCII arrow per Pitfall 6), enforces the no-silent-drop
completeness invariant over the ORIGINAL mol's heavy atoms + the shared bridging
O (, Pitfall 7), and fails closed  on anything out of scope (branched,
trisaccharide non-linear, C-glycoside, polymeric).

WAVE 0 CONTRACT (mirror tests/unit/rules/test_conjugate_controller.py): imports
of the not-yet-built `name_disaccharide` go INSIDE each test body, NOT at module
level, so `pytest --collect-only` succeeds while the engine is RED at run time
until Wave-2 (Plan 183-02) lands. `RDLogger.DisableLog("rdApp.*")` at module top.

Root-cause-only (the contributor guide): assertions are structural (unit recognition,
reducing-end detection, completeness invariant, fail-closed None) — the (1->4)
arrow and trailing-parent checks pin the emitted form, not a string transform.

Verified this session (OPSIN-RT True, ASCII descriptors + ASCII arrow):
  beta-maltose -> α-D-glucopyranosyl-(1->4)-β-D-glucopyranose (W6B-T10:
             a DEFINED reducing-end anomer is cited,; only an
             UNSPECIFIED reducing anomer is omitted,)
  sucrose -> β-D-fructofuranosyl α-D-glucopyranoside (no free hemiacetal)
"""

import pytest
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Verified disaccharide SMILES (OPSIN-RT True this session)
# ---------------------------------------------------------------------------
# beta-Maltose: alpha-(1->4) glucosyl-glucose; the reducing C1 is a DEFINED beta
# anomer (W6B-T10: cited per, previously wrongly dropped).
MALTOSE_SMILES = (
    "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O)O[C@@H]2CO)"
    "[C@H](O)[C@@H](O)[C@@H]1O"
)
# Maltose with an UNSPECIFIED reducing anomer (reducing-C1 stereo removed): the
# anomer is correctly omitted (mutarotation,).
MALTOSE_UNSPEC_SMILES = (
    "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)C(O)O[C@@H]2CO)"
    "[C@H](O)[C@@H](O)[C@@H]1O"
)
# Sucrose: no free hemiacetal (both anomeric carbons in the glycosidic linkage)
# -> glycosyl glycoside. NOTE (Plan 183-02): the SMILES previously copied from
# retained_names.py:350 is stereochemically INCORRECT sucrose (its fructose ring
# is not D-fructofuranose: InChI.../m0 vs real sucrose.../m1) — it round-trips
# to neither real sucrose nor "β-D-fructofuranosyl α-D-glucopyranoside".
# Corrected here to the structurally-correct sucrose (PubChem CID 5988), which the
# assembler derives + OPSIN-round-trips to the expected systematic name. The buggy
# retained_names.py entry is a separate (trivial-name-path) defect, logged for a
# later phase.
SUCROSE_SMILES = (
    "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)"
    "[C@@H](O)[C@@H]1O"
)

MALTOSE_EXPECTED = "α-D-glucopyranosyl-(1->4)-β-D-glucopyranose"
SUCROSE_EXPECTED = "β-D-fructofuranosyl α-D-glucopyranoside"


@pytest.mark.integration
class TestDisaccharide:
    """-04 disaccharide / oligosaccharide naming."""

    def test_glycosylglycose(self):
        """Maltose names as the glycosylglycose form (free hemiacetal,)."""
        from orthonym.rules.oligosaccharides import name_disaccharide

        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        assert mol is not None
        assert name_disaccharide(mol) == MALTOSE_EXPECTED

    def test_glycosyl_glycoside(self):
        """Sucrose names as the glycosyl-glycoside form (no free hemiacetal,)."""
        from orthonym.rules.oligosaccharides import name_disaccharide

        mol = Chem.MolFromSmiles(SUCROSE_SMILES)
        assert mol is not None
        assert name_disaccharide(mol) == SUCROSE_EXPECTED

    def test_unspecified_reducing_end_no_anomer(self):
        """UNSPECIFIED reducing-end anomer -> NO alpha/beta on the parent ."""
        from orthonym.rules.oligosaccharides import name_disaccharide

        mol = Chem.MolFromSmiles(MALTOSE_UNSPEC_SMILES)
        name = name_disaccharide(mol)
        assert name is not None
        # The glucose parent trails as bare "-D-glucopyranose" (no invented anomer).
        assert name.endswith("-D-glucopyranose")
        assert "α-D-glucopyranose" not in name
        assert "β-D-glucopyranose" not in name

    def test_defined_reducing_end_cites_anomer(self):
        """W6B-: a DEFINED reducing-end anomer MUST be cited."""
        from orthonym.rules.oligosaccharides import name_disaccharide

        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        assert name_disaccharide(mol) == "α-D-glucopyranosyl-(1->4)-β-D-glucopyranose"

    def test_linkage_arrow_glyph(self):
        """Linkage locant uses the ASCII (1->4) arrow matching the gold row ."""
        from orthonym.rules.oligosaccharides import name_disaccharide

        mol = Chem.MolFromSmiles(MALTOSE_SMILES)
        name = name_disaccharide(mol)
        assert name is not None
        assert "(1->4)" in name
        # Never the Unicode arrow (pin_strict_eval.normalize does not transliterate).
        assert "→" not in name

    def test_completeness_invariant(self):
        """No-silent-drop: an unclassifiable extra substituent -> None (, Pitfall 7).

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
        """Fail-closed : branched / C-glycoside / non-linear -> None."""
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
    """Wave 6b Task 11: linear reducing oligosaccharide chain +
    1->6 disaccharide; branched / non-reducing fail closed."""

    def test_maltotriose(self):
        from orthonym import name_compound
        assert name_compound(
            "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@@H](O[C@H]3[C@H](O)"
            "[C@@H](O)C(O)O[C@@H]3CO)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O"
        ) == "α-D-glucopyranosyl-(1->4)-α-D-glucopyranosyl-(1->4)-D-glucopyranose"

    def test_isomaltose_1_6(self):
        from orthonym import name_compound
        assert name_compound(
            "OC[C@H]1O[C@H](OC[C@H]2O[C@H](O)[C@H](O)[C@@H](O)[C@@H]2O)"
            "[C@H](O)[C@@H](O)[C@@H]1O"
        ) == "α-D-glucopyranosyl-(1->6)-α-D-glucopyranose"

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
    """ L3-2b: ``_extract_unit_capped`` (the multi-bond generalization of
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
    galactose,...). But the only recognizer that knows a MODIFIED sugar (e.g.
    2-acetamido-2-deoxy-glucopyranose / GlcNAc) is ``lookup_sugar``'s EXACT
    canonical-SMILES dictionary match, which an isotope-tagged SMILES can never
    hit — so every GlcNAc/GalNAc (etc.) unit in an oligosaccharide chain fails
    recognition and the whole chain abstains (measured: a phase L3-2b a trace,
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
        # real chain (L3-2b a trace row idx5 unit).
        capped_glcnac = "CC(=O)N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]1O"
        mol = Chem.MolFromSmiles(capped_glcnac)
        canon = Chem.MolToSmiles(mol)
        assert lookup_sugar(canon) == ("β", "D", "2-acetamido-2-deoxy-glucopyranose")


class TestAminoSugarOligosaccharideChain:
    """ L3-2b: a real hexasaccharide (3x β-D-galactopyranose alternating
    with 3x N-acetyl-β-D-glucosamine, terminal D-glucopyranose reducing end)
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


class TestV33Engine2DecoratedUnitVocabulary:
    """ giants-engine Engine 2 (docs/the workflow tooling/plans/2026-08-19--giants-
    engine.md): extends ``name_monosaccharide_systematic``'s modification
    vocabulary to O-sulfate / O-phosphate-MONOester / N-sulfonate esters (BB
      /), so a decorated GAG-style unit
    (heparin/heparan/chondroitin/dermatan sulfate) no longer voids an
    otherwise-nameable oligosaccharide chain.

    ⚠ VERIFIED (OPSIN 2.9.0, this session): the carbohydrate-specific ``O-``
    prefix (``6-O-sulfo-``/``6-O-phosphono-``) round-trips to the exact
    mono-ester structure; the GENERAL substitutive prefix (``(sulfooxy)``/
    ``(phosphonooxy)``) does NOT -- it makes OPSIN insert an EXTRA oxygen at
    the cited position (an additional substituent atop the position's own
    retained hydroxyl, rather than a substitution of its H), so it silently
    asserts a WRONG molecule if ever reused for a carbohydrate ring position.
    """

    def test_pure_o_ester_defers_to_name_sugar_ester(self):
        """A PURE O-sulfate/O-phosphate ester (no co-occurring deoxy/amino/
        uronic/halo/N-sulfonate) must DECLINE here -- it is the EXISTING,
        already-PIN-tested ``name_sugar_ester``'s job (the BB /
        .1.3 FUNCTIONAL-CLASS suffix form, ``"D-glucopyranose 6-(dihydrogen
        phosphate)"``/``"α-D-glucopyranose 2-(hydrogen sulfate)"``,
        dispatched ahead of this engine). This engine's O- PREFIX form fires
        ONLY when a co-occurring modification makes the suffix form
        inexpressible (test_o_sulfate_plus_uronic_acid_co_occurrence,
        test_n_sulfonate_glucosamine below) -- regression-tested directly:
        see ``TestSugarPhosphateSulfateEster`` in test_systematic_carbohydrate.py."""
        from orthonym.data.sugar_names import name_monosaccharide_systematic

        sulfate = Chem.MolFromSmiles("O[C@H]1O[C@H](COS(=O)(=O)O)[C@@H](O)[C@H](O)[C@H]1O")
        assert name_monosaccharide_systematic(sulfate) is None
        phospho = Chem.MolFromSmiles("O[C@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H](O)[C@H]1O")
        assert name_monosaccharide_systematic(phospho) is None

    def test_n_sulfonate_glucosamine(self):
        from orthonym.data.sugar_names import name_monosaccharide_systematic

        # N-sulfo-α-D-glucosamine (heparin's characteristic GlcNS unit).
        mol = Chem.MolFromSmiles(
            "O[C@H]1O[C@H](COS(=O)(=O)O)[C@@H](O)[C@H](O)[C@H]1NS(=O)(=O)O"
        )
        assert name_monosaccharide_systematic(mol) == (
            "2-deoxy-6-O-sulfo-2-(sulfoamino)-α-D-glucopyranose"
        )

    def test_uronic_acid_with_o_sulfate_co_occurs(self):
        """O-sulfo MUST co-occur with a uronic acid (heparin's IdoA-2-sulfate /
        GlcA-2-sulfate units) -- only the deoxy/amino/halo+uronic combination
        stays vetoed (DEFERRED: KDO/ulosonic uronic+deoxy)."""
        from orthonym.data.sugar_names import name_monosaccharide_systematic

        mol = Chem.MolFromSmiles(
            "O[C@@H]1O[C@@H](C(=O)O)[C@@H](O)[C@H](OS(=O)(=O)O)[C@H]1O"
        )
        name = name_monosaccharide_systematic(mol)
        assert name is not None
        assert "O-sulfo" in name and "uronic acid" in name

    def test_n_acetyl_plus_sulfate_stays_out_of_scope(self):
        """N-acyl is explicitly out of scope for this classifier (a DIFFERENT,
        pre-existing regime, -- must not silently drop the acyl
        group. No regression: still declines (never a wrong/partial name)."""
        from orthonym.data.sugar_names import name_monosaccharide_systematic

        mol = Chem.MolFromSmiles(
            "CC(=O)N[C@@H]1[C@@H](O)[C@@H](O)[C@@H](COS(=O)(=O)O)O[C@H]1O"
        )
        assert name_monosaccharide_systematic(mol) is None


class TestV33Engine2CappedTerminusChain:
    """ giants-engine Engine 2 fix (b): the oligosaccharide chain's reducing
    end may be capped with a simple alkyl/aryl glycoside (e.g. a methyl
    glycoside used as a synthetic capping group in a GAG fragment), not only a
    free hemiacetal. ``_oligo_topology`` accepts this terminus and
    ``name_linear_oligosaccharide`` renders it via the glycoside
    form (``"{cap} n-O-[...]-{glycoside}"``), never the free-sugar ``-ose``
    form. VERIFIED (OPSIN 2.9.0): the ``(1->c')`` suffix chain does NOT combine
    with a glycoside head (``"glycosyl-(1->c')-methyl α-D-
    glucopyranoside"`` fails to parse); the ``n-O-[...]`` prefix form does.
    """

    def test_methyl_glycoside_disaccharide(self):
        from orthonym import name_compound

        # methyl 3-O-β-D-glucopyranosyl-α-D-glucopyranoside (single
        # donor -> no enclosing marks needed,: a bare glycosyl prefix
        # carries no internal locant/parenthetical of its own).
        smi = (
            "CO[C@H]1O[C@H](CO)[C@@H](O)[C@H](O[C@@H]2O[C@H](CO)[C@@H](O)"
            "[C@H](O)[C@H]2O)[C@H]1O"
        )
        name = name_compound(smi)
        assert name == "methyl 3-O-β-D-glucopyranosyl-α-D-glucopyranoside"

    def test_heparin_pentasaccharide_methyl_glycoside(self):
        """A real heparin-fragment pentasaccharide (methyl-capped GlcNS6S --
        IdoA2S -- GlcNS3,6S -- GlcA -- GlcNS6S, all-N-sulfonate/O-sulfate, no
        N-acyl) -- the exact a trace positive that motivated this engine. Full
        InChIKey round-trip verified this session (an InChIKey)."""
        from orthonym import name_compound
        from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

        smi = (
            "CO[C@H]1O[C@H](COS(=O)(=O)O)[C@@H](O[C@@H]2O[C@@H](C(=O)O)"
            "[C@@H](O[C@H]3O[C@H](COS(=O)(=O)O)[C@@H](O[C@@H]4O[C@H](C(=O)O)"
            "[C@@H](O[C@H]5O[C@H](COS(=O)(=O)O)[C@@H](O)[C@H](O)"
            "[C@H]5NS(=O)(=O)O)[C@H](O)[C@H]4O)[C@H](OS(=O)(=O)O)"
            "[C@H]3NS(=O)(=O)O)[C@H](O)[C@H]2OS(=O)(=O)O)[C@H](O)"
            "[C@H]1NS(=O)(=O)O"
        )
        name = name_compound(smi)
        assert name is not None and "unknown" not in name
        result = opsin_roundtrip_check(smi, name)
        assert result["passed"] is True, (name, result)


class TestV33Task1p2DecoratedUnitClassification:
    """.2 (breadth-lever program, a phase): ``_classify_units`` (and
    the shared per-unit cascade every namer in this module goes through --
    ``_recognize_unit`` plus the 3 identical inline copies in
    ``name_linear_oligosaccharide`` / ``name_branched_oligosaccharide`` /
    ``name_nonreducing_oligosaccharide``) used to fail to segment a DECORATED
    ring: a sulfo/O-acyl/N-acyl substituent on a unit made ``lookup_sugar`` /
    ``recognize_sugar_skeleton`` / the (undecorated-only) call to
    ``name_monosaccharide_systematic`` all miss, so the whole unit -- and
    therefore the whole chain -- returned ``None``.

    Root cause (a trace, verified on the ``sulfo-27``/``sulfo-32`` backlog
    witnesses below before this fix): ``_classify_sugar_positions``
    (``data/sugar_names.py``) silently treated a plain-acyl ring oxygen as a
    bare hydroxyl (never idealized, so the clean-parent fingerprint never
    matched) and explicitly failed closed on ANY N-acyl nitrogen (documented
    "N-acyl / N-alkyl / charged N -> out of scope"); separately, even the
    ALREADY-classified O-sulfo/O-phospho case deferred to
    ``name_sugar_ester``'s two-word FUNCTIONAL-CLASS form
    ("β-D-galactopyranose 6-(hydrogen sulfate)") whenever the ester-
    stripped residual was an exact-catalog sugar -- a form that cannot be
    embedded as a glycosyl unit inside a larger disaccharide/oligosaccharide
    name at all.

    Fix: ``_classify_sugar_positions``/``_idealize_to_parent`` now also
    recognize + physically idealize a plain O-acyl ester and a plain N-acyl
    amide (new ``o_acyl``/``n_acyl`` modification categories, alongside the
    existing o_sulfo/o_phospho/n_sulfo/deoxy/amino/uronic/halo ones), and
    ``name_monosaccharide_systematic`` gained an opt-in
    ``for_glycosidic_unit=True`` parameter (passed ONLY by the 4 oligo-
    saccharide per-unit call sites in this module) that (a) threads O-acyl/
    N-acyl into the assembled name as a PREFIX (``6-O-acetyl-``/
    ``2-acetamido-``-style) instead of forcing an early ``None``, and (b)
    skips the catalog-residual "defer to name_sugar_ester" check so a
    sulfo/phospho decoration ALSO composes as a prefix rather than the
    standalone functional-class form. Every other existing caller (the
    default ``for_glycosidic_unit=False``) is BYTE-IDENTICAL: an O-acyl/
    N-acyl-decorated free-standing sugar still returns ``None`` (Task 1.4's
    scope), proven by ``TestV33Engine2DecoratedUnitVocabulary`` above and
    ``test_systematic_carbohydrate.py`` staying fully green.

    A SECOND, independent bug surfaced by the SULFO witnesses (not itself an
    O-acyl/N-acyl issue): ``_name_disaccharide_binary``'s glycosylglycose
    assembly relaxed (stripped the chirality of) the reducing-end anomeric
    centre whenever the unit tuple's own ``anomer``/``config`` fields were
    empty -- true for EVERY ``name_monosaccharide_systematic`` result, since
    that engine folds a KNOWN anomer/config into the base string itself
    rather than returning them as separate fields. Relaxing a centre the
    name explicitly asserts (e.g. "...-6-O-sulfo-β-D-glucopyranose")
    makes the OPSIN round-trip fail (the parsed structure has that centre
    DEFINED; the relaxed comparison target does not) -- fixed by relaxing
    only when the anomeric centre is genuinely UNDEFINED on the input
    structure, never merely because the tuple fields are empty.

    Provenance: ``sulfo-27``/``sulfo-32`` are REAL rows from the
    breadth-lever glycan backlog (``glycan_413.json``, HA 27 / HA 32) --
    genuine two-real-sugar-ring disaccharides, hand-verified (not the
    heuristic ``sugar_glycan`` family tag, which independently confirmed
    false-positives on coumarins/furanones/lactone-bearing terpenoids in
    this same backlog). A corpus-wide search of all 413 rows for a CLEAN
    (non-aglycone-entangled) two-real-sugar-ring O-acyl or N-acyl disaccharide
    found none -- every O-acyl/N-acyl "sugar_glycan"-tagged multi-ring row is
    either a single decorated sugar attached to a macrolide/steroid aglycone
    (not a second monosaccharide ring) or a giant peptide-glycoside hybrid,
    neither of which this module's 2-real-sugar-ring cascade targets. The
    ``o_acyl``/``n_acyl`` recognizer tests below therefore use two small,
    chemically ordinary CONSTRUCTED disaccharides (a 6-O-acetylated maltose
    analog; a maltose analog whose reducing end is a non-catalog N-acyl
    amino sugar, N-propanoyl rather than GlcNAc's own catalogued N-acetyl,
    so the test actually exercises the new code path instead of the
    catalog short-circuit) to isolate and prove each decoration axis --
    every SMILES here is independently OPSIN-full-InChIKey verified via
    ``verify_or_none``, never asserted by construction.
    """

    # Real backlog witness (HA 27): D-galactopyranosyl-(1->4)-6-O-sulfo-D-
    # glucopyranose -- a genuine sulfated disaccharide.
    SULFO_27_SMILES = (
        "O=S(=O)(O)OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
        "[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O"
    )
    # Real backlog witness (HA 32): a methyl glycoside disaccharide with BOTH
    # units 6-O-sulfated.
    SULFO_32_SMILES = (
        "CO[C@@H]1O[C@H](COS(=O)(=O)O)[C@@H](O[C@H]2O[C@H](COS(=O)(=O)O)"
        "[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@H]1O"
    )
    # Constructed: maltose analog with a 6-O-acetyl on the reducing-end ring.
    O_ACYL_DISACCHARIDE_SMILES = (
        "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O)O[C@H]2COC(C)=O)"
        "[C@H](O)[C@@H](O)[C@@H]1O"
    )
    # Constructed: maltose analog whose reducing-end ring is a non-catalog
    # N-acyl (N-propanoyl) amino sugar.
    N_ACYL_DISACCHARIDE_SMILES = (
        "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](NC(=O)CC)[C@H](O)O[C@H]2CO)"
        "[C@H](O)[C@@H](O)[C@@H]1O"
    )

    def test_sulfo_27_backlog_witness_round_trips(self):
        """Before this fix: ``classify_units_ok=False`` (per the Phase-0
        census), ``name_disaccharide`` returned ``None``."""
        from orthonym.rules.oligosaccharides import name_disaccharide
        from orthonym.validation.reconstruct import verify_or_none

        mol = Chem.MolFromSmiles(self.SULFO_27_SMILES)
        assert mol is not None
        name = name_disaccharide(mol)
        assert name is not None, "decorated-ring unit must now be recognized"
        assert "O-sulfo" in name
        assert verify_or_none(name, self.SULFO_27_SMILES) == name

    def test_sulfo_32_backlog_witness_round_trips(self):
        from orthonym.rules.oligosaccharides import name_disaccharide
        from orthonym.validation.reconstruct import verify_or_none

        mol = Chem.MolFromSmiles(self.SULFO_32_SMILES)
        assert mol is not None
        name = name_disaccharide(mol)
        assert name is not None
        assert name.count("O-sulfo") == 2  # both units decorated
        assert verify_or_none(name, self.SULFO_32_SMILES) == name

    def test_o_acyl_disaccharide_round_trips(self):
        from orthonym.rules.oligosaccharides import name_disaccharide
        from orthonym.validation.reconstruct import verify_or_none

        mol = Chem.MolFromSmiles(self.O_ACYL_DISACCHARIDE_SMILES)
        assert mol is not None
        name = name_disaccharide(mol)
        assert name is not None
        assert "O-acetyl" in name
        assert verify_or_none(name, self.O_ACYL_DISACCHARIDE_SMILES) == name

    def test_n_acyl_disaccharide_round_trips(self):
        from orthonym.rules.oligosaccharides import name_disaccharide
        from orthonym.validation.reconstruct import verify_or_none

        mol = Chem.MolFromSmiles(self.N_ACYL_DISACCHARIDE_SMILES)
        assert mol is not None
        name = name_disaccharide(mol)
        assert name is not None
        assert "propanamido" in name
        assert verify_or_none(name, self.N_ACYL_DISACCHARIDE_SMILES) == name

    def test_for_glycosidic_unit_default_is_byte_identical(self):
        """The new ``for_glycosidic_unit`` parameter defaults to ``False`` and
        every EXISTING caller (the free-sugar path) must be unaffected: an
        O-acyl/N-acyl-decorated standalone ring still abstains (Task 1.4's
        scope, not this one)."""
        from orthonym.data.sugar_names import name_monosaccharide_systematic

        o_acyl_mono = Chem.MolFromSmiles("OC[C@H]1O[C@H](O)[C@H](O)[C@H](O)[C@H]1OC(C)=O")
        assert name_monosaccharide_systematic(o_acyl_mono) is None
        n_acyl_mono = Chem.MolFromSmiles(
            "OC[C@H]1O[C@H](O)[C@H](NC(=O)CC)[C@H](O)[C@H]1O"
        )
        assert name_monosaccharide_systematic(n_acyl_mono) is None
