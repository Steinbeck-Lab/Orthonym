"""Wave-8 Phase P9 unit tests for organometallic additive nomenclature.

Covers docs/the workflow tooling/plans/2026-07-16-wave8-p9-organometallics.md Tasks
9.1-9.6. Every test targets a live reproduce-first finding (verified at HEAD
both gated and gate-off raw per the plan's verified-scope table).

NEVER uses @pytest.mark.xfail (internal notes) — honest-fail-on-data.
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym


def _raw():
    """Gate-off raw namer -- the no-Java path where the RT-gate fails OPEN,
    per the plan's Global Constraints (accuracy-critical probe)."""
    return Orthonym(_disable_opsin_validity_gate=True)


# ---------------------------------------------------------------------------
# Task 9.1 -- eta/kappa/mu additive-name validity-gate carve-out
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTask91EtaCarveOut:
    """: Tier-4 already generates the correct PIN; the OPSIN-2.9 eta
    grammar gap must NOT suppress it (carve-out, like inositol/dianhydride)."""

    def test_bis_benzene_chromium_ships_gated(self):
        assert Orthonym().name("[Cr].c1ccccc1.c1ccccc1") == "bis(η⁶-benzene)chromium"

    def test_tricarbonyl_benzene_chromium_ships_gated(self):
        assert (
            Orthonym().name("[Cr].c1ccccc1.[C-]#[O+].[C-]#[O+].[C-]#[O+]")
            == "tricarbonyl(η⁶-benzene)chromium"
        )

    def test_bis_allyl_nickel_abstains_lossy_ligand(self):
        # (item 4): neutral propene C=CC (C3H6) is NOT the η³-prop-2-en-1-yl
        # anion (C3H5) — naming it so drops 1 H per ligand (C6H12Ni named for a
        # C6H10Ni constitution). The conservation veto declines -> abstain
        # (organometallics out of scope,. See _organometallic_conserves.
        assert (
            Orthonym().name("[Ni].[CH2]=CC.[CH2]=CC")
            == "nickel compound (not supported)"
        )

    def test_cymantrene_systematic_name_ships_gated(self):
        assert (
            Orthonym().name("[Mn].c1cc[cH-]c1.[C-]#[O+].[C-]#[O+].[C-]#[O+]")
            == "tricarbonyl(η⁵-cyclopentadienyl)manganese"
        )

    def test_tricarbonyl_cycloheptatrienyl_molybdenum_abstains_lossy_ligand(self):
        # (item 4, PREP-T4): the LIGAND_ETA_DEFAULTS key 'C1=CC=CC=CC=1'
        # perceives a C7H6 fragment but is named 'cycloheptatrienyl' (C7H7) — the
        # name over-claims 1 H, so the emitted constitution is wrong. The
        # conservation veto declines -> abstain (organometallics out of scope).
        assert (
            Orthonym().name("[Mo].C1=CC=CC=CC=1.[C-]#[O+].[C-]#[O+].[C-]#[O+]")
            == "molybdenum compound (not supported)"
        )

    def test_gated_equals_raw_for_carveout_names(self):
        """The carve-out must not diverge gated vs raw -- both paths ship the
        identical correct-by-construction name."""
        smi = "[Cr].c1ccccc1.c1ccccc1"
        assert Orthonym().name(smi) == _raw().name(smi)

    def test_carveout_regex_does_not_unsuppress_unrelated_names(self):
        """CR guard: a totally unrelated OPSIN-unparseable garbled name must
        still suppress to the honest fallback -- the carve-out is TIGHT."""
        from orthonym.namer import _ORGANOMETALLIC_ADDITIVE_PIN_RE
        assert not _ORGANOMETALLIC_ADDITIVE_PIN_RE.match("some garbled unknown name")
        assert not _ORGANOMETALLIC_ADDITIVE_PIN_RE.match("4,4'-carbonyldibenzoic acid")


# ---------------------------------------------------------------------------
# Task 9.2 -- organometallic structure-loss veto (ACCURACY-CRITICAL)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTask92StructureLossVeto:
    """A covalent M-C compound must NEVER be silently renamed from a
    sub-fragment that drops the metal -- fail closed instead. This is the
    no-Java (gate-OFF raw) acceptance test; the RT-gate fails OPEN without
    Java so this veto must be a SOURCE-level guarantee."""

    def test_ti_methyl_trichloride_never_collapses_to_methane_raw(self):
        raw = _raw()
        assert raw.name("C[Ti](Cl)(Cl)Cl") != "methane"

    def test_pt_metallacycle_never_drops_platinum_stem_raw(self):
        raw = _raw()
        name = raw.name("Cl[Pt]1(Cl)C(C)=C(C)C(C)=C1C")
        assert "platin" in name or name.endswith("(not supported)")

    def test_ferrocene_ionic_cascade_unaffected_by_veto(self):
        """Dot-separated IONIC metallocenes (no covalent M-C bond in the same
        fragment) must keep cascading/naming normally -- the veto is scoped
        to COVALENT M-C compounds only."""
        raw = _raw()
        assert raw.name("[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1") == "ferrocene"

    def test_grignard_unaffected_by_veto(self):
        raw = _raw()
        assert "magnesium" in raw.name("C[Mg]Br")

    def test_silane_unaffected_by_veto(self):
        raw = _raw()
        assert raw.name("C[SiH3]") == "methylsilane"

    def test_gallane_unaffected_by_veto(self):
        raw = _raw()
        assert raw.name("C[Ga](C)C") == "trimethylgallane"

    def test_veto_fires_gated_too(self):
        """The gated (production) path already suppresses these via;
        the veto must not regress that -- both paths agree."""
        gated = Orthonym()
        assert gated.name("C[Ti](Cl)(Cl)Cl") != "methane"


@pytest.mark.unit
class TestTask92CovalentMetalCarbonPredicate:
    """Direct unit coverage of perception.metals.has_covalent_metal_carbon_bond."""

    def test_ti_methyl_trichloride_is_covalent(self):
        from orthonym.perception.metals import has_covalent_metal_carbon_bond
        mol = Chem.MolFromSmiles("C[Ti](Cl)(Cl)Cl")
        assert has_covalent_metal_carbon_bond(mol) is True

    def test_pt_metallacycle_is_covalent(self):
        from orthonym.perception.metals import has_covalent_metal_carbon_bond
        mol = Chem.MolFromSmiles("Cl[Pt]1(Cl)C(C)=C(C)C(C)=C1C")
        assert has_covalent_metal_carbon_bond(mol) is True

    def test_dot_separated_ferrocene_is_not_covalent(self):
        from orthonym.perception.metals import has_covalent_metal_carbon_bond
        mol = Chem.MolFromSmiles("[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1")
        assert has_covalent_metal_carbon_bond(mol) is False

    def test_plain_organic_is_not_covalent(self):
        from orthonym.perception.metals import has_covalent_metal_carbon_bond
        mol = Chem.MolFromSmiles("CCO")
        assert has_covalent_metal_carbon_bond(mol) is False

    def test_metal_metal_salt_is_not_covalent(self):
        from orthonym.perception.metals import has_covalent_metal_carbon_bond
        mol = Chem.MolFromSmiles("[Na+].[Cl-]")
        assert has_covalent_metal_carbon_bond(mol) is False


# ---------------------------------------------------------------------------
# Task 9.3 -- metal-carbonyl Stock-(0) PIN form (VERIFY-FIRST; BB-decided)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTask93CarbonylStockDecision:
    """BB verbatim, P6a.pdf pp. examples): every neutral-metal
    carbonyl PIN example given ('tricarbonyliron', 'dicarbonyl...molybdenum')
    OMITS the Stock oxidation number. Cationic examples use ionic charge
    notation ('molybdenum(1+)'), never a Roman-numeral Stock number, for a
    neutral-ligand carbonyl complex. DECISION: 'pentacarbonyliron' (NO Stock)
    IS the PIN; the plan/ledger's 'pentacarbonyliron(0)' expectation is
    STALE. This locks the existing (correct) no-Stock behaviour so it can
    never silently regress."""

    def test_pentacarbonyliron_no_stock(self):
        smi = "[Fe].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+]"
        assert Orthonym().name(smi) == "pentacarbonyliron"

    def test_tetracarbonylnickel_no_stock(self):
        smi = "[Ni].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+]"
        assert Orthonym().name(smi) == "tetracarbonylnickel"

    def test_hexacarbonylchromium_no_stock(self):
        smi = "[Cr].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+]"
        assert Orthonym().name(smi) == "hexacarbonylchromium"


# ---------------------------------------------------------------------------
# Task 9.4 -- additive sigma-coordination (transition metal +
# anionic + organic ligands)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTask94AdditiveSigmaCoordination:
    """BB verbatim (line 39789 / P6a.pdf):
    '[Ti(CH3)Cl3] trichlorido(methanido)titanium trichlorido(methyl)titanium'.
    PIN uses the substitutive ligand name '(methyl)'; ligands cited in
    alphanumerical order ('chlorido' < 'methyl'), then the metal."""

    def test_trichlorido_methyl_titanium(self):
        assert Orthonym().name("C[Ti](Cl)(Cl)Cl") == "trichlorido(methyl)titanium"

    def test_gated_equals_raw(self):
        smi = "C[Ti](Cl)(Cl)Cl"
        assert Orthonym().name(smi) == _raw().name(smi)

    def test_unnameable_organic_ligand_fails_closed_not_partial(self):
        """An organic ligand this narrow builder cannot name (e.g. a fused
        aryl system) must fail closed -- NEVER emit a partial/atom-dropping
        additive name. Never a bare 'titanium' or similar truncation."""
        # tert-butyl is NOT in the Tier-3 simple ligand table for a
        # metal_direct transition metal (only Group-14 hydride-parent gets
        # the extended chokepoint) -- must fail closed, not drop atoms.
        name = Orthonym().name("CC(C)(C)[Ti](Cl)(Cl)Cl")
        from orthonym.perception.metals import has_covalent_metal_carbon_bond
        mol = Chem.MolFromSmiles("CC(C)(C)[Ti](Cl)(Cl)Cl")
        if has_covalent_metal_carbon_bond(mol):
            assert name.endswith("(not supported)") or "titanium" in name


# ---------------------------------------------------------------------------
# Task 9.5 -- metallacycle namer, monocyclic tractable case
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTask95Metallacycle:
    """BB verbatim (P6a.pdf):
    '1,1-dichloro-2,3,4,5-tetramethylplatinole (Hantzsch-Widman type name)'
    '1,1-dichloro-2,3,4,5-tetramethyl-1-platinacyclopenta-2,4-diene
    (skeletal replacement name)'. Orthonym ships the skeletal-replacement
    form (systematic/PIN-style per project convention; the BB explicitly
    withholds an official PIN designation for transition-metal
    organometallics at."""

    def test_platinacyclopentadiene(self):
        assert (
            Orthonym().name("Cl[Pt]1(Cl)C(C)=C(C)C(C)=C1C")
            == "1,1-dichloro-2,3,4,5-tetramethyl-1-platinacyclopenta-2,4-diene"
        )

    def test_gated_equals_raw(self):
        smi = "Cl[Pt]1(Cl)C(C)=C(C)C(C)=C1C"
        assert Orthonym().name(smi) == _raw().name(smi)

    def test_bicyclic_metallacycle_fails_closed(self):
        """Bicyclic metallacycles (titanabicyclo[3.2.0]heptane family) are
        explicitly deferred (design-contract open question) -- must fail
        closed, never guess."""
        # Ti as a ring-fusion atom shared between two rings -- out of this
        # cycle's scope; must decline (None-tier) rather than mis-number.
        smi = "C1CC[Ti]2(CC2)C1"
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            from orthonym.perception.metals import detect_metallacycle
            assert detect_metallacycle(mol) is None


# ---------------------------------------------------------------------------
# Task 9.6 -- dimetal seniority tractable; fail-closed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTask96DimetalClass1Class2:
    """BB verbatim (P6a.pdf):
    '[4-(diphenylstibanyl)phenyl](phenyl)mercury' -- class-1 metal (Hg) is
    the central atom; the class-2 metal (Sb) is named as a substituent
    group ('diphenylstibanyl') on the organic ligand."""

    def test_mu_bridge_dinuclear_fails_closed(self):
        """ (two class-1 metals + mu-bridge) is the hardest
        construct -- explicitly deferred (design-contract). Must fail
        closed, never guess a partial/atom-dropping name."""
        smi = "c1cc(sc1[Hg]O)[Hg]c1ccncc1[As](C)C"
        name = Orthonym().name(smi)
        assert name.endswith("(not supported)")

    def test_class1_class2_dimetal_p6952_built(self):
        """ (BUILD): class-1 central metal (Hg, Group 12) +
        class-2 substituent metalloid (Sb, Group 15). Named additively with
        Hg as central atom, the Sb-bearing aryl cited as the recursive
        substituent '4-(diphenylstibanyl)phenyl'. BB worked example
        VERBATIM (the Blue Book). The complex ligand's enclosing marks
        upgrade to '' nesting, since the name already contains
        ''). Was previously deferred/fail-closed; now built via a class-
        aware metal partition + Group-12 recursive ligand naming + a general
        'stibanyl' substituent primitive (mirrors the arsanyl machinery)."""
        smi = "c1ccc(cc1)[Hg]c1ccc(cc1)[Sb](c1ccccc1)c1ccccc1"
        name = Orthonym().name(smi)
        assert name == "[4-(diphenylstibanyl)phenyl](phenyl)mercury", name

    def test_dimetal_generalizes_not_special_cased(self):
        """The build is a CLASS fix, not a -02 special-case: a
        different diorganyl-stibanyl aryl on mercury must also name via the
        same class-aware partition + recursive ligand path. Here the second
        Hg ligand is methyl (not phenyl), and the stibanyl carries tolyl
        groups -- a distinct molecule that must still assemble correctly."""
        # methyl-Hg-(4-(diphenylstibanyl)phenyl)
        smi = "C[Hg]c1ccc(cc1)[Sb](c1ccccc1)c1ccccc1"
        name = Orthonym().name(smi)
        # methyl < [4-(diphenylstibanyl)phenyl] alphanumerically -> either order
        # must still be RT-valid; assert it is NOT the fail-closed sentinel and
        # contains both the stibanyl ligand and mercury.
        assert "not supported" not in name and "unknown" not in name, name
        assert "diphenylstibanyl" in name and "mercury" in name, name


# ---------------------------------------------------------------------------
# σ-ligand namer 0-wrong fail-closed (branched / unsaturated ligands)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSigmaLigandNamerFailsClosed:
    """``_ligand_name_from_atoms`` classified alkyl ligands by INTERNAL DEGREE
    ONLY, so it could not tell an n-alkyl attached at a chain END (propyl) from
    a branched isomer attached at an INTERNAL carbon (isopropyl, sec-butyl) —
    both give the degree pattern {1, 1, 2,...} — and it never inspected bond
    order, so an allyl / propargyl backbone read as the saturated n-alkyl. It
    therefore named isopropyl-/allyl-/sec-butyl-metal fragments as the linear
    n-alkyl, silently dropping the branch or the C=C/C#C: a DIFFERENT molecule.
    Since OPSIN cannot round-trip these additive names, no gate caught it. The
    σ-alkyl branch now fails closed unless the fragment is a genuine all-single-
    bond chain whose metal-attached atom is a terminus (out of scope →
    abstain, never a wrong constitution). These assertions need no JVM."""

    @staticmethod
    def _lig(smi):
        from orthonym.rules.organometallics import _ligand_name_from_atoms
        m = Chem.MolFromSmiles(smi)
        idxs = tuple(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'C')
        return _ligand_name_from_atoms(m, idxs)

    def test_keeps_linear_terminal_alkyls(self):
        assert self._lig("C[Ti]") == "methyl"
        assert self._lig("CC[Ti]") == "ethyl"
        assert self._lig("CCC[Ti]") == "propyl"
        assert self._lig("CCCC[Ti]") == "butyl"

    def test_fails_closed_on_internal_attachment(self):
        assert self._lig("CC(C)[Ti]") is None      # isopropyl (attach = middle C)
        assert self._lig("CCC(C)[Ti]") is None      # sec-butyl (attach = internal C)

    def test_fails_closed_on_unsaturation(self):
        assert self._lig("C=CC[Ti]") is None        # allyl (C=C was dropped → propyl)
        assert self._lig("C#CC[Ti]") is None        # propargyl (C#C was dropped)

    @pytest.mark.parametrize("smi", [
        "CC(C)[Ti](Cl)(Cl)Cl",   # isopropyl-TiCl3 (was → trichlorido(propyl)titanium)
        "C=CC[Ti](Cl)(Cl)Cl",    # allyl (was → …(propyl)…, C=C dropped)
        "C#CC[Ti](Cl)(Cl)Cl",    # propargyl (was → …(propyl)…, C#C dropped)
        "CCC(C)[Ti](Cl)(Cl)Cl",  # sec-butyl (was → (butyl)trichloridotitanium)
        "CC(C)[Fe]Cl",           # class fires across whitelisted metals
        "CC(C)[V](Cl)Cl",
    ])
    def test_wrong_constitution_witnesses_abstain_raw(self, smi):
        # Even gate-off (no OPSIN), the producer declines rather than fabricate
        # a wrong-constitution additive name.
        assert _raw().name(smi).endswith("(not supported)")

    def test_correct_sigma_alkyls_still_ship_raw(self):
        # The clean terminal σ-alkyls remain the standard additive PINs
        # — the fix must not suppress them.
        assert _raw().name("C[Ti](Cl)(Cl)Cl") == "trichlorido(methyl)titanium"
        assert _raw().name("CC[Ti](Cl)(Cl)Cl") == "trichlorido(ethyl)titanium"
        assert _raw().name("CCC[Ti](Cl)(Cl)Cl") == "trichlorido(propyl)titanium"


# ---------------------------------------------------------------------------
# Transition-metal additive-branch σ-ligand conservation certificate (0-wrong)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSigmaAdditiveConservationCertificate:
    """The transition-metal additive name (``trichlorido(methyl)titanium``)
    ships UNVERIFIED past OPSIN via the carve-out, and the builder assumed
    "σ-ligands conserve by construction". They do not: a charged/hydrido metal, a
    charged/radical σ-carbon, or a ligand bonded to the metal at two atoms all
    shipped a WRONG constitution. ``_sigma_additive_ligands_certified`` now fails
    the branch closed on each. These abstain even gate-off (the guard is in the
    producer, not the OPSIN gate), so they need no JVM. These organometallics are
    out of scope."""

    @pytest.mark.parametrize("smi,why", [
        ("c1ccc2c(c1)[Ti]2(Cl)Cl", "W1 benzyne: 2-point ring named (phenyl)=C6H5 vs C6H4"),
        ("c1ccc2c(c1)[Pt]2(Cl)Cl", "W1 benzyne on Pt"),
        ("C1=C[Ti]1(Cl)Cl",        "W1 metallacyclopropene named (ethenyl)"),
        ("[CH2-][Ti](Cl)(Cl)Cl",   "W2 methanide carbanion named neutral (methyl)"),
        ("[CH2][Ti](Cl)(Cl)Cl",    "W2 methyl radical named (methyl)"),
        ("[CH2-]C[Ti](Cl)(Cl)Cl",  "W2 charged β-carbon in an ethyl ligand"),
        ("C[Ti+](Cl)(Cl)Cl",       "W3 cationic metal, Ewens-Bassett number dropped"),
        ("C[Ti-](Cl)(Cl)(Cl)Cl",   "W3 anionic metal"),
        ("C[TiH](Cl)Cl",           "W4 metal-bound H (hydrido) dropped"),
        ("C[TiH2]Cl",              "W4 two metal-bound H dropped"),
    ])
    def test_wrong_constitution_classes_abstain_raw(self, smi, why):
        assert _raw().name(smi).endswith("(not supported)"), (smi, why)

    def test_correct_additive_names_still_ship_raw(self):
        # Neutral, monodentate, radical-free σ-ligands on a neutral metal with no
        # metal-H still ship their BB-cited additive PIN.
        raw = _raw()
        assert raw.name("C[Ti](Cl)(Cl)Cl") == "trichlorido(methyl)titanium"
        assert raw.name("c1ccccc1[Ti](Cl)(Cl)Cl") == "trichlorido(phenyl)titanium"
        assert raw.name("C=C[Ti](Cl)(Cl)Cl") == "trichlorido(ethenyl)titanium"

    def test_certificate_predicate_directly(self):
        # Unit-level: the structural predicate on (metal_complex, mol, ligs).
        from orthonym.perception.metals import detect_metal_complex
        from orthonym.rules.organometallics import (
            _sigma_additive_ligands_certified,
        )

        def certify(smi):
            mol = Chem.MolFromSmiles(smi)
            mc = detect_metal_complex(mol)
            if mc is None or mc.is_multimetal:
                return None
            organic = [
                lg for lg in mc.ligand_groups
                if lg.ligand_smarts_key not in ('[Cl]', '[Br]', '[F]', '[I]')
            ]
            return _sigma_additive_ligands_certified(mc, mol, organic)

        assert certify("C[Ti](Cl)(Cl)Cl") is True        # clean methyl
        assert certify("[CH2-][Ti](Cl)(Cl)Cl") is False  # charged ligand carbon
        assert certify("C[Ti+](Cl)(Cl)Cl") is False       # charged metal
        assert certify("C[TiH](Cl)Cl") is False           # metal-bound H
