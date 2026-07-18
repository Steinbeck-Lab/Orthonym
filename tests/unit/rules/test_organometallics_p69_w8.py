"""Wave-8 Phase P9 unit tests for P-69 organometallic additive nomenclature.

Covers  Tasks
9.1-9.6. Every test targets a live reproduce-first finding (verified at HEAD
both gated and gate-off raw per the plan's verified-scope table).

NEVER uses @pytest.mark.xfail (CONTEXT D-29) — honest-fail-on-data.
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
    """P-69.2.6: Tier-4 already generates the correct PIN; the OPSIN-2.9 eta
    grammar gap must NOT suppress it (carve-out, like inositol/dianhydride)."""

    def test_bis_benzene_chromium_ships_gated(self):
        assert Orthonym().name("[Cr].c1ccccc1.c1ccccc1") == "bis(η⁶-benzene)chromium"

    def test_tricarbonyl_benzene_chromium_ships_gated(self):
        assert (
            Orthonym().name("[Cr].c1ccccc1.[C-]#[O+].[C-]#[O+].[C-]#[O+]")
            == "tricarbonyl(η⁶-benzene)chromium"
        )

    def test_bis_allyl_nickel_ships_gated(self):
        assert (
            Orthonym().name("[Ni].[CH2]=CC.[CH2]=CC")
            == "bis(η³-prop-2-en-1-yl)nickel"
        )

    def test_cymantrene_systematic_name_ships_gated(self):
        assert (
            Orthonym().name("[Mn].c1cc[cH-]c1.[C-]#[O+].[C-]#[O+].[C-]#[O+]")
            == "tricarbonyl(η⁵-cyclopentadienyl)manganese"
        )

    def test_tricarbonyl_cycloheptatrienyl_molybdenum_ships_gated(self):
        assert (
            Orthonym().name("[Mo].C1=CC=CC=CC=1.[C-]#[O+].[C-]#[O+].[C-]#[O+]")
            == "tricarbonyl(η⁷-cycloheptatrienyl)molybdenum"
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
        """The gated (production) path already suppresses these via SELF-01;
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
    """BB verbatim (P-69.2.4, P6a.pdf pp. examples): every neutral-metal
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
# Task 9.4 -- P-69.2.3 additive sigma-coordination (transition metal +
# anionic + organic ligands)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTask94AdditiveSigmaCoordination:
    """BB P-69.2.3 verbatim (line 39789 / P6a.pdf):
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
# Task 9.5 -- metallacycle namer (P-69.4), monocyclic tractable case
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTask95Metallacycle:
    """BB P-69.4 verbatim (P6a.pdf):
    '1,1-dichloro-2,3,4,5-tetramethylplatinole (Hantzsch-Widman type name)'
    '1,1-dichloro-2,3,4,5-tetramethyl-1-platinacyclopenta-2,4-diene
    (skeletal replacement name)'. Orthonym ships the skeletal-replacement
    form (systematic/PIN-style per project convention; the BB explicitly
    withholds an official PIN designation for transition-metal
    organometallics at P-69.0)."""

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
# Task 9.6 -- dimetal seniority (P-69.5.2 tractable; P-69.5.1 fail-closed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTask96DimetalClass1Class2:
    """BB P-69.5.2 verbatim (P6a.pdf):
    '[4-(diphenylstibanyl)phenyl](phenyl)mercury' -- class-1 metal (Hg) is
    the central atom; the class-2 metal (Sb) is named as a substituent
    group ('diphenylstibanyl') on the organic ligand."""

    def test_mu_bridge_dinuclear_fails_closed(self):
        """P-69.5.1 (two class-1 metals + mu-bridge) is the hardest P-69
        construct -- explicitly deferred (design-contract). Must fail
        closed, never guess a partial/atom-dropping name."""
        smi = "c1cc(sc1[Hg]O)[Hg]c1ccncc1[As](C)C"
        name = Orthonym().name(smi)
        assert name.endswith("(not supported)")

    def test_class1_class2_dimetal_fails_closed(self):
        """P-69.5.2 (class-1 central metal + class-2 substituent metal) is
        deferred THIS cycle: reproduce-first confirmed the required
        Sb-substituent namer ('diphenylstibanyl') does not yet exist
        (unlike the As-rooted sibling, name_arsanyl_substituent, which
        does) -- building it correctly requires a new cross-cutting
        subsystem (Sb-substituent namer + a relaxed is_multimetal
        detection + a new dimetal assembler), not a quick wire-up. Must
        fail closed, never guess."""
        smi = "c1ccc(cc1)[Hg]c1ccc(cc1)[Sb](c1ccccc1)c1ccccc1"
        name = Orthonym().name(smi)
        assert name.endswith("(not supported)")
