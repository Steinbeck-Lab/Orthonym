"""RED unit tests for the systematic monosaccharide engine (Phase 183, WSC-04).

The systematic-mono engine (`data/sugar_names.name_monosaccharide_systematic`,
D-01/D-04) is a *generalization* of the existing `recognize_sugar_skeleton`
fingerprint deriver: for a non-cataloged single sugar ring (deoxy / amino /
uronic) it PHYSICALLY idealizes the ring to its parent aldose/ketose skeleton
(deoxy -> add the missing exocyclic O; uronic -> reduce COOH -> CH2OH; amino ->
ring N -> O), re-runs `rdCIPLabeler.AssignCIPLabels`, looks the idealized
fingerprint up in `_SKELETON_FINGERPRINT_INDEX` to recover (anomer, config,
base), then re-applies the modifications as detachable prefixes / the uronic
suffix with derived locants. It is fail-closed (D-11): any out-of-scope ring
returns None -> existing pipeline.

WAVE 0 CONTRACT (mirror tests/unit/rules/test_conjugate_controller.py): imports
of the not-yet-built `name_monosaccharide_systematic` (and `AMINO_SUGAR_NAMES`
for the Pitfall-3 non-catalog assertion) go INSIDE each test body, NOT at module
level, so `pytest --collect-only` succeeds while the engine is RED at run time
until Wave-1 (Plan 183-01) lands. `RDLogger.DisableLog("rdApp.*")` at module top.

Root-cause-only (the contributor guide): every assertion is structural (physical
idealization, fingerprint recovery, fail-closed None) — never string surgery on
a derived base name.

Verified this session (OPSIN-RT True, ASCII descriptors per Pitfall 6):
  deoxy  : C[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O   -> 6-deoxy-beta-D-glucopyranose
  uronic : O=C(O)[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O -> beta-D-glucopyranuronic acid
  amino  : N[C@@H]1[C@@H](O)[C@H](O)O[C@H](CO)[C@H]1O    -> 3-amino-3-deoxy-beta-D-glucopyranose
           (idealizes to beta-D-glucopyranose; NOT in AMINO_SUGAR_NAMES -> Pitfall 3 OK)
"""

import pytest
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Verified target SMILES (canonical, OPSIN-RT True this session)
# ---------------------------------------------------------------------------
# 6-deoxy-beta-D-glucopyranose (ring-CH3 instead of ring-CH2OH at C6).
DEOXY_SMILES = "C[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
# beta-D-glucopyranuronic acid (C6 oxidized -CH2OH -> -COOH); free acid form (D-10).
URONIC_SMILES = "O=C(O)[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
# 3-amino-3-deoxy-beta-D-glucopyranose: NON-cataloged amino sugar (Pitfall 3).
# Built from clean beta-D-glucopyranose by replacing the C3 exocyclic O with N;
# idealizing N->O provably returns the gluco parent. InChI-matched to OPSIN's
# parse of "3-amino-3-deoxy-beta-D-glucopyranose" this session.
AMINO_SMILES = "N[C@@H]1[C@@H](O)[C@H](O)O[C@H](CO)[C@H]1O"

# Clean parent every idealization must reproduce (Pitfall 1 / RESEARCH §1).
CLEAN_GLUCOPYRANOSE = "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"

# Open-chain aldoheptose (D-05 no-regression: the acyclic substitutive form
# emits correctly today and MUST stay byte-identical; C7+ cyclic is honest-fail
# per Assumption A3). HEAD output pinned below as the expected value.
HEPTOSE_ACYCLIC_SMILES = "OC[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H](O)C=O"
HEPTOSE_ACYCLIC_HEAD_NAME = "(2S,3S,4R,5S,6R)-2,3,4,5,6,7-hexahydroxyheptanal"


def _idealize_to_parent_inline(smiles):
    """Replicate the RESEARCH §1 physical idealization (Pitfall 1) inline.

    deoxy : add an exocyclic O on the bare terminal ring-attached CH3.
    uronic: RemoveAtom the carbonyl =O so COOH -> CH2OH.
    amino : SetAtomicNum(8) on the (single) nitrogen.
    Returns the canonical SMILES of the idealized molecule (a NEW mol so CIP
    re-runs cleanly). This proves the physical-idealization requirement; the
    Wave-1 helper (`_idealize_to_parent`) must reproduce this behaviour.
    """
    m = Chem.RWMol(Chem.MolFromSmiles(smiles))
    has_n = any(a.GetSymbol() == "N" for a in m.GetAtoms())
    has_carbonyl = any(
        b.GetBondType() == Chem.BondType.DOUBLE
        and {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} == {"C", "O"}
        for b in m.GetBonds()
    )
    if has_n:
        for a in m.GetAtoms():
            if a.GetSymbol() == "N":
                a.SetAtomicNum(8)
                break
    elif has_carbonyl:
        # uronic: reduce the ring-attached COOH carbonyl (remove the =O atom).
        cc = next(
            a.GetIdx()
            for a in m.GetAtoms()
            if a.GetSymbol() == "C"
            and sum(1 for n in a.GetNeighbors() if n.GetSymbol() == "O") == 2
        )
        dbl = next(
            b.GetOtherAtom(m.GetAtomWithIdx(cc)).GetIdx()
            for b in m.GetAtomWithIdx(cc).GetBonds()
            if b.GetBondType() == Chem.BondType.DOUBLE
        )
        m.RemoveAtom(dbl)
    else:
        # deoxy: add an exocyclic O on the bare terminal (degree-1) carbon.
        methyl = next(
            a.GetIdx()
            for a in m.GetAtoms()
            if a.GetSymbol() == "C" and a.GetDegree() == 1
        )
        o = m.AddAtom(Chem.Atom(8))
        m.AddBond(methyl, o, Chem.BondType.SINGLE)
    Chem.SanitizeMol(m)
    return Chem.MolToSmiles(m)


@pytest.mark.unit
class TestSystematicMonosaccharide:
    """WSC-04 systematic monosaccharide engine (deoxy / amino / uronic)."""

    def test_deoxy_systematic(self):
        """6-deoxy hexose names with a 'deoxy' detachable prefix, not an oxane."""
        from orthonym.data.sugar_names import name_monosaccharide_systematic

        mol = Chem.MolFromSmiles(DEOXY_SMILES)
        assert mol is not None
        name = name_monosaccharide_systematic(mol)
        assert name is not None, "deoxy sugar must name systematically, not None"
        assert "6-deoxy" in name
        assert "glucopyranose" in name
        # Never the substituted-oxane mis-name (the live defect this fixes).
        assert "oxane" not in name

    def test_uronic_free_acid(self):
        """Free uronic acid names as the OPSIN-parseable '...pyranuronic acid' (D-10)."""
        from orthonym.data.sugar_names import name_monosaccharide_systematic

        mol = Chem.MolFromSmiles(URONIC_SMILES)
        assert mol is not None
        name = name_monosaccharide_systematic(mol)
        # The D-10 free-acid form, NOT 'glucuronopyranose' (OPSIN-unparseable).
        assert name == "beta-D-glucopyranuronic acid"

    def test_amino_systematic(self):
        """A NON-cataloged amino-deoxy sugar names systematically (P-102.5.4, Pitfall 3)."""
        from orthonym.data.sugar_names import (
            AMINO_SUGAR_NAMES,
            name_monosaccharide_systematic,
        )

        # Pitfall 3: the gold target MUST NOT be one of the cataloged amino
        # sugars (those are PROTECT, not systematic targets).
        canon = Chem.CanonSmiles(AMINO_SMILES)
        catalog_keys = {Chem.CanonSmiles(k) for k in AMINO_SUGAR_NAMES}
        assert canon not in catalog_keys, (
            "amino TARGET must be non-cataloged (Pitfall 3); it is in AMINO_SUGAR_NAMES"
        )

        mol = Chem.MolFromSmiles(AMINO_SMILES)
        assert mol is not None
        name = name_monosaccharide_systematic(mol)
        assert name is not None, "non-cataloged amino sugar must name systematically"
        assert "amino" in name
        assert "deoxy" in name
        assert ("pyranose" in name) or ("furanose" in name)

    def test_idealize_to_parent_recovers_clean_fingerprint(self):
        """Physical idealization restores the clean parent skeleton (Pitfall 1).

        For each modified-sugar class the idealized canonical SMILES MUST equal
        clean beta-D-glucopyranose — proving idealization is physical (edit +
        re-CIP), not an analytic 'modification-tolerant fingerprint' (which
        FAILS because the modification re-ranks ring CIP). Where the engine
        exposes a `_idealize_to_parent` helper it is preferred; otherwise the
        inline replication asserts the same physical-idealization contract.
        """
        clean = Chem.CanonSmiles(CLEAN_GLUCOPYRANOSE)
        try:
            from orthonym.data.sugar_names import _idealize_to_parent  # type: ignore

            def idealize(smi):
                rw = _idealize_to_parent(Chem.MolFromSmiles(smi))
                return Chem.MolToSmiles(rw) if rw is not None else None
        except ImportError:
            idealize = _idealize_to_parent_inline

        for smi in (DEOXY_SMILES, URONIC_SMILES, AMINO_SMILES):
            assert idealize(smi) == clean, (
                f"idealization of {smi} did not reproduce clean beta-D-glucopyranose"
            )

    def test_heptose_acyclic_unchanged(self):
        """C7+ acyclic aldose path is byte-identical (D-05 / Assumption A3 no-reg).

        The open-chain substitutive form already emits correctly today; C7+
        cyclic is honest-fail (no _SKELETON_FINGERPRINT_INDEX entry), so the
        acyclic path MUST stay byte-identical. Pinned to the HEAD output.
        """
        from orthonym import name_compound

        assert name_compound(HEPTOSE_ACYCLIC_SMILES) == HEPTOSE_ACYCLIC_HEAD_NAME

    def test_systematic_mono_fail_closed(self):
        """Fail-closed (D-11): None for a non-sugar and an out-of-scope ring."""
        from orthonym.data.sugar_names import name_monosaccharide_systematic

        # Non-sugar (benzene) -> None.
        benzene = Chem.MolFromSmiles("c1ccccc1")
        assert name_monosaccharide_systematic(benzene) is None

        # Out-of-scope ring: a C-glycoside (anomeric carbon bonded to a ring C,
        # no anomeric exocyclic O) has no clean idealized fingerprint -> None.
        c_glycoside = Chem.MolFromSmiles(
            "OC[C@H]1O[C@@H](c2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O"
        )
        assert name_monosaccharide_systematic(c_glycoside) is None


@pytest.mark.unit
class TestCARB03ModifiedMonosaccharides:
    """v23 CARB-03: uronic-acid family completion (P-102.5.6.6), halogeno-deoxy
    (P-102.5.3), and the anomer-unspecified D/L descriptor fix. All targets were
    OPSIN-round-trip verified; imports go inside each test body (module contract).
    """

    def test_uronic_family_beyond_gluco_galacto(self):
        """All 8 aldohexose uronic acids name (was gluco/galacto only)."""
        from orthonym.data.sugar_names import name_monosaccharide_systematic

        cases = {
            # ido (L-iduronic, heparin/dermatan) — not in the old 2-stem map
            "O=C(O)[C@@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O":
                "alpha-L-idopyranuronic acid",
            # manno (D-mannuronic, alginate)
            "O=C(O)[C@H]1O[C@@H](O)[C@@H](O)[C@@H](O)[C@@H]1O":
                "beta-D-mannopyranuronic acid",
            # gulo (L-guluronic, alginate)
            "O=C(O)[C@@H]1O[C@H](O)[C@@H](O)[C@@H](O)[C@@H]1O":
                "beta-L-gulopyranuronic acid",
        }
        for smi, expected in cases.items():
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, smi
            assert name_monosaccharide_systematic(mol) == expected

    def test_gluco_galacto_uronic_unchanged(self):
        """The pre-existing uronic entries stay byte-identical (no regression)."""
        from orthonym.data.sugar_names import name_monosaccharide_systematic

        gluc = Chem.MolFromSmiles("O=C(O)[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O")
        assert name_monosaccharide_systematic(gluc) == "beta-D-glucopyranuronic acid"

    def test_halogeno_deoxy(self):
        """A ring C-OH replaced by a halogen names x-deoxy-x-halogeno (P-102.5.3)."""
        from orthonym.data.sugar_names import name_monosaccharide_systematic

        # 2-deoxy-2-fluoro-D-galactopyranose (anomer unspecified; keeps D).
        fluoro = Chem.MolFromSmiles("OC[C@H]1OC(O)[C@H](F)[C@@H](O)[C@H]1O")
        assert name_monosaccharide_systematic(fluoro) == \
            "2-deoxy-2-fluoro-D-galactopyranose"

    def test_anomer_unspecified_keeps_config(self):
        """D/L is emitted even when the anomeric configuration is undefined."""
        from orthonym.data.sugar_names import name_monosaccharide_systematic

        # 6-deoxy-D-glucopyranose with an unspecified anomeric centre: keep the D.
        deoxy = Chem.MolFromSmiles("OC1O[C@H](C)[C@@H](O)[C@H](O)[C@H]1O")
        name = name_monosaccharide_systematic(deoxy)
        assert name == "6-deoxy-D-glucopyranose", name

    def test_halo_fail_closed_on_non_ring_carbon_halogen(self):
        """A non-sugar molecule with a halogen still fails closed (None)."""
        from orthonym.data.sugar_names import name_monosaccharide_systematic

        assert name_monosaccharide_systematic(Chem.MolFromSmiles("CCCl")) is None


@pytest.mark.unit
class TestSugarPhosphateSulfateEster:
    """W6-P1 free-sugar mono-phosphate / sulfate esters (BB P-102.5.6.1.2/.1.3).

    The sugar is the parent; the ester is cited as ``<locant>-(dihydrogen
    phosphate)`` / ``<locant>-sulfate`` after the sugar name (BB 53209/53231).
    Mono-ester only; di/bis-phosphate and P-O-P bridges fail closed (183/184).
    Every emitted name is OPSIN-RT gated. Descriptors ASCII (Pitfall 6).
    """

    # BB 53209: D-glucopyranose 6-(dihydrogen phosphate) (acid form -OPO(OH)2, C6).
    GLC6P = "O=P(O)(O)OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
    # Corpus: D-mannopyranose 6-(dihydrogen phosphate) (anomer unspecified).
    MAN6P = "O=P(O)(O)OC[C@H]1OC(O)[C@@H](O)[C@@H](O)[C@@H]1O"
    # BB 53231: sugar 2-sulfate. ACID form -OSO3H -> '2-(hydrogen sulfate)';
    # BB's bare '2-sulfate' is the ionized -OSO3(-) (its 'sulfonato' alternative).
    GLC2S = "O=S(=O)(O)O[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]1O"
    # Ionized glucose 2-sulfate (-OSO3(-)) -> bare '2-sulfate' (BB 53231 verbatim).
    GLC2S_ION = "[O-]S(=O)(=O)O[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]1O"
    # BB 53213: glycosyl phosphate at C1. ACID -OPO(OH)2 -> '1-(dihydrogen
    # phosphate)'; ionized -OPO(O-)2 -> bare '1-phosphate'.
    GLC1P = "O=P(O)(O)O[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"

    def test_glucose_6_dihydrogen_phosphate(self):
        from orthonym import name_compound
        assert name_compound(self.GLC6P) == "beta-D-glucopyranose 6-(dihydrogen phosphate)"

    def test_mannose_6_dihydrogen_phosphate(self):
        from orthonym import name_compound
        assert name_compound(self.MAN6P) == "D-mannopyranose 6-(dihydrogen phosphate)"

    def test_glucose_2_hydrogen_sulfate_acid_form(self):
        from orthonym import name_compound
        assert name_compound(self.GLC2S) == "alpha-D-glucopyranose 2-(hydrogen sulfate)"

    def test_protonation_words(self):
        """The ester word is derived in-place from the acid centre's ionization
        (BB 53199): -OSO3(-) sulfate / -OSO3H hydrogen sulfate; phosphate di-anion
        / mono-anion / acid -> phosphate / hydrogen phosphate / dihydrogen phosphate."""
        from orthonym.data.sugar_names import _find_sugar_oxoacid_ester

        def word(smi):
            found = _find_sugar_oxoacid_ester(Chem.MolFromSmiles(smi))
            return found[-1] if found else None

        assert word(self.GLC2S_ION) == "sulfate"
        assert word(self.GLC2S) == "hydrogen sulfate"
        assert word("O=P([O-])([O-])OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O") == "phosphate"
        assert word("O=P([O-])(O)OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O") == "hydrogen phosphate"
        assert word(self.GLC6P) == "dihydrogen phosphate"

    def test_glucose_1_dihydrogen_phosphate_glycosyl(self):
        from orthonym import name_compound
        assert name_compound(self.GLC1P) == "alpha-D-glucopyranose 1-(dihydrogen phosphate)"

    def test_diphosphate_fails_closed(self):
        """A P-O-P (di/pyro-phosphate) bridge is out of scope -> not an ester name."""
        from orthonym.data.sugar_names import name_sugar_ester
        # beta-D-glucopyranose 6-(trihydrogen diphosphate) core (P-O-P): fail closed.
        smi = "O=P(O)(O)OP(=O)(O)OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
        mol = Chem.MolFromSmiles(smi)
        assert name_sugar_ester(mol, Chem.CanonSmiles(smi)) is None

    def test_non_sugar_fails_closed(self):
        """A plain phosphate ester with no sugar ring returns None."""
        from orthonym.data.sugar_names import name_sugar_ester
        smi = "CCOP(=O)(O)O"  # ethyl dihydrogen phosphate
        mol = Chem.MolFromSmiles(smi)
        assert name_sugar_ester(mol, Chem.CanonSmiles(smi)) is None


@pytest.mark.unit
class TestGlycosylamineAndHalide:
    """W6-P2 glycosylamine (P-102.6.1.3) + glycosyl halide (P-102.6.1.5).

    The anomeric -OH is replaced by a bare -NH2 -> ``-osylamine`` head, or by a
    single halogen -> functional-class ``<glycosyl> <halide>``. Fail-closed on
    N-substituted amines, multi-halo, and C2/non-anomeric substitution (those are
    amino/deoxy-halo sugars). Every name OPSIN-RT gated.
    """

    def test_glucosylamine(self):
        from orthonym import name_compound
        assert name_compound("N[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O") == \
            "beta-D-glucopyranosylamine"

    def test_glucosyl_bromide(self):
        from orthonym import name_compound
        assert name_compound("OC[C@H]1O[C@H](Br)[C@H](O)[C@@H](O)[C@@H]1O") == \
            "alpha-D-glucopyranosyl bromide"

    def test_c2_amino_sugar_not_misfired(self):
        """A C2-amino sugar (glucosamine) is NOT a glycosylamine -> stays catalog."""
        from orthonym import name_compound
        assert name_compound("N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]1O") == \
            "alpha-D-glucosamine"

    def test_n_substituted_amine_fails_closed(self):
        """An N-acyl anomeric amine is out of scope for the bare -osylamine head."""
        from orthonym.data.sugar_names import name_glycosylamine
        from rdkit import Chem
        # N-acetyl glycosylamine: anomeric N bears an acetyl -> not bare NH2.
        smi = "CC(=O)N[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"
        assert name_glycosylamine(Chem.MolFromSmiles(smi), Chem.CanonSmiles(smi)) is None


@pytest.mark.unit
class TestSugarOMethyl:
    """W6-P3 O-methyl (O-alkyl) ether sugars (BB P-102.5.6.1): n-O-methyl- prefix.

    Strip-and-name (F-OXANE-DROP-safe: the residual must be a recognized free
    sugar). The anomeric O-methyl is a GLYCOSIDE (methyl glucopyranoside), NOT an
    O-methyl ether -> fail-closed so the glycoside path handles it. OPSIN-RT gated.
    """

    def test_tetra_o_methyl_glucopyranose(self):
        from orthonym import name_compound
        assert name_compound("CO[C@H]1[C@H](O)O[C@@H]([C@H]([C@@H]1OC)OC)COC") == \
            "2,3,4,6-tetra-O-methyl-beta-D-glucopyranose"

    def test_single_o_methyl_glucopyranose(self):
        """A single non-anomeric O-methyl -> '3-O-methyl-…' (no double hyphen)."""
        from orthonym import name_compound
        assert name_compound("CO[C@@H]1[C@H]([C@H](O)O[C@@H]([C@H]1O)CO)O") == \
            "3-O-methyl-beta-D-glucopyranose"

    def test_2_o_methyl_rhamnopyranose(self):
        """2-O-methyl-rhamnose: rhamnopyranose is a cataloged retained sugar, so the
        residual names cleanly (no leading detachable prefix) -> the O-methyl prefix
        composes correctly."""
        from orthonym import name_compound
        assert name_compound("CO[C@@H]1[C@H](O)[C@@H](O)[C@H](C)O[C@H]1O") == \
            "2-O-methyl-alpha-L-rhamnopyranose"

    def test_anomeric_o_methyl_is_glycoside_not_o_methyl(self):
        """Methyl beta-D-glucopyranoside must NOT be named 1-O-methyl-..."""
        from orthonym.data.sugar_names import name_sugar_o_methyl
        from rdkit import Chem
        smi = "CO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"  # methyl beta-D-glucoside
        assert name_sugar_o_methyl(Chem.MolFromSmiles(smi), Chem.CanonSmiles(smi)) is None

    def test_plain_oxane_fails_closed(self):
        from orthonym.data.sugar_names import name_sugar_o_methyl
        from rdkit import Chem
        assert name_sugar_o_methyl(Chem.MolFromSmiles("CC1CCCCO1"), "CC1CCCCO1") is None


class TestW6bMultiplierElision:
    """Wave 6b Task 1 (P-63.1.2): a multiplied -ol/-one/-amine suffix on the
    generic HW/oxane namer must elide the multiplier's terminal 'a'
    (tetra+ol -> tetrol, not tetraol)."""

    def test_c_phenyl_sugar_tetrol_elision(self):
        from orthonym import name_compound
        # 2-C-phenyl sugar -> systematic oxane; -ol multiplier must elide.
        assert name_compound("OC[C@H]1O[C@@H](O)[C@](O)(c2ccccc2)[C@@H](O)[C@@H]1O") == \
            "(2R,3S,4S,5S,6R)-6-(hydroxymethyl)-3-phenyloxane-2,3,4,5-tetrol"
