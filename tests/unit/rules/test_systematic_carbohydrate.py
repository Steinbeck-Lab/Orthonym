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
# W6B-T4: the open-chain aldoheptose now catalogs to its P-102.5.1 carbohydrate
# PIN (configurational-prefix name is preferred over the substitutive
# hexahydroxyheptanal per P-102.2.1); OPSIN-RT confirmed.
HEPTOSE_ACYCLIC_HEAD_NAME = "D-glycero-L-gulo-heptose"


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

    def test_heptose_acyclic_catalog_pin(self):
        """C7 acyclic aldoheptose: W6B-T4 catalogs it to the P-102.5.1
        configurational-prefix carbohydrate PIN (preferred over the substitutive
        hexahydroxyheptanal per P-102.2.1); OPSIN-RT confirmed.
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
            # W6B-T3: _find_sugar_oxoacid_ester now returns a LIST of ester tuples
            # (multi-phosphate support); each tuple is (...,locant, word).
            found = _find_sugar_oxoacid_ester(Chem.MolFromSmiles(smi))
            return found[0][-1] if found else None

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


class TestW6bMultiPhosphateEster:
    """Wave 6b Task 3 (P-102.5.6.1.2): 2-3 mono-phosphate esters -> bis/tris; the
    ketose (fructose) numbering fix also enables fructose mono-phosphates."""

    def test_fructose_1_6_bisphosphate(self):
        from orthonym import name_compound
        assert name_compound("P(=O)(O)(O)OCC1(O)[C@@H](O)[C@H](O)[C@H](O1)COP(=O)(O)O") == \
            "D-fructofuranose 1,6-bis(dihydrogen phosphate)"

    def test_fructose_6_phosphate_ketose_numbering(self):
        from orthonym import name_compound
        assert name_compound("OCC1(O)O[C@H](COP(=O)(O)O)[C@@H](O)[C@@H]1O") == \
            "D-fructofuranose 6-(dihydrogen phosphate)"

    def test_diphosphate_pop_still_fails_closed(self):
        from orthonym.data.sugar_names import name_sugar_ester
        smi = "OP(O)(=O)OP(=O)(O)O[C@@H]1[C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO"  # P-O-P
        assert name_sugar_ester(Chem.MolFromSmiles(smi), Chem.CanonSmiles(smi)) is None


class TestW6bDeoxyHeptoseCatalog:
    """Wave 6b Task 4: 2-deoxypentofuranose (P-102.5.3) + open-chain aldoheptose
    (P-102.5.1) catalogs. The heptose ledger label was mislabeled."""

    def test_2_deoxy_erythro_pentofuranose(self):
        from orthonym import name_compound
        assert name_compound("O[C@H]1C[C@H](O)[C@H](O1)CO") == \
            "2-deoxy-beta-D-erythro-pentofuranose"

    def test_heptose_ledger_is_glycero_ido(self):
        from orthonym import name_compound  # ledger 'D-glycero-D-gluco-heptose' label is WRONG
        assert name_compound("OC[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)C=O") == \
            "D-glycero-L-ido-heptose"

    def test_all_catalog_rows_emit_pin(self):
        from orthonym import name_compound
        from orthonym.data.sugar_names import DEOXY_PENTOSE_NAMES, HEPTOSE_NAMES
        for cat in (DEOXY_PENTOSE_NAMES, HEPTOSE_NAMES):
            for canon, (_a, _c, pin) in cat.items():
                assert name_compound(canon) == pin


class TestW6bAminoalditol:
    """Wave 6b Task 5 (P-102.5.6.5.4): 2-amino-2-deoxy-hexitols catalog. The
    ledger 'D-glucosaminitol' census SMILES has an UNDEFINED C2 -> stays
    fail-closed (no config assignable); only fully-defined forms catalog."""

    def test_amino_deoxy_glucitol(self):
        from orthonym import name_compound
        assert name_compound("N[C@@H](CO)[C@@H](O)[C@H](O)[C@H](O)CO") == \
            "2-amino-2-deoxy-D-glucitol"

    def test_amino_deoxy_galactitol(self):
        from orthonym import name_compound
        assert name_compound("N[C@@H](CO)[C@@H](O)[C@@H](O)[C@H](O)CO") == \
            "2-amino-2-deoxy-D-galactitol"

    def test_undefined_c2_census_not_amino_sugar_name(self):
        from orthonym import name_compound
        # undefined C2 -> a config cannot be assigned; must NOT ship the sugar name.
        got = name_compound("NC(CO)[C@@H](O)[C@H](O)[C@H](O)CO")
        assert got != "2-amino-2-deoxy-D-glucitol"


class TestW6bUlosonicAcid:
    """Wave 6b Task 7 (P-102.5.6.6.3): KDO/KDN/Neu5Ac/Neu5Gc catalog + a
    fail-closed veto so uncataloged 2-ulosonic acids never ship a stereo-dropped
    ring-carboxylic name."""

    def test_kdo(self):
        from orthonym import name_compound
        assert name_compound("C([C@@]1(O)C[C@@H](O)[C@@H](O)[C@H](O1)[C@H](O)CO)(=O)O") == \
            "3-deoxy-alpha-D-manno-oct-2-ulopyranosonic acid"

    def test_kdn(self):
        from orthonym import name_compound
        assert name_compound("C([C@@]1(O)C[C@H](O)[C@@H](O)[C@@H](O1)[C@H](O)[C@H](O)CO)(=O)O") == \
            "3-deoxy-alpha-D-glycero-D-galacto-non-2-ulopyranosonic acid"

    def test_neu5ac(self):
        from orthonym import name_compound
        assert name_compound("CC(=O)N[C@H]1[C@H]([C@H](O)[C@H](O)CO)OC(O)(C(=O)O)C[C@@H]1O") == \
            "N-acetylneuraminic acid"

    def test_uncataloged_ulosonic_fails_closed(self):
        from orthonym import name_compound
        # an uncataloged 2-ulosonic stereoisomer must NOT ship a stereo-dropped name.
        got = name_compound("C([C@@]1(O)C[C@H](O)[C@@H](O)[C@@H](O1)[C@H](O)[C@@H](O)CO)(=O)O")
        assert got == "unknown organic compound"

    def test_ordinary_oxane_carboxylic_unaffected(self):
        from orthonym import name_compound
        assert name_compound("OC(=O)C1CCCCO1") == "oxane-2-carboxylic acid"
        assert name_compound("OC1(C(=O)O)CCCCO1") == "2-hydroxyoxane-2-carboxylic acid"


class TestW6bAldonateAldarateEster:
    """Wave 6b Task 8 (P-102.5.6.6.2.1/.5.3): open-chain aldonate + aldarate
    partial esters; ordinary esters unaffected."""

    def test_propan_2_yl_gluconate(self):
        from orthonym import name_compound
        assert name_compound("O=C([C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO)OC(C)C") == \
            "propan-2-yl D-gluconate"

    def test_methyl_gluconate(self):
        from orthonym import name_compound
        assert name_compound("O=C([C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO)OC") == \
            "methyl D-gluconate"

    def test_aldarate_partial_ester_named(self):
        from orthonym.data.sugar_names import name_aldonate_ester
        from rdkit import Chem
        # W8-P7b.3: the aldarate partial ester is now NAMED (P-102.5.6.6.5.3). The
        # 1-vs-6 ester locant (distinct isomers) is resolved by a HARD OPSIN
        # round-trip of each candidate against the input (opsin_parse fails-CLOSED
        # w/o Java), so no wrong locant can ship. (Was previously deferred.)
        smi = "O=C([C@H](O)[C@@H](O)[C@@H](O)[C@@H](O)C(=O)O)OC"
        assert name_aldonate_ester(
            Chem.MolFromSmiles(smi), Chem.CanonSmiles(smi)
        ) == "1-methyl hydrogen L-altrarate"

    def test_ordinary_esters_unaffected(self):
        from orthonym import name_compound
        assert name_compound("CCOC(C)=O") == "ethyl acetate"
        assert name_compound("CC(O)C(=O)OC") == "methyl 2-hydroxypropanoate"


class TestW6bAminoDeoxyOpenSugar:
    """Wave 6b Task 9 (P-102.5.4.1.2): N-alkylamino / amino open-chain aldose;
    N-acyl fails closed."""

    def test_butylamino_glucose(self):
        from orthonym import name_compound
        assert name_compound("C(CCC)N[C@@H](C=O)[C@@H](O)[C@H](O)[C@H](O)CO") == \
            "2-(butylamino)-2-deoxy-D-glucose"

    def test_primary_amino_glucose(self):
        from orthonym import name_compound
        assert name_compound("N[C@@H](C=O)[C@@H](O)[C@H](O)[C@H](O)CO") == \
            "2-amino-2-deoxy-D-glucose"

    def test_n_acyl_fails_closed(self):
        from orthonym.data.sugar_names import name_amino_deoxy_open_sugar
        from rdkit import Chem
        smi = "CC(=O)N[C@@H](C=O)[C@@H](O)[C@H](O)[C@H](O)CO"  # N-acetyl -> out of scope
        assert name_amino_deoxy_open_sugar(Chem.MolFromSmiles(smi), Chem.CanonSmiles(smi)) is None


class TestW6bGlycosyloxyAglycone:
    """Wave 6b Task 12 (P-102.6.1.2): glycosyloxy on a senior aglycone; a
    non-senior aglycone (phenol/methanol) stays a glycoside (fail-closed)."""

    def test_glucosyloxy_acetophenone(self):
        from orthonym import name_compound
        # ACCURACY: previously the broken legacy (glycosyloxy)acetophenone -> unknown.
        assert name_compound("CC(=O)c1ccc(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc1") == \
            "1-[4-(beta-D-glucopyranosyloxy)phenyl]ethan-1-one"

    def test_phenyl_glucoside_stays_glycoside(self):
        from orthonym.data.sugar_names import name_glycosyloxy_aglycone
        from rdkit import Chem
        # phenol is NOT senior to hydroxy -> glycosyloxy declines (glycoside form used).
        smi = "O([C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O)c1ccccc1"
        assert name_glycosyloxy_aglycone(Chem.MolFromSmiles(smi), Chem.CanonSmiles(smi)) is None

    def test_methyl_glucoside_stays_glycoside(self):
        from orthonym.data.sugar_names import name_glycosyloxy_aglycone
        from rdkit import Chem
        smi = "CO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"
        assert name_glycosyloxy_aglycone(Chem.MolFromSmiles(smi), Chem.CanonSmiles(smi)) is None


class TestW6bCGlycosyl:
    """Wave 6b Task 13 (P-102.6.1.4): C-glycosyl on a senior aglycone. (The n-O-yl
    form P-102.6.2 is deferred fail-closed -- Orthonym names the aglycone
    'ethanoic acid' not the target 'acetic acid', and the '-n-O-yl' construction
    has no clean placeholder; it stays 'unknown', never a wrong name.)"""

    def test_c_glucosyl_phloroglucinol(self):
        from orthonym import name_compound
        assert name_compound("Oc1cc(O)c([C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)c(O)c1") == \
            "2-(beta-D-glucopyranosyl)benzene-1,3,5-triol"

    def test_plain_glucose_unaffected(self):
        from orthonym import name_compound
        assert name_compound("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O") == "beta-D-glucopyranose"

    def test_n_o_yl_named(self):
        from orthonym import name_compound
        # W8-P7b.5 (P-102.6.2): the glycosyloxy n-O-yl is now NAMED (sugar bonded
        # via a non-anomeric ring O -> <anomer>-<config>-glycopyranos-n-O-yl on the
        # senior parent). The locant is RT-verified (fail-closed w/o Java). (Was
        # previously deferred fail-closed to 'unknown'.)
        got = name_compound("OC[C@H]1O[C@@H](O)[C@H](OCC(=O)O)[C@@H](O)[C@H]1O")
        assert got == "(beta-D-galactopyranos-2-O-yl)acetic acid"


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

    def test_c_phenyl_sugar_named_as_pin(self):
        from orthonym import name_compound
        # W8-P7b.4 (P-102.5.6.3.1): the 2-C-phenyl sugar is now named as the
        # carbohydrate PIN (config + anomer RT-verified), not the systematic
        # oxane-tetrol it previously emitted.
        assert name_compound("OC[C@H]1O[C@@H](O)[C@](O)(c2ccccc2)[C@@H](O)[C@@H]1O") == \
            "2-C-phenyl-beta-D-mannopyranose"


class TestW6bSugarAcylEster:
    """Wave 6b Task 2 (P-102.5.6.1.1): free-sugar O-acyl esters -> functional-class
    ester name; anomeric O-acyl (glycosyl ester) and mixed-acyl fail closed."""

    def test_glucose_6_acetate(self):
        from orthonym import name_compound
        assert name_compound("C(C)(=O)OC[C@@H]1[C@H]([C@@H]([C@H]([C@H](O)O1)O)O)O") == \
            "beta-D-glucopyranose 6-acetate"

    def test_glucose_6_benzoate_not_hexyl(self):
        from orthonym import name_compound
        # ACCURACY: previously mis-named '(...)-hexyl benzoate' (sugar dropped).
        assert name_compound("C(C1=CC=CC=C1)(=O)OC[C@@H]1[C@H]([C@@H]([C@H]([C@H](O)O1)O)O)O") == \
            "beta-D-glucopyranose 6-benzoate"

    def test_glucopyranose_tetraacetate(self):
        from orthonym import name_compound
        assert name_compound("C(C)(=O)O[C@H]1[C@H](O)O[C@@H]([C@H]([C@@H]1OC(C)=O)OC(C)=O)COC(C)=O") == \
            "beta-D-glucopyranose 2,3,4,6-tetraacetate"

    def test_anomeric_o_acyl_fails_closed(self):
        from orthonym.data.sugar_names import _find_sugar_acyl_esters
        from rdkit import Chem
        # 1-O-acetyl (anomeric) = glycosyl ester, out of scope -> None.
        smi = "CC(=O)O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"
        assert _find_sugar_acyl_esters(Chem.MolFromSmiles(smi)) is None

    def test_mixed_acyl_fails_closed(self):
        from orthonym.data.sugar_names import _name_sugar_acyl_ester
        from rdkit import Chem
        smi = "CC(=O)O[C@H]1[C@H](O)O[C@@H](CO)[C@H](O)[C@@H]1OC(=O)c1ccccc1"
        assert _name_sugar_acyl_ester(Chem.MolFromSmiles(smi)) is None


@pytest.mark.unit
class TestV33Task1p4FreeSugarNAcylDecoration:
    """v33 Task 1.4 (BB P-102.5.4 / P-102.5.6.2): a STANDALONE free sugar
    bearing N-acyl (amide) decoration, or O-acyl COMBINED with N-acyl on the
    same ring -- the two shapes neither ``name_free_sugar`` (no functional-
    class analog for an amide) nor ``name_sugar_ester``'s O-acyl
    strip-and-recurse (hits the same fence on any N-acyl left in its
    residual) can express. Witnesses are CONSTRUCTED (non-catalog acyl words,
    mirroring Task 1.2's own precedent) because the real ``sugar_glycan``
    backlog family tag is heavily false-positive at this granularity (Task
    1.2/1.3's own finding) -- every real single-ring backlog row found by a
    structural census either already round-trips once isolated, or is
    blocked by an unrelated GIANT aglycone the general engine cannot name
    regardless of sugar recognition (a different lever, not this task's
    scope)."""

    # Free 2-deoxy-2-propanamido-D-glucopyranose analog: N-propanoyl (NOT the
    # catalogued N-acetyl/GlcNAc) at the ring nitrogen, no O-acyl. Built from
    # the catalog's own beta-D-GlcNAc SMILES with acetyl -> propanoyl.
    N_ACYL_ONLY = "CCC(=O)N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]1O"

    # Same ring, PLUS a 6-O-acetyl ester (O-acyl + N-acyl combined).
    O_ACYL_PLUS_N_ACYL = "CCC(=O)N[C@@H]1[C@@H](O)[C@H](O)[C@@H](COC(C)=O)O[C@@H]1O"

    # Furanose ring form diversity: 2-deoxy-2-propanamido-D-ribofuranose
    # analog (no O-acyl).
    N_ACYL_FURANOSE = "OC[C@H]1O[C@@H](O)[C@H](NC(=O)CC)[C@@H]1O"

    def _rt_exact(self, smi, name):
        from rdkit import Chem
        from orthonym.validation.opsin_roundtrip import opsin_parse
        want = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
        opsin_smiles = opsin_parse(name)
        assert opsin_smiles is not None, f"OPSIN could not parse {name!r}"
        got = Chem.MolToInchiKey(Chem.MolFromSmiles(opsin_smiles))
        assert got == want, f"{name!r} round-trips to a DIFFERENT molecule"

    def test_n_acyl_only_pyranose_names_and_round_trips(self):
        from orthonym import name_compound
        name = name_compound(self.N_ACYL_ONLY)
        assert name and "unknown" not in name
        assert "propanamido" in name and "deoxy" in name
        self._rt_exact(self.N_ACYL_ONLY, name)

    def test_o_acyl_plus_n_acyl_combined_names_and_round_trips(self):
        from orthonym import name_compound
        name = name_compound(self.O_ACYL_PLUS_N_ACYL)
        assert name and "unknown" not in name
        assert "propanamido" in name and "acetyl" in name
        self._rt_exact(self.O_ACYL_PLUS_N_ACYL, name)

    def test_n_acyl_furanose_names_and_round_trips(self):
        from orthonym import name_compound
        name = name_compound(self.N_ACYL_FURANOSE)
        assert name and "unknown" not in name
        assert "propanamido" in name and "furanose" in name
        self._rt_exact(self.N_ACYL_FURANOSE, name)

    def test_predicate_detects_n_acyl_only_ring(self):
        from orthonym.data.sugar_names import _has_sugar_n_acyl_amide
        from rdkit import Chem
        assert _has_sugar_n_acyl_amide(Chem.MolFromSmiles(self.N_ACYL_ONLY)) is True

    def test_predicate_false_on_clean_catalog_sugar(self):
        from orthonym.data.sugar_names import _has_sugar_n_acyl_amide
        from rdkit import Chem
        glucose = "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
        assert _has_sugar_n_acyl_amide(Chem.MolFromSmiles(glucose)) is False

    def test_predicate_false_on_nacetyl_glcnac_catalog_amide(self):
        # A CATALOG N-acetyl amide (GlcNAc) should NOT need this detector --
        # lookup_sugar already claims it; the detector being True too is
        # harmless (name_free_sugar wins first), but confirm it does not
        # mis-fire on a non-sugar / no-ring input.
        from orthonym.data.sugar_names import _has_sugar_n_acyl_amide
        from rdkit import Chem
        assert _has_sugar_n_acyl_amide(Chem.MolFromSmiles("CCO")) is False
        assert _has_sugar_n_acyl_amide(None) is False

    def test_functional_class_ester_route_unaffected(self):
        # Regression: the existing pure-O-acyl functional-class route
        # (name_sugar_ester -> _name_sugar_acyl_ester) must stay
        # byte-identical -- this new fallback is tried strictly AFTER it.
        from orthonym import name_compound
        assert name_compound("C(C)(=O)OC[C@@H]1[C@H]([C@@H]([C@H]([C@H](O)O1)O)O)O") == \
            "beta-D-glucopyranose 6-acetate"
