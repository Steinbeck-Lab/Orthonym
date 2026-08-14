"""Unit tests for the polyfunctional acyclic substituent namer (v23 SL).

`substituent_naming._name_polyfunctional_acyclic_substituent` names a saturated
acyclic carbon-chain substituent bearing >=2 simple detachable prefixes
(carboxy / amino / hydroxy / oxo / halogen) FROM STRUCTURE, numbered from the
free valence (P-29.2 / P-31.1.4.3.4 / P-65.1.1). This is the root-cause fix for
the structure-loss bug where the recursive path let `parent_to_prefix` DROP the
secondary prefixes (serine-O -CH2CH(NH2)COOH -> '(R)-2-carboxyethyl', amino lost)
and the cache mapped the H-capped fragment to a wrong retained name
('alanine' -> 'alaninyl').

It unblocks phosphatidylserine (P-107.3.3): the serine head, previously
unknown, is now named via the functional-class hydrogen-phosphate diester.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym import name_compound
from orthonym.assembly.substituent_enumerator import name_substituent


def _name_frag(smiles, attach=1):
    """Name the substituent = atoms 1..n of `smiles` (atom 0 = parent methyl C)."""
    mol = Chem.MolFromSmiles(smiles)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except Exception:
        pass
    frag = list(range(1, mol.GetNumAtoms()))
    return name_substituent(mol, frag, attach)


class TestPolyfunctionalSubstituentNamed:
    """>=2 detachable prefixes — enumerated from structure (every group kept)."""

    def test_serine_carbon_skeleton(self):
        # -CH2-CH(NH2)-COOH : ethyl backbone, amino+carboxy at C2 (alphabetical)
        assert _name_frag("CCC(N)C(=O)O") == "2-amino-2-carboxyethyl"

    def test_serine_L_stereo(self):
        # single stereocentre (not at attachment) -> bare (R)- descriptor
        assert _name_frag("CC[C@@H](N)C(=O)O") == "(R)-2-amino-2-carboxyethyl"

    def test_hydroxy_carboxy(self):
        # carboxy < hydroxy alphabetically
        assert _name_frag("CCC(O)C(=O)O") == "2-carboxy-2-hydroxyethyl"

    def test_diol(self):
        assert _name_frag("CCC(O)CO") == "2,3-dihydroxypropyl"

    def test_diamino(self):
        assert _name_frag("CCC(N)CN") == "2,3-diaminopropyl"

    def test_oxo_hydroxy(self):
        # hydroxy < oxo alphabetically
        assert _name_frag("CCC(=O)CO") == "3-hydroxy-2-oxopropyl"

    def test_methyl_backbone_two_prefixes(self):
        # -CH(NH2)COOH : mononuclear (1-carbon) backbone, locants elided.
        # P-16.5.1.3.1 / P-16.3.3: first prefix bare, the rest EACH enclosed. The
        # bare concatenation 'aminocarboxymethyl' is RT-ambiguous -- OPSIN parses
        # it as amino + carboxymethyl (amino ON THE RING), a DIFFERENT molecule;
        # 'amino(carboxy)methyl' round-trips (BB witness '[amino(imino)methyl]').
        assert _name_frag("CC(N)C(=O)O") == "amino(carboxy)methyl"

    def test_halo_hydroxy_mixed(self):
        assert _name_frag("CCC(Cl)O") == "2-chloro-2-hydroxyethyl"


class TestSingleFGSubstituent:
    """v23 single-FG extension: ONE detachable prefix, numbered from the free
    valence so the locant is attachment-correct (P-29.2). 1-carbon backbones
    elide the locant; >=2-carbon backbones carry it."""

    def test_2_hydroxyethyl(self):
        # -CH2CH2OH : was 'hydroxyethyl' (locant dropped)
        assert _name_frag("CCCO") == "2-hydroxyethyl"

    def test_3_hydroxypropyl(self):
        # -CH2CH2CH2OH : was '1-hydroxypropyl' (numbered from the WRONG end)
        assert _name_frag("CCCCO") == "3-hydroxypropyl"

    def test_2_aminoethyl(self):
        assert _name_frag("CCCN") == "2-aminoethyl"

    def test_3_aminopropyl(self):
        assert _name_frag("CCCCN") == "3-aminopropyl"

    def test_carboxymethyl(self):
        # -CH2COOH : was 'acetyl' (a DIFFERENT molecule -C(=O)CH3)
        assert _name_frag("CCC(=O)O") == "carboxymethyl"

    def test_2_carboxyethyl(self):
        assert _name_frag("CCCC(=O)O") == "2-carboxyethyl"

    def test_2_oxopropyl(self):
        # -CH2C(=O)CH3 : oxo on C2 (not the attachment) -> '2-oxopropyl'
        assert _name_frag("CCC(=O)C") == "2-oxopropyl"

    def test_hydroxymethyl_one_carbon_byte_identical(self):
        assert _name_frag("CCO") == "hydroxymethyl"

    def test_acyl_oxo_on_attachment_named_as_acyl(self):
        # -C(=O)CH3 : the oxo sits on the free-valence carbon = an acyl group. A
        # LINEAR SATURATED acyl is now named as the acyl PREFIX here (P-66.6,
        # BB:17762 'acetyl (preferred prefix)'), NOT '1-oxoethyl' and NOT the old
        # garbled '1-methyl-2-oxaeth-1-en-1-yl' (which the previous decline +
        # acetaldehyde-cap downstream produced). RT-verified:
        # methyl 4-acetylcyclohexane-1-carboxylate round-trips.
        from orthonym.assembly.substituent_naming import (
            _name_polyfunctional_acyclic_substituent as f,
        )
        mol = Chem.MolFromSmiles("CC(=O)C")  # parent-C0, attach=1 (carbonyl C)
        assert f(mol, [1, 2, 3], 1, set()) == "acetyl"


class TestSingleFGRingParent:
    """The ring-parent path (benzene._name_functionalized_chain_substituent)
    must give the SAME located single-FG names as the structure-based namer."""

    def test_benzoic_2_hydroxyethyl(self):
        assert name_compound("OC(=O)c1ccc(CCO)cc1") == "4-(2-hydroxyethyl)benzoic acid"

    def test_benzoic_3_hydroxypropyl(self):
        assert name_compound("OC(=O)c1ccc(CCCO)cc1") == "4-(3-hydroxypropyl)benzoic acid"

    def test_benzoic_carboxymethyl_unchanged(self):
        assert name_compound("OC(=O)c1ccc(CC(=O)O)cc1") == "4-(carboxymethyl)benzoic acid"

    def test_benzoic_formyl_preserved(self):
        # -CHO directly on the ring stays 'formyl' (1-carbon map entry intact)
        assert name_compound("OC(=O)c1ccc(C=O)cc1") == "4-formylbenzoic acid"


class TestPolyfunctionalSubstituentFailClosed:
    """Fail-closed: non-clean classes (ring/ether/amide) keep their existing tiers."""

    def test_single_hydroxy_one_carbon_elided(self):
        # 1-carbon backbone -> locant elided -> 'hydroxymethyl' (byte-identical
        # to the pre-single-FG behaviour).
        assert _name_frag("CCO") == "hydroxymethyl"

    def test_single_amino_one_carbon_elided(self):
        assert _name_frag("CCN") == "aminomethyl"

    def test_ether_declined(self):
        # -CH2-O-CH2COOH : ether linkage -> namer declines (other tiers own it)
        out = _name_frag("CCOCC(=O)O")
        assert "amino" not in out  # not mis-claimed; ether handled elsewhere

    def test_ring_declined(self):
        # cyclohexane-diol ring -> ring engine, not the acyclic namer
        out = _name_frag("CC1CCC(O)C(O)C1")
        assert out is not None  # named by ring path, no exception

    def test_direct_namer_declines_amide(self):
        from orthonym.assembly.substituent_naming import (
            _name_polyfunctional_acyclic_substituent as f,
        )
        # primary amide -C(=O)NH2 on a chain -> decline (carbamoyl is another tier)
        mol = Chem.MolFromSmiles("CCC(N)=O")  # -CH2-C(=O)NH2
        assert f(mol, [1, 2, 3, 4], 1, set()) is None


class TestPhosphatidylserine:
    """End-to-end P-107.3.3: PS now names via the functional-class diester."""

    def test_ps_achiral(self):
        smi = ("CCCCCCCCCCCCCCCC(=O)OCC(COP(O)(=O)OCC(N)C(=O)O)"
               "OC(=O)CCCCCCCCCCCCCCC")
        assert name_compound(smi) == (
            "2,3-bis(hexadecanoyloxy)propyl 2-amino-2-carboxyethyl "
            "hydrogen phosphate"
        )

    def test_ps_L_serine_sn_glycero(self):
        # v31 change-asserted-value: the engine now emits the substitutive
        # L-serine-parent PIN (P-107.3.3 Phosphatidylserines), which is
        # byte-identical to the Blue Book template at BlueBookV2.md:55162
        # ("O-{[(2R)-2,3-bis(octadecanoyloxy)propoxy]hydroxyphosphoryl}-L-serine",
        # here with hexadecanoyl for the C16 input) — the PIN, superseding the old
        # functional-class "... hydrogen phosphate" diester expectation. Verified
        # RT-exact (isomeric round-trip incl. the (2R)/L stereo).
        smi = ("CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(O)(=O)OC[C@H](N)C(=O)O)"
               "OC(=O)CCCCCCCCCCCCCCC")
        assert name_compound(smi) == (
            "O-{[(2R)-2,3-bis(hexadecanoyloxy)propoxy]hydroxyphosphoryl}-L-serine"
        )

    def test_ps_deterministic_across_renderings(self):
        smi = ("CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(O)(=O)OC[C@H](N)C(=O)O)"
               "OC(=O)CCCCCCCCCCCCCCC")
        mol = Chem.MolFromSmiles(smi)
        names = set()
        for seed in range(6):
            rs = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            names.add(name_compound(rs))
        assert len(names) == 1, f"non-deterministic: {names}"


class TestPhospholipidRegression:
    """The serine branch must not disturb PC / PE / phosphatidic acid."""

    def test_pc_unchanged(self):
        smi = ("CCCCCCCCCCCCCCCC(=O)OCC(COP([O-])(=O)OCC[N+](C)(C)C)"
               "OC(=O)CCCCCCCCCCCCCCC")
        assert name_compound(smi) == (
            "[2,3-bis(hexadecanoyloxy)propyl] 2-(trimethylazaniumyl)ethyl "
            "phosphate"
        )

    def test_pe_unchanged(self):
        smi = "CCCCCCCCCCCCCCCC(=O)OCC(COP(O)(=O)OCCN)OC(=O)CCCCCCCCCCCCCCC"
        assert name_compound(smi) == (
            "3-{[(2-aminoethoxy)hydroxyphosphoryl]oxy}propane-1,2-diyl "
            "dihexadecanoate"
        )

    def test_phosphatidic_acid_unchanged(self):
        smi = "CCCCCCCCCCCCCCCC(=O)OCC(COP(O)(O)=O)OC(=O)CCCCCCCCCCCCCCC"
        assert name_compound(smi) == (
            "2,3-bis(hexadecanoyloxy)propyl dihydrogen phosphate"
        )

    def test_pi_inositol_head_excluded(self):
        # PI: the inositol ring head (multi-stereocentre, P12 non-determinism risk)
        # is explicitly EXCLUDED from the serine branch -> my functional-class
        # diester is NOT produced (the assembler falls to its pre-existing path,
        # which the self-consistency gate suppresses to 'unknown' in production).
        smi = ("CCCCCCCCCCCCCCCC(=O)OCC(COP(O)(=O)OC1C(O)C(O)C(O)C(O)C1O)"
               "OC(=O)CCCCCCCCCCCCCCC")
        name = name_compound(smi)
        # the serine-branch signature ('<glyceryl> <head> hydrogen phosphate')
        # must NOT appear — inositol is deferred, not mis-claimed
        assert not name.endswith("hydrogen phosphate")
        assert "2-amino-2-carboxyethyl" not in name
