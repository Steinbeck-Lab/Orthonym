""" a phase — substituent support for the algorithmic 2-component
ortho-fused mancude heterocycle path.

Root cause fixed: ``_try_algorithmic_fusion_name`` previously hard-refused the
moment any exocyclic heavy atom was present, so a 2-component ortho-fused
mancude heterocycle whose core is NOT in the retained catalog lost its name as
soon as it carried a substituent. The fix discovers substituents against the
deterministic peripheral numbering (``compute_fused_numbering``),
applies the lowest-substituent-locant tie-break over the ring-system
automorphisms, and fails closed at source when any exocyclic branch is
unnameable.

Blue Book grounding (the Blue Book Blue Book):
- (a)/(b): ring numbering fixed by heteroatoms (set, then element
  order O > S > Se > Te > N...); substituents are NOT in that list.
- / line 25503: when a choice remains, low locants to detachable
  prefixes, then alphanumerical — the symmetric-parent tie-break.

These assert the NAME contract directly (via ``_try_algorithmic_fusion_name``)
rather than through the full namer's gate, which fails OPEN under OPSIN
subprocess contention (see NEXT-SESSION hazards). Each target PIN was verified
this session to OPSIN-round-trip to the input SMILES via ``scripts/diagnose.py``.
"""
import pytest
from rdkit import Chem

from orthonym.rules.fused_rings import (
    _try_algorithmic_fusion_name,
    _try_polycomponent_fusion_name,
    _exocyclic_atoms_accounted,
)


def _ring_atoms(mol):
    atoms = set()
    for r in mol.GetRingInfo().AtomRings():
        atoms.update(r)
    return atoms


class TestBP4Phase1Targets:
    """The four blueprint targets now emit the expected PIN (were `unknown`)."""

    @pytest.mark.parametrize("smiles,expected", [
        # Symmetric parent: 2- and 5- are equivalent; PIN takes the lower (2).
        ("Cc1cc2ccoc2o1", "2-methylfuro[2,3-b]furan"),
        ("Cc1cc2occc2o1", "2-methylfuro[3,2-b]furan"),
        # Asymmetric parent (O < S): numbering fixed, no substituent freedom;
        # the methyl sits on the S-ring carbon = locant 5.
        ("Cc1cc2ccoc2s1", "5-methylthieno[2,3-b]furan"),
        # Halogen detachable prefix on the symmetric parent -> lowest locant 2.
        ("Clc1cc2ccoc2o1", "2-chlorofuro[2,3-b]furan"),
    ])
    def test_target_pin(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert _try_algorithmic_fusion_name(mol) == expected


class TestBP4Phase1Generalizes:
    """The fix is a whole-class root-cause fix, not the four examples."""

    @pytest.mark.parametrize("smiles,expected", [
        ("Cc1cc2cc(C)oc2o1", "2,5-dimethylfuro[2,3-b]furan"),   # di-substituted
        ("CCc1cc2ccoc2o1", "2-ethylfuro[2,3-b]furan"),          # ethyl
        ("Cc1cc2ccoc2[se]1", "5-methylselenopheno[2,3-b]furan"),  # selenium ring
        ("Fc1cc2ccsc2o1", "2-fluorothieno[2,3-b]furan"),        # halogen, asym
        ("Nc1cc2ccoc2o1", "furo[2,3-b]furan-2-amine"),          # amino -> suffix
    ])
    def test_generalized_pin(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert _try_algorithmic_fusion_name(mol) == expected

    def test_input_order_independent(self):
        """Two SMILES writings of the same molecule give the same PIN
        (the tie-break uses a SMILES-order-independent canonical signature)."""
        a = _try_algorithmic_fusion_name(Chem.MolFromSmiles("Cc1cc2occc2o1"))
        b = _try_algorithmic_fusion_name(Chem.MolFromSmiles("Cc1oc2c(c1)occ2"))
        assert a == b == "2-methylfuro[3,2-b]furan"


class TestBP4Phase1BaselinesUnchanged:
    """Bare systems keep the legacy path byte-for-byte; catalog cores still
    short-circuit before the algorithmic path (regression guard)."""

    @pytest.mark.parametrize("smiles,expected", [
        ("c1cc2ccoc2o1", "furo[2,3-b]furan"),
        ("c1cc2ccoc2s1", "thieno[2,3-b]furan"),
        ("c1cc2occc2o1", "furo[3,2-b]furan"),
    ])
    def test_bare_unchanged(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert _try_algorithmic_fusion_name(mol) == expected


class TestBP4Phase1FailClosed:
    """Critic-required source-level completeness check: never emit a name that
    silently omits a substituent (that would denote a DIFFERENT molecule)."""

    @pytest.mark.parametrize("smiles", [
        "[Si](C)(C)c1cc2ccoc2o1",  # silyl: exotic element, unnameable branch
        "[B](O)(O)c1cc2ccoc2o1",   # boronic acid: exotic element B
    ])
    def test_unnameable_substituent_fails_closed(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert _exocyclic_atoms_accounted(mol, _ring_atoms(mol)) is False
        assert _try_algorithmic_fusion_name(mol) is None

    def test_accounted_true_for_simple_substituent(self):
        mol = Chem.MolFromSmiles("Cc1cc2ccoc2o1")
        assert _exocyclic_atoms_accounted(mol, _ring_atoms(mol)) is True

    def test_accounted_true_for_bare_system(self):
        mol = Chem.MolFromSmiles("c1cc2ccoc2o1")
        assert _exocyclic_atoms_accounted(mol, _ring_atoms(mol)) is True


class TestBP4Phase2Polycomponent:
    """a phase: substituent support for the 3+-component cata-fused star class
    (single heteroring base + >=2 monocyclic children), reusing the identical
    numbering + assembler machinery. Each PIN verified this session to OPSIN
    round-trip via scripts/diagnose.py."""

    @pytest.mark.parametrize("smiles,expected", [
        ("Cc1cc2nc3ccoc3cc2o1", "2-methyldifuro[3,2-b:2',3'-e]pyridine"),
        ("Cc1coc2cc3occc3nc12", "3-methyldifuro[3,2-b:2',3'-e]pyridine"),
        ("Cc1c2occc2nc2ccoc12", "8-methyldifuro[3,2-b:2',3'-e]pyridine"),
        ("Clc1cc2nc3ccoc3cc2o1", "2-chlorodifuro[3,2-b:2',3'-e]pyridine"),
    ])
    def test_substituted_polycomponent_pin(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert _try_polycomponent_fusion_name(mol) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("O1C=CC2=C1C=C1C(=N2)C=CO1", "difuro[3,2-b:2',3'-e]pyridine"),
        ("O1C=CC2=NC3=C(C=C21)SC=C3", "furo[3,2-b]thieno[2,3-e]pyridine"),
    ])
    def test_bare_polycomponent_unchanged(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert _try_polycomponent_fusion_name(mol) == expected

    def test_unnameable_substituent_fails_closed(self):
        mol = Chem.MolFromSmiles("[Si](C)(C)c1cc2nc3ccoc3cc2o1")
        assert mol is not None
        assert _try_polycomponent_fusion_name(mol) is None


class TestBP4Phase3PartialSaturation:
    """a phase (full): partially-saturated 2-component ortho-fused pairs named as
    '<hydro>-<indicatedH>-<mancude parent>' / /. The
    maximum number of noncumulative double bonds is placed into the saturated
    region (a maximum matching of the saturated-carbon subgraph); unmatched
    carbons are indicated hydrogen, each matched pair is one unit of hydro. Covers
    even counts (0 indicated H, byte-identical to the former slice), odd counts
    (indicated-H + hydro mix), saturated ring chalcogens, and prefix substituents.
    Each PIN verified this session to OPSIN round-trip."""

    @pytest.mark.parametrize("smiles,expected", [
        # even sp3 count, 0 indicated H (byte-identical to the former slice):
        # symmetric mancude parent (furo[3,2-b]furan): 2,3 beats the RT-valid 5,6
        ("C1COc2ccoc21", "2,3-dihydrofuro[3,2-b]furan"),
        ("C1Cc2ccoc2O1", "2,3-dihydrofuro[2,3-b]furan"),
        ("O1CCc2sccc21", "2,3-dihydrothieno[3,2-b]furan"),
        ("C1COc2ccsc21", "2,3-dihydrothieno[3,2-b]furan"),
    ])
    def test_partial_saturation_pin(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert _try_algorithmic_fusion_name(mol) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # case (a): ODD sp3 count = indicated-H (lowest locant, + hydro
        ("C1CCc2ccoc2O1", "5,6-dihydro-4H-furo[2,3-b]pyran"),
        ("C1CCOc2ccoc21", "6,7-dihydro-5H-furo[3,2-b]pyran"),
        ("O1CCCc2ccoc21", "5,6-dihydro-4H-furo[2,3-b]pyran"),
        ("O1CCCc2occc21", "6,7-dihydro-5H-furo[3,2-b]pyran"),
    ])
    def test_odd_saturation_indicated_h_mix(self, smiles, expected):
        assert _try_algorithmic_fusion_name(Chem.MolFromSmiles(smiles)) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # case (b): a saturated ring S/Se is an inherent ring atom, not a hydro pos
        ("C1Cc2ccoc2S1", "4,5-dihydrothieno[2,3-b]furan"),
        ("C1Cc2ccoc2[Se]1", "4,5-dihydroselenopheno[2,3-b]furan"),
    ])
    def test_saturated_chalcogen_ring(self, smiles, expected):
        assert _try_algorithmic_fusion_name(Chem.MolFromSmiles(smiles)) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # case (c): prefix-substituted partial saturation
        ("O1CCCc2cc(C)oc21", "2-methyl-5,6-dihydro-4H-furo[2,3-b]pyran"),
        ("O1CC(C)Cc2ccoc21", "5-methyl-5,6-dihydro-4H-furo[2,3-b]pyran"),
        ("CC1CCc2ccoc21", "6-methyl-5,6-dihydro-4H-cyclopenta[b]furan"),
        ("Cc1cc2c(o1)CCC(C)O2", "2,5-dimethyl-6,7-dihydro-5H-furo[3,2-b]pyran"),
    ])
    def test_substituted_partial_saturation(self, smiles, expected):
        assert _try_algorithmic_fusion_name(Chem.MolFromSmiles(smiles)) == expected

    @pytest.mark.parametrize("smiles", [
        "O=C1CCc2ccoc21",          # oxo suffix
        "O1CCCc2cc(C(=O)O)oc21",   # carboxylic-acid suffix
    ])
    def test_suffix_group_fails_closed(self, smiles):
        """A suffix-forming group needs added indicated H (out of scope):
        _try_partial_saturation_name declines (the legacy path then owns it)."""
        from orthonym.rules.fused_rings import (
            _try_partial_saturation_name, get_shared_atoms,
        )
        from orthonym.rules.fusion_descriptors import (
            generate_systematic_name_for_fused_pair,
        )
        mol = Chem.MolFromSmiles(smiles)
        r = mol.GetRingInfo().AtomRings()
        desc = generate_systematic_name_for_fused_pair(
            mol, list(r[0]), list(r[1]), get_shared_atoms(mol, r[0], r[1]))
        assert _try_partial_saturation_name(
            mol, set(r[0]) | set(r[1]), desc) is None

    def test_fully_aromatic_not_treated_as_partial(self):
        """A fully-mancude system must NOT enter the hydro path."""
        from orthonym.rules.fused_rings import _try_partial_saturation_name
        mol = Chem.MolFromSmiles("c1cc2ccoc2o1")
        assert _try_partial_saturation_name(mol, _ring_atoms(mol), "furo[2,3-b]furan") is None


class TestBP3ClusterRDecoratedRingSubstituent:
    """ cluster R: a ring substituent that carries its OWN decorations
    (rooted at a ring atom) now names via free-valence numbering + decoration
    placement, and the pyrazole/imidazole R-bug is fixed on the substituent path.
    Each PIN OPSIN round-trip verified this session via scripts/diagnose.py."""

    @pytest.mark.parametrize("smiles,expected", [
        # bare pyrazolyl — R-bug fix (was mis-id'd as imidazolyl -> unknown)
        ("OC(=O)c1ccc(-c2cc[nH]n2)cc1", "4-(1H-pyrazol-3-yl)benzoic acid"),
        ("OC(=O)c1ccc(-c2ccn[nH]2)cc1", "4-(1H-pyrazol-5-yl)benzoic acid"),
        # decorated pyrazolyl — cluster R recursion
        # 2026-09-25 (pre-existing-failures plan, Task 5) change-asserted-value:
        # cites indicated hydrogen for the mancude parent whatever sits on the pyrrole-type atom; a substituent group keeps it: "(1H-indol-1-yl)acetic acid (PIN)" (the Blue Book). OPSIN RT exact.
        ("OC(=O)c1ccc(-c2cc(C)n(C)n2)cc1",
         "4-(1,5-dimethyl-1H-pyrazol-3-yl)benzoic acid"),
        # genuine imidazole unchanged (non-adjacent N)
        ("OC(=O)c1ccc(-c2cnc[nH]2)cc1", "4-(1H-imidazol-5-yl)benzoic acid"),
        # bare pyridinyl unchanged
        ("OC(=O)c1ccc(-c2cccnc2)cc1", "4-(pyridin-3-yl)benzoic acid"),
    ])
    def test_ring_substituent_pin(self, smiles, expected):
        from orthonym import name_compound
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # diaryl-ketone flagship: decorated heteroaryl substituent on a methanone
        # (routes through the composer ring-substituent-prefix path, now funneled
        # to the chokepoint). Was `unknown`.
        # 2026-09-25 (pre-existing-failures plan, Task 5) change-asserted-value:
        # cites indicated hydrogen for the mancude parent whatever sits on the pyrrole-type atom; a substituent group keeps it: "(1H-indol-1-yl)acetic acid (PIN)" (the Blue Book). OPSIN RT exact.
        ("O=C(c1ccccc1)c1cc(C)n(C)n1",
         "(1,5-dimethyl-1H-pyrazol-3-yl)phenylmethanone"),
        # (item 2): — a one-carbon `methanone` has a single
        # position, so the `-1-` locant is omitted (matches the sibling rows
        # above; RT-verified 2026-09-04, ITEM2-VERIFICATION.md).
        # 2026-09-25 (pre-existing-failures plan, Task 5) change-asserted-value:
        # cites indicated hydrogen for the mancude parent whatever sits on the pyrrole-type atom; a substituent group keeps it: "(1H-indol-1-yl)acetic acid (PIN)" (the Blue Book). OPSIN RT exact.
        ("Cc1nn(C)c(O)c1C(=O)c1ccc(Cl)cc1Cl",
         "(2,4-dichlorophenyl)(5-hydroxy-1,3-dimethyl-1H-pyrazol-4-yl)methanone"),
        # bare heteroaryl methanones unchanged (regression guard)
        ("O=C(c1ccccc1)c1ccncc1", "phenyl(pyridin-4-yl)methanone"),
        ("O=C(c1ccccc1)c1cccnc1", "phenyl(pyridin-3-yl)methanone"),
        # common decorated-aryl-on-chain unchanged (regression guard)
        ("OC(=O)CCc1ccc(Cl)cc1", "3-(4-chlorophenyl)propanoic acid"),
        ("OC(=O)CCc1ccc(C)cc1", "3-(4-methylphenyl)propanoic acid"),
    ])
    def test_ketone_and_chain_ring_substituent(self, smiles, expected):
        from orthonym import name_compound
        assert name_compound(smiles) == expected

    def test_decorated_helper_direct(self):
        """The decorated-ring helper produces the numbered PIN substituent form."""
        from orthonym.rules.ring_substituents import (
            _decorated_heteroaryl_substituent_name,
        )
        # 5-hydroxy-1,3-dimethylpyrazol-4-yl fragment (ring + 2 Me + OH)
        mol = Chem.MolFromSmiles("Cc1nn(C)c(O)c1C(=O)c1ccc(Cl)cc1Cl")
        ri = mol.GetRingInfo()
        pyr = next(r for r in ri.AtomRings()
                   if len(r) == 5
                   and sum(1 for a in r if mol.GetAtomWithIdx(a).GetSymbol() == 'N') == 2)
        ringset = set(pyr)
        attach = next(
            a for a in pyr
            for nb in mol.GetAtomWithIdx(a).GetNeighbors()
            if nb.GetIdx() not in ringset and nb.GetSymbol() == 'C'
            and any(b.GetBondTypeAsDouble() == 2 for b in nb.GetBonds())
        )
        frag = set(pyr)
        for a in pyr:
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                ni = nb.GetIdx()
                if ni in ringset:
                    continue
                if a == attach and nb.GetSymbol() == 'C' and any(
                        b.GetBondTypeAsDouble() == 2 for b in nb.GetBonds()):
                    continue
                frag.add(ni)
        assert _decorated_heteroaryl_substituent_name(
            mol, tuple(frag), tuple(pyr), attach
        ) == "5-hydroxy-1,3-dimethyl-1H-pyrazol-4-yl"  # indicated H: see the rows above
