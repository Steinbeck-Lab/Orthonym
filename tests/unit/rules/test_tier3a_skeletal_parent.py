"""Wave2 T3a — SKELETAL_SUFFIX_PGS extension + constitution-conservation guard.

Three coupled root-cause fixes (P-52.2.8 / P-29.4.2 / P-14.3.4):

1. SKELETAL parent selection: -ol/-thiol/-selenol/-tellurol/-amine/-imine
   decorate a skeletal atom — no exocyclic-carbon suffix form exists (unlike
   -carbaldehyde/-carboxylic acid). The single-carbon ring-neighbour shortcut
   (namer._classify gate + parent_selection single-carbon branch) mis-parented
   OCC1CCCCC1 to ring ('cyclohexan-1-ol', a different molecule, SELF-01-
   suppressed). Extending SKELETAL_SUFFIX_PGS REQUIRED paired
   PG_ATTACHMENT_INDICES overrides (the SMARTS lead with O/N/S; membership
   must test the bearing CARBON or 4-methylcyclohexan-1-ol goes
   false-negative). selenol SMARTS gained the carbon ([SeX2H][#6]).

2. Constitution-conservation guards: (a) benzene._identify_functionalized_chain
   fabricated a linear all-C chain from carbon_count alone — Ar-CH2-SiH2-CH2-Ar'
   emitted as '4-(8-carboxyoctyl)benzoic acid' (Si dropped, far ring
   re-linearised) passing the coverage gate at 0.79; now it emits only for a
   linear acyclic saturated chain with one terminal FG. (b) unnameable benzene
   branches record an explicit sentinel -> name_substituted_benzene declines.
   (c) composer._build_substituted_ring_name returns None (fail closed) when
   the ring carries branches it cannot express; compound branches (-CH2OH) are
   now named via name_substituent_fragment (heals the trimethanol gold).

3. Locant hygiene: a mononuclear (1-atom) chain parent takes no substituent
   locants (phenylmethanol, NOT 1-phenylmethanol; P-14.3.4 Rule 1 through
   format_substituent_prefix); a SUBSTITUTED 2-carbon parent CITES the suffix
   locant on the generic path ('1-cyclohexylethan-1-imine',
   '2-chloroethane-1-selenol' — parallel to the dedicated -ol handler's
   '2-chloroethan-1-ol'), while bare ethanimine/ethaneselenol still elide
   (the FG's own pseudo-branch in features.substituents is excluded).

Reproduce-first notes: all six chain-parent heal targets were fail-closed
'unknown' at HEAD (not wrong names — SELF-01 caught the relocations);
OCC1CCC(O)CC1 (exo-CH2OH + ring-OH) is unknown at HEAD *and* after T3a via a
PRE-EXISTING ring-suffix anchor-walk defect (proven identical in a HEAD
worktree A/B) — not a T3a regression, deferred.
"""

import pytest

from orthonym import name_compound


@pytest.mark.unit
class TestSkeletalSuffixChainParent:
    """Exocyclic skeletal-suffix carbon forces the chain (methan-) parent."""

    @pytest.mark.parametrize("smiles,expected", [
        ("OCC1CCCCC1", "cyclohexylmethanol"),
        ("NCC1CCCCC1", "cyclohexylmethanamine"),
        ("SCC1CCCCC1", "cyclohexylmethanethiol"),
        ("[SeH]CC1CCCCC1", "cyclohexylmethaneselenol"),
        ("[TeH]CC1CCCCC1", "cyclohexylmethanetellurol"),
        ("N=CC1CCCCC1", "cyclohexylmethanimine"),
        ("OCc1ccccc1", "phenylmethanol"),
        ("NCc1ccccc1", "phenylmethanamine"),
        ("OCC1CCC(C)CC1", "(4-methylcyclohexyl)methanol"),
    ])
    def test_chain_parent_heals(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_diaryl_ketone_methanone_parent(self):
        # Non-retained diaryl ketone: single-carbon methanone parent now
        # eligible (benzophenone itself stays retained, tested below).
        assert name_compound("O=C(c1ccccc1)c1ccncc1") == \
            "phenyl(pyridin-4-yl)methanone"

    def test_benzophenone_stays_retained(self):
        assert name_compound("O=C(c1ccccc1)c1ccccc1") == "benzophenone"


@pytest.mark.unit
class TestRingParentGuards:
    """Suffix atom IS a ring atom (or exocyclic-carbon suffix class):
    ring parent MUST hold."""

    @pytest.mark.parametrize("smiles,expected", [
        ("OC(=O)C1CCCCC1", "cyclohexanecarboxylic acid"),
        ("O=CC1CCCCC1", "cyclohexanecarbaldehyde"),
        ("OC1CCCCC1", "cyclohexan-1-ol"),
        ("Oc1ccccc1", "phenol"),
        ("NC1CCCCC1", "cyclohexan-1-amine"),
        ("N=C1CCCCC1", "cyclohexan-1-imine"),
        # PG_ATTACHMENT_INDICES pairing: without the carbon override these
        # go false-negative in is_principal_group_on_ring (the O/N/S atom is
        # never a ring member) and lose the ring parent.
        ("CC1CCC(O)CC1", "4-methylcyclohexan-1-ol"),
        ("CC1CCC(N)CC1", "4-methylcyclohexan-1-amine"),
        ("SC1CCCC1C", "2-methylcyclopentane-1-thiol"),
        ("CNC1CCCCC1", "N-methylcyclohexan-1-amine"),
    ])
    def test_ring_parent_holds(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_secondary_amine_chain_shape_unchanged(self):
        # HEAD-equivalence pin: the secondary_amine [1,2] override must not
        # flip the working methanamine-parent choice.
        assert name_compound("CNCC1CCCCC1") == "N-cyclohexylmethylmethanamine"


@pytest.mark.unit
class TestSuffixLocantOnSubstitutedParent:
    """P-14.3.4: substituents force the suffix locant back on the generic
    path; bare symmetric parents still elide."""

    @pytest.mark.parametrize("smiles,expected", [
        ("CC(=N)C1CCCCC1", "1-cyclohexylethan-1-imine"),
        ("ClCC[SeH]", "2-chloroethane-1-selenol"),
        ("N=CCCl", "2-chloroethan-1-imine"),
    ])
    def test_substituted_parent_cites_locant(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("N=CC", "ethanimine"),
        ("CC[SeH]", "ethaneselenol"),
        ("NCC", "ethanamine"),
        ("CCS", "ethanethiol"),
        ("OCC", "ethanol"),
        ("OOCC", "ethaneperoxol"),
        ("CCS(N)(=O)=O", "ethanesulfonamide"),
        ("CC(=N)C", "propan-2-imine"),
        ("CCC=N", "propan-1-imine"),
    ])
    def test_bare_elision_controls(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestMononuclearParentLocants:
    """P-14.3.4 Rule 1: a 1-atom parent takes no substituent locants."""

    def test_tetrabromomethane_control(self):
        assert name_compound("BrC(Br)(Br)Br") == "tetrabromomethane"

    def test_format_substituent_prefix_empty_locants_no_hyphen(self):
        from orthonym.assembly.naming_utils import format_substituent_prefix
        assert format_substituent_prefix("cyclohexyl", [], 1) == "cyclohexyl"
        assert format_substituent_prefix("methyl", [2], 1) == "2-methyl"


@pytest.mark.unit
class TestConstitutionConservation:
    """Named form must account for every branch atom — else fail closed."""

    def test_si_bridge_stays_unknown(self):
        # Was '4-(8-carboxyoctyl)benzoic acid' raw (Si dropped, ring
        # re-linearised); guard makes the refusal deterministic.
        assert name_compound(
            "OC(=O)c1ccc(C[SiH2]Cc2ccc(C(=O)O)cc2)cc1"
        ) == "unknown organic compound"

    def test_oxy_bridge_stays_unknown(self):
        assert name_compound(
            "Oc1ccc(COCc2ccc(O)cc2)cc1"
        ) == "unknown organic compound"

    def test_functionalized_chain_guard_rejects_si(self):
        from rdkit import Chem
        from orthonym.rules.benzene import _identify_functionalized_chain
        mol = Chem.MolFromSmiles("OC(=O)c1ccc(C[SiH2]Cc2ccc(C(=O)O)cc2)cc1")
        ring = next(
            set(r) for r in mol.GetRingInfo().AtomRings()
            if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)
        )
        # start atom = the benzylic CH2 attached to this ring
        start = next(
            n.GetIdx()
            for a in ring for n in mol.GetAtomWithIdx(a).GetNeighbors()
            if n.GetIdx() not in ring and n.GetSymbol() == 'C'
            and n.GetTotalNumHs() == 2
        )
        assert _identify_functionalized_chain(mol, start, ring) is None

    def test_functionalized_chain_guard_accepts_linear(self):
        from rdkit import Chem
        from orthonym.rules.benzene import _identify_functionalized_chain
        mol = Chem.MolFromSmiles("OC(=O)CCc1ccccc1")
        ring = set(mol.GetRingInfo().AtomRings()[0])
        start = next(
            n.GetIdx()
            for a in ring for n in mol.GetAtomWithIdx(a).GetNeighbors()
            if n.GetIdx() not in ring and n.GetSymbol() == 'C'
        )
        info = _identify_functionalized_chain(mol, start, ring)
        assert info is not None
        assert info['functional_group'] == 'carboxylic_acid'

    def test_unnameable_benzene_branch_declines(self):
        from rdkit import Chem
        from orthonym.rules.benzene import (
            get_benzene_substituents, name_substituted_benzene,
        )
        mol = Chem.MolFromSmiles("OC(=O)c1ccc(C[SiH2]Cc2ccc(C(=O)O)cc2)cc1")
        ring = tuple(next(
            r for r in mol.GetRingInfo().AtomRings()
            if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)
        ))
        subs = get_benzene_substituents(mol, ring)
        assert any(
            s.get('unnameable') for lst in subs.values() for s in lst
        )
        oriented = list(ring)
        assert name_substituted_benzene(mol, ring, oriented, subs) is None

    @pytest.mark.parametrize("smiles,expected", [
        # Compound (hetero-bearing) ring branches now named exactly via the
        # recursive fragment namer instead of being dropped.
        # Wave2 T5a: the multiplicative alcohol-arm PIN now claims this row
        # (P-15.3.2.1); the T3a substitutive form remains the conservation
        # fallback when the multiplicative path declines.
        ("OCc1cc(CO)cc(CO)c1", "(benzene-1,3,5-triyl)trimethanol"),
        ("ClCc1ccccc1CO", "(2-(chloromethyl)phenyl)methanol"),
        # Working functionalized-chain forms must survive the guard.
        ("OC(=O)c1ccc(CCC(=O)O)cc1", "4-(2-carboxyethyl)benzoic acid"),
        ("OCCc1ccc(O)cc1", "4-(2-hydroxyethyl)phenol"),
        ("Oc1ccc(CO)cc1", "4-(hydroxymethyl)phenol"),
    ])
    def test_conservation_heals_and_controls(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestBenzeneControlsUnchanged:
    """Benzene-path spot checks around the sentinel change."""

    @pytest.mark.parametrize("smiles,expected", [
        ("Cc1ccccc1", "toluene"),
        ("CCc1ccccc1", "ethylbenzene"),
        ("Oc1ccc(Cl)cc1", "4-chlorophenol"),
        ("Nc1ccc(C(=O)O)cc1", "4-aminobenzoic acid"),
        ("OC(=O)c1ccccc1", "benzoic acid"),
    ])
    def test_controls(self, smiles, expected):
        assert name_compound(smiles) == expected
