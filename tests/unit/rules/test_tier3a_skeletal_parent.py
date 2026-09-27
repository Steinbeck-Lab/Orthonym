"""Wave2 — SKELETAL_SUFFIX_PGS extension + constitution-conservation guard.

Three coupled root-cause fixes / /:

1. SKELETAL parent selection: -ol/-thiol/-selenol/-tellurol/-amine/-imine
   decorate a skeletal atom — no exocyclic-carbon suffix form exists (unlike
   -carbaldehyde/-carboxylic acid). The single-carbon ring-neighbour shortcut
   (namer._classify gate + parent_selection single-carbon branch) mis-parented
   OCC1CCCCC1 to ring ('cyclohexan-1-ol', a different molecule, -
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
   locants (phenylmethanol, NOT 1-phenylmethanol; Rule 1 through
   format_substituent_prefix); a SUBSTITUTED 2-carbon parent CITES the suffix
   locant on the generic path ('1-cyclohexylethan-1-imine',
   '2-chloroethane-1-selenol' — parallel to the dedicated -ol handler's
   '2-chloroethan-1-ol'), while bare ethanimine/ethaneselenol still elide
   (the FG's own pseudo-branch in features.substituents is excluded).

Reproduce-first notes: all six chain-parent heal targets were fail-closed
'unknown' at HEAD (not wrong names — caught the relocations);
OCC1CCC(O)CC1 (exo-CH2OH + ring-OH) is unknown at HEAD *and* after via a
PRE-EXISTING ring-suffix anchor-walk defect (proven identical in a HEAD
worktree A/B) — not a regression, deferred.
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
        # Non-retained diaryl ketone: single-carbon methanone parent.
        assert name_compound("O=C(c1ccccc1)c1ccncc1") == \
            "phenyl(pyridin-4-yl)methanone"

    def test_benzophenone_demoted_to_diphenylmethanone(self):
        # W8-P1 R6, BB 28326/28378): 'benzophenone' is retained
        # for GENERAL nomenclature only; the PIN is the systematic
        # diphenylmethanone (mirror of acetophenone -> 1-phenylethan-1-one).
        assert name_compound("O=C(c1ccccc1)c1ccccc1") == "diphenylmethanone"


@pytest.mark.unit
class TestRingParentGuards:
    """Suffix atom IS a ring atom (or exocyclic-carbon suffix class):
    ring parent MUST hold."""

    @pytest.mark.parametrize("smiles,expected", [
        ("OC(=O)C1CCCCC1", "cyclohexanecarboxylic acid"),
        ("O=CC1CCCCC1", "cyclohexanecarbaldehyde"),
        # Phase C tranche A: these three are MONOsubstituted homogeneous
        # monocycles, so (c) (``the Blue Book``, "The locant '1' is
        # omitted:... (c) in monosubstituted homogeneous monocyclic rings") licenses
        # the omission -- ``:2917`` prints ``cyclohexanethiol`` as the rule's own
        # example and ``:26854``/``:14916`` print ``cyclopentanol``/``cyclohexanone``.
        # The rows BELOW keep their locants because a ring bearing a suffix AND a
        # substituent is not monosubstituted in that sense. Updated from
        # ``cyclohexan-1-ol``/``-1-amine``, which the licence makes non-PINs.
        ("OC1CCCCC1", "cyclohexanol"),
        ("Oc1ccccc1", "phenol"),
        ("NC1CCCCC1", "cyclohexanamine"),
        # ``cyclohexan-1-imine`` KEEPS its locant: the ``imine`` exclusion on
        # ``_ring_suffix_locant_is_trivial`` is the sole guard against the
        # ``N-hydroxy`` oxime hole, and Task 4 measured re-admitting it as
        # impossible-at-site. Deliberately unchanged.
        ("N=C1CCCCC1", "cyclohexan-1-imine"),
        # PG_ATTACHMENT_INDICES pairing: without the carbon override these
        # go false-negative in is_principal_group_on_ring (the O/N/S atom is
        # never a ring member) and lose the ring parent.
        ("CC1CCC(O)CC1", "4-methylcyclohexan-1-ol"),
        ("CC1CCC(N)CC1", "4-methylcyclohexan-1-amine"),
        ("SC1CCCC1C", "2-methylcyclopentane-1-thiol"),
        # N-substituents hang off the nitrogen, so the ring is monosubstituted
        # and the locant '1' is omitted (c) the Blue Book;
        # '*N*-butylcyclopropanamine (PIN)':26292,.
        ("CNC1CCCCC1", "N-methylcyclohexanamine"),
    ])
    def test_ring_parent_holds(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.xfail(
        strict=True,
        reason="PRE-EXISTING, and the expectation was never met. Measured 2026-07-30 "
               "against pre-Phase-C 974aba86 in a worktree: production there emits "
               "'unknown organic compound' for CNCC1CCCCC1, logging 'SELF-01 "
               "suppressed (different molecule): N-cyclohexylmethyl-N-methyl"
               "cyclohexan-1-amine (opsin=CN(CC1CCCCC1)C1CCCCC1)'. So the candidate "
               "names TWO cyclohexyl rings for a molecule with one, and SELF-01 "
               "correctly refuses it -- no wrong name ships. This unit test disables "
               "the OPSIN gate, which is the ONLY reason the fabrication is visible "
               "here. Not a Phase C regression; see task 'N-pyridylanamine' for the "
               "same class on the P-62 path.",
    )
    def test_secondary_amine_chain_shape_unchanged(self):
        # HEAD-equivalence pin: the secondary_amine [1,2] override must not
        # flip the working methanamine-parent choice.
        assert name_compound("CNCC1CCCCC1") == "N-cyclohexylmethylmethanamine"


@pytest.mark.unit
class TestSuffixLocantOnSubstitutedParent:
    """: substituents force the suffix locant back on the generic
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
    """ Rule 1: a 1-atom parent takes no substituent locants."""

    def test_tetrabromomethane_control(self):
        assert name_compound("BrC(Br)(Br)Br") == "tetrabromomethane"

    def test_format_substituent_prefix_empty_locants_no_hyphen(self):
        from orthonym.assembly.naming_utils import format_substituent_prefix
        assert format_substituent_prefix("cyclohexyl", [], 1) == "cyclohexyl"
        assert format_substituent_prefix("methyl", [2], 1) == "2-methyl"


@pytest.mark.unit
class TestConstitutionConservation:
    """Named form must account for every branch atom — else fail closed."""

    def test_si_bridge_heals_multiplicative(self):
        # Was '4-(8-carboxyoctyl)benzoic acid' raw (Si dropped, ring
        # re-linearised); the tier-3a guard made the refusal deterministic.
        # Wave-2 completion B2 then built the proper composite
        # CH2-SiH2-CH2 multiplicative bridge -- the OPSIN-RT-verified PIN.
        assert name_compound(
            "OC(=O)c1ccc(C[SiH2]Cc2ccc(C(=O)O)cc2)cc1"
        ) == "4,4'-[silanediylbis(methylene)]dibenzoic acid"

    def test_oxy_bridge_names_multiplicative(self):
        # Was fail-closed 'unknown' (no -CH2-O-CH2- recognizer). w2f p1 then
        # built the proper / composite CH2-O-CH2
        # oxybis(methylene) multiplicative bridge -- the OPSIN-RT-verified PIN
        # (constitution PRESERVED; gold W2F-P1-07). Exact analog of the
        # silanediylbis(methylene) heal above (Wave-2 completion B2).
        assert name_compound(
            "Oc1ccc(COCc2ccc(O)cc2)cc1"
        ) == "4,4'-[oxybis(methylene)]diphenol"

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
        #; the substitutive form remains the conservation
        # fallback when the multiplicative path declines.
        ("OCc1cc(CO)cc(CO)c1", "(benzene-1,3,5-triyl)trimethanol"),
        # ⚠ PRE-EXISTING DEFECT, marked so rather than silently red. We emit
        # `(2-chloromethylphenyl)methanol` -- the INNER enclosing marks around
        # `chloromethyl` are lost, so the name reads as `2-chloro` + `methylphenyl`,
        # a different substitution pattern. `chloromethyl` is a compound substituent
        # and requires the marks. Measured 2026-07-30 against pre-Phase-C
        # in a worktree: BYTE-IDENTICAL there, so this is NOT a Phase C
        # regression -- it is the same lost-enclosing-mark class Task 5a fixed for
        # `_HALOALKYL_RE`'s truncated multiplier list, surviving on a different path.
        pytest.param(
            "ClCc1ccccc1CO", "(2-(chloromethyl)phenyl)methanol",
            marks=pytest.mark.xfail(
                strict=True,
                reason="pre-existing lost enclosing marks: emits "
                       "'(2-chloromethylphenyl)methanol'; identical at 974aba86",
            ),
        ),
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
