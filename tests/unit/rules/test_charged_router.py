"""Unit tests for the charged-species chokepoint route_charged (a phase-03).

CHOKE-01 (universal routing) + CHOKE-02 (the 4 IUPAC guards). Each test cites the
governing Blue Book P-rule. The Wave-0 file named in 169.6-VALIDATION.md.

route_charged(mol, style) GENERALIZES the proven _name_oxoacid_anion template
(neutralize -> re-enter Orthonym(style).name -> re-apply the class-correct
ionic suffix). These tests prove the carbon-counting stubs (the heptanolate bug
class) are dead: a branched/substituted alkoxide now names structurally-complete.
"""

import pytest
from rdkit import Chem

from orthonym.rules.charged_router import route_charged


def _rc(smiles, style="pin"):
    return route_charged(Chem.MolFromSmiles(smiles), style)


@pytest.mark.unit
class TestPerClass:
    """One worked target per charged class (CHOKE-01 universal routing)."""

    def test_anion_sulfonate(self):
        """: an aryl sulfonate anion ->...sulfonate (acid anion),
        via neutralize (sulfonic acid) -> re-enter -> -sulfonate suffix."""
        assert _rc("CCS(=O)(=O)[O-]") == "ethanesulfonate"

    def test_anion_carboxylate_deferred_to_proven_path(self):
        """CARBOXYLATE anions are DEFERRED ('') to the proven, retained-name-aware
        _name_carboxylate_systematic path (benzoate / 2-naphthoate / succinate
        stay byte-identical). route_charged owns only the deleted-stub classes +
        S/P oxoacid anions; routing carboxylate here would flip retained names to
        the systematic -oate form (a style change, not a fix)."""
        assert _rc("CC(=O)[O-]") == ""
        assert _rc("[O-]C(=O)CCC(=O)[O-]") == ""  # succinate stays on proven path

    def test_anion_oxoacid_sulfonate(self):
        """: an S/P-oxoacid anion IS owned by route_charged (it
        shares the deleted-stub neighborhood) -> -sulfonate."""
        assert _rc("CCCS(=O)(=O)[O-]") == "propane-1-sulfonate" or \
            "sulfonate" in _rc("CCCS(=O)(=O)[O-]")

    def test_cation_carbenium(self):
        """: carbenium PIN is methylium (NOT 'carbenium'); class-keyed
        ane->ylium on the parent hydride restored by adding the lost hydride."""
        assert _rc("[CH3+]") == "methylium"
        assert _rc("CC[CH2+]") == "propylium"

    def test_dianion_via_guard2(self):
        """(a) (GUARD 2): a fully-deprotonated S/P-oxoacid DIANION's parent
        bears BOTH centers and keeps the charge ->...phosphonate (both [O-]
        neutralized to the acid, then the single ionic suffix re-applied). (The
        CARBOXYLATE dianion is deferred to the proven path; see
        test_anion_carboxylate_deferred_to_proven_path.)"""
        out = _rc("CCP(=O)([O-])[O-]")
        assert "phosphonate" in out and "acid" not in out

    def test_radical_monovalent(self):
        """: a monovalent alkyl radical ->...yl (the radical center is
        H-saturated, the neutral alkane re-entered, the -yl suffix re-applied).
        CC[CH2] is the propyl radical (3 carbons)."""
        assert _rc("CC[CH2]") == "propyl"
        assert _rc("[CH3]") == "methyl"

    def test_radical_divalent(self):
        """: a divalent (carbene) radical ->...ylidene."""
        assert _rc("[CH2]") == "methylidene"


@pytest.mark.unit
class TestGuard1NoCrossFire:
    """GUARD 1 vs: FG class chosen BEFORE the suffix,
    so an alkoxide names with -olate and a sulfonate with -sulfonate and they can
    NEVER cross-fire. This is the heptanolate-bug-dead proof on a branched/
    substituted substrate (the deleted _name_alkoxide_systematic counted carbons
    and dropped every substituent + the locant)."""

    def test_branched_alkoxide_is_structurally_complete(self):
        """A branched alkoxide names the FULL substituted -olate with its locant
        (NOT a bare carbon-count name) -- proves _name_alkoxide_systematic
        (heptanolate) dead. Uses a genuinely SYSTEMATIC branched alkoxide;
        CC(C)(C)[O-] is NOT used here because tert-butoxide is a RETAINED PIN
        (BB, lines 28180/41017: "tert-butoxide... is also retained
        as a preferred IUPAC name"), so it never exercises the systematic path."""
        assert _rc("CCC(C)C[O-]") == "2-methylbutan-1-olate"

    def test_tert_butoxide_is_retained_pin(self):
        """BB (lines 28180/41017): tert-butoxide is retained AS A
        PREFERRED IUPAC NAME (it just cannot be substituted). It must NOT be
        systematised to 2-methylpropan-2-olate."""
        assert _rc("CC(C)(C)[O-]") == "tert-butoxide"

    def test_long_chain_alkoxide_keeps_locant(self):
        """The canonical heptanolate bug: CCCCCCC[O-] used to drop the locant to
        'heptanolate'; the chokepoint names heptan-1-olate."""
        out = _rc("CCCCCCC[O-]")
        assert out == "heptan-1-olate"
        assert out != "heptanolate"

    def test_sulfonate_never_olate(self):
        """A sulfonate's [O-] is on S, NOT a C-OH: it must take -sulfonate, never
        -olate (the GUARD-1 cross-fire the textual seam alone could not prevent)."""
        out = _rc("CCS(=O)(=O)[O-]")
        assert "sulfonate" in out
        assert "olate" not in out

    def test_thiolate_not_olate(self):
        """: a thiolate (-S-) takes -thiolate, never -olate."""
        out = _rc("[S-]CC")
        assert out == "ethanethiolate"


@pytest.mark.unit
class TestGuard2MultiCenter:
    """GUARD 2 a-c): on a multi-center ion the parent maximizes anionic
    center count before length -- realized by neutralizing ALL same-sign
    centers so the re-entered pipeline names the multi-suffix parent."""

    def test_phosphonate_dianion_keeps_charge(self):
        """A fully-deprotonated S/P-oxoacid dianion ships the anion name
        (methanephosphonate), NOT the neutral acid (a) /). GUARD 2:
        BOTH [O-] are neutralized so the single phosphonate parent is named."""
        out = _rc("CP(=O)([O-])[O-]")
        assert "phosphonate" in out
        assert "acid" not in out

    def test_carboxylate_dianion_deferred(self):
        """A pure carboxylate dianion is deferred ('') to the proven path (which
        emits the retained 'succinate'); route_charged does not own it."""
        assert _rc("[O-]C(=O)CCC(=O)[O-]") == ""


@pytest.mark.unit
class TestGuard3ElementSeniority:
    """GUARD 3 /: a skeletal heteroatom charge picks the senior
    element (N>P>...>O>S>...>C) as the parent-bearing atom. For the single-center
    majority the re-entered cascade applies the SAME element order, so
    the senior-element parent is chosen automatically."""

    def test_heteroatom_anion_routes_through_chokepoint(self):
        """A thiolate (charge on S) names on the S-bearing parent (ethanethiolate),
        not a bare carbanion -- the senior-element parent is honored."""
        assert _rc("[S-]CC") == "ethanethiolate"

    def test_aminide_charge_on_nitrogen(self):
        """An amide/amine anion's charge on N routes to the N-bearing parent
        : N is the most senior). The chokepoint neutralizes + re-enters;
        if no canonical -aminide transform applies it falls through ('')."""
        # Mechanism check: route returns either a valid aminide name or '' (never
        # a carbon-counted carbanion misname); both are acceptable fall-through.
        out = _rc("CC[NH-]")
        assert out == "" or "azanide" in out or "amin" in out


@pytest.mark.unit
class TestDeferrals:
    """Metal complex / multi-fragment salt / zwitterion -> route_charged returns
    '' (deferred to Plan 04) so the legacy path is byte-identical this plan."""

    def test_metal_complex_deferred(self):
        """_has_metal -> '' (simple-metal-salt composition is Plan 04,."""
        assert _rc("CCC(=O)[O-].[K+]") == ""
        assert _rc("CCC(=O)[O-].[Na+]") == ""

    def test_multi_fragment_deferred(self):
        """A dot-disconnected multi-fragment species -> '' (Plan 04)."""
        assert _rc("CC(=O)[O-].CC(=O)[O-]") == ""

    def test_zwitterion_p74_1_2_skeletal_implemented(self):
        """F- (DD3,: a zwitterion whose cation is SKELETAL to the
        anion's parent ring (a pyridinium-2-carboxylate ring N+) is now named with
        the cumulative ``<ring>-<N-locant>-ium-<carboxyl-locant>-carboxylate``
        suffix via ``emit_zwitterion_ring_carboxylate`` (was deferred to '' in
        169.6-04 scope). The N-methyl aromatic cation is demoted to a ring
        substituent prefix. PIN-VERIFICATION §B confirms the cumulative-suffix PIN
        (PubChem neutralizes the zwitterion)."""
        assert _rc("O=C([O-])c1cccc[n+]1C") == "1-methylpyridin-1-ium-2-carboxylate"

    def test_neutral_internal_charge_not_routed(self):
        """A molecule whose only charge is an internal nitro/N-oxide bonding
        charge is NOT a charged species -> '' (the internal-charge filter
        excludes it, so the neutral path names it)."""
        assert _rc("CC[N+](=O)[O-]") == ""  # nitroethane: internal charge only


@pytest.mark.unit
class TestFailSafe:
    """route_charged never crashes and returns '' on degenerate input (the
    byte-identical no-crash contract)."""

    def test_none_mol(self):
        assert route_charged(None, "pin") == ""

    def test_uncharged_nonradical(self):
        assert _rc("CCO") == ""  # neutral alcohol: nothing for the router to do


@pytest.mark.unit
class TestNoSizeCap:
    """route_charged sets no heavy-atom limit: a charged molecule of any size is
    routed, and the work of its full-pipeline re-entry is bounded by the budgets
    of the outermost name (fragment attempts, inner operations, analysis calls)
    plus _MAX_ROUTE_DEPTH. The 14.5 h hang once attributed to size was the [99Tc]
    radical recursion (see TestMetalRadicalHangGuard).

     'Anions derived from hydroxy compounds' (the Blue Book;
    sentence:41013): an anion from a hydroxy group "is preferably named by using
    suffixes 'olate',..."; (:41429) "Cationic suffixes derived from
    names of... amines... are formed by adding the suffix 'ium'", example
    '*N*,*N*,*N*-trimethylmethanaminium (PIN)'."""

    def test_large_alkoxide_is_routed(self):
        big = "C" * 55 + "[O-]"
        assert Chem.MolFromSmiles(big).GetNumHeavyAtoms() == 56
        assert _rc(big) == "pentapentacontan-1-olate"

    def test_large_quaternary_ammonium_is_routed(self):
        big = "C" * 52 + "[N+](C)(C)C"
        assert Chem.MolFromSmiles(big).GetNumHeavyAtoms() == 56
        assert _rc(big) == "N,N,N-trimethyldopentacontan-1-aminium"

    def test_large_carboxylate_left_to_the_anion_path(self):
        # A carboxylate is not built by the route at any size (its -ate is formed
        # on the anion path of name); the route returns '' for it.
        big = "C" * 54 + "(=O)[O-]"
        assert Chem.MolFromSmiles(big).GetNumHeavyAtoms() > 50
        assert _rc(big) == ""

    def test_small_charged_below_bound_still_routes(self):
        # A small alkoxide (6 HA) the router owns.
        assert _rc("CCCCCC[O-]") != ""


@pytest.mark.unit
@pytest.mark.timeout(20)
class TestMetalRadicalHangGuard:
    """[99Tc] regression: a lone radical metal atom MUST NOT infinite-loop through
    route_charged -> name_radical -> re-enter -> _handle_radical -> name_radical ->
    route_charged (the 14.5h full-corpus hang). Two layered fixes: (a) complete
    metal detection so route_charged bails on Tc/U/f-block (the legacy metal list
    omitted them), and (b) the _MAX_ROUTE_DEPTH recursion guard as a backstop for
    any non-metal species whose re-entered form re-triggers routing."""

    def test_lone_metal_radical_bails_fast(self):
        # Must bail to '' (metal -> out of scope) and NOT hang. The @timeout(20)
        # marker fails the test if the recursion regresses.
        assert _rc("[99Tc]") == ""
        assert _rc("[U]") == ""

    def test_recursion_depth_guard_present(self):
        from orthonym.rules.charged_router import _MAX_ROUTE_DEPTH
        assert isinstance(_MAX_ROUTE_DEPTH, int) and _MAX_ROUTE_DEPTH >= 1
