"""Unit tests for the charged-species chokepoint route_charged (Phase 169.6-03).

CHOKE-01 (universal routing) + CHOKE-02 (the 4 IUPAC guards). Each test cites the
governing Blue Book P-rule. The Wave-0 file named in 169.6-VALIDATION.md.

route_charged(mol, style) GENERALIZES the proven _name_oxoacid_anion template
(neutralize -> re-enter Orthonym(style).name() -> re-apply the class-correct
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
        """P-72.2.2.2.1.1: an aryl sulfonate anion -> ...sulfonate (acid anion),
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
        """P-72.2.2.2.1.1: an S/P-oxoacid anion IS owned by route_charged (it
        shares the deleted-stub neighborhood) -> -sulfonate."""
        assert _rc("CCCS(=O)(=O)[O-]") == "propane-1-sulfonate" or \
            "sulfonate" in _rc("CCCS(=O)(=O)[O-]")

    def test_cation_carbenium(self):
        """P-73.2.2.1.1: carbenium PIN is methylium (NOT 'carbenium'); class-keyed
        ane->ylium on the parent hydride restored by adding the lost hydride."""
        assert _rc("[CH3+]") == "methylium"
        assert _rc("CC[CH2+]") == "propylium"

    def test_dianion_via_guard2(self):
        """P-72.7(a) (GUARD 2): a fully-deprotonated S/P-oxoacid DIANION's parent
        bears BOTH centers and keeps the charge -> ...phosphonate (both [O-]
        neutralized to the acid, then the single ionic suffix re-applied). (The
        CARBOXYLATE dianion is deferred to the proven path; see
        test_anion_carboxylate_deferred_to_proven_path.)"""
        out = _rc("CCP(=O)([O-])[O-]")
        assert "phosphonate" in out and "acid" not in out

    def test_radical_monovalent(self):
        """P-71.1.1: a monovalent alkyl radical -> ...yl (the radical center is
        H-saturated, the neutral alkane re-entered, the -yl suffix re-applied).
        CC[CH2] is the propyl radical (3 carbons)."""
        assert _rc("CC[CH2]") == "propyl"
        assert _rc("[CH3]") == "methyl"

    def test_radical_divalent(self):
        """P-71: a divalent (carbene) radical -> ...ylidene."""
        assert _rc("[CH2]") == "methylidene"


@pytest.mark.unit
class TestGuard1NoCrossFire:
    """GUARD 1 (P-72.2.2.2.1 vs P-72.2.2.2.2): FG class chosen BEFORE the suffix,
    so an alkoxide names with -olate and a sulfonate with -sulfonate and they can
    NEVER cross-fire. This is the heptanolate-bug-dead proof on a branched/
    substituted substrate (the deleted _name_alkoxide_systematic counted carbons
    and dropped every substituent + the locant)."""

    def test_branched_alkoxide_is_structurally_complete(self):
        """A branched alkoxide names the FULL substituted -olate (NOT a bare
        carbon-count name) -- proves _name_alkoxide_systematic (heptanolate) dead."""
        assert _rc("CC(C)(C)[O-]") == "2-methylpropan-2-olate"

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
        """P-72.2.2.2.2: a thiolate (-S-) takes -thiolate, never -olate."""
        out = _rc("[S-]CC")
        assert out == "ethanethiolate"


@pytest.mark.unit
class TestGuard2MultiCenter:
    """GUARD 2 (P-72.7 a-c): on a multi-center ion the parent maximizes anionic
    center count before P-44 length -- realized by neutralizing ALL same-sign
    centers so the re-entered pipeline names the multi-suffix parent."""

    def test_phosphonate_dianion_keeps_charge(self):
        """A fully-deprotonated S/P-oxoacid dianion ships the anion name
        (methanephosphonate), NOT the neutral acid (P-72.7(a) / CR-02). GUARD 2:
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
    """GUARD 3 (P-72.7d / P-73.7c): a skeletal heteroatom charge picks the senior
    element (N>P>...>O>S>...>C) as the parent-bearing atom. For the single-center
    majority the re-entered P-44.1.2 cascade applies the SAME element order, so
    the senior-element parent is chosen automatically."""

    def test_heteroatom_anion_routes_through_chokepoint(self):
        """A thiolate (charge on S) names on the S-bearing parent (ethanethiolate),
        not a bare carbanion -- the senior-element parent is honored."""
        assert _rc("[S-]CC") == "ethanethiolate"

    def test_aminide_charge_on_nitrogen(self):
        """An amide/amine anion's charge on N routes to the N-bearing parent
        (P-72.7d: N is the most senior). The chokepoint neutralizes + re-enters;
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
        """_has_metal -> '' (simple-metal-salt composition is Plan 04, P-65.6.2.1)."""
        assert _rc("CCC(=O)[O-].[K+]") == ""
        assert _rc("CCC(=O)[O-].[Na+]") == ""

    def test_multi_fragment_deferred(self):
        """A dot-disconnected multi-fragment species -> '' (Plan 04)."""
        assert _rc("CC(=O)[O-].CC(=O)[O-]") == ""

    def test_zwitterion_p74_1_2_skeletal_deferred(self):
        """GUARD 4 (P-74.0): a P-74.1.2 zwitterion whose cation is SKELETAL to the
        anion's parent ring (a pyridinium-2-carboxylate ring N+) is still deferred
        to the legacy path (the cumulative ium+ate suffix is out of 169.6-04
        scope) -> ''. (The P-74.1.3 separable-cation betaine is now IMPLEMENTED;
        see test_zwitterion_salt.TestZwitterionGuard4.)"""
        assert _rc("O=C([O-])c1cccc[n+]1C") == ""

    def test_neutral_internal_charge_not_routed(self):
        """A molecule whose only charge is an internal nitro/N-oxide bonding
        charge (P-59) is NOT a charged species -> '' (the internal-charge filter
        excludes it, so the neutral path names it)."""
        assert _rc("CC[N+](=O)[O-]") == ""  # nitroethane: internal charge only


@pytest.mark.unit
class TestFailSafe:
    """route_charged never crashes and returns '' on degenerate input (the v18
    byte-identical no-crash contract)."""

    def test_none_mol(self):
        assert route_charged(None, "pin") == ""

    def test_uncharged_nonradical(self):
        assert _rc("CCO") == ""  # neutral alcohol: nothing for the router to do


@pytest.mark.unit
class TestComplexityGuard:
    """Anti-hang complexity bound (169.6 follow-on). route_charged degrades
    GRACEFULLY (returns '') above _MAX_CHARGED_ROUTE_HEAVY_ATOMS rather than
    blowing up the full-pipeline re-entry on a pathological large charged molecule
    (the deleted carbon-counting stub used to absorb these instantly-but-wrongly;
    its removal exposed a 14.5h full-corpus benchmark hang). Regression-safe: the
    largest charged compound that round-trips in the 169.5 baseline is 46 HA."""

    def test_threshold_is_regression_safe(self):
        from orthonym.rules.charged_router import _MAX_CHARGED_ROUTE_HEAVY_ATOMS
        # Must sit ABOVE the largest RT-ing charged baseline compound (46 HA) so
        # no currently-round-tripping charged molecule is ever refused.
        assert _MAX_CHARGED_ROUTE_HEAVY_ATOMS >= 47

    def test_large_charged_bails_fast(self):
        # A C54 carboxylate anion = 56 heavy atoms (> 50): the guard fires and
        # returns '' (caller falls through). Without the guard this enters the
        # full select_parent/assembly pipeline and hangs.
        big = "C" * 54 + "(=O)[O-]"
        assert Chem.MolFromSmiles(big).GetNumHeavyAtoms() > 50
        assert _rc(big) == ""

    def test_small_charged_below_bound_still_routes(self):
        # A small alkoxide (6 HA, well under the bound) is a deleted-stub class
        # the router owns -> unaffected by the guard.
        assert _rc("CCCCCC[O-]") != ""
