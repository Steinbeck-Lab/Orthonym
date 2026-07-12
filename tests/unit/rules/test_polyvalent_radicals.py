"""P-29.2 / P-71.2.3 multi-site free-valence (polyradical) PINs (w2f p5 Tasks 1-3).

Names here are ASSERTED as raw strings straight off the rule functions (no OPSIN
in the assertion path). Every expected string is OPSIN-2.9-`-r`-verified in
 §Item-1 §D (radical names need the -r
OPSIN flag for name->structure round-trip; the phase gate scores by exact string,
so these gate cleanly — research §E2).
"""
import pytest
from rdkit import Chem

from orthonym.perception.ions import get_radical_sites
from orthonym.rules.ions import emit_parent_hydride_polyvalent_suffixes


def _mol_centers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    sites = get_radical_sites(mol)
    return mol, [(s["atom_idx"], s["n_electrons"]) for s in sites]


class TestHomogeneousPolyradical:
    @pytest.mark.parametrize("smiles,expected", [
        ("[CH2][CH2]",        "ethane-1,2-diyl"),        # BB P-71.2.3 verbatim; 'e' kept before 'd'
        ("[CH2]C[CH2]",       "propane-1,3-diyl"),       # symmetric
        ("[CH2]CC[CH2]",      "butane-1,4-diyl"),
        ("[CH2][CH]C",        "propane-1,2-diyl"),       # asymmetric set -> lowest-set orientation {1,2}
        ("CC([CH2])[CH2]",    "2-methylpropane-1,3-diyl"),# substituent prefix on the diyl parent
        ("[CH2][CH][CH2]",    "propane-1,2,3-triyl"),    # BB P-71.2.3 verbatim; three 1e sites
        ("C[C]C[C]C",         "pentane-2,4-diylidene"),  # BB P-71.2.3 verbatim; two 2e sites
    ])
    def test_homogeneous(self, smiles, expected):
        mol, centers = _mol_centers(smiles)
        assert emit_parent_hydride_polyvalent_suffixes(mol, centers) == expected

    def test_symmetric_is_orientation_stable(self):
        # both chain ends equivalent -> identical name (determinism sanity)
        mol, centers = _mol_centers("[CH2]C[CH2]")
        assert emit_parent_hydride_polyvalent_suffixes(mol, centers) == "propane-1,3-diyl"


class TestScopeGuardsFailClosed:
    @pytest.mark.parametrize("smiles", [
        "[CH2][N]",     # hetero center -> not all-C
        "[O]CC[O]",     # oxygen centers -> not all-C
        "[CH]1CC[CH]CC1",  # ring centers -> acyclic-only guard
        "[c]1cc[c]cc1",    # aromatic ring -> out of scope
    ])
    def test_out_of_scope_returns_empty(self, smiles):
        mol, centers = _mol_centers(smiles)
        assert emit_parent_hydride_polyvalent_suffixes(mol, centers) == ""

    def test_single_center_returns_empty(self):
        # single site is the single-center primitive's job, not this one
        mol, centers = _mol_centers("C[CH]C")
        assert len(centers) == 1
        assert emit_parent_hydride_polyvalent_suffixes(mol, centers) == ""

    # NOTE: the Task-1-authored ``test_mixed_suffix_deferred_to_task3`` (mixed
    # 1e+2e -> '') was removed here in Task 3: Task 3 deliberately LIFTS that
    # homogeneous-only deferral, and the mixed case is now asserted positively in
    # ``TestMixedSuffix.test_mixed_yl_ylidene`` ([CH2][CH] -> ethan-1-yl-2-ylidene).


class TestRouteChargedMultiSite:
    @pytest.mark.parametrize("smiles,expected", [
        ("[CH2][CH2]",     "ethane-1,2-diyl"),
        ("[CH2]C[CH2]",    "propane-1,3-diyl"),
        ("[CH2][CH][CH2]", "propane-1,2,3-triyl"),
        ("CC([CH2])[CH2]", "2-methylpropane-1,3-diyl"),
    ])
    def test_route_charged_emits_polyradical(self, smiles, expected):
        from orthonym.rules.charged_router import route_charged
        assert route_charged(Chem.MolFromSmiles(smiles), "pin") == expected

    @pytest.mark.parametrize("smiles", ["[O]CC[O]", "[CH2][N]", "[CH]1CC[CH]CC1"])
    def test_route_charged_bails_out_of_scope(self, smiles):
        from orthonym.rules.charged_router import route_charged
        assert route_charged(Chem.MolFromSmiles(smiles), "pin") == ""

    def test_name_radical_no_structure_dropping_oxyl(self):
        # deleted shortcut: [O]CC[O] must NOT return 'ethoxyl' (structure-dropping)
        from orthonym.rules.radicals import name_radical
        assert name_radical(Chem.MolFromSmiles("[O]CC[O]"), style="pin") == ""

    def test_single_site_radical_unchanged(self):
        # PROTECT: single-site path (route_charged -> emit_parent_hydride_cumulative_suffix) byte-identical
        from orthonym.rules.charged_router import route_charged
        assert route_charged(Chem.MolFromSmiles("CC[CH]CC"), "pin") == "pentan-3-yl"
        assert route_charged(Chem.MolFromSmiles("C[CH]C"), "pin") == "propan-2-yl"


class TestMixedSuffix:
    @pytest.mark.parametrize("smiles,expected", [
        ("[CH2][CH]",   "ethan-1-yl-2-ylidene"),   # BB P-71.6 verbatim; yl gets locant 1; 'e' elided before 'y'
        ("[CH]C[CH2]",  "propan-1-yl-3-ylidene"),   # yl-lowest tie-break picks 1-yl over 3-yl
    ])
    def test_mixed_yl_ylidene(self, smiles, expected):
        mol, centers = _mol_centers(smiles)
        assert emit_parent_hydride_polyvalent_suffixes(mol, centers) == expected

    def test_citation_order_yl_before_ylidene(self):
        # even when ylidene holds the lower locant, yl is CITED first (P-29.3.2.2)
        mol, centers = _mol_centers("[CH2][CH]")
        out = emit_parent_hydride_polyvalent_suffixes(mol, centers)
        assert out.index("yl") < out.index("ylidene")

    def test_homogeneous_still_pass(self):
        # regression: Task 1 golds unaffected by the mixed lift
        mol, centers = _mol_centers("[CH2][CH2]")
        assert emit_parent_hydride_polyvalent_suffixes(mol, centers) == "ethane-1,2-diyl"
