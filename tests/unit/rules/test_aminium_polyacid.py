"""Aminium/betaine zwitterions with a MULTI-carboxylate (poly-anion) parent
(a phase).

Generalizes the GUARD-4 zwitterion path
(`charged_router._route_zwitterion` -> `_name_polyacid_zwitterion`) from
exactly-one-anion to >= 2 carboxylate anions on one acyclic skeleton, so a
single cation riding on a poly-acid parent (the glutamate/aspartate
zwitterion ANION -- net charge -1, both carboxyls deprotonated, the amine
still protonated) gets a real name instead of abstaining
("unknown organic compound").

Root cause (measured, a trace): TWO stacked gaps, not the one hypothesized.
1. `detect_species_type` classifies a net-NONZERO mixed-sign fragment as
   `'ion'`, not `'zwitterion'` (its own docstring: "Zwitterion - net zero
   charge but has both + and - atoms"), so `assemble_ion_name`'s `'ion'`
   branch (composer.py) never called `name_zwitterion`/`route_charged` for
   this shape at all -- neither its "cations only" nor "anions only" branch
   matches a MIXED ion_sites, so it fell straight to the empty-string
   fallback.
2. Even if reached, `_route_zwitterion`'s exactly-one-anion scope check
   (`len(cations) != 1 or len(anions) != 1: return ''`) would still have
   declined the 2-carboxylate shape.

Both are fixed: composer.py routes a mixed cation+anion `'ion'` through
`name_zwitterion` (pure ADD, RT-gated); `_name_polyacid_zwitterion` is tried
BEFORE the exactly-one-anion guard for >= 2 carboxylate anions.

RT-gated (`@pytest.mark.opsin_gate`); the top-level /OPSIN gate
(which also checks net-charge preservation) is the 0-wrong backstop.
"""
import pytest
from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# --- the 2 goal molecules (RT-verified via OPSIN 2026-08-17) --------------
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # Glutamate zwitterion anion: [NH3+]-CH(-COO-)-CH2-CH2-COO-.
    # Parent = pentanedioic acid (severed cation); attach locant = min(4, 2) = 2.
    ("[NH3+]C(CCC(=O)[O-])C(=O)[O-]", "2-azaniumylpentanedioate"),
    # Aspartate zwitterion anion: [NH3+]-CH(-COO-)-CH2-COO-.
    # Parent = butanedioic acid (severed cation); attach locant = min(3, 2) = 2.
    ("[NH3+]C(CC(=O)[O-])C(=O)[O-]", "2-azaniumylbutanedioate"),
])
def test_aminium_polyacid_zwitterion(namer, smi, expected):
    assert namer.name(smi) == expected


# --- regressions: untouched paths ----------------------------------------
@pytest.mark.opsin_gate
def test_carboxylate_betaine_unchanged(namer):
    # Single-anion GUARD-4 betaine path is a completely separate
    # branch (len(anions) == 1) -- must stay byte-identical.
    assert namer.name("C[N+](C)(C)CCC(=O)[O-]") == "3-(trimethylazaniumyl)propanoate"


@pytest.mark.opsin_gate
def test_l_carnitine_unchanged(namer):
    assert namer.name("C[N+](C)(C)C[C@H](O)CC(=O)[O-]") == "L-carnitine"


@pytest.mark.opsin_gate
def test_l_serine_zwitterion_unchanged(namer):
    # Single-anion protonated-amine zwitterion -- the established
    # retained/neutral-form amino-acid path (defer), unaffected by the
    # new multi-anion branch (len(anions) == 1 here).
    assert namer.name("[NH3+][C@@H](CO)C(=O)[O-]") == "L-serine"


@pytest.mark.opsin_gate
def test_dicarboxylate_dianion_no_cation_unchanged(namer):
    # Pure poly-anion (no cation at all): route_charged never calls
    # _route_zwitterion when there is no cation site -- a completely
    # separate dispatch path, must stay byte-identical.
    assert namer.name("[O-]C(=O)CCC(=O)[O-]") == "butanedioate"


@pytest.mark.opsin_gate
def test_phase3_choline_sulfate_unchanged(namer):
    assert namer.name("C[N+](C)(C)CCOS(=O)(=O)[O-]") == "2-(trimethylazaniumyl)ethyl sulfate"


@pytest.mark.opsin_gate
def test_phase3_bis_quaternary_ammonium_unchanged(namer):
    # /: the substitutive '-bis(aminium)' name is the PIN
    # (the Blue Book,:42160-42162,:42366); the multiplicative
    # 'hexane-1,6-diylbis(trimethylazanium)' is its general-tier fallback.
    assert namer.name("C[N+](C)(C)CCCCCC[N+](C)(C)C") == \
        "N1,N1,N1,N6,N6,N6-hexamethylhexane-1,6-bis(aminium)"


@pytest.mark.opsin_gate
def test_phase3_sulfonatobenzoate_unchanged(namer):
    assert namer.name("[O-]C(=O)c1ccc(cc1)S(=O)(=O)[O-]") == "4-sulfonatobenzoate"


# --- fail-closed: out-of-scope shapes must never emit a wrong name --------
@pytest.mark.opsin_gate
def test_mixed_carboxylate_sulfonate_zwitterion_failclosed(namer):
    # 1 cation + 1 carboxylate anion + 1 sulfonate anion -- mixed anion
    # types are out of scope for the new poly-carboxylate branch (it only
    # fires when EVERY anion classifies 'carboxylate'). Abstain (or some
    # non-fabricated fallback) is acceptable; a WRONG name is not.
    from orthonym.errors import is_failure_name
    out = namer.name("[NH3+]C(CS(=O)(=O)[O-])C(=O)[O-]")
    assert out is not None
    assert is_failure_name(out), f"expected an honest abstain, got: {out!r}"


@pytest.mark.opsin_gate
def test_two_cations_build_bis_azaniumyl(namer):
    # >1 cation is out of scope for the single-cation GUARD-4 path
    # and for `_name_polyacid_zwitterion` (both require exactly one cation),
    # but charged Slice B's `_name_primary_amine_azaniumyl_zwitterion`
    # (tried BEFORE the single-cation scope check) handles the multi-cation
    # shape: the anion is the parent and each primary -NH3+ is an `azaniumyl`
    # prefix, two of the same kind -> `bis(azaniumyl)`. This is the
    # ionic PIN, replacing the 4782742f neutral over-reach
    # (`2,3-diaminopentanedioic acid`). The builder full-InChIKey RT-gates its
    # own emission (0-wrong); verified independently via OPSIN below.
    from rdkit import Chem
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smi = "[NH3+]C(CC(=O)[O-])C([NH3+])C(=O)[O-]"
    out = namer.name(smi)
    assert out == "2,3-bis(azaniumyl)pentanedioate"
    assert "dioic acid" not in out  # not the non-PIN neutral form
    g = opsin_parse(out)
    assert g and Chem.MolToInchiKey(Chem.MolFromSmiles(g)) == \
        Chem.MolToInchiKey(Chem.MolFromSmiles(smi))


def test_ring_cation_declines_the_new_branch_directly():
    # Unit-level (no JVM needed): a ring-borne cation must be declined by
    # _name_polyacid_zwitterion itself is out of scope here), not
    # silently accepted with a wrong locant/prefix.
    from rdkit import Chem
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.charged_router import _name_polyacid_zwitterion

    smi = "C[n+]1ccc(CC(CC(=O)[O-])C(=O)[O-])cc1"
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    sites = get_ion_sites(mol)
    result = _name_polyacid_zwitterion(
        mol, sites['cations'], sites['anions'], 'pin')
    assert result == ''


# --- a phase review follow-up (Findings A/B/C) -----------------------
@pytest.mark.opsin_gate
def test_tricarboxylate_parent_failclosed(namer):
    # Finding B: a 3-carboxylate parent (`propane-1,2,3-tricarboxylic acid`
    # -- is named `...tricarboxylate` by `name_carboxylate_anion`,
    # which `_parent_has_chain_locants` does NOT recognize (only
    # anoate/enoate/ynoate/dioate). The cation sits on C2 (locant 2, NOT the
    # licensed-omission position), so shipping the prefix unlocanted would
    # silently cite the wrong ring/chain position. RT-verified 2026-08-17
    # (`.venv/bin/python -m orthonym`): before the Finding-B tightening this
    # emitted the unlocanted 'azaniumylpropane-1,2,3-tricarboxylate', which
    # OPSIN parses as a DIFFERENT molecule (NH3+ defaults onto C1) and the
    # outer gate suppressed -> abstain. `_name_polyacid_zwitterion`
    # now declines this shape itself (proactive fail-closed, not just an
    # accidental gate catch).
    from orthonym.errors import is_failure_name
    out = namer.name("[NH3+]C(CC(=O)[O-])(CC(=O)[O-])C(=O)[O-]")
    assert out is not None
    assert is_failure_name(out), f"expected an honest abstain, got: {out!r}"


def test_tricarboxylate_parent_declines_the_new_branch_directly():
    # Unit-level companion to the RT-gated test above: confirm the DECLINE
    # happens inside `_name_polyacid_zwitterion` itself (Finding B), not
    # merely downstream in the outer gate.
    from rdkit import Chem
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.charged_router import _name_polyacid_zwitterion

    smi = "[NH3+]C(CC(=O)[O-])(CC(=O)[O-])C(=O)[O-]"
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    sites = get_ion_sites(mol)
    result = _name_polyacid_zwitterion(
        mol, sites['cations'], sites['anions'], 'pin')
    assert result == ''


@pytest.mark.opsin_gate
def test_compound_cation_prefix_polyacid_branch(namer):
    # Finding C: a COMPOUND (substituted) cation prefix -- a quaternary
    # trimethylazaniumyl, not the bare 'azaniumyl' of the goal molecules --
    # on the poly-acid branch, pinning `enclose_if_compound`'s parens arm
    # here (RT-verified 2026-08-17).
    assert namer.name("C[N+](C)(C)C(CC(=O)[O-])C(=O)[O-]") == \
        "2-(trimethylazaniumyl)butanedioate"


def test_multi_branch_onium_atom_drop_declines_directly():
    # Finding A: a quaternary onium with the two carboxylate anions on
    # DIFFERENT branches (methyl, methyl, -CH2COO- [anions[0]], -CH2COO-
    # [anions[1]]). Before the atom-coverage guard, `parent_attach_idx` was
    # computed from `anions[0]` alone, so severing at that bond stranded
    # anions[1]'s whole branch on the CATION side -- `cation_to_prefix`
    # would happily absorb it as a neutral 'carboxymethyl' N-substituent
    # (measured: `cation_to_prefix(mol, 1, 3)` == 'carboxymethyldimethyl-
    # azaniumyl'), silently dropping that anion's charge. The new
    # `_atom_coverage_ok_for_polyacid_zwitterion` guard must decline this
    # shape before `cation_to_prefix` is even called.
    from rdkit import Chem
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.charged_router import _name_polyacid_zwitterion

    smi = "C[N+](C)(CC(=O)[O-])CC(=O)[O-]"
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    sites = get_ion_sites(mol)
    result = _name_polyacid_zwitterion(
        mol, sites['cations'], sites['anions'], 'pin')
    assert result == ''
