"""a phase: resonance-shifted azide/diazo/diazonium drawings.

RDKit does not normalise resonance forms on ``MolFromSmiles``: each drawing
keeps its own literal bond orders / formal charges. The canonical drawing of
each class already names (a trace, `internal notes`);
its resonance-shifted TWIN abstained before this fix. Acceptance is full
round-trip InChIKey identity to the canonical molecule -- the emitted name
string may legitimately differ (e.g. a demoted systematic form), per the
plan's TDD contract.
"""

from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import name_compound
from orthonym.data.resonance_templates import (
    classify_resonance_chain,
    find_resonance_chains,
)


def _full_rt(smiles: str, name: str) -> bool:
    from orthonym.validation.opsin_roundtrip import opsin_parse
    o = opsin_parse(name)
    if not o:
        return False
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(o))


# ---------------------------------------------------------------------------
# Regression guard: canonical drawings keep their exact existing name.
# ---------------------------------------------------------------------------

def test_canonical_azide_unchanged():
    assert name_compound("CN=[N+]=[N-]") == "azidomethane"


def test_canonical_diazonium_unchanged():
    assert name_compound("c1ccccc1[N+]#N") == "benzenediazonium"


def test_canonical_diazo_unchanged():
    assert name_compound("C=[N+]=[N-]") == "diazomethane"


# ---------------------------------------------------------------------------
# The resonance-shifted twins: must now name and full-InChIKey round-trip to
# the SAME molecule as their canonical counterpart (RED before this fix).
# ---------------------------------------------------------------------------

def test_azide_twin_names_and_round_trips():
    smi = "C[N-][N+]#N"
    name = name_compound(smi)
    assert name and "unknown" not in name, name
    assert _full_rt(smi, name), name
    # same molecule as the canonical drawing
    assert inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles("CN=[N+]=[N-]"))


def test_diazonium_twin_names_and_round_trips():
    smi = "c1ccccc1N=[N+]"
    name = name_compound(smi)
    assert name and "unknown" not in name, name
    assert _full_rt(smi, name), name
    assert inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles("c1ccccc1[N+]#N"))


def test_genuine_radical_cation_not_swept_by_diazonium_carveout():
    """Defense-in-depth (a review review, 2026-08-15): ``c1ccccc1[N+]=N`` is a
    GENUINE open-shell monoradical cation (1 radical electron on a degree-2
    N), not the valence-shortfall artifact the carve-out in
    ``rules.charged_router.route_charged`` targets (2 spurious radical
    electrons on a degree-1 terminal N whose sole bond is a double bond).
    ``classify_cation`` still labels it 'diazonium' (bond-order-only
    classifier), so a carve-out keyed on "1 radical site == 1 cation site,
    classified diazonium" alone is broader than intended and would build a
    wrong ``benzenediazonium`` candidate for a real radical species. The
    tightened carve-out must decline to clear ``radical_sites`` here, so
    ``route_charged``'s ordinary "charged AND radical -> bail" guard fires
    THE PRODUCER ITSELF, never relying on SELF-01 downstream to catch it."""
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules import charged_router as cr

    mol = Chem.MolFromSmiles("c1ccccc1[N+]=N")
    sites = get_ion_sites(mol)
    assert sites['cations'][0]['atom_idx'] is not None  # sanity: a cation exists

    calls = []
    original = cr._name_diazonium

    def _spy(m, cation_idx, style):
        calls.append(cation_idx)
        return original(m, cation_idx, style)

    cr._name_diazonium = _spy
    try:
        result = cr.route_charged(mol, "pin")
    finally:
        cr._name_diazonium = original

    assert result == '', result
    assert calls == [], "route_charged must not reach _name_diazonium for a genuine radical cation"

    # Whatever the overall pipeline ultimately reports, it must never be the
    # diazonium-specific wrong candidate this carve-out used to build.
    assert name_compound("c1ccccc1[N+]=N") != "benzenediazonium"


def test_diazo_twin_names_and_round_trips():
    smi = "[CH2-][N+]#N"
    name = name_compound(smi)
    assert name and "unknown" not in name, name
    assert _full_rt(smi, name), name
    assert inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles("C=[N+]=[N-]"))


# ---------------------------------------------------------------------------
# Vector-table unit tests: closed 3-set, fail-closed on a non-matching chain.
# ---------------------------------------------------------------------------

def test_classify_resonance_chain_all_six_rows():
    assert classify_resonance_chain((1, 2, 2), (0, 0, 1, -1)) == 'azido'
    assert classify_resonance_chain((1, 1, 3), (0, -1, 1, 0)) == 'azido'
    assert classify_resonance_chain((1, 3), (0, 1, 0)) == 'diazonium'
    assert classify_resonance_chain((1, 2), (0, 0, 1)) == 'diazonium'
    assert classify_resonance_chain((2, 2), (0, 1, -1)) == 'diazo'
    assert classify_resonance_chain((1, 3), (-1, 1, 0)) == 'diazo'


def test_classify_resonance_chain_no_false_positive():
    # an ordinary amine chain (single bond, no charge) -- not in the closed set
    assert classify_resonance_chain((1,), (0, 0)) is None
    # a plain hydrazine R-NH-NH2 chain
    assert classify_resonance_chain((1, 1), (0, 0, 0)) is None
    # a protonated terminal hydrazinium R-NH-NH3+
    assert classify_resonance_chain((1, 1), (0, 0, 1)) is None
    # a nitro-shaped vector accidentally offered to the chain classifier
    assert classify_resonance_chain((2, 1), (0, 1, -1)) is None


def test_find_resonance_chains_no_false_positive_on_amine_and_nitro():
    # ordinary amine: no charged N-chain of the closed shape
    mol = Chem.MolFromSmiles("CCN")
    assert find_resonance_chains(mol) == []
    # nitro group branches (2 oxygens off the charged N) -- must not be
    # mis-read as a linear N-chain
    mol = Chem.MolFromSmiles("CC[N+](=O)[O-]")
    assert find_resonance_chains(mol) == []
    # azo compound ArN=NAr': both ends are non-nitrogen attach atoms, so the
    # chain never terminates in N -- must not match
    mol = Chem.MolFromSmiles("c1ccccc1/N=N/c1ccccc1")
    assert find_resonance_chains(mol) == []


def test_find_resonance_chains_free_azide_anion_never_seeds():
    # a free azide anion has no organic attachment -- every atom is nitrogen,
    # so no non-nitrogen atom exists to seed a walk into it.
    mol = Chem.MolFromSmiles("[Na+].[N-]=[N+]=[N-]")
    assert find_resonance_chains(mol) == []


def test_find_resonance_chains_finds_both_drawings():
    canonical = Chem.MolFromSmiles("CN=[N+]=[N-]")
    twin = Chem.MolFromSmiles("C[N-][N+]#N")
    assert [c for c, _ in find_resonance_chains(canonical)] == ['azido']
    assert [c for c, _ in find_resonance_chains(twin)] == ['azido']

    canonical_diazo = Chem.MolFromSmiles("C=[N+]=[N-]")
    twin_diazo = Chem.MolFromSmiles("[CH2-][N+]#N")
    assert [c for c, _ in find_resonance_chains(canonical_diazo)] == ['diazo']
    assert [c for c, _ in find_resonance_chains(twin_diazo)] == ['diazo']
