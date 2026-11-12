"""Findings of the code audit of 2026-11-11 (internal notes), one test each.

A wall-clock time-out (``WallClockTimeout``, a BaseException) can land anywhere. A process-wide table filled in place,
or a placeholder written before its value, then stayed wrong for the rest of the process and gave false PIN labels
('indolizidine', '(2-phosphonoethyl)phosphonic acid'). The carve-outs that ship a name OPSIN cannot read must be bound
to the input, not to the look of the name. The re-entry guard of the oxoacid-ester offer is keyed on the structure.
The staged cascade of the principal chain gives the chains of the full one.
"""
import pytest
from rdkit import Chem

from orthonym.wallclock import WallClockTimeout

pytestmark = pytest.mark.unit


def test_a_time_out_during_the_fused_index_build_leaves_it_empty(monkeypatch):
    from orthonym.data import fused_heterocycles as fh
    fh._build_pattern_index()
    full = (len(fh._REC_BY_SMILES), len(fh._REC_BY_SKELETON), sum(len(v) for v in fh._PATTERN_BUCKETS.values()))
    for attr in ("_PATTERN_BUCKETS", "_REC_BY_SMILES", "_REC_BY_SKELETON"):
        monkeypatch.setattr(fh, attr, {})
    calls = {"n": 0}
    real = fh.ring_skeleton_key

    def cut(pattern):
        calls["n"] += 1
        if calls["n"] == 20:
            raise WallClockTimeout("cut")
        return real(pattern)
    monkeypatch.setattr(fh, "ring_skeleton_key", cut)
    with pytest.raises(WallClockTimeout):
        fh._build_pattern_index()
    assert fh._PATTERN_BUCKETS == {}                      # the guard is empty: the next call builds again
    monkeypatch.setattr(fh, "ring_skeleton_key", real)
    fh._build_pattern_index()
    assert (len(fh._REC_BY_SMILES), len(fh._REC_BY_SKELETON),
            sum(len(v) for v in fh._PATTERN_BUCKETS.values())) == full


def test_a_time_out_leaves_no_placeholder_in_the_linked_phosphonic_cache(monkeypatch):
    from orthonym.rules import multiplicative as mu

    def cut(mol, matches):
        raise WallClockTimeout("cut")
    monkeypatch.setattr(mu, "_linked_phosphonic_acids_name", cut)
    mol = Chem.MolFromSmiles("OP(=O)(O)CCCCCCP(=O)(O)O")
    key = Chem.MolToSmiles(mol)
    mu._LINKED_PHOSPHONIC_CACHE.pop(key, None)
    with pytest.raises(WallClockTimeout):
        mu._try_linked_phosphonic_acids(mol)
    assert key not in mu._LINKED_PHOSPHONIC_CACHE


def test_a_list_name_ships_only_for_its_own_structure():
    from orthonym.data.natural_products import NAME_EXACT_NP_PARENTS, NATURAL_PRODUCT_DERIVATIVES
    from orthonym.namer import _is_exact_match_list_name
    smiles = next(s for s, n in NATURAL_PRODUCT_DERIVATIVES.items() if n == "germacrane")
    assert "germacrane" in NAME_EXACT_NP_PARENTS
    assert _is_exact_match_list_name(smiles, "germacrane")
    assert not _is_exact_match_list_name("CCO", "germacrane")


def test_the_polyacid_carve_out_is_bound_to_its_producer():
    from orthonym.validation.carveout_binding import polyacid_carveout_bound as _polyacid_carveout_bound
    name = "1,3-dicyanatodiphosphoric acid"
    smiles = "N#COP(=O)(O)OP(=O)(O)OC#N"
    opsin_reading = "N#COP(=O)(OC#N)OP(=O)(O)O"       # OPSIN 2.9.0 puts both groups on: an isomer
    assert _polyacid_carveout_bound(name, smiles, opsin_reading)
    assert not _polyacid_carveout_bound(name, "CCO", "CCO")
    assert not _polyacid_carveout_bound("1,3-dichlorodisulfuric acid", "c1ccccc1", "c1ccccc1")
    assert not _polyacid_carveout_bound(name, smiles, "N#COP(=O)(O)OP(=O)(O)O")   # an atom lost


@pytest.mark.opsin_gate
def test_the_oxoacid_offer_does_not_recurse_on_a_copy_of_the_molecule(monkeypatch):
    """pubchem_2000 row 0 (a steroid phosphate): the offer was entered on fresh copies until Python's recursion limit
    (130 calls); keyed on the structure it is entered a bounded number of times and the name is the same."""
    import orthonym.rules.seniority as se
    from tests.support.rt_assert import name_best_effort
    calls = {"n": 0}
    real = se._promoted_oxoacid_ester

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(se, "_promoted_oxoacid_ester", spy)
    row = name_best_effort("CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)COP(=O)(O)O)O)C)O)F)C")
    assert row["name"].endswith("-2-oxoethyl dihydrogen phosphate"), row
    assert calls["n"] < 100, calls


def test_the_staged_chain_cascade_gives_the_full_cascade_chains(monkeypatch):
    """A branched hydrocarbon of many maximal chains (the slow polymer rows of the PubChem time-outs):
    ORTHONYM_P2_VERIFY=1 compares the staged and the full cascade on every call."""
    import orthonym.perception.chains as ch
    from orthonym.perception.functional_groups import detect_functional_groups
    monkeypatch.setenv("ORTHONYM_P2_VERIFY", "1")
    before = len(ch.P2_STAGED_MISMATCHES)
    for smiles in ("CCC(C)(CC(C)(CC(C)(CC(C)(CC(C)(C)C)C)C)C)C", "CC(C)CC(CC(C)C)C(O)CC(C)CC(C)C",
                   "C=CC(C)C(CC=C)C(C)CC#C"):
        mol = Chem.MolFromSmiles(smiles)
        chain = ch.find_principal_chain(mol, detect_functional_groups(mol))
        assert chain
    assert len(ch.P2_STAGED_MISMATCHES) == before


@pytest.mark.opsin_gate
def test_a_stereo_block_with_fusion_letter_locants_is_not_doubled(monkeypatch):
    """CID 22084961 named in its input atom order (the entry point names the RDKit canonical spelling; the input
    order reaches the composer for the spellings it does not rewrite): '(4S,4aS,6R,8aR)-' was not read as a stereo
    block, a second block was put in front, OPSIN refused the name and the PIN was lost."""
    import orthonym.input_mol as im
    from tests.support.rt_assert import name_best_effort
    monkeypatch.setattr(im, "canonical_spelling", lambda s: None)
    row = name_best_effort("C[C@H]1CC(C([C@]2([C@H]1C[C@@H](CC2)C(C)C)C)(C34CC5CC(C3)CC(C5)C4)OC(=O)C=C)C")
    assert row["name"] == ("(4S,4aS,6R,8aR)-1-(adamantan-1-yl)-2,4,8a-trimethyl-6-(propan-2-yl)"
                           "decahydronaphthalen-1-yl prop-2-enoate"), row
    assert row["tier"] == "pin_verified", row
