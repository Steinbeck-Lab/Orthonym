"""task-W2 jar-absent-proper: producer-level atom-coverage close.

The default (PIN) tier used to ship atom-DROPPED wrong names from two producers
that silently discard atoms before emission — caught jar-present by SELF-01/OPSIN
but, jar-absent, by nothing:

  * Witness A — captopril `CC(CS)C(=O)N1CCCC1C(=O)O` -> `pyrrolidine-2-carboxylic
    acid` (the N-acyl chain dropped by rules/heterocycles.py
    get_heterocycle_substituents).
  * Witness B — methyl hydrogen sulfate `COS(=O)(=O)O` -> `methane` (the
    carbon-free sulfate ester dropped at DROP-01 in composer._generate_alkyl_
    prefixes).

The fix wires the shared shape-agnostic E1 partition primitive
(`e1_certificate.verify_atom_coverage`, over `_verify_partition`) at each
producer's return point, fed by atom-index data the producer already computes,
gated UNCONDITIONALLY: an atom-incomplete name VOIDS (the producer declines →
cascade → complete name or abstain), a complete name ships. It must void ONLY
atom-drops — every ordinary correct name still emits, jar-present AND jar-absent.
"""
import pytest

from rdkit import Chem

from orthonym.validation.e1_certificate import verify_atom_coverage


# --------------------------------------------------------------------------
# verify_atom_coverage primitive
# --------------------------------------------------------------------------
class TestVerifyAtomCoverage:
    def test_complete_partition_ok(self):
        mol = Chem.MolFromSmiles("CCO")  # ethanol, heavy atoms {0,1,2}
        v = verify_atom_coverage(mol, "ethanol", [[0, 1, 2]])
        assert v.ok, v.reason

    def test_dropped_atom_voids(self):
        mol = Chem.MolFromSmiles("COS(=O)(=O)O")  # 6 heavy atoms 0..5
        # Only the methyl carbon accounted -> the sulfate {1..5} is a drop.
        v = verify_atom_coverage(mol, "methane", [[0]])
        assert not v.ok
        assert "unbound" in v.reason

    def test_overlapping_groups_deduped_not_voided(self):
        # A benign double-listing of an atom must NOT trip a bound-twice refusal
        # (that would be a false void costing breadth); coverage is still complete.
        mol = Chem.MolFromSmiles("CCO")
        v = verify_atom_coverage(mol, "ethanol", [[0, 1], [1, 2]])  # atom 1 in both
        assert v.ok, v.reason

    def test_charged_not_voided_by_default(self):
        # The template owns the coverage axis, not the G1 charge scope: a
        # legitimately-charged parent whose atoms are fully covered is not voided.
        mol = Chem.MolFromSmiles("CC(=O)[O-]")  # acetate, 4 heavy atoms
        v = verify_atom_coverage(mol, "acetate", [[0, 1, 2, 3]])
        assert v.ok, v.reason


# --------------------------------------------------------------------------
# End-to-end: witnesses + by-construction guard, jar-present AND jar-absent
# --------------------------------------------------------------------------
def _force_jar_absent(monkeypatch):
    """Simulate the FABLE default-tier no-Java config (the hole this task closes)."""
    import orthonym.namer as namer_mod
    import orthonym.validation.atom_coverage as ac_mod
    import orthonym.validation.opsin_roundtrip as rt_mod
    monkeypatch.setattr(namer_mod, "_validity_gate_jar_present", lambda: False)
    monkeypatch.setattr(ac_mod, "find_opsin_jar", lambda: None)
    monkeypatch.setattr(rt_mod, "_find_opsin_jar", lambda *a, **k: None)


CAPTOPRIL = "CC(CS)C(=O)N1CCCC1C(=O)O"
METHYL_SULFATE = "COS(=O)(=O)O"

# The atom-DROPPED wrong strings the two producers used to ship jar-absent.
_WITNESS_WRONG = {
    CAPTOPRIL: "pyrrolidine-2-carboxylic acid",  # drops the N-acyl chain
    METHYL_SULFATE: "methane",                    # drops the sulfate ester
}

# Ordinary correct names that MUST keep emitting — the fix voids only drops.
_GUARD = {
    "CCO": "ethanol",
    "c1ccccc1": "benzene",
    "CC(C)O": "propan-2-ol",
    "CN1CCCC1": "1-methylpyrrolidine",
    "OC(=O)C1CCCN1": "pyrrolidine-2-carboxylic acid",   # proline: SAME string,
    #                                                     here it is CORRECT
    "Cc1ccccn1": "2-methylpyridine",
    "Cc1ccc(=O)[nH]c1": "5-methylpyridin-2(1H)-one",
    "C1CCNCC1": "piperidine",
    "CN1CCCCC1=O": "1-methylpiperidin-2-one",
    "OC(=O)c1cccnc1": "pyridine-3-carboxylic acid",
    "C1COCCN1": "morpholine",
    "CC(=O)O": "acetic acid",
    "Cc1ccccc1": "toluene",
    "Nc1ccccc1": "aniline",
    "CCCC": "butane",
    "CC(C)CC": "2-methylbutane",
}


@pytest.fixture
def namer():
    from orthonym import Orthonym
    return Orthonym(style="pin")


@pytest.mark.parametrize("smiles", list(_WITNESS_WRONG))
def test_witness_no_longer_ships_atom_drop_jar_absent(monkeypatch, smiles):
    """Jar-absent default tier: the witness must NOT ship its atom-dropped name.

    It abstains (`unknown organic compound`) or cascades to an atom-COMPLETE name
    — never the wrong drop.
    """
    _force_jar_absent(monkeypatch)
    from orthonym import Orthonym
    name = Orthonym(style="pin").name(smiles)
    assert name != _WITNESS_WRONG[smiles], (
        f"{smiles} still ships the atom-dropped wrong name {name!r}")


def test_methyl_sulfate_abstains_jar_absent(monkeypatch):
    _force_jar_absent(monkeypatch)
    from orthonym import Orthonym
    assert Orthonym(style="pin").name(METHYL_SULFATE) == "unknown organic compound"


@pytest.mark.parametrize("smiles,expected", list(_GUARD.items()))
def test_by_construction_guard_still_emits_jar_present(namer, smiles, expected):
    """The fix voids ONLY atom-drops: ordinary correct names still emit
    (jar-present, the normal operating config)."""
    assert namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", list(_GUARD.items()))
def test_by_construction_guard_still_emits_jar_absent(monkeypatch, smiles, expected):
    """Same guard with the jar forced absent — the unconditional coverage check
    must not void any of these complete names."""
    _force_jar_absent(monkeypatch)
    from orthonym import Orthonym
    assert Orthonym(style="pin").name(smiles) == expected
