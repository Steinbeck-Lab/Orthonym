"""The cluster tell must measure ATOMS, never the name's character count.

Task Z2. ``eval/cluster.py`` used to flag ``len(name) / heavy_atoms < 0.6`` as
``name_length_collapse``. A character count is anti-correlated with coverage,
so that test got BOTH directions wrong, and each direction has a case here:

  * a correct name that happens to be SHORT was flagged (false positive), and
  * a partial name that happens to be LONG was missed (false negative) -- the
    larger error in practice, 15 of 16 real collapses on the 1000-row split.

These tests pin the measured replacement against both. They construct run rows
directly, so no JVM and no naming run is involved: ``opsin_smiles`` is exactly
what a run record already stores for every row whose name OPSIN could parse.
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cluster import coverage_collapse, structural_tells  # noqa: E402

# Real molecule / real OPSIN parse-back pairs.
CHOLESTEROL = "CC(C)CCCC(C)C1CCC2C1(C)CCC1C2CC=C2CC(O)CCC12C"          # 28 heavy
TROPANE_HOST = ("CC1(C)C2CCC1(C)C(=O)C2OC(=O)C(CO)c1ccccc1."           # 38 heavy
                "CN1C2CCC1CC(O)C2")


def row(name, opsin_smiles, smiles, outcome="constitution_mismatch"):
    return {"smiles": smiles, "name": name, "opsin_smiles": opsin_smiles,
            "outcome": outcome}


def test_short_correct_name_on_a_large_molecule_is_not_flagged():
    """The direction the character count got backwards, case 1.

    Two rows, both correct, both fully covering. What separates them is only
    how verbose the name is:

      'cholest-5-en-3-ol' 17 chars / 28 HA = 0.607 -- cleared the old 0.6
                           threshold by 0.007, i.e. a correct steroid name sat
                           one character away from being reported as a defect;
      'cholesterol' 11 chars / 28 HA = 0.393 -- the old test FLAGGED it.

    Honesty note: the 0.393 row is CONSTRUCTED. No such false positive occurs
    in the committed run records (measured: 0 rows flagged by the length test
    and cleared by this one). The length test's observed damage was blindness,
    not false alarms -- see the long-name case below. This row pins the
    anti-correlation anyway, because a retained name is exactly what the
    pipeline would emit if trivial-name coverage widened.
    """
    mol = Chem.MolFromSmiles(CHOLESTEROL)
    assert mol.GetNumHeavyAtoms() == 28

    systematic = row("cholest-5-en-3-ol", CHOLESTEROL, CHOLESTEROL, outcome="rt_exact")
    assert len("cholest-5-en-3-ol") / 28 == pytest.approx(0.607, abs=0.001)
    assert coverage_collapse(systematic, mol) is False

    retained = row("cholesterol", CHOLESTEROL, CHOLESTEROL, outcome="rt_exact")
    assert len("cholesterol") / 28 < 0.6, "the old length test FLAGGED this"
    assert coverage_collapse(retained, mol) is False


def test_long_name_covering_few_atoms_is_flagged():
    """The direction the character count got backwards, case 2 -- the blind one.

    A verbose name is not a covering name. Here the name is 51 characters on a
    38-heavy-atom molecule (1.34 chars/HA, far above the old 0.6 threshold, so
    the length test saw nothing), while the structure it actually denotes
    carries 15 of the 38 heavy atoms.
    """
    mol = Chem.MolFromSmiles(TROPANE_HOST)
    ha = mol.GetNumHeavyAtoms()
    name = "(3S)-1,2,3,4-tetrahydroisoquinoline-3-carboxylic acid"
    parsed = "OC(=O)[C@@H]1Cc2ccccc2CN1"

    assert len(name) / ha > 0.6, "must be INVISIBLE to the old length test"
    assert coverage_collapse(row(name, parsed, TROPANE_HOST), mol) is True


def test_short_partial_name_is_still_flagged():
    """The one true positive the length test did find is not lost."""
    mol = Chem.MolFromSmiles(TROPANE_HOST)
    r = row("tropane", "[C@H]12CCC[C@H](CC1)N2C", TROPANE_HOST)
    assert coverage_collapse(r, mol) is True


def test_unparsed_name_yields_no_verdict():
    """No structure, no claim. Those rows already carry 'opsin_parse_fail'."""
    mol = Chem.MolFromSmiles(TROPANE_HOST)
    r = row("some-unparseable-name", None, TROPANE_HOST, outcome="opsin_parse_fail")
    assert coverage_collapse(r, mol) is False
    assert "name_coverage_collapse" not in structural_tells(r, mol)


def test_element_identity_is_required_not_just_a_count():
    """A carbon in the name may not stand in for a nitrogen in the molecule.

    sarcosine (CNCC(=O)O) and alanine share a formula AND an element multiset,
    which is why claimed atoms are a per-element intersection rather than a
    bare heavy-atom count. Here the parse-back has the right NUMBER of heavy
    atoms and the wrong elements, and must not count as covered.
    """
    mol = Chem.MolFromSmiles("C" * 34)  # 34 carbons
    ha = mol.GetNumHeavyAtoms()
    all_nitrogen = "N" * 34  # same heavy-atom COUNT, disjoint elements
    parsed = Chem.MolFromSmiles(all_nitrogen)
    assert parsed is not None and parsed.GetNumHeavyAtoms() == ha

    r = row("a-name", all_nitrogen, Chem.MolToSmiles(mol))
    assert coverage_collapse(r, mol) is True, (
        "equal heavy-atom COUNT with disjoint elements is 0% coverage"
    )


def test_tell_is_named_for_what_it_measures():
    """A regression guard on the label itself.

    The old key said 'length'. A tell that reports a measurement must not be
    named after the discarded proxy.
    """
    mol = Chem.MolFromSmiles(TROPANE_HOST)
    tells = structural_tells(row("tropane", "[C@H]12CCC[C@H](CC1)N2C",
                                 TROPANE_HOST), mol)
    assert "name_coverage_collapse" in tells
    assert "name_length_collapse" not in tells
