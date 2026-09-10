#!/usr/bin/env python3
"""WSB-03 (a phase) coverage for the centres CIP-engine bridge.

Covers:
  * centres-ON: with the jar present + Java available, the CIP validation
    suite scores 281/290 via the centres engine (1.2.1 tagged release).
  * graceful RDKit fallback: when _find_centres_jar() returns None (monkeypatch)
    OR Java is absent, centres_label_batch returns None so the caller falls
    back to RDKit -- a missing JVM never hard-fails ().
  * both-endpoint -> RDKit-bond mapping: centres' per-atom E/Z labels at both
    endpoints are applied onto the correct DOUBLE bond's _CIPCode ().
  * label parsing: tetrahedral single label + both-endpoint E/Z token lists.

Tests that need the live jar skip cleanly when Java / the jar is absent.
"""

from pathlib import Path

import pytest
from rdkit import Chem

from orthonym.perception.centres_bridge import (
    PROJECT_ROOT,
    _find_centres_jar,
    _java_available,
    apply_centres_labels,
    centres_label_batch,
    centres_label_mol,
    parse_centres_labels,
)


def _engine_available() -> bool:
    return _find_centres_jar() is not None and _java_available()


# ---------------------------------------------------------------------------
# Pure-Python: jar resolution + label parsing (no JVM needed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_find_centres_jar_at_project_root():
    """The vendored jar resolves at PROJECT_ROOT as centres-cli-1.2.1.jar.

    CIP-UPDATE (2026-09-07): engine reverted 1.5-SNAPSHOT -> 1.2.1 (tagged
    release). _find_centres_jar() globs centres-cli-*.jar and
    picks the highest version, so this asserts the highest vendored jar.
    """
    jar = _find_centres_jar()
    assert jar is not None, "centres jar not vendored at project root"
    assert Path(jar).name == "centres-cli-1.2.1.jar"
    assert Path(jar).parent == PROJECT_ROOT
    # Vendored unmodified -> exact byte size (T-177-03 provenance).
    assert Path(jar).stat().st_size == 2291742


@pytest.mark.integration
def test_parse_centres_labels_tetrahedral():
    """A single tetrahedral token maps the 1-based atom to its descriptor."""
    assert parse_centres_labels("2S") == {2: "S"}
    assert parse_centres_labels("13R 5S") == {13: "R", 5: "S"}


@pytest.mark.integration
def test_parse_centres_labels_both_endpoints():
    """C=C / C=N E/Z is labelled at BOTH 1-based endpoints."""
    assert parse_centres_labels("7E 8E") == {7: "E", 8: "E"}
    assert parse_centres_labels("2E 3E") == {2: "E", 3: "E"}


@pytest.mark.integration
def test_parse_centres_labels_cumulene_marker_skipped():
    """A bare cumulene marker (CT4, no atom index) carries no per-atom mapping."""
    assert parse_centres_labels("CT4") == {}
    # Mixed: the cumulene marker is skipped, the indexed token kept.
    assert parse_centres_labels("CT4 2E") == {2: "E"}


@pytest.mark.integration
def test_parse_centres_labels_empty():
    assert parse_centres_labels("") == {}
    assert parse_centres_labels("   ") == {}


# ---------------------------------------------------------------------------
# Bond mapping () -- no JVM needed (synthetic label map)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_labels_sets_double_bond_cipcode():
    """A synthetic both-endpoint E label lands on the correct DOUBLE bond."""
    # but-2-ene: atoms C0 C1=C2 C3; the double bond is between atoms 1 and 2.
    mol = Chem.MolFromSmiles("C/C=C/C")
    assert mol is not None
    # centres would emit "2E 3E" (1-based atoms 2,3 == 0-based 1,2).
    apply_centres_labels(mol, {2: "E", 3: "E"})
    db = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    assert len(db) == 1
    assert db[0].HasProp("_CIPCode")
    assert db[0].GetProp("_CIPCode") == "E"


@pytest.mark.integration
def test_apply_labels_sets_tetrahedral_atom_cipcode():
    """A single tetrahedral label lands on the atom (1-based -> 0-based)."""
    mol = Chem.MolFromSmiles("C[C@H](O)CC")
    assert mol is not None
    apply_centres_labels(mol, {2: "S"})
    atom = mol.GetAtomWithIdx(1)  # 1-based 2 -> 0-based 1
    assert atom.HasProp("_CIPCode")
    assert atom.GetProp("_CIPCode") == "S"


@pytest.mark.integration
def test_apply_labels_diene_targets_specific_bond():
    """In a diene only the labelled double bond gets the descriptor."""
    # hexa-2,4-diene: C0 C1=C2 C3=C4 C5; two double bonds.
    mol = Chem.MolFromSmiles("C/C=C/C=C/C")
    assert mol is not None
    # Label only the first double bond (1-based atoms 2,3).
    apply_centres_labels(mol, {2: "E", 3: "E"})
    db = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    assert len(db) == 2
    labelled = [b for b in db if b.HasProp("_CIPCode")]
    assert len(labelled) == 1
    b = labelled[0]
    assert {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} == {1, 2}


# ---------------------------------------------------------------------------
# Graceful fallback () -- monkeypatched jar-absent, no JVM needed
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_batch_returns_none_when_jar_absent(monkeypatch):
    """jar-absent -> centres_label_batch returns None (caller falls back)."""
    import orthonym.perception.centres_bridge as cb
    monkeypatch.setattr(cb, "_find_centres_jar", lambda *a, **k: None)
    assert cb.centres_label_batch(["C[C@H](O)CC"]) is None


@pytest.mark.integration
def test_mol_label_returns_false_when_jar_absent(monkeypatch):
    """jar-absent -> centres_label_mol returns False (caller falls back to RDKit)."""
    import orthonym.perception.centres_bridge as cb
    monkeypatch.setattr(cb, "_find_centres_jar", lambda *a, **k: None)
    mol = Chem.MolFromSmiles("C[C@H](O)CC")
    assert cb.centres_label_mol(mol) is False


@pytest.mark.integration
def test_batch_returns_none_when_no_jvm_at_all(monkeypatch):
    """Engine unavailable -> centres_label_batch returns None ().

    There are now TWO ways to reach a JVM, so both must be absent for the engine
    to count as unavailable: the in-process JVM (``jvm_bridge``, JPype) and the
    external ``java`` binary. This asserts the contract itself -- no JVM by
    any route means None, so the caller falls back to RDKit rather than hard-failing.

    Note the SMILES must not already be in the process-level label cache, or the
    cache would serve it before any availability check is reached.
    """
    import orthonym.jvm_bridge as jb
    import orthonym.perception.centres_bridge as cb
    smi = "CC[C@@H](O)CC[C@H](C)Cl"          # distinctive; not cached by other tests
    assert smi not in cb._CENTRES_LABEL_CACHE, "precondition: SMILES not cached"
    monkeypatch.setattr(cb, "_java_available", lambda *a, **k: False)
    monkeypatch.setattr(jb, "centres_available", lambda *a, **k: False)
    assert cb.centres_label_batch([smi]) is None


@pytest.mark.integration
def test_batch_works_without_java_binary_when_inprocess_jvm_available(monkeypatch):
    """An in-process JVM needs no ``java`` BINARY on PATH.

    JPype loads libjvm into this process, so ``_java_available()`` -- which shells
    out to ``java -version`` -- stopped being the operative availability test. This
    locks the intended new behaviour and would catch a regression that silently
    reintroduced a per-call `java` process launch as a precondition.

    Skips when the in-process JVM genuinely is not available (no jpype / no jar).
    """
    import orthonym.jvm_bridge as jb
    import orthonym.perception.centres_bridge as cb
    if not jb.centres_available():
        pytest.skip("in-process JVM (jpype + centres jar) not available")
    # Remove the external binary as an option, so a pass can ONLY come from the
    # in-process JVM. Without this the test would prove nothing.
    monkeypatch.setattr(cb, "_java_available", lambda *a, **k: False)
    smi = "C[C@@H](O)CCCC"                    # distinctive; forces a real engine call
    cb._CENTRES_LABEL_CACHE.pop(smi, None)
    result = cb.centres_label_batch([smi])
    assert result is not None, "in-process engine should have served this"
    assert any(d in ("R", "S", "r", "s") for d in result[smi].values()), result


@pytest.mark.integration
def test_empty_batch_returns_empty_dict():
    """An empty SMILES list short-circuits to {} (no JVM spawned)."""
    assert centres_label_batch([]) == {}


# ---------------------------------------------------------------------------
# Live engine (needs the jar + Java) -- skip cleanly otherwise
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_centres_live_tetrahedral():
    """A known R/S molecule gets a tetrahedral descriptor from the live engine."""
    if not _engine_available():
        pytest.skip("centres jar or Java runtime not available")
    result = centres_label_batch(["C[C@H](O)CC"])
    assert result is not None
    labels = result["C[C@H](O)CC"]
    assert any(d in ("R", "S", "r", "s") for d in labels.values()), labels


@pytest.mark.integration
def test_centres_live_double_bond():
    """A known C=C molecule gets a both-endpoint E/Z label from the live engine."""
    if not _engine_available():
        pytest.skip("centres jar or Java runtime not available")
    result = centres_label_batch(["C/C=C/C"])
    assert result is not None
    labels = result["C/C=C/C"]
    assert any(d in ("E", "Z") for d in labels.values()), labels


@pytest.mark.integration
def test_centres_live_mol_apply_double_bond():
    """End-to-end: centres labels a C=C mol's bond _CIPCode in place."""
    if not _engine_available():
        pytest.skip("centres jar or Java runtime not available")
    mol = Chem.MolFromSmiles("C/C=C/C")
    ran = centres_label_mol(mol)
    assert ran is True
    db = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
    assert db[0].HasProp("_CIPCode")
    assert db[0].GetProp("_CIPCode") in ("E", "Z")


@pytest.mark.integration
def test_centres_engine_scores_281_on_suite():
    """centres-ON scores 281/290 on the CIP validation suite (gate).

    CIP-UPDATE (2026-09-07): engine reverted 1.5-SNAPSHOT -> 1.2.1 (tagged
    release). R/S/E/Z labels byte-identical between jars; 1.2.1 gets the two
    exotic cyclic-cumulene axial M/P labels Orthonym does not consume. See
    test_cip_validation.test_centres_engine_281.

    Single batched JVM invocation () via the shared validation-suite
    scorer.
    """
    if not _engine_available():
        pytest.skip("centres jar or Java runtime not available")
    from tests.integration.test_cip_validation import (
        load_cip_data,
        score_suite_centres,
    )
    cip_data = load_cip_data()
    centres_pass = score_suite_centres(cip_data)
    assert centres_pass == 281, (
        f"centres CIP-suite pass count drifted: expected 281, got {centres_pass}"
    )
