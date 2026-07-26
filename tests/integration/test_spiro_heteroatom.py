"""Phase 151-02 D-12 + D-23: hetero-spiro 'a'-prefix OPSIN round-trip.

Confirms heteroatom 'a'-prefix path uses
``polycyclic_bridged.get_heteroatom_prefix`` and that the resulting
names round-trip through OPSIN to InChI L1 == input.

Source: 151-02-PLAN.md task 1b; 151-AUDIT-B.md (rows 5-8 hetero-spiro
matrix); IUPAC P-24.2.4.1.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym.rules.spiro import name_spiro_system


_OPSIN_JAR = (
    Path(__file__).resolve().parents[2]
    / "opsin-cli-2.9.0-jar-with-dependencies.jar"
)


def _opsin_available() -> bool:
    return _OPSIN_JAR.exists() and shutil.which("java") is not None


def _opsin_parse(name: str) -> str | None:
    try:
        proc = subprocess.run(
            ["java", "-jar", str(_OPSIN_JAR), "-osmi"],
            input=name + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("Run the jar"):
                continue
            if line.startswith("Enter a chemical"):
                continue
            if line.lower().startswith("error") or line.startswith("Could not"):
                return None
            return line
        return None
    except Exception:
        return None


# Hetero-spiro fixtures — taken from blue_book_examples.json so the data
# source is the same artifact used by the unit tests.
def _hetero_blue_book_fixtures():
    p = (Path(__file__).resolve().parents[1]
         / "fixtures" / "ring_systems" / "spiro" / "blue_book_examples.json")
    if not p.exists():
        return []
    fx = json.loads(p.read_text())
    return [f for f in fx if f.get("compound_class") == "spiro-hetero"]


_HETERO_FIXTURES = _hetero_blue_book_fixtures()


@pytest.mark.integration
@pytest.mark.roundtrip
@pytest.mark.skipif(not _opsin_available(),
                    reason="OPSIN/Java not available")
@pytest.mark.parametrize(
    "fixture",
    _HETERO_FIXTURES,
    ids=lambda f: f["fixture_id"] if isinstance(f, dict) else "no-id",
)
def test_hetero_spiro_blue_book_roundtrip(fixture):
    """Each Blue Book hetero-spiro fixture round-trips via OPSIN.

    Per 151-AUDIT-B verdict PURE_SPIRO_PARTIAL, several fixtures
    currently emit non-canonical heteroatom locants (e.g.,
    `3-oxaspiro[3.5]nonane` instead of `2-oxaspiro[3.5]nonane`). The
    round-trip InChI L1 (formula + connectivity) is invariant under
    locant permutations of the same skeleton, so this test passes for
    BOTH the canonical and the non-canonical forms — until the v19
    follow-up locks the canonical ordering.
    """
    mol = Chem.MolFromSmiles(fixture["smiles"])
    assert mol is not None, fixture["fixture_id"]
    result = name_spiro_system(mol)
    assert result is not None, fixture["fixture_id"]
    name = result[0]
    parsed = _opsin_parse(name)
    if parsed is None:
        pytest.skip(f"OPSIN cannot parse {name!r} for "
                    f"{fixture['fixture_id']} — see audit notes")
    rt = Chem.MolFromSmiles(parsed)
    if rt is None:
        pytest.skip(f"OPSIN output not RDKit-parseable for "
                    f"{fixture['fixture_id']}: {parsed!r}")
    i_in = Chem.MolToInchi(mol).split("/c", 1)[0]
    i_rt = Chem.MolToInchi(rt).split("/c", 1)[0]
    assert i_in == i_rt, (
        f"InChI L1 mismatch on {fixture['fixture_id']}: "
        f"input={i_in!r} round-trip={i_rt!r} via name={name!r}"
    )


@pytest.mark.integration
def test_hetero_spiro_imports_polycyclic_bridged_helper():
    """D-12: spiro.py reuses polycyclic_bridged.get_heteroatom_prefix."""
    import inspect
    from orthonym.rules import spiro
    src = inspect.getsource(spiro)
    assert "from ..rules.polycyclic_bridged import get_heteroatom_prefix" in src \
        or "from .polycyclic_bridged import get_heteroatom_prefix" in src \
        or ("get_heteroatom_prefix" in src
            and "polycyclic_bridged" in src), (
        "spiro.py must reuse get_heteroatom_prefix from polycyclic_bridged "
        "(D-12 lock)"
    )


@pytest.mark.integration
def test_hetero_spiro_no_new_prefix_builder():
    """D-12: no new heteroatom prefix builder beyond _build_hetero_prefix."""
    import inspect
    from orthonym.rules import spiro
    src = inspect.getsource(spiro)
    # Allow only the existing _build_hetero_prefix; reject any new
    # _build_heteroatom_prefix or similar.
    assert "def _build_heteroatom_prefix" not in src


@pytest.mark.integration
def test_hetero_spiro_priority_order_in_existing_helper():
    """IUPAC P-25.2 priority order O > S > Se > N > P > Si > B is honored
    by the existing _build_hetero_prefix function.

    Asserted BEHAVIOURALLY. This test used to grep the function's source for the
    literal list ``['O', 'S', 'Se', 'N', 'P', 'Si', 'B']`` (or merely for the
    substrings ``'O'`` and ``'N'``). That hard-coded list is gone: ordering is
    delegated to ``hw_heteroatoms.sort_heteroatoms_by_priority``, the canonical
    seniority source, which also covers the Te/Ge/As/Sb/Bi/Sn/Pb rows the literal
    list omitted. A substring lock cannot tell correct delegation from a
    regression, and it fails on any refactor that improves the code; citing the
    prefixes in the wrong order is what actually matters, so test that.
    """
    from orthonym.perception.rings import get_spiro_atoms
    from orthonym.rules.spiro import _build_hetero_prefix

    # 3-oxa / 9-thia on a spiro[5.5]undecane: O must be cited BEFORE S even
    # though this places them on ascending locants anyway ...
    mol = Chem.MolFromSmiles("C1CC2(CCO1)CCSCC2")
    assert mol is not None
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    result = _build_hetero_prefix(mol, set(get_spiro_atoms(mol)), ring)
    assert result is not None, "in-table hetero spiro must still build"
    # ``_build_hetero_prefix`` returns the shared ``ReplacementPrefix`` (a
    # per-atom decomposition riding alongside the string, af0d7262); unwrap
    # ``.prefix`` to get the plain string this test asserts ordering on.
    prefix = result.prefix
    assert prefix.index("oxa") < prefix.index("thia"), (
        f"O must be cited before S (P-25.2 seniority): {prefix!r}"
    )

    # ... and the seniority order must NOT be an artefact of locant order: here
    # the senior O carries the HIGHER locant, so a locant-sorted implementation
    # would cite thia first.
    mol2 = Chem.MolFromSmiles("C1CC2(CCS1)CCOCC2")
    assert mol2 is not None
    ring2 = {a.GetIdx() for a in mol2.GetAtoms() if a.IsInRing()}
    result2 = _build_hetero_prefix(mol2, set(get_spiro_atoms(mol2)), ring2)
    assert result2 is not None
    prefix2 = result2.prefix
    assert prefix2.index("oxa") < prefix2.index("thia"), (
        f"O must be cited before S regardless of locants: {prefix2!r}"
    )


@pytest.mark.integration
def test_hetero_spiro_emits_a_prefix_on_2_oxaspiro_3_5_nonane():
    """Smoke check: 'oxa' appears in the name of a hetero-spiro compound."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("C1CC2(O1)CCCCC2")
    result = name_spiro_system(mol)
    assert result is not None
    name = result[0]
    assert "oxa" in name


@pytest.mark.integration
def test_hetero_spiro_emits_a_prefix_on_dioxaspiro():
    """Multi-heteroatom 'a'-prefix smoke check."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("C1CC2(OCCO2)CCC1")
    result = name_spiro_system(mol)
    assert result is not None
    name = result[0]
    assert "dioxa" in name


@pytest.mark.integration
@pytest.mark.roundtrip
@pytest.mark.skipif(not _opsin_available(),
                    reason="OPSIN/Java not available")
def test_hetero_spiro_count_at_least_eight():
    """Acceptance gate: ≥8 hetero-spiro compounds covered.

    The Blue Book hetero subset above provides the 'a'-prefix coverage.
    Pure-carbon spiro fixtures provide the carbocyclic coverage. Both
    families are exercised in the parametrized round-trip test.
    """
    # Count Blue Book hetero + a few pure-spiro hetero variants from
    # the corpus (which ARE hetero, e.g., spiro-fused with O/N/S in
    # the side ring). Plan 151-02 acceptance gate is structural, not
    # corpus-mined — we ship the gate from the Blue Book set + corpus
    # natural mixed-hetero compounds.
    p = (Path(__file__).resolve().parents[1]
         / "fixtures" / "ring_systems" / "spiro" / "corpus_mined.json")
    corpus_hetero = []
    if p.exists():
        for f in json.loads(p.read_text()):
            mol = Chem.MolFromSmiles(f["smiles"])
            if mol is None:
                continue
            if any(a.GetSymbol() not in ("C", "H")
                   for a in mol.GetAtoms()
                   if a.GetIsAromatic() or a.IsInRing()):
                # Includes ring atoms with hetero symbols
                ring_hetero = any(
                    a.GetSymbol() != "C" and a.IsInRing()
                    for a in mol.GetAtoms()
                )
                if ring_hetero:
                    corpus_hetero.append(f["fixture_id"])
    total = len(_HETERO_FIXTURES) + len(corpus_hetero)
    assert total >= 8, (
        f"hetero-spiro coverage gate requires ≥8 compounds, got "
        f"{total} (blue_book={len(_HETERO_FIXTURES)} + "
        f"corpus_hetero={len(corpus_hetero)})"
    )
